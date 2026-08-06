"""Versioned semantic fact-support assessment without retrieval-score coercion."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import cast

from rag_mvp.domain.retrieval import RankingEvidence
from rag_mvp.providers.errors import ProviderOperationError
from rag_mvp.providers.models import (
    AttemptStatus,
    Deadline,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingSpaceIdentity,
    ModelAttempt,
    ModelIdentity,
    ProviderCallContext,
    RoutedResult,
    TokenUsage,
)
from rag_mvp.providers.protocols import EmbeddingProvider
from rag_mvp.providers.routing import ModelProviderRouter
from rag_mvp.qa.refusal import FactEvidence
from rag_mvp.retrieval.request import RetrievalRequestError, canonicalize_query

FACT_EVIDENCE_ASSESSOR_VERSION = "semantic-cosine-assertion-conflict-v1"

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$")
_FACT_SEPARATOR = re.compile(
    r"(?:[?\uff1f;\uff1b\n]+|\b(?:and|also)\b|(?:以及|并且|同时))",
    re.IGNORECASE,
)
_NUMBER = re.compile(
    r"(\d+(?:\.\d+)?)\s*(%|percent(?:age)?|days?|天|hours?|小时|yuan|元|usd|dollars?)?",
    re.IGNORECASE,
)
_NEGATIVE_ASSERTION = re.compile(
    r"(?i)\b(?:must\s+not|may\s+not|not\s+allowed|prohibited|forbidden|not\s+required)\b|"
    r"(?:不得|不允许|禁止|无需|不需要|不可以)"
)
_POSITIVE_ASSERTION = re.compile(
    r"(?i)\b(?:must|required|may|allowed|permitted)\b|(?:必须|需要|可以|允许|应当)"
)
_UNIT_ALIASES = {
    "%": "percent",
    "percent": "percent",
    "percentage": "percent",
    "day": "day",
    "days": "day",
    "天": "day",
    "hour": "hour",
    "hours": "hour",
    "小时": "hour",
    "yuan": "currency",
    "元": "currency",
    "usd": "currency",
    "dollar": "currency",
    "dollars": "currency",
}


class EvidenceAssessmentError(ValueError):
    """A stable failure to derive request-scoped fact evidence."""

    def __init__(
        self,
        code: str,
        *,
        provider_attempts: tuple[ModelAttempt, ...] = (),
        unrecorded_provider_attempt_count: int = 0,
    ) -> None:
        self.code = code
        self.provider_attempts = tuple(provider_attempts)
        self.unrecorded_provider_attempt_count = unrecorded_provider_attempt_count
        super().__init__(code)

    @property
    def provider_attempt_count(self) -> int:
        return len(self.provider_attempts) + self.unrecorded_provider_attempt_count

    @property
    def provider_failed_attempt_count(self) -> int:
        return (
            sum(attempt.status is not AttemptStatus.SUCCEEDED for attempt in self.provider_attempts)
            + self.unrecorded_provider_attempt_count
        )

    @property
    def provider_unknown_usage_attempt_count(self) -> int:
        return sum(_attempt_usage_unknown(attempt) for attempt in self.provider_attempts) + (
            self.unrecorded_provider_attempt_count
        )


@dataclass(frozen=True, slots=True)
class FactAssessmentResult:
    facts: tuple[FactEvidence, ...]
    provider_attempts: tuple[ModelAttempt, ...] = ()
    direct_provider_usage: TokenUsage | None = None
    direct_provider_identity: ModelIdentity | None = None

    def __post_init__(self) -> None:
        if (self.direct_provider_usage is None) != (self.direct_provider_identity is None):
            raise ValueError("direct provider usage and identity must be recorded together")

    @property
    def provider_attempt_count(self) -> int:
        return len(self.provider_attempts) + int(self.direct_provider_usage is not None)

    @property
    def provider_failed_attempt_count(self) -> int:
        return sum(
            attempt.status is not AttemptStatus.SUCCEEDED for attempt in self.provider_attempts
        )

    @property
    def provider_unknown_usage_attempt_count(self) -> int:
        unknown = sum(_attempt_usage_unknown(attempt) for attempt in self.provider_attempts)
        if self.direct_provider_usage is not None:
            unknown += int(self.direct_provider_usage.input_tokens is None)
        return unknown


@dataclass(frozen=True, slots=True)
class FactAssessmentConfig:
    candidate_similarity_floor: float = 0.45
    maximum_supporting_chunks: int = 3
    maximum_facts: int = 8
    maximum_candidates: int = 20
    version: str = FACT_EVIDENCE_ASSESSOR_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.candidate_similarity_floor, bool)
            or not isinstance(self.candidate_similarity_floor, (int, float))
            or not math.isfinite(self.candidate_similarity_floor)
            or not 0 < self.candidate_similarity_floor <= 1
        ):
            raise ValueError("candidate_similarity_floor must be finite and in (0, 1]")
        for name in ("maximum_supporting_chunks", "maximum_facts", "maximum_candidates"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.version != FACT_EVIDENCE_ASSESSOR_VERSION:
            raise ValueError("unsupported_fact_assessment_version")


class SemanticFactEvidenceAssessor:
    """Assess fact support with dedicated embeddings and conservative conflict rules."""

    def __init__(
        self,
        embedding: EmbeddingProvider | ModelProviderRouter,
        *,
        required_space: EmbeddingSpaceIdentity | None = None,
        config: FactAssessmentConfig | None = None,
    ) -> None:
        if isinstance(embedding, ModelProviderRouter):
            if not isinstance(required_space, EmbeddingSpaceIdentity):
                raise ValueError("required_embedding_space_missing")
            resolved_space = required_space
        elif isinstance(embedding, EmbeddingProvider):
            resolved_space = embedding.identity if required_space is None else required_space
            if resolved_space != embedding.identity:
                raise ValueError("required_embedding_space_mismatch")
        else:
            raise TypeError("embedding must implement EmbeddingProvider or be a router")
        self._embedding = embedding
        self.required_space = resolved_space
        self.config = config or FactAssessmentConfig()

    @property
    def identity(self) -> str:
        space = self.required_space
        return (
            f"{self.config.version}:{space.provider}/{space.model}/{space.dimension}/"
            f"{space.normalization.value}:{space.adapter_version}:"
            f"floor-{self.config.candidate_similarity_floor:g}"
        )

    async def assess(
        self,
        query: str,
        candidates: Sequence[RankingEvidence],
        *,
        request_id: str,
        revision_id: str,
        deadline: Deadline,
    ) -> tuple[FactEvidence, ...]:
        return (
            await self.assess_with_diagnostics(
                query,
                candidates,
                request_id=request_id,
                revision_id=revision_id,
                deadline=deadline,
            )
        ).facts

    async def assess_with_diagnostics(
        self,
        query: str,
        candidates: Sequence[RankingEvidence],
        *,
        request_id: str,
        revision_id: str,
        deadline: Deadline,
    ) -> FactAssessmentResult:
        if not isinstance(request_id, str) or _SAFE_ID.fullmatch(request_id) is None:
            raise EvidenceAssessmentError("request_id_invalid")
        if not isinstance(revision_id, str) or _SAFE_ID.fullmatch(revision_id) is None:
            raise EvidenceAssessmentError("revision_id_invalid")
        if not isinstance(deadline, Deadline) or deadline.expired:
            raise TimeoutError
        facts = _decompose_query(query, maximum_facts=self.config.maximum_facts)
        evidence = _validated_candidates(
            candidates,
            revision_id=revision_id,
            maximum_candidates=self.config.maximum_candidates,
        )
        if not evidence:
            return FactAssessmentResult(
                tuple(FactEvidence(f"fact-{index}", 0) for index in range(1, len(facts) + 1))
            )

        texts = (*facts, *(candidate.text for candidate in evidence))
        request = EmbeddingRequest(tuple(texts))
        context = ProviderCallContext(request_id, "fact-evidence-assessment", deadline)
        raw_result, provider_attempts, direct_provider_usage = await self._embed(request, context)
        if deadline.expired:
            raise TimeoutError
        try:
            if raw_result.identity != self.required_space or len(raw_result.vectors) != len(texts):
                raise EvidenceAssessmentError("embedding_result_invalid")
            fact_vectors = raw_result.vectors[: len(facts)]
            candidate_vectors = raw_result.vectors[len(facts) :]

            assessments: list[FactEvidence] = []
            for index, fact_vector in enumerate(fact_vectors, start=1):
                ranked = sorted(
                    (
                        (_cosine_score(fact_vector, vector), candidate)
                        for vector, candidate in zip(candidate_vectors, evidence, strict=True)
                    ),
                    key=lambda item: (-item[0], item[1].final_rank, item[1].chunk_id),
                )
                selected = tuple(
                    item for item in ranked if item[0] >= self.config.candidate_similarity_floor
                )[: self.config.maximum_supporting_chunks]
                if not selected:
                    assessments.append(FactEvidence(f"fact-{index}", 0))
                    continue
                score = round(selected[0][0], 6)
                primary = selected[0][1]
                conflicts = tuple(
                    candidate.chunk_id
                    for _, candidate in selected[1:]
                    if candidate.source_id != primary.source_id
                    and _materially_conflicts(primary.text, candidate.text)
                )
                if conflicts:
                    assessments.append(
                        FactEvidence(
                            f"fact-{index}",
                            score,
                            (primary.chunk_id,),
                            conflicts,
                        )
                    )
                else:
                    assessments.append(
                        FactEvidence(
                            f"fact-{index}",
                            score,
                            tuple(candidate.chunk_id for _, candidate in selected),
                        )
                    )
        except EvidenceAssessmentError as error:
            raise EvidenceAssessmentError(
                error.code,
                provider_attempts=provider_attempts,
                unrecorded_provider_attempt_count=int(direct_provider_usage is not None),
            ) from None
        return FactAssessmentResult(
            tuple(assessments),
            provider_attempts,
            direct_provider_usage,
            raw_result.identity.model_identity if direct_provider_usage is not None else None,
        )

    async def _embed(
        self,
        request: EmbeddingRequest,
        context: ProviderCallContext,
    ) -> tuple[EmbeddingResult, tuple[ModelAttempt, ...], TokenUsage | None]:
        if isinstance(self._embedding, ModelProviderRouter):
            try:
                routed = await self._embedding.embed(
                    request,
                    context,
                    required_space=self.required_space,
                )
            except ProviderOperationError as error:
                raise EvidenceAssessmentError(
                    "embedding_provider_failed",
                    provider_attempts=error.attempts,
                ) from None
            if not isinstance(routed, RoutedResult) or not isinstance(
                routed.value, EmbeddingResult
            ):
                raise EvidenceAssessmentError(
                    "embedding_result_invalid",
                    provider_attempts=getattr(routed, "attempts", ()),
                )
            return routed.value, routed.attempts, None
        try:
            result = await self._embedding.embed(request, context)
        except Exception:
            raise EvidenceAssessmentError(
                "embedding_provider_failed",
                unrecorded_provider_attempt_count=1,
            ) from None
        if not isinstance(result, EmbeddingResult):
            raise EvidenceAssessmentError(
                "embedding_result_invalid",
                unrecorded_provider_attempt_count=1,
            )
        return result, (), result.usage


def _attempt_usage_unknown(attempt: ModelAttempt) -> bool:
    if attempt.role.value == "embedding":
        return attempt.usage.input_tokens is None
    return attempt.usage.input_tokens is None or attempt.usage.output_tokens is None


def _decompose_query(query: str, *, maximum_facts: int) -> tuple[str, ...]:
    try:
        canonical = canonicalize_query(query)
    except (RetrievalRequestError, TypeError, ValueError):
        raise EvidenceAssessmentError("query_invalid") from None
    boundary_punctuation = " ,.\u3002\uff0c"
    facts = tuple(
        part.strip(boundary_punctuation)
        for part in _FACT_SEPARATOR.split(canonical)
        if part.strip(boundary_punctuation)
    )
    if not facts:
        raise EvidenceAssessmentError("query_invalid")
    if len(facts) > maximum_facts:
        raise EvidenceAssessmentError("fact_limit_exceeded")
    return facts


def _validated_candidates(
    candidates: object,
    *,
    revision_id: str,
    maximum_candidates: int,
) -> tuple[RankingEvidence, ...]:
    if isinstance(candidates, (str, bytes, bytearray)) or not isinstance(candidates, Sequence):
        raise EvidenceAssessmentError("candidate_registry_invalid")
    values: list[RankingEvidence] = []
    for candidate in cast(Sequence[object], candidates):
        if not isinstance(candidate, RankingEvidence):
            raise EvidenceAssessmentError("candidate_registry_invalid")
        values.append(candidate)
    ordered = tuple(sorted(values, key=lambda candidate: candidate.final_rank))
    ranks = tuple(candidate.final_rank for candidate in ordered)
    if (
        ranks != tuple(range(1, len(ordered) + 1))
        or len({candidate.chunk_id for candidate in ordered}) != len(ordered)
        or any(candidate.revision_id != revision_id for candidate in ordered)
        or len(ordered) > maximum_candidates
    ):
        raise EvidenceAssessmentError("candidate_registry_invalid")
    return ordered


def _cosine_score(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise EvidenceAssessmentError("embedding_result_invalid")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise EvidenceAssessmentError("embedding_result_invalid")
    cosine = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    if not math.isfinite(cosine):
        raise EvidenceAssessmentError("embedding_result_invalid")
    return min(1.0, max(0.0, cosine))


def _materially_conflicts(left: str, right: str) -> bool:
    left_numeric = _single_numeric_assertion(left)
    right_numeric = _single_numeric_assertion(right)
    if (
        left_numeric is not None
        and right_numeric is not None
        and left_numeric[1] == right_numeric[1]
        and left_numeric[0] != right_numeric[0]
    ):
        return True
    left_polarity = _assertion_polarity(left)
    right_polarity = _assertion_polarity(right)
    return left_polarity != 0 and right_polarity != 0 and left_polarity != right_polarity


def _single_numeric_assertion(text: str) -> tuple[Decimal, str] | None:
    matches = _NUMBER.findall(text)
    if len(matches) != 1:
        return None
    raw_value, raw_unit = matches[0]
    try:
        value = Decimal(raw_value)
    except InvalidOperation:
        return None
    unit = _UNIT_ALIASES.get(raw_unit.casefold(), raw_unit.casefold())
    return value, unit


def _assertion_polarity(text: str) -> int:
    if _NEGATIVE_ASSERTION.search(text):
        return -1
    if _POSITIVE_ASSERTION.search(text):
        return 1
    return 0
