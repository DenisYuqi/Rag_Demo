"""Versioned semantic fact-support assessment without retrieval-score coercion."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
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

FACT_EVIDENCE_ASSESSOR_VERSION = "semantic-authority-assertion-rerank-v3"

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$")
_FACT_SEPARATOR = re.compile(
    r"(?:[?\uff1f;\uff1b\n]+|\b(?:and|also)\b|(?:以及|并且|同时))",
    re.IGNORECASE,
)
_ASSERTION_SEPARATOR = re.compile(r"(?<=[\u3002\uff01\uff1f\uff1b;])\s*|(?<=[.!?])\s+|\n{2,}")
_RESPONSE_INSTRUCTION = re.compile(
    r"(?i)^(?:please\s+)?(?:answer|respond|reply)\s+in\s+(?:chinese|english)\b|"
    r"^(?:请)?(?:用|使用)(?:中文|英文)(?:回答|回复|作答)"
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
_CURRENT_EVIDENCE = re.compile(
    r"(?i)(?:\bcurrent(?:ly)?\b.{0,48}\b(?:authoritative|effective|valid)\b|"
    r"\b(?:authoritative|effective|valid)\b.{0,48}\bcurrent(?:ly)?\b|"
    r"\bauthoritative\s+(?:policy|source|document)\b|"
    r"\u5f53\u524d\u6709\u6548|\u73b0\u884c\u6709\u6548|\u5f53\u524d\u751f\u6548|"
    r"\u6743\u5a01\u653f\u7b56)"
)
_WITHDRAWN_EVIDENCE = re.compile(
    r"(?i)(?:\bwithdrawn\b|\bobsolete\b|\bsuperseded\b|\bdraft\b|"
    r"\btraining\s+(?:draft|example)\b|\bnot\s+an?\s+authoritative\b|"
    r"\u5df2\u64a4\u56de|\u64a4\u56de|\u8349\u6848|\u57f9\u8bad\u793a\u4f8b|"
    r"\u4f5c\u5e9f|\u5e9f\u6b62|\u4e0d\u518d\u6709\u6548)"
)
_WITHDRAWN_REFERENCE = re.compile(
    r"(?i)(?:\b(?:draft|withdrawn|obsolete|superseded)\b|"
    r"\u8349\u6848|\u5df2\u64a4\u56de|\u64a4\u56de|\u4f5c\u5e9f|\u5e9f\u6b62)"
)
_WITHDRAWN_SELECTION = re.compile(
    r"(?i)(?:\b(?:choose|select|use|return|give)\b|"
    r"\u9009\u62e9|\u91c7\u7528|\u4f7f\u7528|\u7ed9\u51fa)"
)
_UNQUALIFIED_SELECTION = re.compile(
    r"(?i)(?:\bwithout\s+qualification\b|\bunqualified\b|\bdirectly\b|"
    r"\u76f4\u63a5|\u4e0d\u52a0\u8bf4\u660e|\u65e0\u9700\u8bf4\u660e)"
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


class EvidenceLifecycleStatus(StrEnum):
    """Explicit lifecycle status derived from one reranked child."""

    CURRENT = "current"
    WITHDRAWN = "withdrawn"
    UNSPECIFIED = "unspecified"


@dataclass(frozen=True, slots=True)
class EvidenceAuthorityMetadata:
    """Safe request-scoped authority metadata without retaining source text."""

    chunk_id: str
    status: EvidenceLifecycleStatus
    authority_level: int
    signals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if _SAFE_ID.fullmatch(self.chunk_id) is None:
            raise ValueError("authority_chunk_id_invalid")
        expected_level = {
            EvidenceLifecycleStatus.WITHDRAWN: 0,
            EvidenceLifecycleStatus.UNSPECIFIED: 1,
            EvidenceLifecycleStatus.CURRENT: 2,
        }[self.status]
        if self.authority_level != expected_level:
            raise ValueError("authority_level_invalid")


@dataclass(frozen=True, slots=True)
class FactAssessmentResult:
    facts: tuple[FactEvidence, ...]
    provider_attempts: tuple[ModelAttempt, ...] = ()
    direct_provider_usage: TokenUsage | None = None
    direct_provider_identity: ModelIdentity | None = None
    authority_metadata: tuple[EvidenceAuthorityMetadata, ...] = ()
    authority_resolution_count: int = 0

    def __post_init__(self) -> None:
        if (self.direct_provider_usage is None) != (self.direct_provider_identity is None):
            raise ValueError("direct provider usage and identity must be recorded together")
        object.__setattr__(self, "authority_metadata", tuple(self.authority_metadata))
        if self.authority_resolution_count < 0:
            raise ValueError("authority_resolution_count_invalid")
        if len({item.chunk_id for item in self.authority_metadata}) != len(self.authority_metadata):
            raise ValueError("authority_metadata_duplicate")

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
    assertion_similarity_floor: float = 0.50
    maximum_supporting_chunks: int = 3
    authority_similarity_floor: float = 0.30
    authority_score_bonus: float = 0.15
    authority_rerank_limit: int = 3
    maximum_assertions_per_candidate: int = 12
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
        for name, value in (
            ("assertion_similarity_floor", self.assertion_similarity_floor),
            ("authority_similarity_floor", self.authority_similarity_floor),
            ("authority_score_bonus", self.authority_score_bonus),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= 1
            ):
                raise ValueError(f"{name} must be finite and in (0, 1]")
        for name in (
            "maximum_supporting_chunks",
            "authority_rerank_limit",
            "maximum_assertions_per_candidate",
            "maximum_facts",
            "maximum_candidates",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.authority_rerank_limit > self.maximum_candidates:
            raise ValueError("authority_rerank_limit must not exceed maximum_candidates")
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
            f"floor-{self.config.candidate_similarity_floor:g}:"
            f"assertion-floor-{self.config.assertion_similarity_floor:g}:"
            f"authority-floor-{self.config.authority_similarity_floor:g}:"
            f"authority-bonus-{self.config.authority_score_bonus:g}:"
            f"authority-top-{self.config.authority_rerank_limit}:"
            f"assertions-{self.config.maximum_assertions_per_candidate}"
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

        authority_metadata = tuple(_authority_metadata(candidate) for candidate in evidence)
        authority_by_id = {item.chunk_id: item for item in authority_metadata}

        assertions_by_candidate = tuple(
            _candidate_assertions(
                candidate.text,
                maximum_assertions=self.config.maximum_assertions_per_candidate,
            )
            for candidate in evidence
        )
        assertions = tuple(
            assertion
            for candidate_assertions in assertions_by_candidate
            for assertion in candidate_assertions
        )
        texts = (*facts, *assertions)
        request = EmbeddingRequest(tuple(texts))
        context = ProviderCallContext(request_id, "fact-evidence-assessment", deadline)
        raw_result, provider_attempts, direct_provider_usage = await self._embed(request, context)
        if deadline.expired:
            raise TimeoutError
        try:
            if raw_result.identity != self.required_space or len(raw_result.vectors) != len(texts):
                raise EvidenceAssessmentError("embedding_result_invalid")
            fact_vectors = raw_result.vectors[: len(facts)]
            assertion_vectors = raw_result.vectors[len(facts) :]
            candidate_assertion_vectors: list[tuple[tuple[float, ...], ...]] = []
            offset = 0
            for candidate_assertions in assertions_by_candidate:
                next_offset = offset + len(candidate_assertions)
                candidate_assertion_vectors.append(tuple(assertion_vectors[offset:next_offset]))
                offset = next_offset
            if offset != len(assertion_vectors):
                raise EvidenceAssessmentError("embedding_result_invalid")

            assessments: list[FactEvidence] = []
            authority_resolution_count = 0
            for index, fact_vector in enumerate(fact_vectors, start=1):
                scored_candidates: list[tuple[float, RankingEvidence, str]] = []
                for candidate, candidate_assertions, vectors in zip(
                    evidence,
                    assertions_by_candidate,
                    candidate_assertion_vectors,
                    strict=True,
                ):
                    best_score, best_assertion = max(
                        (
                            (_cosine_score(fact_vector, vector), assertion)
                            for assertion, vector in zip(candidate_assertions, vectors, strict=True)
                        ),
                        key=lambda item: item[0],
                    )
                    scored_candidates.append((best_score, candidate, best_assertion))
                ranked = tuple(
                    sorted(
                        scored_candidates,
                        key=lambda item: (-item[0], item[1].final_rank, item[1].chunk_id),
                    )
                )
                authority_resolution = _resolve_authority_evidence(
                    fact_id=f"fact-{index}",
                    query=query,
                    ranked=ranked,
                    authority_by_id=authority_by_id,
                    config=self.config,
                )
                if authority_resolution is not None:
                    assessments.append(authority_resolution)
                    authority_resolution_count += 1
                    continue
                support_floor = max(
                    self.config.candidate_similarity_floor,
                    self.config.assertion_similarity_floor,
                )
                selected = tuple(item for item in ranked if item[0] >= support_floor)[
                    : self.config.maximum_supporting_chunks
                ]
                if not selected:
                    assessments.append(FactEvidence(f"fact-{index}", 0))
                    continue
                score = round(selected[0][0], 6)
                primary = selected[0][1]
                primary_assertion = selected[0][2]
                conflicts = tuple(
                    candidate.chunk_id
                    for _, candidate, assertion in selected[1:]
                    if candidate.source_id != primary.source_id
                    and _materially_conflicts(primary_assertion, assertion)
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
                            tuple(candidate.chunk_id for _, candidate, _ in selected),
                        )
                    )
        except EvidenceAssessmentError as error:
            raise EvidenceAssessmentError(
                error.code,
                provider_attempts=provider_attempts,
                unrecorded_provider_attempt_count=int(direct_provider_usage is not None),
            ) from None
        return FactAssessmentResult(
            facts=tuple(assessments),
            provider_attempts=provider_attempts,
            direct_provider_usage=direct_provider_usage,
            direct_provider_identity=(
                raw_result.identity.model_identity if direct_provider_usage is not None else None
            ),
            authority_metadata=authority_metadata,
            authority_resolution_count=authority_resolution_count,
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
        and not _RESPONSE_INSTRUCTION.search(part.strip(boundary_punctuation))
    )
    if not facts:
        raise EvidenceAssessmentError("query_invalid")
    if len(facts) > maximum_facts:
        raise EvidenceAssessmentError("fact_limit_exceeded")
    return facts


def _candidate_assertions(text: str, *, maximum_assertions: int) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text)
    assertions = tuple(
        " ".join(part.split()) for part in _ASSERTION_SEPARATOR.split(normalized) if part.strip()
    )
    if not assertions:
        raise EvidenceAssessmentError("candidate_registry_invalid")
    return assertions[:maximum_assertions]


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


def _authority_metadata(candidate: RankingEvidence) -> EvidenceAuthorityMetadata:
    title = unicodedata.normalize("NFKC", candidate.display_title)
    title_withdrawn = _WITHDRAWN_EVIDENCE.search(title)
    title_current = _CURRENT_EVIDENCE.search(title)
    if title_withdrawn is not None:
        return EvidenceAuthorityMetadata(
            candidate.chunk_id,
            EvidenceLifecycleStatus.WITHDRAWN,
            0,
            ("explicit-withdrawn",),
        )
    if title_current is not None:
        return EvidenceAuthorityMetadata(
            candidate.chunk_id,
            EvidenceLifecycleStatus.CURRENT,
            2,
            ("explicit-current",),
        )
    text = unicodedata.normalize("NFKC", candidate.text)
    body_current = _CURRENT_EVIDENCE.search(text)
    body_withdrawn = _WITHDRAWN_EVIDENCE.search(text)
    if body_current is not None and (
        body_withdrawn is None or body_current.start() < body_withdrawn.start()
    ):
        return EvidenceAuthorityMetadata(
            candidate.chunk_id,
            EvidenceLifecycleStatus.CURRENT,
            2,
            ("explicit-current",),
        )
    if body_withdrawn is not None:
        return EvidenceAuthorityMetadata(
            candidate.chunk_id,
            EvidenceLifecycleStatus.WITHDRAWN,
            0,
            ("explicit-withdrawn",),
        )
    return EvidenceAuthorityMetadata(
        candidate.chunk_id,
        EvidenceLifecycleStatus.UNSPECIFIED,
        1,
    )


def _resolve_authority_evidence(
    *,
    fact_id: str,
    query: str,
    ranked: Sequence[tuple[float, RankingEvidence, str]],
    authority_by_id: dict[str, EvidenceAuthorityMetadata],
    config: FactAssessmentConfig,
) -> FactEvidence | None:
    window = tuple(
        (score, candidate, assertion)
        for score, candidate, assertion in ranked
        if candidate.final_rank <= config.authority_rerank_limit
        and score >= config.authority_similarity_floor
    )
    current = tuple(
        item
        for item in window
        if authority_by_id[item[1].chunk_id].status is EvidenceLifecycleStatus.CURRENT
    )
    withdrawn = tuple(
        item
        for item in window
        if authority_by_id[item[1].chunk_id].status is EvidenceLifecycleStatus.WITHDRAWN
    )
    if not current or not withdrawn:
        return None

    if _requests_unqualified_withdrawn_evidence(query):
        involved_current = _unique_ranked(current)
        involved_withdrawn = _unique_ranked(withdrawn)
        primary_score, primary, _ = min(
            involved_current,
            key=lambda item: (
                -min(1.0, item[0] + config.authority_score_bonus),
                item[1].final_rank,
                item[1].chunk_id,
            ),
        )
        return FactEvidence(
            fact_id,
            round(min(1.0, primary_score + config.authority_score_bonus), 6),
            (primary.chunk_id,),
            tuple(item[1].chunk_id for item in involved_withdrawn),
        )

    conflicting_pairs = tuple(
        (current_item, withdrawn_item)
        for current_item in current
        for withdrawn_item in withdrawn
        if max(current_item[0], withdrawn_item[0]) >= config.candidate_similarity_floor
        and _materially_conflicts(current_item[2], withdrawn_item[2])
    )
    if not conflicting_pairs:
        return None

    involved_current = _unique_ranked(tuple(item[0] for item in conflicting_pairs))
    involved_withdrawn = _unique_ranked(tuple(item[1] for item in conflicting_pairs))
    primary_score, primary, _ = min(
        involved_current,
        key=lambda item: (
            -min(1.0, item[0] + config.authority_score_bonus),
            item[1].final_rank,
            item[1].chunk_id,
        ),
    )
    support_score = round(
        min(1.0, primary_score + config.authority_score_bonus),
        6,
    )
    return FactEvidence(
        fact_id,
        support_score,
        tuple(item[1].chunk_id for item in involved_current),
    )


def _unique_ranked(
    values: Sequence[tuple[float, RankingEvidence, str]],
) -> tuple[tuple[float, RankingEvidence, str], ...]:
    by_id: dict[str, tuple[float, RankingEvidence, str]] = {}
    for score, candidate, assertion in values:
        by_id.setdefault(candidate.chunk_id, (score, candidate, assertion))
    return tuple(
        sorted(
            by_id.values(),
            key=lambda item: (-item[0], item[1].final_rank, item[1].chunk_id),
        )
    )


def _requests_unqualified_withdrawn_evidence(query: str) -> bool:
    normalized = unicodedata.normalize("NFKC", query)
    return bool(
        _WITHDRAWN_REFERENCE.search(normalized)
        and _WITHDRAWN_SELECTION.search(normalized)
        and _UNQUALIFIED_SELECTION.search(normalized)
    )


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
