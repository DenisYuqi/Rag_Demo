"""Bounded, deadline-aware optional reranking with deterministic fallback."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

from rag_mvp.domain.retrieval import RetrievalCandidate
from rag_mvp.providers.errors import ProviderError, ProviderOperationError
from rag_mvp.providers.models import (
    Deadline,
    ModelAttempt,
    ModelIdentity,
    ProviderCallContext,
    ProviderErrorCategory,
    RerankCandidate,
    RerankRequest,
    RerankResult,
    RoutedRerankResult,
    TokenUsage,
)
from rag_mvp.providers.protocols import RerankingProvider
from rag_mvp.providers.routing import ModelProviderRouter


class RerankIntegrityError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class RerankTruncationPolicy:
    maximum_query_characters: int = 2048
    maximum_query_tokens: int = 256
    maximum_candidate_characters: int = 2048
    maximum_candidate_tokens: int = 512
    version: str = "unicode-codepoint-prefix-v1"

    def __post_init__(self) -> None:
        for name in (
            "maximum_query_characters",
            "maximum_query_tokens",
            "maximum_candidate_characters",
            "maximum_candidate_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name}_invalid")
        if self.version != "unicode-codepoint-prefix-v1":
            raise ValueError("rerank_truncation_version_invalid")

    def truncate_query(self, query: str) -> str:
        return _bounded_text(
            query,
            min(self.maximum_query_characters, self.maximum_query_tokens),
        )

    def truncate_candidate(self, text: str) -> str:
        return _bounded_text(
            text,
            min(self.maximum_candidate_characters, self.maximum_candidate_tokens),
        )


@dataclass(frozen=True, slots=True)
class RerankStageResult:
    ordered_candidates: tuple[RetrievalCandidate, ...]
    applied: bool
    degraded: bool
    reason: str | None
    attempts: tuple[ModelAttempt, ...]
    route_id: str | None
    identity: ModelIdentity | None
    prompt_version: str | None
    truncation_version: str
    parser_version: str
    usage: TokenUsage | None
    elapsed_ms: float
    submitted_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordered_candidates", tuple(self.ordered_candidates))
        object.__setattr__(self, "attempts", tuple(self.attempts))
        if any(not isinstance(attempt, ModelAttempt) for attempt in self.attempts):
            raise TypeError("rerank attempts must be ModelAttempt values")
        if self.route_id is not None and (
            not isinstance(self.route_id, str) or not self.route_id.strip()
        ):
            raise ValueError("rerank route identity is invalid")
        if self.identity is not None and not isinstance(self.identity, ModelIdentity):
            raise TypeError("rerank identity must be a ModelIdentity")
        if self.prompt_version is not None and (
            not isinstance(self.prompt_version, str) or not self.prompt_version.strip()
        ):
            raise ValueError("rerank prompt identity is invalid")
        if self.usage is not None and not isinstance(self.usage, TokenUsage):
            raise TypeError("rerank usage must be TokenUsage")
        if self.applied and self.degraded:
            raise ValueError("applied reranking cannot be degraded")
        if self.degraded != (self.reason is not None):
            raise ValueError("rerank degradation reason is inconsistent")
        if not math.isfinite(self.elapsed_ms) or self.elapsed_ms < 0:
            raise ValueError("rerank elapsed time must be finite and non-negative")
        if type(self.submitted_count) is not int or self.submitted_count < 0:
            raise ValueError("rerank submitted count must be non-negative")
        if self.submitted_count > len(self.ordered_candidates):
            raise ValueError("rerank submitted count exceeds candidate count")
        if self.applied and self.submitted_count == 0:
            raise ValueError("applied reranking requires submitted candidates")
        if self.submitted_count > 0 and not self.applied and not self.degraded:
            raise ValueError("unapplied reranking with candidates must be degraded")
        metadata = (self.identity, self.prompt_version, self.usage)
        if self.applied and any(value is None for value in metadata):
            raise ValueError("applied reranking requires provider result metadata")

    @property
    def provider_identity(self) -> str | None:
        if self.identity is None:
            return None
        route = f"{self.route_id}:" if self.route_id is not None else ""
        return (
            f"{route}{self.identity.provider}/{self.identity.model}/"
            f"{self.identity.adapter_version}:{self.prompt_version}:"
            f"{self.truncation_version}:{self.parser_version}"
        )


class RerankStage:
    """Invoke one provider/router within a strict subdeadline and preserve base order."""

    PROMPT_VERSION = "listwise-rerank-v1"
    PARSER_VERSION = "exact-id-permutation-v1"

    def __init__(
        self,
        reranker: RerankingProvider | ModelProviderRouter,
        *,
        candidate_limit: int = 10,
        budget_seconds: float = 1.2,
        truncation: RerankTruncationPolicy | None = None,
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        if not isinstance(reranker, (RerankingProvider, ModelProviderRouter)):
            raise TypeError("reranker must implement RerankingProvider or be a router")
        if type(candidate_limit) is not int or candidate_limit < 1:
            raise ValueError("rerank_candidate_limit_invalid")
        if not isinstance(budget_seconds, (int, float)) or isinstance(budget_seconds, bool):
            raise ValueError("rerank_budget_invalid")
        normalized_budget = float(budget_seconds)
        if not math.isfinite(normalized_budget) or normalized_budget <= 0:
            raise ValueError("rerank_budget_invalid")
        if not isinstance(prompt_version, str) or not prompt_version.strip():
            raise ValueError("rerank_prompt_version_invalid")
        self._reranker = reranker
        self.candidate_limit = candidate_limit
        self.budget_seconds = normalized_budget
        self.truncation = truncation or RerankTruncationPolicy()
        self.prompt_version = prompt_version

    async def run(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        context: ProviderCallContext,
    ) -> RerankStageResult:
        if not isinstance(context, ProviderCallContext):
            raise TypeError("context must be a ProviderCallContext")
        base = _candidate_registry(candidates)
        submitted = base[: self.candidate_limit]
        started = context.deadline.clock()
        if not submitted:
            return self._result(
                base,
                applied=False,
                degraded=False,
                reason=None,
                attempts=(),
                route_id=None,
                identity=None,
                prompt_version=None,
                usage=None,
                started=started,
                submitted_count=0,
                clock=context.deadline.clock,
            )

        remaining = context.deadline.expires_at - started
        duration = min(self.budget_seconds, remaining)
        if not math.isfinite(duration) or duration <= 0:
            return self._degraded(
                base,
                "rerank_deadline_exceeded",
                (),
                started,
                len(submitted),
                context.deadline.clock,
                identity=_direct_identity(self._reranker),
                prompt_version=self.prompt_version,
            )
        subdeadline = Deadline(started + duration, context.deadline.clock)
        subcontext = ProviderCallContext(
            request_id=context.request_id,
            operation_id=context.operation_id,
            deadline=subdeadline,
        )
        request = RerankRequest(
            query=self.truncation.truncate_query(query),
            candidates=tuple(
                RerankCandidate(
                    candidate.chunk_id,
                    self.truncation.truncate_candidate(candidate.text),
                )
                for candidate in submitted
            ),
            prompt_version=self.prompt_version,
            max_query_tokens=self.truncation.maximum_query_tokens,
            max_candidate_tokens=self.truncation.maximum_candidate_tokens,
        )

        try:
            async with asyncio.timeout(duration):
                raw_result = await self._reranker.rerank(request, subcontext)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return self._degraded(
                base,
                "rerank_timeout",
                (),
                started,
                len(submitted),
                context.deadline.clock,
                identity=_direct_identity(self._reranker),
                prompt_version=request.prompt_version,
            )
        except ProviderOperationError as error:
            return self._degraded(
                base,
                _provider_reason(error.category),
                error.attempts,
                started,
                len(submitted),
                context.deadline.clock,
                identity=_direct_identity(self._reranker),
                prompt_version=request.prompt_version,
            )
        except ProviderError as error:
            return self._degraded(
                base,
                _provider_reason(error.category),
                (),
                started,
                len(submitted),
                context.deadline.clock,
                identity=_direct_identity(self._reranker),
                prompt_version=request.prompt_version,
            )
        except Exception:
            return self._degraded(
                base,
                "rerank_provider_failure",
                (),
                started,
                len(submitted),
                context.deadline.clock,
                identity=_direct_identity(self._reranker),
                prompt_version=request.prompt_version,
            )

        try:
            if isinstance(self._reranker, ModelProviderRouter):
                if not isinstance(raw_result, RoutedRerankResult):
                    raise RerankIntegrityError("rerank_result_invalid")
                if not raw_result.applied or raw_result.degraded:
                    reason = raw_result.degradation_reason or ProviderErrorCategory.UNAVAILABLE
                    return self._degraded(
                        base,
                        _provider_reason(reason),
                        _safe_attempts(raw_result),
                        started,
                        len(submitted),
                        context.deadline.clock,
                        route_id=_safe_route_id(raw_result),
                        identity=_safe_result_identity(raw_result),
                        prompt_version=(_safe_prompt_version(raw_result) or request.prompt_version),
                        usage=_safe_usage(raw_result),
                    )
                order = _exact_permutation(raw_result.ordered_ids, request.candidate_ids)
                route_id = raw_result.route_id
                identity = raw_result.identity
                result_prompt = raw_result.prompt_version
                usage = raw_result.usage
                attempts = _validated_attempts(raw_result.attempts)
            else:
                if not isinstance(raw_result, RerankResult):
                    raise RerankIntegrityError("rerank_result_invalid")
                if raw_result.identity != self._reranker.identity:
                    raise RerankIntegrityError("rerank_identity_mismatch")
                if raw_result.prompt_version != request.prompt_version:
                    raise RerankIntegrityError("rerank_prompt_mismatch")
                order = _exact_permutation(raw_result.ordered_ids, request.candidate_ids)
                route_id = None
                identity = raw_result.identity
                result_prompt = raw_result.prompt_version
                usage = raw_result.usage
                attempts = ()
            if (
                not isinstance(identity, ModelIdentity)
                or not isinstance(result_prompt, str)
                or not result_prompt.strip()
                or not isinstance(usage, TokenUsage)
                or (
                    route_id is not None and (not isinstance(route_id, str) or not route_id.strip())
                )
            ):
                raise RerankIntegrityError("rerank_metadata_incomplete")
            ordered = _apply_order(base, submitted, order)
        except (RerankIntegrityError, TypeError, ValueError):
            attempts = _safe_attempts(raw_result)
            return self._degraded(
                base,
                "rerank_invalid_output",
                attempts,
                started,
                len(submitted),
                context.deadline.clock,
                route_id=(
                    _safe_route_id(raw_result)
                    if isinstance(raw_result, RoutedRerankResult)
                    else None
                ),
                identity=_safe_result_identity(raw_result),
                prompt_version=(_safe_prompt_version(raw_result) or request.prompt_version),
                usage=_safe_usage(raw_result),
            )

        return self._result(
            ordered,
            applied=True,
            degraded=False,
            reason=None,
            attempts=attempts,
            route_id=route_id,
            identity=identity,
            prompt_version=result_prompt,
            usage=usage,
            started=started,
            submitted_count=len(submitted),
            clock=context.deadline.clock,
        )

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        context: ProviderCallContext,
    ) -> RerankStageResult:
        return await self.run(query, candidates, context)

    def _degraded(
        self,
        base: tuple[RetrievalCandidate, ...],
        reason: str,
        attempts: tuple[ModelAttempt, ...],
        started: float,
        submitted_count: int,
        clock: Callable[[], float],
        route_id: str | None = None,
        identity: ModelIdentity | None = None,
        prompt_version: str | None = None,
        usage: TokenUsage | None = None,
    ) -> RerankStageResult:
        return self._result(
            base,
            applied=False,
            degraded=True,
            reason=reason,
            attempts=attempts,
            route_id=route_id,
            identity=identity,
            prompt_version=prompt_version,
            usage=usage,
            started=started,
            submitted_count=submitted_count,
            clock=clock,
        )

    def _result(
        self,
        candidates: tuple[RetrievalCandidate, ...],
        *,
        applied: bool,
        degraded: bool,
        reason: str | None,
        attempts: tuple[ModelAttempt, ...],
        route_id: str | None,
        identity: ModelIdentity | None,
        prompt_version: str | None,
        usage: TokenUsage | None,
        started: float,
        submitted_count: int,
        clock: Callable[[], float],
    ) -> RerankStageResult:
        current = clock()
        return RerankStageResult(
            ordered_candidates=candidates,
            applied=applied,
            degraded=degraded,
            reason=reason,
            attempts=attempts,
            route_id=route_id,
            identity=identity,
            prompt_version=prompt_version,
            truncation_version=self.truncation.version,
            parser_version=self.PARSER_VERSION,
            usage=usage,
            elapsed_ms=max(0.0, (current - started) * 1000),
            submitted_count=submitted_count,
        )


def validate_rerank_stage_result(
    base_candidates: Sequence[RetrievalCandidate],
    result: RerankStageResult,
) -> tuple[RetrievalCandidate, ...]:
    """Revalidate an exact candidate permutation at the service boundary."""

    base = _candidate_registry(base_candidates)
    if not isinstance(result, RerankStageResult):
        raise RerankIntegrityError("rerank_stage_result_invalid")
    ordered = _candidate_registry(result.ordered_candidates, allow_reranked=True)
    if len(ordered) != len(base) or {item.chunk_id for item in ordered} != {
        item.chunk_id for item in base
    }:
        raise RerankIntegrityError("rerank_candidate_registry_mismatch")
    registry = {candidate.chunk_id: candidate for candidate in base}
    for candidate in ordered:
        original = registry[candidate.chunk_id]
        values = candidate.model_dump()
        reranking_rank = values.pop("reranking_rank")
        original_values = original.model_dump()
        original_values.pop("reranking_rank")
        if values != original_values:
            raise RerankIntegrityError("rerank_candidate_mutated")
        if result.applied:
            position = ordered.index(candidate) + 1
            if position <= result.submitted_count:
                if reranking_rank != position:
                    raise RerankIntegrityError("rerank_rank_invalid")
            elif reranking_rank is not None:
                raise RerankIntegrityError("rerank_rank_pollution")
        elif reranking_rank is not None:
            raise RerankIntegrityError("rerank_rank_pollution")
    if not result.applied and ordered != base:
        raise RerankIntegrityError("unapplied_rerank_changed_order")
    return ordered


def _candidate_registry(
    candidates: object,
    *,
    allow_reranked: bool = False,
) -> tuple[RetrievalCandidate, ...]:
    if isinstance(candidates, (str, bytes, bytearray)) or not isinstance(candidates, Sequence):
        raise RerankIntegrityError("rerank_candidates_invalid")
    raw_candidates = cast(Sequence[object], candidates)
    validated: list[RetrievalCandidate] = []
    seen: set[str] = set()
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, RetrievalCandidate):
            raise RerankIntegrityError("rerank_candidate_invalid")
        candidate = RetrievalCandidate.model_validate(raw_candidate.model_dump())
        if candidate.chunk_id in seen:
            raise RerankIntegrityError("rerank_duplicate_candidate_id")
        if candidate.reranking_rank is not None and not allow_reranked:
            raise RerankIntegrityError("rerank_candidate_already_ranked")
        seen.add(candidate.chunk_id)
        validated.append(candidate)
    return tuple(validated)


def _exact_permutation(order: object, expected: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(order, (str, bytes, bytearray)) or not isinstance(order, Sequence):
        raise RerankIntegrityError("rerank_order_invalid")
    if any(type(value) is not str for value in order):
        raise RerankIntegrityError("rerank_order_invalid")
    normalized = tuple(cast(Sequence[str], order))
    if (
        len(normalized) != len(expected)
        or len(set(normalized)) != len(normalized)
        or set(normalized) != set(expected)
    ):
        raise RerankIntegrityError("rerank_order_invalid")
    return normalized


def _apply_order(
    base: tuple[RetrievalCandidate, ...],
    submitted: tuple[RetrievalCandidate, ...],
    order: tuple[str, ...],
) -> tuple[RetrievalCandidate, ...]:
    registry = {candidate.chunk_id: candidate for candidate in submitted}
    ordered: list[RetrievalCandidate] = []
    for rank, candidate_id in enumerate(order, start=1):
        candidate = registry[candidate_id]
        ordered.append(
            RetrievalCandidate.model_validate({**candidate.model_dump(), "reranking_rank": rank})
        )
    ordered.extend(base[len(submitted) :])
    return tuple(ordered)


def _bounded_text(text: str, limit: int) -> str:
    if not isinstance(text, str) or not text.strip():
        raise RerankIntegrityError("rerank_text_invalid")
    bounded = text[:limit]
    if bounded.strip():
        return bounded
    return text.lstrip()[:limit]


def _provider_reason(category: ProviderErrorCategory) -> str:
    return f"rerank_provider_{category.value}"


def _direct_identity(
    reranker: RerankingProvider | ModelProviderRouter,
) -> ModelIdentity | None:
    if isinstance(reranker, ModelProviderRouter):
        return None
    try:
        identity = reranker.identity
    except Exception:
        return None
    return identity if isinstance(identity, ModelIdentity) else None


def _validated_attempts(attempts: object) -> tuple[ModelAttempt, ...]:
    if isinstance(attempts, (str, bytes, bytearray)) or not isinstance(attempts, Sequence):
        raise RerankIntegrityError("rerank_attempts_invalid")
    values = tuple(attempts)
    if any(not isinstance(attempt, ModelAttempt) for attempt in values):
        raise RerankIntegrityError("rerank_attempts_invalid")
    return cast(tuple[ModelAttempt, ...], values)


def _safe_route_id(result: object) -> str | None:
    route_id = getattr(result, "route_id", None)
    return route_id if isinstance(route_id, str) and route_id.strip() else None


def _safe_result_identity(result: object) -> ModelIdentity | None:
    identity = getattr(result, "identity", None)
    return identity if isinstance(identity, ModelIdentity) else None


def _safe_prompt_version(result: object) -> str | None:
    prompt_version = getattr(result, "prompt_version", None)
    return prompt_version if isinstance(prompt_version, str) and prompt_version.strip() else None


def _safe_usage(result: object) -> TokenUsage | None:
    usage = getattr(result, "usage", None)
    return usage if isinstance(usage, TokenUsage) else None


def _safe_attempts(result: object) -> tuple[ModelAttempt, ...]:
    try:
        return _validated_attempts(getattr(result, "attempts", ()))
    except RerankIntegrityError:
        return ()
