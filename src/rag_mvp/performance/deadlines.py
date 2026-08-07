"""Total QA deadlines, per-stage budgets, and optional-stage degradation."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from rag_mvp.config.settings import Settings
from rag_mvp.providers.models import Deadline


class DeadlineExceededError(TimeoutError):
    """The request's hard deadline no longer permits safe completion."""

    code = "deadline-expired"
    retryable = False

    def __init__(self, stage: str | None = None) -> None:
        self.stage = stage
        super().__init__(self.code)


class StageDeadlineExceededError(DeadlineExceededError):
    """A stage exhausted its own budget before the total deadline."""

    code = "stage-deadline-expired"

    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        self.args = (self.code,)


@dataclass(frozen=True, slots=True)
class QALatencyBudgets:
    """Configurable budgets for every measured QA stage.

    Retrieval stages may overlap, so their individual budgets are not summed. The
    hard request deadline always wins over a stage budget.
    """

    total_seconds: float = 9.5
    queue_seconds: float = 0.2
    validation_seconds: float = 0.8
    embedding_seconds: float = 0.8
    dense_retrieval_seconds: float = 0.8
    bm25_seconds: float = 0.8
    fusion_seconds: float = 0.2
    rerank_seconds: float = 3.0
    evidence_assessment_seconds: float = 4.0
    generation_seconds: float = 6.0
    grounding_seconds: float = 0.3
    redaction_seconds: float = 0.2
    serialization_seconds: float = 0.1
    finalization_seconds: float = 0.6

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not _positive_finite(value):
                raise ValueError(f"{name} must be positive and finite")
            if name != "total_seconds" and value > self.total_seconds:
                raise ValueError(f"{name} cannot exceed total_seconds")

    @classmethod
    def from_settings(cls, settings: Settings) -> QALatencyBudgets:
        """Resolve budgets from the existing runtime deadline settings."""

        if not isinstance(settings, Settings):
            raise TypeError("settings must be Settings")
        scale = settings.qa_deadline_seconds / 9.5

        def stage(field: str, default: float) -> float:
            value = float(getattr(settings, field))
            return value if field in settings.model_fields_set else default * scale

        return cls(
            total_seconds=settings.qa_deadline_seconds,
            queue_seconds=stage("qa_queue_budget_seconds", 0.2),
            validation_seconds=stage("qa_validation_budget_seconds", 0.8),
            embedding_seconds=stage("qa_embedding_budget_seconds", 0.8),
            dense_retrieval_seconds=stage("qa_dense_retrieval_budget_seconds", 0.8),
            bm25_seconds=stage("qa_bm25_budget_seconds", 0.8),
            fusion_seconds=stage("qa_fusion_budget_seconds", 0.2),
            rerank_seconds=settings.rerank_deadline_seconds,
            evidence_assessment_seconds=stage("qa_evidence_assessment_budget_seconds", 4.0),
            generation_seconds=stage("qa_generation_budget_seconds", 6.0),
            grounding_seconds=stage("qa_grounding_budget_seconds", 0.3),
            redaction_seconds=stage("qa_redaction_budget_seconds", 0.2),
            serialization_seconds=stage("qa_serialization_budget_seconds", 0.1),
            finalization_seconds=stage("qa_finalization_budget_seconds", 0.6),
        )

    def for_stage(self, stage: str) -> float:
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("stage must be non-empty")
        normalized = stage.strip().lower().replace("-", "_")
        aliases = {
            "admission": "queue",
            "dense": "dense_retrieval",
            "lexical": "bm25",
            "reranking": "rerank",
            "evidence": "evidence_assessment",
            "citation": "grounding",
        }
        normalized = aliases.get(normalized, normalized)
        attribute = f"{normalized}_seconds"
        value = getattr(self, attribute, None)
        if not isinstance(value, int | float):
            raise KeyError(stage)
        return float(value)


@dataclass(frozen=True, slots=True)
class OptionalStageResult[T]:
    value: T
    degraded: bool
    degradation_reason: str | None
    elapsed_ms: float

    def __post_init__(self) -> None:
        if self.degraded != (self.degradation_reason is not None):
            raise ValueError("degradation reason is inconsistent")
        if not math.isfinite(self.elapsed_ms) or self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be finite and non-negative")


class DeadlineController:
    """One monotonic hard deadline shared by all child stages."""

    def __init__(
        self,
        budgets: QALatencyBudgets | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        started_at: float | None = None,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.budgets = budgets or QALatencyBudgets()
        self._clock = clock
        self.started_at = clock() if started_at is None else started_at
        if not math.isfinite(self.started_at):
            raise ValueError("started_at must be finite")
        self.expires_at = self.started_at + self.budgets.total_seconds

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> DeadlineController:
        return cls(QALatencyBudgets.from_settings(settings), clock=clock)

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self.started_at)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.expires_at - self._clock())

    @property
    def expired(self) -> bool:
        return self.remaining_seconds <= 0

    @property
    def deadline(self) -> Deadline:
        return Deadline(self.expires_at, self._clock)

    def stage_timeout(
        self,
        stage: str,
        *,
        reserve_seconds: float = 0.0,
        require_full_budget: bool = False,
    ) -> float:
        _require_non_negative_finite(reserve_seconds, "reserve_seconds")
        stage_budget = self.budgets.for_stage(stage)
        available = self.remaining_seconds - reserve_seconds
        if available <= 0:
            raise DeadlineExceededError(stage)
        if require_full_budget and available < stage_budget:
            raise DeadlineExceededError(stage)
        return min(stage_budget, available)

    def child_deadline(
        self,
        stage: str,
        *,
        reserve_seconds: float = 0.0,
        require_full_budget: bool = False,
    ) -> Deadline:
        duration = self.stage_timeout(
            stage,
            reserve_seconds=reserve_seconds,
            require_full_budget=require_full_budget,
        )
        return Deadline(self._clock() + duration, self._clock)

    async def run_required[T](
        self,
        stage: str,
        operation: Callable[[], Awaitable[T]],
        *,
        reserve_seconds: float = 0.0,
        require_full_budget: bool = False,
    ) -> T:
        if not callable(operation):
            raise TypeError("operation must be callable")
        timeout_seconds = self.stage_timeout(
            stage,
            reserve_seconds=reserve_seconds,
            require_full_budget=require_full_budget,
        )
        try:
            async with asyncio.timeout(timeout_seconds):
                result = await operation()
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            if self.expired or self.remaining_seconds <= reserve_seconds:
                raise DeadlineExceededError(stage) from None
            raise StageDeadlineExceededError(stage) from None
        if self.expired:
            raise DeadlineExceededError(stage)
        return result

    async def run_with_deadline[T](
        self,
        stage: str,
        operation: Callable[[Deadline], Awaitable[T]],
        *,
        reserve_seconds: float = 0.0,
        require_full_budget: bool = False,
    ) -> T:
        if not callable(operation):
            raise TypeError("operation must be callable")
        child = self.child_deadline(
            stage,
            reserve_seconds=reserve_seconds,
            require_full_budget=require_full_budget,
        )
        return await self.run_required(
            stage,
            lambda: operation(child),
            reserve_seconds=reserve_seconds,
            require_full_budget=require_full_budget,
        )

    async def run_generation[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        """Start required generation only when its budget plus finalization fits."""

        return await self.run_required(
            "generation",
            operation,
            reserve_seconds=self.budgets.finalization_seconds,
            require_full_budget=True,
        )

    async def run_remaining[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        """Run required downstream work inside the remaining hard deadline."""

        if not callable(operation):
            raise TypeError("operation must be callable")
        timeout_seconds = self.remaining_seconds
        if timeout_seconds <= 0:
            raise DeadlineExceededError
        try:
            async with asyncio.timeout(timeout_seconds):
                result = await operation()
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            raise DeadlineExceededError from None
        if self.expired:
            raise DeadlineExceededError
        return result

    def run_sync_required[T](self, stage: str, operation: Callable[[], T]) -> T:
        """Fail closed when a short synchronous release stage overruns its budget."""

        if not callable(operation):
            raise TypeError("operation must be callable")
        timeout_seconds = self.stage_timeout(stage)
        started = self._clock()
        result = operation()
        elapsed = max(0.0, self._clock() - started)
        if self.expired:
            raise DeadlineExceededError(stage)
        if elapsed > timeout_seconds:
            raise StageDeadlineExceededError(stage)
        return result

    async def run_optional[T](
        self,
        stage: str,
        operation: Callable[[], Awaitable[T]],
        *,
        fallback: T,
        timeout_reason: str | None = None,
        failure_reason: str | None = None,
        reserve_seconds: float = 0.0,
    ) -> OptionalStageResult[T]:
        started = self._clock()
        prefix = _degradation_prefix(stage)
        try:
            value = await self.run_required(
                stage,
                operation,
                reserve_seconds=reserve_seconds,
            )
        except asyncio.CancelledError:
            raise
        except (DeadlineExceededError, StageDeadlineExceededError):
            return OptionalStageResult(
                fallback,
                True,
                timeout_reason or f"{prefix}_timeout",
                self._elapsed_ms(started),
            )
        except Exception:
            return OptionalStageResult(
                fallback,
                True,
                failure_reason or f"{prefix}_failure",
                self._elapsed_ms(started),
            )
        return OptionalStageResult(value, False, None, self._elapsed_ms(started))

    async def run_optional_reranker[T](
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        base_ranking: T,
    ) -> OptionalStageResult[T]:
        return await self.run_optional(
            "rerank",
            operation,
            fallback=base_ranking,
            timeout_reason="rerank_timeout",
            failure_reason="rerank_failure",
        )

    def ensure_remaining(self, required_seconds: float = 0.0) -> None:
        _require_non_negative_finite(required_seconds, "required_seconds")
        if self.remaining_seconds <= required_seconds:
            raise DeadlineExceededError

    def _elapsed_ms(self, started: float) -> float:
        return max(0.0, (self._clock() - started) * 1000)


StageBudgets = QALatencyBudgets
RequestDeadline = DeadlineController


def _positive_finite(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _require_non_negative_finite(value: object, name: str) -> None:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative")


def _degradation_prefix(stage: str) -> str:
    normalized = stage.strip().lower().replace("-", "_")
    return "rerank" if normalized in {"rerank", "reranking"} else normalized
