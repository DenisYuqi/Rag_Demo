"""Persist content-free provider-attempt accounting."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

from rag_mvp.domain.evaluation import ModelAttempt as PersistedModelAttempt
from rag_mvp.domain.evaluation import (
    ModelAttemptStatus,
    ModelRole,
    TokenUsage,
)
from rag_mvp.providers.models import ModelAttempt, ProviderErrorCategory
from rag_mvp.storage.repositories import ProviderUsageRepository

_EVALUATION_RUN_ID: ContextVar[str | None] = ContextVar(
    "rag_mvp_evaluation_run_id",
    default=None,
)
_SAFE_EVALUATION_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$")


@contextmanager
def evaluation_run_attempt_context(run_id: str) -> Iterator[None]:
    """Bind provider attempts to one immutable evaluation run in this async context."""

    if _SAFE_EVALUATION_RUN_ID.fullmatch(run_id) is None:
        raise ValueError("evaluation_run_context_invalid")
    token = _EVALUATION_RUN_ID.set(run_id)
    try:
        yield
    finally:
        _EVALUATION_RUN_ID.reset(token)


@contextmanager
def unbound_evaluation_attempt_context() -> Iterator[None]:
    """Explicitly mark shared setup work as not belonging to any evaluation run."""

    token = _EVALUATION_RUN_ID.set(None)
    try:
        yield
    finally:
        _EVALUATION_RUN_ID.reset(token)


def current_evaluation_run_id() -> str | None:
    """Return the current run binding without process-global mutable state."""

    return _EVALUATION_RUN_ID.get()


class PersistentAttemptRecorder:
    """Map provider-layer attempts to the persisted diagnostics contract."""

    def __init__(
        self,
        repository: ProviderUsageRepository,
        *,
        attempt_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._repository = repository
        self._attempt_id_factory = attempt_id_factory or (lambda: f"attempt_{uuid4().hex}")

    def record(self, attempt: ModelAttempt) -> None:
        status = ModelAttemptStatus(attempt.status.value)
        if attempt.error_category in {
            ProviderErrorCategory.TIMEOUT,
            ProviderErrorCategory.DEADLINE_EXCEEDED,
        }:
            status = ModelAttemptStatus.TIMED_OUT
        self._repository.record(
            PersistedModelAttempt(
                attempt_id=self._attempt_id_factory(),
                operation_id=attempt.operation_id,
                request_id=attempt.request_id,
                run_id=current_evaluation_run_id(),
                role=ModelRole(attempt.role.value),
                provider=attempt.provider,
                model=attempt.model,
                status=status,
                attempt_number=attempt.attempt_number,
                fallback=attempt.is_fallback,
                latency_ms=attempt.latency_ms,
                usage=TokenUsage(
                    input_tokens=attempt.usage.input_tokens,
                    output_tokens=attempt.usage.output_tokens,
                    total_tokens_reported=attempt.usage.total_tokens,
                ),
                safe_error_category=(
                    attempt.error_category.value if attempt.error_category is not None else None
                ),
            )
        )


__all__ = [
    "PersistentAttemptRecorder",
    "current_evaluation_run_id",
    "evaluation_run_attempt_context",
    "unbound_evaluation_attempt_context",
]
