"""Persist content-free provider-attempt accounting."""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from rag_mvp.domain.evaluation import ModelAttempt as PersistedModelAttempt
from rag_mvp.domain.evaluation import (
    ModelAttemptStatus,
    ModelRole,
    TokenUsage,
)
from rag_mvp.providers.models import ModelAttempt, ProviderErrorCategory
from rag_mvp.storage.repositories import ProviderUsageRepository


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
