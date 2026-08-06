from __future__ import annotations

from pathlib import Path

from rag_mvp.domain.evaluation import ModelAttemptStatus
from rag_mvp.providers.models import (
    AttemptStatus,
    ModelAttempt,
    ProviderErrorCategory,
    ProviderRole,
    TokenUsage,
)
from rag_mvp.providers.persistence import PersistentAttemptRecorder
from rag_mvp.storage.database import Database
from rag_mvp.storage.repositories import ProviderUsageRepository


def test_provider_attempts_are_mapped_to_safe_persistent_usage(tmp_path: Path) -> None:
    database = Database(tmp_path / "metadata.sqlite3")
    database.initialize()
    repository = ProviderUsageRepository(database)
    attempt_ids = iter(("attempt-success", "attempt-timeout"))
    recorder = PersistentAttemptRecorder(
        repository,
        attempt_id_factory=lambda: next(attempt_ids),
    )

    recorder.record(
        ModelAttempt(
            request_id="request-1",
            operation_id="generation-1",
            attempt_number=1,
            route_id="openai-generation",
            role=ProviderRole.GENERATION,
            provider="openai",
            model="model-1",
            latency_ms=12.5,
            status=AttemptStatus.SUCCEEDED,
            is_fallback=False,
            usage=TokenUsage(input_tokens=10, output_tokens=4),
        )
    )
    recorder.record(
        ModelAttempt(
            request_id="request-1",
            operation_id="generation-1",
            attempt_number=2,
            route_id="openai-generation",
            role=ProviderRole.GENERATION,
            provider="openai",
            model="model-1",
            latency_ms=25,
            status=AttemptStatus.FAILED,
            is_fallback=False,
            error_category=ProviderErrorCategory.TIMEOUT,
        )
    )

    attempts = repository.list_for_request("request-1")
    assert [attempt.status for attempt in attempts] == [
        ModelAttemptStatus.SUCCEEDED,
        ModelAttemptStatus.TIMED_OUT,
    ]
    assert attempts[0].usage.known_total == 14
    assert attempts[1].usage.known_total is None
    assert attempts[1].safe_error_category == "timeout"
