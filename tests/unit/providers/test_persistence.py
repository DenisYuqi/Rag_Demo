from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rag_mvp.domain.evaluation import ModelAttemptStatus
from rag_mvp.providers.models import (
    AttemptStatus,
    ModelAttempt,
    ProviderErrorCategory,
    ProviderRole,
    TokenUsage,
)
from rag_mvp.providers.persistence import (
    PersistentAttemptRecorder,
    current_evaluation_run_id,
    evaluation_run_attempt_context,
    unbound_evaluation_attempt_context,
)
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

    with evaluation_run_attempt_context("evaluation-run-1"):
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
    assert attempts[0].run_id == "evaluation-run-1"
    assert attempts[1].usage.known_total is None
    assert attempts[1].safe_error_category == "timeout"


@pytest.mark.parametrize("unsafe_run_id", ["../foreign", "run id", "run\nforeign"])
def test_evaluation_run_attempt_context_rejects_unsafe_ids(unsafe_run_id: str) -> None:
    with (
        pytest.raises(ValueError, match="evaluation_run_context_invalid"),
        evaluation_run_attempt_context(unsafe_run_id),
    ):
        raise AssertionError("unsafe context was entered")


def test_unbound_setup_context_restores_nested_evaluation_binding() -> None:
    assert current_evaluation_run_id() is None
    with evaluation_run_attempt_context("evaluation-run-1"):
        assert current_evaluation_run_id() == "evaluation-run-1"
        with unbound_evaluation_attempt_context():
            assert current_evaluation_run_id() is None
            with evaluation_run_attempt_context("evaluation-run-2"):
                assert current_evaluation_run_id() == "evaluation-run-2"
            assert current_evaluation_run_id() is None
        assert current_evaluation_run_id() == "evaluation-run-1"
    assert current_evaluation_run_id() is None


@pytest.mark.asyncio
async def test_unbound_setup_context_is_concurrently_isolated() -> None:
    async def observe(run_id: str) -> tuple[str | None, str | None, str | None]:
        with evaluation_run_attempt_context(run_id):
            before = current_evaluation_run_id()
            await asyncio.sleep(0)
            with unbound_evaluation_attempt_context():
                during = current_evaluation_run_id()
                await asyncio.sleep(0)
            after = current_evaluation_run_id()
        return before, during, after

    observed = await asyncio.gather(observe("evaluation-a"), observe("evaluation-b"))
    assert tuple(observed) == (
        ("evaluation-a", None, "evaluation-a"),
        ("evaluation-b", None, "evaluation-b"),
    )
    assert current_evaluation_run_id() is None
