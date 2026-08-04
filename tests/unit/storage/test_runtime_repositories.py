from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from rag_mvp.domain.evaluation import (
    EvaluationRun,
    EvaluationRunStatus,
    ModelAttempt,
    ModelAttemptStatus,
    ModelRole,
    ReportManifest,
    TokenUsage,
)
from rag_mvp.domain.qa import (
    ConversationRole,
    ConversationSession,
    ConversationTurn,
    RequestDiagnostic,
    SessionStatus,
)
from rag_mvp.storage.database import Database
from rag_mvp.storage.repositories import (
    RepositoryConflict,
    RuntimeRepositories,
    SessionOwnershipError,
)


@pytest.fixture
def database(tmp_path: Path) -> Database:
    value = Database(tmp_path / "metadata.sqlite3")
    value.initialize()
    return value


def _run() -> EvaluationRun:
    return EvaluationRun(
        run_id="run-1",
        dataset_id="mvp",
        dataset_version="1.0.0",
        dataset_hash="dataset-123",
        corpus_version="corpus-v1",
        configuration_id="config-v1",
        code_revision="code-v1",
        scorer_versions={"faithfulness": "v1"},
        cache_policy="acceptance-bypass",
        total_cases=10,
    )


def test_session_turns_owner_check_and_reset(database: Database) -> None:
    repositories = RuntimeRepositories.from_database(database)
    session = ConversationSession(session_id="session-1", owner_id="owner-a")
    repositories.sessions.create(session)
    repositories.sessions.append_turn(
        ConversationTurn(
            turn_id="turn-1",
            session_id="session-1",
            ordinal=0,
            role=ConversationRole.USER,
            content="What is the policy?",
        )
    )

    assert repositories.sessions.require_owned("session-1", "owner-a") == session
    with pytest.raises(SessionOwnershipError):
        repositories.sessions.require_owned("session-1", "owner-b")
    with pytest.raises(RepositoryConflict):
        repositories.sessions.append_turn(
            ConversationTurn(
                turn_id="turn-2",
                session_id="session-1",
                ordinal=2,
                role=ConversationRole.USER,
                content="Skipped an ordinal",
            )
        )

    reset = repositories.sessions.reset("session-1")
    assert reset.status is SessionStatus.RESET
    assert repositories.sessions.list_turns("session-1") == []
    assert len(repositories.sessions.list_turns("session-1", include_reset_history=True)) == 1


def test_request_diagnostics_respect_expiry(database: Database) -> None:
    repositories = RuntimeRepositories.from_database(database)
    now = datetime(2026, 8, 4, tzinfo=UTC)
    expired = RequestDiagnostic(
        request_id="request-expired",
        outcome="answer",
        expires_at=now - timedelta(seconds=1),
    )
    current = RequestDiagnostic(
        request_id="request-current",
        outcome="refusal",
        safe_error_category="insufficient-evidence",
        expires_at=now + timedelta(hours=1),
    )
    repositories.request_diagnostics.save(expired)
    repositories.request_diagnostics.save(current)

    assert repositories.request_diagnostics.get("request-expired", now=now) is None
    assert repositories.request_diagnostics.get("request-current", now=now) == current
    assert repositories.request_diagnostics.purge_expired(now=now) == 1


def test_provider_usage_records_every_attempt_and_unknown_usage(database: Database) -> None:
    repositories = RuntimeRepositories.from_database(database)
    failed = ModelAttempt(
        attempt_id="attempt-1",
        operation_id="operation-1",
        request_id="request-1",
        role=ModelRole.GENERATION,
        provider="primary",
        model="chat-v1",
        status=ModelAttemptStatus.TIMED_OUT,
        latency_ms=900,
        usage=TokenUsage(),
        safe_error_category="timeout",
    )
    fallback = ModelAttempt(
        attempt_id="attempt-2",
        operation_id="operation-1",
        request_id="request-1",
        role=ModelRole.GENERATION,
        provider="secondary",
        model="chat-v1",
        status=ModelAttemptStatus.SUCCEEDED,
        attempt_number=2,
        fallback=True,
        latency_ms=100,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        estimated_cost=Decimal("0.001"),
        currency="USD",
    )
    repositories.provider_usage.record(failed)
    repositories.provider_usage.record(fallback)

    attempts = repositories.provider_usage.list_for_request("request-1")
    assert {attempt.attempt_id for attempt in attempts} == {"attempt-1", "attempt-2"}
    assert attempts[0].usage.known_total is None
    assert repositories.provider_usage.get("attempt-2") == fallback


def test_evaluation_run_and_report_manifest_survive_reopen(database: Database) -> None:
    repositories = RuntimeRepositories.from_database(database)
    run = _run()
    repositories.evaluation_runs.create(run)
    running = EvaluationRun.model_validate(
        {
            **run.model_dump(),
            "status": EvaluationRunStatus.RUNNING,
            "completed_cases": 2,
            "updated_at": datetime(2026, 8, 4, 2, tzinfo=UTC),
        }
    )
    repositories.evaluation_runs.update(running)
    manifest = ReportManifest(
        run_id=run.run_id,
        schema_version="v1",
        json_report_path="reports/run-1/report.json",
        html_report_path="reports/run-1/report.html",
        content_hash="report-123",
    )
    repositories.report_manifests.save(manifest)

    reopened = RuntimeRepositories.from_database(Database(database.path))

    assert reopened.evaluation_runs.get("run-1") == running
    assert reopened.report_manifests.get("run-1") == manifest
