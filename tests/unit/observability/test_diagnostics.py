from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rag_mvp.domain.qa import RequestDiagnostic
from rag_mvp.observability.diagnostics import (
    DiagnosticRetention,
    DiagnosticSafetyError,
    SafeRequestDiagnosticStore,
)
from rag_mvp.storage.database import Database


@pytest.fixture
def database(tmp_path: Path) -> Database:
    value = Database(tmp_path / "metadata.sqlite3")
    value.initialize()
    return value


def _diagnostic(request_id: str, created_at: datetime) -> RequestDiagnostic:
    return RequestDiagnostic(
        request_id=request_id,
        session_id="session-1",
        trace_id="trace-1",
        outcome="answer",
        stage_timings_ms={"generation": 12.0, "raw-question": 99.0},
        cache_status={"retrieval": "bypass", "raw-question": "hit"},
        model_identities={
            "generation": "model person@example.com",
            "raw-question": "secret text",
        },
        token_counts={"input": 10, "raw-question": 20},
        metadata={
            "configuration_id": "config person@example.com",
            "question": "person@example.com",
        },
        created_at=created_at,
    )


def test_diagnostics_are_redacted_allowlisted_and_persisted(database: Database) -> None:
    now = datetime(2026, 8, 7, tzinfo=UTC)
    store = SafeRequestDiagnosticStore(database)
    saved = store.save(_diagnostic("request-1", now), now=now)

    assert saved.stage_timings_ms == {"generation": 12.0}
    assert saved.cache_status == {"retrieval": "bypass"}
    assert saved.model_identities == {"generation": "model [REDACTED_EMAIL]"}
    assert saved.token_counts == {"input": 10}
    assert saved.metadata == {"configuration_id": "config [REDACTED_EMAIL]"}
    assert saved.expires_at == now + timedelta(hours=24)
    assert "person@example.com" not in saved.model_dump_json()

    reopened = SafeRequestDiagnosticStore(Database(database.path))
    assert reopened.get("request-1", now=now) == saved
    assert reopened.get("person@example.com", now=now) is None


def test_refusal_guidance_telemetry_keeps_only_stable_safe_fields(
    database: Database,
) -> None:
    now = datetime(2026, 8, 7, tzinfo=UTC)
    unsafe_fixture = "person@example.com"
    diagnostic = _diagnostic("request-guided-refusal", now).model_copy(
        update={
            "outcome": "refusal",
            "metadata": {
                "refusal_reason_code": "prompt-injection",
                "refusal_guidance_reason_code": "prompt-injection",
                "refusal_guidance_template_id": ("refusal-guidance-v1.prompt-injection.en"),
                "refusal_guidance_catalog_version": "refusal-guidance-v1",
                "refusal_guidance_present": True,
                "refusal_guidance_language": "en",
                "input_policy": "override_policy",
                "question": f"raw trigger {unsafe_fixture}",
            },
        }
    )

    saved = SafeRequestDiagnosticStore(database).save(diagnostic, now=now)

    assert saved.metadata == {
        "refusal_reason_code": "prompt-injection",
        "refusal_guidance_reason_code": "prompt-injection",
        "refusal_guidance_template_id": "refusal-guidance-v1.prompt-injection.en",
        "refusal_guidance_catalog_version": "refusal-guidance-v1",
        "refusal_guidance_present": True,
        "refusal_guidance_language": "en",
    }
    assert "input_policy" not in saved.metadata
    assert unsafe_fixture not in saved.model_dump_json()


def test_retention_bounds_count_and_expiry(database: Database) -> None:
    now = datetime(2026, 8, 7, tzinfo=UTC)
    store = SafeRequestDiagnosticStore(
        database,
        retention=DiagnosticRetention(max_entries=2, ttl=timedelta(minutes=5)),
    )
    for offset in range(3):
        created = now + timedelta(seconds=offset)
        store.save(_diagnostic(f"request-{offset}", created), now=created)

    assert store.count() == 2
    assert store.get("request-0", now=now + timedelta(seconds=3)) is None
    assert store.get("request-1", now=now + timedelta(seconds=3)) is not None
    assert store.purge(now=now + timedelta(minutes=6)) == 2
    assert store.count() == 0


def test_diagnostics_fail_closed_without_complete_redaction(database: Database) -> None:
    store = SafeRequestDiagnosticStore(database, redactor=None)
    now = datetime(2026, 8, 7, tzinfo=UTC)

    with pytest.raises(DiagnosticSafetyError, match="redaction is unavailable"):
        store.save(_diagnostic("request-1", now), now=now)
    assert store.count() == 0


def test_duplicate_request_id_cannot_overwrite_diagnostics(database: Database) -> None:
    store = SafeRequestDiagnosticStore(database)
    now = datetime(2026, 8, 7, tzinfo=UTC)
    original = store.save(_diagnostic("request-1", now), now=now)

    with pytest.raises(DiagnosticSafetyError, match="already been persisted"):
        store.save(_diagnostic("request-1", now + timedelta(seconds=1)))

    assert store.get("request-1", now=now) == original
