from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_mvp.domain.ingestion import (
    Document,
    DocumentKind,
    IngestionJob,
    IngestionJobStatus,
    IngestionStage,
)
from rag_mvp.storage.database import Database
from rag_mvp.storage.repositories import (
    DocumentRepository,
    IngestionJobRepository,
    RepositoryConflict,
)

CREATED_AT = datetime(2026, 8, 6, 9, tzinfo=UTC)


@pytest.fixture
def repository(tmp_path: Path) -> IngestionJobRepository:
    database = Database(tmp_path / "metadata.sqlite3")
    database.initialize()
    _create_source(database)
    return IngestionJobRepository(database)


def _queued() -> IngestionJob:
    return IngestionJob(
        job_id="job-1",
        source_key="employee-handbook",
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
    )


def _create_source(database: Database) -> None:
    DocumentRepository(database).create(
        Document(
            source_id="src_1",
            source_key="registered-source",
            display_title="Registered Source",
            media_type="text/plain",
            kind=DocumentKind.TEXT,
            created_at=CREATED_AT,
            updated_at=CREATED_AT,
        )
    )


def _replace(job: IngestionJob, **changes: object) -> IngestionJob:
    return IngestionJob.model_validate({**job.model_dump(), **changes})


def test_job_transitions_merge_safe_diagnostics_and_survive_reopen(tmp_path: Path) -> None:
    database = Database(tmp_path / "metadata.sqlite3")
    database.initialize()
    _create_source(database)
    jobs = IngestionJobRepository(database)
    queued = _queued()
    jobs.create(queued)

    validating = jobs.transition(
        _replace(
            queued,
            status=IngestionJobStatus.PROCESSING,
            stage=IngestionStage.VALIDATING,
            stage_timings_ms={"validating": 4.5},
            warnings=("mime_fallback",),
        ),
        updated_at=CREATED_AT + timedelta(seconds=1),
    )
    extracting = jobs.transition(
        _replace(
            validating,
            stage=IngestionStage.EXTRACTING,
            source_id="src_1",
            stage_timings_ms={"extracting": 12.0},
            warnings=("ocr_partial",),
        ),
        updated_at=CREATED_AT + timedelta(seconds=2),
    )
    failed = jobs.transition(
        _replace(
            extracting,
            status=IngestionJobStatus.FAILED,
            stage=IngestionStage.FAILED,
            safe_error_code="ocr_unavailable",
        ),
        updated_at=CREATED_AT + timedelta(seconds=3),
    )

    assert failed.failed_stage is IngestionStage.EXTRACTING
    assert failed.stage_timings_ms == {"validating": 4.5, "extracting": 12.0}
    assert failed.warnings == ("mime_fallback", "ocr_partial")
    assert jobs.list_nonterminal() == []

    reopened = IngestionJobRepository(Database(database.path))
    assert reopened.get(failed.job_id) == failed
    assert reopened.transition(failed) == failed


def test_processing_stage_cannot_regress_and_failed_update_rolls_back(
    repository: IngestionJobRepository,
) -> None:
    queued = _queued()
    repository.create(queued)
    extracting = repository.transition(
        _replace(
            queued,
            status=IngestionJobStatus.PROCESSING,
            stage=IngestionStage.EXTRACTING,
        ),
        updated_at=CREATED_AT + timedelta(seconds=1),
    )

    with pytest.raises(RepositoryConflict, match="cannot regress"):
        repository.transition(
            _replace(extracting, stage=IngestionStage.VALIDATING),
            updated_at=CREATED_AT + timedelta(seconds=2),
        )

    assert repository.get(queued.job_id) == extracting


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_key": "other-source"}, "identity fields"),
        ({"created_at": CREATED_AT - timedelta(seconds=1)}, "identity fields"),
        ({"source_id": None}, "source_id cannot be cleared"),
        ({"document_version": None}, "document_version cannot be cleared"),
        ({"active_index_revision": None}, "active_index_revision cannot be cleared"),
        ({"ocr_page_count": 1}, "ocr_page_count cannot decrease"),
        ({"chunk_count": 2}, "chunk_count cannot decrease"),
    ],
)
def test_assigned_results_and_job_identity_cannot_be_cleared_or_changed(
    repository: IngestionJobRepository,
    changes: dict[str, object],
    message: str,
) -> None:
    queued = _queued()
    repository.create(queued)
    processing = repository.transition(
        _replace(
            queued,
            status=IngestionJobStatus.PROCESSING,
            stage=IngestionStage.INDEXING,
            source_id="src_1",
            document_version=2,
            active_index_revision="rev_1",
            ocr_page_count=3,
            chunk_count=5,
        ),
        updated_at=CREATED_AT + timedelta(seconds=1),
    )

    with pytest.raises(RepositoryConflict, match=message):
        repository.transition(
            _replace(processing, **changes),
            updated_at=CREATED_AT + timedelta(seconds=2),
        )


def test_terminal_job_is_idempotent_but_immutable(repository: IngestionJobRepository) -> None:
    queued = _queued()
    repository.create(queued)
    processing = repository.transition(
        _replace(
            queued,
            status=IngestionJobStatus.PROCESSING,
            stage=IngestionStage.PUBLISHING,
            source_id="src_1",
            document_version=1,
        ),
        updated_at=CREATED_AT + timedelta(seconds=1),
    )
    succeeded = repository.transition(
        _replace(
            processing,
            status=IngestionJobStatus.SUCCEEDED,
            stage=IngestionStage.COMPLETE,
            active_index_revision="rev_1",
        ),
        updated_at=CREATED_AT + timedelta(seconds=2),
    )

    assert repository.transition(succeeded) == succeeded
    with pytest.raises(RepositoryConflict, match="terminal ingestion jobs are immutable"):
        repository.transition(_replace(succeeded, warnings=("late_warning",)))


def test_updated_at_must_advance_when_explicit(repository: IngestionJobRepository) -> None:
    queued = _queued()
    repository.create(queued)

    with pytest.raises(RepositoryConflict, match="must advance"):
        repository.transition(
            _replace(
                queued,
                status=IngestionJobStatus.PROCESSING,
                stage=IngestionStage.VALIDATING,
            ),
            updated_at=CREATED_AT,
        )


@pytest.mark.parametrize(
    "values",
    [
        {"status": IngestionJobStatus.QUEUED, "stage": IngestionStage.EXTRACTING},
        {"status": IngestionJobStatus.PROCESSING, "stage": IngestionStage.QUEUED},
        {
            "status": IngestionJobStatus.SUCCEEDED,
            "stage": IngestionStage.PUBLISHING,
        },
        {
            "status": IngestionJobStatus.FAILED,
            "stage": IngestionStage.EXTRACTING,
            "safe_error_code": "extract_failed",
        },
        {"stage_timings_ms": {"document_text": 1.0}},
        {"warnings": ("contains document text",)},
        {
            "status": IngestionJobStatus.FAILED,
            "stage": IngestionStage.FAILED,
            "safe_error_code": "unsafe/value",
        },
    ],
)
def test_job_model_rejects_invalid_states_and_unbounded_diagnostics(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        IngestionJob.model_validate(
            {
                "job_id": "job-1",
                "source_key": "employee-handbook",
                **values,
            }
        )


def test_persisted_failed_payload_without_failed_stage_remains_readable(tmp_path: Path) -> None:
    database = Database(tmp_path / "metadata.sqlite3")
    database.initialize()
    failed = IngestionJob(
        job_id="legacy-job",
        source_key="legacy-source",
        status=IngestionJobStatus.FAILED,
        stage=IngestionStage.FAILED,
        safe_error_code="legacy_failure",
    )
    payload = failed.model_dump(mode="json")
    payload.pop("failed_stage")
    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO ingestion_jobs(job_id, source_key, source_id, status, payload_json)
            VALUES (?, ?, NULL, ?, ?)
            """,
            (
                failed.job_id,
                failed.source_key,
                failed.status.value,
                json.dumps(payload),
            ),
        )

    assert IngestionJobRepository(database).get(failed.job_id) == failed
