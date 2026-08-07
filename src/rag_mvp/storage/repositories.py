"""SQLite repositories for knowledge and runtime metadata."""

from __future__ import annotations

import builtins
import hashlib
import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_mvp.domain.evaluation import EvaluationRun, ModelAttempt, ReportManifest
from rag_mvp.domain.ingestion import (
    PROCESSING_INGESTION_STAGES,
    Document,
    DocumentVersion,
    IndexRevision,
    IndexRevisionStatus,
    IngestionJob,
    IngestionJobStatus,
    IngestionStage,
    ParentChunk,
)
from rag_mvp.domain.qa import (
    ConversationSession,
    ConversationTurn,
    RequestDiagnostic,
    SessionStatus,
)
from rag_mvp.evaluation.json_report import canonical_json_value
from rag_mvp.storage.database import Database

if TYPE_CHECKING:
    from rag_mvp.evaluation.comparison import (
        ComparisonCandidateHistory,
        ComparisonCandidateReference,
        ComparisonResult,
        ComparisonSharedSetupEvidence,
        ComparisonSuite,
    )
    from rag_mvp.evaluation.experiment import ExperimentAxis


class RepositoryError(RuntimeError):
    """Base class for normalized persistence failures."""


class RepositoryConflict(RepositoryError):
    """Raised when a stable identifier or unique domain value already exists."""


class RepositoryNotFound(RepositoryError):
    """Raised when a requested persistent entity does not exist."""


class SessionOwnershipError(RepositoryError):
    """Raised when an owner attempts to access another owner's session."""


_EXPECTED_ACTIVE_REVISION_UNSET = object()


def _now() -> datetime:
    return datetime.now(UTC)


def _decode[TModel: BaseModel](model_type: type[TModel], row: sqlite3.Row) -> TModel:
    payload = cast(str, row["payload_json"])
    return model_type.model_validate_json(payload)


@contextmanager
def _read_connection(
    database: Database,
    connection: sqlite3.Connection | None,
) -> Iterator[sqlite3.Connection]:
    if connection is not None:
        yield connection
        return
    with database.connection() as opened:
        yield opened


@contextmanager
def _write_connection(
    database: Database,
    connection: sqlite3.Connection | None,
) -> Iterator[sqlite3.Connection]:
    if connection is not None:
        yield connection
        return
    with database.transaction() as opened:
        yield opened


def _raise_conflict(entity: str, error: sqlite3.IntegrityError) -> None:
    raise RepositoryConflict(f"{entity} conflicts with an existing record") from error


class DocumentRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def create(
        self,
        document: Document,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        try:
            with _write_connection(self._database, connection) as active:
                active.execute(
                    """
                    INSERT INTO documents(
                        source_id, source_key, active_version, deleted_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        document.source_id,
                        document.source_key,
                        document.active_version,
                        document.deleted_at.isoformat() if document.deleted_at else None,
                        document.model_dump_json(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            _raise_conflict("document", error)

    def update(
        self,
        document: Document,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        with _write_connection(self._database, connection) as active:
            cursor = active.execute(
                """
                UPDATE documents
                SET source_key = ?, active_version = ?, deleted_at = ?, payload_json = ?
                WHERE source_id = ?
                """,
                (
                    document.source_key,
                    document.active_version,
                    document.deleted_at.isoformat() if document.deleted_at else None,
                    document.model_dump_json(),
                    document.source_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RepositoryNotFound(f"document {document.source_id!r} was not found")

    def get(
        self,
        source_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> Document | None:
        with _read_connection(self._database, connection) as active:
            row = active.execute(
                "SELECT payload_json FROM documents WHERE source_id = ?",
                (source_id,),
            ).fetchone()
        return None if row is None else _decode(Document, row)

    def get_by_source_key(
        self,
        source_key: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> Document | None:
        with _read_connection(self._database, connection) as active:
            row = active.execute(
                "SELECT payload_json FROM documents WHERE source_key = ?",
                (source_key,),
            ).fetchone()
        return None if row is None else _decode(Document, row)

    def list(
        self,
        *,
        include_deleted: bool = False,
        connection: sqlite3.Connection | None = None,
    ) -> list[Document]:
        query = "SELECT payload_json FROM documents"
        if not include_deleted:
            query += " WHERE deleted_at IS NULL"
        query += " ORDER BY source_id"
        with _read_connection(self._database, connection) as active:
            rows = active.execute(query).fetchall()
        return [_decode(Document, row) for row in rows]

    def list_active(
        self,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> builtins.list[Document]:
        with _read_connection(self._database, connection) as active:
            rows = active.execute(
                """
                SELECT payload_json FROM documents
                WHERE deleted_at IS NULL AND active_version IS NOT NULL
                ORDER BY source_id
                """
            ).fetchall()
        return [_decode(Document, row) for row in rows]

    def get_active_mapping(
        self,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> dict[str, int]:
        with _read_connection(self._database, connection) as active:
            rows = active.execute(
                """
                SELECT source_id, active_version FROM documents
                WHERE deleted_at IS NULL AND active_version IS NOT NULL
                ORDER BY source_id
                """
            ).fetchall()
        mapping: dict[str, int] = {}
        for row in rows:
            source_id = row["source_id"]
            version = row["active_version"]
            if not isinstance(source_id, str) or not source_id or type(version) is not int:
                raise RepositoryError("active document mapping is invalid")
            mapping[source_id] = version
        return mapping

    def add_version(
        self,
        version: DocumentVersion,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        try:
            with _write_connection(self._database, connection) as active:
                active.execute(
                    """
                    INSERT INTO document_versions(
                        source_id, version, content_digest, derivation_config_digest, payload_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        version.source_id,
                        version.version,
                        version.content_digest,
                        version.derivation_config_digest,
                        version.model_dump_json(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            _raise_conflict("document version", error)

    def get_version(
        self,
        source_id: str,
        version: int,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> DocumentVersion | None:
        with _read_connection(self._database, connection) as active:
            row = active.execute(
                """
                SELECT payload_json FROM document_versions
                WHERE source_id = ? AND version = ?
                """,
                (source_id, version),
            ).fetchone()
        return None if row is None else _decode(DocumentVersion, row)

    def list_versions(
        self,
        source_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> builtins.list[DocumentVersion]:
        with _read_connection(self._database, connection) as active:
            rows = active.execute(
                """
                SELECT payload_json FROM document_versions
                WHERE source_id = ? ORDER BY version
                """,
                (source_id,),
            ).fetchall()
        return [_decode(DocumentVersion, row) for row in rows]

    def get_latest_version(
        self,
        source_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> DocumentVersion | None:
        with _read_connection(self._database, connection) as active:
            row = active.execute(
                """
                SELECT payload_json FROM document_versions
                WHERE source_id = ? ORDER BY version DESC LIMIT 1
                """,
                (source_id,),
            ).fetchone()
        return None if row is None else _decode(DocumentVersion, row)

    def latest_version(
        self,
        source_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> DocumentVersion | None:
        return self.get_latest_version(source_id, connection=connection)

    def next_version(
        self,
        source_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> int:
        with _read_connection(self._database, connection) as active:
            row = active.execute(
                """
                SELECT COALESCE(MAX(version), 0) + 1 AS next_version
                FROM document_versions WHERE source_id = ?
                """,
                (source_id,),
            ).fetchone()
        return 1 if row is None else int(row["next_version"])

    def set_active_version(
        self,
        source_id: str,
        version: int,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> Document:
        with _write_connection(self._database, connection) as active:
            document = self.get(source_id, connection=active)
            if document is None:
                raise RepositoryNotFound(f"document {source_id!r} was not found")
            if self.get_version(source_id, version, connection=active) is None:
                raise RepositoryNotFound(f"document version {source_id!r}/{version} was not found")
            updated = Document.model_validate(
                {
                    **document.model_dump(),
                    "active_version": version,
                    "updated_at": _now(),
                    "deleted_at": None,
                }
            )
            self.update(updated, connection=active)
            return updated

    def mark_deleted(
        self,
        source_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> Document:
        with _write_connection(self._database, connection) as active:
            document = self.get(source_id, connection=active)
            if document is None:
                raise RepositoryNotFound(f"document {source_id!r} was not found")
            timestamp = _now()
            updated = Document.model_validate(
                {
                    **document.model_dump(),
                    "active_version": None,
                    "updated_at": timestamp,
                    "deleted_at": timestamp,
                }
            )
            self.update(updated, connection=active)
            return updated

    def delete_permanently(
        self,
        source_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> bool:
        """Physically remove one document and all of its version metadata."""

        with _write_connection(self._database, connection) as active:
            document = self.get(source_id, connection=active)
            if document is None:
                return False
            active.execute(
                "DELETE FROM document_versions WHERE source_id = ?",
                (source_id,),
            )
            cursor = active.execute(
                "DELETE FROM documents WHERE source_id = ?",
                (source_id,),
            )
            return cursor.rowcount == 1


class IngestionJobRepository:
    _ALLOWED_TRANSITIONS: ClassVar[dict[IngestionJobStatus, frozenset[IngestionJobStatus]]] = {
        IngestionJobStatus.QUEUED: frozenset(
            {IngestionJobStatus.QUEUED, IngestionJobStatus.PROCESSING, IngestionJobStatus.FAILED}
        ),
        IngestionJobStatus.PROCESSING: frozenset(
            {
                IngestionJobStatus.PROCESSING,
                IngestionJobStatus.SUCCEEDED,
                IngestionJobStatus.FAILED,
            }
        ),
        IngestionJobStatus.SUCCEEDED: frozenset({IngestionJobStatus.SUCCEEDED}),
        IngestionJobStatus.FAILED: frozenset({IngestionJobStatus.FAILED}),
    }
    _STAGE_ORDER: ClassVar[dict[IngestionStage, int]] = {
        stage: index for index, stage in enumerate(PROCESSING_INGESTION_STAGES)
    }

    def __init__(self, database: Database) -> None:
        self._database = database

    def create(
        self,
        job: IngestionJob,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        if job.status is not IngestionJobStatus.QUEUED:
            raise RepositoryConflict("new ingestion jobs must be queued")
        try:
            with _write_connection(self._database, connection) as active:
                active.execute(
                    """
                    INSERT INTO ingestion_jobs(job_id, source_key, source_id, status, payload_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        job.job_id,
                        job.source_key,
                        job.source_id,
                        job.status.value,
                        job.model_dump_json(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            _raise_conflict("ingestion job", error)

    def get(
        self,
        job_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> IngestionJob | None:
        with _read_connection(self._database, connection) as active:
            row = active.execute(
                "SELECT payload_json FROM ingestion_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return None if row is None else _decode(IngestionJob, row)

    def update(
        self,
        job: IngestionJob,
        *,
        updated_at: datetime | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> IngestionJob:
        return self.transition(job, updated_at=updated_at, connection=connection)

    def transition(
        self,
        job: IngestionJob,
        *,
        updated_at: datetime | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> IngestionJob:
        with _write_connection(self._database, connection) as active:
            existing = self.get(job.job_id, connection=active)
            if existing is None:
                raise RepositoryNotFound(f"ingestion job {job.job_id!r} was not found")
            if job.source_key != existing.source_key or job.created_at != existing.created_at:
                raise RepositoryConflict("ingestion job identity fields are immutable")
            if job.operation is not existing.operation:
                raise RepositoryConflict("ingestion job operation is immutable")
            if job.status not in self._ALLOWED_TRANSITIONS[existing.status]:
                raise RepositoryConflict(
                    f"invalid ingestion transition {existing.status.value} -> {job.status.value}"
                )
            if (
                existing.status is IngestionJobStatus.PROCESSING
                and job.status is IngestionJobStatus.PROCESSING
                and self._STAGE_ORDER[job.stage] < self._STAGE_ORDER[existing.stage]
            ):
                raise RepositoryConflict("ingestion stages cannot regress")

            self._validate_result_fields(existing, job)
            merged = self._merge_transition(existing, job)
            if existing.status in {
                IngestionJobStatus.SUCCEEDED,
                IngestionJobStatus.FAILED,
            }:
                unchanged = IngestionJob.model_validate(
                    {**merged.model_dump(), "updated_at": existing.updated_at}
                )
                if unchanged != existing:
                    raise RepositoryConflict("terminal ingestion jobs are immutable")
                return existing

            timestamp = self._next_timestamp(existing, job, updated_at)
            transitioned = IngestionJob.model_validate(
                {**merged.model_dump(), "updated_at": timestamp}
            )
            cursor = active.execute(
                """
                UPDATE ingestion_jobs
                SET source_id = ?, status = ?, payload_json = ?
                WHERE job_id = ?
                """,
                (
                    transitioned.source_id,
                    transitioned.status.value,
                    transitioned.model_dump_json(),
                    transitioned.job_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RepositoryNotFound(f"ingestion job {job.job_id!r} was not found")
            return transitioned

    @staticmethod
    def _validate_result_fields(existing: IngestionJob, requested: IngestionJob) -> None:
        for field_name in ("source_id", "document_version", "active_index_revision"):
            current = getattr(existing, field_name)
            replacement = getattr(requested, field_name)
            if current is not None and replacement is None:
                raise RepositoryConflict(f"ingestion job field {field_name} cannot be cleared")
            if current is not None and replacement != current:
                raise RepositoryConflict(f"ingestion job field {field_name} is already assigned")
        for field_name in ("ocr_page_count", "chunk_count"):
            if getattr(requested, field_name) < getattr(existing, field_name):
                raise RepositoryConflict(f"ingestion job field {field_name} cannot decrease")

    @staticmethod
    def _merge_transition(existing: IngestionJob, requested: IngestionJob) -> IngestionJob:
        timings = {**existing.stage_timings_ms, **requested.stage_timings_ms}
        warnings = tuple(dict.fromkeys((*existing.warnings, *requested.warnings)))
        failed_stage = requested.failed_stage
        if requested.status is IngestionJobStatus.FAILED:
            origin = existing.failed_stage
            if origin is None and existing.status is not IngestionJobStatus.FAILED:
                origin = existing.stage
            if failed_stage is not None and failed_stage is not origin:
                raise RepositoryConflict("failed_stage must match the originating stage")
            failed_stage = origin
        try:
            return IngestionJob.model_validate(
                {
                    **requested.model_dump(),
                    "stage_timings_ms": timings,
                    "warnings": warnings,
                    "failed_stage": failed_stage,
                }
            )
        except ValueError as error:
            raise RepositoryConflict("merged ingestion diagnostics exceed safe bounds") from error

    @staticmethod
    def _next_timestamp(
        existing: IngestionJob,
        requested: IngestionJob,
        explicit: datetime | None,
    ) -> datetime:
        if explicit is not None:
            if explicit.tzinfo is None or explicit.utcoffset() is None:
                raise RepositoryConflict("updated_at must be timezone-aware")
            if explicit <= existing.updated_at:
                raise RepositoryConflict("updated_at must advance")
            return explicit
        if requested.updated_at > existing.updated_at:
            return requested.updated_at
        current = _now()
        return max(current, existing.updated_at + timedelta(microseconds=1))

    def list_for_source(self, source_key: str) -> list[IngestionJob]:
        with self._database.connection() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM ingestion_jobs
                WHERE source_key = ? ORDER BY rowid
                """,
                (source_key,),
            ).fetchall()
        return [_decode(IngestionJob, row) for row in rows]

    def list(self) -> list[IngestionJob]:
        with self._database.connection() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM ingestion_jobs ORDER BY rowid"
            ).fetchall()
        return [_decode(IngestionJob, row) for row in rows]

    def list_nonterminal(
        self,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> builtins.list[IngestionJob]:
        with _read_connection(self._database, connection) as active:
            rows = active.execute(
                """
                SELECT payload_json FROM ingestion_jobs
                WHERE status IN (?, ?) ORDER BY rowid
                """,
                (IngestionJobStatus.QUEUED.value, IngestionJobStatus.PROCESSING.value),
            ).fetchall()
        return [_decode(IngestionJob, row) for row in rows]

    def list_nonterminal_through(
        self,
        job_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> builtins.list[IngestionJob]:
        with _read_connection(self._database, connection) as active:
            target = active.execute(
                "SELECT rowid FROM ingestion_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if target is None:
                raise RepositoryNotFound(f"ingestion job {job_id!r} was not found")
            rows = active.execute(
                """
                SELECT payload_json FROM ingestion_jobs
                WHERE status IN (?, ?) AND rowid <= ? ORDER BY rowid
                """,
                (
                    IngestionJobStatus.QUEUED.value,
                    IngestionJobStatus.PROCESSING.value,
                    int(target["rowid"]),
                ),
            ).fetchall()
        return [_decode(IngestionJob, row) for row in rows]

    def requeue_interrupted(
        self,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> int:
        """Reset processing jobs for replay while retaining durable result assignments."""

        with _write_connection(self._database, connection) as active:
            rows = active.execute(
                """
                SELECT payload_json FROM ingestion_jobs
                WHERE status = ? ORDER BY rowid
                """,
                (IngestionJobStatus.PROCESSING.value,),
            ).fetchall()
            for row in rows:
                job = _decode(IngestionJob, row)
                requeued = IngestionJob.model_validate(
                    {
                        **job.model_dump(),
                        "status": IngestionJobStatus.QUEUED,
                        "stage": IngestionStage.QUEUED,
                        "updated_at": max(_now(), job.updated_at + timedelta(microseconds=1)),
                    }
                )
                active.execute(
                    """
                    UPDATE ingestion_jobs SET status = ?, payload_json = ? WHERE job_id = ?
                    """,
                    (requeued.status.value, requeued.model_dump_json(), requeued.job_id),
                )
            return len(rows)


class IndexRevisionRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    @property
    def database(self) -> Database:
        return self._database

    def create(
        self,
        revision: IndexRevision,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        try:
            with _write_connection(self._database, connection) as active:
                active.execute(
                    """
                    INSERT INTO index_revisions(revision_id, status, payload_json)
                    VALUES (?, ?, ?)
                    """,
                    (revision.revision_id, revision.status.value, revision.model_dump_json()),
                )
        except sqlite3.IntegrityError as error:
            _raise_conflict("index revision", error)

    def get(
        self,
        revision_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> IndexRevision | None:
        with _read_connection(self._database, connection) as active:
            row = active.execute(
                "SELECT payload_json FROM index_revisions WHERE revision_id = ?",
                (revision_id,),
            ).fetchone()
        return None if row is None else _decode(IndexRevision, row)

    def list(self) -> list[IndexRevision]:
        with self._database.connection() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM index_revisions ORDER BY rowid"
            ).fetchall()
        return [_decode(IndexRevision, row) for row in rows]

    def mark_staged_failed(
        self,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> int:
        """Abandon staged rows without deleting evidence or changing the active pointer."""

        with _write_connection(self._database, connection) as active:
            rows = active.execute(
                "SELECT payload_json FROM index_revisions WHERE status = ? ORDER BY rowid",
                (IndexRevisionStatus.STAGED.value,),
            ).fetchall()
            for row in rows:
                revision = _decode(IndexRevision, row)
                failed = IndexRevision.model_validate(
                    {**revision.model_dump(), "status": IndexRevisionStatus.FAILED}
                )
                active.execute(
                    """
                    UPDATE index_revisions SET status = ?, payload_json = ?
                    WHERE revision_id = ? AND status = ?
                    """,
                    (
                        failed.status.value,
                        failed.model_dump_json(),
                        failed.revision_id,
                        IndexRevisionStatus.STAGED.value,
                    ),
                )
            return len(rows)

    def mark_failed(
        self,
        revision_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> IndexRevision:
        with _write_connection(self._database, connection) as active:
            revision = self.get(revision_id, connection=active)
            if revision is None:
                raise RepositoryNotFound(f"index revision {revision_id!r} was not found")
            if revision.status is IndexRevisionStatus.FAILED:
                return revision
            if revision.status is not IndexRevisionStatus.STAGED:
                raise RepositoryConflict("only a staged index revision can be failed")
            failed = IndexRevision.model_validate(
                {**revision.model_dump(), "status": IndexRevisionStatus.FAILED}
            )
            active.execute(
                """
                UPDATE index_revisions SET status = ?, payload_json = ? WHERE revision_id = ?
                """,
                (failed.status.value, failed.model_dump_json(), failed.revision_id),
            )
            return failed

    def get_active(
        self,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> IndexRevision | None:
        with _read_connection(self._database, connection) as active:
            row = active.execute(
                """
                SELECT
                    manifest.revision_id AS manifest_revision_id,
                    revision.status AS stored_status,
                    revision.payload_json
                FROM active_index_manifest AS manifest
                LEFT JOIN index_revisions AS revision
                    ON revision.revision_id = manifest.revision_id
                WHERE manifest.singleton_id = 1
                """,
            ).fetchone()
        if row is None:
            return None
        if row["stored_status"] != IndexRevisionStatus.ACTIVE.value or row["payload_json"] is None:
            raise RepositoryError("active index manifest is invalid")
        revision = _decode(IndexRevision, row)
        if (
            revision.status is not IndexRevisionStatus.ACTIVE
            or revision.revision_id != row["manifest_revision_id"]
        ):
            raise RepositoryError("active index manifest is invalid")
        return revision

    def get_active_revision_id(
        self,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> str | None:
        with _read_connection(self._database, connection) as active:
            row = active.execute(
                """
                SELECT revision_id FROM active_index_manifest WHERE singleton_id = 1
                """
            ).fetchone()
        return None if row is None else str(row["revision_id"])

    def list_active_status_ids(
        self,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> builtins.list[str]:
        with _read_connection(self._database, connection) as active:
            rows = active.execute(
                "SELECT revision_id FROM index_revisions WHERE status = ? ORDER BY rowid",
                (IndexRevisionStatus.ACTIVE.value,),
            ).fetchall()
        return [str(row["revision_id"]) for row in rows]

    def publish(
        self,
        revision_id: str,
        *,
        published_at: datetime | None = None,
        expected_active_revision_id: str | object | None = _EXPECTED_ACTIVE_REVISION_UNSET,
        ingestion_job_id: str | None = None,
        job_ocr_page_count: int | None = None,
        job_chunk_count: int | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> IndexRevision:
        with _write_connection(self._database, connection) as active:
            revision = self.get(revision_id, connection=active)
            if revision is None:
                raise RepositoryNotFound(f"index revision {revision_id!r} was not found")
            current = self.get_active(connection=active)
            current_id = None if current is None else current.revision_id
            if expected_active_revision_id is not _EXPECTED_ACTIVE_REVISION_UNSET:
                expected = cast(str | None, expected_active_revision_id)
                if expected is not None and (not isinstance(expected, str) or not expected):
                    raise ValueError("expected_active_revision_id must be non-empty or None")
                if current_id != expected:
                    raise RepositoryConflict("active index revision changed")
            if current is not None and current.revision_id == revision_id:
                return current
            if revision.status is not IndexRevisionStatus.STAGED:
                raise RepositoryConflict(
                    f"only a staged index revision can be published, got {revision.status.value}"
                )

            documents = DocumentRepository(self._database)
            all_documents = {
                document.source_id: document
                for document in documents.list(include_deleted=True, connection=active)
            }
            for source_id, version in revision.active_sources.items():
                if source_id not in all_documents:
                    raise RepositoryNotFound(f"document {source_id!r} was not found")
                if documents.get_version(source_id, version, connection=active) is None:
                    raise RepositoryNotFound(
                        f"document version {source_id!r}/{version} was not found"
                    )

            previously_active = documents.list_active(connection=active)
            for source_id, version in sorted(revision.active_sources.items()):
                documents.set_active_version(source_id, version, connection=active)
            retained_source_ids = set(revision.active_sources)
            for document in previously_active:
                if document.source_id not in retained_source_ids:
                    documents.mark_deleted(document.source_id, connection=active)

            timestamp = published_at or _now()
            if current is not None:
                superseded = IndexRevision.model_validate(
                    {**current.model_dump(), "status": IndexRevisionStatus.SUPERSEDED}
                )
                active.execute(
                    """
                    UPDATE index_revisions SET status = ?, payload_json = ?
                    WHERE revision_id = ?
                    """,
                    (
                        superseded.status.value,
                        superseded.model_dump_json(),
                        superseded.revision_id,
                    ),
                )

            published = IndexRevision.model_validate(
                {
                    **revision.model_dump(),
                    "status": IndexRevisionStatus.ACTIVE,
                    "published_at": timestamp,
                }
            )
            active.execute(
                """
                UPDATE index_revisions SET status = ?, payload_json = ?
                WHERE revision_id = ?
                """,
                (published.status.value, published.model_dump_json(), published.revision_id),
            )
            active.execute(
                """
                INSERT INTO active_index_manifest(singleton_id, revision_id, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(singleton_id) DO UPDATE SET
                    revision_id = excluded.revision_id,
                    updated_at = excluded.updated_at
                """,
                (published.revision_id, timestamp.isoformat()),
            )

            if ingestion_job_id is not None:
                jobs = IngestionJobRepository(self._database)
                job = jobs.get(ingestion_job_id, connection=active)
                if job is None:
                    raise RepositoryNotFound(f"ingestion job {ingestion_job_id!r} was not found")
                completed = IngestionJob.model_validate(
                    {
                        **job.model_dump(),
                        "status": IngestionJobStatus.SUCCEEDED,
                        "stage": IngestionStage.COMPLETE,
                        "ocr_page_count": (
                            job.ocr_page_count if job_ocr_page_count is None else job_ocr_page_count
                        ),
                        "chunk_count": (
                            revision.chunk_count if job_chunk_count is None else job_chunk_count
                        ),
                        "active_index_revision": published.revision_id,
                    }
                )
                jobs.transition(completed, connection=active)
            elif job_ocr_page_count is not None or job_chunk_count is not None:
                raise ValueError("job counts require ingestion_job_id")
            return published


class SessionRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    @property
    def database(self) -> Database:
        return self._database

    def create(
        self,
        session: ConversationSession,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        try:
            with _write_connection(self._database, connection) as active:
                active.execute(
                    """
                    INSERT INTO sessions(session_id, owner_id, status, payload_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        session.session_id,
                        session.owner_id,
                        session.status.value,
                        session.model_dump_json(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            _raise_conflict("session", error)

    def get(
        self,
        session_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> ConversationSession | None:
        with _read_connection(self._database, connection) as active:
            row = active.execute(
                "SELECT payload_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return None if row is None else _decode(ConversationSession, row)

    def require_owned(
        self,
        session_id: str,
        owner_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> ConversationSession:
        session = self.get(session_id, connection=connection)
        if session is None:
            raise RepositoryNotFound(f"session {session_id!r} was not found")
        if session.owner_id != owner_id:
            raise SessionOwnershipError(f"session {session_id!r} belongs to another owner")
        return session

    def append_turn(
        self,
        turn: ConversationTurn,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        try:
            with _write_connection(self._database, connection) as active:
                session = self.get(turn.session_id, connection=active)
                if session is None:
                    raise RepositoryNotFound(f"session {turn.session_id!r} was not found")
                if session.status is not SessionStatus.ACTIVE:
                    raise RepositoryConflict("cannot append a turn to a reset session")
                row = active.execute(
                    """
                    SELECT COALESCE(MAX(ordinal), -1) + 1 AS next_ordinal
                    FROM session_turns WHERE session_id = ?
                    """,
                    (turn.session_id,),
                ).fetchone()
                next_ordinal = 0 if row is None else int(row["next_ordinal"])
                if turn.ordinal != next_ordinal:
                    raise RepositoryConflict(
                        f"turn ordinal must be {next_ordinal}, got {turn.ordinal}"
                    )
                active.execute(
                    """
                    INSERT INTO session_turns(turn_id, session_id, ordinal, payload_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (turn.turn_id, turn.session_id, turn.ordinal, turn.model_dump_json()),
                )
        except sqlite3.IntegrityError as error:
            _raise_conflict("conversation turn", error)

    def list_turns(
        self,
        session_id: str,
        *,
        include_reset_history: bool = False,
        connection: sqlite3.Connection | None = None,
    ) -> list[ConversationTurn]:
        with _read_connection(self._database, connection) as active:
            session = self.get(session_id, connection=active)
            if session is None:
                raise RepositoryNotFound(f"session {session_id!r} was not found")
            if session.status is SessionStatus.RESET and not include_reset_history:
                return []
            rows = active.execute(
                """
                SELECT payload_json FROM session_turns
                WHERE session_id = ? ORDER BY ordinal
                """,
                (session_id,),
            ).fetchall()
        return [_decode(ConversationTurn, row) for row in rows]

    def reset(
        self,
        session_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> ConversationSession:
        with _write_connection(self._database, connection) as active:
            session = self.get(session_id, connection=active)
            if session is None:
                raise RepositoryNotFound(f"session {session_id!r} was not found")
            timestamp = _now()
            reset = ConversationSession.model_validate(
                {
                    **session.model_dump(),
                    "status": SessionStatus.RESET,
                    "updated_at": timestamp,
                    "reset_at": timestamp,
                }
            )
            active.execute(
                """
                UPDATE sessions SET status = ?, payload_json = ? WHERE session_id = ?
                """,
                (reset.status.value, reset.model_dump_json(), reset.session_id),
            )
            return reset


class RequestDiagnosticRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def save(
        self,
        diagnostic: RequestDiagnostic,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        with _write_connection(self._database, connection) as active:
            active.execute(
                """
                INSERT INTO request_diagnostics(
                    request_id, session_id, created_at, expires_at, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at,
                    payload_json = excluded.payload_json
                """,
                (
                    diagnostic.request_id,
                    diagnostic.session_id,
                    diagnostic.created_at.isoformat(),
                    diagnostic.expires_at.isoformat() if diagnostic.expires_at else None,
                    diagnostic.model_dump_json(),
                ),
            )

    def get(self, request_id: str, *, now: datetime | None = None) -> RequestDiagnostic | None:
        with self._database.connection() as connection:
            row = connection.execute(
                "SELECT payload_json FROM request_diagnostics WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        if row is None:
            return None
        diagnostic = _decode(RequestDiagnostic, row)
        if diagnostic.expires_at is not None and diagnostic.expires_at <= (now or _now()):
            return None
        return diagnostic

    def purge_expired(self, *, now: datetime | None = None) -> int:
        timestamp = (now or _now()).isoformat()
        with self._database.transaction() as connection:
            cursor = connection.execute(
                """
                DELETE FROM request_diagnostics
                WHERE expires_at IS NOT NULL AND expires_at <= ?
                """,
                (timestamp,),
            )
            return cursor.rowcount


class ProviderUsageRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def record(
        self,
        attempt: ModelAttempt,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        try:
            with _write_connection(self._database, connection) as active:
                active.execute(
                    """
                    INSERT INTO provider_usage(
                        attempt_id, request_id, run_id, created_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        attempt.attempt_id,
                        attempt.request_id,
                        attempt.run_id,
                        attempt.created_at.isoformat(),
                        attempt.model_dump_json(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            _raise_conflict("provider attempt", error)

    def get(self, attempt_id: str) -> ModelAttempt | None:
        with self._database.connection() as connection:
            row = connection.execute(
                "SELECT payload_json FROM provider_usage WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
        return None if row is None else _decode(ModelAttempt, row)

    def list_for_request(self, request_id: str) -> list[ModelAttempt]:
        return self._list_by("request_id", request_id)

    def list_for_run(self, run_id: str) -> list[ModelAttempt]:
        return self._list_by("run_id", run_id)

    def _list_by(self, column: str, value: str) -> list[ModelAttempt]:
        if column == "request_id":
            query = (
                "SELECT payload_json FROM provider_usage WHERE request_id = ? ORDER BY created_at"
            )
        elif column == "run_id":
            query = "SELECT payload_json FROM provider_usage WHERE run_id = ? ORDER BY created_at"
        else:
            raise ValueError("unsupported provider-usage lookup")
        with self._database.connection() as connection:
            rows = connection.execute(query, (value,)).fetchall()
        return [_decode(ModelAttempt, row) for row in rows]


class EvaluationRunRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def create(
        self,
        run: EvaluationRun,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        try:
            with _write_connection(self._database, connection) as active:
                active.execute(
                    """
                    INSERT INTO evaluation_runs(
                        run_id, status, created_at, updated_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run.run_id,
                        run.status.value,
                        run.created_at.isoformat(),
                        run.updated_at.isoformat(),
                        run.model_dump_json(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            _raise_conflict("evaluation run", error)

    def get(
        self,
        run_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> EvaluationRun | None:
        with _read_connection(self._database, connection) as active:
            row = active.execute(
                "SELECT payload_json FROM evaluation_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return None if row is None else _decode(EvaluationRun, row)

    def update(
        self,
        run: EvaluationRun,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        with _write_connection(self._database, connection) as active:
            cursor = active.execute(
                """
                UPDATE evaluation_runs SET status = ?, updated_at = ?, payload_json = ?
                WHERE run_id = ?
                """,
                (run.status.value, run.updated_at.isoformat(), run.model_dump_json(), run.run_id),
            )
            if cursor.rowcount != 1:
                raise RepositoryNotFound(f"evaluation run {run.run_id!r} was not found")

    def list(self) -> list[EvaluationRun]:
        with self._database.connection() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM evaluation_runs ORDER BY created_at"
            ).fetchall()
        return [_decode(EvaluationRun, row) for row in rows]


class ReportManifestRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def save(
        self,
        manifest: ReportManifest,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        try:
            with _write_connection(self._database, connection) as active:
                active.execute(
                    """
                    INSERT INTO report_manifests(run_id, created_at, payload_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        created_at = excluded.created_at,
                        payload_json = excluded.payload_json
                    """,
                    (
                        manifest.run_id,
                        manifest.created_at.isoformat(),
                        manifest.model_dump_json(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            _raise_conflict("report manifest", error)

    def get(self, run_id: str) -> ReportManifest | None:
        with self._database.connection() as connection:
            row = connection.execute(
                "SELECT payload_json FROM report_manifests WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return None if row is None else _decode(ReportManifest, row)


class ComparisonSelectionIdentity(BaseModel):
    """One safe fixed identity inherited from an upstream comparison plan."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=255, pattern=r"^[a-z][a-z0-9_.-]*$")
    value: str = Field(min_length=1, max_length=4096)


class ComparisonSelectionRecord(BaseModel):
    """Append-only upstream selection bound to one verified comparison result."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal["comparison-selection-v1"] = "comparison-selection-v1"
    selection_sequence: int = Field(ge=1)
    comparison_id: str = Field(min_length=1, max_length=255)
    axis: Literal["generation-model", "retrieval-strategy", "cache-behavior"]
    plan_id: str = Field(min_length=1, max_length=255)
    plan_content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    result_content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    selected_variant_id: str = Field(min_length=1, max_length=255)
    selected_axis_value: str = Field(min_length=1, max_length=4096)
    selected_configuration_id: str = Field(min_length=1, max_length=255)
    selected_evaluation_run_id: str = Field(min_length=1, max_length=255)
    upstream_identities: tuple[ComparisonSelectionIdentity, ...] = ()
    created_at: datetime

    @model_validator(mode="after")
    def validate_timestamp(self) -> Self:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("comparison_selection_timestamp_naive")
        names = tuple(item.name for item in self.upstream_identities)
        if (
            len(names) != len(set(names))
            or tuple(sorted(names)) != names
            or any(not name.startswith("upstream.") for name in names)
        ):
            raise ValueError("comparison_selection_upstream_identity_invalid")
        return self


class ComparisonRepository:
    """Append-only persisted comparison suites, results, and axis selections."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def create(
        self,
        suite: ComparisonSuite,
        evaluation_runs: Sequence[EvaluationRun],
    ) -> None:
        """Atomically persist all prepared candidate runs and immutable suite revision zero."""

        self._validate_initial_suite(suite)
        runs = {item.run_id: item for item in evaluation_runs}
        references = tuple(item.reference for item in suite.candidates)
        if len(runs) != len(evaluation_runs) or set(runs) != {
            item.evaluation_run_id for item in references
        }:
            raise RepositoryConflict("comparison candidate run set does not match the suite")
        for reference in references:
            self._validate_evaluation_run(suite, reference, runs[reference.evaluation_run_id])

        try:
            with self._database.transaction() as connection:
                run_repository = EvaluationRunRepository(self._database)
                for run in evaluation_runs:
                    run_repository.create(run, connection=connection)
                connection.execute(
                    """
                    INSERT INTO comparison_plans(
                        comparison_id, plan_id, plan_content_hash, axis, created_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        suite.comparison_id,
                        suite.plan.plan_id,
                        suite.plan_content_hash,
                        suite.plan.axis.value,
                        suite.created_at.isoformat(),
                        suite.plan.model_dump_json(),
                    ),
                )
                for history in suite.candidates:
                    reference = history.reference
                    connection.execute(
                        """
                        INSERT INTO comparison_candidate_bindings(
                            comparison_id, variant_id, axis_value, configuration_id,
                            evaluation_run_id, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            suite.comparison_id,
                            reference.variant_id,
                            reference.axis_value,
                            reference.configuration_id,
                            reference.evaluation_run_id,
                            suite.created_at.isoformat(),
                        ),
                    )
                    snapshot = history.snapshots[0]
                    connection.execute(
                        """
                        INSERT INTO comparison_candidates(
                            comparison_id, variant_id, revision, evaluation_run_id,
                            status, recorded_at, payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            suite.comparison_id,
                            reference.variant_id,
                            snapshot.sequence,
                            reference.evaluation_run_id,
                            snapshot.status.value,
                            snapshot.recorded_at.isoformat(),
                            snapshot.model_dump_json(),
                        ),
                    )
                progress = suite.progress_history[0]
                connection.execute(
                    """
                    INSERT INTO comparison_runs(
                        comparison_id, revision, status, recorded_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        suite.comparison_id,
                        progress.sequence,
                        suite.status.value,
                        progress.recorded_at.isoformat(),
                        suite.model_dump_json(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            _raise_conflict("comparison", error)

    def append(self, suite: ComparisonSuite) -> None:
        """Append one validated suite revision and any matching candidate snapshot."""

        try:
            with self._database.transaction() as connection:
                stored = self._get_verified(suite.comparison_id, connection)
                if stored is None:
                    raise RepositoryNotFound("comparison was not found")
                changed = self._validate_successor(stored, suite)
                progress = suite.progress_history[-1]
                for history in changed:
                    snapshot = history.snapshots[-1]
                    connection.execute(
                        """
                        INSERT INTO comparison_candidates(
                            comparison_id, variant_id, revision, evaluation_run_id,
                            status, recorded_at, payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            suite.comparison_id,
                            history.reference.variant_id,
                            snapshot.sequence,
                            history.reference.evaluation_run_id,
                            snapshot.status.value,
                            snapshot.recorded_at.isoformat(),
                            snapshot.model_dump_json(),
                        ),
                    )
                connection.execute(
                    """
                    INSERT INTO comparison_runs(
                        comparison_id, revision, status, recorded_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        suite.comparison_id,
                        progress.sequence,
                        suite.status.value,
                        progress.recorded_at.isoformat(),
                        suite.model_dump_json(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            _raise_conflict("comparison history", error)

    def get(self, comparison_id: str) -> ComparisonSuite | None:
        with self._database.connection() as connection:
            return self._get_verified(comparison_id, connection)

    def list(self) -> builtins.list[ComparisonSuite]:
        with self._database.connection() as connection:
            rows = connection.execute(
                "SELECT comparison_id FROM comparison_plans ORDER BY created_at DESC, comparison_id"
            ).fetchall()
            values = [self._get_verified(str(row["comparison_id"]), connection) for row in rows]
        return [item for item in values if item is not None]

    def save_result(self, result: ComparisonResult) -> None:
        """Persist one immutable deterministic result after binding it to the stored suite."""

        result_hash = _comparison_result_content_hash(result)
        try:
            with self._database.transaction() as connection:
                suite = self._get_verified(result.comparison_id, connection)
                if suite is None:
                    raise RepositoryNotFound("comparison was not found")
                self._validate_result_binding(suite, result)
                setup = self._get_shared_setup_verified(result.comparison_id, connection)
                if setup is None or setup != result.shared_setup:
                    raise RepositoryConflict(
                        "comparison result shared setup evidence binding failed"
                    )
                connection.execute(
                    """
                    INSERT INTO comparison_results(
                        comparison_id, plan_content_hash, result_content_hash,
                        completed_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        result.comparison_id,
                        result.plan_content_hash,
                        result_hash,
                        result.completed_at.isoformat(),
                        result.model_dump_json(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            _raise_conflict("comparison result", error)

    def save_shared_setup(self, evidence: ComparisonSharedSetupEvidence) -> None:
        """Persist one immutable setup ledger, including failure spend without a result."""

        from rag_mvp.evaluation.comparison import validate_comparison_shared_setup

        evidence_hash = _comparison_shared_setup_content_hash(evidence)
        try:
            with self._database.transaction() as connection:
                suite = self._get_verified(evidence.comparison_id, connection)
                if suite is None:
                    raise RepositoryNotFound("comparison was not found")
                validate_comparison_shared_setup(
                    evidence,
                    comparison_id=suite.comparison_id,
                    plan=suite.plan,
                )
                if evidence.recorded_at < suite.created_at:
                    raise RepositoryConflict("comparison setup evidence predates the suite")
                connection.execute(
                    """
                    INSERT INTO comparison_shared_setup_evidence(
                        comparison_id, plan_content_hash, status,
                        provider_calls_complete, provider_call_count, known_partial_cost,
                        total_cost, currency, recorded_at, evidence_content_hash, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        evidence.comparison_id,
                        evidence.plan_content_hash,
                        evidence.status.value,
                        int(evidence.provider_calls_complete),
                        (
                            evidence.provider_call_count
                            if isinstance(evidence.provider_call_count, int)
                            else None
                        ),
                        str(evidence.known_partial_cost),
                        None if evidence.total_cost is None else str(evidence.total_cost),
                        evidence.currency,
                        evidence.recorded_at.isoformat(),
                        evidence_hash,
                        evidence.model_dump_json(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            _raise_conflict("comparison shared setup evidence", error)

    def get_shared_setup(
        self,
        comparison_id: str,
    ) -> ComparisonSharedSetupEvidence | None:
        with self._database.connection() as connection:
            return self._get_shared_setup_verified(comparison_id, connection)

    def get_result(self, comparison_id: str) -> ComparisonResult | None:
        with self._database.connection() as connection:
            return self._get_result_verified(comparison_id, connection)

    def append_selection(
        self,
        result: ComparisonResult,
        *,
        created_at: datetime | None = None,
    ) -> ComparisonSelectionRecord:
        """Append a recommended result to axis history after exact persisted-result validation."""

        if (
            result.recommendation.state.value != "recommended"
            or result.recommendation.selected_variant_id is None
        ):
            raise RepositoryConflict("comparison result has no deterministic selection")
        if result.axis.value == "cache-behavior":
            raise RepositoryConflict("cache comparison cannot become an upstream selection")
        selected = next(
            item
            for item in result.candidates
            if item.reference.variant_id == result.recommendation.selected_variant_id
        )
        timestamp = created_at or _now()
        if timestamp < result.completed_at:
            raise RepositoryConflict("comparison selection predates its result")
        try:
            with self._database.transaction() as connection:
                persisted = self._get_result_verified(result.comparison_id, connection)
                if persisted is None:
                    raise RepositoryNotFound("comparison result was not found")
                if persisted != result:
                    raise RepositoryConflict("comparison selection result binding failed")
                result_row = connection.execute(
                    "SELECT result_content_hash FROM comparison_results WHERE comparison_id = ?",
                    (result.comparison_id,),
                ).fetchone()
                if result_row is None:
                    raise RepositoryNotFound("comparison result was not found")
                result_hash = str(result_row["result_content_hash"])
                latest_suite = self._get_verified(result.comparison_id, connection)
                if latest_suite is None or latest_suite.status.value != "completed":
                    raise RepositoryConflict(
                        "comparison selection requires a completed latest suite"
                    )
                sequence_row = connection.execute(
                    "SELECT COALESCE(MAX(selection_sequence), 0) + 1 AS next_sequence "
                    "FROM comparison_selections"
                ).fetchone()
                sequence = int(sequence_row["next_sequence"])
                selection = ComparisonSelectionRecord(
                    selection_sequence=sequence,
                    comparison_id=result.comparison_id,
                    axis=result.axis,
                    plan_id=result.plan_id,
                    plan_content_hash=result.plan_content_hash,
                    result_content_hash=result_hash,
                    selected_variant_id=selected.reference.variant_id,
                    selected_axis_value=selected.reference.axis_value,
                    selected_configuration_id=selected.reference.configuration_id,
                    selected_evaluation_run_id=selected.reference.evaluation_run_id,
                    upstream_identities=tuple(
                        ComparisonSelectionIdentity(name=item.name, value=item.value)
                        for item in result.plan.fixed_identities.controlled
                        if item.name.startswith("upstream.")
                    ),
                    created_at=timestamp,
                )
                connection.execute(
                    """
                    INSERT INTO comparison_selections(
                        selection_sequence, comparison_id, axis, plan_id, plan_content_hash,
                        result_content_hash, selected_variant_id, selected_axis_value,
                        selected_configuration_id, selected_evaluation_run_id,
                        created_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        selection.selection_sequence,
                        selection.comparison_id,
                        selection.axis,
                        selection.plan_id,
                        selection.plan_content_hash,
                        selection.result_content_hash,
                        selection.selected_variant_id,
                        selection.selected_axis_value,
                        selection.selected_configuration_id,
                        selection.selected_evaluation_run_id,
                        selection.created_at.isoformat(),
                        selection.model_dump_json(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            _raise_conflict("comparison selection", error)
        return selection

    def save_selection(
        self,
        result: ComparisonResult,
        *,
        created_at: datetime | None = None,
    ) -> ComparisonSelectionRecord:
        """Compatibility name for the append-only selection boundary."""

        return self.append_selection(result, created_at=created_at)

    def get_selection(self, axis: ExperimentAxis) -> ComparisonSelectionRecord | None:
        with self._database.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM comparison_selections
                WHERE axis = ? ORDER BY selection_sequence DESC LIMIT 1
                """,
                (axis.value,),
            ).fetchone()
            return None if row is None else self._decode_selection_verified(row, connection)

    def list_selections(
        self,
        axis: ExperimentAxis | None = None,
    ) -> builtins.list[ComparisonSelectionRecord]:
        query = "SELECT * FROM comparison_selections"
        parameters: tuple[str, ...] = ()
        if axis is not None:
            query += " WHERE axis = ?"
            parameters = (axis.value,)
        query += " ORDER BY selection_sequence"
        with self._database.connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
            return [self._decode_selection_verified(row, connection) for row in rows]

    @staticmethod
    def _validate_initial_suite(suite: ComparisonSuite) -> None:
        if (
            suite.status.value != "queued"
            or len(suite.progress_history) != 1
            or suite.progress_history[0].sequence != 0
            or any(
                len(item.snapshots) != 1 or item.snapshots[0].sequence != 0
                for item in suite.candidates
            )
        ):
            raise RepositoryConflict("comparison suite is not an initial revision")

    @staticmethod
    def _validate_evaluation_run(
        suite: ComparisonSuite,
        reference: ComparisonCandidateReference,
        run: EvaluationRun,
    ) -> None:
        fixed = suite.plan.fixed_identities
        expected_cases = fixed.case_count * suite.plan.repeat_order_policy.repeats_per_case
        if (
            run.run_id != reference.evaluation_run_id
            or run.configuration_id != reference.configuration_id
            or run.dataset_id != fixed.dataset_id
            or run.dataset_version != fixed.dataset_version
            or run.dataset_hash != fixed.dataset_hash
            or run.corpus_version != fixed.corpus_version
            or run.cache_policy != suite.plan.cache_policy.value
            or run.total_cases != expected_cases
        ):
            raise RepositoryConflict("comparison candidate run identity does not match the plan")

    @staticmethod
    def _validate_successor(
        stored: ComparisonSuite,
        successor: ComparisonSuite,
    ) -> tuple[ComparisonCandidateHistory, ...]:
        if (
            successor.plan != stored.plan
            or successor.plan_content_hash != stored.plan_content_hash
            or successor.created_at != stored.created_at
            or successor.progress_history[:-1] != stored.progress_history
            or len(successor.progress_history) != len(stored.progress_history) + 1
            or tuple(item.reference for item in successor.candidates)
            != tuple(item.reference for item in stored.candidates)
        ):
            raise RepositoryConflict("comparison history is not an append-only successor")
        changed = []
        for old, new in zip(stored.candidates, successor.candidates, strict=True):
            if new.snapshots[: len(old.snapshots)] != old.snapshots or len(new.snapshots) not in {
                len(old.snapshots),
                len(old.snapshots) + 1,
            }:
                raise RepositoryConflict("comparison candidate history is not append-only")
            if len(new.snapshots) == len(old.snapshots) + 1:
                changed.append(new)
        if len(changed) > 1:
            raise RepositoryConflict("one comparison revision may change only one candidate")
        return tuple(changed)

    @staticmethod
    def _validate_result_binding(suite: ComparisonSuite, result: ComparisonResult) -> None:
        if (
            suite.status.value != "completed"
            or result.plan != suite.plan
            or result.plan_content_hash != suite.plan_content_hash
            or tuple(item.reference for item in result.candidates)
            != tuple(item.reference for item in suite.candidates)
        ):
            raise RepositoryConflict("comparison result does not match the persisted suite")

    def _get_verified(
        self,
        comparison_id: str,
        connection: sqlite3.Connection,
    ) -> ComparisonSuite | None:
        from rag_mvp.evaluation.comparison import (
            ComparisonCandidateSnapshot,
            ComparisonSuite,
        )
        from rag_mvp.evaluation.experiment import ExperimentPlan

        run_rows = connection.execute(
            """
            SELECT revision, status, recorded_at, payload_json FROM comparison_runs
            WHERE comparison_id = ? ORDER BY revision
            """,
            (comparison_id,),
        ).fetchall()
        if not run_rows:
            return None
        suite_history = tuple(_decode(ComparisonSuite, row) for row in run_rows)
        suite = suite_history[-1]
        if len(run_rows) != len(suite.progress_history):
            raise RepositoryError("comparison run history integrity check failed")
        previous: ComparisonSuite | None = None
        for index, (row, revision) in enumerate(zip(run_rows, suite_history, strict=True)):
            progress = revision.progress_history[-1]
            if (
                int(row["revision"]) != index
                or progress.sequence != index
                or str(row["status"]) != revision.status.value
                or str(row["recorded_at"]) != progress.recorded_at.isoformat()
                or revision.progress_history != suite.progress_history[: index + 1]
            ):
                raise RepositoryError("comparison run history integrity check failed")
            if previous is not None:
                try:
                    self._validate_successor(previous, revision)
                except RepositoryConflict:
                    raise RepositoryError("comparison run history integrity check failed") from None
            previous = revision
        plan_row = connection.execute(
            """
            SELECT plan_id, plan_content_hash, axis, created_at, payload_json
            FROM comparison_plans WHERE comparison_id = ?
            """,
            (comparison_id,),
        ).fetchone()
        bindings = connection.execute(
            """
            SELECT binding.variant_id, binding.axis_value, binding.configuration_id,
                   binding.evaluation_run_id, run.payload_json AS evaluation_run_payload
            FROM comparison_candidate_bindings AS binding
            JOIN evaluation_runs AS run ON run.run_id = binding.evaluation_run_id
            WHERE binding.comparison_id = ? ORDER BY binding.rowid
            """,
            (comparison_id,),
        ).fetchall()
        if plan_row is None:
            raise RepositoryError("comparison plan integrity check failed")
        plan = ExperimentPlan.model_validate_json(str(plan_row["payload_json"]))
        expected_bindings = tuple(
            (
                item.reference.variant_id,
                item.reference.axis_value,
                item.reference.configuration_id,
                item.reference.evaluation_run_id,
            )
            for item in suite.candidates
        )
        actual_bindings = tuple(
            (
                str(item["variant_id"]),
                str(item["axis_value"]),
                str(item["configuration_id"]),
                str(item["evaluation_run_id"]),
            )
            for item in bindings
        )
        if (
            suite.comparison_id != comparison_id
            or plan != suite.plan
            or str(plan_row["plan_id"]) != suite.plan.plan_id
            or str(plan_row["plan_content_hash"]) != suite.plan_content_hash
            or str(plan_row["axis"]) != suite.plan.axis.value
            or str(plan_row["created_at"]) != suite.created_at.isoformat()
            or actual_bindings != expected_bindings
        ):
            raise RepositoryError("comparison persisted identity integrity check failed")
        for history, binding in zip(suite.candidates, bindings, strict=True):
            run = EvaluationRun.model_validate_json(str(binding["evaluation_run_payload"]))
            try:
                self._validate_evaluation_run(suite, history.reference, run)
            except RepositoryConflict:
                raise RepositoryError(
                    "comparison candidate evaluation identity integrity check failed"
                ) from None
        for history in suite.candidates:
            snapshot_rows = connection.execute(
                """
                SELECT revision, evaluation_run_id, status, recorded_at, payload_json
                FROM comparison_candidates
                WHERE comparison_id = ? AND variant_id = ? ORDER BY revision
                """,
                (comparison_id, history.reference.variant_id),
            ).fetchall()
            snapshots = tuple(_decode(ComparisonCandidateSnapshot, item) for item in snapshot_rows)
            columns_valid = all(
                int(item["revision"]) == snapshot.sequence
                and str(item["evaluation_run_id"]) == history.reference.evaluation_run_id
                and str(item["status"]) == snapshot.status.value
                and str(item["recorded_at"]) == snapshot.recorded_at.isoformat()
                for item, snapshot in zip(snapshot_rows, snapshots, strict=True)
            )
            if snapshots != history.snapshots or not columns_valid:
                raise RepositoryError("comparison candidate history integrity check failed")
        return suite

    def _get_result_verified(
        self,
        comparison_id: str,
        connection: sqlite3.Connection,
    ) -> ComparisonResult | None:
        from rag_mvp.evaluation.comparison import ComparisonResult, ComparisonSuite

        row = connection.execute(
            """
            SELECT comparison_id, plan_content_hash, result_content_hash,
                   completed_at, payload_json
            FROM comparison_results WHERE comparison_id = ?
            """,
            (comparison_id,),
        ).fetchone()
        if row is None:
            return None
        result = _decode(ComparisonResult, row)
        result_payload = json.loads(str(row["payload_json"]))
        if not isinstance(result_payload, dict):
            raise RepositoryError("comparison result payload integrity check failed")
        raw_setup = result_payload.get("shared_setup")
        legacy_setup = isinstance(raw_setup, dict) and ("provider_calls_complete" not in raw_setup)
        legacy_cost_state = "known_partial_cost" not in result_payload
        suite = self._get_verified(comparison_id, connection)
        if (
            suite is None
            or str(row["comparison_id"]) != result.comparison_id
            or str(row["plan_content_hash"]) != result.plan_content_hash
            or str(row["result_content_hash"])
            != _comparison_result_content_hash(
                result,
                legacy_shared_setup=legacy_setup,
                legacy_cost_state=legacy_cost_state,
            )
            or str(row["completed_at"]) != result.completed_at.isoformat()
        ):
            raise RepositoryError("comparison result integrity check failed")
        binding_suite = suite
        if suite.status.value != "completed":
            completed_row = connection.execute(
                """
                SELECT payload_json FROM comparison_runs
                WHERE comparison_id = ? AND revision = ?
                """,
                (comparison_id, len(suite.progress_history) - 2),
            ).fetchone()
            if completed_row is None:
                raise RepositoryError("comparison result historical binding check failed")
            binding_suite = _decode(ComparisonSuite, completed_row)
            if (
                binding_suite.status.value != "completed"
                or suite.status.value != "failed"
                or suite.progress_history[:-1] != binding_suite.progress_history
                or suite.candidates != binding_suite.candidates
                or suite.plan != binding_suite.plan
            ):
                raise RepositoryError("comparison result historical binding check failed")
        self._validate_result_binding(binding_suite, result)
        setup = self._get_shared_setup_verified(comparison_id, connection)
        if setup is None or setup != result.shared_setup:
            raise RepositoryError("comparison result shared setup integrity check failed")
        return result

    def _get_shared_setup_verified(
        self,
        comparison_id: str,
        connection: sqlite3.Connection,
    ) -> ComparisonSharedSetupEvidence | None:
        from rag_mvp.evaluation.comparison import (
            ComparisonSharedSetupEvidence,
            validate_comparison_shared_setup,
        )

        row = connection.execute(
            """
            SELECT comparison_id, plan_content_hash, status, provider_call_count,
                   provider_calls_complete, known_partial_cost, total_cost,
                   currency, recorded_at,
                   evidence_content_hash, payload_json
            FROM comparison_shared_setup_evidence WHERE comparison_id = ?
            """,
            (comparison_id,),
        ).fetchone()
        if row is None:
            return None
        evidence = _decode(ComparisonSharedSetupEvidence, row)
        suite = self._get_verified(comparison_id, connection)
        if suite is None:
            raise RepositoryError("comparison shared setup suite binding failed")
        try:
            validate_comparison_shared_setup(
                evidence,
                comparison_id=suite.comparison_id,
                plan=suite.plan,
            )
        except ValueError:
            raise RepositoryError("comparison shared setup plan binding failed") from None
        payload = str(row["payload_json"])
        decoded_payload = json.loads(payload)
        if not isinstance(decoded_payload, dict):
            raise RepositoryError("comparison shared setup payload integrity check failed")
        legacy_payload = "provider_calls_complete" not in decoded_payload
        expected_columns = (
            evidence.comparison_id,
            evidence.plan_content_hash,
            evidence.status.value,
            int(evidence.provider_calls_complete),
            (
                evidence.provider_call_count
                if isinstance(evidence.provider_call_count, int)
                else None
            ),
            str(evidence.known_partial_cost),
            None if evidence.total_cost is None else str(evidence.total_cost),
            evidence.currency,
            evidence.recorded_at.isoformat(),
            _comparison_shared_setup_content_hash(evidence, legacy=legacy_payload),
        )
        actual_columns = (
            str(row["comparison_id"]),
            str(row["plan_content_hash"]),
            str(row["status"]),
            int(row["provider_calls_complete"]),
            (None if row["provider_call_count"] is None else int(row["provider_call_count"])),
            str(row["known_partial_cost"]),
            None if row["total_cost"] is None else str(row["total_cost"]),
            str(row["currency"]),
            str(row["recorded_at"]),
            str(row["evidence_content_hash"]),
        )
        if actual_columns != expected_columns or evidence.recorded_at < suite.created_at:
            raise RepositoryError("comparison shared setup integrity check failed")
        return evidence

    def _decode_selection_verified(
        self,
        row: sqlite3.Row,
        connection: sqlite3.Connection,
    ) -> ComparisonSelectionRecord:
        selection = _decode(ComparisonSelectionRecord, row)
        columns = (
            int(row["selection_sequence"]),
            str(row["comparison_id"]),
            str(row["axis"]),
            str(row["plan_id"]),
            str(row["plan_content_hash"]),
            str(row["result_content_hash"]),
            str(row["selected_variant_id"]),
            str(row["selected_axis_value"]),
            str(row["selected_configuration_id"]),
            str(row["selected_evaluation_run_id"]),
            str(row["created_at"]),
        )
        expected = (
            selection.selection_sequence,
            selection.comparison_id,
            selection.axis,
            selection.plan_id,
            selection.plan_content_hash,
            selection.result_content_hash,
            selection.selected_variant_id,
            selection.selected_axis_value,
            selection.selected_configuration_id,
            selection.selected_evaluation_run_id,
            selection.created_at.isoformat(),
        )
        result = self._get_result_verified(selection.comparison_id, connection)
        if result is None:
            raise RepositoryError("comparison selection result integrity check failed")
        result_row = connection.execute(
            "SELECT result_content_hash FROM comparison_results WHERE comparison_id = ?",
            (selection.comparison_id,),
        ).fetchone()
        if result_row is None:
            raise RepositoryError("comparison selection result integrity check failed")
        selected = next(
            (
                item
                for item in result.candidates
                if item.reference.variant_id == selection.selected_variant_id
            ),
            None,
        )
        latest_suite = self._get_verified(selection.comparison_id, connection)
        if (
            columns != expected
            or selection.axis != result.axis.value
            or selection.plan_id != result.plan_id
            or selection.plan_content_hash != result.plan_content_hash
            or selection.result_content_hash != str(result_row["result_content_hash"])
            or result.recommendation.state.value != "recommended"
            or result.recommendation.selected_variant_id != selection.selected_variant_id
            or selected is None
            or selected.reference.axis_value != selection.selected_axis_value
            or selected.reference.configuration_id != selection.selected_configuration_id
            or selected.reference.evaluation_run_id != selection.selected_evaluation_run_id
            or selection.upstream_identities
            != tuple(
                ComparisonSelectionIdentity(name=item.name, value=item.value)
                for item in result.plan.fixed_identities.controlled
                if item.name.startswith("upstream.")
            )
            or latest_suite is None
            or latest_suite.status.value != "completed"
        ):
            raise RepositoryError("comparison selection integrity check failed")
        return selection


def _comparison_result_content_hash(
    result: ComparisonResult,
    *,
    legacy_shared_setup: bool = False,
    legacy_cost_state: bool = False,
) -> str:
    exclude: dict[str, object] = {}
    if legacy_shared_setup:
        exclude["shared_setup"] = {"provider_calls_complete"}
    if legacy_cost_state:
        cost_fields = {
            "known_partial_cost",
            "cost_complete",
            "cost_unknown_reasons",
        }
        exclude.update({field: True for field in cost_fields})
        exclude["candidates"] = {
            "__all__": {
                **{field: True for field in cost_fields},
                "source_evidence": {
                    **{field: True for field in cost_fields},
                    "logical_attempts": {"__all__": {field: True for field in cost_fields}},
                },
            }
        }
    content = canonical_json_value(
        result.model_dump(
            mode="json",
            exclude=cast(Any, exclude) if exclude else None,
        )
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _comparison_shared_setup_content_hash(
    evidence: ComparisonSharedSetupEvidence,
    *,
    legacy: bool = False,
) -> str:
    content = canonical_json_value(
        evidence.model_dump(
            mode="json",
            exclude={"provider_calls_complete"} if legacy else None,
        )
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


class ParentChunkRepository:
    """Immutable parent text inventories scoped to one index revision."""

    _LOOKUP_BATCH_SIZE = 400

    def __init__(self, database: Database) -> None:
        self._database = database

    def insert_many(
        self,
        revision_id: str,
        parents: Sequence[ParentChunk],
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        values = tuple(parents)
        if not revision_id:
            raise ValueError("revision_id must be non-empty")
        if len({parent.parent_chunk_id for parent in values}) != len(values):
            raise RepositoryConflict("parent chunk identifiers must be unique")
        if len(
            {(parent.source_id, parent.document_version, parent.ordinal) for parent in values}
        ) != len(values):
            raise RepositoryConflict("parent chunk ordinals must be unique per source version")
        rows = [
            (
                revision_id,
                parent.parent_chunk_id,
                parent.source_id,
                parent.document_version,
                parent.ordinal,
                parent.text,
                parent.content_digest,
                parent.locator.model_dump_json(),
                parent.token_count,
            )
            for parent in values
        ]
        try:
            with _write_connection(self._database, connection) as active:
                active.executemany(
                    """
                    INSERT INTO parent_chunks(
                        revision_id, parent_chunk_id, source_id, document_version,
                        ordinal, text, content_digest, locator_json, token_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        except sqlite3.IntegrityError as error:
            _raise_conflict("parent chunk", error)

    def get_many(
        self,
        revision_id: str,
        parent_chunk_ids: Sequence[str],
        *,
        connection: sqlite3.Connection | None = None,
    ) -> dict[str, ParentChunk]:
        requested = tuple(dict.fromkeys(parent_chunk_ids))
        if not revision_id:
            raise ValueError("revision_id must be non-empty")
        if any(not parent_id for parent_id in requested):
            raise ValueError("parent chunk identifiers must be non-empty")
        if not requested:
            return {}
        parents: dict[str, ParentChunk] = {}
        with _read_connection(self._database, connection) as active:
            for start in range(0, len(requested), self._LOOKUP_BATCH_SIZE):
                batch = requested[start : start + self._LOOKUP_BATCH_SIZE]
                placeholders = ",".join("?" for _ in batch)
                query = f"""
                    SELECT parent_chunk_id, source_id, document_version, ordinal,
                           text, content_digest, locator_json, token_count
                    FROM parent_chunks
                    WHERE revision_id = ? AND parent_chunk_id IN ({placeholders})
                    """  # noqa: S608 - placeholders are generated, not user input
                rows = active.execute(
                    query,
                    (revision_id, *batch),
                ).fetchall()
                for row in rows:
                    parent = self._decode_parent(row)
                    parents[parent.parent_chunk_id] = parent
        return parents

    def list_for_revision(
        self,
        revision_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> tuple[ParentChunk, ...]:
        if not revision_id:
            raise ValueError("revision_id must be non-empty")
        with _read_connection(self._database, connection) as active:
            rows = active.execute(
                """
                SELECT parent_chunk_id, source_id, document_version, ordinal,
                       text, content_digest, locator_json, token_count
                FROM parent_chunks
                WHERE revision_id = ?
                ORDER BY source_id, document_version, ordinal, parent_chunk_id
                """,
                (revision_id,),
            ).fetchall()
        return tuple(self._decode_parent(row) for row in rows)

    def inventory(
        self,
        revision_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> tuple[int, str, dict[str, str]]:
        # Keep storage importable while the retrieval package initializes.
        from rag_mvp.retrieval.snapshot import parent_chunk_record_digest, parent_set_digest

        parents = self.list_for_revision(revision_id, connection=connection)
        records = {parent.parent_chunk_id: parent_chunk_record_digest(parent) for parent in parents}
        return len(parents), parent_set_digest(records), records

    def delete_revision(
        self,
        revision_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> int:
        with _write_connection(self._database, connection) as active:
            cursor = active.execute(
                "DELETE FROM parent_chunks WHERE revision_id = ?",
                (revision_id,),
            )
        return cursor.rowcount

    @staticmethod
    def _decode_parent(row: sqlite3.Row) -> ParentChunk:
        try:
            text = str(row["text"])
            digest = str(row["content_digest"])
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != digest:
                raise RepositoryError("parent chunk content digest mismatch")
            parent = ParentChunk(
                parent_chunk_id=str(row["parent_chunk_id"]),
                source_id=str(row["source_id"]),
                document_version=int(row["document_version"]),
                ordinal=int(row["ordinal"]),
                text=text,
                content_digest=digest,
                locator=json.loads(str(row["locator_json"])),
                token_count=int(row["token_count"]),
            )
            return parent.model_copy(update={"text": text})
        except RepositoryError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RepositoryError("parent chunk record is invalid") from error


@dataclass(frozen=True, slots=True)
class KnowledgeRepositories:
    documents: DocumentRepository
    ingestion_jobs: IngestionJobRepository
    index_revisions: IndexRevisionRepository
    parent_chunks: ParentChunkRepository

    @classmethod
    def from_database(cls, database: Database) -> KnowledgeRepositories:
        return cls(
            documents=DocumentRepository(database),
            ingestion_jobs=IngestionJobRepository(database),
            index_revisions=IndexRevisionRepository(database),
            parent_chunks=ParentChunkRepository(database),
        )


@dataclass(frozen=True, slots=True)
class RuntimeRepositories:
    sessions: SessionRepository
    request_diagnostics: RequestDiagnosticRepository
    provider_usage: ProviderUsageRepository
    evaluation_runs: EvaluationRunRepository
    report_manifests: ReportManifestRepository
    comparisons: ComparisonRepository

    @classmethod
    def from_database(cls, database: Database) -> RuntimeRepositories:
        return cls(
            sessions=SessionRepository(database),
            request_diagnostics=RequestDiagnosticRepository(database),
            provider_usage=ProviderUsageRepository(database),
            evaluation_runs=EvaluationRunRepository(database),
            report_manifests=ReportManifestRepository(database),
            comparisons=ComparisonRepository(database),
        )
