"""SQLite repositories for knowledge and runtime metadata."""

from __future__ import annotations

import builtins
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import ClassVar, cast

from pydantic import BaseModel

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
)
from rag_mvp.domain.qa import (
    ConversationSession,
    ConversationTurn,
    RequestDiagnostic,
    SessionStatus,
)
from rag_mvp.storage.database import Database


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
                SELECT revision.payload_json
                FROM active_index_manifest AS manifest
                JOIN index_revisions AS revision
                    ON revision.revision_id = manifest.revision_id
                WHERE manifest.singleton_id = 1 AND revision.status = ?
                """,
                (IndexRevisionStatus.ACTIVE.value,),
            ).fetchone()
        return None if row is None else _decode(IndexRevision, row)

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

    def require_owned(self, session_id: str, owner_id: str) -> ConversationSession:
        session = self.get(session_id)
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
    ) -> list[ConversationTurn]:
        session = self.get(session_id)
        if session is None:
            raise RepositoryNotFound(f"session {session_id!r} was not found")
        if session.status is SessionStatus.RESET and not include_reset_history:
            return []
        with self._database.connection() as connection:
            rows = connection.execute(
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


@dataclass(frozen=True, slots=True)
class KnowledgeRepositories:
    documents: DocumentRepository
    ingestion_jobs: IngestionJobRepository
    index_revisions: IndexRevisionRepository

    @classmethod
    def from_database(cls, database: Database) -> KnowledgeRepositories:
        return cls(
            documents=DocumentRepository(database),
            ingestion_jobs=IngestionJobRepository(database),
            index_revisions=IndexRevisionRepository(database),
        )


@dataclass(frozen=True, slots=True)
class RuntimeRepositories:
    sessions: SessionRepository
    request_diagnostics: RequestDiagnosticRepository
    provider_usage: ProviderUsageRepository
    evaluation_runs: EvaluationRunRepository
    report_manifests: ReportManifestRepository

    @classmethod
    def from_database(cls, database: Database) -> RuntimeRepositories:
        return cls(
            sessions=SessionRepository(database),
            request_diagnostics=RequestDiagnosticRepository(database),
            provider_usage=ProviderUsageRepository(database),
            evaluation_runs=EvaluationRunRepository(database),
            report_manifests=ReportManifestRepository(database),
        )
