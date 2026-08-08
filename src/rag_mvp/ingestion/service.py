"""Durable single-writer orchestration for corpus mutations and startup recovery."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import re
import shutil
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import uuid4

from rag_mvp.domain.ingestion import (
    Chunk,
    DeleteCommand,
    Document,
    DocumentKind,
    DocumentVersion,
    IndexRevision,
    IndexRevisionStatus,
    IngestionCommand,
    IngestionJob,
    IngestionJobStatus,
    IngestionOperation,
    IngestionStage,
    ParentChunk,
    ReindexCommand,
    UploadCommand,
)
from rag_mvp.ingestion.chunking import (
    ChunkingConfig,
    chunk_document,
    chunk_document_hierarchy,
)
from rag_mvp.ingestion.embedding import EmbeddingStage, EmbeddingStageError
from rag_mvp.ingestion.extractors import (
    ExtractedDocument,
    ExtractionError,
    OcrAdapter,
    PageUsabilityPolicy,
    TesseractOcrAdapter,
    extract_pdf,
    extract_utf8_text,
)
from rag_mvp.ingestion.indexing import (
    INDEX_EXTRACTION_VERSION,
    IndexingError,
    RevisionPublisher,
    RevisionStager,
)
from rag_mvp.ingestion.normalization import (
    NORMALIZATION_VERSION,
    normalize_document,
)
from rag_mvp.ingestion.validation import (
    UploadValidationError,
    ValidatedUpload,
    validate_upload,
)
from rag_mvp.ingestion.versioning import (
    DeletedSourceError,
    JsonValue,
    SourceVersionDisposition,
    SourceVersioningService,
    derivation_config_digest,
)
from rag_mvp.performance.worker_pools import RagWorkerPools, default_worker_pools
from rag_mvp.providers.models import Deadline, ProviderCallContext
from rag_mvp.providers.protocols import EmbeddingProvider
from rag_mvp.retrieval.tokenizer import BILINGUAL_TOKENIZER_IDENTITY
from rag_mvp.storage.artifacts import (
    ArtifactCorruptError,
    ArtifactNotFoundError,
    ArtifactStore,
    ArtifactStoreError,
    StoredVersionArtifacts,
)
from rag_mvp.storage.database import Database
from rag_mvp.storage.embedding_cache import EmbeddingCache
from rag_mvp.storage.layout import DataLayout, UnsafeDataPathError
from rag_mvp.storage.repositories import (
    KnowledgeRepositories,
    RepositoryConflict,
    RepositoryError,
    RepositoryNotFound,
)

_SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_CODE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_MUTATION_LOCKS: dict[Path, asyncio.Lock] = {}


class IngestionSubmissionError(ValueError):
    """A bounded submission failure that is safe to expose."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class IngestionRecoveryError(RuntimeError):
    """A fail-closed startup validation error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    active_revision_id: str | None
    abandoned_revision_count: int
    requeued_job_count: int
    replayed_job_count: int
    failed_job_count: int


class IngestionService:
    """Serialize complete corpus mutations and retain enough intent for replay."""

    def __init__(
        self,
        *,
        layout: DataLayout,
        database: Database,
        repositories: KnowledgeRepositories,
        artifacts: ArtifactStore,
        versioning: SourceVersioningService,
        stager: RevisionStager,
        publisher: RevisionPublisher,
        ocr: OcrAdapter,
        derivation_config: Mapping[str, JsonValue],
        chunking_config: ChunkingConfig | None = None,
        upload_max_bytes: int = 25 * 1024 * 1024,
        ocr_languages: str = "chi_sim+eng",
        page_usability: PageUsabilityPolicy | None = None,
        operation_deadline_seconds: float = 300.0,
        mutation_lock: asyncio.Lock | None = None,
        job_id_factory: Callable[[], str] | None = None,
        revision_id_factory: Callable[[], str] | None = None,
        owned_embedding_cache: EmbeddingCache | None = None,
        worker_pools: RagWorkerPools | None = None,
    ) -> None:
        if upload_max_bytes < 1:
            raise ValueError("upload_max_bytes_invalid")
        if operation_deadline_seconds <= 0:
            raise ValueError("operation_deadline_invalid")
        if worker_pools is not None and not isinstance(worker_pools, RagWorkerPools):
            raise TypeError("worker_pools must be RagWorkerPools")
        self._layout = layout
        self._database = database
        self._repositories = repositories
        self._artifacts = artifacts
        self._versioning = versioning
        self._stager = stager
        self._publisher = publisher
        self._ocr = ocr
        self._derivation_config = dict(derivation_config)
        self._derivation_digest = derivation_config_digest(derivation_config)
        self._chunking = chunking_config or ChunkingConfig()
        self._upload_max_bytes = upload_max_bytes
        self._ocr_languages = ocr_languages
        self._page_usability = page_usability or PageUsabilityPolicy()
        self._operation_deadline_seconds = operation_deadline_seconds
        self._mutation_lock = mutation_lock or _MUTATION_LOCKS.setdefault(
            layout.root,
            asyncio.Lock(),
        )
        self._job_id_factory = job_id_factory or (lambda: f"job_{uuid4().hex}")
        self._revision_id_factory = revision_id_factory or (lambda: f"rev_{uuid4().hex}")
        self._owned_embedding_cache = owned_embedding_cache
        resolved_worker_pools = worker_pools or default_worker_pools()
        self._chroma_worker_pool = resolved_worker_pools.chroma
        self._ocr_worker_pool = resolved_worker_pools.ocr

    @classmethod
    def create(
        cls,
        data_root: Path | str,
        embedding_provider: EmbeddingProvider,
        *,
        ocr: OcrAdapter | None = None,
        chunking_config: ChunkingConfig | None = None,
        upload_max_bytes: int = 25 * 1024 * 1024,
        ocr_languages: str = "chi_sim+eng",
        page_usability: PageUsabilityPolicy | None = None,
        derivation_config: Mapping[str, JsonValue] | None = None,
        embedding_batch_size: int = 128,
        worker_pools: RagWorkerPools | None = None,
    ) -> IngestionService:
        """Compose the persistent implementation used by tests and the application."""

        layout = DataLayout.from_root(data_root)
        layout.initialize()
        database = Database(layout.metadata_db)
        database.initialize()
        repositories = KnowledgeRepositories.from_database(database)
        artifacts = ArtifactStore(layout)
        resolved_ocr = ocr or TesseractOcrAdapter()
        resolved_chunking = chunking_config or ChunkingConfig()
        resolved_usability = page_usability or PageUsabilityPolicy()
        resolved_config = (
            _default_derivation_config(
                ocr=resolved_ocr,
                ocr_languages=ocr_languages,
                usability=resolved_usability,
                chunking=resolved_chunking,
            )
            if derivation_config is None
            else derivation_config
        )
        cache = EmbeddingCache(layout.directory("caches") / "embeddings.sqlite3")
        embeddings = EmbeddingStage(
            embedding_provider,
            cache,
            batch_size=embedding_batch_size,
        )
        versioning = SourceVersioningService(database, repositories.documents, artifacts)
        stager = RevisionStager(
            layout,
            embeddings,
            extraction_version=INDEX_EXTRACTION_VERSION,
            chunking_version=resolved_chunking.version,
            worker_pools=worker_pools,
        )
        publisher = RevisionPublisher(
            layout,
            repositories.index_revisions,
            repositories.parent_chunks,
        )
        return cls(
            layout=layout,
            database=database,
            repositories=repositories,
            artifacts=artifacts,
            versioning=versioning,
            stager=stager,
            publisher=publisher,
            ocr=resolved_ocr,
            derivation_config=resolved_config,
            chunking_config=resolved_chunking,
            upload_max_bytes=upload_max_bytes,
            ocr_languages=ocr_languages,
            page_usability=resolved_usability,
            owned_embedding_cache=cache,
            worker_pools=worker_pools,
        )

    @property
    def repositories(self) -> KnowledgeRepositories:
        return self._repositories

    @property
    def data_root(self) -> Path:
        return self._layout.root

    @property
    def upload_max_bytes(self) -> int:
        return self._upload_max_bytes

    @property
    def chunking_config(self) -> ChunkingConfig:
        """Return the immutable chunking identity used by this runtime."""

        return self._chunking

    @property
    def derivation_config(self) -> Mapping[str, JsonValue]:
        """Return a detached copy of the non-secret document derivation identity."""

        return copy.deepcopy(self._derivation_config)

    async def publish_prechunked_snapshot(
        self,
        *,
        revision_id: str,
        parents: Sequence[ParentChunk],
        chunks: Sequence[Chunk],
        titles: Mapping[str, str],
        active_sources: Mapping[str, int],
        request_id: str,
        cleanup_failed: bool = False,
    ) -> IndexRevision:
        """Publish a fully verified pre-chunked snapshot through the production index path.

        This narrow hook exists for immutable evaluation corpora whose chunks have
        already been reproduced and compared against the production chunker. Document
        and version metadata must be registered before this method is called. Callers
        owning an otherwise-empty target may request removal of an unpublished failed
        revision so the complete higher-level installation can be retried atomically.
        """

        if type(cleanup_failed) is not bool:
            raise TypeError("cleanup_failed must be a boolean")
        async with self._mutation_lock:
            normalized_titles = dict(titles)
            normalized_sources = dict(active_sources)
            self._validate_prechunked_metadata(normalized_titles, normalized_sources)
            expected_active_id = self._repositories.index_revisions.get_active_revision_id()
            context = ProviderCallContext(
                request_id=request_id,
                operation_id=revision_id,
                deadline=Deadline.after(self._operation_deadline_seconds),
            )
            revision = await self._stager.stage(
                revision_id,
                tuple(chunks),
                normalized_titles,
                normalized_sources,
                context,
                parents=tuple(parents),
            )
            try:
                with self._database.transaction() as connection:
                    self._repositories.index_revisions.create(revision, connection=connection)
                    self._repositories.parent_chunks.insert_many(
                        revision_id,
                        tuple(parents),
                        connection=connection,
                    )
            except BaseException:
                await self._remove_revision_artifacts(revision_id)
                raise
            try:
                await self._chroma_worker_pool.run_cancel_safe(
                    self._publisher.validate,
                    revision,
                )
                return await self._chroma_worker_pool.run_cancel_safe(
                    self._publisher.publish,
                    revision_id,
                    expected_active_revision_id=expected_active_id,
                )
            except BaseException:
                with suppress(Exception):
                    self._repositories.index_revisions.mark_failed(revision_id)
                if (
                    cleanup_failed
                    and self._repositories.index_revisions.get_active_revision_id() != revision_id
                ):
                    self._delete_unpublished_revision(revision_id)
                    await self._remove_revision_artifacts(revision_id)
                raise

    def _validate_prechunked_metadata(
        self,
        titles: Mapping[str, str],
        active_sources: Mapping[str, int],
    ) -> None:
        if set(titles) != set(active_sources):
            raise IngestionSubmissionError("prechunked_metadata_invalid")
        for source_id, version_number in active_sources.items():
            document = self._repositories.documents.get(source_id)
            version = self._repositories.documents.get_version(source_id, version_number)
            if (
                document is None
                or document.deleted_at is not None
                or version is None
                or version.source_id != source_id
                or version.version != version_number
                or version.derivation_config_digest != self._derivation_digest
                or titles.get(source_id) != document.display_title
            ):
                raise IngestionSubmissionError("prechunked_metadata_invalid")

    def _delete_unpublished_revision(self, revision_id: str) -> None:
        with self._database.transaction() as connection:
            if (
                self._repositories.index_revisions.get_active_revision_id(connection=connection)
                == revision_id
            ):
                raise RepositoryConflict("cannot delete an active index revision")
            revision = self._repositories.index_revisions.get(
                revision_id,
                connection=connection,
            )
            if revision is None:
                return
            if revision.status not in {
                IndexRevisionStatus.STAGED,
                IndexRevisionStatus.FAILED,
            }:
                raise RepositoryConflict("only an unpublished index revision can be deleted")
            connection.execute(
                "DELETE FROM index_revisions WHERE revision_id = ?",
                (revision_id,),
            )

    async def _remove_revision_artifacts(self, revision_id: str) -> None:
        revision_path = self._layout.index_revision_path(revision_id)
        if revision_path.exists():
            await self._chroma_worker_pool.run_cancel_safe(
                shutil.rmtree,
                revision_path,
            )

    def close(self) -> None:
        if self._owned_embedding_cache is not None:
            self._owned_embedding_cache.close()
            self._owned_embedding_cache = None

    def submit_upload(
        self,
        filename: str,
        content: bytes,
        *,
        source_key: str | None = None,
        declared_media_type: str | None = None,
        display_title: str | None = None,
    ) -> IngestionJob:
        upload = validate_upload(
            filename,
            content,
            declared_media_type=declared_media_type,
            max_bytes=self._upload_max_bytes,
        )
        safe_key = _stable_source_key(source_key or filename.casefold())
        job_id = self._new_opaque_id(self._job_id_factory, "job_id")
        title = _validated_title(display_title or upload.filename)
        command = UploadCommand(
            job_id=job_id,
            source_key=safe_key,
            filename=upload.filename,
            display_title=title,
            media_type=upload.media_type,
            kind=upload.kind,
            payload_size=len(upload.content),
            payload_digest=hashlib.sha256(upload.content).hexdigest(),
            derivation_config_digest=self._derivation_digest,
        )
        return self._persist_submission(command, upload_content=upload.content)

    def submit_reindex(self) -> IngestionJob:
        job_id = self._new_opaque_id(self._job_id_factory, "job_id")
        command = ReindexCommand(
            job_id=job_id,
            source_key="corpus",
            derivation_config_digest=self._derivation_digest,
        )
        return self._persist_submission(command)

    def submit_delete(self, source_id: str) -> IngestionJob:
        try:
            self._layout.source_artifact_relative_path(source_id, 1, "source.txt")
        except UnsafeDataPathError:
            raise IngestionSubmissionError("source_id_invalid") from None
        document = self._repositories.documents.get(source_id)
        if document is None or document.deleted_at is not None or document.active_version is None:
            raise IngestionSubmissionError("source_not_active")
        job_id = self._new_opaque_id(self._job_id_factory, "job_id")
        command = DeleteCommand(
            job_id=job_id,
            source_key=_stable_source_key(document.source_key),
            source_id=document.source_id,
        )
        return self._persist_submission(command)

    def get_job(self, job_id: str) -> IngestionJob | None:
        return self._repositories.ingestion_jobs.get(job_id)

    def list_active_documents(self) -> tuple[str | None, tuple[Document, ...]]:
        """Read one manifest-consistent view of documents visible to retrieval."""

        with self._database.transaction(immediate=False) as connection:
            active = self._repositories.index_revisions.get_active(connection=connection)
            if active is None:
                if self._repositories.documents.list_active(connection=connection):
                    raise IngestionRecoveryError("active_revision_missing")
                return None, ()
            documents: list[Document] = []
            for source_id, version in sorted(active.active_sources.items()):
                document = self._repositories.documents.get(source_id, connection=connection)
                if (
                    document is None
                    or document.deleted_at is not None
                    or document.active_version != version
                ):
                    raise IngestionRecoveryError("active_document_metadata_mismatch")
                documents.append(document)
            return active.revision_id, tuple(documents)

    async def run(self, job_id: str) -> IngestionJob:
        async with self._mutation_lock:
            target = self._repositories.ingestion_jobs.get(job_id)
            if target is None:
                raise RepositoryNotFound("ingestion_job_not_found")
            if target.status in {IngestionJobStatus.SUCCEEDED, IngestionJobStatus.FAILED}:
                return target
            for pending in self._repositories.ingestion_jobs.list_nonterminal_through(job_id):
                await self._execute_safely(pending)
            result = self._repositories.ingestion_jobs.get(job_id)
            if result is None:
                raise RepositoryNotFound("ingestion_job_not_found")
            return result

    async def recover_startup(self) -> RecoveryReport:
        async with self._mutation_lock:
            active_id = await self._validate_active_state()
            try:
                active_revision = self._repositories.index_revisions.get_active()
                if active_revision is not None:
                    legacy_deleted = tuple(
                        document
                        for document in self._repositories.documents.list(include_deleted=True)
                        if document.deleted_at is not None
                    )
                    for document in legacy_deleted:
                        await self._purge_deleted_source(
                            None,
                            document.source_id,
                            active_revision,
                        )
                jobs = {job.job_id: job for job in self._repositories.ingestion_jobs.list()}
                for command_job_id in self._artifacts.list_command_job_ids():
                    command_job = jobs.get(command_job_id)
                    if command_job is None or command_job.status in {
                        IngestionJobStatus.SUCCEEDED,
                        IngestionJobStatus.FAILED,
                    }:
                        with suppress(ArtifactStoreError, OSError, UnsafeDataPathError):
                            self._artifacts.delete_command(command_job_id)
                with self._database.transaction() as connection:
                    abandoned = self._repositories.index_revisions.mark_staged_failed(
                        connection=connection
                    )
                    requeued = self._repositories.ingestion_jobs.requeue_interrupted(
                        connection=connection
                    )
                pending = self._repositories.ingestion_jobs.list_nonterminal()
            except IngestionRecoveryError:
                raise
            except Exception:
                raise IngestionRecoveryError("recovery_metadata_invalid") from None
            failed = 0
            for job in pending:
                result = await self._execute_safely(job)
                failed += result.status is IngestionJobStatus.FAILED
            current = self._repositories.index_revisions.get_active()
            return RecoveryReport(
                active_revision_id=(current.revision_id if current is not None else active_id),
                abandoned_revision_count=abandoned,
                requeued_job_count=requeued,
                replayed_job_count=len(pending),
                failed_job_count=failed,
            )

    def _persist_submission(
        self,
        command: IngestionCommand,
        *,
        upload_content: bytes | None = None,
    ) -> IngestionJob:
        job = IngestionJob(
            job_id=command.job_id,
            source_key=command.source_key,
            operation=command.operation,
            created_at=command.submitted_at,
            updated_at=command.submitted_at,
        )
        self._artifacts.write_command(command, upload_content=upload_content)
        try:
            self._repositories.ingestion_jobs.create(job)
        except BaseException:
            self._artifacts.delete_command(command.job_id)
            raise
        return job

    async def _execute_safely(self, job: IngestionJob) -> IngestionJob:
        try:
            command = self._artifacts.load_command(job.job_id)
            if (
                command.source_key != job.source_key
                or command.operation is not job.operation
                or command.submitted_at != job.created_at
            ):
                raise ArtifactCorruptError("command_job_mismatch")
            if (
                isinstance(command, (UploadCommand, ReindexCommand))
                and command.derivation_config_digest != self._derivation_digest
            ):
                raise ArtifactCorruptError("command_config_changed")
            result, _ = await self._execute(job, command)
            with suppress(ArtifactStoreError, OSError, UnsafeDataPathError):
                self._artifacts.delete_command(job.job_id)
            return result
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                raise
            for revision in self._repositories.index_revisions.list():
                if (
                    revision.ingestion_job_id == job.job_id
                    and revision.status is IndexRevisionStatus.STAGED
                ):
                    with suppress(RepositoryError):
                        self._repositories.index_revisions.mark_failed(revision.revision_id)
            failed = self._fail_job(job.job_id, _safe_error_code(error))
            with suppress(ArtifactStoreError, OSError, UnsafeDataPathError):
                self._artifacts.delete_command(job.job_id)
            return failed

    async def _execute(
        self,
        initial_job: IngestionJob,
        command: IngestionCommand,
    ) -> tuple[IngestionJob, str | None]:
        active_at_start = self._repositories.index_revisions.get_active()
        expected_active_id = active_at_start.revision_id if active_at_start is not None else None
        job = initial_job
        if (
            isinstance(command, DeleteCommand)
            and active_at_start is not None
            and active_at_start.ingestion_job_id == job.job_id
            and command.source_id not in active_at_start.active_sources
        ):
            document = self._repositories.documents.get(command.source_id)
            if document is None or document.deleted_at is not None:
                job, started = self._enter_stage(job, IngestionStage.PUBLISHING)
                job = self._record_stage(job, IngestionStage.PUBLISHING, started)
                completed = await self._purge_deleted_source(
                    job,
                    command.source_id,
                    active_at_start,
                )
                if completed is None:
                    raise IngestionRecoveryError("delete_job_completion_missing")
                return completed, active_at_start.revision_id
        if (
            isinstance(command, UploadCommand)
            and active_at_start is not None
            and not self._stager.is_compatible(active_at_start)
        ):
            raise IngestionSubmissionError("reindex_required")
        if isinstance(command, UploadCommand):
            job, versions, duplicate = await self._prepare_upload(job, command)
            if duplicate:
                if active_at_start is None:
                    raise IngestionRecoveryError("active_revision_missing")
                return self._complete_duplicate(job, active_at_start.revision_id), None
        elif isinstance(command, ReindexCommand):
            job, versions = await self._prepare_reindex(job)
        else:
            job, versions = self._prepare_delete(job, command)

        if any(
            version.derivation_config_digest != self._derivation_digest
            for version in versions.values()
        ):
            raise IngestionSubmissionError("reindex_required")

        job, parents, chunks, titles, active_sources = self._build_snapshot(job, versions)
        revision_id = self._new_opaque_id(self._revision_id_factory, "revision_id")
        context = ProviderCallContext(
            request_id=job.job_id,
            operation_id=revision_id,
            deadline=Deadline.after(self._operation_deadline_seconds),
        )
        job, embedding_started = self._enter_stage(job, IngestionStage.EMBEDDING)
        current_job = job
        stage_started = embedding_started

        def progress(phase: str) -> None:
            nonlocal current_job, stage_started
            if phase == "indexing":
                current_job = self._record_stage(
                    current_job,
                    IngestionStage.EMBEDDING,
                    stage_started,
                )
                current_job, stage_started = self._enter_stage(
                    current_job,
                    IngestionStage.INDEXING,
                )

        try:
            revision = await self._stager.stage(
                revision_id,
                chunks,
                titles,
                active_sources,
                context,
                parents=parents,
                ingestion_job_id=job.job_id,
                progress_hook=progress,
            )
        except BaseException:
            self._record_stage(current_job, current_job.stage, stage_started)
            raise
        job = self._record_stage(current_job, current_job.stage, stage_started)
        try:
            with self._database.transaction() as connection:
                self._repositories.index_revisions.create(revision, connection=connection)
                self._repositories.parent_chunks.insert_many(
                    revision_id,
                    parents,
                    connection=connection,
                )
        except BaseException:
            await self._remove_revision_artifacts(revision_id)
            raise

        job, publishing_started = self._enter_stage(job, IngestionStage.PUBLISHING)
        await self._chroma_worker_pool.run_cancel_safe(self._publisher.validate, revision)
        job = self._record_stage(job, IngestionStage.PUBLISHING, publishing_started)
        published = await self._chroma_worker_pool.run_cancel_safe(
            self._publisher.publish,
            revision_id,
            expected_active_revision_id=expected_active_id,
            ingestion_job_id=(None if isinstance(command, DeleteCommand) else job.job_id),
            job_ocr_page_count=(None if isinstance(command, DeleteCommand) else job.ocr_page_count),
            job_chunk_count=(None if isinstance(command, DeleteCommand) else job.chunk_count),
        )
        if isinstance(command, DeleteCommand):
            completed = await self._purge_deleted_source(job, command.source_id, published)
            if completed is None:
                raise IngestionRecoveryError("delete_job_completion_missing")
            return completed, published.revision_id
        persisted_job = self._repositories.ingestion_jobs.get(job.job_id)
        if persisted_job is None:
            raise RepositoryNotFound("ingestion_job_not_found")
        return persisted_job, published.revision_id

    async def _purge_deleted_source(
        self,
        job: IngestionJob | None,
        source_id: str,
        active_revision: IndexRevision,
    ) -> IngestionJob | None:
        """Remove retired index snapshots, artifacts, and SQLite document rows."""

        persisted_active = self._repositories.index_revisions.get_active()
        if (
            persisted_active is None
            or persisted_active.revision_id != active_revision.revision_id
            or source_id in persisted_active.active_sources
        ):
            raise IngestionRecoveryError("delete_purge_not_safe")

        versions = self._repositories.documents.list_versions(source_id)
        retired_revisions = tuple(
            revision
            for revision in self._repositories.index_revisions.list()
            if revision.revision_id != active_revision.revision_id
            and source_id in revision.active_sources
        )

        for revision in retired_revisions:
            await self._remove_revision_artifacts(revision.revision_id)
        for version in versions:
            self._artifacts.cleanup(
                StoredVersionArtifacts(
                    source_artifact_path=version.source_artifact_path,
                    canonical_artifact_path=version.canonical_artifact_path,
                )
            )

        with self._database.transaction() as connection:
            current = self._repositories.index_revisions.get_active(connection=connection)
            if (
                current is None
                or current.revision_id != active_revision.revision_id
                or source_id in current.active_sources
            ):
                raise IngestionRecoveryError("delete_purge_not_safe")

            completed: IngestionJob | None = None
            if job is not None:
                current_job = self._repositories.ingestion_jobs.get(
                    job.job_id,
                    connection=connection,
                )
                if current_job is None:
                    raise RepositoryNotFound("ingestion_job_not_found")
                completed = IngestionJob.model_validate(
                    {
                        **current_job.model_dump(),
                        "status": IngestionJobStatus.SUCCEEDED,
                        "stage": IngestionStage.COMPLETE,
                        "active_index_revision": active_revision.revision_id,
                    }
                )
                completed = self._repositories.ingestion_jobs.transition(
                    completed,
                    connection=connection,
                )

            for revision in retired_revisions:
                persisted = self._repositories.index_revisions.get(
                    revision.revision_id,
                    connection=connection,
                )
                if persisted is None:
                    continue
                if persisted.status is IndexRevisionStatus.ACTIVE:
                    raise IngestionRecoveryError("delete_purge_not_safe")
                if source_id in persisted.active_sources:
                    connection.execute(
                        "DELETE FROM index_revisions WHERE revision_id = ?",
                        (persisted.revision_id,),
                    )
            self._repositories.documents.delete_permanently(
                source_id,
                connection=connection,
            )
            return completed

    async def _prepare_upload(
        self,
        job: IngestionJob,
        command: UploadCommand,
    ) -> tuple[IngestionJob, dict[str, DocumentVersion], bool]:
        job, started = self._enter_stage(job, IngestionStage.VALIDATING)
        content = self._artifacts.load_upload(
            command.job_id,
            expected_size=command.payload_size,
            expected_digest=command.payload_digest,
        )
        upload = validate_upload(
            command.filename,
            content,
            declared_media_type=command.media_type,
            max_bytes=self._upload_max_bytes,
        )
        if upload.kind is not command.kind or upload.media_type != command.media_type:
            raise ArtifactCorruptError("command_upload_mismatch")
        job = self._record_stage(job, IngestionStage.VALIDATING, started)

        if any(
            self._require_version(
                document.source_id, document.active_version
            ).derivation_config_digest
            != self._derivation_digest
            for document in self._repositories.documents.list_active()
        ):
            raise IngestionSubmissionError("reindex_required")

        if job.source_id is not None and job.document_version is not None:
            version = self._require_assigned_upload_version(job, command)
            canonical = self._artifacts.load_canonical(version)
            job, started = self._enter_stage(job, IngestionStage.EXTRACTING)
            job = self._record_stage(
                job,
                IngestionStage.EXTRACTING,
                started,
                ocr_page_count=max(job.ocr_page_count, canonical.ocr_page_count),
            )
            job = self._zero_stage(job, IngestionStage.NORMALIZING)
            active = self._repositories.documents.get(version.source_id)
            duplicate = active is not None and active.active_version == version.version
            return job, self._desired_versions(version), duplicate

        job, started = self._enter_stage(job, IngestionStage.EXTRACTING)
        extracted = await self._extract(upload)
        job = self._record_stage(
            job,
            IngestionStage.EXTRACTING,
            started,
            ocr_page_count=extracted.ocr_page_count,
        )
        job, started = self._enter_stage(job, IngestionStage.NORMALIZING)
        normalized = normalize_document(extracted)
        existing = self._repositories.documents.get_by_source_key(command.source_key)
        if existing is not None and existing.kind is not upload.kind:
            raise IngestionSubmissionError("source_kind_mismatch")
        registration = self._versioning.register(
            source_key=command.source_key,
            upload=upload,
            extracted_document=normalized,
            derivation_config=self._derivation_config,
            display_title=command.display_title,
        )
        job = self._record_stage(
            job,
            IngestionStage.NORMALIZING,
            started,
            source_id=registration.document.source_id,
            document_version=registration.version.version,
        )
        return (
            job,
            self._desired_versions(registration.version),
            registration.disposition is SourceVersionDisposition.DUPLICATE,
        )

    async def _prepare_reindex(
        self,
        job: IngestionJob,
    ) -> tuple[IngestionJob, dict[str, DocumentVersion]]:
        job = self._zero_stage(job, IngestionStage.VALIDATING)
        active_documents = self._repositories.documents.list_active()
        versions: dict[str, DocumentVersion] = {}
        job, extraction_started = self._enter_stage(job, IngestionStage.EXTRACTING)
        ocr_count = 0
        normalized_documents: list[tuple[Document, ValidatedUpload, ExtractedDocument]] = []
        for document in active_documents:
            if document.active_version is None:
                raise IngestionRecoveryError("active_document_invalid")
            active_version = self._require_version(document.source_id, document.active_version)
            if active_version.derivation_config_digest == self._derivation_digest:
                self._load_canonical_for(document, active_version)
                versions[document.source_id] = active_version
                continue
            assigned = self._assigned_reindex_version(
                job,
                document.source_id,
            )
            if assigned is not None:
                if assigned.derivation_config_digest != self._derivation_digest:
                    raise IngestionRecoveryError("job_version_mismatch")
                canonical = self._load_canonical_for(document, assigned)
                ocr_count += canonical.ocr_page_count
                versions[document.source_id] = assigned
                continue
            content = self._artifacts.load_source(active_version)
            upload = validate_upload(
                active_version.original_filename,
                content,
                declared_media_type=active_version.media_type,
                max_bytes=max(self._upload_max_bytes, active_version.size_bytes),
            )
            extracted = await self._extract(upload)
            ocr_count += extracted.ocr_page_count
            normalized_documents.append((document, upload, normalize_document(extracted)))

        job = self._record_stage(
            job,
            IngestionStage.EXTRACTING,
            extraction_started,
            ocr_page_count=max(job.ocr_page_count, ocr_count),
        )
        job, normalization_started = self._enter_stage(job, IngestionStage.NORMALIZING)
        for document, upload, raw_normalized in normalized_documents:
            normalized = raw_normalized
            registration = self._versioning.register(
                source_key=document.source_key,
                upload=upload,
                extracted_document=normalized,
                derivation_config=self._derivation_config,
                display_title=document.display_title,
            )
            versions[document.source_id] = registration.version
        if len(versions) == 1:
            only = next(iter(versions.values()))
            job = self._record_stage(
                job,
                IngestionStage.NORMALIZING,
                normalization_started,
                source_id=only.source_id,
                document_version=only.version,
            )
        else:
            job = self._record_stage(
                job,
                IngestionStage.NORMALIZING,
                normalization_started,
            )
        return job, versions

    def _prepare_delete(
        self,
        job: IngestionJob,
        command: DeleteCommand,
    ) -> tuple[IngestionJob, dict[str, DocumentVersion]]:
        job, started = self._enter_stage(job, IngestionStage.VALIDATING)
        document = self._repositories.documents.get(command.source_id)
        if document is None or document.deleted_at is not None or document.active_version is None:
            raise IngestionSubmissionError("source_not_active")
        if _stable_source_key(document.source_key) != command.source_key:
            raise ArtifactCorruptError("command_source_mismatch")
        job = self._record_stage(
            job,
            IngestionStage.VALIDATING,
            started,
            source_id=document.source_id,
            document_version=document.active_version,
        )
        job = self._zero_stage(job, IngestionStage.EXTRACTING)
        job = self._zero_stage(job, IngestionStage.NORMALIZING)
        versions = {
            item.source_id: self._require_version(item.source_id, item.active_version)
            for item in self._repositories.documents.list_active()
            if item.source_id != command.source_id and item.active_version is not None
        }
        return job, versions

    def _build_snapshot(
        self,
        job: IngestionJob,
        versions: Mapping[str, DocumentVersion],
    ) -> tuple[
        IngestionJob,
        tuple[ParentChunk, ...],
        tuple[Chunk, ...],
        dict[str, str],
        dict[str, int],
    ]:
        job, started = self._enter_stage(job, IngestionStage.CHUNKING)
        parents: list[ParentChunk] = []
        chunks: list[Chunk] = []
        source_chunk_counts: dict[str, int] = {}
        titles: dict[str, str] = {}
        active_sources: dict[str, int] = {}
        for source_id in sorted(versions):
            version = versions[source_id]
            document = self._repositories.documents.get(source_id)
            if document is None:
                raise RepositoryNotFound("snapshot_document_missing")
            canonical = self._load_canonical_for(document, version)
            hierarchy = chunk_document_hierarchy(
                canonical,
                source_id=source_id,
                document_version=version.version,
                config=self._chunking,
            )
            parents.extend(hierarchy.parents)
            chunks.extend(hierarchy.children)
            source_chunk_counts[source_id] = len(hierarchy.children)
            titles[source_id] = document.display_title
            active_sources[source_id] = version.version
        if job.operation is IngestionOperation.UPLOAD and job.source_id is not None:
            job_chunk_count = source_chunk_counts.get(job.source_id, 0)
        elif job.operation is IngestionOperation.DELETE:
            job_chunk_count = 0
        else:
            job_chunk_count = len(chunks)
        job = self._record_stage(
            job,
            IngestionStage.CHUNKING,
            started,
            chunk_count=job_chunk_count,
        )
        return job, tuple(parents), tuple(chunks), titles, active_sources

    def _desired_versions(self, candidate: DocumentVersion) -> dict[str, DocumentVersion]:
        desired = {
            document.source_id: self._require_version(
                document.source_id,
                document.active_version,
            )
            for document in self._repositories.documents.list_active()
            if document.active_version is not None
        }
        desired[candidate.source_id] = candidate
        return desired

    def _require_assigned_upload_version(
        self,
        job: IngestionJob,
        command: UploadCommand,
    ) -> DocumentVersion:
        if job.source_id is None or job.document_version is None:
            raise IngestionRecoveryError("job_version_incomplete")
        document = self._repositories.documents.get(job.source_id)
        version = self._require_version(job.source_id, job.document_version)
        if (
            document is None
            or document.source_key != command.source_key
            or document.kind is not command.kind
            or version.derivation_config_digest != command.derivation_config_digest
        ):
            raise IngestionRecoveryError("job_version_mismatch")
        self._load_canonical_for(document, version)
        return version

    def _assigned_reindex_version(
        self,
        job: IngestionJob,
        source_id: str,
    ) -> DocumentVersion | None:
        if job.source_id != source_id or job.document_version is None:
            return None
        version = self._require_version(source_id, job.document_version)
        if version.derivation_config_digest != self._derivation_digest:
            raise IngestionRecoveryError("job_version_mismatch")
        return version

    def _complete_duplicate(self, job: IngestionJob, active_revision_id: str) -> IngestionJob:
        if job.source_id is None or job.document_version is None:
            raise IngestionRecoveryError("job_version_incomplete")
        version = self._require_version(job.source_id, job.document_version)
        job, started = self._enter_stage(job, IngestionStage.CHUNKING)
        duplicate_chunks = chunk_document(
            self._artifacts.load_canonical(version),
            source_id=version.source_id,
            document_version=version.version,
            config=self._chunking,
        )
        job = self._record_stage(
            job,
            IngestionStage.CHUNKING,
            started,
            chunk_count=len(duplicate_chunks),
        )
        for stage in (
            IngestionStage.EMBEDDING,
            IngestionStage.INDEXING,
            IngestionStage.PUBLISHING,
        ):
            job = self._zero_stage(job, stage)
        completed = IngestionJob.model_validate(
            {
                **job.model_dump(),
                "status": IngestionJobStatus.SUCCEEDED,
                "stage": IngestionStage.COMPLETE,
                "active_index_revision": active_revision_id,
                "warnings": (*job.warnings, "duplicate"),
            }
        )
        return self._repositories.ingestion_jobs.transition(completed)

    async def _validate_active_state(self) -> str | None:
        try:
            pointer_id = self._repositories.index_revisions.get_active_revision_id()
            active_status_ids = self._repositories.index_revisions.list_active_status_ids()
            active = self._repositories.index_revisions.get_active()
            all_documents = self._repositories.documents.list(include_deleted=True)
            active_documents = self._repositories.documents.list_active()
            indexed_mapping = self._repositories.documents.get_active_mapping()
            payload_active_documents = {
                document.source_id: document
                for document in all_documents
                if document.deleted_at is None and document.active_version is not None
            }
            if {document.source_id for document in active_documents} != set(
                payload_active_documents
            ) or any(
                document.deleted_at is not None or document.active_version is None
                for document in active_documents
            ):
                raise IngestionRecoveryError("active_document_metadata_mismatch")
            mapping = {
                document.source_id: document.active_version
                for document in payload_active_documents.values()
            }
            if mapping != indexed_mapping:
                raise IngestionRecoveryError("active_document_mapping_mismatch")
            if active is None:
                if pointer_id is not None or mapping or active_status_ids:
                    raise IngestionRecoveryError("active_revision_missing")
                return None
            if active_status_ids != [active.revision_id]:
                raise IngestionRecoveryError("active_revision_status_invalid")
            if pointer_id != active.revision_id:
                raise IngestionRecoveryError("active_revision_pointer_mismatch")
            if active.status is not IndexRevisionStatus.ACTIVE:
                raise IngestionRecoveryError("active_revision_status_invalid")
            if mapping != active.active_sources:
                raise IngestionRecoveryError("active_document_mapping_mismatch")
            for source_id, version_number in active.active_sources.items():
                version = self._require_version(source_id, version_number)
                self._load_canonical_for(payload_active_documents[source_id], version)
            await self._chroma_worker_pool.run_cancel_safe(
                self._publisher.validate_artifacts,
                active,
            )
            return active.revision_id
        except IngestionRecoveryError:
            raise
        except Exception:
            raise IngestionRecoveryError("active_revision_invalid") from None

    async def _extract(self, upload: ValidatedUpload) -> ExtractedDocument:
        if upload.kind is DocumentKind.PDF:
            return await self._ocr_worker_pool.run(
                extract_pdf,
                upload.content,
                ocr=self._ocr,
                languages=self._ocr_languages,
                usability=self._page_usability,
            )
        return extract_utf8_text(upload.content, kind=upload.kind)

    def _require_version(self, source_id: str, version: int | None) -> DocumentVersion:
        if version is None:
            raise RepositoryNotFound("document_version_missing")
        result = self._repositories.documents.get_version(source_id, version)
        if result is None:
            raise RepositoryNotFound("document_version_missing")
        if result.source_id != source_id or result.version != version:
            raise RepositoryError("document version identity is invalid")
        return result

    def _load_canonical_for(
        self,
        document: Document,
        version: DocumentVersion,
    ) -> ExtractedDocument:
        canonical = self._artifacts.load_canonical(version)
        if version.source_id != document.source_id or canonical.kind is not document.kind:
            raise ArtifactCorruptError("canonical_artifact_kind_mismatch")
        return canonical

    def _enter_stage(
        self,
        job: IngestionJob,
        stage: IngestionStage,
    ) -> tuple[IngestionJob, float]:
        processing = IngestionJob.model_validate(
            {
                **job.model_dump(),
                "status": IngestionJobStatus.PROCESSING,
                "stage": stage,
            }
        )
        return self._repositories.ingestion_jobs.transition(processing), time.monotonic()

    def _record_stage(
        self,
        job: IngestionJob,
        stage: IngestionStage,
        started: float,
        **updates: object,
    ) -> IngestionJob:
        elapsed = max(0.0, (time.monotonic() - started) * 1000)
        updated = IngestionJob.model_validate(
            {
                **job.model_dump(),
                **updates,
                "stage_timings_ms": {stage.value: elapsed},
            }
        )
        return self._repositories.ingestion_jobs.transition(updated)

    def _zero_stage(self, job: IngestionJob, stage: IngestionStage) -> IngestionJob:
        job, started = self._enter_stage(job, stage)
        return self._record_stage(job, stage, started)

    def _fail_job(self, job_id: str, code: str) -> IngestionJob:
        current = self._repositories.ingestion_jobs.get(job_id)
        if current is None:
            raise RepositoryNotFound("ingestion_job_not_found")
        if current.status in {IngestionJobStatus.SUCCEEDED, IngestionJobStatus.FAILED}:
            return current
        active = self._repositories.index_revisions.get_active()
        failed = IngestionJob.model_validate(
            {
                **current.model_dump(),
                "status": IngestionJobStatus.FAILED,
                "stage": IngestionStage.FAILED,
                "safe_error_code": code,
                "active_index_revision": (
                    current.active_index_revision
                    if current.active_index_revision is not None or active is None
                    else active.revision_id
                ),
            }
        )
        return self._repositories.ingestion_jobs.transition(failed)

    def _new_opaque_id(self, factory: Callable[[], str], field_name: str) -> str:
        value = factory()
        try:
            if field_name == "job_id":
                self._layout.job_path(value)
            else:
                self._layout.index_revision_path(value)
        except UnsafeDataPathError:
            raise ValueError(f"{field_name}_factory_invalid") from None
        return value


def _default_derivation_config(
    *,
    ocr: OcrAdapter,
    ocr_languages: str,
    usability: PageUsabilityPolicy,
    chunking: ChunkingConfig,
) -> dict[str, JsonValue]:
    return cast(
        dict[str, JsonValue],
        {
            "extraction": {"version": INDEX_EXTRACTION_VERSION},
            "ocr": {
                "adapter_version": ocr.version,
                "languages": ocr_languages,
                "usability_version": usability.version,
                "minimum_alphanumeric_characters": usability.minimum_alphanumeric_characters,
                "minimum_printable_ratio": usability.minimum_printable_ratio,
            },
            "normalization": {"version": NORMALIZATION_VERSION},
            "chunking": {
                "version": chunking.version,
                "tokenizer_version": chunking.tokenizer_version,
                "target_tokens": chunking.target_tokens,
                "overlap_tokens": chunking.overlap_tokens,
                "parent_target_tokens": chunking.parent_target_tokens,
            },
            "tokenizer": {"version": BILINGUAL_TOKENIZER_IDENTITY},
        },
    )


def _stable_source_key(value: str) -> str:
    try:
        normalized = unicodedata.normalize("NFC", value).strip()
        encoded = normalized.encode("utf-8")
    except (TypeError, UnicodeError):
        raise IngestionSubmissionError("source_key_invalid") from None
    if not normalized:
        raise IngestionSubmissionError("source_key_invalid")
    if _SAFE_KEY.fullmatch(normalized):
        return normalized
    return "key_" + hashlib.sha256(encoded).hexdigest()[:32]


def _validated_title(value: str) -> str:
    try:
        normalized = unicodedata.normalize("NFC", value).strip()
        encoded = normalized.encode("utf-8")
    except (TypeError, UnicodeError):
        raise IngestionSubmissionError("display_title_invalid") from None
    if not normalized or len(encoded) > 255:
        raise IngestionSubmissionError("display_title_invalid")
    if any(unicodedata.category(character) == "Cc" for character in normalized):
        raise IngestionSubmissionError("display_title_invalid")
    return normalized


def _safe_error_code(error: BaseException) -> str:
    known = (
        UploadValidationError,
        ExtractionError,
        EmbeddingStageError,
        IndexingError,
        ArtifactNotFoundError,
        ArtifactCorruptError,
        IngestionSubmissionError,
        IngestionRecoveryError,
    )
    if isinstance(error, known):
        code = getattr(error, "code", "")
        if isinstance(code, str) and _SAFE_CODE.fullmatch(code):
            return code
    if isinstance(error, DeletedSourceError):
        return "source_deleted"
    if isinstance(error, RepositoryConflict):
        return "repository_conflict"
    if isinstance(error, RepositoryNotFound):
        return "repository_not_found"
    if isinstance(error, (ArtifactStoreError, UnsafeDataPathError)):
        return "artifact_invalid"
    return "ingestion_internal_error"
