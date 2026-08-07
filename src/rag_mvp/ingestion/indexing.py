"""Complete-snapshot index staging, validation, and atomic publication."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from rag_mvp.domain.ingestion import (
    Chunk,
    EmbeddingSpaceIdentity,
    IndexRevision,
    IndexRevisionStatus,
    ParentChunk,
)
from rag_mvp.ingestion.chunking import CHUNKING_VERSION
from rag_mvp.ingestion.embedding import EmbeddingStage
from rag_mvp.performance.worker_pools import RagWorkerPools, default_worker_pools
from rag_mvp.providers.models import ProviderCallContext
from rag_mvp.retrieval.bm25 import PersistentBm25Index
from rag_mvp.retrieval.dense import PersistentChromaIndex
from rag_mvp.retrieval.identity import domain_embedding_identity
from rag_mvp.retrieval.snapshot import (
    RECORD_DIGEST_ALGORITHM,
    chunk_record_digest,
    chunk_set_digest,
    parent_chunk_record_digest,
    parent_set_digest,
)
from rag_mvp.retrieval.tokenizer import BilingualTokenizer
from rag_mvp.storage.layout import DataLayout
from rag_mvp.storage.repositories import (
    IndexRevisionRepository,
    ParentChunkRepository,
    RepositoryNotFound,
)

INDEX_EXTRACTION_VERSION = "extraction-v1"

type FailureHook = Callable[[str], None]
type ProgressHook = Callable[[str], None]

_EXPECTED_ACTIVE_UNSET = object()


class IndexingError(ValueError):
    """A safe full-snapshot validation or staging error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class RevisionStager:
    """Build a closed, parity-validated dense and lexical snapshot."""

    def __init__(
        self,
        layout: DataLayout,
        embedding_stage: EmbeddingStage,
        *,
        extraction_version: str = INDEX_EXTRACTION_VERSION,
        chunking_version: str = CHUNKING_VERSION,
        tokenizer: BilingualTokenizer | None = None,
        failure_hook: FailureHook | None = None,
        worker_pools: RagWorkerPools | None = None,
    ) -> None:
        if not extraction_version or not chunking_version:
            raise ValueError("index_version_invalid")
        self._layout = layout
        self._embedding_stage = embedding_stage
        self._embedding_space = _embedding_identity(embedding_stage)
        self._extraction_version = extraction_version
        self._chunking_version = chunking_version
        self._tokenizer = tokenizer or BilingualTokenizer()
        self._failure_hook = failure_hook
        resolved_worker_pools = worker_pools or default_worker_pools()
        self._chroma_worker_pool = resolved_worker_pools.chroma
        self._bm25_worker_pool = resolved_worker_pools.bm25

    def is_compatible(self, revision: IndexRevision) -> bool:
        return (
            revision.embedding_space == self._embedding_space
            and revision.extraction_version == self._extraction_version
            and revision.chunking_version == self._chunking_version
            and revision.tokenizer_version == self._tokenizer.version
        )

    async def stage(
        self,
        revision_id: str,
        chunks: Sequence[Chunk],
        titles: Mapping[str, str],
        active_sources: Mapping[str, int],
        context: ProviderCallContext,
        *,
        parents: Sequence[ParentChunk] = (),
        ingestion_job_id: str | None = None,
        progress_hook: ProgressHook | None = None,
    ) -> IndexRevision:
        if _embedding_identity(self._embedding_stage) != self._embedding_space:
            raise IndexingError("embedding_identity_changed")
        ordered_chunks = tuple(chunks)
        ordered_parents = tuple(parents)
        normalized_titles = dict(titles)
        normalized_sources = dict(active_sources)
        expected_digests, expected_parent_digests = _validate_snapshot_inputs(
            ordered_parents,
            ordered_chunks,
            normalized_titles,
            normalized_sources,
        )
        expected_ids = frozenset(expected_digests)
        expected_set_digest = chunk_set_digest(expected_digests)
        expected_parent_set_digest = parent_set_digest(expected_parent_digests)
        revision_path = self._layout.index_revision_path(revision_id)
        dense_path = self._layout.dense_index_path(revision_id)
        lexical_path = self._layout.lexical_index_path(revision_id)
        if revision_path.exists():
            raise IndexingError("revision_path_exists")

        if progress_hook is not None:
            progress_hook("embedding")
        embedding_result = await self._embedding_stage.embed(ordered_chunks, context)
        if progress_hook is not None:
            progress_hook("indexing")
        if _embedding_identity(self._embedding_stage) != self._embedding_space:
            raise IndexingError("embedding_identity_changed")
        created_revision_path = False
        try:
            self._layout.index_revisions.mkdir(mode=0o700, parents=True, exist_ok=True)
            revision_path.mkdir(mode=0o700, exist_ok=False)
            created_revision_path = True

            await self._chroma_worker_pool.run_cancel_safe(
                _write_dense_snapshot,
                dense_path,
                revision_id,
                self._embedding_space,
                ordered_chunks,
                embedding_result.vectors,
                normalized_titles,
            )
            self._run_hook("after_dense")

            lexical = await self._bm25_worker_pool.run_cancel_safe(
                _write_lexical_snapshot,
                lexical_path,
                revision_id,
                ordered_chunks,
                normalized_titles,
                self._tokenizer,
            )
            self._run_hook("after_bm25")

            await self._chroma_worker_pool.run_cancel_safe(
                _validate_written_snapshot,
                dense_path,
                lexical_path,
                revision_id,
                self._embedding_space,
                expected_ids,
                expected_digests,
                expected_set_digest,
            )
            self._run_hook("after_parity")

            return IndexRevision(
                revision_id=revision_id,
                status=IndexRevisionStatus.STAGED,
                active_sources=normalized_sources,
                chunk_set_digest=expected_set_digest,
                embedding_space=self._embedding_space,
                extraction_version=self._extraction_version,
                chunking_version=self._chunking_version,
                tokenizer_version=self._tokenizer.version,
                dense_index_path=self._layout.dense_index_relative_path(revision_id),
                lexical_index_path=self._layout.lexical_index_relative_path(revision_id),
                chunk_count=len(expected_ids),
                parent_chunk_count=len(expected_parent_digests),
                parent_chunk_set_digest=expected_parent_set_digest,
                dense_schema_version=PersistentChromaIndex.SCHEMA_VERSION,
                dense_metric=PersistentChromaIndex.METRIC,
                lexical_schema_version=PersistentBm25Index.SNAPSHOT_SCHEMA,
                lexical_algorithm_version=PersistentBm25Index.ALGORITHM_VERSION,
                lexical_k1=lexical.k1,
                lexical_b=lexical.b,
                record_digest_algorithm=RECORD_DIGEST_ALGORITHM,
                ingestion_job_id=ingestion_job_id,
            )
        except BaseException:
            if created_revision_path:
                await self._chroma_worker_pool.run(
                    shutil.rmtree,
                    revision_path,
                    ignore_errors=True,
                )
            raise

    def _run_hook(self, phase: str) -> None:
        if self._failure_hook is not None:
            self._failure_hook(phase)


class RevisionPublisher:
    """Revalidate a staged snapshot and publish all metadata atomically."""

    def __init__(
        self,
        layout: DataLayout,
        revisions: IndexRevisionRepository,
        parents: ParentChunkRepository | None = None,
        *,
        failure_hook: FailureHook | None = None,
    ) -> None:
        self._layout = layout
        self._revisions = revisions
        self._parents = parents or ParentChunkRepository(revisions.database)
        self._failure_hook = failure_hook

    def publish(
        self,
        revision_id: str,
        *,
        expected_active_revision_id: str | object | None = _EXPECTED_ACTIVE_UNSET,
        ingestion_job_id: str | None = None,
        job_ocr_page_count: int | None = None,
        job_chunk_count: int | None = None,
    ) -> IndexRevision:
        revision = self._revisions.get(revision_id)
        if revision is None:
            raise RepositoryNotFound(f"index revision {revision_id!r} was not found")
        self.validate(revision)
        self._run_hook("pretransaction")

        with self._revisions.database.transaction() as connection:
            if expected_active_revision_id is _EXPECTED_ACTIVE_UNSET:
                published = self._revisions.publish(
                    revision_id,
                    connection=connection,
                    ingestion_job_id=ingestion_job_id,
                    job_ocr_page_count=job_ocr_page_count,
                    job_chunk_count=job_chunk_count,
                )
            else:
                expected = expected_active_revision_id
                if expected is not None and not isinstance(expected, str):
                    raise TypeError("expected_active_revision_id must be a string or None")
                published = self._revisions.publish(
                    revision_id,
                    connection=connection,
                    expected_active_revision_id=expected,
                    ingestion_job_id=ingestion_job_id,
                    job_ocr_page_count=job_ocr_page_count,
                    job_chunk_count=job_chunk_count,
                )
            self._run_hook("inside_transaction")
            return published

    def validate(self, revision: IndexRevision) -> None:
        if revision.status is not IndexRevisionStatus.STAGED:
            raise IndexingError("revision_not_staged")
        self.validate_artifacts(revision)

    def validate_artifacts(self, revision: IndexRevision) -> None:
        """Validate immutable artifacts for staged or already-active revisions."""

        if revision.status not in {IndexRevisionStatus.STAGED, IndexRevisionStatus.ACTIVE}:
            raise IndexingError("revision_status_invalid")
        expected_dense_path = self._layout.dense_index_path(revision.revision_id)
        expected_lexical_path = self._layout.lexical_index_path(revision.revision_id)
        if (
            self._layout.resolve_artifact_path(revision.dense_index_path) != expected_dense_path
            or self._layout.resolve_artifact_path(revision.lexical_index_path)
            != expected_lexical_path
        ):
            raise IndexingError("revision_path_mismatch")
        if (
            revision.dense_schema_version != PersistentChromaIndex.SCHEMA_VERSION
            or revision.dense_metric != PersistentChromaIndex.METRIC
            or revision.lexical_schema_version != PersistentBm25Index.SNAPSHOT_SCHEMA
            or revision.lexical_algorithm_version != PersistentBm25Index.ALGORITHM_VERSION
            or revision.record_digest_algorithm != RECORD_DIGEST_ALGORITHM
        ):
            raise IndexingError("revision_index_identity_mismatch")

        with PersistentChromaIndex.open_existing(
            expected_dense_path,
            revision_id=revision.revision_id,
            identity=revision.embedding_space,
        ) as dense:
            lexical = PersistentBm25Index.load(
                expected_lexical_path,
                expected_revision_id=revision.revision_id,
            )
            if (
                lexical.tokenizer_identity != revision.tokenizer_version
                or lexical.k1 != revision.lexical_k1
                or lexical.b != revision.lexical_b
            ):
                raise IndexingError("revision_lexical_config_mismatch")
            if any(
                revision.active_sources.get(record.chunk.source_id) != record.chunk.document_version
                for record in lexical.records
            ):
                raise IndexingError("revision_active_sources_mismatch")
            if {record.chunk.source_id for record in lexical.records} != set(
                revision.active_sources
            ):
                raise IndexingError("revision_active_sources_mismatch")
            _validate_parity(
                expected_ids=dense.chunk_ids,
                expected_digests=dense.record_digests,
                expected_chunk_set_digest=revision.chunk_set_digest,
                dense=dense,
                lexical=lexical,
            )
            if len(dense.chunk_ids) != revision.chunk_count:
                raise IndexingError("revision_chunk_count_mismatch")
            parent_count, parent_digest, _ = self._parents.inventory(revision.revision_id)
            if (
                parent_count != revision.parent_chunk_count
                or parent_digest != revision.parent_chunk_set_digest
            ):
                raise IndexingError("revision_parent_inventory_mismatch")
            parent_records = self._parents.get_many(
                revision.revision_id,
                tuple(record.chunk.parent_chunk_id for record in lexical.records),
            )
            referenced_parent_ids = {record.chunk.parent_chunk_id for record in lexical.records}
            if set(parent_records) != referenced_parent_ids or len(parent_records) != parent_count:
                raise IndexingError("revision_parent_reference_mismatch")
            for record in lexical.records:
                parent = parent_records[record.chunk.parent_chunk_id]
                if (
                    parent.source_id != record.chunk.source_id
                    or parent.document_version != record.chunk.document_version
                ):
                    raise IndexingError("revision_parent_reference_mismatch")

    def _run_hook(self, phase: str) -> None:
        if self._failure_hook is not None:
            self._failure_hook(phase)


def _write_dense_snapshot(
    dense_path: Path,
    revision_id: str,
    identity: EmbeddingSpaceIdentity,
    chunks: tuple[Chunk, ...],
    vectors: Sequence[Sequence[float]],
    titles: Mapping[str, str],
) -> None:
    dense: PersistentChromaIndex | None = None
    try:
        dense = PersistentChromaIndex.create_new(
            dense_path,
            revision_id=revision_id,
            identity=identity,
        )
        dense.add(chunks, vectors, titles)
        dense.seal()
    finally:
        if dense is not None:
            dense.close()


def _write_lexical_snapshot(
    lexical_path: Path,
    revision_id: str,
    chunks: tuple[Chunk, ...],
    titles: Mapping[str, str],
    tokenizer: BilingualTokenizer,
) -> PersistentBm25Index:
    lexical = PersistentBm25Index.build(
        chunks,
        titles,
        revision_id=revision_id,
        tokenizer=tokenizer,
    )
    lexical.save_new(lexical_path)
    return lexical


def _validate_written_snapshot(
    dense_path: Path,
    lexical_path: Path,
    revision_id: str,
    identity: EmbeddingSpaceIdentity,
    expected_ids: frozenset[str],
    expected_digests: Mapping[str, str],
    expected_chunk_set_digest: str,
) -> None:
    with PersistentChromaIndex.open_existing(
        dense_path,
        revision_id=revision_id,
        identity=identity,
    ) as reopened_dense:
        reopened_lexical = PersistentBm25Index.load(
            lexical_path,
            expected_revision_id=revision_id,
        )
        _validate_parity(
            expected_ids=expected_ids,
            expected_digests=expected_digests,
            expected_chunk_set_digest=expected_chunk_set_digest,
            dense=reopened_dense,
            lexical=reopened_lexical,
        )


def _validate_snapshot_inputs(
    parents: tuple[ParentChunk, ...],
    chunks: tuple[Chunk, ...],
    titles: Mapping[str, str],
    active_sources: Mapping[str, int],
) -> tuple[dict[str, str], dict[str, str]]:
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise IndexingError("duplicate_chunk_id")
    if len({parent.parent_chunk_id for parent in parents}) != len(parents):
        raise IndexingError("duplicate_parent_chunk_id")
    if any(
        not isinstance(source_id, str)
        or not source_id
        or isinstance(version, bool)
        or not isinstance(version, int)
        or version < 1
        for source_id, version in active_sources.items()
    ):
        raise IndexingError("active_sources_invalid")
    if any(not isinstance(title, str) or not title for title in titles.values()):
        raise IndexingError("display_title_invalid")
    chunk_source_ids = {chunk.source_id for chunk in chunks}
    if chunk_source_ids != set(active_sources):
        raise IndexingError("active_source_chunk_mismatch")
    if not set(active_sources) <= titles.keys():
        raise IndexingError("missing_display_title")
    parents_by_id = {parent.parent_chunk_id: parent for parent in parents}
    referenced_parent_ids = {chunk.parent_chunk_id for chunk in chunks}
    if referenced_parent_ids != set(parents_by_id):
        raise IndexingError("parent_child_inventory_mismatch")
    parent_source_ids = {parent.source_id for parent in parents}
    if parent_source_ids != set(active_sources):
        raise IndexingError("active_source_parent_mismatch")

    digests: dict[str, str] = {}
    try:
        for chunk in chunks:
            if active_sources.get(chunk.source_id) != chunk.document_version:
                raise IndexingError("chunk_not_in_active_sources")
            title = titles[chunk.source_id]
            parent = parents_by_id[chunk.parent_chunk_id]
            if (
                parent.source_id != chunk.source_id
                or parent.document_version != chunk.document_version
            ):
                raise IndexingError("chunk_parent_mismatch")
            digests[chunk.chunk_id] = chunk_record_digest(chunk, title)
    except KeyError:
        raise IndexingError("missing_display_title") from None
    parent_digests: dict[str, str] = {}
    for parent in parents:
        if active_sources.get(parent.source_id) != parent.document_version:
            raise IndexingError("parent_not_in_active_sources")
        parent_digests[parent.parent_chunk_id] = parent_chunk_record_digest(parent)
    return digests, parent_digests


def _validate_parity(
    *,
    expected_ids: frozenset[str],
    expected_digests: Mapping[str, str],
    expected_chunk_set_digest: str,
    dense: PersistentChromaIndex,
    lexical: PersistentBm25Index,
) -> None:
    if dense.chunk_ids != expected_ids:
        raise IndexingError("dense_chunk_id_mismatch")
    if lexical.chunk_ids != expected_ids:
        raise IndexingError("bm25_chunk_id_mismatch")
    if dense.record_digests != dict(expected_digests):
        raise IndexingError("dense_record_digest_mismatch")
    if lexical.record_digests != dict(expected_digests):
        raise IndexingError("bm25_record_digest_mismatch")
    if (
        dense.inventory_digest != expected_chunk_set_digest
        or lexical.chunk_set_digest != expected_chunk_set_digest
    ):
        raise IndexingError("chunk_set_digest_mismatch")


def _embedding_identity(stage: EmbeddingStage) -> EmbeddingSpaceIdentity:
    try:
        return domain_embedding_identity(stage.identity)
    except Exception:
        raise IndexingError("embedding_identity_invalid") from None
