from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from rag_mvp.domain.ingestion import (
    Chunk,
    ChunkLocator,
    Document,
    DocumentKind,
    DocumentVersion,
    ExtractionMethod,
    IndexRevision,
    IndexRevisionStatus,
    IngestionJob,
    IngestionJobStatus,
    IngestionStage,
    ParentChunk,
)
from rag_mvp.ingestion.chunking import token_spans
from rag_mvp.ingestion.embedding import EmbeddingStage
from rag_mvp.ingestion.indexing import IndexingError, RevisionPublisher, RevisionStager
from rag_mvp.providers.fakes import DeterministicEmbeddingProvider
from rag_mvp.providers.models import Deadline, ProviderCallContext
from rag_mvp.retrieval.bm25 import PersistentBm25Index
from rag_mvp.retrieval.dense import PersistentChromaIndex
from rag_mvp.storage.database import Database
from rag_mvp.storage.embedding_cache import EmbeddingCache
from rag_mvp.storage.layout import DataLayout
from rag_mvp.storage.repositories import KnowledgeRepositories, RepositoryConflict


class InjectedFailure(RuntimeError):
    pass


def _fail_at(target: str) -> Callable[[str], None]:
    def fail(phase: str) -> None:
        if phase == target:
            raise InjectedFailure(phase)

    return fail


def _context(operation: str) -> ProviderCallContext:
    return ProviderCallContext(
        request_id=f"request-{operation}",
        operation_id=operation,
        deadline=Deadline.after(30),
    )


def _chunk(chunk_id: str, source_id: str, version: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        parent_chunk_id=f"parent-{chunk_id}",
        source_id=source_id,
        document_version=version,
        ordinal=0,
        text=text,
        content_digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        locator=ChunkLocator(pages=(version,)),
    )


def _parents(chunks: tuple[Chunk, ...]) -> tuple[ParentChunk, ...]:
    return tuple(
        ParentChunk(
            parent_chunk_id=chunk.parent_chunk_id,
            source_id=chunk.source_id,
            document_version=chunk.document_version,
            ordinal=chunk.ordinal,
            text=chunk.text,
            content_digest=chunk.content_digest,
            locator=chunk.locator,
            token_count=len(token_spans(chunk.text)),
        )
        for chunk in chunks
    )


def _register(
    database: Database,
    repositories: KnowledgeRepositories,
    revision: IndexRevision,
    parents: tuple[ParentChunk, ...],
) -> None:
    with database.transaction() as connection:
        repositories.index_revisions.create(revision, connection=connection)
        repositories.parent_chunks.insert_many(
            revision.revision_id,
            parents,
            connection=connection,
        )


def _document(source_id: str, source_key: str, title: str) -> Document:
    return Document(
        source_id=source_id,
        source_key=source_key,
        display_title=title,
        media_type="text/plain",
        kind=DocumentKind.TEXT,
    )


def _version(source_id: str, version: int) -> DocumentVersion:
    return DocumentVersion(
        source_id=source_id,
        version=version,
        content_digest=f"content-{source_id}-{version}",
        derivation_config_digest="derivation-v1",
        original_filename=f"{source_id}.txt",
        media_type="text/plain",
        size_bytes=10,
        source_artifact_path=f"sources/{source_id}/{version}/source.txt",
        canonical_artifact_path=f"canonical/{source_id}/{version}/document.json",
        extraction_method=ExtractionMethod.TEXT,
    )


async def test_publication_rejects_revision_without_persisted_parents(tmp_path: Path) -> None:
    layout = DataLayout.from_root(tmp_path / "missing-parent")
    layout.initialize()
    database = Database(layout.metadata_db)
    database.initialize()
    repositories = KnowledgeRepositories.from_database(database)
    chunk = _chunk("chunk-one", "source-one", 1, "parent integrity")

    with EmbeddingCache(layout.directory("caches") / "embeddings.sqlite3") as cache:
        revision = await RevisionStager(
            layout,
            EmbeddingStage(DeterministicEmbeddingProvider(), cache),
        ).stage(
            "revision-missing-parent-row",
            (chunk,),
            {"source-one": "One"},
            {"source-one": 1},
            _context("missing-parent-row"),
            parents=_parents((chunk,)),
        )

    repositories.index_revisions.create(revision)
    with pytest.raises(IndexingError, match="revision_parent_inventory_mismatch"):
        RevisionPublisher(layout, repositories.index_revisions).validate(revision)


async def test_publication_is_atomic_across_all_failure_points_and_success(
    tmp_path: Path,
) -> None:
    layout = DataLayout.from_root(tmp_path / "data")
    layout.initialize()
    database = Database(layout.metadata_db)
    database.initialize()
    repositories = KnowledgeRepositories.from_database(database)

    repositories.documents.create(_document("source-a", "policy-a", "Policy A"))
    repositories.documents.add_version(_version("source-a", 1))
    repositories.documents.add_version(_version("source-a", 2))
    repositories.documents.create(_document("source-b", "policy-b", "Policy B"))
    repositories.documents.add_version(_version("source-b", 1))

    old_chunks = (
        _chunk("chunk-a-v1", "source-a", 1, "Policy code OLD-101 grants ten leave days"),
        _chunk("chunk-b-v1", "source-b", 1, "Legacy policy RETAIN-222 remains active"),
    )
    new_chunks = (
        _chunk("chunk-a-v2", "source-a", 2, "Policy code NEW-202 grants twelve leave days"),
    )
    old_titles = {"source-a": "Policy A", "source-b": "Policy B"}
    new_titles = {"source-a": "Policy A"}

    with EmbeddingCache(layout.directory("caches") / "embeddings.sqlite3") as cache:
        provider = DeterministicEmbeddingProvider()
        embeddings = EmbeddingStage(provider, cache)
        stager = RevisionStager(layout, embeddings)
        revision_one = await stager.stage(
            "revision-1",
            old_chunks,
            old_titles,
            {"source-a": 1, "source-b": 1},
            _context("stage-one"),
            parents=_parents(old_chunks),
        )
        old_vector = (await embeddings.embed(old_chunks, _context("old-vector"))).vectors[0]
        _register(database, repositories, revision_one, _parents(old_chunks))
        published_one = RevisionPublisher(layout, repositories.index_revisions).publish(
            revision_one.revision_id,
            expected_active_revision_id=None,
        )
        assert published_one.status is IndexRevisionStatus.ACTIVE
        assert not layout.active_manifest.exists()

        async def assert_revision_one_unchanged() -> None:
            active = repositories.index_revisions.get_active()
            assert active is not None
            assert active.revision_id == "revision-1"
            source_a = repositories.documents.get("source-a")
            source_b = repositories.documents.get("source-b")
            assert source_a is not None and source_a.active_version == 1
            assert source_a.deleted_at is None
            assert source_b is not None and source_b.active_version == 1
            assert source_b.deleted_at is None
            assert not layout.active_manifest.exists()
            await assert_old_revision_queryable(old_vector, layout, revision_one)

        for phase in ("after_dense", "after_bm25", "after_parity"):
            failed_id = f"failed-{phase.replace('_', '-')}"
            failing_stager = RevisionStager(
                layout,
                embeddings,
                failure_hook=_fail_at(phase),
            )
            with pytest.raises(InjectedFailure, match=phase):
                await failing_stager.stage(
                    failed_id,
                    new_chunks,
                    new_titles,
                    {"source-a": 2},
                    _context(failed_id),
                    parents=_parents(new_chunks),
                )
            assert not layout.index_revision_path(failed_id).exists()
            await assert_revision_one_unchanged()

        pretransaction = await stager.stage(
            "revision-pretransaction",
            new_chunks,
            new_titles,
            {"source-a": 2},
            _context("pretransaction"),
            parents=_parents(new_chunks),
        )
        _register(database, repositories, pretransaction, _parents(new_chunks))
        with pytest.raises(InjectedFailure, match="pretransaction"):
            RevisionPublisher(
                layout,
                repositories.index_revisions,
                failure_hook=_fail_at("pretransaction"),
            ).publish(
                pretransaction.revision_id,
                expected_active_revision_id="revision-1",
            )
        assert repositories.index_revisions.get(pretransaction.revision_id) == pretransaction
        await assert_revision_one_unchanged()

        inside = await stager.stage(
            "revision-inside-transaction",
            new_chunks,
            new_titles,
            {"source-a": 2},
            _context("inside-transaction"),
            parents=_parents(new_chunks),
        )
        _register(database, repositories, inside, _parents(new_chunks))
        with pytest.raises(InjectedFailure, match="inside_transaction"):
            RevisionPublisher(
                layout,
                repositories.index_revisions,
                failure_hook=_fail_at("inside_transaction"),
            ).publish(
                inside.revision_id,
                expected_active_revision_id="revision-1",
            )
        assert repositories.index_revisions.get(inside.revision_id) == inside
        await assert_revision_one_unchanged()

        stale = await stager.stage(
            "revision-stale-base",
            new_chunks,
            new_titles,
            {"source-a": 2},
            _context("stale-base"),
            parents=_parents(new_chunks),
        )
        _register(database, repositories, stale, _parents(new_chunks))
        with pytest.raises(RepositoryConflict, match="active index revision changed"):
            RevisionPublisher(layout, repositories.index_revisions).publish(
                stale.revision_id,
                expected_active_revision_id=None,
            )
        await assert_revision_one_unchanged()
        with pytest.raises(RepositoryConflict, match="active index revision changed"):
            RevisionPublisher(layout, repositories.index_revisions).publish(
                stale.revision_id,
                expected_active_revision_id="revision-does-not-exist",
            )
        assert repositories.index_revisions.get(stale.revision_id) == stale
        await assert_revision_one_unchanged()

        revision_two = await stager.stage(
            "revision-2",
            new_chunks,
            new_titles,
            {"source-a": 2},
            _context("stage-two"),
            parents=_parents(new_chunks),
        )
        _register(database, repositories, revision_two, _parents(new_chunks))
        queued = IngestionJob(
            job_id="job-publish-two",
            source_key="policy-a",
            source_id="source-a",
            document_version=2,
        )
        repositories.ingestion_jobs.create(queued)
        processing = IngestionJob.model_validate(
            {
                **queued.model_dump(),
                "status": IngestionJobStatus.PROCESSING,
                "stage": IngestionStage.PUBLISHING,
            }
        )
        repositories.ingestion_jobs.transition(processing)

        published_two = RevisionPublisher(layout, repositories.index_revisions).publish(
            revision_two.revision_id,
            expected_active_revision_id="revision-1",
            ingestion_job_id=queued.job_id,
            job_ocr_page_count=3,
        )

    assert published_two.status is IndexRevisionStatus.ACTIVE
    assert repositories.index_revisions.get_active() == published_two
    old_revision = repositories.index_revisions.get("revision-1")
    assert old_revision is not None
    assert old_revision.status is IndexRevisionStatus.SUPERSEDED
    source_a = repositories.documents.get("source-a")
    source_b = repositories.documents.get("source-b")
    assert source_a is not None and source_a.active_version == 2
    assert source_a.deleted_at is None
    assert source_b is not None and source_b.active_version is None
    assert source_b.deleted_at is not None
    job = repositories.ingestion_jobs.get("job-publish-two")
    assert job is not None
    assert job.status is IngestionJobStatus.SUCCEEDED
    assert job.stage is IngestionStage.COMPLETE
    assert job.ocr_page_count == 3
    assert job.chunk_count == revision_two.chunk_count
    assert job.active_index_revision == "revision-2"
    assert not layout.active_manifest.exists()

    await assert_old_revision_queryable(old_vector, layout, revision_one)
    lexical_two = PersistentBm25Index.load(
        layout.lexical_index_path(revision_two.revision_id),
        expected_revision_id=revision_two.revision_id,
    )
    assert (await lexical_two.search("NEW-202", 5))[0].chunk_id == "chunk-a-v2"
    assert await lexical_two.search("RETAIN-222", 5) == ()


async def assert_old_revision_queryable(
    old_vector: tuple[float, ...],
    layout: DataLayout,
    revision: IndexRevision,
) -> None:
    with PersistentChromaIndex.open_existing(
        layout.dense_index_path(revision.revision_id),
        revision_id=revision.revision_id,
        identity=revision.embedding_space,
    ) as dense:
        dense_results = dense.query(
            old_vector,
            query_identity=revision.embedding_space,
            limit=2,
        )
    lexical = PersistentBm25Index.load(
        layout.lexical_index_path(revision.revision_id),
        expected_revision_id=revision.revision_id,
    )
    lexical_results = await lexical.search("OLD-101", 5)
    assert dense_results[0].chunk_id == "chunk-a-v1"
    assert lexical_results[0].chunk_id == "chunk-a-v1"
