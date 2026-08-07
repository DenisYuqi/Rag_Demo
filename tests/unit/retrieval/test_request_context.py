from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag_mvp.domain.ingestion import (
    Chunk,
    ChunkLocator,
    Document,
    DocumentKind,
    DocumentVersion,
    EmbeddingSpaceIdentity,
    ExtractionMethod,
    IndexRevision,
    IndexRevisionStatus,
    ParentChunk,
)
from rag_mvp.domain.retrieval import RetrievalMode
from rag_mvp.ingestion.chunking import token_spans
from rag_mvp.ingestion.embedding import EmbeddingStage
from rag_mvp.ingestion.indexing import RevisionPublisher, RevisionStager
from rag_mvp.providers.fakes import DeterministicEmbeddingProvider
from rag_mvp.providers.models import Deadline, ProviderCallContext
from rag_mvp.retrieval.binding import BoundRetrievalSnapshotFactory
from rag_mvp.retrieval.request import (
    RetrievalRequestContext,
    RetrievalRequestError,
    canonicalize_query,
)
from rag_mvp.storage.database import Database
from rag_mvp.storage.embedding_cache import EmbeddingCache
from rag_mvp.storage.layout import DataLayout
from rag_mvp.storage.repositories import KnowledgeRepositories


def _provider_context(operation: str) -> ProviderCallContext:
    return ProviderCallContext(
        request_id=f"request-{operation}",
        operation_id=operation,
        deadline=Deadline.after(30),
    )


def _chunk(chunk_id: str, text: str, *, version: int = 1) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        parent_chunk_id=f"parent-{chunk_id}",
        source_id="source-policy",
        document_version=version,
        ordinal=0,
        text=text,
        content_digest=hashlib.sha256(text.encode()).hexdigest(),
        locator=ChunkLocator(pages=(version,)),
    )


def _parent(chunk: Chunk) -> ParentChunk:
    return ParentChunk(
        parent_chunk_id=chunk.parent_chunk_id,
        source_id=chunk.source_id,
        document_version=chunk.document_version,
        ordinal=chunk.ordinal,
        text=chunk.text,
        content_digest=chunk.content_digest,
        locator=chunk.locator,
        token_count=len(token_spans(chunk.text)),
    )


def _register(
    repositories: KnowledgeRepositories,
    revision: IndexRevision,
    chunk: Chunk,
) -> None:
    with repositories.index_revisions.database.transaction() as connection:
        repositories.index_revisions.create(revision, connection=connection)
        repositories.parent_chunks.insert_many(
            revision.revision_id,
            (_parent(chunk),),
            connection=connection,
        )


def _initialize_repository(
    tmp_path: Path,
) -> tuple[DataLayout, KnowledgeRepositories, EmbeddingCache, RevisionStager]:
    layout = DataLayout.from_root(tmp_path / "data")
    layout.initialize()
    database = Database(layout.metadata_db)
    database.initialize()
    repositories = KnowledgeRepositories.from_database(database)
    repositories.documents.create(
        Document(
            source_id="source-policy",
            source_key="policy",
            display_title="Policy",
            media_type="text/plain",
            kind=DocumentKind.TEXT,
        )
    )
    repositories.documents.add_version(
        DocumentVersion(
            source_id="source-policy",
            version=1,
            content_digest="document-content-v1",
            derivation_config_digest="derivation-config-v1",
            original_filename="policy.txt",
            media_type="text/plain",
            size_bytes=10,
            source_artifact_path="sources/source-policy/1/source.txt",
            canonical_artifact_path="canonical/source-policy/1/document.json",
            extraction_method=ExtractionMethod.TEXT,
        )
    )
    cache = EmbeddingCache(layout.directory("caches") / "embeddings.sqlite3")
    stager = RevisionStager(
        layout,
        EmbeddingStage(DeterministicEmbeddingProvider(), cache),
    )
    return layout, repositories, cache, stager


async def _stage(
    stager: RevisionStager,
    revision_id: str,
    chunk: Chunk,
) -> IndexRevision:
    return await stager.stage(
        revision_id,
        (chunk,),
        {"source-policy": "Policy"},
        {"source-policy": 1},
        _provider_context(revision_id),
        parents=(_parent(chunk),),
    )


def _uncommitted_revision(status: IndexRevisionStatus) -> IndexRevision:
    return IndexRevision(
        revision_id=f"revision-{status.value}",
        status=status,
        active_sources={},
        chunk_set_digest="empty-chunk-set",
        embedding_space=EmbeddingSpaceIdentity(
            provider_alias="test",
            model="embedding",
            dimension=3,
            normalization="none",
            adapter_version="v1",
        ),
        extraction_version="v1",
        chunking_version="v1",
        tokenizer_version="tokenizer-v1",
        dense_index_path=f"indexes/revisions/revision-{status.value}/chroma",
        lexical_index_path=f"indexes/revisions/revision-{status.value}/bm25.json",
    )


def test_query_is_unicode_canonical_and_normal_whitespace_is_collapsed() -> None:
    assert canonicalize_query("  Cafe\u0301\n\tpolicy\r\u2003terms  ") == "Café policy terms"


@pytest.mark.parametrize("query", ["a\x00b", "a\x1fb", "a\u200bb", "a\ud800b"])
def test_query_rejects_controls_format_characters_and_surrogates(query: str) -> None:
    with pytest.raises(RetrievalRequestError, match="invalid_query"):
        canonicalize_query(query)


def test_query_checks_raw_length_before_whitespace_collapse() -> None:
    with pytest.raises(RetrievalRequestError, match="query_too_long"):
        canonicalize_query("a   b", maximum_characters=3)
    with pytest.raises(RetrievalRequestError, match="query_too_long"):
        canonicalize_query("\u0344", maximum_characters=1)
    assert canonicalize_query("a b", maximum_characters=3) == "a b"


@pytest.mark.parametrize("maximum", [0, -1, True, 2.5, "10"])
def test_query_limit_must_be_a_positive_integer(maximum: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        canonicalize_query("valid", maximum_characters=maximum)  # type: ignore[arg-type]


def test_query_requires_a_string() -> None:
    with pytest.raises(RetrievalRequestError, match="invalid_query"):
        canonicalize_query(123)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("values", "code"),
    [
        (("bad id", "query", "dense", "revision-1"), "invalid_request_id"),
        (("request-1", "query", "unknown", "revision-1"), "invalid_retrieval_mode"),
        (("request-1", "query", "dense", "../revision"), "invalid_revision_id"),
        (("request-1", "\x00", "dense", "revision-1"), "invalid_query"),
    ],
)
def test_direct_context_construction_cannot_bypass_validation(
    values: tuple[object, object, object, object],
    code: str,
) -> None:
    with pytest.raises(RetrievalRequestError, match=code):
        RetrievalRequestContext(*values)  # type: ignore[arg-type]


def test_direct_context_cannot_inject_a_revision_identity() -> None:
    with pytest.raises(TypeError):
        RetrievalRequestContext(
            "request-1",
            "policy",
            RetrievalMode.DENSE,
            "revision-active",
            revision=_uncommitted_revision(IndexRevisionStatus.STAGED),
        )  # type: ignore[call-arg]


def test_normal_bind_rejects_no_active_and_caller_forged_revision_ids() -> None:
    with pytest.raises(RetrievalRequestError, match="index_not_ready"):
        RetrievalRequestContext.bind(
            request_id="request-1",
            query="policy",
            mode="dense",
        )
    with pytest.raises(RetrievalRequestError, match="untrusted_revision_binding"):
        RetrievalRequestContext.bind(
            request_id="request-1",
            query="policy",
            mode="dense",
            active_revision_id="revision-forged",
        )


async def test_snapshot_context_honors_configured_limit_above_default(tmp_path: Path) -> None:
    layout, repositories, cache, stager = _initialize_repository(tmp_path)
    chunk = _chunk("chunk", "policy")
    staged = await _stage(stager, "revision-limit", chunk)
    _register(repositories, staged, chunk)
    RevisionPublisher(layout, repositories.index_revisions).publish(
        staged.revision_id,
        expected_active_revision_id=None,
    )
    with BoundRetrievalSnapshotFactory(layout, repositories.index_revisions).bind() as snapshot:
        query = "x" * 4097
        context = RetrievalRequestContext.from_snapshot(
            request_id="request-limit",
            query=query,
            mode="dense",
            snapshot=snapshot,
            maximum_characters=5000,
        )
        assert context.query == query
    cache.close()


def test_factory_rejects_no_active_and_uncommitted_revisions(tmp_path: Path) -> None:
    layout = DataLayout.from_root(tmp_path / "data")
    layout.initialize()
    database = Database(layout.metadata_db)
    database.initialize()
    factory = BoundRetrievalSnapshotFactory(
        layout,
        KnowledgeRepositories.from_database(database).index_revisions,
    )

    with pytest.raises(RetrievalRequestError, match="index_not_ready"):
        factory.bind()
    for status in (IndexRevisionStatus.STAGED, IndexRevisionStatus.FAILED):
        revision = _uncommitted_revision(status)
        factory.revisions.create(revision)
        with pytest.raises(RetrievalRequestError, match="revision_not_committed"):
            factory.open_committed(revision.revision_id)
    with pytest.raises(RetrievalRequestError, match="index_not_ready"):
        factory.bind()


def test_active_manifest_with_missing_artifacts_has_stable_error(tmp_path: Path) -> None:
    layout = DataLayout.from_root(tmp_path / "data")
    layout.initialize()
    database = Database(layout.metadata_db)
    database.initialize()
    repositories = KnowledgeRepositories.from_database(database)
    staged = _uncommitted_revision(IndexRevisionStatus.STAGED)
    repositories.index_revisions.create(staged)
    repositories.index_revisions.publish(
        staged.revision_id,
        published_at=datetime.now(UTC),
        expected_active_revision_id=None,
    )

    with pytest.raises(RetrievalRequestError) as captured:
        BoundRetrievalSnapshotFactory(layout, repositories.index_revisions).bind()
    assert captured.value.code == "index_artifact_missing"


async def test_bound_snapshot_survives_publication_and_owns_dense_handle(
    tmp_path: Path,
) -> None:
    layout, repositories, cache, stager = _initialize_repository(tmp_path)
    publisher = RevisionPublisher(layout, repositories.index_revisions)
    first_chunk = _chunk("chunk-old", "OLD-101 policy")
    first = await _stage(stager, "revision-1", first_chunk)
    _register(repositories, first, first_chunk)
    published_first = publisher.publish(first.revision_id, expected_active_revision_id=None)
    factory = BoundRetrievalSnapshotFactory(layout, repositories.index_revisions)

    snapshot = factory.bind()
    context = RetrievalRequestContext.from_snapshot(
        request_id="request-1",
        query="  OLD-101\npolicy ",
        mode="hybrid",
        snapshot=snapshot,
    )
    second_chunk = _chunk("chunk-new", "NEW-202 policy")
    second = await _stage(stager, "revision-2", second_chunk)
    _register(repositories, second, second_chunk)
    publisher.publish(second.revision_id, expected_active_revision_id=published_first.revision_id)

    persisted_first = repositories.index_revisions.get(first.revision_id)
    assert persisted_first is not None
    assert persisted_first.status is IndexRevisionStatus.SUPERSEDED
    assert snapshot.revision.status is IndexRevisionStatus.ACTIVE
    assert context.revision == snapshot.revision
    assert context.query == "OLD-101 policy"
    assert context.mode is RetrievalMode.HYBRID
    context.assert_matches_snapshot(snapshot)
    assert (await snapshot.bm25.search("OLD-101", 5))[0].chunk_id == "chunk-old"
    assert await snapshot.bm25.search("NEW-202", 5) == ()
    with factory.bind() as current:
        assert current.revision_id == "revision-2"
        with pytest.raises(RetrievalRequestError, match="snapshot_context_mismatch"):
            context.assert_matches_snapshot(current)

    snapshot.close()
    assert snapshot.is_closed
    with pytest.raises(RetrievalRequestError, match="snapshot_context_mismatch"):
        context.assert_matches_snapshot(snapshot)
    with pytest.raises(ValueError, match="dense_index_closed"):
        snapshot.dense.query(
            [0.0] * snapshot.revision.embedding_space.dimension,
            query_identity=snapshot.revision.embedding_space,
            limit=1,
        )
    with factory.open_committed(persisted_first.revision_id) as reopened_superseded:
        assert reopened_superseded.revision.status is IndexRevisionStatus.SUPERSEDED
        assert (await reopened_superseded.bm25.search("OLD-101", 5))[0].chunk_id == "chunk-old"
    cache.close()


async def test_empty_active_snapshot_is_valid(tmp_path: Path) -> None:
    layout = DataLayout.from_root(tmp_path / "data")
    layout.initialize()
    database = Database(layout.metadata_db)
    database.initialize()
    repositories = KnowledgeRepositories.from_database(database)
    with EmbeddingCache(layout.directory("caches") / "embeddings.sqlite3") as cache:
        staged = await RevisionStager(
            layout,
            EmbeddingStage(DeterministicEmbeddingProvider(), cache),
        ).stage(
            "revision-empty",
            (),
            {},
            {},
            _provider_context("empty"),
            parents=(),
        )
        repositories.index_revisions.create(staged)
        RevisionPublisher(layout, repositories.index_revisions).publish(
            staged.revision_id,
            expected_active_revision_id=None,
        )

        with BoundRetrievalSnapshotFactory(layout, repositories.index_revisions).bind() as snapshot:
            assert snapshot.revision_id == "revision-empty"
            assert snapshot.dense.chunk_ids == frozenset()
            assert snapshot.bm25.chunk_ids == frozenset()


async def test_corrupt_active_artifact_has_stable_error(tmp_path: Path) -> None:
    layout, repositories, cache, stager = _initialize_repository(tmp_path)
    chunk = _chunk("chunk", "policy")
    staged = await _stage(stager, "revision-corrupt", chunk)
    _register(repositories, staged, chunk)
    RevisionPublisher(layout, repositories.index_revisions).publish(
        staged.revision_id,
        expected_active_revision_id=None,
    )
    layout.lexical_index_path(staged.revision_id).write_text("{corrupt", encoding="utf-8")

    with pytest.raises(RetrievalRequestError) as captured:
        BoundRetrievalSnapshotFactory(layout, repositories.index_revisions).bind()
    assert captured.value.code == "index_artifact_invalid"
    assert captured.value.detail_code == "invalid_snapshot"
    cache.close()


async def test_corrupt_active_manifest_has_distinct_stable_error(tmp_path: Path) -> None:
    layout, repositories, cache, stager = _initialize_repository(tmp_path)
    chunk = _chunk("chunk", "policy")
    staged = await _stage(stager, "revision-manifest", chunk)
    _register(repositories, staged, chunk)
    RevisionPublisher(layout, repositories.index_revisions).publish(
        staged.revision_id,
        expected_active_revision_id=None,
    )
    with repositories.index_revisions.database.transaction() as connection:
        connection.execute(
            "UPDATE index_revisions SET status = ? WHERE revision_id = ?",
            (IndexRevisionStatus.SUPERSEDED.value, staged.revision_id),
        )

    with pytest.raises(RetrievalRequestError) as captured:
        BoundRetrievalSnapshotFactory(layout, repositories.index_revisions).bind()
    assert captured.value.code == "index_manifest_invalid"
    cache.close()
