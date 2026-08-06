from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_mvp.domain.ingestion import (
    Chunk,
    ChunkLocator,
    Document,
    DocumentKind,
    DocumentVersion,
    EmbeddingSpaceIdentity,
    ExtractionMethod,
)
from rag_mvp.ingestion.embedding import EmbeddingStage
from rag_mvp.ingestion.indexing import RevisionStager
from rag_mvp.providers.fakes import DeterministicEmbeddingProvider
from rag_mvp.providers.models import (
    Deadline,
    EmbeddingRequest,
    EmbeddingResult,
    NormalizationPolicy,
    ProviderCallContext,
)
from rag_mvp.providers.models import (
    EmbeddingSpaceIdentity as ProviderEmbeddingSpaceIdentity,
)
from rag_mvp.providers.resilience import RetryPolicy
from rag_mvp.providers.routing import ModelProviderRouter, ProviderRoute
from rag_mvp.retrieval.binding import BoundRetrievalSnapshot, BoundRetrievalSnapshotFactory
from rag_mvp.retrieval.dense import DenseIndexError, PersistentChromaIndex
from rag_mvp.retrieval.identity import provider_embedding_identity
from rag_mvp.retrieval.query_dense import BoundDenseRetriever
from rag_mvp.storage.database import Database
from rag_mvp.storage.embedding_cache import EmbeddingCache
from rag_mvp.storage.layout import DataLayout
from rag_mvp.storage.repositories import KnowledgeRepositories


def _identity(model: str = "embedding-model") -> EmbeddingSpaceIdentity:
    return EmbeddingSpaceIdentity(
        provider_alias="test",
        model=model,
        dimension=3,
        normalization="none",
        adapter_version="v1",
    )


def _provider_identity(model: str = "embedding-model") -> ProviderEmbeddingSpaceIdentity:
    return ProviderEmbeddingSpaceIdentity(
        provider="test",
        model=model,
        dimension=3,
        normalization=NormalizationPolicy.NONE,
        adapter_version="v1",
    )


def _chunk(chunk_id: str, text: str, ordinal: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_id="source-1",
        document_version=1,
        ordinal=ordinal,
        text=text,
        content_digest=hashlib.sha256(text.encode()).hexdigest(),
        locator=ChunkLocator(pages=(1,)),
    )


def _context() -> ProviderCallContext:
    return ProviderCallContext(
        request_id="request-query",
        operation_id="dense-query",
        deadline=Deadline.after(30),
    )


async def _bound_snapshot(tmp_path: Path) -> BoundRetrievalSnapshot:
    layout = DataLayout.from_root(tmp_path / "data")
    layout.initialize()
    database = Database(layout.metadata_db)
    database.initialize()
    repositories = KnowledgeRepositories.from_database(database)
    repositories.documents.create(
        Document(
            source_id="source-1",
            source_key="policy",
            display_title="Handbook",
            media_type="text/plain",
            kind=DocumentKind.TEXT,
        )
    )
    repositories.documents.add_version(
        DocumentVersion(
            source_id="source-1",
            version=1,
            content_digest="document-content-v1",
            derivation_config_digest="derivation-config-v1",
            original_filename="policy.txt",
            media_type="text/plain",
            size_bytes=10,
            source_artifact_path="sources/source-1/1/source.txt",
            canonical_artifact_path="canonical/source-1/1/document.json",
            extraction_method=ExtractionMethod.TEXT,
        )
    )
    provider = DeterministicEmbeddingProvider(_provider_identity())
    with EmbeddingCache(layout.directory("caches") / "embeddings.sqlite3") as cache:
        staged = await RevisionStager(layout, EmbeddingStage(provider, cache)).stage(
            "revision-bound",
            (
                _chunk("chunk-a", "  annual leave\n", 0),
                _chunk("chunk-b", "network security", 1),
            ),
            {"source-1": "Handbook"},
            {"source-1": 1},
            _context(),
        )
    repositories.index_revisions.create(staged)
    repositories.index_revisions.publish(
        staged.revision_id,
        published_at=datetime.now(UTC),
        expected_active_revision_id=None,
    )
    return BoundRetrievalSnapshotFactory(layout, repositories.index_revisions).bind()


class ResultProvider:
    def __init__(self, result: object) -> None:
        self._identity = _provider_identity()
        self.result = result
        self.call_count = 0
        self.requests: list[EmbeddingRequest] = []
        self.contexts: list[ProviderCallContext] = []

    @property
    def identity(self) -> ProviderEmbeddingSpaceIdentity:
        return self._identity

    async def embed(self, request: EmbeddingRequest, context: ProviderCallContext) -> object:
        self.call_count += 1
        self.requests.append(request)
        self.contexts.append(context)
        return self.result


async def test_chroma_search_is_persistent_and_deterministic(tmp_path: Path) -> None:
    index = PersistentChromaIndex(
        tmp_path / "chroma",
        collection_name="revision_1",
        identity=_identity(),
    )
    index.add(
        (_chunk("chunk-a", "annual leave"), _chunk("chunk-b", "network security", 1)),
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        {"source-1": "Handbook"},
    )

    reopened = PersistentChromaIndex(
        tmp_path / "chroma",
        collection_name="revision_1",
        identity=_identity(),
    )
    results = reopened.query([1.0, 0.0, 0.0], query_identity=_identity(), limit=2)

    assert reopened.chunk_ids == {"chunk-a", "chunk-b"}
    assert [result.chunk_id for result in results] == ["chunk-a", "chunk-b"]
    assert [result.dense_rank for result in results] == [1, 2]
    assert results[0].revision_id == "revision_1"
    assert results[0].ordinal == 0
    assert results[0].content_digest == _chunk("chunk-a", "annual leave").content_digest
    assert results[0].record_digest == reopened.record_digests["chunk-a"]
    reopened.close()


async def test_incompatible_query_identity_fails_before_query(tmp_path: Path) -> None:
    index = PersistentChromaIndex(
        tmp_path / "chroma",
        collection_name="revision_1",
        identity=_identity(),
    )

    with pytest.raises(DenseIndexError, match="embedding_identity_mismatch"):
        index.query([1.0, 0.0, 0.0], query_identity=_identity("other-model"), limit=1)
    index.close()


async def test_bound_dense_retriever_embeds_exactly_one_query_and_returns_identity(
    tmp_path: Path,
) -> None:
    snapshot = await _bound_snapshot(tmp_path)
    provider = DeterministicEmbeddingProvider(
        provider_embedding_identity(snapshot.revision.embedding_space)
    )
    retriever = BoundDenseRetriever(snapshot, provider, _context())

    results = await retriever.search("annual leave", 2)

    assert provider.call_count == 1
    assert retriever.revision_id == snapshot.revision_id
    whitespace_result = next(result for result in results if result.chunk_id == "chunk-a")
    assert whitespace_result.text == "  annual leave\n"
    assert whitespace_result.revision_id == snapshot.revision_id
    assert whitespace_result.content_digest is not None
    assert whitespace_result.record_digest is not None
    snapshot.close()


async def test_bound_dense_submits_one_canonical_text_with_call_context(tmp_path: Path) -> None:
    snapshot = await _bound_snapshot(tmp_path)
    identity = provider_embedding_identity(snapshot.revision.embedding_space)
    provider = ResultProvider(EmbeddingResult(((1.0, 0.0, 0.0),), identity))
    context = _context()
    retriever = BoundDenseRetriever(snapshot, provider, context)  # type: ignore[arg-type]

    await retriever.search("  policy\nterms ", 1)

    assert provider.call_count == 1
    assert provider.requests[0].texts == ("policy terms",)
    assert provider.contexts == [context]
    snapshot.close()


async def test_direct_provider_mismatch_makes_zero_provider_and_chroma_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = await _bound_snapshot(tmp_path)
    provider = DeterministicEmbeddingProvider(_provider_identity("other-model"))
    query_calls = 0

    def query_spy(*args: object, **kwargs: object) -> tuple[object, ...]:
        nonlocal query_calls
        del args, kwargs
        query_calls += 1
        return ()

    monkeypatch.setattr(snapshot.dense, "query", query_spy)
    with pytest.raises(DenseIndexError, match="embedding_identity_mismatch"):
        BoundDenseRetriever(snapshot, provider, _context())

    assert provider.call_count == 0
    assert query_calls == 0
    snapshot.close()


async def test_router_without_compatible_space_makes_zero_provider_and_chroma_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = await _bound_snapshot(tmp_path)
    provider = DeterministicEmbeddingProvider(_provider_identity("other-model"))
    router = ModelProviderRouter(
        embedding_routes=(
            ProviderRoute(
                "incompatible",
                provider,
                RetryPolicy(attempt_timeout_seconds=1),
            ),
        )
    )
    query_calls = 0

    def query_spy(*args: object, **kwargs: object) -> tuple[object, ...]:
        nonlocal query_calls
        del args, kwargs
        query_calls += 1
        return ()

    monkeypatch.setattr(snapshot.dense, "query", query_spy)
    retriever = BoundDenseRetriever(snapshot, router, _context())
    with pytest.raises(DenseIndexError, match="embedding_identity_mismatch"):
        await retriever.search("policy", 1)

    assert provider.call_count == 0
    assert query_calls == 0
    snapshot.close()


@pytest.mark.parametrize(
    ("result_factory", "code"),
    [
        (
            lambda identity: SimpleNamespace(
                identity=_provider_identity("other-model"),
                vectors=((0.0, 0.0, 0.0),),
            ),
            "embedding_identity_mismatch",
        ),
        (lambda identity: SimpleNamespace(identity=identity, vectors=()), "count_mismatch"),
        (
            lambda identity: SimpleNamespace(identity=identity, vectors=((0.0,),)),
            "dimension_mismatch",
        ),
        (
            lambda identity: SimpleNamespace(
                identity=identity,
                vectors=((0.0, float("nan"), 0.0),),
            ),
            "vector_nonfinite",
        ),
    ],
)
async def test_bound_dense_rejects_malformed_provider_results_before_chroma(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result_factory: object,
    code: str,
) -> None:
    snapshot = await _bound_snapshot(tmp_path)
    identity = provider_embedding_identity(snapshot.revision.embedding_space)
    result = result_factory(identity)  # type: ignore[operator]
    provider = ResultProvider(result)
    query_calls = 0

    def query_spy(*args: object, **kwargs: object) -> tuple[object, ...]:
        nonlocal query_calls
        del args, kwargs
        query_calls += 1
        return ()

    monkeypatch.setattr(snapshot.dense, "query", query_spy)
    retriever = BoundDenseRetriever(snapshot, provider, _context())  # type: ignore[arg-type]
    with pytest.raises(DenseIndexError, match=code):
        await retriever.search("policy", 1)

    assert provider.call_count == 1
    assert query_calls == 0
    snapshot.close()


def _create_tied_index(
    path: Path,
    revision_id: str,
    insertion_order: tuple[str, ...],
) -> None:
    chunks = {
        "chunk-a": _chunk("chunk-a", "alpha", 0),
        "chunk-b": _chunk("chunk-b", "beta", 1),
        "chunk-c": _chunk("chunk-c", "gamma", 2),
    }
    index = PersistentChromaIndex.create_new(path, revision_id=revision_id, identity=_identity())
    index.add(
        tuple(chunks[chunk_id] for chunk_id in insertion_order),
        tuple((1.0, 0.0, 0.0) for _ in insertion_order),
        {"source-1": "Handbook"},
    )
    index.seal()
    index.close()


async def test_identical_vector_cutoff_is_global_stable_and_insertion_independent(
    tmp_path: Path,
) -> None:
    expected = ["chunk-a", "chunk-b"]
    for number, insertion_order in enumerate(
        (("chunk-c", "chunk-a", "chunk-b"), ("chunk-b", "chunk-c", "chunk-a")),
        start=1,
    ):
        path = tmp_path / f"chroma-{number}"
        revision_id = f"revision-{number}"
        _create_tied_index(path, revision_id, insertion_order)

        for _ in range(3):
            with PersistentChromaIndex.open_existing(
                path,
                revision_id=revision_id,
                identity=_identity(),
            ) as reopened:
                results = reopened.query(
                    [1.0, 0.0, 0.0],
                    query_identity=_identity(),
                    limit=2,
                )
                assert [candidate.chunk_id for candidate in results] == expected


async def test_boundary_whitespace_survives_chunk_json_and_dense_reopen(tmp_path: Path) -> None:
    text = "  leading and trailing\n"
    chunk = _chunk("chunk-space", text)
    assert Chunk.model_validate_json(chunk.model_dump_json()).text == text
    path = tmp_path / "chroma-space"
    index = PersistentChromaIndex.create_new(
        path,
        revision_id="revision-space",
        identity=_identity(),
    )
    index.add((chunk,), ((1.0, 0.0, 0.0),), {"source-1": "Handbook"})
    index.seal()
    index.close()

    with PersistentChromaIndex.open_existing(
        path,
        revision_id="revision-space",
        identity=_identity(),
    ) as reopened:
        result = reopened.query([1.0, 0.0, 0.0], query_identity=_identity(), limit=1)[0]
        assert result.text == text
        assert result.ordinal == chunk.ordinal
        assert result.content_digest == chunk.content_digest
        assert result.record_digest == reopened.record_digests[chunk.chunk_id]


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_dense_limit_must_be_a_positive_integer(tmp_path: Path, limit: object) -> None:
    index = PersistentChromaIndex(
        tmp_path / "chroma-limit",
        collection_name="revision_limit",
        identity=_identity(),
    )
    with pytest.raises(ValueError, match="limit must be positive"):
        index.query([1.0, 0.0, 0.0], query_identity=_identity(), limit=limit)  # type: ignore[arg-type]
    index.close()
