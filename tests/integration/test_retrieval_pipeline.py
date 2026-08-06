from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from rag_mvp.domain.ingestion import ChunkLocator, IndexRevisionStatus, IngestionJobStatus
from rag_mvp.domain.retrieval import CacheOutcome, RetrievalMode, RetrievalResult
from rag_mvp.ingestion.chunking import ChunkingConfig
from rag_mvp.ingestion.service import IngestionService
from rag_mvp.providers.models import (
    Deadline,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingSpaceIdentity,
    NormalizationPolicy,
    ProviderCallContext,
    TokenUsage,
)
from rag_mvp.retrieval.binding import BoundRetrievalSnapshot, BoundRetrievalSnapshotFactory
from rag_mvp.retrieval.bm25 import LexicalIndexError
from rag_mvp.retrieval.collection import BoundBm25Retriever
from rag_mvp.retrieval.dense import DenseIndexError
from rag_mvp.retrieval.query_dense import BoundDenseRetriever, DenseSearchResult
from rag_mvp.retrieval.request import RetrievalRequestContext
from rag_mvp.retrieval.service import (
    RetrievalLimits,
    RetrievalService,
    RetrievalUnavailableError,
)
from rag_mvp.storage.layout import DataLayout

_IDENTITY = EmbeddingSpaceIdentity(
    provider="concept-test",
    model="bilingual-concepts-v1",
    dimension=8,
    normalization=NormalizationPolicy.L2,
    adapter_version="integration-v1",
)
_CONCEPTS = (
    (("sabbatical benefit", "长期休假福利"), 0),
    (("multifactor authentication", "双因素验证"), 1),
    (("release ledger",), 2),
    (("运维台账",), 3),
    (("neutral anchor",), 4),
    (("blue state", "historical state"), 5),
    (("green state", "current state"), 6),
)


class NeverOcr:
    version = "never-ocr-v1"

    def recognize(self, png_bytes: bytes, *, languages: str) -> str:
        del png_bytes, languages
        raise AssertionError("text-only integration tests must not invoke OCR")


class ConceptEmbeddingProvider:
    """Small deterministic semantic space with explicit bilingual concepts."""

    def __init__(
        self,
        identity: EmbeddingSpaceIdentity = _IDENTITY,
        *,
        blocked_query: str | None = None,
        query_started: asyncio.Event | None = None,
        query_release: asyncio.Event | None = None,
    ) -> None:
        self._identity = identity
        self._blocked_query = blocked_query
        self._query_started = query_started
        self._query_release = query_release
        self.call_count = 0

    @property
    def identity(self) -> EmbeddingSpaceIdentity:
        return self._identity

    async def embed(
        self,
        request: EmbeddingRequest,
        context: ProviderCallContext,
    ) -> EmbeddingResult:
        del context
        self.call_count += 1
        vectors = []
        for text in request.texts:
            if text == self._blocked_query:
                assert self._query_started is not None and self._query_release is not None
                self._query_started.set()
                await self._query_release.wait()
            vectors.append(_concept_vector(text, self.identity.dimension))
        return EmbeddingResult(
            tuple(vectors),
            self.identity,
            TokenUsage(
                input_tokens=sum(max(1, len(text) // 4) for text in request.texts),
                output_tokens=0,
            ),
        )


class _FaultingDenseRetriever:
    def __init__(self, delegate: BoundDenseRetriever) -> None:
        self._delegate = delegate

    @property
    def revision_id(self) -> str:
        return self._delegate.revision_id

    async def search(self, query: str, limit: int) -> tuple[object, ...]:
        del query, limit
        raise DenseIndexError("injected_dense_unavailable")

    async def search_with_diagnostics(self, query: str, limit: int) -> DenseSearchResult:
        del query, limit
        raise DenseIndexError("injected_dense_unavailable")


class _FaultingBm25Retriever:
    def __init__(self, delegate: BoundBm25Retriever) -> None:
        self._delegate = delegate

    @property
    def revision_id(self) -> str:
        return self._delegate.revision_id

    async def search(self, query: str, limit: int) -> tuple[object, ...]:
        del query, limit
        raise LexicalIndexError("injected_bm25_unavailable")


@dataclass(frozen=True, slots=True)
class IndexedSource:
    source_id: str
    version: int
    title: str
    text: str

    @property
    def locator(self) -> ChunkLocator:
        return ChunkLocator(char_start=0, char_end=len(self.text))


def _concept_vector(text: str, dimension: int) -> tuple[float, ...]:
    folded = text.casefold()
    concept = 4
    for terms, index in _CONCEPTS:
        if any(term.casefold() in folded for term in terms):
            concept = index
            break
    return tuple(1.0 if index == concept else 0.0 for index in range(dimension))


def _ingestion(root: Path, provider: ConceptEmbeddingProvider) -> IngestionService:
    return IngestionService.create(
        root,
        provider,
        ocr=NeverOcr(),
        chunking_config=ChunkingConfig(target_tokens=256, overlap_tokens=0),
    )


async def _publish_text(
    service: IngestionService,
    *,
    source_key: str,
    title: str,
    text: str,
) -> IndexedSource:
    submitted = service.submit_upload(
        f"{source_key}.txt",
        text.encode("utf-8"),
        source_key=source_key,
        declared_media_type="text/plain",
        display_title=title,
    )
    completed = await service.run(submitted.job_id)
    assert completed.status is IngestionJobStatus.SUCCEEDED
    assert completed.source_id is not None
    assert completed.document_version is not None
    return IndexedSource(
        source_id=completed.source_id,
        version=completed.document_version,
        title=title,
        text=text,
    )


def _factory(root: Path, service: IngestionService) -> BoundRetrievalSnapshotFactory:
    return BoundRetrievalSnapshotFactory(
        DataLayout.from_root(root),
        service.repositories.index_revisions,
    )


def _service(
    snapshot: BoundRetrievalSnapshot,
    provider: ConceptEmbeddingProvider,
    *,
    request_id: str,
    allow_degradation: bool = False,
    dense_limit: int = 5,
    lexical_limit: int = 5,
    final_limit: int = 5,
) -> RetrievalService:
    return RetrievalService.from_snapshot(
        snapshot,
        provider,
        ProviderCallContext(request_id, "retrieval", Deadline.after(30)),
        limits=RetrievalLimits(
            dense=dense_limit,
            lexical=lexical_limit,
            rerank=max(5, final_limit),
            final=final_limit,
        ),
        allow_single_retriever_degradation=allow_degradation,
    )


def _request(
    snapshot: BoundRetrievalSnapshot,
    *,
    request_id: str,
    query: str,
    mode: RetrievalMode,
) -> RetrievalRequestContext:
    return RetrievalRequestContext.from_snapshot(
        request_id=request_id,
        query=query,
        mode=mode,
        snapshot=snapshot,
    )


async def _retrieve(
    snapshot: BoundRetrievalSnapshot,
    provider: ConceptEmbeddingProvider,
    *,
    request_id: str,
    query: str,
    mode: RetrievalMode,
    dense_limit: int = 5,
    lexical_limit: int = 5,
    final_limit: int = 5,
) -> RetrievalResult:
    service = _service(
        snapshot,
        provider,
        request_id=request_id,
        dense_limit=dense_limit,
        lexical_limit=lexical_limit,
        final_limit=final_limit,
    )
    try:
        return await service.retrieve(
            _request(
                snapshot,
                request_id=request_id,
                query=query,
                mode=mode,
            )
        )
    finally:
        service.close()


def _assert_persisted_identity(snapshot: BoundRetrievalSnapshot) -> None:
    revision = snapshot.revision
    assert snapshot.revision_id == snapshot.dense.revision_id == snapshot.bm25.revision_id
    assert snapshot.dense.identity == revision.embedding_space
    assert snapshot.bm25.tokenizer_identity == revision.tokenizer_version
    assert snapshot.bm25.algorithm_version == revision.lexical_algorithm_version
    assert snapshot.bm25.k1 == revision.lexical_k1
    assert snapshot.bm25.b == revision.lexical_b
    assert snapshot.dense.chunk_ids == snapshot.bm25.chunk_ids
    assert snapshot.dense.inventory_digest == snapshot.bm25.chunk_set_digest
    assert snapshot.dense.inventory_digest == revision.chunk_set_digest
    assert len(snapshot.dense.chunk_ids) == revision.chunk_count


def _assert_no_retrieval_cache(result: RetrievalResult) -> None:
    assert result.diagnostics.cache_status == {
        "query_embedding": CacheOutcome.NOT_APPLICABLE,
        "retrieval": CacheOutcome.NOT_APPLICABLE,
        "rerank": CacheOutcome.NOT_APPLICABLE,
        "final": CacheOutcome.NOT_APPLICABLE,
    }


def _assert_exact_evidence(result: RetrievalResult, source: IndexedSource) -> None:
    evidence = next(item for item in result.evidence if item.source_id == source.source_id)
    assert evidence.text == source.text
    assert evidence.display_title == source.title
    assert evidence.document_version == source.version
    assert evidence.locator == source.locator
    assert evidence.revision_id == result.diagnostics.index_revision


@pytest.mark.integration
async def test_persisted_pipeline_semantic_exact_degraded_identity_and_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "data"
    ingestion_provider = ConceptEmbeddingProvider()
    ingestion = _ingestion(root, ingestion_provider)
    english_semantic = await _publish_text(
        ingestion,
        source_key="english-semantic",
        title="English Benefits Guide",
        text="The sabbatical benefit permits a twelve-week paid absence after six years.",
    )
    chinese_semantic = await _publish_text(
        ingestion,
        source_key="chinese-semantic",
        title="中文安全手册",
        text="双因素验证是所有远程管理员登录的强制要求。",
    )
    english_exact = await _publish_text(
        ingestion,
        source_key="english-exact",
        title="Release Register",
        text="The release ledger records identifier ZXQ-741 for the cobalt deployment.",
    )
    chinese_exact = await _publish_text(
        ingestion,
        source_key="chinese-exact",
        title="运维标准",
        text="运维台账规定量子密钥轮换必须每季度执行。",
    )
    await _publish_text(
        ingestion,
        source_key="dense-anchor",
        title="Unrelated Notes",
        text="A neutral anchor describes office plant maintenance.",
    )
    active = ingestion.repositories.index_revisions.get_active()
    assert active is not None
    revision_count = len(ingestion.repositories.index_revisions.list())

    factory = _factory(root, ingestion)
    snapshot = factory.bind()
    _assert_persisted_identity(snapshot)
    query_provider = ingestion_provider
    chinese_query = await _retrieve(
        snapshot,
        query_provider,
        request_id="semantic-zh-en",
        query="长期休假福利如何安排?",
        mode=RetrievalMode.DENSE,
        final_limit=1,
    )
    english_query = await _retrieve(
        snapshot,
        query_provider,
        request_id="semantic-en-zh",
        query="multifactor authentication requirement",
        mode=RetrievalMode.DENSE,
        final_limit=1,
    )

    chinese_lexical = await snapshot.bm25.search("长期休假福利如何安排?", 5)
    english_lexical = await snapshot.bm25.search("multifactor authentication requirement", 5)
    assert english_semantic.source_id not in {item.source_id for item in chinese_lexical}
    assert chinese_semantic.source_id not in {item.source_id for item in english_lexical}
    assert chinese_query.evidence[0].source_id == english_semantic.source_id
    assert english_query.evidence[0].source_id == chinese_semantic.source_id
    assert all(
        item.bm25_rank is None for item in (*chinese_query.evidence, *english_query.evidence)
    )
    _assert_exact_evidence(chinese_query, english_semantic)
    _assert_exact_evidence(english_query, chinese_semantic)
    _assert_no_retrieval_cache(chinese_query)
    _assert_no_retrieval_cache(english_query)

    for request_id, query, expected in (
        ("exact-english", "ZXQ-741", english_exact),
        ("exact-chinese", "量子密钥轮换", chinese_exact),
    ):
        result = await _retrieve(
            snapshot,
            query_provider,
            request_id=request_id,
            query=query,
            mode=RetrievalMode.HYBRID,
            dense_limit=1,
            final_limit=2,
        )
        evidence = next(item for item in result.evidence if item.source_id == expected.source_id)
        assert evidence.final_rank <= 2
        assert evidence.dense_rank is None and evidence.dense_score is None
        assert evidence.bm25_rank == 1
        assert evidence.bm25_score is not None and evidence.bm25_score > 0
        assert evidence.rrf_score == pytest.approx(1 / 61)
        assert result.diagnostics.candidate_counts["dense"] == 1
        assert result.diagnostics.candidate_counts["bm25"] >= 1
        assert result.diagnostics.candidate_counts["fused"] >= 2
        _assert_exact_evidence(result, expected)
        _assert_no_retrieval_cache(result)

    query_calls = 0
    original_query = snapshot.dense.query

    def counted_query(*args: object, **kwargs: object) -> tuple[object, ...]:
        nonlocal query_calls
        query_calls += 1
        return original_query(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(snapshot.dense, "query", counted_query)
    incompatible_identity = replace(_IDENTITY, model="different-space-v1")
    incompatible = ConceptEmbeddingProvider(incompatible_identity)
    with pytest.raises(DenseIndexError, match="embedding_identity_mismatch"):
        _service(snapshot, incompatible, request_id="incompatible-query")
    assert incompatible.call_count == 0
    assert query_calls == 0

    compatible = await _retrieve(
        snapshot,
        query_provider,
        request_id="compatible-query",
        query="长期休假福利",
        mode=RetrievalMode.DENSE,
        final_limit=1,
    )
    assert compatible.evidence[0].source_id == english_semantic.source_id
    assert query_calls == 1

    for failed_channel, query, expected_source in (
        ("dense", "ZXQ-741", english_exact),
        ("bm25", "长期休假福利", english_semantic),
    ):
        strict_id = f"strict-{failed_channel}"
        strict = _service(snapshot, query_provider, request_id=strict_id)
        assert isinstance(strict._dense, BoundDenseRetriever)
        assert isinstance(strict._lexical, BoundBm25Retriever)
        if failed_channel == "dense":
            strict._dense = _FaultingDenseRetriever(strict._dense)  # type: ignore[assignment]
        else:
            strict._lexical = _FaultingBm25Retriever(strict._lexical)  # type: ignore[assignment]
        with pytest.raises(RetrievalUnavailableError) as captured:
            await strict.retrieve(
                _request(
                    snapshot,
                    request_id=strict_id,
                    query=query,
                    mode=RetrievalMode.HYBRID,
                )
            )
        assert captured.value.failed_stages == (failed_channel,)
        strict.close()

        degraded_id = f"degraded-{failed_channel}"
        degraded = _service(
            snapshot,
            query_provider,
            request_id=degraded_id,
            allow_degradation=True,
        )
        assert isinstance(degraded._dense, BoundDenseRetriever)
        assert isinstance(degraded._lexical, BoundBm25Retriever)
        if failed_channel == "dense":
            degraded._dense = _FaultingDenseRetriever(degraded._dense)  # type: ignore[assignment]
        else:
            degraded._lexical = _FaultingBm25Retriever(degraded._lexical)  # type: ignore[assignment]
        degraded_result = await degraded.retrieve(
            _request(
                snapshot,
                request_id=degraded_id,
                query=query,
                mode=RetrievalMode.HYBRID,
            )
        )
        degraded.close()

        assert any(item.source_id == expected_source.source_id for item in degraded_result.evidence)
        assert degraded_result.diagnostics.failed_stages == (failed_channel,)
        assert degraded_result.diagnostics.degradation_reasons == (f"{failed_channel}_unavailable",)
        assert degraded_result.diagnostics.candidate_counts[failed_channel] == 0
        surviving_channel = "bm25" if failed_channel == "dense" else "dense"
        assert degraded_result.diagnostics.candidate_counts[surviving_channel] > 0
        assert degraded_result.diagnostics.provider_identities.keys() >= {
            "embedding",
            "dense",
            "bm25",
            "rrf",
        }
        assert all(
            (item.dense_rank is None if failed_channel == "dense" else item.bm25_rank is None)
            for item in degraded_result.evidence
        )
        assert "injected_" not in degraded_result.diagnostics.model_dump_json()
        _assert_exact_evidence(degraded_result, expected_source)
        _assert_no_retrieval_cache(degraded_result)

    snapshot.close()
    ingestion.close()

    recovery_provider = ConceptEmbeddingProvider()
    restarted = _ingestion(root, recovery_provider)
    report = await restarted.recover_startup()
    assert report.active_revision_id == active.revision_id
    assert report.replayed_job_count == 0
    assert recovery_provider.call_count == 0
    assert len(restarted.repositories.index_revisions.list()) == revision_count

    restarted_snapshot = _factory(root, restarted).bind()
    _assert_persisted_identity(restarted_snapshot)
    restarted_result = await _retrieve(
        restarted_snapshot,
        recovery_provider,
        request_id="restart-query",
        query="multifactor authentication requirement",
        mode=RetrievalMode.DENSE,
        final_limit=1,
    )
    assert recovery_provider.call_count == 1
    assert restarted_result.diagnostics.index_revision == active.revision_id
    _assert_exact_evidence(restarted_result, chinese_semantic)
    _assert_no_retrieval_cache(restarted_result)
    restarted_snapshot.close()
    restarted.close()


@pytest.mark.integration
async def test_last_source_deletion_publishes_empty_revision_without_stale_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    provider = ConceptEmbeddingProvider()
    ingestion = _ingestion(root, provider)
    source = await _publish_text(
        ingestion,
        source_key="only-source",
        title="Only Source",
        text="The sabbatical benefit is the only indexed policy.",
    )
    populated = ingestion.repositories.index_revisions.get_active()
    assert populated is not None and populated.chunk_count == 1

    deleted = await ingestion.run(ingestion.submit_delete(source.source_id).job_id)
    empty = ingestion.repositories.index_revisions.get_active()
    assert deleted.status is IngestionJobStatus.SUCCEEDED
    assert empty is not None and empty.revision_id != populated.revision_id
    assert empty.active_sources == {} and empty.chunk_count == 0

    snapshot = _factory(root, ingestion).bind()
    _assert_persisted_identity(snapshot)
    result = await _retrieve(
        snapshot,
        provider,
        request_id="empty-corpus",
        query="长期休假福利",
        mode=RetrievalMode.HYBRID,
    )

    assert result.evidence == ()
    assert result.diagnostics.index_revision == empty.revision_id
    assert result.diagnostics.candidate_counts == {
        "dense": 0,
        "bm25": 0,
        "fused": 0,
        "reranked": 0,
        "final": 0,
    }
    assert source.text not in result.model_dump_json()
    _assert_no_retrieval_cache(result)
    snapshot.close()
    ingestion.close()


@pytest.mark.integration
async def test_concurrent_publication_keeps_paused_request_on_bound_revision(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    query_started = asyncio.Event()
    query_release = asyncio.Event()
    provider = ConceptEmbeddingProvider(
        blocked_query="historical state",
        query_started=query_started,
        query_release=query_release,
    )
    ingestion = _ingestion(root, provider)
    version_one = await _publish_text(
        ingestion,
        source_key="atomic-source",
        title="Atomic Publication Guide",
        text="The blue state contains revision-one-only text OLD-STATE-1.",
    )
    revision_one = ingestion.repositories.index_revisions.get_active()
    assert revision_one is not None
    factory = _factory(root, ingestion)
    old_snapshot = factory.bind()
    _assert_persisted_identity(old_snapshot)
    old_service = _service(
        old_snapshot,
        provider,
        request_id="paused-old-request",
        dense_limit=1,
        final_limit=1,
    )
    old_task = asyncio.create_task(
        old_service.retrieve(
            _request(
                old_snapshot,
                request_id="paused-old-request",
                query="historical state",
                mode=RetrievalMode.DENSE,
            )
        )
    )
    await asyncio.wait_for(query_started.wait(), timeout=5)

    try:
        version_two = await _publish_text(
            ingestion,
            source_key="atomic-source",
            title="Atomic Publication Guide",
            text="The green state contains revision-two-only text NEW-STATE-2.",
        )
        revision_two = ingestion.repositories.index_revisions.get_active()
        persisted_one = ingestion.repositories.index_revisions.get(revision_one.revision_id)
        assert revision_two is not None and revision_two.revision_id != revision_one.revision_id
        assert persisted_one is not None
        assert persisted_one.status is IndexRevisionStatus.SUPERSEDED
        assert version_two.source_id == version_one.source_id
        assert version_two.version == 2

        query_release.set()
        old_result = await asyncio.wait_for(old_task, timeout=5)
    finally:
        query_release.set()
        if not old_task.done():
            old_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await old_task
        old_service.close()

    assert [item.text for item in old_result.evidence] == [version_one.text]
    assert [item.document_version for item in old_result.evidence] == [1]
    assert old_result.diagnostics.index_revision == revision_one.revision_id
    assert all("NEW-STATE-2" not in item.text for item in old_result.evidence)
    _assert_exact_evidence(old_result, version_one)
    _assert_no_retrieval_cache(old_result)

    fresh_snapshot = factory.bind()
    _assert_persisted_identity(fresh_snapshot)
    fresh_result = await _retrieve(
        fresh_snapshot,
        provider,
        request_id="fresh-new-request",
        query="current state",
        mode=RetrievalMode.DENSE,
        dense_limit=1,
        final_limit=1,
    )
    assert [item.text for item in fresh_result.evidence] == [version_two.text]
    assert [item.document_version for item in fresh_result.evidence] == [2]
    assert fresh_result.diagnostics.index_revision == revision_two.revision_id
    assert all("OLD-STATE-1" not in item.text for item in fresh_result.evidence)
    _assert_exact_evidence(fresh_result, version_two)
    _assert_no_retrieval_cache(fresh_result)

    old_snapshot.close()
    fresh_snapshot.close()
    ingestion.close()


def test_concept_vectors_are_finite_unit_vectors_with_explicit_bilingual_pairs() -> None:
    pairs: Sequence[tuple[str, str]] = (
        ("sabbatical benefit", "长期休假福利"),
        ("multifactor authentication", "双因素验证"),
    )
    for english, chinese in pairs:
        english_vector = _concept_vector(english, _IDENTITY.dimension)
        chinese_vector = _concept_vector(chinese, _IDENTITY.dimension)
        assert english_vector == chinese_vector
        assert all(math.isfinite(value) for value in english_vector)
        assert math.sqrt(sum(value * value for value in english_vector)) == pytest.approx(1.0)
