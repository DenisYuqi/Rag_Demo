from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest
from retrieval_test_helpers import build_bound_snapshot, candidate

from rag_mvp.domain.evaluation import ModelAttemptStatus
from rag_mvp.domain.retrieval import CacheOutcome, RetrievalCandidate, RetrievalMode
from rag_mvp.providers.errors import ProviderError
from rag_mvp.providers.fakes import DeterministicEmbeddingProvider
from rag_mvp.providers.models import (
    AttemptStatus,
    Deadline,
    EmbeddingRequest,
    EmbeddingResult,
    ModelAttempt,
    ModelIdentity,
    ProviderCallContext,
    ProviderErrorCategory,
    ProviderRole,
    RerankRequest,
    RerankResult,
)
from rag_mvp.providers.resilience import RetryPolicy
from rag_mvp.providers.routing import ModelProviderRouter, ProviderRoute
from rag_mvp.retrieval.bm25 import LexicalIndexError
from rag_mvp.retrieval.identity import provider_embedding_identity
from rag_mvp.retrieval.request import RetrievalRequestContext, RetrievalRequestError
from rag_mvp.retrieval.service import (
    RetrievalLimits,
    RetrievalService,
    RetrievalUnavailableError,
    _provider_attempt_evidence,
)


class SpyRetriever:
    def __init__(self, results: tuple[RetrievalCandidate, ...], *, fails: bool = False) -> None:
        self.results = results
        self.fails = fails
        self.calls = 0
        self.limits: list[int] = []

    async def search(self, query: str, limit: int) -> tuple[RetrievalCandidate, ...]:
        del query
        self.calls += 1
        self.limits.append(limit)
        if self.fails:
            raise RuntimeError("safe fake failure")
        return self.results


class ReverseLegacyReranker:
    identity = "reverse-v1"

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
    ) -> tuple[str, ...]:
        del query
        return tuple(candidate.chunk_id for candidate in reversed(candidates))


def test_provider_timeout_attempt_uses_timed_out_evidence_status() -> None:
    attempt = ModelAttempt(
        request_id="request-timeout",
        operation_id="qa-retrieval",
        attempt_number=1,
        route_id="embedding-primary",
        role=ProviderRole.EMBEDDING,
        provider="test",
        model="embedding-model",
        latency_ms=1,
        status=AttemptStatus.FAILED,
        is_fallback=False,
        error_category=ProviderErrorCategory.TIMEOUT,
    )

    assert _provider_attempt_evidence(attempt).status is ModelAttemptStatus.TIMED_OUT


class ReverseProviderReranker:
    identity = ModelIdentity("test", "reverse", "adapter-v1")

    def __init__(self) -> None:
        self.calls = 0

    async def rerank(
        self,
        request: RerankRequest,
        context: ProviderCallContext,
    ) -> RerankResult:
        del context
        self.calls += 1
        return RerankResult(
            tuple(reversed(request.candidate_ids)),
            self.identity,
            request.prompt_version,
        )


class FailingEmbeddingProvider:
    def __init__(self, identity: object) -> None:
        self.identity = identity
        self.calls = 0

    async def embed(
        self,
        request: EmbeddingRequest,
        context: ProviderCallContext,
    ) -> EmbeddingResult:
        del request, context
        self.calls += 1
        raise ProviderError(ProviderErrorCategory.SERVER)


def _provider_context(request_id: str = "request") -> ProviderCallContext:
    return ProviderCallContext(request_id, "retrieval", Deadline.after(10))


def _request(snapshot: object, mode: RetrievalMode) -> RetrievalRequestContext:
    return RetrievalRequestContext.from_snapshot(
        request_id="request",
        query="policy",
        mode=mode,
        snapshot=snapshot,  # type: ignore[arg-type]
    )


async def test_dense_mode_does_not_call_lexical_or_reranker_legacy() -> None:
    dense = SpyRetriever((candidate("a", dense_rank=1),))
    lexical = SpyRetriever(())
    service = RetrievalService(
        dense=dense,
        lexical=lexical,
        reranker=ReverseLegacyReranker(),
    )

    result = await service.retrieve(
        RetrievalRequestContext("req", "query", RetrievalMode.DENSE, "rev")
    )

    assert result.evidence[0].chunk_id == "a"
    assert dense.calls == 1
    assert lexical.calls == 0


async def test_hybrid_degrades_to_one_successful_retriever_legacy() -> None:
    dense = SpyRetriever((), fails=True)
    lexical = SpyRetriever((candidate("lexical", bm25_rank=1),))
    service = RetrievalService(dense=dense, lexical=lexical)

    result = await service.retrieve(
        RetrievalRequestContext("req", "query", RetrievalMode.HYBRID, "rev")
    )

    assert result.evidence[0].chunk_id == "lexical"
    assert result.diagnostics.failed_stages == ("dense",)
    assert result.diagnostics.effective_mode is RetrievalMode.HYBRID


async def test_hybrid_rerank_applies_valid_order_legacy() -> None:
    dense = SpyRetriever((candidate("a", dense_rank=1), candidate("b", dense_rank=2)))
    service = RetrievalService(
        dense=dense,
        lexical=SpyRetriever(()),
        reranker=ReverseLegacyReranker(),
    )

    result = await service.retrieve(
        RetrievalRequestContext("req", "query", RetrievalMode.HYBRID_RERANK, "rev")
    )

    assert [item.chunk_id for item in result.evidence] == ["b", "a"]


async def test_production_dense_mode_is_snapshot_bound_and_skips_bm25_rerank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, source_kinds = await build_bound_snapshot(tmp_path)
    embedding = DeterministicEmbeddingProvider(
        provider_embedding_identity(snapshot.revision.embedding_space)
    )
    reranker = ReverseProviderReranker()

    async def forbidden_bm25(query: str, limit: int) -> tuple[RetrievalCandidate, ...]:
        del query, limit
        raise AssertionError("BM25 must not run in dense mode")

    monkeypatch.setattr(snapshot.bm25, "search", forbidden_bm25)
    service = RetrievalService.from_snapshot(
        snapshot,
        embedding,
        _provider_context(),
        source_kinds=source_kinds,
        reranker=reranker,
        limits=RetrievalLimits(dense=2, lexical=2, rerank=2, final=2),
    )

    result = await service.retrieve(_request(snapshot, RetrievalMode.DENSE))

    assert embedding.call_count == 1
    assert reranker.calls == 0
    assert all(item.bm25_rank is None and item.rrf_score is None for item in result.evidence)
    assert result.diagnostics.effective_mode is RetrievalMode.DENSE
    assert result.diagnostics.index_revision == snapshot.revision_id
    snapshot.close()


async def test_production_hybrid_runs_both_channels_and_reports_complete_diagnostics(
    tmp_path: Path,
) -> None:
    snapshot, source_kinds = await build_bound_snapshot(tmp_path)
    embedding = DeterministicEmbeddingProvider(
        provider_embedding_identity(snapshot.revision.embedding_space)
    )
    service = RetrievalService.from_snapshot(
        snapshot,
        embedding,
        _provider_context(),
        source_kinds=source_kinds,
        limits=RetrievalLimits(dense=2, lexical=2, rerank=2, final=2),
    )

    result = await service.retrieve(_request(snapshot, RetrievalMode.HYBRID))

    diagnostics = result.diagnostics
    assert embedding.call_count == 1
    assert diagnostics.requested_mode is RetrievalMode.HYBRID
    assert diagnostics.effective_mode is RetrievalMode.HYBRID
    assert diagnostics.candidate_counts.keys() == {
        "dense",
        "bm25",
        "fused",
        "reranked",
        "final",
    }
    assert diagnostics.candidate_counts["dense"] <= 2
    assert diagnostics.candidate_counts["bm25"] <= 2
    assert diagnostics.candidate_counts["final"] <= 2
    assert diagnostics.stage_timings_ms.keys() >= {
        "query_embedding",
        "dense",
        "bm25",
        "fusion",
        "rerank",
        "total",
    }
    assert diagnostics.provider_identities.keys() >= {"embedding", "dense", "bm25", "rrf"}
    assert set(diagnostics.cache_status.values()) == {CacheOutcome.NOT_APPLICABLE}
    serialized = diagnostics.model_dump_json()
    assert "annual leave policy alpha" not in serialized
    assert "network security policy beta" not in serialized
    snapshot.close()


async def test_retrieval_diagnostics_count_failed_provider_fallback_attempts(
    tmp_path: Path,
) -> None:
    snapshot, source_kinds = await build_bound_snapshot(tmp_path)
    identity = provider_embedding_identity(snapshot.revision.embedding_space)
    primary = FailingEmbeddingProvider(identity)
    fallback = DeterministicEmbeddingProvider(identity)
    router = ModelProviderRouter(
        embedding_routes=(
            ProviderRoute("embedding-primary", primary, RetryPolicy(1, max_retries=0)),
            ProviderRoute("embedding-fallback", fallback, RetryPolicy(1, max_retries=0)),
        )
    )
    service = RetrievalService.from_snapshot(
        snapshot,
        router,
        _provider_context(),
        source_kinds=source_kinds,
        limits=RetrievalLimits(dense=2, lexical=2, rerank=2, final=2),
    )

    result = await service.retrieve(_request(snapshot, RetrievalMode.HYBRID))

    assert result.diagnostics.provider_attempt_counts == {"embedding": 2}
    assert result.diagnostics.provider_failed_attempt_counts == {"embedding": 1}
    snapshot.close()


async def test_hybrid_rerank_requires_configuration_and_applies_when_present(
    tmp_path: Path,
) -> None:
    snapshot, source_kinds = await build_bound_snapshot(tmp_path)
    embedding = DeterministicEmbeddingProvider(
        provider_embedding_identity(snapshot.revision.embedding_space)
    )
    limits = RetrievalLimits(dense=2, lexical=2, rerank=2, final=2)
    unconfigured = RetrievalService.from_snapshot(
        snapshot,
        embedding,
        _provider_context(),
        source_kinds=source_kinds,
        limits=limits,
    )

    fallback = await unconfigured.retrieve(_request(snapshot, RetrievalMode.HYBRID_RERANK))

    assert fallback.diagnostics.effective_mode is RetrievalMode.HYBRID
    assert fallback.diagnostics.degradation_reasons == ("reranker_not_configured",)

    reranker = ReverseProviderReranker()
    configured = RetrievalService.from_snapshot(
        snapshot,
        embedding,
        _provider_context(),
        source_kinds=source_kinds,
        reranker=reranker,
        limits=limits,
    )
    result = await configured.retrieve(_request(snapshot, RetrievalMode.HYBRID_RERANK))

    assert reranker.calls == 1
    assert result.diagnostics.effective_mode is RetrievalMode.HYBRID_RERANK
    assert result.diagnostics.candidate_counts["reranked"] == 2
    assert result.diagnostics.provider_identities["reranker"].startswith("test/reverse/adapter-v1")
    assert [item.reranking_rank for item in result.evidence] == [1, 2]
    snapshot.close()


async def test_router_without_rerank_routes_does_not_invent_provider_attempt(
    tmp_path: Path,
) -> None:
    snapshot, source_kinds = await build_bound_snapshot(tmp_path)
    embedding = DeterministicEmbeddingProvider(
        provider_embedding_identity(snapshot.revision.embedding_space)
    )
    service = RetrievalService.from_snapshot(
        snapshot,
        embedding,
        _provider_context(),
        source_kinds=source_kinds,
        reranker=ModelProviderRouter(),
        limits=RetrievalLimits(dense=2, lexical=2, rerank=2, final=2),
    )

    result = await service.retrieve(_request(snapshot, RetrievalMode.HYBRID_RERANK))

    assert result.diagnostics.degradation_reasons == ("rerank_provider_unavailable",)
    assert "reranker" not in result.diagnostics.provider_attempt_counts
    assert all(
        attempt.role.value != "reranking" for attempt in result.diagnostics.provider_attempts
    )
    snapshot.close()


async def test_single_normalized_failure_requires_explicit_degradation(
    tmp_path: Path,
) -> None:
    snapshot, source_kinds = await build_bound_snapshot(tmp_path)
    identity = provider_embedding_identity(snapshot.revision.embedding_space)
    failing = FailingEmbeddingProvider(identity)
    limits = RetrievalLimits(dense=2, lexical=2, rerank=2, final=2)
    strict = RetrievalService.from_snapshot(
        snapshot,
        failing,  # type: ignore[arg-type]
        _provider_context(),
        source_kinds=source_kinds,
        limits=limits,
    )

    with pytest.raises(RetrievalUnavailableError) as caught:
        await strict.retrieve(_request(snapshot, RetrievalMode.HYBRID))
    assert caught.value.code == "retrieval_unavailable"
    assert caught.value.failed_stages == ("dense",)
    assert caught.value.provider_attempt_count == 1
    assert caught.value.provider_failed_attempt_count == 1
    assert caught.value.provider_unknown_usage_attempt_count == 1

    degraded = RetrievalService.from_snapshot(
        snapshot,
        failing,  # type: ignore[arg-type]
        _provider_context(),
        source_kinds=source_kinds,
        limits=limits,
        allow_single_retriever_degradation=True,
    )
    result = await degraded.retrieve(_request(snapshot, RetrievalMode.HYBRID))

    assert result.evidence
    assert result.diagnostics.failed_stages == ("dense",)
    assert result.diagnostics.degradation_reasons == ("dense_unavailable",)
    assert result.diagnostics.provider_attempt_counts == {"embedding": 1}
    assert result.diagnostics.provider_failed_attempt_counts == {"embedding": 1}
    assert result.diagnostics.provider_unknown_usage_attempt_counts == {"embedding": 1}
    assert len(result.diagnostics.provider_attempts) == 1
    assert all(item.dense_rank is None for item in result.evidence)
    snapshot.close()


async def test_both_retrievers_failing_never_degrades(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, source_kinds = await build_bound_snapshot(tmp_path)
    failing = FailingEmbeddingProvider(
        provider_embedding_identity(snapshot.revision.embedding_space)
    )

    async def fail_bm25(query: str, limit: int) -> tuple[RetrievalCandidate, ...]:
        del query, limit
        raise LexicalIndexError("lexical_query_failed")

    monkeypatch.setattr(snapshot.bm25, "search", fail_bm25)
    service = RetrievalService.from_snapshot(
        snapshot,
        failing,  # type: ignore[arg-type]
        _provider_context(),
        source_kinds=source_kinds,
        allow_single_retriever_degradation=True,
    )

    with pytest.raises(RetrievalUnavailableError) as caught:
        await service.retrieve(_request(snapshot, RetrievalMode.HYBRID))
    assert caught.value.failed_stages == ("dense", "bm25")
    assert caught.value.provider_attempt_count == 1
    assert caught.value.provider_failed_attempt_count == 1
    snapshot.close()


async def test_context_must_match_exact_snapshot_binding(tmp_path: Path) -> None:
    first, first_kinds = await build_bound_snapshot(tmp_path / "first", revision_id="revision-one")
    second, _ = await build_bound_snapshot(tmp_path / "second", revision_id="revision-two")
    service = RetrievalService.from_snapshot(
        first,
        DeterministicEmbeddingProvider(provider_embedding_identity(first.revision.embedding_space)),
        _provider_context(),
        source_kinds=first_kinds,
    )

    with pytest.raises(RetrievalRequestError, match="snapshot_context_mismatch"):
        await service.retrieve(_request(second, RetrievalMode.DENSE))
    first.close()
    second.close()


async def test_service_enforces_limits_even_for_bad_legacy_doubles() -> None:
    dense = SpyRetriever(tuple(candidate(f"c-{rank}", dense_rank=rank) for rank in range(1, 6)))
    service = RetrievalService(
        dense=dense,
        lexical=SpyRetriever(()),
        limits=RetrievalLimits(dense=2, lexical=2, rerank=2, final=1),
    )

    result = await service.retrieve(
        RetrievalRequestContext("request", "query", RetrievalMode.DENSE, "revision")
    )

    assert dense.limits == [2]
    assert len(result.evidence) == 1
    assert result.diagnostics.candidate_counts["dense"] == 2


async def test_parent_cancellation_is_not_converted_to_degradation() -> None:
    started = asyncio.Event()

    class BlockingRetriever:
        async def search(self, query: str, limit: int) -> tuple[RetrievalCandidate, ...]:
            del query, limit
            started.set()
            await asyncio.Event().wait()
            return ()

    service = RetrievalService(dense=BlockingRetriever(), lexical=SpyRetriever(()))
    task = asyncio.create_task(
        service.retrieve(
            RetrievalRequestContext("request", "query", RetrievalMode.HYBRID, "revision")
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_service_snapshot_ownership_is_explicit(tmp_path: Path) -> None:
    snapshot, _ = await build_bound_snapshot(tmp_path)
    service = RetrievalService.from_snapshot(
        snapshot,
        DeterministicEmbeddingProvider(
            provider_embedding_identity(snapshot.revision.embedding_space)
        ),
        _provider_context(),
        owns_snapshot=True,
    )

    await service.retrieve(_request(snapshot, RetrievalMode.DENSE))
    assert not snapshot.is_closed
    service.close()
    assert snapshot.is_closed
