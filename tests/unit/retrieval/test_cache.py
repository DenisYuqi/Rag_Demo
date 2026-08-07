from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from retrieval_test_helpers import build_bound_snapshot

from rag_mvp.config.settings import Settings
from rag_mvp.domain.ingestion import EmbeddingSpaceIdentity
from rag_mvp.domain.retrieval import CacheOutcome, CachePolicy, RetrievalMode
from rag_mvp.providers.errors import ProviderError
from rag_mvp.providers.fakes import DeterministicEmbeddingProvider, DeterministicRerankingProvider
from rag_mvp.providers.models import (
    Deadline,
    EmbeddingRequest,
    EmbeddingResult,
    ProviderCallContext,
    ProviderErrorCategory,
)
from rag_mvp.retrieval.cache import (
    BoundedCacheLookup,
    BoundedCacheWrite,
    BoundedTtlCache,
    RetrievalCacheIdentity,
    RetrievalCachePayload,
    RetrievalCacheWriteStatus,
    RetrievalResultCache,
)
from rag_mvp.retrieval.fusion import RrfConfig
from rag_mvp.retrieval.identity import provider_embedding_identity
from rag_mvp.retrieval.request import (
    QUERY_CANONICALIZATION_VERSION,
    RetrievalRequestContext,
)
from rag_mvp.retrieval.service import (
    DEGRADATION_POLICY_VERSION,
    RRF_TIE_POLICY_VERSION,
    RetrievalService,
    RetrievalUnavailableError,
)


class _FailingEmbeddingProvider:
    def __init__(self, identity: Any, *, fail_once: bool = False) -> None:
        self.identity = identity
        self.fail_once = fail_once
        self.calls = 0
        self._delegate = DeterministicEmbeddingProvider(identity)

    async def embed(
        self,
        request: EmbeddingRequest,
        context: ProviderCallContext,
    ) -> EmbeddingResult:
        self.calls += 1
        if not self.fail_once or self.calls == 1:
            raise ProviderError(ProviderErrorCategory.SERVER)
        return await self._delegate.embed(request, context)


class _ExplodingCacheBackend(BoundedTtlCache[str]):
    def lookup(self, key: str, *, now: float | None = None) -> BoundedCacheLookup[str]:
        del key, now
        raise RuntimeError("cache read unavailable")

    def put_with_outcome(
        self,
        key: str,
        value: str,
        *,
        now: float | None = None,
    ) -> BoundedCacheWrite:
        del key, value, now
        raise RuntimeError("cache write unavailable")


class _BlockingEmbeddingProvider:
    def __init__(self, identity: Any) -> None:
        self.identity = identity
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self._delegate = DeterministicEmbeddingProvider(identity)

    async def embed(
        self,
        request: EmbeddingRequest,
        context: ProviderCallContext,
    ) -> EmbeddingResult:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return await self._delegate.embed(request, context)


def _cache_settings(
    tmp_path: Path,
    *,
    allow_single_retriever_degradation: bool = False,
    reranking_model: str | None = None,
) -> Settings:
    return Settings(
        data_root=tmp_path,
        retrieval_cache_enabled=True,
        allow_single_retriever_degradation=allow_single_retriever_degradation,
        reranking_model=reranking_model,
        _env_file=None,
    )


def _result_cache(
    *,
    maximum_entries: int = 8,
    ttl_seconds: float = 60,
    clock: Callable[[], float] | None = None,
    backend: BoundedTtlCache[str] | None = None,
) -> RetrievalResultCache:
    if clock is None:
        return RetrievalResultCache(
            configuration_id="configuration-v1",
            maximum_entries=maximum_entries,
            ttl_seconds=ttl_seconds,
            backend=backend,
        )
    return RetrievalResultCache(
        configuration_id="configuration-v1",
        maximum_entries=maximum_entries,
        ttl_seconds=ttl_seconds,
        backend=backend,
        clock=clock,
    )


def _identity() -> RetrievalCacheIdentity:
    return RetrievalCacheIdentity(
        canonical_query="policy",
        canonicalization_version=QUERY_CANONICALIZATION_VERSION,
        configuration_id="configuration-v1",
        revision_id="revision-1",
        chunk_set_digest="chunk-set-digest-v1",
        chunk_set_digest_algorithm="chunk-set-sha256-v1",
        record_digest_algorithm="chunk-record-sha256-v1",
        extraction_version="extraction-v1",
        chunking_version="chunking-v1",
        mode=RetrievalMode.HYBRID_RERANK,
        embedding_identity=EmbeddingSpaceIdentity(
            provider_alias="provider",
            model="embed-model",
            dimension=3,
            normalization="none",
            adapter_version="embed-adapter-v1",
        ),
        dense_schema_version="chroma-v1",
        dense_metric="cosine",
        bm25_tokenizer_identity="tokenizer-v1",
        bm25_schema_version="bm25-schema-v1",
        bm25_algorithm_version="bm25-okapi-v1",
        bm25_k1=1.5,
        bm25_b=0.75,
        rrf_version="rrf-v1",
        rrf_k=60,
        dense_weight=1.0,
        lexical_weight=1.0,
        rrf_tie_policy=RRF_TIE_POLICY_VERSION,
        reranker_route_id="rerank-route",
        reranker_provider="provider",
        reranker_model="rerank-model",
        reranker_adapter_version="rerank-adapter-v1",
        reranker_prompt_version="prompt-v1",
        reranker_truncation_version="truncate-v1",
        reranker_parser_version="parser-v1",
        reranker_maximum_query_characters=2048,
        reranker_maximum_query_tokens=256,
        reranker_maximum_candidate_characters=2048,
        reranker_maximum_candidate_tokens=512,
        reranker_budget_seconds=1.2,
        allow_single_retriever_degradation=False,
        degradation_policy_version=DEGRADATION_POLICY_VERSION,
        dense_limit=20,
        lexical_limit=20,
        rerank_limit=10,
        final_limit=5,
        evidence_schema_version="evidence-v1",
        result_schema_version="result-v1",
        safety_version="safety-v1",
    )


@pytest.mark.parametrize(
    "change",
    [
        {"canonical_query": "other"},
        {"canonicalization_version": "canonical-v2"},
        {"configuration_id": "configuration-v2"},
        {"revision_id": "revision-2"},
        {"chunk_set_digest": "chunk-set-digest-v2"},
        {"chunk_set_digest_algorithm": "chunk-set-v2"},
        {"record_digest_algorithm": "record-v2"},
        {"extraction_version": "extraction-v2"},
        {"chunking_version": "chunking-v2"},
        {"mode": RetrievalMode.HYBRID},
        {
            "embedding_identity": EmbeddingSpaceIdentity(
                provider_alias="provider",
                model="embed-model-v2",
                dimension=3,
                normalization="none",
                adapter_version="embed-adapter-v1",
            )
        },
        {"dense_schema_version": "chroma-v2"},
        {"dense_metric": "l2"},
        {"bm25_tokenizer_identity": "tokenizer-v2"},
        {"bm25_schema_version": "bm25-schema-v2"},
        {"bm25_algorithm_version": "bm25-okapi-v2"},
        {"bm25_k1": 1.6},
        {"bm25_b": 0.5},
        {"rrf_version": "rrf-v2"},
        {"rrf_k": 30},
        {"dense_weight": 2.0},
        {"lexical_weight": 2.0},
        {"rrf_tie_policy": "tie-v2"},
        {"reranker_route_id": "route-v2"},
        {"reranker_model": "rerank-model-v2"},
        {"reranker_adapter_version": "rerank-adapter-v2"},
        {"reranker_prompt_version": "prompt-v2"},
        {"reranker_truncation_version": "truncate-v2"},
        {"reranker_parser_version": "parser-v2"},
        {"reranker_maximum_query_characters": 1024},
        {"reranker_maximum_query_tokens": 128},
        {"reranker_maximum_candidate_characters": 1024},
        {"reranker_maximum_candidate_tokens": 256},
        {"reranker_budget_seconds": 0.8},
        {"allow_single_retriever_degradation": True},
        {"degradation_policy_version": "degradation-v2"},
        {"dense_limit": 19},
        {"lexical_limit": 19},
        {"rerank_limit": 9},
        {"final_limit": 4},
        {"evidence_schema_version": "evidence-v2"},
        {"result_schema_version": "result-v2"},
        {"safety_version": "safety-v2"},
    ],
)
def test_every_result_affecting_change_produces_a_distinct_key(
    change: dict[str, object],
) -> None:
    base = _identity()
    changed = replace(base, **change)  # type: ignore[arg-type]

    assert base.key != changed.key
    assert base.canonical_json != changed.canonical_json


def test_canonical_json_is_stable_and_contains_structured_embedding_identity() -> None:
    first = _identity()
    second = _identity()

    assert first.canonical_json == second.canonical_json
    assert first.key == second.key
    assert '"embedding_identity":{"adapter_version":"embed-adapter-v1"' in first.canonical_json
    assert '"query_digest"' in first.canonical_json
    assert '"canonical_query"' not in first.canonical_json
    assert first.canonical_query not in first.key
    assert "canonical_query=" not in repr(first)


@pytest.mark.parametrize(
    "change",
    [
        {"canonical_query": ""},
        {"canonical_query": " policy "},
        {"embedding_identity": "embed-v1"},
        {"bm25_k1": float("nan")},
        {"bm25_b": float("inf")},
        {"dense_weight": 0.0},
        {"rrf_k": True},
        {"final_limit": 11},
        {"allow_single_retriever_degradation": 1},
        {"reranker_model": None},
        {"reranker_budget_seconds": float("nan")},
    ],
)
def test_cache_identity_rejects_incomplete_invalid_or_nonfinite_values(
    change: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(_identity(), **change)  # type: ignore[arg-type]


def test_ttl_size_lru_and_injected_clock_are_bounded() -> None:
    now = [0.0]
    cache: BoundedTtlCache[str] = BoundedTtlCache(
        maximum_entries=2,
        ttl_seconds=10,
        clock=lambda: now[0],
    )
    cache.put("first", "one")
    now[0] = 1
    cache.put("second", "two")
    assert cache.get("first") == "one"
    cache.put("third", "three")

    assert cache.get("second") is None
    assert cache.get("first") == "one"
    assert cache.get("third") == "three"
    now[0] = 10
    assert cache.get("first") is None
    assert len(cache) == 1


def test_guarded_put_never_writes_failed_cancelled_or_degraded_results() -> None:
    cache: BoundedTtlCache[str] = BoundedTtlCache(maximum_entries=4, ttl_seconds=10)

    assert not cache.put_if_cacheable(
        "failed", "value", succeeded=False, cancelled=False, degraded=False
    )
    assert not cache.put_if_cacheable(
        "cancelled", "value", succeeded=True, cancelled=True, degraded=False
    )
    assert not cache.put_if_cacheable(
        "degraded", "value", succeeded=True, cancelled=False, degraded=True
    )
    assert cache.put_if_cacheable(
        "success", "value", succeeded=True, cancelled=False, degraded=False
    )
    assert len(cache) == 1
    assert cache.get("success") == "value"


def test_cache_is_thread_safe_and_never_exceeds_size_bound() -> None:
    cache: BoundedTtlCache[int] = BoundedTtlCache(maximum_entries=8, ttl_seconds=30)

    def writer(offset: int) -> None:
        for number in range(100):
            key = f"{offset}-{number}"
            cache.put(key, number)
            cache.get(key)

    threads = [threading.Thread(target=writer, args=(number,)) for number in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(cache) <= 8


async def test_absent_caches_do_not_affect_retrieval_and_report_not_applicable(
    tmp_path: Path,
) -> None:
    snapshot, source_kinds = await build_bound_snapshot(tmp_path)
    provider = DeterministicEmbeddingProvider(
        provider_embedding_identity(snapshot.revision.embedding_space)
    )
    service = RetrievalService.from_snapshot(
        snapshot,
        provider,
        ProviderCallContext("request", "retrieval", Deadline.after(10)),
        source_kinds=source_kinds,
    )
    context = RetrievalRequestContext.from_snapshot(
        request_id="request",
        query="policy",
        mode=RetrievalMode.HYBRID,
        snapshot=snapshot,
    )

    first = await service.retrieve(context)
    second = await service.retrieve(context)

    assert provider.call_count == 2
    assert first.evidence == second.evidence
    assert set(first.diagnostics.cache_status.values()) == {CacheOutcome.NOT_APPLICABLE}
    assert set(second.diagnostics.cache_status.values()) == {CacheOutcome.NOT_APPLICABLE}
    snapshot.close()


async def test_explicit_bypass_is_distinct_from_absent_cache_status(tmp_path: Path) -> None:
    snapshot, source_kinds = await build_bound_snapshot(tmp_path)
    provider = DeterministicEmbeddingProvider(
        provider_embedding_identity(snapshot.revision.embedding_space)
    )
    service = RetrievalService.from_snapshot(
        snapshot,
        provider,
        ProviderCallContext("request", "retrieval", Deadline.after(10)),
        source_kinds=source_kinds,
    )
    context = RetrievalRequestContext.from_snapshot(
        request_id="request",
        query="policy",
        mode=RetrievalMode.HYBRID,
        snapshot=snapshot,
        cache_policy=CachePolicy.BYPASS,
    )

    result = await service.retrieve(context)

    assert context.cache_policy is CachePolicy.BYPASS
    assert set(result.diagnostics.cache_status.values()) == {CacheOutcome.NOT_APPLICABLE}
    snapshot.close()


async def test_corpus_revision_and_configuration_changes_produce_new_keys(
    tmp_path: Path,
) -> None:
    first, _ = await build_bound_snapshot(tmp_path / "first", revision_id="revision-one")
    second, _ = await build_bound_snapshot(tmp_path / "second", revision_id="revision-two")
    base = _identity()
    first_identity = replace(
        base,
        revision_id=first.revision_id,
        chunk_set_digest=first.revision.chunk_set_digest,
        embedding_identity=first.revision.embedding_space,
        dense_schema_version=first.revision.dense_schema_version,
        dense_metric=first.revision.dense_metric,
        bm25_tokenizer_identity=first.revision.tokenizer_version,
        bm25_schema_version=first.revision.lexical_schema_version,
        bm25_algorithm_version=first.revision.lexical_algorithm_version,
        bm25_k1=first.revision.lexical_k1,
        bm25_b=first.revision.lexical_b,
    )
    second_identity = replace(
        first_identity,
        revision_id=second.revision_id,
        chunk_set_digest=second.revision.chunk_set_digest,
    )

    assert first_identity.key != second_identity.key
    assert first_identity.key != replace(first_identity, dense_weight=2.0).key
    first.close()
    second.close()


async def test_enabled_cache_eliminates_provider_calls_and_rebinds_safe_diagnostics(
    tmp_path: Path,
) -> None:
    snapshot, _ = await build_bound_snapshot(tmp_path)
    provider = DeterministicEmbeddingProvider(
        provider_embedding_identity(snapshot.revision.embedding_space)
    )
    cache = _result_cache()
    settings = _cache_settings(tmp_path)

    first_service = RetrievalService.from_snapshot(
        snapshot,
        provider,
        ProviderCallContext("request-one", "retrieval", Deadline.after(10)),
        settings=settings,
        cache=cache,
    )
    first = await first_service.retrieve(
        RetrievalRequestContext.from_snapshot(
            request_id="request-one",
            query="private lookup 8675309",
            mode=RetrievalMode.HYBRID,
            snapshot=snapshot,
        )
    )
    second_service = RetrievalService.from_snapshot(
        snapshot,
        provider,
        ProviderCallContext("request-two", "retrieval", Deadline.after(10)),
        settings=settings,
        cache=cache,
    )
    second = await second_service.retrieve(
        RetrievalRequestContext.from_snapshot(
            request_id="request-two",
            query="private   lookup\n8675309",
            mode=RetrievalMode.HYBRID,
            snapshot=snapshot,
        )
    )

    assert provider.call_count == 1
    assert first.evidence == second.evidence
    assert first.diagnostics.index_revision == second.diagnostics.index_revision
    assert first.diagnostics.cache_status["retrieval"] is CacheOutcome.MISS
    assert second.diagnostics.cache_status["retrieval"] is CacheOutcome.HIT
    assert second.diagnostics.request_id == "request-two"
    assert second.diagnostics.provider_attempts == ()
    assert second.diagnostics.provider_attempt_counts == {}
    assert second.diagnostics.provider_usage == {}
    counters = cache.metrics.snapshot()
    assert (counters.eligible_lookups, counters.hits, counters.misses, counters.writes) == (
        2,
        1,
        1,
        1,
    )
    assert counters.hit_rate == 0.5
    snapshot.close()


async def test_cache_hit_eliminates_embedding_and_reranking_provider_calls(
    tmp_path: Path,
) -> None:
    snapshot, _ = await build_bound_snapshot(tmp_path)
    embedding = DeterministicEmbeddingProvider(
        provider_embedding_identity(snapshot.revision.embedding_space)
    )
    reranker = DeterministicRerankingProvider()
    cache = _result_cache()
    settings = _cache_settings(tmp_path, reranking_model=reranker.identity.model)

    for request_id in ("rerank-request-one", "rerank-request-two"):
        service = RetrievalService.from_snapshot(
            snapshot,
            embedding,
            ProviderCallContext(request_id, "retrieval", Deadline.after(10)),
            reranker=reranker,
            settings=settings,
            cache=cache,
        )
        result = await service.retrieve(
            RetrievalRequestContext.from_snapshot(
                request_id=request_id,
                query="policy",
                mode=RetrievalMode.HYBRID_RERANK,
                snapshot=snapshot,
            )
        )

    assert result.diagnostics.cache_status["retrieval"] is CacheOutcome.HIT
    assert embedding.call_count == 1
    assert reranker.call_count == 1
    snapshot.close()


async def test_bypass_neither_reads_nor_writes_and_does_not_change_rate_denominator(
    tmp_path: Path,
) -> None:
    snapshot, _ = await build_bound_snapshot(tmp_path)
    provider = DeterministicEmbeddingProvider(
        provider_embedding_identity(snapshot.revision.embedding_space)
    )
    cache = _result_cache()
    settings = _cache_settings(tmp_path)

    async def retrieve(request_id: str, policy: CachePolicy) -> CacheOutcome:
        service = RetrievalService.from_snapshot(
            snapshot,
            provider,
            ProviderCallContext(request_id, "retrieval", Deadline.after(10)),
            settings=settings,
            cache=cache,
        )
        result = await service.retrieve(
            RetrievalRequestContext.from_snapshot(
                request_id=request_id,
                query="policy",
                mode=RetrievalMode.HYBRID,
                snapshot=snapshot,
                cache_policy=policy,
            )
        )
        return result.diagnostics.cache_status["retrieval"]

    assert await retrieve("request-prime", CachePolicy.USE) is CacheOutcome.MISS
    assert await retrieve("request-bypass", CachePolicy.BYPASS) is CacheOutcome.BYPASS
    assert await retrieve("request-hit", CachePolicy.USE) is CacheOutcome.HIT

    assert provider.call_count == 2
    counters = cache.metrics.snapshot()
    assert counters.eligible_lookups == 2
    assert counters.bypasses == 1
    assert counters.hit_rate == 0.5
    snapshot.close()


async def test_ttl_expiration_and_lru_eviction_force_uncached_retrieval(
    tmp_path: Path,
) -> None:
    snapshot, _ = await build_bound_snapshot(tmp_path)
    provider = DeterministicEmbeddingProvider(
        provider_embedding_identity(snapshot.revision.embedding_space)
    )
    now = [0.0]
    cache = _result_cache(maximum_entries=1, ttl_seconds=10, clock=lambda: now[0])
    settings = _cache_settings(tmp_path)

    async def retrieve(request_id: str, query: str) -> CacheOutcome:
        service = RetrievalService.from_snapshot(
            snapshot,
            provider,
            ProviderCallContext(request_id, "retrieval", Deadline.after(10)),
            settings=settings,
            cache=cache,
        )
        result = await service.retrieve(
            RetrievalRequestContext.from_snapshot(
                request_id=request_id,
                query=query,
                mode=RetrievalMode.HYBRID,
                snapshot=snapshot,
            )
        )
        return result.diagnostics.cache_status["retrieval"]

    assert await retrieve("request-one", "first query") is CacheOutcome.MISS
    assert await retrieve("request-two", "second query") is CacheOutcome.MISS
    assert await retrieve("request-three", "first query") is CacheOutcome.MISS
    assert cache.metrics.snapshot().evictions == 2
    now[0] = 11
    assert await retrieve("request-four", "first query") is CacheOutcome.MISS
    counters = cache.metrics.snapshot()
    assert counters.expirations == 1
    assert provider.call_count == 4
    snapshot.close()


async def test_failed_and_degraded_retrievals_are_never_cached(tmp_path: Path) -> None:
    snapshot, _ = await build_bound_snapshot(tmp_path)
    identity = provider_embedding_identity(snapshot.revision.embedding_space)
    cache = _result_cache()
    settings = _cache_settings(tmp_path, allow_single_retriever_degradation=True)
    failure_then_success = _FailingEmbeddingProvider(identity, fail_once=True)

    failed_service = RetrievalService.from_snapshot(
        snapshot,
        failure_then_success,
        ProviderCallContext("request-failed", "retrieval", Deadline.after(10)),
        settings=settings,
        cache=cache,
    )
    with pytest.raises(RetrievalUnavailableError):
        await failed_service.retrieve(
            RetrievalRequestContext.from_snapshot(
                request_id="request-failed",
                query="retry me",
                mode=RetrievalMode.DENSE,
                snapshot=snapshot,
            )
        )

    retry_service = RetrievalService.from_snapshot(
        snapshot,
        failure_then_success,
        ProviderCallContext("request-retry", "retrieval", Deadline.after(10)),
        settings=settings,
        cache=cache,
    )
    retry = await retry_service.retrieve(
        RetrievalRequestContext.from_snapshot(
            request_id="request-retry",
            query="retry me",
            mode=RetrievalMode.DENSE,
            snapshot=snapshot,
        )
    )
    assert retry.diagnostics.cache_status["retrieval"] is CacheOutcome.MISS
    assert failure_then_success.calls == 2

    always_failing = _FailingEmbeddingProvider(identity)
    for request_id in ("request-degraded-one", "request-degraded-two"):
        degraded_service = RetrievalService.from_snapshot(
            snapshot,
            always_failing,
            ProviderCallContext(request_id, "retrieval", Deadline.after(10)),
            settings=settings,
            cache=cache,
        )
        degraded = await degraded_service.retrieve(
            RetrievalRequestContext.from_snapshot(
                request_id=request_id,
                query="lexical fallback only",
                mode=RetrievalMode.HYBRID,
                snapshot=snapshot,
            )
        )
        assert degraded.diagnostics.degradation_reasons
        assert degraded.diagnostics.cache_status["retrieval"] is CacheOutcome.MISS
    assert always_failing.calls == 2
    assert cache.metrics.snapshot().writes == 1
    snapshot.close()


async def test_cancelled_retrieval_is_not_cached_and_later_request_retries(
    tmp_path: Path,
) -> None:
    snapshot, _ = await build_bound_snapshot(tmp_path)
    provider = _BlockingEmbeddingProvider(
        provider_embedding_identity(snapshot.revision.embedding_space)
    )
    cache = _result_cache()
    settings = _cache_settings(tmp_path)
    cancelled_service = RetrievalService.from_snapshot(
        snapshot,
        provider,
        ProviderCallContext("request-cancelled", "retrieval", Deadline.after(10)),
        settings=settings,
        cache=cache,
    )
    cancelled_context = RetrievalRequestContext.from_snapshot(
        request_id="request-cancelled",
        query="cancel then retry",
        mode=RetrievalMode.DENSE,
        snapshot=snapshot,
    )
    task = asyncio.create_task(cancelled_service.retrieve(cancelled_context))
    await asyncio.wait_for(provider.started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cache.metrics.snapshot().writes == 0

    provider.release.set()
    retry_service = RetrievalService.from_snapshot(
        snapshot,
        provider,
        ProviderCallContext("request-after-cancel", "retrieval", Deadline.after(10)),
        settings=settings,
        cache=cache,
    )
    retry = await retry_service.retrieve(
        RetrievalRequestContext.from_snapshot(
            request_id="request-after-cancel",
            query="cancel then retry",
            mode=RetrievalMode.DENSE,
            snapshot=snapshot,
        )
    )

    assert retry.diagnostics.cache_status["retrieval"] is CacheOutcome.MISS
    assert provider.calls == 2
    assert cache.metrics.snapshot().writes == 1
    snapshot.close()


async def test_cache_read_and_write_errors_fail_open_with_safe_counters(tmp_path: Path) -> None:
    snapshot, _ = await build_bound_snapshot(tmp_path)
    provider = DeterministicEmbeddingProvider(
        provider_embedding_identity(snapshot.revision.embedding_space)
    )
    backend = _ExplodingCacheBackend(maximum_entries=2, ttl_seconds=10)
    cache = _result_cache(backend=backend)
    settings = _cache_settings(tmp_path)
    service = RetrievalService.from_snapshot(
        snapshot,
        provider,
        ProviderCallContext("request-error", "retrieval", Deadline.after(10)),
        settings=settings,
        cache=cache,
    )

    result = await service.retrieve(
        RetrievalRequestContext.from_snapshot(
            request_id="request-error",
            query="policy",
            mode=RetrievalMode.HYBRID,
            snapshot=snapshot,
        )
    )

    assert result.evidence
    assert provider.call_count == 1
    assert result.diagnostics.cache_status["retrieval"] is CacheOutcome.ERROR
    counters = cache.metrics.snapshot()
    assert (counters.eligible_lookups, counters.misses, counters.errors) == (1, 1, 2)
    assert counters.hit_rate == 0
    assert "policy" not in repr(counters)
    snapshot.close()


async def test_corrupt_entry_is_replaced_without_retaining_the_raw_query(tmp_path: Path) -> None:
    snapshot, _ = await build_bound_snapshot(tmp_path)
    provider = DeterministicEmbeddingProvider(
        provider_embedding_identity(snapshot.revision.embedding_space)
    )
    backend = BoundedTtlCache[str](maximum_entries=2, ttl_seconds=60)
    cache = _result_cache(backend=backend)
    settings = _cache_settings(tmp_path)
    service = RetrievalService.from_snapshot(
        snapshot,
        provider,
        ProviderCallContext("request-corrupt", "retrieval", Deadline.after(10)),
        settings=settings,
        cache=cache,
    )
    raw_query = "private lookup 8675309"
    context = RetrievalRequestContext.from_snapshot(
        request_id="request-corrupt",
        query=raw_query,
        mode=RetrievalMode.HYBRID,
        snapshot=snapshot,
    )
    key = service._cache_identity(context).key
    backend.put(key, "not valid cache JSON")

    result = await service.retrieve(context)

    assert result.evidence
    assert result.diagnostics.cache_status["retrieval"] is CacheOutcome.ERROR
    assert provider.call_count == 1
    assert cache.metrics.snapshot().errors == 1
    stored = backend.get(key)
    assert stored is not None
    assert raw_query not in key
    assert raw_query not in stored
    snapshot.close()


async def test_schema_valid_forged_snapshot_record_fails_open_without_release(
    tmp_path: Path,
) -> None:
    snapshot, _ = await build_bound_snapshot(tmp_path)
    provider = DeterministicEmbeddingProvider(
        provider_embedding_identity(snapshot.revision.embedding_space)
    )
    backend = BoundedTtlCache[str](maximum_entries=2, ttl_seconds=60)
    cache = _result_cache(backend=backend)
    settings = _cache_settings(tmp_path)
    first_service = RetrievalService.from_snapshot(
        snapshot,
        provider,
        ProviderCallContext("request-prime-forgery", "retrieval", Deadline.after(10)),
        settings=settings,
        cache=cache,
    )
    first_context = RetrievalRequestContext.from_snapshot(
        request_id="request-prime-forgery",
        query="policy",
        mode=RetrievalMode.HYBRID,
        snapshot=snapshot,
    )
    first = await first_service.retrieve(first_context)
    key = first_service._cache_identity(first_context).key
    raw_forged = RetrievalCachePayload.from_result(first).model_dump()
    raw_forged["evidence"][0].update(
        {
            "chunk_id": "chunk-forged",
            "source_id": "forged-source",
            "display_title": "Forged person@example.com",
            "text": "FORGED person@example.com",
        }
    )
    forged = RetrievalCachePayload.model_validate(raw_forged)
    assert cache.store(key, forged) is RetrievalCacheWriteStatus.STORED

    second_service = RetrievalService.from_snapshot(
        snapshot,
        provider,
        ProviderCallContext("request-read-forgery", "retrieval", Deadline.after(10)),
        settings=settings,
        cache=cache,
    )
    second = await second_service.retrieve(
        RetrievalRequestContext.from_snapshot(
            request_id="request-read-forgery",
            query="policy",
            mode=RetrievalMode.HYBRID,
            snapshot=snapshot,
        )
    )

    released = second.model_dump_json()
    assert second.evidence == first.evidence
    assert second.diagnostics.cache_status["retrieval"] is CacheOutcome.ERROR
    assert provider.call_count == 2
    assert "chunk-forged" not in released
    assert "forged-source" not in released
    assert "FORGED" not in released
    assert "person@example.com" not in released
    replaced = backend.get(key)
    assert replaced is not None
    assert "chunk-forged" not in replaced
    assert "forged-source" not in replaced
    assert "person@example.com" not in replaced
    counters = cache.metrics.snapshot()
    assert (counters.eligible_lookups, counters.hits, counters.misses, counters.errors) == (
        2,
        0,
        2,
        1,
    )
    snapshot.close()


@pytest.mark.parametrize("mutation", ["score", "order", "counts", "provider_identity"])
async def test_authenticated_envelope_rejects_schema_valid_backend_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    snapshot, _ = await build_bound_snapshot(tmp_path)
    provider = DeterministicEmbeddingProvider(
        provider_embedding_identity(snapshot.revision.embedding_space)
    )
    backend = BoundedTtlCache[str](maximum_entries=2, ttl_seconds=60)
    cache = _result_cache(backend=backend)
    settings = _cache_settings(tmp_path)
    first_service = RetrievalService.from_snapshot(
        snapshot,
        provider,
        ProviderCallContext("request-prime-envelope", "retrieval", Deadline.after(10)),
        settings=settings,
        cache=cache,
    )
    first_context = RetrievalRequestContext.from_snapshot(
        request_id="request-prime-envelope",
        query="policy",
        mode=RetrievalMode.HYBRID,
        snapshot=snapshot,
    )
    first = await first_service.retrieve(first_context)
    key = first_service._cache_identity(first_context).key
    stored = backend.get(key)
    assert stored is not None
    envelope = json.loads(stored)
    payload = json.loads(envelope["payload"])
    if mutation == "score":
        payload["evidence"][0]["dense_score"] = 123.456
    elif mutation == "order":
        assert len(payload["evidence"]) >= 2
        payload["evidence"][0], payload["evidence"][1] = (
            payload["evidence"][1],
            payload["evidence"][0],
        )
        payload["evidence"][0]["final_rank"] = 1
        payload["evidence"][1]["final_rank"] = 2
    elif mutation == "counts":
        payload["candidate_counts"]["dense"] += 1
    else:
        payload["provider_identities"]["embedding"] = "person@example.com"
    RetrievalCachePayload.model_validate(payload)
    envelope["payload"] = json.dumps(payload, separators=(",", ":"))
    backend.put(key, json.dumps(envelope, separators=(",", ":")))

    second_service = RetrievalService.from_snapshot(
        snapshot,
        provider,
        ProviderCallContext("request-mutated-envelope", "retrieval", Deadline.after(10)),
        settings=settings,
        cache=cache,
    )
    second = await second_service.retrieve(
        RetrievalRequestContext.from_snapshot(
            request_id="request-mutated-envelope",
            query="policy",
            mode=RetrievalMode.HYBRID,
            snapshot=snapshot,
        )
    )

    assert second.evidence == first.evidence
    assert second.diagnostics.cache_status["retrieval"] is CacheOutcome.ERROR
    assert provider.call_count == 2
    assert cache.metrics.snapshot().errors == 1
    assert "person@example.com" not in second.model_dump_json()
    snapshot.close()


async def test_revision_activation_prunes_once_and_rejects_obsolete_inflight_write(
    tmp_path: Path,
) -> None:
    first, _ = await build_bound_snapshot(
        tmp_path / "first",
        revision_id="revision-first",
    )
    second, _ = await build_bound_snapshot(
        tmp_path / "second",
        revision_id="revision-second",
    )
    assert first.revision.published_at is not None
    assert second.revision.published_at is not None
    assert second.revision.published_at > first.revision.published_at
    backend = BoundedTtlCache[str](maximum_entries=8, ttl_seconds=60)
    cache = _result_cache(backend=backend)
    settings = _cache_settings(tmp_path)
    first_provider = DeterministicEmbeddingProvider(
        provider_embedding_identity(first.revision.embedding_space)
    )

    prime_service = RetrievalService.from_snapshot(
        first,
        first_provider,
        ProviderCallContext("revision-prime", "retrieval", Deadline.after(10)),
        settings=settings,
        cache=cache,
    )
    prime = await prime_service.retrieve(
        RetrievalRequestContext.from_snapshot(
            request_id="revision-prime",
            query="policy",
            mode=RetrievalMode.HYBRID,
            snapshot=first,
        )
    )
    same_revision_service = RetrievalService.from_snapshot(
        first,
        first_provider,
        ProviderCallContext("revision-same", "retrieval", Deadline.after(10)),
        settings=settings,
        cache=cache,
    )
    same_revision = await same_revision_service.retrieve(
        RetrievalRequestContext.from_snapshot(
            request_id="revision-same",
            query="policy",
            mode=RetrievalMode.HYBRID,
            snapshot=first,
        )
    )
    assert prime.diagnostics.cache_status["retrieval"] is CacheOutcome.MISS
    assert same_revision.diagnostics.cache_status["retrieval"] is CacheOutcome.HIT
    assert len(backend) == 1

    blocking = _BlockingEmbeddingProvider(
        provider_embedding_identity(first.revision.embedding_space)
    )
    obsolete_service = RetrievalService.from_snapshot(
        first,
        blocking,
        ProviderCallContext("revision-obsolete", "retrieval", Deadline.after(10)),
        settings=settings,
        cache=cache,
    )
    obsolete_task = asyncio.create_task(
        obsolete_service.retrieve(
            RetrievalRequestContext.from_snapshot(
                request_id="revision-obsolete",
                query="obsolete in flight",
                mode=RetrievalMode.HYBRID,
                snapshot=first,
            )
        )
    )
    await asyncio.wait_for(blocking.started.wait(), timeout=5)

    second_provider = DeterministicEmbeddingProvider(
        provider_embedding_identity(second.revision.embedding_space)
    )
    second_service = RetrievalService.from_snapshot(
        second,
        second_provider,
        ProviderCallContext("revision-new", "retrieval", Deadline.after(10)),
        settings=settings,
        cache=cache,
    )
    assert cache.active_revision_id == second.revision_id
    assert len(backend) == 0
    new_result = await second_service.retrieve(
        RetrievalRequestContext.from_snapshot(
            request_id="revision-new",
            query="new revision",
            mode=RetrievalMode.HYBRID,
            snapshot=second,
        )
    )
    assert new_result.diagnostics.cache_status["retrieval"] is CacheOutcome.MISS
    assert len(backend) == 1

    blocking.release.set()
    obsolete_result = await asyncio.wait_for(obsolete_task, timeout=5)
    assert obsolete_result.diagnostics.cache_status["retrieval"] is CacheOutcome.MISS
    assert len(backend) == 1
    assert cache.metrics.snapshot().writes == 2

    RetrievalService.from_snapshot(
        first,
        first_provider,
        ProviderCallContext("revision-old-rebind", "retrieval", Deadline.after(10)),
        settings=settings,
        cache=cache,
    )
    assert cache.active_revision_id == second.revision_id
    assert len(backend) == 1
    first.close()
    second.close()


async def test_revision_and_retrieval_configuration_are_isolated(tmp_path: Path) -> None:
    first, _ = await build_bound_snapshot(tmp_path / "first", revision_id="revision-one")
    second, _ = await build_bound_snapshot(tmp_path / "second", revision_id="revision-two")
    cache = _result_cache()
    settings = _cache_settings(tmp_path)
    provider_one = DeterministicEmbeddingProvider(
        provider_embedding_identity(first.revision.embedding_space)
    )
    provider_two = DeterministicEmbeddingProvider(
        provider_embedding_identity(second.revision.embedding_space)
    )

    async def retrieve(
        request_id: str,
        snapshot: Any,
        provider: DeterministicEmbeddingProvider,
        rrf: RrfConfig,
    ) -> CacheOutcome:
        service = RetrievalService.from_snapshot(
            snapshot,
            provider,
            ProviderCallContext(request_id, "retrieval", Deadline.after(10)),
            settings=settings,
            cache=cache,
            rrf=rrf,
        )
        result = await service.retrieve(
            RetrievalRequestContext.from_snapshot(
                request_id=request_id,
                query="policy",
                mode=RetrievalMode.HYBRID,
                snapshot=snapshot,
            )
        )
        return result.diagnostics.cache_status["retrieval"]

    base_rrf = RrfConfig(dense_weight=1, lexical_weight=1)
    changed_rrf = RrfConfig(dense_weight=2, lexical_weight=1)
    assert await retrieve("request-one", first, provider_one, base_rrf) is CacheOutcome.MISS
    assert await retrieve("request-two", first, provider_one, changed_rrf) is CacheOutcome.MISS
    assert await retrieve("request-three", second, provider_two, base_rrf) is CacheOutcome.MISS
    assert provider_one.call_count == 2
    assert provider_two.call_count == 1
    assert cache.metrics.snapshot().hits == 0
    first.close()
    second.close()
