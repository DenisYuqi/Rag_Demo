from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import pytest
from retrieval_test_helpers import build_bound_snapshot

from rag_mvp.domain.ingestion import EmbeddingSpaceIdentity
from rag_mvp.domain.retrieval import CacheOutcome, CachePolicy, RetrievalMode
from rag_mvp.providers.fakes import DeterministicEmbeddingProvider
from rag_mvp.providers.models import Deadline, ProviderCallContext
from rag_mvp.retrieval.cache import BoundedTtlCache, RetrievalCacheIdentity
from rag_mvp.retrieval.identity import provider_embedding_identity
from rag_mvp.retrieval.request import (
    QUERY_CANONICALIZATION_VERSION,
    RetrievalRequestContext,
)
from rag_mvp.retrieval.service import (
    DEGRADATION_POLICY_VERSION,
    RRF_TIE_POLICY_VERSION,
    RetrievalService,
)


def _identity() -> RetrievalCacheIdentity:
    return RetrievalCacheIdentity(
        canonical_query="policy",
        canonicalization_version=QUERY_CANONICALIZATION_VERSION,
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
