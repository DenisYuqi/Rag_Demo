from __future__ import annotations

from dataclasses import replace

from rag_mvp.retrieval.cache import BoundedTtlCache, RetrievalCacheIdentity


def _identity() -> RetrievalCacheIdentity:
    return RetrievalCacheIdentity(
        canonical_query="policy",
        revision_id="revision-1",
        mode="hybrid-rerank",
        embedding_identity="embed-v1",
        bm25_version="bm25-v1",
        rrf_version="rrf-v1",
        rrf_k=60,
        dense_weight=1.0,
        lexical_weight=1.0,
        reranker_identity="rerank-v1",
        reranker_prompt_version="prompt-v1",
        dense_limit=20,
        lexical_limit=20,
        rerank_limit=10,
        final_limit=5,
    )


def test_every_versioned_change_produces_a_cache_miss() -> None:
    base = _identity()

    assert base.key != replace(base, revision_id="revision-2").key
    assert base.key != replace(base, dense_weight=2.0).key
    assert base.key != replace(base, reranker_prompt_version="prompt-v2").key
    assert base.key != replace(base, final_limit=4).key


def test_ttl_and_size_are_bounded() -> None:
    cache: BoundedTtlCache[str] = BoundedTtlCache(maximum_entries=1, ttl_seconds=10)
    cache.put("first", "one", now=0)
    cache.put("second", "two", now=1)

    assert cache.get("first", now=2) is None
    assert cache.get("second", now=2) == "two"
    assert cache.get("second", now=11) is None
