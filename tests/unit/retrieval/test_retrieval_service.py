from __future__ import annotations

from collections.abc import Sequence

from conftest import candidate

from rag_mvp.domain.retrieval import RetrievalCandidate, RetrievalMode
from rag_mvp.retrieval.request import RetrievalRequestContext
from rag_mvp.retrieval.service import RetrievalService


class SpyRetriever:
    def __init__(self, results: tuple[RetrievalCandidate, ...], *, fails: bool = False) -> None:
        self.results = results
        self.fails = fails
        self.calls = 0

    async def search(self, query: str, limit: int) -> tuple[RetrievalCandidate, ...]:
        self.calls += 1
        if self.fails:
            raise RuntimeError("safe fake failure")
        return self.results


class ReverseReranker:
    identity = "reverse-v1"

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
    ) -> tuple[str, ...]:
        return tuple(candidate.chunk_id for candidate in reversed(candidates))


async def test_dense_mode_does_not_call_lexical_or_reranker() -> None:
    dense = SpyRetriever((candidate("a", dense_rank=1),))
    lexical = SpyRetriever(())
    service = RetrievalService(dense=dense, lexical=lexical, reranker=ReverseReranker())

    result = await service.retrieve(
        RetrievalRequestContext("req", "query", RetrievalMode.DENSE, "rev")
    )

    assert result.evidence[0].chunk_id == "a"
    assert dense.calls == 1
    assert lexical.calls == 0


async def test_hybrid_degrades_to_one_successful_retriever() -> None:
    dense = SpyRetriever((), fails=True)
    lexical = SpyRetriever((candidate("lexical", bm25_rank=1),))
    service = RetrievalService(dense=dense, lexical=lexical)

    result = await service.retrieve(
        RetrievalRequestContext("req", "query", RetrievalMode.HYBRID, "rev")
    )

    assert result.evidence[0].chunk_id == "lexical"
    assert result.diagnostics.failed_stages == ("dense",)
    assert result.diagnostics.effective_mode is RetrievalMode.HYBRID


async def test_hybrid_rerank_applies_valid_order() -> None:
    dense = SpyRetriever((candidate("a", dense_rank=1), candidate("b", dense_rank=2)))
    service = RetrievalService(
        dense=dense,
        lexical=SpyRetriever(()),
        reranker=ReverseReranker(),
    )

    result = await service.retrieve(
        RetrievalRequestContext("req", "query", RetrievalMode.HYBRID_RERANK, "rev")
    )

    assert [item.chunk_id for item in result.evidence] == ["b", "a"]
