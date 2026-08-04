from __future__ import annotations

import asyncio
from collections.abc import Sequence

from conftest import candidate

from rag_mvp.domain.retrieval import RetrievalCandidate, RetrievalMode
from rag_mvp.retrieval.request import RetrievalRequestContext
from rag_mvp.retrieval.service import RetrievalService


class StaticRetriever:
    def __init__(self, results: tuple[RetrievalCandidate, ...]) -> None:
        self.results = results

    async def search(self, query: str, limit: int) -> tuple[RetrievalCandidate, ...]:
        return self.results


class InvalidReranker:
    identity = "invalid-v1"

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
    ) -> tuple[str, ...]:
        return (candidates[0].chunk_id, "invented")


class SlowReranker:
    identity = "slow-v1"

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
    ) -> tuple[str, ...]:
        await asyncio.sleep(1)
        return tuple(candidate.chunk_id for candidate in candidates)


async def test_invalid_rerank_falls_back_to_rrf() -> None:
    dense = StaticRetriever((candidate("a", dense_rank=1), candidate("b", dense_rank=2)))
    lexical = StaticRetriever(())
    service = RetrievalService(dense=dense, lexical=lexical, reranker=InvalidReranker())

    result = await service.retrieve(
        RetrievalRequestContext("req", "query", RetrievalMode.HYBRID_RERANK, "rev")
    )

    assert [item.chunk_id for item in result.evidence] == ["a", "b"]
    assert "rerank_degraded" in result.diagnostics.degradation_reasons


async def test_rerank_timeout_falls_back_without_waiting_past_budget() -> None:
    dense = StaticRetriever((candidate("a", dense_rank=1),))
    service = RetrievalService(
        dense=dense,
        lexical=StaticRetriever(()),
        reranker=SlowReranker(),
        rerank_deadline_seconds=0.01,
    )

    result = await service.retrieve(
        RetrievalRequestContext("req", "query", RetrievalMode.HYBRID_RERANK, "rev")
    )

    assert result.evidence[0].chunk_id == "a"
    assert "rerank_degraded" in result.diagnostics.degradation_reasons
