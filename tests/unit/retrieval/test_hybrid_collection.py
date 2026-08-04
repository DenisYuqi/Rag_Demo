from __future__ import annotations

import asyncio

from conftest import candidate

from rag_mvp.domain.retrieval import RetrievalMode
from rag_mvp.retrieval.request import RetrievalRequestContext
from rag_mvp.retrieval.service import RetrievalService


class OverlapRetriever:
    def __init__(
        self,
        result: tuple[object, ...],
        barrier: asyncio.Event,
        counter: list[int],
    ) -> None:
        self._result = result
        self._barrier = barrier
        self._counter = counter

    async def search(self, query: str, limit: int) -> tuple[object, ...]:
        self._counter[0] += 1
        if self._counter[0] == 2:
            self._barrier.set()
        await asyncio.wait_for(self._barrier.wait(), timeout=0.2)
        return self._result


async def test_hybrid_collects_in_parallel_and_merges_chunk_ids() -> None:
    barrier = asyncio.Event()
    counter = [0]
    dense = OverlapRetriever((candidate("shared", dense_rank=1),), barrier, counter)
    lexical = OverlapRetriever((candidate("shared", bm25_rank=1),), barrier, counter)
    service = RetrievalService(dense=dense, lexical=lexical)
    context = RetrievalRequestContext("req", "policy", RetrievalMode.HYBRID, "rev")

    result = await service.retrieve(context)

    assert counter[0] == 2
    assert len(result.evidence) == 1
    assert result.evidence[0].dense_rank == 1
    assert result.evidence[0].bm25_rank == 1
