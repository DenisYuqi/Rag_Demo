from __future__ import annotations

from conftest import candidate

from rag_mvp.domain.retrieval import RetrievalMode
from rag_mvp.retrieval.request import RetrievalRequestContext
from rag_mvp.retrieval.service import RetrievalService


class StaticRetriever:
    async def search(self, query: str, limit: int) -> tuple[object, ...]:
        return (candidate("pdf-evidence", dense_rank=1, page=7),)


async def test_evidence_contains_exact_text_locator_and_real_scores() -> None:
    service = RetrievalService(dense=StaticRetriever(), lexical=StaticRetriever())

    result = await service.retrieve(
        RetrievalRequestContext("req", "policy", RetrievalMode.DENSE, "rev-1")
    )

    evidence = result.evidence[0]
    assert evidence.text == "Evidence for pdf-evidence"
    assert evidence.locator.pages == (7,)
    assert evidence.final_rank == 1
    assert evidence.dense_score == 1.0
    assert evidence.bm25_score is None
