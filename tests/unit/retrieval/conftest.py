from __future__ import annotations

from rag_mvp.domain.ingestion import ChunkLocator
from rag_mvp.domain.retrieval import RetrievalCandidate


def candidate(
    chunk_id: str,
    *,
    dense_rank: int | None = None,
    bm25_rank: int | None = None,
    page: int = 1,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        source_id="source-1",
        display_title="Policy",
        document_version=1,
        locator=ChunkLocator(pages=(page,)),
        text=f"Evidence for {chunk_id}",
        dense_rank=dense_rank,
        dense_score=1.0 / dense_rank if dense_rank else None,
        bm25_rank=bm25_rank,
        bm25_score=1.0 / bm25_rank if bm25_rank else None,
    )
