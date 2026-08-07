from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass

import pytest

from rag_mvp.domain.ingestion import ChunkLocator
from rag_mvp.domain.retrieval import RetrievalCandidate
from rag_mvp.retrieval.collection import collect_hybrid_candidates
from rag_mvp.retrieval.fusion import CandidateIntegrityError


def _candidate(
    chunk_id: str,
    *,
    channel: str,
    rank: int,
    score: float = 1.0,
    revision_id: str = "revision-1",
    text: str | None = None,
    source_id: str = "source-1",
    rrf_score: float | None = None,
    reranking_rank: int | None = None,
    other_rank: int | None = None,
) -> RetrievalCandidate:
    exact_text = text if text is not None else f"  Evidence for {chunk_id}\n"
    digest = hashlib.sha256(exact_text.encode()).hexdigest()
    values: dict[str, object] = {
        "chunk_id": chunk_id,
        "parent_chunk_id": f"parent-{chunk_id}",
        "source_id": source_id,
        "display_title": "Policy",
        "document_version": 1,
        "locator": ChunkLocator(pages=(1,)),
        "text": exact_text,
        "revision_id": revision_id,
        "ordinal": 0,
        "content_digest": digest,
        "record_digest": hashlib.sha256(f"record:{chunk_id}:{digest}".encode()).hexdigest(),
        "rrf_score": rrf_score,
        "reranking_rank": reranking_rank,
    }
    if channel == "dense":
        values.update(dense_rank=rank, dense_score=score)
        if other_rank is not None:
            values.update(bm25_rank=other_rank, bm25_score=score)
    else:
        values.update(bm25_rank=rank, bm25_score=score)
        if other_rank is not None:
            values.update(dense_rank=other_rank, dense_score=score)
    return RetrievalCandidate.model_validate(values)


@dataclass
class StaticRetriever:
    revision_id: str
    results: tuple[RetrievalCandidate, ...]
    barrier: asyncio.Event | None = None
    counter: list[int] | None = None
    calls: int = 0
    seen_query: str | None = None
    seen_limit: int | None = None

    async def search(self, query: str, limit: int) -> tuple[RetrievalCandidate, ...]:
        self.calls += 1
        self.seen_query = query
        self.seen_limit = limit
        if self.barrier is not None and self.counter is not None:
            self.counter[0] += 1
            if self.counter[0] == 2:
                self.barrier.set()
            await asyncio.wait_for(self.barrier.wait(), timeout=0.2)
        return self.results


async def test_collects_in_parallel_caps_results_and_preserves_zero_scores() -> None:
    barrier = asyncio.Event()
    counter = [0]
    dense = StaticRetriever(
        "revision-1",
        (
            _candidate("shared", channel="dense", rank=1, score=0.0),
            _candidate("dense-only", channel="dense", rank=2, score=0.0),
            _candidate("capped", channel="dense", rank=3),
        ),
        barrier,
        counter,
    )
    bm25 = StaticRetriever(
        "revision-1",
        (
            _candidate("shared", channel="bm25", rank=1, score=0.0),
            _candidate("bm25-only", channel="bm25", rank=2, score=0.0),
            _candidate("also-capped", channel="bm25", rank=3),
        ),
        barrier,
        counter,
    )

    result = await collect_hybrid_candidates(
        "  policy\nterms ",
        dense=dense,
        bm25=bm25,
        dense_limit=2,
        bm25_limit=2,
    )

    assert counter[0] == 2
    assert dense.seen_query == bm25.seen_query == "policy terms"
    assert dense.seen_limit == bm25.seen_limit == 2
    assert [item.chunk_id for item in result.dense] == ["shared", "dense-only"]
    assert [item.chunk_id for item in result.bm25] == ["shared", "bm25-only"]
    assert {item.chunk_id for item in result.merged} == {
        "shared",
        "dense-only",
        "bm25-only",
    }
    shared = next(item for item in result.merged if item.chunk_id == "shared")
    assert shared.dense_rank == 1 and shared.bm25_rank == 1
    assert shared.dense_score == 0.0
    assert shared.bm25_score == 0.0
    assert shared.text == "  Evidence for shared\n"


async def test_retriever_revision_mismatch_fails_before_search() -> None:
    dense = StaticRetriever("revision-1", ())
    bm25 = StaticRetriever("revision-2", ())

    with pytest.raises(ValueError, match="mixed_retriever_revisions"):
        await collect_hybrid_candidates(
            "policy",
            dense=dense,
            bm25=bm25,
            dense_limit=2,
            bm25_limit=2,
        )

    assert dense.calls == bm25.calls == 0


async def test_candidate_revision_mismatch_fails_closed() -> None:
    dense = StaticRetriever(
        "revision-1",
        (_candidate("dense", channel="dense", rank=1, revision_id="revision-2"),),
    )
    bm25 = StaticRetriever("revision-1", ())

    with pytest.raises(CandidateIntegrityError, match="dense_revision_mismatch"):
        await collect_hybrid_candidates(
            "policy",
            dense=dense,
            bm25=bm25,
            dense_limit=2,
            bm25_limit=2,
        )


class BlockingRetriever:
    revision_id = "revision-1"

    def __init__(self, started: asyncio.Event) -> None:
        self.started = started
        self.cancelled = False

    async def search(self, query: str, limit: int) -> tuple[RetrievalCandidate, ...]:
        del query, limit
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


async def test_external_cancellation_propagates_to_both_retrievers() -> None:
    dense_started = asyncio.Event()
    bm25_started = asyncio.Event()
    dense = BlockingRetriever(dense_started)
    bm25 = BlockingRetriever(bm25_started)
    task = asyncio.create_task(
        collect_hybrid_candidates(
            "policy",
            dense=dense,
            bm25=bm25,
            dense_limit=2,
            bm25_limit=2,
        )
    )
    await asyncio.gather(dense_started.wait(), bm25_started.wait())
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert dense.cancelled
    assert bm25.cancelled


@pytest.mark.parametrize(
    ("dense_results", "bm25_results", "code"),
    [
        (
            (
                _candidate("duplicate", channel="dense", rank=1),
                _candidate("duplicate", channel="dense", rank=2),
            ),
            (),
            "dense_duplicate_chunk_id",
        ),
        (
            (
                _candidate("first", channel="dense", rank=1),
                _candidate("second", channel="dense", rank=1),
            ),
            (),
            "dense_rank_invalid",
        ),
        (
            (_candidate("first", channel="dense", rank=2),),
            (),
            "dense_rank_invalid",
        ),
        (
            (_candidate("first", channel="dense", rank=1, other_rank=1),),
            (),
            "dense_provenance_invalid",
        ),
        (
            (
                _candidate("first", channel="dense", rank=1).model_copy(
                    update={"dense_score": None}
                ),
            ),
            (),
            "dense_score_missing",
        ),
        (
            (_candidate("first", channel="dense", rank=1, rrf_score=0.1),),
            (),
            "dense_stage_fields_invalid",
        ),
        (
            (),
            (_candidate("first", channel="bm25", rank=1, reranking_rank=1),),
            "bm25_stage_fields_invalid",
        ),
    ],
)
async def test_collection_rejects_duplicate_rank_provenance_and_stage_errors(
    dense_results: tuple[RetrievalCandidate, ...],
    bm25_results: tuple[RetrievalCandidate, ...],
    code: str,
) -> None:
    with pytest.raises(CandidateIntegrityError, match=code):
        await collect_hybrid_candidates(
            "policy",
            dense=StaticRetriever("revision-1", dense_results),
            bm25=StaticRetriever("revision-1", bm25_results),
            dense_limit=3,
            bm25_limit=3,
        )


async def test_collection_rejects_incomplete_and_mismatched_candidate_identity() -> None:
    incomplete = _candidate("incomplete", channel="dense", rank=1)
    incomplete = RetrievalCandidate.model_validate(
        {**incomplete.model_dump(), "record_digest": None}
    )
    with pytest.raises(CandidateIntegrityError, match="dense_identity_incomplete"):
        await collect_hybrid_candidates(
            "policy",
            dense=StaticRetriever("revision-1", (incomplete,)),
            bm25=StaticRetriever("revision-1", ()),
            dense_limit=2,
            bm25_limit=2,
        )

    dense_shared = _candidate("shared", channel="dense", rank=1, text="dense text")
    bm25_shared = _candidate("shared", channel="bm25", rank=1, text="different text")
    with pytest.raises(CandidateIntegrityError, match="chunk_metadata_mismatch"):
        await collect_hybrid_candidates(
            "policy",
            dense=StaticRetriever("revision-1", (dense_shared,)),
            bm25=StaticRetriever("revision-1", (bm25_shared,)),
            dense_limit=2,
            bm25_limit=2,
        )
