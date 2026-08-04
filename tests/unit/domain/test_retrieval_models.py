from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_mvp.domain.ingestion import ChunkLocator
from rag_mvp.domain.retrieval import (
    CacheOutcome,
    RankingEvidence,
    RetrievalCandidate,
    RetrievalDiagnostics,
    RetrievalMode,
    RetrievalResult,
)


def _candidate_values() -> dict[str, object]:
    return {
        "chunk_id": "chunk-1",
        "source_id": "source-1",
        "display_title": "Policy",
        "document_version": 1,
        "locator": ChunkLocator(pages=(3,)),
        "text": "Grounded evidence",
        "dense_rank": 1,
        "dense_score": 0.8,
        "bm25_rank": 2,
        "bm25_score": 4.2,
        "rrf_score": 0.03,
    }


def test_retrieval_modes_use_stable_wire_values() -> None:
    assert [mode.value for mode in RetrievalMode] == [
        "dense",
        "hybrid",
        "hybrid-rerank",
    ]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_candidate_rejects_non_finite_provider_scores(score: float) -> None:
    with pytest.raises(ValidationError):
        RetrievalCandidate.model_validate({**_candidate_values(), "dense_score": score})


def test_ranked_result_and_diagnostics_round_trip() -> None:
    evidence = RankingEvidence.model_validate({**_candidate_values(), "final_rank": 1})
    diagnostics = RetrievalDiagnostics(
        request_id="request-1",
        requested_mode=RetrievalMode.HYBRID_RERANK,
        effective_mode=RetrievalMode.HYBRID,
        index_revision="revision-1",
        candidate_counts={"dense": 20, "bm25": 20, "final": 1},
        stage_timings_ms={"dense": 12.5},
        cache_status={"retrieval": CacheOutcome.MISS},
        provider_identities={"embedding": "primary/embed-v1"},
        degradation_reasons=("reranker-timeout",),
    )
    result = RetrievalResult(evidence=(evidence,), diagnostics=diagnostics)

    decoded = RetrievalResult.model_validate_json(result.model_dump_json())

    assert decoded == result
    assert decoded.evidence[0].locator.pages == (3,)
    assert decoded.diagnostics.degradation_reasons == ("reranker-timeout",)
