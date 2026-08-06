from __future__ import annotations

import pytest
from retrieval_test_helpers import candidate

from rag_mvp.domain.ingestion import ChunkLocator
from rag_mvp.domain.retrieval import RetrievalCandidate
from rag_mvp.retrieval.fusion import CandidateIntegrityError, RrfConfig, weighted_rrf


def test_weighted_rrf_matches_rank_formula_for_sparse_out_of_order_ranks() -> None:
    results = weighted_rrf(
        (candidate("both", dense_rank=1), candidate("dense", dense_rank=20)),
        (candidate("both", bm25_rank=9), candidate("lexical", bm25_rank=1)),
        config=RrfConfig(k=60, dense_weight=1.0, lexical_weight=2.0),
    )

    both = next(item for item in results if item.chunk_id == "both")
    assert both.rrf_score == pytest.approx(1 / 61 + 2 / 69)
    assert both.dense_score == 1.0
    assert both.bm25_score == pytest.approx(1 / 9)


def test_rrf_uses_only_ranks_not_raw_scores() -> None:
    results = weighted_rrf(
        (
            candidate("low-score", dense_rank=1).model_copy(update={"dense_score": -999.0}),
            candidate("high-score", dense_rank=2).model_copy(update={"dense_score": 999.0}),
        ),
        (),
    )

    assert [item.chunk_id for item in results] == ["low-score", "high-score"]
    assert results[0].rrf_score == pytest.approx(1 / 61)
    assert results[1].rrf_score == pytest.approx(1 / 62)


def test_rrf_permits_tied_explicit_source_ranks() -> None:
    results = weighted_rrf(
        (candidate("b", dense_rank=4), candidate("a", dense_rank=4)),
        (),
    )

    assert [item.chunk_id for item in results] == ["a", "b"]


def test_equal_score_prefers_better_individual_rank() -> None:
    results = weighted_rrf(
        (candidate("best-rank", dense_rank=1),),
        (candidate("worse-rank", bm25_rank=62),),
        config=RrfConfig(k=60, dense_weight=1.0, lexical_weight=2.0),
    )

    assert results[0].rrf_score == results[1].rrf_score
    assert [item.chunk_id for item in results] == ["best-rank", "worse-rank"]


def test_exact_score_and_best_rank_tie_breaks_by_chunk_id() -> None:
    results = weighted_rrf(
        (candidate("b", dense_rank=7),),
        (candidate("a", bm25_rank=7),),
    )

    assert [item.chunk_id for item in results] == ["a", "b"]


@pytest.mark.parametrize(
    "values",
    [
        {"k": 0},
        {"k": True},
        {"k": 1.5},
        {"dense_weight": 0.0},
        {"dense_weight": True},
        {"dense_weight": "1"},
        {"dense_weight": float("nan")},
        {"lexical_weight": float("inf")},
        {"lexical_weight": float("-inf")},
        {"version": "weighted-rrf-v2"},
        {"version": 1},
    ],
)
def test_rrf_config_rejects_invalid_types_values_and_versions(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="rrf_"):
        RrfConfig(**values)  # type: ignore[arg-type]


def test_rrf_requires_config_instance() -> None:
    with pytest.raises(TypeError, match="RrfConfig"):
        weighted_rrf((candidate("a", dense_rank=1),), (), config={})  # type: ignore[arg-type]


def test_rrf_rejects_rankless_candidate() -> None:
    rankless = RetrievalCandidate(
        chunk_id="rankless",
        source_id="source-1",
        display_title="Policy",
        document_version=1,
        locator=ChunkLocator(pages=(1,)),
        text="Evidence",
    )

    with pytest.raises(CandidateIntegrityError, match="dense_rank_invalid"):
        weighted_rrf((rankless,), ())


@pytest.mark.parametrize("rank", [0, -1, True, 1.0, "1"])
def test_rrf_revalidates_candidates_and_rejects_rank_bypass(rank: object) -> None:
    invalid = RetrievalCandidate.model_construct(
        chunk_id="invalid",
        source_id="source-1",
        display_title="Policy",
        document_version=1,
        locator=ChunkLocator(pages=(1,)),
        text="Evidence",
        dense_rank=rank,
        dense_score=1.0,
    )

    with pytest.raises(CandidateIntegrityError, match="dense_rank_invalid"):
        weighted_rrf((invalid,), ())


@pytest.mark.parametrize(
    "invalid",
    [
        candidate("prefused", dense_rank=1).model_copy(update={"rrf_score": 0.1}),
        candidate("reranked", dense_rank=1).model_copy(update={"reranking_rank": 1}),
        candidate("fabricated", dense_rank=1).model_copy(
            update={"bm25_rank": 1, "bm25_score": 0.0}
        ),
    ],
)
def test_rrf_rejects_fabricated_later_or_other_channel_fields(
    invalid: RetrievalCandidate,
) -> None:
    with pytest.raises(CandidateIntegrityError):
        weighted_rrf((invalid,), ())
