from __future__ import annotations

import pytest
from conftest import candidate

from rag_mvp.retrieval.fusion import RrfConfig, weighted_rrf


def test_weighted_rrf_matches_rank_formula() -> None:
    results = weighted_rrf(
        (candidate("both", dense_rank=1), candidate("dense", dense_rank=2)),
        (candidate("both", bm25_rank=2), candidate("lexical", bm25_rank=1)),
        config=RrfConfig(k=60, dense_weight=1.0, lexical_weight=2.0),
    )

    assert results[0].chunk_id == "both"
    assert results[0].rrf_score == pytest.approx(1 / 61 + 2 / 62)


def test_rrf_tie_breaks_by_best_rank_then_chunk_id() -> None:
    results = weighted_rrf(
        (candidate("b", dense_rank=1), candidate("a", dense_rank=1)),
        (),
    )

    assert [item.chunk_id for item in results] == ["a", "b"]
