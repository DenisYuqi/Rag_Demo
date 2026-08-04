from __future__ import annotations

import pytest

from rag_mvp.domain.retrieval import RetrievalMode
from rag_mvp.retrieval.request import (
    RetrievalRequestContext,
    RetrievalRequestError,
    canonicalize_query,
)


def test_query_is_unicode_canonical_and_whitespace_bounded() -> None:
    assert canonicalize_query("  Cafe\u0301\n policy  ") == "Café policy"


def test_request_binds_one_revision_snapshot() -> None:
    active_pointer = "revision-1"
    context = RetrievalRequestContext.bind(
        request_id="request-1",
        query="policy",
        mode="hybrid",
        active_revision_id=active_pointer,
    )
    active_pointer = "revision-2"

    assert context.revision_id == "revision-1"
    assert context.mode is RetrievalMode.HYBRID
    assert active_pointer == "revision-2"


@pytest.mark.parametrize(
    ("query", "revision", "code"),
    [(" ", "rev", "empty_query"), ("valid", None, "index_not_ready")],
)
def test_invalid_request_has_stable_error(query: str, revision: str | None, code: str) -> None:
    with pytest.raises(RetrievalRequestError, match=code):
        RetrievalRequestContext.bind(
            request_id="request-1",
            query=query,
            mode="dense",
            active_revision_id=revision,
        )
