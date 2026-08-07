import hashlib
from collections.abc import Sequence

import pytest

from rag_mvp.domain.ingestion import ChunkLocator, ParentChunk
from rag_mvp.domain.retrieval import RankingEvidence
from rag_mvp.ingestion.chunking import token_spans
from rag_mvp.qa.context import ContextBuilder, ContextSelectionError


class _Parents:
    def __init__(self, values: tuple[ParentChunk, ...]) -> None:
        self.values = {value.parent_chunk_id: value for value in values}

    def get_many(
        self,
        revision_id: str,
        parent_chunk_ids: Sequence[str],
    ) -> dict[str, ParentChunk]:
        assert revision_id == "revision-parent-context"
        return {
            parent_id: self.values[parent_id]
            for parent_id in parent_chunk_ids
            if parent_id in self.values
        }


def _parent(parent_id: str, text: str, *, source_id: str = "source-shared") -> ParentChunk:
    return ParentChunk(
        parent_chunk_id=parent_id,
        source_id=source_id,
        document_version=1,
        ordinal=0,
        text=text,
        content_digest=hashlib.sha256(text.encode()).hexdigest(),
        locator=ChunkLocator(pages=(1,)),
        token_count=len(token_spans(text)),
    )


def _evidence(chunk_id: str, final_rank: int, text: str) -> RankingEvidence:
    return RankingEvidence(
        chunk_id=chunk_id,
        parent_chunk_id=chunk_id,
        source_id=f"source-{chunk_id}",
        display_title=f"Policy {chunk_id}",
        document_version=1,
        locator=ChunkLocator(pages=(final_rank,)),
        text=text,
        final_rank=final_rank,
    )


def test_context_uses_final_rank_order_and_chunk_count_limit() -> None:
    evidence = (
        _evidence("chunk-3", 3, "third ranked evidence"),
        _evidence("chunk-1", 1, "first ranked evidence"),
        _evidence("chunk-2", 2, "second ranked evidence"),
    )

    result = ContextBuilder(maximum_chunks=2).build(evidence)

    assert [chunk.chunk_id for chunk in result.chunks] == ["chunk-1", "chunk-2"]
    assert [chunk.final_rank for chunk in result.chunks] == [1, 2]
    assert result.available_evidence_count == 3
    assert result.omitted_evidence_count == 1


def test_per_chunk_limit_truncates_at_bilingual_token_boundaries() -> None:
    result = ContextBuilder(
        maximum_chunks=2,
        maximum_tokens_per_chunk=3,
        maximum_total_tokens=6,
    ).build(
        (
            _evidence("english", 1, "one two three four"),
            _evidence("chinese", 2, "员工休假政策"),
        )
    )

    english, chinese = result.chunks
    assert english.text.rstrip() == "one two three"
    assert chinese.text == "员工休"
    assert english.token_count == chinese.token_count == 3
    assert english.truncated and chinese.truncated
    assert result.total_tokens == 6
    assert result.truncated_chunk_count == 2


def test_total_budget_is_applied_in_rank_order_and_can_partially_fill_last_chunk() -> None:
    result = ContextBuilder(
        maximum_chunks=3,
        maximum_tokens_per_chunk=3,
        maximum_total_tokens=5,
    ).build(
        (
            _evidence("first", 1, "one two three four"),
            _evidence("second", 2, "five six seven eight"),
            _evidence("third", 3, "nine ten eleven twelve"),
        )
    )

    assert [chunk.chunk_id for chunk in result.chunks] == ["first", "second"]
    assert [chunk.token_count for chunk in result.chunks] == [3, 2]
    assert len(token_spans(result.chunks[1].text)) == 2
    assert result.total_tokens == 5
    assert result.omitted_evidence_count == 1


def test_context_keeps_original_evidence_for_later_citation_validation() -> None:
    evidence = _evidence("policy", 1, "one two three four")

    context = (
        ContextBuilder(
            maximum_tokens_per_chunk=2,
            maximum_total_tokens=2,
        )
        .build((evidence,))
        .chunks[0]
    )

    assert context.text.rstrip() == "one two"
    assert context.evidence is evidence
    assert context.evidence.text == "one two three four"
    assert context.evidence.locator.pages == (1,)


def test_empty_evidence_produces_an_empty_bounded_selection() -> None:
    result = ContextBuilder().build(())

    assert result.chunks == ()
    assert result.total_tokens == 0
    assert result.available_evidence_count == 0
    assert result.omitted_evidence_count == 0


@pytest.mark.parametrize(
    ("evidence", "code"),
    [
        (
            (
                _evidence("first", 1, "first"),
                _evidence("third", 3, "third"),
            ),
            "evidence_ranks_invalid",
        ),
        (
            (
                _evidence("duplicate", 1, "first"),
                _evidence("duplicate", 2, "second"),
            ),
            "duplicate_evidence_chunk",
        ),
    ],
)
def test_context_rejects_invalid_ranked_evidence(
    evidence: tuple[RankingEvidence, ...],
    code: str,
) -> None:
    with pytest.raises(ContextSelectionError, match=code):
        ContextBuilder().build(evidence)


def test_context_expands_and_deduplicates_parents_but_keeps_best_child_identity() -> None:
    parent = _parent("parent-shared", "full parent context with surrounding policy details")
    evidence = (
        RankingEvidence(
            chunk_id="child-best",
            parent_chunk_id=parent.parent_chunk_id,
            source_id=parent.source_id,
            display_title="Policy",
            document_version=1,
            locator=ChunkLocator(pages=(1,)),
            text="policy details",
            revision_id="revision-parent-context",
            final_rank=1,
        ),
        RankingEvidence(
            chunk_id="child-second",
            parent_chunk_id=parent.parent_chunk_id,
            source_id=parent.source_id,
            display_title="Policy",
            document_version=1,
            locator=ChunkLocator(pages=(1,)),
            text="surrounding policy",
            revision_id="revision-parent-context",
            final_rank=2,
        ),
    )

    result = ContextBuilder(parent_resolver=_Parents((parent,))).build(evidence)

    assert len(result.chunks) == 1
    assert result.chunks[0].text == parent.text
    assert result.chunks[0].chunk_id == "child-best"
    assert result.chunks[0].parent_chunk_id == parent.parent_chunk_id
    assert result.available_evidence_count == 1


def test_context_fails_closed_when_parent_is_missing() -> None:
    evidence = RankingEvidence(
        chunk_id="child-missing",
        parent_chunk_id="parent-missing",
        source_id="source-shared",
        display_title="Policy",
        document_version=1,
        locator=ChunkLocator(pages=(1,)),
        text="child evidence",
        revision_id="revision-parent-context",
        final_rank=1,
    )

    with pytest.raises(ContextSelectionError, match="parent_chunk_missing"):
        ContextBuilder(parent_resolver=_Parents(())).build((evidence,))
