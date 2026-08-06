from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rag_mvp.domain.ingestion import ChunkLocator
from rag_mvp.domain.qa import Citation
from rag_mvp.domain.retrieval import RankingEvidence
from rag_mvp.qa.citations import ParsedAnswer, StructuredAnswerParser
from rag_mvp.qa.context import ContextBuilder
from rag_mvp.qa.grounding import GroundingValidationError, GroundingValidator
from rag_mvp.qa.prompt import GENERATOR_OUTPUT_SCHEMA_VERSION


def _evidence(
    chunk_id: str,
    rank: int,
    *,
    text: str | None = None,
    revision_id: str = "revision-current",
) -> RankingEvidence:
    return RankingEvidence(
        chunk_id=chunk_id,
        source_id=f"source-{rank}",
        display_title=f"Policy {rank}",
        document_version=rank,
        locator=ChunkLocator(pages=(rank,)),
        text=text or f"Evidence {rank}",
        revision_id=revision_id,
        final_rank=rank,
    )


def _parse(
    answer: str,
    claims: list[dict[str, object]],
    evidence: tuple[RankingEvidence, ...],
    *,
    revision_id: str = "revision-current",
) -> ParsedAnswer:
    context = ContextBuilder().build(evidence)
    return StructuredAnswerParser().parse(
        json.dumps(
            {
                "schema_version": GENERATOR_OUTPUT_SCHEMA_VERSION,
                "answer": answer,
                "claims": claims,
            }
        ),
        context=context,
        expected_revision_id=revision_id,
    )


def test_valid_answer_has_complete_ordered_claim_coverage() -> None:
    candidates = (_evidence("chunk-1", 1), _evidence("chunk-2", 2))
    parsed = _parse(
        "Employees receive ten days. Carryover is limited.",
        [
            {"text": "Employees receive ten days.", "citation_chunk_ids": ["chunk-1"]},
            {"text": "Carryover is limited.", "citation_chunk_ids": ["chunk-2"]},
        ],
        candidates,
    )

    result = GroundingValidator().validate(
        parsed,
        request_id="request-1",
        revision_id="revision-current",
        candidates=candidates,
    )

    assert result.answer == parsed.answer
    assert result.claims == parsed.claims
    assert result.citations == parsed.citations
    assert result.request_id == "request-1"


@pytest.mark.parametrize(
    ("answer", "claims"),
    [
        (
            "Employees receive ten days. Remote work is always allowed.",
            [
                {
                    "text": "Employees receive ten days.",
                    "citation_chunk_ids": ["chunk-1"],
                }
            ],
        ),
        (
            "Carryover is limited. Employees receive ten days.",
            [
                {
                    "text": "Employees receive ten days.",
                    "citation_chunk_ids": ["chunk-1"],
                },
                {"text": "Carryover is limited.", "citation_chunk_ids": ["chunk-2"]},
            ],
        ),
    ],
)
def test_uncovered_or_reordered_factual_units_withhold_complete_answer(
    answer: str,
    claims: list[dict[str, object]],
) -> None:
    candidates = (_evidence("chunk-1", 1), _evidence("chunk-2", 2))
    parsed = _parse(answer, claims, candidates)

    with pytest.raises(
        GroundingValidationError,
        match="factual_unit_coverage_invalid",
    ) as captured:
        GroundingValidator().validate(
            parsed,
            request_id="request-1",
            revision_id="revision-current",
            candidates=candidates,
        )

    assert answer not in str(captured.value)


def test_citation_must_belong_to_the_current_request_candidates() -> None:
    parsed = _parse(
        "Employees receive ten days.",
        [{"text": "Employees receive ten days.", "citation_chunk_ids": ["chunk-1"]}],
        (_evidence("chunk-1", 1),),
    )
    other_request_candidates = (_evidence("chunk-2", 1),)

    with pytest.raises(GroundingValidationError, match="citation_not_in_request"):
        GroundingValidator().validate(
            parsed,
            request_id="request-2",
            revision_id="revision-current",
            candidates=other_request_candidates,
        )


def test_citation_metadata_must_match_request_candidate_exactly() -> None:
    candidate = _evidence("chunk-1", 1)
    parsed = _parse(
        "Employees receive ten days.",
        [{"text": "Employees receive ten days.", "citation_chunk_ids": ["chunk-1"]}],
        (candidate,),
    )
    tampered = replace(
        parsed,
        citations=(
            Citation(
                source_title="Fabricated title",
                document_version=1,
                chunk_id="chunk-1",
                locator=ChunkLocator(pages=(99,)),
            ),
        ),
    )

    with pytest.raises(GroundingValidationError, match="citation_metadata_mismatch"):
        GroundingValidator().validate(
            tampered,
            request_id="request-1",
            revision_id="revision-current",
            candidates=(candidate,),
        )


def test_stale_candidate_registry_is_rejected_before_release() -> None:
    stale = _evidence("chunk-1", 1, revision_id="revision-old")
    parsed = _parse(
        "Employees receive ten days.",
        [{"text": "Employees receive ten days.", "citation_chunk_ids": ["chunk-1"]}],
        (stale,),
        revision_id="revision-old",
    )

    with pytest.raises(GroundingValidationError, match="candidate_registry_invalid"):
        GroundingValidator().validate(
            parsed,
            request_id="request-1",
            revision_id="revision-current",
            candidates=(stale,),
        )


def test_runtime_validation_does_not_attempt_semantic_scoring() -> None:
    candidate = _evidence("chunk-1", 1, text="Annual leave policy evidence")
    parsed = _parse(
        "The moon is made of cheese.",
        [{"text": "The moon is made of cheese.", "citation_chunk_ids": ["chunk-1"]}],
        (candidate,),
    )

    result = GroundingValidator().validate(
        parsed,
        request_id="request-1",
        revision_id="revision-current",
        candidates=(candidate,),
    )

    assert result.answer == "The moon is made of cheese."
