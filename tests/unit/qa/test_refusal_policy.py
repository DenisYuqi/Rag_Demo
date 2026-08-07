from __future__ import annotations

import pytest

from rag_mvp.domain.ingestion import ChunkLocator
from rag_mvp.domain.qa import RefusalReason
from rag_mvp.domain.retrieval import RankingEvidence
from rag_mvp.qa.refusal import (
    EvidenceDecisionCode,
    EvidenceDecisionKind,
    FactEvidence,
    RefusalPolicy,
    RefusalPolicyError,
)


def _candidate(
    chunk_id: str,
    rank: int,
    *,
    revision_id: str = "revision-current",
) -> RankingEvidence:
    return RankingEvidence(
        chunk_id=chunk_id,
        source_id=f"source-{rank}",
        display_title=f"Policy {rank}",
        document_version=1,
        locator=ChunkLocator(pages=(rank,)),
        text=f"Evidence {rank}",
        revision_id=revision_id,
        final_rank=rank,
    )


def test_all_supported_facts_are_answerable() -> None:
    candidates = (_candidate("chunk-1", 1), _candidate("chunk-2", 2))

    decision = RefusalPolicy(minimum_support_score=0.7).decide(
        (
            FactEvidence("fact-leave", 0.9, ("chunk-1",)),
            FactEvidence("fact-carryover", 0.7, ("chunk-2",)),
        ),
        candidates=candidates,
        revision_id="revision-current",
    )

    assert decision.kind is EvidenceDecisionKind.ANSWER
    assert decision.code is EvidenceDecisionCode.ANSWERABLE
    assert decision.answer_allowed
    assert decision.reason is None
    assert decision.supported_fact_ids == ("fact-leave", "fact-carryover")
    assert decision.citation_chunk_ids == ("chunk-1", "chunk-2")


def test_mixed_support_produces_a_partial_answer_decision() -> None:
    candidates = (_candidate("chunk-1", 1), _candidate("chunk-2", 2))

    decision = RefusalPolicy(minimum_support_score=0.7).decide(
        (
            FactEvidence("fact-supported", 0.8, ("chunk-1",)),
            FactEvidence("fact-weak", 0.4, ("chunk-2",)),
            FactEvidence("fact-absent", 0),
        ),
        candidates=candidates,
        revision_id="revision-current",
    )

    assert decision.kind is EvidenceDecisionKind.PARTIAL
    assert decision.code is EvidenceDecisionCode.PARTIAL_EVIDENCE
    assert decision.answer_allowed
    assert decision.supported_fact_ids == ("fact-supported",)
    assert decision.unsupported_fact_ids == ("fact-weak", "fact-absent")
    assert decision.citation_chunk_ids == ("chunk-1",)


def test_absent_evidence_returns_stable_low_confidence_refusal() -> None:
    decision = RefusalPolicy().decide(
        (FactEvidence("fact-absent", 0),),
        candidates=(),
        revision_id="revision-current",
    )

    assert decision.kind is EvidenceDecisionKind.REFUSAL
    assert decision.code is EvidenceDecisionCode.LOW_CONFIDENCE
    assert decision.reason is RefusalReason.LOW_CONFIDENCE
    assert decision.requires_refusal
    assert decision.citation_chunk_ids == ()


def test_default_calibration_accepts_supported_partial_evidence() -> None:
    candidate = _candidate("chunk-1", 1)

    decision = RefusalPolicy().decide(
        (
            FactEvidence("fact-supported", 0.55, ("chunk-1",)),
            FactEvidence("fact-unsupported", 0),
        ),
        candidates=(candidate,),
        revision_id="revision-current",
    )

    assert decision.kind is EvidenceDecisionKind.PARTIAL
    assert decision.supported_fact_ids == ("fact-supported",)


def test_below_threshold_evidence_is_insufficient_at_a_stable_boundary() -> None:
    candidate = _candidate("chunk-1", 1)
    policy = RefusalPolicy(minimum_support_score=0.7)

    below = policy.decide(
        (FactEvidence("fact-1", 0.699, ("chunk-1",)),),
        candidates=(candidate,),
        revision_id="revision-current",
    )
    boundary = policy.decide(
        (FactEvidence("fact-1", 0.7, ("chunk-1",)),),
        candidates=(candidate,),
        revision_id="revision-current",
    )

    assert below.reason is RefusalReason.LOW_CONFIDENCE
    assert boundary.kind is EvidenceDecisionKind.ANSWER


def test_unresolved_conflict_wins_and_retains_both_evidence_sides() -> None:
    candidates = (_candidate("chunk-current", 1), _candidate("chunk-conflict", 2))

    decision = RefusalPolicy().decide(
        (
            FactEvidence(
                "fact-policy-date",
                0.95,
                ("chunk-current",),
                ("chunk-conflict",),
            ),
        ),
        candidates=candidates,
        revision_id="revision-current",
    )

    assert decision.kind is EvidenceDecisionKind.REFUSAL
    assert decision.code is EvidenceDecisionCode.CONFLICTING_EVIDENCE
    assert decision.reason is RefusalReason.CONFLICTING_EVIDENCE
    assert decision.conflicting_fact_ids == ("fact-policy-date",)
    assert decision.citation_chunk_ids == ("chunk-current", "chunk-conflict")


def test_fact_evidence_must_be_scoped_to_current_request() -> None:
    with pytest.raises(RefusalPolicyError, match="fact_evidence_not_in_request"):
        RefusalPolicy().decide(
            (FactEvidence("fact-1", 0.9, ("chunk-invented",)),),
            candidates=(_candidate("chunk-1", 1),),
            revision_id="revision-current",
        )


@pytest.mark.parametrize("threshold", [0, -0.1, 1.1, float("nan"), True])
def test_policy_rejects_invalid_calibration_threshold(threshold: object) -> None:
    with pytest.raises(ValueError, match="minimum_support_score"):
        RefusalPolicy(minimum_support_score=threshold)  # type: ignore[arg-type]
