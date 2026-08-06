from __future__ import annotations

import math

import pytest

from rag_mvp.evaluation.grounding_metrics import (
    CONTEXT_PRECISION_SCORER_VERSION,
    FAITHFULNESS_SCORER_VERSION,
    ContextPrecisionScorer,
    EvidenceVerdict,
    FactSupportAssessment,
    FaithfulnessScorer,
    MetricInputError,
    MetricName,
    aggregate_metric,
)


def _fact(
    fact_id: str,
    supported: bool,
    evidence_ids: tuple[str, ...] = (),
) -> FactSupportAssessment:
    return FactSupportAssessment(
        fact_id=fact_id,
        supported=supported,
        rationale="support_verified" if supported else "support_not_verified",
        evidence_chunk_ids=evidence_ids,
    )


def test_faithfulness_uses_all_factual_units_and_retains_audit_evidence() -> None:
    result = FaithfulnessScorer().score(
        case_id="case-en-1",
        answerable=True,
        response_outcome="answer",
        factual_units=(
            _fact("fact-1", True, ("chunk-1",)),
            _fact("fact-2", False),
            _fact("fact-3", True, ("chunk-2", "chunk-3")),
        ),
    )

    assert result.metric is MetricName.FAITHFULNESS
    assert result.scorer_version == FAITHFULNESS_SCORER_VERSION
    assert result.eligible
    assert result.score == 2 / 3
    assert result.numerator == 2
    assert result.denominator == 3
    assert result.rationale == "supported_factual_units=2; factual_units=3"
    assert tuple(item.reference_id for item in result.evidence) == (
        "fact-1",
        "fact-2",
        "fact-3",
    )
    assert tuple(item.verdict for item in result.evidence) == (
        EvidenceVerdict.SUPPORTED,
        EvidenceVerdict.UNSUPPORTED,
        EvidenceVerdict.SUPPORTED,
    )
    assert result.evidence[2].evidence_references == ("chunk-2", "chunk-3")


@pytest.mark.parametrize(
    ("answerable", "outcome", "facts", "reason"),
    [
        (False, "answer", (_fact("fact-1", True, ("chunk-1",)),), "case_not_answerable"),
        (True, "refusal", (), "response_not_answer"),
        (True, "answer", (), "no_factual_units"),
    ],
)
def test_faithfulness_applies_strict_eligibility(
    answerable: bool,
    outcome: str,
    facts: tuple[FactSupportAssessment, ...],
    reason: str,
) -> None:
    result = FaithfulnessScorer().score(
        case_id="case-1",
        answerable=answerable,
        response_outcome=outcome,
        factual_units=facts,
    )

    assert not result.eligible
    assert result.score is None
    assert result.denominator is None
    assert result.evidence == ()
    assert result.rationale == reason


def test_faithfulness_rejects_unadjudicated_or_duplicate_facts() -> None:
    with pytest.raises(ValueError, match="supported_fact_requires_evidence"):
        _fact("fact-1", True)

    duplicate = _fact("fact-1", True, ("chunk-1",))
    with pytest.raises(MetricInputError, match="duplicate_factual_unit"):
        FaithfulnessScorer().score(
            case_id="case-1",
            answerable=True,
            response_outcome="answer",
            factual_units=(duplicate, duplicate),
        )


def test_context_precision_is_rank_aware_average_precision() -> None:
    result = ContextPrecisionScorer().score(
        case_id="case-zh-1",
        answerable=True,
        retrieved_evidence_ids=("chunk-a", "chunk-noise", "chunk-b"),
        authoritative_evidence_ids=("chunk-a", "chunk-b"),
    )

    expected_numerator = 1.0 + (2 / 3)
    assert result.metric is MetricName.CONTEXT_PRECISION
    assert result.scorer_version == CONTEXT_PRECISION_SCORER_VERSION
    assert result.eligible
    assert math.isclose(result.numerator or 0, expected_numerator)
    assert math.isclose(result.score or 0, expected_numerator / 2)
    assert result.denominator == 2
    assert tuple(item.rank for item in result.evidence) == (1, 2, 3)
    assert tuple(item.verdict for item in result.evidence) == (
        EvidenceVerdict.RELEVANT,
        EvidenceVerdict.IRRELEVANT,
        EvidenceVerdict.RELEVANT,
    )
    assert "average_precision_contribution=" in result.rationale


@pytest.mark.parametrize("retrieved", [(), ("chunk-noise",)])
def test_missing_authoritative_context_is_an_eligible_zero(
    retrieved: tuple[str, ...],
) -> None:
    result = ContextPrecisionScorer().score(
        case_id="case-1",
        answerable=True,
        retrieved_evidence_ids=retrieved,
        authoritative_evidence_ids=("chunk-authoritative",),
    )

    assert result.eligible
    assert result.score == 0
    assert result.denominator == 1
    assert result.evidence[-1].reference_id == "chunk-authoritative"
    assert result.evidence[-1].verdict is EvidenceVerdict.MISSING


def test_context_precision_requires_an_answerable_case_and_authoritative_mapping() -> None:
    non_answerable = ContextPrecisionScorer().score(
        case_id="case-refusal",
        answerable=False,
        retrieved_evidence_ids=(),
        authoritative_evidence_ids=(),
    )
    missing_mapping = ContextPrecisionScorer().score(
        case_id="case-answerable",
        answerable=True,
        retrieved_evidence_ids=("chunk-1",),
        authoritative_evidence_ids=(),
    )

    assert not non_answerable.eligible
    assert non_answerable.rationale == "case_not_answerable"
    assert not missing_mapping.eligible
    assert missing_mapping.rationale == "no_authoritative_evidence"

    with pytest.raises(ValueError, match="retrieved_evidence_ids_duplicate"):
        ContextPrecisionScorer().score(
            case_id="case-answerable",
            answerable=True,
            retrieved_evidence_ids=("chunk-1", "chunk-1"),
            authoritative_evidence_ids=("chunk-1",),
        )


def test_aggregate_uses_only_eligible_cases_without_rounding() -> None:
    scorer = FaithfulnessScorer()
    results = (
        scorer.score(
            case_id="case-1",
            answerable=True,
            response_outcome="answer",
            factual_units=(
                _fact("fact-1", True, ("chunk-1",)),
                _fact("fact-2", False),
                _fact("fact-3", False),
            ),
        ),
        scorer.score(
            case_id="case-2",
            answerable=True,
            response_outcome="answer",
            factual_units=(_fact("fact-1", True, ("chunk-1",)),),
        ),
        scorer.score(
            case_id="case-3",
            answerable=False,
            response_outcome="refusal",
            factual_units=(),
        ),
    )

    aggregate = aggregate_metric(
        results,
        metric=MetricName.FAITHFULNESS,
        scorer_version=scorer.version,
    )

    assert aggregate.eligible_cases == 2
    assert aggregate.total_cases == 3
    assert aggregate.score == ((1 / 3) + 1.0) / 2
    assert aggregate.value == aggregate.score


def test_aggregate_with_no_eligible_cases_keeps_value_unknown() -> None:
    result = FaithfulnessScorer().score(
        case_id="case-1",
        answerable=False,
        response_outcome="refusal",
        factual_units=(),
    )

    aggregate = aggregate_metric(
        (result,),
        metric=MetricName.FAITHFULNESS,
        scorer_version=FAITHFULNESS_SCORER_VERSION,
    )

    assert aggregate.eligible_cases == 0
    assert aggregate.score is None
