from __future__ import annotations

import pytest

from rag_mvp.evaluation.answer_metrics import (
    ANSWER_COMPLETENESS_SCORER_VERSION,
    REFUSAL_APPROPRIATENESS_SCORER_VERSION,
    STYLE_CONSISTENCY_SCORER_VERSION,
    AnswerCompletenessScorer,
    RefusalAppropriatenessScorer,
    StyleAssessment,
    StyleConsistencyScorer,
)
from rag_mvp.evaluation.grounding_metrics import (
    EvidenceVerdict,
    MetricInputError,
    MetricName,
    aggregate_metric,
)


def _style(expectation_id: str, satisfied: bool) -> StyleAssessment:
    return StyleAssessment(
        expectation_id=expectation_id,
        satisfied=satisfied,
        rationale="check_satisfied" if satisfied else "check_violated",
    )


def test_answer_completeness_scores_expected_fact_coverage() -> None:
    result = AnswerCompletenessScorer().score(
        case_id="case-1",
        answerable=True,
        response_outcome="answer",
        expected_fact_ids=("fact-1", "fact-2", "fact-3"),
        covered_fact_ids=("fact-1", "fact-3"),
    )

    assert result.metric is MetricName.ANSWER_COMPLETENESS
    assert result.scorer_version == ANSWER_COMPLETENESS_SCORER_VERSION
    assert result.score == 2 / 3
    assert result.rationale == "covered_expected_facts=2; expected_facts=3"
    assert tuple(item.verdict for item in result.evidence) == (
        EvidenceVerdict.COVERED,
        EvidenceVerdict.MISSING,
        EvidenceVerdict.COVERED,
    )


def test_answerable_refusal_is_complete_zero_not_an_excluded_case() -> None:
    result = AnswerCompletenessScorer().score(
        case_id="case-1",
        answerable=True,
        response_outcome="refusal",
        expected_fact_ids=("fact-1", "fact-2"),
        covered_fact_ids=(),
    )

    assert result.eligible
    assert result.score == 0
    assert all(item.verdict is EvidenceVerdict.MISSING for item in result.evidence)


def test_completeness_exact_eighty_percent_is_preserved() -> None:
    result = AnswerCompletenessScorer().score(
        case_id="case-1",
        answerable=True,
        response_outcome="answer",
        expected_fact_ids=("fact-1", "fact-2", "fact-3", "fact-4", "fact-5"),
        covered_fact_ids=("fact-1", "fact-2", "fact-3", "fact-4"),
    )

    assert result.score == 0.8


def test_completeness_has_strict_eligibility_and_expected_fact_registry() -> None:
    unanswerable = AnswerCompletenessScorer().score(
        case_id="case-1",
        answerable=False,
        response_outcome="refusal",
        expected_fact_ids=(),
        covered_fact_ids=(),
    )
    no_denominator = AnswerCompletenessScorer().score(
        case_id="case-2",
        answerable=True,
        response_outcome="answer",
        expected_fact_ids=(),
        covered_fact_ids=(),
    )

    assert not unanswerable.eligible
    assert unanswerable.rationale == "case_not_answerable"
    assert not no_denominator.eligible
    assert no_denominator.rationale == "no_expected_facts"
    with pytest.raises(MetricInputError, match="covered_fact_not_expected"):
        AnswerCompletenessScorer().score(
            case_id="case-3",
            answerable=True,
            response_outcome="answer",
            expected_fact_ids=("fact-1",),
            covered_fact_ids=("fact-invented",),
        )


def test_style_consistency_scores_only_applicable_expectations() -> None:
    result = StyleConsistencyScorer().score(
        case_id="case-1",
        response_outcome="answer",
        assessments=(
            _style("same-language", True),
            _style("concise", False),
            _style("citation-style", True),
        ),
    )

    assert result.metric is MetricName.STYLE_CONSISTENCY
    assert result.scorer_version == STYLE_CONSISTENCY_SCORER_VERSION
    assert result.score == 2 / 3
    assert result.evidence[1].rationale == "check_violated"


def test_style_exact_eighty_percent_and_missing_denominator() -> None:
    result = StyleConsistencyScorer().score(
        case_id="case-1",
        response_outcome="refusal",
        assessments=tuple(_style(f"style-{index}", index < 4) for index in range(5)),
    )
    missing = StyleConsistencyScorer().score(
        case_id="case-2",
        response_outcome="answer",
        assessments=(),
    )

    assert result.score == 0.8
    assert not missing.eligible
    assert missing.rationale == "no_applicable_style_expectations"


@pytest.mark.parametrize(
    (
        "expected_refusal",
        "outcome",
        "expected_reason",
        "actual_reason",
        "expected_score",
        "expected_rationale",
    ),
    [
        (False, "answer", None, None, 1.0, "expected_answer_emitted"),
        (
            False,
            "refusal",
            None,
            "insufficient-evidence",
            0.0,
            "answer_expected_but_refusal_emitted",
        ),
        (True, "answer", None, None, 0.0, "refusal_expected_but_answer_emitted"),
        (
            True,
            "refusal",
            "unsafe-request",
            "unsafe-request",
            1.0,
            "expected_refusal_emitted",
        ),
        (
            True,
            "refusal",
            "unsafe-request",
            "insufficient-evidence",
            0.0,
            "refusal_reason_mismatch",
        ),
    ],
)
def test_refusal_appropriateness_includes_inappropriate_refusals(
    expected_refusal: bool,
    outcome: str,
    expected_reason: str | None,
    actual_reason: str | None,
    expected_score: float,
    expected_rationale: str,
) -> None:
    result = RefusalAppropriatenessScorer().score(
        case_id="case-1",
        expected_refusal=expected_refusal,
        response_outcome=outcome,
        expected_reason=expected_reason,
        actual_reason=actual_reason,
    )

    assert result.metric is MetricName.REFUSAL_APPROPRIATENESS
    assert result.scorer_version == REFUSAL_APPROPRIATENESS_SCORER_VERSION
    assert result.eligible
    assert result.score == expected_score
    assert result.rationale == expected_rationale


def test_response_errors_are_not_silently_scored_as_appropriate() -> None:
    result = RefusalAppropriatenessScorer().score(
        case_id="case-1",
        expected_refusal=True,
        response_outcome="error",
    )

    assert not result.eligible
    assert result.score is None
    assert result.rationale == "response_error"


def test_refusal_aggregate_preserves_exact_eighty_percent_boundary() -> None:
    scorer = RefusalAppropriatenessScorer()
    results = tuple(
        scorer.score(
            case_id=f"case-{index}",
            expected_refusal=index < 4,
            response_outcome="refusal",
            actual_reason="insufficient-evidence",
        )
        for index in range(5)
    )

    aggregate = aggregate_metric(
        results,
        metric=MetricName.REFUSAL_APPROPRIATENESS,
        scorer_version=scorer.version,
    )

    assert aggregate.score == 0.8
