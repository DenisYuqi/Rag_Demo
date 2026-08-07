from __future__ import annotations

import pytest

from rag_mvp.domain.evaluation import (
    EvidenceComparisonOperator,
    GateStatus,
    MetricObservation,
    MetricObservationStatus,
    UnavailableValue,
)
from rag_mvp.evaluation.grounding_metrics import MetricAggregate, MetricName
from rag_mvp.evaluation.quality_gate import (
    ADVANCED_QUALITY_GATE_VERSION,
    QUALITY_GATE_VERSION,
    AdvancedMetricName,
    AdvancedQualityGate,
    QualityGate,
    ThresholdOperator,
)


def _aggregate(metric: MetricName, value: float | None, *, eligible: int = 1) -> MetricAggregate:
    return MetricAggregate(
        metric=metric,
        scorer_version=f"{metric.value}-v1",
        score=value,
        eligible_cases=eligible,
        total_cases=max(eligible, 1),
    )


def _passing_values() -> dict[MetricName, MetricAggregate]:
    return {
        MetricName.FAITHFULNESS: _aggregate(MetricName.FAITHFULNESS, 0.8500000000000001),
        MetricName.CONTEXT_PRECISION: _aggregate(
            MetricName.CONTEXT_PRECISION,
            0.7000000000000001,
        ),
        MetricName.ANSWER_COMPLETENESS: _aggregate(
            MetricName.ANSWER_COMPLETENESS,
            0.80,
        ),
        MetricName.STYLE_CONSISTENCY: _aggregate(MetricName.STYLE_CONSISTENCY, 0.80),
        MetricName.REFUSAL_APPROPRIATENESS: _aggregate(
            MetricName.REFUSAL_APPROPRIATENESS,
            0.80,
        ),
    }


def _advanced_observation(
    metric: AdvancedMetricName,
    value: float,
    *,
    denominator: int = 100,
) -> MetricObservation:
    return MetricObservation(
        metric_id=metric.value,
        unit="ratio",
        value=value,
        numerator=value * denominator,
        denominator=denominator,
        eligible=True,
        scorer_version=f"{metric.value}-v2",
        status=MetricObservationStatus.OBSERVED,
    )


def _advanced_passing_values() -> dict[AdvancedMetricName, MetricObservation]:
    return {
        AdvancedMetricName.FAITHFULNESS: _advanced_observation(
            AdvancedMetricName.FAITHFULNESS, 0.85
        ),
        AdvancedMetricName.CONTEXT_PRECISION: _advanced_observation(
            AdvancedMetricName.CONTEXT_PRECISION, 0.70
        ),
        AdvancedMetricName.ANSWER_COMPLIANCE: _advanced_observation(
            AdvancedMetricName.ANSWER_COMPLIANCE, 0.90
        ),
        AdvancedMetricName.STYLE: _advanced_observation(AdvancedMetricName.STYLE, 0.85),
        AdvancedMetricName.REFUSAL_APPROPRIATENESS: _advanced_observation(
            AdvancedMetricName.REFUSAL_APPROPRIATENESS, 0.90
        ),
    }


def test_all_five_unrounded_thresholds_pass_independently() -> None:
    result = QualityGate().evaluate(_passing_values())

    assert result.version == QUALITY_GATE_VERSION
    assert result.valid
    assert result.passed
    assert result.failed_metrics == ()
    assert tuple(decision.operator for decision in result.decisions) == (
        ThresholdOperator.GREATER_THAN,
        ThresholdOperator.GREATER_THAN,
        ThresholdOperator.GREATER_THAN_OR_EQUAL,
        ThresholdOperator.GREATER_THAN_OR_EQUAL,
        ThresholdOperator.GREATER_THAN_OR_EQUAL,
    )


def test_incomplete_case_executions_invalidate_otherwise_passing_gate() -> None:
    result = QualityGate().evaluate(
        _passing_values(),
        case_executions_complete=False,
    )

    assert not result.case_executions_complete
    assert not result.valid
    assert not result.passed
    assert all(decision.valid and decision.passed for decision in result.decisions)


@pytest.mark.parametrize(
    "metric",
    [MetricName.FAITHFULNESS, MetricName.CONTEXT_PRECISION],
)
def test_strict_quality_boundaries_fail_at_exact_equality(metric: MetricName) -> None:
    values = _passing_values()
    boundary = 0.85 if metric is MetricName.FAITHFULNESS else 0.70
    values[metric] = _aggregate(metric, boundary)

    result = QualityGate().evaluate(values)
    decision = next(item for item in result.decisions if item.metric is metric)

    assert result.valid
    assert not result.passed
    assert not decision.passed
    assert decision.value == boundary
    assert decision.rationale == "threshold_failed"


@pytest.mark.parametrize(
    "metric",
    [
        MetricName.ANSWER_COMPLETENESS,
        MetricName.STYLE_CONSISTENCY,
        MetricName.REFUSAL_APPROPRIATENESS,
    ],
)
def test_inclusive_quality_boundaries_pass_at_exact_equality(metric: MetricName) -> None:
    values = _passing_values()
    values[metric] = _aggregate(metric, 0.80)

    result = QualityGate().evaluate(values)
    decision = next(item for item in result.decisions if item.metric is metric)

    assert result.passed
    assert decision.passed


@pytest.mark.parametrize(
    ("metric", "failing_value"),
    [
        (MetricName.FAITHFULNESS, 0.8499999999999999),
        (MetricName.CONTEXT_PRECISION, 0.6999999999999999),
        (MetricName.ANSWER_COMPLETENESS, 0.7999999999999999),
        (MetricName.STYLE_CONSISTENCY, 0.7999999999999999),
        (MetricName.REFUSAL_APPROPRIATENESS, 0.7999999999999999),
    ],
)
def test_one_failed_metric_cannot_be_compensated_by_other_perfect_scores(
    metric: MetricName,
    failing_value: float,
) -> None:
    values = {required: _aggregate(required, 1.0) for required in MetricName}
    values[metric] = _aggregate(metric, failing_value)

    result = QualityGate().evaluate(values)

    assert result.valid
    assert not result.passed
    assert result.failed_metrics == (metric,)


def test_missing_metric_makes_gate_invalid_instead_of_passing() -> None:
    values = _passing_values()
    del values[MetricName.FAITHFULNESS]

    result = QualityGate().evaluate(values)
    decision = result.decisions[0]

    assert not result.valid
    assert not result.passed
    assert not decision.valid
    assert decision.rationale == "required_metric_missing"


def test_metric_without_eligible_cases_makes_gate_invalid() -> None:
    values = _passing_values()
    values[MetricName.CONTEXT_PRECISION] = _aggregate(
        MetricName.CONTEXT_PRECISION,
        None,
        eligible=0,
    )

    result = QualityGate().evaluate(values)
    decision = result.decisions[1]

    assert not result.valid
    assert not result.passed
    assert decision.value is None
    assert decision.rationale == "required_metric_has_no_eligible_cases"


def test_advanced_v2_profile_uses_five_inclusive_unrounded_thresholds() -> None:
    result = AdvancedQualityGate().evaluate(_advanced_passing_values())

    assert result.profile_version == ADVANCED_QUALITY_GATE_VERSION
    assert result.status is GateStatus.PASSED
    assert result.valid and result.passed
    assert tuple(item.metric_id for item in result.observations) == tuple(
        metric.value for metric in AdvancedMetricName
    )
    assert all(
        item.operator is EvidenceComparisonOperator.GREATER_THAN_OR_EQUAL
        for item in result.observations
    )
    assert tuple(item.threshold for item in result.observations) == (
        0.85,
        0.70,
        0.90,
        0.85,
        0.90,
    )


@pytest.mark.parametrize("metric", list(AdvancedMetricName))
def test_advanced_v2_gate_has_no_weighted_compensation(metric: AdvancedMetricName) -> None:
    values = {required: _advanced_observation(required, 1.0) for required in AdvancedMetricName}
    threshold = {
        AdvancedMetricName.FAITHFULNESS: 0.85,
        AdvancedMetricName.CONTEXT_PRECISION: 0.70,
        AdvancedMetricName.ANSWER_COMPLIANCE: 0.90,
        AdvancedMetricName.STYLE: 0.85,
        AdvancedMetricName.REFUSAL_APPROPRIATENESS: 0.90,
    }[metric]
    values[metric] = _advanced_observation(metric, threshold - 1e-12, denominator=1)

    result = AdvancedQualityGate().evaluate(values)

    assert result.valid
    assert not result.passed
    assert result.status is GateStatus.FAILED
    assert result.failure_reasons == (f"{metric.value}-threshold-not-met",)


def test_advanced_v2_gate_requires_nonzero_complete_denominator_evidence() -> None:
    values = _advanced_passing_values()
    missing = UnavailableValue(reason="not-recorded-in-v1")
    values[AdvancedMetricName.ANSWER_COMPLIANCE] = MetricObservation(
        metric_id=AdvancedMetricName.ANSWER_COMPLIANCE.value,
        unit="ratio",
        value=0.95,
        numerator=missing,
        denominator=10,
        eligible=True,
        threshold=0.90,
        operator=EvidenceComparisonOperator.GREATER_THAN_OR_EQUAL,
        scorer_version="legacy-v1",
        status=MetricObservationStatus.UNAVAILABLE,
    )

    result = AdvancedQualityGate().evaluate(values)

    assert not result.valid
    assert not result.passed
    assert result.status is GateStatus.UNAVAILABLE
    assert result.failure_reasons == ("answer-compliance-unavailable",)
    decision = result.observations[2]
    assert decision.status is MetricObservationStatus.UNAVAILABLE
    assert isinstance(decision.denominator, UnavailableValue)


def test_advanced_v2_gate_missing_metric_and_incomplete_execution_are_explicit() -> None:
    values = _advanced_passing_values()
    del values[AdvancedMetricName.STYLE]

    result = AdvancedQualityGate().evaluate(values, case_executions_complete=False)

    assert not result.valid
    assert result.failure_reasons == (
        "case-executions-incomplete",
        "style-unavailable",
    )
