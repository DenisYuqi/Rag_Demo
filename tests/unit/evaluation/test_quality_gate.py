from __future__ import annotations

import pytest

from rag_mvp.evaluation.grounding_metrics import MetricAggregate, MetricName
from rag_mvp.evaluation.quality_gate import (
    QUALITY_GATE_VERSION,
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
