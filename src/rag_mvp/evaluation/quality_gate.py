"""Unrounded, non-compensating quality acceptance gate."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import cast

from rag_mvp.evaluation.grounding_metrics import MetricAggregate, MetricName

QUALITY_GATE_VERSION = "rag-quality-thresholds-v1"


class QualityGateInputError(ValueError):
    """A stable error raised for malformed aggregate inputs."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ThresholdOperator(StrEnum):
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="


@dataclass(frozen=True, slots=True)
class QualityThreshold:
    value: float
    operator: ThresholdOperator

    def __post_init__(self) -> None:
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, (int, float))
            or not math.isfinite(self.value)
            or not 0 <= self.value <= 1
        ):
            raise ValueError("quality_threshold_invalid")
        object.__setattr__(self, "value", float(self.value))
        try:
            resolved_operator = ThresholdOperator(self.operator)
        except (TypeError, ValueError):
            raise ValueError("quality_operator_invalid") from None
        object.__setattr__(self, "operator", resolved_operator)


QUALITY_THRESHOLDS: Mapping[MetricName, QualityThreshold] = MappingProxyType(
    {
        MetricName.FAITHFULNESS: QualityThreshold(0.85, ThresholdOperator.GREATER_THAN),
        MetricName.CONTEXT_PRECISION: QualityThreshold(0.70, ThresholdOperator.GREATER_THAN),
        MetricName.ANSWER_COMPLETENESS: QualityThreshold(
            0.80,
            ThresholdOperator.GREATER_THAN_OR_EQUAL,
        ),
        MetricName.STYLE_CONSISTENCY: QualityThreshold(
            0.80,
            ThresholdOperator.GREATER_THAN_OR_EQUAL,
        ),
        MetricName.REFUSAL_APPROPRIATENESS: QualityThreshold(
            0.80,
            ThresholdOperator.GREATER_THAN_OR_EQUAL,
        ),
    }
)


@dataclass(frozen=True, slots=True)
class QualityGateDecision:
    metric: MetricName
    scorer_version: str | None
    value: float | None
    eligible_cases: int
    threshold: float
    operator: ThresholdOperator
    valid: bool
    passed: bool
    rationale: str

    def __post_init__(self) -> None:
        try:
            resolved_metric = MetricName(self.metric)
            resolved_operator = ThresholdOperator(self.operator)
        except (TypeError, ValueError):
            raise ValueError("quality_decision_identity_invalid") from None
        object.__setattr__(self, "metric", resolved_metric)
        object.__setattr__(self, "operator", resolved_operator)
        if type(self.eligible_cases) is not int or self.eligible_cases < 0:
            raise ValueError("quality_decision_eligibility_invalid")
        if type(self.valid) is not bool or type(self.passed) is not bool:
            raise ValueError("quality_decision_status_invalid")
        if self.passed and not self.valid:
            raise ValueError("invalid_quality_decision_cannot_pass")
        if not isinstance(self.rationale, str) or not self.rationale:
            raise ValueError("quality_decision_rationale_invalid")


@dataclass(frozen=True, slots=True)
class QualityGateResult:
    version: str
    valid: bool
    passed: bool
    decisions: tuple[QualityGateDecision, ...]
    case_executions_complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("quality_gate_version_invalid")
        if type(self.valid) is not bool or type(self.passed) is not bool:
            raise ValueError("quality_gate_status_invalid")
        if type(self.case_executions_complete) is not bool:
            raise ValueError("quality_gate_execution_completeness_invalid")
        resolved_decisions = tuple(self.decisions)
        if len(resolved_decisions) != len(QUALITY_THRESHOLDS) or any(
            not isinstance(decision, QualityGateDecision) for decision in resolved_decisions
        ):
            raise ValueError("quality_gate_decisions_invalid")
        if tuple(decision.metric for decision in resolved_decisions) != tuple(QUALITY_THRESHOLDS):
            raise ValueError("quality_gate_decision_order_invalid")
        if self.valid != (
            self.case_executions_complete
            and all(decision.valid for decision in resolved_decisions)
        ):
            raise ValueError("quality_gate_validity_mismatch")
        if self.passed != (self.valid and all(decision.passed for decision in resolved_decisions)):
            raise ValueError("quality_gate_result_mismatch")
        object.__setattr__(self, "decisions", resolved_decisions)

    @property
    def failed_metrics(self) -> tuple[MetricName, ...]:
        return tuple(decision.metric for decision in self.decisions if not decision.passed)


class QualityGate:
    """Evaluate every required metric independently using its exact raw value."""

    version = QUALITY_GATE_VERSION

    def evaluate(
        self,
        aggregates: Mapping[MetricName | str, MetricAggregate],
        *,
        case_executions_complete: bool = True,
    ) -> QualityGateResult:
        values = _aggregates(aggregates)
        decisions = tuple(
            self._decision(metric, QUALITY_THRESHOLDS[metric], values.get(metric))
            for metric in QUALITY_THRESHOLDS
        )
        valid = case_executions_complete and all(decision.valid for decision in decisions)
        passed = valid and all(decision.passed for decision in decisions)
        return QualityGateResult(
            version=self.version,
            valid=valid,
            passed=passed,
            decisions=decisions,
            case_executions_complete=case_executions_complete,
        )

    @staticmethod
    def _decision(
        metric: MetricName,
        threshold: QualityThreshold,
        aggregate: MetricAggregate | None,
    ) -> QualityGateDecision:
        if aggregate is None:
            return QualityGateDecision(
                metric=metric,
                scorer_version=None,
                value=None,
                eligible_cases=0,
                threshold=threshold.value,
                operator=threshold.operator,
                valid=False,
                passed=False,
                rationale="required_metric_missing",
            )
        if aggregate.eligible_cases == 0 or aggregate.score is None:
            return QualityGateDecision(
                metric=metric,
                scorer_version=aggregate.scorer_version,
                value=None,
                eligible_cases=aggregate.eligible_cases,
                threshold=threshold.value,
                operator=threshold.operator,
                valid=False,
                passed=False,
                rationale="required_metric_has_no_eligible_cases",
            )

        value = aggregate.score
        if threshold.operator is ThresholdOperator.GREATER_THAN:
            passed = value > threshold.value
        else:
            passed = value >= threshold.value
        return QualityGateDecision(
            metric=metric,
            scorer_version=aggregate.scorer_version,
            value=value,
            eligible_cases=aggregate.eligible_cases,
            threshold=threshold.value,
            operator=threshold.operator,
            valid=True,
            passed=passed,
            rationale="threshold_passed" if passed else "threshold_failed",
        )


def _aggregates(
    values: object,
) -> dict[MetricName, MetricAggregate]:
    if not isinstance(values, Mapping):
        raise QualityGateInputError("quality_aggregates_invalid")
    resolved: dict[MetricName, MetricAggregate] = {}
    for raw_metric, raw_aggregate in cast(Mapping[object, object], values).items():
        if isinstance(raw_metric, MetricName):
            metric = raw_metric
        elif isinstance(raw_metric, str):
            try:
                metric = MetricName(raw_metric)
            except ValueError:
                raise QualityGateInputError("unknown_quality_metric") from None
        else:
            raise QualityGateInputError("unknown_quality_metric")
        if not isinstance(raw_aggregate, MetricAggregate):
            raise QualityGateInputError("quality_aggregate_invalid")
        if raw_aggregate.metric is not metric:
            raise QualityGateInputError("quality_aggregate_identity_mismatch")
        if metric in resolved:
            raise QualityGateInputError("duplicate_quality_metric")
        resolved[metric] = raw_aggregate
    return resolved
