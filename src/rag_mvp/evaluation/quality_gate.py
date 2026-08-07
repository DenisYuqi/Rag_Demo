"""Unrounded, non-compensating quality acceptance gate."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import cast

from rag_mvp.domain.evaluation import (
    EvidenceComparisonOperator,
    GateResult,
    GateStatus,
    MetricObservation,
    MetricObservationStatus,
    UnavailableValue,
)
from rag_mvp.evaluation.grounding_metrics import MetricAggregate, MetricName

QUALITY_GATE_VERSION = "rag-quality-thresholds-v1"
ADVANCED_QUALITY_GATE_VERSION = "rag-advanced-quality-thresholds-v2"
ADVANCED_QUALITY_GATE_ID = "advanced-quality"


class QualityGateInputError(ValueError):
    """A stable error raised for malformed aggregate inputs."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ThresholdOperator(StrEnum):
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="


class AdvancedMetricName(StrEnum):
    FAITHFULNESS = "faithfulness"
    CONTEXT_PRECISION = "context-precision"
    ANSWER_COMPLIANCE = "answer-compliance"
    STYLE = "style"
    REFUSAL_APPROPRIATENESS = "refusal-appropriateness"


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

ADVANCED_QUALITY_THRESHOLDS: Mapping[AdvancedMetricName, float] = MappingProxyType(
    {
        AdvancedMetricName.FAITHFULNESS: 0.85,
        AdvancedMetricName.CONTEXT_PRECISION: 0.70,
        AdvancedMetricName.ANSWER_COMPLIANCE: 0.90,
        AdvancedMetricName.STYLE: 0.85,
        AdvancedMetricName.REFUSAL_APPROPRIATENESS: 0.90,
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
            self.case_executions_complete and all(decision.valid for decision in resolved_decisions)
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


class AdvancedQualityGate:
    """Evaluate the schema-v2 advanced profile without changing the v1 quality gate."""

    version = ADVANCED_QUALITY_GATE_VERSION
    gate_id = ADVANCED_QUALITY_GATE_ID

    def evaluate(
        self,
        observations: Mapping[AdvancedMetricName | str, MetricObservation],
        *,
        case_executions_complete: bool = True,
    ) -> GateResult:
        if type(case_executions_complete) is not bool:
            raise QualityGateInputError("quality_execution_completeness_invalid")
        values = _advanced_observations(observations)
        decisions = tuple(
            _advanced_decision(metric, ADVANCED_QUALITY_THRESHOLDS[metric], values.get(metric))
            for metric in AdvancedMetricName
        )
        complete = all(
            decision.status is not MetricObservationStatus.UNAVAILABLE for decision in decisions
        )
        valid = case_executions_complete and complete
        passed = valid and all(
            decision.status is MetricObservationStatus.PASSED for decision in decisions
        )
        failure_reasons: list[str] = []
        if not case_executions_complete:
            failure_reasons.append("case-executions-incomplete")
        for metric, decision in zip(AdvancedMetricName, decisions, strict=True):
            if decision.status is MetricObservationStatus.UNAVAILABLE:
                failure_reasons.append(f"{metric.value}-unavailable")
            elif decision.status is MetricObservationStatus.FAILED:
                failure_reasons.append(f"{metric.value}-threshold-not-met")
        return GateResult(
            gate_id=self.gate_id,
            profile_version=self.version,
            status=(
                GateStatus.UNAVAILABLE
                if not valid
                else GateStatus.PASSED
                if passed
                else GateStatus.FAILED
            ),
            valid=valid,
            passed=passed,
            case_executions_complete=case_executions_complete,
            observations=decisions,
            failure_reasons=tuple(failure_reasons),
        )


def _advanced_decision(
    metric: AdvancedMetricName,
    threshold: float,
    observation: MetricObservation | None,
) -> MetricObservation:
    missing = UnavailableValue(reason="required-evidence-missing")
    if observation is None:
        return MetricObservation(
            metric_id=metric.value,
            unit="ratio",
            value=missing,
            numerator=missing,
            denominator=missing,
            eligible=False,
            threshold=threshold,
            operator=EvidenceComparisonOperator.GREATER_THAN_OR_EQUAL,
            scorer_version=missing,
            status=MetricObservationStatus.UNAVAILABLE,
        )

    value = observation.value
    numerator = observation.numerator
    denominator = observation.denominator
    scorer_version = observation.scorer_version
    complete = (
        observation.eligible
        and observation.unit == "ratio"
        and isinstance(value, float)
        and isinstance(numerator, float)
        and isinstance(denominator, int)
        and denominator > 0
        and isinstance(scorer_version, str)
        and 0 <= value <= 1
        and 0 <= numerator <= denominator
        and math.isclose(value, numerator / denominator, rel_tol=1e-12, abs_tol=1e-12)
    )
    if not complete:
        unavailable = UnavailableValue(reason="invalid-aggregate-evidence")
        return MetricObservation(
            metric_id=metric.value,
            unit="ratio",
            value=unavailable,
            numerator=unavailable,
            denominator=unavailable,
            eligible=False,
            threshold=threshold,
            operator=EvidenceComparisonOperator.GREATER_THAN_OR_EQUAL,
            scorer_version=(scorer_version if isinstance(scorer_version, str) else unavailable),
            status=MetricObservationStatus.UNAVAILABLE,
            evidence_references=observation.evidence_references,
        )
    assert isinstance(value, float)
    assert isinstance(numerator, float)
    assert isinstance(denominator, int)
    assert isinstance(scorer_version, str)
    passed = value >= threshold
    return MetricObservation(
        metric_id=metric.value,
        unit="ratio",
        value=value,
        numerator=numerator,
        denominator=denominator,
        eligible=True,
        threshold=threshold,
        operator=EvidenceComparisonOperator.GREATER_THAN_OR_EQUAL,
        scorer_version=scorer_version,
        status=(MetricObservationStatus.PASSED if passed else MetricObservationStatus.FAILED),
        evidence_references=observation.evidence_references,
    )


def _advanced_observations(
    values: object,
) -> dict[AdvancedMetricName, MetricObservation]:
    if not isinstance(values, Mapping):
        raise QualityGateInputError("quality_observations_invalid")
    resolved: dict[AdvancedMetricName, MetricObservation] = {}
    for raw_metric, raw_observation in cast(Mapping[object, object], values).items():
        try:
            metric = (
                raw_metric
                if isinstance(raw_metric, AdvancedMetricName)
                else AdvancedMetricName(cast(str, raw_metric))
            )
        except (TypeError, ValueError):
            raise QualityGateInputError("unknown_advanced_quality_metric") from None
        if not isinstance(raw_observation, MetricObservation):
            raise QualityGateInputError("quality_observation_invalid")
        if raw_observation.metric_id != metric.value:
            raise QualityGateInputError("quality_observation_identity_mismatch")
        if metric in resolved:
            raise QualityGateInputError("duplicate_advanced_quality_metric")
        resolved[metric] = raw_observation
    return resolved


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
