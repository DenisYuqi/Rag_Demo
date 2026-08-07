"""Production assembly of advanced schema-v2 quality evidence."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import cast

from rag_mvp.domain.evaluation import (
    GateResult,
    MetricObservation,
    MetricObservationStatus,
    UnavailableValue,
)
from rag_mvp.domain.qa import StreamEventKind
from rag_mvp.evaluation.answer_metrics import (
    ANSWER_COMPLIANCE_SCORER_VERSION,
    AnswerComplianceAggregate,
    AnswerComplianceResult,
    AnswerComplianceScorer,
    GuidedRefusalAppropriatenessScorer,
    aggregate_answer_compliance,
)
from rag_mvp.evaluation.compliance import assess_compliance_obligations
from rag_mvp.evaluation.dataset import (
    Answerability,
    ChallengeTag,
    EvaluationCaseV2,
    EvaluationDataset,
)
from rag_mvp.evaluation.grounding_metrics import (
    AdjudicatedFaithfulnessScorer,
    MetricName,
    MetricResult,
)
from rag_mvp.evaluation.quality_gate import (
    AdvancedMetricName,
    AdvancedQualityGate,
)
from rag_mvp.evaluation.runner import PersistedCaseResult
from rag_mvp.evaluation.scoring import (
    ADVANCED_SCORING_PIPELINE_VERSION,
    EvaluationScorecard,
    EvaluationScorer,
)
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, Redactor

_ADVANCED_TO_V1 = {
    AdvancedMetricName.FAITHFULNESS: MetricName.FAITHFULNESS,
    AdvancedMetricName.CONTEXT_PRECISION: MetricName.CONTEXT_PRECISION,
    AdvancedMetricName.STYLE: MetricName.STYLE_CONSISTENCY,
    AdvancedMetricName.REFUSAL_APPROPRIATENESS: MetricName.REFUSAL_APPROPRIATENESS,
}


class AdvancedScoringError(ValueError):
    """Stable fail-closed error for incomplete schema-v2 scoring inputs."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class AdvancedCategoryScore:
    category_id: str
    case_ids: tuple[str, ...]
    observations: tuple[MetricObservation, ...]

    def __post_init__(self) -> None:
        if not self.category_id or not self.case_ids:
            raise ValueError("advanced_category_identity_invalid")
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("advanced_category_cases_duplicate")
        if tuple(item.metric_id for item in self.observations) != tuple(
            metric.value for metric in AdvancedMetricName
        ):
            raise ValueError("advanced_category_metric_order_invalid")


@dataclass(frozen=True, slots=True)
class AdvancedEvaluationScorecard:
    run_id: str
    dataset_id: str
    dataset_version: str
    case_ids: tuple[str, ...]
    failed_case_ids: tuple[str, ...]
    legacy: EvaluationScorecard
    compliance: AnswerComplianceAggregate
    observations: tuple[MetricObservation, ...]
    gate: GateResult
    categories: tuple[AdvancedCategoryScore, ...]

    def __post_init__(self) -> None:
        if self.run_id != self.legacy.run_id:
            raise ValueError("advanced_scorecard_run_identity_mismatch")
        if self.case_ids != self.legacy.case_ids:
            raise ValueError("advanced_scorecard_case_identity_mismatch")
        if self.failed_case_ids != self.legacy.failed_case_ids:
            raise ValueError("advanced_scorecard_failure_identity_mismatch")
        metric_ids = tuple(metric.value for metric in AdvancedMetricName)
        if tuple(item.metric_id for item in self.observations) != metric_ids:
            raise ValueError("advanced_scorecard_metric_order_invalid")
        if self.gate.observations != self.observations:
            raise ValueError("advanced_scorecard_gate_evidence_mismatch")


def score_evaluation_v2(
    dataset: EvaluationDataset,
    results: tuple[PersistedCaseResult, ...],
    *,
    redactor: Redactor = DEFAULT_REDACTOR,
) -> AdvancedEvaluationScorecard:
    """Score all advanced metrics from immutable case evidence, never from defaults."""

    cases = tuple(dataset.cases)
    if not cases or any(not isinstance(case, EvaluationCaseV2) for case in cases):
        raise AdvancedScoringError("evaluation_dataset_v2_required")
    v2_cases = cast(tuple[EvaluationCaseV2, ...], cases)
    _validate_persisted_case_set(v2_cases, results)
    legacy = EvaluationScorer(
        redactor=redactor,
        faithfulness=AdjudicatedFaithfulnessScorer(),
        refusal=GuidedRefusalAppropriatenessScorer(),
        scoring_version=ADVANCED_SCORING_PIPELINE_VERSION,
    ).score(dataset, _normalized_v2_language_results(v2_cases, results))
    result_by_case = {result.case_id: result for result in results}
    compliance_results: list[AnswerComplianceResult] = []
    for case in v2_cases:
        persisted = result_by_case.get(case.case_id)
        if persisted is None:
            raise AdvancedScoringError("persisted_case_set_mismatch")
        execution = persisted.execution
        if not persisted.succeeded or execution is None:
            compliance_results.append(
                AnswerComplianceResult(
                    case_id=case.case_id,
                    scorer_version=ANSWER_COMPLIANCE_SCORER_VERSION,
                    scored=False,
                    eligible=False,
                    score=None,
                    numerator=None,
                    denominator=None,
                    rationale="case-execution-unavailable",
                )
            )
            continue
        outcome = (
            "answer"
            if execution.event.kind is StreamEventKind.ANSWER
            else "refusal"
            if execution.event.kind is StreamEventKind.REFUSAL
            else "error"
        )
        compliance_results.append(
            AnswerComplianceScorer().score(
                case_id=case.case_id,
                answerable=case.answerability is Answerability.ANSWERABLE,
                response_outcome=outcome,
                assessments=assess_compliance_obligations(
                    case,
                    execution.event,
                    redactor=redactor,
                ),
            )
        )
    compliance = aggregate_answer_compliance(tuple(compliance_results))
    raw_observations = _observations(
        legacy,
        compliance,
        selected_case_ids=legacy.case_ids,
    )
    gate = AdvancedQualityGate().evaluate(
        {item.metric_id: item for item in raw_observations},
        case_executions_complete=not legacy.failed_case_ids,
    )
    category_case_ids: dict[ChallengeTag, list[str]] = defaultdict(list)
    for case in v2_cases:
        for tag in case.challenge_tags:
            category_case_ids[tag].append(case.case_id)
    categories = tuple(
        AdvancedCategoryScore(
            category_id=tag.value,
            case_ids=tuple(category_case_ids[tag]),
            observations=_observations(
                legacy,
                compliance,
                selected_case_ids=tuple(category_case_ids[tag]),
            ),
        )
        for tag in sorted(category_case_ids, key=lambda item: item.value)
    )
    return AdvancedEvaluationScorecard(
        run_id=legacy.run_id,
        dataset_id=legacy.dataset_id,
        dataset_version=legacy.dataset_version,
        case_ids=legacy.case_ids,
        failed_case_ids=legacy.failed_case_ids,
        legacy=legacy,
        compliance=compliance,
        observations=gate.observations,
        gate=gate,
        categories=categories,
    )


def _observations(
    legacy: EvaluationScorecard,
    compliance: AnswerComplianceAggregate,
    *,
    selected_case_ids: tuple[str, ...],
) -> tuple[MetricObservation, ...]:
    selected = set(selected_case_ids)
    compliance_by_case = {result.case_id: result for result in compliance.case_results}
    observations: list[MetricObservation] = []
    for metric in AdvancedMetricName:
        if metric is AdvancedMetricName.ANSWER_COMPLIANCE:
            values = tuple(
                result
                for case_id, result in compliance_by_case.items()
                if case_id in selected and result.eligible
            )
            scores = tuple(cast(float, result.score) for result in values)
            scorer_version = compliance.scorer_version
            references = tuple(result.case_id for result in values)
        else:
            v1_metric = _ADVANCED_TO_V1[metric]
            metric_results = tuple(
                result
                for result in legacy.per_case
                if result.case_id in selected and result.metric is v1_metric and result.eligible
            )
            scores = tuple(cast(float, result.score) for result in metric_results)
            scorer_version = _metric_scorer_version(legacy, v1_metric, metric_results)
            references = tuple(result.case_id for result in metric_results)
        denominator = len(scores)
        if denominator == 0:
            unavailable = UnavailableValue(reason="no-eligible-cases")
            observations.append(
                MetricObservation(
                    metric_id=metric.value,
                    unit="ratio",
                    value=unavailable,
                    numerator=unavailable,
                    denominator=unavailable,
                    eligible=False,
                    scorer_version=scorer_version,
                    status=MetricObservationStatus.UNAVAILABLE,
                    evidence_references=references,
                )
            )
            continue
        numerator = float(sum(scores))
        observations.append(
            MetricObservation(
                metric_id=metric.value,
                unit="ratio",
                value=numerator / denominator,
                numerator=numerator,
                denominator=denominator,
                eligible=True,
                scorer_version=scorer_version,
                status=MetricObservationStatus.OBSERVED,
                evidence_references=references,
            )
        )
    return tuple(observations)


def _metric_scorer_version(
    legacy: EvaluationScorecard,
    metric: MetricName,
    results: tuple[MetricResult, ...],
) -> str:
    if results:
        versions = {result.scorer_version for result in results}
        if len(versions) != 1:
            raise AdvancedScoringError("metric_scorer_version_mismatch")
        return next(iter(versions))
    aggregate = next(item for item in legacy.aggregates if item.metric is metric)
    return aggregate.scorer_version


def _normalized_v2_language_results(
    cases: tuple[EvaluationCaseV2, ...],
    results: tuple[PersistedCaseResult, ...],
) -> tuple[PersistedCaseResult, ...]:
    """Normalize locale labels for v2 style scoring without changing persisted evidence."""

    expected_by_case = {case.case_id: case.language.value for case in cases}
    normalized: list[PersistedCaseResult] = []
    for result in results:
        execution = result.execution
        if execution is None:
            normalized.append(result)
            continue
        declared = execution.event.response_language.casefold()
        expected = expected_by_case[result.case_id]
        equivalent = (
            (expected == "zh" and (declared == "zh" or declared.startswith("zh-")))
            or (expected == "en" and (declared == "en" or declared.startswith("en-")))
            or declared == expected
        )
        if not equivalent or execution.event.response_language == expected:
            normalized.append(result)
            continue
        event = execution.event.model_copy(update={"response_language": expected})
        normalized.append(
            result.model_copy(update={"execution": execution.model_copy(update={"event": event})})
        )
    return tuple(normalized)


def _validate_persisted_case_set(
    cases: tuple[EvaluationCaseV2, ...],
    results: tuple[PersistedCaseResult, ...],
) -> None:
    expected_ids = tuple(case.case_id for case in cases)
    persisted_ids = tuple(result.case_id for result in results)
    if (
        len(expected_ids) != len(set(expected_ids))
        or len(persisted_ids) != len(set(persisted_ids))
        or len(expected_ids) != len(persisted_ids)
        or set(expected_ids) != set(persisted_ids)
    ):
        raise AdvancedScoringError("persisted_case_set_mismatch")


__all__ = [
    "AdvancedCategoryScore",
    "AdvancedEvaluationScorecard",
    "AdvancedScoringError",
    "score_evaluation_v2",
]
