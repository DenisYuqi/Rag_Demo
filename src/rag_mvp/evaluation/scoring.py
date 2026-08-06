"""Deterministic orchestration from persisted QA outcomes to evaluation scores."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

from rag_mvp.domain.qa import StreamEventKind, ValidatedStreamEvent
from rag_mvp.evaluation.answer_metrics import (
    AnswerCompletenessScorer,
    RefusalAppropriatenessScorer,
    StyleAssessment,
    StyleConsistencyScorer,
)
from rag_mvp.evaluation.dataset import (
    Answerability,
    EvaluationCase,
    EvaluationDataset,
    EvaluationLanguage,
    StyleExpectation,
)
from rag_mvp.evaluation.grounding_metrics import (
    ContextPrecisionScorer,
    FactSupportAssessment,
    FaithfulnessScorer,
    MetricAggregate,
    MetricName,
    MetricResult,
    aggregate_metric,
)
from rag_mvp.evaluation.quality_gate import QualityGate, QualityGateResult
from rag_mvp.evaluation.runner import EvaluationCaseExecution, PersistedCaseResult
from rag_mvp.safety.redactor import RedactionError, Redactor

SCORING_PIPELINE_VERSION = "deterministic-evaluation-scoring-v1"
MAX_CONCISE_CHARACTERS = 1_200
MAX_REFUSAL_CONCISE_CHARACTERS = 240

_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")
_METRIC_ORDER = tuple(MetricName)


class EvaluationScoringError(ValueError):
    """A stable, content-free scoring orchestration error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class EvaluationScorecard:
    """All per-case scores, aggregates, and the resulting quality decision."""

    run_id: str
    dataset_id: str
    dataset_version: str
    case_ids: tuple[str, ...]
    failed_case_ids: tuple[str, ...]
    per_case: tuple[MetricResult, ...]
    aggregates: tuple[MetricAggregate, ...]
    quality_gate: QualityGateResult
    scoring_version: str = SCORING_PIPELINE_VERSION

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.run_id,
                self.dataset_id,
                self.dataset_version,
                self.scoring_version,
            )
        ):
            raise ValueError("scorecard_identity_invalid")
        case_ids = tuple(self.case_ids)
        failed_case_ids = tuple(self.failed_case_ids)
        if not case_ids or len(set(case_ids)) != len(case_ids):
            raise ValueError("scorecard_case_ids_invalid")
        if len(set(failed_case_ids)) != len(failed_case_ids) or not set(failed_case_ids).issubset(
            case_ids
        ):
            raise ValueError("scorecard_failed_case_ids_invalid")

        per_case = tuple(self.per_case)
        expected_identities = tuple(
            (case_id, metric) for case_id in case_ids for metric in _METRIC_ORDER
        )
        if len(per_case) != len(expected_identities) or any(
            not isinstance(result, MetricResult) for result in per_case
        ):
            raise ValueError("scorecard_per_case_invalid")
        if tuple((result.case_id, result.metric) for result in per_case) != expected_identities:
            raise ValueError("scorecard_per_case_order_invalid")

        aggregates = tuple(self.aggregates)
        if len(aggregates) != len(_METRIC_ORDER) or any(
            not isinstance(aggregate, MetricAggregate) for aggregate in aggregates
        ):
            raise ValueError("scorecard_aggregates_invalid")
        if tuple(aggregate.metric for aggregate in aggregates) != _METRIC_ORDER:
            raise ValueError("scorecard_aggregate_order_invalid")
        if any(aggregate.total_cases != len(case_ids) for aggregate in aggregates):
            raise ValueError("scorecard_aggregate_case_count_invalid")
        if not isinstance(self.quality_gate, QualityGateResult):
            raise ValueError("scorecard_quality_gate_invalid")
        if self.quality_gate.case_executions_complete != (not failed_case_ids):
            raise ValueError("scorecard_quality_gate_execution_mismatch")

        object.__setattr__(self, "case_ids", case_ids)
        object.__setattr__(self, "failed_case_ids", failed_case_ids)
        object.__setattr__(self, "per_case", per_case)
        object.__setattr__(self, "aggregates", aggregates)

    def metrics_for_case(self, case_id: str) -> tuple[MetricResult, ...]:
        """Return the five metrics in stable gate order for one case."""

        if case_id not in self.case_ids:
            raise KeyError(case_id)
        return tuple(result for result in self.per_case if result.case_id == case_id)

    @property
    def per_case_by_id(self) -> dict[str, dict[MetricName, MetricResult]]:
        return {
            case_id: {result.metric: result for result in self.metrics_for_case(case_id)}
            for case_id in self.case_ids
        }

    @property
    def aggregates_by_metric(self) -> dict[MetricName, MetricAggregate]:
        return {aggregate.metric: aggregate for aggregate in self.aggregates}


@dataclass(frozen=True, slots=True)
class EvaluationScorer:
    """Turn trusted dataset cases and immutable runner results into a scorecard."""

    redactor: Redactor = field(default_factory=Redactor, repr=False)
    faithfulness: FaithfulnessScorer = field(default_factory=FaithfulnessScorer)
    context_precision: ContextPrecisionScorer = field(default_factory=ContextPrecisionScorer)
    completeness: AnswerCompletenessScorer = field(default_factory=AnswerCompletenessScorer)
    style: StyleConsistencyScorer = field(default_factory=StyleConsistencyScorer)
    refusal: RefusalAppropriatenessScorer = field(default_factory=RefusalAppropriatenessScorer)
    quality_gate: QualityGate = field(default_factory=QualityGate)

    @property
    def scorer_versions(self) -> Mapping[MetricName, str]:
        return MappingProxyType(
            {
                MetricName.FAITHFULNESS: self.faithfulness.version,
                MetricName.CONTEXT_PRECISION: self.context_precision.version,
                MetricName.ANSWER_COMPLETENESS: self.completeness.version,
                MetricName.STYLE_CONSISTENCY: self.style.version,
                MetricName.REFUSAL_APPROPRIATENESS: self.refusal.version,
            }
        )

    def score(
        self,
        dataset: EvaluationDataset,
        results: Sequence[PersistedCaseResult],
    ) -> EvaluationScorecard:
        if not isinstance(dataset, EvaluationDataset):
            raise EvaluationScoringError("evaluation_dataset_invalid")
        registry, run_id = _result_registry(dataset, results)
        per_case: list[MetricResult] = []
        failed_case_ids: list[str] = []

        for case in dataset.cases:
            persisted = registry[case.case_id]
            if not persisted.succeeded:
                failed_case_ids.append(case.case_id)
                per_case.extend(self._error_results(case.case_id))
                continue
            execution = persisted.execution
            if execution is None:
                raise EvaluationScoringError("successful_case_execution_missing")
            _validate_execution(case, persisted, execution)
            per_case.extend(self._score_case(case, execution))

        aggregates = tuple(
            aggregate_metric(
                tuple(result for result in per_case if result.metric is metric),
                metric=metric,
                scorer_version=self.scorer_versions[metric],
            )
            for metric in _METRIC_ORDER
        )
        quality_gate = self.quality_gate.evaluate(
            {aggregate.metric: aggregate for aggregate in aggregates},
            case_executions_complete=not failed_case_ids,
        )
        return EvaluationScorecard(
            run_id=run_id,
            dataset_id=dataset.manifest.dataset_id,
            dataset_version=dataset.manifest.version,
            case_ids=tuple(case.case_id for case in dataset.cases),
            failed_case_ids=tuple(failed_case_ids),
            per_case=tuple(per_case),
            aggregates=aggregates,
            quality_gate=quality_gate,
        )

    def _score_case(
        self,
        case: EvaluationCase,
        execution: EvaluationCaseExecution,
    ) -> tuple[MetricResult, ...]:
        event = execution.event
        outcome = _outcome(event)
        answerable = case.answerability is Answerability.ANSWERABLE
        context_ids = set(execution.context_chunk_ids)
        factual_units = _fact_support_assessments(event, context_ids)
        covered_fact_ids = _covered_fact_ids(case, event, context_ids)
        actual_reason = event.reason.value if event.reason is not None else None

        return (
            self.faithfulness.score(
                case_id=case.case_id,
                answerable=answerable,
                response_outcome=outcome,
                factual_units=factual_units,
            ),
            self.context_precision.score(
                case_id=case.case_id,
                answerable=answerable,
                retrieved_evidence_ids=execution.retrieved_chunk_ids,
                authoritative_evidence_ids=case.authoritative_evidence_ids,
            ),
            self.completeness.score(
                case_id=case.case_id,
                answerable=answerable,
                response_outcome=outcome,
                expected_fact_ids=tuple(fact.fact_id for fact in case.expected_facts),
                covered_fact_ids=covered_fact_ids,
            ),
            self.style.score(
                case_id=case.case_id,
                response_outcome=outcome,
                assessments=self._style_assessments(case, event),
            ),
            self.refusal.score(
                case_id=case.case_id,
                expected_refusal=not answerable,
                response_outcome=outcome,
                actual_reason=actual_reason,
            ),
        )

    def _style_assessments(
        self,
        case: EvaluationCase,
        event: ValidatedStreamEvent,
    ) -> tuple[StyleAssessment, ...]:
        return tuple(
            self._style_assessment(expectation, case=case, event=event)
            for expectation in case.style_expectations
        )

    def _style_assessment(
        self,
        expectation: StyleExpectation,
        *,
        case: EvaluationCase,
        event: ValidatedStreamEvent,
    ) -> StyleAssessment:
        content = (event.content or "").strip()
        if expectation is StyleExpectation.ANSWER_IN_REQUEST_LANGUAGE:
            satisfied = _matches_language(
                content,
                expected=case.language,
                declared=event.response_language,
            )
            rationale = (
                "response_language_matches_case" if satisfied else "response_language_mismatch"
            )
        elif expectation is StyleExpectation.CITATIONS_REQUIRED:
            satisfied = (
                event.kind is StreamEventKind.ANSWER
                and bool(event.claims)
                and bool(event.citations)
            )
            rationale = "citations_present" if satisfied else "citations_missing"
        elif expectation is StyleExpectation.CONCISE:
            satisfied = bool(content) and len(content) <= MAX_CONCISE_CHARACTERS
            rationale = "response_within_concise_limit" if satisfied else "response_too_long"
        elif expectation is StyleExpectation.REFUSAL_CONCISE:
            satisfied = (
                event.kind is StreamEventKind.REFUSAL
                and bool(content)
                and len(content) <= MAX_REFUSAL_CONCISE_CHARACTERS
            )
            rationale = "refusal_within_concise_limit" if satisfied else "refusal_not_concise"
        elif expectation is StyleExpectation.PII_REDACTED:
            satisfied, rationale = self._pii_redacted(event)
        else:  # pragma: no cover - exhaustive for a trusted versioned dataset enum
            raise EvaluationScoringError("style_expectation_unsupported")
        return StyleAssessment(
            expectation_id=expectation.value,
            satisfied=satisfied,
            rationale=rationale,
        )

    def _pii_redacted(self, event: ValidatedStreamEvent) -> tuple[bool, str]:
        try:
            has_sensitive_output = any(
                self.redactor.detect(value) for value in _visible_output_strings(event)
            )
        except RedactionError:
            return False, "pii_verification_unavailable"
        if has_sensitive_output:
            return False, "raw_supported_pii_detected"
        return True, "no_raw_supported_pii_detected"

    def _error_results(self, case_id: str) -> tuple[MetricResult, ...]:
        return tuple(
            MetricResult(
                case_id=case_id,
                metric=metric,
                scorer_version=self.scorer_versions[metric],
                eligible=False,
                score=None,
                numerator=None,
                denominator=None,
                rationale="case_execution_error",
            )
            for metric in _METRIC_ORDER
        )


def score_evaluation(
    dataset: EvaluationDataset,
    results: Sequence[PersistedCaseResult],
    *,
    redactor: Redactor | None = None,
) -> EvaluationScorecard:
    """Convenience entry point for deterministic scoring and quality gating."""

    scorer = EvaluationScorer(redactor=redactor) if redactor is not None else EvaluationScorer()
    return scorer.score(dataset, results)


def _result_registry(
    dataset: EvaluationDataset,
    results: object,
) -> tuple[dict[str, PersistedCaseResult], str]:
    if isinstance(results, (str, bytes, bytearray)) or not isinstance(results, Sequence):
        raise EvaluationScoringError("persisted_case_results_invalid")
    values = tuple(cast(Sequence[object], results))
    if not values or any(not isinstance(result, PersistedCaseResult) for result in values):
        raise EvaluationScoringError("persisted_case_results_invalid")
    typed = cast(tuple[PersistedCaseResult, ...], values)
    case_ids = tuple(result.case_id for result in typed)
    if len(set(case_ids)) != len(case_ids):
        raise EvaluationScoringError("duplicate_persisted_case_result")
    expected_case_ids = {case.case_id for case in dataset.cases}
    if set(case_ids) != expected_case_ids:
        raise EvaluationScoringError("persisted_case_set_mismatch")
    run_ids = {result.run_id for result in typed}
    if len(run_ids) != 1:
        raise EvaluationScoringError("persisted_run_identity_mismatch")

    registry = {result.case_id: result for result in typed}
    for case in dataset.cases:
        persisted = registry[case.case_id]
        execution = persisted.execution
        if execution is not None:
            _validate_execution(case, persisted, execution)
    return registry, next(iter(run_ids))


def _validate_execution(
    case: EvaluationCase,
    persisted: PersistedCaseResult,
    execution: EvaluationCaseExecution,
) -> None:
    event = execution.event
    if (
        execution.case_id != case.case_id
        or event.request_id != execution.request_id
        or event.session_id != execution.session_id
    ):
        raise EvaluationScoringError("case_execution_identity_mismatch")
    retrieved = execution.retrieved_chunk_ids
    context = execution.context_chunk_ids
    if len(set(retrieved)) != len(retrieved) or len(set(context)) != len(context):
        raise EvaluationScoringError("case_execution_evidence_duplicate")
    if not set(context).issubset(retrieved):
        raise EvaluationScoringError("case_context_not_in_retrieval")
    if persisted.succeeded:
        if event.kind not in {StreamEventKind.ANSWER, StreamEventKind.REFUSAL}:
            raise EvaluationScoringError("successful_case_outcome_invalid")
    elif event.kind is not StreamEventKind.ERROR:
        raise EvaluationScoringError("failed_case_outcome_invalid")


def _outcome(event: ValidatedStreamEvent) -> str:
    if event.kind is StreamEventKind.ANSWER:
        return "answer"
    if event.kind is StreamEventKind.REFUSAL:
        return "refusal"
    if event.kind is StreamEventKind.ERROR:
        return "error"
    raise EvaluationScoringError("terminal_event_kind_invalid")


def _fact_support_assessments(
    event: ValidatedStreamEvent,
    context_ids: set[str],
) -> tuple[FactSupportAssessment, ...]:
    if event.kind is not StreamEventKind.ANSWER:
        return ()
    return tuple(
        FactSupportAssessment(
            fact_id=f"claim-{ordinal:04d}",
            supported=set(claim.citation_chunk_ids).issubset(context_ids),
            rationale=(
                "claim_citations_in_runner_context"
                if set(claim.citation_chunk_ids).issubset(context_ids)
                else "claim_citation_lacks_runner_context_proof"
            ),
            evidence_chunk_ids=tuple(dict.fromkeys(claim.citation_chunk_ids)),
        )
        for ordinal, claim in enumerate(event.claims, start=1)
    )


def _covered_fact_ids(
    case: EvaluationCase,
    event: ValidatedStreamEvent,
    context_ids: set[str],
) -> tuple[str, ...]:
    if event.kind is not StreamEventKind.ANSWER:
        return ()
    proven_citation_ids = {
        citation_id
        for claim in event.claims
        for citation_id in claim.citation_chunk_ids
        if citation_id in context_ids
    }
    return tuple(
        fact.fact_id
        for fact in case.expected_facts
        if proven_citation_ids.intersection(fact.evidence_ids)
    )


def _matches_language(
    content: str,
    *,
    expected: EvaluationLanguage,
    declared: str,
) -> bool:
    if not content:
        return False
    has_han = _HAN.search(content) is not None
    has_latin = _LATIN.search(content) is not None
    if expected is EvaluationLanguage.CHINESE:
        return declared == EvaluationLanguage.CHINESE.value and has_han
    if expected is EvaluationLanguage.ENGLISH:
        return declared == EvaluationLanguage.ENGLISH.value and has_latin
    return declared in {
        EvaluationLanguage.CHINESE.value,
        EvaluationLanguage.ENGLISH.value,
        EvaluationLanguage.MIXED.value,
    } and bool(has_han and has_latin)


def _visible_output_strings(event: ValidatedStreamEvent) -> tuple[str, ...]:
    values: list[str] = []
    if event.content:
        values.append(event.content)
    values.extend(claim.text for claim in event.claims)
    for citation in event.citations:
        values.append(citation.source_title)
        values.extend(citation.locator.section_path)
    return tuple(dict.fromkeys(value for value in values if value))


__all__ = [
    "MAX_CONCISE_CHARACTERS",
    "MAX_REFUSAL_CONCISE_CHARACTERS",
    "SCORING_PIPELINE_VERSION",
    "EvaluationScorecard",
    "EvaluationScorer",
    "EvaluationScoringError",
    "score_evaluation",
]
