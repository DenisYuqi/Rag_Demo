"""Build truthful comparison-candidate evidence from persisted normal evaluation runs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import cast

from rag_mvp.domain.evaluation import (
    GateResult,
    MetricObservation,
    MetricObservationStatus,
    ModelAttempt,
    ModelAttemptStatus,
    ProviderAttemptEvidence,
    TokenUsage,
    UnavailableValue,
)
from rag_mvp.domain.qa import StreamEventKind
from rag_mvp.domain.retrieval import CacheOutcome, CachePolicy, RetrievalMode
from rag_mvp.evaluation.comparison import (
    ComparisonCandidateEvidence,
    ComparisonCandidateReference,
    ComparisonDomainError,
    ComparisonIdentityProjection,
    ComparisonLogicalAttempt,
    ComparisonLogicalAttemptStatus,
    ComparisonProviderAttempt,
    RerankerCaseEvidence,
    build_comparison_candidate_evidence,
    build_comparison_selection_eligibility_gate,
    project_evaluation_identity,
)
from rag_mvp.evaluation.comparison_schedule import build_comparison_schedule
from rag_mvp.evaluation.dataset import ChallengeTag, EvaluationCaseV2, EvaluationDataset
from rag_mvp.evaluation.experiment import ExperimentPlan, FixedIdentity
from rag_mvp.evaluation.grounding_metrics import (
    TEXT_SUPPORT_MATCHER_VERSION,
    TEXT_SUPPORT_NORMALIZATION_VERSION,
    MetricName,
)
from rag_mvp.evaluation.quality_gate import AdvancedMetricName, AdvancedQualityGate
from rag_mvp.evaluation.report_builder import case_ids_content_hash
from rag_mvp.evaluation.report_v2 import CategoryResultV2
from rag_mvp.evaluation.runner import (
    EvaluationCaseInput,
    EvaluationRunPlan,
    PersistedCaseResult,
)
from rag_mvp.evaluation.scoring_v2 import AdvancedEvaluationScorecard, score_evaluation_v2
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, Redactor


class ComparisonEvidenceBuildError(RuntimeError):
    """Stable fail-closed native candidate-evidence construction error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class PersistedProviderLedgerSummary:
    """Exact actual provider ledger totals, retaining known partial cost."""

    provider_call_count: int
    known_partial_cost: Decimal
    total_cost: Decimal | None
    currency: str
    cost_complete: bool
    cost_unknown_reasons: tuple[str, ...]


def summarize_persisted_provider_attempts(
    experiment_plan: ExperimentPlan,
    evaluation_plan: EvaluationRunPlan,
    attempts: Sequence[ModelAttempt],
) -> PersistedProviderLedgerSummary:
    """Price every persisted run attempt without treating a reservation as usage."""

    values = tuple(attempts)
    attempt_ids = tuple(item.attempt_id for item in values)
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ComparisonEvidenceBuildError("comparison_provider_attempt_duplicate")
    if any(item.run_id != evaluation_plan.run_id for item in values):
        raise ComparisonEvidenceBuildError("comparison_provider_run_mismatch")
    priced = tuple(
        _price_provider_attempt(
            experiment_plan,
            evaluation_plan,
            attempt,
            logical_attempt_id=f"partial-ledger-{ordinal + 1}",
        )
        for ordinal, attempt in enumerate(values)
    )
    known_partial = sum(
        (item.known_partial_cost for item in priced),
        start=Decimal(0),
    )
    complete = all(item.complete and item.total_cost is not None for item in priced)
    unknown_reasons = tuple(sorted({reason for item in priced for reason in item.unknown_reasons}))
    return PersistedProviderLedgerSummary(
        provider_call_count=len(priced),
        known_partial_cost=known_partial,
        total_cost=(
            sum(
                (cast(Decimal, item.total_cost) for item in priced),
                start=Decimal(0),
            )
            if complete
            else None
        ),
        currency=experiment_plan.pricing.currency,
        cost_complete=complete,
        cost_unknown_reasons=unknown_reasons,
    )


def build_persisted_candidate_evidence(
    *,
    comparison_id: str,
    experiment_plan: ExperimentPlan,
    reference: ComparisonCandidateReference,
    dataset: EvaluationDataset,
    evaluation_plan: EvaluationRunPlan,
    results: Sequence[PersistedCaseResult],
    provider_attempts_by_request: Mapping[str, Sequence[ModelAttempt]],
    identity_projection: ComparisonIdentityProjection | None = None,
    redactor: Redactor = DEFAULT_REDACTOR,
) -> ComparisonCandidateEvidence:
    """Recompute quality, logical attempts, tokens, and cost from immutable evidence."""

    if (
        evaluation_plan.run_id != reference.evaluation_run_id
        or evaluation_plan.identity.configuration_id != reference.configuration_id
    ):
        raise ComparisonEvidenceBuildError("comparison_candidate_run_identity_mismatch")
    validate_candidate_plan_binding(
        experiment_plan,
        reference,
        dataset,
        evaluation_plan,
    )
    selected_case_ids = _selected_dataset_case_ids(dataset, evaluation_plan)
    ordered_results = _ordered_results(evaluation_plan, results)
    scorecards = _score_repetitions(
        dataset,
        evaluation_plan,
        ordered_results,
        selected_case_ids,
        redactor,
    )
    quality_metrics, quality_gate, category_results = _aggregate_quality(scorecards)
    logical_attempts: list[ComparisonLogicalAttempt] = []
    provider_attempts: list[ComparisonProviderAttempt] = []
    reranker_evidence: list[RerankerCaseEvidence] = []
    tags_by_case = {
        case.case_id: case.challenge_tags
        for case in dataset.cases
        if isinstance(case, EvaluationCaseV2)
    }
    consumed_requests: set[str] = set()
    for order_index, (case_input, result) in enumerate(
        zip(evaluation_plan.cases, ordered_results, strict=True)
    ):
        logical_id = f"logical-{order_index + 1}"
        attempts: tuple[ModelAttempt, ...] = ()
        if result.execution is not None:
            request_id = result.execution.request_id
            if request_id in consumed_requests:
                raise ComparisonEvidenceBuildError("comparison_provider_request_duplicate")
            if request_id not in provider_attempts_by_request:
                raise ComparisonEvidenceBuildError("comparison_provider_request_missing")
            consumed_requests.add(request_id)
            attempts = tuple(provider_attempts_by_request[request_id])
            if any(
                item.request_id != request_id or item.run_id != evaluation_plan.run_id
                for item in attempts
            ):
                raise ComparisonEvidenceBuildError("comparison_provider_request_mismatch")
        priced = tuple(
            _priced_provider_attempt(
                experiment_plan,
                evaluation_plan,
                attempt,
                logical_attempt_id=logical_id,
            )
            for attempt in attempts
        )
        cache_outcome = _cache_outcome(result)
        _validate_required_provider_roles(case_input, result, priced, cache_outcome)
        logical_attempts.append(
            _logical_attempt(
                result,
                case_id=case_input.source_case_id or case_input.case_id,
                repeat_index=case_input.repeat_index,
                order_index=order_index,
                logical_attempt_id=logical_id,
                provider_attempts=priced,
                pricing_currency=experiment_plan.pricing.currency,
                cache_outcome=cache_outcome,
                cache_policy=evaluation_plan.identity.cache_policy,
            )
        )
        provider_attempts.extend(priced)
        reranker = _reranker_case_evidence(
            reference.variant_id,
            case_input.source_case_id or case_input.case_id,
            result,
            priced,
            logical_attempt_id=logical_id,
            rerank_sensitive=(
                ChallengeTag.RERANK_SENSITIVE
                in tags_by_case.get(case_input.source_case_id or case_input.case_id, ())
            ),
        )
        if reranker is not None:
            reranker_evidence.append(reranker)
    unexpected_requests = set(provider_attempts_by_request) - consumed_requests
    if unexpected_requests:
        raise ComparisonEvidenceBuildError("comparison_provider_request_unbound")
    projection = _candidate_identity_projection(
        experiment_plan,
        reference,
        dataset,
        evaluation_plan,
    )
    if identity_projection is not None and identity_projection != projection:
        raise ComparisonEvidenceBuildError("comparison_candidate_identity_projection_mismatch")
    try:
        selection_gate = build_comparison_selection_eligibility_gate(
            tuple(logical_attempts),
            tuple(provider_attempts),
            expected_logical_attempt_count=len(evaluation_plan.cases),
        )
        return build_comparison_candidate_evidence(
            comparison_id=comparison_id,
            plan=experiment_plan,
            reference=reference,
            identity_projection=projection,
            expected_case_ids=selected_case_ids,
            logical_attempts=logical_attempts,
            provider_attempts=provider_attempts,
            quality_metrics=quality_metrics,
            gates=(quality_gate, selection_gate),
            category_results=category_results,
            reranker_evidence=reranker_evidence,
        )
    except (ComparisonDomainError, TypeError, ValueError) as error:
        code = getattr(error, "code", "comparison_candidate_evidence_invalid")
        raise ComparisonEvidenceBuildError(str(code)) from None


def _ordered_results(
    plan: EvaluationRunPlan,
    results: Sequence[PersistedCaseResult],
) -> tuple[PersistedCaseResult, ...]:
    by_case = {item.case_id: item for item in results}
    expected = tuple(item.case_id for item in plan.cases)
    if (
        len(by_case) != len(results)
        or set(by_case) != set(expected)
        or any(item.run_id != plan.run_id for item in results)
    ):
        raise ComparisonEvidenceBuildError("comparison_candidate_case_set_mismatch")
    ordered = tuple(by_case[case_id] for case_id in expected)
    if any(
        result.execution is not None
        and (
            result.execution.case_id != case_input.case_id
            or result.execution.cache_policy is not plan.identity.cache_policy
        )
        for case_input, result in zip(plan.cases, ordered, strict=True)
    ):
        raise ComparisonEvidenceBuildError("comparison_candidate_execution_identity_mismatch")
    return ordered


def validate_candidate_plan_binding(
    experiment_plan: ExperimentPlan,
    reference: ComparisonCandidateReference,
    dataset: EvaluationDataset,
    evaluation_plan: EvaluationRunPlan,
) -> None:
    """Cross-bind one normal run plan to its immutable comparison declaration."""

    fixed = experiment_plan.fixed_identities
    corpus = dataset.corpus.manifest
    identity = evaluation_plan.identity
    variant = next(
        (item for item in experiment_plan.variants if item.variant_id == reference.variant_id),
        None,
    )
    if (
        evaluation_plan.run_id != reference.evaluation_run_id
        or evaluation_plan.identity.configuration_id != reference.configuration_id
        or variant is None
        or variant.axis_value != reference.axis_value
        or variant.configuration_id != reference.configuration_id
    ):
        raise ComparisonEvidenceBuildError("comparison_candidate_variant_mismatch")
    if (
        (fixed.dataset_id, fixed.dataset_version, fixed.dataset_hash)
        != (dataset.manifest.dataset_id, dataset.manifest.version, dataset.manifest.content_hash)
        or (fixed.corpus_id, fixed.corpus_version, fixed.corpus_hash)
        != (corpus.snapshot_id, corpus.version, corpus.content_hash)
        or (
            identity.dataset_id,
            identity.dataset_version,
            identity.dataset_hash,
            identity.corpus_version,
            identity.corpus_hash,
        )
        != (
            fixed.dataset_id,
            fixed.dataset_version,
            fixed.dataset_hash,
            fixed.corpus_version,
            fixed.corpus_hash,
        )
    ):
        raise ComparisonEvidenceBuildError("comparison_candidate_dataset_identity_mismatch")
    repeats = experiment_plan.repeat_order_policy.repeats_per_case
    if len(evaluation_plan.cases) != fixed.case_count * repeats:
        raise ComparisonEvidenceBuildError("comparison_candidate_case_set_mismatch")
    selected_case_order = _selected_dataset_case_ids(dataset, evaluation_plan)
    if len(selected_case_order) != fixed.case_count:
        raise ComparisonEvidenceBuildError("comparison_candidate_case_set_mismatch")
    source_case_ids = tuple(item.source_case_id or item.case_id for item in evaluation_plan.cases)
    expected_case_ids = set(selected_case_order)
    if fixed.case_set_hash != case_ids_content_hash(selected_case_order):
        raise ComparisonEvidenceBuildError("comparison_candidate_case_set_hash_mismatch")
    repetitions: dict[int, list[str]] = defaultdict(list)
    schedule = build_comparison_schedule(
        experiment_plan,
        dataset,
        selected_case_ids=selected_case_order,
    )
    execution_ids = {
        (item.repetition, item.dataset_case_id): item.execution_case_id
        for item in schedule.steps
        if item.variant_id == reference.variant_id
    }
    dataset_cases = {item.case_id: item for item in dataset.cases}
    observed_schedule: list[tuple[int, str, str]] = []
    for item, source_case_id in zip(evaluation_plan.cases, source_case_ids, strict=True):
        if source_case_id in repetitions[item.repeat_index]:
            raise ComparisonEvidenceBuildError("comparison_candidate_case_set_mismatch")
        source = dataset_cases.get(source_case_id)
        if (
            source is None
            or execution_ids.get((item.repeat_index, source_case_id)) != item.case_id
            or item.question != source.question
            or item.language != source.language.value
            or tuple((turn.role.value, turn.content) for turn in item.history)
            != tuple((turn.role, turn.content) for turn in source.history)
        ):
            raise ComparisonEvidenceBuildError("comparison_candidate_case_binding_mismatch")
        repetitions[item.repeat_index].append(source_case_id)
        observed_schedule.append((item.repeat_index, source_case_id, item.case_id))
    expected_schedule = tuple(
        (item.repetition, item.dataset_case_id, item.execution_case_id)
        for item in schedule.steps
        if item.variant_id == reference.variant_id
    )
    if tuple(observed_schedule) != expected_schedule:
        raise ComparisonEvidenceBuildError("comparison_candidate_schedule_order_mismatch")
    expected_repetitions = set(range(repeats))
    if set(repetitions) != expected_repetitions or any(
        set(values) != expected_case_ids for values in repetitions.values()
    ):
        raise ComparisonEvidenceBuildError("comparison_candidate_case_set_mismatch")
    if (
        identity.pricing_version != experiment_plan.pricing.pricing_version
        or identity.cache_policy is not experiment_plan.cache_policy
    ):
        raise ComparisonEvidenceBuildError("comparison_candidate_policy_identity_mismatch")
    projection = _candidate_identity_projection(
        experiment_plan,
        reference,
        dataset,
        evaluation_plan,
    )
    actual = projection.identity_map()
    expected_controlled = {
        item.name: item.value for item in experiment_plan.fixed_identities.controlled
    }
    if any(actual.get(name) != value for name, value in expected_controlled.items()):
        raise ComparisonEvidenceBuildError("comparison_candidate_controlled_identity_mismatch")
    if actual.get(experiment_plan.axis.identity_name) != variant.axis_value:
        raise ComparisonEvidenceBuildError("comparison_candidate_axis_identity_mismatch")
    if experiment_plan.axis.value == "retrieval-strategy" and any(
        item.retrieval_mode.value != variant.axis_value for item in evaluation_plan.cases
    ):
        raise ComparisonEvidenceBuildError("comparison_candidate_retrieval_mode_mismatch")
    projected_retrieval_mode = actual.get("retrieval.mode")
    if experiment_plan.axis.value != "retrieval-strategy" and any(
        item.retrieval_mode.value != projected_retrieval_mode for item in evaluation_plan.cases
    ):
        raise ComparisonEvidenceBuildError("comparison_candidate_retrieval_mode_mismatch")


def _candidate_identity_projection(
    experiment_plan: ExperimentPlan,
    reference: ComparisonCandidateReference,
    dataset: EvaluationDataset,
    evaluation_plan: EvaluationRunPlan,
) -> ComparisonIdentityProjection:
    variant = next(
        item for item in experiment_plan.variants if item.variant_id == reference.variant_id
    )
    cache_behavior = variant.axis_value if experiment_plan.axis.value == "cache-behavior" else None
    projected = project_evaluation_identity(
        reference.variant_id,
        evaluation_plan.identity,
        corpus_id=dataset.corpus.manifest.snapshot_id,
        case_set_hash=experiment_plan.fixed_identities.case_set_hash,
        cache_behavior=cache_behavior,
    )
    upstream = tuple(
        item
        for item in experiment_plan.fixed_identities.controlled
        if item.name.startswith("upstream.")
    )
    existing = {item.name for item in projected.identities}
    if any(item.name in existing for item in upstream):
        raise ComparisonEvidenceBuildError("comparison_upstream_identity_collision")
    if not upstream:
        return projected
    return ComparisonIdentityProjection.model_validate(
        projected.model_copy(
            update={
                "identities": (
                    *projected.identities,
                    *(FixedIdentity(name=item.name, value=item.value) for item in upstream),
                )
            }
        )
    )


def _selected_dataset_case_ids(
    dataset: EvaluationDataset,
    evaluation_plan: EvaluationRunPlan,
) -> tuple[str, ...]:
    full_case_ids = tuple(item.case_id for item in dataset.cases)
    observed = {item.source_case_id or item.case_id for item in evaluation_plan.cases}
    selected = tuple(case_id for case_id in full_case_ids if case_id in observed)
    if not selected or not observed.issubset(set(full_case_ids)) or len(selected) != len(observed):
        raise ComparisonEvidenceBuildError("comparison_candidate_case_set_mismatch")
    return selected


def _score_repetitions(
    dataset: EvaluationDataset,
    plan: EvaluationRunPlan,
    results: tuple[PersistedCaseResult, ...],
    selected_case_ids: tuple[str, ...],
    redactor: Redactor,
) -> tuple[AdvancedEvaluationScorecard, ...]:
    selected_set = set(selected_case_ids)
    selected_cases = tuple(item for item in dataset.cases if item.case_id in selected_set)
    if tuple(item.case_id for item in selected_cases) != selected_case_ids:
        raise ComparisonEvidenceBuildError("comparison_candidate_case_set_mismatch")
    scoring_dataset = dataset.model_copy(update={"cases": selected_cases})
    grouped: dict[int, list[PersistedCaseResult]] = defaultdict(list)
    for case_input, result in zip(plan.cases, results, strict=True):
        source_case_id = case_input.source_case_id or case_input.case_id
        execution = result.execution
        grouped[case_input.repeat_index].append(
            result.model_copy(
                update={
                    "case_id": source_case_id,
                    "execution": (
                        None
                        if execution is None
                        else execution.model_copy(update={"case_id": source_case_id})
                    ),
                }
            )
        )
    expected_repeats = set(range(max(grouped, default=-1) + 1))
    if set(grouped) != expected_repeats:
        raise ComparisonEvidenceBuildError("comparison_repeat_set_mismatch")
    try:
        scorecards = tuple(
            score_evaluation_v2(
                scoring_dataset,
                tuple(grouped[index]),
                redactor=redactor,
            )
            for index in sorted(grouped)
        )
    except (TypeError, ValueError) as error:
        code = getattr(error, "code", "comparison_advanced_quality_unavailable")
        raise ComparisonEvidenceBuildError(str(code)) from None
    for scorecard in scorecards:
        _validate_scorecard_versions(plan, scorecard)
    return scorecards


def _validate_scorecard_versions(
    plan: EvaluationRunPlan,
    scorecard: AdvancedEvaluationScorecard,
) -> None:
    aggregate_versions = {
        item.metric.value: item.scorer_version for item in scorecard.legacy.aggregates
    }
    expected = {
        MetricName.FAITHFULNESS.value: aggregate_versions.get(MetricName.FAITHFULNESS.value),
        MetricName.CONTEXT_PRECISION.value: aggregate_versions.get(
            MetricName.CONTEXT_PRECISION.value
        ),
        MetricName.ANSWER_COMPLETENESS.value: aggregate_versions.get(
            MetricName.ANSWER_COMPLETENESS.value
        ),
        MetricName.STYLE_CONSISTENCY.value: aggregate_versions.get(
            MetricName.STYLE_CONSISTENCY.value
        ),
        MetricName.REFUSAL_APPROPRIATENESS.value: aggregate_versions.get(
            MetricName.REFUSAL_APPROPRIATENESS.value
        ),
        "answer-compliance": scorecard.compliance.scorer_version,
        "scoring-pipeline": scorecard.legacy.scoring_version,
        "quality-gate": scorecard.legacy.quality_gate.version,
        "advanced-quality-gate": scorecard.gate.profile_version,
        "evaluation-backend": "legacy-v1",
        "faithfulness-text-matcher": TEXT_SUPPORT_MATCHER_VERSION,
        "faithfulness-text-normalization": TEXT_SUPPORT_NORMALIZATION_VERSION,
    }
    if (
        any(value is None for value in expected.values())
        or plan.identity.scorer_versions != expected
    ):
        raise ComparisonEvidenceBuildError("comparison_scorer_version_mismatch")
    observation_versions = {item.metric_id: item.scorer_version for item in scorecard.observations}
    expected_observation_versions = {
        AdvancedMetricName.FAITHFULNESS.value: expected[MetricName.FAITHFULNESS.value],
        AdvancedMetricName.CONTEXT_PRECISION.value: expected[MetricName.CONTEXT_PRECISION.value],
        AdvancedMetricName.ANSWER_COMPLIANCE.value: expected["answer-compliance"],
        AdvancedMetricName.STYLE.value: expected[MetricName.STYLE_CONSISTENCY.value],
        AdvancedMetricName.REFUSAL_APPROPRIATENESS.value: expected[
            MetricName.REFUSAL_APPROPRIATENESS.value
        ],
    }
    if observation_versions != expected_observation_versions:
        raise ComparisonEvidenceBuildError("comparison_scorer_version_mismatch")


def _aggregate_quality(
    scorecards: tuple[AdvancedEvaluationScorecard, ...],
) -> tuple[tuple[MetricObservation, ...], GateResult, tuple[CategoryResultV2, ...]]:
    if not scorecards:
        raise ComparisonEvidenceBuildError("comparison_advanced_quality_unavailable")
    raw = tuple(
        _aggregate_observation(
            tuple(scorecard.observations[index] for scorecard in scorecards),
            repeat_prefix="quality",
        )
        for index, _ in enumerate(scorecards[0].observations)
    )
    complete = all(not scorecard.failed_case_ids for scorecard in scorecards)
    gate = AdvancedQualityGate().evaluate(
        {item.metric_id: item for item in raw},
        case_executions_complete=complete,
    )
    category_ids = tuple(item.category_id for item in scorecards[0].categories)
    if any(
        tuple(item.category_id for item in value.categories) != category_ids for value in scorecards
    ):
        raise ComparisonEvidenceBuildError("comparison_category_set_mismatch")
    categories = tuple(
        CategoryResultV2(
            category_id=category_id,
            case_count=sum(len(scorecard.categories[index].case_ids) for scorecard in scorecards),
            observations=tuple(
                _aggregate_observation(
                    tuple(
                        scorecard.categories[index].observations[metric_index]
                        for scorecard in scorecards
                    ),
                    repeat_prefix=f"category-{category_id}",
                )
                for metric_index, _ in enumerate(scorecards[0].categories[index].observations)
            ),
        )
        for index, category_id in enumerate(category_ids)
    )
    return gate.observations, gate, categories


def _aggregate_observation(
    observations: tuple[MetricObservation, ...],
    *,
    repeat_prefix: str,
) -> MetricObservation:
    first = observations[0]
    if any(
        item.metric_id != first.metric_id
        or item.unit != first.unit
        or item.scorer_version != first.scorer_version
        for item in observations
    ):
        raise ComparisonEvidenceBuildError("comparison_metric_identity_mismatch")
    references = tuple(
        f"{repeat_prefix}-repeat-{repeat_index + 1}-{reference}"
        for repeat_index, item in enumerate(observations)
        for reference in item.evidence_references
    )
    if len(references) != len(set(references)):
        raise ComparisonEvidenceBuildError("comparison_metric_reference_duplicate")
    if any(
        not item.eligible
        or not isinstance(item.value, float)
        or not isinstance(item.numerator, float)
        or not isinstance(item.denominator, int)
        or item.denominator <= 0
        for item in observations
    ):
        unavailable = UnavailableValue(reason="repeat-evidence-unavailable")
        return MetricObservation(
            metric_id=first.metric_id,
            unit=first.unit,
            value=unavailable,
            numerator=unavailable,
            denominator=unavailable,
            eligible=False,
            scorer_version=first.scorer_version,
            status=MetricObservationStatus.UNAVAILABLE,
            evidence_references=references,
        )
    numerator = sum(cast(float, item.numerator) for item in observations)
    denominator = sum(cast(int, item.denominator) for item in observations)
    return MetricObservation(
        metric_id=first.metric_id,
        unit=first.unit,
        value=numerator / denominator,
        numerator=numerator,
        denominator=denominator,
        eligible=True,
        scorer_version=first.scorer_version,
        status=MetricObservationStatus.OBSERVED,
        evidence_references=references,
    )


def _priced_provider_attempt(
    plan: ExperimentPlan,
    evaluation_plan: EvaluationRunPlan,
    attempt: ModelAttempt,
    *,
    logical_attempt_id: str,
) -> ComparisonProviderAttempt:
    return _price_provider_attempt(
        plan,
        evaluation_plan,
        attempt,
        logical_attempt_id=logical_attempt_id,
    )


def _price_provider_attempt(
    plan: ExperimentPlan,
    evaluation_plan: EvaluationRunPlan,
    attempt: ModelAttempt,
    *,
    logical_attempt_id: str,
) -> ComparisonProviderAttempt:
    expected_provider = evaluation_plan.identity.provider_identities.get(attempt.role.value)
    expected_model = evaluation_plan.identity.model_identities.get(attempt.role.value)
    if attempt.provider != expected_provider or attempt.model != expected_model:
        raise ComparisonEvidenceBuildError("comparison_provider_variant_identity_mismatch")
    rate = next(
        (
            item
            for item in plan.pricing.rate_card
            if item.role.value == attempt.role.value
            and item.provider == attempt.provider
            and item.model == attempt.model
        ),
        None,
    )
    if rate is None:
        raise ComparisonEvidenceBuildError("comparison_provider_pricing_missing")
    evidence = ProviderAttemptEvidence(
        operation_id=attempt.operation_id,
        attempt_number=attempt.attempt_number,
        role=attempt.role,
        provider=attempt.provider,
        model=attempt.model,
        status=attempt.status,
        fallback=attempt.fallback,
        latency_ms=attempt.latency_ms,
        safe_error_category=attempt.safe_error_category,
        usage=TokenUsage(
            input_tokens=attempt.usage.input_tokens,
            output_tokens=attempt.usage.output_tokens,
            total_tokens_reported=attempt.usage.total_tokens_reported,
        ),
    )
    try:
        priced = ComparisonProviderAttempt.create(
            attempt_reference=attempt.attempt_id,
            logical_attempt_id=logical_attempt_id,
            evaluation_run_id=evaluation_plan.run_id,
            evidence=evidence,
            latency_ms=attempt.latency_ms,
            pricing_version=plan.pricing.pricing_version,
            pricing_hash=plan.pricing.pricing_hash,
            currency=plan.pricing.currency,
            input_per_million=rate.input_per_million,
            output_per_million=rate.output_per_million,
            pricing_source_reference=rate.source_reference,
        )
    except (TypeError, ValueError):
        raise ComparisonEvidenceBuildError("comparison_provider_evidence_invalid") from None
    return priced


def _logical_attempt(
    result: PersistedCaseResult,
    *,
    case_id: str,
    repeat_index: int,
    order_index: int,
    logical_attempt_id: str,
    provider_attempts: tuple[ComparisonProviderAttempt, ...],
    pricing_currency: str,
    cache_outcome: CacheOutcome,
    cache_policy: CachePolicy,
) -> ComparisonLogicalAttempt:
    if result.logical_latency_ms is None:
        raise ComparisonEvidenceBuildError("comparison_logical_latency_unavailable")
    execution = result.execution
    succeeded = result.succeeded and execution is not None
    timed_out = any(
        item.evidence.status is ModelAttemptStatus.TIMED_OUT for item in provider_attempts
    )
    status = (
        ComparisonLogicalAttemptStatus.SUCCEEDED
        if succeeded
        else ComparisonLogicalAttemptStatus.TIMEOUT
        if timed_out
        else ComparisonLogicalAttemptStatus.ERROR
    )
    input_applicable = tuple(
        item for item in provider_attempts if item.input_per_million is not None
    )
    output_applicable = tuple(
        item for item in provider_attempts if item.output_per_million is not None
    )
    input_values = tuple(item.evidence.usage.input_tokens for item in input_applicable)
    output_values = tuple(item.evidence.usage.output_tokens for item in output_applicable)
    input_tokens = (
        sum(cast(int, item) for item in input_values)
        if all(item is not None for item in input_values)
        else None
    )
    output_tokens = (
        sum(cast(int, item) for item in output_values)
        if all(item is not None for item in output_values)
        else None
    )
    known_partial_cost = sum(
        (item.known_partial_cost for item in provider_attempts),
        start=Decimal(0),
    )
    cost_complete = all(item.complete for item in provider_attempts)
    estimated_cost = known_partial_cost if cost_complete else None
    cost_unknown_reasons = tuple(
        sorted({reason for item in provider_attempts for reason in item.unknown_reasons})
    )
    terminal_kind = "error"
    if (
        execution is not None
        and succeeded
        and execution.event.kind
        in {
            StreamEventKind.ANSWER,
            StreamEventKind.REFUSAL,
        }
    ):
        terminal_kind = execution.event.kind.value
    if (
        execution is not None
        and execution.event.kind is StreamEventKind.ANSWER
        and not any(
            item.evidence.role.value == "generation"
            and item.evidence.status is ModelAttemptStatus.SUCCEEDED
            for item in provider_attempts
        )
    ):
        raise ComparisonEvidenceBuildError("comparison_answer_generation_evidence_missing")
    index_revision, retrieval_digest = _retrieval_equivalence_evidence(
        result,
        cache_outcome,
    )
    return ComparisonLogicalAttempt(
        attempt_id=logical_attempt_id,
        case_id=case_id,
        repeat_index=repeat_index,
        order_index=order_index,
        status=status,
        latency_ms=result.logical_latency_ms,
        terminal_kind=terminal_kind,
        cache_policy=cache_policy,
        cache_outcome=cache_outcome,
        index_revision_id=index_revision,
        retrieved_chunk_ids=() if execution is None else execution.retrieved_chunk_ids,
        context_chunk_ids=() if execution is None else execution.context_chunk_ids,
        retrieval_evidence_digest=retrieval_digest,
        safe_error_code=None if succeeded else result.safe_error_code or "evaluation-case-failed",
        provider_attempt_references=tuple(item.attempt_reference for item in provider_attempts),
        provider_failed_attempt_count=sum(
            item.evidence.status is not ModelAttemptStatus.SUCCEEDED for item in provider_attempts
        ),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost=estimated_cost,
        known_partial_cost=known_partial_cost,
        cost_complete=cost_complete,
        cost_unknown_reasons=cost_unknown_reasons,
        currency=pricing_currency,
        degradation_codes=(
            () if execution is None else execution.event.diagnostics.degradation_reasons
        ),
        completed_at=result.completed_at,
    )


def _validate_required_provider_roles(
    case_input: EvaluationCaseInput,
    result: PersistedCaseResult,
    provider_attempts: tuple[ComparisonProviderAttempt, ...],
    cache_outcome: CacheOutcome,
) -> None:
    execution = result.execution
    if cache_outcome is CacheOutcome.HIT and any(
        item.evidence.operation_id == "qa-retrieval" for item in provider_attempts
    ):
        raise ComparisonEvidenceBuildError("comparison_cache_hit_provider_call_present")
    if (
        execution is None
        or not result.succeeded
        or execution.event.kind is not StreamEventKind.ANSWER
    ):
        return

    def successful(role: str, operation_id: str) -> bool:
        return any(
            item.evidence.role.value == role
            and item.evidence.operation_id == operation_id
            and item.evidence.status is ModelAttemptStatus.SUCCEEDED
            and item.evidence.safe_error_category is None
            for item in provider_attempts
        )

    if not successful("generation", "qa-generation"):
        raise ComparisonEvidenceBuildError("comparison_answer_generation_evidence_missing")
    cache_hit = execution.cache_policy is CachePolicy.USE and cache_outcome is CacheOutcome.HIT
    if not cache_hit and not successful("embedding", "qa-retrieval"):
        raise ComparisonEvidenceBuildError("comparison_answer_embedding_evidence_missing")
    if (
        not cache_hit
        and case_input.retrieval_mode is RetrievalMode.HYBRID_RERANK
        and not any(
            item.evidence.role.value == "reranking" and item.evidence.operation_id == "qa-retrieval"
            for item in provider_attempts
        )
    ):
        raise ComparisonEvidenceBuildError("comparison_answer_reranking_evidence_missing")


def _cache_outcome(result: PersistedCaseResult) -> CacheOutcome:
    execution = result.execution
    if execution is None:
        return CacheOutcome.ERROR
    raw = execution.event.diagnostics.cache_status.get("retrieval")
    try:
        outcome = CacheOutcome(raw) if raw is not None else None
    except ValueError:
        raise ComparisonEvidenceBuildError("comparison_cache_outcome_invalid") from None
    if execution.cache_policy is CachePolicy.BYPASS and outcome is CacheOutcome.NOT_APPLICABLE:
        outcome = CacheOutcome.BYPASS
    reason = execution.event.reason
    pre_retrieval_refusal = reason is not None and reason.value in {
        "prompt-injection",
        "safety",
        "unsafe-request",
    }
    if pre_retrieval_refusal:
        expected = (
            CacheOutcome.BYPASS
            if execution.cache_policy is CachePolicy.BYPASS
            else CacheOutcome.NOT_APPLICABLE
        )
        if outcome is not None and outcome is not expected:
            raise ComparisonEvidenceBuildError("comparison_cache_outcome_policy_mismatch")
        return expected
    if outcome is CacheOutcome.NOT_APPLICABLE and not pre_retrieval_refusal:
        outcome = None
    if outcome is None and not result.succeeded:
        return (
            CacheOutcome.BYPASS
            if execution.cache_policy is CachePolicy.BYPASS
            else CacheOutcome.ERROR
        )
    if outcome is None:
        raise ComparisonEvidenceBuildError("comparison_cache_outcome_unavailable")
    if execution.cache_policy is CachePolicy.BYPASS and outcome is not CacheOutcome.BYPASS:
        raise ComparisonEvidenceBuildError("comparison_cache_outcome_policy_mismatch")
    return outcome


def _retrieval_equivalence_evidence(
    result: PersistedCaseResult,
    cache_outcome: CacheOutcome,
) -> tuple[str | UnavailableValue, str | UnavailableValue]:
    execution = result.execution
    unavailable_revision = UnavailableValue(reason="index-revision-unavailable")
    unavailable_digest = UnavailableValue(reason="retrieval-evidence-digest-unavailable")
    if execution is None:
        return unavailable_revision, unavailable_digest
    revision = execution.event.diagnostics.metadata.get("index_revision")
    index_revision = revision if isinstance(revision, str) and revision else unavailable_revision
    digest = execution.retrieval_evidence_digest or unavailable_digest
    required = result.succeeded and (
        cache_outcome in {CacheOutcome.HIT, CacheOutcome.MISS}
        or execution.event.kind is StreamEventKind.ANSWER
    )
    if required and (
        isinstance(index_revision, UnavailableValue) or isinstance(digest, UnavailableValue)
    ):
        raise ComparisonEvidenceBuildError("comparison_retrieval_equivalence_unavailable")
    return index_revision, digest


def _reranker_case_evidence(
    variant_id: str,
    execution_case_id: str,
    result: PersistedCaseResult,
    provider_attempts: tuple[ComparisonProviderAttempt, ...],
    *,
    logical_attempt_id: str,
    rerank_sensitive: bool,
) -> RerankerCaseEvidence | None:
    execution = result.execution
    if execution is None or not (execution.pre_rerank_chunk_ids or execution.post_rerank_chunk_ids):
        return None
    reranking = tuple(item for item in provider_attempts if item.evidence.role.value == "reranking")
    return RerankerCaseEvidence(
        candidate_variant_id=variant_id,
        case_id=execution_case_id,
        logical_attempt_id=logical_attempt_id,
        rerank_sensitive=rerank_sensitive,
        pre_rerank_chunk_ids=execution.pre_rerank_chunk_ids,
        post_rerank_chunk_ids=execution.post_rerank_chunk_ids,
        # The runtime does not claim a counterfactual pre-rerank context selection.
        pre_rerank_context_chunk_ids=(),
        selected_context_chunk_ids=execution.context_chunk_ids,
        reranking_attempt_references=tuple(item.attempt_reference for item in reranking),
        successful_reranking_attempt_references=tuple(
            item.attempt_reference
            for item in reranking
            if item.evidence.status is ModelAttemptStatus.SUCCEEDED
        ),
    )


__all__ = [
    "ComparisonEvidenceBuildError",
    "PersistedProviderLedgerSummary",
    "build_persisted_candidate_evidence",
    "summarize_persisted_provider_attempts",
    "validate_candidate_plan_binding",
]
