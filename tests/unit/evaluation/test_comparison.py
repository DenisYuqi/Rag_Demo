from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from rag_mvp.domain import (
    ArtifactDescriptor,
    EvidenceComparisonOperator,
    GateResult,
    GateStatus,
    MetricObservation,
    MetricObservationStatus,
    ModelAttemptStatus,
    ModelRole,
    ProviderAttemptEvidence,
    TokenUsage,
    UnavailableValue,
)
from rag_mvp.domain.retrieval import CacheOutcome, CachePolicy
from rag_mvp.evaluation.comparison import (
    COMPARISON_RERANKER_BENEFIT_PROFILE_ID,
    COMPARISON_RESULT_SCHEMA_VERSION,
    ComparisonArtifactManifest,
    ComparisonCandidateEvidence,
    ComparisonCandidateStatus,
    ComparisonCompatibility,
    ComparisonCompatibilityIssue,
    ComparisonDomainError,
    ComparisonIdentityProjection,
    ComparisonLogicalAttempt,
    ComparisonLogicalAttemptStatus,
    ComparisonProviderAttempt,
    ComparisonRecommendationState,
    ComparisonResult,
    ComparisonSharedSetupAttempt,
    ComparisonSharedSetupEvidence,
    ComparisonSharedSetupStatus,
    ComparisonStatus,
    ComparisonSuite,
    RerankerCaseEvidence,
    ResolvedComparisonArtifact,
    adapt_verified_evaluation_report,
    aggregate_comparison_result,
    build_comparison_candidate_evidence,
    canonical_comparison_manifest,
    create_comparison_suite,
    project_evaluation_identity,
    resolve_comparison_artifact,
    seal_comparison_candidate_evidence,
    validate_comparison_compatibility,
    validate_comparison_plan_safe_values,
    validate_comparison_shared_setup,
)
from rag_mvp.evaluation.experiment import (
    DeterministicSelectionPolicy,
    ExperimentAxis,
    ExperimentFixedIdentities,
    ExperimentGateProfile,
    ExperimentOrderPolicy,
    ExperimentPlan,
    ExperimentPricingProvenance,
    ExperimentPricingRate,
    ExperimentVariant,
    FinalTieBreak,
    FixedIdentity,
    PricingRole,
    RepeatOrderPolicy,
    SelectionCriterion,
    SelectionDirection,
)
from rag_mvp.evaluation.report_builder import case_ids_content_hash
from rag_mvp.evaluation.runner import EvaluationEnvironment, EvaluationRunIdentity

_NOW = datetime(2026, 8, 7, tzinfo=UTC)
_HASH_A = "sha256:" + ("a" * 64)
_HASH_B = "sha256:" + ("b" * 64)
_HASH_C = case_ids_content_hash(("case-1",))
_SOURCE = "https://pricing.example.test/comparison-v1"


def _rates() -> tuple[ExperimentPricingRate, ...]:
    return (
        ExperimentPricingRate(
            role=PricingRole.EMBEDDING,
            provider="provider-a",
            model="embed-v1",
            input_per_million=Decimal("0.02"),
            source_reference=_SOURCE,
        ),
        ExperimentPricingRate(
            role=PricingRole.GENERATION,
            provider="provider-a",
            model="gen-a",
            input_per_million=Decimal("1"),
            output_per_million=Decimal("2"),
            source_reference=_SOURCE,
        ),
        ExperimentPricingRate(
            role=PricingRole.GENERATION,
            provider="provider-a",
            model="gen-b",
            input_per_million=Decimal("2"),
            output_per_million=Decimal("4"),
            source_reference=_SOURCE,
        ),
        ExperimentPricingRate(
            role=PricingRole.RERANKING,
            provider="provider-a",
            model="rerank-v1",
            input_per_million=Decimal("1"),
            output_per_million=Decimal("1"),
            source_reference=_SOURCE,
        ),
    )


def _plan(
    axis: ExperimentAxis = ExperimentAxis.GENERATION_MODEL,
    *,
    maximum_provider_calls: int = 20,
    maximum_cost: Decimal = Decimal("1"),
) -> ExperimentPlan:
    axis_values = {
        ExperimentAxis.GENERATION_MODEL: ("gen-a", "gen-b"),
        ExperimentAxis.RETRIEVAL_STRATEGY: ("dense", "hybrid", "hybrid-rerank"),
        ExperimentAxis.CACHE_BEHAVIOR: ("cold", "warm"),
    }[axis]
    gates = ("quality",)
    controlled = [
        FixedIdentity(name="prompt.generation", value="prompt-v2"),
        FixedIdentity(name="provider.generation", value="provider-a"),
        FixedIdentity(name="provider.embedding", value="provider-a"),
        FixedIdentity(name="model.embedding", value="embed-v1"),
    ]
    if axis is not ExperimentAxis.GENERATION_MODEL:
        controlled.append(FixedIdentity(name="generation.model", value="gen-a"))
    if axis is not ExperimentAxis.RETRIEVAL_STRATEGY:
        controlled.extend(
            (
                FixedIdentity(name="retrieval.mode", value="dense"),
                FixedIdentity(name="provider.reranking", value="provider-a"),
                FixedIdentity(name="model.reranking", value="rerank-v1"),
            )
        )
    if axis is not ExperimentAxis.CACHE_BEHAVIOR:
        controlled.append(FixedIdentity(name="cache.behavior", value="bypass"))
    return ExperimentPlan.create(
        plan_id=f"{axis.value}-plan",
        display_name=f"{axis.value} plan",
        axis=axis,
        fixed_identities=ExperimentFixedIdentities(
            dataset_id="acceptance-v2",
            dataset_version="2.0.0",
            dataset_hash=_HASH_A,
            corpus_id="corpus-v2",
            corpus_version="2.0.0",
            corpus_hash=_HASH_B,
            case_set_hash=_HASH_C,
            case_count=1,
            controlled=tuple(controlled),
        ),
        variants=tuple(
            ExperimentVariant(
                variant_id=f"variant-{index}",
                display_name=f"Variant {index}",
                axis_value=value,
                configuration_id=f"semantic-config-{index}",
            )
            for index, value in enumerate(axis_values)
        ),
        baseline_variant_id="variant-0",
        repeat_order_policy=RepeatOrderPolicy(
            repeats_per_case=1,
            order_policy=ExperimentOrderPolicy.SEEDED_INTERLEAVED,
            seed=7,
        ),
        cache_policy=(
            CachePolicy.USE if axis is ExperimentAxis.CACHE_BEHAVIOR else CachePolicy.BYPASS
        ),
        pricing=ExperimentPricingProvenance(
            pricing_version="comparison-pricing-v1",
            pricing_hash=_HASH_A,
            currency="USD",
            source_references=(_SOURCE,),
            rate_card=_rates(),
        ),
        maximum_provider_calls=maximum_provider_calls,
        maximum_cost=maximum_cost,
        gate_profile=ExperimentGateProfile(
            profile_id="comparison-gates",
            profile_version="1.0.0",
            profile_hash=_HASH_B,
            mandatory_gate_ids=gates,
        ),
        selection_policy=DeterministicSelectionPolicy(
            policy_id="comparison-policy",
            policy_version="1.0.0",
            required_gate_ids=gates,
            tie_breakers=(
                SelectionCriterion(
                    metric="quality",
                    direction=SelectionDirection.MAXIMIZE,
                ),
            ),
            final_tie_break=FinalTieBreak.BASELINE_FIRST,
        ),
    )


def _suite(plan: ExperimentPlan | None = None) -> ComparisonSuite:
    active = plan or _plan()
    return create_comparison_suite(
        "comparison-1",
        active,
        {item.variant_id: f"run-{item.variant_id}" for item in active.variants},
        created_at=_NOW,
    )


def _shared_setup(
    plan: ExperimentPlan,
    *,
    comparison_id: str = "comparison-1",
    status: ComparisonSharedSetupStatus = ComparisonSharedSetupStatus.REUSED,
    attempts: tuple[ComparisonSharedSetupAttempt, ...] = (),
    safe_error_code: str | None = None,
    provider_calls_complete: bool = True,
) -> ComparisonSharedSetupEvidence:
    return ComparisonSharedSetupEvidence.create(
        comparison_id=comparison_id,
        plan=plan,
        status=status,
        attempts=attempts,
        safe_error_code=safe_error_code,
        provider_calls_complete=provider_calls_complete,
        recorded_at=_NOW,
    )


def _setup_attempt(
    plan: ExperimentPlan,
    *,
    attempt_number: int,
    status: ModelAttemptStatus,
    input_tokens: int | None = 100,
    source_run_id: str | None = None,
    recorded_at: datetime | None = None,
) -> ComparisonSharedSetupAttempt:
    binding = _shared_setup(plan)
    rate = next(item for item in plan.pricing.rate_card if item.role is PricingRole.EMBEDDING)
    return ComparisonSharedSetupAttempt.create(
        attempt_reference=f"setup-attempt-{attempt_number}",
        setup_id=binding.setup_id,
        request_id=binding.request_id,
        index_revision_id=binding.index_revision_id,
        source_run_id=source_run_id,
        evidence=ProviderAttemptEvidence(
            operation_id=binding.index_revision_id,
            attempt_number=attempt_number,
            role=ModelRole.EMBEDDING,
            provider=rate.provider,
            model=rate.model,
            status=status,
            latency_ms=12.0,
            safe_error_category=(
                None if status is ModelAttemptStatus.SUCCEEDED else "provider-timeout"
            ),
            usage=TokenUsage(input_tokens=input_tokens),
        ),
        latency_ms=12.0,
        pricing_version=plan.pricing.pricing_version,
        pricing_hash=plan.pricing.pricing_hash,
        currency=plan.pricing.currency,
        input_per_million=rate.input_per_million,
        output_per_million=rate.output_per_million,
        pricing_source_reference=rate.source_reference,
        recorded_at=recorded_at or _NOW,
    )


def _projection(
    plan: ExperimentPlan,
    variant_index: int,
    *,
    controlled_overrides: dict[str, str] | None = None,
) -> ComparisonIdentityProjection:
    variant = plan.variants[variant_index]
    values = {item.name: item.value for item in plan.fixed_identities.controlled}
    values.update(controlled_overrides or {})
    values[plan.axis.identity_name] = variant.axis_value
    if plan.axis is ExperimentAxis.RETRIEVAL_STRATEGY and variant.axis_value == "hybrid-rerank":
        values["provider.reranking"] = "provider-a"
        values["model.reranking"] = "rerank-v1"
    return ComparisonIdentityProjection(
        variant_id=variant.variant_id,
        configuration_id=variant.configuration_id,
        dataset_id=plan.fixed_identities.dataset_id,
        dataset_version=plan.fixed_identities.dataset_version,
        dataset_hash=plan.fixed_identities.dataset_hash,
        corpus_id=plan.fixed_identities.corpus_id,
        corpus_version=plan.fixed_identities.corpus_version,
        corpus_hash=plan.fixed_identities.corpus_hash,
        case_set_hash=plan.fixed_identities.case_set_hash,
        identities=tuple(FixedIdentity(name=name, value=value) for name, value in values.items()),
    )


def _quality_observation(value: float = 1.0) -> MetricObservation:
    return MetricObservation(
        metric_id="quality",
        unit="ratio",
        value=value,
        numerator=value,
        denominator=1,
        eligible=True,
        threshold=0.8,
        operator=EvidenceComparisonOperator.GREATER_THAN_OR_EQUAL,
        scorer_version="quality-v1",
        status=(MetricObservationStatus.PASSED if value >= 0.8 else MetricObservationStatus.FAILED),
    )


def _quality_gate() -> GateResult:
    return GateResult(
        gate_id="quality",
        profile_version="1.0.0",
        status=GateStatus.PASSED,
        valid=True,
        passed=True,
        case_executions_complete=True,
        observations=(_quality_observation(),),
    )


def _provider(
    plan: ExperimentPlan,
    *,
    reference: str,
    logical_attempt_id: str = "attempt-1",
    role: ModelRole = ModelRole.GENERATION,
    model: str = "gen-a",
    input_tokens: int | None = 100,
    output_tokens: int | None = 20,
    status: ModelAttemptStatus = ModelAttemptStatus.SUCCEEDED,
    evaluation_run_id: str = "run-variant-0",
    operation_id: str | None = None,
) -> ComparisonProviderAttempt:
    rate = next(
        item
        for item in plan.pricing.rate_card
        if item.role.value == role.value and item.model == model
    )
    evidence = ProviderAttemptEvidence(
        operation_id=operation_id
        or ("qa-generation" if role is ModelRole.GENERATION else "qa-retrieval"),
        role=role,
        provider="provider-a",
        model=model,
        status=status,
        latency_ms=10.0,
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )
    return ComparisonProviderAttempt.create(
        attempt_reference=reference,
        logical_attempt_id=logical_attempt_id,
        evaluation_run_id=evaluation_run_id,
        evidence=evidence,
        latency_ms=10.0,
        pricing_version=plan.pricing.pricing_version,
        pricing_hash=plan.pricing.pricing_hash,
        currency=plan.pricing.currency,
        input_per_million=rate.input_per_million,
        output_per_million=rate.output_per_million,
        pricing_source_reference=rate.source_reference,
    )


def _logical_attempt(
    providers: tuple[ComparisonProviderAttempt, ...],
    *,
    status: ComparisonLogicalAttemptStatus = ComparisonLogicalAttemptStatus.SUCCEEDED,
    safe_error_code: str | None = None,
    cache_policy: CachePolicy = CachePolicy.BYPASS,
    cache_outcome: CacheOutcome = CacheOutcome.BYPASS,
    terminal_kind: str | None = None,
) -> ComparisonLogicalAttempt:
    input_applicable = tuple(item for item in providers if item.input_per_million is not None)
    output_applicable = tuple(item for item in providers if item.output_per_million is not None)
    input_values = tuple(item.evidence.usage.input_tokens for item in input_applicable)
    output_values = tuple(item.evidence.usage.output_tokens for item in output_applicable)
    complete_cost = all(item.total_cost is not None for item in providers)
    known_partial_cost = sum(
        (item.known_partial_cost for item in providers),
        Decimal(0),
    )
    cost_unknown_reasons = tuple(
        sorted({reason for item in providers for reason in item.unknown_reasons})
    )
    return ComparisonLogicalAttempt(
        attempt_id="attempt-1",
        case_id="case-1",
        repeat_index=0,
        order_index=0,
        status=status,
        latency_ms=100.0,
        terminal_kind=terminal_kind
        or ("answer" if status is ComparisonLogicalAttemptStatus.SUCCEEDED else "error"),
        cache_policy=cache_policy,
        cache_outcome=cache_outcome,
        index_revision_id="index-revision-v1",
        retrieved_chunk_ids=("chunk-a", "chunk-b"),
        context_chunk_ids=("chunk-a",),
        retrieval_evidence_digest=_HASH_B,
        safe_error_code=safe_error_code,
        provider_attempt_references=tuple(item.attempt_reference for item in providers),
        provider_failed_attempt_count=sum(
            item.evidence.status is not ModelAttemptStatus.SUCCEEDED for item in providers
        ),
        input_tokens=(
            sum(item for item in input_values if item is not None)
            if all(item is not None for item in input_values)
            else None
        ),
        output_tokens=(
            sum(item for item in output_values if item is not None)
            if all(item is not None for item in output_values)
            else None
        ),
        estimated_cost=(
            sum((item.total_cost for item in providers if item.total_cost is not None), Decimal(0))
            if complete_cost
            else None
        ),
        known_partial_cost=known_partial_cost,
        cost_complete=complete_cost,
        cost_unknown_reasons=cost_unknown_reasons,
        currency="USD",
        completed_at=_NOW,
    )


def _candidate_evidence(
    plan: ExperimentPlan,
    providers: tuple[ComparisonProviderAttempt, ...],
    *,
    variant_index: int = 0,
    quality_value: float = 1.0,
    auto_embedding: bool | None = None,
    reranker_evidence: tuple[RerankerCaseEvidence, ...] = (),
    terminal_kind: str | None = None,
) -> ComparisonCandidateEvidence:
    suite = _suite(plan)
    reference = suite.candidates[variant_index].reference
    resolved_providers = tuple(providers)
    should_embed = (
        plan.cache_policy is CachePolicy.BYPASS
        or (
            plan.axis is ExperimentAxis.CACHE_BEHAVIOR
            and plan.variants[variant_index].axis_value == "cold"
        )
        if auto_embedding is None
        else auto_embedding
    )
    if should_embed and not any(
        item.evidence.role is ModelRole.EMBEDDING for item in resolved_providers
    ):
        resolved_providers = (
            _provider(
                plan,
                reference=f"provider-embedding-{variant_index}",
                role=ModelRole.EMBEDDING,
                model="embed-v1",
                input_tokens=100,
                output_tokens=None,
                evaluation_run_id=reference.evaluation_run_id,
            ),
            *resolved_providers,
        )
    cache_outcome = (
        CacheOutcome.BYPASS
        if plan.cache_policy is CachePolicy.BYPASS
        else CacheOutcome.MISS
        if plan.variants[variant_index].axis_value == "cold"
        else CacheOutcome.HIT
    )
    return build_comparison_candidate_evidence(
        comparison_id=suite.comparison_id,
        plan=plan,
        reference=reference,
        identity_projection=_projection(plan, variant_index),
        expected_case_ids=("case-1",),
        logical_attempts=(
            _logical_attempt(
                resolved_providers,
                cache_policy=plan.cache_policy,
                cache_outcome=cache_outcome,
                terminal_kind=terminal_kind,
            ),
        ),
        provider_attempts=resolved_providers,
        quality_metrics=(_quality_observation(quality_value),),
        gates=(_quality_gate(),),
        reranker_evidence=reranker_evidence,
        generated_at=_NOW,
    )


def test_suite_histories_are_append_only_and_preserve_partial_failure() -> None:
    suite = _suite()
    running = suite.transition_candidate(
        "variant-0",
        status=ComparisonCandidateStatus.RUNNING,
        completed_cases=0,
        failed_cases=0,
        provider_calls=0,
        recorded_at=_NOW + timedelta(seconds=1),
    )
    first_done = running.transition_candidate(
        "variant-0",
        status=ComparisonCandidateStatus.COMPLETED,
        completed_cases=1,
        failed_cases=0,
        provider_calls=2,
        incurred_cost=Decimal("0.01"),
        currency="USD",
        recorded_at=_NOW + timedelta(seconds=2),
    )
    second_running = first_done.transition_candidate(
        "variant-1",
        status=ComparisonCandidateStatus.RUNNING,
        completed_cases=0,
        failed_cases=0,
        provider_calls=0,
        recorded_at=_NOW + timedelta(seconds=3),
    )
    completed = second_running.transition_candidate(
        "variant-1",
        status=ComparisonCandidateStatus.COMPLETED,
        completed_cases=0,
        failed_cases=1,
        provider_calls=2,
        incurred_cost=Decimal("0.02"),
        currency="USD",
        recorded_at=_NOW + timedelta(seconds=4),
    )

    assert completed.status is ComparisonStatus.COMPLETED
    assert completed.partial_failure
    assert completed.latest_progress.completed_candidates == 2
    assert completed.latest_progress.failed_cases == 1
    assert completed.plan_content_hash == completed.plan.content_hash
    forged_status: dict[str, Any] = completed.model_dump(mode="json")
    forged_status["status"] = "running"
    forged_status["progress_history"][-1]["status"] = "running"
    with pytest.raises(ValidationError, match="status_not_derived"):
        ComparisonSuite.model_validate(forged_status)

    future_snapshot: dict[str, Any] = completed.model_dump(mode="json")
    future_snapshot["candidates"][1]["snapshots"][-1]["recorded_at"] = (
        completed.updated_at + timedelta(seconds=1)
    ).isoformat()
    with pytest.raises(ValidationError, match="timestamp_binding_invalid"):
        ComparisonSuite.model_validate(future_snapshot)

    finalized_failure = completed.fail(
        "publication-failed",
        recorded_at=_NOW + timedelta(seconds=5),
    )
    assert finalized_failure.status is ComparisonStatus.FAILED
    interrupted = completed.fail(
        "comparison-interrupted",
        recorded_at=_NOW + timedelta(seconds=5),
    )
    assert interrupted.status is ComparisonStatus.FAILED
    with pytest.raises(ValueError, match="not_finalization"):
        completed.fail("operator-cancelled", recorded_at=_NOW + timedelta(seconds=5))


def test_candidate_cost_history_retains_real_lower_bound_and_continues_after_timeout() -> None:
    suite = _suite().transition_candidate(
        "variant-0",
        status=ComparisonCandidateStatus.RUNNING,
        completed_cases=0,
        failed_cases=0,
        provider_calls=72,
        incurred_cost=Decimal("0.01870680"),
        currency="USD",
        recorded_at=_NOW + timedelta(seconds=1),
    )
    incomplete = suite.transition_candidate(
        "variant-0",
        status=ComparisonCandidateStatus.RUNNING,
        completed_cases=0,
        failed_cases=1,
        provider_calls=74,
        incurred_cost=None,
        known_partial_cost=Decimal("0.01870706"),
        cost_complete=False,
        cost_unknown_reasons=("input-usage-unknown",),
        currency="USD",
        recorded_at=_NOW + timedelta(seconds=2),
    )
    continued = incomplete.transition_candidate(
        "variant-0",
        status=ComparisonCandidateStatus.RUNNING,
        completed_cases=0,
        failed_cases=1,
        provider_calls=76,
        incurred_cost=None,
        known_partial_cost=Decimal("0.01880706"),
        cost_complete=False,
        cost_unknown_reasons=("input-usage-unknown",),
        currency="USD",
        recorded_at=_NOW + timedelta(seconds=3),
    )

    latest = continued.candidates[0].latest
    assert latest.sequence == 3
    assert latest.known_partial_cost == Decimal("0.01880706")
    assert latest.incurred_cost is None
    assert latest.cost_complete is False
    assert continued.latest_progress.known_partial_cost == Decimal("0.01880706")

    with pytest.raises(ValidationError, match="cost_history_not_monotonic"):
        incomplete.transition_candidate(
            "variant-0",
            status=ComparisonCandidateStatus.RUNNING,
            completed_cases=0,
            failed_cases=1,
            provider_calls=76,
            incurred_cost=Decimal("0.01880706"),
            known_partial_cost=Decimal("0.01880706"),
            cost_complete=True,
            cost_unknown_reasons=(),
            currency="USD",
            recorded_at=_NOW + timedelta(seconds=3),
        )
    with pytest.raises(ValidationError, match="cost_history_not_monotonic"):
        incomplete.transition_candidate(
            "variant-0",
            status=ComparisonCandidateStatus.RUNNING,
            completed_cases=0,
            failed_cases=1,
            provider_calls=76,
            incurred_cost=None,
            known_partial_cost=Decimal("0.01870705"),
            cost_complete=False,
            cost_unknown_reasons=("input-usage-unknown",),
            currency="USD",
            recorded_at=_NOW + timedelta(seconds=3),
        )


def test_legacy_numeric_candidate_cost_is_only_a_lower_bound_after_reopen() -> None:
    suite = _suite().transition_candidate(
        "variant-0",
        status=ComparisonCandidateStatus.RUNNING,
        completed_cases=0,
        failed_cases=1,
        provider_calls=74,
        incurred_cost=Decimal("0.01870680"),
        currency="USD",
        recorded_at=_NOW + timedelta(seconds=1),
    )
    payload: dict[str, Any] = suite.model_dump(mode="json")
    for history in payload["candidates"]:
        for snapshot in history["snapshots"]:
            snapshot.pop("known_partial_cost")
            snapshot.pop("cost_complete")
            snapshot.pop("cost_unknown_reasons")
    for progress in payload["progress_history"]:
        progress.pop("known_partial_cost")
        progress.pop("cost_complete")
        progress.pop("cost_unknown_reasons")

    reopened = ComparisonSuite.model_validate(payload)
    latest = reopened.candidates[0].latest
    assert latest.known_partial_cost == Decimal("0.01870680")
    assert latest.incurred_cost is None
    assert latest.cost_complete is False
    assert latest.cost_unknown_reasons == ("legacy-cost-completeness-unavailable",)
    assert reopened.latest_progress.known_partial_cost == Decimal("0.01870680")
    assert reopened.latest_progress.cost_complete is False


@pytest.mark.parametrize("axis", tuple(ExperimentAxis))
def test_compatibility_allows_only_axis_and_distinct_predeclared_configuration_ids(
    axis: ExperimentAxis,
) -> None:
    plan = _plan(axis)
    projections = tuple(_projection(plan, index) for index, _ in enumerate(plan.variants))

    compatibility = validate_comparison_compatibility(plan, projections)

    assert compatibility.compatible
    assert compatibility.issues == ()


def test_non_axis_identity_difference_is_reported_without_deltas() -> None:
    plan = _plan()
    projections = tuple(
        _projection(
            plan,
            index,
            controlled_overrides={"prompt.generation": "prompt-v3" if index else "prompt-v2"},
        )
        for index, _ in enumerate(plan.variants)
    )

    compatibility = validate_comparison_compatibility(plan, projections)

    assert not compatibility.compatible
    assert compatibility.issues[0].identity_name == "prompt.generation"
    assert compatibility.issues[0].variant_id == "variant-1"


def test_shared_setup_ledger_prices_retries_and_is_included_in_result_caps() -> None:
    plan = _plan(maximum_provider_calls=5)
    first = _setup_attempt(
        plan,
        attempt_number=1,
        status=ModelAttemptStatus.FAILED,
        input_tokens=100,
        recorded_at=_NOW - timedelta(seconds=2),
    )
    second = _setup_attempt(
        plan,
        attempt_number=2,
        status=ModelAttemptStatus.SUCCEEDED,
        input_tokens=200,
        recorded_at=_NOW - timedelta(seconds=1),
    )
    setup = _shared_setup(
        plan,
        status=ComparisonSharedSetupStatus.COMPLETED,
        attempts=(first, second),
    )
    evidence = _verified_reports(plan, quality_values=(0.9, 1.0))
    suite = _terminal_suite(plan, evidence)
    reports = {
        item.variant_id: seal_comparison_candidate_evidence(
            suite.candidates[index].reference,
            evidence[index],
        )
        for index, item in enumerate(plan.variants)
    }

    result = aggregate_comparison_result(
        suite,
        _compatibility(plan),
        reports,
        shared_setup=setup,
    )

    assert setup.provider_call_count == 2
    assert setup.known_partial_cost == Decimal("0.000006")
    assert setup.total_cost == Decimal("0.000006")
    assert result.provider_call_count == setup.provider_call_count + sum(
        item.provider_call_count for item in result.candidates
    )
    assert result.total_cost == setup.total_cost + sum(
        (item.total_cost for item in result.candidates if item.total_cost is not None),
        start=Decimal(0),
    )
    assert result.candidates[1].metrics[0].baseline_delta == pytest.approx(0.1)
    assert result.recommendation.state is ComparisonRecommendationState.NO_RECOMMENDATION
    assert result.recommendation.rationale_codes == ("provider-call-cap-exceeded",)


def test_shared_setup_unknown_usage_keeps_lower_bound_without_blocking_selection() -> None:
    plan = _plan()
    unknown = _setup_attempt(
        plan,
        attempt_number=1,
        status=ModelAttemptStatus.SUCCEEDED,
        input_tokens=None,
        recorded_at=_NOW - timedelta(seconds=1),
    )
    unknown_setup = _shared_setup(
        plan,
        status=ComparisonSharedSetupStatus.COMPLETED,
        attempts=(unknown,),
    )
    evidence = _verified_reports(plan)
    suite = _terminal_suite(plan, evidence)
    reports = {
        item.variant_id: seal_comparison_candidate_evidence(
            suite.candidates[index].reference,
            evidence[index],
        )
        for index, item in enumerate(plan.variants)
    }

    result = aggregate_comparison_result(
        suite,
        _compatibility(plan),
        reports,
        shared_setup=unknown_setup,
    )

    assert unknown_setup.known_partial_cost == 0
    assert unknown_setup.total_cost is None
    assert unknown_setup.unknown_reasons == ("input-usage-unknown",)
    assert result.total_cost is None
    assert result.recommendation.state is ComparisonRecommendationState.RECOMMENDED
    assert "comparison-cost-lower-bound-only" in result.recommendation.rationale_codes

    successful = _setup_attempt(
        plan,
        attempt_number=1,
        status=ModelAttemptStatus.SUCCEEDED,
        recorded_at=_NOW - timedelta(seconds=1),
    )
    failed_after_embedding = _shared_setup(
        plan,
        status=ComparisonSharedSetupStatus.FAILED,
        attempts=(successful,),
        safe_error_code="index-publication-failed",
    )
    failed_before_provider = _shared_setup(
        plan,
        status=ComparisonSharedSetupStatus.FAILED,
        safe_error_code="setup-validation-failed",
    )
    assert failed_after_embedding.total_cost == successful.total_cost
    assert failed_before_provider.provider_call_count == 0
    with pytest.raises(ComparisonDomainError, match="shared_setup_not_ready"):
        aggregate_comparison_result(
            suite,
            _compatibility(plan),
            reports,
            shared_setup=failed_after_embedding,
        )


def test_failed_shared_setup_can_preserve_an_explicit_unavailable_aggregate() -> None:
    plan = _plan()
    unavailable = _shared_setup(
        plan,
        status=ComparisonSharedSetupStatus.FAILED,
        safe_error_code="setup-ledger-integrity-failed",
        provider_calls_complete=False,
    )

    assert unavailable.provider_calls_complete is False
    assert isinstance(unavailable.provider_call_count, UnavailableValue)
    assert unavailable.provider_call_count.reason == ("setup-ledger-integrity-unavailable")
    assert unavailable.known_partial_cost == 0
    assert unavailable.total_cost is None
    assert unavailable.cost_complete is False
    assert unavailable.unknown_reasons == ("setup-ledger-integrity-unavailable",)
    assert (
        ComparisonSharedSetupEvidence.model_validate_json(unavailable.model_dump_json())
        == unavailable
    )

    raw: dict[str, Any] = unavailable.model_dump(mode="json")
    raw["provider_call_count"] = 0
    with pytest.raises(ValidationError, match="aggregate_mismatch"):
        ComparisonSharedSetupEvidence.model_validate(raw)

    for status in ("reused", "completed"):
        raw = unavailable.model_dump(mode="json")
        raw["status"] = status
        raw["safe_error_code"] = None
        with pytest.raises(ValidationError, match="aggregate_unavailable_not_failed"):
            ComparisonSharedSetupEvidence.model_validate(raw)


def test_shared_setup_rejects_foreign_binding_tampering_and_bounds_setup_id() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="source_run_must_be_unbound"):
        _setup_attempt(
            plan,
            attempt_number=1,
            status=ModelAttemptStatus.SUCCEEDED,
            source_run_id="candidate-run",
        )
    attempt = _setup_attempt(
        plan,
        attempt_number=1,
        status=ModelAttemptStatus.SUCCEEDED,
        recorded_at=_NOW - timedelta(seconds=1),
    )
    setup = _shared_setup(
        plan,
        status=ComparisonSharedSetupStatus.COMPLETED,
        attempts=(attempt,),
    )
    raw: dict[str, Any] = setup.model_dump(mode="json")
    raw["request_id"] = "foreign-request"
    with pytest.raises(ValidationError, match="identity_mismatch"):
        ComparisonSharedSetupEvidence.model_validate(raw)
    raw = setup.model_dump(mode="json")
    raw["attempts"][0]["recorded_at"] = (_NOW + timedelta(seconds=1)).isoformat()
    with pytest.raises(ValidationError, match="attempt_history_invalid"):
        ComparisonSharedSetupEvidence.model_validate(raw)
    raw = setup.model_dump(mode="json")
    raw["status"] = "reused"
    with pytest.raises(ValidationError, match="reuse_has_provider_attempts"):
        ComparisonSharedSetupEvidence.model_validate(raw)

    for field, value in (("provider", "provider-b"), ("model", "embed-v2")):
        raw = setup.model_dump(mode="json")
        raw["attempts"][0]["evidence"][field] = value
        forged = ComparisonSharedSetupEvidence.model_validate(raw)
        with pytest.raises(ComparisonDomainError, match="rate_plan_mismatch"):
            validate_comparison_shared_setup(
                forged,
                comparison_id="comparison-1",
                plan=plan,
            )
    raw = setup.model_dump(mode="json")
    raw["attempts"][0]["evidence"]["role"] = "generation"
    with pytest.raises(ValidationError, match="role_invalid"):
        ComparisonSharedSetupEvidence.model_validate(raw)
    raw = setup.model_dump(mode="json")
    raw["attempts"][0]["evidence"]["operation_id"] = "foreign-revision"
    with pytest.raises(ValidationError, match="revision_mismatch"):
        ComparisonSharedSetupEvidence.model_validate(raw)

    longest = _shared_setup(plan, comparison_id="c" * 255)
    assert len(longest.setup_id) <= 255
    assert longest.setup_id == _shared_setup(plan, comparison_id="c" * 255).setup_id


def test_provider_ledger_recomputes_latency_tokens_cost_and_rejects_metric_tampering() -> None:
    plan = _plan()
    provider = _provider(plan, reference="provider-1")
    evidence = _candidate_evidence(plan, (provider,))
    metrics = {item.metric_id: item for item in evidence.metrics}

    assert metrics["comparison-logical-all-p90-ms"].value == 100.0
    assert metrics["comparison-logical-all-p90-ms"].denominator == 1
    assert metrics["comparison-input-tokens"].value == 200.0
    assert metrics["comparison-output-tokens"].value == 20.0
    assert evidence.provider_call_count == 2
    assert evidence.total_cost == Decimal("0.000142")
    assert metrics["comparison-cost-per-1000-logical-attempts"].unit == (
        "USD-per-1000-logical-attempts"
    )
    assert metrics["comparison-cost-per-1000-successes"].unit == "USD-per-1000-successes"

    raw: dict[str, Any] = evidence.model_dump(mode="json")
    metric = next(
        item for item in raw["metrics"] if item["metric_id"] == "comparison-logical-all-p90-ms"
    )
    metric["value"] = 99.0
    metric["numerator"] = 99.0
    with pytest.raises(ValidationError, match="attempt_metric_ledger_mismatch"):
        ComparisonCandidateEvidence.model_validate(raw)
    swapped_units: dict[str, Any] = evidence.model_dump(mode="json")
    logical_cost = next(
        item
        for item in swapped_units["metrics"]
        if item["metric_id"] == "comparison-cost-per-1000-logical-attempts"
    )
    logical_cost["unit"] = "USD-per-1000-successes"
    with pytest.raises(ValidationError, match="attempt_metric_ledger_mismatch"):
        ComparisonCandidateEvidence.model_validate(swapped_units)


def test_answer_provider_roles_are_operation_bound_and_plan_aware() -> None:
    plan = _plan()
    generation = _provider(plan, reference="provider-generation")
    with pytest.raises(ComparisonDomainError, match="embedding_evidence_missing"):
        _candidate_evidence(plan, (generation,), auto_embedding=False)

    wrong_embedding = _provider(
        plan,
        reference="provider-fact-assessment",
        role=ModelRole.EMBEDDING,
        model="embed-v1",
        input_tokens=20,
        output_tokens=None,
        operation_id="fact-evidence-assessment",
    )
    with pytest.raises(ComparisonDomainError, match="embedding_evidence_missing"):
        _candidate_evidence(
            plan,
            (wrong_embedding, generation),
            auto_embedding=False,
        )

    retrieval_plan = _plan(ExperimentAxis.RETRIEVAL_STRATEGY)
    retrieval_generation = _provider(
        retrieval_plan,
        reference="provider-generation",
        evaluation_run_id="run-variant-2",
    )
    with pytest.raises(ComparisonDomainError, match="reranking_evidence_missing"):
        _candidate_evidence(
            retrieval_plan,
            (retrieval_generation,),
            variant_index=2,
        )


def test_unknown_usage_is_unavailable_and_embedding_output_is_not_required() -> None:
    plan = _plan()
    embedding = _provider(
        plan,
        reference="provider-embed",
        role=ModelRole.EMBEDDING,
        model="embed-v1",
        input_tokens=40,
        output_tokens=None,
    )
    generation = _provider(plan, reference="provider-generation")
    reranking = _provider(
        plan,
        reference="provider-rerank",
        role=ModelRole.RERANKING,
        model="rerank-v1",
        input_tokens=30,
        output_tokens=5,
    )
    complete = _candidate_evidence(plan, (embedding, generation, reranking))
    complete_metrics = {item.metric_id: item for item in complete.metrics}
    assert complete_metrics["comparison-output-tokens"].value == 25.0
    assert complete_metrics["comparison-output-tokens"].denominator == 3

    unknown_generation = _provider(
        plan,
        reference="provider-unknown",
        output_tokens=None,
    )
    unknown = _candidate_evidence(plan, (unknown_generation,))
    unknown_metrics = {item.metric_id: item for item in unknown.metrics}
    assert isinstance(unknown_metrics["comparison-output-tokens"].value, UnavailableValue)
    assert unknown_metrics["comparison-output-tokens"].denominator == 2
    assert isinstance(
        unknown_metrics["comparison-cost-per-1000-logical-attempts"].value,
        UnavailableValue,
    )
    assert unknown.total_cost is None
    assert unknown.cost_complete is False
    assert unknown.cost_unknown_reasons == ("output-usage-unknown",)
    assert unknown.known_partial_cost == sum(
        (item.known_partial_cost for item in unknown.provider_attempts),
        Decimal(0),
    )
    assert unknown.logical_attempts[0].known_partial_cost == unknown.known_partial_cost
    tampered = unknown.model_dump(mode="json")
    tampered["logical_attempts"][0]["known_partial_cost"] = "0"
    with pytest.raises(ValidationError, match="logical_provider_aggregate_mismatch"):
        ComparisonCandidateEvidence.model_validate(tampered)


def test_incomplete_candidate_cost_keeps_lower_bound_while_caps_remain_fail_closed() -> None:
    plan = _plan(maximum_cost=Decimal("1"))
    evidence = list(_verified_reports(plan))
    unknown_generation = _provider(
        plan,
        reference="provider-incomplete-generation",
        output_tokens=None,
    )
    evidence[0] = _candidate_evidence(plan, (unknown_generation,), variant_index=0)
    evidence_tuple = tuple(evidence)
    suite = _terminal_suite(plan, evidence_tuple)
    reports = {
        item.variant_id: seal_comparison_candidate_evidence(
            suite.candidates[index].reference,
            evidence_tuple[index],
        )
        for index, item in enumerate(plan.variants)
    }

    result = aggregate_comparison_result(
        suite,
        _compatibility(plan),
        reports,
        shared_setup=_shared_setup(plan),
    )

    assert result.known_partial_cost == sum(
        (item.known_partial_cost for item in result.candidates),
        Decimal(0),
    )
    assert result.total_cost is None
    assert result.cost_complete is False
    assert result.cost_unknown_reasons == ("output-usage-unknown",)
    assert result.recommendation.state is ComparisonRecommendationState.RECOMMENDED
    assert "comparison-cost-lower-bound-only" in result.recommendation.rationale_codes

    capped_plan = _plan(maximum_cost=Decimal("0.000001"))
    capped_evidence = list(_verified_reports(capped_plan))
    capped_unknown = _provider(
        capped_plan,
        reference="provider-capped-incomplete",
        output_tokens=None,
    )
    capped_evidence[0] = _candidate_evidence(
        capped_plan,
        (capped_unknown,),
        variant_index=0,
    )
    capped_tuple = tuple(capped_evidence)
    capped_suite = _terminal_suite(capped_plan, capped_tuple)
    capped_reports = {
        item.variant_id: seal_comparison_candidate_evidence(
            capped_suite.candidates[index].reference,
            capped_tuple[index],
        )
        for index, item in enumerate(capped_plan.variants)
    }
    capped = aggregate_comparison_result(
        capped_suite,
        _compatibility(capped_plan),
        capped_reports,
        shared_setup=_shared_setup(capped_plan),
    )
    assert capped.known_partial_cost > capped.plan.maximum_cost
    assert capped.recommendation.rationale_codes == ("comparison-cost-cap-exceeded",)


def test_reranker_proof_is_bound_to_real_successful_provider_attempt() -> None:
    plan = _plan()
    generation = _provider(plan, reference="provider-generation")
    reranking = _provider(
        plan,
        reference="provider-rerank",
        role=ModelRole.RERANKING,
        model="rerank-v1",
        input_tokens=30,
        output_tokens=5,
    )
    proof = RerankerCaseEvidence(
        candidate_variant_id="variant-0",
        case_id="case-1",
        logical_attempt_id="attempt-1",
        rerank_sensitive=True,
        pre_rerank_chunk_ids=("chunk-a", "chunk-b"),
        post_rerank_chunk_ids=("chunk-b", "chunk-a"),
        pre_rerank_context_chunk_ids=(),
        selected_context_chunk_ids=("chunk-b",),
        reranking_attempt_references=("provider-rerank",),
        successful_reranking_attempt_references=("provider-rerank",),
    )
    evidence = _candidate_evidence(
        plan,
        (generation, reranking),
        reranker_evidence=(proof,),
    )
    assert evidence.reranker_evidence[0].discriminating

    raw: dict[str, Any] = evidence.model_dump(mode="json")
    raw["reranker_evidence"][0]["successful_reranking_attempt_references"] = ["fake-ref"]
    with pytest.raises(ValidationError):
        ComparisonCandidateEvidence.model_validate(raw)


def _verified_reports(
    plan: ExperimentPlan,
    *,
    quality_values: tuple[float, ...] | None = None,
) -> tuple[ComparisonCandidateEvidence, ...]:
    values = quality_values or tuple(1.0 - (index / 10) for index in range(len(plan.variants)))
    evidence: list[ComparisonCandidateEvidence] = []
    for index, variant in enumerate(plan.variants):
        model = variant.axis_value if plan.axis is ExperimentAxis.GENERATION_MODEL else "gen-a"
        provider = _provider(
            plan,
            reference=f"provider-{variant.variant_id}",
            model=model,
            evaluation_run_id=f"run-{variant.variant_id}",
        )
        providers = [provider]
        if plan.axis is ExperimentAxis.RETRIEVAL_STRATEGY and variant.axis_value == "hybrid-rerank":
            providers.append(
                _provider(
                    plan,
                    reference=f"provider-reranking-{variant.variant_id}",
                    role=ModelRole.RERANKING,
                    model="rerank-v1",
                    input_tokens=20,
                    output_tokens=2,
                    evaluation_run_id=f"run-{variant.variant_id}",
                )
            )
        evidence.append(
            _candidate_evidence(
                plan,
                tuple(providers),
                variant_index=index,
                quality_value=values[index],
            )
        )
    return tuple(evidence)


def _terminal_suite(
    plan: ExperimentPlan,
    evidence: tuple[ComparisonCandidateEvidence | None, ...],
) -> ComparisonSuite:
    suite = _suite(plan)
    timestamp = _NOW
    for history, candidate_evidence in zip(suite.candidates, evidence, strict=True):
        timestamp += timedelta(seconds=1)
        suite = suite.transition_candidate(
            history.reference.variant_id,
            status=ComparisonCandidateStatus.RUNNING,
            completed_cases=0,
            failed_cases=0,
            provider_calls=0,
            recorded_at=timestamp,
        )
        timestamp += timedelta(seconds=1)
        if candidate_evidence is None:
            suite = suite.transition_candidate(
                history.reference.variant_id,
                status=ComparisonCandidateStatus.FAILED,
                completed_cases=0,
                failed_cases=1,
                provider_calls=0,
                safe_error_code="candidate-execution-failed",
                recorded_at=timestamp,
            )
            continue
        succeeded = sum(
            item.status is ComparisonLogicalAttemptStatus.SUCCEEDED
            for item in candidate_evidence.logical_attempts
        )
        suite = suite.transition_candidate(
            history.reference.variant_id,
            status=ComparisonCandidateStatus.COMPLETED,
            completed_cases=succeeded,
            failed_cases=candidate_evidence.failed_case_count,
            provider_calls=candidate_evidence.provider_call_count,
            incurred_cost=candidate_evidence.total_cost,
            known_partial_cost=candidate_evidence.known_partial_cost,
            cost_complete=candidate_evidence.cost_complete,
            cost_unknown_reasons=candidate_evidence.cost_unknown_reasons,
            currency=candidate_evidence.currency,
            recorded_at=timestamp,
        )
    return suite


def _compatibility(plan: ExperimentPlan) -> ComparisonCompatibility:
    return validate_comparison_compatibility(
        plan,
        tuple(_projection(plan, index) for index, _ in enumerate(plan.variants)),
    )


def test_aggregation_recomputes_deltas_gates_and_deterministic_selection() -> None:
    plan = _plan()
    evidence = _verified_reports(plan, quality_values=(0.9, 1.0))
    suite = _terminal_suite(plan, evidence)
    reports = {
        item.variant_id: seal_comparison_candidate_evidence(
            suite.candidates[index].reference,
            evidence[index],
        )
        for index, item in enumerate(plan.variants)
    }

    result = aggregate_comparison_result(
        suite,
        _compatibility(plan),
        reports,
        shared_setup=_shared_setup(plan),
        completed_at=suite.updated_at,
    )

    assert result.recommendation.state is ComparisonRecommendationState.RECOMMENDED
    assert result.recommendation.selected_variant_id == "variant-1"
    metrics = {item.metric_id: item for item in result.candidates[1].metrics}
    assert metrics["quality"].baseline_delta == pytest.approx(0.1)
    assert result.gates == tuple(
        gate for candidate in result.candidates for gate in candidate.gates
    )

    raw: dict[str, Any] = result.model_dump(mode="json")
    quality = next(
        item for item in raw["candidates"][1]["metrics"] if item["metric_id"] == "quality"
    )
    quality["baseline_delta"] = 999.0
    with pytest.raises(ValidationError, match="baseline_delta_mismatch"):
        ComparisonResult.model_validate(raw)
    raw = result.model_dump(mode="json")
    raw["recommendation"] = {
        "state": "recommended",
        "selected_variant_id": "variant-0",
        "rationale_codes": ["manual-selection"],
    }
    with pytest.raises(ValidationError, match="recommendation_not_deterministic"):
        ComparisonResult.model_validate(raw)


def test_aggregation_preserves_failed_candidate_and_caps_force_no_recommendation() -> None:
    plan = _plan(maximum_provider_calls=1)
    complete = _verified_reports(plan)
    evidence: tuple[ComparisonCandidateEvidence | None, ...] = (complete[0], None)
    suite = _terminal_suite(plan, evidence)
    reports = {
        "variant-0": seal_comparison_candidate_evidence(
            suite.candidates[0].reference,
            complete[0],
        ),
        "variant-1": None,
    }

    result = aggregate_comparison_result(
        suite,
        _compatibility(plan),
        reports,
        shared_setup=_shared_setup(plan),
    )

    assert result.recommendation.state is ComparisonRecommendationState.NO_RECOMMENDATION
    assert "candidate-evidence-incomplete" in result.recommendation.rationale_codes
    assert result.candidates[1].status is ComparisonCandidateStatus.FAILED
    assert isinstance(result.candidates[1].metrics[0].value, UnavailableValue)

    capped_evidence = _verified_reports(plan)
    capped_suite = _terminal_suite(plan, capped_evidence)
    capped_reports = {
        item.variant_id: seal_comparison_candidate_evidence(
            capped_suite.candidates[index].reference,
            capped_evidence[index],
        )
        for index, item in enumerate(plan.variants)
    }
    capped = aggregate_comparison_result(
        capped_suite,
        _compatibility(plan),
        capped_reports,
        shared_setup=_shared_setup(plan),
    )
    assert capped.recommendation.rationale_codes == ("provider-call-cap-exceeded",)


def test_non_discriminating_reranker_cannot_be_selected() -> None:
    plan = _plan(ExperimentAxis.RETRIEVAL_STRATEGY)
    evidence = _verified_reports(plan, quality_values=(0.8, 0.9, 1.0))
    suite = _terminal_suite(plan, evidence)
    reports = {
        item.variant_id: seal_comparison_candidate_evidence(
            suite.candidates[index].reference,
            evidence[index],
        )
        for index, item in enumerate(plan.variants)
    }

    result = aggregate_comparison_result(
        suite,
        _compatibility(plan),
        reports,
        shared_setup=_shared_setup(plan),
    )

    assert result.recommendation.selected_variant_id == "variant-1"
    assert result.candidates[2].reference.axis_value == "hybrid-rerank"
    assert "reranker-non-discriminating-excluded" in result.recommendation.rationale_codes


def test_discriminating_reranker_requires_predeclared_minimum_quality_benefit() -> None:
    source = _plan(ExperimentAxis.RETRIEVAL_STRATEGY)
    plan = ExperimentPlan.create(
        **{
            **source.model_dump(exclude={"content_hash"}),
            "gate_profile": source.gate_profile.model_copy(
                update={"profile_id": COMPARISON_RERANKER_BENEFIT_PROFILE_ID}
            ),
        }
    )
    evidence = list(_verified_reports(plan, quality_values=(0.8, 0.9, 0.9)))
    generation = _provider(
        plan,
        reference="provider-rerank-generation",
        evaluation_run_id="run-variant-2",
    )
    reranking = _provider(
        plan,
        reference="provider-rerank-minimum-benefit",
        role=ModelRole.RERANKING,
        model="rerank-v1",
        input_tokens=20,
        output_tokens=2,
        evaluation_run_id="run-variant-2",
    )
    proof = RerankerCaseEvidence(
        candidate_variant_id="variant-2",
        case_id="case-1",
        logical_attempt_id="attempt-1",
        rerank_sensitive=True,
        pre_rerank_chunk_ids=("chunk-a", "chunk-b"),
        post_rerank_chunk_ids=("chunk-b", "chunk-a"),
        pre_rerank_context_chunk_ids=(),
        selected_context_chunk_ids=("chunk-b",),
        reranking_attempt_references=(reranking.attempt_reference,),
        successful_reranking_attempt_references=(reranking.attempt_reference,),
    )
    evidence[2] = _candidate_evidence(
        plan,
        (generation, reranking),
        variant_index=2,
        quality_value=0.9,
        reranker_evidence=(proof,),
    )
    candidate_evidence = tuple(evidence)
    suite = _terminal_suite(plan, candidate_evidence)
    reports = {
        item.variant_id: seal_comparison_candidate_evidence(
            suite.candidates[index].reference,
            candidate_evidence[index],
        )
        for index, item in enumerate(plan.variants)
    }

    result = aggregate_comparison_result(
        suite,
        _compatibility(plan),
        reports,
        shared_setup=_shared_setup(plan),
    )

    assert result.recommendation.selected_variant_id == "variant-1"
    assert "reranker-minimum-quality-benefit-not-met" in (result.recommendation.rationale_codes)


def test_cache_axis_pairs_miss_hit_and_rejects_retrieval_evidence_drift() -> None:
    plan = _plan(ExperimentAxis.CACHE_BEHAVIOR)
    cold = _verified_reports(plan, quality_values=(1.0, 1.0))[0]
    warm_generation = _provider(
        plan,
        reference="provider-warm-generation",
        evaluation_run_id="run-variant-1",
    )
    unrelated_warm_attempt = _provider(
        plan,
        reference="provider-warm-fact-assessment",
        role=ModelRole.EMBEDDING,
        model="embed-v1",
        input_tokens=10,
        output_tokens=None,
        evaluation_run_id="run-variant-1",
        operation_id="fact-evidence-assessment",
    )
    warm = _candidate_evidence(
        plan,
        (warm_generation, unrelated_warm_attempt),
        variant_index=1,
        quality_value=1.0,
        auto_embedding=False,
    )
    evidence = (cold, warm)
    suite = _terminal_suite(plan, evidence)
    reports = {
        item.variant_id: seal_comparison_candidate_evidence(
            suite.candidates[index].reference,
            evidence[index],
        )
        for index, item in enumerate(plan.variants)
    }

    result = aggregate_comparison_result(
        suite,
        _compatibility(plan),
        reports,
        shared_setup=_shared_setup(plan),
    )
    observations = {item.metric_id: item for item in result.cache_observations}
    assert observations["comparison-cache-hit-rate"].value == 1.0
    assert observations["comparison-cache-hit-rate"].denominator == 1
    assert observations["comparison-cache-embedding-call-reduction"].value == 1.0
    assert observations["comparison-cache-warm-hits"].value == 1.0
    assert observations["comparison-cache-warm-hits"].unit == "hits"
    assert result.recommendation.state is ComparisonRecommendationState.RECOMMENDED

    retrieval_on_hit = _provider(
        plan,
        reference="provider-warm-retrieval",
        role=ModelRole.EMBEDDING,
        model="embed-v1",
        input_tokens=10,
        output_tokens=None,
        evaluation_run_id="run-variant-1",
    )
    with pytest.raises(ComparisonDomainError, match="cache_hit_provider_attempt_present"):
        _candidate_evidence(
            plan,
            (warm_generation, retrieval_on_hit),
            variant_index=1,
            auto_embedding=False,
        )
    with pytest.raises(ComparisonDomainError, match="cache_hit_provider_attempt_present"):
        _candidate_evidence(
            plan,
            (warm_generation, retrieval_on_hit),
            variant_index=1,
            auto_embedding=False,
            terminal_kind="refusal",
        )

    tampered_raw: dict[str, Any] = evidence[1].model_dump(mode="json")
    tampered_raw["logical_attempts"][0]["retrieved_chunk_ids"] = ["chunk-b", "chunk-a"]
    tampered_warm = ComparisonCandidateEvidence.model_validate(tampered_raw)
    tampered_reports = {
        "variant-0": reports["variant-0"],
        "variant-1": seal_comparison_candidate_evidence(
            suite.candidates[1].reference,
            tampered_warm,
        ),
    }
    tampered = aggregate_comparison_result(
        suite,
        _compatibility(plan),
        tampered_reports,
        shared_setup=_shared_setup(plan),
    )
    assert tampered.recommendation.state is ComparisonRecommendationState.NO_RECOMMENDATION
    assert "cache-retrieval-equivalence-mismatch" in tampered.recommendation.rationale_codes
    tampered_observations = {item.metric_id: item for item in tampered.cache_observations}
    assert tampered_observations["comparison-cache-retrieval-equivalence-rate"].value == 0.0
    assert tampered_observations["comparison-cache-retrieval-equivalence-rate"].denominator == 1
    assert tampered_observations["comparison-cache-hit-rate"].value == 1.0
    assert tampered_observations["comparison-cache-embedding-call-reduction"].value == 1.0
    assert all(
        item.status is MetricObservationStatus.OBSERVED for item in tampered.cache_observations
    )


def test_cache_hit_or_miss_refusal_requires_retrieval_equivalence_evidence() -> None:
    plan = _plan(ExperimentAxis.CACHE_BEHAVIOR)
    warm = _verified_reports(plan)[1]
    raw: dict[str, Any] = warm.model_dump(mode="json")
    attempt = raw["logical_attempts"][0]
    attempt["terminal_kind"] = "refusal"
    attempt["index_revision_id"] = {
        "status": "unavailable",
        "reason": "not-recorded",
    }
    attempt["retrieval_evidence_digest"] = {
        "status": "unavailable",
        "reason": "not-recorded",
    }
    with pytest.raises(ValidationError, match="retrieval_equivalence_evidence_missing"):
        ComparisonCandidateEvidence.model_validate(raw)


def test_failed_cache_miss_preserves_unavailable_retrieval_equivalence() -> None:
    plan = _plan(ExperimentAxis.CACHE_BEHAVIOR)
    warm = _verified_reports(plan)[1]
    raw: dict[str, Any] = warm.model_dump(mode="json")
    attempt = raw["logical_attempts"][0]
    attempt["status"] = "error"
    attempt["terminal_kind"] = "error"
    attempt["safe_error_code"] = "evaluation-case-failed"
    attempt["index_revision_id"] = {
        "status": "unavailable",
        "reason": "index-revision-unavailable",
    }
    attempt["retrieval_evidence_digest"] = {
        "status": "unavailable",
        "reason": "retrieval-evidence-digest-unavailable",
    }

    logical_attempt = ComparisonLogicalAttempt.model_validate(attempt)

    assert logical_attempt.safe_error_code == "evaluation-case-failed"


def _run_identity(*, include_secret_key: bool = False) -> EvaluationRunIdentity:
    providers = {"generation": "provider-a"}
    if include_secret_key:
        providers["api-key"] = "sk-secret-value"
    return EvaluationRunIdentity(
        dataset_id="acceptance-v2",
        dataset_version="2.0.0",
        dataset_hash=_HASH_A,
        corpus_version="2.0.0",
        corpus_hash=_HASH_B,
        configuration_id="semantic-config-0",
        code_revision="revision-v1",
        prompt_versions={"generation": "prompt-v2"},
        provider_identities=providers,
        model_identities={"generation": "gen-a"},
        generation_settings={"temperature": 0},
        embedding_identity={"model": "embed-v1"},
        chunking_identity={"chunking_version": "structure-page-token-v1"},
        retrieval_configuration={"mode": "dense"},
        scorer_versions={"advanced-quality-gate": "advanced-quality-v2"},
        pricing_version="comparison-pricing-v1",
        random_seeds={"case-order": 7},
        environment=EvaluationEnvironment(
            python_version="3.13.0",
            platform="Windows-AMD64",
            deployment="test",
        ),
        cache_policy=CachePolicy.BYPASS,
    )


def test_identity_projection_uses_strict_allowlist_without_echoing_secret() -> None:
    projected = project_evaluation_identity(
        "variant-0",
        _run_identity(),
        corpus_id="corpus-v2",
        case_set_hash=_HASH_C,
    )
    assert projected.identity_map()["chunking.chunking_version"] == "structure-page-token-v1"

    with pytest.raises(ComparisonDomainError) as error:
        project_evaluation_identity(
            "variant-0",
            _run_identity(include_secret_key=True),
            corpus_id="corpus-v2",
            case_set_hash=_HASH_C,
        )
    assert str(error.value) == "comparison_identity_key_not_allowlisted"
    assert "sk-secret-value" not in str(error.value)

    direct = _projection(_plan(), 0).model_dump(mode="json")
    direct["identities"].append({"name": "provider.api-key", "value": "sk-direct-secret-value"})
    with pytest.raises(ValidationError) as direct_error:
        ComparisonIdentityProjection.model_validate(direct)
    assert "sk-direct-secret-value" not in str(direct_error.value)


def test_compatibility_and_plan_boundaries_hide_unsafe_values_but_allow_unicode_labels() -> None:
    secret = "sk-direct-secret-value"
    with pytest.raises(ValidationError) as issue_error:
        ComparisonCompatibilityIssue.model_validate(
            {
                "variant_id": "variant-0",
                "code": "controlled-identity-mismatch",
                "identity_name": "provider.generation",
                "expected": secret,
                "actual": "provider-a",
            }
        )
    assert secret not in str(issue_error.value)

    private_path = "D:\\private\\comparison.json"
    with pytest.raises(ValidationError) as controlled_error:
        ComparisonCompatibility.model_validate(
            {
                "compatible": True,
                "axis": ExperimentAxis.GENERATION_MODEL,
                "controlled_dimensions": ({"name": "prompt.generation", "value": private_path},),
                "issues": (),
            }
        )
    assert private_path not in str(controlled_error.value)

    base = _plan()
    bilingual = ExperimentPlan.create(
        **{
            **base.model_dump(mode="python", exclude={"content_hash"}),
            "display_name": "Generation model comparison / 生成模型对比",
        }
    )
    assert validate_comparison_plan_safe_values(bilingual) is bilingual

    unsafe = ExperimentPlan.create(
        **{
            **base.model_dump(mode="python", exclude={"content_hash"}),
            "display_name": secret,
        }
    )
    with pytest.raises(ValidationError) as plan_error:
        _suite(unsafe)
    assert "comparison_display_value_not_safe" in str(plan_error.value)
    assert secret not in str(plan_error.value)


@pytest.mark.parametrize(
    ("schema_version", "relative_path"),
    (("wrong-schema", "evaluation-report.json"), ("2.0.0", "wrong-report.json")),
)
def test_source_report_adapter_rejects_wrong_schema_or_path(
    schema_version: str,
    relative_path: str,
) -> None:
    content = b"{}"
    descriptor = ArtifactDescriptor(
        schema_version=schema_version,
        artifact_id="evaluation-report-json",
        format="json",
        media_type="application/json",
        relative_path=relative_path,
        sha256_digest=f"sha256:{hashlib.sha256(content).hexdigest()}",
        byte_size=len(content),
        created_at=_NOW,
    )
    suite = _suite()
    with pytest.raises(ComparisonDomainError, match="artifact_integrity_failed"):
        adapt_verified_evaluation_report(
            suite.candidates[0].reference,
            descriptor,
            content,
            comparison_id=suite.comparison_id,
            plan=suite.plan,
            run_identity=_run_identity(),
            corpus_id="corpus-v2",
            expected_case_ids=("case-1",),
        )


def _artifact(
    artifact_id: str,
    schema_version: str,
    media_type: str,
    relative_path: str,
    content: bytes,
) -> ArtifactDescriptor:
    return ArtifactDescriptor(
        schema_version=schema_version,
        artifact_id=artifact_id,
        format=relative_path.rsplit(".", 1)[-1],
        media_type=media_type,
        relative_path=relative_path,
        sha256_digest=f"sha256:{hashlib.sha256(content).hexdigest()}",
        byte_size=len(content),
        created_at=_NOW,
    )


def test_manifest_hash_contract_and_resolved_bytes_are_fail_closed() -> None:
    plan = _plan()
    payloads = {
        "comparison-plan-json": b"plan",
        "comparison-report-json": b"json",
        "comparison-report-html": b"html",
        "comparison-report-txt": b"text",
        "comparison-report-csv": b"csv",
        "comparison-candidate-variant-0": b"candidate-0",
        "comparison-candidate-variant-1": b"candidate-1",
    }
    contracts = {
        "comparison-plan-json": ("experiment-plan-v1", "application/json", "comparison-plan.json"),
        "comparison-report-json": (
            COMPARISON_RESULT_SCHEMA_VERSION,
            "application/json",
            "comparison-report.json",
        ),
        "comparison-report-html": (
            "comparison-report-html-v1",
            "text/html",
            "comparison-report.html",
        ),
        "comparison-report-txt": (
            "comparison-report-text-v1",
            "text/plain",
            "comparison-report.txt",
        ),
        "comparison-report-csv": (
            "comparison-report-csv-v1",
            "text/csv",
            "comparison-report.csv",
        ),
        "comparison-candidate-variant-0": (
            "comparison-candidate-evidence-v1",
            "application/json",
            "candidates/variant-0.json",
        ),
        "comparison-candidate-variant-1": (
            "comparison-candidate-evidence-v1",
            "application/json",
            "candidates/variant-1.json",
        ),
    }
    artifacts = tuple(
        _artifact(artifact_id, *contracts[artifact_id], content)
        for artifact_id, content in payloads.items()
    )
    manifest = ComparisonArtifactManifest.create(
        comparison_id="comparison-1",
        plan=plan,
        artifacts=artifacts,
        created_at=_NOW,
    )

    assert canonical_comparison_manifest(manifest).endswith(b"\n")
    resolved = resolve_comparison_artifact(
        manifest,
        "comparison-report-json",
        payloads["comparison-report-json"],
    )
    assert isinstance(resolved, ResolvedComparisonArtifact)
    with pytest.raises(ComparisonDomainError, match="integrity_failed"):
        resolve_comparison_artifact(manifest, "comparison-report-json", b"tampered")
    raw = manifest.model_dump(mode="json")
    raw["artifacts"][0]["relative_path"] = "wrong.json"
    with pytest.raises(ValidationError, match="artifact_contract_invalid"):
        ComparisonArtifactManifest.model_validate(raw)


def test_manifest_allows_only_available_candidate_artifact_subset() -> None:
    plan = _plan()
    payloads = {
        "comparison-plan-json": b"plan",
        "comparison-report-json": b"json",
        "comparison-report-html": b"html",
        "comparison-report-txt": b"text",
        "comparison-report-csv": b"csv",
        "comparison-candidate-variant-0": b"candidate-0",
    }
    contracts = {
        "comparison-plan-json": ("experiment-plan-v1", "application/json", "comparison-plan.json"),
        "comparison-report-json": (
            COMPARISON_RESULT_SCHEMA_VERSION,
            "application/json",
            "comparison-report.json",
        ),
        "comparison-report-html": (
            "comparison-report-html-v1",
            "text/html",
            "comparison-report.html",
        ),
        "comparison-report-txt": (
            "comparison-report-text-v1",
            "text/plain",
            "comparison-report.txt",
        ),
        "comparison-report-csv": (
            "comparison-report-csv-v1",
            "text/csv",
            "comparison-report.csv",
        ),
        "comparison-candidate-variant-0": (
            "comparison-candidate-evidence-v1",
            "application/json",
            "candidates/variant-0.json",
        ),
    }
    artifacts = tuple(
        _artifact(artifact_id, *contracts[artifact_id], content)
        for artifact_id, content in payloads.items()
    )

    manifest = ComparisonArtifactManifest.create(
        comparison_id="comparison-1",
        plan=plan,
        artifacts=artifacts,
        created_at=_NOW,
    )

    assert manifest.candidate_variant_ids == ("variant-0", "variant-1")
    assert {
        item.artifact_id
        for item in manifest.artifacts
        if item.artifact_id.startswith("comparison-candidate-")
    } == {"comparison-candidate-variant-0"}
    extra = _artifact(
        "comparison-candidate-foreign",
        "comparison-candidate-evidence-v1",
        "application/json",
        "candidates/foreign.json",
        b"foreign",
    )
    with pytest.raises(ValidationError, match="candidate_artifact_set_mismatch"):
        ComparisonArtifactManifest.create(
            comparison_id="comparison-1",
            plan=plan,
            artifacts=(*artifacts, extra),
            created_at=_NOW,
        )
