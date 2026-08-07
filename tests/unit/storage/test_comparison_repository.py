from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from rag_mvp.domain.evaluation import (
    EvaluationRun,
    EvidenceComparisonOperator,
    GateResult,
    GateStatus,
    MetricObservation,
    MetricObservationStatus,
    ModelAttemptStatus,
    ModelRole,
    ProviderAttemptEvidence,
    TokenUsage,
)
from rag_mvp.domain.retrieval import CacheOutcome, CachePolicy
from rag_mvp.evaluation.comparison import (
    ComparisonCandidateStatus,
    ComparisonCompatibility,
    ComparisonIdentityProjection,
    ComparisonLogicalAttempt,
    ComparisonLogicalAttemptStatus,
    ComparisonProviderAttempt,
    ComparisonRecommendationState,
    ComparisonResult,
    ComparisonSharedSetupEvidence,
    ComparisonSharedSetupStatus,
    ComparisonSuite,
    VerifiedCandidateReport,
    aggregate_comparison_result,
    build_comparison_candidate_evidence,
    create_comparison_suite,
    seal_comparison_candidate_evidence,
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
from rag_mvp.evaluation.json_report import canonical_json_value
from rag_mvp.evaluation.report_builder import case_ids_content_hash
from rag_mvp.storage.database import Database
from rag_mvp.storage.repositories import (
    ComparisonRepository,
    EvaluationRunRepository,
    RepositoryConflict,
    RepositoryError,
)

_NOW = datetime(2026, 8, 7, tzinfo=UTC)
_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64


@pytest.fixture
def database(tmp_path: Path) -> Database:
    value = Database(tmp_path / "metadata.sqlite3")
    value.initialize()
    return value


def _plan() -> ExperimentPlan:
    source = "https://pricing.example.test/comparison-v1"
    return ExperimentPlan.create(
        plan_id="generation-model-comparison-v1",
        display_name="Generation model comparison / 生成模型对比",
        axis=ExperimentAxis.GENERATION_MODEL,
        fixed_identities=ExperimentFixedIdentities(
            dataset_id="acceptance-v2",
            dataset_version="2.0.0",
            dataset_hash=_HASH_A,
            corpus_id="corpus-v2",
            corpus_version="2.0.0",
            corpus_hash=_HASH_B,
            case_set_hash=case_ids_content_hash(("case-1",)),
            case_count=1,
            controlled=(
                FixedIdentity(name="prompt.generation", value="prompt-v2"),
                FixedIdentity(name="provider.generation", value="provider-a"),
                FixedIdentity(name="provider.embedding", value="provider-a"),
                FixedIdentity(name="model.embedding", value="embedding-a"),
                FixedIdentity(name="retrieval.mode", value="hybrid"),
                FixedIdentity(name="cache.behavior", value="bypass"),
            ),
        ),
        variants=(
            ExperimentVariant(
                variant_id="model-a",
                display_name="Model A",
                axis_value="generation-a",
                configuration_id="semantic-config-a",
            ),
            ExperimentVariant(
                variant_id="model-b",
                display_name="Model B",
                axis_value="generation-b",
                configuration_id="semantic-config-b",
            ),
        ),
        baseline_variant_id="model-a",
        repeat_order_policy=RepeatOrderPolicy(
            repeats_per_case=1,
            order_policy=ExperimentOrderPolicy.SEEDED_INTERLEAVED,
            seed=7,
        ),
        cache_policy=CachePolicy.BYPASS,
        pricing=ExperimentPricingProvenance(
            pricing_version="comparison-pricing-v1",
            pricing_hash=_HASH_A,
            currency="USD",
            source_references=(source,),
            rate_card=(
                ExperimentPricingRate(
                    role=PricingRole.EMBEDDING,
                    provider="provider-a",
                    model="embedding-a",
                    input_per_million=Decimal("0.02"),
                    source_reference=source,
                ),
                ExperimentPricingRate(
                    role=PricingRole.GENERATION,
                    provider="provider-a",
                    model="generation-a",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                    source_reference=source,
                ),
                ExperimentPricingRate(
                    role=PricingRole.GENERATION,
                    provider="provider-a",
                    model="generation-b",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                    source_reference=source,
                ),
            ),
        ),
        maximum_provider_calls=20,
        maximum_cost=Decimal("1"),
        gate_profile=ExperimentGateProfile(
            profile_id="comparison-gates-v1",
            profile_version="1.0.0",
            profile_hash=_HASH_B,
            mandatory_gate_ids=("quality",),
        ),
        selection_policy=DeterministicSelectionPolicy(
            policy_id="comparison-selection-v1",
            policy_version="1.0.0",
            required_gate_ids=("quality",),
            tie_breakers=(
                SelectionCriterion(
                    metric="quality",
                    direction=SelectionDirection.MAXIMIZE,
                ),
            ),
            final_tie_break=FinalTieBreak.BASELINE_FIRST,
        ),
    )


def _suite(
    comparison_id: str = "comparison-1",
    run_prefix: str = "candidate",
) -> ComparisonSuite:
    plan = _plan()
    return create_comparison_suite(
        comparison_id,
        plan,
        {"model-a": f"{run_prefix}-run-a", "model-b": f"{run_prefix}-run-b"},
        created_at=_NOW,
    )


def _shared_setup(suite: ComparisonSuite) -> ComparisonSharedSetupEvidence:
    return ComparisonSharedSetupEvidence.create(
        comparison_id=suite.comparison_id,
        plan=suite.plan,
        status=ComparisonSharedSetupStatus.REUSED,
        attempts=(),
        recorded_at=_NOW,
    )


def _runs(suite: ComparisonSuite) -> tuple[EvaluationRun, ...]:
    return tuple(
        EvaluationRun(
            run_id=item.reference.evaluation_run_id,
            dataset_id=suite.plan.fixed_identities.dataset_id,
            dataset_version=suite.plan.fixed_identities.dataset_version,
            dataset_hash=suite.plan.fixed_identities.dataset_hash,
            corpus_version=suite.plan.fixed_identities.corpus_version,
            configuration_id=item.reference.configuration_id,
            code_revision="revision-v1",
            scorer_versions={"quality": "quality-v1"},
            cache_policy=suite.plan.cache_policy,
            total_cases=1,
            created_at=_NOW,
            updated_at=_NOW,
        )
        for item in suite.candidates
    )


def _candidate_report(
    suite: ComparisonSuite,
    variant_index: int,
    quality_value: float,
) -> VerifiedCandidateReport:
    reference = suite.candidates[variant_index].reference
    variant = suite.plan.variants[variant_index]
    identities = (
        *suite.plan.fixed_identities.controlled,
        FixedIdentity(name=suite.plan.axis.identity_name, value=variant.axis_value),
    )
    projection = ComparisonIdentityProjection(
        variant_id=variant.variant_id,
        configuration_id=variant.configuration_id,
        dataset_id=suite.plan.fixed_identities.dataset_id,
        dataset_version=suite.plan.fixed_identities.dataset_version,
        dataset_hash=suite.plan.fixed_identities.dataset_hash,
        corpus_id=suite.plan.fixed_identities.corpus_id,
        corpus_version=suite.plan.fixed_identities.corpus_version,
        corpus_hash=suite.plan.fixed_identities.corpus_hash,
        case_set_hash=suite.plan.fixed_identities.case_set_hash,
        identities=identities,
    )
    embedding_rate = next(
        item for item in suite.plan.pricing.rate_card if item.role is PricingRole.EMBEDDING
    )
    generation_rate = next(
        item
        for item in suite.plan.pricing.rate_card
        if item.role is PricingRole.GENERATION and item.model == variant.axis_value
    )
    embedding = ComparisonProviderAttempt.create(
        attempt_reference=f"{variant.variant_id}-embedding",
        logical_attempt_id="logical-1",
        evaluation_run_id=reference.evaluation_run_id,
        evidence=ProviderAttemptEvidence(
            operation_id="qa-retrieval",
            role=ModelRole.EMBEDDING,
            provider="provider-a",
            model="embedding-a",
            status=ModelAttemptStatus.SUCCEEDED,
            latency_ms=5.0,
            usage=TokenUsage(input_tokens=10),
        ),
        latency_ms=5.0,
        pricing_version=suite.plan.pricing.pricing_version,
        pricing_hash=suite.plan.pricing.pricing_hash,
        currency=suite.plan.pricing.currency,
        input_per_million=embedding_rate.input_per_million,
        output_per_million=embedding_rate.output_per_million,
        pricing_source_reference=embedding_rate.source_reference,
    )
    generation = ComparisonProviderAttempt.create(
        attempt_reference=f"{variant.variant_id}-generation",
        logical_attempt_id="logical-1",
        evaluation_run_id=reference.evaluation_run_id,
        evidence=ProviderAttemptEvidence(
            operation_id="qa-generation",
            role=ModelRole.GENERATION,
            provider="provider-a",
            model=variant.axis_value,
            status=ModelAttemptStatus.SUCCEEDED,
            latency_ms=10.0,
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        ),
        latency_ms=10.0,
        pricing_version=suite.plan.pricing.pricing_version,
        pricing_hash=suite.plan.pricing.pricing_hash,
        currency=suite.plan.pricing.currency,
        input_per_million=generation_rate.input_per_million,
        output_per_million=generation_rate.output_per_million,
        pricing_source_reference=generation_rate.source_reference,
    )
    assert embedding.total_cost is not None
    assert generation.total_cost is not None
    total_cost = embedding.total_cost + generation.total_cost
    logical = ComparisonLogicalAttempt(
        attempt_id="logical-1",
        case_id="case-1",
        repeat_index=0,
        order_index=0,
        status=ComparisonLogicalAttemptStatus.SUCCEEDED,
        latency_ms=100.0,
        terminal_kind="answer",
        cache_policy=CachePolicy.BYPASS,
        cache_outcome=CacheOutcome.BYPASS,
        index_revision_id="index-v1",
        retrieved_chunk_ids=("chunk-a",),
        context_chunk_ids=("chunk-a",),
        retrieval_evidence_digest=_HASH_B,
        provider_attempt_references=(embedding.attempt_reference, generation.attempt_reference),
        provider_failed_attempt_count=0,
        input_tokens=20,
        output_tokens=5,
        estimated_cost=total_cost,
        currency="USD",
        completed_at=_NOW,
    )
    quality = MetricObservation(
        metric_id="quality",
        unit="ratio",
        value=quality_value,
        numerator=quality_value,
        denominator=1,
        eligible=True,
        threshold=0.8,
        operator=EvidenceComparisonOperator.GREATER_THAN_OR_EQUAL,
        scorer_version="quality-v1",
        status=MetricObservationStatus.PASSED,
    )
    gate = GateResult(
        gate_id="quality",
        profile_version="1.0.0",
        status=GateStatus.PASSED,
        valid=True,
        passed=True,
        case_executions_complete=True,
        observations=(quality,),
    )
    evidence = build_comparison_candidate_evidence(
        comparison_id=suite.comparison_id,
        plan=suite.plan,
        reference=reference,
        identity_projection=projection,
        expected_case_ids=("case-1",),
        logical_attempts=(logical,),
        provider_attempts=(embedding, generation),
        quality_metrics=(quality,),
        gates=(gate,),
        generated_at=_NOW,
    )
    return seal_comparison_candidate_evidence(reference, evidence)


def _complete_recommended_result(
    repository: ComparisonRepository,
    suite: ComparisonSuite,
    *,
    completed_offset: int = 0,
) -> ComparisonResult:
    reports = (_candidate_report(suite, 0, 0.9), _candidate_report(suite, 1, 1.0))
    current = suite
    for index, report in enumerate(reports):
        assert report.evidence.total_cost is not None
        current = current.transition_candidate(
            report.reference.variant_id,
            status=ComparisonCandidateStatus.RUNNING,
            completed_cases=0,
            failed_cases=0,
            provider_calls=0,
            recorded_at=_NOW + timedelta(seconds=completed_offset + index * 2 + 1),
        )
        repository.append(current)
        current = current.transition_candidate(
            report.reference.variant_id,
            status=ComparisonCandidateStatus.COMPLETED,
            completed_cases=1,
            failed_cases=0,
            provider_calls=report.evidence.provider_call_count,
            incurred_cost=report.evidence.total_cost,
            currency=report.evidence.currency,
            recorded_at=_NOW + timedelta(seconds=completed_offset + index * 2 + 2),
        )
        repository.append(current)
    compatibility = ComparisonCompatibility(
        compatible=True,
        axis=current.plan.axis,
        controlled_dimensions=current.plan.fixed_identities.controlled,
    )
    shared_setup = _shared_setup(current)
    repository.save_shared_setup(shared_setup)
    return aggregate_comparison_result(
        current,
        compatibility,
        {item.reference.variant_id: item for item in reports},
        shared_setup=shared_setup,
        completed_at=_NOW + timedelta(seconds=completed_offset + 5),
    )


def test_atomic_create_append_and_reopen_preserve_exact_histories(database: Database) -> None:
    repository = ComparisonRepository(database)
    suite = _suite()
    repository.create(suite, _runs(suite))

    running = suite.transition_candidate(
        "model-a",
        status=ComparisonCandidateStatus.RUNNING,
        completed_cases=0,
        failed_cases=0,
        provider_calls=0,
        recorded_at=_NOW + timedelta(seconds=1),
    )
    repository.append(running)

    reopened = ComparisonRepository(Database(database.path))
    assert reopened.get(suite.comparison_id) == running
    assert reopened.list() == [running]
    assert EvaluationRunRepository(database).get("candidate-run-a") is not None
    with pytest.raises(RepositoryConflict):
        reopened.append(running)


def test_failed_shared_setup_spend_is_restart_safe_without_a_result(database: Database) -> None:
    repository = ComparisonRepository(database)
    suite = _suite()
    repository.create(suite, _runs(suite))
    failed = ComparisonSharedSetupEvidence.create(
        comparison_id=suite.comparison_id,
        plan=suite.plan,
        status=ComparisonSharedSetupStatus.FAILED,
        attempts=(),
        safe_error_code="setup-validation-failed",
        provider_calls_complete=False,
        recorded_at=_NOW + timedelta(seconds=1),
    )

    repository.save_shared_setup(failed)

    reopened = ComparisonRepository(Database(database.path))
    assert reopened.get_shared_setup(suite.comparison_id) == failed
    assert reopened.get_shared_setup(suite.comparison_id).provider_calls_complete is False
    assert reopened.get_shared_setup(suite.comparison_id).total_cost is None
    assert reopened.get_result(suite.comparison_id) is None
    with pytest.raises(RepositoryConflict):
        reopened.save_shared_setup(failed)

    with database.transaction() as connection:
        connection.execute(
            """
            UPDATE comparison_shared_setup_evidence
            SET provider_call_count = 1
            WHERE comparison_id = ?
            """,
            (suite.comparison_id,),
        )
    with pytest.raises(RepositoryError, match="setup integrity"):
        reopened.get_shared_setup(suite.comparison_id)


def test_v4_shared_setup_reopens_when_a_value_matches_the_v5_field_name(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "legacy-v4.sqlite3")
    database.initialize(target_version=4)
    suite = _suite(comparison_id="provider_calls_complete")
    ComparisonRepository(database).create(suite, _runs(suite))
    failed = ComparisonSharedSetupEvidence.create(
        comparison_id=suite.comparison_id,
        plan=suite.plan,
        status=ComparisonSharedSetupStatus.FAILED,
        attempts=(),
        safe_error_code="setup-validation-failed",
        recorded_at=_NOW + timedelta(seconds=1),
    )
    legacy_payload = canonical_json_value(
        failed.model_dump(mode="json", exclude={"provider_calls_complete"})
    )
    legacy_hash = f"sha256:{hashlib.sha256(legacy_payload.encode()).hexdigest()}"
    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO comparison_shared_setup_evidence(
                comparison_id, plan_content_hash, status, provider_call_count,
                known_partial_cost, total_cost, currency, recorded_at,
                evidence_content_hash, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                failed.comparison_id,
                failed.plan_content_hash,
                failed.status.value,
                failed.provider_call_count,
                str(failed.known_partial_cost),
                str(failed.total_cost),
                failed.currency,
                failed.recorded_at.isoformat(),
                legacy_hash,
                legacy_payload,
            ),
        )

    database.initialize()

    assert ComparisonRepository(database).get_shared_setup(suite.comparison_id) == failed


def test_legacy_result_hash_reopens_and_new_selection_uses_stored_hash(
    database: Database,
) -> None:
    repository = ComparisonRepository(database)
    suite = _suite()
    repository.create(suite, _runs(suite))
    result = _complete_recommended_result(repository, suite)
    repository.save_result(result)
    legacy_result = result.model_dump(mode="json")
    shared_setup = legacy_result["shared_setup"]
    assert isinstance(shared_setup, dict)
    shared_setup.pop("provider_calls_complete")
    legacy_payload = canonical_json_value(legacy_result)
    legacy_hash = f"sha256:{hashlib.sha256(legacy_payload.encode()).hexdigest()}"
    with database.transaction() as connection:
        connection.execute(
            """
            UPDATE comparison_results
            SET result_content_hash = ?, payload_json = ?
            WHERE comparison_id = ?
            """,
            (legacy_hash, legacy_payload, suite.comparison_id),
        )

    reopened = ComparisonRepository(Database(database.path))
    assert reopened.get_result(suite.comparison_id) == result

    selection = reopened.save_selection(
        result,
        created_at=result.completed_at + timedelta(seconds=1),
    )
    assert selection.result_content_hash == legacy_hash
    assert reopened.get_selection(ExperimentAxis.GENERATION_MODEL) == selection


def test_atomic_create_rolls_back_every_new_candidate_run_on_conflict(database: Database) -> None:
    suite = _suite()
    runs = _runs(suite)
    EvaluationRunRepository(database).create(runs[1])

    with pytest.raises(RepositoryConflict):
        ComparisonRepository(database).create(suite, runs)

    assert EvaluationRunRepository(database).get(runs[0].run_id) is None
    with database.connection() as connection:
        plan_count = int(connection.execute("SELECT COUNT(*) FROM comparison_plans").fetchone()[0])
    assert plan_count == 0


def test_read_rejects_tampered_run_and_candidate_history_columns(database: Database) -> None:
    repository = ComparisonRepository(database)
    suite = _suite()
    repository.create(suite, _runs(suite))
    with database.transaction() as connection:
        connection.execute(
            "UPDATE comparison_runs SET status = 'completed' WHERE comparison_id = ?",
            (suite.comparison_id,),
        )
    with pytest.raises(RepositoryError, match="run history integrity"):
        repository.get(suite.comparison_id)

    other_database = Database(database.path.parent / "candidate.sqlite3")
    other_database.initialize()
    other = ComparisonRepository(other_database)
    other.create(suite, _runs(suite))
    with other_database.transaction() as connection:
        connection.execute(
            """
            UPDATE comparison_candidates SET recorded_at = ?
            WHERE comparison_id = ? AND variant_id = 'model-a'
            """,
            ((_NOW + timedelta(days=1)).isoformat(), suite.comparison_id),
        )
    with pytest.raises(RepositoryError, match="candidate history integrity"):
        other.get(suite.comparison_id)

    identity_database = Database(database.path.parent / "identity.sqlite3")
    identity_database.initialize()
    identity_repository = ComparisonRepository(identity_database)
    identity_runs = _runs(suite)
    identity_repository.create(suite, identity_runs)
    tampered_run = identity_runs[0].model_copy(update={"configuration_id": "tampered-config"})
    with identity_database.transaction() as connection:
        connection.execute(
            "UPDATE evaluation_runs SET payload_json = ? WHERE run_id = ?",
            (tampered_run.model_dump_json(), tampered_run.run_id),
        )
    with pytest.raises(RepositoryError, match="evaluation identity integrity"):
        identity_repository.get(suite.comparison_id)


def test_result_is_immutable_and_no_recommendation_cannot_create_selection(
    database: Database,
) -> None:
    repository = ComparisonRepository(database)
    suite = _suite()
    repository.create(suite, _runs(suite))
    first = suite.transition_candidate(
        "model-a",
        status=ComparisonCandidateStatus.FAILED,
        completed_cases=0,
        failed_cases=1,
        provider_calls=0,
        safe_error_code="candidate-failed",
        recorded_at=_NOW + timedelta(seconds=1),
    )
    repository.append(first)
    completed = first.transition_candidate(
        "model-b",
        status=ComparisonCandidateStatus.FAILED,
        completed_cases=0,
        failed_cases=1,
        provider_calls=0,
        safe_error_code="candidate-failed",
        recorded_at=_NOW + timedelta(seconds=2),
    )
    repository.append(completed)
    compatibility = ComparisonCompatibility(
        compatible=True,
        axis=completed.plan.axis,
        controlled_dimensions=completed.plan.fixed_identities.controlled,
    )
    result = aggregate_comparison_result(
        completed,
        compatibility,
        {"model-a": None, "model-b": None},
        shared_setup=_shared_setup(completed),
        completed_at=_NOW + timedelta(seconds=3),
    )
    assert result.recommendation.state is ComparisonRecommendationState.NO_RECOMMENDATION

    repository.save_shared_setup(result.shared_setup)
    repository.save_result(result)
    assert ComparisonRepository(Database(database.path)).get_result(result.comparison_id) == result
    with pytest.raises(RepositoryConflict):
        repository.save_result(result)
    with pytest.raises(RepositoryConflict, match="no deterministic selection"):
        repository.save_selection(result)


def test_selection_history_is_axis_indexed_restart_safe_and_fully_verified(
    database: Database,
) -> None:
    repository = ComparisonRepository(database)
    first_suite = _suite()
    repository.create(first_suite, _runs(first_suite))
    first_result = _complete_recommended_result(repository, first_suite)
    repository.save_result(first_result)
    first_selection = repository.save_selection(
        first_result, created_at=_NOW + timedelta(seconds=6)
    )
    assert first_selection.selected_variant_id == "model-b"

    reopened = ComparisonRepository(Database(database.path))
    assert reopened.get_selection(ExperimentAxis.GENERATION_MODEL) == first_selection

    second_suite = _suite("comparison-2", "candidate-2")
    reopened.create(second_suite, _runs(second_suite))
    second_result = _complete_recommended_result(reopened, second_suite, completed_offset=10)
    reopened.save_result(second_result)
    second_selection = reopened.save_selection(
        second_result,
        created_at=_NOW + timedelta(seconds=16),
    )
    assert second_selection.selection_sequence == first_selection.selection_sequence + 1
    assert reopened.get_selection(ExperimentAxis.GENERATION_MODEL) == second_selection
    assert reopened.list_selections(ExperimentAxis.GENERATION_MODEL) == [
        first_selection,
        second_selection,
    ]

    cache_forgery = second_result.model_copy(update={"axis": ExperimentAxis.CACHE_BEHAVIOR})
    with pytest.raises(RepositoryConflict, match="cannot become an upstream selection"):
        reopened.save_selection(cache_forgery)

    with database.transaction() as connection:
        connection.execute(
            """
            UPDATE comparison_selections SET selected_configuration_id = 'tampered-config'
            WHERE comparison_id = ?
            """,
            (second_result.comparison_id,),
        )
    with pytest.raises(RepositoryError, match="selection integrity"):
        reopened.get_selection(ExperimentAxis.GENERATION_MODEL)


def test_result_survives_finalization_failure_but_selection_is_not_exposed(
    database: Database,
) -> None:
    repository = ComparisonRepository(database)
    suite = _suite()
    repository.create(suite, _runs(suite))
    result = _complete_recommended_result(repository, suite)
    repository.save_result(result)
    repository.save_selection(result, created_at=_NOW + timedelta(seconds=6))
    completed = repository.get(suite.comparison_id)
    assert completed is not None and completed.status.value == "completed"
    failed = completed.fail(
        "publication-failed",
        recorded_at=_NOW + timedelta(seconds=7),
    )
    repository.append(failed)

    reopened = ComparisonRepository(Database(database.path))
    assert reopened.get_result(result.comparison_id) == result
    with pytest.raises(RepositoryConflict, match="completed latest suite"):
        reopened.save_selection(result, created_at=_NOW + timedelta(seconds=8))
    with pytest.raises(RepositoryError, match="selection integrity"):
        reopened.get_selection(ExperimentAxis.GENERATION_MODEL)
