from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from test_comparison import (
    _candidate_evidence,
    _compatibility,
    _plan,
    _provider,
    _setup_attempt,
    _shared_setup,
    _suite,
    _terminal_suite,
    _verified_reports,
)

from rag_mvp.config.settings import Settings
from rag_mvp.domain.evaluation import (
    EvaluationRun,
    EvaluationRunStatus,
    ModelAttemptStatus,
    UnavailableValue,
)
from rag_mvp.evaluation.application import EvaluationApplicationService
from rag_mvp.evaluation.comparison import (
    ComparisonCandidateStatus,
    ComparisonLogicalAttemptStatus,
    ComparisonSharedSetupEvidence,
    ComparisonSharedSetupStatus,
    aggregate_comparison_result,
    seal_comparison_candidate_evidence,
)
from rag_mvp.evaluation.comparison_application import (
    ComparisonPlanCatalogEntry,
    ComparisonRunEntry,
    ComparisonSummary,
)
from rag_mvp.evaluation.experiment import ExperimentAxis
from rag_mvp.evaluation.plan import EvaluationDatasetRegistry
from rag_mvp.storage.database import Database
from rag_mvp.storage.repositories import ComparisonRepository, EvaluationRunRepository
from rag_mvp.ui.callbacks import WorkbenchCallbacks
from rag_mvp.ui.models import BrowserSessionState
from rag_mvp.ui.services import WorkbenchServices


def test_plan_and_run_dtos_are_path_free_and_denominator_bearing() -> None:
    plan = _plan()
    suite = _suite(plan)

    catalog = ComparisonPlanCatalogEntry.from_plan(
        plan,
        launchable=True,
    )
    run = ComparisonRunEntry.from_suite(suite)

    assert catalog.maximum_logical_calls == (
        catalog.planned_case_count * catalog.repeats_per_case * len(catalog.variants)
    )
    assert catalog.cost_estimate_status == "unavailable"
    assert catalog.cost_estimate is None
    assert catalog.cost_cap == plan.maximum_cost
    assert run.remaining_candidates == len(plan.variants)
    assert "path" not in catalog.model_dump_json().casefold()


def test_missing_result_is_never_available_or_passing() -> None:
    suite = _suite()

    summary = ComparisonSummary.from_evidence(suite, None)

    assert summary.evidence_status == "unavailable"
    assert summary.gate_status == "unavailable"
    assert summary.compatibility_state == "unavailable"
    assert all(not item.metrics for item in summary.candidates)
    assert summary.shared_setup.status == "unavailable"
    assert summary.shared_setup.provider_calls_complete is False
    assert summary.shared_setup.cost_complete is False
    assert summary.shared_setup.unknown_reasons == ("setup-evidence-not-recorded",)
    assert isinstance(summary.provider_call_count, UnavailableValue)
    payload = summary.model_dump_json()
    assert "setup_id" not in payload
    assert "request_id" not in payload
    assert "attempt_reference" not in payload
    assert "index_revision_id" not in payload


def test_exact_zero_candidate_work_keeps_completed_setup_total_exact() -> None:
    plan = _plan()
    suite = _suite(plan)
    setup = _shared_setup(
        plan,
        status=ComparisonSharedSetupStatus.COMPLETED,
        attempts=(
            _setup_attempt(
                plan,
                attempt_number=1,
                status=ModelAttemptStatus.SUCCEEDED,
                input_tokens=752,
            ),
        ),
    )

    run = ComparisonRunEntry.from_suite(suite, setup)
    summary = ComparisonSummary.from_evidence(suite, None, setup)

    assert run.provider_calls == 1
    assert run.known_partial_cost == Decimal("0.00001504")
    assert run.incurred_cost == Decimal("0.00001504")
    assert run.cost_complete is True
    assert run.currency == "USD"
    assert all(candidate.total_cost == 0 for candidate in summary.candidates)
    assert all(candidate.cost_complete for candidate in summary.candidates)
    assert all(candidate.currency == "USD" for candidate in summary.candidates)
    assert summary.known_partial_cost == Decimal("0.00001504")
    assert summary.total_cost == Decimal("0.00001504")
    assert summary.cost_complete is True
    assert summary.cost_unknown_reasons == ()


def test_failed_unknown_setup_projects_unavailable_totals_without_internal_ids() -> None:
    suite = _suite()
    setup = ComparisonSharedSetupEvidence.create(
        comparison_id=suite.comparison_id,
        plan=suite.plan,
        status=ComparisonSharedSetupStatus.FAILED,
        attempts=(),
        safe_error_code="comparison-shared-setup-ledger-mismatch",
        provider_calls_complete=False,
    )

    summary = ComparisonSummary.from_evidence(suite, None, setup)

    assert summary.shared_setup.status == "failed"
    assert summary.shared_setup.provider_calls_complete is False
    assert isinstance(summary.shared_setup.provider_call_count, UnavailableValue)
    assert summary.shared_setup.known_partial_cost == 0
    assert isinstance(summary.shared_setup.total_cost, UnavailableValue)
    assert isinstance(summary.provider_call_count, UnavailableValue)
    assert summary.shared_setup.unknown_reasons == ("setup-ledger-integrity-unavailable",)
    payload = summary.model_dump_json()
    assert "setup_id" not in payload
    assert "request_id" not in payload
    assert "attempt_reference" not in payload
    assert "index_revision_id" not in payload


def test_cache_result_metrics_are_exposed_at_summary_and_warm_candidate() -> None:
    plan = _plan(ExperimentAxis.CACHE_BEHAVIOR)
    evidence = _verified_reports(plan, quality_values=(1.0, 1.0))
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

    summary = ComparisonSummary.from_evidence(suite, result)

    metric_ids = {item.metric_id for item in summary.comparison_metrics}
    assert {
        "comparison-cache-hit-rate",
        "comparison-cache-retrieval-provider-call-reduction",
        "comparison-cache-latency-delta",
        "comparison-cache-retrieval-equivalence-rate",
    }.issubset(metric_ids)
    warm = next(item for item in summary.candidates if item.axis_value == "warm")
    warm_metric_ids = {item.metric_id for item in warm.metrics}
    assert metric_ids.issubset(warm_metric_ids)
    hit_rate = next(
        item for item in summary.comparison_metrics if item.metric_id == "comparison-cache-hit-rate"
    )
    assert hit_rate.value == 1.0
    assert hit_rate.denominator == 1
    assert summary.evidence_status == "available"
    assert summary.gate_status == "passed"
    assert summary.shared_setup.status == "reused"
    assert summary.shared_setup.provider_call_count == 0
    assert summary.shared_setup.known_partial_cost == 0
    assert summary.shared_setup.total_cost == 0
    assert summary.shared_setup.currency == "USD"
    assert summary.shared_setup.provider_calls_complete is True
    assert summary.shared_setup.cost_complete is True
    assert summary.provider_call_count == result.provider_call_count
    assert summary.total_cost == result.total_cost
    assert summary.currency == result.currency


def test_incomplete_candidate_cost_exposes_lower_bound_without_overriding_gate() -> None:
    plan = _plan()
    evidence = list(_verified_reports(plan, quality_values=(1.0, 1.0)))
    evidence[0] = _candidate_evidence(
        plan,
        (_provider(plan, reference="provider-incomplete", output_tokens=None),),
        variant_index=0,
        quality_value=1.0,
    )
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

    summary = ComparisonSummary.from_evidence(suite, result)

    assert result.total_cost is None
    assert summary.known_partial_cost == result.known_partial_cost
    assert summary.known_partial_cost > Decimal(0)
    assert isinstance(summary.total_cost, UnavailableValue)
    assert summary.cost_complete is False
    assert summary.cost_unknown_reasons == ("output-usage-unknown",)
    assert summary.gate_status == "passed"
    assert summary.recommendation.state == "recommended"
    incomplete = summary.candidates[0]
    assert incomplete.known_partial_cost == evidence_tuple[0].known_partial_cost
    assert incomplete.total_cost is None
    assert incomplete.cost_complete is False


@pytest.mark.asyncio
async def test_persisted_cache_result_renders_after_fresh_service_restart(
    tmp_path: Path,
) -> None:
    plan = _plan(ExperimentAxis.CACHE_BEHAVIOR)
    evidence = _verified_reports(plan, quality_values=(1.0, 1.0))
    suite = _suite(plan)
    database_path = tmp_path / "metadata.sqlite3"
    database = Database(database_path)
    database.initialize()
    comparison_repository = ComparisonRepository(database)
    evaluation_repository = EvaluationRunRepository(database)
    evaluation_runs = tuple(
        EvaluationRun(
            run_id=history.reference.evaluation_run_id,
            dataset_id=plan.fixed_identities.dataset_id,
            dataset_version=plan.fixed_identities.dataset_version,
            dataset_hash=plan.fixed_identities.dataset_hash,
            corpus_version=plan.fixed_identities.corpus_version,
            configuration_id=history.reference.configuration_id,
            code_revision="comparison-restart-test",
            scorer_versions={"quality": "1.0.0"},
            cache_policy=plan.cache_policy.value,
            total_cases=(
                plan.fixed_identities.case_count * plan.repeat_order_policy.repeats_per_case
            ),
            created_at=suite.created_at,
            updated_at=suite.created_at,
        )
        for history in suite.candidates
    )
    comparison_repository.create(suite, evaluation_runs)

    timestamp = suite.created_at
    initial_histories = suite.candidates
    for history, candidate_evidence in zip(initial_histories, evidence, strict=True):
        timestamp += timedelta(seconds=1)
        suite = suite.transition_candidate(
            history.reference.variant_id,
            status=ComparisonCandidateStatus.RUNNING,
            completed_cases=0,
            failed_cases=0,
            provider_calls=0,
            recorded_at=timestamp,
        )
        comparison_repository.append(suite)
        timestamp += timedelta(seconds=1)
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
            currency=candidate_evidence.currency,
            recorded_at=timestamp,
        )
        comparison_repository.append(suite)

    sealed = {
        item.variant_id: seal_comparison_candidate_evidence(
            suite.candidates[index].reference,
            evidence[index],
        )
        for index, item in enumerate(plan.variants)
    }
    result = aggregate_comparison_result(
        suite,
        _compatibility(plan),
        sealed,
        shared_setup=_shared_setup(plan),
    )
    comparison_repository.save_shared_setup(result.shared_setup)
    comparison_repository.save_result(result)
    for run in evaluation_runs:
        evaluation_repository.update(
            EvaluationRun.model_validate(
                {
                    **run.model_dump(),
                    "status": EvaluationRunStatus.COMPLETED,
                    "completed_cases": run.total_cases,
                    "updated_at": suite.updated_at,
                }
            )
        )

    class RestartCatalog:
        def list(self) -> tuple[ComparisonPlanCatalogEntry, ...]:
            return (ComparisonPlanCatalogEntry.from_plan(plan, launchable=True),)

        def prepare(self, comparison_id: str, plan_id: str) -> None:
            del comparison_id, plan_id
            raise AssertionError("read-only restart must not prepare provider work")

    class NoProviderExecutor:
        calls = 0

        async def execute(self, *args: object) -> None:
            del args
            self.calls += 1
            raise AssertionError("read-only restart must not execute provider work")

    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    no_provider = NoProviderExecutor()
    restarted_database = Database(database_path)
    restarted_database.initialize()
    service = EvaluationApplicationService(
        registry=EvaluationDatasetRegistry(dataset_root),
        settings=Settings(
            _env_file=None,
            data_root=tmp_path / "online",
            evaluation_dataset_root=dataset_root,
            workbench_enabled=False,
        ),
        repository=EvaluationRunRepository(restarted_database),
        run_artifacts_root=tmp_path / "evaluation-runs",
        executor=no_provider,
        comparison_catalog=RestartCatalog(),  # type: ignore[arg-type]
        comparison_repository=ComparisonRepository(restarted_database),
        comparison_executor=no_provider,
    )

    await service.startup()
    restarted_summary = service.comparison_summary(suite.comparison_id)
    restarted_entry = service.get_comparison(suite.comparison_id)
    restarted_list = service.list_comparisons()
    assert restarted_summary is not None
    assert restarted_entry is not None
    assert restarted_list == (restarted_entry,)
    assert restarted_entry.provider_calls == result.provider_call_count
    assert restarted_entry.known_partial_cost == result.known_partial_cost
    assert restarted_entry.incurred_cost == result.total_cost
    assert restarted_entry.cost_complete is True
    assert restarted_summary.shared_setup.status == "reused"
    assert restarted_summary.shared_setup.provider_call_count == 0
    assert restarted_summary.provider_call_count == result.provider_call_count
    assert restarted_summary.total_cost == result.total_cost
    assert restarted_summary.currency == result.currency
    summary_json = restarted_summary.model_dump_json()
    assert "setup_id" not in summary_json
    assert "request_id" not in summary_json
    assert "attempt_reference" not in summary_json
    assert "index_revision_id" not in summary_json
    rendered = WorkbenchCallbacks(WorkbenchServices(evaluations=service)).refresh_comparisons(
        BrowserSessionState.create()
    )

    metric_ids = {row[0] for row in rendered.comparison_metric_rows}
    setup_rows = {(row[0], row[1]): row[2] for row in rendered.shared_setup_rows}
    assert {
        "comparison-cache-hit-rate",
        "comparison-cache-retrieval-provider-call-reduction",
        "comparison-cache-latency-delta",
        "comparison-cache-retrieval-equivalence-rate",
    }.issubset(metric_ids)
    assert setup_rows[("shared-setup", "status")] == "reused"
    assert setup_rows[("shared-setup", "provider-call-count")] == "0"
    assert setup_rows[("inclusive-comparison", "provider-call-count")] == str(
        result.provider_call_count
    )
    assert "CONFIRMED" in rendered.cache_conclusion_markdown
    assert plan.fixed_identities.corpus_version in rendered.cache_conclusion_markdown
    assert rendered.history_rows[0][20:22] == ("available", "passed")
    assert no_provider.calls == 0
    assert str(tmp_path) not in repr(rendered)
    assert "file:" not in rendered.artifact_links_markdown
