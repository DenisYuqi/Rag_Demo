"""Production registered-comparison catalog and normal-run coordinator."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol, cast

from rag_mvp.api.qa import QARuntimeServices
from rag_mvp.config.settings import Settings
from rag_mvp.domain import (
    EvaluationRun,
    EvaluationRunStatus,
    IndexRevision,
    ModelAttempt,
    ModelAttemptStatus,
    ProviderAttemptEvidence,
    TokenUsage,
    UnavailableValue,
)
from rag_mvp.domain._base import utc_now
from rag_mvp.evaluation.application import EvaluationRunStore
from rag_mvp.evaluation.comparison import (
    ComparisonCandidateStatus,
    ComparisonSharedSetupAttempt,
    ComparisonSharedSetupEvidence,
    ComparisonSharedSetupStatus,
    ComparisonStatus,
    ComparisonSuite,
    VerifiedCandidateReport,
    aggregate_comparison_result,
    comparison_shared_setup_id,
    create_comparison_suite,
    seal_comparison_candidate_evidence,
    validate_comparison_compatibility,
)
from rag_mvp.evaluation.comparison_application import (
    ComparisonPlanCatalogEntry,
    ComparisonPlanVariantEntry,
    PreparedComparisonLaunch,
)
from rag_mvp.evaluation.comparison_artifacts import ComparisonArtifactCatalog
from rag_mvp.evaluation.comparison_evidence import (
    PersistedProviderLedgerSummary,
    build_persisted_candidate_evidence,
    summarize_persisted_provider_attempts,
)
from rag_mvp.evaluation.comparison_plans import (
    ComparisonPlanCatalogContext,
    ComparisonPlanMaterializationContext,
    MaterializedComparisonCandidate,
    MaterializedComparisonPlan,
    RegisteredComparisonCandidateSummary,
    RegisteredComparisonPlanError,
    RegisteredComparisonPlanRegistry,
    RegisteredComparisonPlanSummary,
    UpstreamComparisonSelection,
    UpstreamSelections,
)
from rag_mvp.evaluation.corpus import (
    EvaluationCorpusInstaller,
    InstalledEvaluationCorpus,
)
from rag_mvp.evaluation.dataset import EvaluationDataset
from rag_mvp.evaluation.environment import EvaluationIndexReuseKey
from rag_mvp.evaluation.experiment import ExperimentAxis, ExperimentPlan
from rag_mvp.evaluation.plan import (
    EvaluationDatasetRegistry,
    EvaluationPlanError,
    evaluation_scorer_versions,
)
from rag_mvp.evaluation.runner import (
    EvaluationCaseExecutor,
    EvaluationCaseInput,
    EvaluationRunner,
    PersistedCaseResult,
    ProductionQAExecutor,
)
from rag_mvp.evaluation.work_budget import ProviderWorkEstimate
from rag_mvp.ingestion.service import IngestionService
from rag_mvp.performance.admission import QAAdmissionController
from rag_mvp.performance.deadlines import QALatencyBudgets
from rag_mvp.providers.models import ModelAttempt as ProviderModelAttempt
from rag_mvp.providers.models import ProviderErrorCategory
from rag_mvp.providers.persistence import unbound_evaluation_attempt_context
from rag_mvp.providers.resilience import capture_provider_attempts
from rag_mvp.safety.redactor import Redactor
from rag_mvp.storage.layout import DataLayout
from rag_mvp.storage.repositories import (
    ComparisonRepository,
    ProviderUsageRepository,
    RuntimeRepositories,
)

_DEFAULT_DATASET_ID = "original-pdf-acceptance"
_DEFAULT_DATASET_VERSION = "2.0.0"
_SAFE_OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$")
_SAFE_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


class ComparisonProductionError(RuntimeError):
    """Stable fail-closed production comparison error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _ComparisonSharedSetupExecutionError(ComparisonProductionError):
    def __init__(
        self,
        code: str,
        shared_setup: ComparisonSharedSetupEvidence,
    ) -> None:
        self.shared_setup = shared_setup
        super().__init__(code)


class _ComparisonSharedSetupCancelled(asyncio.CancelledError):
    def __init__(self, shared_setup: ComparisonSharedSetupEvidence) -> None:
        self.shared_setup = shared_setup
        super().__init__("comparison-interrupted")


class ComparisonRuntimeComposition(Protocol):
    @property
    def ingestion(self) -> IngestionService: ...

    @property
    def qa(self) -> QARuntimeServices: ...

    @property
    def retrieval_cache(self) -> object | None: ...


type ComparisonCompositionFactory = Callable[[Settings, Redactor], ComparisonRuntimeComposition]


class ComparisonCorpusInstaller(Protocol):
    async def install(self, dataset: EvaluationDataset) -> InstalledEvaluationCorpus: ...


type ComparisonCorpusInstallerFactory = Callable[[IngestionService], ComparisonCorpusInstaller]
type ComparisonCaseExecutorFactory = Callable[
    [MaterializedComparisonCandidate, QARuntimeServices, Redactor],
    EvaluationCaseExecutor,
]


def _default_composition_factory(
    settings: Settings,
    redactor: Redactor,
) -> ComparisonRuntimeComposition:
    from rag_mvp.api.composition import compose_openai_services

    return compose_openai_services(settings, redactor, include_evaluation=False)


def _default_corpus_installer_factory(
    ingestion: IngestionService,
) -> ComparisonCorpusInstaller:
    return EvaluationCorpusInstaller(ingestion)


def _default_case_executor_factory(
    candidate: MaterializedComparisonCandidate,
    services: QARuntimeServices,
    redactor: Redactor,
) -> EvaluationCaseExecutor:
    del candidate
    return ProductionQAExecutor(services, redactor)


@dataclass(frozen=True, slots=True)
class ProductionComparisonExecutionContext:
    comparison_id: str
    dataset: EvaluationDataset
    materialized: MaterializedComparisonPlan
    workspace: Path
    online_manifest_digest: str | None


@dataclass(slots=True)
class RegisteredComparisonLaunchCatalog:
    """Materialize only registered plans, then prepare every normal run before admission."""

    registry: RegisteredComparisonPlanRegistry
    datasets: EvaluationDatasetRegistry
    settings: Settings
    evaluation_repository: EvaluationRunStore
    comparison_repository: ComparisonRepository
    run_artifacts_root: Path
    dataset_id: str = _DEFAULT_DATASET_ID
    dataset_version: str = _DEFAULT_DATASET_VERSION

    def __post_init__(self) -> None:
        self.run_artifacts_root = Path(self.run_artifacts_root).resolve()

    def list(self) -> tuple[ComparisonPlanCatalogEntry, ...]:
        dataset = self._dataset()
        selections = self._upstream_selections()
        summaries = self.registry.list(
            ComparisonPlanCatalogContext(
                dataset=dataset,
                settings=self.settings,
                upstream_selections=selections,
            )
        )
        values: list[ComparisonPlanCatalogEntry] = []
        for summary in summaries:
            if summary.launchable:
                preview_id = f"preview-{summary.plan_id}"
                preview_settings = self._isolated_settings(preview_id)
                run_ids = _candidate_run_ids(preview_id, summary.candidates)
                try:
                    materialized = self.registry.resolve(
                        summary.plan_id,
                        ComparisonPlanMaterializationContext(
                            dataset=dataset,
                            settings=preview_settings,
                            upstream_selections=selections,
                            comparison_id=preview_id,
                            candidate_run_ids=run_ids,
                        ),
                    )
                except RegisteredComparisonPlanError as error:
                    values.append(self._blocked_entry(dataset, summary, (error.code,)))
                    continue
                values.append(
                    ComparisonPlanCatalogEntry.from_plan(
                        materialized.plan,
                        launchable=True,
                        conservative_cost_estimate=(materialized.preflight.snapshot.reserved_cost),
                    )
                )
                continue
            values.append(self._blocked_entry(dataset, summary, summary.blocking_codes))
        return tuple(values)

    def prepare(self, comparison_id: str, plan_id: str) -> PreparedComparisonLaunch:
        dataset = self._dataset()
        selections = self._upstream_selections()
        isolated = self._isolated_settings(comparison_id)
        workspace = isolated.data_root.resolve()
        if workspace.exists() or workspace.is_symlink():
            raise ComparisonProductionError("comparison-workspace-exists")
        summaries = {
            item.plan_id: item
            for item in self.registry.list(
                ComparisonPlanCatalogContext(
                    dataset=dataset,
                    settings=isolated,
                    upstream_selections=selections,
                )
            )
        }
        summary = summaries.get(plan_id)
        if summary is None:
            raise ComparisonProductionError("comparison_plan_not_found")
        if not summary.launchable:
            code = (
                summary.blocking_codes[0]
                if summary.blocking_codes
                else "comparison-prerequisite-missing"
            )
            raise ComparisonProductionError(code)
        candidate_run_ids = _candidate_run_ids(comparison_id, summary.candidates)
        try:
            materialized = self.registry.resolve(
                plan_id,
                ComparisonPlanMaterializationContext(
                    dataset=dataset,
                    settings=isolated,
                    upstream_selections=selections,
                    comparison_id=comparison_id,
                    candidate_run_ids=candidate_run_ids,
                ),
            )
            suite = create_comparison_suite(
                comparison_id,
                materialized.plan,
                candidate_run_ids,
            )
            runner = EvaluationRunner(
                self.evaluation_repository,
                self.run_artifacts_root,
                None,
            )
            runs = tuple(
                runner.prepare(candidate.evaluation_plan) for candidate in materialized.candidates
            )
        except RegisteredComparisonPlanError:
            raise
        except Exception as error:
            code = getattr(error, "code", "comparison-plan-materialization-invalid")
            raise ComparisonProductionError(str(code)) from None
        return PreparedComparisonLaunch(
            suite=suite,
            evaluation_runs=runs,
            execution_context=ProductionComparisonExecutionContext(
                comparison_id=comparison_id,
                dataset=dataset,
                materialized=materialized,
                workspace=workspace,
                online_manifest_digest=_online_manifest_digest(self.settings.data_root),
            ),
        )

    def _dataset(self) -> EvaluationDataset:
        try:
            return self.datasets.resolve(self.dataset_id, self.dataset_version)
        except EvaluationPlanError as error:
            raise ComparisonProductionError(error.code) from None

    def _upstream_selections(self) -> UpstreamSelections:
        generation = self.comparison_repository.get_selection(ExperimentAxis.GENERATION_MODEL)
        retrieval = self.comparison_repository.get_selection(ExperimentAxis.RETRIEVAL_STRATEGY)
        return UpstreamSelections(
            generation_model=(
                None if generation is None else UpstreamComparisonSelection.from_record(generation)
            ),
            retrieval_strategy=(
                None if retrieval is None else UpstreamComparisonSelection.from_record(retrieval)
            ),
        )

    def _isolated_settings(self, comparison_id: str) -> Settings:
        workspace = _suite_workspace(self.settings.data_root, comparison_id)
        return self.settings.model_copy(
            update={
                "data_root": workspace,
                "workbench_enabled": False,
            }
        )

    @staticmethod
    def _blocked_entry(
        dataset: EvaluationDataset,
        summary: RegisteredComparisonPlanSummary,
        blocking_codes: tuple[str, ...],
    ) -> ComparisonPlanCatalogEntry:
        unavailable = UnavailableValue(reason="comparison-plan-not-materialized")
        corpus = dataset.corpus.manifest
        plan_hash = summary.plan_content_hash
        return ComparisonPlanCatalogEntry(
            experiment_plan_id=summary.plan_id,
            plan_content_hash=unavailable if plan_hash is None else plan_hash,
            display_name=summary.display_name,
            axis=summary.axis.value,
            dataset_id=dataset.manifest.dataset_id,
            dataset_version=dataset.manifest.version,
            dataset_hash=dataset.manifest.content_hash,
            corpus_id=corpus.snapshot_id,
            corpus_version=corpus.version,
            corpus_hash=corpus.content_hash,
            case_set_hash=summary.case_set_hash,
            planned_case_count=summary.case_count,
            variants=tuple(
                ComparisonPlanVariantEntry(
                    variant_id=item.variant_id,
                    display_name=item.display_name,
                    axis_value=item.axis_value,
                    configuration_id=unavailable,
                )
                for item in summary.candidates
            ),
            baseline_variant_id=summary.baseline_variant_id,
            repeats_per_case=summary.repeats_per_case,
            maximum_logical_calls=summary.planned_logical_attempts,
            maximum_provider_calls=summary.maximum_provider_calls,
            cache_policy=summary.cache_policy.value,
            cost_estimate_status="unavailable",
            cost_estimate=None,
            cost_cap=summary.maximum_cost,
            currency=summary.pricing_currency,
            launchable=False,
            blocking_codes=blocking_codes,
        )


@dataclass(slots=True)
class _CandidateRuntime:
    candidate: MaterializedComparisonCandidate
    runner: EvaluationRunner
    composition: ComparisonRuntimeComposition
    services: QARuntimeServices
    admission: QAAdmissionController


@dataclass(slots=True)
class ProductionComparisonJobExecutor:
    """Execute normal candidate runs under one pre-reserved comparison budget."""

    settings: Settings
    evaluation_repository: EvaluationRunStore
    comparison_repository: ComparisonRepository
    run_artifacts_root: Path
    artifact_catalog: ComparisonArtifactCatalog
    redactor: Redactor
    composition_factory: ComparisonCompositionFactory = field(
        default=_default_composition_factory,
        repr=False,
    )
    corpus_installer_factory: ComparisonCorpusInstallerFactory = field(
        default=_default_corpus_installer_factory,
        repr=False,
    )
    case_executor_factory: ComparisonCaseExecutorFactory = field(
        default=_default_case_executor_factory,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.run_artifacts_root = Path(self.run_artifacts_root).resolve()

    async def execute(self, launch: PreparedComparisonLaunch) -> None:
        context = launch.execution_context
        if not isinstance(context, ProductionComparisonExecutionContext):
            raise ComparisonProductionError("comparison-execution-context-invalid")
        materialized = context.materialized
        suite = self.comparison_repository.get(launch.suite.comparison_id)
        if (
            context.comparison_id != launch.suite.comparison_id
            or suite != launch.suite
            or materialized.plan != suite.plan
        ):
            raise ComparisonProductionError("comparison-persisted-launch-mismatch")
        runtimes: dict[str, _CandidateRuntime] = {}
        compositions: list[ComparisonRuntimeComposition] = []
        admissions: list[QAAdmissionController] = []
        reports: dict[str, VerifiedCandidateReport] = {}
        shared_setup: ComparisonSharedSetupEvidence | None = None
        try:
            self._validate_runtime_context(context)
            runtimes, compositions, admissions, shared_setup = await self._prepare_runtimes(context)
            self.comparison_repository.save_shared_setup(shared_setup)
            suite = self._start_candidates(suite, materialized, runtimes)
            completed_steps: dict[str, int] = {
                item.variant_id: 0 for item in materialized.candidates
            }
            total_steps = {
                item.variant_id: len(item.evaluation_plan.cases) for item in materialized.candidates
            }
            cases = {
                item.variant_id: {case.case_id: case for case in item.evaluation_plan.cases}
                for item in materialized.candidates
            }
            failed_variants: set[str] = set()
            for step in materialized.schedule.steps:
                if step.variant_id in failed_variants:
                    continue
                runtime = runtimes[step.variant_id]
                try:
                    await runtime.runner.execute_case(
                        runtime.candidate.evaluation_plan,
                        cases[step.variant_id][step.execution_case_id],
                    )
                    completed_steps[step.variant_id] += 1
                    suite = self._record_running_progress(suite, runtime)
                    if completed_steps[step.variant_id] == total_steps[step.variant_id]:
                        report, suite = self._complete_candidate(
                            suite,
                            context,
                            runtime,
                        )
                        reports[step.variant_id] = report
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    failed_variants.add(step.variant_id)
                    self._fail_evaluation_run(runtime.candidate.evaluation_plan.run_id)
                    suite = self._record_candidate_failure(suite, runtime, error)
            if any(
                item.latest.status is ComparisonCandidateStatus.RUNNING for item in suite.candidates
            ):
                raise ComparisonProductionError("comparison-schedule-incomplete")
            compatibility = validate_comparison_compatibility(
                materialized.plan,
                tuple(item.identity_projection for item in materialized.candidates),
            )
            complete_report_set = {
                item.variant_id: reports.get(item.variant_id) for item in materialized.plan.variants
            }
            result = aggregate_comparison_result(
                suite,
                compatibility,
                complete_report_set,
                shared_setup=shared_setup,
            )
            _require_online_manifest_unchanged(
                self.settings.data_root,
                context.online_manifest_digest,
            )
            self.comparison_repository.save_result(result)
            try:
                self.artifact_catalog.publish(suite, result, reports)
            except Exception as error:
                raise ComparisonProductionError(
                    "publication-comparison-artifacts-failed"
                ) from error
            if (
                result.recommendation.selected_variant_id is not None
                and result.axis is not ExperimentAxis.CACHE_BEHAVIOR
            ):
                self.comparison_repository.save_selection(result)
        except asyncio.CancelledError as error:
            failure_code = "comparison-interrupted"
            setup = (
                _shared_setup_from_error(error)
                or shared_setup
                or _prepare_failure_setup(
                    context,
                    None,
                    setup_verified=False,
                    safe_error_code=failure_code,
                )
            )
            self._save_failed_shared_setup(setup)
            self._fail_nonterminal_evaluation_runs(
                materialized,
                failure_code,
            )
            self._reconcile_nonterminal_candidates(
                context.comparison_id,
                materialized,
                runtimes,
                status=ComparisonCandidateStatus.INTERRUPTED,
                safe_error_code=failure_code,
            )
            raise
        except Exception as error:
            failure_code = _safe_candidate_error_code(error)
            setup = (
                _shared_setup_from_error(error)
                or shared_setup
                or _prepare_failure_setup(
                    context,
                    None,
                    setup_verified=False,
                    safe_error_code=failure_code,
                )
            )
            self._save_failed_shared_setup(setup)
            self._fail_nonterminal_evaluation_runs(
                materialized,
                failure_code,
            )
            self._reconcile_nonterminal_candidates(
                context.comparison_id,
                materialized,
                runtimes,
                status=ComparisonCandidateStatus.FAILED,
                safe_error_code=failure_code,
            )
            raise
        finally:
            for admission in admissions:
                await admission.close()
            for composition in reversed(compositions):
                await _close_composition(composition)

    def _validate_runtime_context(self, context: ProductionComparisonExecutionContext) -> None:
        workspace = context.workspace.resolve()
        expected_workspace = _suite_workspace(
            self.settings.data_root,
            context.comparison_id,
        ).resolve()
        if workspace != expected_workspace:
            raise ComparisonProductionError("comparison-workspace-unsafe")
        expected_scorers = evaluation_scorer_versions(context.dataset)
        for candidate in context.materialized.candidates:
            if (
                candidate.settings.data_root.resolve() != workspace
                or candidate.evaluation_plan.identity.runtime_configuration_id
                != candidate.settings.runtime_configuration_identity
                or candidate.evaluation_plan.identity.configuration_id
                != candidate.settings.evaluation_configuration_identity
                or candidate.evaluation_plan.identity.scorer_versions != expected_scorers
            ):
                code = (
                    "comparison-scorer-version-mismatch"
                    if candidate.evaluation_plan.identity.scorer_versions != expected_scorers
                    else "comparison-runtime-identity-mismatch"
                )
                raise ComparisonProductionError(code)
        keys = {
            EvaluationIndexReuseKey.from_plan(candidate.evaluation_plan, context.dataset).digest
            for candidate in context.materialized.candidates
        }
        if len(keys) != 1:
            raise ComparisonProductionError("comparison-index-reuse-key-mismatch")
        if context.materialized.plan.axis is ExperimentAxis.CACHE_BEHAVIOR:
            settings = tuple(item.settings for item in context.materialized.candidates)
            if any(item != settings[0] for item in settings[1:]):
                raise ComparisonProductionError("comparison-cache-settings-mismatch")

    async def _prepare_runtimes(
        self,
        context: ProductionComparisonExecutionContext,
    ) -> tuple[
        dict[str, _CandidateRuntime],
        list[ComparisonRuntimeComposition],
        list[QAAdmissionController],
        ComparisonSharedSetupEvidence,
    ]:
        materialized = context.materialized
        if context.workspace.exists() or context.workspace.is_symlink():
            raise ComparisonProductionError("comparison-workspace-exists")
        index_estimates = materialized.preflight.estimates[
            -materialized.preflight.index_build_count :
        ]
        if len(index_estimates) != 1:
            raise ComparisonProductionError("comparison-index-estimate-mismatch")
        case_estimates = materialized.preflight.estimates[
            : materialized.preflight.logical_attempt_count
        ]
        estimate_offset = 0
        compositions_by_runtime: dict[str, ComparisonRuntimeComposition] = {}
        services_by_runtime: dict[str, tuple[QARuntimeServices, QAAdmissionController]] = {}
        compositions: list[ComparisonRuntimeComposition] = []
        admissions: list[QAAdmissionController] = []
        runtimes: dict[str, _CandidateRuntime] = {}
        installed = False
        setup_verified = False
        shared_setup: ComparisonSharedSetupEvidence | None = None
        try:
            for candidate in materialized.candidates:
                runtime_id = candidate.settings.runtime_configuration_identity
                composition = compositions_by_runtime.get(runtime_id)
                if composition is None:
                    composition = self.composition_factory(candidate.settings, self.redactor)
                    compositions_by_runtime[runtime_id] = composition
                    compositions.append(composition)
                    admission = QAAdmissionController(
                        candidate.settings.qa_max_active,
                        candidate.settings.qa_max_queue,
                    )
                    services = replace(
                        composition.qa,
                        admission=admission,
                        latency_budgets=QALatencyBudgets.from_settings(candidate.settings),
                    )
                    services_by_runtime[runtime_id] = (services, admission)
                    admissions.append(admission)
                    ingestion = composition.ingestion
                    _require_ingestion_workspace(ingestion, context.workspace)
                    materialized.preflight.budget.require_reserved((index_estimates[0],))
                    key = EvaluationIndexReuseKey.from_plan(
                        candidate.evaluation_plan,
                        context.dataset,
                    )
                    if not installed:
                        installer = self.corpus_installer_factory(ingestion)
                        installed_corpus, shared_setup = await _install_shared_corpus(
                            installer,
                            ingestion,
                            context,
                        )
                        key.verify_revision(installed_corpus.revision)
                        setup_verified = True
                        installed = True
                    else:
                        revision = _active_index_revision(ingestion)
                        if revision is None:
                            raise ComparisonProductionError("comparison-index-reuse-unavailable")
                        key.verify_revision(revision)
                services, admission = services_by_runtime[runtime_id]
                count = len(candidate.evaluation_plan.cases)
                estimates = tuple(case_estimates[estimate_offset : estimate_offset + count])
                estimate_offset += count
                estimator = _reserved_case_estimator(
                    candidate.evaluation_plan.cases,
                    estimates,
                )
                runner = EvaluationRunner(
                    self.evaluation_repository,
                    self.run_artifacts_root,
                    self.case_executor_factory(candidate, services, self.redactor),
                    work_budget=materialized.preflight.budget,
                    case_work_estimator=estimator,
                    work_reservations_prepared=True,
                )
                runtimes[candidate.variant_id] = _CandidateRuntime(
                    candidate=candidate,
                    runner=runner,
                    composition=composition,
                    services=services,
                    admission=admission,
                )
            if estimate_offset != len(case_estimates):
                raise ComparisonProductionError("comparison-case-estimate-mismatch")
            if materialized.plan.axis is ExperimentAxis.CACHE_BEHAVIOR:
                compositions_for_candidates = tuple(
                    runtimes[item.variant_id].composition for item in materialized.candidates
                )
                if (
                    any(
                        item is not compositions_for_candidates[0]
                        for item in compositions_for_candidates
                    )
                    or compositions_for_candidates[0].retrieval_cache is None
                ):
                    raise ComparisonProductionError("comparison-cache-coordinator-invalid")
            if shared_setup is None:
                raise ComparisonProductionError("comparison-shared-setup-missing")
            return runtimes, compositions, admissions, shared_setup
        except BaseException as error:
            for admission in admissions:
                with suppress(Exception):
                    await admission.close()
            for composition in reversed(compositions):
                with suppress(Exception):
                    await _close_composition(composition)
            if isinstance(
                error,
                (_ComparisonSharedSetupExecutionError, _ComparisonSharedSetupCancelled),
            ):
                raise
            failure_code = (
                "comparison-interrupted"
                if isinstance(error, asyncio.CancelledError)
                else _safe_candidate_error_code(error)
                if isinstance(error, Exception)
                else "comparison-setup-failed"
            )
            failure_setup = _prepare_failure_setup(
                context,
                shared_setup,
                setup_verified=setup_verified,
                safe_error_code=failure_code,
            )
            if isinstance(error, asyncio.CancelledError):
                raise _ComparisonSharedSetupCancelled(failure_setup) from error
            if isinstance(error, Exception):
                raise _ComparisonSharedSetupExecutionError(
                    failure_code,
                    failure_setup,
                ) from error
            raise

    def _start_candidates(
        self,
        suite: ComparisonSuite,
        materialized: MaterializedComparisonPlan,
        runtimes: Mapping[str, _CandidateRuntime],
    ) -> ComparisonSuite:
        for candidate in materialized.candidates:
            runtimes[candidate.variant_id].runner.start(candidate.evaluation_plan)
            suite = suite.transition_candidate(
                candidate.variant_id,
                status=ComparisonCandidateStatus.RUNNING,
                completed_cases=0,
                failed_cases=0,
                provider_calls=0,
                recorded_at=_next_timestamp(suite),
            )
            self.comparison_repository.append(suite)
        return suite

    def _record_running_progress(
        self,
        suite: ComparisonSuite,
        runtime: _CandidateRuntime,
    ) -> ComparisonSuite:
        run = self.evaluation_repository.get(runtime.candidate.evaluation_plan.run_id)
        if run is None:
            raise ComparisonProductionError("comparison-evaluation-run-missing")
        attempts = self._provider_attempts(runtime)
        ledger = self._provider_ledger_summary(suite, runtime, attempts)
        suite = suite.transition_candidate(
            runtime.candidate.variant_id,
            status=ComparisonCandidateStatus.RUNNING,
            completed_cases=run.completed_cases,
            failed_cases=run.failed_cases,
            provider_calls=len(attempts),
            incurred_cost=ledger.total_cost,
            known_partial_cost=ledger.known_partial_cost,
            cost_complete=ledger.cost_complete,
            cost_unknown_reasons=ledger.cost_unknown_reasons,
            currency=ledger.currency,
            recorded_at=_next_timestamp(suite),
        )
        self.comparison_repository.append(suite)
        return suite

    def _complete_candidate(
        self,
        suite: ComparisonSuite,
        context: ProductionComparisonExecutionContext,
        runtime: _CandidateRuntime,
    ) -> tuple[VerifiedCandidateReport, ComparisonSuite]:
        plan = runtime.candidate.evaluation_plan
        runtime.runner.complete(plan)
        results = runtime.runner.load_case_results(plan.run_id)
        attempts_by_request = self._provider_attempts_by_request(runtime, results)
        reference = next(
            item.reference
            for item in suite.candidates
            if item.reference.variant_id == runtime.candidate.variant_id
        )
        evidence = build_persisted_candidate_evidence(
            comparison_id=suite.comparison_id,
            experiment_plan=context.materialized.plan,
            reference=reference,
            dataset=context.dataset,
            evaluation_plan=plan,
            results=results,
            provider_attempts_by_request=attempts_by_request,
            identity_projection=runtime.candidate.identity_projection,
            redactor=self.redactor,
        )
        report = seal_comparison_candidate_evidence(reference, evidence)
        suite = suite.transition_candidate(
            runtime.candidate.variant_id,
            status=ComparisonCandidateStatus.COMPLETED,
            completed_cases=sum(item.succeeded for item in results),
            failed_cases=sum(not item.succeeded for item in results),
            provider_calls=evidence.provider_call_count,
            incurred_cost=evidence.total_cost,
            known_partial_cost=evidence.known_partial_cost,
            cost_complete=evidence.cost_complete,
            cost_unknown_reasons=evidence.cost_unknown_reasons,
            currency=evidence.currency,
            recorded_at=_next_timestamp(suite),
        )
        self.comparison_repository.append(suite)
        return report, suite

    def _record_candidate_failure(
        self,
        suite: ComparisonSuite,
        runtime: _CandidateRuntime,
        error: Exception,
    ) -> ComparisonSuite:
        run = self.evaluation_repository.get(runtime.candidate.evaluation_plan.run_id)
        completed = 0 if run is None else run.completed_cases
        failed = 1 if run is None else max(1, run.failed_cases)
        attempts = self._provider_attempts(runtime)
        ledger = self._provider_ledger_summary(suite, runtime, attempts)
        safe_code = _safe_candidate_error_code(error)
        suite = suite.transition_candidate(
            runtime.candidate.variant_id,
            status=ComparisonCandidateStatus.FAILED,
            completed_cases=completed,
            failed_cases=failed,
            provider_calls=len(attempts),
            incurred_cost=ledger.total_cost,
            known_partial_cost=ledger.known_partial_cost,
            cost_complete=ledger.cost_complete,
            cost_unknown_reasons=ledger.cost_unknown_reasons,
            currency=ledger.currency,
            safe_error_code=safe_code,
            recorded_at=_next_timestamp(suite),
        )
        self.comparison_repository.append(suite)
        return suite

    def _fail_evaluation_run(
        self,
        run_id: str,
        *,
        safe_error_code: str = "candidate-execution-failed",
        include_completed: bool = True,
    ) -> None:
        run = self.evaluation_repository.get(run_id)
        terminal = {EvaluationRunStatus.FAILED, EvaluationRunStatus.INVALID}
        if not include_completed:
            terminal.add(EvaluationRunStatus.COMPLETED)
        if run is None or run.status in terminal:
            return
        self.evaluation_repository.update(
            EvaluationRun.model_validate(
                {
                    **run.model_dump(mode="python"),
                    "status": EvaluationRunStatus.FAILED,
                    "safe_error_code": safe_error_code,
                    "updated_at": utc_now(),
                }
            )
        )

    def _fail_nonterminal_evaluation_runs(
        self,
        materialized: MaterializedComparisonPlan,
        safe_error_code: str,
    ) -> None:
        for candidate in materialized.candidates:
            self._fail_evaluation_run(
                candidate.evaluation_plan.run_id,
                safe_error_code=safe_error_code,
                include_completed=False,
            )

    def _save_failed_shared_setup(
        self,
        evidence: ComparisonSharedSetupEvidence,
    ) -> None:
        existing = self.comparison_repository.get_shared_setup(evidence.comparison_id)
        if existing is None:
            self.comparison_repository.save_shared_setup(evidence)
        elif existing != evidence:
            raise ComparisonProductionError("comparison-shared-setup-persisted-mismatch")

    def _reconcile_nonterminal_candidates(
        self,
        comparison_id: str,
        materialized: MaterializedComparisonPlan,
        runtimes: Mapping[str, _CandidateRuntime],
        *,
        status: ComparisonCandidateStatus,
        safe_error_code: str,
    ) -> None:
        suite = self.comparison_repository.get(comparison_id)
        if suite is None or suite.status in {
            ComparisonStatus.FAILED,
            ComparisonStatus.INVALID,
        }:
            return
        for history in suite.candidates:
            latest = history.latest
            if latest.status in {
                ComparisonCandidateStatus.COMPLETED,
                ComparisonCandidateStatus.FAILED,
                ComparisonCandidateStatus.INTERRUPTED,
            }:
                continue
            candidate = next(
                item
                for item in materialized.candidates
                if item.variant_id == history.reference.variant_id
            )
            run = self.evaluation_repository.get(candidate.evaluation_plan.run_id)
            runtime = runtimes.get(candidate.variant_id)
            provider_calls = latest.provider_calls
            incurred_cost = latest.incurred_cost
            known_partial_cost = latest.known_partial_cost
            cost_complete = latest.cost_complete
            cost_unknown_reasons = latest.cost_unknown_reasons
            currency = latest.currency
            if runtime is not None:
                attempts = self._provider_attempts(runtime)
                ledger = self._provider_ledger_summary(suite, runtime, attempts)
                provider_calls = len(attempts)
                incurred_cost = ledger.total_cost
                known_partial_cost = ledger.known_partial_cost
                cost_complete = ledger.cost_complete
                cost_unknown_reasons = ledger.cost_unknown_reasons
                currency = ledger.currency
            suite = suite.transition_candidate(
                candidate.variant_id,
                status=status,
                completed_cases=(latest.completed_cases if run is None else run.completed_cases),
                failed_cases=(latest.failed_cases if run is None else run.failed_cases),
                provider_calls=provider_calls,
                incurred_cost=incurred_cost,
                known_partial_cost=known_partial_cost,
                cost_complete=cost_complete,
                cost_unknown_reasons=cost_unknown_reasons,
                currency=currency,
                safe_error_code=safe_error_code,
                recorded_at=_next_timestamp(suite),
            )
            self.comparison_repository.append(suite)
        if suite.status is ComparisonStatus.COMPLETED:
            suite = suite.fail(
                f"result-{safe_error_code}",
                recorded_at=_next_timestamp(suite),
            )
            self.comparison_repository.append(suite)

    @staticmethod
    def _provider_repository(runtime: _CandidateRuntime) -> ProviderUsageRepository:
        database = runtime.composition.ingestion.repositories.index_revisions.database
        return RuntimeRepositories.from_database(database).provider_usage

    def _provider_attempts(self, runtime: _CandidateRuntime) -> tuple[ModelAttempt, ...]:
        repository = self._provider_repository(runtime)
        return tuple(repository.list_for_run(runtime.candidate.evaluation_plan.run_id))

    @staticmethod
    def _provider_ledger_summary(
        suite: ComparisonSuite,
        runtime: _CandidateRuntime,
        attempts: Sequence[ModelAttempt],
    ) -> PersistedProviderLedgerSummary:
        return summarize_persisted_provider_attempts(
            suite.plan,
            runtime.candidate.evaluation_plan,
            attempts,
        )

    def _provider_attempts_by_request(
        self,
        runtime: _CandidateRuntime,
        results: Sequence[PersistedCaseResult],
    ) -> dict[str, tuple[ModelAttempt, ...]]:
        repository = self._provider_repository(runtime)
        attempts = tuple(repository.list_for_run(runtime.candidate.evaluation_plan.run_id))
        grouped: dict[str, list[ModelAttempt]] = {}
        for attempt in attempts:
            if attempt.request_id is None:
                raise ComparisonProductionError("comparison-provider-request-missing")
            grouped.setdefault(attempt.request_id, []).append(attempt)
        values: dict[str, tuple[ModelAttempt, ...]] = {}
        for result in results:
            execution = result.execution
            if execution is not None:
                values[execution.request_id] = tuple(grouped.pop(execution.request_id, ()))
        for request_id, unbound in grouped.items():
            values[request_id] = tuple(unbound)
        return values


def _candidate_run_ids(
    comparison_id: str,
    candidates: Sequence[RegisteredComparisonCandidateSummary],
) -> dict[str, str]:
    values: dict[str, str] = {}
    prefix = comparison_id[:180]
    for index, candidate in enumerate(candidates):
        variant_id = candidate.variant_id
        digest = hashlib.sha256(f"{comparison_id}\0{index}\0{variant_id}".encode()).hexdigest()[:16]
        run_id = f"{prefix}-candidate-{index + 1}-{digest}"
        if _SAFE_OPAQUE_ID.fullmatch(run_id) is None:
            raise ComparisonProductionError("comparison-candidate-run-id-invalid")
        values[variant_id] = run_id
    return values


def _suite_workspace(data_root: Path, comparison_id: str) -> Path:
    if _SAFE_OPAQUE_ID.fullmatch(comparison_id) is None:
        raise ComparisonProductionError("comparison-id-invalid")
    declared_online = Path(data_root).expanduser().absolute()
    declared_parent = declared_online / "evaluations" / "suites"
    declared_workspace = declared_parent / comparison_id / "runtime"
    _reject_symlink_components(
        declared_online,
        (
            declared_online,
            declared_online / "evaluations",
            declared_parent,
            declared_parent / comparison_id,
            declared_workspace,
        ),
    )
    online = declared_online.resolve()
    parent = declared_parent.resolve()
    workspace = declared_workspace.resolve()
    if (
        not parent.is_relative_to(online)
        or not workspace.is_relative_to(parent)
        or workspace == online
        or workspace.is_relative_to((online / "indexes").resolve())
    ):
        raise ComparisonProductionError("comparison-workspace-unsafe")
    return workspace


def _online_manifest_digest(data_root: Path) -> str | None:
    path = DataLayout.from_root(data_root).active_manifest
    if path.is_symlink():
        raise ComparisonProductionError("comparison-online-index-unsafe")
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError:
        raise ComparisonProductionError("comparison-online-index-unavailable") from None
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _require_online_manifest_unchanged(data_root: Path, expected: str | None) -> None:
    if _online_manifest_digest(data_root) != expected:
        raise ComparisonProductionError("comparison-online-index-mutated")


def _next_timestamp(suite: ComparisonSuite) -> datetime:
    return max(utc_now(), suite.updated_at + timedelta(microseconds=1))


def _safe_candidate_error_code(error: Exception) -> str:
    code = getattr(error, "code", None)
    return (
        code
        if isinstance(code, str) and _SAFE_ERROR_CODE.fullmatch(code) is not None
        else "candidate-execution-failed"
    )


def _shared_setup_from_error(
    error: BaseException,
) -> ComparisonSharedSetupEvidence | None:
    value = getattr(error, "shared_setup", None)
    return value if isinstance(value, ComparisonSharedSetupEvidence) else None


def _reject_symlink_components(root: Path, paths: Sequence[Path]) -> None:
    for path in paths:
        try:
            if path.is_symlink():
                raise ComparisonProductionError("comparison-workspace-unsafe")
        except OSError:
            raise ComparisonProductionError("comparison-workspace-unavailable") from None
        if path != root and not path.absolute().is_relative_to(root.absolute()):
            raise ComparisonProductionError("comparison-workspace-unsafe")


def _reserved_case_estimator(
    cases: Sequence[EvaluationCaseInput],
    estimates: Sequence[ProviderWorkEstimate],
) -> Callable[[EvaluationCaseInput], ProviderWorkEstimate]:
    values = {case.case_id: estimate for case, estimate in zip(cases, estimates, strict=True)}

    def estimate(case: EvaluationCaseInput) -> ProviderWorkEstimate:
        try:
            return values[case.case_id]
        except KeyError:
            raise ComparisonProductionError("comparison-case-estimate-missing") from None

    return estimate


def _require_ingestion_workspace(
    ingestion: IngestionService,
    expected: Path,
) -> None:
    if ingestion.data_root.resolve() != expected.resolve():
        raise ComparisonProductionError("comparison-index-isolation-failed")


def _active_index_revision(ingestion: IngestionService) -> IndexRevision | None:
    return ingestion.repositories.index_revisions.get_active()


async def _install_shared_corpus(
    installer: ComparisonCorpusInstaller,
    ingestion: IngestionService,
    context: ProductionComparisonExecutionContext,
) -> tuple[InstalledEvaluationCorpus, ComparisonSharedSetupEvidence]:
    installed: InstalledEvaluationCorpus | None = None
    failure: BaseException | None = None
    with unbound_evaluation_attempt_context(), capture_provider_attempts() as captured:
        try:
            installed = await installer.install(context.dataset)
        except BaseException as error:
            failure = error
    plan = context.materialized.plan
    request_id = _setup_request_id(plan.fixed_identities.corpus_hash)
    try:
        repository = _provider_repository_for_ingestion(ingestion)
        persisted = tuple(repository.list_for_request(request_id))
        _validate_setup_capture(captured.attempts, persisted, request_id=request_id)
        attempts = _shared_setup_attempts(
            context.comparison_id,
            plan,
            f"rev_eval_{plan.fixed_identities.corpus_hash.removeprefix('sha256:')}",
            persisted,
        )
    except Exception as error:
        unavailable = ComparisonSharedSetupEvidence.create(
            comparison_id=context.comparison_id,
            plan=plan,
            status=ComparisonSharedSetupStatus.FAILED,
            attempts=(),
            safe_error_code="comparison-shared-setup-ledger-mismatch",
            provider_calls_complete=False,
        )
        raise _ComparisonSharedSetupExecutionError(
            "comparison-shared-setup-ledger-mismatch",
            unavailable,
        ) from error
    failure_code = (
        None
        if failure is None
        else "comparison-interrupted"
        if isinstance(failure, asyncio.CancelledError)
        else _safe_candidate_error_code(cast(Exception, failure))
    )
    evidence = ComparisonSharedSetupEvidence.create(
        comparison_id=context.comparison_id,
        plan=plan,
        status=(
            ComparisonSharedSetupStatus.COMPLETED
            if failure is None
            else ComparisonSharedSetupStatus.FAILED
        ),
        attempts=attempts,
        safe_error_code=failure_code,
    )
    if failure is not None:
        if isinstance(failure, asyncio.CancelledError):
            raise _ComparisonSharedSetupCancelled(evidence) from failure
        if not isinstance(failure, Exception):
            raise failure
        raise _ComparisonSharedSetupExecutionError(
            cast(str, failure_code),
            evidence,
        ) from failure
    if installed is None:
        raise ComparisonProductionError("comparison-shared-setup-install-missing")
    if evidence.index_revision_id != installed.revision.revision_id:
        failed = ComparisonSharedSetupEvidence.create(
            comparison_id=context.comparison_id,
            plan=plan,
            status=ComparisonSharedSetupStatus.FAILED,
            attempts=attempts,
            safe_error_code="comparison-shared-setup-revision-mismatch",
        )
        raise _ComparisonSharedSetupExecutionError(
            "comparison-shared-setup-revision-mismatch",
            failed,
        )
    return installed, evidence


def _prepare_failure_setup(
    context: ProductionComparisonExecutionContext,
    shared_setup: ComparisonSharedSetupEvidence | None,
    *,
    setup_verified: bool,
    safe_error_code: str,
) -> ComparisonSharedSetupEvidence:
    if shared_setup is not None and setup_verified:
        return shared_setup
    return ComparisonSharedSetupEvidence.create(
        comparison_id=context.comparison_id,
        plan=context.materialized.plan,
        status=ComparisonSharedSetupStatus.FAILED,
        attempts=() if shared_setup is None else shared_setup.attempts,
        safe_error_code=safe_error_code,
    )


def _setup_request_id(corpus_hash: str) -> str:
    return f"eval_corpus_{corpus_hash.removeprefix('sha256:')}"


def _provider_repository_for_ingestion(
    ingestion: IngestionService,
) -> ProviderUsageRepository:
    database = ingestion.repositories.index_revisions.database
    return RuntimeRepositories.from_database(database).provider_usage


def _validate_setup_capture(
    captured: Sequence[ProviderModelAttempt],
    persisted: Sequence[ModelAttempt],
    *,
    request_id: str,
) -> None:
    if len(captured) != len(persisted):
        raise ComparisonProductionError("comparison-shared-setup-ledger-mismatch")
    if len({item.attempt_id for item in persisted}) != len(persisted):
        raise ComparisonProductionError("comparison-shared-setup-attempt-duplicate")
    for source, stored in zip(captured, persisted, strict=True):
        expected_status = ModelAttemptStatus(source.status.value)
        if source.error_category in {
            ProviderErrorCategory.TIMEOUT,
            ProviderErrorCategory.DEADLINE_EXCEEDED,
        }:
            expected_status = ModelAttemptStatus.TIMED_OUT
        if (
            source.request_id != request_id
            or stored.request_id != request_id
            or stored.run_id is not None
            or stored.operation_id != source.operation_id
            or stored.attempt_number != source.attempt_number
            or stored.role.value != source.role.value
            or stored.provider != source.provider
            or stored.model != source.model
            or stored.status is not expected_status
            or stored.fallback != source.is_fallback
            or stored.latency_ms != source.latency_ms
            or stored.usage.input_tokens != source.usage.input_tokens
            or stored.usage.output_tokens != source.usage.output_tokens
            or stored.usage.total_tokens_reported != source.usage.total_tokens
            or stored.safe_error_category
            != (source.error_category.value if source.error_category is not None else None)
        ):
            raise ComparisonProductionError("comparison-shared-setup-ledger-mismatch")


def _shared_setup_attempts(
    comparison_id: str,
    plan: ExperimentPlan,
    revision_id: str,
    attempts: Sequence[ModelAttempt],
) -> tuple[ComparisonSharedSetupAttempt, ...]:
    controlled = {item.name: item.value for item in plan.fixed_identities.controlled}
    expected_provider = controlled.get("provider.embedding")
    expected_model = controlled.get("model.embedding")
    rate = next(
        (
            item
            for item in plan.pricing.rate_card
            if item.role.value == "embedding"
            and item.provider == expected_provider
            and item.model == expected_model
        ),
        None,
    )
    if rate is None:
        raise ComparisonProductionError("comparison-shared-setup-pricing-missing")
    setup_id = comparison_shared_setup_id(comparison_id)
    values: list[ComparisonSharedSetupAttempt] = []
    for attempt in attempts:
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
        request_id = attempt.request_id
        if request_id is None:
            raise ComparisonProductionError("comparison-shared-setup-request-missing")
        try:
            values.append(
                ComparisonSharedSetupAttempt.create(
                    attempt_reference=attempt.attempt_id,
                    setup_id=setup_id,
                    request_id=request_id,
                    index_revision_id=revision_id,
                    source_run_id=attempt.run_id,
                    evidence=evidence,
                    latency_ms=attempt.latency_ms,
                    pricing_version=plan.pricing.pricing_version,
                    pricing_hash=plan.pricing.pricing_hash,
                    currency=plan.pricing.currency,
                    input_per_million=rate.input_per_million,
                    output_per_million=rate.output_per_million,
                    pricing_source_reference=rate.source_reference,
                    recorded_at=attempt.created_at,
                )
            )
        except (TypeError, ValueError):
            raise ComparisonProductionError("comparison-shared-setup-attempt-invalid") from None
    return tuple(values)


async def _close_composition(composition: ComparisonRuntimeComposition) -> None:
    ingestion = getattr(composition, "ingestion", None)
    close_ingestion = getattr(ingestion, "close", None)
    if callable(close_ingestion):
        close_ingestion()
    await composition.qa.close()


__all__ = [
    "ComparisonProductionError",
    "ProductionComparisonExecutionContext",
    "ProductionComparisonJobExecutor",
    "RegisteredComparisonLaunchCatalog",
]
