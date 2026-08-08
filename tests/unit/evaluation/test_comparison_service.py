from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import pytest
from test_comparison import (
    _artifact,
    _compatibility,
    _plan,
    _shared_setup,
    _suite,
    _terminal_suite,
    _verified_reports,
)

from rag_mvp.config.settings import Settings
from rag_mvp.domain.evaluation import EvaluationRun
from rag_mvp.evaluation.application import (
    ComparisonConflictError,
    ComparisonNotFoundError,
    ComparisonUnavailableError,
    ComparisonValidationError,
    EvaluationApplicationService,
    EvaluationCapacityError,
)
from rag_mvp.evaluation.comparison import (
    COMPARISON_RESULT_SCHEMA_VERSION,
    ComparisonArtifactManifest,
    ComparisonResult,
    ComparisonSharedSetupEvidence,
    ComparisonSuite,
    ResolvedComparisonArtifact,
    aggregate_comparison_result,
    canonical_candidate_evidence,
    resolve_comparison_artifact,
    seal_comparison_candidate_evidence,
)
from rag_mvp.evaluation.comparison_application import (
    ComparisonPlanCatalogEntry,
    PreparedComparisonLaunch,
)
from rag_mvp.evaluation.dataset import EvaluationDataset
from rag_mvp.evaluation.json_report import canonical_json_value
from rag_mvp.evaluation.plan import EvaluationDatasetRegistry
from rag_mvp.evaluation.runner import EvaluationRunPlan

_DATASETS = Path(__file__).resolve().parents[3] / "evaluations" / "datasets"


@dataclass
class MemoryEvaluationRepository:
    runs: dict[str, EvaluationRun] = field(default_factory=dict)

    def create(self, run: EvaluationRun) -> None:
        self.runs[run.run_id] = run

    def get(self, run_id: str) -> EvaluationRun | None:
        return self.runs.get(run_id)

    def update(self, run: EvaluationRun) -> None:
        self.runs[run.run_id] = run

    def list(self) -> list[EvaluationRun]:
        return list(self.runs.values())


class NoopEvaluationExecutor:
    async def execute(self, plan: EvaluationRunPlan, dataset: EvaluationDataset) -> None:
        del plan, dataset


@dataclass
class MemoryComparisonRepository:
    suite: ComparisonSuite | None = None
    result: ComparisonResult | None = None
    shared_setup: ComparisonSharedSetupEvidence | None = None
    evaluation_runs: tuple[EvaluationRun, ...] = ()
    create_calls: int = 0

    def create(
        self,
        suite: ComparisonSuite,
        evaluation_runs: Sequence[EvaluationRun],
    ) -> None:
        self.create_calls += 1
        if self.suite is not None:
            raise RepositoryConflict("duplicate")
        self.suite = suite
        self.evaluation_runs = tuple(evaluation_runs)

    def append(self, suite: ComparisonSuite) -> None:
        self.suite = suite

    def get(self, comparison_id: str) -> ComparisonSuite | None:
        if self.suite is None or self.suite.comparison_id != comparison_id:
            return None
        return self.suite

    def list(self) -> tuple[ComparisonSuite, ...]:
        return () if self.suite is None else (self.suite,)

    def get_result(self, comparison_id: str) -> ComparisonResult | None:
        if self.suite is None or self.suite.comparison_id != comparison_id:
            return None
        return self.result

    def get_shared_setup(
        self,
        comparison_id: str,
    ) -> ComparisonSharedSetupEvidence | None:
        if self.suite is None or self.suite.comparison_id != comparison_id:
            return None
        return self.shared_setup


class RepositoryConflict(RuntimeError):
    pass


@dataclass
class StaticComparisonCatalog:
    launch: PreparedComparisonLaunch
    error: Exception | None = None
    prepare_calls: int = 0

    def list(self) -> tuple[ComparisonPlanCatalogEntry, ...]:
        return (
            ComparisonPlanCatalogEntry.from_plan(
                self.launch.suite.plan,
                launchable=True,
            ),
        )

    def prepare(self, comparison_id: str, plan_id: str) -> PreparedComparisonLaunch:
        self.prepare_calls += 1
        if self.error is not None:
            raise self.error
        assert comparison_id == self.launch.suite.comparison_id
        assert plan_id == self.launch.suite.plan.plan_id
        return self.launch


@dataclass
class BlockingComparisonExecutor:
    repository: MemoryComparisonRepository
    calls: int = 0
    observed_atomic_rows: bool = False
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def execute(self, launch: PreparedComparisonLaunch) -> None:
        self.calls += 1
        self.observed_atomic_rows = (
            self.repository.suite == launch.suite
            and self.repository.evaluation_runs == launch.evaluation_runs
        )
        self.entered.set()
        await self.release.wait()


@dataclass
class StaticComparisonArtifactStore:
    manifest_value: ComparisonArtifactManifest
    payloads: dict[str, bytes]
    resolve_calls: int = 0

    def manifest(self, comparison_id: str) -> ComparisonArtifactManifest | None:
        return self.manifest_value if comparison_id == self.manifest_value.comparison_id else None

    def resolve(
        self,
        comparison_id: str,
        artifact_id: str,
    ) -> ResolvedComparisonArtifact | None:
        if comparison_id != self.manifest_value.comparison_id:
            return None
        content = self.payloads.get(artifact_id)
        if content is None:
            return None
        self.resolve_calls += 1
        return resolve_comparison_artifact(self.manifest_value, artifact_id, content)


class StableError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__("details-must-not-escape")


def _evaluation_runs(suite: ComparisonSuite) -> tuple[EvaluationRun, ...]:
    fixed = suite.plan.fixed_identities
    total_cases = fixed.case_count * suite.plan.repeat_order_policy.repeats_per_case
    return tuple(
        EvaluationRun(
            run_id=item.reference.evaluation_run_id,
            dataset_id=fixed.dataset_id,
            dataset_version=fixed.dataset_version,
            dataset_hash=fixed.dataset_hash,
            corpus_version=fixed.corpus_version,
            configuration_id=item.reference.configuration_id,
            code_revision="revision-1",
            scorer_versions={"comparison": "comparison-v1"},
            cache_policy=suite.plan.cache_policy.value,
            total_cases=total_cases,
        )
        for item in suite.candidates
    )


def _launch(suite: ComparisonSuite | None = None) -> PreparedComparisonLaunch:
    value = suite or _suite()
    return PreparedComparisonLaunch(
        suite=value,
        evaluation_runs=_evaluation_runs(value),
        execution_context=object(),
    )


def _service(
    tmp_path: Path,
    catalog: StaticComparisonCatalog,
    repository: MemoryComparisonRepository,
    executor: BlockingComparisonExecutor,
    *,
    artifact_store: StaticComparisonArtifactStore | None = None,
    comparison_id: str = "comparison-1",
    shutdown_grace_seconds: float = 0,
) -> EvaluationApplicationService:
    return EvaluationApplicationService(
        registry=EvaluationDatasetRegistry(_DATASETS),
        settings=Settings(
            _env_file=None,
            data_root=tmp_path / "online",
            evaluation_dataset_root=_DATASETS,
            workbench_enabled=False,
        ),
        repository=MemoryEvaluationRepository(),
        run_artifacts_root=tmp_path / "runs",
        executor=NoopEvaluationExecutor(),
        maximum_active_jobs=1,
        shutdown_grace_seconds=shutdown_grace_seconds,
        comparison_catalog=catalog,
        comparison_repository=repository,
        comparison_executor=executor,
        comparison_artifact_store=artifact_store,
        comparison_id_factory=lambda: comparison_id,
    )


@pytest.mark.asyncio
async def test_start_persists_every_candidate_before_shared_supervisor_work(
    tmp_path: Path,
) -> None:
    launch = _launch()
    repository = MemoryComparisonRepository()
    catalog = StaticComparisonCatalog(launch)
    executor = BlockingComparisonExecutor(repository)
    service = _service(tmp_path, catalog, repository, executor)
    await service.startup()

    queued = await service.start_comparison(launch.suite.plan.plan_id)
    await executor.entered.wait()

    assert queued.comparison_id == launch.suite.comparison_id
    assert executor.observed_atomic_rows
    assert repository.create_calls == 1
    assert {item.run_id for item in repository.evaluation_runs} == {
        item.reference.evaluation_run_id for item in launch.suite.candidates
    }
    with pytest.raises(ComparisonConflictError, match="comparison_duplicate"):
        await service.start_comparison(launch.suite.plan.plan_id)
    with pytest.raises(EvaluationCapacityError, match="evaluation_capacity"):
        await service.start("mvp-bilingual-rag", "1.0.0")

    executor.release.set()
    await service.wait_comparison(queued.comparison_id)
    await service.close()


@pytest.mark.asyncio
async def test_startup_interrupts_stale_comparison_without_executor_work(
    tmp_path: Path,
) -> None:
    launch = _launch()
    repository = MemoryComparisonRepository(suite=launch.suite)
    catalog = StaticComparisonCatalog(launch)
    executor = BlockingComparisonExecutor(repository)
    service = _service(tmp_path, catalog, repository, executor)

    await service.startup()

    assert repository.suite is not None
    assert repository.suite.status.value == "failed"
    assert repository.suite.safe_error_code == "comparison-interrupted"
    assert all(
        item.latest.status.value == "interrupted"
        and item.latest.safe_error_code == "comparison-interrupted"
        for item in repository.suite.candidates
    )
    assert executor.calls == 0
    await service.close()


@pytest.mark.asyncio
async def test_generated_id_and_catalog_failures_are_typed(
    tmp_path: Path,
) -> None:
    launch = _launch()
    repository = MemoryComparisonRepository()
    catalog = StaticComparisonCatalog(launch)
    executor = BlockingComparisonExecutor(repository)
    invalid_id_service = _service(
        tmp_path,
        catalog,
        repository,
        executor,
        comparison_id="../unsafe",
    )
    await invalid_id_service.startup()
    with pytest.raises(ComparisonUnavailableError, match="comparison_id_invalid"):
        await invalid_id_service.start_comparison(launch.suite.plan.plan_id)
    assert catalog.prepare_calls == 0
    await invalid_id_service.close()

    cases = (
        ("comparison_plan_not_found", ComparisonNotFoundError),
        ("comparison_exact_pricing_missing", ComparisonConflictError),
        ("comparison_registered_dataset_identity_mismatch", ComparisonConflictError),
        ("provider_call_cap_exceeded", ComparisonValidationError),
        ("comparison_runtime_unavailable", ComparisonUnavailableError),
    )
    for index, (code, expected) in enumerate(cases):
        failing_catalog = StaticComparisonCatalog(launch, StableError(code))
        service = _service(
            tmp_path / str(index),
            failing_catalog,
            MemoryComparisonRepository(),
            BlockingComparisonExecutor(MemoryComparisonRepository()),
        )
        await service.startup()
        with pytest.raises(expected, match=code):
            await service.start_comparison(launch.suite.plan.plan_id)
        await service.close()


def _result_bundle() -> tuple[
    ComparisonSuite,
    ComparisonResult,
    ComparisonArtifactManifest,
    dict[str, bytes],
]:
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
    )
    payloads = {
        "comparison-plan-json": (canonical_json_value(plan.model_dump(mode="json")) + "\n").encode(
            "utf-8"
        ),
        "comparison-report-json": (
            canonical_json_value(result.model_dump(mode="json")) + "\n"
        ).encode("utf-8"),
        "comparison-report-html": b"<html><body>comparison</body></html>",
        "comparison-report-txt": b"comparison\n",
        "comparison-report-csv": b"metric,value\nquality,1\n",
        **{
            sealed.descriptor.artifact_id: canonical_candidate_evidence(sealed.evidence)
            for sealed in reports.values()
        },
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
        **{
            sealed.descriptor.artifact_id: (
                sealed.descriptor.schema_version,
                sealed.descriptor.media_type,
                sealed.descriptor.relative_path,
            )
            for sealed in reports.values()
        },
    }
    manifest = ComparisonArtifactManifest.create(
        comparison_id=suite.comparison_id,
        plan=plan,
        artifacts=tuple(
            _artifact(artifact_id, *contracts[artifact_id], content)
            for artifact_id, content in payloads.items()
        ),
        created_at=suite.updated_at,
    )
    return suite, result, manifest, payloads


def test_artifacts_require_matching_persisted_result_and_canonical_bytes(
    tmp_path: Path,
) -> None:
    suite, result, manifest, payloads = _result_bundle()
    repository = MemoryComparisonRepository(suite=suite)
    store = StaticComparisonArtifactStore(manifest, payloads)
    launch = _launch(_suite())
    service = _service(
        tmp_path,
        StaticComparisonCatalog(launch),
        repository,
        BlockingComparisonExecutor(repository),
        artifact_store=store,
    )

    assert service.comparison_manifest(suite.comparison_id) is None
    assert store.resolve_calls == 0

    repository.result = result
    view = service.comparison_manifest(suite.comparison_id)
    assert view is not None
    assert "path" not in view.model_dump_json().casefold()
    report = service.comparison_artifact(suite.comparison_id, "comparison-report-json")
    assert report is not None and report.content == payloads["comparison-report-json"]

    repository.suite = suite.fail(
        "publication-failed",
        recorded_at=suite.updated_at + timedelta(microseconds=1),
    )
    assert service.comparison_manifest(suite.comparison_id) is None
    assert service.comparison_artifact(suite.comparison_id, "comparison-report-json") is None
    repository.suite = suite

    noncanonical = dict(payloads)
    noncanonical["comparison-report-json"] = (result.model_dump_json(indent=2) + "\n").encode(
        "utf-8"
    )
    _, _, noncanonical_manifest, _ = _result_bundle()
    replacement = _artifact(
        "comparison-report-json",
        COMPARISON_RESULT_SCHEMA_VERSION,
        "application/json",
        "comparison-report.json",
        noncanonical["comparison-report-json"],
    )
    noncanonical_manifest = ComparisonArtifactManifest.create(
        comparison_id=suite.comparison_id,
        plan=suite.plan,
        artifacts=tuple(
            replacement if item.artifact_id == replacement.artifact_id else item
            for item in noncanonical_manifest.artifacts
        ),
        created_at=suite.updated_at,
    )
    service.comparison_artifact_store = StaticComparisonArtifactStore(
        noncanonical_manifest,
        noncanonical,
    )
    with pytest.raises(
        RuntimeError,
        match="comparison_artifact_integrity_failed",
    ):
        service.comparison_manifest(suite.comparison_id)


@pytest.mark.asyncio
async def test_repository_duplicate_is_a_typed_conflict(tmp_path: Path) -> None:
    launch = _launch()
    repository = MemoryComparisonRepository(suite=launch.suite)
    catalog = StaticComparisonCatalog(launch)
    executor = BlockingComparisonExecutor(repository)
    service = _service(tmp_path, catalog, repository, executor)
    await service.startup()

    with pytest.raises(ComparisonConflictError, match="comparison_duplicate"):
        await service.start_comparison(launch.suite.plan.plan_id)
    await service.close()
