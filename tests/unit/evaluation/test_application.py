from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest
from test_json_report import valid_report

from rag_mvp.config.settings import Settings
from rag_mvp.domain.evaluation import EvaluationRun, EvaluationRunStatus
from rag_mvp.domain.qa import RefusalReason, StreamEventKind, ValidatedStreamEvent
from rag_mvp.domain.retrieval import CachePolicy
from rag_mvp.evaluation.application import (
    EvaluationApplicationError,
    EvaluationApplicationService,
    EvaluationArtifactDescriptor,
    EvaluationArtifactManifest,
    EvaluationCapacityError,
    EvaluationConflictError,
    ResolvedEvaluationArtifact,
)
from rag_mvp.evaluation.dataset import EvaluationDataset
from rag_mvp.evaluation.json_report import canonical_report_document
from rag_mvp.evaluation.plan import EvaluationDatasetRegistry, build_evaluation_plan
from rag_mvp.evaluation.production import ProductionEvaluationJobExecutor
from rag_mvp.evaluation.runner import (
    EvaluationCaseExecution,
    EvaluationCaseInput,
    EvaluationRunner,
    EvaluationRunPlan,
)
from rag_mvp.safety.redactor import DEFAULT_REDACTOR
from rag_mvp.storage.repositories import EvaluationRunRepository, ReportManifestRepository

_DATASETS = Path(__file__).resolve().parents[3] / "evaluations" / "datasets"


@dataclass
class MemoryRunRepository:
    runs: dict[str, EvaluationRun] = field(default_factory=dict)

    def create(self, run: EvaluationRun) -> None:
        if run.run_id in self.runs:
            raise ValueError("duplicate")
        self.runs[run.run_id] = run

    def get(self, run_id: str) -> EvaluationRun | None:
        return self.runs.get(run_id)

    def update(self, run: EvaluationRun) -> None:
        if run.run_id not in self.runs:
            raise ValueError("missing")
        self.runs[run.run_id] = run

    def list(self) -> list[EvaluationRun]:
        return list(self.runs.values())


@dataclass
class SafeCaseExecutor:
    fail: bool = False

    async def execute(
        self,
        case: EvaluationCaseInput,
        *,
        owner_id: str,
        cache_policy: CachePolicy,
    ) -> EvaluationCaseExecution:
        assert cache_policy is CachePolicy.BYPASS
        if self.fail:
            raise RuntimeError("raw question and person@example.com must never escape")
        event = ValidatedStreamEvent(
            request_id=f"request_{case.case_id}",
            session_id=f"session_{case.case_id}",
            sequence=0,
            kind=StreamEventKind.REFUSAL,
            response_language="en",
            content="Safe refusal.",
            reason=RefusalReason.INSUFFICIENT_EVIDENCE,
            terminal=True,
        )
        return EvaluationCaseExecution(
            case_id=case.case_id,
            owner_id=owner_id,
            session_id=event.session_id,
            request_id=event.request_id,
            event=event,
            latency_ms=1,
        )


@dataclass
class BlockingJobExecutor:
    repository: MemoryRunRepository
    root: Path
    case_executor: SafeCaseExecutor = field(default_factory=SafeCaseExecutor)
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def execute(self, plan: EvaluationRunPlan, dataset: EvaluationDataset) -> None:
        del dataset
        self.entered.set()
        await self.release.wait()
        await EvaluationRunner(self.repository, self.root, self.case_executor).execute(plan)


@dataclass
class ReportFailingJobExecutor:
    repository: MemoryRunRepository
    root: Path

    async def execute(self, plan: EvaluationRunPlan, dataset: EvaluationDataset) -> None:
        del dataset
        await EvaluationRunner(self.repository, self.root, SafeCaseExecutor()).execute(plan)
        raise RuntimeError("sensitive report writer detail")


@dataclass
class StaticLegacyReportStore:
    artifact: ResolvedEvaluationArtifact
    calls: int = 0

    def resolve(
        self,
        run_id: str,
        report_format: str,
    ) -> ResolvedEvaluationArtifact | None:
        del run_id
        self.calls += 1
        return self.artifact if report_format == "json" else None


@dataclass
class StaticArtifactStore:
    manifest_value: EvaluationArtifactManifest
    artifacts: dict[str, ResolvedEvaluationArtifact] = field(default_factory=dict)

    def manifest(self, run_id: str) -> EvaluationArtifactManifest | None:
        del run_id
        return self.manifest_value

    def resolve(
        self,
        run_id: str,
        artifact_id: str,
    ) -> ResolvedEvaluationArtifact | None:
        del run_id
        return self.artifacts.get(artifact_id)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_root=tmp_path / "online",
        evaluation_dataset_root=_DATASETS,
        workbench_enabled=False,
    )


def _report_run(status: EvaluationRunStatus) -> EvaluationRun:
    report = valid_report()
    provenance = cast(dict[str, object], report["provenance"])
    dataset = cast(dict[str, object], provenance["dataset"])
    return EvaluationRun(
        run_id=str(report["run_id"]),
        status=status,
        dataset_id=str(dataset["id"]),
        dataset_version=str(dataset["version"]),
        dataset_hash=str(dataset["content_hash"]),
        corpus_version="corpus-v1",
        configuration_id="config-001",
        code_revision="abc12345",
        scorer_versions={"faithfulness": "faithfulness-v1"},
        cache_policy="bypass",
        total_cases=4,
        completed_cases=4,
        safe_error_code=(
            "evaluation_publication_failed" if status is EvaluationRunStatus.FAILED else None
        ),
    )


def _legacy_report_artifact() -> ResolvedEvaluationArtifact:
    return ResolvedEvaluationArtifact(
        artifact_id="evaluation-report-json",
        content=canonical_report_document(valid_report()),
        media_type="application/json",
        filename="evaluation-report.json",
    )


def _service(
    tmp_path: Path,
    repository: MemoryRunRepository,
    executor: object,
    *,
    ids: list[str] | None = None,
    limit: int = 1,
    shutdown_grace: float = 0,
) -> EvaluationApplicationService:
    values = iter(ids or ["eval_one", "eval_two", "eval_three"])
    return EvaluationApplicationService(
        registry=EvaluationDatasetRegistry(_DATASETS),
        settings=_settings(tmp_path),
        repository=repository,
        run_artifacts_root=tmp_path / "runs",
        executor=executor,  # type: ignore[arg-type]
        maximum_active_jobs=limit,
        shutdown_grace_seconds=shutdown_grace,
        run_id_factory=lambda: next(values),
    )


@pytest.mark.asyncio
async def test_catalog_queue_duplicate_capacity_and_missing_evidence_state(
    tmp_path: Path,
) -> None:
    repository = MemoryRunRepository()
    executor = BlockingJobExecutor(repository, tmp_path / "runs")
    service = _service(tmp_path, repository, executor)
    await service.startup()

    datasets = service.datasets()
    plans = service.plans()
    assert {item.dataset_id for item in datasets} == {
        "mvp-bilingual-rag",
        "original-pdf-acceptance",
    }
    assert all(item.maximum_logical_calls == item.planned_case_count for item in plans)
    assert all(item.maximum_provider_calls >= item.maximum_logical_calls for item in plans)
    assert all(item.cost_estimate_status == "unavailable" for item in plans)

    queued = await service.start("mvp-bilingual-rag", "1.0.0")
    await executor.entered.wait()
    assert queued.status is EvaluationRunStatus.QUEUED
    persisted_plan = EvaluationRunner(repository, tmp_path / "runs", None).load_plan(queued.run_id)
    assert (
        persisted_plan.identity.configuration_id
        == service.settings.evaluation_configuration_identity
    )

    with pytest.raises(EvaluationConflictError, match="evaluation_duplicate"):
        await service.start("mvp-bilingual-rag", "1.0.0")
    with pytest.raises(EvaluationCapacityError, match="evaluation_capacity"):
        await service.start("original-pdf-acceptance", "2.0.0")

    executor.release.set()
    completed = await service.wait(queued.run_id)
    assert completed is not None and completed.status is EvaluationRunStatus.COMPLETED
    summary = service.summary(queued.run_id)
    assert summary is not None
    assert summary.evidence_status == "unavailable"
    assert summary.gate_status == "unavailable"
    assert summary.completed_at is not None
    await service.close()


@pytest.mark.asyncio
async def test_restart_reconciles_stale_run_without_reusing_it(tmp_path: Path) -> None:
    repository = MemoryRunRepository()
    settings = _settings(tmp_path)
    dataset = EvaluationDatasetRegistry(_DATASETS).resolve("mvp-bilingual-rag", "1.0.0")
    plan = build_evaluation_plan(dataset, settings, "stale_run")
    runner = EvaluationRunner(repository, tmp_path / "runs", None)
    runner.queue(plan)
    original_plan = (tmp_path / "runs" / "stale_run" / "plan.json").read_bytes()
    executor = BlockingJobExecutor(repository, tmp_path / "runs")
    service = _service(tmp_path, repository, executor)

    await service.startup()

    reconciled = repository.get("stale_run")
    assert reconciled is not None
    assert reconciled.status is EvaluationRunStatus.FAILED
    assert reconciled.safe_error_code == "evaluation_interrupted"
    assert (tmp_path / "runs" / "stale_run" / "plan.json").read_bytes() == original_plan
    await service.close()


@pytest.mark.asyncio
async def test_close_cancels_after_own_grace_and_persists_interruption(tmp_path: Path) -> None:
    repository = MemoryRunRepository()
    executor = BlockingJobExecutor(repository, tmp_path / "runs")
    service = _service(tmp_path, repository, executor, shutdown_grace=0)
    await service.startup()
    queued = await service.start("mvp-bilingual-rag", "1.0.0")
    await executor.entered.wait()

    await service.close()

    interrupted = repository.get(queued.run_id)
    assert interrupted is not None
    assert interrupted.status is EvaluationRunStatus.FAILED
    assert interrupted.safe_error_code == "evaluation_interrupted"


@pytest.mark.asyncio
async def test_post_case_report_failure_cannot_remain_completed(tmp_path: Path) -> None:
    repository = MemoryRunRepository()
    executor = ReportFailingJobExecutor(repository, tmp_path / "runs")
    service = _service(tmp_path, repository, executor)
    await service.startup()

    queued = await service.start("mvp-bilingual-rag", "1.0.0")
    failed = await service.wait(queued.run_id)

    assert failed is not None
    assert failed.status is EvaluationRunStatus.FAILED
    assert failed.safe_error_code == "evaluation_execution_failed"
    summary = service.summary(queued.run_id)
    assert summary is not None and summary.evidence_status == "incomplete"
    await service.close()


def test_failed_run_cannot_retain_passing_report_state_or_download(tmp_path: Path) -> None:
    run = _report_run(EvaluationRunStatus.FAILED)
    repository = MemoryRunRepository({run.run_id: run})
    report_store = StaticLegacyReportStore(_legacy_report_artifact())
    service = _service(tmp_path, repository, object())
    service.legacy_report_store = report_store  # type: ignore[assignment]

    summary = service.summary(run.run_id)

    assert summary is not None
    assert summary.status is EvaluationRunStatus.FAILED
    assert summary.evidence_status == "incomplete"
    assert summary.gate_status == "unavailable"
    assert service.report(run.run_id, "json") is None
    assert report_store.calls == 0


def test_legacy_report_download_rejects_repository_identity_mismatch(
    tmp_path: Path,
) -> None:
    run = _report_run(EvaluationRunStatus.COMPLETED).model_copy(
        update={"configuration_id": "config-foreign"}
    )
    repository = MemoryRunRepository({run.run_id: run})
    service = _service(tmp_path, repository, object())
    service.legacy_report_store = StaticLegacyReportStore(  # type: ignore[assignment]
        _legacy_report_artifact()
    )

    with pytest.raises(
        EvaluationApplicationError,
        match="evaluation_report_identity_failed",
    ):
        service.report(run.run_id, "json")


def test_v2_artifact_manifest_is_bound_to_persisted_run_configuration(
    tmp_path: Path,
) -> None:
    run = _report_run(EvaluationRunStatus.COMPLETED)
    repository = MemoryRunRepository({run.run_id: run})
    service = _service(tmp_path, repository, object())
    service.artifact_store = StaticArtifactStore(  # type: ignore[assignment]
        EvaluationArtifactManifest(
            run_id=run.run_id,
            configuration_id="config-foreign",
            manifest_content_hash="sha256:" + "a" * 64,
            artifacts=(),
        )
    )

    with pytest.raises(
        EvaluationApplicationError,
        match="evaluation_artifact_identity_failed",
    ):
        service.artifact_manifest(run.run_id)


def test_v2_report_selection_uses_exact_artifact_id_not_shared_json_format(
    tmp_path: Path,
) -> None:
    run = _report_run(EvaluationRunStatus.COMPLETED)
    repository = MemoryRunRepository({run.run_id: run})
    report_artifact = _legacy_report_artifact()
    report_digest = f"sha256:{sha256(report_artifact.content).hexdigest()}"
    dictionary_content = b'{"fields":[]}\n'
    service = _service(tmp_path, repository, object())
    service.artifact_store = StaticArtifactStore(  # type: ignore[assignment]
        EvaluationArtifactManifest(
            run_id=run.run_id,
            configuration_id=run.configuration_id,
            manifest_content_hash="sha256:" + "a" * 64,
            artifacts=(
                EvaluationArtifactDescriptor(
                    artifact_id="structured-log-field-dictionary",
                    schema_version="structured-log-field-dictionary-v1",
                    format="json",
                    media_type="application/json",
                    sha256_digest=f"sha256:{sha256(dictionary_content).hexdigest()}",
                    byte_size=len(dictionary_content),
                    created_at=run.updated_at,
                ),
                EvaluationArtifactDescriptor(
                    artifact_id="evaluation-report-json",
                    schema_version="evaluation-report-v1",
                    format="json",
                    media_type="application/json",
                    sha256_digest=report_digest,
                    byte_size=len(report_artifact.content),
                    created_at=run.updated_at,
                ),
            ),
        ),
        artifacts={"evaluation-report-json": report_artifact},
    )

    selected = service.report(run.run_id, "json")

    assert selected is not None
    assert selected.artifact_id == "evaluation-report-json"
    assert selected.content == report_artifact.content


@pytest.mark.asyncio
async def test_failed_case_diagnostics_are_allowlisted_and_tagged(tmp_path: Path) -> None:
    repository = MemoryRunRepository()
    executor = BlockingJobExecutor(
        repository,
        tmp_path / "runs",
        case_executor=SafeCaseExecutor(fail=True),
    )
    executor.release.set()
    service = _service(tmp_path, repository, executor)
    await service.startup()
    queued = await service.start("mvp-bilingual-rag", "1.0.0")
    await service.wait(queued.run_id)

    cases = service.failed_cases(queued.run_id)
    assert cases
    assert all(item.safe_error_code == "case_execution_failed" for item in cases)
    assert all(item.tags for item in cases)
    assert all(item.metric_contributions == () for item in cases)
    serialized = "".join(item.model_dump_json() for item in cases)
    assert "person@example.com" not in serialized
    assert "raw question" not in serialized
    assert str(tmp_path) not in serialized
    await service.close()


def test_production_plan_identity_uses_exact_isolated_runtime_without_online_index_mutation(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path).model_copy(
        update={"workbench_enabled": False, "retrieval_cache_enabled": False}
    )
    online_active = settings.data_root / "indexes" / "active.json"
    online_active.parent.mkdir(parents=True)
    online_active.write_bytes(b'{"revision_id":"online-sentinel"}\n')
    repository = MemoryRunRepository()
    executor = ProductionEvaluationJobExecutor(
        settings=settings,
        repository=cast(EvaluationRunRepository, repository),
        report_repository=cast(ReportManifestRepository, object()),
        run_artifacts_root=tmp_path / "runs",
        redactor=DEFAULT_REDACTOR,
    )
    isolated = executor.isolated_settings("identity_run")
    dataset = EvaluationDatasetRegistry(_DATASETS).resolve("mvp-bilingual-rag", "1.0.0")
    plan = build_evaluation_plan(dataset, isolated, "identity_run")

    assert plan.identity.configuration_id == isolated.evaluation_configuration_identity
    assert isolated.data_root != settings.data_root
    assert isolated.data_root.is_relative_to(settings.data_root / "evaluations" / "workspaces")
    assert not isolated.data_root.is_relative_to(settings.data_root / "indexes")
    assert online_active.read_bytes() == b'{"revision_id":"online-sentinel"}\n'
