from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from rag_mvp.config.settings import Settings
from rag_mvp.domain.evaluation import EvaluationRun
from rag_mvp.evaluation.application import EvaluationApplicationService
from rag_mvp.evaluation.dataset import EvaluationDataset
from rag_mvp.evaluation.plan import EvaluationDatasetRegistry
from rag_mvp.evaluation.release_evidence import VerifiedReleaseEvidenceStore
from rag_mvp.evaluation.runner import EvaluationRunPlan
from rag_mvp.safety.redactor import DEFAULT_REDACTOR
from rag_mvp.ui.evaluation_dashboard import render_evaluation_dashboard
from rag_mvp.ui.models import BrowserSessionState

_ROOT = Path(__file__).resolve().parents[3]
_RELEASES = _ROOT / "evaluations" / "releases"
_DATASETS = _ROOT / "evaluations" / "datasets"
_RUN_ID = "20260807t030340z-954bb3e2-quality"


@dataclass
class _MemoryRuns:
    values: dict[str, EvaluationRun] = field(default_factory=dict)

    def create(self, run: EvaluationRun) -> None:
        self.values[run.run_id] = run

    def get(self, run_id: str) -> EvaluationRun | None:
        return self.values.get(run_id)

    def update(self, run: EvaluationRun) -> None:
        self.values[run.run_id] = run

    def list(self) -> list[EvaluationRun]:
        return list(self.values.values())


class _NoopExecutor:
    async def execute(self, plan: EvaluationRunPlan, dataset: EvaluationDataset) -> None:
        del plan, dataset


def _service(tmp_path: Path) -> EvaluationApplicationService:
    return EvaluationApplicationService(
        registry=EvaluationDatasetRegistry(_DATASETS),
        settings=Settings(
            _env_file=None,
            data_root=tmp_path / "data",
            evaluation_dataset_root=_DATASETS,
            evaluation_release_root=_RELEASES,
            workbench_enabled=False,
        ),
        repository=_MemoryRuns(),
        run_artifacts_root=tmp_path / "runs",
        executor=_NoopExecutor(),
        release_store=VerifiedReleaseEvidenceStore(_RELEASES),
    )


def test_sealed_release_is_the_default_read_only_evaluation_evidence(tmp_path: Path) -> None:
    service = _service(tmp_path)

    runs = service.list_runs()
    assert runs[0].run_id == _RUN_ID
    assert service.get(_RUN_ID) == runs[0]
    summary = service.summary(_RUN_ID)
    assert summary is not None
    assert summary.evidence_status == "available"
    assert summary.gate_status == "passed"

    rendered = render_evaluation_dashboard(
        service,
        redactor=DEFAULT_REDACTOR,
        state=BrowserSessionState(owner_id="owner-release-test"),
    )
    assert rendered.selected_run_id == _RUN_ID
    assert "4.61s" in rendered.kpi_html
    assert "USD 1.4996" in rendered.kpi_html
    assert "99.02%" in rendered.kpi_html
    assert "Security / 安全" in rendered.kpi_html
    assert "PASS" in rendered.kpi_html
    compliance = next(row for row in rendered.quality_rows if row[0] == "answer-compliance")
    assert "not-recorded-in-v1-release" in str(compliance[1])
    assert "sealed-release-v1" in str(rendered.run_rows[0])
    assert "/api/v1/evaluations/" in rendered.artifact_links_markdown


def test_sealed_release_reports_are_hash_verified_and_downloadable(tmp_path: Path) -> None:
    service = _service(tmp_path)

    manifest = service.artifact_manifest(_RUN_ID)
    assert manifest is not None
    assert {item.artifact_id for item in manifest.artifacts} == {
        "evaluation-report-json",
        "evaluation-report-html",
    }
    report = service.report(_RUN_ID, "json")
    assert report is not None
    assert report.content.startswith(b'{"$schema"')
    assert service.report(_RUN_ID, "html") is not None


def test_tampered_release_is_not_listed(tmp_path: Path) -> None:
    copied_root = tmp_path / "releases"
    source = next(_RELEASES.iterdir())
    target = copied_root / source.name
    shutil.copytree(source, target)
    report = target / "evaluation-report.json"
    report.write_bytes(report.read_bytes() + b"\n")

    store = VerifiedReleaseEvidenceStore(copied_root)

    assert store.list() == ()
    assert store.get(_RUN_ID) is None
