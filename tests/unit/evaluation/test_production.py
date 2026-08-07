from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from test_json_report import valid_report

from rag_mvp.config.settings import Settings
from rag_mvp.domain.evaluation import ReportManifest
from rag_mvp.evaluation import production
from rag_mvp.evaluation.html_report import write_html_report
from rag_mvp.evaluation.json_report import report_content_hash, write_json_report
from rag_mvp.evaluation.production import (
    EvaluationProductionError,
    VerifiedLegacyReportStore,
)
from rag_mvp.safety.redactor import DEFAULT_REDACTOR


@dataclass
class MemoryReportRepository:
    manifest: ReportManifest | None

    def get(self, run_id: str) -> ReportManifest | None:
        if self.manifest is None or self.manifest.run_id != run_id:
            return None
        return self.manifest


def _published_report(root: Path) -> tuple[str, ReportManifest]:
    report = valid_report()
    run_id = str(report["run_id"])
    run_root = root / run_id
    run_root.mkdir(parents=True)
    json_path = write_json_report(report, run_root / "evaluation-report.json")
    html_path = write_html_report(report, run_root / "evaluation-report.html")
    return run_id, ReportManifest(
        run_id=run_id,
        schema_version=str(report["schema_version"]),
        json_report_path=str(json_path),
        html_report_path=str(html_path),
        content_hash=report_content_hash(report),
    )


def test_verified_legacy_report_store_returns_bytes_without_a_path(tmp_path: Path) -> None:
    run_id, manifest = _published_report(tmp_path / "runs")
    store = VerifiedLegacyReportStore(
        MemoryReportRepository(manifest),  # type: ignore[arg-type]
        tmp_path / "runs",
    )

    artifact = store.resolve(run_id, "json")

    assert artifact is not None
    assert artifact.artifact_id == "evaluation-report-json"
    assert artifact.filename == "evaluation-report.json"
    assert artifact.media_type == "application/json"
    assert str(tmp_path).encode() not in artifact.content
    assert not hasattr(artifact, "path")


def test_verified_legacy_report_store_returns_the_exact_verified_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id, manifest = _published_report(tmp_path / "runs")
    report_path = tmp_path / "runs" / run_id / "evaluation-report.json"
    verified_content = report_path.read_bytes()
    original_hash = production.report_content_hash

    def mutate_after_verification(report: object) -> str:
        digest = original_hash(report)  # type: ignore[arg-type]
        report_path.write_bytes(b'{"unverified":"replacement"}\n')
        return digest

    monkeypatch.setattr(production, "report_content_hash", mutate_after_verification)
    store = VerifiedLegacyReportStore(
        MemoryReportRepository(manifest),  # type: ignore[arg-type]
        tmp_path / "runs",
    )

    artifact = store.resolve(run_id, "json")

    assert artifact is not None
    assert artifact.content == verified_content
    assert artifact.content != report_path.read_bytes()


def test_verified_legacy_report_store_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    run_id, manifest = _published_report(tmp_path / "runs")
    outside = tmp_path / "outside.json"
    outside.write_bytes((tmp_path / "runs" / run_id / "evaluation-report.json").read_bytes())
    tampered = manifest.model_copy(update={"json_report_path": str(outside)})
    store = VerifiedLegacyReportStore(
        MemoryReportRepository(tampered),  # type: ignore[arg-type]
        tmp_path / "runs",
    )

    with pytest.raises(EvaluationProductionError, match="evaluation_report_integrity_failed"):
        store.resolve(run_id, "json")


def test_verified_legacy_report_store_rejects_symlinked_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id, manifest = _published_report(tmp_path / "runs")
    report_path = tmp_path / "runs" / run_id / "evaluation-report.json"
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path == report_path or original_is_symlink(path),
    )
    store = VerifiedLegacyReportStore(
        MemoryReportRepository(manifest),  # type: ignore[arg-type]
        tmp_path / "runs",
    )

    with pytest.raises(EvaluationProductionError, match="evaluation_report_integrity_failed"):
        store.resolve(run_id, "json")


def test_isolated_settings_rejects_workspace_parent_resolved_outside_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(_env_file=None, data_root=tmp_path / "online")
    executor = production.ProductionEvaluationJobExecutor(
        settings=settings,
        repository=object(),  # type: ignore[arg-type]
        report_repository=object(),  # type: ignore[arg-type]
        run_artifacts_root=tmp_path / "runs",
        redactor=DEFAULT_REDACTOR,
    )
    online_root = settings.data_root.resolve()
    workspace_parent = online_root / "evaluations" / "workspaces"
    outside = (tmp_path / "outside" / "workspaces").resolve()
    original_resolve = Path.resolve

    def redirected_resolve(path: Path, strict: bool = False) -> Path:
        if path == workspace_parent:
            return outside
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", redirected_resolve)

    with pytest.raises(EvaluationProductionError, match="evaluation_workspace_unsafe"):
        executor.isolated_settings("safe-run")
