from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

import pytest
from fastapi.testclient import TestClient

from rag_mvp.api.app import create_app
from rag_mvp.config.settings import Settings
from rag_mvp.domain.evaluation import EvaluationRun, EvaluationRunStatus
from rag_mvp.evaluation.application import (
    EvaluationApplicationService,
    EvaluationArtifactDescriptor,
    EvaluationArtifactManifest,
    EvaluationDatasetCatalogEntry,
    EvaluationPlanCatalogEntry,
    EvaluationRunSummary,
    ResolvedEvaluationArtifact,
)
from rag_mvp.evaluation.operations_v2 import (
    OperationsEvidenceInputV2,
    build_operations_summary_v2,
    render_operations_csv,
    render_operations_text,
)
from rag_mvp.evaluation.plan import EvaluationDatasetRegistry
from rag_mvp.evaluation.report_v2 import canonical_report_document_v2, parse_report_v2
from rag_mvp.ui.callbacks import WorkbenchCallbacks
from rag_mvp.ui.models import BrowserSessionState
from rag_mvp.ui.services import WorkbenchServices
from rag_mvp.ui.workbench import create_workbench

pytestmark = pytest.mark.ui

_ROOT = Path(__file__).resolve().parents[2]
_CREATED_AT = datetime(2026, 8, 7, 3, 4, 5, tzinfo=UTC)
_RAW_REPORT = json.loads((_ROOT / "tests" / "fixtures" / "evaluation-report-v2.json").read_bytes())
_RAW_REPORT["operations_summary"] = build_operations_summary_v2(
    OperationsEvidenceInputV2(
        run_id="acceptance-run-v2",
        configuration_id="configuration-v2",
        total_logical_requests=2,
        successful_logical_requests=2,
        all_attempt_latency_ms=(100.0, 200.0),
        provider_attempt_count=2,
        input_tokens=2000,
        output_tokens=1000,
        cache_hits=0,
        cache_eligible_lookups=0,
        refusals=0,
        answered_requests=2,
        compliant_answers=2,
        scored_answers=2,
        total_cost=Decimal("0.012"),
        currency="USD",
        source_artifact_ids=("attempt-ledger",),
        generated_at=_CREATED_AT,
    )
).model_dump(mode="json")
_PARSED = parse_report_v2(_RAW_REPORT)
_REPORT = canonical_report_document_v2(_PARSED)


def _completed_run() -> EvaluationRun:
    return EvaluationRun(
        run_id=_PARSED.run_id,
        status=EvaluationRunStatus.COMPLETED,
        dataset_id=_PARSED.provenance.dataset_id,
        dataset_version=_PARSED.provenance.dataset_version,
        dataset_hash=_PARSED.provenance.dataset_content_hash,
        corpus_version=_PARSED.provenance.corpus_version,
        configuration_id=_PARSED.configuration_id,
        code_revision=_PARSED.provenance.code_revision,
        scorer_versions={"advanced": "v2"},
        cache_policy="bypass",
        total_cases=24,
        completed_cases=23,
        failed_cases=1,
        created_at=_CREATED_AT,
        updated_at=_CREATED_AT,
    )


def _descriptor(
    artifact_id: str,
    artifact_format: str,
    media_type: str,
    content: bytes,
) -> EvaluationArtifactDescriptor:
    return EvaluationArtifactDescriptor(
        artifact_id=artifact_id,
        schema_version="evidence-v2",
        format=artifact_format,
        media_type=media_type,
        sha256_digest="sha256:" + "a" * 64,
        byte_size=len(content),
        created_at=_CREATED_AT,
    )


class TypedEvaluationGateway:
    def __init__(
        self,
        *,
        with_manifest: bool = True,
        run: EvaluationRun | None = None,
        evidence_status: Literal["available", "incomplete", "unavailable"] = "available",
        gate_status: Literal["passed", "failed", "unavailable"] = "passed",
    ) -> None:
        self.starts: list[tuple[str, str | None]] = []
        self._runs = [run or _completed_run()]
        self._evidence_status = evidence_status
        self._gate_status = gate_status
        text = render_operations_text(_PARSED.operations_summary).encode()
        csv = render_operations_csv(_PARSED.operations_summary).encode()
        self._artifacts = {
            "evaluation-report-json": ResolvedEvaluationArtifact(
                "evaluation-report-json",
                _REPORT,
                "application/json",
                "evaluation-report.json",
            ),
            "operations-summary-txt": ResolvedEvaluationArtifact(
                "operations-summary-txt",
                text,
                "text/plain",
                "operations-summary.txt",
            ),
            "operations-summary-csv": ResolvedEvaluationArtifact(
                "operations-summary-csv",
                csv,
                "text/csv",
                "operations-summary.csv",
            ),
        }
        self._manifest = (
            EvaluationArtifactManifest(
                run_id=_PARSED.run_id,
                configuration_id=_PARSED.configuration_id,
                manifest_content_hash="sha256:" + "b" * 64,
                artifacts=(
                    _descriptor(
                        "evaluation-report-json",
                        "json",
                        "application/json",
                        _REPORT,
                    ),
                    _descriptor(
                        "operations-summary-txt",
                        "txt",
                        "text/plain",
                        text,
                    ),
                    _descriptor(
                        "operations-summary-csv",
                        "csv",
                        "text/csv",
                        csv,
                    ),
                ),
            )
            if with_manifest
            else None
        )

    async def start(
        self,
        dataset_id: str,
        dataset_version: str | None = None,
    ) -> EvaluationRun:
        self.starts.append((dataset_id, dataset_version))
        run = EvaluationRun(
            run_id="queued-run",
            status=EvaluationRunStatus.QUEUED,
            dataset_id=dataset_id,
            dataset_version=dataset_version or "2.0.0",
            dataset_hash=_PARSED.provenance.dataset_content_hash,
            corpus_version=_PARSED.provenance.corpus_version,
            configuration_id="queued-configuration",
            code_revision="revision-v2",
            scorer_versions={"advanced": "v2"},
            cache_policy="bypass",
            total_cases=24,
        )
        self._runs.insert(0, run)
        return run

    def get_run(self, run_id: str) -> EvaluationRun | None:
        return next((item for item in self._runs if item.run_id == run_id), None)

    def list_runs(self) -> tuple[EvaluationRun, ...]:
        return tuple(self._runs)

    def datasets(self) -> tuple[EvaluationDatasetCatalogEntry, ...]:
        return (
            EvaluationDatasetCatalogEntry(
                dataset_id=_PARSED.provenance.dataset_id,
                dataset_version=_PARSED.provenance.dataset_version,
                schema_version="evaluation-dataset-v2",
                content_hash=_PARSED.provenance.dataset_content_hash,
                corpus_version=_PARSED.provenance.corpus_version,
                corpus_hash=_PARSED.provenance.corpus_content_hash,
                case_count=24,
                languages=("en", "zh-CN"),
            ),
        )

    def plans(self) -> tuple[EvaluationPlanCatalogEntry, ...]:
        return (
            EvaluationPlanCatalogEntry(
                dataset_id=_PARSED.provenance.dataset_id,
                dataset_version=_PARSED.provenance.dataset_version,
                planned_case_count=24,
                maximum_logical_calls=24,
                maximum_provider_calls=240,
                maximum_active_jobs=1,
            ),
        )

    def summary(self, run_id: str) -> EvaluationRunSummary | None:
        run = self.get_run(run_id)
        if run is None:
            return None
        available = run.status is EvaluationRunStatus.COMPLETED
        return EvaluationRunSummary.from_run(
            run,
            corpus_hash=_PARSED.provenance.corpus_content_hash,
            evidence_status=self._evidence_status if available else "unavailable",
            gate_status=self._gate_status if available else "unavailable",
        )

    def failed_cases(self, run_id: str) -> tuple[dict[str, object], ...]:
        if run_id != _PARSED.run_id:
            return ()
        return (
            {
                "case_id": "case-safe",
                "safe_error_code": "review person@example.com",
                "tags": ("grounding",),
                "metric_contributions": ("faithfulness:failed",),
                "refusal_reason": "low-confidence",
                "citation_chunk_ids": ("chunk-safe",),
                "request_id": "request-safe",
                "trace_id": "trace-safe",
                "outcome": "refusal",
                "raw_prompt": "never render this prompt",
            },
        )

    def artifact_manifest(self, run_id: str) -> EvaluationArtifactManifest | None:
        return self._manifest if run_id == _PARSED.run_id else None

    def artifact(
        self,
        run_id: str,
        artifact_id: str,
    ) -> ResolvedEvaluationArtifact | None:
        return self._artifacts.get(artifact_id) if run_id == _PARSED.run_id else None

    def compare_runs(self, baseline_run_id: str, candidate_run_id: str) -> dict[str, object]:
        return {"baseline": baseline_run_id, "candidate": candidate_run_id}


class StaticRunRepository:
    def __init__(self, run: EvaluationRun) -> None:
        self.run = run

    def create(self, run: EvaluationRun) -> None:
        self.run = run

    def get(self, run_id: str) -> EvaluationRun | None:
        return self.run if self.run.run_id == run_id else None

    def update(self, run: EvaluationRun) -> None:
        self.run = run

    def list(self) -> list[EvaluationRun]:
        return [self.run]


class StaticLegacyReportStore:
    def resolve(
        self,
        run_id: str,
        report_format: str,
    ) -> ResolvedEvaluationArtifact | None:
        if run_id != _PARSED.run_id:
            return None
        if report_format == "json":
            return ResolvedEvaluationArtifact(
                "evaluation-report-json",
                _REPORT,
                "application/json",
                "evaluation-report.json",
            )
        if report_format == "html":
            return ResolvedEvaluationArtifact(
                "evaluation-report-html",
                b"<!doctype html><title>Validated evaluation</title>",
                "text/html",
                "evaluation-report.html",
            )
        return None


def test_typed_views_render_validated_evidence_without_paths_or_private_fields() -> None:
    gateway = TypedEvaluationGateway()
    callbacks = WorkbenchCallbacks(WorkbenchServices(evaluations=gateway))

    rendered = callbacks.refresh_evaluations(BrowserSessionState.create())
    metrics = {row[0]: row for row in rendered.overview_rows}
    operations = {row[0]: row for row in rendered.operations_rows}
    quality = {row[0]: row for row in rendered.quality_rows}
    performance = {row[0]: row for row in rendered.performance_rows}
    cost = {row[0]: row for row in rendered.cost_rows}

    assert rendered.start_enabled
    assert rendered.selected_run_id == _PARSED.run_id
    assert metrics["answer-compliance"][5] == "24"
    assert metrics["all-attempt-latency-p90"][3] == "<= 10000.0"
    assert metrics["successful-only-latency-p95"][5] == "2"
    assert metrics["input-tokens"][5] == "2"
    assert metrics["cost-per-1000-successes"][5] == "2"
    assert operations["cache-hit-rate"][1].startswith("unavailable")
    assert operations["cache-hit-rate"][5].startswith("unavailable")
    assert tuple(quality) == (
        "faithfulness",
        "context-precision",
        "answer-compliance",
        "style",
        "refusal-appropriateness",
    )
    assert {row[0] for row in rendered.quality_plot_rows} == set(quality)
    assert {row[0] for row in rendered.latency_plot_rows} == {
        "all attempts",
        "successful only",
    }
    assert performance["all-attempt-latency-p95"][5] == "2"
    assert cost["cost-per-1000-logical-attempts"][5] == "2"
    assert "Quality gates / 质量门槛" in rendered.kpi_html
    assert "All-attempt P95 / 全请求 P95" in rendered.kpi_html
    assert any(
        row[0] == "cpu-utilization" and row[3] == "unavailable" for row in rendered.system_rows
    )
    assert tuple(row[0] for row in rendered.cache_rows) == (
        "cache-hits",
        "cache-eligible-lookups",
        "cache-hit-rate",
    )
    assert rendered.failure_rows[0][1] == "review [REDACTED_EMAIL]"
    assert "raw_prompt" not in repr(rendered)
    assert "never render this prompt" not in repr(rendered)
    assert "/api/v1/evaluations/acceptance-run-v2/artifacts/" in (rendered.artifact_links_markdown)
    assert "/operations-summary-txt" in rendered.operations_links_markdown
    assert "/operations-summary-csv" in rendered.operations_links_markdown
    assert "\\" not in rendered.artifact_links_markdown
    assert "D:\\" not in repr(rendered)
    assert "C:\\" not in repr(rendered)


@pytest.mark.asyncio
async def test_refresh_is_read_only_and_explicit_start_is_non_blocking_and_polled() -> None:
    gateway = TypedEvaluationGateway()
    callbacks = WorkbenchCallbacks(WorkbenchServices(evaluations=gateway))
    state = BrowserSessionState.create()
    initial = callbacks.refresh_evaluations(state)

    for _ in range(3):
        callbacks.preview_evaluation_plan(
            initial.selected_dataset_key,
            initial.selected_plan_key,
            initial.selected_run_id,
            state,
        )
    assert gateway.starts == []

    started = await callbacks.start_registered_evaluation(
        initial.selected_dataset_key,
        initial.selected_plan_key,
        state,
    )

    assert gateway.starts == [(_PARSED.provenance.dataset_id, _PARSED.provenance.dataset_version)]
    assert started.selected_run_id == "queued-run"
    assert started.poll_active
    assert "queued" in started.progress_markdown
    assert started.artifact_rows == ()
    assert "unavailable" in started.artifact_links_markdown

    restarted_browser = WorkbenchCallbacks(
        WorkbenchServices(evaluations=gateway)
    ).refresh_evaluations(BrowserSessionState.create())
    assert any(choice[1] == "queued-run" for choice in restarted_browser.run_choices)


def test_completed_legacy_evidence_uses_only_same_origin_compatibility_links() -> None:
    gateway = TypedEvaluationGateway(with_manifest=False)
    rendered = WorkbenchCallbacks(WorkbenchServices(evaluations=gateway)).refresh_evaluations(
        BrowserSessionState.create()
    )

    assert rendered.artifact_rows == ()
    assert all(row[6] == "unavailable" for row in rendered.overview_rows)
    assert all(str(row[5]).startswith("unavailable") for row in rendered.overview_rows)
    assert "Evidence unavailable" in rendered.gate_markdown
    assert "/api/v1/reports/acceptance-run-v2.json" in rendered.artifact_links_markdown
    assert "/api/v1/reports/acceptance-run-v2.html" in rendered.artifact_links_markdown
    assert "file:" not in rendered.artifact_links_markdown
    assert "\\" not in rendered.artifact_links_markdown


def test_gate_states_never_infer_success_and_poll_only_active_runs() -> None:
    passing = WorkbenchCallbacks(
        WorkbenchServices(evaluations=TypedEvaluationGateway())
    ).refresh_evaluations(BrowserSessionState.create())
    incomplete = WorkbenchCallbacks(
        WorkbenchServices(
            evaluations=TypedEvaluationGateway(
                with_manifest=False,
                evidence_status="incomplete",
                gate_status="unavailable",
            )
        )
    ).refresh_evaluations(BrowserSessionState.create())
    unavailable = WorkbenchCallbacks(
        WorkbenchServices(
            evaluations=TypedEvaluationGateway(
                with_manifest=False,
                evidence_status="unavailable",
                gate_status="unavailable",
            )
        )
    ).refresh_evaluations(BrowserSessionState.create())
    failed_run = _completed_run().model_copy(
        update={
            "status": EvaluationRunStatus.FAILED,
            "safe_error_code": "evaluation_execution_failed",
        }
    )
    failed = WorkbenchCallbacks(
        WorkbenchServices(
            evaluations=TypedEvaluationGateway(
                with_manifest=False,
                run=failed_run,
                evidence_status="unavailable",
                gate_status="unavailable",
            )
        )
    ).refresh_evaluations(BrowserSessionState.create())

    assert "PASS" in passing.gate_markdown
    assert "incomplete" in incomplete.gate_markdown
    assert "unavailable" in unavailable.gate_markdown
    assert "Run failed" in failed.gate_markdown
    assert all(not item.poll_active for item in (passing, incomplete, unavailable, failed))
    assert "compatibility" not in incomplete.artifact_links_markdown
    assert "compatibility" not in unavailable.artifact_links_markdown


def test_workbench_page_construction_is_read_only_and_has_no_compare_or_local_download(
    tmp_path: Path,
) -> None:
    gateway = TypedEvaluationGateway()
    blocks = create_workbench(
        Settings(_env_file=None, data_root=tmp_path),
        WorkbenchServices(evaluations=gateway),
    )
    config = blocks.get_config_file()
    button_values = {
        component["props"].get("value")
        for component in config["components"]
        if component["type"] == "button"
    }

    assert gateway.starts == []
    assert not any("Compare" in str(value) for value in button_values)
    assert not any(component["type"] == "downloadbutton" for component in config["components"])


def test_real_application_legacy_store_links_target_verified_api_download(
    tmp_path: Path,
) -> None:
    settings = Settings(
        _env_file=None,
        data_root=tmp_path,
        evaluation_dataset_root=_ROOT / "evaluations" / "datasets",
        workbench_enabled=False,
    )
    service = EvaluationApplicationService(
        registry=EvaluationDatasetRegistry(settings.evaluation_dataset_root),
        settings=settings,
        repository=StaticRunRepository(_completed_run()),
        run_artifacts_root=tmp_path / "runs",
        executor=object(),  # type: ignore[arg-type]
        legacy_report_store=StaticLegacyReportStore(),
    )
    rendered = WorkbenchCallbacks(WorkbenchServices(evaluations=service)).refresh_evaluations(
        BrowserSessionState.create()
    )

    assert rendered.artifact_rows == ()
    assert "/api/v1/reports/acceptance-run-v2.json" in rendered.artifact_links_markdown
    assert "/api/v1/reports/acceptance-run-v2.html" in rendered.artifact_links_markdown

    with TestClient(
        create_app(settings, evaluation_service=service),
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/api/v1/reports/acceptance-run-v2.json")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.content == _REPORT
    assert str(tmp_path) not in response.text
