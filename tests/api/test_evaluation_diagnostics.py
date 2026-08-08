from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from rag_mvp.api.app import create_app
from rag_mvp.api.evaluation_diagnostics import ReportArtifact, ReportFormat
from rag_mvp.config.settings import Settings
from rag_mvp.domain.evaluation import EvaluationRun
from rag_mvp.domain.qa import RequestDiagnostic
from rag_mvp.evaluation.application import (
    EvaluationArtifactDescriptor,
    EvaluationArtifactManifest,
    EvaluationCapacityError,
    EvaluationConflictError,
    EvaluationDatasetCatalogEntry,
    EvaluationPlanCatalogEntry,
    EvaluationRunSummary,
    FailedCaseDiagnostic,
    FailedMetricContribution,
    ResolvedEvaluationArtifact,
)


def _run(run_id: str = "run_test") -> EvaluationRun:
    return EvaluationRun(
        run_id=run_id,
        dataset_id="mvp-v1",
        dataset_version="1.0.0",
        dataset_hash="dataset_hash_1234",
        corpus_version="corpus-v1",
        configuration_id="config_test",
        code_revision="revision_test",
        scorer_versions={"faithfulness": "v1"},
        cache_policy="bypass-final",
        total_cases=7,
    )


@dataclass
class FakeEvaluationService:
    runs: dict[str, EvaluationRun] = field(default_factory=dict)
    reports: dict[tuple[str, str], ReportArtifact] = field(default_factory=dict)
    starts: list[tuple[str, str]] = field(default_factory=list)
    start_error: RuntimeError | None = None
    artifact_manifest_value: EvaluationArtifactManifest | None = None
    artifacts: dict[str, ResolvedEvaluationArtifact] = field(default_factory=dict)

    async def start(self, *, dataset_id: str, dataset_version: str) -> EvaluationRun:
        if self.start_error is not None:
            raise self.start_error
        self.starts.append((dataset_id, dataset_version))
        run = _run()
        self.runs[run.run_id] = run
        return run

    def get(self, run_id: str) -> EvaluationRun | None:
        return self.runs.get(run_id)

    def list(self) -> tuple[EvaluationRun, ...]:
        return tuple(self.runs.values())

    def datasets(self) -> tuple[EvaluationDatasetCatalogEntry, ...]:
        return (
            EvaluationDatasetCatalogEntry(
                dataset_id="mvp-v1",
                dataset_version="1.0.0",
                schema_version="rag-evaluation-dataset-v1",
                content_hash="sha256:" + "1" * 64,
                corpus_version="1.0.0",
                corpus_hash="sha256:" + "2" * 64,
                case_count=7,
                languages=("en", "zh"),
            ),
        )

    def plans(self) -> tuple[EvaluationPlanCatalogEntry, ...]:
        return (
            EvaluationPlanCatalogEntry(
                dataset_id="mvp-v1",
                dataset_version="1.0.0",
                planned_case_count=7,
                maximum_logical_calls=7,
                maximum_provider_calls=70,
                maximum_active_jobs=1,
            ),
        )

    def summary(self, run_id: str) -> EvaluationRunSummary | None:
        run = self.get(run_id)
        if run is None:
            return None
        return EvaluationRunSummary.from_run(
            run,
            corpus_hash="sha256:" + "2" * 64,
            evidence_status="unavailable",
            gate_status="unavailable",
        )

    def failed_cases(self, run_id: str) -> tuple[FailedCaseDiagnostic, ...]:
        if self.get(run_id) is None:
            return ()
        return (
            FailedCaseDiagnostic(
                case_id="case-safe",
                safe_error_code="case_execution_failed",
                request_id="request-safe",
                trace_id="trace-safe",
                outcome="error",
                citation_chunk_ids=("chunk-safe",),
                tags=("rerank-sensitive",),
                metric_contributions=(
                    FailedMetricContribution(
                        metric_id="answer-compliance",
                        status="failed",
                        value=0.5,
                        numerator=1,
                        denominator=2,
                    ),
                ),
            ),
        )

    def artifact_manifest(self, run_id: str) -> EvaluationArtifactManifest | None:
        return self.artifact_manifest_value if self.get(run_id) is not None else None

    def artifact(self, run_id: str, artifact_id: str) -> ResolvedEvaluationArtifact | None:
        return self.artifacts.get(artifact_id) if self.get(run_id) is not None else None

    def report(self, run_id: str, report_format: ReportFormat) -> ReportArtifact | None:
        return self.reports.get((run_id, report_format))


@dataclass
class FakeDiagnosticService:
    diagnostics: dict[str, RequestDiagnostic] = field(default_factory=dict)

    def get(self, request_id: str) -> RequestDiagnostic | None:
        return self.diagnostics.get(request_id)


def _client(
    tmp_path: object,
    *,
    evaluations: FakeEvaluationService | None = None,
    evaluation_profiles: dict[str, FakeEvaluationService] | None = None,
    diagnostics: FakeDiagnosticService | None = None,
) -> TestClient:
    settings = Settings(
        _env_file=None,
        data_root=tmp_path,
        workbench_enabled=False,
    )
    return TestClient(
        create_app(
            settings,
            evaluation_service=evaluations,
            evaluation_profile_services=evaluation_profiles,
            diagnostics_service=diagnostics,
        )
    )


def test_start_and_get_evaluation(tmp_path: object) -> None:
    service = FakeEvaluationService()
    with _client(tmp_path, evaluations=service) as client:
        started = client.post(
            "/api/v1/evaluations",
            json={"dataset_id": "mvp-v1", "dataset_version": "1.0.0"},
        )
        fetched = client.get("/api/v1/evaluations/run_test")

    assert started.status_code == 202
    assert started.headers["location"] == "/api/v1/evaluations/run_test"
    assert started.headers["cache-control"] == "no-store"
    assert started.json()["status"] == "queued"
    assert service.starts == [("mvp-v1", "1.0.0")]
    assert fetched.status_code == 200
    assert fetched.json()["total_cases"] == 7


def test_evaluation_routes_select_an_isolated_retrieval_profile(tmp_path: object) -> None:
    openai = FakeEvaluationService(runs={"run_openai": _run("run_openai")})
    bge = FakeEvaluationService(runs={"run_bge": _run("run_bge")})
    with _client(
        tmp_path,
        evaluations=openai,
        evaluation_profiles={"openai-api": openai, "bge-local": bge},
    ) as client:
        default_runs = client.get("/api/v1/evaluations")
        bge_runs = client.get("/api/v1/evaluations?retrieval_profile=bge-local")
        started = client.post(
            "/api/v1/evaluations?retrieval_profile=bge-local",
            json={"dataset_id": "mvp-v1", "dataset_version": "1.0.0"},
        )
        unknown = client.get("/api/v1/evaluations?retrieval_profile=unknown")

    assert default_runs.json()["runs"][0]["run_id"] == "run_openai"
    assert bge_runs.json()["runs"][0]["run_id"] == "run_bge"
    assert started.status_code == 202
    assert started.headers["location"] == (
        "/api/v1/evaluations/run_test?retrieval_profile=bge-local"
    )
    assert bge.starts == [("mvp-v1", "1.0.0")]
    assert openai.starts == []
    assert unknown.status_code == 503
    assert unknown.json() == {"error": {"code": "evaluation_unavailable"}}


def test_evaluation_routes_fail_safely_when_unavailable_or_missing(tmp_path: object) -> None:
    with _client(tmp_path) as client:
        unavailable = client.post(
            "/api/v1/evaluations",
            json={"dataset_id": "mvp-v1", "dataset_version": "1.0.0"},
        )

    with _client(tmp_path, evaluations=FakeEvaluationService()) as client:
        missing = client.get("/api/v1/evaluations/unknown")

    assert unavailable.status_code == 503
    assert unavailable.json() == {"error": {"code": "evaluation_unavailable"}}
    assert missing.status_code == 404
    assert missing.json() == {"error": {"code": "evaluation_not_found"}}


def test_catalog_list_summary_and_failed_cases_are_typed_no_store_and_path_free(
    tmp_path: object,
) -> None:
    service = FakeEvaluationService(runs={"run_test": _run()})
    with _client(tmp_path, evaluations=service) as client:
        datasets = client.get("/api/v1/evaluation-datasets")
        plans = client.get("/api/v1/evaluation-plans")
        runs = client.get("/api/v1/evaluations")
        summary = client.get("/api/v1/evaluations/run_test/summary")
        failed = client.get("/api/v1/evaluations/run_test/failed-cases")

    for response in (datasets, plans, runs, summary, failed):
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "relative_path" not in response.text
        assert str(tmp_path) not in response.text
    assert plans.json()["plans"][0]["maximum_logical_calls"] == 7
    assert plans.json()["plans"][0]["cost_estimate_status"] == "unavailable"
    assert summary.json()["evidence_status"] == "unavailable"
    assert summary.json()["gate_status"] == "unavailable"
    assert failed.json()["cases"][0]["tags"] == ["rerank-sensitive"]
    assert failed.json()["cases"][0]["metric_contributions"][0]["metric_id"] == (
        "answer-compliance"
    )


def test_start_maps_duplicate_and_capacity_to_stable_errors(tmp_path: object) -> None:
    duplicate = FakeEvaluationService(start_error=EvaluationConflictError("evaluation_duplicate"))
    capacity = FakeEvaluationService(start_error=EvaluationCapacityError("evaluation_capacity"))
    payload = {"dataset_id": "mvp-v1", "dataset_version": "1.0.0"}

    with _client(tmp_path, evaluations=duplicate) as client:
        duplicate_response = client.post("/api/v1/evaluations", json=payload)
    with _client(tmp_path, evaluations=capacity) as client:
        capacity_response = client.post("/api/v1/evaluations", json=payload)

    assert duplicate_response.status_code == 409
    assert duplicate_response.json() == {"error": {"code": "evaluation_duplicate"}}
    assert capacity_response.status_code == 429
    assert capacity_response.json() == {"error": {"code": "evaluation_capacity"}}


def test_opaque_artifact_manifest_and_jsonl_download_never_expose_path(
    tmp_path: object,
) -> None:
    now = datetime.now(UTC)
    descriptor = EvaluationArtifactDescriptor(
        artifact_id="privacy-safe-log-sample",
        schema_version="privacy-safe-log-sample-v1",
        format="jsonl",
        media_type="application/x-ndjson",
        sha256_digest="sha256:" + "a" * 64,
        byte_size=17,
        created_at=now,
    )
    service = FakeEvaluationService(
        runs={"run_test": _run()},
        artifact_manifest_value=EvaluationArtifactManifest(
            run_id="run_test",
            configuration_id="config_test",
            manifest_content_hash="sha256:" + "b" * 64,
            artifacts=(descriptor,),
        ),
        artifacts={
            descriptor.artifact_id: ResolvedEvaluationArtifact(
                artifact_id=descriptor.artifact_id,
                content=b'{"event":"safe"}\n',
                media_type=descriptor.media_type,
                filename="privacy-safe-sample-v1.jsonl",
            )
        },
    )
    with _client(tmp_path, evaluations=service) as client:
        manifest = client.get("/api/v1/evaluations/run_test/artifacts")
        download = client.get("/api/v1/evaluations/run_test/artifacts/privacy-safe-log-sample")

    assert manifest.status_code == 200
    assert "relative_path" not in manifest.text
    assert str(tmp_path) not in manifest.text
    assert download.status_code == 200
    assert download.headers["cache-control"] == "no-store"
    assert download.headers["x-content-type-options"] == "nosniff"
    assert download.headers["content-type"].startswith("application/x-ndjson")
    assert download.headers["content-disposition"] == (
        'attachment; filename="privacy-safe-sample-v1.jsonl"'
    )


def test_artifact_download_rejects_media_type_valid_for_a_different_artifact(
    tmp_path: object,
) -> None:
    artifact_id = "privacy-safe-log-sample"
    service = FakeEvaluationService(
        runs={"run_test": _run()},
        artifacts={
            artifact_id: ResolvedEvaluationArtifact(
                artifact_id=artifact_id,
                content=b"<!doctype html><title>wrong contract</title>",
                media_type="text/html",
                filename="privacy-safe-sample-v1.jsonl",
            )
        },
    )

    with _client(tmp_path, evaluations=service) as client:
        response = client.get("/api/v1/evaluations/run_test/artifacts/privacy-safe-log-sample")

    assert response.status_code == 503
    assert response.json() == {"error": {"code": "evaluation_artifact_unavailable"}}


def test_download_reports_uses_validated_content_contract(tmp_path: object) -> None:
    service = FakeEvaluationService(
        runs={"run_test": _run()},
        reports={
            ("run_test", "json"): ReportArtifact(
                b'{"run_id":"run_test"}',
                "application/json",
                "run_test.json",
            ),
            ("run_test", "html"): ReportArtifact(
                b"<!doctype html><title>Safe report</title>",
                "text/html",
                "run_test.html",
            ),
        },
    )
    with _client(tmp_path, evaluations=service) as client:
        json_response = client.get("/api/v1/reports/run_test.json")
        html_response = client.get("/api/v1/reports/run_test.html")
        invalid = client.get("/api/v1/reports/run_test.xml")
        v2_manifest = client.get("/api/v1/evaluations/run_test/artifacts")

    assert json_response.status_code == 200
    assert json_response.json() == {"run_id": "run_test"}
    assert json_response.headers["content-disposition"] == 'attachment; filename="run_test.json"'
    assert html_response.status_code == 200
    assert html_response.headers["content-type"].startswith("text/html")
    assert invalid.status_code == 422
    assert invalid.json() == {"error": {"code": "request_invalid"}}
    assert v2_manifest.status_code == 404
    assert v2_manifest.json() == {"error": {"code": "evaluation_artifact_manifest_not_found"}}


def test_diagnostic_is_allowlisted_redacted_and_not_cached(tmp_path: object) -> None:
    now = datetime.now(UTC)
    diagnostic = RequestDiagnostic(
        request_id="request_test",
        session_id="session_test",
        trace_id="trace_test",
        outcome="answer",
        stage_timings_ms={"generation": 12.5},
        cache_status={"retrieval": "miss"},
        model_identities={"generation": "model-safe"},
        token_counts={"input": 12, "output": 3},
        metadata={
            "index_revision": "person@example.com",
            "question": "do not expose person@example.com",
        },
        created_at=now,
        expires_at=now + timedelta(hours=1),
    )
    service = FakeDiagnosticService({diagnostic.request_id: diagnostic})
    with _client(tmp_path, diagnostics=service) as client:
        response = client.get("/api/v1/diagnostics/requests/request_test")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    serialized = response.text
    assert "person@example.com" not in serialized
    assert "question" not in serialized
    assert response.json()["metadata"]["index_revision"] == "[REDACTED_EMAIL]"


def test_missing_diagnostic_and_disabled_reader_are_distinct(tmp_path: object) -> None:
    with _client(tmp_path) as client:
        unavailable = client.get("/api/v1/diagnostics/requests/request_test")
    with _client(tmp_path, diagnostics=FakeDiagnosticService()) as client:
        missing = client.get("/api/v1/diagnostics/requests/request_test")

    assert unavailable.status_code == 503
    assert unavailable.json() == {"error": {"code": "diagnostics_unavailable"}}
    assert missing.status_code == 404
    assert missing.json() == {"error": {"code": "diagnostic_not_found"}}
