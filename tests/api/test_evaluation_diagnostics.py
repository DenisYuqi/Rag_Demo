from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from rag_mvp.api.app import create_app
from rag_mvp.api.evaluation_diagnostics import ReportArtifact, ReportFormat
from rag_mvp.config.settings import Settings
from rag_mvp.domain.evaluation import EvaluationRun
from rag_mvp.domain.qa import RequestDiagnostic


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

    async def start(self, *, dataset_id: str, dataset_version: str) -> EvaluationRun:
        self.starts.append((dataset_id, dataset_version))
        run = _run()
        self.runs[run.run_id] = run
        return run

    def get(self, run_id: str) -> EvaluationRun | None:
        return self.runs.get(run_id)

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


def test_download_reports_uses_validated_content_contract(tmp_path: object) -> None:
    service = FakeEvaluationService(
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
        }
    )
    with _client(tmp_path, evaluations=service) as client:
        json_response = client.get("/api/v1/reports/run_test.json")
        html_response = client.get("/api/v1/reports/run_test.html")
        invalid = client.get("/api/v1/reports/run_test.xml")

    assert json_response.status_code == 200
    assert json_response.json() == {"run_id": "run_test"}
    assert json_response.headers["content-disposition"] == 'attachment; filename="run_test.json"'
    assert html_response.status_code == 200
    assert html_response.headers["content-type"].startswith("text/html")
    assert invalid.status_code == 422
    assert invalid.json() == {"error": {"code": "request_invalid"}}


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
