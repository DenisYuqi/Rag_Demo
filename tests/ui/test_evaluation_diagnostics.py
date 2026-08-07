from __future__ import annotations

from pathlib import Path

import pytest

from rag_mvp.domain.evaluation import EvaluationRun, EvaluationRunStatus
from rag_mvp.domain.qa import RequestDiagnostic
from rag_mvp.ui.callbacks import WorkbenchCallbacks
from rag_mvp.ui.models import BrowserSessionState
from rag_mvp.ui.services import (
    EvaluationCompatibilityError,
    HealthComponent,
    WorkbenchServices,
)

pytestmark = pytest.mark.ui


def _run(run_id: str, *, configuration_id: str = "config_accepted") -> EvaluationRun:
    return EvaluationRun(
        run_id=run_id,
        status=EvaluationRunStatus.COMPLETED,
        dataset_id="mvp-v1",
        dataset_version="1.0.0",
        dataset_hash="dataset_hash_1234",
        corpus_version="corpus-v1",
        configuration_id=configuration_id,
        code_revision="revision_test",
        scorer_versions={"faithfulness": "v1"},
        cache_policy="bypass-final",
        total_cases=3,
        completed_cases=2,
        failed_cases=1,
    )


class FakeEvaluationGateway:
    def __init__(self, root: Path) -> None:
        self.runs = {
            "run_baseline": _run("run_baseline"),
            "run_candidate": _run("run_candidate"),
            "run_incompatible": _run(
                "run_incompatible",
                configuration_id="config_incompatible",
            ),
        }
        self.starts: list[tuple[str, str | None]] = []
        del root

    async def start(
        self,
        dataset_id: str,
        dataset_version: str | None = None,
    ) -> EvaluationRun:
        self.starts.append((dataset_id, dataset_version))
        run = _run("run_started")
        self.runs[run.run_id] = run
        return run

    def get_run(self, run_id: str) -> EvaluationRun | None:
        return self.runs.get(run_id)

    def list_runs(self) -> tuple[EvaluationRun, ...]:
        return tuple(self.runs.values())

    def failed_cases(self, run_id: str) -> tuple[dict[str, object], ...]:
        if run_id not in self.runs:
            return ()
        return (
            {
                "case_id": "case_03",
                "safe_reason": "Contact person@example.com was not grounded.",
            },
        )

    def compare_runs(
        self,
        baseline_run_id: str,
        candidate_run_id: str,
    ) -> dict[str, object]:
        baseline = self.runs[baseline_run_id]
        candidate = self.runs[candidate_run_id]
        if baseline.configuration_id != candidate.configuration_id:
            raise EvaluationCompatibilityError()
        return {
            "baseline_run_id": baseline_run_id,
            "candidate_run_id": candidate_run_id,
            "faithfulness_delta": 0.08,
            "note": "Reviewed by person@example.com",
        }


class FakeDiagnosticsGateway:
    def __init__(self, diagnostic: RequestDiagnostic) -> None:
        self.diagnostic = diagnostic

    def health(self) -> tuple[HealthComponent, ...]:
        return (
            HealthComponent("storage", True),
            HealthComponent("generation", False, "provider person@example.com unavailable"),
        )

    def get_request(self, request_id: str) -> RequestDiagnostic | None:
        if request_id == self.diagnostic.request_id:
            return self.diagnostic
        return None


@pytest.mark.asyncio
async def test_legacy_evaluation_callbacks_are_redacted(tmp_path: Path) -> None:
    gateway = FakeEvaluationGateway(tmp_path)
    callbacks = WorkbenchCallbacks(WorkbenchServices(evaluations=gateway))
    state = BrowserSessionState.create()

    started = await callbacks.start_evaluation("mvp-v1", "1.0.0", state)

    assert gateway.starts == [("mvp-v1", "1.0.0")]
    assert started.state.evaluation_run_id == "run_started"
    assert any(row[0] == "run_started" for row in started.run_rows)
    assert started.failure_rows == (("case_03", "Contact [REDACTED_EMAIL] was not grounded."),)
    assert "person@example.com" not in repr(started)

    compared = callbacks.compare_evaluations("run_baseline", "run_candidate", state)

    assert "faithfulness_delta" in compared.metrics_markdown
    assert "0.08" in compared.metrics_markdown
    assert "Reviewed by [REDACTED_EMAIL]" in compared.metrics_markdown
    assert "person@example.com" not in repr(compared)
    assert "Compatible runs compared" in compared.status_markdown

    incompatible = callbacks.compare_evaluations("run_baseline", "run_incompatible", state)
    assert incompatible.metrics_markdown == ""
    assert "Runs are incompatible" in incompatible.status_markdown


def test_diagnostics_health_and_request_trace_are_allowlisted_and_redacted() -> None:
    diagnostic = RequestDiagnostic(
        request_id="request_test",
        session_id="session_test",
        trace_id="trace_test",
        outcome="answer",
        stage_timings_ms={"retrieval": 3.5},
        cache_status={"retrieval": "miss"},
        model_identities={"generation": "model person@example.com"},
        token_counts={"input": 11, "output": 4},
        metadata={
            "index_revision": "revision person@example.com",
            "citation_count": 2,
            "question": "Never expose person@example.com",
            "credential": "not-an-output-field",
        },
    )
    callbacks = WorkbenchCallbacks(
        WorkbenchServices(diagnostics=FakeDiagnosticsGateway(diagnostic))
    )

    health = callbacks.refresh_health()
    request = callbacks.inspect_request("request_test")
    request_values = dict(request.request_rows)

    assert health.health_rows == (
        ("storage", True, ""),
        ("generation", False, "provider [REDACTED_EMAIL] unavailable"),
    )
    assert "person@example.com" not in repr(health)
    assert request_values["request_id"] == "request_test"
    assert request_values["stage_timings_ms"] == {"retrieval": 3.5}
    assert request_values["model_identities"] == {"generation": "model [REDACTED_EMAIL]"}
    assert request_values["metadata"] == {
        "index_revision": "revision [REDACTED_EMAIL]",
        "citation_count": 2,
    }
    assert "question" not in repr(request.request_rows)
    assert "credential" not in repr(request.request_rows)
    assert "person@example.com" not in repr(request)


def test_missing_diagnostic_is_distinct_from_an_unavailable_backend() -> None:
    diagnostic = RequestDiagnostic(request_id="request_test", outcome="answer")
    available = WorkbenchCallbacks(
        WorkbenchServices(diagnostics=FakeDiagnosticsGateway(diagnostic))
    )
    unavailable = WorkbenchCallbacks(WorkbenchServices())

    missing = available.inspect_request("request_missing")
    disabled = unavailable.inspect_request("request_missing")

    assert "Request not found" in missing.status_markdown
    assert "capability is unavailable" in disabled.status_markdown
