from __future__ import annotations

import asyncio
import json
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rag_mvp.api.app import RuntimeState, create_app
from rag_mvp.api.qa import QARuntimeServices
from rag_mvp.config.settings import Settings
from rag_mvp.domain.ingestion import (
    ChunkLocator,
    IngestionJob,
    IngestionJobStatus,
    IngestionStage,
)
from rag_mvp.domain.qa import AnswerClaim, Citation, QAAnswer, SafeQADiagnostics
from rag_mvp.domain.retrieval import CachePolicy, RetrievalMode
from rag_mvp.observability.diagnostics import SafeRequestDiagnosticStore
from rag_mvp.observability.logging import bind_correlation_context, configure_logging
from rag_mvp.observability.metrics import RAGMetrics
from rag_mvp.observability.runtime import PipelineTelemetry
from rag_mvp.observability.tracing import RAGTracer
from rag_mvp.qa.grounding import ValidatedGroundedAnswer
from rag_mvp.qa.orchestrator import OrchestratedResponse
from rag_mvp.qa.sessions import ConversationService
from rag_mvp.qa.streaming import CompleteResponseEmitter
from rag_mvp.storage.database import Database
from rag_mvp.storage.repositories import SessionRepository

pytestmark = pytest.mark.integration

REQUEST_ID = "request-observability"
TRACE_ID = "d" * 32
RAW_EMAIL = "person@example.com"
RAW_SECRET = "password=correct-horse-battery-staple"
RAW_QUESTION = f"Private observability question for {RAW_EMAIL}; {RAW_SECRET}"
RAW_ANSWER = f"Unredacted policy answer for {RAW_EMAIL}."


class ObservableOrchestrator:
    """Small QA gateway that exercises the production stage-observer contract."""

    def __init__(self) -> None:
        self._observer: PipelineTelemetry | None = None

    def set_stage_observer(self, observer: PipelineTelemetry | None) -> None:
        self._observer = observer

    async def run(
        self,
        *,
        request_id: str,
        session_id: str,
        owner_id: str,
        question: str,
        mode: RetrievalMode | str,
        requested_language: str | None = None,
        cache_policy: CachePolicy | str = CachePolicy.USE,
    ) -> OrchestratedResponse:
        del owner_id, question, mode, requested_language, cache_policy
        if self._observer is None:
            raise RuntimeError("stage_observer_missing")
        for stage in ("retrieval", "evidence_assessment", "generation"):
            async with self._observer.stage(stage):
                await asyncio.sleep(0)

        citation = Citation(
            source_title=f"Private handbook {RAW_EMAIL}",
            document_version=1,
            chunk_id="chunk-observability",
            locator=ChunkLocator(pages=(1,)),
        )
        claim = AnswerClaim(
            text=RAW_ANSWER,
            citation_chunk_ids=(citation.chunk_id,),
        )
        diagnostics = SafeQADiagnostics(
            stage_timings_ms={
                "retrieval": 4.0,
                "evidence_assessment": 2.0,
                "generation": 6.0,
            },
            cache_status={"retrieval": "bypass"},
            model_identities={"generation": f"chat-model {RAW_EMAIL}"},
            token_counts={"generation-input": 12, "generation-output": 4},
            metadata={
                "index_revision": "revision-observability",
                "effective_mode": "hybrid",
                "private_note": RAW_SECRET,
            },
        )
        response = QAAnswer(
            request_id=request_id,
            session_id=session_id,
            response_language="en",
            answer=RAW_ANSWER,
            claims=(claim,),
            citations=(citation,),
            diagnostics=diagnostics,
        )
        grounded = ValidatedGroundedAnswer(
            request_id=request_id,
            revision_id="revision-observability",
            answer=RAW_ANSWER,
            claims=(claim,),
            citations=(citation,),
        )
        return OrchestratedResponse._create(response, grounded_answer=grounded)


@dataclass(frozen=True, slots=True)
class ObservabilityHarness:
    settings: Settings
    runtime: RuntimeState
    diagnostics: SafeRequestDiagnosticStore
    logs: StringIO
    exporter: InMemorySpanExporter


def _harness(tmp_path: Path) -> tuple[FastAPI, ObservabilityHarness]:
    settings = Settings(
        data_root=tmp_path / "data",
        environment="test",
        workbench_enabled=False,
        _env_file=None,
    )
    database = Database(tmp_path / "runtime.sqlite3")
    database.initialize()
    conversations = ConversationService(SessionRepository(database))
    diagnostics = SafeRequestDiagnosticStore(database)
    services = QARuntimeServices(
        conversations=conversations,
        orchestrator=ObservableOrchestrator(),
        emitter=CompleteResponseEmitter(conversations),
    )
    logs = StringIO()
    with redirect_stdout(logs):
        app = create_app(
            settings,
            qa_services=services,
            diagnostics_service=diagnostics,
        )
    runtime = cast(RuntimeState, app.state.runtime)
    assert runtime.telemetry is not None

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    runtime.telemetry.tracer = RAGTracer(provider.get_tracer("observability-integration"))
    return app, ObservabilityHarness(settings, runtime, diagnostics, logs, exporter)


def _json_logs(stream: StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


def test_qa_telemetry_is_correlated_across_all_surfaces_and_content_free(
    tmp_path: Path,
) -> None:
    app, harness = _harness(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/qa",
            headers={"x-request-id": REQUEST_ID, "x-trace-id": TRACE_ID},
            json={
                "owner_id": "owner-observability",
                "question": RAW_QUESTION,
                "mode": "hybrid",
                "requested_language": "en",
            },
        )
        metrics_text = client.get("/metrics").text

    assert response.status_code == 200
    event = json.loads(response.text)
    assert event["request_id"] == REQUEST_ID
    assert event["kind"] == "answer"
    assert RAW_EMAIL not in response.text

    spans = harness.exporter.get_finished_spans()
    spans_by_name = {span.name: span for span in spans}
    expected_names = {
        "rag.request",
        "rag.stage.retrieval",
        "rag.stage.evidence-assessment",
        "rag.stage.generation",
    }
    assert expected_names.issubset(spans_by_name)
    assert {
        "rag.stage.queue",
        "rag.stage.safety",
        "rag.stage.redaction",
        "rag.stage.serialization",
    }.issubset(spans_by_name)
    root = spans_by_name["rag.request"]
    assert f"{root.context.trace_id:032x}" == TRACE_ID
    assert root.attributes["rag.request.id"] == REQUEST_ID
    assert root.attributes["rag.outcome"] == "answer"
    for name in expected_names - {"rag.request"}:
        span = spans_by_name[name]
        assert f"{span.context.trace_id:032x}" == TRACE_ID
        assert span.parent is not None
        assert span.parent.span_id == root.context.span_id

    logs = _json_logs(harness.logs)
    qa_logs = [entry for entry in logs if str(entry.get("event", "")).startswith("qa.")]
    assert qa_logs
    assert {entry["request_id"] for entry in qa_logs} == {REQUEST_ID}
    assert {entry["trace_id"] for entry in qa_logs} == {TRACE_ID}
    stage_logs = {
        cast(str, entry["stage"]) for entry in qa_logs if entry.get("event") == "qa.stage.completed"
    }
    assert {
        "retrieval",
        "evidence-assessment",
        "generation",
        "queue",
        "safety",
        "redaction",
        "serialization",
    }.issubset(stage_logs)

    assert 'rag_mvp_qa_requests_total{error_category="none",outcome="answer"} 1.0' in metrics_text
    for stage in ("retrieval", "evidence-assessment", "generation"):
        assert f'rag_mvp_stage_duration_seconds_count{{stage="{stage}"}} 1.0' in metrics_text
    assert 'rag_mvp_cache_access_total{cache="retrieval",outcome="bypass"} 1.0' in metrics_text
    assert 'rag_mvp_provider_tokens_total{direction="input",role="generation"} 12.0' in metrics_text
    assert "rag_mvp_qa_in_flight 0.0" in metrics_text

    diagnostic = harness.diagnostics.get(REQUEST_ID)
    assert diagnostic is not None
    assert diagnostic.request_id == REQUEST_ID
    assert diagnostic.trace_id == TRACE_ID
    assert diagnostic.outcome == "answer"
    assert {
        key: diagnostic.stage_timings_ms[key]
        for key in ("retrieval", "evidence_assessment", "generation")
    } == {
        "retrieval": 4.0,
        "evidence_assessment": 2.0,
        "generation": 6.0,
    }
    assert set(diagnostic.stage_timings_ms) <= {
        "retrieval",
        "evidence_assessment",
        "generation",
        "queue",
        "safety",
        "redaction",
        "serialization",
    }
    assert diagnostic.model_identities == {"generation": "chat-model [REDACTED_EMAIL]"}
    assert diagnostic.token_counts == {"generation-input": 12, "generation-output": 4}

    span_evidence = repr(
        [
            {
                "name": span.name,
                "attributes": dict(span.attributes),
                "events": tuple(span.events),
            }
            for span in spans
        ]
    )
    diagnostic_evidence = diagnostic.model_dump_json()
    telemetry_surfaces = "\n".join(
        (harness.logs.getvalue(), metrics_text, span_evidence, diagnostic_evidence)
    )
    for prohibited in (RAW_QUESTION, RAW_ANSWER, RAW_EMAIL, RAW_SECRET):
        assert prohibited not in telemetry_surfaces


@pytest.mark.asyncio
async def test_ingestion_completion_and_evaluation_hook_are_safe_and_timed(
    tmp_path: Path,
) -> None:
    settings = Settings(
        data_root=tmp_path / "data",
        environment="test",
        workbench_enabled=False,
        _env_file=None,
    )
    logs = StringIO()
    configure_logging(
        service=settings.service_name,
        service_version=settings.service_version,
        config_version=settings.configuration_identity,
        stream=logs,
    )
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    metrics = RAGMetrics()
    telemetry = PipelineTelemetry(
        settings,
        metrics=metrics,
        tracer=RAGTracer(provider.get_tracer("background-observability")),
    )
    sensitive_source = f"source-{RAW_EMAIL}-{RAW_SECRET}"
    telemetry.record_ingestion(
        IngestionJob(
            job_id="job-observability",
            source_key=sensitive_source,
            status=IngestionJobStatus.SUCCEEDED,
            stage=IngestionStage.COMPLETE,
            chunk_count=7,
            ocr_page_count=2,
            stage_timings_ms={"validating": 2.0, "complete": 3.0},
        )
    )

    evaluation_request_id = "evaluation-observability"
    evaluation_trace_id = "e" * 32
    with bind_correlation_context(evaluation_request_id, evaluation_trace_id):
        async with telemetry.tracer.request_span(
            request_id=evaluation_request_id,
            operation="evaluation",
            run_id="run-observability",
            trace_id=evaluation_trace_id,
        ):
            async with telemetry.stage("evaluation"):
                await asyncio.sleep(0)

    rendered_metrics = metrics.render().decode("utf-8")
    assert 'rag_mvp_stage_duration_seconds_count{stage="ingestion"} 1.0' in rendered_metrics
    events = _json_logs(logs)
    ingestion_event = next(event for event in events if event["event"] == "ingestion.job.completed")
    assert ingestion_event["outcome"] == "succeeded"
    assert ingestion_event["counts"] == {"chunks": 7, "ocr_pages": 2}
    evaluation_event = next(event for event in events if event["event"] == "qa.stage.completed")
    assert evaluation_event["stage"] == "evaluation"
    assert evaluation_event["request_id"] == evaluation_request_id
    assert evaluation_event["trace_id"] == evaluation_trace_id

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert set(spans) == {"rag.request", "rag.stage.evaluation"}
    assert spans["rag.stage.evaluation"].parent is not None
    assert spans["rag.stage.evaluation"].parent.span_id == spans["rag.request"].context.span_id
    background_surfaces = logs.getvalue() + rendered_metrics + repr(spans)
    assert sensitive_source not in background_surfaces
    assert RAW_EMAIL not in background_surfaces
    assert RAW_SECRET not in background_surfaces
