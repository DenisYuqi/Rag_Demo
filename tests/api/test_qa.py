from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_mvp.api.app import create_app
from rag_mvp.api.qa import NDJSON_MEDIA_TYPE, QARuntimeServices, stream_qa_events
from rag_mvp.config.settings import Settings
from rag_mvp.domain.ingestion import ChunkLocator
from rag_mvp.domain.qa import (
    AnswerClaim,
    Citation,
    QAAnswer,
    QAError,
    QAErrorCode,
    QARefusal,
    RefusalReason,
    SafeQADiagnostics,
)
from rag_mvp.observability.runtime import PipelineTelemetry
from rag_mvp.performance.admission import QAAdmissionController
from rag_mvp.performance.deadlines import QALatencyBudgets
from rag_mvp.qa.grounding import ValidatedGroundedAnswer
from rag_mvp.qa.orchestrator import OrchestratedResponse
from rag_mvp.qa.sessions import ConversationService
from rag_mvp.qa.streaming import CompleteResponseEmitter
from rag_mvp.safety.models import RedactionResult
from rag_mvp.safety.output import SAFE_UNAVAILABLE_MESSAGE
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, Redactor
from rag_mvp.storage.database import Database
from rag_mvp.storage.repositories import SessionRepository

pytestmark = pytest.mark.api


class ScriptedOrchestrator:
    def __init__(self, outcome: str) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, object]] = []

    async def run(self, **kwargs: object) -> OrchestratedResponse:
        self.calls.append(kwargs)
        request_id = cast(str, kwargs["request_id"])
        session_id = cast(str, kwargs["session_id"])
        diagnostics = SafeQADiagnostics(
            trace_id="trace-safe",
            stage_timings_ms={"retrieval": 4.5},
            metadata={"notice": "person@example.com"},
        )
        citation = Citation(
            source_title="Handbook owner@example.com",
            document_version=2,
            chunk_id="chunk-1",
            locator=ChunkLocator(pages=(7,)),
        )
        if self.outcome == "answer":
            answer = "Employees receive ten days. Contact person@example.com."
            claims = (AnswerClaim(text=answer, citation_chunk_ids=(citation.chunk_id,)),)
            grounded = ValidatedGroundedAnswer(
                request_id=request_id,
                revision_id="revision-current",
                answer=answer,
                claims=claims,
                citations=(citation,),
            )
            response = QAAnswer(
                request_id=request_id,
                session_id=session_id,
                response_language="en",
                answer=answer,
                claims=claims,
                citations=(citation,),
                diagnostics=diagnostics,
            )
            return OrchestratedResponse._create(response, grounded_answer=grounded)
        if self.outcome == "refusal":
            return OrchestratedResponse._create(
                QARefusal(
                    request_id=request_id,
                    session_id=session_id,
                    response_language="en",
                    message="The available evidence is insufficient.",
                    reason=RefusalReason.INSUFFICIENT_EVIDENCE,
                    diagnostics=diagnostics,
                )
            )
        return OrchestratedResponse._create(
            QAError(
                request_id=request_id,
                session_id=session_id,
                response_language="en",
                message="The request reached its deadline.",
                code=QAErrorCode.DEADLINE_EXPIRED,
                retryable=True,
                diagnostics=diagnostics,
            )
        )


class ExplodingOrchestrator:
    async def run(self, **kwargs: object) -> OrchestratedResponse:
        del kwargs
        raise RuntimeError("private person@example.com must not cross the API")


class BlockingOrchestrator:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def run(self, **kwargs: object) -> OrchestratedResponse:
        del kwargs
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("blocking orchestrator unexpectedly returned")


class DelayedEmitter:
    def __init__(self, delegate: CompleteResponseEmitter, delay_seconds: float) -> None:
        self._delegate = delegate
        self._delay_seconds = delay_seconds

    @property
    def ready(self) -> bool:
        return self._delegate.ready

    def emit(self, outcome: OrchestratedResponse, *, owner_id: str) -> object:
        time.sleep(self._delay_seconds)
        return self._delegate.emit(outcome, owner_id=owner_id)


class DelayedRedactor(Redactor):
    def __init__(self, delay_seconds: float) -> None:
        super().__init__()
        self._delay_seconds = delay_seconds

    def redact(self, text: str) -> RedactionResult:
        time.sleep(self._delay_seconds)
        return super().redact(text)


class MalformedEmitter:
    ready = True

    def emit(self, outcome: OrchestratedResponse, *, owner_id: str) -> object:
        del owner_id
        response = outcome.response
        return (
            {
                "request_id": response.request_id,
                "session_id": response.session_id,
                "sequence": 1,
                "kind": "error",
                "response_language": response.response_language,
                "content": "private person@example.com",
                "error_code": "internal",
                "retryable": False,
                "terminal": True,
            },
        )


class SplitSensitiveEmitter:
    ready = True

    def emit(self, outcome: OrchestratedResponse, *, owner_id: str) -> object:
        del owner_id
        response = outcome.response
        citation = {
            "source_title": "Handbook",
            "document_version": 1,
            "chunk_id": "chunk-1",
            "locator": {
                "pages": [1],
                "section_path": [],
                "character_start": None,
                "character_end": None,
            },
        }

        def sentence(sequence: int, content: str) -> dict[str, object]:
            return {
                "request_id": response.request_id,
                "session_id": response.session_id,
                "sequence": sequence,
                "kind": "sentence",
                "response_language": response.response_language,
                "content": content,
                "claims": [{"text": content, "citation_chunk_ids": ["chunk-1"]}],
                "citations": [citation],
                "terminal": False,
            }

        return (
            sentence(0, "Contact person@"),
            sentence(1, "example.com"),
            {
                "request_id": response.request_id,
                "session_id": response.session_id,
                "sequence": 2,
                "kind": "done",
                "response_language": response.response_language,
                "terminal": True,
            },
        )


@dataclass(frozen=True, slots=True)
class QAHarness:
    client: TestClient
    conversations: ConversationService
    orchestrator: ScriptedOrchestrator


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_root=tmp_path / "data", environment="test", _env_file=None)


def _conversations(tmp_path: Path) -> ConversationService:
    database = Database(tmp_path / "metadata.sqlite3")
    database.initialize()
    return ConversationService(SessionRepository(database))


def _latency_budgets(
    *,
    total_seconds: float = 0.2,
    queue_seconds: float = 0.02,
    redaction_seconds: float = 0.02,
    serialization_seconds: float = 0.02,
) -> QALatencyBudgets:
    return QALatencyBudgets(
        total_seconds=total_seconds,
        queue_seconds=queue_seconds,
        validation_seconds=0.02,
        embedding_seconds=0.02,
        dense_retrieval_seconds=0.02,
        bm25_seconds=0.02,
        fusion_seconds=0.02,
        rerank_seconds=0.02,
        evidence_assessment_seconds=0.02,
        generation_seconds=0.02,
        grounding_seconds=0.02,
        redaction_seconds=redaction_seconds,
        serialization_seconds=serialization_seconds,
        finalization_seconds=0.02,
    )


async def _next_stream_event(stream: Any) -> dict[str, object]:
    raw = await asyncio.wait_for(anext(stream), timeout=1)
    await stream.aclose()
    return cast(dict[str, object], json.loads(raw))


def _app(
    tmp_path: Path,
    orchestrator: object,
    *,
    emitter: object | None = None,
) -> tuple[FastAPI, ConversationService]:
    conversations = _conversations(tmp_path)
    services = QARuntimeServices(
        conversations=conversations,
        orchestrator=orchestrator,  # type: ignore[arg-type]
        emitter=emitter or CompleteResponseEmitter(conversations),  # type: ignore[arg-type]
    )
    return create_app(_settings(tmp_path), qa_services=services), conversations


def _event(response: object) -> dict[str, object]:
    http_response = cast("_HttpResponse", response)
    lines = http_response.text.splitlines()
    assert len(lines) == 1
    return cast(dict[str, object], json.loads(lines[0]))


@pytest.fixture
def qa_harness(tmp_path: Path) -> QAHarness:
    orchestrator = ScriptedOrchestrator("answer")
    app, conversations = _app(tmp_path, orchestrator)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield QAHarness(client, conversations, orchestrator)


def test_answer_stream_is_atomic_redacted_and_correlated(qa_harness: QAHarness) -> None:
    response = qa_harness.client.post(
        "/api/v1/qa",
        json={
            "owner_id": "owner-1",
            "question": "What is the leave policy?",
            "mode": "dense",
            "requested_language": "en",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(NDJSON_MEDIA_TYPE)
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-rag-instance-id"]
    event = _event(response)
    assert event["kind"] == "answer"
    assert event["terminal"] is True
    assert event["sequence"] == 0
    assert event["request_id"] == response.headers["x-request-id"]
    assert cast(str, event["session_id"]).startswith("session_")
    assert "x-session-id" not in response.headers
    assert event["content"] == "Employees receive ten days. Contact [REDACTED_EMAIL]."
    assert event["claims"][0]["citation_chunk_ids"] == ["chunk-1"]
    assert event["citations"][0]["source_title"] == "Handbook [REDACTED_EMAIL]"
    assert event["diagnostics"]["metadata"]["notice"] == "[REDACTED_EMAIL]"
    assert "person@example.com" not in response.text
    assert "owner@example.com" not in response.text
    assert "error_code" not in event
    call = qa_harness.orchestrator.calls[0]
    assert call["mode"] == "dense"
    assert call["cache_policy"] == "use"
    turns = qa_harness.conversations.list_turns(cast(str, event["session_id"]), "owner-1")
    assert turns[-1].content == event["content"]


@pytest.mark.asyncio
async def test_structural_request_id_is_not_redacted_as_a_payment_card(
    tmp_path: Path,
) -> None:
    conversations = _conversations(tmp_path)
    session = conversations.create_session("owner-1")
    request_id = "request_4111111111111111"
    services = QARuntimeServices(
        conversations=conversations,
        orchestrator=ScriptedOrchestrator("answer"),
        emitter=CompleteResponseEmitter(conversations),
    )
    stream = stream_qa_events(
        services,
        request_id=request_id,
        session_id=session.session_id,
        owner_id="owner-1",
        question="What is the policy?",
        mode="hybrid",
        requested_language="en",
        response_language="en",
        redactor=DEFAULT_REDACTOR,
    )

    event = await _next_stream_event(stream)

    assert event["kind"] == "answer"
    assert event["request_id"] == request_id
    assert "person@example.com" not in json.dumps(event)


@pytest.mark.parametrize(
    ("outcome", "kind", "detail_name", "detail_value"),
    [
        ("refusal", "refusal", "reason", "insufficient-evidence"),
        ("error", "error", "error_code", "deadline-expired"),
    ],
)
def test_refusal_and_error_keep_machine_readable_outcomes(
    tmp_path: Path,
    outcome: str,
    kind: str,
    detail_name: str,
    detail_value: str,
) -> None:
    app, _ = _app(tmp_path, ScriptedOrchestrator(outcome))
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/qa",
            json={"owner_id": "owner-1", "question": "What is unknown?"},
        )

    event = _event(response)
    assert response.status_code == 200
    assert event["kind"] == kind
    assert event[detail_name] == detail_value
    assert event["terminal"] is True
    if outcome == "error":
        assert event["retryable"] is True


def test_existing_session_is_reused_without_cross_owner_output(qa_harness: QAHarness) -> None:
    first = qa_harness.client.post(
        "/api/v1/qa",
        json={"owner_id": "owner-1", "question": "First question"},
    )
    session_id = cast(str, _event(first)["session_id"])

    continued = qa_harness.client.post(
        "/api/v1/qa",
        json={
            "owner_id": "owner-1",
            "session_id": session_id,
            "question": "Follow-up question",
        },
    )
    isolated = qa_harness.client.post(
        "/api/v1/qa",
        json={
            "owner_id": "owner-2",
            "session_id": session_id,
            "question": "Cross-owner question",
        },
    )

    assert _event(continued)["session_id"] == session_id
    isolated_event = _event(isolated)
    assert isolated_event["kind"] == "error"
    assert isolated_event["error_code"] == "safety-unavailable"
    assert "Employees receive" not in isolated.text


def test_response_language_must_match_the_request_contract(qa_harness: QAHarness) -> None:
    response = qa_harness.client.post(
        "/api/v1/qa",
        json={
            "owner_id": "owner-1",
            "question": "请说明休假政策。",
            "requested_language": "zh-CN",
        },
    )

    event = _event(response)
    assert event["kind"] == "error"
    assert event["response_language"] == "zh-CN"
    assert event["error_code"] == "safety-unavailable"
    assert "Employees receive" not in response.text


@pytest.mark.parametrize(
    ("failure", "error_code", "retryable"),
    [
        ("exception", "internal", False),
        ("malformed", "safety-unavailable", True),
        ("split-sensitive", "safety-unavailable", True),
    ],
)
def test_internal_or_malformed_stream_fails_closed(
    tmp_path: Path,
    failure: str,
    error_code: str,
    retryable: bool,
) -> None:
    if failure == "exception":
        app, _ = _app(tmp_path, ExplodingOrchestrator())
    elif failure == "malformed":
        app, _ = _app(
            tmp_path,
            ScriptedOrchestrator("answer"),
            emitter=MalformedEmitter(),
        )
    else:
        app, _ = _app(
            tmp_path,
            ScriptedOrchestrator("answer"),
            emitter=SplitSensitiveEmitter(),
        )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/qa",
            json={"owner_id": "owner-1", "question": "private person@example.com"},
        )

    event = _event(response)
    assert response.status_code == 200
    assert event["kind"] == "error"
    assert event["content"] == SAFE_UNAVAILABLE_MESSAGE
    assert event["error_code"] == error_code
    assert event["retryable"] is retryable
    assert event["sequence"] == 0
    assert "person@example.com" not in response.text
    assert "person@" not in response.text
    assert "example.com" not in response.text


@pytest.mark.asyncio
async def test_stream_cancellation_propagates_without_emitting_fallback(tmp_path: Path) -> None:
    conversations = _conversations(tmp_path)
    session = conversations.create_session("owner-1")
    orchestrator = BlockingOrchestrator()
    services = QARuntimeServices(
        conversations=conversations,
        orchestrator=orchestrator,
        emitter=CompleteResponseEmitter(conversations),
    )
    stream = stream_qa_events(
        services,
        request_id="request-1",
        session_id=session.session_id,
        owner_id="owner-1",
        question="What is the policy?",
        mode="hybrid",
        requested_language="en",
        response_language="en",
        redactor=DEFAULT_REDACTOR,
    )
    next_event = asyncio.create_task(anext(stream))
    await asyncio.wait_for(orchestrator.started.wait(), timeout=1)

    next_event.cancel()

    with pytest.raises(asyncio.CancelledError):
        await next_event
    assert orchestrator.cancelled.is_set()
    await stream.aclose()
    assert conversations.list_turns(session.session_id, "owner-1") == ()


@pytest.mark.asyncio
async def test_queue_budget_timeout_returns_retryable_capacity_event(tmp_path: Path) -> None:
    conversations = _conversations(tmp_path)
    session = conversations.create_session("owner-1")
    admission = QAAdmissionController(max_active=1, max_queue=1)
    occupied = await admission.acquire()
    orchestrator = ScriptedOrchestrator("answer")
    services = QARuntimeServices(
        conversations=conversations,
        orchestrator=orchestrator,
        emitter=CompleteResponseEmitter(conversations),
        admission=admission,
        latency_budgets=_latency_budgets(queue_seconds=0.005),
    )
    stream = stream_qa_events(
        services,
        request_id="request-queue-timeout",
        session_id=session.session_id,
        owner_id="owner-1",
        question="What is the policy?",
        mode="hybrid",
        requested_language="en",
        response_language="en",
        redactor=DEFAULT_REDACTOR,
    )

    try:
        event = await _next_stream_event(stream)
    finally:
        await occupied.release()
        await admission.close()

    assert event["kind"] == "error"
    assert event["error_code"] == "capacity"
    assert event["retryable"] is True
    assert event["content"] == SAFE_UNAVAILABLE_MESSAGE
    assert orchestrator.calls == []
    snapshot = await admission.snapshot()
    assert snapshot.active == 0
    assert snapshot.queued == 0


@pytest.mark.asyncio
async def test_total_deadline_cancels_orchestrator_and_returns_safe_error(tmp_path: Path) -> None:
    conversations = _conversations(tmp_path)
    session = conversations.create_session("owner-1")
    orchestrator = BlockingOrchestrator()
    services = QARuntimeServices(
        conversations=conversations,
        orchestrator=orchestrator,
        emitter=CompleteResponseEmitter(conversations),
        latency_budgets=_latency_budgets(total_seconds=0.03),
    )
    stream = stream_qa_events(
        services,
        request_id="request-hard-deadline",
        session_id=session.session_id,
        owner_id="owner-1",
        question="What is the policy?",
        mode="hybrid",
        requested_language="en",
        response_language="en",
        redactor=DEFAULT_REDACTOR,
    )

    event = await _next_stream_event(stream)

    assert orchestrator.cancelled.is_set()
    assert event["kind"] == "error"
    assert event["error_code"] == "deadline-expired"
    assert event["retryable"] is True
    assert event["content"] == SAFE_UNAVAILABLE_MESSAGE
    assert conversations.list_turns(session.session_id, "owner-1") == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("slow_stage", ["redaction", "serialization"])
async def test_synchronous_release_stage_overrun_fails_closed(
    tmp_path: Path,
    slow_stage: str,
) -> None:
    conversations = _conversations(tmp_path)
    session = conversations.create_session("owner-1")
    emitter: object = CompleteResponseEmitter(conversations)
    redactor: Redactor = DEFAULT_REDACTOR
    budgets = _latency_budgets()
    if slow_stage == "redaction":
        emitter = DelayedEmitter(cast(CompleteResponseEmitter, emitter), 0.02)
        budgets = _latency_budgets(redaction_seconds=0.005)
    else:
        redactor = DelayedRedactor(0.01)
        budgets = _latency_budgets(serialization_seconds=0.005)
    services = QARuntimeServices(
        conversations=conversations,
        orchestrator=ScriptedOrchestrator("answer"),
        emitter=cast(CompleteResponseEmitter, emitter),
        telemetry=PipelineTelemetry(_settings(tmp_path)),
        latency_budgets=budgets,
    )
    stream = stream_qa_events(
        services,
        request_id=f"request-{slow_stage}-timeout",
        session_id=session.session_id,
        owner_id="owner-1",
        question="What is the policy?",
        mode="hybrid",
        requested_language="en",
        response_language="en",
        redactor=redactor,
    )

    event = await _next_stream_event(stream)

    assert event["kind"] == "error"
    assert event["error_code"] == "deadline-expired"
    assert event["retryable"] is True
    assert event["content"] == SAFE_UNAVAILABLE_MESSAGE
    assert "person@example.com" not in json.dumps(event)


@pytest.mark.asyncio
async def test_http_disconnect_cancels_the_active_qa_pipeline(tmp_path: Path) -> None:
    orchestrator = BlockingOrchestrator()
    app, _ = _app(tmp_path, orchestrator)
    body = json.dumps({"owner_id": "owner-1", "question": "What is the policy?"}).encode()
    receive_count = 0
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal receive_count
        receive_count += 1
        if receive_count == 1:
            return {"type": "http.request", "body": body, "more_body": False}
        await orchestrator.started.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/v1/qa",
        "raw_path": b"/api/v1/qa",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "state": {},
    }
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app(scope, receive, send), timeout=2)

    assert orchestrator.cancelled.is_set()
    response_body = b"".join(
        cast(bytes, message.get("body", b""))
        for message in sent
        if message["type"] == "http.response.body"
    )
    assert SAFE_UNAVAILABLE_MESSAGE.encode() not in response_body


def test_qa_readiness_requires_the_actual_release_gate(tmp_path: Path) -> None:
    conversations = _conversations(tmp_path)
    services = QARuntimeServices(
        conversations=conversations,
        orchestrator=ScriptedOrchestrator("answer"),
        emitter=CompleteResponseEmitter(conversations, redactor=None),
    )
    app = create_app(_settings(tmp_path), qa_services=services)

    with TestClient(app, raise_server_exceptions=False) as client:
        readiness = client.get("/readyz")
        response = client.post(
            "/api/v1/qa",
            json={"owner_id": "owner-1", "question": "Question"},
        )

    assert readiness.status_code == 503
    assert {
        "name": "qa",
        "ready": False,
        "reason": "qa_release_unavailable",
    } in readiness.json()["components"]
    assert response.status_code == 503
    assert response.json() == {"error": {"code": "qa_unavailable"}}


def test_unavailable_runtime_and_invalid_request_use_content_free_http_errors(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app, raise_server_exceptions=False) as client:
        unavailable = client.post(
            "/api/v1/qa",
            json={"owner_id": "owner-1", "question": "Question"},
        )
    configured_app, _ = _app(tmp_path, ScriptedOrchestrator("answer"))
    with TestClient(configured_app, raise_server_exceptions=False) as client:
        invalid = client.post(
            "/api/v1/qa",
            json={"owner_id": "owner-1", "question": "person@example.com", "mode": "bad"},
        )
        unsafe_session = client.post(
            "/api/v1/qa",
            json={
                "owner_id": "owner-1",
                "session_id": "123-45-6789",
                "question": "Question",
            },
        )

    assert unavailable.status_code == 503
    assert unavailable.json() == {"error": {"code": "qa_unavailable"}}
    assert invalid.status_code == 422
    assert invalid.json() == {"error": {"code": "request_invalid"}}
    assert "person@example.com" not in invalid.text
    assert unsafe_session.status_code == 422
    assert unsafe_session.json() == {"error": {"code": "request_invalid"}}
    assert "123-45-6789" not in unsafe_session.text


def test_openapi_declares_qa_request_and_ndjson_event_contract(qa_harness: QAHarness) -> None:
    schema = qa_harness.client.get("/openapi.json").json()
    operation = schema["paths"]["/api/v1/qa"]["post"]

    assert operation["requestBody"]["content"]["application/json"]
    assert set(operation["responses"]["200"]["content"]) == {NDJSON_MEDIA_TYPE}
    assert operation["responses"]["200"]["content"][NDJSON_MEDIA_TYPE]["schema"] == {
        "type": "string",
        "description": "UTF-8 NDJSON containing one validated terminal event.",
    }
    assert operation["responses"]["200"]["content"][NDJSON_MEDIA_TYPE]["x-ndjson-item-schema"] == {
        "$ref": "#/components/schemas/ValidatedStreamEvent"
    }
    event_properties = schema["components"]["schemas"]["ValidatedStreamEvent"]["properties"]
    assert set(event_properties) >= {
        "request_id",
        "session_id",
        "sequence",
        "kind",
        "content",
        "claims",
        "citations",
        "reason",
        "error_code",
        "retryable",
        "diagnostics",
        "terminal",
    }
    for status_code in ("422", "500", "503"):
        assert set(operation["responses"][status_code]["content"]) == {"application/json"}


class _HttpResponse(Protocol):
    text: str
