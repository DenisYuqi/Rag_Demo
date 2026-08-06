from __future__ import annotations

import asyncio
import json
from io import StringIO

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_mvp.observability.logging import (
    RequestTraceContextMiddleware,
    SafeErrorCategory,
    bind_correlation_context,
    classify_exception,
    configure_logging,
    current_correlation_context,
    get_logger,
)


def _events(stream: StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


def test_structlog_emits_allowlisted_correlated_json_without_content() -> None:
    stream = StringIO()
    configure_logging(
        service="rag-mvp",
        service_version="0.1.0",
        config_version="config-v1",
        stream=stream,
    )
    fixture = "person@example.com"

    with bind_correlation_context("request-1", "a" * 32):
        get_logger("qa").info(
            "qa.completed",
            outcome="answer",
            stage_duration_ms=12.5,
            counts={"citations": 2},
            question=fixture,
            answer=f"answer for {fixture}",
            unknown="not-allowed",
        )

    event = _events(stream)[0]
    assert event["event"] == "qa.completed"
    assert event["level"] == "info"
    assert event["service"] == "rag-mvp"
    assert event["service_version"] == "0.1.0"
    assert event["config_version"] == "config-v1"
    assert event["request_id"] == "request-1"
    assert event["trace_id"] == "a" * 32
    assert event["operation"] == "qa"
    assert "timestamp" in event
    assert fixture not in stream.getvalue()
    assert "question" not in event
    assert "answer" not in event
    assert "unknown" not in event


@pytest.mark.asyncio
async def test_correlation_context_propagates_to_async_tasks_and_restores() -> None:
    async def read_context() -> object:
        await asyncio.sleep(0)
        return current_correlation_context()

    assert current_correlation_context() is None
    with bind_correlation_context("request-async", "b" * 32) as expected:
        assert await asyncio.create_task(read_context()) == expected
    assert current_correlation_context() is None


def test_safe_exception_classification_never_uses_exception_message() -> None:
    fixture = "Bearer abcdefghijklmnop"
    error = RuntimeError(fixture)
    assert classify_exception(error) is SafeErrorCategory.INTERNAL

    stream = StringIO()
    configure_logging(
        service="rag-mvp",
        service_version="0.1.0",
        config_version="config-v1",
        stream=stream,
    )
    get_logger().error(
        "qa.failed",
        outcome="failed",
        safe_error_category=classify_exception(error).value,
    )
    assert fixture not in stream.getvalue()
    assert _events(stream)[0]["safe_error_category"] == "internal"


def test_http_middleware_binds_and_returns_correlation_headers() -> None:
    stream = StringIO()
    configure_logging(
        service="rag-mvp",
        service_version="0.1.0",
        config_version="config-v1",
        stream=stream,
    )
    app = FastAPI()
    app.add_middleware(RequestTraceContextMiddleware)

    @app.get("/probe")
    async def probe() -> dict[str, str]:
        context = current_correlation_context()
        assert context is not None
        return {"request_id": context.request_id, "trace_id": context.trace_id}

    response = TestClient(app).get(
        "/probe",
        headers={"x-request-id": "request-client", "x-trace-id": "c" * 32},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "request-client"
    assert response.headers["x-trace-id"] == "c" * 32
    assert response.json() == {"request_id": "request-client", "trace_id": "c" * 32}
    events = _events(stream)
    assert [event["event"] for event in events] == [
        "http.request.started",
        "http.request.completed",
    ]


def test_http_middleware_replaces_an_invalid_zero_trace_id() -> None:
    app = FastAPI()
    app.add_middleware(RequestTraceContextMiddleware)

    @app.get("/probe")
    async def probe() -> dict[str, str]:
        context = current_correlation_context()
        assert context is not None
        return {"trace_id": context.trace_id}

    response = TestClient(app).get("/probe", headers={"x-trace-id": "0" * 32})

    assert response.status_code == 200
    assert response.headers["x-trace-id"] != "0" * 32
    assert response.json()["trace_id"] == response.headers["x-trace-id"]
