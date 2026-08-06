from __future__ import annotations

import asyncio
import importlib.util
import io
import json
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rag_mvp.config.settings import Settings
from rag_mvp.observability.logging import SafeErrorCategory
from rag_mvp.observability.metrics import PipelineStage, QAOutcome
from rag_mvp.observability.tracing import (
    RAGTracer,
    TelemetryConfigurationError,
    create_rag_tracer,
    tracing_readiness_errors,
)


class _TrackingSpanExporter(InMemorySpanExporter):
    def __init__(self) -> None:
        super().__init__()
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        super().shutdown()


def _tracer() -> tuple[RAGTracer, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return RAGTracer(provider.get_tracer("test-rag"), provider=provider), exporter


@pytest.mark.asyncio
async def test_root_and_async_stage_spans_preserve_parent_context_without_content() -> None:
    tracer, exporter = _tracer()

    async def provider_call() -> None:
        async with tracer.stage_span(
            PipelineStage.GENERATION,
            provider_alias="primary",
            model_alias="chat-v1",
        ) as span:
            await asyncio.sleep(0)
            span.set_token_usage(input_tokens=10, output_tokens=4)

    async with tracer.request_span(
        request_id="request-1",
        operation="qa",
        config_version="config-v1",
    ) as root:
        root.set_outcome(QAOutcome.ANSWER)
        await asyncio.create_task(provider_call())

    spans = {span.name: span for span in exporter.get_finished_spans()}
    root_span = spans["rag.request"]
    stage_span = spans["rag.stage.generation"]
    assert root_span.parent is None
    assert stage_span.parent is not None
    assert stage_span.parent.span_id == root_span.context.span_id
    assert root_span.attributes["rag.request.id"] == "request-1"
    assert stage_span.attributes["rag.provider.alias"] == "primary"
    assert stage_span.attributes["rag.token.input"] == 10
    forbidden = {"question", "answer", "prompt", "content", "document.text"}
    assert forbidden.isdisjoint(root_span.attributes)
    assert forbidden.isdisjoint(stage_span.attributes)


@pytest.mark.asyncio
async def test_exception_records_only_safe_category() -> None:
    tracer, exporter = _tracer()
    fixture = "person@example.com"

    with pytest.raises(RuntimeError, match=fixture):
        async with tracer.request_span(request_id="request-2"):
            raise RuntimeError(fixture)

    span = exporter.get_finished_spans()[0]
    assert span.attributes["error.type"] == SafeErrorCategory.INTERNAL.value
    assert fixture not in repr(span.attributes)
    assert not span.events


@pytest.mark.asyncio
@pytest.mark.parametrize("unsafe_alias", ["person@example.com", "198.51.100.23"])
async def test_arbitrary_content_cannot_be_added_as_alias(unsafe_alias: str) -> None:
    tracer, _ = _tracer()
    with pytest.raises(ValueError, match=r"(safe identifier|opaque identifier)"):
        async with tracer.stage_span(
            PipelineStage.GENERATION,
            model_alias=unsafe_alias,
        ):
            pass


@pytest.mark.asyncio
async def test_none_exporter_disables_recording_but_preserves_correlation(tmp_path: Path) -> None:
    tracer = create_rag_tracer(
        Settings(data_root=tmp_path, telemetry_exporter="none", _env_file=None)
    )

    async with (
        tracer.request_span(request_id="request-disabled") as root,
        tracer.stage_span(PipelineStage.RETRIEVAL) as child,
    ):
        assert child.reference.trace_id == root.reference.trace_id

    assert tracer.exporter_name == "none"
    assert not tracer.enabled
    assert int(root.reference.trace_id, 16) != 0
    assert tracer.force_flush(100)
    tracer.shutdown()
    tracer.shutdown()


@pytest.mark.asyncio
async def test_console_exporter_emits_correlated_redacted_json(tmp_path: Path) -> None:
    output = io.StringIO()
    tracer = create_rag_tracer(
        Settings(data_root=tmp_path, telemetry_exporter="console", _env_file=None),
        console_stream=output,
    )
    async with tracer.request_span(
        request_id="request-console",
        operation="qa",
        config_version="config-v1",
        trace_id="a" * 32,
    ) as root:
        async with tracer.stage_span(
            PipelineStage.GENERATION,
            provider_alias="primary",
            model_alias="chat-v1",
        ) as child:
            child.set_token_usage(input_tokens=12, output_tokens=3)
        root.set_outcome(QAOutcome.ANSWER)

    assert tracer.force_flush(5_000)
    records = [json.loads(line) for line in output.getvalue().splitlines() if line]
    assert len(records) == 2
    assert {record["trace_id"] for record in records} == {"a" * 32}
    assert {record["event_name"] for record in records} == {"trace.span.completed"}
    root_record = next(record for record in records if record["request_id"] == "request-console")
    child_record = next(record for record in records if record["stage"] == "generation")
    assert child_record["metadata"]["parent_span_id"] == root_record["span_id"]
    assert root_record["metadata"]["parent_span_id"] is None
    assert child_record["token_usage"] == {"input": 12, "output": 3}
    assert child_record["provider"] == "primary"
    assert "question" not in output.getvalue()
    tracer.shutdown()


@pytest.mark.asyncio
async def test_otlp_exporter_uses_explicit_endpoint_and_flushes(tmp_path: Path) -> None:
    exporter = _TrackingSpanExporter()
    captured: dict[str, str | float] = {}

    def exporter_factory(endpoint: str, timeout_seconds: float) -> _TrackingSpanExporter:
        captured["endpoint"] = endpoint
        captured["timeout_seconds"] = timeout_seconds
        return exporter

    tracer = create_rag_tracer(
        Settings(
            data_root=tmp_path,
            telemetry_exporter="otlp",
            telemetry_otlp_traces_endpoint="http://collector:4318/v1/traces",
            telemetry_export_timeout_seconds=2.5,
            _env_file=None,
        ),
        otlp_exporter_factory=exporter_factory,
    )
    async with (
        tracer.request_span(request_id="request-otlp") as root,
        tracer.stage_span(PipelineStage.RETRIEVAL) as child,
    ):
        pass

    assert tracer.force_flush(5_000)
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert spans[0].context.trace_id == spans[1].context.trace_id
    assert root.reference.trace_id == child.reference.trace_id
    assert captured == {
        "endpoint": "http://collector:4318/v1/traces",
        "timeout_seconds": 2.5,
    }
    tracer.shutdown()
    tracer.shutdown()
    assert exporter.shutdown_calls == 1


def test_otlp_missing_endpoint_is_an_explicit_safe_configuration_error(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path, telemetry_exporter="otlp", _env_file=None)

    assert settings.telemetry_readiness_errors() == ("telemetry_otlp_endpoint_missing",)
    assert tracing_readiness_errors(settings) == ("telemetry_otlp_endpoint_missing",)
    with pytest.raises(
        TelemetryConfigurationError,
        match=r"^telemetry_otlp_endpoint_missing$",
    ):
        create_rag_tracer(settings)


def test_otlp_missing_dependency_is_an_explicit_safe_configuration_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        data_root=tmp_path,
        telemetry_exporter="otlp",
        telemetry_otlp_traces_endpoint="http://collector:4318/v1/traces",
        _env_file=None,
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    assert tracing_readiness_errors(settings) == ("telemetry_otlp_dependency_missing",)
    with pytest.raises(
        TelemetryConfigurationError,
        match=r"^telemetry_otlp_dependency_missing$",
    ):
        create_rag_tracer(settings)


def test_otlp_initialization_failure_does_not_chain_raw_error_content(tmp_path: Path) -> None:
    fixture = "person@example.com"
    settings = Settings(
        data_root=tmp_path,
        telemetry_exporter="otlp",
        telemetry_otlp_traces_endpoint="http://collector:4318/v1/traces",
        _env_file=None,
    )

    def failing_factory(_endpoint: str, _timeout_seconds: float) -> InMemorySpanExporter:
        raise RuntimeError(fixture)

    with pytest.raises(
        TelemetryConfigurationError,
        match=r"^telemetry_otlp_initialization_failed$",
    ) as captured:
        create_rag_tracer(settings, otlp_exporter_factory=failing_factory)

    assert captured.value.__cause__ is None
    assert fixture not in str(captured.value)


def test_exporter_rejects_unsafe_resource_identity_with_safe_error(tmp_path: Path) -> None:
    settings = Settings(
        data_root=tmp_path,
        telemetry_exporter="console",
        service_name="unsafe service name",
        _env_file=None,
    )

    with pytest.raises(
        TelemetryConfigurationError,
        match=r"^telemetry_resource_identity_invalid$",
    ) as captured:
        create_rag_tracer(settings, console_stream=io.StringIO())

    assert captured.value.__cause__ is None
