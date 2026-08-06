from __future__ import annotations

import asyncio

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rag_mvp.observability.logging import SafeErrorCategory
from rag_mvp.observability.metrics import PipelineStage, QAOutcome
from rag_mvp.observability.tracing import RAGTracer


def _tracer() -> tuple[RAGTracer, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return RAGTracer(provider.get_tracer("test-rag")), exporter


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
async def test_arbitrary_content_cannot_be_added_as_alias() -> None:
    tracer, _ = _tracer()
    with pytest.raises(ValueError, match="opaque identifier"):
        async with tracer.stage_span(
            PipelineStage.GENERATION,
            model_alias="person@example.com",
        ):
            pass
