"""Privacy-safe OpenTelemetry root and stage spans."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from secrets import randbits
from typing import Final

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import (
    NonRecordingSpan,
    Span,
    SpanContext,
    Status,
    StatusCode,
    TraceFlags,
    Tracer,
    TraceState,
    set_span_in_context,
)

from rag_mvp.observability.logging import (
    SafeErrorCategory,
    classify_exception,
    is_safe_identifier,
)
from rag_mvp.observability.metrics import (
    CacheOutcome,
    DegradationReason,
    PipelineStage,
    QAOutcome,
)

_ATTRIBUTE_LIMIT: Final[int] = 128


def _safe_identifier(value: str, field: str) -> str:
    if len(value) > _ATTRIBUTE_LIMIT or not is_safe_identifier(value):
        raise ValueError(f"{field} must be a bounded opaque identifier")
    return value


def _non_negative(value: int, field: str) -> int:
    if isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class TraceReference:
    """Content-free identifiers suitable for diagnostics and evidence bundles."""

    trace_id: str
    span_id: str


class SafeSpan:
    """Narrow span facade that cannot attach arbitrary content attributes."""

    def __init__(self, span: Span) -> None:
        self._span = span
        self._errored = False

    @property
    def reference(self) -> TraceReference:
        context = self._span.get_span_context()
        return TraceReference(
            trace_id=f"{context.trace_id:032x}",
            span_id=f"{context.span_id:016x}",
        )

    def set_outcome(self, outcome: QAOutcome | str) -> None:
        self._span.set_attribute("rag.outcome", QAOutcome(outcome).value)

    def set_error(self, category: SafeErrorCategory | str) -> None:
        safe_category = SafeErrorCategory(category)
        self._span.set_attribute("error.type", safe_category.value)
        self._span.set_status(Status(StatusCode.ERROR))
        self._errored = True

    @property
    def errored(self) -> bool:
        return self._errored

    def set_token_usage(
        self,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        if input_tokens is not None:
            self._span.set_attribute(
                "rag.token.input",
                _non_negative(input_tokens, "input_tokens"),
            )
        if output_tokens is not None:
            self._span.set_attribute(
                "rag.token.output",
                _non_negative(output_tokens, "output_tokens"),
            )

    def set_cache_outcome(self, outcome: CacheOutcome | str) -> None:
        self._span.set_attribute("rag.cache.outcome", CacheOutcome(outcome).value)

    def set_degradation(self, reason: DegradationReason | str) -> None:
        self._span.set_attribute("rag.degradation.reason", DegradationReason(reason).value)


class RAGTracer:
    """Create one QA root span and bounded child spans with async propagation."""

    def __init__(self, tracer: Tracer | None = None) -> None:
        self._tracer = tracer or trace.get_tracer("rag_mvp.observability", "1")

    @asynccontextmanager
    async def request_span(
        self,
        *,
        request_id: str,
        operation: str = "qa",
        run_id: str | None = None,
        config_version: str | None = None,
        trace_id: str | None = None,
    ) -> AsyncIterator[SafeSpan]:
        attributes: dict[str, str] = {
            "rag.request.id": _safe_identifier(request_id, "request_id"),
            "rag.operation": _safe_identifier(operation, "operation"),
        }
        if run_id is not None:
            attributes["rag.run.id"] = _safe_identifier(run_id, "run_id")
        if config_version is not None:
            attributes["rag.config.version"] = _safe_identifier(
                config_version,
                "config_version",
            )
        parent_context = _parent_context(trace_id) if trace_id is not None else None
        async with self._span("rag.request", attributes, context=parent_context) as span:
            yield span

    @asynccontextmanager
    async def stage_span(
        self,
        stage: PipelineStage | str,
        *,
        provider_alias: str | None = None,
        model_alias: str | None = None,
        cache_outcome: CacheOutcome | str | None = None,
        degradation_reason: DegradationReason | str | None = None,
    ) -> AsyncIterator[SafeSpan]:
        safe_stage = PipelineStage(stage)
        attributes: dict[str, str] = {"rag.stage": safe_stage.value}
        if provider_alias is not None:
            attributes["rag.provider.alias"] = _safe_identifier(
                provider_alias,
                "provider_alias",
            )
        if model_alias is not None:
            attributes["rag.model.alias"] = _safe_identifier(model_alias, "model_alias")
        if cache_outcome is not None:
            attributes["rag.cache.outcome"] = CacheOutcome(cache_outcome).value
        if degradation_reason is not None:
            attributes["rag.degradation.reason"] = DegradationReason(degradation_reason).value
        async with self._span(f"rag.stage.{safe_stage.value}", attributes) as span:
            yield span

    @asynccontextmanager
    async def _span(
        self,
        name: str,
        attributes: dict[str, str],
        *,
        context: Context | None = None,
    ) -> AsyncIterator[SafeSpan]:
        with self._tracer.start_as_current_span(
            name,
            context=context,
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as raw_span:
            span = SafeSpan(raw_span)
            try:
                yield span
            except BaseException as error:
                span.set_error(classify_exception(error))
                raise
            else:
                if not span.errored:
                    raw_span.set_status(Status(StatusCode.OK))


def _parent_context(trace_id: str) -> Context:
    normalized = trace_id.lower()
    if len(normalized) != 32:
        raise ValueError("trace_id must be a 32-character hexadecimal identifier")
    try:
        numeric_trace_id = int(normalized, 16)
    except ValueError:
        raise ValueError("trace_id must be a 32-character hexadecimal identifier") from None
    if numeric_trace_id == 0:
        raise ValueError("trace_id must not be all zeroes")
    parent = NonRecordingSpan(
        SpanContext(
            trace_id=numeric_trace_id,
            span_id=randbits(64) or 1,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
            trace_state=TraceState(),
        )
    )
    return set_span_in_context(parent)
