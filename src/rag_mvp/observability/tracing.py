"""Privacy-safe OpenTelemetry root and stage spans."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, TextIO

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
)
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.trace import (
    Span,
    Status,
    StatusCode,
    Tracer,
    TracerProvider,
)

from rag_mvp.config.settings import Settings
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
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, RedactionError
from rag_mvp.safety.telemetry import TelemetryFilter

_ATTRIBUTE_LIMIT: Final[int] = 128
_INSTRUMENTATION_NAME: Final[str] = "rag_mvp.observability"
_INSTRUMENTATION_VERSION: Final[str] = "1"
_OTLP_HTTP_MODULE: Final[str] = "opentelemetry.exporter.otlp.proto.http.trace_exporter"
_REQUESTED_TRACE_ID: ContextVar[int | None] = ContextVar(
    "rag_mvp_requested_trace_id",
    default=None,
)

OTLPExporterFactory = Callable[[str, float], SpanExporter]


class _CorrelatedIdGenerator(RandomIdGenerator):
    """Generate a true root span with the HTTP correlation trace ID when supplied."""

    def generate_trace_id(self) -> int:
        return _REQUESTED_TRACE_ID.get() or super().generate_trace_id()


class TelemetryConfigurationError(RuntimeError):
    """Stable, content-free telemetry configuration failure."""

    safe_error_category = SafeErrorCategory.UNAVAILABLE

    def __init__(self, reason: str) -> None:
        self.reason = _safe_identifier(reason, "reason")
        super().__init__(self.reason)


def _safe_identifier(value: str, field: str) -> str:
    if len(value) > _ATTRIBUTE_LIMIT or not is_safe_identifier(value):
        raise ValueError(f"{field} must be a bounded opaque identifier")
    try:
        contains_sensitive_value = bool(DEFAULT_REDACTOR.detect(value))
    except RedactionError as error:
        raise ValueError(f"{field} could not be verified as a safe identifier") from error
    if contains_sensitive_value:
        raise ValueError(f"{field} must be a content-free opaque identifier")
    return value


def _non_negative(value: int, field: str) -> int:
    if isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def tracing_readiness_errors(settings: Settings) -> tuple[str, ...]:
    """Return safe local configuration errors without contacting an OTLP collector."""

    errors = list(settings.telemetry_readiness_errors())
    if settings.telemetry_exporter == "otlp" and not errors:
        try:
            dependency_available = importlib.util.find_spec(_OTLP_HTTP_MODULE) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            dependency_available = False
        if not dependency_available:
            errors.append("telemetry_otlp_dependency_missing")
    return tuple(errors)


def create_rag_tracer(
    settings: Settings,
    *,
    console_stream: TextIO | None = None,
    otlp_exporter_factory: OTLPExporterFactory | None = None,
) -> RAGTracer:
    """Build an isolated tracer provider from validated application settings.

    The provider is intentionally not installed globally. This keeps unrelated library
    instrumentation out of the privacy-safe RAG export surface while normal OpenTelemetry
    context propagation still links the root and child spans created by :class:`RAGTracer`.
    """

    errors = tracing_readiness_errors(settings)
    if errors:
        raise TelemetryConfigurationError(errors[0])

    exporter_name = settings.telemetry_exporter
    if exporter_name == "none":
        no_export_provider = SDKTracerProvider(
            sampler=ALWAYS_OFF,
            id_generator=_CorrelatedIdGenerator(),
        )
        return RAGTracer(
            no_export_provider.get_tracer(
                _INSTRUMENTATION_NAME,
                _INSTRUMENTATION_VERSION,
            ),
            provider=no_export_provider,
            enabled=False,
            exporter_name="none",
            owns_provider=True,
        )

    try:
        resource = Resource.create(
            {
                "service.name": _safe_identifier(settings.service_name, "service_name"),
                "service.version": _safe_identifier(
                    settings.service_version,
                    "service_version",
                ),
                "deployment.environment.name": settings.environment,
                "rag.config.version": settings.configuration_identity,
            }
        )
    except ValueError:
        raise TelemetryConfigurationError("telemetry_resource_identity_invalid") from None
    sdk_provider = SDKTracerProvider(
        resource=resource,
        id_generator=_CorrelatedIdGenerator(),
    )
    if exporter_name == "console":
        telemetry_filter = TelemetryFilter()

        def formatter(span: ReadableSpan) -> str:
            return _format_console_span(span, telemetry_filter)

        exporter: SpanExporter = ConsoleSpanExporter(
            out=console_stream or sys.stdout,
            formatter=formatter,
        )
    else:
        endpoint = settings.telemetry_otlp_traces_endpoint
        if endpoint is None:  # Defensive: readiness validation above owns this invariant.
            raise TelemetryConfigurationError("telemetry_otlp_endpoint_missing")
        exporter_factory = otlp_exporter_factory or _create_otlp_exporter
        try:
            exporter = exporter_factory(endpoint, settings.telemetry_export_timeout_seconds)
        except TelemetryConfigurationError:
            raise
        except (ImportError, ModuleNotFoundError):
            raise TelemetryConfigurationError("telemetry_otlp_dependency_missing") from None
        except Exception:
            raise TelemetryConfigurationError("telemetry_otlp_initialization_failed") from None

    sdk_provider.add_span_processor(BatchSpanProcessor(exporter))
    return RAGTracer(
        sdk_provider.get_tracer(_INSTRUMENTATION_NAME, _INSTRUMENTATION_VERSION),
        provider=sdk_provider,
        enabled=True,
        exporter_name=exporter_name,
        owns_provider=True,
    )


def _create_otlp_exporter(endpoint: str, timeout_seconds: float) -> SpanExporter:
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except (ImportError, ModuleNotFoundError):
        raise TelemetryConfigurationError("telemetry_otlp_dependency_missing") from None
    return OTLPSpanExporter(endpoint=endpoint, timeout=timeout_seconds)


def _format_console_span(span: ReadableSpan, telemetry_filter: TelemetryFilter) -> str:
    context = span.context
    if context is None or not context.is_valid or span.start_time is None or span.end_time is None:
        return ""
    attributes = span.attributes or {}
    parent_span_id = (
        f"{span.parent.span_id:016x}" if span.parent is not None and span.parent.is_valid else None
    )
    event: dict[str, object] = {
        "timestamp": datetime.fromtimestamp(span.end_time / 1_000_000_000, tz=UTC).isoformat(),
        "service": span.resource.attributes.get("service.name", "rag-mvp"),
        "service_version": span.resource.attributes.get("service.version", "unknown"),
        "config_version": attributes.get(
            "rag.config.version",
            span.resource.attributes.get("rag.config.version", "unknown"),
        ),
        "event_name": "trace.span.completed",
        "trace_id": f"{context.trace_id:032x}",
        "span_id": f"{context.span_id:016x}",
        "operation": attributes.get("rag.operation", span.name),
        "stage": attributes.get("rag.stage"),
        "outcome": attributes.get("rag.outcome"),
        "status": span.status.status_code.name.lower(),
        "duration_ms": max(0.0, (span.end_time - span.start_time) / 1_000_000),
        "request_id": attributes.get("rag.request.id"),
        "run_id": attributes.get("rag.run.id"),
        "provider": attributes.get("rag.provider.alias"),
        "model": attributes.get("rag.model.alias"),
        "cache_outcome": attributes.get("rag.cache.outcome"),
        "degraded_reason": attributes.get("rag.degradation.reason"),
        "safe_error_category": attributes.get("error.type"),
        "token_usage": {
            "input": attributes.get("rag.token.input"),
            "output": attributes.get("rag.token.output"),
        },
        "metadata": {
            "span_name": span.name,
            "parent_span_id": parent_span_id,
        },
    }
    filtered = telemetry_filter.filter(event)
    if filtered is None:
        return ""
    return json.dumps(filtered, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"


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

    def set_provider_alias(self, alias: str) -> None:
        self._span.set_attribute(
            "rag.provider.alias",
            _safe_identifier(alias, "provider_alias"),
        )

    def set_model_alias(self, alias: str) -> None:
        self._span.set_attribute("rag.model.alias", _safe_identifier(alias, "model_alias"))


class RAGTracer:
    """Create one QA root span and bounded child spans with async propagation."""

    def __init__(
        self,
        tracer: Tracer | None = None,
        *,
        provider: TracerProvider | None = None,
        enabled: bool = True,
        exporter_name: str = "global",
        owns_provider: bool = False,
    ) -> None:
        resolved_provider = provider
        if tracer is None:
            resolved_provider = resolved_provider or trace.get_tracer_provider()
            tracer = resolved_provider.get_tracer(
                _INSTRUMENTATION_NAME,
                _INSTRUMENTATION_VERSION,
            )
        self._provider = resolved_provider
        self._tracer = tracer
        self._enabled = enabled
        self._exporter_name = _safe_identifier(exporter_name, "exporter_name")
        self._owns_provider = owns_provider
        self._shutdown = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def exporter_name(self) -> str:
        return self._exporter_name

    def force_flush(self, timeout_millis: int) -> bool:
        """Flush the configured provider when it exposes the SDK hook."""

        if type(timeout_millis) is not int or timeout_millis < 1:
            raise ValueError("timeout_millis must be a positive integer")
        if self._shutdown:
            return True
        force_flush = getattr(self._provider, "force_flush", None)
        if not callable(force_flush):
            return True
        result = force_flush(timeout_millis=timeout_millis)
        return result is not False

    def shutdown(self) -> None:
        """Flush and close an exporter provider owned by this tracer factory."""

        if self._shutdown:
            return
        self._shutdown = True
        if not self._owns_provider:
            return
        shutdown = getattr(self._provider, "shutdown", None)
        if callable(shutdown):
            shutdown()

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
        requested_trace_id = _numeric_trace_id(trace_id) if trace_id is not None else None
        token = _REQUESTED_TRACE_ID.set(requested_trace_id)
        try:
            # An empty OTel context makes this the true trace root. The isolated
            # provider's ID generator preserves the already-issued HTTP trace ID.
            async with self._span("rag.request", attributes, context=Context()) as span:
                yield span
        finally:
            _REQUESTED_TRACE_ID.reset(token)

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


def _numeric_trace_id(trace_id: str) -> int:
    normalized = trace_id.lower()
    if len(normalized) != 32:
        raise ValueError("trace_id must be a 32-character hexadecimal identifier")
    try:
        numeric_trace_id = int(normalized, 16)
    except ValueError:
        raise ValueError("trace_id must be a 32-character hexadecimal identifier") from None
    if numeric_trace_id == 0:
        raise ValueError("trace_id must not be all zeroes")
    return numeric_trace_id
