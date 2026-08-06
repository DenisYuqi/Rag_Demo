"""Privacy-safe structured logging and HTTP correlation context."""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import sys
import time
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, TextIO, cast

import structlog
from opentelemetry import trace
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from structlog.typing import EventDict, FilteringBoundLogger, WrappedLogger

from rag_mvp.safety.telemetry import TelemetryFilter

_SAFE_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_TRACE_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"^[0-9a-fA-F]{32}$")
_REQUEST_ID: ContextVar[str | None] = ContextVar("rag_mvp_request_id", default=None)
_TRACE_ID: ContextVar[str | None] = ContextVar("rag_mvp_trace_id", default=None)


class SafeErrorCategory(StrEnum):
    """Stable error categories that never include unrestricted exception content."""

    VALIDATION = "validation"
    CAPACITY = "capacity"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate-limit"
    DEPENDENCY = "dependency"
    UNAVAILABLE = "unavailable"
    INTERNAL = "internal"


@dataclass(frozen=True, slots=True)
class CorrelationContext:
    """Request and trace identifiers propagated through asynchronous work."""

    request_id: str
    trace_id: str


def is_safe_identifier(value: str) -> bool:
    """Return whether a value is suitable for a content-free correlation field."""

    return bool(_SAFE_IDENTIFIER.fullmatch(value))


def _validated_identifier(value: str, *, field: str) -> str:
    if not is_safe_identifier(value):
        raise ValueError(f"{field} must be a bounded opaque identifier")
    return value


def _validated_trace_id(value: str) -> str:
    if not _TRACE_IDENTIFIER.fullmatch(value):
        raise ValueError("trace_id must be a 32-character hexadecimal identifier")
    return value.lower()


def current_correlation_context() -> CorrelationContext | None:
    """Return the correlation context currently bound to this async execution path."""

    request_id = _REQUEST_ID.get()
    trace_id = _TRACE_ID.get()
    if request_id is None or trace_id is None:
        return None
    return CorrelationContext(request_id=request_id, trace_id=trace_id)


@contextmanager
def bind_correlation_context(request_id: str, trace_id: str) -> Iterator[CorrelationContext]:
    """Bind validated identifiers and restore the prior context on exit."""

    context = CorrelationContext(
        request_id=_validated_identifier(request_id, field="request_id"),
        trace_id=_validated_trace_id(trace_id),
    )
    request_token = _REQUEST_ID.set(context.request_id)
    trace_token = _TRACE_ID.set(context.trace_id)
    try:
        yield context
    finally:
        _TRACE_ID.reset(trace_token)
        _REQUEST_ID.reset(request_token)


def classify_exception(error: BaseException) -> SafeErrorCategory:
    """Classify an exception without inspecting or exporting its message."""

    declared = getattr(error, "safe_error_category", None)
    if isinstance(declared, SafeErrorCategory):
        return declared
    if isinstance(declared, str):
        try:
            return SafeErrorCategory(declared)
        except ValueError:
            pass

    if isinstance(error, asyncio.CancelledError):
        return SafeErrorCategory.CANCELLED
    if isinstance(error, TimeoutError):
        return SafeErrorCategory.TIMEOUT
    if isinstance(error, PermissionError):
        return SafeErrorCategory.AUTHENTICATION
    if isinstance(error, ConnectionError):
        return SafeErrorCategory.DEPENDENCY
    if isinstance(error, ValueError | TypeError):
        return SafeErrorCategory.VALIDATION

    class_name = type(error).__name__.casefold()
    if "capacity" in class_name or "queuefull" in class_name:
        return SafeErrorCategory.CAPACITY
    if "ratelimit" in class_name or "rate_limit" in class_name:
        return SafeErrorCategory.RATE_LIMIT
    if "auth" in class_name or "permission" in class_name:
        return SafeErrorCategory.AUTHENTICATION
    if "timeout" in class_name or "deadline" in class_name:
        return SafeErrorCategory.TIMEOUT
    if "unavailable" in class_name or "notready" in class_name:
        return SafeErrorCategory.UNAVAILABLE
    return SafeErrorCategory.INTERNAL


class _StaticFields:
    def __init__(self, *, service: str, service_version: str, config_version: str) -> None:
        self._fields = {
            "service": _validated_identifier(service, field="service"),
            "service_version": _validated_identifier(service_version, field="service_version"),
            "config_version": _validated_identifier(config_version, field="config_version"),
        }

    def __call__(
        self,
        logger: WrappedLogger,
        method_name: str,
        event_dict: EventDict,
    ) -> EventDict:
        del logger, method_name
        for key, value in self._fields.items():
            event_dict.setdefault(key, value)
        context = current_correlation_context()
        if context is not None:
            event_dict.setdefault("request_id", context.request_id)
            event_dict.setdefault("trace_id", context.trace_id)
        return event_dict


class _AllowlistedEvent:
    def __init__(self, telemetry_filter: TelemetryFilter) -> None:
        self._filter = telemetry_filter

    def __call__(
        self,
        logger: WrappedLogger,
        method_name: str,
        event_dict: EventDict,
    ) -> EventDict:
        del logger, method_name
        filtered = self._filter.filter(event_dict)
        if filtered is None:
            raise structlog.DropEvent
        return cast(EventDict, filtered)


def configure_logging(
    *,
    service: str,
    service_version: str,
    config_version: str,
    level: str = "INFO",
    stream: TextIO | None = None,
    telemetry_filter: TelemetryFilter | None = None,
) -> TelemetryFilter:
    """Configure deterministic JSON logging and return its fail-closed filter."""

    normalized_level = level.upper()
    numeric_level = getattr(logging, normalized_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError("unsupported log level")
    active_filter = telemetry_filter or TelemetryFilter()
    structlog.configure(
        processors=[
            _StaticFields(
                service=service,
                service_version=service_version,
                config_version=config_version,
            ),
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
            _AllowlistedEvent(active_filter),
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(file=stream or sys.stdout),
        cache_logger_on_first_use=False,
    )
    return active_filter


def get_logger(component: str | None = None) -> FilteringBoundLogger:
    """Return a structured logger with an optional safe operation binding."""

    logger = cast(FilteringBoundLogger, structlog.get_logger())
    if component is None:
        return logger
    return logger.bind(operation=_validated_identifier(component, field="component"))


def _active_trace_id() -> str | None:
    span_context = trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return None
    return f"{span_context.trace_id:032x}"


def _request_id(headers: Headers) -> str:
    candidate = headers.get("x-request-id")
    if candidate is not None and is_safe_identifier(candidate):
        return candidate
    return f"request-{secrets.token_hex(12)}"


def _trace_id(headers: Headers) -> str:
    active = _active_trace_id()
    if active is not None:
        return active
    candidate = headers.get("x-trace-id")
    if candidate is not None and _TRACE_IDENTIFIER.fullmatch(candidate) and int(candidate, 16) != 0:
        return candidate.lower()
    return secrets.token_hex(16)


class RequestTraceContextMiddleware:
    """Bind request/trace IDs and emit content-minimized request lifecycle events."""

    def __init__(self, app: ASGIApp, *, logger: FilteringBoundLogger | None = None) -> None:
        self._app = app
        self._logger = logger or get_logger("http.request")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        request_id = _request_id(headers)
        trace_id = _trace_id(headers)
        started = time.perf_counter()
        status_code = 500

        async def send_correlated(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                response_headers = MutableHeaders(scope=message)
                if "x-request-id" not in response_headers:
                    response_headers["x-request-id"] = request_id
                if "x-trace-id" not in response_headers:
                    response_headers["x-trace-id"] = trace_id
            await send(message)

        with bind_correlation_context(request_id, trace_id):
            self._logger.info(
                "http.request.started",
                operation="http.request",
                outcome="started",
            )
            try:
                await self._app(scope, receive, send_correlated)
            except asyncio.CancelledError as error:
                self._log_failure(error, started)
                raise
            except Exception as error:
                self._log_failure(error, started)
                raise
            else:
                self._logger.info(
                    "http.request.completed",
                    operation="http.request",
                    outcome="succeeded" if status_code < 500 else "failed",
                    status=status_code,
                    duration_ms=(time.perf_counter() - started) * 1_000,
                )

    def _log_failure(self, error: BaseException, started: float) -> None:
        self._logger.error(
            "http.request.failed",
            operation="http.request",
            outcome="failed",
            safe_error_category=classify_exception(error).value,
            duration_ms=(time.perf_counter() - started) * 1_000,
        )


def safe_event(
    event_name: str,
    **fields: object,
) -> Mapping[str, object]:
    """Build an event mapping while refusing known raw-content field names."""

    event = _validated_identifier(event_name, field="event_name")
    payload: MutableMapping[str, object] = {"event_name": event}
    payload.update(fields)
    filtered = TelemetryFilter().filter(payload)
    if filtered is None:
        raise ValueError("event could not be safely filtered")
    return filtered
