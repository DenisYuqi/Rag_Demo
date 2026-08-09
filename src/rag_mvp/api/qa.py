"""Grounded QA HTTP route and fail-closed validated stream transport."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, cast
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, Request, status
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from pydantic.json_schema import models_json_schema

from rag_mvp.api.errors import ApiError
from rag_mvp.api.readiness import ComponentStatus, ReadinessRegistry
from rag_mvp.api.schemas import SESSION_ID_PATTERN, QARequestBody
from rag_mvp.config.settings import Settings
from rag_mvp.domain.qa import (
    QAErrorCode,
    StreamEventKind,
    ValidatedStreamEvent,
)
from rag_mvp.domain.retrieval import CachePolicy, RetrievalMode
from rag_mvp.observability.logging import current_correlation_context
from rag_mvp.observability.runtime import PipelineTelemetry, RequestObservation
from rag_mvp.performance.admission import (
    AdmissionClosedError,
    AdmissionLease,
    AdmissionRejectedError,
    QAAdmissionController,
)
from rag_mvp.performance.deadlines import (
    DeadlineController,
    DeadlineExceededError,
    QALatencyBudgets,
    StageDeadlineExceededError,
)
from rag_mvp.performance.load_report import INSTANCE_ID_HEADER
from rag_mvp.qa.orchestrator import OrchestratedResponse
from rag_mvp.qa.query_rewrite import select_response_language
from rag_mvp.qa.sessions import ConversationService
from rag_mvp.safety.output import SAFE_UNAVAILABLE_MESSAGE, redact_output
from rag_mvp.safety.redactor import Redactor
from rag_mvp.storage.repositories import RepositoryError


class QAOrchestratorGateway(Protocol):
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
    ) -> OrchestratedResponse: ...


class QAEventEmitter(Protocol):
    @property
    def ready(self) -> bool: ...

    def emit(
        self,
        outcome: OrchestratedResponse,
        *,
        owner_id: str,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class ProfileLoadStatus:
    """Content-free progress for a retrieval profile's required model assets."""

    state: Literal["loading", "ready", "failed"]
    completed_steps: int
    total_steps: int
    active_step: str | None = None
    safe_error_code: str | None = None

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("profile load total_steps must be positive")
        if not 0 <= self.completed_steps <= self.total_steps:
            raise ValueError("profile load completed_steps is out of range")
        if self.state == "ready" and self.completed_steps != self.total_steps:
            raise ValueError("ready profile load status must be complete")
        if self.state == "failed" and not self.safe_error_code:
            raise ValueError("failed profile load status requires a safe error code")

    @classmethod
    def ready(cls) -> ProfileLoadStatus:
        return cls("ready", 1, 1)

    @property
    def is_ready(self) -> bool:
        return self.state == "ready"

    @property
    def progress_percent(self) -> int:
        return round((self.completed_steps / self.total_steps) * 100)


@dataclass(frozen=True, slots=True)
class QARuntimeServices:
    conversations: ConversationService
    orchestrator: QAOrchestratorGateway
    emitter: QAEventEmitter
    readiness_probe: Callable[[], tuple[bool, str | None]] | None = None
    close_callback: Callable[[], Awaitable[None]] | None = None
    admission: QAAdmissionController | None = None
    telemetry: PipelineTelemetry | None = None
    latency_budgets: QALatencyBudgets | None = None
    profile_load_status_probe: Callable[[], ProfileLoadStatus] | None = None

    def profile_load_status(self) -> ProfileLoadStatus:
        if self.profile_load_status_probe is None:
            return ProfileLoadStatus.ready()
        try:
            return self.profile_load_status_probe()
        except Exception:
            return ProfileLoadStatus("failed", 0, 1, safe_error_code="profile-load-status-error")

    def check_readiness(self) -> tuple[bool, str | None]:
        try:
            if self.emitter.ready is not True:
                return False, "qa_release_unavailable"
            if self.readiness_probe is None:
                return True, None
            ready, reason = self.readiness_probe()
            if ready:
                return True, None
            return False, reason or "qa_unavailable"
        except Exception:
            return False, "qa_check_failed"

    async def close(self) -> None:
        if self.close_callback is not None:
            await self.close_callback()


@dataclass(frozen=True, slots=True)
class QARuntimeReadinessCheck:
    services: QARuntimeServices
    name: str = "qa"

    def check(self) -> ComponentStatus:
        ready, reason = self.services.check_readiness()
        return ComponentStatus(self.name, ready, reason)


class QAApiRuntime(Protocol):
    settings: Settings
    instance_identity: str
    readiness: ReadinessRegistry
    accepting_traffic: bool
    qa_services: QARuntimeServices | None
    redactor: Redactor | None


router = APIRouter(prefix="/api/v1", tags=["qa"])

NDJSON_MEDIA_TYPE = "application/x-ndjson"
_REQUIRED_READINESS = ("configuration", "providers", "safety", "storage", "qa")
_, _STREAM_EVENT_JSON_SCHEMA = models_json_schema(
    [(ValidatedStreamEvent, "serialization")],
    ref_template="#/components/schemas/{model}",
)
_STREAM_EVENT_SCHEMAS = cast(dict[str, dict[str, Any]], _STREAM_EVENT_JSON_SCHEMA["$defs"])


class _NDJSONStreamingResponse(StreamingResponse):
    media_type = NDJSON_MEDIA_TYPE


_QA_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_200_OK: {
        "description": "One validated QA event per NDJSON line.",
        "content": {
            NDJSON_MEDIA_TYPE: {
                "schema": {
                    "type": "string",
                    "description": "UTF-8 NDJSON containing one validated terminal event.",
                },
                "x-ndjson-item-schema": {"$ref": "#/components/schemas/ValidatedStreamEvent"},
            }
        },
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "content": {
            "application/json": {"schema": {"$ref": "#/components/schemas/ApiErrorResponse"}}
        }
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "content": {
            "application/json": {"schema": {"$ref": "#/components/schemas/ApiErrorResponse"}}
        }
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "content": {
            "application/json": {"schema": {"$ref": "#/components/schemas/ApiErrorResponse"}}
        }
    },
}


class StreamContractError(ValueError):
    """Content-free marker for a malformed internal stream."""


def _runtime(request: Request) -> QAApiRuntime:
    return cast(QAApiRuntime, request.app.state.runtime)


def install_qa_openapi_contract(app: FastAPI) -> None:
    base_openapi = app.openapi

    def openapi() -> dict[str, Any]:
        schema = base_openapi()
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        for name, definition in _STREAM_EVENT_SCHEMAS.items():
            components.setdefault(name, definition)
        return schema

    app.openapi = openapi  # type: ignore[method-assign]


def _require_qa(request: Request) -> QARuntimeServices:
    runtime = _runtime(request)
    redactor = runtime.redactor
    if (
        not runtime.accepting_traffic
        or runtime.qa_services is None
        or redactor is None
        or not redactor.fully_configured
    ):
        raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "qa_unavailable")
    try:
        ready = all(runtime.readiness.get(name).check().ready for name in _REQUIRED_READINESS)
    except Exception:
        ready = False
    if not ready:
        raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "qa_unavailable")
    return runtime.qa_services


def _safe_failure_event(
    request_id: str,
    session_id: str,
    response_language: str,
    *,
    error_code: QAErrorCode,
    retryable: bool,
) -> ValidatedStreamEvent:
    return ValidatedStreamEvent(
        request_id=request_id,
        session_id=session_id,
        sequence=0,
        kind=StreamEventKind.ERROR,
        response_language=response_language,
        content=SAFE_UNAVAILABLE_MESSAGE,
        error_code=error_code,
        retryable=retryable,
        terminal=True,
    )


def _validated_events(
    raw_events: object,
    *,
    request_id: str,
    session_id: str,
    response_language: str,
    redactor: Redactor,
) -> tuple[ValidatedStreamEvent, ...]:
    if not isinstance(raw_events, tuple) or len(raw_events) != 1:
        raise StreamContractError("stream_events_invalid")
    try:
        raw_event = ValidatedStreamEvent.model_validate(raw_events[0])
        _validate_stream_event_contract(
            raw_event,
            request_id=request_id,
            session_id=session_id,
            response_language=response_language,
        )
        redacted = _restore_stream_structural_ids(
            redact_output(raw_event, redactor=redactor),
            raw_event,
        )
        event = ValidatedStreamEvent.model_validate(redacted)
    except (TypeError, ValueError, ValidationError, RecursionError):
        raise StreamContractError("stream_event_invalid") from None
    _validate_stream_event_contract(
        event,
        request_id=request_id,
        session_id=session_id,
        response_language=response_language,
    )
    return (event,)


def _validate_stream_event_contract(
    event: ValidatedStreamEvent,
    *,
    request_id: str,
    session_id: str,
    response_language: str,
) -> None:
    if (
        event.request_id != request_id
        or event.session_id != session_id
        or event.response_language != response_language
        or event.sequence != 0
    ):
        raise StreamContractError("stream_identity_invalid")
    if not event.terminal or event.kind not in {
        StreamEventKind.ANSWER,
        StreamEventKind.REFUSAL,
        StreamEventKind.ERROR,
    }:
        raise StreamContractError("terminal_event_invalid")


def _restore_stream_structural_ids(
    redacted: object,
    original: ValidatedStreamEvent,
) -> dict[str, object]:
    """Keep trusted envelope IDs out of natural-language PII heuristics."""

    if not isinstance(redacted, dict):
        raise StreamContractError("stream_event_invalid")
    payload = dict(redacted)
    payload["request_id"] = original.request_id
    payload["session_id"] = original.session_id
    payload["response_language"] = original.response_language

    claims = payload.get("claims")
    if not isinstance(claims, list) or len(claims) != len(original.claims):
        raise StreamContractError("stream_event_invalid")
    for raw_claim, claim in zip(claims, original.claims, strict=True):
        if not isinstance(raw_claim, dict):
            raise StreamContractError("stream_event_invalid")
        raw_claim["citation_chunk_ids"] = list(claim.citation_chunk_ids)

    citations = payload.get("citations")
    if not isinstance(citations, list) or len(citations) != len(original.citations):
        raise StreamContractError("stream_event_invalid")
    for raw_citation, citation in zip(citations, original.citations, strict=True):
        if not isinstance(raw_citation, dict):
            raise StreamContractError("stream_event_invalid")
        raw_citation["chunk_id"] = citation.chunk_id
    return payload


async def stream_qa_events(
    services: QARuntimeServices,
    *,
    request_id: str,
    session_id: str,
    owner_id: str,
    question: str,
    mode: RetrievalMode | str,
    requested_language: str | None,
    response_language: str,
    redactor: Redactor,
    cache_policy: CachePolicy | str = CachePolicy.USE,
    deadline_controller: DeadlineController | None = None,
) -> AsyncIterator[bytes]:
    resolved_cache_policy = CachePolicy(cache_policy)
    deadline = deadline_controller
    if deadline is None and services.latency_budgets is not None:
        deadline = DeadlineController(services.latency_budgets)
    async with _observation(services.telemetry, request_id) as observation:
        try:

            async def operation() -> tuple[tuple[ValidatedStreamEvent, ...], float]:
                return await _produce_validated_events(
                    services,
                    request_id=request_id,
                    session_id=session_id,
                    owner_id=owner_id,
                    question=question,
                    mode=mode,
                    requested_language=requested_language,
                    response_language=response_language,
                    redactor=redactor,
                    cache_policy=resolved_cache_policy,
                    deadline=deadline,
                )

            if deadline is None:
                events, queue_duration_ms = await operation()
            else:
                events, queue_duration_ms = await deadline.run_remaining(operation)
        except DeadlineExceededError:
            events = (
                _safe_failure_event(
                    request_id,
                    session_id,
                    response_language,
                    error_code=QAErrorCode.DEADLINE_EXPIRED,
                    retryable=True,
                ),
            )
            queue_duration_ms = 0.0
        if queue_duration_ms > 0:
            events = tuple(_with_queue_timing(event, queue_duration_ms) for event in events)
        if observation is not None:
            observation.complete(events[0])

    for event in events:
        yield f"{event.model_dump_json(exclude_none=True)}\n".encode()


async def _produce_validated_events(
    services: QARuntimeServices,
    *,
    request_id: str,
    session_id: str,
    owner_id: str,
    question: str,
    mode: RetrievalMode | str,
    requested_language: str | None,
    response_language: str,
    redactor: Redactor,
    cache_policy: CachePolicy,
    deadline: DeadlineController | None,
) -> tuple[tuple[ValidatedStreamEvent, ...], float]:
    admission = services.admission
    lease = None
    active_recorded = False
    queue_started = asyncio.get_running_loop().time()
    try:
        if admission is not None:
            try:

                async def acquire() -> AdmissionLease:
                    return await admission.acquire()

                if services.telemetry is not None:
                    services.telemetry.metrics.set_queue_depth(admission.queued_count)
                    async with services.telemetry.stage("queue"):
                        lease = (
                            await deadline.run_required("queue", acquire)
                            if deadline is not None
                            else await admission.acquire()
                        )
                else:
                    lease = (
                        await deadline.run_required("queue", acquire)
                        if deadline is not None
                        else await admission.acquire()
                    )
            except StageDeadlineExceededError:
                if services.telemetry is not None:
                    services.telemetry.metrics.record_queue_rejection("capacity")
                return (
                    (
                        _safe_failure_event(
                            request_id,
                            session_id,
                            response_language,
                            error_code=QAErrorCode.CAPACITY,
                            retryable=True,
                        ),
                    ),
                    max(
                        0.0,
                        (asyncio.get_running_loop().time() - queue_started) * 1_000,
                    ),
                )
        queue_duration_ms = max(
            0.0,
            (asyncio.get_running_loop().time() - queue_started) * 1_000,
        )
        if services.telemetry is not None:
            services.telemetry.metrics.pipeline_started()
            active_recorded = True

        async def run_orchestrator() -> OrchestratedResponse:
            return await services.orchestrator.run(
                request_id=request_id,
                session_id=session_id,
                owner_id=owner_id,
                question=question,
                mode=mode,
                requested_language=requested_language,
                cache_policy=cache_policy,
            )

        outcome = (
            await deadline.run_remaining(run_orchestrator)
            if deadline is not None
            else await run_orchestrator()
        )
        try:
            if services.telemetry is None:
                raw_events = services.emitter.emit(outcome, owner_id=owner_id)
                events = _validated_events(
                    raw_events,
                    request_id=request_id,
                    session_id=session_id,
                    response_language=response_language,
                    redactor=redactor,
                )
            else:
                async with (
                    services.telemetry.stage("safety"),
                    services.telemetry.stage("redaction"),
                ):
                    raw_events = (
                        deadline.run_sync_required(
                            "redaction",
                            lambda: services.emitter.emit(outcome, owner_id=owner_id),
                        )
                        if deadline is not None
                        else services.emitter.emit(outcome, owner_id=owner_id)
                    )
                async with services.telemetry.stage("serialization"):

                    def serialize() -> tuple[ValidatedStreamEvent, ...]:
                        return _validated_events(
                            raw_events,
                            request_id=request_id,
                            session_id=session_id,
                            response_language=response_language,
                            redactor=redactor,
                        )

                    events = (
                        deadline.run_sync_required("serialization", serialize)
                        if deadline is not None
                        else serialize()
                    )
        except asyncio.CancelledError:
            raise
        except DeadlineExceededError:
            raise
        except Exception:
            events = (
                _safe_failure_event(
                    request_id,
                    session_id,
                    response_language,
                    error_code=QAErrorCode.SAFETY_UNAVAILABLE,
                    retryable=True,
                ),
            )
        return events, queue_duration_ms
    except asyncio.CancelledError:
        raise
    except (AdmissionRejectedError, AdmissionClosedError):
        if services.telemetry is not None:
            services.telemetry.metrics.record_queue_rejection(
                "shutdown"
                if isinstance(admission, QAAdmissionController) and admission.closed
                else "capacity"
            )
        return (
            (
                _safe_failure_event(
                    request_id,
                    session_id,
                    response_language,
                    error_code=QAErrorCode.CAPACITY,
                    retryable=True,
                ),
            ),
            max(0.0, (asyncio.get_running_loop().time() - queue_started) * 1_000),
        )
    except DeadlineExceededError:
        raise
    except Exception:
        return (
            (
                _safe_failure_event(
                    request_id,
                    session_id,
                    response_language,
                    error_code=QAErrorCode.INTERNAL,
                    retryable=False,
                ),
            ),
            max(0.0, (asyncio.get_running_loop().time() - queue_started) * 1_000),
        )
    finally:
        if active_recorded and services.telemetry is not None:
            services.telemetry.metrics.pipeline_finished()
        if lease is not None:
            await lease.release()
        if admission is not None and services.telemetry is not None:
            services.telemetry.metrics.set_queue_depth(admission.queued_count)


@asynccontextmanager
async def _observation(
    telemetry: PipelineTelemetry | None,
    request_id: str,
) -> AsyncIterator[RequestObservation | None]:
    if telemetry is None:
        yield None
        return
    async with telemetry.request(request_id) as observation:
        yield observation


def _with_queue_timing(
    event: ValidatedStreamEvent,
    duration_ms: float,
) -> ValidatedStreamEvent:
    diagnostics = event.diagnostics.model_copy(
        update={
            "stage_timings_ms": {
                **event.diagnostics.stage_timings_ms,
                "queue": duration_ms,
            }
        }
    )
    return event.model_copy(update={"diagnostics": diagnostics})


@router.post(
    "/qa",
    response_class=_NDJSONStreamingResponse,
    responses=_QA_RESPONSES,
    summary="Run grounded question answering",
)
async def answer_question(
    payload: QARequestBody,
    request: Request,
    services: Annotated[QARuntimeServices, Depends(_require_qa)],
) -> _NDJSONStreamingResponse:
    runtime = _runtime(request)
    deadline = (
        DeadlineController(services.latency_budgets)
        if services.latency_budgets is not None
        else None
    )
    session_id = payload.session_id
    if session_id is None:
        try:
            session_id = services.conversations.create_session(payload.owner_id).session_id
        except RepositoryError:
            raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "qa_unavailable") from None
    if re.fullmatch(SESSION_ID_PATTERN, session_id) is None:
        raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "qa_unavailable")
    correlation = current_correlation_context()
    request_id = correlation.request_id if correlation is not None else f"request_{uuid4().hex}"
    mode = payload.mode or runtime.settings.default_retrieval_mode
    cache_policy_header = request.headers.get("x-rag-cache-policy", CachePolicy.USE.value)
    try:
        cache_policy = CachePolicy(cache_policy_header)
    except ValueError:
        raise ApiError(status.HTTP_422_UNPROCESSABLE_CONTENT, "cache_policy_invalid") from None
    try:
        response_language = select_response_language(
            payload.question,
            requested_language=payload.requested_language,
        )
    except Exception:
        response_language = "en"
    stream = stream_qa_events(
        services,
        request_id=request_id,
        session_id=session_id,
        owner_id=payload.owner_id,
        question=payload.question,
        mode=mode,
        requested_language=payload.requested_language,
        response_language=response_language,
        redactor=cast(Redactor, runtime.redactor),
        cache_policy=cache_policy,
        deadline_controller=deadline,
    )
    return _NDJSONStreamingResponse(
        stream,
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
            "X-Request-ID": request_id,
            "X-RAG-Cache-Policy": cache_policy.value,
            INSTANCE_ID_HEADER: runtime.instance_identity,
        },
    )
