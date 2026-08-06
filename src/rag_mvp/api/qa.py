"""Grounded QA HTTP route and fail-closed validated stream transport."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any, Protocol, cast
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
class QARuntimeServices:
    conversations: ConversationService
    orchestrator: QAOrchestratorGateway
    emitter: QAEventEmitter
    readiness_probe: Callable[[], tuple[bool, str | None]] | None = None
    close_callback: Callable[[], Awaitable[None]] | None = None

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
        redacted = redact_output(raw_events[0], redactor=redactor)
        event = ValidatedStreamEvent.model_validate(redacted)
    except (TypeError, ValueError, ValidationError, RecursionError):
        raise StreamContractError("stream_event_invalid") from None
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
    return (event,)


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
) -> AsyncIterator[bytes]:
    events: tuple[ValidatedStreamEvent, ...]
    try:
        outcome = await services.orchestrator.run(
            request_id=request_id,
            session_id=session_id,
            owner_id=owner_id,
            question=question,
            mode=mode,
            requested_language=requested_language,
            cache_policy=CachePolicy.USE,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        events = (
            _safe_failure_event(
                request_id,
                session_id,
                response_language,
                error_code=QAErrorCode.INTERNAL,
                retryable=False,
            ),
        )
    else:
        try:
            raw_events = services.emitter.emit(outcome, owner_id=owner_id)
            events = _validated_events(
                raw_events,
                request_id=request_id,
                session_id=session_id,
                response_language=response_language,
                redactor=redactor,
            )
        except asyncio.CancelledError:
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

    for event in events:
        yield f"{event.model_dump_json(exclude_none=True)}\n".encode()


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
    session_id = payload.session_id
    if session_id is None:
        try:
            session_id = services.conversations.create_session(payload.owner_id).session_id
        except RepositoryError:
            raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "qa_unavailable") from None
    if re.fullmatch(SESSION_ID_PATTERN, session_id) is None:
        raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "qa_unavailable")
    request_id = f"request_{uuid4().hex}"
    mode = payload.mode or runtime.settings.default_retrieval_mode
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
    )
    return _NDJSONStreamingResponse(
        stream,
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
            "X-Request-ID": request_id,
        },
    )
