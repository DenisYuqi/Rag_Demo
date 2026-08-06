"""Narrow backend protocols and adapters shared by every workbench view."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import ValidationError

from rag_mvp.api.qa import QARuntimeServices, stream_qa_events
from rag_mvp.config.settings import Settings
from rag_mvp.domain.evaluation import EvaluationRun
from rag_mvp.domain.ingestion import Document, IngestionJob
from rag_mvp.domain.qa import Citation, RequestDiagnostic, ValidatedStreamEvent
from rag_mvp.domain.retrieval import RetrievalMode
from rag_mvp.ingestion.service import IngestionService
from rag_mvp.qa.query_rewrite import select_response_language
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, Redactor

from .models import ChatServiceResult, ReportDownload, SourcePreview, UploadPayload


class SourcePreviewLookup(Protocol):
    async def get_previews(
        self,
        request_id: str,
        citations: Sequence[Citation],
    ) -> Mapping[str, str]: ...


class ChatGateway(Protocol):
    async def submit(
        self,
        *,
        owner_id: str,
        session_id: str | None,
        question: str,
        mode: RetrievalMode,
    ) -> ChatServiceResult: ...

    def reset(self, *, owner_id: str, session_id: str | None) -> str: ...


class DocumentGateway(Protocol):
    def submit_upload(self, payload: UploadPayload) -> IngestionJob: ...

    def submit_reindex(self) -> IngestionJob: ...

    def submit_delete(self, source_id: str) -> IngestionJob: ...

    async def run_job(self, job_id: str) -> IngestionJob: ...

    def get_job(self, job_id: str) -> IngestionJob | None: ...

    def list_active_documents(self) -> tuple[str | None, tuple[Document, ...]]: ...

    def list_jobs(self) -> Sequence[IngestionJob]: ...


class EvaluationCompatibilityError(ValueError):
    def __init__(self, code: str = "evaluation_runs_incompatible") -> None:
        self.code = code
        super().__init__(code)


class EvaluationGateway(Protocol):
    async def start(
        self,
        dataset_id: str,
        dataset_version: str | None = None,
    ) -> EvaluationRun: ...

    def get_run(self, run_id: str) -> EvaluationRun | None: ...

    def list_runs(self) -> Sequence[EvaluationRun]: ...

    def failed_cases(self, run_id: str) -> Sequence[Mapping[str, object]]: ...

    def compare_runs(
        self,
        baseline_run_id: str,
        candidate_run_id: str,
    ) -> Mapping[str, object]: ...

    def get_report(
        self,
        run_id: str,
        format: Literal["json", "html"],
    ) -> ReportDownload | None: ...


class RequestDiagnosticsLookup(Protocol):
    def get(self, request_id: str) -> RequestDiagnostic | None: ...


@dataclass(frozen=True, slots=True)
class HealthComponent:
    name: str
    ready: bool
    reason: str | None = None


class DiagnosticsGateway(Protocol):
    def health(self) -> Sequence[HealthComponent]: ...

    def get_request(self, request_id: str) -> RequestDiagnostic | None: ...


@dataclass(frozen=True, slots=True)
class WorkbenchServices:
    chat: ChatGateway | None = None
    documents: DocumentGateway | None = None
    evaluations: EvaluationGateway | None = None
    diagnostics: DiagnosticsGateway | None = None
    redactor: Redactor | None = DEFAULT_REDACTOR


@dataclass(frozen=True, slots=True)
class SharedQAGateway:
    """Use the same orchestrator and release emitter as the HTTP QA endpoint."""

    services: QARuntimeServices
    redactor: Redactor = DEFAULT_REDACTOR
    preview_lookup: SourcePreviewLookup | None = None

    async def submit(
        self,
        *,
        owner_id: str,
        session_id: str | None,
        question: str,
        mode: RetrievalMode,
    ) -> ChatServiceResult:
        resolved_session_id = session_id
        if resolved_session_id is None:
            resolved_session_id = self.services.conversations.create_session(owner_id).session_id
        else:
            self.services.conversations.get_session(resolved_session_id, owner_id)
        request_id = f"request_{uuid4().hex}"
        response_language = select_response_language(question)
        stream = stream_qa_events(
            self.services,
            request_id=request_id,
            session_id=resolved_session_id,
            owner_id=owner_id,
            question=question,
            mode=mode,
            requested_language=None,
            response_language=response_language,
            redactor=self.redactor,
        )
        payloads: list[bytes] = []
        async for payload in stream:
            payloads.append(payload)
        if len(payloads) != 1:
            raise ValueError("validated_stream_contract_invalid")
        try:
            event = ValidatedStreamEvent.model_validate(json.loads(payloads[0]))
        except (TypeError, ValueError, ValidationError):
            raise ValueError("validated_stream_contract_invalid") from None
        if event.request_id != request_id or event.session_id != resolved_session_id:
            raise ValueError("validated_stream_identity_invalid")
        previews: tuple[SourcePreview, ...]
        if self.preview_lookup is None:
            previews = tuple(SourcePreview(citation=citation) for citation in event.citations)
        else:
            values = await self.preview_lookup.get_previews(request_id, event.citations)
            previews = tuple(
                SourcePreview(citation=citation, preview=values.get(citation.chunk_id))
                for citation in event.citations
            )
        return ChatServiceResult(event=event, previews=previews)

    def reset(self, *, owner_id: str, session_id: str | None) -> str:
        if session_id is not None:
            self.services.conversations.reset_session(session_id, owner_id)
        return self.services.conversations.create_session(owner_id).session_id


@dataclass(frozen=True, slots=True)
class SharedDocumentGateway:
    """Expose the production ingestion service through UI-sized operations."""

    service: IngestionService

    def submit_upload(self, payload: UploadPayload) -> IngestionJob:
        return self.service.submit_upload(
            payload.filename,
            payload.content,
            source_key=payload.source_key,
            declared_media_type=payload.declared_media_type,
            display_title=payload.display_title,
        )

    def submit_reindex(self) -> IngestionJob:
        return self.service.submit_reindex()

    def submit_delete(self, source_id: str) -> IngestionJob:
        return self.service.submit_delete(source_id)

    async def run_job(self, job_id: str) -> IngestionJob:
        return await self.service.run(job_id)

    def get_job(self, job_id: str) -> IngestionJob | None:
        return self.service.get_job(job_id)

    def list_active_documents(self) -> tuple[str | None, tuple[Document, ...]]:
        return self.service.list_active_documents()

    def list_jobs(self) -> Sequence[IngestionJob]:
        return self.service.repositories.ingestion_jobs.list()


@dataclass(frozen=True, slots=True)
class RuntimeDiagnosticsGateway:
    """Safe health and request lookup adapter for the future diagnostics service."""

    request_lookup: RequestDiagnosticsLookup
    readiness_report: Callable[[], tuple[bool, Sequence[object]]]

    def health(self) -> Sequence[HealthComponent]:
        _, raw_components = self.readiness_report()
        values: list[HealthComponent] = []
        for raw in raw_components:
            name = getattr(raw, "name", None)
            ready = getattr(raw, "ready", None)
            reason = getattr(raw, "reason", None)
            if not isinstance(name, str) or not isinstance(ready, bool):
                raise ValueError("health_component_invalid")
            if reason is not None and not isinstance(reason, str):
                raise ValueError("health_component_invalid")
            values.append(HealthComponent(name=name, ready=ready, reason=reason))
        return tuple(values)

    def get_request(self, request_id: str) -> RequestDiagnostic | None:
        return self.request_lookup.get(request_id)


def configured_workbench_services(
    *,
    settings: Settings,
    qa: QARuntimeServices | None,
    ingestion: IngestionService | None,
    diagnostics: DiagnosticsGateway | None = None,
    evaluations: EvaluationGateway | None = None,
    redactor: Redactor | None = DEFAULT_REDACTOR,
) -> WorkbenchServices:
    """Compose available backends without claiming unavailable future capabilities."""

    chat = SharedQAGateway(qa, redactor) if qa and redactor else None
    documents = SharedDocumentGateway(ingestion) if ingestion is not None else None
    del settings
    return WorkbenchServices(
        chat=chat,
        documents=documents,
        evaluations=evaluations,
        diagnostics=diagnostics,
        redactor=redactor,
    )


type AsyncRunStarter = Callable[[str, str | None], Awaitable[EvaluationRun]]
type ReportPathResolver = Callable[[str, Literal["json", "html"]], Path | None]
