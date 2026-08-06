"""Provider-neutral state and render models for the Gradio workbench."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from rag_mvp.domain.qa import Citation, ValidatedStreamEvent


def _new_owner_id() -> str:
    return f"ui_owner_{uuid4().hex}"


@dataclass(frozen=True, slots=True)
class BrowserSessionState:
    """Mutable browser concerns represented as an immutable, session-local value."""

    owner_id: str
    session_id: str | None = None
    evaluation_run_id: str | None = None
    active_request_id: str | None = None

    @classmethod
    def create(cls) -> BrowserSessionState:
        return cls(owner_id=_new_owner_id())

    def with_session(self, session_id: str | None) -> BrowserSessionState:
        return replace(self, session_id=session_id, active_request_id=None)

    def with_evaluation(self, run_id: str | None) -> BrowserSessionState:
        return replace(self, evaluation_run_id=run_id)

    def with_active_request(self, request_id: str | None) -> BrowserSessionState:
        return replace(self, active_request_id=request_id)


@dataclass(frozen=True, slots=True)
class SourcePreview:
    citation: Citation
    preview: str | None = None


@dataclass(frozen=True, slots=True)
class ChatServiceResult:
    event: ValidatedStreamEvent
    previews: tuple[SourcePreview, ...] = ()


@dataclass(frozen=True, slots=True)
class ChatRender:
    history: tuple[dict[str, str], ...]
    state: BrowserSessionState
    citations_markdown: str
    previews_markdown: str
    status_markdown: str


@dataclass(frozen=True, slots=True)
class UploadPayload:
    filename: str
    content: bytes
    declared_media_type: str | None = None
    source_key: str | None = None
    display_title: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentsRender:
    document_rows: tuple[tuple[Any, ...], ...]
    job_rows: tuple[tuple[Any, ...], ...]
    status_markdown: str


@dataclass(frozen=True, slots=True)
class EvaluationRender:
    run_rows: tuple[tuple[Any, ...], ...]
    failure_rows: tuple[tuple[Any, ...], ...]
    metrics_markdown: str
    status_markdown: str
    state: BrowserSessionState


@dataclass(frozen=True, slots=True)
class ReportDownload:
    run_id: str
    format: str
    path: Path


@dataclass(frozen=True, slots=True)
class DiagnosticsRender:
    health_rows: tuple[tuple[Any, ...], ...]
    request_rows: tuple[tuple[Any, ...], ...]
    status_markdown: str
