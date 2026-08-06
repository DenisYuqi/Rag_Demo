"""Privacy-safe callback controller for the Gradio workbench."""

from __future__ import annotations

import json
import mimetypes
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

import anyio

from rag_mvp.domain.evaluation import EvaluationRun
from rag_mvp.domain.ingestion import Document, IngestionJob, IngestionJobStatus
from rag_mvp.domain.qa import RequestDiagnostic, StreamEventKind, ValidatedStreamEvent
from rag_mvp.domain.retrieval import RetrievalMode
from rag_mvp.safety.output import redact_output
from rag_mvp.safety.redactor import Redactor

from .models import (
    BrowserSessionState,
    ChatRender,
    DiagnosticsRender,
    DocumentsRender,
    EvaluationRender,
    UploadPayload,
)
from .services import EvaluationCompatibilityError, WorkbenchServices

SAFE_UI_ERROR = "Request could not be completed safely. / 请求无法安全完成。"
SAFE_UNAVAILABLE = "This capability is unavailable. / 此功能暂不可用。"
SAFE_CANCELLED = "Request cancelled; no pending text was shown. / 请求已取消, 未显示待验证文本。"

_SAFE_DIAGNOSTIC_METADATA = frozenset(
    {
        "candidate_count",
        "citation_count",
        "configuration_id",
        "currency",
        "estimated_cost",
        "index_revision",
        "redaction_count",
        "retrieval_mode",
    }
)


def _state(value: BrowserSessionState | None) -> BrowserSessionState:
    return value if isinstance(value, BrowserSessionState) else BrowserSessionState.create()


def _history(value: Sequence[Mapping[str, object]] | None) -> tuple[dict[str, str], ...]:
    if value is None:
        return ()
    result: list[dict[str, str]] = []
    for item in value:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            result.append({"role": role, "content": content})
    return tuple(result)


def _safe_value(value: object, redactor: Redactor | None) -> object:
    if redactor is None or not redactor.fully_configured:
        raise ValueError("output_redaction_unavailable")
    return redact_output(value, redactor=redactor)


def _safe_text(value: object, redactor: Redactor | None) -> str:
    redacted = _safe_value(value, redactor)
    if not isinstance(redacted, str):
        raise ValueError("safe_text_invalid")
    return redacted


def _locator(citation: object) -> str:
    locator = getattr(citation, "locator", None)
    pages = getattr(locator, "pages", ())
    section_path = getattr(locator, "section_path", ())
    char_start = getattr(locator, "char_start", None)
    char_end = getattr(locator, "char_end", None)
    if pages:
        return "pages " + ", ".join(str(page) for page in pages)
    if section_path:
        return " / ".join(str(part) for part in section_path)
    if char_start is not None and char_end is not None:
        return f"chars {char_start}-{char_end}"
    return "locator unavailable"


class WorkbenchCallbacks:
    """Small UI adapter whose methods are directly unit-testable."""

    def __init__(self, services: WorkbenchServices) -> None:
        self.services = services

    def new_session(self) -> BrowserSessionState:
        return BrowserSessionState.create()

    async def submit_chat(
        self,
        question: str,
        mode: RetrievalMode | str,
        history: Sequence[Mapping[str, object]] | None,
        state: BrowserSessionState | None,
    ) -> ChatRender:
        current = _state(state)
        prior = _history(history)
        if self.services.chat is None:
            return ChatRender(prior, current, "", "", SAFE_UNAVAILABLE)
        if not isinstance(question, str) or not question.strip():
            return ChatRender(prior, current, "", "", "Enter a question. / 请输入问题。")
        correlation = current.active_request_id or f"ui_{uuid4().hex}"
        try:
            selected_mode = RetrievalMode(mode)
            result = await self.services.chat.submit(
                owner_id=current.owner_id,
                session_id=current.session_id,
                question=question,
                mode=selected_mode,
            )
            event = result.event
            if isinstance(event, ValidatedStreamEvent):
                correlation = _safe_text(event.request_id, self.services.redactor)
            if (
                not isinstance(event, ValidatedStreamEvent)
                or not event.terminal
                or event.kind
                not in {StreamEventKind.ANSWER, StreamEventKind.REFUSAL, StreamEventKind.ERROR}
            ):
                raise ValueError("validated_event_invalid")
            safe_question = _safe_text(question, self.services.redactor)
            safe_content = _safe_text(event.content or SAFE_UI_ERROR, self.services.redactor)
            citations: list[str] = []
            previews: list[str] = []
            for index, citation in enumerate(event.citations, start=1):
                title = _safe_text(citation.source_title, self.services.redactor)
                location = _safe_text(_locator(citation), self.services.redactor)
                citations.append(
                    f"[{index}] **{title}**, v{citation.document_version}, "
                    f"{location} (`{citation.chunk_id}`)"
                )
            for index, preview in enumerate(result.previews, start=1):
                title = _safe_text(preview.citation.source_title, self.services.redactor)
                body = (
                    "Preview unavailable. / 预览不可用。"
                    if preview.preview is None
                    else _safe_text(preview.preview, self.services.redactor)
                )
                previews.append(f"<details><summary>[{index}] {title}</summary>{body}</details>")
            markers = (
                ""
                if not citations
                else " " + " ".join(f"[{i}]" for i in range(1, len(citations) + 1))
            )
            rendered_content = safe_content + markers
            updated_history = (
                *prior,
                {"role": "user", "content": safe_question},
                {"role": "assistant", "content": rendered_content},
            )
            updated_state = current.with_session(event.session_id).with_active_request(None)
            if event.kind is StreamEventKind.ANSWER:
                status = f"Answer validated. / 回答已验证。 Request: `{event.request_id}`"
            elif event.kind is StreamEventKind.REFUSAL:
                reason = "unknown" if event.reason is None else event.reason.value
                status = f"Refusal / 拒绝: `{reason}`. Request: `{event.request_id}`"
            else:
                code = "internal" if event.error_code is None else event.error_code.value
                status = f"Safe error / 安全错误: `{code}`. Request: `{event.request_id}`"
            return ChatRender(
                updated_history,
                updated_state,
                "\n\n".join(citations),
                "\n\n".join(previews),
                status,
            )
        except Exception:
            return ChatRender(
                prior,
                current.with_active_request(None),
                "",
                "",
                f"{SAFE_UI_ERROR} Correlation: `{correlation}`",
            )

    def reset_chat(self, state: BrowserSessionState | None) -> ChatRender:
        current = _state(state)
        if self.services.chat is None:
            return ChatRender((), current, "", "", SAFE_UNAVAILABLE)
        try:
            session_id = self.services.chat.reset(
                owner_id=current.owner_id,
                session_id=current.session_id,
            )
            return ChatRender(
                (),
                current.with_session(session_id),
                "",
                "",
                "Session reset. / 会话已重置。",
            )
        except Exception:
            return ChatRender((), current.with_session(None), "", "", SAFE_UI_ERROR)

    def cancel_chat(
        self,
        history: Sequence[Mapping[str, object]] | None,
        state: BrowserSessionState | None,
    ) -> ChatRender:
        current = _state(state).with_active_request(None)
        return ChatRender(_history(history), current, "", "", SAFE_CANCELLED)

    def refresh_documents(self) -> DocumentsRender:
        service = self.services.documents
        if service is None:
            return DocumentsRender((), (), SAFE_UNAVAILABLE)
        try:
            revision_id, documents = service.list_active_documents()
            jobs = service.list_jobs()
            document_rows = tuple(self._document_row(document) for document in documents)
            job_rows = tuple(self._job_row(job) for job in jobs)
            revision = revision_id or "none"
            return DocumentsRender(
                document_rows,
                job_rows,
                f"Active revision / 当前索引: `{revision}`",
            )
        except Exception:
            return DocumentsRender((), (), SAFE_UI_ERROR)

    async def upload_document(
        self,
        file_path: str | None,
        source_key: str | None,
        display_title: str | None,
    ) -> DocumentsRender:
        service = self.services.documents
        if service is None:
            return DocumentsRender((), (), SAFE_UNAVAILABLE)
        if not file_path:
            return DocumentsRender((), (), "Select a document. / 请选择文档。")
        try:
            path = Path(file_path)
            payload = UploadPayload(
                filename=path.name,
                content=await anyio.Path(path).read_bytes(),
                declared_media_type=mimetypes.guess_type(path.name)[0],
                source_key=source_key or None,
                display_title=display_title or None,
            )
            submitted = service.submit_upload(payload)
            completed = await service.run_job(submitted.job_id)
            return self._documents_after_job(completed, operation="upload")
        except Exception:
            return DocumentsRender((), (), SAFE_UI_ERROR)

    async def reindex_documents(self) -> DocumentsRender:
        service = self.services.documents
        if service is None:
            return DocumentsRender((), (), SAFE_UNAVAILABLE)
        try:
            completed = await service.run_job(service.submit_reindex().job_id)
            return self._documents_after_job(completed, operation="reindex")
        except Exception:
            return DocumentsRender((), (), SAFE_UI_ERROR)

    async def delete_document(self, source_id: str, confirmed: bool) -> DocumentsRender:
        service = self.services.documents
        if service is None:
            return DocumentsRender((), (), SAFE_UNAVAILABLE)
        if not confirmed:
            return DocumentsRender((), (), "Confirm deletion first. / 请先确认删除。")
        try:
            completed = await service.run_job(service.submit_delete(source_id).job_id)
            if completed.status is not IngestionJobStatus.SUCCEEDED:
                raise ValueError("delete_not_published")
            refreshed = self.refresh_documents()
            if any(row and row[0] == source_id for row in refreshed.document_rows):
                raise ValueError("source_still_active")
            return DocumentsRender(
                refreshed.document_rows,
                refreshed.job_rows,
                f"Deletion published. / 删除已发布。 Job: `{completed.job_id}`",
            )
        except Exception:
            return DocumentsRender((), (), SAFE_UI_ERROR)

    async def start_evaluation(
        self,
        dataset_id: str,
        dataset_version: str | None,
        state: BrowserSessionState | None,
    ) -> EvaluationRender:
        current = _state(state)
        service = self.services.evaluations
        if service is None:
            return EvaluationRender((), (), "", SAFE_UNAVAILABLE, current)
        try:
            run = await service.start(dataset_id, dataset_version or None)
            return self.refresh_evaluations(current.with_evaluation(run.run_id))
        except Exception:
            return EvaluationRender((), (), "", SAFE_UI_ERROR, current)

    def refresh_evaluations(self, state: BrowserSessionState | None) -> EvaluationRender:
        current = _state(state)
        service = self.services.evaluations
        if service is None:
            return EvaluationRender((), (), "", SAFE_UNAVAILABLE, current)
        try:
            runs = tuple(service.list_runs())
            selected = current.evaluation_run_id
            failures: Sequence[Mapping[str, object]] = ()
            if selected:
                failures = service.failed_cases(selected)
            rows = tuple(self._run_row(run) for run in runs)
            failure_rows = tuple(
                tuple(_safe_text(value, self.services.redactor) for value in row.values())
                for row in failures
            )
            status = (
                "No runs. / 暂无运行。"
                if not rows
                else f"{len(rows)} run(s). / {len(rows)} 个运行。"
            )
            return EvaluationRender(rows, failure_rows, "", status, current)
        except Exception:
            return EvaluationRender((), (), "", SAFE_UI_ERROR, current)

    def compare_evaluations(
        self,
        baseline_run_id: str,
        candidate_run_id: str,
        state: BrowserSessionState | None,
    ) -> EvaluationRender:
        current = _state(state)
        service = self.services.evaluations
        if service is None:
            return EvaluationRender((), (), "", SAFE_UNAVAILABLE, current)
        try:
            comparison = service.compare_runs(baseline_run_id, candidate_run_id)
            safe_comparison = _safe_value(dict(comparison), self.services.redactor)
            metrics = (
                "```json\n" + json.dumps(safe_comparison, ensure_ascii=False, indent=2) + "\n```"
            )
            refreshed = self.refresh_evaluations(current)
            return EvaluationRender(
                refreshed.run_rows,
                refreshed.failure_rows,
                metrics,
                "Compatible runs compared. / 已比较兼容运行。",
                current,
            )
        except EvaluationCompatibilityError:
            return EvaluationRender((), (), "", "Runs are incompatible. / 运行不兼容。", current)
        except Exception:
            return EvaluationRender((), (), "", SAFE_UI_ERROR, current)

    def report_path(self, run_id: str, report_format: Literal["json", "html"]) -> str | None:
        service = self.services.evaluations
        if service is None:
            return None
        try:
            report = service.get_report(run_id, report_format)
            if report is None or not report.path.is_file():
                return None
            return str(report.path)
        except Exception:
            return None

    def refresh_health(self) -> DiagnosticsRender:
        service = self.services.diagnostics
        if service is None:
            return DiagnosticsRender((), (), SAFE_UNAVAILABLE)
        try:
            rows = tuple(
                (
                    _safe_text(component.name, self.services.redactor),
                    component.ready,
                    _safe_text(component.reason or "", self.services.redactor),
                )
                for component in service.health()
            )
            return DiagnosticsRender(rows, (), "Health refreshed. / 健康状态已刷新。")
        except Exception:
            return DiagnosticsRender((), (), SAFE_UI_ERROR)

    def inspect_request(self, request_id: str) -> DiagnosticsRender:
        service = self.services.diagnostics
        if service is None:
            return DiagnosticsRender((), (), SAFE_UNAVAILABLE)
        try:
            diagnostic = service.get_request(request_id)
            if diagnostic is None:
                return DiagnosticsRender((), (), "Request not found. / 未找到请求。")
            safe = self._safe_diagnostic(diagnostic)
            rows = tuple((key, value) for key, value in safe.items())
            return DiagnosticsRender((), rows, "Request loaded. / 请求已加载。")
        except Exception:
            return DiagnosticsRender((), (), SAFE_UI_ERROR)

    def _documents_after_job(self, completed: IngestionJob, *, operation: str) -> DocumentsRender:
        refreshed = self.refresh_documents()
        if completed.status is not IngestionJobStatus.SUCCEEDED:
            code = completed.safe_error_code or "job_failed"
            return DocumentsRender(
                refreshed.document_rows,
                refreshed.job_rows,
                f"{operation} failed safely / {operation} 安全失败: `{code}`",
            )
        return DocumentsRender(
            refreshed.document_rows,
            refreshed.job_rows,
            f"{operation} complete / {operation} 完成. Job: `{completed.job_id}`",
        )

    def _document_row(self, document: Document) -> tuple[object, ...]:
        return (
            document.source_id,
            _safe_text(document.display_title, self.services.redactor),
            document.active_version or "",
            document.kind.value,
            document.media_type,
        )

    @staticmethod
    def _job_row(job: IngestionJob) -> tuple[object, ...]:
        return (
            job.job_id,
            job.operation.value,
            job.status.value,
            job.stage.value,
            job.source_id or "",
            job.document_version or "",
            job.ocr_page_count,
            job.chunk_count,
            job.active_index_revision or "",
        )

    @staticmethod
    def _run_row(run: EvaluationRun) -> tuple[object, ...]:
        return (
            run.run_id,
            run.status.value,
            run.dataset_id,
            run.dataset_version,
            run.completed_cases,
            run.failed_cases,
            run.total_cases,
        )

    def _safe_diagnostic(self, diagnostic: RequestDiagnostic) -> dict[str, object]:
        payload: dict[str, object] = {
            "request_id": diagnostic.request_id,
            "session_id": diagnostic.session_id,
            "trace_id": diagnostic.trace_id,
            "outcome": diagnostic.outcome,
            "safe_error_category": diagnostic.safe_error_category,
            "stage_timings_ms": dict(diagnostic.stage_timings_ms),
            "cache_status": dict(diagnostic.cache_status),
            "model_identities": dict(diagnostic.model_identities),
            "token_counts": dict(diagnostic.token_counts),
            "metadata": {
                key: value
                for key, value in diagnostic.metadata.items()
                if key in _SAFE_DIAGNOSTIC_METADATA
            },
        }
        safe = _safe_value(payload, self.services.redactor)
        if not isinstance(safe, dict):
            raise ValueError("diagnostic_output_invalid")
        return cast(dict[str, object], safe)
