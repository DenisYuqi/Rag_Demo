"""Document ingestion and active-corpus HTTP routes."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Annotated, Any, Protocol, cast

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Request,
    Response,
    UploadFile,
    status,
)

from rag_mvp.api.errors import ApiError
from rag_mvp.api.readiness import ReadinessRegistry
from rag_mvp.api.schemas import (
    ActiveDocumentResponse,
    ApiErrorResponse,
    DocumentListResponse,
    IngestionJobResponse,
)
from rag_mvp.config.settings import Settings
from rag_mvp.ingestion.service import IngestionService
from rag_mvp.ingestion.validation import UploadValidationError
from rag_mvp.safety.redactor import Redactor
from rag_mvp.safety.streaming import SafeStream


class DocumentApiRuntime(Protocol):
    settings: Settings
    readiness: ReadinessRegistry
    accepting_traffic: bool
    ingestion_service: IngestionService | None
    redactor: Redactor | None

    def schedule_ingestion(self, job_id: str) -> Awaitable[None]: ...


router = APIRouter(prefix="/api/v1", tags=["documents"])

_READ_ERRORS: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"model": ApiErrorResponse},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ApiErrorResponse},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ApiErrorResponse},
}
_MUTATION_ERRORS: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"model": ApiErrorResponse},
    status.HTTP_409_CONFLICT: {"model": ApiErrorResponse},
    status.HTTP_413_CONTENT_TOO_LARGE: {"model": ApiErrorResponse},
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"model": ApiErrorResponse},
    status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ApiErrorResponse},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ApiErrorResponse},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ApiErrorResponse},
}


def _runtime(request: Request) -> DocumentApiRuntime:
    return cast(DocumentApiRuntime, request.app.state.runtime)


def _require_ingestion(request: Request) -> IngestionService:
    runtime = _runtime(request)
    if not runtime.accepting_traffic or runtime.ingestion_service is None:
        raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "ingestion_unavailable")
    try:
        ready = runtime.readiness.get("ingestion").check().ready
    except Exception:
        ready = False
    if not ready:
        raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "ingestion_unavailable")
    return runtime.ingestion_service


def _job_location(job_id: str) -> str:
    return f"/api/v1/ingestion-jobs/{job_id}"


def _safe_title(value: str, redactor: Redactor | None) -> str:
    if redactor is None or not redactor.fully_configured:
        raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "output_redaction_unavailable")
    stream = SafeStream(redactor=redactor, max_buffer_chars=1024)
    try:
        pieces = (*stream.push(value), *stream.finish())
    except Exception:
        raise ApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "output_redaction_unavailable",
        ) from None
    if stream.failed:
        raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "output_redaction_unavailable")
    return "".join(pieces)


@router.post(
    "/documents",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=_MUTATION_ERRORS,
    summary="Upload a document for ingestion",
)
async def upload_document(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File()],
    service: Annotated[IngestionService, Depends(_require_ingestion)],
    source_key: Annotated[str | None, Form()] = None,
    display_title: Annotated[str | None, Form()] = None,
) -> IngestionJobResponse:
    try:
        content = await file.read(service.upload_max_bytes + 1)
    finally:
        await file.close()
    if len(content) > service.upload_max_bytes:
        raise UploadValidationError("document_too_large")
    job = service.submit_upload(
        file.filename or "",
        content,
        source_key=source_key,
        declared_media_type=file.content_type,
        display_title=display_title,
    )
    response.headers["Location"] = _job_location(job.job_id)
    background_tasks.add_task(_runtime(request).schedule_ingestion, job.job_id)
    return IngestionJobResponse.from_domain(job)


@router.get(
    "/ingestion-jobs/{job_id}",
    response_model=IngestionJobResponse,
    responses=_READ_ERRORS,
    summary="Get ingestion job status",
)
async def get_ingestion_job(
    job_id: str,
    service: Annotated[IngestionService, Depends(_require_ingestion)],
) -> IngestionJobResponse:
    job = service.get_job(job_id)
    if job is None:
        raise ApiError(status.HTTP_404_NOT_FOUND, "ingestion_job_not_found")
    return IngestionJobResponse.from_domain(job)


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    responses=_READ_ERRORS,
    summary="List active documents",
)
async def list_documents(
    request: Request,
    service: Annotated[IngestionService, Depends(_require_ingestion)],
) -> DocumentListResponse:
    revision_id, documents = service.list_active_documents()
    redactor = _runtime(request).redactor
    return DocumentListResponse(
        active_index_revision=revision_id,
        documents=tuple(
            ActiveDocumentResponse.from_domain(
                document,
                display_title=_safe_title(document.display_title, redactor),
            )
            for document in documents
        ),
    )


@router.delete(
    "/documents/{source_id}",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=_MUTATION_ERRORS,
    summary="Remove a document from the active corpus",
)
async def delete_document(
    source_id: str,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    service: Annotated[IngestionService, Depends(_require_ingestion)],
) -> IngestionJobResponse:
    job = service.submit_delete(source_id)
    response.headers["Location"] = _job_location(job.job_id)
    background_tasks.add_task(_runtime(request).schedule_ingestion, job.job_id)
    return IngestionJobResponse.from_domain(job)


@router.post(
    "/index/rebuild",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=_MUTATION_ERRORS,
    summary="Rebuild the active index",
)
async def rebuild_index(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    service: Annotated[IngestionService, Depends(_require_ingestion)],
) -> IngestionJobResponse:
    job = service.submit_reindex()
    response.headers["Location"] = _job_location(job.job_id)
    background_tasks.add_task(_runtime(request).schedule_ingestion, job.job_id)
    return IngestionJobResponse.from_domain(job)
