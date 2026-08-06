"""Shared, allowlisted HTTP response contracts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from rag_mvp.domain.ingestion import (
    Document,
    DocumentKind,
    IngestionJob,
    IngestionJobStatus,
    IngestionOperation,
    IngestionStage,
)
from rag_mvp.domain.retrieval import RetrievalMode
from rag_mvp.retrieval.request import DEFAULT_MAXIMUM_QUERY_CHARACTERS

SafeApiCode = Annotated[
    str,
    Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_.-]*$"),
]
OpaqueApiId = Annotated[
    str,
    Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$"),
]
SESSION_ID_PATTERN = r"^session_[0-9a-f]{32}$"
SessionApiId = Annotated[
    str,
    Field(min_length=40, max_length=40, pattern=SESSION_ID_PATTERN),
]


class ApiSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ApiErrorDetail(ApiSchema):
    code: SafeApiCode


class ApiErrorResponse(ApiSchema):
    error: ApiErrorDetail


class QARequestBody(ApiSchema):
    owner_id: OpaqueApiId
    session_id: SessionApiId | None = None
    question: Annotated[str, Field(min_length=1, max_length=DEFAULT_MAXIMUM_QUERY_CHARACTERS)]
    mode: RetrievalMode | None = None
    requested_language: Literal["en", "zh-CN"] | None = None

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class IngestionJobResponse(ApiSchema):
    job_id: str
    operation: IngestionOperation
    status: IngestionJobStatus
    stage: IngestionStage
    source_id: str | None = None
    document_version: int | None = None
    ocr_page_count: int
    chunk_count: int
    active_index_revision: str | None = None
    stage_timings_ms: dict[str, float]
    warnings: tuple[SafeApiCode, ...]
    safe_error_code: SafeApiCode | None = None
    failed_stage: IngestionStage | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @classmethod
    def from_domain(cls, job: IngestionJob) -> Self:
        return cls(
            job_id=job.job_id,
            operation=job.operation,
            status=job.status,
            stage=job.stage,
            source_id=job.source_id,
            document_version=job.document_version,
            ocr_page_count=job.ocr_page_count,
            chunk_count=job.chunk_count,
            active_index_revision=job.active_index_revision,
            stage_timings_ms=dict(job.stage_timings_ms),
            warnings=job.warnings,
            safe_error_code=job.safe_error_code,
            failed_stage=job.failed_stage,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )


class ActiveDocumentResponse(ApiSchema):
    source_id: str
    display_title: str
    media_type: str
    kind: DocumentKind
    active_version: Annotated[int, Field(gt=0)]
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @classmethod
    def from_domain(cls, document: Document, *, display_title: str) -> Self:
        if document.active_version is None or document.deleted_at is not None:
            raise ValueError("document is not active")
        return cls(
            source_id=document.source_id,
            display_title=display_title,
            media_type=document.media_type,
            kind=document.kind,
            active_version=document.active_version,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )


class DocumentListResponse(ApiSchema):
    active_index_revision: str | None
    documents: tuple[ActiveDocumentResponse, ...]
