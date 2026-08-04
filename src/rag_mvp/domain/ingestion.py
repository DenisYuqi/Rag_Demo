"""Knowledge-ingestion domain contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import AwareDatetime, Field, model_validator

from rag_mvp.domain._base import (
    Digest,
    DomainModel,
    Identifier,
    NonEmptyText,
    NonNegativeFiniteFloat,
    utc_now,
)


class DocumentKind(StrEnum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"


class ExtractionMethod(StrEnum):
    NATIVE = "native"
    OCR = "ocr"
    TEXT = "text"
    MIXED = "mixed"


class IngestionJobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class IngestionStage(StrEnum):
    QUEUED = "queued"
    VALIDATING = "validating"
    EXTRACTING = "extracting"
    NORMALIZING = "normalizing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    PUBLISHING = "publishing"
    COMPLETE = "complete"
    FAILED = "failed"


class IndexRevisionStatus(StrEnum):
    STAGED = "staged"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    FAILED = "failed"


class ChunkLocator(DomainModel):
    """A stable source locator; at least one locator form is required."""

    pages: tuple[int, ...] = ()
    section_path: tuple[str, ...] = ()
    char_start: Annotated[int, Field(ge=0)] | None = None
    char_end: Annotated[int, Field(gt=0)] | None = None

    @model_validator(mode="after")
    def validate_locator(self) -> ChunkLocator:
        if any(page < 1 for page in self.pages):
            raise ValueError("page numbers must be positive")
        if len(set(self.pages)) != len(self.pages):
            raise ValueError("page numbers must be unique")
        has_range = self.char_start is not None or self.char_end is not None
        if has_range:
            if self.char_start is None or self.char_end is None:
                raise ValueError("char_start and char_end must be provided together")
            if self.char_end <= self.char_start:
                raise ValueError("char_end must be greater than char_start")
        if not self.pages and not self.section_path and not has_range:
            raise ValueError("at least one source locator is required")
        return self


class EmbeddingSpaceIdentity(DomainModel):
    provider_alias: Identifier
    model: Identifier
    dimension: Annotated[int, Field(gt=0)]
    normalization: Identifier
    adapter_version: Identifier


class Document(DomainModel):
    source_id: Identifier
    source_key: Identifier
    display_title: Identifier
    media_type: Identifier
    kind: DocumentKind
    active_version: Annotated[int, Field(gt=0)] | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)
    deleted_at: AwareDatetime | None = None


class DocumentVersion(DomainModel):
    source_id: Identifier
    version: Annotated[int, Field(gt=0)]
    content_digest: Digest
    derivation_config_digest: Digest
    original_filename: Identifier
    media_type: Identifier
    size_bytes: Annotated[int, Field(ge=0)]
    source_artifact_path: NonEmptyText
    canonical_artifact_path: NonEmptyText
    extraction_method: ExtractionMethod
    created_at: AwareDatetime = Field(default_factory=utc_now)


class Chunk(DomainModel):
    chunk_id: Identifier
    source_id: Identifier
    document_version: Annotated[int, Field(gt=0)]
    ordinal: Annotated[int, Field(ge=0)]
    text: NonEmptyText
    content_digest: Digest
    locator: ChunkLocator
    language_hint: str | None = None
    token_count: Annotated[int, Field(gt=0)] | None = None


class IndexRevision(DomainModel):
    revision_id: Identifier
    status: IndexRevisionStatus = IndexRevisionStatus.STAGED
    active_sources: dict[str, int]
    chunk_set_digest: Digest
    embedding_space: EmbeddingSpaceIdentity
    extraction_version: Identifier
    chunking_version: Identifier
    tokenizer_version: Identifier
    dense_index_path: NonEmptyText
    lexical_index_path: NonEmptyText
    created_at: AwareDatetime = Field(default_factory=utc_now)
    published_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_active_sources(self) -> IndexRevision:
        if any(not source_id or version < 1 for source_id, version in self.active_sources.items()):
            raise ValueError("active source IDs must be non-empty and versions must be positive")
        if self.status is IndexRevisionStatus.ACTIVE and self.published_at is None:
            raise ValueError("an active revision requires published_at")
        return self


class IngestionJob(DomainModel):
    job_id: Identifier
    source_key: Identifier
    status: IngestionJobStatus = IngestionJobStatus.QUEUED
    stage: IngestionStage = IngestionStage.QUEUED
    source_id: str | None = None
    document_version: Annotated[int, Field(gt=0)] | None = None
    ocr_page_count: Annotated[int, Field(ge=0)] = 0
    chunk_count: Annotated[int, Field(ge=0)] = 0
    active_index_revision: str | None = None
    stage_timings_ms: dict[str, NonNegativeFiniteFloat] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    safe_error_code: str | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_terminal_state(self) -> IngestionJob:
        if self.status is IngestionJobStatus.FAILED and not self.safe_error_code:
            raise ValueError("a failed ingestion job requires safe_error_code")
        if (
            self.status is IngestionJobStatus.SUCCEEDED
            and self.stage is not IngestionStage.COMPLETE
        ):
            raise ValueError("a succeeded ingestion job must be complete")
        return self


def replace_timestamp(value: datetime | None = None) -> datetime:
    """Compatibility helper for services that need a single update timestamp."""

    return value or utc_now()
