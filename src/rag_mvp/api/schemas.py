"""Shared, allowlisted HTTP response contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from rag_mvp.domain.evaluation import EvaluationRun, EvaluationRunStatus
from rag_mvp.domain.ingestion import (
    Document,
    DocumentKind,
    IngestionJob,
    IngestionJobStatus,
    IngestionOperation,
    IngestionStage,
)
from rag_mvp.domain.qa import RequestDiagnostic
from rag_mvp.domain.retrieval import RetrievalMode
from rag_mvp.retrieval.request import DEFAULT_MAXIMUM_QUERY_CHARACTERS

if TYPE_CHECKING:
    from rag_mvp.evaluation.application import (
        EvaluationArtifactDescriptor,
        EvaluationArtifactManifest,
        EvaluationDatasetCatalogEntry,
        EvaluationPlanCatalogEntry,
        EvaluationRunSummary,
        FailedCaseDiagnostic,
        FailedMetricContribution,
    )

SafeApiCode = Annotated[
    str,
    Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_.-]*$"),
]
OpaqueApiId = Annotated[
    str,
    Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$"),
]
VersionApiId = Annotated[
    str,
    Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$"),
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


class EvaluationStartRequest(ApiSchema):
    dataset_id: OpaqueApiId
    dataset_version: VersionApiId
    plan_id: OpaqueApiId | None = None


class EvaluationRunResponse(ApiSchema):
    run_id: OpaqueApiId
    status: EvaluationRunStatus
    dataset_id: OpaqueApiId
    dataset_version: VersionApiId
    configuration_id: OpaqueApiId
    total_cases: Annotated[int, Field(ge=0)]
    completed_cases: Annotated[int, Field(ge=0)]
    failed_cases: Annotated[int, Field(ge=0)]
    safe_error_code: SafeApiCode | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @classmethod
    def from_domain(cls, run: EvaluationRun) -> Self:
        return cls(
            run_id=run.run_id,
            status=run.status,
            dataset_id=run.dataset_id,
            dataset_version=run.dataset_version,
            configuration_id=run.configuration_id,
            total_cases=run.total_cases,
            completed_cases=run.completed_cases,
            failed_cases=run.failed_cases,
            safe_error_code=run.safe_error_code,
            created_at=run.created_at,
            updated_at=run.updated_at,
        )


class EvaluationRunListResponse(ApiSchema):
    runs: tuple[EvaluationRunResponse, ...]


class EvaluationDatasetCatalogEntryResponse(ApiSchema):
    dataset_id: OpaqueApiId
    dataset_version: VersionApiId
    schema_version: VersionApiId
    content_hash: str
    corpus_version: VersionApiId
    corpus_hash: str
    case_count: Annotated[int, Field(gt=0)]
    languages: tuple[VersionApiId, ...]

    @classmethod
    def from_domain(cls, item: EvaluationDatasetCatalogEntry) -> Self:
        return cls(**item.model_dump())


class EvaluationDatasetCatalogResponse(ApiSchema):
    datasets: tuple[EvaluationDatasetCatalogEntryResponse, ...]


class EvaluationPlanCatalogEntryResponse(ApiSchema):
    plan_id: OpaqueApiId
    plan_version: VersionApiId
    kind: OpaqueApiId
    dataset_id: OpaqueApiId
    dataset_version: VersionApiId
    planned_case_count: Annotated[int, Field(gt=0)]
    candidate_count: Literal[1]
    maximum_logical_calls: Annotated[int, Field(gt=0)]
    maximum_provider_calls: Annotated[int, Field(gt=0)]
    cache_policy: OpaqueApiId
    cost_estimate_status: Literal["unavailable"]
    cost_estimate: None
    cost_cap: None
    maximum_active_jobs: Annotated[int, Field(gt=0)]

    @classmethod
    def from_domain(cls, item: EvaluationPlanCatalogEntry) -> Self:
        return cls(**item.model_dump())


class EvaluationPlanCatalogResponse(ApiSchema):
    plans: tuple[EvaluationPlanCatalogEntryResponse, ...]


class EvaluationRunSummaryResponse(ApiSchema):
    run_id: OpaqueApiId
    status: EvaluationRunStatus
    dataset_id: OpaqueApiId
    dataset_version: VersionApiId
    dataset_hash: str
    corpus_version: VersionApiId
    corpus_hash: str | None
    plan_id: OpaqueApiId
    plan_version: VersionApiId
    configuration_id: str
    code_revision: str
    cache_policy: OpaqueApiId
    total_cases: Annotated[int, Field(ge=0)]
    completed_cases: Annotated[int, Field(ge=0)]
    failed_cases: Annotated[int, Field(ge=0)]
    remaining_cases: Annotated[int, Field(ge=0)]
    safe_error_code: SafeApiCode | None = None
    evidence_status: Literal["available", "incomplete", "unavailable"]
    gate_status: Literal["passed", "failed", "unavailable"]
    completed_at: AwareDatetime | None
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @classmethod
    def from_domain(cls, item: EvaluationRunSummary) -> Self:
        return cls(**item.model_dump())


class FailedMetricContributionResponse(ApiSchema):
    metric_id: OpaqueApiId
    status: Literal["passed", "failed", "unavailable"]
    value: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None
    numerator: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    denominator: Annotated[int, Field(gt=0)] | None = None

    @classmethod
    def from_domain(cls, item: FailedMetricContribution) -> Self:
        return cls(**item.model_dump())


class FailedCaseDiagnosticResponse(ApiSchema):
    case_id: OpaqueApiId
    safe_error_code: SafeApiCode
    request_id: OpaqueApiId | None = None
    trace_id: OpaqueApiId | None = None
    outcome: OpaqueApiId | None = None
    refusal_reason: OpaqueApiId | None = None
    citation_chunk_ids: tuple[OpaqueApiId, ...] = ()
    tags: tuple[OpaqueApiId, ...] = ()
    metric_contributions: tuple[FailedMetricContributionResponse, ...] = ()

    @classmethod
    def from_domain(cls, item: FailedCaseDiagnostic) -> Self:
        return cls(**item.model_dump())


class FailedCaseListResponse(ApiSchema):
    run_id: OpaqueApiId
    cases: tuple[FailedCaseDiagnosticResponse, ...]


class EvaluationArtifactDescriptorResponse(ApiSchema):
    artifact_id: OpaqueApiId
    schema_version: VersionApiId
    format: OpaqueApiId
    media_type: str
    sha256_digest: str
    byte_size: Annotated[int, Field(ge=0)]
    created_at: AwareDatetime

    @classmethod
    def from_domain(cls, item: EvaluationArtifactDescriptor) -> Self:
        return cls(**item.model_dump())


class EvaluationArtifactManifestResponse(ApiSchema):
    run_id: OpaqueApiId
    configuration_id: str
    manifest_content_hash: str
    artifacts: tuple[EvaluationArtifactDescriptorResponse, ...]

    @classmethod
    def from_domain(cls, item: EvaluationArtifactManifest) -> Self:
        return cls(
            run_id=item.run_id,
            configuration_id=item.configuration_id,
            manifest_content_hash=item.manifest_content_hash,
            artifacts=tuple(
                EvaluationArtifactDescriptorResponse.from_domain(descriptor)
                for descriptor in item.artifacts
            ),
        )


class RequestDiagnosticResponse(ApiSchema):
    request_id: OpaqueApiId
    session_id: str | None = None
    trace_id: str | None = None
    outcome: OpaqueApiId
    safe_error_category: SafeApiCode | None = None
    stage_timings_ms: dict[str, float]
    cache_status: dict[str, str]
    model_identities: dict[str, str]
    token_counts: dict[str, Annotated[int, Field(ge=0)]]
    metadata: dict[str, str | int | float | bool | None]
    created_at: AwareDatetime

    @classmethod
    def from_domain(
        cls,
        diagnostic: RequestDiagnostic,
        *,
        metadata: dict[str, str | int | float | bool | None],
    ) -> Self:
        return cls(
            request_id=diagnostic.request_id,
            session_id=diagnostic.session_id,
            trace_id=diagnostic.trace_id,
            outcome=diagnostic.outcome,
            safe_error_category=diagnostic.safe_error_category,
            stage_timings_ms=dict(diagnostic.stage_timings_ms),
            cache_status=dict(diagnostic.cache_status),
            model_identities=dict(diagnostic.model_identities),
            token_counts=dict(diagnostic.token_counts),
            metadata=metadata,
            created_at=diagnostic.created_at,
        )
