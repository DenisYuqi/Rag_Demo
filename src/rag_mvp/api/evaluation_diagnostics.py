"""Evaluation, report, and privacy-safe request-diagnostics HTTP routes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, cast

from fastapi import APIRouter, Depends, Request, Response, status

from rag_mvp.api.errors import ApiError
from rag_mvp.api.schemas import (
    ApiErrorResponse,
    EvaluationArtifactManifestResponse,
    EvaluationDatasetCatalogEntryResponse,
    EvaluationDatasetCatalogResponse,
    EvaluationPlanCatalogEntryResponse,
    EvaluationPlanCatalogResponse,
    EvaluationRunListResponse,
    EvaluationRunResponse,
    EvaluationRunSummaryResponse,
    EvaluationStartRequest,
    FailedCaseDiagnosticResponse,
    FailedCaseListResponse,
    RequestDiagnosticResponse,
)
from rag_mvp.domain.evaluation import EvaluationRun
from rag_mvp.domain.qa import RequestDiagnostic
from rag_mvp.evaluation.artifacts_v2 import (
    ArtifactNotFoundError,
    artifact_download_filename_v2,
    artifact_media_type_v2,
)
from rag_mvp.safety.output import redact_output
from rag_mvp.safety.redactor import Redactor

if TYPE_CHECKING:
    from rag_mvp.evaluation.application import (
        EvaluationArtifactManifest,
        EvaluationDatasetCatalogEntry,
        EvaluationPlanCatalogEntry,
        EvaluationRunSummary,
        FailedCaseDiagnostic,
        ResolvedEvaluationArtifact,
    )

ReportFormat = Literal["json", "html"]
STANDARD_EVALUATION_PLAN_ID = "standard-evaluation-v1"


@dataclass(frozen=True, slots=True)
class ReportArtifact:
    """A validated report body returned without exposing a filesystem path."""

    content: bytes
    media_type: str
    filename: str

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("report content must not be empty")
        if self.media_type not in {"application/json", "text/html"}:
            raise ValueError("unsupported report media type")
        if not self.filename or any(character in self.filename for character in '\\/:\r\n"'):
            raise ValueError("report filename is unsafe")


class EvaluationOperations(Protocol):
    async def start(self, *, dataset_id: str, dataset_version: str) -> EvaluationRun: ...

    def get(self, run_id: str) -> EvaluationRun | None: ...

    def list(self) -> Sequence[EvaluationRun]: ...

    def datasets(self) -> Sequence[EvaluationDatasetCatalogEntry]: ...

    def plans(self) -> Sequence[EvaluationPlanCatalogEntry]: ...

    def summary(self, run_id: str) -> EvaluationRunSummary | None: ...

    def failed_cases(self, run_id: str) -> Sequence[FailedCaseDiagnostic]: ...

    def artifact_manifest(self, run_id: str) -> EvaluationArtifactManifest | None: ...

    def artifact(self, run_id: str, artifact_id: str) -> ResolvedEvaluationArtifact | None: ...

    def report(self, run_id: str, report_format: ReportFormat) -> DownloadArtifact | None: ...


class DownloadArtifact(Protocol):
    @property
    def content(self) -> bytes: ...

    @property
    def media_type(self) -> str: ...

    @property
    def filename(self) -> str: ...


class DiagnosticOperations(Protocol):
    def get(self, request_id: str) -> RequestDiagnostic | None: ...


class EvaluationApiRuntime(Protocol):
    accepting_traffic: bool
    evaluation_service: EvaluationOperations | None
    diagnostics_service: DiagnosticOperations | None
    redactor: Redactor | None


router = APIRouter(prefix="/api/v1", tags=["evaluation", "diagnostics"])

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"model": ApiErrorResponse},
    status.HTTP_409_CONFLICT: {"model": ApiErrorResponse},
    status.HTTP_429_TOO_MANY_REQUESTS: {"model": ApiErrorResponse},
    status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ApiErrorResponse},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ApiErrorResponse},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ApiErrorResponse},
}
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


def _runtime(request: Request) -> EvaluationApiRuntime:
    return cast(EvaluationApiRuntime, request.app.state.runtime)


def _require_evaluation(request: Request) -> EvaluationOperations:
    runtime = _runtime(request)
    if not runtime.accepting_traffic or runtime.evaluation_service is None:
        raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "evaluation_unavailable")
    return runtime.evaluation_service


def _require_diagnostics(request: Request) -> DiagnosticOperations:
    runtime = _runtime(request)
    redactor = runtime.redactor
    if (
        not runtime.accepting_traffic
        or runtime.diagnostics_service is None
        or redactor is None
        or not redactor.fully_configured
    ):
        raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "diagnostics_unavailable")
    return runtime.diagnostics_service


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"


@router.get(
    "/evaluation-datasets",
    response_model=EvaluationDatasetCatalogResponse,
    responses=_ERROR_RESPONSES,
    summary="List validated evaluation datasets",
)
async def list_evaluation_datasets(
    response: Response,
    service: Annotated[EvaluationOperations, Depends(_require_evaluation)],
) -> EvaluationDatasetCatalogResponse:
    try:
        items = tuple(
            EvaluationDatasetCatalogEntryResponse.from_domain(item) for item in service.datasets()
        )
    except RuntimeError:
        raise ApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "evaluation_catalog_unavailable",
        ) from None
    _no_store(response)
    return EvaluationDatasetCatalogResponse(datasets=items)


@router.get(
    "/evaluation-plans",
    response_model=EvaluationPlanCatalogResponse,
    responses=_ERROR_RESPONSES,
    summary="List registered evaluation plans",
)
async def list_evaluation_plans(
    response: Response,
    service: Annotated[EvaluationOperations, Depends(_require_evaluation)],
) -> EvaluationPlanCatalogResponse:
    items = tuple(EvaluationPlanCatalogEntryResponse.from_domain(item) for item in service.plans())
    _no_store(response)
    return EvaluationPlanCatalogResponse(plans=items)


@router.get(
    "/evaluations",
    response_model=EvaluationRunListResponse,
    responses=_ERROR_RESPONSES,
    summary="List persisted evaluations",
)
async def list_evaluations(
    response: Response,
    service: Annotated[EvaluationOperations, Depends(_require_evaluation)],
) -> EvaluationRunListResponse:
    runs = tuple(EvaluationRunResponse.from_domain(run) for run in service.list())
    _no_store(response)
    return EvaluationRunListResponse(runs=runs)


@router.post(
    "/evaluations",
    response_model=EvaluationRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=_ERROR_RESPONSES,
    summary="Start a versioned RAG evaluation",
)
async def start_evaluation(
    payload: EvaluationStartRequest,
    response: Response,
    service: Annotated[EvaluationOperations, Depends(_require_evaluation)],
) -> EvaluationRunResponse:
    if payload.plan_id not in {None, STANDARD_EVALUATION_PLAN_ID}:
        raise ApiError(status.HTTP_422_UNPROCESSABLE_CONTENT, "evaluation_plan_invalid")
    try:
        run = await service.start(
            dataset_id=payload.dataset_id,
            dataset_version=payload.dataset_version,
        )
    except RuntimeError as error:
        raw_code = getattr(error, "code", None)
        if raw_code == "evaluation_duplicate":
            raise ApiError(status.HTTP_409_CONFLICT, "evaluation_duplicate") from None
        if raw_code == "evaluation_capacity":
            raise ApiError(status.HTTP_429_TOO_MANY_REQUESTS, "evaluation_capacity") from None
        code = (
            raw_code
            if raw_code
            in {
                "evaluation_dataset_not_found",
                "evaluation_dataset_ambiguous",
                "evaluation_plan_not_found",
                "dataset_settings_derivation_mismatch",
            }
            else "evaluation_start_invalid"
        )
        raise ApiError(status.HTTP_422_UNPROCESSABLE_CONTENT, code) from None
    except ValueError:
        raise ApiError(status.HTTP_422_UNPROCESSABLE_CONTENT, "dataset_invalid") from None
    response.headers["Location"] = f"/api/v1/evaluations/{run.run_id}"
    _no_store(response)
    return EvaluationRunResponse.from_domain(run)


@router.get(
    "/evaluations/{run_id}",
    response_model=EvaluationRunResponse,
    responses=_ERROR_RESPONSES,
    summary="Get evaluation progress and status",
)
async def get_evaluation(
    run_id: str,
    response: Response,
    service: Annotated[EvaluationOperations, Depends(_require_evaluation)],
) -> EvaluationRunResponse:
    run = service.get(run_id)
    if run is None:
        raise ApiError(status.HTTP_404_NOT_FOUND, "evaluation_not_found")
    _no_store(response)
    return EvaluationRunResponse.from_domain(run)


@router.get(
    "/evaluations/{run_id}/summary",
    response_model=EvaluationRunSummaryResponse,
    responses=_ERROR_RESPONSES,
    summary="Get a privacy-safe evaluation summary",
)
async def get_evaluation_summary(
    run_id: str,
    response: Response,
    service: Annotated[EvaluationOperations, Depends(_require_evaluation)],
) -> EvaluationRunSummaryResponse:
    item = service.summary(run_id)
    if item is None:
        raise ApiError(status.HTTP_404_NOT_FOUND, "evaluation_not_found")
    _no_store(response)
    return EvaluationRunSummaryResponse.from_domain(item)


@router.get(
    "/evaluations/{run_id}/failed-cases",
    response_model=FailedCaseListResponse,
    responses=_ERROR_RESPONSES,
    summary="List allowlisted failed-case diagnostics",
)
async def get_evaluation_failed_cases(
    run_id: str,
    response: Response,
    service: Annotated[EvaluationOperations, Depends(_require_evaluation)],
) -> FailedCaseListResponse:
    if service.get(run_id) is None:
        raise ApiError(status.HTTP_404_NOT_FOUND, "evaluation_not_found")
    cases = tuple(
        FailedCaseDiagnosticResponse.from_domain(item) for item in service.failed_cases(run_id)
    )
    _no_store(response)
    return FailedCaseListResponse(run_id=run_id, cases=cases)


@router.get(
    "/evaluations/{run_id}/artifacts",
    response_model=EvaluationArtifactManifestResponse,
    responses=_ERROR_RESPONSES,
    summary="Get the verified artifact manifest",
)
async def get_evaluation_artifact_manifest(
    run_id: str,
    response: Response,
    service: Annotated[EvaluationOperations, Depends(_require_evaluation)],
) -> EvaluationArtifactManifestResponse:
    if service.get(run_id) is None:
        raise ApiError(status.HTTP_404_NOT_FOUND, "evaluation_not_found")
    try:
        manifest = service.artifact_manifest(run_id)
    except RuntimeError:
        raise ApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "evaluation_artifacts_unavailable",
        ) from None
    if manifest is None:
        raise ApiError(status.HTTP_404_NOT_FOUND, "evaluation_artifact_manifest_not_found")
    _no_store(response)
    return EvaluationArtifactManifestResponse.from_domain(manifest)


@router.get(
    "/evaluations/{run_id}/artifacts/{artifact_id}",
    responses=_ERROR_RESPONSES,
    summary="Download one verified evaluation artifact",
)
async def download_evaluation_artifact(
    run_id: str,
    artifact_id: str,
    service: Annotated[EvaluationOperations, Depends(_require_evaluation)],
) -> Response:
    if service.get(run_id) is None:
        raise ApiError(status.HTTP_404_NOT_FOUND, "evaluation_not_found")
    try:
        artifact = service.artifact(run_id, artifact_id)
    except RuntimeError:
        raise ApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "evaluation_artifact_unavailable",
        ) from None
    if artifact is None:
        raise ApiError(status.HTTP_404_NOT_FOUND, "evaluation_artifact_not_found")
    try:
        expected_filename = artifact_download_filename_v2(artifact_id)
        expected_media_type = artifact_media_type_v2(artifact_id)
    except ArtifactNotFoundError:
        raise ApiError(status.HTTP_404_NOT_FOUND, "evaluation_artifact_not_found") from None
    if (
        artifact.artifact_id != artifact_id
        or artifact.filename != expected_filename
        or artifact.media_type != expected_media_type
    ):
        raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "evaluation_artifact_unavailable")
    return Response(
        content=artifact.content,
        media_type=artifact.media_type,
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f'attachment; filename="{artifact.filename}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get(
    "/reports/{run_id}.{report_format}",
    responses=_ERROR_RESPONSES,
    summary="Download a completed evaluation report",
)
async def download_report(
    run_id: str,
    report_format: ReportFormat,
    service: Annotated[EvaluationOperations, Depends(_require_evaluation)],
) -> Response:
    try:
        artifact = service.report(run_id, report_format)
    except RuntimeError:
        raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "report_unavailable") from None
    if artifact is None:
        raise ApiError(status.HTTP_404_NOT_FOUND, "report_not_found")
    expected_type = "application/json" if report_format == "json" else "text/html"
    if artifact.media_type != expected_type:
        raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "report_unavailable")
    return Response(
        content=artifact.content,
        media_type=artifact.media_type,
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f'attachment; filename="{artifact.filename}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get(
    "/diagnostics/requests/{request_id}",
    response_model=RequestDiagnosticResponse,
    responses=_ERROR_RESPONSES,
    summary="Get privacy-safe evidence for a recent request",
)
async def get_request_diagnostic(
    request_id: str,
    request: Request,
    response: Response,
    service: Annotated[DiagnosticOperations, Depends(_require_diagnostics)],
) -> RequestDiagnosticResponse:
    diagnostic = service.get(request_id)
    if diagnostic is None:
        raise ApiError(status.HTTP_404_NOT_FOUND, "diagnostic_not_found")
    metadata = {
        key: value for key, value in diagnostic.metadata.items() if key in _SAFE_DIAGNOSTIC_METADATA
    }
    safe_response = RequestDiagnosticResponse.from_domain(diagnostic, metadata=metadata)
    runtime = _runtime(request)
    try:
        redacted = redact_output(safe_response, redactor=cast(Redactor, runtime.redactor))
        validated = RequestDiagnosticResponse.model_validate(redacted)
    except (TypeError, ValueError, RecursionError):
        raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "diagnostics_unavailable") from None
    response.headers["Cache-Control"] = "no-store"
    return validated
