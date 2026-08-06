"""Evaluation, report, and privacy-safe request-diagnostics HTTP routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, cast

from fastapi import APIRouter, Depends, Request, Response, status

from rag_mvp.api.errors import ApiError
from rag_mvp.api.schemas import (
    ApiErrorResponse,
    EvaluationRunResponse,
    EvaluationStartRequest,
    RequestDiagnosticResponse,
)
from rag_mvp.domain.evaluation import EvaluationRun
from rag_mvp.domain.qa import RequestDiagnostic
from rag_mvp.safety.output import redact_output
from rag_mvp.safety.redactor import Redactor

ReportFormat = Literal["json", "html"]


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

    def report(self, run_id: str, report_format: ReportFormat) -> ReportArtifact | None: ...


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
    try:
        run = await service.start(
            dataset_id=payload.dataset_id,
            dataset_version=payload.dataset_version,
        )
    except ValueError:
        raise ApiError(status.HTTP_422_UNPROCESSABLE_CONTENT, "dataset_invalid") from None
    response.headers["Location"] = f"/api/v1/evaluations/{run.run_id}"
    response.headers["Cache-Control"] = "no-store"
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
    response.headers["Cache-Control"] = "no-store"
    return EvaluationRunResponse.from_domain(run)


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
    artifact = service.report(run_id, report_format)
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
