"""Registered comparison catalog, execution, summary, and artifact routes."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from typing import Annotated, Any, Protocol, cast

from fastapi import APIRouter, Depends, Request, Response, status

from rag_mvp.api.errors import ApiError
from rag_mvp.api.schemas import ApiErrorResponse, ApiSchema, OpaqueApiId
from rag_mvp.evaluation.comparison_application import (
    ComparisonApplicationError,
    ComparisonArtifactManifestView,
    ComparisonCapacityError,
    ComparisonConflictError,
    ComparisonNotFoundError,
    ComparisonPlanCatalogEntry,
    ComparisonRunEntry,
    ComparisonSummary,
    ComparisonUnavailableError,
    ComparisonValidationError,
    ResolvedComparisonDownload,
)


class ComparisonStartRequest(ApiSchema):
    experiment_plan_id: OpaqueApiId


class ComparisonPlanCatalogResponse(ApiSchema):
    plans: tuple[ComparisonPlanCatalogEntry, ...]


class ComparisonRunListResponse(ApiSchema):
    comparisons: tuple[ComparisonRunEntry, ...]


class ComparisonOperations(Protocol):
    def comparison_plans(self) -> Sequence[ComparisonPlanCatalogEntry]: ...

    async def start_comparison(self, experiment_plan_id: str) -> ComparisonRunEntry: ...

    def get_comparison(self, comparison_id: str) -> ComparisonRunEntry | None: ...

    def list_comparisons(self) -> Sequence[ComparisonRunEntry]: ...

    def comparison_summary(self, comparison_id: str) -> ComparisonSummary | None: ...

    def comparison_manifest(
        self,
        comparison_id: str,
    ) -> ComparisonArtifactManifestView | None: ...

    def comparison_artifact(
        self,
        comparison_id: str,
        artifact_id: str,
    ) -> ResolvedComparisonDownload | None: ...


class ComparisonApiRuntime(Protocol):
    accepting_traffic: bool
    evaluation_service: object | None


router = APIRouter(prefix="/api/v1", tags=["comparisons"])

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"model": ApiErrorResponse},
    status.HTTP_409_CONFLICT: {"model": ApiErrorResponse},
    status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ApiErrorResponse},
    status.HTTP_429_TOO_MANY_REQUESTS: {"model": ApiErrorResponse},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ApiErrorResponse},
}
_REQUIRED_OPERATIONS = (
    "comparison_plans",
    "start_comparison",
    "get_comparison",
    "list_comparisons",
    "comparison_summary",
    "comparison_manifest",
    "comparison_artifact",
)


def _runtime(request: Request) -> ComparisonApiRuntime:
    return cast(ComparisonApiRuntime, request.app.state.runtime)


def _require_comparison(request: Request) -> ComparisonOperations:
    runtime = _runtime(request)
    service = runtime.evaluation_service
    if (
        not runtime.accepting_traffic
        or service is None
        or any(not callable(getattr(service, name, None)) for name in _REQUIRED_OPERATIONS)
    ):
        raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "comparison_unavailable")
    return cast(ComparisonOperations, service)


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"


def _start_error(error: ComparisonApplicationError) -> ApiError:
    if isinstance(error, ComparisonNotFoundError):
        return ApiError(status.HTTP_404_NOT_FOUND, "comparison_plan_not_found")
    if isinstance(error, ComparisonCapacityError):
        return ApiError(status.HTTP_429_TOO_MANY_REQUESTS, "comparison_capacity")
    if isinstance(error, ComparisonConflictError):
        code = (
            "comparison_duplicate"
            if error.code == "comparison_duplicate"
            else "comparison_prerequisite_unmet"
        )
        return ApiError(status.HTTP_409_CONFLICT, code)
    if isinstance(error, ComparisonValidationError):
        return ApiError(status.HTTP_422_UNPROCESSABLE_CONTENT, "comparison_preflight_invalid")
    if isinstance(error, ComparisonUnavailableError):
        return ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "comparison_unavailable")
    return ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "comparison_unavailable")


@router.get(
    "/comparison-plans",
    response_model=ComparisonPlanCatalogResponse,
    responses=_ERROR_RESPONSES,
    summary="List registered controlled comparison plans",
)
async def list_comparison_plans(
    response: Response,
    service: Annotated[ComparisonOperations, Depends(_require_comparison)],
) -> ComparisonPlanCatalogResponse:
    try:
        plans = tuple(service.comparison_plans())
    except ComparisonApplicationError:
        raise ApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "comparison_catalog_unavailable",
        ) from None
    except Exception:
        raise ApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "comparison_catalog_unavailable",
        ) from None
    _no_store(response)
    return ComparisonPlanCatalogResponse(plans=plans)


@router.get(
    "/comparisons",
    response_model=ComparisonRunListResponse,
    responses=_ERROR_RESPONSES,
    summary="List immutable comparison runs",
)
async def list_comparisons(
    response: Response,
    service: Annotated[ComparisonOperations, Depends(_require_comparison)],
) -> ComparisonRunListResponse:
    try:
        comparisons = tuple(service.list_comparisons())
    except Exception:
        raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "comparison_unavailable") from None
    _no_store(response)
    return ComparisonRunListResponse(comparisons=comparisons)


@router.post(
    "/comparisons",
    response_model=ComparisonRunEntry,
    status_code=status.HTTP_202_ACCEPTED,
    responses=_ERROR_RESPONSES,
    summary="Start one registered controlled comparison",
)
async def start_comparison(
    payload: ComparisonStartRequest,
    response: Response,
    service: Annotated[ComparisonOperations, Depends(_require_comparison)],
) -> ComparisonRunEntry:
    try:
        run = await service.start_comparison(payload.experiment_plan_id)
    except ComparisonApplicationError as error:
        raise _start_error(error) from None
    except Exception:
        raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "comparison_unavailable") from None
    response.headers["Location"] = f"/api/v1/comparisons/{run.comparison_id}"
    _no_store(response)
    return run


@router.get(
    "/comparisons/{comparison_id}",
    response_model=ComparisonRunEntry,
    responses=_ERROR_RESPONSES,
    summary="Get comparison progress and status",
)
async def get_comparison(
    comparison_id: OpaqueApiId,
    response: Response,
    service: Annotated[ComparisonOperations, Depends(_require_comparison)],
) -> ComparisonRunEntry:
    try:
        run = service.get_comparison(comparison_id)
    except Exception:
        raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "comparison_unavailable") from None
    if run is None:
        raise ApiError(status.HTTP_404_NOT_FOUND, "comparison_not_found")
    _no_store(response)
    return run


@router.get(
    "/comparisons/{comparison_id}/summary",
    response_model=ComparisonSummary,
    responses=_ERROR_RESPONSES,
    summary="Get denominator-bearing comparison evidence",
)
async def get_comparison_summary(
    comparison_id: OpaqueApiId,
    response: Response,
    service: Annotated[ComparisonOperations, Depends(_require_comparison)],
) -> ComparisonSummary:
    try:
        if service.get_comparison(comparison_id) is None:
            raise ApiError(status.HTTP_404_NOT_FOUND, "comparison_not_found")
        summary = service.comparison_summary(comparison_id)
    except ApiError:
        raise
    except Exception:
        raise ApiError(status.HTTP_503_SERVICE_UNAVAILABLE, "comparison_unavailable") from None
    if summary is None:
        raise ApiError(status.HTTP_404_NOT_FOUND, "comparison_summary_not_found")
    _no_store(response)
    return summary


@router.get(
    "/comparisons/{comparison_id}/artifacts",
    response_model=ComparisonArtifactManifestView,
    responses=_ERROR_RESPONSES,
    summary="Get the verified path-free comparison artifact manifest",
)
async def get_comparison_artifact_manifest(
    comparison_id: OpaqueApiId,
    response: Response,
    service: Annotated[ComparisonOperations, Depends(_require_comparison)],
) -> ComparisonArtifactManifestView:
    try:
        if service.get_comparison(comparison_id) is None:
            raise ApiError(status.HTTP_404_NOT_FOUND, "comparison_not_found")
        manifest = service.comparison_manifest(comparison_id)
    except ApiError:
        raise
    except Exception:
        raise ApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "comparison_artifact_unavailable",
        ) from None
    if manifest is None:
        raise ApiError(
            status.HTTP_404_NOT_FOUND,
            "comparison_artifact_manifest_not_found",
        )
    _no_store(response)
    return manifest


@router.get(
    "/comparisons/{comparison_id}/artifacts/{artifact_id}",
    responses=_ERROR_RESPONSES,
    summary="Download one verified immutable comparison artifact",
)
async def download_comparison_artifact(
    comparison_id: OpaqueApiId,
    artifact_id: OpaqueApiId,
    service: Annotated[ComparisonOperations, Depends(_require_comparison)],
) -> Response:
    try:
        if service.get_comparison(comparison_id) is None:
            raise ApiError(status.HTTP_404_NOT_FOUND, "comparison_not_found")
        manifest = service.comparison_manifest(comparison_id)
        if manifest is None:
            raise ApiError(
                status.HTTP_404_NOT_FOUND,
                "comparison_artifact_manifest_not_found",
            )
        descriptor = next(
            (item for item in manifest.artifacts if item.artifact_id == artifact_id),
            None,
        )
        if descriptor is None:
            raise ApiError(status.HTTP_404_NOT_FOUND, "comparison_artifact_not_found")
        artifact = service.comparison_artifact(comparison_id, artifact_id)
    except ApiError:
        raise
    except Exception:
        raise ApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "comparison_artifact_unavailable",
        ) from None
    if artifact is None:
        raise ApiError(status.HTTP_404_NOT_FOUND, "comparison_artifact_not_found")
    digest = f"sha256:{sha256(artifact.content).hexdigest()}"
    if (
        artifact.artifact_id != artifact_id
        or artifact.media_type != descriptor.media_type
        or len(artifact.content) != descriptor.byte_size
        or digest != descriptor.sha256_digest
        or not artifact.filename.endswith(f".{descriptor.format}")
    ):
        raise ApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "comparison_artifact_unavailable",
        )
    return Response(
        content=artifact.content,
        media_type=artifact.media_type,
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f'attachment; filename="{artifact.filename}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


__all__ = [
    "ComparisonOperations",
    "ComparisonPlanCatalogResponse",
    "ComparisonRunListResponse",
    "ComparisonStartRequest",
    "router",
]
