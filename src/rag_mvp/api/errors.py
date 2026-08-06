"""Content-free API exception normalization."""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rag_mvp.api.schemas import ApiErrorDetail, ApiErrorResponse
from rag_mvp.ingestion.service import IngestionSubmissionError
from rag_mvp.ingestion.validation import UploadValidationError
from rag_mvp.storage.repositories import RepositoryConflict, RepositoryError

_UPLOAD_STATUS = {
    "document_too_large": status.HTTP_413_CONTENT_TOO_LARGE,
    "media_type_mismatch": status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    "unsupported_extension": status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
}
_KNOWN_UPLOAD_CODES = {
    "document_too_large",
    "empty_document",
    "encrypted_pdf",
    "invalid_filename",
    "invalid_utf8",
    "malformed_pdf",
    "media_type_mismatch",
    "unsupported_extension",
}
_KNOWN_SUBMISSION_CODES = {
    "display_title_invalid",
    "source_id_invalid",
    "source_key_invalid",
    "source_not_active",
}


class ApiError(Exception):
    def __init__(self, status_code: int, code: str) -> None:
        self.status_code = status_code
        self.code = code
        super().__init__(code)


def _response(status_code: int, code: str) -> JSONResponse:
    body = ApiErrorResponse(error=ApiErrorDetail(code=code))
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, error: ApiError) -> JSONResponse:
        del request
        return _response(error.status_code, error.code)

    @app.exception_handler(UploadValidationError)
    async def upload_error_handler(
        request: Request,
        error: UploadValidationError,
    ) -> JSONResponse:
        del request
        code = error.code if error.code in _KNOWN_UPLOAD_CODES else "request_invalid"
        return _response(_UPLOAD_STATUS.get(code, status.HTTP_422_UNPROCESSABLE_CONTENT), code)

    @app.exception_handler(IngestionSubmissionError)
    async def submission_error_handler(
        request: Request,
        error: IngestionSubmissionError,
    ) -> JSONResponse:
        del request
        if error.code in {"source_id_invalid", "source_not_active"}:
            return _response(status.HTTP_404_NOT_FOUND, "source_not_active")
        code = error.code if error.code in _KNOWN_SUBMISSION_CODES else "request_invalid"
        return _response(status.HTTP_422_UNPROCESSABLE_CONTENT, code)

    @app.exception_handler(RepositoryConflict)
    async def repository_conflict_handler(
        request: Request,
        error: RepositoryConflict,
    ) -> JSONResponse:
        del request, error
        return _response(status.HTTP_409_CONFLICT, "submission_conflict")

    @app.exception_handler(RepositoryError)
    async def repository_error_handler(
        request: Request,
        error: RepositoryError,
    ) -> JSONResponse:
        del request, error
        return _response(status.HTTP_503_SERVICE_UNAVAILABLE, "ingestion_unavailable")

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        del request, error
        return _response(status.HTTP_422_UNPROCESSABLE_CONTENT, "request_invalid")

    @app.exception_handler(Exception)
    async def internal_error_handler(request: Request, error: Exception) -> JSONResponse:
        del request, error
        return _response(status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error")
