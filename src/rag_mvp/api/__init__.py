"""HTTP API composition with cycle-safe lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag_mvp.api.app import create_app, create_executable_app
    from rag_mvp.api.qa import NDJSON_MEDIA_TYPE, QARuntimeServices
    from rag_mvp.api.schemas import (
        ActiveDocumentResponse,
        ApiErrorResponse,
        DocumentListResponse,
        IngestionJobResponse,
        QARequestBody,
    )

_EXPORTS = {
    "NDJSON_MEDIA_TYPE": ("rag_mvp.api.qa", "NDJSON_MEDIA_TYPE"),
    "QARuntimeServices": ("rag_mvp.api.qa", "QARuntimeServices"),
    "ActiveDocumentResponse": ("rag_mvp.api.schemas", "ActiveDocumentResponse"),
    "ApiErrorResponse": ("rag_mvp.api.schemas", "ApiErrorResponse"),
    "DocumentListResponse": ("rag_mvp.api.schemas", "DocumentListResponse"),
    "IngestionJobResponse": ("rag_mvp.api.schemas", "IngestionJobResponse"),
    "QARequestBody": ("rag_mvp.api.schemas", "QARequestBody"),
    "create_app": ("rag_mvp.api.app", "create_app"),
    "create_executable_app": ("rag_mvp.api.app", "create_executable_app"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))


__all__ = [
    "NDJSON_MEDIA_TYPE",
    "ActiveDocumentResponse",
    "ApiErrorResponse",
    "DocumentListResponse",
    "IngestionJobResponse",
    "QARequestBody",
    "QARuntimeServices",
    "create_app",
    "create_executable_app",
]
