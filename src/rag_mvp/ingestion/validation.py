"""Fail-fast upload validation before any persistent artifact is created."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from rag_mvp.domain.ingestion import DocumentKind


class UploadValidationError(ValueError):
    """A safe upload rejection suitable for an API reason code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ValidatedUpload(BaseModel):
    model_config = ConfigDict(frozen=True)

    filename: str
    media_type: str
    kind: DocumentKind
    content: bytes


_EXTENSIONS: dict[str, tuple[DocumentKind, frozenset[str]]] = {
    ".pdf": (DocumentKind.PDF, frozenset({"application/pdf"})),
    ".md": (
        DocumentKind.MARKDOWN,
        frozenset({"text/markdown", "text/plain", "text/x-markdown"}),
    ),
    ".markdown": (
        DocumentKind.MARKDOWN,
        frozenset({"text/markdown", "text/plain", "text/x-markdown"}),
    ),
    ".txt": (DocumentKind.TEXT, frozenset({"text/plain"})),
}


def _validate_filename(filename: str) -> str:
    if not filename or len(filename.encode("utf-8")) > 255:
        raise UploadValidationError("invalid_filename")
    if Path(filename).name != filename or filename in {".", ".."}:
        raise UploadValidationError("invalid_filename")
    if any(ord(character) < 32 or character == "\x7f" for character in filename):
        raise UploadValidationError("invalid_filename")
    return filename


def _detect_media_type(content: bytes, kind: DocumentKind) -> str:
    if content.startswith(b"%PDF-"):
        return "application/pdf"
    if kind is DocumentKind.PDF:
        return "application/octet-stream"
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise UploadValidationError("invalid_utf8") from error
    if not decoded.strip():
        raise UploadValidationError("empty_document")
    return "text/plain"


def validate_upload(
    filename: str,
    content: bytes,
    *,
    declared_media_type: str | None,
    max_bytes: int,
) -> ValidatedUpload:
    """Validate an in-memory upload without writing any rejected data."""
    safe_filename = _validate_filename(filename)
    extension = Path(safe_filename).suffix.casefold()
    if extension not in _EXTENSIONS:
        raise UploadValidationError("unsupported_extension")
    if not content:
        raise UploadValidationError("empty_document")
    if len(content) > max_bytes:
        raise UploadValidationError("document_too_large")

    kind, accepted_media_types = _EXTENSIONS[extension]
    detected_media_type = _detect_media_type(content, kind)
    normalized_declared = (declared_media_type or "").split(";", maxsplit=1)[0].strip().casefold()
    guessed_media_type = mimetypes.guess_type(safe_filename)[0]

    if detected_media_type not in accepted_media_types:
        raise UploadValidationError("media_type_mismatch")
    if normalized_declared and normalized_declared not in accepted_media_types:
        raise UploadValidationError("media_type_mismatch")
    if kind is DocumentKind.PDF and guessed_media_type != "application/pdf":
        raise UploadValidationError("media_type_mismatch")

    media_type = "application/pdf" if kind is DocumentKind.PDF else normalized_declared
    if not media_type or media_type not in accepted_media_types:
        media_type = "text/markdown" if kind is DocumentKind.MARKDOWN else "text/plain"
    return ValidatedUpload(
        filename=safe_filename,
        media_type=media_type,
        kind=kind,
        content=content,
    )
