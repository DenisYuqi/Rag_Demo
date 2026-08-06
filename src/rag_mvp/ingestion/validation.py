"""Fail-fast upload validation before any persistent artifact is created."""

from __future__ import annotations

import mimetypes
import unicodedata
from contextlib import suppress
from pathlib import Path

import fitz
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

_WINDOWS_INVALID_FILENAME_CHARACTERS = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED_FILENAMES = frozenset(
    {
        "aux",
        "clock$",
        "con",
        "conin$",
        "conout$",
        "nul",
        "prn",
        *(f"com{number}" for number in range(1, 10)),
        *(f"com{number}" for number in "¹²³"),
        *(f"lpt{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in "¹²³"),
    }
)
_ALLOWED_TEXT_CONTROL_BYTES = frozenset({9, 10, 13})


def _validate_filename(filename: str) -> str:
    if not filename:
        raise UploadValidationError("invalid_filename")
    try:
        encoded_filename = filename.encode("utf-8")
    except UnicodeEncodeError as error:
        raise UploadValidationError("invalid_filename") from error
    if len(encoded_filename) > 255:
        raise UploadValidationError("invalid_filename")
    if filename in {".", ".."} or filename[-1] in {" ", "."}:
        raise UploadValidationError("invalid_filename")
    if any(unicodedata.category(character) == "Cc" for character in filename):
        raise UploadValidationError("invalid_filename")
    if any(character in _WINDOWS_INVALID_FILENAME_CHARACTERS for character in filename):
        raise UploadValidationError("invalid_filename")
    windows_stem = filename.split(".", maxsplit=1)[0].rstrip(" .").casefold()
    if windows_stem in _WINDOWS_RESERVED_FILENAMES:
        raise UploadValidationError("invalid_filename")
    return filename


def _validate_pdf_structure(content: bytes) -> None:
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as error:
        raise UploadValidationError("malformed_pdf") from error

    try:
        if document.needs_pass:
            raise UploadValidationError("encrypted_pdf")
        if document.is_repaired:
            raise UploadValidationError("malformed_pdf")
        page_count = document.page_count
        if page_count < 1:
            raise UploadValidationError("malformed_pdf")
        for page_index in range(page_count):
            document.load_page(page_index)
    except UploadValidationError:
        raise
    except Exception as error:
        raise UploadValidationError("malformed_pdf") from error
    finally:
        with suppress(Exception):
            document.close()


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
    if any(
        (byte < 32 and byte not in _ALLOWED_TEXT_CONTROL_BYTES) or byte == 127 for byte in content
    ):
        raise UploadValidationError("media_type_mismatch")
    if any(
        character not in "\t\n\r" and unicodedata.category(character) == "Cc"
        for character in decoded
    ):
        raise UploadValidationError("media_type_mismatch")
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
    if kind is DocumentKind.PDF:
        _validate_pdf_structure(content)

    media_type = "application/pdf" if kind is DocumentKind.PDF else normalized_declared
    if not media_type or media_type not in accepted_media_types:
        media_type = "text/markdown" if kind is DocumentKind.MARKDOWN else "text/plain"
    return ValidatedUpload(
        filename=safe_filename,
        media_type=media_type,
        kind=kind,
        content=content,
    )
