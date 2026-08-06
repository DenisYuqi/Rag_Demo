from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from rag_mvp.domain.ingestion import DocumentKind
from rag_mvp.ingestion.validation import UploadValidationError, validate_upload


def _pdf_bytes(*, encrypted: bool = False) -> bytes:
    document = fitz.open()
    document.new_page().insert_text((72, 72), "Validated PDF content")
    if encrypted:
        content = document.tobytes(
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="owner-secret",
            user_pw="user-secret",
        )
    else:
        content = document.tobytes()
    document.close()
    return content


def test_supported_upload_at_exact_limit_is_accepted() -> None:
    content = "中文 and English".encode()
    result = validate_upload(
        "policy.md",
        content,
        declared_media_type="text/markdown; charset=utf-8",
        max_bytes=len(content),
    )

    assert result.kind is DocumentKind.MARKDOWN
    assert result.content == content


@pytest.mark.parametrize(
    ("filename", "content", "media_type", "code"),
    [
        ("empty.txt", b"", "text/plain", "empty_document"),
        ("blank.txt", b"  \n", "text/plain", "empty_document"),
        ("bad.exe", b"hello", "text/plain", "unsupported_extension"),
        ("../escape.txt", b"hello", "text/plain", "invalid_filename"),
        ("CON.txt", b"hello", "text/plain", "invalid_filename"),
        ("CONOUT$.txt", b"hello", "text/plain", "invalid_filename"),
        ("COM¹.txt", b"hello", "text/plain", "invalid_filename"),
        ("aux.policy.md", b"hello", "text/markdown", "invalid_filename"),
        ("policy.txt.", b"hello", "text/plain", "invalid_filename"),
        ("policy.txt ", b"hello", "text/plain", "invalid_filename"),
        ("policy:old.txt", b"hello", "text/plain", "invalid_filename"),
        ("bad\ud800.txt", b"hello", "text/plain", "invalid_filename"),
        ("fake.pdf", b"ordinary text", "application/pdf", "media_type_mismatch"),
        ("fake.txt", b"%PDF-1.7", "text/plain", "media_type_mismatch"),
        ("binary.txt", b"ordinary\x00text", "text/plain", "media_type_mismatch"),
        ("control.txt", b"ordinary\xc2\x85text", "text/plain", "media_type_mismatch"),
        ("bad.txt", b"\xff\xfe", "text/plain", "invalid_utf8"),
    ],
)
def test_invalid_uploads_fail_with_safe_codes(
    filename: str,
    content: bytes,
    media_type: str,
    code: str,
) -> None:
    with pytest.raises(UploadValidationError, match=code):
        validate_upload(filename, content, declared_media_type=media_type, max_bytes=1024)


def test_oversized_upload_is_rejected() -> None:
    with pytest.raises(UploadValidationError, match="document_too_large"):
        validate_upload("large.txt", b"ab", declared_media_type="text/plain", max_bytes=1)


def test_normal_text_whitespace_controls_are_accepted() -> None:
    content = b"one\ttwo\r\nthree"

    upload = validate_upload(
        "notes.txt",
        content,
        declared_media_type="text/plain",
        max_bytes=len(content),
    )

    assert upload.content == content


def test_structurally_valid_pdf_is_accepted() -> None:
    content = _pdf_bytes()

    upload = validate_upload(
        "policy.pdf",
        content,
        declared_media_type="application/pdf",
        max_bytes=len(content),
    )

    assert upload.content == content


@pytest.mark.parametrize(
    ("content", "code"),
    [
        (b"%PDF-1.7\n%%EOF", "malformed_pdf"),
        (_pdf_bytes(encrypted=True), "encrypted_pdf"),
    ],
)
def test_rejected_pdf_never_reaches_storage(
    tmp_path: Path,
    content: bytes,
    code: str,
) -> None:
    artifact = tmp_path / "sources" / "upload.pdf"

    with pytest.raises(UploadValidationError, match=code):
        upload = validate_upload(
            "upload.pdf",
            content,
            declared_media_type="application/pdf",
            max_bytes=len(content),
        )
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(upload.content)

    assert not artifact.exists()
