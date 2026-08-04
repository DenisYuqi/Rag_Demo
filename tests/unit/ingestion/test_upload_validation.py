from __future__ import annotations

import pytest

from rag_mvp.domain.ingestion import DocumentKind
from rag_mvp.ingestion.validation import UploadValidationError, validate_upload


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
        ("fake.pdf", b"ordinary text", "application/pdf", "media_type_mismatch"),
        ("fake.txt", b"%PDF-1.7", "text/plain", "media_type_mismatch"),
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
