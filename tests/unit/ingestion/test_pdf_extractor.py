from __future__ import annotations

import fitz
import pytest

from rag_mvp.domain.ingestion import ExtractionMethod
from rag_mvp.ingestion.extractors import ExtractionError, extract_pdf


class FailingOcr:
    version = "test"

    def recognize(self, png_bytes: bytes, *, languages: str) -> str:
        raise AssertionError("OCR must not be called for usable native text")


def _native_pdf() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "This digital policy page contains enough native English text for extraction.",
    )
    content = document.tobytes()
    document.close()
    return content


def test_native_pdf_preserves_page_and_skips_ocr() -> None:
    result = extract_pdf(_native_pdf(), ocr=FailingOcr())

    assert result.ocr_page_count == 0
    assert result.blocks[0].page_number == 1
    assert result.blocks[0].extraction_method is ExtractionMethod.NATIVE
    assert "digital policy" in result.blocks[0].text


def test_malformed_pdf_has_safe_error() -> None:
    with pytest.raises(ExtractionError, match="malformed_pdf"):
        extract_pdf(b"%PDF-corrupt", ocr=FailingOcr())


def test_encrypted_pdf_without_password_has_safe_error() -> None:
    document = fitz.open()
    document.new_page().insert_text((72, 72), "Protected document")
    content = document.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw="user-secret",
    )
    document.close()

    with pytest.raises(ExtractionError, match="encrypted_pdf"):
        extract_pdf(content, ocr=FailingOcr())
