from __future__ import annotations

import fitz
import pytest

from rag_mvp.domain.ingestion import ExtractionMethod
from rag_mvp.ingestion.extractors import ExtractionError, extract_pdf


class RecordingOcr:
    version = "fake-ocr-v1"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def recognize(self, png_bytes: bytes, *, languages: str) -> str:
        self.calls.append(languages)
        return self.responses[len(self.calls) - 1]


def _mixed_pdf() -> bytes:
    document = fitz.open()
    first = document.new_page()
    first.insert_text(
        (72, 72),
        "This first page has enough searchable native policy text for extraction.",
    )
    document.new_page()
    content = document.tobytes()
    document.close()
    return content


def test_mixed_pdf_keeps_page_order_and_ocr_diagnostics() -> None:
    ocr = RecordingOcr(["扫描页 recovered policy content"])

    result = extract_pdf(_mixed_pdf(), ocr=ocr)

    assert [block.page_number for block in result.blocks] == [1, 2]
    assert [block.extraction_method for block in result.blocks] == [
        ExtractionMethod.NATIVE,
        ExtractionMethod.OCR,
    ]
    assert result.ocr_page_count == 1
    assert ocr.calls == ["chi_sim+eng"]


def test_all_pages_empty_fails_without_publication_payload() -> None:
    document = fitz.open()
    document.new_page()
    content = document.tobytes()
    document.close()

    with pytest.raises(ExtractionError, match="no_usable_text"):
        extract_pdf(content, ocr=RecordingOcr([" "]))
