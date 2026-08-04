from __future__ import annotations

from rag_mvp.domain.ingestion import DocumentKind
from rag_mvp.ingestion.extractors import ExtractedBlock, ExtractedDocument
from rag_mvp.ingestion.normalization import (
    canonical_document_digest,
    normalize_document,
    normalize_text,
)


def test_unicode_and_line_normalization_is_idempotent() -> None:
    raw = "\ufeffCafe\u0301  \r\n\r\n\r\n制度\r"
    first = normalize_text(raw)

    assert first == "Café\n\n制度"
    assert normalize_text(first) == first


def test_repeated_page_headers_and_footers_are_removed_conservatively() -> None:
    document = ExtractedDocument(
        kind=DocumentKind.PDF,
        blocks=tuple(
            ExtractedBlock(text=f"CONFIDENTIAL\nPage body {page}\nInternal", page_number=page)
            for page in range(1, 4)
        ),
    )

    normalized = normalize_document(document)

    assert [block.text for block in normalized.blocks] == [
        "Page body 1",
        "Page body 2",
        "Page body 3",
    ]


def test_canonical_digest_is_stable_for_equivalent_unicode() -> None:
    first = ExtractedDocument(
        kind=DocumentKind.TEXT,
        blocks=(ExtractedBlock(text=normalize_text("Cafe\u0301\r\nPolicy")),),
    )
    second = ExtractedDocument(
        kind=DocumentKind.TEXT,
        blocks=(ExtractedBlock(text=normalize_text("Café\nPolicy")),),
    )

    assert canonical_document_digest(first) == canonical_document_digest(second)
