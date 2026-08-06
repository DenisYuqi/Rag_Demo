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


def test_document_normalization_is_idempotent_without_losing_inner_content() -> None:
    document = ExtractedDocument(
        kind=DocumentKind.PDF,
        blocks=tuple(
            ExtractedBlock(
                text=f"OUTER HEADER\nKeep this header-like line\nBody {page}\n"
                "Keep this footer-like line\nOUTER FOOTER",
                page_number=page,
            )
            for page in range(1, 4)
        ),
    )

    first = normalize_document(document)
    second = normalize_document(first)

    assert second is first
    assert all("Keep this header-like line" in block.text for block in first.blocks)
    assert all("Keep this footer-like line" in block.text for block in first.blocks)


def test_normalization_preserves_ambiguous_single_line_page_content() -> None:
    document = ExtractedDocument(
        kind=DocumentKind.PDF,
        blocks=tuple(ExtractedBlock(text="Policy", page_number=page) for page in range(1, 4)),
    )

    normalized = normalize_document(document)

    assert [block.text for block in normalized.blocks] == ["Policy", "Policy", "Policy"]


def test_header_footer_cleanup_does_not_erase_edge_only_pages() -> None:
    document = ExtractedDocument(
        kind=DocumentKind.PDF,
        blocks=tuple(
            ExtractedBlock(text="Policy title\nRequired notice", page_number=page)
            for page in range(1, 4)
        ),
    )

    normalized = normalize_document(document)

    assert [block.text for block in normalized.blocks] == [
        "Policy title\nRequired notice",
        "Policy title\nRequired notice",
        "Policy title\nRequired notice",
    ]


def test_section_paths_are_normalized_to_nfc() -> None:
    document = ExtractedDocument(
        kind=DocumentKind.MARKDOWN,
        blocks=(ExtractedBlock(text="Policy", section_path=("Cafe\u0301", "制度")),),
    )

    normalized = normalize_document(document)

    assert normalized.blocks[0].section_path == ("Café", "制度")


def test_canonical_digest_is_stable_for_equivalent_unicode() -> None:
    first = ExtractedDocument(
        kind=DocumentKind.TEXT,
        blocks=(ExtractedBlock(text="Cafe\u0301\r\nPolicy"),),
    )
    second = ExtractedDocument(
        kind=DocumentKind.TEXT,
        blocks=(ExtractedBlock(text="Café\nPolicy"),),
    )

    assert canonical_document_digest(first) == canonical_document_digest(second)


def test_canonical_digest_uses_unambiguous_section_serialization() -> None:
    joined_section = ExtractedDocument(
        kind=DocumentKind.MARKDOWN,
        blocks=(ExtractedBlock(text="Policy", section_path=("a/b",)),),
    )
    nested_section = ExtractedDocument(
        kind=DocumentKind.MARKDOWN,
        blocks=(ExtractedBlock(text="Policy", section_path=("a", "b")),),
    )

    assert canonical_document_digest(joined_section) != canonical_document_digest(nested_section)
