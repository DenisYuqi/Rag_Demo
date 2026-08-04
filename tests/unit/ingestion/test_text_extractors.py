from __future__ import annotations

from rag_mvp.domain.ingestion import DocumentKind
from rag_mvp.ingestion.extractors import extract_utf8_text


def test_plain_text_preserves_chinese_and_english() -> None:
    document = extract_utf8_text("制度 Policy\r\n第二行".encode(), kind=DocumentKind.TEXT)

    assert document.text == "制度 Policy\r\n第二行"


def test_markdown_tracks_nested_heading_paths() -> None:
    content = b"# Handbook\nWelcome\n## Leave\nAnnual leave rules\n# Security\nMFA required"
    document = extract_utf8_text(content, kind=DocumentKind.MARKDOWN)

    assert [block.section_path for block in document.blocks] == [
        ("Handbook",),
        ("Handbook", "Leave"),
        ("Security",),
    ]
    assert document.blocks[1].text == "Annual leave rules"
