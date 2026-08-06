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
    assert document.blocks[1].text == "## Leave\nAnnual leave rules"
    assert "# Handbook" in document.text
    assert "# Security" in document.text


def test_markdown_heading_inside_fence_is_content_not_structure() -> None:
    content = b"# Real\nIntro\n```markdown\n# Example only\n```\n## Child\nDetails"

    document = extract_utf8_text(content, kind=DocumentKind.MARKDOWN)

    assert [block.section_path for block in document.blocks] == [
        ("Real",),
        ("Real", "Child"),
    ]
    assert "# Example only" in document.blocks[0].text
    assert document.blocks[1].text == "## Child\nDetails"


def test_markdown_heading_only_document_remains_searchable() -> None:
    document = extract_utf8_text("# 制度 Policy\n## Leave".encode(), kind=DocumentKind.MARKDOWN)

    assert document.text == "# 制度 Policy\n\n## Leave"
