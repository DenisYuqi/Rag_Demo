"""Canonical Unicode normalization and stable content identities."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import replace

from rag_mvp.ingestion.extractors import ExtractedBlock, ExtractedDocument

NORMALIZATION_VERSION = "unicode-nfc-lines-v1"
_TRAILING_HORIZONTAL_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    canonical = unicodedata.normalize("NFC", text.lstrip("\ufeff"))
    canonical = canonical.replace("\r\n", "\n").replace("\r", "\n")
    canonical = _TRAILING_HORIZONTAL_SPACE.sub("", canonical)
    canonical = _EXCESS_BLANK_LINES.sub("\n\n", canonical)
    return canonical.strip()


def _edge_lines(block: ExtractedBlock) -> tuple[str | None, str | None]:
    lines = [line.strip() for line in normalize_text(block.text).splitlines() if line.strip()]
    return (lines[0], lines[-1]) if lines else (None, None)


def normalize_document(document: ExtractedDocument) -> ExtractedDocument:
    """Normalize blocks and remove only headers/footers repeated on every page."""
    page_blocks = [block for block in document.blocks if block.page_number is not None]
    repeated_headers: set[str] = set()
    repeated_footers: set[str] = set()
    if len(page_blocks) >= 3:
        edges = [_edge_lines(block) for block in page_blocks]
        header_counts = Counter(header for header, _ in edges if header)
        footer_counts = Counter(footer for _, footer in edges if footer)
        repeated_headers = {
            line for line, count in header_counts.items() if count == len(page_blocks)
        }
        repeated_footers = {
            line for line, count in footer_counts.items() if count == len(page_blocks)
        }

    normalized_blocks: list[ExtractedBlock] = []
    for block in document.blocks:
        text = normalize_text(block.text)
        lines = text.splitlines()
        if block.page_number is not None and lines:
            if lines[0].strip() in repeated_headers:
                lines = lines[1:]
            if lines and lines[-1].strip() in repeated_footers:
                lines = lines[:-1]
            text = normalize_text("\n".join(lines))
        if text:
            normalized_blocks.append(replace(block, text=text))
    return replace(document, blocks=tuple(normalized_blocks))


def canonical_document_digest(document: ExtractedDocument) -> str:
    pieces = [
        f"page={block.page_number or 0};section={'/'.join(block.section_path)}\n{block.text}"
        for block in document.blocks
    ]
    return hashlib.sha256("\n\n".join(pieces).encode("utf-8")).hexdigest()
