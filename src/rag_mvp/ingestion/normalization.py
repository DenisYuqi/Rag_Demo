"""Canonical Unicode normalization and stable content identities."""

from __future__ import annotations

import hashlib
import json
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
    lines = [line.strip() for line in block.text.splitlines() if line.strip()]
    return (lines[0], lines[-1]) if lines else (None, None)


def normalize_document(document: ExtractedDocument) -> ExtractedDocument:
    """Normalize blocks and remove only headers/footers repeated on every page."""
    if document.normalization_version == NORMALIZATION_VERSION:
        return document

    normalized_blocks = tuple(
        replace(
            block,
            text=normalize_text(block.text),
            section_path=tuple(unicodedata.normalize("NFC", part) for part in block.section_path),
        )
        for block in document.blocks
    )
    page_blocks = [block for block in normalized_blocks if block.page_number is not None]
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
        ambiguous_edges = repeated_headers & repeated_footers
        repeated_headers -= ambiguous_edges
        repeated_footers -= ambiguous_edges

    cleaned_blocks: list[ExtractedBlock] = []
    for block in normalized_blocks:
        text = block.text
        lines = text.splitlines()
        if block.page_number is not None and lines:
            first = 1 if lines[0].strip() in repeated_headers else 0
            last = len(lines) - (1 if lines[-1].strip() in repeated_footers else 0)
            if any(line.strip() for line in lines[first:last]):
                text = normalize_text("\n".join(lines[first:last]))
        cleaned_blocks.append(replace(block, text=text))
    return replace(
        document,
        blocks=tuple(cleaned_blocks),
        normalization_version=NORMALIZATION_VERSION,
    )


def canonical_document_digest(document: ExtractedDocument) -> str:
    normalized = normalize_document(document)
    payload = {
        "blocks": [
            {
                "page_number": block.page_number,
                "section_path": list(block.section_path),
                "text": block.text,
            }
            for block in normalized.blocks
        ],
        "format": "canonical-document-v1",
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
