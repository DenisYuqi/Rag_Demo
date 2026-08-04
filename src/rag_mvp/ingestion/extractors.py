"""Deterministic text and page-aware PDF extraction with selective OCR."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Protocol

import fitz
import pytesseract
from PIL import Image

from rag_mvp.domain.ingestion import DocumentKind, ExtractionMethod


class ExtractionError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class ExtractedBlock:
    text: str
    page_number: int | None = None
    section_path: tuple[str, ...] = ()
    extraction_method: ExtractionMethod = ExtractionMethod.TEXT


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    kind: DocumentKind
    blocks: tuple[ExtractedBlock, ...]
    ocr_page_count: int = 0

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks if block.text)


@dataclass(frozen=True, slots=True)
class PageUsabilityPolicy:
    version: str = "page-usability-v1"
    minimum_alphanumeric_characters: int = 20
    minimum_printable_ratio: float = 0.70

    def is_usable(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        semantic_count = sum(character.isalnum() for character in stripped)
        printable_ratio = sum(character.isprintable() for character in stripped) / len(stripped)
        return (
            semantic_count >= self.minimum_alphanumeric_characters
            and printable_ratio >= self.minimum_printable_ratio
        )


class OcrAdapter(Protocol):
    @property
    def version(self) -> str: ...

    def recognize(self, png_bytes: bytes, *, languages: str) -> str: ...


@dataclass(frozen=True, slots=True)
class TesseractOcrAdapter:
    version: str = "tesseract-pytesseract-v1"

    def recognize(self, png_bytes: bytes, *, languages: str) -> str:
        with Image.open(BytesIO(png_bytes)) as image:
            return str(pytesseract.image_to_string(image, lang=languages))


_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def extract_utf8_text(content: bytes, *, kind: DocumentKind) -> ExtractedDocument:
    if kind not in {DocumentKind.TEXT, DocumentKind.MARKDOWN}:
        raise ExtractionError("unsupported_text_kind")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ExtractionError("invalid_utf8") from error
    if not text.strip():
        raise ExtractionError("no_usable_text")
    if kind is DocumentKind.TEXT:
        return ExtractedDocument(kind=kind, blocks=(ExtractedBlock(text=text),))

    headings: list[str] = []
    current_lines: list[str] = []
    blocks: list[ExtractedBlock] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            blocks.append(ExtractedBlock(text=body, section_path=tuple(headings)))
        current_lines.clear()

    for line in text.splitlines():
        match = _MARKDOWN_HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            headings[level - 1 :] = [match.group(2).strip()]
            continue
        current_lines.append(line)
    flush()
    if not blocks:
        raise ExtractionError("no_usable_text")
    return ExtractedDocument(kind=kind, blocks=tuple(blocks))


def extract_pdf(
    content: bytes,
    *,
    ocr: OcrAdapter,
    languages: str = "chi_sim+eng",
    usability: PageUsabilityPolicy | None = None,
    render_dpi: int = 200,
) -> ExtractedDocument:
    policy = usability or PageUsabilityPolicy()
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as error:
        raise ExtractionError("malformed_pdf") from error

    try:
        if document.needs_pass:
            raise ExtractionError("encrypted_pdf")
        if document.page_count < 1:
            raise ExtractionError("no_usable_text")
        blocks: list[ExtractedBlock] = []
        ocr_page_count = 0
        zoom = render_dpi / 72
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            native = page.get_text("text")
            if policy.is_usable(native):
                text = native
                method = ExtractionMethod.NATIVE
            else:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                try:
                    text = ocr.recognize(pixmap.tobytes("png"), languages=languages)
                except Exception as error:
                    raise ExtractionError("ocr_failed") from error
                method = ExtractionMethod.OCR
                ocr_page_count += 1
            if text.strip():
                blocks.append(
                    ExtractedBlock(
                        text=text,
                        page_number=page_index + 1,
                        extraction_method=method,
                    )
                )
        if not blocks:
            raise ExtractionError("no_usable_text")
        return ExtractedDocument(
            kind=DocumentKind.PDF,
            blocks=tuple(blocks),
            ocr_page_count=ocr_page_count,
        )
    finally:
        document.close()


def extraction_methods(blocks: Sequence[ExtractedBlock]) -> tuple[ExtractionMethod, ...]:
    """Return stable per-block diagnostics without exposing extracted text."""
    return tuple(block.extraction_method for block in blocks)
