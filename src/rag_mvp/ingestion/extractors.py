"""Deterministic text and page-aware PDF extraction with selective OCR."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from contextlib import suppress
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
    normalization_version: str | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks if block.text)


@dataclass(frozen=True, slots=True)
class PageUsabilityPolicy:
    version: str = "page-usability-v1"
    minimum_alphanumeric_characters: int = 20
    minimum_printable_ratio: float = 0.70

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be non-empty")
        if (
            isinstance(self.minimum_alphanumeric_characters, bool)
            or not isinstance(self.minimum_alphanumeric_characters, int)
            or self.minimum_alphanumeric_characters < 1
        ):
            raise ValueError("minimum_alphanumeric_characters must be a positive integer")
        if (
            isinstance(self.minimum_printable_ratio, bool)
            or not isinstance(self.minimum_printable_ratio, (int, float))
            or not math.isfinite(self.minimum_printable_ratio)
            or not 0 <= self.minimum_printable_ratio <= 1
        ):
            raise ValueError("minimum_printable_ratio must be finite and between zero and one")

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


_MARKDOWN_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*$")
_MARKDOWN_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,}).*$")


def _markdown_heading_text(match: re.Match[str]) -> str:
    return re.sub(r"[ \t]+#+[ \t]*$", "", match.group(2)).strip()


def _closes_markdown_fence(line: str, marker: str) -> bool:
    character = re.escape(marker[0])
    return re.fullmatch(rf" {{0,3}}{character}{{{len(marker)},}}[ \t]*", line) is not None


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
    fence_marker: str | None = None

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            blocks.append(ExtractedBlock(text=body, section_path=tuple(headings)))
        current_lines.clear()

    for line in text.splitlines():
        if fence_marker is not None:
            current_lines.append(line)
            if _closes_markdown_fence(line, fence_marker):
                fence_marker = None
            continue

        fence_match = _MARKDOWN_FENCE.match(line)
        if fence_match:
            fence_marker = fence_match.group(1)
            current_lines.append(line)
            continue

        match = _MARKDOWN_HEADING.match(line)
        if match:
            heading_text = _markdown_heading_text(match)
            if not heading_text:
                current_lines.append(line)
                continue
            flush()
            level = len(match.group(1))
            headings[level - 1 :] = [heading_text]
            current_lines.append(line)
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
        try:
            needs_pass = document.needs_pass
            is_repaired = document.is_repaired
            page_count = document.page_count
        except Exception as error:
            raise ExtractionError("malformed_pdf") from error
        if needs_pass:
            raise ExtractionError("encrypted_pdf")
        if is_repaired:
            raise ExtractionError("malformed_pdf")
        if page_count < 1:
            raise ExtractionError("no_usable_text")
        blocks: list[ExtractedBlock] = []
        ocr_page_count = 0
        zoom = render_dpi / 72
        for page_index in range(page_count):
            try:
                page = document.load_page(page_index)
                native = page.get_text("text")
                if not isinstance(native, str):
                    raise TypeError
            except Exception as error:
                raise ExtractionError("pdf_page_failed") from error
            if policy.is_usable(native):
                text = native
                method = ExtractionMethod.NATIVE
            else:
                try:
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                    png_bytes = pixmap.tobytes("png")
                except Exception as error:
                    raise ExtractionError("pdf_page_failed") from error
                native_fallback = _usable_native_fallback(native, policy)
                ocr_page_count += 1
                try:
                    ocr_text: object = ocr.recognize(png_bytes, languages=languages)
                except Exception as error:
                    if native_fallback:
                        blocks.append(
                            ExtractedBlock(
                                text=native,
                                page_number=page_index + 1,
                                extraction_method=ExtractionMethod.NATIVE,
                            )
                        )
                        continue
                    raise ExtractionError("ocr_failed") from error
                if not isinstance(ocr_text, str):
                    if native_fallback:
                        text = native
                        method = ExtractionMethod.NATIVE
                        blocks.append(
                            ExtractedBlock(
                                text=text,
                                page_number=page_index + 1,
                                extraction_method=method,
                            )
                        )
                        continue
                    raise ExtractionError("ocr_failed")
                if policy.is_usable(ocr_text):
                    text = ocr_text
                    method = ExtractionMethod.OCR
                elif native_fallback:
                    text = native
                    method = ExtractionMethod.NATIVE
                else:
                    continue
            if text:
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
    except ExtractionError:
        raise
    except Exception as error:
        raise ExtractionError("malformed_pdf") from error
    finally:
        with suppress(Exception):
            document.close()


def _usable_native_fallback(text: str, policy: PageUsabilityPolicy) -> bool:
    stripped = text.strip()
    if not stripped or not any(character.isalnum() for character in stripped):
        return False
    printable_ratio = sum(character.isprintable() for character in stripped) / len(stripped)
    return printable_ratio >= policy.minimum_printable_ratio


def extraction_methods(blocks: Sequence[ExtractedBlock]) -> tuple[ExtractionMethod, ...]:
    """Return stable per-block diagnostics without exposing extracted text."""
    return tuple(block.extraction_method for block in blocks)
