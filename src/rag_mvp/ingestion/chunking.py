"""Bounded, overlapping, deterministic structure/page-aware chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from rag_mvp.domain.ingestion import Chunk, ChunkLocator
from rag_mvp.ingestion.extractors import ExtractedDocument

CHUNKING_VERSION = "structure-page-token-v1"
TOKENIZER_VERSION = "unicode-word-cjk-v1"
_TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z0-9]+(?:[._'-][A-Za-z0-9]+)*|[^\s]",
    re.UNICODE,
)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_tokens: int = 500
    overlap_tokens: int = 80
    version: str = CHUNKING_VERSION
    tokenizer_version: str = TOKENIZER_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.target_tokens, bool)
            or not isinstance(self.target_tokens, int)
            or isinstance(self.overlap_tokens, bool)
            or not isinstance(self.overlap_tokens, int)
        ):
            raise ValueError("target_tokens and overlap_tokens must be integers")
        if self.target_tokens < 1:
            raise ValueError("target_tokens must be positive")
        if self.overlap_tokens < 0 or self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be non-negative and below target_tokens")


def token_spans(text: str) -> tuple[tuple[int, int], ...]:
    return tuple((match.start(), match.end()) for match in _TOKEN_PATTERN.finditer(text))


def _chunk_id(
    source_id: str,
    document_version: int,
    ordinal: int,
    content_digest: str,
) -> str:
    raw = f"{source_id}:{document_version}:{ordinal}:{content_digest}"
    return "chk_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def chunk_document(
    document: ExtractedDocument,
    *,
    source_id: str,
    document_version: int,
    config: ChunkingConfig | None = None,
) -> tuple[Chunk, ...]:
    resolved = config or ChunkingConfig()
    chunks: list[Chunk] = []
    document_offset = 0
    has_prior_block_text = False

    for block in document.blocks:
        if not block.text:
            continue
        if has_prior_block_text:
            document_offset += 2
        spans = token_spans(block.text)
        if not spans:
            document_offset += len(block.text)
            has_prior_block_text = True
            continue
        step = resolved.target_tokens - resolved.overlap_tokens
        for token_start in range(0, len(spans), step):
            token_end = min(token_start + resolved.target_tokens, len(spans))
            char_start = 0 if token_start == 0 else spans[token_start][0]
            char_end = len(block.text) if token_end == len(spans) else spans[token_end][0]
            text = block.text[char_start:char_end]
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            ordinal = len(chunks)
            pages = (block.page_number,) if block.page_number is not None else ()
            locator = ChunkLocator(
                pages=pages,
                section_path=block.section_path,
                char_start=document_offset + char_start if not pages else None,
                char_end=document_offset + char_end if not pages else None,
            )
            chunk = Chunk(
                chunk_id=_chunk_id(source_id, document_version, ordinal, digest),
                source_id=source_id,
                document_version=document_version,
                ordinal=ordinal,
                text=text,
                content_digest=digest,
                locator=locator,
                token_count=token_end - token_start,
            )
            # The shared domain model trims strings; restore the exact validated source slice.
            chunks.append(chunk.model_copy(update={"text": text}))
            if token_end == len(spans):
                break
        document_offset += len(block.text)
        has_prior_block_text = True
    return tuple(chunks)
