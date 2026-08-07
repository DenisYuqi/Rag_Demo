"""Deterministic structure-aware parent and overlapping child chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from rag_mvp.domain.ingestion import Chunk, ChunkLocator, ParentChunk
from rag_mvp.ingestion.extractors import ExtractedDocument

CHUNKING_VERSION = "structure-page-parent-child-token-v1"
TOKENIZER_VERSION = "unicode-word-cjk-v1"
_TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z0-9]+(?:[._'-][A-Za-z0-9]+)*|[^\s]",
    re.UNICODE,
)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_tokens: int = 512
    overlap_tokens: int = 128
    parent_target_tokens: int = 1536
    version: str = CHUNKING_VERSION
    tokenizer_version: str = TOKENIZER_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.target_tokens, bool)
            or not isinstance(self.target_tokens, int)
            or isinstance(self.overlap_tokens, bool)
            or not isinstance(self.overlap_tokens, int)
            or isinstance(self.parent_target_tokens, bool)
            or not isinstance(self.parent_target_tokens, int)
        ):
            raise ValueError("chunk token bounds must be integers")
        if self.target_tokens < 1:
            raise ValueError("target_tokens must be positive")
        if self.overlap_tokens < 0 or self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be non-negative and below target_tokens")
        if self.parent_target_tokens < self.target_tokens:
            raise ValueError("parent_target_tokens must be at least target_tokens")


@dataclass(frozen=True, slots=True)
class ChunkedDocument:
    parents: tuple[ParentChunk, ...]
    children: tuple[Chunk, ...]


def token_spans(text: str) -> tuple[tuple[int, int], ...]:
    return tuple((match.start(), match.end()) for match in _TOKEN_PATTERN.finditer(text))


def _chunk_id(
    source_id: str,
    document_version: int,
    ordinal: int,
    content_digest: str,
    parent_chunk_id: str,
) -> str:
    raw = f"{source_id}:{document_version}:{ordinal}:{content_digest}:{parent_chunk_id}"
    return "chk_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _parent_chunk_id(
    source_id: str,
    document_version: int,
    ordinal: int,
    content_digest: str,
) -> str:
    raw = f"{source_id}:{document_version}:{ordinal}:{content_digest}"
    return "par_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _character_bounds(
    text: str,
    spans: tuple[tuple[int, int], ...],
    token_start: int,
    token_end: int,
) -> tuple[int, int]:
    char_start = 0 if token_start == 0 else spans[token_start][0]
    char_end = len(text) if token_end == len(spans) else spans[token_end][0]
    return char_start, char_end


def chunk_document(
    document: ExtractedDocument,
    *,
    source_id: str,
    document_version: int,
    config: ChunkingConfig | None = None,
) -> tuple[Chunk, ...]:
    """Return indexed child chunks for callers that do not need parent persistence."""

    return chunk_document_hierarchy(
        document,
        source_id=source_id,
        document_version=document_version,
        config=config,
    ).children


def chunk_document_hierarchy(
    document: ExtractedDocument,
    *,
    source_id: str,
    document_version: int,
    config: ChunkingConfig | None = None,
) -> ChunkedDocument:
    resolved = config or ChunkingConfig()
    parents: list[ParentChunk] = []
    children: list[Chunk] = []
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
        for parent_token_start in range(0, len(spans), resolved.parent_target_tokens):
            parent_token_end = min(
                parent_token_start + resolved.parent_target_tokens,
                len(spans),
            )
            parent_char_start, parent_char_end = _character_bounds(
                block.text,
                spans,
                parent_token_start,
                parent_token_end,
            )
            parent_text = block.text[parent_char_start:parent_char_end]
            parent_digest = hashlib.sha256(parent_text.encode("utf-8")).hexdigest()
            parent_ordinal = len(parents)
            parent_id = _parent_chunk_id(
                source_id,
                document_version,
                parent_ordinal,
                parent_digest,
            )
            pages = (block.page_number,) if block.page_number is not None else ()
            parent_locator = ChunkLocator(
                pages=pages,
                section_path=block.section_path,
                char_start=document_offset + parent_char_start if not pages else None,
                char_end=document_offset + parent_char_end if not pages else None,
            )
            parent = ParentChunk(
                parent_chunk_id=parent_id,
                source_id=source_id,
                document_version=document_version,
                ordinal=parent_ordinal,
                text=parent_text,
                content_digest=parent_digest,
                locator=parent_locator,
                token_count=parent_token_end - parent_token_start,
            )
            parents.append(parent.model_copy(update={"text": parent_text}))

            child_spans = token_spans(parent_text)
            child_step = resolved.target_tokens - resolved.overlap_tokens
            for child_token_start in range(0, len(child_spans), child_step):
                child_token_end = min(
                    child_token_start + resolved.target_tokens,
                    len(child_spans),
                )
                child_char_start, child_char_end = _character_bounds(
                    parent_text,
                    child_spans,
                    child_token_start,
                    child_token_end,
                )
                child_text = parent_text[child_char_start:child_char_end]
                child_digest = hashlib.sha256(child_text.encode("utf-8")).hexdigest()
                child_ordinal = len(children)
                child_locator = ChunkLocator(
                    pages=pages,
                    section_path=block.section_path,
                    char_start=(
                        document_offset + parent_char_start + child_char_start
                        if not pages
                        else None
                    ),
                    char_end=(
                        document_offset + parent_char_start + child_char_end if not pages else None
                    ),
                )
                child = Chunk(
                    chunk_id=_chunk_id(
                        source_id,
                        document_version,
                        child_ordinal,
                        child_digest,
                        parent_id,
                    ),
                    parent_chunk_id=parent_id,
                    source_id=source_id,
                    document_version=document_version,
                    ordinal=child_ordinal,
                    text=child_text,
                    content_digest=child_digest,
                    locator=child_locator,
                    token_count=child_token_end - child_token_start,
                )
                children.append(child.model_copy(update={"text": child_text}))
                if child_token_end == len(child_spans):
                    break
        document_offset += len(block.text)
        has_prior_block_text = True
    return ChunkedDocument(parents=tuple(parents), children=tuple(children))


def chunk_document_legacy(
    document: ExtractedDocument,
    *,
    source_id: str,
    document_version: int,
    target_tokens: int,
    overlap_tokens: int,
) -> tuple[Chunk, ...]:
    """Reproduce immutable pre-parent evaluation corpora without indexing them."""

    if target_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= target_tokens:
        raise ValueError("legacy chunk bounds are invalid")
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
        step = target_tokens - overlap_tokens
        for token_start in range(0, len(spans), step):
            token_end = min(token_start + target_tokens, len(spans))
            char_start, char_end = _character_bounds(
                block.text,
                spans,
                token_start,
                token_end,
            )
            text = block.text[char_start:char_end]
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            ordinal = len(chunks)
            raw = f"{source_id}:{document_version}:{ordinal}:{digest}"
            chunk_id = "chk_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
            pages = (block.page_number,) if block.page_number is not None else ()
            locator = ChunkLocator(
                pages=pages,
                section_path=block.section_path,
                char_start=document_offset + char_start if not pages else None,
                char_end=document_offset + char_end if not pages else None,
            )
            chunk = Chunk(
                chunk_id=chunk_id,
                parent_chunk_id=chunk_id,
                source_id=source_id,
                document_version=document_version,
                ordinal=ordinal,
                text=text,
                content_digest=digest,
                locator=locator,
                token_count=token_end - token_start,
            )
            chunks.append(chunk.model_copy(update={"text": text}))
            if token_end == len(spans):
                break
        document_offset += len(block.text)
        has_prior_block_text = True
    return tuple(chunks)
