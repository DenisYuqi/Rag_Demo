"""Rank-preserving, token-bounded context selection for grounded generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from rag_mvp.domain.retrieval import RankingEvidence
from rag_mvp.ingestion.chunking import TOKENIZER_VERSION, token_spans

CONTEXT_SELECTION_VERSION = "ranked-token-prefix-v1"
CONTEXT_TOKENIZER_VERSION = TOKENIZER_VERSION


class ContextSelectionError(ValueError):
    """A stable validation failure for retrieved generation context."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ContextChunk:
    evidence: RankingEvidence = field(repr=False)
    text: str = field(repr=False)
    token_count: int
    original_token_count: int
    truncated: bool

    @property
    def chunk_id(self) -> str:
        return self.evidence.chunk_id

    @property
    def final_rank(self) -> int:
        return self.evidence.final_rank


@dataclass(frozen=True, slots=True)
class ContextSelection:
    chunks: tuple[ContextChunk, ...]
    total_tokens: int
    available_evidence_count: int
    omitted_evidence_count: int
    truncated_chunk_count: int
    tokenizer_version: str = CONTEXT_TOKENIZER_VERSION
    version: str = CONTEXT_SELECTION_VERSION


class ContextBuilder:
    """Select highest-ranked evidence without exceeding any configured text budget."""

    def __init__(
        self,
        *,
        maximum_chunks: int = 5,
        maximum_tokens_per_chunk: int = 500,
        maximum_total_tokens: int = 2000,
    ) -> None:
        for name, value in (
            ("maximum_chunks", maximum_chunks),
            ("maximum_tokens_per_chunk", maximum_tokens_per_chunk),
            ("maximum_total_tokens", maximum_total_tokens),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.maximum_chunks = maximum_chunks
        self.maximum_tokens_per_chunk = maximum_tokens_per_chunk
        self.maximum_total_tokens = maximum_total_tokens

    def build(self, evidence: Sequence[RankingEvidence]) -> ContextSelection:
        ordered = self._validated_evidence(evidence)
        chunks: list[ContextChunk] = []
        remaining_tokens = self.maximum_total_tokens

        for item in ordered[: self.maximum_chunks]:
            if remaining_tokens == 0:
                break
            spans = token_spans(item.text)
            original_token_count = len(spans)
            if original_token_count == 0:
                raise ContextSelectionError("evidence_text_has_no_tokens")
            token_limit = min(self.maximum_tokens_per_chunk, remaining_tokens)
            token_count = min(original_token_count, token_limit)
            truncated = token_count < original_token_count
            text = item.text if not truncated else item.text[: spans[token_count][0]]
            if not text.strip():
                raise ContextSelectionError("context_text_empty")
            chunks.append(
                ContextChunk(
                    evidence=item,
                    text=text,
                    token_count=token_count,
                    original_token_count=original_token_count,
                    truncated=truncated,
                )
            )
            remaining_tokens -= token_count

        selected = tuple(chunks)
        return ContextSelection(
            chunks=selected,
            total_tokens=sum(chunk.token_count for chunk in selected),
            available_evidence_count=len(ordered),
            omitted_evidence_count=len(ordered) - len(selected),
            truncated_chunk_count=sum(chunk.truncated for chunk in selected),
        )

    @staticmethod
    def _validated_evidence(
        evidence: object,
    ) -> tuple[RankingEvidence, ...]:
        if isinstance(evidence, (str, bytes, bytearray)) or not isinstance(evidence, Sequence):
            raise ContextSelectionError("evidence_sequence_invalid")
        items: list[RankingEvidence] = []
        for item in cast(Sequence[object], evidence):
            if not isinstance(item, RankingEvidence):
                raise ContextSelectionError("evidence_sequence_invalid")
            items.append(item)
        ordered = tuple(sorted(items, key=lambda item: item.final_rank))
        ranks = tuple(item.final_rank for item in ordered)
        if ranks != tuple(range(1, len(ordered) + 1)):
            raise ContextSelectionError("evidence_ranks_invalid")
        chunk_ids = tuple(item.chunk_id for item in ordered)
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ContextSelectionError("duplicate_evidence_chunk")
        return ordered
