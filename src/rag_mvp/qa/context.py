"""Rank-preserving, token-bounded context selection for grounded generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

from rag_mvp.domain.ingestion import ParentChunk
from rag_mvp.domain.retrieval import RankingEvidence
from rag_mvp.ingestion.chunking import TOKENIZER_VERSION, token_spans

CONTEXT_SELECTION_VERSION = "parent-expanded-ranked-token-prefix-v1"
CONTEXT_TOKENIZER_VERSION = TOKENIZER_VERSION


class ContextSelectionError(ValueError):
    """A stable validation failure for retrieved generation context."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ContextChunk:
    evidence: RankingEvidence = field(repr=False)
    parent_chunk_id: str
    text: str = field(repr=False)
    rank: int
    token_count: int
    original_token_count: int
    truncated: bool

    @property
    def chunk_id(self) -> str:
        return self.evidence.chunk_id

    @property
    def final_rank(self) -> int:
        return self.rank


class ParentChunkResolver(Protocol):
    def get_many(
        self,
        revision_id: str,
        parent_chunk_ids: Sequence[str],
    ) -> Mapping[str, ParentChunk]: ...


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
        parent_resolver: ParentChunkResolver | None = None,
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
        self.parent_resolver = parent_resolver

    def build(self, evidence: Sequence[RankingEvidence]) -> ContextSelection:
        ordered = self._validated_evidence(evidence)
        distinct: list[RankingEvidence] = []
        seen_parents: set[str] = set()
        for item in ordered:
            if item.parent_chunk_id not in seen_parents:
                seen_parents.add(item.parent_chunk_id)
                distinct.append(item)
        parents = self._resolve_parents(tuple(distinct))
        chunks: list[ContextChunk] = []
        remaining_tokens = self.maximum_total_tokens

        for item in distinct[: self.maximum_chunks]:
            if remaining_tokens == 0:
                break
            parent = parents[item.parent_chunk_id]
            spans = token_spans(parent.text)
            original_token_count = len(spans)
            if original_token_count == 0:
                raise ContextSelectionError("evidence_text_has_no_tokens")
            token_limit = min(self.maximum_tokens_per_chunk, remaining_tokens)
            token_count = min(original_token_count, token_limit)
            truncated = token_count < original_token_count
            text = parent.text if not truncated else parent.text[: spans[token_count][0]]
            if not text.strip():
                raise ContextSelectionError("context_text_empty")
            chunks.append(
                ContextChunk(
                    evidence=item,
                    parent_chunk_id=parent.parent_chunk_id,
                    text=text,
                    rank=len(chunks) + 1,
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
            available_evidence_count=len(distinct),
            omitted_evidence_count=len(distinct) - len(selected),
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

    def _resolve_parents(
        self,
        evidence: tuple[RankingEvidence, ...],
    ) -> Mapping[str, ParentChunk]:
        if not evidence:
            return {}
        if self.parent_resolver is None:
            parents: dict[str, ParentChunk] = {}
            for item in evidence:
                if item.parent_chunk_id != item.chunk_id:
                    raise ContextSelectionError("parent_resolver_missing")
                spans = token_spans(item.text)
                if not spans:
                    raise ContextSelectionError("evidence_text_has_no_tokens")
                parents[item.parent_chunk_id] = ParentChunk.model_construct(
                    parent_chunk_id=item.parent_chunk_id,
                    source_id=item.source_id,
                    document_version=item.document_version,
                    ordinal=item.ordinal or 0,
                    text=item.text,
                    content_digest=item.content_digest or "inline-parent-context",
                    locator=item.locator,
                    token_count=len(spans),
                )
            return parents
        revision_ids = {item.revision_id for item in evidence}
        if len(revision_ids) != 1 or None in revision_ids:
            raise ContextSelectionError("parent_revision_invalid")
        revision_id = cast(str, next(iter(revision_ids)))
        parent_ids = tuple(item.parent_chunk_id for item in evidence)
        try:
            resolved = dict(self.parent_resolver.get_many(revision_id, parent_ids))
        except Exception:
            raise ContextSelectionError("parent_lookup_failed") from None
        if set(resolved) != set(parent_ids):
            raise ContextSelectionError("parent_chunk_missing")
        for item in evidence:
            parent = resolved[item.parent_chunk_id]
            if (
                parent.source_id != item.source_id
                or parent.document_version != item.document_version
            ):
                raise ContextSelectionError("parent_chunk_mismatch")
        return resolved
