"""Retrieval and ranking domain contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field, field_validator

from rag_mvp.domain._base import (
    DomainModel,
    FiniteFloat,
    Identifier,
    NonEmptyText,
    NonNegativeFiniteFloat,
)
from rag_mvp.domain.ingestion import ChunkLocator


class RetrievalMode(StrEnum):
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid-rerank"


class CacheOutcome(StrEnum):
    HIT = "hit"
    MISS = "miss"
    BYPASS = "bypass"
    NOT_APPLICABLE = "not-applicable"


class RetrievalCandidate(DomainModel):
    chunk_id: Identifier
    source_id: Identifier
    display_title: Identifier
    document_version: Annotated[int, Field(gt=0)]
    locator: ChunkLocator
    text: NonEmptyText
    dense_rank: Annotated[int, Field(gt=0)] | None = None
    dense_score: FiniteFloat | None = None
    bm25_rank: Annotated[int, Field(gt=0)] | None = None
    bm25_score: FiniteFloat | None = None
    rrf_score: NonNegativeFiniteFloat | None = None
    reranking_rank: Annotated[int, Field(gt=0)] | None = None

    @field_validator("rrf_score")
    @classmethod
    def rrf_is_nonnegative(cls, value: float | None) -> float | None:
        return value


class RankingEvidence(RetrievalCandidate):
    final_rank: Annotated[int, Field(gt=0)]


class RetrievalDiagnostics(DomainModel):
    request_id: Identifier
    requested_mode: RetrievalMode
    effective_mode: RetrievalMode
    index_revision: Identifier
    candidate_counts: dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    stage_timings_ms: dict[str, NonNegativeFiniteFloat] = Field(default_factory=dict)
    cache_status: dict[str, CacheOutcome] = Field(default_factory=dict)
    provider_identities: dict[str, str] = Field(default_factory=dict)
    degradation_reasons: tuple[str, ...] = ()
    failed_stages: tuple[str, ...] = ()


class RetrievalResult(DomainModel):
    evidence: tuple[RankingEvidence, ...]
    diagnostics: RetrievalDiagnostics
