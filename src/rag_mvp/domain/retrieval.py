"""Retrieval and ranking domain contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field, StringConstraints, field_validator

from rag_mvp.domain._base import (
    Digest,
    DomainModel,
    FiniteFloat,
    Identifier,
    NonNegativeFiniteFloat,
)
from rag_mvp.domain.evaluation import TokenUsage
from rag_mvp.domain.ingestion import ChunkLocator

RETRIEVAL_EVIDENCE_SCHEMA_VERSION = "ranking-evidence-v1"
RETRIEVAL_RESULT_SCHEMA_VERSION = "retrieval-result-v1"
SafeDiagnosticCode = Annotated[
    str,
    Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_.-]*$"),
]


class RetrievalMode(StrEnum):
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid-rerank"


class CacheOutcome(StrEnum):
    HIT = "hit"
    MISS = "miss"
    BYPASS = "bypass"
    NOT_APPLICABLE = "not-applicable"


class CachePolicy(StrEnum):
    USE = "use"
    BYPASS = "bypass"


class RetrievalCandidate(DomainModel):
    chunk_id: Identifier
    source_id: Identifier
    display_title: Identifier
    document_version: Annotated[int, Field(gt=0)]
    locator: ChunkLocator
    text: Annotated[str, StringConstraints(strip_whitespace=False, min_length=1)]
    revision_id: Identifier | None = None
    ordinal: Annotated[int, Field(ge=0)] | None = None
    content_digest: Digest | None = None
    record_digest: Digest | None = None
    dense_rank: Annotated[int, Field(gt=0)] | None = None
    dense_score: FiniteFloat | None = None
    bm25_rank: Annotated[int, Field(gt=0)] | None = None
    bm25_score: FiniteFloat | None = None
    rrf_score: NonNegativeFiniteFloat | None = None
    reranking_rank: Annotated[int, Field(gt=0)] | None = None

    @field_validator("text")
    @classmethod
    def text_has_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("candidate text must not be blank")
        return value

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
    provider_usage: dict[str, TokenUsage] = Field(default_factory=dict)
    provider_attempt_counts: dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    degradation_reasons: Annotated[tuple[SafeDiagnosticCode, ...], Field(max_length=16)] = ()
    failed_stages: Annotated[tuple[SafeDiagnosticCode, ...], Field(max_length=8)] = ()


class RetrievalResult(DomainModel):
    evidence: tuple[RankingEvidence, ...]
    diagnostics: RetrievalDiagnostics
