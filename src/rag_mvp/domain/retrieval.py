"""Retrieval and ranking domain contracts."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, StringConstraints, field_validator, model_validator

from rag_mvp.domain._base import (
    Digest,
    DomainModel,
    FiniteFloat,
    Identifier,
    NonNegativeFiniteFloat,
)
from rag_mvp.domain.evaluation import ProviderAttemptEvidence, TokenUsage
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
    ERROR = "error"
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
    provider_failed_attempt_counts: dict[str, Annotated[int, Field(ge=0)]] = Field(
        default_factory=dict
    )
    provider_unknown_usage_attempt_counts: dict[str, Annotated[int, Field(ge=0)]] = Field(
        default_factory=dict
    )
    provider_attempts: tuple[ProviderAttemptEvidence, ...] = ()
    pre_rerank_chunk_ids: tuple[Identifier, ...] = ()
    post_rerank_chunk_ids: tuple[Identifier, ...] = ()
    degradation_reasons: Annotated[tuple[SafeDiagnosticCode, ...], Field(max_length=16)] = ()
    failed_stages: Annotated[tuple[SafeDiagnosticCode, ...], Field(max_length=8)] = ()

    @model_validator(mode="after")
    def validate_provider_attempt_counts(self) -> Self:
        for label, counts in (
            ("failed", self.provider_failed_attempt_counts),
            ("unknown usage", self.provider_unknown_usage_attempt_counts),
        ):
            if any(
                role not in self.provider_attempt_counts
                or count > self.provider_attempt_counts[role]
                for role, count in counts.items()
            ):
                raise ValueError(f"{label} provider attempts exceed the attempt ledger")
        for label, chunk_ids in (
            ("pre-rerank", self.pre_rerank_chunk_ids),
            ("post-rerank", self.post_rerank_chunk_ids),
        ):
            if len(chunk_ids) != len(set(chunk_ids)):
                raise ValueError(f"{label} chunk identifiers must be unique")
        if bool(self.pre_rerank_chunk_ids) != bool(self.post_rerank_chunk_ids):
            raise ValueError("pre/post-rerank evidence must be present together")
        if self.pre_rerank_chunk_ids and set(self.pre_rerank_chunk_ids) != set(
            self.post_rerank_chunk_ids
        ):
            raise ValueError("pre/post-rerank evidence must contain the same candidates")
        return self


class RetrievalResult(DomainModel):
    evidence: tuple[RankingEvidence, ...]
    diagnostics: RetrievalDiagnostics


def retrieval_evidence_digest(result: RetrievalResult) -> str:
    """Hash deterministic validated retrieval output without retaining query or text."""

    if not isinstance(result, RetrievalResult):
        raise TypeError("retrieval result is required")
    diagnostics = result.diagnostics
    payload = {
        "schema_version": "retrieval-evidence-digest-v1",
        "index_revision": diagnostics.index_revision,
        "requested_mode": diagnostics.requested_mode.value,
        "effective_mode": diagnostics.effective_mode.value,
        "candidate_counts": diagnostics.candidate_counts,
        "pre_rerank_chunk_ids": diagnostics.pre_rerank_chunk_ids,
        "post_rerank_chunk_ids": diagnostics.post_rerank_chunk_ids,
        "evidence": [item.model_dump(mode="json") for item in result.evidence],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"
