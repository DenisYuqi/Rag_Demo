"""Candidate integrity checks and weighted reciprocal-rank fusion."""

from __future__ import annotations

from dataclasses import dataclass

from rag_mvp.domain.retrieval import RetrievalCandidate


class CandidateIntegrityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RrfConfig:
    k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    version: str = "weighted-rrf-v1"

    def __post_init__(self) -> None:
        if self.k < 1 or self.dense_weight <= 0 or self.lexical_weight <= 0:
            raise ValueError("RRF parameters must be positive")


def _merge_pair(
    existing: RetrievalCandidate,
    incoming: RetrievalCandidate,
) -> RetrievalCandidate:
    identity_fields = ("source_id", "display_title", "document_version", "locator", "text")
    if any(getattr(existing, field) != getattr(incoming, field) for field in identity_fields):
        raise CandidateIntegrityError("chunk_metadata_mismatch")
    return existing.model_copy(
        update={
            "dense_rank": existing.dense_rank or incoming.dense_rank,
            "dense_score": existing.dense_score
            if existing.dense_score is not None
            else incoming.dense_score,
            "bm25_rank": existing.bm25_rank or incoming.bm25_rank,
            "bm25_score": existing.bm25_score
            if existing.bm25_score is not None
            else incoming.bm25_score,
        }
    )


def weighted_rrf(
    dense: tuple[RetrievalCandidate, ...],
    lexical: tuple[RetrievalCandidate, ...],
    *,
    config: RrfConfig | None = None,
) -> tuple[RetrievalCandidate, ...]:
    resolved = config or RrfConfig()
    merged: dict[str, RetrievalCandidate] = {}
    for candidate in (*dense, *lexical):
        merged[candidate.chunk_id] = (
            _merge_pair(merged[candidate.chunk_id], candidate)
            if candidate.chunk_id in merged
            else candidate
        )

    scored: list[RetrievalCandidate] = []
    for candidate in merged.values():
        score = 0.0
        if candidate.dense_rank is not None:
            score += resolved.dense_weight / (resolved.k + candidate.dense_rank)
        if candidate.bm25_rank is not None:
            score += resolved.lexical_weight / (resolved.k + candidate.bm25_rank)
        scored.append(candidate.model_copy(update={"rrf_score": score}))

    def sort_key(candidate: RetrievalCandidate) -> tuple[float, int, str]:
        ranks = [rank for rank in (candidate.dense_rank, candidate.bm25_rank) if rank]
        return (-(candidate.rrf_score or 0.0), min(ranks), candidate.chunk_id)

    return tuple(sorted(scored, key=sort_key))
