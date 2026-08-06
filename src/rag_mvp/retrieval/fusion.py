"""Strict candidate integrity checks and weighted reciprocal-rank fusion."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from pydantic import ValidationError

from rag_mvp.domain.retrieval import RetrievalCandidate


class CandidateIntegrityError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class RrfConfig:
    k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    version: str = "weighted-rrf-v1"

    def __post_init__(self) -> None:
        if type(self.k) is not int or self.k < 1:
            raise ValueError("rrf_k_invalid")
        for name in ("dense_weight", "lexical_weight"):
            value = getattr(self, name)
            if type(value) not in {int, float}:
                raise ValueError(f"rrf_{name}_invalid")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized <= 0:
                raise ValueError(f"rrf_{name}_invalid")
            object.__setattr__(self, name, normalized)
        if type(self.version) is not str or self.version != "weighted-rrf-v1":
            raise ValueError("rrf_version_invalid")


def validate_ranked_channel(
    candidates: object,
    *,
    channel: str,
    expected_revision_id: str | None = None,
    require_complete_identity: bool = False,
    require_positional_ranks: bool = False,
    require_scores: bool = False,
) -> tuple[RetrievalCandidate, ...]:
    """Validate rank provenance and return fresh domain-validated candidates."""

    if channel not in {"dense", "bm25"}:
        raise ValueError("unsupported_retrieval_channel")
    if isinstance(candidates, (str, bytes, bytearray)) or not isinstance(candidates, Sequence):
        raise CandidateIntegrityError(f"{channel}_result_invalid")
    raw_candidates = cast(Sequence[object], candidates)
    validated: list[RetrievalCandidate] = []
    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    for position, raw_candidate in enumerate(raw_candidates, start=1):
        if not isinstance(raw_candidate, RetrievalCandidate):
            raise CandidateIntegrityError(f"{channel}_candidate_invalid")
        raw_rank = raw_candidate.dense_rank if channel == "dense" else raw_candidate.bm25_rank
        if type(raw_rank) is not int or raw_rank < 1:
            raise CandidateIntegrityError(f"{channel}_rank_invalid")
        try:
            candidate = RetrievalCandidate.model_validate(raw_candidate.model_dump())
        except ValidationError:
            raise CandidateIntegrityError(f"{channel}_candidate_invalid") from None
        rank = raw_rank
        score = candidate.dense_score if channel == "dense" else candidate.bm25_score
        other_rank = candidate.bm25_rank if channel == "dense" else candidate.dense_rank
        other_score = candidate.bm25_score if channel == "dense" else candidate.dense_score
        if require_positional_ranks and (rank != position or rank in seen_ranks):
            raise CandidateIntegrityError(f"{channel}_rank_invalid")
        if require_scores and score is None:
            raise CandidateIntegrityError(f"{channel}_score_missing")
        if other_rank is not None or other_score is not None:
            raise CandidateIntegrityError(f"{channel}_provenance_invalid")
        if candidate.rrf_score is not None or candidate.reranking_rank is not None:
            raise CandidateIntegrityError(f"{channel}_stage_fields_invalid")
        if candidate.chunk_id in seen_ids:
            raise CandidateIntegrityError(f"{channel}_duplicate_chunk_id")
        if expected_revision_id is not None and candidate.revision_id != expected_revision_id:
            raise CandidateIntegrityError(f"{channel}_revision_mismatch")
        if require_complete_identity and any(
            value is None
            for value in (
                candidate.revision_id,
                candidate.ordinal,
                candidate.content_digest,
                candidate.record_digest,
            )
        ):
            raise CandidateIntegrityError(f"{channel}_identity_incomplete")
        seen_ids.add(candidate.chunk_id)
        if require_positional_ranks:
            seen_ranks.add(rank)
        validated.append(candidate)
    return tuple(validated)


def merge_ranked_candidates(
    dense: Sequence[RetrievalCandidate],
    lexical: Sequence[RetrievalCandidate],
    *,
    expected_revision_id: str | None = None,
    require_complete_identity: bool = False,
    require_positional_ranks: bool = False,
    require_scores: bool = False,
) -> tuple[RetrievalCandidate, ...]:
    """Join two validated channels by chunk ID without losing zero-valued scores."""

    validated_dense = validate_ranked_channel(
        dense,
        channel="dense",
        expected_revision_id=expected_revision_id,
        require_complete_identity=require_complete_identity,
        require_positional_ranks=require_positional_ranks,
        require_scores=require_scores,
    )
    validated_lexical = validate_ranked_channel(
        lexical,
        channel="bm25",
        expected_revision_id=expected_revision_id,
        require_complete_identity=require_complete_identity,
        require_positional_ranks=require_positional_ranks,
        require_scores=require_scores,
    )
    _validate_revision_coherence(validated_dense, validated_lexical)

    merged: dict[str, RetrievalCandidate] = {
        candidate.chunk_id: candidate for candidate in validated_dense
    }
    for candidate in validated_lexical:
        existing = merged.get(candidate.chunk_id)
        if existing is None:
            merged[candidate.chunk_id] = candidate
            continue
        if _candidate_identity(existing) != _candidate_identity(candidate):
            raise CandidateIntegrityError("chunk_metadata_mismatch")
        merged[candidate.chunk_id] = RetrievalCandidate.model_validate(
            {
                **existing.model_dump(),
                "bm25_rank": candidate.bm25_rank,
                "bm25_score": candidate.bm25_score,
            }
        )
    return tuple(merged.values())


def weighted_rrf(
    dense: Sequence[RetrievalCandidate],
    lexical: Sequence[RetrievalCandidate],
    *,
    config: RrfConfig | None = None,
) -> tuple[RetrievalCandidate, ...]:
    resolved = RrfConfig() if config is None else config
    if not isinstance(resolved, RrfConfig):
        raise TypeError("config must be an RrfConfig")
    merged = merge_ranked_candidates(dense, lexical)

    scored: list[RetrievalCandidate] = []
    for candidate in merged:
        if candidate.dense_rank is None and candidate.bm25_rank is None:
            raise CandidateIntegrityError("candidate_source_rank_missing")
        score = 0.0
        if candidate.dense_rank is not None:
            score += resolved.dense_weight / (resolved.k + candidate.dense_rank)
        if candidate.bm25_rank is not None:
            score += resolved.lexical_weight / (resolved.k + candidate.bm25_rank)
        scored.append(
            RetrievalCandidate.model_validate({**candidate.model_dump(), "rrf_score": score})
        )

    def sort_key(candidate: RetrievalCandidate) -> tuple[float, int, str]:
        if candidate.rrf_score is None:
            raise CandidateIntegrityError("rrf_score_missing")
        ranks = [rank for rank in (candidate.dense_rank, candidate.bm25_rank) if rank is not None]
        if not ranks:
            raise CandidateIntegrityError("candidate_source_rank_missing")
        return (-candidate.rrf_score, min(ranks), candidate.chunk_id)

    return tuple(sorted(scored, key=sort_key))


def _candidate_identity(candidate: RetrievalCandidate) -> tuple[object, ...]:
    return (
        candidate.chunk_id,
        candidate.source_id,
        candidate.display_title,
        candidate.document_version,
        candidate.locator,
        candidate.text,
        candidate.revision_id,
        candidate.ordinal,
        candidate.content_digest,
        candidate.record_digest,
    )


def _validate_revision_coherence(
    dense: tuple[RetrievalCandidate, ...],
    lexical: tuple[RetrievalCandidate, ...],
) -> None:
    revisions = {candidate.revision_id for candidate in (*dense, *lexical)}
    if len(revisions) > 1:
        raise CandidateIntegrityError("mixed_candidate_revisions")
