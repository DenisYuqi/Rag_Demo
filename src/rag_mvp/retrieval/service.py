"""Async dense/hybrid/reranked retrieval orchestration."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rag_mvp.domain.retrieval import (
    CacheOutcome,
    RankingEvidence,
    RetrievalCandidate,
    RetrievalDiagnostics,
    RetrievalMode,
    RetrievalResult,
)
from rag_mvp.retrieval.fusion import RrfConfig, weighted_rrf
from rag_mvp.retrieval.request import RetrievalRequestContext


class CandidateRetriever(Protocol):
    async def search(self, query: str, limit: int) -> tuple[RetrievalCandidate, ...]: ...


class CandidateReranker(Protocol):
    @property
    def identity(self) -> str: ...

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
    ) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class RetrievalLimits:
    dense: int = 20
    lexical: int = 20
    rerank: int = 10
    final: int = 5

    def __post_init__(self) -> None:
        if min(self.dense, self.lexical, self.rerank, self.final) < 1:
            raise ValueError("retrieval limits must be positive")
        if self.final > self.rerank:
            raise ValueError("final limit cannot exceed rerank limit")


class RetrievalUnavailableError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class RetrievalService:
    def __init__(
        self,
        *,
        dense: CandidateRetriever,
        lexical: CandidateRetriever,
        reranker: CandidateReranker | None = None,
        limits: RetrievalLimits | None = None,
        rrf: RrfConfig | None = None,
        rerank_deadline_seconds: float = 1.2,
        allow_single_retriever_degradation: bool = True,
    ) -> None:
        self._dense = dense
        self._lexical = lexical
        self._reranker = reranker
        self._limits = limits or RetrievalLimits()
        self._rrf = rrf or RrfConfig()
        self._rerank_deadline = rerank_deadline_seconds
        self._allow_degradation = allow_single_retriever_degradation

    async def _hybrid(
        self,
        query: str,
    ) -> tuple[tuple[RetrievalCandidate, ...], tuple[str, ...], dict[str, int]]:
        dense_result, lexical_result = await asyncio.gather(
            self._dense.search(query, self._limits.dense),
            self._lexical.search(query, self._limits.lexical),
            return_exceptions=True,
        )
        failures: list[str] = []
        if isinstance(dense_result, BaseException):
            failures.append("dense")
            dense_candidates: tuple[RetrievalCandidate, ...] = ()
        else:
            dense_candidates = dense_result
        if isinstance(lexical_result, BaseException):
            failures.append("bm25")
            lexical_candidates: tuple[RetrievalCandidate, ...] = ()
        else:
            lexical_candidates = lexical_result
        if len(failures) == 2 or (failures and not self._allow_degradation):
            raise RetrievalUnavailableError("retrieval_unavailable")
        return (
            weighted_rrf(dense_candidates, lexical_candidates, config=self._rrf),
            tuple(failures),
            {"dense": len(dense_candidates), "bm25": len(lexical_candidates)},
        )

    async def retrieve(self, context: RetrievalRequestContext) -> RetrievalResult:
        started = time.monotonic()
        failed_stages: tuple[str, ...] = ()
        degradation: list[str] = []
        counts: dict[str, int] = {}
        effective_mode = context.mode

        if context.mode is RetrievalMode.DENSE:
            try:
                candidates = await self._dense.search(context.query, self._limits.dense)
            except Exception as error:
                raise RetrievalUnavailableError("retrieval_unavailable") from error
            counts["dense"] = len(candidates)
        else:
            candidates, failed_stages, counts = await self._hybrid(context.query)
            if failed_stages:
                degradation.extend(f"{stage}_unavailable" for stage in failed_stages)
                effective_mode = RetrievalMode.HYBRID

        if (
            context.mode is RetrievalMode.HYBRID_RERANK
            and candidates
            and self._reranker is not None
            and not failed_stages
        ):
            rerank_candidates = candidates[: self._limits.rerank]
            try:
                ordering = await asyncio.wait_for(
                    self._reranker.rerank(context.query, rerank_candidates),
                    timeout=self._rerank_deadline,
                )
                expected = {candidate.chunk_id for candidate in rerank_candidates}
                if len(ordering) != len(expected) or set(ordering) != expected:
                    raise ValueError("invalid_reranking_permutation")
                positions = {chunk_id: rank for rank, chunk_id in enumerate(ordering, start=1)}
                reranked = [
                    candidate.model_copy(update={"reranking_rank": positions[candidate.chunk_id]})
                    for candidate in rerank_candidates
                ]
                reranked.sort(key=lambda candidate: candidate.reranking_rank or 0)
                candidates = tuple(reranked) + candidates[self._limits.rerank :]
                counts["reranked"] = len(rerank_candidates)
            except (TimeoutError, ValueError):
                degradation.append("rerank_degraded")
                effective_mode = RetrievalMode.HYBRID
        elif context.mode is RetrievalMode.HYBRID_RERANK and self._reranker is None:
            degradation.append("reranker_unavailable")
            effective_mode = RetrievalMode.HYBRID

        evidence = tuple(
            RankingEvidence(**candidate.model_dump(), final_rank=rank)
            for rank, candidate in enumerate(candidates[: self._limits.final], start=1)
        )
        counts["fused"] = len(candidates)
        counts["final"] = len(evidence)
        diagnostics = RetrievalDiagnostics(
            request_id=context.request_id,
            requested_mode=context.mode,
            effective_mode=effective_mode,
            index_revision=context.revision_id,
            candidate_counts=counts,
            stage_timings_ms={"total": (time.monotonic() - started) * 1000},
            cache_status={"final": CacheOutcome.BYPASS},
            provider_identities={
                "reranker": self._reranker.identity if self._reranker is not None else "unavailable"
            },
            degradation_reasons=tuple(degradation),
            failed_stages=failed_stages,
        )
        return RetrievalResult(evidence=evidence, diagnostics=diagnostics)
