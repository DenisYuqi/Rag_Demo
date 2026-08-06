"""Revision-bound dense, hybrid, and optional-rerank orchestration."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, Self, cast

from rag_mvp.config.settings import Settings
from rag_mvp.domain.evaluation import TokenUsage as DiagnosticTokenUsage
from rag_mvp.domain.ingestion import DocumentKind
from rag_mvp.domain.retrieval import (
    CacheOutcome,
    RankingEvidence,
    RetrievalCandidate,
    RetrievalDiagnostics,
    RetrievalMode,
    RetrievalResult,
)
from rag_mvp.providers.models import ProviderCallContext, TokenUsage
from rag_mvp.providers.protocols import EmbeddingProvider, RerankingProvider
from rag_mvp.providers.routing import ModelProviderRouter
from rag_mvp.retrieval.binding import BoundRetrievalSnapshot
from rag_mvp.retrieval.collection import (
    BoundBm25Retriever,
    HybridCollectionError,
    collect_hybrid_candidates,
)
from rag_mvp.retrieval.evidence import EvidenceAssembler, assemble_legacy_evidence
from rag_mvp.retrieval.fusion import (
    CandidateIntegrityError,
    RrfConfig,
    validate_ranked_channel,
    weighted_rrf,
)
from rag_mvp.retrieval.query_dense import BoundDenseRetriever
from rag_mvp.retrieval.request import RetrievalRequestContext
from rag_mvp.retrieval.rerank import (
    RerankIntegrityError,
    RerankStage,
    RerankStageResult,
    validate_rerank_stage_result,
)

RRF_TIE_POLICY_VERSION = "rrf-score-best-rank-chunk-id-v1"
DEGRADATION_POLICY_VERSION = "single-normalized-retriever-v1"


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
        for name in ("dense", "lexical", "rerank", "final"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError("retrieval limits must be positive integers")
        if self.final > self.rerank:
            raise ValueError("final limit cannot exceed rerank limit")


class RetrievalUnavailableError(RuntimeError):
    def __init__(self, code: str, *, failed_stages: tuple[str, ...] = ()) -> None:
        super().__init__(code)
        self.code = code
        self.failed_stages = failed_stages


@dataclass(frozen=True, slots=True)
class _RetrievalStages:
    ordered: tuple[RetrievalCandidate, ...]
    dense: tuple[RetrievalCandidate, ...]
    bm25: tuple[RetrievalCandidate, ...]
    fused: tuple[RetrievalCandidate, ...]
    rerank: RerankStageResult | None
    failed_stages: tuple[str, ...]
    degradation_reasons: tuple[str, ...]
    effective_mode: RetrievalMode
    timings: dict[str, float]
    embedding_usage: TokenUsage | None = None
    embedding_attempt_count: int = 0


class RetrievalService:
    """One-request retrieval service bound to either a snapshot or legacy test doubles."""

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
        """Compatibility constructor for existing provenance-free test doubles.

        Production composition must use :meth:`from_snapshot`, which enforces manifest,
        binding-token, source-kind, provider-context, and candidate provenance invariants.
        """

        if not callable(getattr(dense, "search", None)) or not callable(
            getattr(lexical, "search", None)
        ):
            raise TypeError("dense and lexical retrievers must provide async search")
        if type(allow_single_retriever_degradation) is not bool:
            raise ValueError("allow_single_retriever_degradation_invalid")
        if not isinstance(rerank_deadline_seconds, (int, float)) or isinstance(
            rerank_deadline_seconds, bool
        ):
            raise ValueError("rerank_deadline_invalid")
        rerank_budget = float(rerank_deadline_seconds)
        if not math.isfinite(rerank_budget) or rerank_budget <= 0:
            raise ValueError("rerank_deadline_invalid")
        self._dense = dense
        self._lexical = lexical
        self._legacy_reranker = reranker
        self._limits = limits or RetrievalLimits()
        self._rrf = rrf or RrfConfig()
        self._rerank_deadline = rerank_budget
        self._allow_degradation = allow_single_retriever_degradation
        self._snapshot: BoundRetrievalSnapshot | None = None
        self._provider_context: ProviderCallContext | None = None
        self._rerank_stage: RerankStage | None = None
        self._evidence_assembler: EvidenceAssembler | None = None
        self._owns_snapshot = False
        self._closed = False

    @classmethod
    def from_snapshot(
        cls,
        snapshot: BoundRetrievalSnapshot,
        embedding: EmbeddingProvider | ModelProviderRouter,
        provider_context: ProviderCallContext,
        *,
        source_kinds: Mapping[str, DocumentKind | str] | None = None,
        reranker: RerankingProvider | ModelProviderRouter | None = None,
        settings: Settings | None = None,
        limits: RetrievalLimits | None = None,
        rrf: RrfConfig | None = None,
        rerank_deadline_seconds: float | None = None,
        allow_single_retriever_degradation: bool | None = None,
        owns_snapshot: bool = False,
    ) -> RetrievalService:
        """Compose the production service around one validated immutable snapshot."""

        if not isinstance(snapshot, BoundRetrievalSnapshot) or snapshot.is_closed:
            raise ValueError("invalid_snapshot_binding")
        if not isinstance(provider_context, ProviderCallContext):
            raise TypeError("provider_context must be a ProviderCallContext")
        if type(owns_snapshot) is not bool:
            raise ValueError("owns_snapshot_invalid")
        if settings is not None and not isinstance(settings, Settings):
            raise TypeError("settings must be Settings")
        if settings is not None and settings.retrieval_cache_enabled:
            raise ValueError("retrieval_cache_not_implemented")
        if settings is not None and reranker is not None and settings.reranking_model is None:
            raise ValueError("reranking_model_not_configured")
        resolved_source_kinds: Mapping[str, DocumentKind | str] = snapshot.source_kinds
        if source_kinds is not None:
            try:
                supplied_source_kinds = {
                    source_id: DocumentKind(kind) for source_id, kind in source_kinds.items()
                }
            except (AttributeError, TypeError, ValueError):
                raise ValueError("source_kind_mapping_invalid") from None
            if supplied_source_kinds != dict(snapshot.source_kinds):
                raise ValueError("source_kind_mapping_mismatch")
            resolved_source_kinds = supplied_source_kinds

        resolved_limits = limits or (
            RetrievalLimits(
                dense=settings.dense_candidate_limit,
                lexical=settings.lexical_candidate_limit,
                rerank=settings.rerank_candidate_limit,
                final=settings.context_chunk_limit,
            )
            if settings is not None
            else RetrievalLimits()
        )
        resolved_rrf = rrf or (
            RrfConfig(
                k=settings.rrf_k,
                dense_weight=settings.dense_weight,
                lexical_weight=settings.lexical_weight,
            )
            if settings is not None
            else RrfConfig()
        )
        resolved_budget = (
            rerank_deadline_seconds
            if rerank_deadline_seconds is not None
            else settings.rerank_deadline_seconds
            if settings is not None
            else 1.2
        )
        resolved_degradation = (
            allow_single_retriever_degradation
            if allow_single_retriever_degradation is not None
            else settings.allow_single_retriever_degradation
            if settings is not None
            else False
        )
        service = cls(
            dense=BoundDenseRetriever(snapshot, embedding, provider_context),
            lexical=BoundBm25Retriever(snapshot),
            limits=resolved_limits,
            rrf=resolved_rrf,
            rerank_deadline_seconds=resolved_budget,
            allow_single_retriever_degradation=resolved_degradation,
        )
        service._snapshot = snapshot
        service._provider_context = provider_context
        service._legacy_reranker = None
        service._rerank_stage = (
            RerankStage(
                reranker,
                candidate_limit=resolved_limits.rerank,
                budget_seconds=resolved_budget,
            )
            if reranker is not None
            else None
        )
        service._evidence_assembler = EvidenceAssembler(
            snapshot,
            resolved_source_kinds,
            final_limit=resolved_limits.final,
            rrf=resolved_rrf,
        )
        service._owns_snapshot = owns_snapshot
        return service

    @property
    def snapshot(self) -> BoundRetrievalSnapshot | None:
        return self._snapshot

    @property
    def owns_snapshot(self) -> bool:
        return self._owns_snapshot

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def retrieve(self, context: RetrievalRequestContext) -> RetrievalResult:
        if self._closed:
            raise RetrievalUnavailableError("retrieval_service_closed")
        if not isinstance(context, RetrievalRequestContext):
            raise TypeError("context must be a RetrievalRequestContext")
        if self._snapshot is not None:
            context.assert_matches_snapshot(self._snapshot)
            provider_context = self._required_provider_context()
            if provider_context.request_id != context.request_id:
                raise RetrievalUnavailableError("provider_context_mismatch")
        mode = RetrievalMode(context.mode)
        started = time.monotonic()
        try:
            stages = (
                await self._retrieve_bound(context)
                if self._snapshot is not None
                else await self._retrieve_legacy(context)
            )
            evidence = self._assemble_evidence(context, stages)
        except asyncio.CancelledError:
            raise
        except RetrievalUnavailableError:
            raise
        except (CandidateIntegrityError, RerankIntegrityError, ValueError, TypeError):
            raise RetrievalUnavailableError("retrieval_result_invalid") from None

        counts = {
            "dense": len(stages.dense),
            "bm25": len(stages.bm25),
            "fused": len(stages.fused),
            "reranked": (
                stages.rerank.submitted_count
                if stages.rerank is not None and stages.rerank.applied
                else 0
            ),
            "final": len(evidence),
        }
        timings = {
            **stages.timings,
            "total": max(0.0, (time.monotonic() - started) * 1000),
        }
        identities = self._provider_identities(mode, stages.rerank)
        usage: dict[str, DiagnosticTokenUsage] = {}
        if stages.embedding_usage is not None:
            usage["embedding"] = _diagnostic_usage(stages.embedding_usage)
        if stages.rerank is not None and stages.rerank.usage is not None:
            usage["reranker"] = _diagnostic_usage(stages.rerank.usage)
        attempt_counts: dict[str, int] = {}
        if stages.embedding_attempt_count:
            attempt_counts["embedding"] = stages.embedding_attempt_count
        if stages.rerank is not None:
            attempt_counts["reranker"] = len(stages.rerank.attempts)
        diagnostics = RetrievalDiagnostics(
            request_id=context.request_id,
            requested_mode=context.mode,
            effective_mode=stages.effective_mode,
            index_revision=context.revision_id,
            candidate_counts=counts,
            stage_timings_ms=timings,
            cache_status=_absent_cache_status(),
            provider_identities=identities,
            provider_usage=usage,
            provider_attempt_counts=attempt_counts,
            degradation_reasons=stages.degradation_reasons,
            failed_stages=stages.failed_stages,
        )
        return RetrievalResult(evidence=evidence, diagnostics=diagnostics)

    async def _retrieve_bound(self, context: RetrievalRequestContext) -> _RetrievalStages:
        mode = RetrievalMode(context.mode)
        if mode is RetrievalMode.DENSE:
            started = time.monotonic()
            try:
                detailed = await cast(BoundDenseRetriever, self._dense).search_with_diagnostics(
                    context.query,
                    self._limits.dense,
                )
                dense = _validate_production_channel(
                    detailed.candidates,
                    limit=self._limits.dense,
                    channel="dense",
                    revision_id=context.revision_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                raise RetrievalUnavailableError(
                    "retrieval_unavailable",
                    failed_stages=("dense",),
                ) from None
            return _RetrievalStages(
                ordered=dense,
                dense=dense,
                bm25=(),
                fused=(),
                rerank=None,
                failed_stages=(),
                degradation_reasons=(),
                effective_mode=RetrievalMode.DENSE,
                timings={
                    "query_embedding": detailed.embedding_elapsed_ms,
                    "dense": detailed.index_elapsed_ms,
                    "bm25": 0.0,
                    "fusion": 0.0,
                    "rerank": 0.0,
                    "retrieval": max(0.0, (time.monotonic() - started) * 1000),
                },
                embedding_usage=detailed.usage,
                embedding_attempt_count=len(detailed.attempts),
            )

        try:
            collection = await collect_hybrid_candidates(
                context.query,
                dense=cast(BoundDenseRetriever, self._dense),
                bm25=cast(BoundBm25Retriever, self._lexical),
                dense_limit=self._limits.dense,
                bm25_limit=self._limits.lexical,
                allow_single_retriever_degradation=self._allow_degradation,
            )
        except asyncio.CancelledError:
            raise
        except HybridCollectionError as error:
            raise RetrievalUnavailableError(
                "retrieval_unavailable",
                failed_stages=error.failed_stages,
            ) from None
        except Exception:
            raise RetrievalUnavailableError("retrieval_result_invalid") from None

        fusion_started = time.monotonic()
        try:
            fused = weighted_rrf(collection.dense, collection.bm25, config=self._rrf)
        except (CandidateIntegrityError, TypeError, ValueError):
            raise RetrievalUnavailableError("retrieval_result_invalid") from None
        fusion_ms = max(0.0, (time.monotonic() - fusion_started) * 1000)
        degradation = tuple(f"{stage}_unavailable" for stage in collection.failed_stages)
        effective_mode = mode
        rerank_result: RerankStageResult | None = None
        rerank_attempted = False
        ordered = fused
        rerank_ms = 0.0

        if mode is RetrievalMode.HYBRID_RERANK:
            if collection.failed_stages:
                effective_mode = RetrievalMode.HYBRID
                degradation += ("rerank_skipped_retriever_degraded",)
            elif self._rerank_stage is None:
                effective_mode = RetrievalMode.HYBRID
                degradation += ("reranker_not_configured",)
            elif fused:
                rerank_attempted = True
                rerank_result = await self._rerank_stage.run(
                    context.query,
                    fused,
                    self._required_provider_context(),
                )
                rerank_ms = rerank_result.elapsed_ms
                try:
                    ordered = validate_rerank_stage_result(fused, rerank_result)
                except RerankIntegrityError:
                    ordered = fused
                    rerank_result = None
                    effective_mode = RetrievalMode.HYBRID
                    degradation += ("rerank_invalid_stage_result",)
                else:
                    if rerank_result.degraded or not rerank_result.applied:
                        effective_mode = RetrievalMode.HYBRID
                        degradation += (rerank_result.reason or "rerank_not_applied",)

        failed_stages = collection.failed_stages
        if rerank_attempted and effective_mode is RetrievalMode.HYBRID:
            failed_stages += ("rerank",)

        return _RetrievalStages(
            ordered=ordered,
            dense=collection.dense,
            bm25=collection.bm25,
            fused=fused,
            rerank=rerank_result,
            failed_stages=failed_stages,
            degradation_reasons=degradation,
            effective_mode=effective_mode,
            timings={
                "query_embedding": collection.embedding_elapsed_ms,
                "dense": collection.dense_elapsed_ms,
                "bm25": collection.bm25_elapsed_ms,
                "fusion": fusion_ms,
                "rerank": rerank_ms,
            },
            embedding_usage=collection.embedding_usage,
            embedding_attempt_count=len(collection.embedding_attempts),
        )

    async def _retrieve_legacy(self, context: RetrievalRequestContext) -> _RetrievalStages:
        mode = RetrievalMode(context.mode)
        timings = {
            "query_embedding": 0.0,
            "dense": 0.0,
            "bm25": 0.0,
            "fusion": 0.0,
            "rerank": 0.0,
        }
        if mode is RetrievalMode.DENSE:
            started = time.monotonic()
            try:
                raw_dense = await self._dense.search(context.query, self._limits.dense)
                dense = _validate_legacy_channel(
                    raw_dense,
                    limit=self._limits.dense,
                    channel="dense",
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                raise RetrievalUnavailableError(
                    "retrieval_unavailable",
                    failed_stages=("dense",),
                ) from None
            timings["dense"] = max(0.0, (time.monotonic() - started) * 1000)
            return _RetrievalStages(
                dense,
                dense,
                (),
                (),
                None,
                (),
                (),
                RetrievalMode.DENSE,
                timings,
            )

        dense_started = time.monotonic()
        dense_result, lexical_result = await asyncio.gather(
            self._dense.search(context.query, self._limits.dense),
            self._lexical.search(context.query, self._limits.lexical),
            return_exceptions=True,
        )
        for result in (dense_result, lexical_result):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, BaseException) and not isinstance(result, Exception):
                raise result
        timings["dense"] = max(0.0, (time.monotonic() - dense_started) * 1000)
        failures = tuple(
            stage
            for stage, result in (("dense", dense_result), ("bm25", lexical_result))
            if isinstance(result, Exception)
        )
        if len(failures) == 2 or (failures and not self._allow_degradation):
            raise RetrievalUnavailableError(
                "retrieval_unavailable",
                failed_stages=failures,
            )
        dense = (
            ()
            if isinstance(dense_result, Exception)
            else _validate_legacy_channel(
                dense_result,
                limit=self._limits.dense,
                channel="dense",
            )
        )
        bm25 = (
            ()
            if isinstance(lexical_result, Exception)
            else _validate_legacy_channel(
                lexical_result,
                limit=self._limits.lexical,
                channel="bm25",
            )
        )
        fusion_started = time.monotonic()
        fused = weighted_rrf(dense, bm25, config=self._rrf)
        timings["fusion"] = max(0.0, (time.monotonic() - fusion_started) * 1000)
        degradation = tuple(f"{stage}_unavailable" for stage in failures)
        effective_mode = mode
        ordered = fused
        if mode is RetrievalMode.HYBRID_RERANK:
            if failures:
                effective_mode = RetrievalMode.HYBRID
                degradation += ("rerank_skipped_retriever_degraded",)
            elif self._legacy_reranker is None:
                effective_mode = RetrievalMode.HYBRID
                degradation += ("reranker_not_configured",)
            elif fused:
                rerank_started = time.monotonic()
                submitted = fused[: self._limits.rerank]
                try:
                    async with asyncio.timeout(self._rerank_deadline):
                        raw_order = await self._legacy_reranker.rerank(
                            context.query,
                            submitted,
                        )
                    order = _validate_legacy_order(raw_order, submitted)
                    registry = {candidate.chunk_id: candidate for candidate in submitted}
                    ordered_prefix = tuple(
                        RetrievalCandidate.model_validate(
                            {
                                **registry[candidate_id].model_dump(),
                                "reranking_rank": rank,
                            }
                        )
                        for rank, candidate_id in enumerate(order, start=1)
                    )
                    ordered = ordered_prefix + fused[len(submitted) :]
                except asyncio.CancelledError:
                    raise
                except Exception:
                    ordered = fused
                    effective_mode = RetrievalMode.HYBRID
                    degradation += ("rerank_degraded",)
                timings["rerank"] = max(
                    0.0,
                    (time.monotonic() - rerank_started) * 1000,
                )
        return _RetrievalStages(
            ordered,
            dense,
            bm25,
            fused,
            None,
            failures,
            degradation,
            effective_mode,
            timings,
        )

    def _assemble_evidence(
        self,
        context: RetrievalRequestContext,
        stages: _RetrievalStages,
    ) -> tuple[RankingEvidence, ...]:
        final_candidates = stages.ordered[: self._limits.final]
        if self._evidence_assembler is None:
            return assemble_legacy_evidence(
                final_candidates,
                final_limit=self._limits.final,
            )
        return self._evidence_assembler.assemble(
            final_candidates,
            mode=stages.effective_mode,
            dense_candidates=stages.dense,
            bm25_candidates=stages.bm25,
            fused_candidates=stages.fused,
            rerank_result=(
                stages.rerank if stages.rerank is not None and stages.rerank.applied else None
            ),
        )

    def _provider_identities(
        self,
        mode: RetrievalMode,
        rerank: RerankStageResult | None,
    ) -> dict[str, str]:
        if self._snapshot is None:
            identities: dict[str, str] = {}
            if self._legacy_reranker is not None:
                try:
                    identity = self._legacy_reranker.identity
                except Exception:
                    identity = None
                if isinstance(identity, str) and identity:
                    identities["reranker"] = identity
            return identities
        revision = self._snapshot.revision
        embedding = revision.embedding_space
        identities = {
            "embedding": _canonical_identity(
                {
                    "provider": embedding.provider_alias,
                    "model": embedding.model,
                    "dimension": embedding.dimension,
                    "normalization": embedding.normalization,
                    "adapter": embedding.adapter_version,
                }
            ),
            "dense": _canonical_identity(
                {
                    "schema": revision.dense_schema_version,
                    "metric": revision.dense_metric,
                }
            ),
        }
        if mode is not RetrievalMode.DENSE:
            identities["bm25"] = _canonical_identity(
                {
                    "tokenizer": revision.tokenizer_version,
                    "schema": revision.lexical_schema_version,
                    "algorithm": revision.lexical_algorithm_version,
                    "k1": revision.lexical_k1,
                    "b": revision.lexical_b,
                }
            )
            identities["rrf"] = _canonical_identity(
                {
                    "version": self._rrf.version,
                    "k": self._rrf.k,
                    "dense_weight": self._rrf.dense_weight,
                    "lexical_weight": self._rrf.lexical_weight,
                    "tie_policy": RRF_TIE_POLICY_VERSION,
                }
            )
        if rerank is not None and rerank.provider_identity is not None:
            identities["reranker"] = rerank.provider_identity
        return identities

    def _required_provider_context(self) -> ProviderCallContext:
        if self._provider_context is None:
            raise RetrievalUnavailableError("provider_context_missing")
        return self._provider_context

    def close(self) -> None:
        if self._closed:
            return
        if self._owns_snapshot and self._snapshot is not None:
            self._snapshot.close()
        self._closed = True

    def __enter__(self) -> Self:
        if self._closed:
            raise RetrievalUnavailableError("retrieval_service_closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


def _validate_production_channel(
    result: object,
    *,
    limit: int,
    channel: str,
    revision_id: str,
) -> tuple[RetrievalCandidate, ...]:
    bounded = _bounded_candidates(result, limit, channel)
    return validate_ranked_channel(
        bounded,
        channel=channel,
        expected_revision_id=revision_id,
        require_complete_identity=True,
        require_positional_ranks=True,
        require_scores=True,
    )


def _validate_legacy_channel(
    result: object,
    *,
    limit: int,
    channel: str,
) -> tuple[RetrievalCandidate, ...]:
    bounded = _bounded_candidates(result, limit, channel)
    return validate_ranked_channel(bounded, channel=channel)


def _bounded_candidates(
    result: object,
    limit: int,
    channel: str,
) -> tuple[RetrievalCandidate, ...]:
    if isinstance(result, (str, bytes, bytearray)) or not isinstance(result, Sequence):
        raise CandidateIntegrityError(f"{channel}_result_invalid")
    values = cast(Sequence[object], result)
    bounded = tuple(values[:limit])
    if any(not isinstance(candidate, RetrievalCandidate) for candidate in bounded):
        raise CandidateIntegrityError(f"{channel}_candidate_invalid")
    return cast(tuple[RetrievalCandidate, ...], bounded)


def _validate_legacy_order(
    order: object,
    candidates: Sequence[RetrievalCandidate],
) -> tuple[str, ...]:
    if isinstance(order, (str, bytes, bytearray)) or not isinstance(order, Sequence):
        raise ValueError("invalid_reranking_permutation")
    if any(type(candidate_id) is not str for candidate_id in order):
        raise ValueError("invalid_reranking_permutation")
    normalized = tuple(cast(Sequence[str], order))
    expected = tuple(candidate.chunk_id for candidate in candidates)
    if (
        len(normalized) != len(expected)
        or len(set(normalized)) != len(normalized)
        or set(normalized) != set(expected)
    ):
        raise ValueError("invalid_reranking_permutation")
    return normalized


def _diagnostic_usage(usage: TokenUsage) -> DiagnosticTokenUsage:
    return DiagnosticTokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens_reported=usage.total_tokens,
    )


def _absent_cache_status() -> dict[str, CacheOutcome]:
    return {
        "query_embedding": CacheOutcome.NOT_APPLICABLE,
        "retrieval": CacheOutcome.NOT_APPLICABLE,
        "rerank": CacheOutcome.NOT_APPLICABLE,
        "final": CacheOutcome.NOT_APPLICABLE,
    }


def _canonical_identity(value: Mapping[str, object]) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
