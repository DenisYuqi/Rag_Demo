"""Strict parallel collection from revision-bound dense and BM25 retrievers."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from rag_mvp.domain.retrieval import RetrievalCandidate
from rag_mvp.performance.worker_pools import BoundedWorkerPool
from rag_mvp.providers.errors import ProviderError, ProviderOperationError
from rag_mvp.providers.models import ModelAttempt, TokenUsage
from rag_mvp.retrieval.bm25 import (
    LexicalIndexError,
    default_bm25_worker_pool,
)
from rag_mvp.retrieval.dense import DenseIndexError
from rag_mvp.retrieval.fusion import merge_ranked_candidates, validate_ranked_channel
from rag_mvp.retrieval.query_dense import DenseSearchResult
from rag_mvp.retrieval.request import canonicalize_query


class RevisionBoundRetriever(Protocol):
    @property
    def revision_id(self) -> str: ...

    async def search(self, query: str, limit: int) -> tuple[RetrievalCandidate, ...]: ...


@dataclass(frozen=True, slots=True)
class HybridCandidateCollection:
    revision_id: str
    dense: tuple[RetrievalCandidate, ...]
    bm25: tuple[RetrievalCandidate, ...]
    merged: tuple[RetrievalCandidate, ...]
    failed_stages: tuple[str, ...] = ()
    embedding_elapsed_ms: float = 0.0
    dense_elapsed_ms: float = 0.0
    bm25_elapsed_ms: float = 0.0
    embedding_attempts: tuple[ModelAttempt, ...] = ()
    embedding_usage: TokenUsage | None = None


class HybridCollectionError(RuntimeError):
    def __init__(self, code: str, failed_stages: tuple[str, ...]) -> None:
        self.code = code
        self.failed_stages = failed_stages
        super().__init__(code)


class _TimedChannelError(Exception):
    def __init__(self, cause: Exception, elapsed_ms: float) -> None:
        self.cause = cause
        self.elapsed_ms = elapsed_ms
        super().__init__("retriever_failed")


class BoundBm25Retriever:
    """Expose the captured BM25 snapshot through the revision-bound protocol."""

    def __init__(
        self,
        snapshot: object,
        *,
        worker_pool: BoundedWorkerPool | None = None,
    ) -> None:
        from rag_mvp.retrieval.binding import BoundRetrievalSnapshot

        if not isinstance(snapshot, BoundRetrievalSnapshot) or snapshot.is_closed:
            raise ValueError("invalid_snapshot_binding")
        self._snapshot = snapshot
        self._worker_pool = worker_pool or default_bm25_worker_pool()
        self._snapshot.bm25.configure_worker_pool(self._worker_pool)

    @property
    def revision_id(self) -> str:
        return self._snapshot.revision_id

    async def search(self, query: str, limit: int) -> tuple[RetrievalCandidate, ...]:
        if self._snapshot.is_closed:
            raise ValueError("snapshot_closed")
        return await self._snapshot.bm25.search(query, limit)


async def collect_hybrid_candidates(
    query: str,
    *,
    dense: RevisionBoundRetriever,
    bm25: RevisionBoundRetriever,
    dense_limit: int,
    bm25_limit: int,
    allow_single_retriever_degradation: bool = False,
) -> HybridCandidateCollection:
    """Collect both channels concurrently and fail on any provenance inconsistency."""

    canonical_query = canonicalize_query(query)
    _validate_limit(dense_limit, "dense_limit")
    _validate_limit(bm25_limit, "bm25_limit")
    dense_revision = _retriever_revision(dense, "dense")
    bm25_revision = _retriever_revision(bm25, "bm25")
    if dense_revision != bm25_revision:
        raise ValueError("mixed_retriever_revisions")

    if type(allow_single_retriever_degradation) is not bool:
        raise ValueError("allow_single_retriever_degradation_invalid")
    dense_result, bm25_result = await asyncio.gather(
        _search_dense(dense, canonical_query, dense_limit),
        _search_channel(bm25, canonical_query, bm25_limit),
        return_exceptions=True,
    )
    for result in (dense_result, bm25_result):
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, BaseException) and not isinstance(result, Exception):
            raise result
    dense_value = cast(DenseSearchResult | _TimedChannelError, dense_result)
    bm25_value = cast(
        tuple[tuple[RetrievalCandidate, ...], float] | _TimedChannelError,
        bm25_result,
    )
    failed_stages = tuple(
        stage
        for stage, result in (("dense", dense_value), ("bm25", bm25_value))
        if isinstance(result, Exception)
    )
    for stage, result in (("dense", dense_value), ("bm25", bm25_value)):
        if isinstance(result, _TimedChannelError) and not _normalized_failure(
            stage,
            result.cause,
        ):
            raise HybridCollectionError("retrieval_unavailable", failed_stages) from None
    if len(failed_stages) == 2 or (failed_stages and not allow_single_retriever_degradation):
        raise HybridCollectionError("retrieval_unavailable", failed_stages)
    if (
        _retriever_revision(dense, "dense") != dense_revision
        or _retriever_revision(bm25, "bm25") != dense_revision
    ):
        raise ValueError("retriever_revision_changed")
    bounded_dense = (
        ()
        if isinstance(dense_value, _TimedChannelError)
        else _bounded_result(dense_value.candidates, dense_limit, "dense")
    )
    bounded_bm25 = (
        ()
        if isinstance(bm25_value, _TimedChannelError)
        else _bounded_result(bm25_value[0], bm25_limit, "bm25")
    )
    validated_dense = validate_ranked_channel(
        bounded_dense,
        channel="dense",
        expected_revision_id=dense_revision,
        require_complete_identity=True,
        require_positional_ranks=True,
        require_scores=True,
    )
    validated_bm25 = validate_ranked_channel(
        bounded_bm25,
        channel="bm25",
        expected_revision_id=dense_revision,
        require_complete_identity=True,
        require_positional_ranks=True,
        require_scores=True,
    )
    merged = merge_ranked_candidates(
        validated_dense,
        validated_bm25,
        expected_revision_id=dense_revision,
        require_complete_identity=True,
        require_positional_ranks=True,
        require_scores=True,
    )
    return HybridCandidateCollection(
        revision_id=dense_revision,
        dense=validated_dense,
        bm25=validated_bm25,
        merged=merged,
        failed_stages=failed_stages,
        embedding_elapsed_ms=(
            0.0 if isinstance(dense_value, _TimedChannelError) else dense_value.embedding_elapsed_ms
        ),
        dense_elapsed_ms=(
            dense_value.elapsed_ms
            if isinstance(dense_value, _TimedChannelError)
            else dense_value.index_elapsed_ms
        ),
        bm25_elapsed_ms=(
            bm25_value.elapsed_ms if isinstance(bm25_value, _TimedChannelError) else bm25_value[1]
        ),
        embedding_attempts=(
            () if isinstance(dense_value, _TimedChannelError) else dense_value.attempts
        ),
        embedding_usage=(
            None if isinstance(dense_value, _TimedChannelError) else dense_value.usage
        ),
    )


async def _search_dense(
    retriever: RevisionBoundRetriever,
    query: str,
    limit: int,
) -> DenseSearchResult:
    started = time.monotonic()
    detailed = getattr(retriever, "search_with_diagnostics", None)
    try:
        if callable(detailed):
            result = await detailed(query, limit)
            if not isinstance(result, DenseSearchResult):
                raise ValueError("dense_diagnostics_invalid")
            return result
        candidates = await retriever.search(query, limit)
        return DenseSearchResult(
            candidates=candidates,
            embedding_elapsed_ms=0.0,
            index_elapsed_ms=max(0.0, (time.monotonic() - started) * 1000),
            attempts=(),
            usage=TokenUsage(),
        )
    except asyncio.CancelledError:
        raise
    except Exception as error:
        raise _TimedChannelError(
            error,
            max(0.0, (time.monotonic() - started) * 1000),
        ) from None


async def _search_channel(
    retriever: RevisionBoundRetriever,
    query: str,
    limit: int,
) -> tuple[tuple[RetrievalCandidate, ...], float]:
    started = time.monotonic()
    try:
        candidates = await retriever.search(query, limit)
        return candidates, max(0.0, (time.monotonic() - started) * 1000)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        raise _TimedChannelError(
            error,
            max(0.0, (time.monotonic() - started) * 1000),
        ) from None


def _normalized_failure(stage: str, error: Exception) -> bool:
    if stage == "dense":
        return isinstance(error, (DenseIndexError, ProviderError, ProviderOperationError))
    return isinstance(error, LexicalIndexError)


def _bounded_result(
    result: object,
    limit: int,
    channel: str,
) -> tuple[RetrievalCandidate, ...]:
    if isinstance(result, (str, bytes, bytearray)) or not isinstance(result, Sequence):
        raise ValueError(f"{channel}_result_invalid")
    values = cast(Sequence[object], result)
    bounded = tuple(values[:limit])
    if any(not isinstance(candidate, RetrievalCandidate) for candidate in bounded):
        raise ValueError(f"{channel}_result_invalid")
    return cast(tuple[RetrievalCandidate, ...], bounded)


def _retriever_revision(retriever: RevisionBoundRetriever, channel: str) -> str:
    try:
        revision_id = retriever.revision_id
    except Exception:
        raise ValueError(f"{channel}_revision_invalid") from None
    if not isinstance(revision_id, str) or not revision_id:
        raise ValueError(f"{channel}_revision_invalid")
    return revision_id


def _validate_limit(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name}_invalid")
