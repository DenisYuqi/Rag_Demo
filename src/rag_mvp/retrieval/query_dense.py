"""Compatible query embedding and dense retrieval bound to one snapshot."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Protocol, cast

from rag_mvp.domain.retrieval import RetrievalCandidate
from rag_mvp.providers.errors import ProviderOperationError
from rag_mvp.providers.models import (
    EmbeddingRequest,
    EmbeddingResult,
    ModelAttempt,
    ProviderCallContext,
    ProviderErrorCategory,
    RoutedResult,
    TokenUsage,
)
from rag_mvp.providers.protocols import EmbeddingProvider
from rag_mvp.providers.routing import ModelProviderRouter
from rag_mvp.retrieval.binding import BoundRetrievalSnapshot
from rag_mvp.retrieval.dense import DenseIndexError
from rag_mvp.retrieval.identity import (
    EmbeddingIdentityError,
    domain_embedding_identity,
    provider_embedding_identity,
)
from rag_mvp.retrieval.request import canonicalize_query


@dataclass(frozen=True, slots=True)
class DenseSearchResult:
    candidates: tuple[RetrievalCandidate, ...]
    embedding_elapsed_ms: float
    index_elapsed_ms: float
    attempts: tuple[ModelAttempt, ...]
    usage: TokenUsage


class BoundDenseRetriever:
    """Embed one canonical query and search only the captured Chroma revision."""

    def __init__(
        self,
        snapshot: BoundRetrievalSnapshot,
        embedding: EmbeddingProvider | ModelProviderRouter,
        context: ProviderCallContext,
    ) -> None:
        if not isinstance(snapshot, BoundRetrievalSnapshot) or snapshot.is_closed:
            raise DenseIndexError("invalid_snapshot_binding")
        if not isinstance(context, ProviderCallContext):
            raise TypeError("context must be a ProviderCallContext")
        self._snapshot = snapshot
        self._embedding = embedding
        self._context = context
        try:
            self._required_provider_identity = provider_embedding_identity(
                snapshot.revision.embedding_space
            )
        except EmbeddingIdentityError:
            raise DenseIndexError("embedding_identity_invalid") from None

        if not isinstance(embedding, ModelProviderRouter):
            try:
                identity = embedding.identity
            except Exception:
                raise DenseIndexError("embedding_identity_invalid") from None
            try:
                persisted_identity = domain_embedding_identity(identity)
            except EmbeddingIdentityError:
                raise DenseIndexError("embedding_identity_invalid") from None
            if persisted_identity != snapshot.revision.embedding_space:
                raise DenseIndexError("embedding_identity_mismatch")

    @property
    def revision_id(self) -> str:
        return self._snapshot.revision_id

    async def search(self, query: str, limit: int) -> tuple[RetrievalCandidate, ...]:
        return (await self.search_with_diagnostics(query, limit)).candidates

    async def search_with_diagnostics(self, query: str, limit: int) -> DenseSearchResult:
        if self._snapshot.is_closed:
            raise DenseIndexError("dense_index_closed")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be positive")
        if self._context.deadline.expired:
            raise DenseIndexError("embedding_deadline_exceeded")
        canonical_query = canonicalize_query(query)
        if not isinstance(self._embedding, ModelProviderRouter):
            try:
                current_identity = domain_embedding_identity(self._embedding.identity)
            except Exception:
                raise DenseIndexError("embedding_identity_invalid") from None
            if current_identity != self._snapshot.revision.embedding_space:
                raise DenseIndexError("embedding_identity_mismatch")
        request = EmbeddingRequest((canonical_query,))
        embedding_started = self._context.deadline.clock()
        attempts: tuple[ModelAttempt, ...] = ()
        try:
            remaining = self._context.deadline.remaining_seconds
            if remaining <= 0:
                raise DenseIndexError("embedding_deadline_exceeded")
            async with asyncio.timeout(remaining):
                if isinstance(self._embedding, ModelProviderRouter):
                    routed = await self._embedding.embed(
                        request,
                        self._context,
                        required_space=self._required_provider_identity,
                    )
                    result = _routed_embedding_result(routed)
                    attempts = routed.attempts
                else:
                    result = await self._embedding.embed(request, self._context)
        except DenseIndexError:
            raise
        except TimeoutError:
            raise DenseIndexError("embedding_deadline_exceeded") from None
        except ProviderOperationError as error:
            if error.category is ProviderErrorCategory.INCOMPATIBLE_RESPONSE:
                raise DenseIndexError("embedding_identity_mismatch") from None
            raise DenseIndexError("query_embedding_failed") from None
        except Exception:
            raise DenseIndexError("query_embedding_failed") from None

        vector = _validated_query_vector(
            result,
            expected_identity=self._required_provider_identity,
        )
        usage = getattr(result, "usage", None)
        if not isinstance(usage, TokenUsage):
            raise DenseIndexError("query_embedding_result_invalid")
        embedding_elapsed_ms = max(
            0.0,
            (self._context.deadline.clock() - embedding_started) * 1000,
        )
        index_started = self._context.deadline.clock()
        candidates = self._snapshot.dense.query(
            vector,
            query_identity=self._snapshot.revision.embedding_space,
            limit=limit,
        )
        return DenseSearchResult(
            candidates=candidates,
            embedding_elapsed_ms=embedding_elapsed_ms,
            index_elapsed_ms=max(
                0.0,
                (self._context.deadline.clock() - index_started) * 1000,
            ),
            attempts=attempts,
            usage=usage,
        )


QueryDenseRetriever = BoundDenseRetriever


class _EmbeddingResultLike(Protocol):
    identity: object
    vectors: object


def _routed_embedding_result(
    routed: RoutedResult[EmbeddingResult],
) -> EmbeddingResult:
    if not isinstance(routed, RoutedResult) or not isinstance(routed.value, EmbeddingResult):
        raise DenseIndexError("query_embedding_result_invalid")
    return routed.value


def _validated_query_vector(
    result: object,
    *,
    expected_identity: object,
) -> tuple[float, ...]:
    try:
        candidate = cast(_EmbeddingResultLike, result)
        identity = candidate.identity
        raw_vectors = candidate.vectors
    except (AttributeError, TypeError):
        raise DenseIndexError("query_embedding_result_invalid") from None
    if identity != expected_identity:
        raise DenseIndexError("embedding_identity_mismatch")
    if not isinstance(raw_vectors, Sequence) or isinstance(raw_vectors, (str, bytes, bytearray)):
        raise DenseIndexError("query_embedding_result_invalid")
    vectors = cast(Sequence[object], raw_vectors)
    if len(vectors) != 1:
        raise DenseIndexError("query_embedding_count_mismatch")
    raw_vector = vectors[0]
    if not isinstance(raw_vector, Sequence) or isinstance(raw_vector, (str, bytes, bytearray)):
        raise DenseIndexError("query_embedding_vector_invalid")
    values = cast(Sequence[object], raw_vector)
    dimension = getattr(expected_identity, "dimension", None)
    if len(values) != dimension:
        raise DenseIndexError("query_embedding_dimension_mismatch")
    vector: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise DenseIndexError("query_embedding_vector_nonfinite")
        try:
            normalized = float(value)
        except (OverflowError, TypeError, ValueError):
            raise DenseIndexError("query_embedding_vector_nonfinite") from None
        if not math.isfinite(normalized):
            raise DenseIndexError("query_embedding_vector_nonfinite")
        vector.append(normalized)
    return tuple(vector)
