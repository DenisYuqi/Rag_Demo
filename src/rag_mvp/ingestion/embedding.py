"""Validated, deduplicated, cache-backed document embedding stage."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import cast

from rag_mvp.domain.ingestion import Chunk
from rag_mvp.providers.errors import ProviderError
from rag_mvp.providers.models import (
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingSpaceIdentity,
    ProviderCallContext,
)
from rag_mvp.providers.protocols import EmbeddingProvider
from rag_mvp.storage.embedding_cache import EmbeddingCache, EmbeddingVector


class EmbeddingStageError(ValueError):
    """A safe validation failure that never includes document text."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class EmbeddingStageResult:
    vectors: tuple[EmbeddingVector, ...] = field(repr=False)
    cache_hit_count: int
    cache_miss_count: int
    unique_content_count: int
    provider_call_count: int


class EmbeddingStage:
    """Embed unique chunk content and fan vectors back to original chunk order."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        cache: EmbeddingCache,
        *,
        batch_size: int = 128,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("embedding_stage_batch_size_invalid")
        self._provider = provider
        self._cache = cache
        self._batch_size = batch_size

    @property
    def identity(self) -> EmbeddingSpaceIdentity:
        return _provider_identity(self._provider)

    async def embed(
        self,
        chunks: Sequence[Chunk],
        context: ProviderCallContext,
    ) -> EmbeddingStageResult:
        """Return validated vectors and content-safe aggregate diagnostics."""

        ordered_chunks = tuple(chunks)
        identity = _provider_identity(self._provider)

        texts_by_digest: dict[str, str] = {}
        digest_text_pairs: list[tuple[str, str]] = []
        ordered_digests: list[str] = []
        for chunk in ordered_chunks:
            try:
                content_digest = chunk.content_digest
                text = chunk.text
            except AttributeError:
                raise EmbeddingStageError("embedding_chunk_invalid") from None
            if not isinstance(content_digest, str) or not isinstance(text, str):
                raise EmbeddingStageError("embedding_chunk_invalid")
            if content_digest in texts_by_digest and texts_by_digest[content_digest] != text:
                raise EmbeddingStageError("embedding_chunk_digest_collision")
            texts_by_digest.setdefault(content_digest, text)
            digest_text_pairs.append((content_digest, text))
            ordered_digests.append(content_digest)

        for content_digest, text in digest_text_pairs:
            try:
                computed_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            except UnicodeEncodeError:
                raise EmbeddingStageError("embedding_chunk_text_invalid") from None
            if computed_digest != content_digest:
                raise EmbeddingStageError("embedding_chunk_digest_mismatch")

        vectors_by_digest: dict[str, EmbeddingVector] = {}
        missing_digests: list[str] = []
        cache_hit_count = 0
        for content_digest in texts_by_digest:
            vector = self._cache.get(identity, content_digest)
            if vector is None:
                missing_digests.append(content_digest)
            else:
                vectors_by_digest[content_digest] = vector
                cache_hit_count += 1

        pending_cache_writes: dict[str, EmbeddingVector] = {}
        provider_call_count = 0
        for start in range(0, len(missing_digests), self._batch_size):
            batch_digests = missing_digests[start : start + self._batch_size]
            request = EmbeddingRequest(tuple(texts_by_digest[digest] for digest in batch_digests))
            provider_call_count += 1
            try:
                result = await self._provider.embed(request, context)
                batch_vectors = _validate_provider_result(
                    result,
                    expected_identity=identity,
                    expected_count=len(batch_digests),
                )
            except (EmbeddingStageError, ProviderError):
                raise
            except Exception:
                raise EmbeddingStageError("embedding_provider_failed") from None
            pending_cache_writes.update(zip(batch_digests, batch_vectors, strict=True))

        if pending_cache_writes:
            if _provider_identity(self._provider) != identity:
                raise EmbeddingStageError("embedding_provider_identity_changed")
            self._cache.put_many(identity, pending_cache_writes)
            vectors_by_digest.update(pending_cache_writes)

        return EmbeddingStageResult(
            vectors=tuple(vectors_by_digest[digest] for digest in ordered_digests),
            cache_hit_count=cache_hit_count,
            cache_miss_count=len(missing_digests),
            unique_content_count=len(texts_by_digest),
            provider_call_count=provider_call_count,
        )


def _provider_identity(provider: EmbeddingProvider) -> EmbeddingSpaceIdentity:
    try:
        identity = provider.identity
    except Exception:
        raise EmbeddingStageError("embedding_provider_identity_invalid") from None
    if not isinstance(identity, EmbeddingSpaceIdentity):
        raise EmbeddingStageError("embedding_provider_identity_invalid")
    return identity


def _validate_provider_result(
    result: EmbeddingResult,
    *,
    expected_identity: EmbeddingSpaceIdentity,
    expected_count: int,
) -> tuple[EmbeddingVector, ...]:
    try:
        result_identity = result.identity
        raw_vectors: object = result.vectors
    except (AttributeError, TypeError):
        raise EmbeddingStageError("embedding_provider_result_invalid") from None
    if (
        not isinstance(result_identity, EmbeddingSpaceIdentity)
        or result_identity != expected_identity
    ):
        raise EmbeddingStageError("embedding_provider_identity_mismatch")
    if not isinstance(raw_vectors, Sequence) or isinstance(raw_vectors, (str, bytes, bytearray)):
        raise EmbeddingStageError("embedding_provider_result_invalid")
    vectors = cast(Sequence[object], raw_vectors)
    if len(vectors) != expected_count:
        raise EmbeddingStageError("embedding_provider_count_mismatch")

    validated: list[EmbeddingVector] = []
    for raw_vector in vectors:
        if not isinstance(raw_vector, Sequence) or isinstance(raw_vector, (str, bytes, bytearray)):
            raise EmbeddingStageError("embedding_provider_vector_invalid")
        values = cast(Sequence[object], raw_vector)
        if len(values) != expected_identity.dimension:
            raise EmbeddingStageError("embedding_provider_dimension_mismatch")
        vector: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise EmbeddingStageError("embedding_provider_vector_nonfinite")
            try:
                normalized_value = float(value)
            except (OverflowError, ValueError):
                raise EmbeddingStageError("embedding_provider_vector_nonfinite") from None
            if not math.isfinite(normalized_value):
                raise EmbeddingStageError("embedding_provider_vector_nonfinite")
            vector.append(normalized_value)
        validated.append(tuple(vector))
    return tuple(validated)
