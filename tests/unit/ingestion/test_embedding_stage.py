from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from rag_mvp.domain.ingestion import Chunk, ChunkLocator
from rag_mvp.ingestion.embedding import EmbeddingStage, EmbeddingStageError
from rag_mvp.providers.models import (
    Deadline,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingSpaceIdentity,
    NormalizationPolicy,
    ProviderCallContext,
)
from rag_mvp.storage.embedding_cache import EmbeddingCache


def _identity(
    *,
    provider: str = "recording",
    model: str = "embedding-v1",
    dimension: int = 2,
    normalization: NormalizationPolicy = NormalizationPolicy.NONE,
    adapter_version: str = "adapter-v1",
) -> EmbeddingSpaceIdentity:
    return EmbeddingSpaceIdentity(
        provider=provider,
        model=model,
        dimension=dimension,
        normalization=normalization,
        adapter_version=adapter_version,
    )


def _context() -> ProviderCallContext:
    return ProviderCallContext(
        request_id="request-embedding",
        operation_id="operation-embedding",
        deadline=Deadline.after(10),
    )


def _chunk(text: str, ordinal: int, *, digest: str | None = None) -> Chunk:
    content_digest = digest or hashlib.sha256(text.encode("utf-8")).hexdigest()
    chunk = Chunk(
        chunk_id=f"chunk-{ordinal}",
        parent_chunk_id=f"parent-{ordinal}",
        source_id="source-1",
        document_version=1,
        ordinal=ordinal,
        text=text,
        content_digest=content_digest,
        locator=ChunkLocator(char_start=ordinal * 10, char_end=ordinal * 10 + len(text)),
    )
    return chunk.model_copy(update={"text": text})


def _vector(text: str, dimension: int = 2) -> tuple[float, ...]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return tuple(float(digest[index]) for index in range(dimension))


def _bypassed_result(
    identity: EmbeddingSpaceIdentity,
    vectors: object,
) -> EmbeddingResult:
    result = object.__new__(EmbeddingResult)
    object.__setattr__(result, "identity", identity)
    object.__setattr__(result, "vectors", vectors)
    return result


class RecordingProvider:
    def __init__(
        self,
        identity: EmbeddingSpaceIdentity | None = None,
        behavior: Callable[[EmbeddingRequest, int], EmbeddingResult] | None = None,
    ) -> None:
        self._identity = identity or _identity()
        self._behavior = behavior
        self.requests: list[EmbeddingRequest] = []

    @property
    def identity(self) -> EmbeddingSpaceIdentity:
        return self._identity

    async def embed(
        self,
        request: EmbeddingRequest,
        context: ProviderCallContext,
    ) -> EmbeddingResult:
        del context
        self.requests.append(request)
        if self._behavior is not None:
            return self._behavior(request, len(self.requests))
        return EmbeddingResult(
            tuple(_vector(text, self.identity.dimension) for text in request.texts),
            self.identity,
        )


async def test_duplicate_chunks_use_one_provider_input_and_fan_out(tmp_path: Path) -> None:
    provider = RecordingProvider()
    with EmbeddingCache(tmp_path / "embeddings.sqlite3") as cache:
        result = await EmbeddingStage(provider, cache, batch_size=8).embed(
            (_chunk("same", 0), _chunk("other", 1), _chunk("same", 2)),
            _context(),
        )

    assert [request.texts for request in provider.requests] == [("same", "other")]
    assert result.vectors == (_vector("same"), _vector("other"), _vector("same"))
    assert result.cache_hit_count == 0
    assert result.cache_miss_count == 2
    assert result.unique_content_count == 2
    assert result.provider_call_count == 1


async def test_cache_persists_across_calls_and_reopen_with_no_all_hit_call(
    tmp_path: Path,
) -> None:
    path = tmp_path / "embeddings.sqlite3"
    chunks = (_chunk("one", 0), _chunk("two", 1))
    first_provider = RecordingProvider()
    with EmbeddingCache(path) as cache:
        first = await EmbeddingStage(first_provider, cache).embed(chunks, _context())
        second = await EmbeddingStage(first_provider, cache).embed(chunks, _context())

    reopened_provider = RecordingProvider()
    with EmbeddingCache(path) as reopened:
        third = await EmbeddingStage(reopened_provider, reopened).embed(chunks, _context())

    assert first.provider_call_count == 1
    assert second.cache_hit_count == 2
    assert second.cache_miss_count == 0
    assert second.provider_call_count == 0
    assert third.vectors == first.vectors
    assert third.cache_hit_count == 2
    assert reopened_provider.requests == []


async def test_mixed_hits_and_misses_preserve_input_order(tmp_path: Path) -> None:
    provider = RecordingProvider()
    cached = _chunk("cached", 0)
    missing = _chunk("missing", 1)
    with EmbeddingCache(tmp_path / "embeddings.sqlite3") as cache:
        cache.put(provider.identity, cached.content_digest, _vector(cached.text))
        result = await EmbeddingStage(provider, cache).embed(
            (missing, cached, missing),
            _context(),
        )

    assert [request.texts for request in provider.requests] == [("missing",)]
    assert result.vectors == (_vector("missing"), _vector("cached"), _vector("missing"))
    assert (result.cache_hit_count, result.cache_miss_count) == (1, 1)


async def test_unique_misses_are_batched(tmp_path: Path) -> None:
    provider = RecordingProvider()
    chunks = tuple(_chunk(f"text-{index}", index) for index in range(5))
    with EmbeddingCache(tmp_path / "embeddings.sqlite3") as cache:
        result = await EmbeddingStage(provider, cache, batch_size=2).embed(chunks, _context())

    assert [request.texts for request in provider.requests] == [
        ("text-0", "text-1"),
        ("text-2", "text-3"),
        ("text-4",),
    ]
    assert result.provider_call_count == 3
    assert result.vectors == tuple(_vector(chunk.text) for chunk in chunks)


@pytest.mark.parametrize(
    "changed_identity",
    [
        _identity(provider="other-provider"),
        _identity(model="embedding-v2"),
        _identity(dimension=3),
        _identity(normalization=NormalizationPolicy.L2),
        _identity(adapter_version="adapter-v2"),
    ],
)
async def test_every_embedding_identity_field_change_is_a_cache_miss(
    tmp_path: Path,
    changed_identity: EmbeddingSpaceIdentity,
) -> None:
    path = tmp_path / "embeddings.sqlite3"
    chunk = _chunk("identity-sensitive", 0)
    original = RecordingProvider()
    with EmbeddingCache(path) as cache:
        await EmbeddingStage(original, cache).embed((chunk,), _context())

        changed_provider = RecordingProvider(changed_identity)
        result = await EmbeddingStage(changed_provider, cache).embed((chunk,), _context())

    assert result.cache_miss_count == 1
    assert len(changed_provider.requests) == 1


@pytest.mark.parametrize(
    ("behavior", "error_code"),
    [
        (
            lambda request, call: (_ for _ in ()).throw(RuntimeError("raw document content")),
            "embedding_provider_failed",
        ),
        (
            lambda request, call: _bypassed_result(_identity(), ()),
            "embedding_provider_count_mismatch",
        ),
        (
            lambda request, call: _bypassed_result(_identity(), ((1.0,),)),
            "embedding_provider_dimension_mismatch",
        ),
        (
            lambda request, call: _bypassed_result(_identity(), ((float("nan"), 1.0),)),
            "embedding_provider_vector_nonfinite",
        ),
        (
            lambda request, call: _bypassed_result(_identity(model="wrong"), ((1.0, 2.0),)),
            "embedding_provider_identity_mismatch",
        ),
    ],
)
async def test_invalid_provider_results_do_not_poison_cache(
    tmp_path: Path,
    behavior: Callable[[EmbeddingRequest, int], EmbeddingResult],
    error_code: str,
) -> None:
    path = tmp_path / "embeddings.sqlite3"
    chunk = _chunk("sensitive-document-text", 0)
    provider = RecordingProvider(behavior=behavior)

    with EmbeddingCache(path) as cache:
        with pytest.raises(EmbeddingStageError, match=error_code) as caught:
            await EmbeddingStage(provider, cache).embed((chunk,), _context())
        assert chunk.text not in str(caught.value)
        assert "raw document content" not in str(caught.value)

    retry = RecordingProvider()
    with EmbeddingCache(path) as reopened:
        result = await EmbeddingStage(retry, reopened).embed((chunk,), _context())

    assert result.cache_miss_count == 1
    assert len(retry.requests) == 1


async def test_later_batch_failure_does_not_cache_earlier_batches(tmp_path: Path) -> None:
    chunks = (_chunk("first", 0), _chunk("second", 1))

    def fail_second_batch(request: EmbeddingRequest, call: int) -> EmbeddingResult:
        if call == 2:
            raise RuntimeError("second batch failed")
        return EmbeddingResult(tuple(_vector(text) for text in request.texts), _identity())

    with EmbeddingCache(tmp_path / "embeddings.sqlite3") as cache:
        with pytest.raises(EmbeddingStageError, match="embedding_provider_failed"):
            await EmbeddingStage(
                RecordingProvider(behavior=fail_second_batch),
                cache,
                batch_size=1,
            ).embed(chunks, _context())

        retry = RecordingProvider()
        result = await EmbeddingStage(retry, cache).embed(chunks, _context())

    assert result.cache_hit_count == 0
    assert result.cache_miss_count == 2


async def test_digest_mismatch_is_rejected_before_cache_or_provider_use(tmp_path: Path) -> None:
    provider = RecordingProvider()
    bad = _chunk("actual", 0, digest=hashlib.sha256(b"different").hexdigest())
    with (
        EmbeddingCache(tmp_path / "embeddings.sqlite3") as cache,
        pytest.raises(EmbeddingStageError, match="embedding_chunk_digest_mismatch"),
    ):
        await EmbeddingStage(provider, cache).embed((bad,), _context())

    assert provider.requests == []


async def test_one_digest_mapped_to_different_text_is_rejected(tmp_path: Path) -> None:
    digest = hashlib.sha256(b"first").hexdigest()
    chunks = (_chunk("first", 0, digest=digest), _chunk("second", 1, digest=digest))
    provider = RecordingProvider()
    with (
        EmbeddingCache(tmp_path / "embeddings.sqlite3") as cache,
        pytest.raises(EmbeddingStageError, match="embedding_chunk_digest_collision"),
    ):
        await EmbeddingStage(provider, cache).embed(chunks, _context())

    assert provider.requests == []


def test_cache_enforces_ttl_and_size_bounds(tmp_path: Path) -> None:
    current = [0.0]
    identity = _identity()
    first = _chunk("first", 0)
    second = _chunk("second", 1)
    with EmbeddingCache(
        tmp_path / "embeddings.sqlite3",
        max_entries=1,
        ttl_seconds=10,
        now=lambda: current[0],
    ) as cache:
        cache.put(identity, first.content_digest, _vector(first.text))
        current[0] = 1
        cache.put(identity, second.content_digest, _vector(second.text))
        assert cache.get(identity, first.content_digest) is None
        assert cache.get(identity, second.content_digest) == _vector(second.text)

        current[0] = 11
        assert cache.get(identity, second.content_digest) is None
        assert len(cache) == 0


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("identity_json", "{}"),
        ("dimension", 3),
        ("vector_json", "[NaN,1.0]"),
    ],
)
def test_corrupt_rows_are_removed_as_cache_misses(
    tmp_path: Path,
    column: str,
    value: object,
) -> None:
    path = tmp_path / "embeddings.sqlite3"
    identity = _identity()
    chunk = _chunk("corruptible", 0)
    with EmbeddingCache(path) as cache:
        cache.put(identity, chunk.content_digest, _vector(chunk.text))
        with sqlite3.connect(path) as connection:
            connection.execute(
                f"UPDATE embedding_cache SET {column} = ?",  # noqa: S608 - fixed test parameters
                (value,),
            )

        assert cache.get(identity, chunk.content_digest) is None
        assert len(cache) == 0


@pytest.mark.parametrize("vector", [(1.0,), (float("inf"), 1.0)])
def test_cache_rejects_invalid_vectors_on_write(
    tmp_path: Path,
    vector: tuple[float, ...],
) -> None:
    chunk = _chunk("write-validation", 0)
    with EmbeddingCache(tmp_path / "embeddings.sqlite3") as cache:
        with pytest.raises(ValueError, match="embedding_cache_vector"):
            cache.put(_identity(), chunk.content_digest, vector)
        assert len(cache) == 0


async def test_cache_database_never_stores_raw_chunk_text(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.sqlite3"
    chunk = _chunk("RAW_CHUNK_TEXT_SENTINEL_7f52", 0)
    with EmbeddingCache(path) as cache:
        await EmbeddingStage(RecordingProvider(), cache).embed((chunk,), _context())
        with sqlite3.connect(path) as connection:
            assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)

    assert chunk.text.encode("utf-8") not in path.read_bytes()
