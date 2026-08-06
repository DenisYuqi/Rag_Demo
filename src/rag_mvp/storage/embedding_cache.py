"""Persistent, bounded cache for content-addressed document embeddings."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from pathlib import Path
from types import TracebackType
from typing import Self, cast

from rag_mvp.providers.models import EmbeddingSpaceIdentity, NormalizationPolicy

type EmbeddingVector = tuple[float, ...]

_DEFAULT_MAX_ENTRIES = 100_000
_DEFAULT_TTL_SECONDS = 30 * 24 * 60 * 60
_MAX_BUSY_TIMEOUT_MS = 60_000


class EmbeddingCacheError(ValueError):
    """A safe cache configuration or write-validation error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _identity_payload(identity: EmbeddingSpaceIdentity) -> dict[str, str | int]:
    if not isinstance(identity, EmbeddingSpaceIdentity):
        raise EmbeddingCacheError("embedding_cache_identity_invalid")
    normalization = identity.normalization
    if not isinstance(normalization, NormalizationPolicy):
        raise EmbeddingCacheError("embedding_cache_identity_invalid")
    normalization_value = normalization.value
    if (
        not isinstance(identity.provider, str)
        or not identity.provider
        or not isinstance(identity.model, str)
        or not identity.model
        or isinstance(identity.dimension, bool)
        or not isinstance(identity.dimension, int)
        or identity.dimension <= 0
        or not normalization_value
        or not isinstance(identity.adapter_version, str)
        or not identity.adapter_version
    ):
        raise EmbeddingCacheError("embedding_cache_identity_invalid")
    return {
        "provider": identity.provider,
        "model": identity.model,
        "dimension": identity.dimension,
        "normalization": normalization_value,
        "adapter_version": identity.adapter_version,
    }


def _canonical_identity(identity: EmbeddingSpaceIdentity) -> tuple[dict[str, str | int], str]:
    payload = _identity_payload(identity)
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return payload, serialized


def _cache_key(identity_payload: Mapping[str, str | int], content_digest: str) -> str:
    if not isinstance(content_digest, str) or not content_digest:
        raise EmbeddingCacheError("embedding_cache_digest_invalid")
    serialized = json.dumps(
        {
            "content_digest": content_digest,
            "embedding_space": identity_payload,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validated_vector(vector: object, dimension: int) -> EmbeddingVector:
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes, bytearray)):
        raise EmbeddingCacheError("embedding_cache_vector_invalid")
    values = cast(Sequence[object], vector)
    if len(values) != dimension:
        raise EmbeddingCacheError("embedding_cache_vector_dimension_mismatch")
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise EmbeddingCacheError("embedding_cache_vector_nonfinite")
        try:
            normalized_value = float(value)
        except (OverflowError, ValueError):
            raise EmbeddingCacheError("embedding_cache_vector_nonfinite") from None
        if not math.isfinite(normalized_value):
            raise EmbeddingCacheError("embedding_cache_vector_nonfinite")
        normalized.append(normalized_value)
    return tuple(normalized)


class EmbeddingCache:
    """A self-initializing SQLite cache isolated from application metadata."""

    def __init__(
        self,
        path: Path,
        *,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        busy_timeout_ms: int = 5_000,
        now: Callable[[], float] = time.time,
    ) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise EmbeddingCacheError("embedding_cache_max_entries_invalid")
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
            or not math.isfinite(ttl_seconds)
            or ttl_seconds <= 0
        ):
            raise EmbeddingCacheError("embedding_cache_ttl_invalid")
        if (
            isinstance(busy_timeout_ms, bool)
            or not isinstance(busy_timeout_ms, int)
            or not 1 <= busy_timeout_ms <= _MAX_BUSY_TIMEOUT_MS
        ):
            raise EmbeddingCacheError("embedding_cache_busy_timeout_invalid")

        self.path = Path(path)
        self._max_entries = max_entries
        self._ttl_seconds = float(ttl_seconds)
        self._busy_timeout_ms = busy_timeout_ms
        self._now = now
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: sqlite3.Connection | None = sqlite3.connect(
            self.path,
            timeout=busy_timeout_ms / 1_000,
            isolation_level=None,
        )
        try:
            self._initialize()
        except BaseException:
            self._connection.close()
            self._connection = None
            raise

    def _initialize(self) -> None:
        connection = self._open_connection()
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()
        if journal_mode is None or str(journal_mode[0]).casefold() != "wal":
            raise RuntimeError("embedding_cache_wal_unavailable")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                cache_key TEXT PRIMARY KEY,
                identity_json TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                dimension INTEGER NOT NULL CHECK (dimension > 0),
                vector_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                accessed_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_embedding_cache_eviction
            ON embedding_cache(accessed_at, created_at, cache_key)
            """
        )

    def get(
        self,
        identity: EmbeddingSpaceIdentity,
        content_digest: str,
    ) -> EmbeddingVector | None:
        """Return one validated vector, treating and deleting corrupt rows as misses."""

        identity_payload, identity_json = _canonical_identity(identity)
        key = _cache_key(identity_payload, content_digest)
        current = self._current_time()
        with self._lock:
            connection = self._open_connection()
            row = connection.execute(
                """
                SELECT
                    identity_json,
                    content_digest,
                    dimension,
                    vector_json,
                    created_at,
                    accessed_at
                FROM embedding_cache
                WHERE cache_key = ?
                """,
                (key,),
            ).fetchone()
            if row is None:
                return None
            try:
                vector = self._validate_row(
                    row,
                    identity_json=identity_json,
                    content_digest=content_digest,
                    dimension=identity.dimension,
                    current=current,
                )
            except Exception:
                connection.execute("DELETE FROM embedding_cache WHERE cache_key = ?", (key,))
                return None
            connection.execute(
                "UPDATE embedding_cache SET accessed_at = ? WHERE cache_key = ?",
                (current, key),
            )
            return vector

    def put(
        self,
        identity: EmbeddingSpaceIdentity,
        content_digest: str,
        vector: Sequence[float],
    ) -> None:
        """Validate and atomically persist one vector."""

        self.put_many(identity, {content_digest: vector})

    def put_many(
        self,
        identity: EmbeddingSpaceIdentity,
        vectors: Mapping[str, Sequence[float]],
    ) -> None:
        """Validate every vector before atomically writing and enforcing bounds."""

        identity_payload, identity_json = _canonical_identity(identity)
        prepared: list[tuple[str, str, str]] = []
        for content_digest, raw_vector in vectors.items():
            key = _cache_key(identity_payload, content_digest)
            vector = _validated_vector(raw_vector, identity.dimension)
            vector_json = json.dumps(vector, allow_nan=False, separators=(",", ":"))
            prepared.append((key, content_digest, vector_json))
        current = self._current_time()

        with self._lock:
            connection = self._open_connection()
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._purge_expired(connection, current)
                for key, content_digest, vector_json in prepared:
                    connection.execute(
                        """
                        INSERT INTO embedding_cache(
                            cache_key,
                            identity_json,
                            content_digest,
                            dimension,
                            vector_json,
                            created_at,
                            accessed_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(cache_key) DO UPDATE SET
                            identity_json = excluded.identity_json,
                            content_digest = excluded.content_digest,
                            dimension = excluded.dimension,
                            vector_json = excluded.vector_json,
                            created_at = excluded.created_at,
                            accessed_at = excluded.accessed_at
                        """,
                        (
                            key,
                            identity_json,
                            content_digest,
                            identity.dimension,
                            vector_json,
                            current,
                            current,
                        ),
                    )
                row = connection.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()
                entry_count = 0 if row is None else int(row[0])
                excess = entry_count - self._max_entries
                if excess > 0:
                    connection.execute(
                        """
                        DELETE FROM embedding_cache
                        WHERE cache_key IN (
                            SELECT cache_key
                            FROM embedding_cache
                            ORDER BY accessed_at, created_at, cache_key
                            LIMIT ?
                        )
                        """,
                        (excess,),
                    )
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    def __len__(self) -> int:
        current = self._current_time()
        with self._lock:
            connection = self._open_connection()
            self._purge_expired(connection, current)
            row = connection.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()
            return 0 if row is None else int(row[0])

    def close(self) -> None:
        """Close the persistent connection; calling close repeatedly is safe."""

        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __enter__(self) -> Self:
        self._open_connection()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def _open_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("embedding_cache_closed")
        return self._connection

    def _current_time(self) -> float:
        current = float(self._now())
        if not math.isfinite(current):
            raise EmbeddingCacheError("embedding_cache_clock_invalid")
        return current

    def _purge_expired(self, connection: sqlite3.Connection, current: float) -> None:
        connection.execute(
            "DELETE FROM embedding_cache WHERE created_at <= ?",
            (current - self._ttl_seconds,),
        )

    def _validate_row(
        self,
        row: sqlite3.Row,
        *,
        identity_json: str,
        content_digest: str,
        dimension: int,
        current: float,
    ) -> EmbeddingVector:
        if row["identity_json"] != identity_json or row["content_digest"] != content_digest:
            raise EmbeddingCacheError("embedding_cache_row_identity_mismatch")
        stored_dimension = row["dimension"]
        if not isinstance(stored_dimension, int) or stored_dimension != dimension:
            raise EmbeddingCacheError("embedding_cache_row_dimension_mismatch")
        created_at = row["created_at"]
        if (
            isinstance(created_at, bool)
            or not isinstance(created_at, Real)
            or not math.isfinite(created_at)
            or current >= float(created_at) + self._ttl_seconds
        ):
            raise EmbeddingCacheError("embedding_cache_row_expired_or_invalid")
        vector_json = row["vector_json"]
        if not isinstance(vector_json, str):
            raise EmbeddingCacheError("embedding_cache_row_vector_invalid")
        try:
            vector: object = json.loads(vector_json)
        except (json.JSONDecodeError, TypeError):
            raise EmbeddingCacheError("embedding_cache_row_vector_invalid") from None
        return _validated_vector(vector, dimension)
