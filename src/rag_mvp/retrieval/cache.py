"""Bounded TTL caches with version-complete deterministic keys."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class RetrievalCacheIdentity:
    canonical_query: str
    revision_id: str
    mode: str
    embedding_identity: str
    bm25_version: str
    rrf_version: str
    rrf_k: int
    dense_weight: float
    lexical_weight: float
    reranker_identity: str | None
    reranker_prompt_version: str | None
    dense_limit: int
    lexical_limit: int
    rerank_limit: int
    final_limit: int

    @property
    def key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class BoundedTtlCache[T]:
    def __init__(self, *, maximum_entries: int, ttl_seconds: float) -> None:
        if maximum_entries < 1 or ttl_seconds <= 0:
            raise ValueError("cache bounds must be positive")
        self._maximum_entries = maximum_entries
        self._ttl_seconds = ttl_seconds
        self._values: OrderedDict[str, tuple[float, T]] = OrderedDict()

    def get(self, key: str, *, now: float | None = None) -> T | None:
        current = time.monotonic() if now is None else now
        item = self._values.get(key)
        if item is None:
            return None
        expires_at, value = item
        if current >= expires_at:
            del self._values[key]
            return None
        self._values.move_to_end(key)
        return value

    def put(self, key: str, value: T, *, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        self._values[key] = (current + self._ttl_seconds, value)
        self._values.move_to_end(key)
        while len(self._values) > self._maximum_entries:
            self._values.popitem(last=False)

    def __len__(self) -> int:
        return len(self._values)
