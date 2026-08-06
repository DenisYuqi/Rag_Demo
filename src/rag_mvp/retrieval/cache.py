"""Future-safe retrieval cache identities and bounded in-memory storage."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from pydantic import ValidationError

from rag_mvp.domain.ingestion import EmbeddingSpaceIdentity
from rag_mvp.domain.retrieval import RetrievalMode
from rag_mvp.retrieval.request import canonicalize_query

RETRIEVAL_CACHE_IDENTITY_VERSION = "retrieval-cache-key-v2"


@dataclass(frozen=True, slots=True)
class RetrievalCacheIdentity:
    canonical_query: str
    canonicalization_version: str
    revision_id: str
    chunk_set_digest: str
    chunk_set_digest_algorithm: str
    record_digest_algorithm: str
    extraction_version: str
    chunking_version: str
    mode: RetrievalMode | str
    embedding_identity: EmbeddingSpaceIdentity
    dense_schema_version: str
    dense_metric: str
    bm25_tokenizer_identity: str
    bm25_schema_version: str
    bm25_algorithm_version: str
    bm25_k1: float
    bm25_b: float
    rrf_version: str
    rrf_k: int
    dense_weight: float
    lexical_weight: float
    rrf_tie_policy: str
    reranker_route_id: str | None
    reranker_provider: str | None
    reranker_model: str | None
    reranker_adapter_version: str | None
    reranker_prompt_version: str | None
    reranker_truncation_version: str | None
    reranker_parser_version: str | None
    reranker_maximum_query_characters: int | None
    reranker_maximum_query_tokens: int | None
    reranker_maximum_candidate_characters: int | None
    reranker_maximum_candidate_tokens: int | None
    reranker_budget_seconds: float | None
    allow_single_retriever_degradation: bool
    degradation_policy_version: str
    dense_limit: int
    lexical_limit: int
    rerank_limit: int
    final_limit: int
    evidence_schema_version: str
    result_schema_version: str
    safety_version: str
    identity_version: str = RETRIEVAL_CACHE_IDENTITY_VERSION

    def __post_init__(self) -> None:
        for name in (
            "canonical_query",
            "canonicalization_version",
            "revision_id",
            "chunk_set_digest",
            "chunk_set_digest_algorithm",
            "record_digest_algorithm",
            "extraction_version",
            "chunking_version",
            "dense_schema_version",
            "dense_metric",
            "bm25_tokenizer_identity",
            "bm25_schema_version",
            "bm25_algorithm_version",
            "rrf_version",
            "rrf_tie_policy",
            "degradation_policy_version",
            "evidence_schema_version",
            "result_schema_version",
            "safety_version",
            "identity_version",
        ):
            _require_nonempty(getattr(self, name), name)
        try:
            canonical_query = canonicalize_query(
                self.canonical_query,
                maximum_characters=len(self.canonical_query),
            )
        except ValueError:
            raise ValueError("canonical_query is invalid") from None
        if canonical_query != self.canonical_query:
            raise ValueError("canonical_query must already be canonical")
        if self.identity_version != RETRIEVAL_CACHE_IDENTITY_VERSION:
            raise ValueError("retrieval cache identity version is unsupported")
        try:
            mode = RetrievalMode(self.mode)
        except (TypeError, ValueError):
            raise ValueError("cache retrieval mode is invalid") from None
        object.__setattr__(self, "mode", mode)
        if not isinstance(self.embedding_identity, EmbeddingSpaceIdentity):
            raise TypeError("embedding_identity must be an EmbeddingSpaceIdentity")
        try:
            embedding_identity = EmbeddingSpaceIdentity.model_validate(
                self.embedding_identity.model_dump()
            )
        except (TypeError, ValueError, ValidationError):
            raise ValueError("embedding_identity is invalid") from None
        object.__setattr__(self, "embedding_identity", embedding_identity)
        for name in ("bm25_k1", "bm25_b", "dense_weight", "lexical_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be finite")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, normalized)
        if self.bm25_k1 <= 0 or not 0 <= self.bm25_b <= 1:
            raise ValueError("BM25 parameters are invalid")
        if self.dense_weight <= 0 or self.lexical_weight <= 0:
            raise ValueError("RRF weights must be positive")
        if type(self.rrf_k) is not int or self.rrf_k < 1:
            raise ValueError("rrf_k must be a positive integer")
        for name in ("dense_limit", "lexical_limit", "rerank_limit", "final_limit"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.final_limit > self.rerank_limit:
            raise ValueError("final_limit cannot exceed rerank_limit")
        if type(self.allow_single_retriever_degradation) is not bool:
            raise TypeError("allow_single_retriever_degradation must be a boolean")

        reranker_required = (
            self.reranker_provider,
            self.reranker_model,
            self.reranker_adapter_version,
            self.reranker_prompt_version,
            self.reranker_truncation_version,
            self.reranker_parser_version,
            self.reranker_maximum_query_characters,
            self.reranker_maximum_query_tokens,
            self.reranker_maximum_candidate_characters,
            self.reranker_maximum_candidate_tokens,
            self.reranker_budget_seconds,
        )
        if any(value is None for value in reranker_required) and any(
            value is not None for value in reranker_required
        ):
            raise ValueError("reranker cache identity is incomplete")
        for name in (
            "reranker_route_id",
            "reranker_provider",
            "reranker_model",
            "reranker_adapter_version",
            "reranker_prompt_version",
            "reranker_truncation_version",
            "reranker_parser_version",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_nonempty(value, name)
        if self.reranker_route_id is not None and self.reranker_provider is None:
            raise ValueError("reranker route requires a complete reranker identity")
        for name in (
            "reranker_maximum_query_characters",
            "reranker_maximum_query_tokens",
            "reranker_maximum_candidate_characters",
            "reranker_maximum_candidate_tokens",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.reranker_budget_seconds is not None:
            value = self.reranker_budget_seconds
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("reranker_budget_seconds must be positive and finite")
            normalized_budget = float(value)
            if not math.isfinite(normalized_budget) or normalized_budget <= 0:
                raise ValueError("reranker_budget_seconds must be positive and finite")
            object.__setattr__(self, "reranker_budget_seconds", normalized_budget)

    @property
    def canonical_json(self) -> str:
        payload: Mapping[str, object] = {
            "identity_version": self.identity_version,
            "canonical_query": self.canonical_query,
            "canonicalization_version": self.canonicalization_version,
            "revision_id": self.revision_id,
            "chunk_set_digest": self.chunk_set_digest,
            "chunk_set_digest_algorithm": self.chunk_set_digest_algorithm,
            "record_digest_algorithm": self.record_digest_algorithm,
            "extraction_version": self.extraction_version,
            "chunking_version": self.chunking_version,
            "mode": RetrievalMode(self.mode).value,
            "embedding_identity": self.embedding_identity.model_dump(mode="json"),
            "dense": {
                "schema_version": self.dense_schema_version,
                "metric": self.dense_metric,
            },
            "bm25": {
                "tokenizer_identity": self.bm25_tokenizer_identity,
                "schema_version": self.bm25_schema_version,
                "algorithm_version": self.bm25_algorithm_version,
                "k1": self.bm25_k1,
                "b": self.bm25_b,
            },
            "rrf": {
                "version": self.rrf_version,
                "k": self.rrf_k,
                "dense_weight": self.dense_weight,
                "lexical_weight": self.lexical_weight,
                "tie_policy": self.rrf_tie_policy,
            },
            "reranker": (
                None
                if self.reranker_provider is None
                else {
                    "route_id": self.reranker_route_id,
                    "provider": self.reranker_provider,
                    "model": self.reranker_model,
                    "adapter_version": self.reranker_adapter_version,
                    "prompt_version": self.reranker_prompt_version,
                    "truncation_version": self.reranker_truncation_version,
                    "parser_version": self.reranker_parser_version,
                    "maximum_query_characters": self.reranker_maximum_query_characters,
                    "maximum_query_tokens": self.reranker_maximum_query_tokens,
                    "maximum_candidate_characters": self.reranker_maximum_candidate_characters,
                    "maximum_candidate_tokens": self.reranker_maximum_candidate_tokens,
                    "budget_seconds": self.reranker_budget_seconds,
                }
            ),
            "degradation": {
                "allow_single_retriever": self.allow_single_retriever_degradation,
                "policy_version": self.degradation_policy_version,
            },
            "limits": {
                "dense": self.dense_limit,
                "lexical": self.lexical_limit,
                "rerank": self.rerank_limit,
                "final": self.final_limit,
            },
            "schemas": {
                "evidence": self.evidence_schema_version,
                "result": self.result_schema_version,
                "safety": self.safety_version,
            },
        }
        return json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def key(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


class BoundedTtlCache[T]:
    """Thread-safe TTL/LRU cache with an injectable monotonic clock."""

    def __init__(
        self,
        *,
        maximum_entries: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(maximum_entries) is not int or maximum_entries < 1:
            raise ValueError("maximum_entries must be a positive integer")
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)):
            raise ValueError("ttl_seconds must be positive and finite")
        normalized_ttl = float(ttl_seconds)
        if not math.isfinite(normalized_ttl) or normalized_ttl <= 0:
            raise ValueError("ttl_seconds must be positive and finite")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._maximum_entries = maximum_entries
        self._ttl_seconds = normalized_ttl
        self._clock = clock
        self._values: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str, *, now: float | None = None) -> T | None:
        _require_nonempty(key, "key")
        current = self._now(now)
        with self._lock:
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
        _require_nonempty(key, "key")
        current = self._now(now)
        with self._lock:
            self._purge_expired(current)
            self._values[key] = (current + self._ttl_seconds, value)
            self._values.move_to_end(key)
            while len(self._values) > self._maximum_entries:
                self._values.popitem(last=False)

    def put_if_cacheable(
        self,
        key: str,
        value: T,
        *,
        succeeded: bool,
        cancelled: bool,
        degraded: bool,
        now: float | None = None,
    ) -> bool:
        _require_nonempty(key, "key")
        for name, flag in (
            ("succeeded", succeeded),
            ("cancelled", cancelled),
            ("degraded", degraded),
        ):
            if type(flag) is not bool:
                raise TypeError(f"{name} must be a boolean")
        if not succeeded or cancelled or degraded:
            return False
        self.put(key, value, now=now)
        return True

    def clear(self) -> None:
        with self._lock:
            self._values.clear()

    def __len__(self) -> int:
        current = self._now(None)
        with self._lock:
            self._purge_expired(current)
            return len(self._values)

    def _now(self, value: float | None) -> float:
        current = self._clock() if value is None else value
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            raise ValueError("cache clock value must be finite")
        normalized = float(current)
        if not math.isfinite(normalized):
            raise ValueError("cache clock value must be finite")
        return normalized

    def _purge_expired(self, current: float) -> None:
        expired = [key for key, (expires_at, _) in self._values.items() if current >= expires_at]
        for key in expired:
            del self._values[key]


def _require_nonempty(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
