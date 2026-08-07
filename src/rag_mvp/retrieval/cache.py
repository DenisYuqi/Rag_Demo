"""Future-safe retrieval cache identities and bounded in-memory storage."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import ValidationError, model_validator

from rag_mvp.domain._base import DomainModel
from rag_mvp.domain.ingestion import EmbeddingSpaceIdentity
from rag_mvp.domain.retrieval import RankingEvidence, RetrievalMode, RetrievalResult
from rag_mvp.retrieval.request import canonicalize_query

RETRIEVAL_CACHE_IDENTITY_VERSION = "retrieval-cache-key-v2"
_RETRIEVAL_CACHE_ENVELOPE_VERSION = "retrieval-cache-envelope-v1"
_CACHE_KEY = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class RetrievalCacheIdentity:
    canonical_query: str = field(repr=False)
    canonicalization_version: str
    configuration_id: str
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
            "configuration_id",
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
    def query_digest(self) -> str:
        """Return a one-way digest; raw normalized queries never enter stored keys."""

        return hashlib.sha256(self.canonical_query.encode("utf-8")).hexdigest()

    @property
    def canonical_json(self) -> str:
        payload: Mapping[str, object] = {
            "identity_version": self.identity_version,
            "query_digest": self.query_digest,
            "canonicalization_version": self.canonicalization_version,
            "configuration_id": self.configuration_id,
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


class BoundedCacheLookupStatus(StrEnum):
    HIT = "hit"
    MISS = "miss"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class BoundedCacheLookup[T]:
    status: BoundedCacheLookupStatus
    value: T | None = None


@dataclass(frozen=True, slots=True)
class BoundedCacheWrite:
    expired_count: int = 0
    eviction_count: int = 0


class RetrievalCachePayload(DomainModel):
    """Request-neutral, validated retrieval evidence retained by the cache."""

    evidence: tuple[RankingEvidence, ...]
    requested_mode: RetrievalMode
    effective_mode: RetrievalMode
    index_revision: str
    candidate_counts: dict[str, int]
    provider_identities: dict[str, str]
    pre_rerank_chunk_ids: tuple[str, ...]
    post_rerank_chunk_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_stable_evidence(self) -> Self:
        if any(count < 0 for count in self.candidate_counts.values()):
            raise ValueError("cache candidate counts must be non-negative")
        if any(not name or not identity for name, identity in self.provider_identities.items()):
            raise ValueError("cache provider identities must be non-empty")
        if any(item.revision_id != self.index_revision for item in self.evidence):
            raise ValueError("cached evidence revision mismatch")
        if tuple(item.final_rank for item in self.evidence) != tuple(
            range(1, len(self.evidence) + 1)
        ):
            raise ValueError("cached evidence ranks are invalid")
        for label, chunk_ids in (
            ("pre-rerank", self.pre_rerank_chunk_ids),
            ("post-rerank", self.post_rerank_chunk_ids),
        ):
            if len(chunk_ids) != len(set(chunk_ids)):
                raise ValueError(f"cached {label} evidence is invalid")
        if set(self.pre_rerank_chunk_ids) != set(self.post_rerank_chunk_ids):
            raise ValueError("cached pre/post-rerank candidates differ")
        return self

    @classmethod
    def from_result(cls, result: RetrievalResult) -> RetrievalCachePayload:
        if not isinstance(result, RetrievalResult):
            raise TypeError("result must be a RetrievalResult")
        diagnostics = result.diagnostics
        if diagnostics.degradation_reasons or diagnostics.failed_stages:
            raise ValueError("degraded retrieval results are not cacheable")
        return cls(
            evidence=result.evidence,
            requested_mode=diagnostics.requested_mode,
            effective_mode=diagnostics.effective_mode,
            index_revision=diagnostics.index_revision,
            candidate_counts=dict(diagnostics.candidate_counts),
            provider_identities=dict(diagnostics.provider_identities),
            pre_rerank_chunk_ids=diagnostics.pre_rerank_chunk_ids,
            post_rerank_chunk_ids=diagnostics.post_rerank_chunk_ids,
        )


@dataclass(frozen=True, slots=True)
class RetrievalCacheCounterSnapshot:
    configuration_id: str
    eligible_lookups: int
    hits: int
    misses: int
    bypasses: int
    expirations: int
    evictions: int
    errors: int
    writes: int

    @property
    def hit_rate(self) -> float | None:
        if self.eligible_lookups == 0:
            return None
        return self.hits / self.eligible_lookups


class RetrievalCacheMetrics:
    """Thread-safe bounded-label counters; methods never accept query content."""

    def __init__(self, configuration_id: str) -> None:
        _require_nonempty(configuration_id, "configuration_id")
        self._configuration_id = configuration_id
        self._eligible_lookups = 0
        self._hits = 0
        self._misses = 0
        self._bypasses = 0
        self._expirations = 0
        self._evictions = 0
        self._errors = 0
        self._writes = 0
        self._lock = threading.Lock()

    def record_hit(self) -> None:
        with self._lock:
            self._eligible_lookups += 1
            self._hits += 1

    def record_miss(self, *, expirations: int = 0, error: bool = False) -> None:
        if type(expirations) is not int or expirations < 0:
            raise ValueError("expirations must be a non-negative integer")
        if type(error) is not bool:
            raise TypeError("error must be a boolean")
        with self._lock:
            self._eligible_lookups += 1
            self._misses += 1
            self._expirations += expirations
            self._errors += int(error)

    def record_bypass(self) -> None:
        with self._lock:
            self._bypasses += 1

    def record_write(
        self,
        *,
        expirations: int = 0,
        evictions: int = 0,
    ) -> None:
        for name, value in (("expirations", expirations), ("evictions", evictions)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        with self._lock:
            self._writes += 1
            self._expirations += expirations
            self._evictions += evictions

    def record_error(self) -> None:
        with self._lock:
            self._errors += 1

    def snapshot(self) -> RetrievalCacheCounterSnapshot:
        with self._lock:
            return RetrievalCacheCounterSnapshot(
                configuration_id=self._configuration_id,
                eligible_lookups=self._eligible_lookups,
                hits=self._hits,
                misses=self._misses,
                bypasses=self._bypasses,
                expirations=self._expirations,
                evictions=self._evictions,
                errors=self._errors,
                writes=self._writes,
            )


class RetrievalCacheReadStatus(StrEnum):
    HIT = "hit"
    MISS = "miss"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class RetrievalCacheRead:
    status: RetrievalCacheReadStatus
    payload: RetrievalCachePayload | None = None


class RetrievalCacheWriteStatus(StrEnum):
    STORED = "stored"
    SKIPPED_OBSOLETE = "skipped_obsolete"
    ERROR = "error"


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
        return self.lookup(key, now=now).value

    def lookup(self, key: str, *, now: float | None = None) -> BoundedCacheLookup[T]:
        _require_nonempty(key, "key")
        current = self._now(now)
        with self._lock:
            item = self._values.get(key)
            if item is None:
                return BoundedCacheLookup(BoundedCacheLookupStatus.MISS)
            expires_at, value = item
            if current >= expires_at:
                del self._values[key]
                return BoundedCacheLookup(BoundedCacheLookupStatus.EXPIRED)
            self._values.move_to_end(key)
            return BoundedCacheLookup(BoundedCacheLookupStatus.HIT, value)

    def put(self, key: str, value: T, *, now: float | None = None) -> None:
        self.put_with_outcome(key, value, now=now)

    def put_with_outcome(
        self,
        key: str,
        value: T,
        *,
        now: float | None = None,
    ) -> BoundedCacheWrite:
        _require_nonempty(key, "key")
        current = self._now(now)
        with self._lock:
            expired_count = self._purge_expired(current)
            self._values[key] = (current + self._ttl_seconds, value)
            self._values.move_to_end(key)
            eviction_count = 0
            while len(self._values) > self._maximum_entries:
                self._values.popitem(last=False)
                eviction_count += 1
            return BoundedCacheWrite(
                expired_count=expired_count,
                eviction_count=eviction_count,
            )

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

    def delete(self, key: str) -> None:
        _require_nonempty(key, "key")
        with self._lock:
            self._values.pop(key, None)

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

    def _purge_expired(self, current: float) -> int:
        expired = [key for key, (expires_at, _) in self._values.items() if current >= expires_at]
        for key in expired:
            del self._values[key]
        return len(expired)


class RetrievalResultCache:
    """Fail-open adapter over a bounded cache of request-neutral JSON payloads."""

    def __init__(
        self,
        *,
        configuration_id: str,
        maximum_entries: int,
        ttl_seconds: float,
        backend: BoundedTtlCache[str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        _require_nonempty(configuration_id, "configuration_id")
        self._configuration_id = configuration_id
        self._backend = (
            backend
            if backend is not None
            else BoundedTtlCache[str](
                maximum_entries=maximum_entries,
                ttl_seconds=ttl_seconds,
                clock=clock,
            )
        )
        self._metrics = RetrievalCacheMetrics(configuration_id)
        self._integrity_key = secrets.token_bytes(32)
        self._revision_lock = threading.RLock()
        self._active_revision_id: str | None = None
        self._active_revision_published_at: datetime | None = None

    @property
    def configuration_id(self) -> str:
        return self._configuration_id

    @property
    def metrics(self) -> RetrievalCacheMetrics:
        return self._metrics

    @property
    def active_revision_id(self) -> str | None:
        with self._revision_lock:
            return self._active_revision_id

    def activate_revision(self, revision_id: str, *, published_at: datetime) -> bool:
        """Atomically activate the newest revision and reclaim obsolete entries.

        Returns ``True`` only when this call advances the active revision. Older
        bindings remain usable without cache access, while same-revision bindings
        preserve useful entries.
        """

        _require_nonempty(revision_id, "revision_id")
        _require_aware_datetime(published_at, "published_at")
        with self._revision_lock:
            if self._active_revision_id == revision_id:
                if self._active_revision_published_at != published_at:
                    raise ValueError("revision publication identity mismatch")
                return False
            if (
                self._active_revision_published_at is not None
                and published_at <= self._active_revision_published_at
            ):
                return False
            try:
                self._backend.clear()
            except Exception:
                self._metrics.record_error()
            self._active_revision_id = revision_id
            self._active_revision_published_at = published_at
            return True

    def lookup(
        self,
        key: str,
        *,
        revision_id: str,
        requested_mode: RetrievalMode | str,
        validator: Callable[[RetrievalCachePayload], None] | None = None,
    ) -> RetrievalCacheRead:
        _require_cache_digest(key)
        _require_nonempty(revision_id, "revision_id")
        if validator is not None and not callable(validator):
            raise TypeError("validator must be callable")
        try:
            mode = RetrievalMode(requested_mode)
        except (TypeError, ValueError):
            raise ValueError("requested_mode is invalid") from None
        with self._revision_lock:
            if self._active_revision_id != revision_id:
                self._metrics.record_miss()
                return RetrievalCacheRead(RetrievalCacheReadStatus.MISS)
            try:
                lookup = self._backend.lookup(key)
            except Exception:
                self._metrics.record_miss(error=True)
                return RetrievalCacheRead(RetrievalCacheReadStatus.ERROR)
            if lookup.status is BoundedCacheLookupStatus.EXPIRED:
                self._metrics.record_miss(expirations=1)
                return RetrievalCacheRead(RetrievalCacheReadStatus.MISS)
            if lookup.status is BoundedCacheLookupStatus.MISS:
                self._metrics.record_miss()
                return RetrievalCacheRead(RetrievalCacheReadStatus.MISS)
            try:
                if not isinstance(lookup.value, str):
                    raise TypeError("cached retrieval payload is not JSON")
                payload = self._authenticated_payload(key, lookup.value)
                if payload.index_revision != revision_id or payload.requested_mode is not mode:
                    raise ValueError("cached retrieval identity mismatch")
                if validator is not None:
                    validator(payload)
            except Exception:
                with suppress(Exception):
                    self._backend.delete(key)
                self._metrics.record_miss(error=True)
                return RetrievalCacheRead(RetrievalCacheReadStatus.ERROR)
            self._metrics.record_hit()
            return RetrievalCacheRead(RetrievalCacheReadStatus.HIT, payload)

    def store(self, key: str, payload: RetrievalCachePayload) -> RetrievalCacheWriteStatus:
        _require_cache_digest(key)
        if not isinstance(payload, RetrievalCachePayload):
            raise TypeError("payload must be a RetrievalCachePayload")
        with self._revision_lock:
            if self._active_revision_id != payload.index_revision:
                return RetrievalCacheWriteStatus.SKIPPED_OBSOLETE
            try:
                serialized_payload = payload.model_dump_json()
                serialized = json.dumps(
                    {
                        "version": _RETRIEVAL_CACHE_ENVELOPE_VERSION,
                        "payload": serialized_payload,
                        "signature": self._payload_signature(key, serialized_payload),
                    },
                    ensure_ascii=True,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                outcome = self._backend.put_with_outcome(key, serialized)
            except Exception:
                self._metrics.record_error()
                return RetrievalCacheWriteStatus.ERROR
            self._metrics.record_write(
                expirations=outcome.expired_count,
                evictions=outcome.eviction_count,
            )
            return RetrievalCacheWriteStatus.STORED

    def _authenticated_payload(self, key: str, serialized: str) -> RetrievalCachePayload:
        envelope = json.loads(serialized)
        if not isinstance(envelope, dict) or set(envelope) != {
            "version",
            "payload",
            "signature",
        }:
            raise ValueError("cached retrieval envelope is invalid")
        if envelope["version"] != _RETRIEVAL_CACHE_ENVELOPE_VERSION:
            raise ValueError("cached retrieval envelope version is invalid")
        serialized_payload = envelope["payload"]
        signature = envelope["signature"]
        if not isinstance(serialized_payload, str) or not isinstance(signature, str):
            raise TypeError("cached retrieval envelope fields are invalid")
        expected = self._payload_signature(key, serialized_payload)
        if not hmac.compare_digest(signature, expected):
            raise ValueError("cached retrieval envelope authentication failed")
        return RetrievalCachePayload.model_validate_json(serialized_payload)

    def _payload_signature(self, key: str, serialized_payload: str) -> str:
        authenticated = f"{key}\0{serialized_payload}".encode()
        return hmac.new(self._integrity_key, authenticated, hashlib.sha256).hexdigest()

    def record_bypass(self) -> None:
        self._metrics.record_bypass()

    def record_lookup_error(self) -> None:
        """Record identity/build failures as eligible fail-open cache misses."""

        self._metrics.record_miss(error=True)

    def clear(self) -> None:
        with self._revision_lock:
            try:
                self._backend.clear()
            except Exception:
                self._metrics.record_error()


def _require_cache_digest(key: object) -> None:
    if not isinstance(key, str) or _CACHE_KEY.fullmatch(key) is None:
        raise ValueError("cache key must be a SHA-256 digest")


def _require_nonempty(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_aware_datetime(value: object, name: str) -> None:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
