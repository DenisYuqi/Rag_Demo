"""Bounded-cardinality Prometheus metrics for the RAG pipeline."""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from decimal import Decimal
from enum import StrEnum

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class MetricLabelError(ValueError):
    """Raised when a metric label is outside its fixed vocabulary."""


class QAOutcome(StrEnum):
    ANSWER = "answer"
    REFUSAL = "refusal"
    ERROR = "error"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class PipelineStage(StrEnum):
    QUEUE = "queue"
    VALIDATION = "validation"
    SAFETY = "safety"
    QUERY_EMBEDDING = "query-embedding"
    EMBEDDING = "embedding"
    RETRIEVAL = "retrieval"
    DENSE = "dense"
    BM25 = "bm25"
    FUSION = "fusion"
    RERANKING = "reranking"
    EVIDENCE_ASSESSMENT = "evidence-assessment"
    GENERATION = "generation"
    GROUNDING = "grounding"
    REDACTION = "redaction"
    FINALIZATION = "finalization"
    SERIALIZATION = "serialization"
    INGESTION = "ingestion"
    EVALUATION = "evaluation"


class CacheName(StrEnum):
    DOCUMENT_EMBEDDING = "document-embedding"
    QUERY_EMBEDDING = "query-embedding"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    ANSWER = "answer"


class CacheOutcome(StrEnum):
    HIT = "hit"
    MISS = "miss"
    BYPASS = "bypass"
    DISABLED = "disabled"
    NOT_APPLICABLE = "not-applicable"
    ERROR = "error"


class ModelRoleLabel(StrEnum):
    EMBEDDING = "embedding"
    GENERATION = "generation"
    RERANKING = "reranking"
    EVALUATION = "evaluation"


class TokenDirection(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


class RetrievalCountKind(StrEnum):
    DENSE = "dense"
    LEXICAL = "lexical"
    FUSED = "fused"
    RERANKED = "reranked"
    CONTEXT = "context"
    CITATION = "citation"


class DegradationReason(StrEnum):
    RERANK_TIMEOUT = "rerank-timeout"
    RERANK_FAILURE = "rerank-failure"
    SINGLE_RETRIEVER = "single-retriever"
    PROVIDER_FALLBACK = "provider-fallback"
    DEADLINE_MARGIN = "deadline-margin"
    OTHER = "other"


_ERROR_CATEGORIES = frozenset(
    {
        "none",
        "validation",
        "capacity",
        "timeout",
        "cancelled",
        "authentication",
        "rate-limit",
        "dependency",
        "unavailable",
        "internal",
    }
)
_QUEUE_REASONS = frozenset({"capacity", "shutdown"})
_CURRENCIES = frozenset({"USD", "CNY", "EUR", "GBP", "JPY"})


def _enum_value[T: StrEnum](value: T | str, enum_type: type[T], field: str) -> str:
    try:
        return enum_type(value).value
    except ValueError as error:
        raise MetricLabelError(f"unsupported {field} label") from error


def _allowed(value: str, choices: Iterable[str], field: str) -> str:
    if value not in choices:
        raise MetricLabelError(f"unsupported {field} label")
    return value


def _non_negative(value: int | float | Decimal, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be non-negative")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return converted


class RAGMetrics:
    """Metrics facade that makes unbounded labels impossible through its public API."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self.qa_requests = Counter(
            "rag_mvp_qa_requests_total",
            "QA requests by terminal outcome and safe error category.",
            ("outcome", "error_category"),
            registry=self.registry,
        )
        self.qa_duration = Histogram(
            "rag_mvp_qa_duration_seconds",
            "Complete QA request latency.",
            ("outcome",),
            buckets=(0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 15, 30),
            registry=self.registry,
        )
        self.qa_in_flight = Gauge(
            "rag_mvp_qa_in_flight",
            "Currently active QA pipelines.",
            registry=self.registry,
        )
        self.qa_queue_depth = Gauge(
            "rag_mvp_qa_queue_depth",
            "Currently queued QA requests.",
            registry=self.registry,
        )
        self.queue_rejections = Counter(
            "rag_mvp_qa_queue_rejections_total",
            "QA requests rejected before admission.",
            ("reason",),
            registry=self.registry,
        )
        self.stage_duration = Histogram(
            "rag_mvp_stage_duration_seconds",
            "Latency for bounded RAG stages.",
            ("stage",),
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
            registry=self.registry,
        )
        self.cache_access = Counter(
            "rag_mvp_cache_access_total",
            "Cache outcomes by bounded cache identity.",
            ("cache", "outcome"),
            registry=self.registry,
        )
        self.retrieval_counts = Histogram(
            "rag_mvp_retrieval_candidates",
            "Candidate and evidence counts by pipeline point.",
            ("kind",),
            buckets=(0, 1, 2, 3, 5, 10, 20, 50, 100),
            registry=self.registry,
        )
        self.degradations = Counter(
            "rag_mvp_degradations_total",
            "Safe bounded degradation reasons.",
            ("reason",),
            registry=self.registry,
        )
        self.tokens = Counter(
            "rag_mvp_provider_tokens_total",
            "Provider-reported token usage.",
            ("role", "direction"),
            registry=self.registry,
        )
        self.estimated_cost = Counter(
            "rag_mvp_estimated_cost_total",
            "Known estimated provider cost; unknown cost is not recorded as zero.",
            ("role", "currency"),
            registry=self.registry,
        )
        self.telemetry_drops = Counter(
            "rag_mvp_telemetry_dropped_total",
            "Events dropped because safe filtering could not be established.",
            registry=self.registry,
        )

    def record_qa(
        self,
        *,
        outcome: QAOutcome | str,
        duration_seconds: float,
        error_category: str = "none",
    ) -> None:
        safe_outcome = _enum_value(outcome, QAOutcome, "outcome")
        safe_error = _allowed(error_category, _ERROR_CATEGORIES, "error_category")
        duration = _non_negative(duration_seconds, "duration_seconds")
        self.qa_requests.labels(outcome=safe_outcome, error_category=safe_error).inc()
        self.qa_duration.labels(outcome=safe_outcome).observe(duration)

    @asynccontextmanager
    async def active_pipeline(self) -> AsyncIterator[None]:
        """Track one active pipeline and always restore the gauge after cancellation."""

        self.pipeline_started()
        try:
            yield
        finally:
            self.pipeline_finished()

    def pipeline_started(self) -> None:
        self.qa_in_flight.inc()

    def pipeline_finished(self) -> None:
        self.qa_in_flight.dec()

    def set_queue_depth(self, depth: int) -> None:
        self.qa_queue_depth.set(_non_negative(depth, "queue depth"))

    def record_queue_rejection(self, reason: str = "capacity") -> None:
        self.queue_rejections.labels(reason=_allowed(reason, _QUEUE_REASONS, "reason")).inc()

    def observe_stage(self, stage: PipelineStage | str, duration_seconds: float) -> None:
        safe_stage = _enum_value(stage, PipelineStage, "stage")
        self.stage_duration.labels(stage=safe_stage).observe(
            _non_negative(duration_seconds, "duration_seconds")
        )

    def record_cache(self, cache: CacheName | str, outcome: CacheOutcome | str) -> None:
        self.cache_access.labels(
            cache=_enum_value(cache, CacheName, "cache"),
            outcome=_enum_value(outcome, CacheOutcome, "cache outcome"),
        ).inc()

    def observe_retrieval_count(self, kind: RetrievalCountKind | str, count: int) -> None:
        self.retrieval_counts.labels(
            kind=_enum_value(kind, RetrievalCountKind, "retrieval count")
        ).observe(_non_negative(count, "retrieval count"))

    def record_degradation(self, reason: DegradationReason | str) -> None:
        self.degradations.labels(
            reason=_enum_value(reason, DegradationReason, "degradation reason")
        ).inc()

    def record_tokens(
        self,
        *,
        role: ModelRoleLabel | str,
        direction: TokenDirection | str,
        count: int,
    ) -> None:
        self.tokens.labels(
            role=_enum_value(role, ModelRoleLabel, "model role"),
            direction=_enum_value(direction, TokenDirection, "token direction"),
        ).inc(_non_negative(count, "token count"))

    def record_cost(
        self,
        *,
        role: ModelRoleLabel | str,
        currency: str,
        amount: Decimal,
    ) -> None:
        self.estimated_cost.labels(
            role=_enum_value(role, ModelRoleLabel, "model role"),
            currency=_allowed(currency.upper(), _CURRENCIES, "currency"),
        ).inc(_non_negative(amount, "cost"))

    def record_telemetry_drop(self) -> None:
        self.telemetry_drops.inc()

    def render(self) -> bytes:
        """Render this isolated registry in Prometheus exposition format."""

        return generate_latest(self.registry)
