"""Runtime telemetry coordination for validated QA and background ingestion."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Protocol

from structlog.typing import FilteringBoundLogger

from rag_mvp.config.settings import Settings
from rag_mvp.domain.ingestion import IngestionJob, IngestionJobStatus
from rag_mvp.domain.qa import (
    QAErrorCode,
    RequestDiagnostic,
    SafeQADiagnostics,
    StreamEventKind,
    ValidatedStreamEvent,
)
from rag_mvp.observability.logging import (
    SafeErrorCategory,
    bind_correlation_context,
    classify_exception,
    current_correlation_context,
    get_logger,
)
from rag_mvp.observability.metrics import (
    CacheName,
    CacheOutcome,
    DegradationReason,
    ModelRoleLabel,
    PipelineStage,
    QAOutcome,
    RAGMetrics,
    RetrievalCountKind,
    TokenDirection,
)
from rag_mvp.observability.tracing import RAGTracer, SafeSpan


class DiagnosticSink(Protocol):
    def save(self, diagnostic: RequestDiagnostic) -> RequestDiagnostic: ...


_STAGE_LABELS: dict[str, PipelineStage] = {
    "queue": PipelineStage.QUEUE,
    "validation": PipelineStage.VALIDATION,
    "safety": PipelineStage.SAFETY,
    "query_embedding": PipelineStage.QUERY_EMBEDDING,
    "embedding": PipelineStage.EMBEDDING,
    "retrieval": PipelineStage.RETRIEVAL,
    "dense": PipelineStage.DENSE,
    "bm25": PipelineStage.BM25,
    "fusion": PipelineStage.FUSION,
    "rerank": PipelineStage.RERANKING,
    "reranking": PipelineStage.RERANKING,
    "evidence_assessment": PipelineStage.EVIDENCE_ASSESSMENT,
    "generation": PipelineStage.GENERATION,
    "grounding": PipelineStage.GROUNDING,
    "redaction": PipelineStage.REDACTION,
    "finalization": PipelineStage.FINALIZATION,
    "serialization": PipelineStage.SERIALIZATION,
    "ingestion": PipelineStage.INGESTION,
    "evaluation": PipelineStage.EVALUATION,
}
_CACHE_LABELS: dict[str, CacheName] = {
    "document_embedding": CacheName.DOCUMENT_EMBEDDING,
    "document-embedding": CacheName.DOCUMENT_EMBEDDING,
    "query_embedding": CacheName.QUERY_EMBEDDING,
    "query-embedding": CacheName.QUERY_EMBEDDING,
    "retrieval": CacheName.RETRIEVAL,
    "rerank": CacheName.RERANKING,
    "reranking": CacheName.RERANKING,
    "final": CacheName.ANSWER,
    "answer": CacheName.ANSWER,
}
_ROLE_LABELS: dict[str, ModelRoleLabel] = {
    "embedding": ModelRoleLabel.EMBEDDING,
    "generation": ModelRoleLabel.GENERATION,
    "reranker": ModelRoleLabel.RERANKING,
    "reranking": ModelRoleLabel.RERANKING,
    "evaluation": ModelRoleLabel.EVALUATION,
}
_DIRECT_STAGE_METRICS = frozenset(
    {
        "queue",
        "validation",
        "safety",
        "retrieval",
        "evidence_assessment",
        "generation",
        "grounding",
        "redaction",
        "finalization",
        "serialization",
    }
)


@dataclass(slots=True)
class RequestObservation:
    """One QA trace kept open until a validated terminal event is available."""

    telemetry: PipelineTelemetry
    request_id: str
    started_at: float
    root_span: SafeSpan
    trace_id: str
    completed: bool = False

    def complete(self, event: ValidatedStreamEvent) -> None:
        """Record only the validated event contract; telemetry failures stay non-fatal."""

        if self.completed:
            return
        self.completed = True
        try:
            self.telemetry._record_terminal(self, event)
        except Exception:
            self.telemetry.metrics.record_telemetry_drop()


class PipelineTelemetry:
    """Keep logs, metrics, traces, and persisted diagnostics on one correlation."""

    def __init__(
        self,
        settings: Settings,
        *,
        metrics: RAGMetrics | None = None,
        tracer: RAGTracer | None = None,
        diagnostics: DiagnosticSink | None = None,
    ) -> None:
        self.settings = settings
        self.metrics = metrics or RAGMetrics()
        self.tracer = tracer or RAGTracer()
        self.diagnostics = diagnostics

    @property
    def logger(self) -> FilteringBoundLogger:
        """Resolve against the current process logging configuration."""

        return get_logger("qa")

    @property
    def ingestion_logger(self) -> FilteringBoundLogger:
        """Resolve against the current process logging configuration."""

        return get_logger("ingestion")

    @asynccontextmanager
    async def request(self, request_id: str) -> AsyncIterator[RequestObservation]:
        outer = current_correlation_context()
        parent_trace_id = outer.trace_id if outer is not None else None
        started = time.perf_counter()
        async with self.tracer.request_span(
            request_id=request_id,
            operation="qa",
            config_version=self.settings.configuration_identity,
            trace_id=parent_trace_id,
        ) as root_span:
            trace_id = root_span.reference.trace_id
            with bind_correlation_context(request_id, trace_id):
                observation = RequestObservation(self, request_id, started, root_span, trace_id)
                self.logger.info("qa.request.started", outcome="started")
                try:
                    yield observation
                except asyncio.CancelledError:
                    self._record_aborted(
                        observation,
                        outcome=QAOutcome.CANCELLED,
                        category=SafeErrorCategory.CANCELLED,
                    )
                    raise
                except Exception:
                    self._record_aborted(
                        observation,
                        outcome=QAOutcome.ERROR,
                        category=SafeErrorCategory.INTERNAL,
                    )
                    raise
                else:
                    if not observation.completed:
                        self._record_aborted(
                            observation,
                            outcome=QAOutcome.ERROR,
                            category=SafeErrorCategory.INTERNAL,
                        )

    @asynccontextmanager
    async def stage(self, stage: str) -> AsyncIterator[None]:
        """Create one content-free child span and correlated timing event."""

        normalized = stage.strip().lower().replace("-", "_")
        label = _STAGE_LABELS.get(normalized)
        if label is None:
            raise ValueError("stage is not allowlisted")
        started = time.perf_counter()
        async with self.tracer.stage_span(label):
            try:
                yield
            except BaseException as error:
                duration_seconds = max(0.0, time.perf_counter() - started)
                self.metrics.observe_stage(label, duration_seconds)
                self.logger.error(
                    "qa.stage.failed",
                    stage=label.value,
                    outcome="failed",
                    safe_error_category=classify_exception(error).value,
                    stage_duration_ms=duration_seconds * 1_000,
                )
                raise
            else:
                duration_seconds = max(0.0, time.perf_counter() - started)
                self.metrics.observe_stage(label, duration_seconds)
                self.logger.info(
                    "qa.stage.completed",
                    stage=label.value,
                    outcome="succeeded",
                    stage_duration_ms=duration_seconds * 1_000,
                )

    def record_ingestion(self, job: IngestionJob) -> None:
        """Emit content-free completion evidence for a background ingestion job."""

        duration_ms = sum(job.stage_timings_ms.values())
        self.metrics.observe_stage(PipelineStage.INGESTION, duration_ms / 1_000)
        outcome = "succeeded" if job.status is IngestionJobStatus.SUCCEEDED else "failed"
        self.ingestion_logger.info(
            "ingestion.job.completed",
            operation=job.operation.value,
            outcome=outcome,
            stage_duration_ms=duration_ms,
            counts={"chunks": job.chunk_count, "ocr_pages": job.ocr_page_count},
            safe_error_category=job.safe_error_code,
        )

    async def flush(self, timeout_seconds: float) -> bool:
        """Bound exporter and logging flushes during graceful shutdown."""

        if isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        timeout_millis = max(1, int(timeout_seconds * 1_000))

        def flush_sync() -> bool:
            flushed = self.tracer.force_flush(timeout_millis)
            for handler in tuple(logging.getLogger().handlers):
                handler.flush()
            sys.stdout.flush()
            sys.stderr.flush()
            return flushed

        try:
            async with asyncio.timeout(timeout_seconds):
                return await asyncio.to_thread(flush_sync)
        except TimeoutError:
            return False

    def _record_terminal(
        self,
        observation: RequestObservation,
        event: ValidatedStreamEvent,
    ) -> None:
        outcome, error_category = _event_outcome(event)
        duration_seconds = max(0.0, time.perf_counter() - observation.started_at)
        self.metrics.record_qa(
            outcome=outcome,
            duration_seconds=duration_seconds,
            error_category=error_category.value if error_category is not None else "none",
        )
        diagnostics = event.diagnostics
        for name, duration_ms in diagnostics.stage_timings_ms.items():
            stage = _STAGE_LABELS.get(name)
            if stage is not None and name not in _DIRECT_STAGE_METRICS:
                self.metrics.observe_stage(stage, duration_ms / 1_000)
        for name, raw_outcome in diagnostics.cache_status.items():
            cache = _CACHE_LABELS.get(name)
            try:
                cache_outcome = CacheOutcome(raw_outcome)
            except ValueError:
                continue
            if cache is not None:
                self.metrics.record_cache(cache, cache_outcome)
        for key, count in diagnostics.token_counts.items():
            role_name, separator, direction_name = key.partition("-")
            role = _ROLE_LABELS.get(role_name)
            if role is None or not separator or direction_name not in {"input", "output"}:
                continue
            self.metrics.record_tokens(
                role=role,
                direction=TokenDirection(direction_name),
                count=count,
            )
        for reason in diagnostics.degradation_reasons:
            self.metrics.record_degradation(_degradation(reason))
        count_labels = {
            "dense_candidate_count": RetrievalCountKind.DENSE,
            "lexical_candidate_count": RetrievalCountKind.LEXICAL,
            "fused_candidate_count": RetrievalCountKind.FUSED,
            "reranked_candidate_count": RetrievalCountKind.RERANKED,
            "context_count": RetrievalCountKind.CONTEXT,
        }
        for key, label in count_labels.items():
            candidate_count_value = diagnostics.metadata.get(key)
            if isinstance(candidate_count_value, int) and not isinstance(
                candidate_count_value, bool
            ):
                self.metrics.observe_retrieval_count(label, candidate_count_value)
        self.metrics.observe_retrieval_count(RetrievalCountKind.CITATION, len(event.citations))

        observation.root_span.set_outcome(outcome)
        if error_category is not None:
            observation.root_span.set_error(error_category)
        _enrich_root_span(observation.root_span, diagnostics)

        metadata = _diagnostic_metadata(event, self.settings)
        diagnostic = RequestDiagnostic(
            request_id=event.request_id,
            session_id=event.session_id,
            trace_id=observation.trace_id,
            outcome=outcome.value,
            safe_error_category=(error_category.value if error_category is not None else None),
            stage_timings_ms=dict(diagnostics.stage_timings_ms),
            cache_status=dict(diagnostics.cache_status),
            model_identities=dict(diagnostics.model_identities),
            token_counts=dict(diagnostics.token_counts),
            metadata=metadata,
        )
        if self.diagnostics is not None:
            try:
                self.diagnostics.save(diagnostic)
            except Exception:
                self.metrics.record_telemetry_drop()
        self.logger.info(
            "qa.request.completed",
            outcome=outcome.value,
            safe_error_category=(error_category.value if error_category is not None else None),
            duration_ms=duration_seconds * 1_000,
            cache_status=dict(diagnostics.cache_status),
            counts={"citations": len(event.citations)},
            model_identity=dict(diagnostics.model_identities),
            token_usage=dict(diagnostics.token_counts),
            degraded_reason=list(diagnostics.degradation_reasons),
        )

    def _record_aborted(
        self,
        observation: RequestObservation,
        *,
        outcome: QAOutcome,
        category: SafeErrorCategory,
    ) -> None:
        if observation.completed:
            return
        observation.completed = True
        duration = max(0.0, time.perf_counter() - observation.started_at)
        self.metrics.record_qa(
            outcome=outcome,
            duration_seconds=duration,
            error_category=category.value,
        )
        observation.root_span.set_outcome(outcome)
        observation.root_span.set_error(category)
        self.logger.error(
            "qa.request.failed",
            outcome=outcome.value,
            safe_error_category=category.value,
            duration_ms=duration * 1_000,
        )


def _event_outcome(
    event: ValidatedStreamEvent,
) -> tuple[QAOutcome, SafeErrorCategory | None]:
    if event.kind is StreamEventKind.ANSWER:
        return QAOutcome.ANSWER, None
    if event.kind is StreamEventKind.REFUSAL:
        return QAOutcome.REFUSAL, None
    if event.error_code is QAErrorCode.DEADLINE_EXPIRED:
        return QAOutcome.TIMEOUT, SafeErrorCategory.TIMEOUT
    if event.error_code is QAErrorCode.CAPACITY:
        return QAOutcome.ERROR, SafeErrorCategory.CAPACITY
    if event.error_code in {
        QAErrorCode.INDEX_NOT_READY,
        QAErrorCode.RETRIEVAL_UNAVAILABLE,
        QAErrorCode.DEPENDENCY_FAILURE,
    }:
        return QAOutcome.ERROR, SafeErrorCategory.DEPENDENCY
    if event.error_code is QAErrorCode.SAFETY_UNAVAILABLE:
        return QAOutcome.ERROR, SafeErrorCategory.UNAVAILABLE
    return QAOutcome.ERROR, SafeErrorCategory.INTERNAL


def _diagnostic_metadata(
    event: ValidatedStreamEvent,
    settings: Settings,
) -> dict[str, str | int | float | bool | None]:
    source = event.diagnostics.metadata
    metadata: dict[str, str | int | float | bool | None] = {
        "configuration_id": settings.configuration_identity,
        "citation_count": len(event.citations),
    }
    if isinstance(source.get("index_revision"), str):
        metadata["index_revision"] = source["index_revision"]
    mode = source.get("effective_mode") or source.get("requested_mode")
    if isinstance(mode, str):
        metadata["retrieval_mode"] = mode
    candidate_count = source.get("candidate_count")
    if isinstance(candidate_count, int) and not isinstance(candidate_count, bool):
        metadata["candidate_count"] = candidate_count
    for key in (
        "context_count",
        "dense_candidate_count",
        "lexical_candidate_count",
        "fused_candidate_count",
        "reranked_candidate_count",
    ):
        count = source.get(key)
        if isinstance(count, int) and not isinstance(count, bool):
            metadata[key] = count
    for key in (
        "refusal_reason_code",
        "refusal_guidance_reason_code",
        "refusal_guidance_template_id",
        "refusal_guidance_catalog_version",
        "refusal_guidance_language",
    ):
        value = source.get(key)
        if isinstance(value, str):
            metadata[key] = value
    guidance_present = source.get("refusal_guidance_present")
    if type(guidance_present) is bool:
        metadata["refusal_guidance_present"] = guidance_present
    return metadata


def _degradation(reason: str) -> DegradationReason:
    normalized = reason.replace("_", "-")
    if "rerank" in normalized and "timeout" in normalized:
        return DegradationReason.RERANK_TIMEOUT
    if "rerank" in normalized:
        return DegradationReason.RERANK_FAILURE
    if "fallback" in normalized:
        return DegradationReason.PROVIDER_FALLBACK
    if "dense" in normalized or "bm25" in normalized or "retriever" in normalized:
        return DegradationReason.SINGLE_RETRIEVER
    if "deadline" in normalized:
        return DegradationReason.DEADLINE_MARGIN
    return DegradationReason.OTHER


def _enrich_root_span(span: SafeSpan, diagnostics: SafeQADiagnostics) -> None:
    """Attach bounded aggregate diagnostics to the correlated QA root span."""

    token_counts = diagnostics.token_counts
    if isinstance(token_counts, dict):
        input_tokens = 0
        output_tokens = 0
        has_input = False
        has_output = False
        for name, count in token_counts.items():
            if not isinstance(name, str) or type(count) is not int or count < 0:
                continue
            normalized = name.strip().casefold().replace("_", "-")
            if normalized == "input" or normalized.endswith("-input"):
                input_tokens += count
                has_input = True
            elif normalized == "output" or normalized.endswith("-output"):
                output_tokens += count
                has_output = True
        span.set_token_usage(
            input_tokens=input_tokens if has_input else None,
            output_tokens=output_tokens if has_output else None,
        )

    cache_status = diagnostics.cache_status
    if isinstance(cache_status, dict):
        outcomes: list[CacheOutcome] = []
        for raw_outcome in cache_status.values():
            try:
                outcomes.append(CacheOutcome(raw_outcome))
            except (TypeError, ValueError):
                continue
        selected_cache = (
            CacheOutcome.BYPASS
            if CacheOutcome.BYPASS in outcomes
            else outcomes[0]
            if outcomes and len(set(outcomes)) == 1
            else None
        )
        if selected_cache is not None:
            span.set_cache_outcome(selected_cache)

    degradation_reasons = diagnostics.degradation_reasons
    if isinstance(degradation_reasons, tuple) and degradation_reasons:
        span.set_degradation(_degradation(degradation_reasons[0]))

    identities = diagnostics.model_identities
    if isinstance(identities, dict):
        for role in ("generation", "reranker", "reranking", "embedding"):
            identity = identities.get(role)
            if not isinstance(identity, str):
                continue
            parts = identity.split("/")
            provider_alias = parts[0]
            model_alias = parts[1] if len(parts) > 1 else identity
            attached = False
            try:
                span.set_provider_alias(provider_alias)
                attached = True
            except ValueError:
                pass
            try:
                span.set_model_alias(model_alias)
                attached = True
            except ValueError:
                pass
            if attached:
                break
