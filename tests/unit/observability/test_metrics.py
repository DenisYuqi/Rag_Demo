from __future__ import annotations

from decimal import Decimal

import pytest
from prometheus_client import CollectorRegistry

from rag_mvp.observability.metrics import (
    CacheName,
    CacheOutcome,
    DegradationReason,
    MetricLabelError,
    ModelRoleLabel,
    PipelineStage,
    QAOutcome,
    RAGMetrics,
    RetrievalCountKind,
    TokenDirection,
)


def test_metrics_cover_qa_stages_cache_tokens_cost_and_degradation() -> None:
    metrics = RAGMetrics(CollectorRegistry())
    metrics.record_qa(outcome=QAOutcome.ANSWER, duration_seconds=1.25)
    metrics.observe_stage(PipelineStage.GENERATION, 0.75)
    metrics.record_cache(CacheName.RETRIEVAL, CacheOutcome.BYPASS)
    metrics.observe_retrieval_count(RetrievalCountKind.CONTEXT, 5)
    metrics.record_degradation(DegradationReason.RERANK_TIMEOUT)
    metrics.record_tokens(
        role=ModelRoleLabel.GENERATION,
        direction=TokenDirection.INPUT,
        count=120,
    )
    metrics.record_cost(
        role=ModelRoleLabel.GENERATION,
        currency="USD",
        amount=Decimal("0.0012"),
    )
    metrics.set_queue_depth(2)
    metrics.record_queue_rejection()

    rendered = metrics.render().decode("utf-8")
    assert 'rag_mvp_qa_requests_total{error_category="none",outcome="answer"} 1.0' in rendered
    assert 'rag_mvp_stage_duration_seconds_count{stage="generation"} 1.0' in rendered
    assert 'rag_mvp_cache_access_total{cache="retrieval",outcome="bypass"} 1.0' in rendered
    assert 'rag_mvp_provider_tokens_total{direction="input",role="generation"} 120.0' in rendered
    assert 'rag_mvp_estimated_cost_total{currency="USD",role="generation"} 0.0012' in rendered
    assert 'rag_mvp_degradations_total{reason="rerank-timeout"} 1.0' in rendered
    assert "request_id" not in rendered
    assert "question" not in rendered


@pytest.mark.asyncio
async def test_active_pipeline_gauge_is_restored() -> None:
    metrics = RAGMetrics(CollectorRegistry())
    async with metrics.active_pipeline():
        assert "rag_mvp_qa_in_flight 1.0" in metrics.render().decode("utf-8")
    assert "rag_mvp_qa_in_flight 0.0" in metrics.render().decode("utf-8")


def test_unbounded_or_unknown_labels_are_rejected() -> None:
    metrics = RAGMetrics(CollectorRegistry())

    with pytest.raises(MetricLabelError):
        metrics.record_qa(
            outcome="request-person@example.com",
            duration_seconds=1,
        )
    with pytest.raises(MetricLabelError):
        metrics.observe_stage("question-123", 1)
    with pytest.raises(MetricLabelError):
        metrics.record_cost(
            role=ModelRoleLabel.GENERATION,
            currency="request-1",
            amount=Decimal("1"),
        )
