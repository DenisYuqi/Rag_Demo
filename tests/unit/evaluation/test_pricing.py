from __future__ import annotations

from decimal import Decimal

from rag_mvp.domain.evaluation import (
    ModelAttempt,
    ModelAttemptStatus,
    ModelRole,
    TokenUsage,
)
from rag_mvp.evaluation.pricing import (
    OPENAI_STANDARD_PRICING_VERSION,
    openai_standard_pricing_catalog,
)


def _attempt(role: ModelRole, model: str, usage: TokenUsage) -> ModelAttempt:
    return ModelAttempt(
        attempt_id=f"attempt-{role.value}",
        operation_id="operation",
        request_id="request",
        role=role,
        provider="openai-compatible-test",
        model=model,
        status=ModelAttemptStatus.SUCCEEDED,
        latency_ms=1,
        usage=usage,
    )


def test_pinned_catalog_prices_generation_and_embedding_without_fake_output_tokens() -> None:
    catalog = openai_standard_pricing_catalog(
        provider="openai-compatible-test",
        models=("gpt-5.4", "text-embedding-3-small"),
    )
    generation = catalog.estimate_attempt(
        _attempt(
            ModelRole.GENERATION,
            "gpt-5.4",
            TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000),
        )
    )
    embedding = catalog.estimate_attempt(
        _attempt(
            ModelRole.EMBEDDING,
            "text-embedding-3-small",
            TokenUsage(input_tokens=1_000_000),
        )
    )

    assert catalog.version == OPENAI_STANDARD_PRICING_VERSION
    assert generation.complete is True
    assert generation.estimated_cost == Decimal("17.50")
    assert embedding.complete is True
    assert embedding.estimated_cost == Decimal("0.02")


def test_unknown_model_is_explicitly_unpriced() -> None:
    catalog = openai_standard_pricing_catalog(
        provider="openai-compatible-test",
        models=("unknown-model",),
    )
    estimate = catalog.estimate_attempt(
        _attempt(
            ModelRole.GENERATION,
            "unknown-model",
            TokenUsage(input_tokens=1, output_tokens=1),
        )
    )

    assert estimate.complete is False
    assert estimate.estimated_cost is None
    assert tuple(reason.value for reason in estimate.unknown_reasons) == ("pricing-not-found",)


def test_pinned_catalog_prices_gpt_4_1_mini_acceptance_model() -> None:
    catalog = openai_standard_pricing_catalog(
        provider="openai-compatible-test",
        models=("gpt-4.1-mini",),
    )

    estimate = catalog.estimate_attempt(
        _attempt(
            ModelRole.GENERATION,
            "gpt-4.1-mini",
            TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000),
        )
    )

    assert estimate.complete is True
    assert estimate.estimated_cost == Decimal("2.00")
