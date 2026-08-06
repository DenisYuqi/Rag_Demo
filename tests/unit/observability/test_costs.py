from __future__ import annotations

from decimal import Decimal

from rag_mvp.domain.evaluation import (
    ModelAttempt,
    ModelAttemptStatus,
    ModelPricing,
    ModelRole,
    TokenUsage,
)
from rag_mvp.observability.costs import (
    PricingCatalog,
    UnknownCostReason,
    project_per_thousand_calls,
)


def _attempt(
    attempt_id: str,
    *,
    request_id: str = "request-1",
    run_id: str = "run-1",
    provider: str = "primary",
    model: str = "chat-v1",
    usage: TokenUsage | None = None,
) -> ModelAttempt:
    return ModelAttempt(
        attempt_id=attempt_id,
        operation_id="operation-1",
        request_id=request_id,
        run_id=run_id,
        role=ModelRole.GENERATION,
        provider=provider,
        model=model,
        status=ModelAttemptStatus.SUCCEEDED,
        latency_ms=10,
        usage=usage or TokenUsage(input_tokens=1_000, output_tokens=500),
    )


def _catalog(*entries: ModelPricing) -> PricingCatalog:
    return PricingCatalog(version="pricing-2026-08", entries=tuple(entries))


def test_per_attempt_request_run_and_per_thousand_costs_are_exact() -> None:
    catalog = _catalog(
        ModelPricing(
            pricing_version="pricing-2026-08",
            provider="primary",
            model="chat-v1",
            currency="USD",
            input_per_million=Decimal("2"),
            output_per_million=Decimal("8"),
        )
    )
    attempts = (_attempt("attempt-1"), _attempt("attempt-2"))

    per_attempt = catalog.estimate_attempt(attempts[0])
    assert per_attempt.complete is True
    assert per_attempt.estimated_cost == Decimal("0.006")

    request = catalog.aggregate_request(attempts, request_id="request-1")
    run = catalog.aggregate_run(attempts, run_id="run-1")
    assert request.estimated_cost == Decimal("0.012")
    assert request.input_tokens == 2_000
    assert request.output_tokens == 1_000
    assert run.estimated_cost == request.estimated_cost
    assert run.roles[0].estimated_cost == Decimal("0.012")

    projected = project_per_thousand_calls(run, successful_calls=2)
    assert projected.complete is True
    assert projected.estimated_cost_per_1000 == Decimal("6.000")
    assert projected.currency == "USD"


def test_unknown_usage_or_pricing_is_never_represented_as_zero() -> None:
    catalog = _catalog(
        ModelPricing(
            pricing_version="pricing-2026-08",
            provider="primary",
            model="chat-v1",
            currency="USD",
            input_per_million=Decimal("2"),
            output_per_million=Decimal("8"),
        )
    )
    unknown_usage = catalog.estimate_attempt(
        _attempt("attempt-unknown", usage=TokenUsage(input_tokens=10))
    )
    unknown_pricing = catalog.estimate_attempt(_attempt("attempt-unpriced", provider="secondary"))

    assert unknown_usage.estimated_cost is None
    assert unknown_usage.known_cost == Decimal("0.00002")
    assert UnknownCostReason.OUTPUT_USAGE_UNKNOWN in unknown_usage.unknown_reasons
    assert unknown_pricing.estimated_cost is None
    assert unknown_pricing.known_cost is None
    assert unknown_pricing.unknown_reasons == (UnknownCostReason.PRICING_NOT_FOUND,)

    aggregate = catalog.aggregate_request(
        (
            _attempt("attempt-known"),
            _attempt("attempt-unpriced", provider="secondary"),
        ),
        request_id="request-1",
    )
    assert aggregate.complete is False
    assert aggregate.estimated_cost is None
    assert (
        project_per_thousand_calls(
            aggregate,
            successful_calls=1,
        ).estimated_cost_per_1000
        is None
    )


def test_pricing_catalog_requires_one_frozen_version() -> None:
    mismatched = ModelPricing(
        pricing_version="old",
        provider="primary",
        model="chat-v1",
        currency="USD",
        input_per_million=Decimal("1"),
        output_per_million=Decimal("1"),
    )

    try:
        _catalog(mismatched)
    except ValueError as error:
        assert "version" in str(error)
    else:
        raise AssertionError("a mismatched pricing version must fail")
