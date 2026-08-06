from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_mvp.domain.evaluation import (
    ModelAttemptStatus,
    ModelRole,
    ProviderAttemptEvidence,
    TokenUsage,
)
from rag_mvp.performance.load_report import (
    LoadAcceptanceThresholds,
    LoadAttempt,
    LoadAttemptStatus,
    LoadReport,
    build_load_report,
    build_warmup_summary,
)
from rag_mvp.performance.pricing import (
    PerformancePricingEvidence,
    PerformanceRolePricing,
    calculate_performance_cost,
)

_START = datetime(2026, 8, 7, tzinfo=UTC)


def _attempt(
    identifier: str,
    *,
    warmup: bool = False,
    token_counts: dict[str, int] | None = None,
    provider_attempt_count: int = 2,
) -> LoadAttempt:
    started_at = _START if warmup else _START + timedelta(seconds=1)
    provider_attempts = (
        ProviderAttemptEvidence(
            operation_id="qa-retrieval",
            route_id="embedding-primary",
            role=ModelRole.EMBEDDING,
            provider="openai",
            model="text-embedding-3-small",
            status=ModelAttemptStatus.SUCCEEDED,
            usage=TokenUsage(input_tokens=100),
        ),
        ProviderAttemptEvidence(
            operation_id="qa-generation",
            route_id="generation-primary",
            role=ModelRole.GENERATION,
            provider="openai",
            model="gpt-4.1-mini",
            status=ModelAttemptStatus.SUCCEEDED,
            usage=TokenUsage(input_tokens=200, output_tokens=50),
        ),
    )
    return LoadAttempt(
        attempt_id=identifier,
        logical_request_id=f"logical-{identifier}",
        scenario_id="policy",
        status=LoadAttemptStatus.SUCCEEDED,
        started_at=started_at,
        completed_at=started_at + timedelta(milliseconds=100),
        latency_ms=100,
        http_status_code=200,
        request_id=f"request-{identifier}",
        trace_id=f"trace-{identifier}",
        instance_identity="instance-test-1",
        terminal_kind="answer",
        provider_attempt_count=provider_attempt_count,
        provider_unknown_usage_attempt_count=0,
        provider_attempts=provider_attempts,
        token_counts=token_counts
        or {
            "embedding-input": 100,
            "generation-input": 200,
            "generation-output": 50,
        },
        model_identities={
            "embedding": "text-embedding-3-small",
            "generation": "gpt-4.1-mini",
        },
        stage_timings_ms={
            "validation": 1,
            "retrieval": 20,
            "evidence_assessment": 10,
            "generation": 50,
            "finalization": 1,
            "total": 100,
        },
        cache_status={"request-policy": "bypass", "retrieval": "bypass"},
    )


def _report() -> LoadReport:
    warmup_attempt = _attempt("warmup-1", warmup=True)
    return build_load_report(
        run_id="pricing-run-1",
        started_at=_START + timedelta(seconds=1),
        completed_at=_START + timedelta(seconds=2),
        duration_ms=1_000,
        instance_count=1,
        configured_concurrency=1,
        observed_peak_concurrency=1,
        cache_policy="bypass",
        workload_scenario_ids=("policy",),
        warmup=build_warmup_summary(
            (warmup_attempt,),
            readiness_passed=True,
            configured_attempts=1,
            started_at=_START,
            completed_at=_START + timedelta(milliseconds=100),
            duration_ms=100,
        ),
        attempts=(_attempt("measured-1"),),
        thresholds=LoadAcceptanceThresholds(
            required_concurrency=1,
            minimum_successes=1,
        ),
    )


def _pricing() -> PerformancePricingEvidence:
    return PerformancePricingEvidence(
        pricing_version="openai-standard-2026-08-07",
        currency="USD",
        rates=(
            PerformanceRolePricing(
                role=ModelRole.EMBEDDING,
                provider="openai",
                model="text-embedding-3-small",
                input_per_million=Decimal("0.02"),
            ),
            PerformanceRolePricing(
                role=ModelRole.GENERATION,
                provider="openai",
                model="gpt-4.1-mini",
                input_per_million=Decimal("0.40"),
                output_per_million=Decimal("1.60"),
            ),
        ),
        source_references=(
            "https://developers.openai.com/api/docs/models/gpt-4.1-mini",
            "https://developers.openai.com/api/docs/models/text-embedding-3-small",
        ),
        assumptions=("standard non-batch API rates",),
    )


def test_calculates_exact_measured_cost_and_projection() -> None:
    evidence = calculate_performance_cost(_report(), _pricing())

    assert evidence.complete is True
    assert evidence.input_tokens == 300
    assert evidence.output_tokens == 50
    assert evidence.provider_attempt_count == 2
    assert evidence.known_cost == Decimal("0.000162")
    assert evidence.estimated_cost == Decimal("0.000162")
    assert evidence.cost_per_1000_calls == Decimal("0.162000")
    assert evidence.unknown_reasons == ()


def test_provider_free_refusal_run_has_exact_zero_cost() -> None:
    report = _report()
    values = report.attempts[0].model_dump()
    values.update(
        {
            "terminal_kind": "refusal",
            "provider_attempt_count": 0,
            "provider_failed_attempt_count": 0,
            "provider_unknown_usage_attempt_count": 0,
            "provider_attempts": (),
            "token_counts": {},
            "model_identities": {},
        }
    )
    provider_free = LoadAttempt.model_validate(values)
    report = report.model_copy(
        update={
            "attempts": (provider_free,),
            "token_totals": {},
            "model_identities": {},
        }
    )

    evidence = calculate_performance_cost(report, _pricing())

    assert evidence.complete is True
    assert evidence.provider_attempt_count == 0
    assert evidence.known_cost == Decimal(0)
    assert evidence.estimated_cost == Decimal(0)
    assert evidence.cost_per_1000_calls == Decimal(0)
    assert evidence.unknown_reasons == ()


def test_missing_role_rate_is_explicitly_incomplete() -> None:
    pricing = _pricing().model_copy(update={"rates": (_pricing().rates[1],)})

    evidence = calculate_performance_cost(_report(), pricing)

    assert evidence.complete is False
    assert evidence.known_cost == Decimal("0.00016")
    assert evidence.estimated_cost is None
    assert evidence.cost_per_1000_calls is None
    assert evidence.unknown_reasons == ("pricing-rate-missing",)


def test_unscoped_usage_remains_in_directional_totals_but_is_unpriced() -> None:
    report = _report()
    report = report.model_copy(update={"token_totals": {"input": 3, "output": 2}})

    evidence = calculate_performance_cost(report, _pricing())

    assert evidence.input_tokens == 3
    assert evidence.output_tokens == 2
    assert evidence.provider_attempt_count == 2
    assert evidence.complete is False
    assert evidence.unknown_reasons == ("pricing-rate-missing",)


def test_rate_card_rejects_duplicate_or_whitespace_source_references() -> None:
    values = _pricing().model_dump(mode="json")
    values["source_references"] = ["https://example.test/rate card"]

    with pytest.raises(ValidationError, match="source references"):
        PerformancePricingEvidence.model_validate(values)


def test_fallback_is_priced_by_its_exact_provider_and_model_rate() -> None:
    report = _report()
    base = report.attempts[0]
    fallback = ProviderAttemptEvidence(
        operation_id="qa-generation",
        attempt_number=1,
        route_id="generation-fallback",
        role=ModelRole.GENERATION,
        provider="fallback-provider",
        model="fallback-model",
        status=ModelAttemptStatus.FAILED,
        fallback=True,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )
    values = base.model_dump()
    values.update(
        {
            "provider_attempt_count": 3,
            "provider_failed_attempt_count": 1,
            "provider_attempts": (*base.provider_attempts, fallback),
            "token_counts": {
                "embedding-input": 100,
                "generation-input": 210,
                "generation-output": 55,
            },
        }
    )
    measured = LoadAttempt.model_validate(values)
    report = report.model_copy(
        update={
            "attempts": (measured,),
            "token_totals": measured.token_counts,
        }
    )
    pricing = _pricing().model_copy(
        update={
            "rates": (
                *_pricing().rates,
                PerformanceRolePricing(
                    role=ModelRole.GENERATION,
                    provider="fallback-provider",
                    model="fallback-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            )
        }
    )

    evidence = calculate_performance_cost(report, pricing)

    assert evidence.complete is True
    assert evidence.provider_attempt_count == 3
    assert evidence.known_cost == Decimal("0.000182")


def test_unknown_attempt_usage_makes_cost_explicitly_incomplete() -> None:
    report = _report()
    base = report.attempts[0]
    generation = base.provider_attempts[1].model_copy(
        update={"usage": TokenUsage(input_tokens=200)}
    )
    values = base.model_dump()
    values.update(
        {
            "terminal_kind": "refusal",
            "provider_unknown_usage_attempt_count": 1,
            "provider_attempts": (base.provider_attempts[0], generation),
            "token_counts": {
                "embedding-input": 100,
                "generation-input": 200,
            },
        }
    )
    measured = LoadAttempt.model_validate(values)
    report = report.model_copy(
        update={"attempts": (measured,), "token_totals": measured.token_counts}
    )

    evidence = calculate_performance_cost(report, _pricing())

    assert evidence.complete is False
    assert evidence.estimated_cost is None
    assert evidence.unknown_reasons == ("provider-usage-unknown",)


def test_unverifiable_http_attempt_makes_cost_explicitly_incomplete() -> None:
    report = _report()
    successful = report.attempts[0]
    failed = LoadAttempt(
        attempt_id="measured-unverifiable",
        logical_request_id="logical-unverifiable",
        status=LoadAttemptStatus.TIMEOUT,
        started_at=successful.started_at,
        completed_at=successful.completed_at,
        latency_ms=successful.latency_ms,
        request_id="load-timeout-request",
        trace_id="a" * 32,
        safe_error_code="http-attempt-timeout",
        retryable=True,
        provider_evidence_complete=False,
    )
    report = build_load_report(
        run_id=report.run_id,
        started_at=report.started_at,
        completed_at=report.completed_at,
        duration_ms=report.duration_ms,
        instance_count=report.instance_count,
        configured_concurrency=report.configured_concurrency,
        observed_peak_concurrency=report.observed_peak_concurrency,
        cache_policy=report.cache_policy,
        workload_scenario_ids=report.workload_scenario_ids,
        warmup=report.warmup,
        attempts=(*report.attempts, failed),
        thresholds=report.thresholds,
    )

    evidence = calculate_performance_cost(report, _pricing())

    assert evidence.complete is False
    assert evidence.estimated_cost is None
    assert "provider-attempt-evidence-missing" in evidence.unknown_reasons


def test_pinned_openai_rate_card_prices_mini_to_gpt54_fallback_exactly() -> None:
    pricing_path = (
        Path(__file__).resolve().parents[3]
        / "evaluations"
        / "pricing"
        / "openai-standard-2026-08-07.json"
    )
    pricing = PerformancePricingEvidence.model_validate_json(
        pricing_path.read_text(encoding="utf-8")
    )
    report = _report()
    base = report.attempts[0]
    provider = "openai-compatible-d9617135d6fdd0a2"
    embedding = base.provider_attempts[0].model_copy(update={"provider": provider})
    mini = base.provider_attempts[1].model_copy(
        update={
            "provider": provider,
            "status": ModelAttemptStatus.FAILED,
        }
    )
    gpt54 = ProviderAttemptEvidence(
        operation_id="qa-generation",
        attempt_number=2,
        route_id="generation-fallback",
        role=ModelRole.GENERATION,
        provider=provider,
        model="gpt-5.4",
        status=ModelAttemptStatus.SUCCEEDED,
        fallback=True,
        usage=TokenUsage(input_tokens=120, output_tokens=30),
    )
    values = base.model_dump()
    values.update(
        {
            "provider_attempt_count": 3,
            "provider_failed_attempt_count": 1,
            "provider_attempts": (embedding, mini, gpt54),
            "token_counts": {
                "embedding-input": 100,
                "generation-input": 320,
                "generation-output": 80,
            },
        }
    )
    measured = LoadAttempt.model_validate(values)
    report = build_load_report(
        run_id=report.run_id,
        started_at=report.started_at,
        completed_at=report.completed_at,
        duration_ms=report.duration_ms,
        instance_count=report.instance_count,
        configured_concurrency=report.configured_concurrency,
        observed_peak_concurrency=report.observed_peak_concurrency,
        cache_policy=report.cache_policy,
        workload_scenario_ids=report.workload_scenario_ids,
        warmup=report.warmup,
        attempts=(measured,),
        thresholds=report.thresholds,
    )

    evidence = calculate_performance_cost(report, pricing)

    assert evidence.complete is True
    assert evidence.provider_attempt_count == 3
    assert evidence.known_cost == Decimal("0.000912")


def test_zero_rate_with_nonzero_usage_is_never_complete_cost_evidence() -> None:
    values = _pricing().model_dump()
    rates = list(values["rates"])
    generation = dict(rates[1])
    generation["input_per_million"] = Decimal(0)
    generation["output_per_million"] = Decimal(0)
    rates[1] = generation
    values["rates"] = rates
    pricing = PerformancePricingEvidence.model_validate(values)

    evidence = calculate_performance_cost(_report(), pricing)

    assert evidence.complete is False
    assert evidence.estimated_cost is None
    assert "zero-price-with-nonzero-usage" in evidence.unknown_reasons
