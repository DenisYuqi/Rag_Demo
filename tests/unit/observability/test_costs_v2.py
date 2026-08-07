from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from rag_mvp.domain.evaluation import (
    ModelAttemptStatus,
    ModelRole,
    ProviderAttemptEvidence,
    TokenUsage,
)
from rag_mvp.observability.costs_v2 import (
    CostEvidenceV2,
    CostUnknownReason,
    EvidenceAvailability,
    ExactPricingRateV2,
    PricingProvenanceV2,
    TokenDirection,
    build_cost_evidence_v2,
)
from rag_mvp.performance.load_report import LoadAttempt, LoadAttemptStatus

_START = datetime(2026, 8, 7, tzinfo=UTC)


def _pricing(*, model: str = "chat-v2") -> PricingProvenanceV2:
    return PricingProvenanceV2.create(
        pricing_version="pricing-2026-08",
        currency="USD",
        rates=(
            ExactPricingRateV2(
                role=ModelRole.GENERATION,
                provider="primary",
                model=model,
                input_per_million=Decimal("2"),
                output_per_million=Decimal("8"),
            ),
        ),
        source_references=("https://pricing.example/provider/model-card-v2",),
    )


def _provider_attempt(
    *,
    status: ModelAttemptStatus = ModelAttemptStatus.SUCCEEDED,
    usage: TokenUsage | None = None,
    model: str = "chat-v2",
    fallback: bool = False,
) -> ProviderAttemptEvidence:
    return ProviderAttemptEvidence(
        operation_id="qa-generation",
        attempt_number=1,
        route_id="fallback" if fallback else "primary",
        role=ModelRole.GENERATION,
        provider="primary",
        model=model,
        status=status,
        fallback=fallback,
        usage=usage or TokenUsage(input_tokens=1_000, output_tokens=500),
    )


def _attempt(
    identifier: str,
    *,
    logical_request_id: str | None = None,
    attempt_number: int = 1,
    retry_of_attempt_id: str | None = None,
    failed: bool = False,
    provider_attempts: tuple[ProviderAttemptEvidence, ...] | None = None,
    provider_evidence_complete: bool = True,
) -> LoadAttempt:
    providers = (
        provider_attempts
        if provider_attempts is not None
        else (
            _provider_attempt(
                status=(ModelAttemptStatus.FAILED if failed else ModelAttemptStatus.SUCCEEDED)
            ),
        )
    )
    token_counts: dict[str, int] = {}
    for provider in providers:
        if provider.usage.input_tokens is not None:
            token_counts["generation-input"] = (
                token_counts.get("generation-input", 0) + provider.usage.input_tokens
            )
        if provider.usage.output_tokens is not None:
            token_counts["generation-output"] = (
                token_counts.get("generation-output", 0) + provider.usage.output_tokens
            )
    unknown_usage_count = sum(
        provider.usage.input_tokens is None or provider.usage.output_tokens is None
        for provider in providers
    )
    started_at = _START + timedelta(milliseconds=len(identifier))
    return LoadAttempt(
        attempt_id=identifier,
        logical_request_id=logical_request_id or f"logical-{identifier}",
        scenario_id="policy",
        attempt_number=attempt_number,
        retry_of_attempt_id=retry_of_attempt_id,
        status=(LoadAttemptStatus.TERMINAL_ERROR if failed else LoadAttemptStatus.SUCCEEDED),
        started_at=started_at,
        completed_at=started_at + timedelta(milliseconds=10),
        latency_ms=10,
        http_status_code=200,
        request_id=f"request-{identifier}",
        trace_id=f"trace-{identifier}",
        instance_identity="instance-v2",
        terminal_kind="error" if failed else "refusal",
        safe_error_code="capacity" if failed else None,
        retryable=failed,
        provider_attempt_count=len(providers),
        provider_failed_attempt_count=sum(
            provider.status is not ModelAttemptStatus.SUCCEEDED for provider in providers
        ),
        provider_unknown_usage_attempt_count=unknown_usage_count,
        provider_evidence_complete=provider_evidence_complete,
        provider_attempts=providers,
        stage_timings_ms={"validation": 1, "retrieval": 5, "total": 10},
        token_counts=token_counts,
        model_identities={"generation": "chat-v2"} if providers else {},
        cache_status={"request-policy": "bypass", "retrieval": "bypass"},
    )


def test_prices_every_provider_attempt_and_both_logical_denominators() -> None:
    first = _attempt("http-1", logical_request_id="logical-1", failed=True)
    retry = _attempt(
        "http-2",
        logical_request_id="logical-1",
        attempt_number=2,
        retry_of_attempt_id="http-1",
    )
    second = _attempt("http-3", logical_request_id="logical-2")

    evidence = build_cost_evidence_v2((first, retry, second), pricing=_pricing())

    assert evidence.provider_attempt_count == 3
    assert tuple(item.status for item in evidence.provider_attempts) == (
        ModelAttemptStatus.FAILED,
        ModelAttemptStatus.SUCCEEDED,
        ModelAttemptStatus.SUCCEEDED,
    )
    assert evidence.logical_attempt_count == 2
    assert evidence.successful_logical_attempt_count == 2
    assert evidence.total_cost == Decimal("0.018")
    assert evidence.cost_per_1000_logical_attempts.per_1000 == Decimal("9")
    assert evidence.cost_per_1000_successes.per_1000 == Decimal("9")
    assert evidence.complete is True
    totals = {(item.role, item.direction): item for item in evidence.role_direction_tokens}
    assert totals[(ModelRole.GENERATION, TokenDirection.INPUT)].total_tokens == 3_000
    assert totals[(ModelRole.GENERATION, TokenDirection.OUTPUT)].total_tokens == 1_500


def test_missing_usage_fails_closed_without_discarding_known_partial_cost() -> None:
    provider = _provider_attempt(usage=TokenUsage(input_tokens=1_000))
    attempt = _attempt("http-unknown-usage", provider_attempts=(provider,))

    evidence = build_cost_evidence_v2((attempt,), pricing=_pricing())

    assert evidence.known_partial_cost == Decimal("0.002")
    assert evidence.total_cost is None
    assert evidence.total_cost_status is EvidenceAvailability.UNAVAILABLE
    assert evidence.cost_per_1000_logical_attempts.per_1000 is None
    assert evidence.cost_per_1000_successes.per_1000 is None
    assert CostUnknownReason.OUTPUT_USAGE_UNKNOWN in evidence.unknown_reasons
    output = next(
        item for item in evidence.role_direction_tokens if item.direction is TokenDirection.OUTPUT
    )
    assert output.known_tokens == 0
    assert output.total_tokens is None
    assert output.status is EvidenceAvailability.UNAVAILABLE


def test_exact_role_provider_model_rate_is_required() -> None:
    attempt = _attempt("http-unpriced")

    evidence = build_cost_evidence_v2((attempt,), pricing=_pricing(model="other-model"))

    assert evidence.provider_attempts[0].pricing_rate is None
    assert evidence.total_cost is None
    assert evidence.complete is False
    assert CostUnknownReason.PRICING_RATE_MISSING in evidence.unknown_reasons


def test_missing_provider_ledger_fails_closed_for_the_entire_measured_cost() -> None:
    priced = _attempt("http-priced")
    unverifiable = _attempt(
        "http-unverifiable",
        failed=True,
        provider_attempts=(),
        provider_evidence_complete=False,
    )

    evidence = build_cost_evidence_v2(
        (priced, unverifiable),
        pricing=_pricing(),
    )

    assert evidence.provider_attempt_count == 1
    assert evidence.unverifiable_http_attempt_ids == ("http-unverifiable",)
    assert evidence.known_partial_cost == Decimal("0.006")
    assert evidence.total_cost is None
    assert evidence.cost_per_1000_logical_attempts.per_1000 is None
    assert CostUnknownReason.PROVIDER_ATTEMPT_EVIDENCE_MISSING in evidence.unknown_reasons


def test_zero_success_denominator_is_unavailable_not_zero() -> None:
    failed = _attempt("http-failed", failed=True)

    evidence = build_cost_evidence_v2((failed,), pricing=_pricing())

    assert evidence.total_cost == Decimal("0.006")
    assert evidence.cost_per_1000_logical_attempts.per_1000 == Decimal("6")
    assert evidence.cost_per_1000_successes.denominator == 0
    assert evidence.cost_per_1000_successes.per_1000 is None
    assert evidence.cost_per_1000_successes.status is EvidenceAvailability.UNAVAILABLE
    assert evidence.cost_per_1000_successes.unknown_reasons == (
        CostUnknownReason.SUCCESS_DENOMINATOR_ZERO,
    )
    assert evidence.complete is False


def test_verified_provider_free_refusal_has_exact_zero_cost() -> None:
    refusal = _attempt("http-provider-free", provider_attempts=())

    evidence = build_cost_evidence_v2((refusal,), pricing=_pricing())

    assert evidence.provider_attempt_count == 0
    assert evidence.total_cost == Decimal(0)
    assert evidence.cost_per_1000_logical_attempts.per_1000 == Decimal(0)
    assert evidence.cost_per_1000_successes.per_1000 == Decimal(0)
    assert evidence.complete is True
    assert evidence.unknown_reasons == ()


def test_pricing_digest_and_derived_provider_cost_are_tamper_evident() -> None:
    pricing_payload = _pricing().model_dump()
    pricing_payload["pricing_version"] = "tampered-pricing"
    with pytest.raises(ValidationError, match="pricing provenance digest mismatch"):
        PricingProvenanceV2.model_validate(pricing_payload)

    evidence = build_cost_evidence_v2((_attempt("http-tamper"),), pricing=_pricing())
    evidence_payload = evidence.model_dump()
    attempts = evidence_payload["provider_attempts"]
    assert isinstance(attempts, tuple)
    first = dict(attempts[0])
    first["total_cost"] = Decimal("1")
    evidence_payload["provider_attempts"] = (first,)
    with pytest.raises(ValidationError, match="exact pricing"):
        CostEvidenceV2.model_validate(evidence_payload)
