from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from rag_mvp.domain.evaluation import (
    ModelAttemptStatus,
    ModelRole,
    ProviderAttemptEvidence,
    TokenUsage,
)
from rag_mvp.observability.costs_v2 import (
    CostUnknownReason,
    ExactPricingRateV2,
    PricingProvenanceV2,
    build_cost_evidence_v2,
)
from rag_mvp.performance.load_report import LoadAttempt, LoadAttemptStatus
from rag_mvp.performance.pricing import (
    OPENAI_COMPARISON_PRICING_ASSUMPTIONS,
    OPENAI_COMPARISON_PRICING_PROVIDER,
    OPENAI_COMPARISON_PRICING_SOURCES,
    OPENAI_COMPARISON_PRICING_VERSION,
    PerformancePricingEvidence,
    PricingPreflightError,
    preflight_openai_comparison_pricing,
)
from rag_mvp.performance.run_load_test import _load_pricing

_CANONICAL_PRICING_DIGEST = (
    "sha256:a714aae47f476d98a9927d566adfca1682edcc66a6b396e5c8b06be672c603e6"
)
_IMMUTABLE_FILE_DIGEST = "sha256:27828dc583b2efc322553e3d6b3efd8f38391bc138929370578cadbc60024c83"
_START = datetime(2026, 8, 7, tzinfo=UTC)


def _artifact_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "evaluations"
        / "pricing"
        / "openai-comparison-standard-2026-08-07-v1.json"
    )


def _pricing() -> PerformancePricingEvidence:
    return PerformancePricingEvidence.model_validate_json(
        _artifact_path().read_text(encoding="utf-8")
    )


def _cost_pricing(
    pricing: PerformancePricingEvidence,
    *,
    include_reranking: bool = True,
) -> PricingProvenanceV2:
    rates = tuple(
        ExactPricingRateV2.model_validate(rate.model_dump())
        for rate in pricing.rates
        if include_reranking or rate.role is not ModelRole.RERANKING
    )
    return PricingProvenanceV2.create(
        pricing_version=pricing.pricing_version,
        currency=pricing.currency,
        rates=rates,
        source_references=pricing.source_references,
    )


def _reranking_attempt(*, model: str = "gpt-4.1-mini") -> LoadAttempt:
    provider_attempt = ProviderAttemptEvidence(
        operation_id="qa-reranking",
        attempt_number=1,
        route_id="reranking-primary",
        role=ModelRole.RERANKING,
        provider=OPENAI_COMPARISON_PRICING_PROVIDER,
        model=model,
        status=ModelAttemptStatus.SUCCEEDED,
        usage=TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000),
    )
    return LoadAttempt(
        attempt_id="comparison-http-attempt-1",
        logical_request_id="comparison-logical-attempt-1",
        scenario_id="rerank-sensitive",
        status=LoadAttemptStatus.SUCCEEDED,
        started_at=_START,
        completed_at=_START + timedelta(milliseconds=10),
        latency_ms=10,
        http_status_code=200,
        request_id="comparison-request-1",
        trace_id="comparison-trace-1",
        instance_identity="comparison-instance-1",
        terminal_kind="refusal",
        provider_attempt_count=1,
        provider_attempts=(provider_attempt,),
        token_counts={
            "reranking-input": 1_000_000,
            "reranking-output": 1_000_000,
        },
        model_identities={"reranking": model},
        cache_status={"request-policy": "bypass"},
    )


def test_versioned_comparison_artifact_has_stable_schema_sources_and_digests() -> None:
    path = _artifact_path()
    pricing = _load_pricing(path)

    assert pricing.pricing_version == OPENAI_COMPARISON_PRICING_VERSION
    assert pricing.source_references == OPENAI_COMPARISON_PRICING_SOURCES
    assert pricing.assumptions == OPENAI_COMPARISON_PRICING_ASSUMPTIONS
    assert all(reference.startswith("https://") for reference in pricing.source_references)
    assert preflight_openai_comparison_pricing(pricing) == _CANONICAL_PRICING_DIGEST
    assert f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}" == _IMMUTABLE_FILE_DIGEST


def test_preflight_requires_exact_reranking_identity_and_authoritative_rate() -> None:
    pricing = _pricing()
    without_reranking = pricing.model_copy(
        update={
            "rates": tuple(rate for rate in pricing.rates if rate.role is not ModelRole.RERANKING)
        }
    )
    with pytest.raises(
        PricingPreflightError,
        match="comparison-pricing-rate-identity-mismatch",
    ):
        preflight_openai_comparison_pricing(without_reranking)

    rates = list(pricing.rates)
    reranking_index = next(
        index for index, rate in enumerate(rates) if rate.role is ModelRole.RERANKING
    )
    rates[reranking_index] = rates[reranking_index].model_copy(
        update={"input_per_million": Decimal("0.41")}
    )
    with pytest.raises(
        PricingPreflightError,
        match="comparison-pricing-rate-value-mismatch",
    ):
        preflight_openai_comparison_pricing(pricing.model_copy(update={"rates": tuple(rates)}))


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ({"role": "unknown-provider-role"}, "safe contract"),
        ({"model": "gpt-unknown"}, "safe contract"),
        (
            {"source_references": ["http://developers.openai.com/api/docs/models/gpt-5.4"]},
            "safe contract",
        ),
    ),
)
def test_loader_fails_closed_for_unknown_identity_or_unapproved_source(
    tmp_path: Path,
    mutation: dict[str, object],
    expected: str,
) -> None:
    payload = json.loads(_artifact_path().read_text(encoding="utf-8"))
    if "source_references" in mutation:
        payload["source_references"] = mutation["source_references"]
    else:
        reranking = next(rate for rate in payload["rates"] if rate["role"] == "reranking")
        reranking.update(mutation)
    path = tmp_path / "tampered-comparison-pricing.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        _load_pricing(path)


def test_reranking_attempt_uses_explicit_role_rate_and_never_generation_alias() -> None:
    pricing = _pricing()
    attempt = _reranking_attempt()

    priced = build_cost_evidence_v2((attempt,), pricing=_cost_pricing(pricing))

    assert priced.complete is True
    assert priced.provider_attempts[0].pricing_rate is not None
    assert priced.provider_attempts[0].pricing_rate.role is ModelRole.RERANKING
    assert priced.total_cost == Decimal("2.00")

    generation_only = build_cost_evidence_v2(
        (attempt,),
        pricing=_cost_pricing(pricing, include_reranking=False),
    )
    assert generation_only.provider_attempts[0].pricing_rate is None
    assert generation_only.total_cost is None
    assert CostUnknownReason.PRICING_RATE_MISSING in generation_only.unknown_reasons


def test_unknown_reranking_model_is_unpriced_even_when_role_is_known() -> None:
    evidence = build_cost_evidence_v2(
        (_reranking_attempt(model="gpt-unknown"),),
        pricing=_cost_pricing(_pricing()),
    )

    assert evidence.provider_attempts[0].pricing_rate is None
    assert evidence.total_cost is None
    assert evidence.complete is False
    assert CostUnknownReason.PRICING_RATE_MISSING in evidence.unknown_reasons
