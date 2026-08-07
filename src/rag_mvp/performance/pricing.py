"""Post-run cost calculation for HTTP performance evidence."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import Field, field_validator

from rag_mvp.domain._base import DomainModel, Identifier
from rag_mvp.domain.evaluation import ModelRole
from rag_mvp.performance.evidence_bundle import (
    PerformanceCostEvidence,
    PerformanceRateEvidence,
    canonical_pricing_evidence_digest,
)
from rag_mvp.performance.load_report import LoadReport

_ONE_MILLION = Decimal(1_000_000)
_ONE_THOUSAND = Decimal(1_000)
_SAFE_ROLE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")

OPENAI_COMPARISON_PRICING_VERSION = "openai-comparison-standard-2026-08-07-v1"
OPENAI_COMPARISON_PRICING_PROVIDER = "openai-compatible-d9617135d6fdd0a2"
OPENAI_COMPARISON_PRICING_SOURCES = (
    "https://developers.openai.com/api/docs/models/gpt-4.1-mini",
    "https://developers.openai.com/api/docs/models/gpt-5.4",
    "https://developers.openai.com/api/docs/models/text-embedding-3-small",
)
OPENAI_COMPARISON_PRICING_ASSUMPTIONS = (
    "standard non-batch and non-priority API pricing",
    "no cached-input discount is claimed",
    "the reranking adapter uses gpt-4.1-mini chat text-token rates but remains an "
    "explicit reranking role; no generation-role alias is permitted",
)

_OPENAI_COMPARISON_RATES = {
    (ModelRole.EMBEDDING, "text-embedding-3-small"): (Decimal("0.02"), None),
    (ModelRole.GENERATION, "gpt-4.1-mini"): (Decimal("0.40"), Decimal("1.60")),
    (ModelRole.GENERATION, "gpt-5.4"): (Decimal("2.50"), Decimal("15.00")),
    (ModelRole.RERANKING, "gpt-4.1-mini"): (Decimal("0.40"), Decimal("1.60")),
}


class PricingPreflightError(ValueError):
    """Safe fail-closed error raised before comparison provider traffic."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class PerformanceRolePricing(PerformanceRateEvidence):
    """Exact per-million-token rate bound to provider, model, and role."""


class PerformancePricingEvidence(DomainModel):
    """Pinned rate card supplied to the load runner before traffic starts."""

    pricing_version: Identifier
    currency: Identifier
    rates: Annotated[tuple[PerformanceRolePricing, ...], Field(min_length=1)]
    source_references: Annotated[tuple[str, ...], Field(min_length=1)]
    assumptions: tuple[str, ...] = ()

    @field_validator("rates")
    @classmethod
    def validate_rates(
        cls,
        value: tuple[PerformanceRolePricing, ...],
    ) -> tuple[PerformanceRolePricing, ...]:
        identities = {(rate.role, rate.provider, rate.model) for rate in value}
        if len(identities) != len(value):
            raise ValueError("performance pricing identities must be unique")
        return tuple(sorted(value, key=lambda rate: (rate.role.value, rate.provider, rate.model)))

    @field_validator("source_references")
    @classmethod
    def validate_sources(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(
            not item or len(item) > 500 or any(character.isspace() for character in item)
            for item in normalized
        ) or len(normalized) != len(set(normalized)):
            raise ValueError("pricing source references must be unique bounded references")
        return normalized


def preflight_openai_comparison_pricing(pricing: PerformancePricingEvidence) -> str:
    """Validate the immutable official comparison card and return its digest.

    The role is part of every rate identity.  In particular, a generation rate
    for ``gpt-4.1-mini`` cannot satisfy the required reranking identity.
    """

    if pricing.pricing_version != OPENAI_COMPARISON_PRICING_VERSION:
        raise PricingPreflightError("comparison-pricing-version-mismatch")
    if pricing.currency != "USD":
        raise PricingPreflightError("comparison-pricing-currency-mismatch")
    if pricing.source_references != OPENAI_COMPARISON_PRICING_SOURCES:
        raise PricingPreflightError("comparison-pricing-sources-mismatch")
    for reference in pricing.source_references:
        parsed = urlsplit(reference)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "developers.openai.com"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
            or parsed.query
            or parsed.fragment
        ):
            raise PricingPreflightError("comparison-pricing-source-not-allowlisted")
    if pricing.assumptions != OPENAI_COMPARISON_PRICING_ASSUMPTIONS:
        raise PricingPreflightError("comparison-pricing-assumptions-mismatch")

    actual = {
        (rate.role, rate.provider, rate.model): (
            rate.input_per_million,
            rate.output_per_million,
        )
        for rate in pricing.rates
    }
    expected = {
        (role, OPENAI_COMPARISON_PRICING_PROVIDER, model): amounts
        for (role, model), amounts in _OPENAI_COMPARISON_RATES.items()
    }
    if actual.keys() != expected.keys():
        raise PricingPreflightError("comparison-pricing-rate-identity-mismatch")
    if actual != expected:
        raise PricingPreflightError("comparison-pricing-rate-value-mismatch")

    return canonical_pricing_evidence_digest(
        pricing_version=pricing.pricing_version,
        currency=pricing.currency,
        rate_card=pricing.rates,
        source_references=pricing.source_references,
    )


def calculate_performance_cost(
    report: LoadReport,
    pricing: PerformancePricingEvidence,
) -> PerformanceCostEvidence:
    """Calculate measured-traffic cost after exact response usage is known."""

    input_tokens = 0
    output_tokens = 0
    known_parts: list[Decimal] = []
    reasons: list[str] = []
    for name, count in report.token_totals.items():
        role, direction = _token_role_and_direction(name)
        if direction == "input":
            input_tokens += count
        elif direction == "output":
            output_tokens += count
        else:
            reasons.append("token-direction-unknown")
        if role is None:
            reasons.append("pricing-rate-missing")
    rate_index = {(rate.role, rate.provider, rate.model): rate for rate in pricing.rates}
    provider_attempt_count = 0
    for http_attempt in report.attempts:
        provider_attempt_count += len(http_attempt.provider_attempts)
        for provider_attempt in http_attempt.provider_attempts:
            rate = rate_index.get(
                (provider_attempt.role, provider_attempt.provider, provider_attempt.model)
            )
            if rate is None:
                reasons.append("pricing-rate-missing")
            if provider_attempt.usage.input_tokens is None:
                reasons.append("provider-usage-unknown")
            else:
                if rate is not None and rate.input_per_million is not None:
                    if provider_attempt.usage.input_tokens > 0 and rate.input_per_million == 0:
                        reasons.append("zero-price-with-nonzero-usage")
                    known_parts.append(
                        Decimal(provider_attempt.usage.input_tokens)
                        * rate.input_per_million
                        / _ONE_MILLION
                    )
                elif rate is not None:
                    reasons.append("input-price-missing")
            if provider_attempt.role is ModelRole.EMBEDDING:
                continue
            if provider_attempt.usage.output_tokens is None:
                reasons.append("provider-usage-unknown")
            else:
                if rate is not None and rate.output_per_million is not None:
                    if provider_attempt.usage.output_tokens > 0 and rate.output_per_million == 0:
                        reasons.append("zero-price-with-nonzero-usage")
                    known_parts.append(
                        Decimal(provider_attempt.usage.output_tokens)
                        * rate.output_per_million
                        / _ONE_MILLION
                    )
                elif rate is not None:
                    reasons.append("output-price-missing")

    if any(attempt.provider_unknown_usage_attempt_count for attempt in report.attempts):
        reasons.append("provider-usage-unknown")
    if any(not attempt.provider_evidence_complete for attempt in report.attempts):
        reasons.append("provider-attempt-evidence-missing")
    if report.success_count == 0:
        reasons.append("successful-calls-missing")

    unknown_reasons = tuple(dict.fromkeys(reasons))
    complete = not unknown_reasons
    known_cost = sum(known_parts, start=Decimal(0))
    estimated_cost = known_cost if complete else None
    cost_per_1000_calls = (
        estimated_cost * _ONE_THOUSAND / Decimal(report.success_count)
        if estimated_cost is not None and report.success_count > 0
        else None
    )
    assumptions = (
        "only measured HTTP attempts are included; warm-up traffic is reported separately",
        "all provider attempts, including internal retries and fallbacks, are counted",
        "the observed successful-call mix is projected linearly to 1,000 calls",
        *pricing.assumptions,
        *(f"pricing source: {reference}" for reference in pricing.source_references),
    )
    rate_card = tuple(
        PerformanceRateEvidence.model_validate(rate.model_dump()) for rate in pricing.rates
    )
    pricing_evidence_digest = canonical_pricing_evidence_digest(
        pricing_version=pricing.pricing_version,
        currency=pricing.currency,
        rate_card=rate_card,
        source_references=pricing.source_references,
    )
    return PerformanceCostEvidence(
        pricing_version=pricing.pricing_version,
        pricing_evidence_digest=pricing_evidence_digest,
        source_references=pricing.source_references,
        currency=pricing.currency,
        complete=complete,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        known_cost=known_cost,
        estimated_cost=estimated_cost,
        cost_per_1000_calls=cost_per_1000_calls,
        provider_attempt_count=provider_attempt_count,
        rate_card=rate_card,
        unknown_reasons=unknown_reasons,
        assumptions=assumptions,
    )


def _token_role_and_direction(name: str) -> tuple[str | None, str | None]:
    normalized = name.strip().casefold().replace("_", "-")
    if normalized in {"input", "output"}:
        return None, normalized
    for direction in ("input", "output"):
        suffix = f"-{direction}"
        if normalized.endswith(suffix):
            role = normalized[: -len(suffix)]
            return (role if _SAFE_ROLE.fullmatch(role) is not None else None), direction
    return None, None


__all__ = [
    "OPENAI_COMPARISON_PRICING_ASSUMPTIONS",
    "OPENAI_COMPARISON_PRICING_PROVIDER",
    "OPENAI_COMPARISON_PRICING_SOURCES",
    "OPENAI_COMPARISON_PRICING_VERSION",
    "PerformancePricingEvidence",
    "PerformanceRolePricing",
    "PricingPreflightError",
    "calculate_performance_cost",
    "preflight_openai_comparison_pricing",
]
