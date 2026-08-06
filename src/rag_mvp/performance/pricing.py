"""Post-run cost calculation for HTTP performance evidence."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Annotated

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
    "PerformancePricingEvidence",
    "PerformanceRolePricing",
    "calculate_performance_cost",
]
