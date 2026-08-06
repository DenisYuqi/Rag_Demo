"""Versioned provider pricing and complete cost aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from rag_mvp.domain.evaluation import ModelAttempt, ModelPricing, ModelRole

_ONE_MILLION = Decimal(1_000_000)
_ONE_THOUSAND = Decimal(1_000)


class UnknownCostReason(StrEnum):
    NO_ATTEMPTS = "no-attempts"
    PRICING_NOT_FOUND = "pricing-not-found"
    INPUT_USAGE_UNKNOWN = "input-usage-unknown"
    OUTPUT_USAGE_UNKNOWN = "output-usage-unknown"
    INPUT_PRICE_UNKNOWN = "input-price-unknown"
    OUTPUT_PRICE_UNKNOWN = "output-price-unknown"
    CURRENCY_MISMATCH = "currency-mismatch"


@dataclass(frozen=True, slots=True)
class AttemptCostEstimate:
    attempt_id: str
    request_id: str | None
    run_id: str | None
    role: ModelRole
    pricing_version: str
    currency: str | None
    input_tokens: int | None
    output_tokens: int | None
    known_cost: Decimal | None
    estimated_cost: Decimal | None
    complete: bool
    unknown_reasons: tuple[UnknownCostReason, ...]


@dataclass(frozen=True, slots=True)
class RoleCost:
    role: ModelRole
    known_cost: Decimal | None
    estimated_cost: Decimal | None
    complete: bool


@dataclass(frozen=True, slots=True)
class CostAggregate:
    scope: str
    scope_id: str
    pricing_version: str
    currency: str | None
    attempt_count: int
    priced_attempt_count: int
    input_tokens: int | None
    output_tokens: int | None
    known_cost: Decimal | None
    estimated_cost: Decimal | None
    complete: bool
    unknown_reasons: tuple[UnknownCostReason, ...]
    roles: tuple[RoleCost, ...]


@dataclass(frozen=True, slots=True)
class PerThousandCallEstimate:
    pricing_version: str
    successful_calls: int
    currency: str | None
    known_cost_per_1000: Decimal | None
    estimated_cost_per_1000: Decimal | None
    complete: bool
    assumptions: tuple[str, ...]
    unknown_reasons: tuple[UnknownCostReason, ...]


class PricingCatalog:
    """Immutable pricing table addressed by provider and exact model identity."""

    def __init__(self, *, version: str, entries: tuple[ModelPricing, ...]) -> None:
        if not version.strip():
            raise ValueError("pricing catalog version must not be empty")
        prices: dict[tuple[str, str], ModelPricing] = {}
        for entry in entries:
            if entry.pricing_version != version:
                raise ValueError("pricing entry version does not match catalog version")
            key = (entry.provider, entry.model)
            if key in prices:
                raise ValueError("pricing entries must have unique provider/model identities")
            prices[key] = entry
        self.version = version
        self._prices = prices

    def lookup(self, provider: str, model: str) -> ModelPricing | None:
        return self._prices.get((provider, model))

    def estimate_attempt(self, attempt: ModelAttempt) -> AttemptCostEstimate:
        pricing = self.lookup(attempt.provider, attempt.model)
        if pricing is None:
            return AttemptCostEstimate(
                attempt_id=attempt.attempt_id,
                request_id=attempt.request_id,
                run_id=attempt.run_id,
                role=attempt.role,
                pricing_version=self.version,
                currency=None,
                input_tokens=attempt.usage.input_tokens,
                output_tokens=attempt.usage.output_tokens,
                known_cost=None,
                estimated_cost=None,
                complete=False,
                unknown_reasons=(UnknownCostReason.PRICING_NOT_FOUND,),
            )

        reasons: list[UnknownCostReason] = []
        known_parts: list[Decimal] = []
        if attempt.usage.input_tokens is None:
            reasons.append(UnknownCostReason.INPUT_USAGE_UNKNOWN)
        elif pricing.input_per_million is None:
            reasons.append(UnknownCostReason.INPUT_PRICE_UNKNOWN)
        else:
            known_parts.append(
                Decimal(attempt.usage.input_tokens) * pricing.input_per_million / _ONE_MILLION
            )

        if attempt.role is not ModelRole.EMBEDDING:
            if attempt.usage.output_tokens is None:
                reasons.append(UnknownCostReason.OUTPUT_USAGE_UNKNOWN)
            elif pricing.output_per_million is None:
                reasons.append(UnknownCostReason.OUTPUT_PRICE_UNKNOWN)
            else:
                known_parts.append(
                    Decimal(attempt.usage.output_tokens)
                    * pricing.output_per_million
                    / _ONE_MILLION
                )

        known_cost = sum(known_parts, start=Decimal(0)) if known_parts else None
        complete = not reasons
        return AttemptCostEstimate(
            attempt_id=attempt.attempt_id,
            request_id=attempt.request_id,
            run_id=attempt.run_id,
            role=attempt.role,
            pricing_version=self.version,
            currency=pricing.currency,
            input_tokens=attempt.usage.input_tokens,
            output_tokens=attempt.usage.output_tokens,
            known_cost=known_cost,
            estimated_cost=known_cost if complete else None,
            complete=complete,
            unknown_reasons=tuple(reasons),
        )

    def aggregate_request(
        self,
        attempts: tuple[ModelAttempt, ...],
        *,
        request_id: str,
    ) -> CostAggregate:
        if any(attempt.request_id != request_id for attempt in attempts):
            raise ValueError("request aggregation received an attempt from another request")
        return self._aggregate(attempts, scope="request", scope_id=request_id)

    def aggregate_run(
        self,
        attempts: tuple[ModelAttempt, ...],
        *,
        run_id: str,
    ) -> CostAggregate:
        if any(attempt.run_id != run_id for attempt in attempts):
            raise ValueError("run aggregation received an attempt from another run")
        return self._aggregate(attempts, scope="run", scope_id=run_id)

    def _aggregate(
        self,
        attempts: tuple[ModelAttempt, ...],
        *,
        scope: str,
        scope_id: str,
    ) -> CostAggregate:
        estimates = tuple(self.estimate_attempt(attempt) for attempt in attempts)
        if not estimates:
            return CostAggregate(
                scope=scope,
                scope_id=scope_id,
                pricing_version=self.version,
                currency=None,
                attempt_count=0,
                priced_attempt_count=0,
                input_tokens=None,
                output_tokens=None,
                known_cost=None,
                estimated_cost=None,
                complete=False,
                unknown_reasons=(UnknownCostReason.NO_ATTEMPTS,),
                roles=(),
            )

        currencies = {estimate.currency for estimate in estimates if estimate.currency is not None}
        currency = next(iter(currencies)) if len(currencies) == 1 else None
        reasons = {reason for estimate in estimates for reason in estimate.unknown_reasons}
        if len(currencies) > 1:
            reasons.add(UnknownCostReason.CURRENCY_MISMATCH)
        can_sum_currency = len(currencies) == 1
        known_values = [
            estimate.known_cost for estimate in estimates if estimate.known_cost is not None
        ]
        known_cost = (
            sum(known_values, start=Decimal(0)) if known_values and can_sum_currency else None
        )
        complete = all(estimate.complete for estimate in estimates) and can_sum_currency
        estimated_cost = known_cost if complete else None

        role_costs = tuple(
            self._role_cost(role, tuple(item for item in estimates if item.role is role), currency)
            for role in ModelRole
            if any(item.role is role for item in estimates)
        )
        return CostAggregate(
            scope=scope,
            scope_id=scope_id,
            pricing_version=self.version,
            currency=currency,
            attempt_count=len(estimates),
            priced_attempt_count=sum(item.currency is not None for item in estimates),
            input_tokens=_sum_known_tokens(tuple(item.input_tokens for item in estimates)),
            output_tokens=_sum_known_tokens(tuple(item.output_tokens for item in estimates)),
            known_cost=known_cost,
            estimated_cost=estimated_cost,
            complete=complete,
            unknown_reasons=tuple(sorted(reasons, key=str)),
            roles=role_costs,
        )

    @staticmethod
    def _role_cost(
        role: ModelRole,
        estimates: tuple[AttemptCostEstimate, ...],
        aggregate_currency: str | None,
    ) -> RoleCost:
        currencies = {item.currency for item in estimates if item.currency is not None}
        same_currency = len(currencies) == 1 and next(iter(currencies)) == aggregate_currency
        known_values = [item.known_cost for item in estimates if item.known_cost is not None]
        known_cost = sum(known_values, start=Decimal(0)) if known_values and same_currency else None
        complete = bool(estimates) and same_currency and all(item.complete for item in estimates)
        return RoleCost(
            role=role,
            known_cost=known_cost,
            estimated_cost=known_cost if complete else None,
            complete=complete,
        )


def _sum_known_tokens(values: tuple[int | None, ...]) -> int | None:
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def project_per_thousand_calls(
    aggregate: CostAggregate,
    *,
    successful_calls: int,
) -> PerThousandCallEstimate:
    """Normalize an aggregate cost; incomplete inputs remain explicitly unknown."""

    if isinstance(successful_calls, bool) or successful_calls <= 0:
        raise ValueError("successful_calls must be positive")
    multiplier = _ONE_THOUSAND / Decimal(successful_calls)
    known = aggregate.known_cost * multiplier if aggregate.known_cost is not None else None
    estimated = (
        aggregate.estimated_cost * multiplier
        if aggregate.estimated_cost is not None and aggregate.complete
        else None
    )
    return PerThousandCallEstimate(
        pricing_version=aggregate.pricing_version,
        successful_calls=successful_calls,
        currency=aggregate.currency,
        known_cost_per_1000=known,
        estimated_cost_per_1000=estimated,
        complete=aggregate.complete,
        assumptions=(
            "all provider attempts, including retries and fallbacks, are included",
            "the observed successful-call mix is projected linearly to 1,000 calls",
        ),
        unknown_reasons=aggregate.unknown_reasons,
    )
