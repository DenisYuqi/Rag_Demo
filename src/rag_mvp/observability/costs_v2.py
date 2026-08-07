"""Schema-v2 provider cost evidence derived from immutable HTTP attempts.

The v1 cost helpers remain unchanged.  This module deliberately keeps the v2
contract separate so existing reports retain their original serialization and
successful-call denominator semantics.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from rag_mvp.domain._base import DomainModel, Identifier
from rag_mvp.domain.evaluation import (
    ModelAttemptStatus,
    ModelRole,
    ProviderAttemptEvidence,
    TokenUsage,
)
from rag_mvp.performance.load_report import LoadAttempt

COST_EVIDENCE_SCHEMA_VERSION: Literal["provider-cost-evidence-v2"] = "provider-cost-evidence-v2"
PRICING_PROVENANCE_SCHEMA_VERSION: Literal["pricing-provenance-v2"] = "pricing-provenance-v2"
_ONE_MILLION = Decimal(1_000_000)
_ONE_THOUSAND = Decimal(1_000)
_SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"

type NonNegativeDecimal = Annotated[Decimal, Field(ge=0)]
type NonNegativeInteger = Annotated[int, Field(ge=0)]


class EvidenceAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not-applicable"


class TokenDirection(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


class CostUnknownReason(StrEnum):
    PROVIDER_ATTEMPT_EVIDENCE_MISSING = "provider-attempt-evidence-missing"
    PRICING_RATE_MISSING = "pricing-rate-missing"
    INPUT_USAGE_UNKNOWN = "input-usage-unknown"
    OUTPUT_USAGE_UNKNOWN = "output-usage-unknown"
    INPUT_RATE_UNKNOWN = "input-rate-unknown"
    OUTPUT_RATE_UNKNOWN = "output-rate-unknown"
    LOGICAL_ATTEMPT_DENOMINATOR_ZERO = "logical-attempt-denominator-zero"
    SUCCESS_DENOMINATOR_ZERO = "success-denominator-zero"
    COST_INCOMPLETE = "cost-incomplete"


class ExactPricingRateV2(DomainModel):
    """One exact role/provider/model rate used for evidence calculation."""

    role: ModelRole
    provider: Identifier
    model: Identifier
    input_per_million: NonNegativeDecimal | None = None
    output_per_million: NonNegativeDecimal | None = None

    @model_validator(mode="after")
    def require_a_declared_direction(self) -> Self:
        if self.input_per_million is None and self.output_per_million is None:
            raise ValueError("an exact pricing rate requires at least one direction")
        return self


class PricingProvenanceV2(DomainModel):
    """Pinned rate card and source references protected by a canonical digest."""

    schema_version: Literal["pricing-provenance-v2"] = PRICING_PROVENANCE_SCHEMA_VERSION
    pricing_version: Identifier
    currency: Identifier
    rates: tuple[ExactPricingRateV2, ...]
    source_references: tuple[Annotated[str, Field(min_length=1, max_length=500)], ...]
    digest: Annotated[str, Field(pattern=_SHA256_PATTERN)]

    @classmethod
    def create(
        cls,
        *,
        pricing_version: str,
        currency: str,
        rates: Sequence[ExactPricingRateV2 | Mapping[str, object]],
        source_references: Sequence[str],
    ) -> PricingProvenanceV2:
        normalized_rates = tuple(ExactPricingRateV2.model_validate(rate) for rate in rates)
        normalized_sources = tuple(source_references)
        digest = canonical_pricing_provenance_digest(
            pricing_version=pricing_version,
            currency=currency,
            rates=normalized_rates,
            source_references=normalized_sources,
        )
        return cls(
            pricing_version=pricing_version,
            currency=currency,
            rates=normalized_rates,
            source_references=normalized_sources,
            digest=digest,
        )

    @model_validator(mode="after")
    def validate_provenance(self) -> Self:
        identities = {(rate.role, rate.provider, rate.model) for rate in self.rates}
        if not self.rates or len(identities) != len(self.rates):
            raise ValueError("pricing provenance requires unique exact rates")
        if (
            not self.source_references
            or len(set(self.source_references)) != len(self.source_references)
            or any(
                not reference.strip() or any(character.isspace() for character in reference)
                for reference in self.source_references
            )
        ):
            raise ValueError("pricing provenance source references are invalid")
        expected = canonical_pricing_provenance_digest(
            pricing_version=self.pricing_version,
            currency=self.currency,
            rates=self.rates,
            source_references=self.source_references,
        )
        if self.digest != expected:
            raise ValueError("pricing provenance digest mismatch")
        return self

    def lookup(self, attempt: ProviderAttemptEvidence) -> ExactPricingRateV2 | None:
        return next(
            (
                rate
                for rate in self.rates
                if (rate.role, rate.provider, rate.model)
                == (attempt.role, attempt.provider, attempt.model)
            ),
            None,
        )


class ProviderAttemptCostV2(DomainModel):
    """Cost result for every measured provider attempt, including failures."""

    http_attempt_id: Identifier
    logical_request_id: Identifier
    provider_attempt_ordinal: Annotated[int, Field(gt=0)]
    operation_id: Identifier
    attempt_number: Annotated[int, Field(gt=0)]
    role: ModelRole
    provider: Identifier
    model: Identifier
    status: ModelAttemptStatus
    fallback: bool
    usage: TokenUsage
    pricing_rate: ExactPricingRateV2 | None
    known_partial_cost: NonNegativeDecimal
    total_cost: NonNegativeDecimal | None
    complete: bool
    unknown_reasons: tuple[CostUnknownReason, ...]

    @model_validator(mode="after")
    def validate_completeness(self) -> Self:
        if self.complete != (self.total_cost is not None and not self.unknown_reasons):
            raise ValueError("provider-attempt cost completeness is inconsistent")
        if not self.complete and (self.total_cost is not None or not self.unknown_reasons):
            raise ValueError("incomplete provider-attempt cost must be unavailable with a reason")
        return self


class RoleDirectionTokenTotalV2(DomainModel):
    role: ModelRole
    direction: TokenDirection
    provider_attempt_count: NonNegativeInteger
    unknown_usage_attempt_count: NonNegativeInteger
    known_tokens: NonNegativeInteger
    total_tokens: NonNegativeInteger | None
    status: EvidenceAvailability
    unknown_reasons: tuple[CostUnknownReason, ...]

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        if self.unknown_usage_attempt_count > self.provider_attempt_count:
            raise ValueError("unknown token usage exceeds the provider-attempt denominator")
        if self.status is EvidenceAvailability.AVAILABLE:
            if (
                self.total_tokens is None
                or self.total_tokens != self.known_tokens
                or self.unknown_usage_attempt_count
                or self.unknown_reasons
            ):
                raise ValueError("available token total is inconsistent")
        elif self.status is EvidenceAvailability.NOT_APPLICABLE:
            if self.total_tokens is not None or self.unknown_usage_attempt_count:
                raise ValueError("not-applicable token total is inconsistent")
        elif self.total_tokens is not None or not self.unknown_reasons:
            raise ValueError("unavailable token total requires a reason")
        return self


class NormalizedCostV2(DomainModel):
    denominator_kind: Literal["logical-attempts", "successful-logical-attempts"]
    denominator: NonNegativeInteger
    per_1000: NonNegativeDecimal | None
    status: EvidenceAvailability
    unknown_reasons: tuple[CostUnknownReason, ...]

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        if self.status is EvidenceAvailability.AVAILABLE:
            if self.denominator <= 0 or self.per_1000 is None or self.unknown_reasons:
                raise ValueError("available normalized cost is inconsistent")
        elif self.status is not EvidenceAvailability.UNAVAILABLE:
            raise ValueError("normalized cost can only be available or unavailable")
        elif self.per_1000 is not None or not self.unknown_reasons:
            raise ValueError("unavailable normalized cost requires a reason")
        return self


class CostEvidenceV2(DomainModel):
    """Fail-closed measured cost evidence with both required denominators."""

    schema_version: Literal["provider-cost-evidence-v2"] = COST_EVIDENCE_SCHEMA_VERSION
    pricing: PricingProvenanceV2
    provider_attempt_count: NonNegativeInteger
    provider_attempts: tuple[ProviderAttemptCostV2, ...]
    unverifiable_http_attempt_ids: tuple[Identifier, ...]
    role_direction_tokens: tuple[RoleDirectionTokenTotalV2, ...]
    logical_attempt_count: NonNegativeInteger
    successful_logical_attempt_count: NonNegativeInteger
    known_partial_cost: NonNegativeDecimal
    total_cost: NonNegativeDecimal | None
    total_cost_status: EvidenceAvailability
    cost_per_1000_logical_attempts: NormalizedCostV2
    cost_per_1000_successes: NormalizedCostV2
    complete: bool
    unknown_reasons: tuple[CostUnknownReason, ...]

    @model_validator(mode="after")
    def validate_derived_values(self) -> Self:
        if self.successful_logical_attempt_count > self.logical_attempt_count:
            raise ValueError("successful logical attempts exceed the denominator")
        if self.provider_attempt_count != len(self.provider_attempts):
            raise ValueError("provider-attempt count disagrees with the cost ledger")
        if len(set(self.unverifiable_http_attempt_ids)) != len(self.unverifiable_http_attempt_ids):
            raise ValueError("unverifiable HTTP attempt IDs must be unique")
        for entry in self.provider_attempts:
            expected = _price_provider_attempt(
                http_attempt_id=entry.http_attempt_id,
                logical_request_id=entry.logical_request_id,
                provider_attempt_ordinal=entry.provider_attempt_ordinal,
                attempt=ProviderAttemptEvidence(
                    operation_id=entry.operation_id,
                    attempt_number=entry.attempt_number,
                    role=entry.role,
                    provider=entry.provider,
                    model=entry.model,
                    status=entry.status,
                    fallback=entry.fallback,
                    usage=entry.usage,
                ),
                pricing=self.pricing,
            )
            if entry != expected:
                raise ValueError("provider-attempt cost disagrees with exact pricing")
        expected_tokens = _role_direction_totals(self.provider_attempts)
        if self.role_direction_tokens != expected_tokens:
            raise ValueError("role/direction token totals disagree with provider attempts")
        known_partial = sum(
            (attempt.known_partial_cost for attempt in self.provider_attempts),
            start=Decimal(0),
        )
        if self.known_partial_cost != known_partial:
            raise ValueError("known partial cost disagrees with provider attempts")
        blocking_reasons = {
            reason for attempt in self.provider_attempts for reason in attempt.unknown_reasons
        }
        if self.unverifiable_http_attempt_ids:
            blocking_reasons.add(CostUnknownReason.PROVIDER_ATTEMPT_EVIDENCE_MISSING)
        provider_reasons = tuple(
            sorted(
                blocking_reasons,
                key=str,
            )
        )
        expected_total = (
            sum(
                (
                    attempt.total_cost
                    for attempt in self.provider_attempts
                    if attempt.total_cost is not None
                ),
                start=Decimal(0),
            )
            if not provider_reasons
            else None
        )
        expected_total_status = (
            EvidenceAvailability.AVAILABLE
            if expected_total is not None
            else EvidenceAvailability.UNAVAILABLE
        )
        if (self.total_cost, self.total_cost_status) != (expected_total, expected_total_status):
            raise ValueError("total cost availability disagrees with provider attempts")
        expected_attempt_projection = _normalized_cost(
            total_cost=expected_total,
            denominator=self.logical_attempt_count,
            denominator_kind="logical-attempts",
            denominator_reason=CostUnknownReason.LOGICAL_ATTEMPT_DENOMINATOR_ZERO,
            provider_reasons=provider_reasons,
        )
        expected_success_projection = _normalized_cost(
            total_cost=expected_total,
            denominator=self.successful_logical_attempt_count,
            denominator_kind="successful-logical-attempts",
            denominator_reason=CostUnknownReason.SUCCESS_DENOMINATOR_ZERO,
            provider_reasons=provider_reasons,
        )
        if self.cost_per_1000_logical_attempts != expected_attempt_projection:
            raise ValueError("logical-attempt cost denominator is inconsistent")
        if self.cost_per_1000_successes != expected_success_projection:
            raise ValueError("successful-attempt cost denominator is inconsistent")
        expected_reasons = set(provider_reasons)
        if self.logical_attempt_count == 0:
            expected_reasons.add(CostUnknownReason.LOGICAL_ATTEMPT_DENOMINATOR_ZERO)
        if self.successful_logical_attempt_count == 0:
            expected_reasons.add(CostUnknownReason.SUCCESS_DENOMINATOR_ZERO)
        if expected_total is None and self.provider_attempts and not provider_reasons:
            expected_reasons.add(CostUnknownReason.COST_INCOMPLETE)
        normalized_reasons = tuple(sorted(expected_reasons, key=str))
        expected_complete = (
            expected_total_status is EvidenceAvailability.AVAILABLE
            and expected_attempt_projection.status is EvidenceAvailability.AVAILABLE
            and expected_success_projection.status is EvidenceAvailability.AVAILABLE
        )
        if self.unknown_reasons != normalized_reasons or self.complete is not expected_complete:
            raise ValueError("cost evidence completeness is inconsistent")
        return self


def canonical_pricing_provenance_digest(
    *,
    pricing_version: str,
    currency: str,
    rates: Sequence[ExactPricingRateV2 | Mapping[str, object]],
    source_references: Sequence[str],
) -> str:
    normalized_rates = tuple(ExactPricingRateV2.model_validate(rate) for rate in rates)
    payload = {
        "schema_version": PRICING_PROVENANCE_SCHEMA_VERSION,
        "pricing_version": pricing_version,
        "currency": currency,
        "rates": [
            rate.model_dump(mode="json")
            for rate in sorted(
                normalized_rates,
                key=lambda item: (item.role.value, item.provider, item.model),
            )
        ],
        "source_references": sorted(source_references),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return f"sha256:{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"


def build_cost_evidence_v2(
    attempts: Sequence[LoadAttempt],
    *,
    pricing: PricingProvenanceV2,
) -> CostEvidenceV2:
    """Price every provider call in measured traffic and expose both denominators."""

    records = tuple(attempts)
    if any(not isinstance(attempt, LoadAttempt) for attempt in records):
        raise TypeError("cost evidence attempts must be LoadAttempt values")
    if len({attempt.attempt_id for attempt in records}) != len(records):
        raise ValueError("cost evidence HTTP attempt IDs must be unique")
    logical_ids = {attempt.logical_request_id for attempt in records}
    successful_ids = {attempt.logical_request_id for attempt in records if attempt.succeeded}
    unverifiable_http_attempt_ids = tuple(
        attempt.attempt_id for attempt in records if not attempt.provider_evidence_complete
    )
    provider_costs: list[ProviderAttemptCostV2] = []
    ordinal = 0
    for http_attempt in records:
        for provider_attempt in http_attempt.provider_attempts:
            ordinal += 1
            provider_costs.append(
                _price_provider_attempt(
                    http_attempt_id=http_attempt.attempt_id,
                    logical_request_id=http_attempt.logical_request_id,
                    provider_attempt_ordinal=ordinal,
                    attempt=provider_attempt,
                    pricing=pricing,
                )
            )
    cost_records = tuple(provider_costs)
    blocking_reasons = {reason for attempt in cost_records for reason in attempt.unknown_reasons}
    if unverifiable_http_attempt_ids:
        blocking_reasons.add(CostUnknownReason.PROVIDER_ATTEMPT_EVIDENCE_MISSING)
    provider_reasons = tuple(sorted(blocking_reasons, key=str))
    total_cost = (
        sum(
            (attempt.total_cost for attempt in cost_records if attempt.total_cost is not None),
            start=Decimal(0),
        )
        if not provider_reasons
        else None
    )
    attempt_projection = _normalized_cost(
        total_cost=total_cost,
        denominator=len(logical_ids),
        denominator_kind="logical-attempts",
        denominator_reason=CostUnknownReason.LOGICAL_ATTEMPT_DENOMINATOR_ZERO,
        provider_reasons=provider_reasons,
    )
    success_projection = _normalized_cost(
        total_cost=total_cost,
        denominator=len(successful_ids),
        denominator_kind="successful-logical-attempts",
        denominator_reason=CostUnknownReason.SUCCESS_DENOMINATOR_ZERO,
        provider_reasons=provider_reasons,
    )
    aggregate_reasons = set(provider_reasons)
    if not logical_ids:
        aggregate_reasons.add(CostUnknownReason.LOGICAL_ATTEMPT_DENOMINATOR_ZERO)
    if not successful_ids:
        aggregate_reasons.add(CostUnknownReason.SUCCESS_DENOMINATOR_ZERO)
    complete = (
        total_cost is not None
        and attempt_projection.status is EvidenceAvailability.AVAILABLE
        and success_projection.status is EvidenceAvailability.AVAILABLE
    )
    return CostEvidenceV2(
        pricing=pricing,
        provider_attempt_count=len(cost_records),
        provider_attempts=cost_records,
        unverifiable_http_attempt_ids=unverifiable_http_attempt_ids,
        role_direction_tokens=_role_direction_totals(cost_records),
        logical_attempt_count=len(logical_ids),
        successful_logical_attempt_count=len(successful_ids),
        known_partial_cost=sum(
            (attempt.known_partial_cost for attempt in cost_records),
            start=Decimal(0),
        ),
        total_cost=total_cost,
        total_cost_status=(
            EvidenceAvailability.AVAILABLE
            if total_cost is not None
            else EvidenceAvailability.UNAVAILABLE
        ),
        cost_per_1000_logical_attempts=attempt_projection,
        cost_per_1000_successes=success_projection,
        complete=complete,
        unknown_reasons=tuple(sorted(aggregate_reasons, key=str)),
    )


def _price_provider_attempt(
    *,
    http_attempt_id: str,
    logical_request_id: str,
    provider_attempt_ordinal: int,
    attempt: ProviderAttemptEvidence,
    pricing: PricingProvenanceV2,
) -> ProviderAttemptCostV2:
    rate = pricing.lookup(attempt)
    reasons: list[CostUnknownReason] = []
    known_parts: list[Decimal] = []
    if rate is None:
        reasons.append(CostUnknownReason.PRICING_RATE_MISSING)
    else:
        if attempt.usage.input_tokens is None:
            reasons.append(CostUnknownReason.INPUT_USAGE_UNKNOWN)
        elif rate.input_per_million is None:
            reasons.append(CostUnknownReason.INPUT_RATE_UNKNOWN)
        else:
            known_parts.append(
                Decimal(attempt.usage.input_tokens) * rate.input_per_million / _ONE_MILLION
            )
        output_applicable = (
            attempt.role is not ModelRole.EMBEDDING
            or rate.output_per_million is not None
            or attempt.usage.output_tokens not in {None, 0}
        )
        if output_applicable:
            if attempt.usage.output_tokens is None:
                reasons.append(CostUnknownReason.OUTPUT_USAGE_UNKNOWN)
            elif rate.output_per_million is None:
                reasons.append(CostUnknownReason.OUTPUT_RATE_UNKNOWN)
            else:
                known_parts.append(
                    Decimal(attempt.usage.output_tokens) * rate.output_per_million / _ONE_MILLION
                )
    normalized_reasons = tuple(dict.fromkeys(reasons))
    known_partial = sum(known_parts, start=Decimal(0))
    total = known_partial if not normalized_reasons else None
    return ProviderAttemptCostV2(
        http_attempt_id=http_attempt_id,
        logical_request_id=logical_request_id,
        provider_attempt_ordinal=provider_attempt_ordinal,
        operation_id=attempt.operation_id,
        attempt_number=attempt.attempt_number,
        role=attempt.role,
        provider=attempt.provider,
        model=attempt.model,
        status=attempt.status,
        fallback=attempt.fallback,
        usage=attempt.usage,
        pricing_rate=rate,
        known_partial_cost=known_partial,
        total_cost=total,
        complete=not normalized_reasons,
        unknown_reasons=normalized_reasons,
    )


def _role_direction_totals(
    attempts: Sequence[ProviderAttemptCostV2],
) -> tuple[RoleDirectionTokenTotalV2, ...]:
    totals: list[RoleDirectionTokenTotalV2] = []
    roles = sorted({attempt.role for attempt in attempts}, key=lambda role: role.value)
    for role in roles:
        role_attempts = tuple(attempt for attempt in attempts if attempt.role is role)
        for direction in TokenDirection:
            values = tuple(
                (
                    attempt.usage.input_tokens
                    if direction is TokenDirection.INPUT
                    else attempt.usage.output_tokens
                )
                for attempt in role_attempts
            )
            known = sum(value for value in values if value is not None)
            unknown = sum(value is None for value in values)
            not_applicable = (
                role is ModelRole.EMBEDDING
                and direction is TokenDirection.OUTPUT
                and all(value is None for value in values)
            )
            reason = (
                CostUnknownReason.INPUT_USAGE_UNKNOWN
                if direction is TokenDirection.INPUT
                else CostUnknownReason.OUTPUT_USAGE_UNKNOWN
            )
            totals.append(
                RoleDirectionTokenTotalV2(
                    role=role,
                    direction=direction,
                    provider_attempt_count=len(role_attempts),
                    unknown_usage_attempt_count=0 if not_applicable else unknown,
                    known_tokens=known,
                    total_tokens=(None if unknown or not_applicable else known),
                    status=(
                        EvidenceAvailability.NOT_APPLICABLE
                        if not_applicable
                        else EvidenceAvailability.UNAVAILABLE
                        if unknown
                        else EvidenceAvailability.AVAILABLE
                    ),
                    unknown_reasons=(() if not_applicable or not unknown else (reason,)),
                )
            )
    return tuple(totals)


def _normalized_cost(
    *,
    total_cost: Decimal | None,
    denominator: int,
    denominator_kind: Literal["logical-attempts", "successful-logical-attempts"],
    denominator_reason: CostUnknownReason,
    provider_reasons: Sequence[CostUnknownReason],
) -> NormalizedCostV2:
    reasons = list(provider_reasons)
    if denominator == 0:
        reasons.append(denominator_reason)
    if total_cost is None and not reasons:
        reasons.append(CostUnknownReason.COST_INCOMPLETE)
    normalized_reasons = tuple(dict.fromkeys(reasons))
    per_1000 = (
        total_cost * _ONE_THOUSAND / Decimal(denominator)
        if total_cost is not None and denominator > 0 and not normalized_reasons
        else None
    )
    available = per_1000 is not None
    return NormalizedCostV2(
        denominator_kind=denominator_kind,
        denominator=denominator,
        per_1000=per_1000,
        status=(EvidenceAvailability.AVAILABLE if available else EvidenceAvailability.UNAVAILABLE),
        unknown_reasons=normalized_reasons,
    )


def _json_default(value: object) -> object:
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise TypeError("pricing provenance contains a non-finite decimal")
        normalized = value.normalize()
        return "0" if normalized == 0 else format(normalized, "f")
    raise TypeError("pricing provenance contains an unsupported value")


__all__ = [
    "COST_EVIDENCE_SCHEMA_VERSION",
    "PRICING_PROVENANCE_SCHEMA_VERSION",
    "CostEvidenceV2",
    "CostUnknownReason",
    "EvidenceAvailability",
    "ExactPricingRateV2",
    "NormalizedCostV2",
    "PricingProvenanceV2",
    "ProviderAttemptCostV2",
    "RoleDirectionTokenTotalV2",
    "TokenDirection",
    "build_cost_evidence_v2",
    "canonical_pricing_provenance_digest",
]
