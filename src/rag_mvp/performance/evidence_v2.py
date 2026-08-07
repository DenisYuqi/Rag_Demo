"""Schema-v2 performance evidence with unbiased latency denominators.

This module is additive.  The v1 ``LoadReport`` and performance bundle remain
unchanged and readable with their original successful-only latency field.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from datetime import datetime
from itertools import pairwise
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from rag_mvp.domain._base import DomainModel, Identifier, NonNegativeFiniteFloat, utc_now
from rag_mvp.observability.costs_v2 import (
    CostEvidenceV2,
    EvidenceAvailability,
    PricingProvenanceV2,
    build_cost_evidence_v2,
)
from rag_mvp.performance.load_report import LoadAttempt, nearest_rank_percentile

PERFORMANCE_EVIDENCE_SCHEMA_VERSION: Literal["performance-evidence-v2"] = "performance-evidence-v2"
ATTEMPT_LEDGER_DIGEST_VERSION = "http-attempt-ledger-sha256-v2"
OFFICIAL_P90_SCOPE: Literal["all-measured-http-attempts"] = "all-measured-http-attempts"
DEFAULT_OFFICIAL_P90_THRESHOLD_MS = 10_000.0
_SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"

type NonNegativeInteger = Annotated[int, Field(ge=0)]


class LatencyPercentilesV2(DomainModel):
    """Nearest-rank latency values, never interpolated."""

    count: Annotated[int, Field(gt=0)]
    p50_ms: NonNegativeFiniteFloat
    p90_ms: NonNegativeFiniteFloat
    p95_ms: NonNegativeFiniteFloat
    p99_ms: NonNegativeFiniteFloat

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        values = (self.p50_ms, self.p90_ms, self.p95_ms, self.p99_ms)
        if any(left > right for left, right in pairwise(values)):
            raise ValueError("latency percentiles are not monotonic")
        return self


class LatencyScopesV2(DomainModel):
    all_attempts: LatencyPercentilesV2 | None
    successful_attempts: LatencyPercentilesV2 | None


class AttemptRateV2(DomainModel):
    numerator: NonNegativeInteger
    denominator: NonNegativeInteger
    value: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None
    status: EvidenceAvailability

    @model_validator(mode="after")
    def validate_rate(self) -> Self:
        if self.numerator > self.denominator:
            raise ValueError("rate numerator exceeds its denominator")
        if self.denominator == 0:
            if self.value is not None or self.status is not EvidenceAvailability.UNAVAILABLE:
                raise ValueError("zero-denominator rate must be unavailable")
        else:
            expected = self.numerator / self.denominator
            if (
                self.value is None
                or self.status is not EvidenceAvailability.AVAILABLE
                or not math.isclose(self.value, expected, rel_tol=0, abs_tol=1e-15)
            ):
                raise ValueError("available rate disagrees with its denominator")
        return self


class OfficialP90GateV2(DomainModel):
    metric: Literal["p90_latency_ms"] = "p90_latency_ms"
    scope: Literal["all-measured-http-attempts"] = OFFICIAL_P90_SCOPE
    operator: Literal["<="] = "<="
    threshold_ms: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    observed_ms: NonNegativeFiniteFloat | None
    denominator: NonNegativeInteger
    status: EvidenceAvailability
    passed: bool | None

    @model_validator(mode="after")
    def validate_gate(self) -> Self:
        if self.denominator == 0:
            if (
                self.observed_ms is not None
                or self.passed is not None
                or self.status is not EvidenceAvailability.UNAVAILABLE
            ):
                raise ValueError("zero-denominator P90 gate must be unavailable")
        elif (
            self.observed_ms is None
            or self.status is not EvidenceAvailability.AVAILABLE
            or self.passed is not (self.observed_ms <= self.threshold_ms)
        ):
            raise ValueError("official P90 gate is inconsistent")
        return self


class WarmupPerformanceEvidenceV2(DomainModel):
    """Traffic explicitly excluded from every official measured denominator."""

    scope: Literal["warm-up-excluded"] = "warm-up-excluded"
    attempts: tuple[LoadAttempt, ...]
    attempt_ledger_digest: Annotated[str, Field(pattern=_SHA256_PATTERN)]
    attempt_count: NonNegativeInteger
    success_count: NonNegativeInteger
    failure_count: NonNegativeInteger
    latency_ms: LatencyScopesV2

    @model_validator(mode="after")
    def validate_from_ledger(self) -> Self:
        _validate_attempt_ledger(self.attempts)
        expected_digest = canonical_attempt_ledger_digest(self.attempts, scope="warm-up")
        if self.attempt_ledger_digest != expected_digest:
            raise ValueError("warm-up attempt-ledger digest mismatch")
        successes = tuple(attempt for attempt in self.attempts if attempt.succeeded)
        if (
            self.attempt_count != len(self.attempts)
            or self.success_count != len(successes)
            or self.failure_count != len(self.attempts) - len(successes)
        ):
            raise ValueError("warm-up counts disagree with the attempt ledger")
        expected_latency = LatencyScopesV2(
            all_attempts=_latency_summary(self.attempts),
            successful_attempts=_latency_summary(successes),
        )
        if self.latency_ms != expected_latency:
            raise ValueError("warm-up latency disagrees with the attempt ledger")
        return self


class MeasuredPerformanceEvidenceV2(DomainModel):
    """Measured HTTP attempts with all-attempt and successful-only views."""

    scope: Literal["measured"] = "measured"
    attempts: tuple[LoadAttempt, ...]
    attempt_ledger_digest: Annotated[str, Field(pattern=_SHA256_PATTERN)]
    http_attempt_count: NonNegativeInteger
    logical_attempt_count: NonNegativeInteger
    successful_http_attempt_count: NonNegativeInteger
    successful_logical_attempt_count: NonNegativeInteger
    failed_http_attempt_count: NonNegativeInteger
    error_rate: AttemptRateV2
    latency_ms: LatencyScopesV2
    official_p90_gate: OfficialP90GateV2

    @model_validator(mode="after")
    def validate_from_ledger(self) -> Self:
        _validate_attempt_ledger(self.attempts)
        expected_digest = canonical_attempt_ledger_digest(self.attempts, scope="measured")
        if self.attempt_ledger_digest != expected_digest:
            raise ValueError("measured attempt-ledger digest mismatch")
        successes = tuple(attempt for attempt in self.attempts if attempt.succeeded)
        logical_ids = {attempt.logical_request_id for attempt in self.attempts}
        successful_logical_ids = {attempt.logical_request_id for attempt in successes}
        failures = len(self.attempts) - len(successes)
        if (
            self.http_attempt_count != len(self.attempts)
            or self.logical_attempt_count != len(logical_ids)
            or self.successful_http_attempt_count != len(successes)
            or self.successful_logical_attempt_count != len(successful_logical_ids)
            or self.failed_http_attempt_count != failures
        ):
            raise ValueError("measured counts disagree with the attempt ledger")
        expected_rate = _attempt_rate(failures, len(self.attempts))
        if self.error_rate != expected_rate:
            raise ValueError("measured error rate disagrees with all HTTP attempts")
        all_latency = _latency_summary(self.attempts)
        expected_latency = LatencyScopesV2(
            all_attempts=all_latency,
            successful_attempts=_latency_summary(successes),
        )
        if self.latency_ms != expected_latency:
            raise ValueError("measured latency disagrees with the attempt ledger")
        expected_gate = _official_p90_gate(
            all_latency,
            denominator=len(self.attempts),
            threshold_ms=self.official_p90_gate.threshold_ms,
        )
        if self.official_p90_gate != expected_gate:
            raise ValueError("official P90 gate must use all measured HTTP attempts")
        return self


class PerformanceEvidenceV2(DomainModel):
    """Complete v2 performance/cost record derived from two immutable ledgers."""

    schema_version: Literal["performance-evidence-v2"] = PERFORMANCE_EVIDENCE_SCHEMA_VERSION
    run_id: Identifier
    configuration_id: Identifier
    created_at: AwareDatetime
    warmup: WarmupPerformanceEvidenceV2
    measured: MeasuredPerformanceEvidenceV2
    cost: CostEvidenceV2

    @model_validator(mode="after")
    def validate_cross_section_parity(self) -> Self:
        warmup_ids = {attempt.attempt_id for attempt in self.warmup.attempts}
        measured_ids = {attempt.attempt_id for attempt in self.measured.attempts}
        if warmup_ids & measured_ids:
            raise ValueError("warm-up and measured attempt ledgers overlap")
        warmup_logical_ids = {attempt.logical_request_id for attempt in self.warmup.attempts}
        measured_logical_ids = {attempt.logical_request_id for attempt in self.measured.attempts}
        if warmup_logical_ids & measured_logical_ids:
            raise ValueError("warm-up and measured logical requests overlap")
        expected_cost = build_cost_evidence_v2(
            self.measured.attempts,
            pricing=self.cost.pricing,
        )
        if self.cost != expected_cost:
            raise ValueError("cost evidence disagrees with measured provider attempts")
        return self


def build_performance_evidence_v2(
    *,
    run_id: str,
    configuration_id: str,
    warmup_attempts: Sequence[LoadAttempt],
    measured_attempts: Sequence[LoadAttempt],
    pricing: PricingProvenanceV2,
    official_p90_threshold_ms: float = DEFAULT_OFFICIAL_P90_THRESHOLD_MS,
    created_at: datetime | None = None,
) -> PerformanceEvidenceV2:
    """Build evidence while keeping warm-up out of official latency and cost."""

    threshold = _positive_finite(official_p90_threshold_ms, "official P90 threshold")
    warmup_records = tuple(warmup_attempts)
    measured_records = tuple(measured_attempts)
    _validate_attempt_ledger(warmup_records)
    _validate_attempt_ledger(measured_records)
    if {attempt.attempt_id for attempt in warmup_records} & {
        attempt.attempt_id for attempt in measured_records
    }:
        raise ValueError("warm-up and measured attempt ledgers overlap")
    if {attempt.logical_request_id for attempt in warmup_records} & {
        attempt.logical_request_id for attempt in measured_records
    }:
        raise ValueError("warm-up and measured logical requests overlap")
    warmup_successes = tuple(attempt for attempt in warmup_records if attempt.succeeded)
    measured_successes = tuple(attempt for attempt in measured_records if attempt.succeeded)
    measured_logical_ids = {attempt.logical_request_id for attempt in measured_records}
    successful_logical_ids = {attempt.logical_request_id for attempt in measured_successes}
    measured_failures = len(measured_records) - len(measured_successes)
    all_measured_latency = _latency_summary(measured_records)
    warmup = WarmupPerformanceEvidenceV2(
        attempts=warmup_records,
        attempt_ledger_digest=canonical_attempt_ledger_digest(
            warmup_records,
            scope="warm-up",
        ),
        attempt_count=len(warmup_records),
        success_count=len(warmup_successes),
        failure_count=len(warmup_records) - len(warmup_successes),
        latency_ms=LatencyScopesV2(
            all_attempts=_latency_summary(warmup_records),
            successful_attempts=_latency_summary(warmup_successes),
        ),
    )
    measured = MeasuredPerformanceEvidenceV2(
        attempts=measured_records,
        attempt_ledger_digest=canonical_attempt_ledger_digest(
            measured_records,
            scope="measured",
        ),
        http_attempt_count=len(measured_records),
        logical_attempt_count=len(measured_logical_ids),
        successful_http_attempt_count=len(measured_successes),
        successful_logical_attempt_count=len(successful_logical_ids),
        failed_http_attempt_count=measured_failures,
        error_rate=_attempt_rate(measured_failures, len(measured_records)),
        latency_ms=LatencyScopesV2(
            all_attempts=all_measured_latency,
            successful_attempts=_latency_summary(measured_successes),
        ),
        official_p90_gate=_official_p90_gate(
            all_measured_latency,
            denominator=len(measured_records),
            threshold_ms=threshold,
        ),
    )
    return PerformanceEvidenceV2(
        run_id=run_id,
        configuration_id=configuration_id,
        created_at=created_at or utc_now(),
        warmup=warmup,
        measured=measured,
        cost=build_cost_evidence_v2(measured_records, pricing=pricing),
    )


def canonical_attempt_ledger_digest(
    attempts: Sequence[LoadAttempt],
    *,
    scope: Literal["warm-up", "measured"],
) -> str:
    records = tuple(attempts)
    payload = {
        "digest_version": ATTEMPT_LEDGER_DIGEST_VERSION,
        "scope": scope,
        "attempts": [attempt.model_dump(mode="json") for attempt in records],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"


def _latency_summary(attempts: Sequence[LoadAttempt]) -> LatencyPercentilesV2 | None:
    samples = tuple(attempt.latency_ms for attempt in attempts)
    if not samples:
        return None
    return LatencyPercentilesV2(
        count=len(samples),
        p50_ms=nearest_rank_percentile(samples, 50),
        p90_ms=nearest_rank_percentile(samples, 90),
        p95_ms=nearest_rank_percentile(samples, 95),
        p99_ms=nearest_rank_percentile(samples, 99),
    )


def _attempt_rate(numerator: int, denominator: int) -> AttemptRateV2:
    return AttemptRateV2(
        numerator=numerator,
        denominator=denominator,
        value=(numerator / denominator if denominator else None),
        status=(
            EvidenceAvailability.AVAILABLE if denominator else EvidenceAvailability.UNAVAILABLE
        ),
    )


def _official_p90_gate(
    summary: LatencyPercentilesV2 | None,
    *,
    denominator: int,
    threshold_ms: float,
) -> OfficialP90GateV2:
    observed = None if summary is None else summary.p90_ms
    return OfficialP90GateV2(
        threshold_ms=threshold_ms,
        observed_ms=observed,
        denominator=denominator,
        status=(
            EvidenceAvailability.AVAILABLE
            if observed is not None and denominator
            else EvidenceAvailability.UNAVAILABLE
        ),
        passed=(None if observed is None or not denominator else observed <= threshold_ms),
    )


def _validate_attempt_ledger(attempts: Sequence[LoadAttempt]) -> None:
    if any(not isinstance(attempt, LoadAttempt) for attempt in attempts):
        raise TypeError("performance evidence attempts must be LoadAttempt values")
    registry = {attempt.attempt_id: attempt for attempt in attempts}
    if len(registry) != len(attempts):
        raise ValueError("performance attempt ledger contains duplicate attempt IDs")
    logical_attempt_numbers: dict[str, set[int]] = {}
    successful_logical_ids: set[str] = set()
    for attempt in attempts:
        numbers = logical_attempt_numbers.setdefault(attempt.logical_request_id, set())
        if attempt.attempt_number in numbers:
            raise ValueError("performance attempt ledger contains duplicate retry ordinals")
        numbers.add(attempt.attempt_number)
        if attempt.succeeded:
            if attempt.logical_request_id in successful_logical_ids:
                raise ValueError("a logical request cannot have multiple successful attempts")
            successful_logical_ids.add(attempt.logical_request_id)
        predecessor_id = attempt.retry_of_attempt_id
        if predecessor_id is not None:
            predecessor = registry.get(predecessor_id)
            if (
                predecessor is None
                or predecessor.logical_request_id != attempt.logical_request_id
                or predecessor.attempt_number + 1 != attempt.attempt_number
                or predecessor.succeeded
                or not predecessor.retryable
            ):
                raise ValueError("performance attempt retry chain is invalid")
    for numbers in logical_attempt_numbers.values():
        if numbers != set(range(1, max(numbers, default=0) + 1)):
            raise ValueError("performance attempt retry ordinals are not contiguous")


def _positive_finite(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be positive and finite")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return normalized


__all__ = [
    "ATTEMPT_LEDGER_DIGEST_VERSION",
    "DEFAULT_OFFICIAL_P90_THRESHOLD_MS",
    "OFFICIAL_P90_SCOPE",
    "PERFORMANCE_EVIDENCE_SCHEMA_VERSION",
    "AttemptRateV2",
    "LatencyPercentilesV2",
    "LatencyScopesV2",
    "MeasuredPerformanceEvidenceV2",
    "OfficialP90GateV2",
    "PerformanceEvidenceV2",
    "WarmupPerformanceEvidenceV2",
    "build_performance_evidence_v2",
    "canonical_attempt_ledger_digest",
]
