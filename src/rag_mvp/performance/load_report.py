"""Auditable load-attempt records and acceptance-summary calculations."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Self

from pydantic import AwareDatetime, Field, model_validator

from rag_mvp.domain._base import Digest, DomainModel, Identifier, NonNegativeFiniteFloat
from rag_mvp.domain.evaluation import (
    ModelAttemptStatus,
    ModelRole,
    ProviderAttemptEvidence,
)

LOAD_REPORT_VERSION = "http-load-report-v1"
ACCEPTANCE_CONCURRENCY = 5
MINIMUM_SUCCESSFUL_REQUESTS = 500
P90_LATENCY_THRESHOLD_MS = 10_000.0
ERROR_RATE_THRESHOLD = 0.01
CACHE_POLICY_HEADER = "X-RAG-Cache-Policy"
INSTANCE_ID_HEADER = "X-RAG-Instance-ID"
CACHE_BYPASS_POLICY = "bypass"

_CACHE_HIT_VALUES = frozenset(
    {
        "cached",
        "fresh-hit",
        "hit",
        "stale-hit",
    }
)


def _minimum_provider_attempts_from_tokens(token_counts: Mapping[str, int]) -> int:
    roles: set[str] = set()
    unscoped_usage = False
    for name in token_counts:
        normalized = name.strip().casefold().replace("_", "-")
        if normalized in {"input", "output"}:
            unscoped_usage = True
            continue
        for suffix in ("-input", "-output"):
            if normalized.endswith(suffix) and normalized[: -len(suffix)]:
                roles.add(normalized[: -len(suffix)])
                break
    return len(roles) + int(unscoped_usage)


class LoadAttemptStatus(StrEnum):
    """Content-free outcome for one HTTP request attempt."""

    SUCCEEDED = "succeeded"
    HTTP_ERROR = "http-error"
    TERMINAL_ERROR = "terminal-error"
    TIMEOUT = "timeout"
    INCOMPLETE = "incomplete"
    MALFORMED = "malformed"
    TRANSPORT_ERROR = "transport-error"


class LoadAttempt(DomainModel):
    """One measured or warm-up attempt, excluding request/response content."""

    attempt_id: Identifier
    logical_request_id: Identifier
    scenario_id: Identifier | None = None
    attempt_number: Annotated[int, Field(gt=0)] = 1
    retry_of_attempt_id: Identifier | None = None
    status: LoadAttemptStatus
    started_at: AwareDatetime
    completed_at: AwareDatetime
    latency_ms: NonNegativeFiniteFloat
    http_status_code: Annotated[int, Field(ge=100, le=599)] | None = None
    request_id: Identifier | None = None
    trace_id: Identifier | None = None
    instance_identity: Identifier | None = None
    terminal_kind: Annotated[str, Field(pattern=r"^(answer|refusal|error)$")] | None = None
    safe_error_code: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.-]{0,63}$")] | None = None
    retryable: bool = False
    provider_attempt_count: Annotated[int, Field(ge=0)] = 0
    provider_failed_attempt_count: Annotated[int, Field(ge=0)] = 0
    provider_unknown_usage_attempt_count: Annotated[int, Field(ge=0)] = 0
    provider_evidence_complete: bool = True
    provider_attempts: tuple[ProviderAttemptEvidence, ...] = ()
    stage_timings_ms: dict[str, NonNegativeFiniteFloat] = Field(default_factory=dict)
    token_counts: dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    model_identities: dict[str, str] = Field(default_factory=dict)
    cache_status: dict[str, str] = Field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status is LoadAttemptStatus.SUCCEEDED

    @property
    def cache_satisfied(self) -> bool:
        return any(_is_cache_hit(value) for value in self.cache_status.values())

    @model_validator(mode="after")
    def validate_attempt(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError("attempt completion precedes its start")
        if self.attempt_number == 1 and self.retry_of_attempt_id is not None:
            raise ValueError("a first attempt cannot reference a retry predecessor")
        if self.attempt_number > 1 and self.retry_of_attempt_id is None:
            raise ValueError("a retry must reference its preceding attempt")
        if self.provider_failed_attempt_count > self.provider_attempt_count:
            raise ValueError("failed provider attempts exceed the provider attempt count")
        if self.provider_unknown_usage_attempt_count > self.provider_attempt_count:
            raise ValueError("unknown-usage attempts exceed the provider attempt count")
        if len(self.provider_attempts) != self.provider_attempt_count:
            raise ValueError("provider attempt ledger count disagrees with diagnostics")
        if len(set(self.provider_attempts)) != len(self.provider_attempts):
            raise ValueError("provider attempt ledger contains duplicate records")
        if (
            sum(
                attempt.status is not ModelAttemptStatus.SUCCEEDED
                for attempt in self.provider_attempts
            )
            != self.provider_failed_attempt_count
        ):
            raise ValueError("provider failure count disagrees with the attempt ledger")
        if (
            sum(_provider_usage_unknown(attempt) for attempt in self.provider_attempts)
            != self.provider_unknown_usage_attempt_count
        ):
            raise ValueError("unknown provider usage count disagrees with the attempt ledger")
        if _provider_token_counts(self.provider_attempts) != self.token_counts:
            raise ValueError("token counts disagree with the provider attempt ledger")
        if self.provider_attempt_count < _minimum_provider_attempts_from_tokens(self.token_counts):
            raise ValueError("provider attempts do not cover token-bearing provider roles")
        if self.succeeded:
            if not self.provider_evidence_complete:
                raise ValueError("a successful attempt requires complete provider evidence")
            if self.terminal_kind not in {"answer", "refusal"}:
                raise ValueError("a successful attempt requires a complete terminal outcome")
            if self.terminal_kind == "answer" and not _has_complete_generation_attempt(
                self.provider_attempts
            ):
                raise ValueError(
                    "a successful answer requires a complete successful generation attempt"
                )
            if self.safe_error_code is not None or self.retryable:
                raise ValueError("a successful attempt cannot carry error details")
            if self.http_status_code is None or not 200 <= self.http_status_code < 300:
                raise ValueError("a successful attempt requires a successful HTTP status")
        elif self.safe_error_code is None:
            raise ValueError("a failed attempt requires a safe error code")
        return self


class LatencySummary(DomainModel):
    """Nearest-rank latency summary in milliseconds."""

    count: Annotated[int, Field(gt=0)]
    minimum: NonNegativeFiniteFloat
    p50: NonNegativeFiniteFloat
    p90: NonNegativeFiniteFloat
    p95: NonNegativeFiniteFloat
    p99: NonNegativeFiniteFloat
    maximum: NonNegativeFiniteFloat

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        values = (
            self.minimum,
            self.p50,
            self.p90,
            self.p95,
            self.p99,
            self.maximum,
        )
        if any(left > right for left, right in pairwise(values)):
            raise ValueError("latency percentiles are not monotonic")
        return self


class LoadAcceptanceThresholds(DomainModel):
    """Frozen single-instance acceptance thresholds."""

    required_concurrency: Annotated[int, Field(gt=0)] = ACCEPTANCE_CONCURRENCY
    minimum_successes: Annotated[int, Field(gt=0)] = MINIMUM_SUCCESSFUL_REQUESTS
    maximum_p90_latency_ms: NonNegativeFiniteFloat = P90_LATENCY_THRESHOLD_MS
    maximum_error_rate_exclusive: Annotated[float, Field(gt=0, le=1)] = ERROR_RATE_THRESHOLD
    required_instance_count: Annotated[int, Field(gt=0)] = 1
    required_cache_policy: str = CACHE_BYPASS_POLICY


class WarmupSummary(DomainModel):
    """Traffic excluded from the measured latency sample."""

    readiness_passed: bool
    configured_attempts: Annotated[int, Field(ge=0)]
    attempts: tuple[LoadAttempt, ...]
    success_count: Annotated[int, Field(ge=0)]
    error_count: Annotated[int, Field(ge=0)]
    started_at: AwareDatetime
    completed_at: AwareDatetime
    duration_ms: NonNegativeFiniteFloat

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        successes = sum(attempt.succeeded for attempt in self.attempts)
        if self.completed_at < self.started_at:
            raise ValueError("warm-up completion precedes its start")
        if self.success_count != successes:
            raise ValueError("warm-up success count disagrees with attempt records")
        if self.error_count != len(self.attempts) - successes:
            raise ValueError("warm-up error count disagrees with attempt records")
        if len(self.attempts) > self.configured_attempts:
            raise ValueError("warm-up attempt count exceeds the configured count")
        return self


class LoadReport(DomainModel):
    """Calculated source record for a load run before evidence publication."""

    report_version: str = LOAD_REPORT_VERSION
    run_id: Identifier
    started_at: AwareDatetime
    completed_at: AwareDatetime
    duration_ms: NonNegativeFiniteFloat
    instance_count: Annotated[int, Field(gt=0)]
    configured_concurrency: Annotated[int, Field(gt=0)]
    observed_peak_concurrency: Annotated[int, Field(ge=0)]
    cache_policy: str
    workload_digest: Digest | None = None
    workload_scenario_ids: tuple[Identifier, ...]
    cache_header_name: str = CACHE_POLICY_HEADER
    warmup: WarmupSummary
    attempts: tuple[LoadAttempt, ...]
    attempt_count: Annotated[int, Field(ge=0)]
    success_count: Annotated[int, Field(ge=0)]
    error_count: Annotated[int, Field(ge=0)]
    retry_attempt_count: Annotated[int, Field(ge=0)]
    cache_satisfied_attempt_count: Annotated[int, Field(ge=0)]
    error_rate: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    throughput_successes_per_second: NonNegativeFiniteFloat
    complete_latency_ms: LatencySummary | None
    stage_latency_ms: dict[str, LatencySummary]
    token_totals: dict[str, Annotated[int, Field(ge=0)]]
    model_identities: dict[str, str]
    representative_trace_references: tuple[Identifier, ...]
    thresholds: LoadAcceptanceThresholds
    invalid_reasons: tuple[Identifier, ...]
    failure_reasons: tuple[Identifier, ...]
    valid: bool
    passed: bool

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        successes = sum(attempt.succeeded for attempt in self.attempts)
        errors = len(self.attempts) - successes
        retries = sum(attempt.attempt_number > 1 for attempt in self.attempts)
        cache_hits = sum(attempt.cache_satisfied for attempt in self.attempts)
        expected_error_rate = errors / len(self.attempts) if self.attempts else 0.0
        if self.completed_at < self.started_at:
            raise ValueError("load-run completion precedes its start")
        if self.attempt_count != len(self.attempts):
            raise ValueError("attempt count disagrees with attempt records")
        if (self.success_count, self.error_count) != (successes, errors):
            raise ValueError("success or error count disagrees with attempt records")
        if self.retry_attempt_count != retries:
            raise ValueError("retry count disagrees with attempt records")
        if self.cache_satisfied_attempt_count != cache_hits:
            raise ValueError("cache-hit count disagrees with attempt records")
        if not math.isclose(self.error_rate, expected_error_rate, rel_tol=0, abs_tol=1e-15):
            raise ValueError("error rate disagrees with the complete attempt denominator")
        if self.observed_peak_concurrency > self.configured_concurrency:
            raise ValueError("observed concurrency exceeds the configured worker count")
        if len(self.workload_scenario_ids) != len(set(self.workload_scenario_ids)):
            raise ValueError("workload scenario IDs must be unique")
        expected_latency = summarize_latency(
            attempt.latency_ms for attempt in self.attempts if attempt.succeeded
        )
        if self.complete_latency_ms != expected_latency:
            raise ValueError("complete latency summary disagrees with successful attempts")
        return self


def nearest_rank_percentile(values: Iterable[float], percentile: float) -> float:
    """Return a nearest-rank percentile without interpolation.

    For ``N`` observations the selected one-based rank is ``ceil(P / 100 * N)``.
    This is deliberately not NumPy's interpolated default.
    """

    if isinstance(percentile, bool) or not math.isfinite(percentile):
        raise ValueError("percentile must be finite")
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("latency samples must be finite non-negative numbers")
        sample = float(value)
        if not math.isfinite(sample) or sample < 0:
            raise ValueError("latency samples must be finite non-negative numbers")
        normalized.append(sample)
    if not normalized:
        raise ValueError("at least one latency sample is required")
    normalized.sort()
    rank = math.ceil(percentile / 100 * len(normalized))
    return normalized[rank - 1]


def summarize_latency(values: Iterable[float]) -> LatencySummary | None:
    samples = tuple(float(value) for value in values)
    if not samples:
        return None
    return LatencySummary(
        count=len(samples),
        minimum=min(samples),
        p50=nearest_rank_percentile(samples, 50),
        p90=nearest_rank_percentile(samples, 90),
        p95=nearest_rank_percentile(samples, 95),
        p99=nearest_rank_percentile(samples, 99),
        maximum=max(samples),
    )


def summarize_stages(attempts: Sequence[LoadAttempt]) -> dict[str, LatencySummary]:
    samples: dict[str, list[float]] = {}
    for attempt in attempts:
        for stage, latency_ms in attempt.stage_timings_ms.items():
            samples.setdefault(stage, []).append(latency_ms)
    return {
        stage: summary
        for stage, values in sorted(samples.items())
        if (summary := summarize_latency(values)) is not None
    }


def build_warmup_summary(
    attempts: Sequence[LoadAttempt],
    *,
    readiness_passed: bool,
    configured_attempts: int,
    started_at: datetime,
    completed_at: datetime,
    duration_ms: float,
) -> WarmupSummary:
    records = tuple(attempts)
    successes = sum(attempt.succeeded for attempt in records)
    return WarmupSummary(
        readiness_passed=readiness_passed,
        configured_attempts=configured_attempts,
        attempts=records,
        success_count=successes,
        error_count=len(records) - successes,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=duration_ms,
    )


def build_load_report(
    *,
    run_id: str,
    started_at: datetime,
    completed_at: datetime,
    duration_ms: float,
    instance_count: int,
    configured_concurrency: int,
    observed_peak_concurrency: int,
    cache_policy: str,
    workload_digest: str | None = None,
    workload_scenario_ids: Sequence[str],
    warmup: WarmupSummary,
    attempts: Sequence[LoadAttempt],
    thresholds: LoadAcceptanceThresholds | None = None,
) -> LoadReport:
    """Build every decision from the immutable attempt ledger."""

    records = tuple(attempts)
    resolved_thresholds = thresholds or LoadAcceptanceThresholds()
    success_count = sum(attempt.succeeded for attempt in records)
    error_count = len(records) - success_count
    error_rate = error_count / len(records) if records else 0.0
    retries = sum(attempt.attempt_number > 1 for attempt in records)
    cache_hits = sum(attempt.cache_satisfied for attempt in records)
    latency = summarize_latency(attempt.latency_ms for attempt in records if attempt.succeeded)
    token_totals = _sum_integer_maps(attempt.token_counts for attempt in records)
    model_identities = _consistent_model_identities(records)
    traces = _representative_trace_references(records)
    successful_records = tuple(attempt for attempt in records if attempt.succeeded)

    invalid_reasons: list[str] = []
    if not warmup.readiness_passed:
        invalid_reasons.append("readiness-not-passed")
    if (
        warmup.configured_attempts <= 0
        or len(warmup.attempts) != warmup.configured_attempts
        or warmup.error_count > 0
    ):
        invalid_reasons.append("warmup-incomplete")
    if configured_concurrency != resolved_thresholds.required_concurrency:
        invalid_reasons.append("concurrency-not-fixed-five")
    if observed_peak_concurrency < resolved_thresholds.required_concurrency:
        invalid_reasons.append("five-concurrent-requests-not-observed")
    if instance_count != resolved_thresholds.required_instance_count:
        invalid_reasons.append("instance-count-not-one")
    successful_instance_identities = {
        attempt.instance_identity
        for attempt in successful_records
        if attempt.instance_identity is not None
    }
    if any(attempt.instance_identity is None for attempt in successful_records):
        invalid_reasons.append("instance-identity-evidence-missing")
    if len(successful_instance_identities) > 1 or (
        successful_instance_identities and len(successful_instance_identities) != instance_count
    ):
        invalid_reasons.append("instance-identity-mismatch")
    successful_scenario_ids = {
        attempt.scenario_id for attempt in successful_records if attempt.scenario_id is not None
    }
    if not workload_scenario_ids:
        invalid_reasons.append("workload-scenario-ids-missing")
    elif not set(workload_scenario_ids).issubset(successful_scenario_ids):
        invalid_reasons.append("scenario-success-coverage-missing")
    if any(
        attempt.scenario_id is not None and attempt.scenario_id not in set(workload_scenario_ids)
        for attempt in records
    ):
        invalid_reasons.append("scenario-identity-mismatch")
    if cache_policy != resolved_thresholds.required_cache_policy:
        invalid_reasons.append("cache-policy-not-bypass")
    if cache_hits:
        invalid_reasons.append("cache-satisfied-request-observed")
    if any(
        attempt.cache_status.get("request-policy") != resolved_thresholds.required_cache_policy
        for attempt in successful_records
    ):
        invalid_reasons.append("cache-policy-evidence-missing")
    if success_count < resolved_thresholds.minimum_successes:
        invalid_reasons.append("minimum-success-count-not-met")
    if latency is None or latency.count != success_count:
        invalid_reasons.append("complete-latency-evidence-missing")
    if any(
        not _successful_attempt_stage_evidence_complete(attempt) for attempt in successful_records
    ):
        invalid_reasons.append("stage-evidence-missing")

    valid = not invalid_reasons
    failure_reasons = list(invalid_reasons)
    if latency is None or latency.p90 > resolved_thresholds.maximum_p90_latency_ms:
        failure_reasons.append("p90-latency-threshold-not-met")
    if error_rate >= resolved_thresholds.maximum_error_rate_exclusive:
        failure_reasons.append("error-rate-threshold-not-met")
    passed = valid and not failure_reasons
    duration_seconds = max(0.0, duration_ms / 1_000)
    throughput = success_count / duration_seconds if duration_seconds > 0 else 0.0
    return LoadReport(
        run_id=run_id,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=duration_ms,
        instance_count=instance_count,
        configured_concurrency=configured_concurrency,
        observed_peak_concurrency=observed_peak_concurrency,
        cache_policy=cache_policy,
        workload_digest=workload_digest,
        workload_scenario_ids=tuple(workload_scenario_ids),
        warmup=warmup,
        attempts=records,
        attempt_count=len(records),
        success_count=success_count,
        error_count=error_count,
        retry_attempt_count=retries,
        cache_satisfied_attempt_count=cache_hits,
        error_rate=error_rate,
        throughput_successes_per_second=throughput,
        complete_latency_ms=latency,
        stage_latency_ms=summarize_stages(successful_records),
        token_totals=token_totals,
        model_identities=model_identities,
        representative_trace_references=traces,
        thresholds=resolved_thresholds,
        invalid_reasons=tuple(dict.fromkeys(invalid_reasons)),
        failure_reasons=tuple(dict.fromkeys(failure_reasons)),
        valid=valid,
        passed=passed,
    )


def _sum_integer_maps(values: Iterable[Mapping[str, int]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for value in values:
        for name, count in value.items():
            totals[name] = totals.get(name, 0) + count
    return dict(sorted(totals.items()))


def _consistent_model_identities(attempts: Sequence[LoadAttempt]) -> dict[str, str]:
    identities: dict[str, str] = {}
    inconsistent: set[str] = set()
    for attempt in attempts:
        for role, identity in attempt.model_identities.items():
            previous = identities.get(role)
            if previous is not None and previous != identity:
                inconsistent.add(role)
            else:
                identities[role] = identity
    return {
        role: identity for role, identity in sorted(identities.items()) if role not in inconsistent
    }


def _representative_trace_references(
    attempts: Sequence[LoadAttempt],
) -> tuple[Identifier, ...]:
    """Select correlation evidence before adding a small general sample.

    The order makes the bounded list retain the successful P90 and maximum
    latency attempts plus failure/retry examples, even for runs with hundreds
    of requests.
    """

    successful = sorted(
        (attempt for attempt in attempts if attempt.succeeded),
        key=lambda attempt: attempt.latency_ms,
    )
    selected: list[LoadAttempt] = []
    if successful:
        p90_index = math.ceil(0.90 * len(successful)) - 1
        selected.extend((successful[p90_index], successful[-1]))
    failed = next((attempt for attempt in attempts if not attempt.succeeded), None)
    if failed is not None:
        selected.append(failed)
    retry = next((attempt for attempt in attempts if attempt.attempt_number > 1), None)
    if retry is not None:
        selected.append(retry)
    selected.extend(attempts)
    return tuple(
        dict.fromkeys(attempt.trace_id for attempt in selected if attempt.trace_id is not None)
    )[:20]


def _is_cache_hit(value: str) -> bool:
    normalized = value.strip().casefold().replace("_", "-")
    return normalized in _CACHE_HIT_VALUES or normalized.endswith("-hit")


def _provider_usage_unknown(attempt: ProviderAttemptEvidence) -> bool:
    if attempt.usage.input_tokens is None:
        return True
    return attempt.role is not ModelRole.EMBEDDING and attempt.usage.output_tokens is None


def _has_complete_generation_attempt(
    attempts: Sequence[ProviderAttemptEvidence],
) -> bool:
    return any(
        attempt.role is ModelRole.GENERATION
        and attempt.status is ModelAttemptStatus.SUCCEEDED
        and attempt.usage.input_tokens is not None
        and attempt.usage.output_tokens is not None
        for attempt in attempts
    )


def _successful_attempt_stage_evidence_complete(attempt: LoadAttempt) -> bool:
    required = {"validation", "total"}
    if attempt.terminal_kind == "answer":
        required.update(
            {
                "retrieval",
                "evidence_assessment",
                "generation",
                "finalization",
            }
        )
    else:
        if attempt.provider_attempts or "retrieval" in attempt.stage_timings_ms:
            required.add("retrieval")
    return required.issubset(attempt.stage_timings_ms)


def _provider_token_counts(
    attempts: Sequence[ProviderAttemptEvidence],
) -> dict[str, int]:
    totals: dict[str, int] = {}
    for attempt in attempts:
        role = attempt.role.value
        if attempt.usage.input_tokens is not None:
            key = f"{role}-input"
            totals[key] = totals.get(key, 0) + attempt.usage.input_tokens
        if attempt.usage.output_tokens is not None:
            key = f"{role}-output"
            totals[key] = totals.get(key, 0) + attempt.usage.output_tokens
    return dict(sorted(totals.items()))


__all__ = [
    "ACCEPTANCE_CONCURRENCY",
    "CACHE_BYPASS_POLICY",
    "CACHE_POLICY_HEADER",
    "ERROR_RATE_THRESHOLD",
    "INSTANCE_ID_HEADER",
    "LOAD_REPORT_VERSION",
    "MINIMUM_SUCCESSFUL_REQUESTS",
    "P90_LATENCY_THRESHOLD_MS",
    "LatencySummary",
    "LoadAcceptanceThresholds",
    "LoadAttempt",
    "LoadAttemptStatus",
    "LoadReport",
    "WarmupSummary",
    "build_load_report",
    "build_warmup_summary",
    "nearest_rank_percentile",
    "summarize_latency",
    "summarize_stages",
]
