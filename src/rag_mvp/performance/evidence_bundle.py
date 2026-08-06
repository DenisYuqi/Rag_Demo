"""Versioned, privacy-safe performance evidence bundles."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any, Never, Self, cast

from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, model_validator

from rag_mvp.domain._base import DomainModel, Identifier
from rag_mvp.domain.evaluation import ModelRole
from rag_mvp.performance.load_report import (
    ACCEPTANCE_CONCURRENCY,
    CACHE_BYPASS_POLICY,
    CACHE_POLICY_HEADER,
    ERROR_RATE_THRESHOLD,
    LOAD_REPORT_VERSION,
    MINIMUM_SUCCESSFUL_REQUESTS,
    P90_LATENCY_THRESHOLD_MS,
    LoadAttempt,
    LoadReport,
    summarize_latency,
    summarize_stages,
)
from rag_mvp.safety.output import OutputRedactionError, redact_output
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, Redactor

EVIDENCE_SCHEMA_VERSION = "1.0.0"
EVIDENCE_SCHEMA_URI = "https://rag-mvp.local/schemas/performance-evidence-v1.schema.json"
EVIDENCE_SCHEMA_RESOURCE = "performance-evidence-v1.schema.json"
MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
PricingEvidenceDigest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
PricingSourceReference = Annotated[
    str,
    Field(min_length=1, max_length=500, pattern=r"^\S+$"),
]
_SUCCESS_REQUIRED_STAGES = frozenset({"validation", "total"})
_ANSWER_REQUIRED_STAGES = frozenset(
    {
        "retrieval",
        "evidence_assessment",
        "generation",
        "finalization",
    }
)
_RETRIEVAL_EVIDENCE_STAGES = frozenset(
    {
        "embedding",
        "query_embedding",
        "dense",
        "bm25",
        "fusion",
        "rerank",
        "reranking",
        "retrieval",
        "evidence_assessment",
    }
)


@dataclass(frozen=True, slots=True)
class EvidenceIssue:
    """Content-free JSON pointer and validation keyword."""

    path: str
    keyword: str
    message: str


class EvidenceValidationError(ValueError):
    def __init__(self, issues: Sequence[EvidenceIssue]) -> None:
        normalized = tuple(issues)
        if not normalized:
            raise ValueError("evidence validation requires at least one issue")
        self.issues = normalized
        first = normalized[0]
        super().__init__(
            f"performance evidence invalid at {first.path} ({first.keyword}); "
            f"{len(normalized)} issue(s)"
        )


class EvidenceSerializationError(ValueError):
    """Raised when evidence is not safely representable as bounded JSON."""


class EvidenceWriteError(OSError):
    """Raised when an immutable evidence artifact cannot be published."""


class PerformanceEvidenceIdentity(DomainModel):
    code_revision: Identifier
    configuration_id: Identifier
    service_version: Identifier
    model_identities: dict[str, str]
    instance_identity: Identifier | None = None
    pricing_evidence_digest: PricingEvidenceDigest | None = None

    @model_validator(mode="after")
    def require_models(self) -> Self:
        if not self.model_identities or any(
            not value.strip() for value in self.model_identities.values()
        ):
            raise ValueError("performance evidence requires model identities")
        return self


class PerformanceEvidenceReferences(DomainModel):
    metrics: tuple[Identifier, ...] = ()
    logs: tuple[Identifier, ...] = ()
    representative_traces: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def require_unique_references(self) -> Self:
        for values in (self.metrics, self.logs, self.representative_traces):
            if len(values) != len(set(values)):
                raise ValueError("evidence references must be unique")
        return self


class PerformanceRateEvidence(DomainModel):
    """One exact provider/model/role rate in the pinned rate card."""

    role: ModelRole
    provider: Identifier
    model: Identifier
    input_per_million: Annotated[Decimal, Field(ge=0)] | None = None
    output_per_million: Annotated[Decimal, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def require_rates(self) -> Self:
        if self.input_per_million is None and self.output_per_million is None:
            raise ValueError("a performance rate requires at least one token price")
        return self


class PerformanceCostEvidence(DomainModel):
    pricing_version: Identifier
    pricing_evidence_digest: PricingEvidenceDigest | None = None
    source_references: tuple[PricingSourceReference, ...] = ()
    currency: str | None = None
    complete: bool = False
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    known_cost: Annotated[Decimal, Field(ge=0)] | None = None
    estimated_cost: Annotated[Decimal, Field(ge=0)] | None = None
    cost_per_1000_calls: Annotated[Decimal, Field(ge=0)] | None = None
    provider_attempt_count: Annotated[int, Field(ge=0)] = 0
    rate_card: tuple[PerformanceRateEvidence, ...] = ()
    unknown_reasons: tuple[Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.-]{0,63}$")], ...] = ()
    assumptions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_completeness(self) -> Self:
        required = (
            self.currency,
            self.input_tokens,
            self.output_tokens,
            self.known_cost,
            self.estimated_cost,
            self.cost_per_1000_calls,
        )
        if self.complete and (any(value is None for value in required) or self.unknown_reasons):
            raise ValueError("complete cost evidence contains unknown values")
        if self.complete and not self.rate_card:
            raise ValueError("complete cost evidence requires its pinned rate card")
        if self.complete and (self.pricing_evidence_digest is None or not self.source_references):
            raise ValueError("complete cost evidence requires pinned pricing provenance")
        if not self.complete and not self.unknown_reasons:
            raise ValueError("incomplete cost evidence requires an unknown reason")
        identities = {(rate.role, rate.provider, rate.model) for rate in self.rate_card}
        if len(identities) != len(self.rate_card):
            raise ValueError("performance rate-card identities must be unique")
        if len(self.source_references) != len(set(self.source_references)):
            raise ValueError("pricing source references must be unique")
        has_pricing_provenance = bool(
            self.pricing_evidence_digest or self.rate_card or self.source_references
        )
        if has_pricing_provenance:
            if (
                self.pricing_evidence_digest is None
                or self.currency is None
                or not self.rate_card
                or not self.source_references
            ):
                raise ValueError("pricing provenance is incomplete")
            expected_digest = canonical_pricing_evidence_digest(
                pricing_version=self.pricing_version,
                currency=self.currency,
                rate_card=self.rate_card,
                source_references=self.source_references,
            )
            if self.pricing_evidence_digest != expected_digest:
                raise ValueError("pricing evidence digest disagrees with the pinned rate card")
        return self


def canonical_pricing_evidence_digest(
    *,
    pricing_version: str,
    currency: str,
    rate_card: Sequence[PerformanceRateEvidence | Mapping[str, object]],
    source_references: Sequence[str],
) -> str:
    """Return a stable digest for exact rates and their declared sources."""

    if not isinstance(pricing_version, str) or not isinstance(currency, str):
        raise ValueError("pricing version and currency are required for provenance")
    normalized_version = pricing_version.strip()
    normalized_currency = currency.strip()
    if not normalized_version or not normalized_currency:
        raise ValueError("pricing version and currency are required for provenance")
    rates = tuple(PerformanceRateEvidence.model_validate(rate) for rate in rate_card)
    identities = {(rate.role, rate.provider, rate.model) for rate in rates}
    if not rates or len(identities) != len(rates):
        raise ValueError("pricing provenance requires unique rate-card identities")
    if any(not isinstance(reference, str) for reference in source_references):
        raise ValueError("pricing provenance requires string source references")
    sources = tuple(sorted(reference.strip() for reference in source_references))
    if (
        not sources
        or len(sources) != len(set(sources))
        or any(
            not reference
            or len(reference) > 500
            or any(character.isspace() for character in reference)
            for reference in sources
        )
    ):
        raise ValueError("pricing provenance requires unique bounded source references")
    canonical_rates = [
        {
            "role": rate.role.value,
            "provider": rate.provider,
            "model": rate.model,
            "input_per_million": _canonical_decimal_text(rate.input_per_million),
            "output_per_million": _canonical_decimal_text(rate.output_per_million),
        }
        for rate in sorted(
            rates,
            key=lambda item: (item.role.value, item.provider, item.model),
        )
    ]
    payload = json.dumps(
        {
            "pricing_version": normalized_version,
            "currency": normalized_currency,
            "rates": canonical_rates,
            "source_references": list(sources),
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def load_performance_evidence_schema() -> dict[str, Any]:
    return deepcopy(_packaged_schema())


@cache
def _packaged_schema() -> dict[str, Any]:
    resource = files("rag_mvp.performance").joinpath("schemas", EVIDENCE_SCHEMA_RESOURCE)
    raw = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("packaged performance evidence schema is not an object")
    schema = cast(dict[str, Any], raw)
    Draft202012Validator.check_schema(schema)
    return schema


@cache
def _validator() -> Draft202012Validator:
    return Draft202012Validator(_packaged_schema(), format_checker=FormatChecker())


def build_performance_evidence_bundle(
    report: LoadReport,
    *,
    identity: PerformanceEvidenceIdentity,
    references: PerformanceEvidenceReferences,
    cost: PerformanceCostEvidence | None = None,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """Map a calculated load report to the frozen machine-readable contract."""

    if not isinstance(report, LoadReport):
        raise TypeError("report must be a LoadReport")
    resolved_cost = cost or PerformanceCostEvidence(
        pricing_version="unconfigured",
        unknown_reasons=("cost-evidence-not-supplied",),
    )
    traces = tuple(
        dict.fromkeys((*references.representative_traces, *report.representative_trace_references))
    )
    resolved_references = PerformanceEvidenceReferences(
        metrics=references.metrics,
        logs=references.logs,
        representative_traces=traces,
    )
    successful_attempts = tuple(attempt for attempt in report.attempts if attempt.succeeded)
    attempts_with_usage = sum(bool(attempt.token_counts) for attempt in report.attempts)
    token_complete = (
        bool(successful_attempts)
        and not any(attempt.provider_unknown_usage_attempt_count for attempt in report.attempts)
        and all(attempt.provider_evidence_complete for attempt in report.attempts)
    )
    token_unknown_reasons: list[str] = []
    if not token_complete:
        if any(not attempt.provider_evidence_complete for attempt in report.attempts):
            token_unknown_reasons.append("provider-attempt-evidence-missing")
        elif any(attempt.provider_unknown_usage_attempt_count for attempt in report.attempts):
            token_unknown_reasons.append("provider-usage-unknown")
        else:
            token_unknown_reasons.append("attempt-token-usage-missing")

    extra_invalid_reasons: list[str] = []
    if not report.stage_latency_ms or any(
        not _successful_attempt_stage_evidence_complete(attempt) for attempt in successful_attempts
    ):
        extra_invalid_reasons.append("stage-evidence-missing")
    if not resolved_references.metrics:
        extra_invalid_reasons.append("metric-reference-missing")
    if not resolved_references.logs:
        extra_invalid_reasons.append("log-reference-missing")
    extra_invalid_reasons.extend(
        _instance_identity_invalid_reasons(
            (*report.warmup.attempts, *report.attempts),
            expected_instance_identity=identity.instance_identity,
            declared_instance_count=report.instance_count,
        )
    )
    if identity.instance_identity is None:
        extra_invalid_reasons.append("instance-identity-missing")
    if not resolved_cost.complete:
        extra_invalid_reasons.append("cost-evidence-incomplete")
    extra_invalid_reasons.extend(
        _trace_reference_invalid_reasons(
            report.attempts,
            resolved_references.representative_traces,
        )
    )
    if _model_identity_mismatch(report.attempts, identity.model_identities):
        extra_invalid_reasons.append("model-identity-mismatch")
    invalid_reasons = tuple(dict.fromkeys((*report.invalid_reasons, *extra_invalid_reasons)))
    failure_reasons = tuple(dict.fromkeys((*report.failure_reasons, *extra_invalid_reasons)))
    valid = not invalid_reasons
    passed = valid and report.passed

    payload: dict[str, object] = {
        "$schema": EVIDENCE_SCHEMA_URI,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "run_id": report.run_id,
        "generated_at": (generated_at or datetime.now(UTC)).isoformat(),
        "identity": {
            "code_revision": identity.code_revision,
            "configuration_id": identity.configuration_id,
            "service_version": identity.service_version,
            "load_harness_version": LOAD_REPORT_VERSION,
            "model_identities": dict(identity.model_identities),
            "instance_identity": identity.instance_identity,
            "pricing_evidence_digest": identity.pricing_evidence_digest,
        },
        "execution": {
            "started_at": report.started_at.isoformat(),
            "completed_at": report.completed_at.isoformat(),
            "duration_ms": report.duration_ms,
            "instance_count": report.instance_count,
            "configured_concurrency": report.configured_concurrency,
            "observed_peak_concurrency": report.observed_peak_concurrency,
            "throughput_successes_per_second": report.throughput_successes_per_second,
            "workload_digest": report.workload_digest,
            "scenario_ids": list(report.workload_scenario_ids),
            "cache_policy": {
                "policy": report.cache_policy,
                "request_header": report.cache_header_name,
                "request_value": report.cache_policy,
                "satisfied_attempts": report.cache_satisfied_attempt_count,
            },
        },
        "warm_up": {
            "readiness_passed": report.warmup.readiness_passed,
            "excluded_from_measurement": True,
            "configured_attempts": report.warmup.configured_attempts,
            "attempt_count": len(report.warmup.attempts),
            "successes": report.warmup.success_count,
            "errors": report.warmup.error_count,
            "started_at": report.warmup.started_at.isoformat(),
            "completed_at": report.warmup.completed_at.isoformat(),
            "duration_ms": report.warmup.duration_ms,
            "records": [_attempt_value(attempt) for attempt in report.warmup.attempts],
        },
        "attempts": {
            "total": report.attempt_count,
            "successful": report.success_count,
            "errors": report.error_count,
            "retry_attempts": report.retry_attempt_count,
            "error_rate": report.error_rate,
            "denominator": "all-http-attempts-including-retries",
            "records": [_attempt_value(attempt) for attempt in report.attempts],
        },
        "latency_ms": {
            "method": "nearest-rank",
            "sample_scope": "successful-complete-http-attempts",
            "complete": _model_value(report.complete_latency_ms),
            "stages": {
                stage: _model_value(summary) for stage, summary in report.stage_latency_ms.items()
            },
        },
        "tokens": {
            "complete": token_complete,
            "attempts_with_usage": attempts_with_usage,
            "totals": dict(report.token_totals),
            "unknown_reasons": token_unknown_reasons,
        },
        "cost": resolved_cost.model_dump(mode="json"),
        "thresholds": {
            "instance_count": {
                "operator": "==",
                "value": report.thresholds.required_instance_count,
            },
            "concurrency": {
                "operator": "==",
                "value": report.thresholds.required_concurrency,
            },
            "successful_requests": {
                "operator": ">=",
                "value": report.thresholds.minimum_successes,
            },
            "p90_latency_ms": {
                "operator": "<=",
                "value": report.thresholds.maximum_p90_latency_ms,
            },
            "error_rate": {
                "operator": "<",
                "value": report.thresholds.maximum_error_rate_exclusive,
            },
            "cache_policy": {"operator": "==", "value": CACHE_BYPASS_POLICY},
        },
        "evidence_references": resolved_references.model_dump(mode="json"),
        "decision": {
            "valid": valid,
            "passed": passed,
            "invalid_reasons": list(invalid_reasons),
            "failure_reasons": list(failure_reasons),
        },
    }
    return validate_performance_evidence_bundle(payload)


def validate_performance_evidence_bundle(
    bundle: Mapping[str, object] | BaseModel,
) -> dict[str, object]:
    normalized = _normalize(bundle)
    issues = [
        EvidenceIssue(
            path=_json_pointer(tuple(error.absolute_path)),
            keyword=str(error.validator),
            message="performance evidence does not match schema",
        )
        for error in sorted(
            _validator().iter_errors(normalized),
            key=lambda item: (tuple(str(part) for part in item.absolute_path), item.validator),
        )
    ]
    if issues:
        raise EvidenceValidationError(issues)
    semantic = _semantic_issues(normalized)
    if semantic:
        raise EvidenceValidationError(semantic)
    return normalized


def prepare_performance_evidence_bundle(
    bundle: Mapping[str, object] | BaseModel,
    *,
    redactor: Redactor = DEFAULT_REDACTOR,
) -> dict[str, object]:
    normalized = _normalize(bundle)
    try:
        redacted = redact_output(normalized, redactor=redactor)
    except (OutputRedactionError, TypeError, ValueError, RecursionError) as error:
        raise EvidenceSerializationError("performance evidence redaction failed") from error
    if not isinstance(redacted, dict):
        raise EvidenceSerializationError("redacted performance evidence is not an object")
    return validate_performance_evidence_bundle(cast(Mapping[str, object], redacted))


def canonical_performance_evidence_json(
    bundle: Mapping[str, object] | BaseModel,
) -> str:
    normalized = validate_performance_evidence_bundle(bundle)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_performance_evidence_bundle(
    bundle: Mapping[str, object] | BaseModel,
    output_path: Path | str,
    *,
    redactor: Redactor = DEFAULT_REDACTOR,
) -> Path:
    target = Path(output_path)
    if target.suffix.casefold() != ".json":
        raise EvidenceWriteError("performance evidence path must end in .json")
    prepared = prepare_performance_evidence_bundle(bundle, redactor=redactor)
    payload = canonical_performance_evidence_json(prepared) + "\n"
    return _atomic_write_text(target, payload)


def load_performance_evidence_bundle(path: Path | str) -> dict[str, object]:
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as error:
        raise EvidenceSerializationError("performance evidence is unavailable") from error
    if size <= 0 or size > MAX_EVIDENCE_BYTES:
        raise EvidenceSerializationError("performance evidence size is outside allowed bounds")
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceSerializationError("performance evidence is not valid UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise EvidenceSerializationError("performance evidence document is not an object")
    return validate_performance_evidence_bundle(cast(Mapping[str, object], raw))


def _semantic_issues(bundle: dict[str, object]) -> tuple[EvidenceIssue, ...]:
    issues: list[EvidenceIssue] = []
    attempts_section = cast(dict[str, object], bundle["attempts"])
    raw_records = cast(list[object], attempts_section["records"])
    attempts = _validated_attempts(raw_records, "/attempts/records", issues)
    warmup_section = cast(dict[str, object], bundle["warm_up"])
    warmup_records = _validated_attempts(
        cast(list[object], warmup_section["records"]),
        "/warm_up/records",
        issues,
    )
    if issues:
        return tuple(issues)

    _check_unique_attempts(attempts, "/attempts/records", issues)
    _check_unique_attempts(warmup_records, "/warm_up/records", issues)
    _check_retry_chains(attempts, issues)
    success_count = sum(attempt.succeeded for attempt in attempts)
    error_count = len(attempts) - success_count
    retry_count = sum(attempt.attempt_number > 1 for attempt in attempts)
    expected_counts = {
        "total": len(attempts),
        "successful": success_count,
        "errors": error_count,
        "retry_attempts": retry_count,
    }
    for field, expected in expected_counts.items():
        if attempts_section[field] != expected:
            issues.append(_issue(f"/attempts/{field}", "attempt-parity"))
    expected_error_rate = error_count / len(attempts) if attempts else 0.0
    if not _float_equal(attempts_section["error_rate"], expected_error_rate):
        issues.append(_issue("/attempts/error_rate", "attempt-denominator"))

    warmup_successes = sum(attempt.succeeded for attempt in warmup_records)
    warmup_expected = {
        "attempt_count": len(warmup_records),
        "successes": warmup_successes,
        "errors": len(warmup_records) - warmup_successes,
    }
    for field, expected in warmup_expected.items():
        if warmup_section[field] != expected:
            issues.append(_issue(f"/warm_up/{field}", "warmup-parity"))

    latency_section = cast(dict[str, object], bundle["latency_ms"])
    expected_latency = summarize_latency(
        attempt.latency_ms for attempt in attempts if attempt.succeeded
    )
    if latency_section["complete"] != _model_value(expected_latency):
        issues.append(_issue("/latency_ms/complete", "nearest-rank-parity"))
    expected_stages = {
        stage: _model_value(summary)
        for stage, summary in summarize_stages(
            tuple(attempt for attempt in attempts if attempt.succeeded)
        ).items()
    }
    if latency_section["stages"] != expected_stages:
        issues.append(_issue("/latency_ms/stages", "stage-summary-parity"))

    token_section = cast(dict[str, object], bundle["tokens"])
    attempts_with_usage = sum(bool(attempt.token_counts) for attempt in attempts)
    token_totals: dict[str, int] = {}
    for attempt in attempts:
        for name, count in attempt.token_counts.items():
            token_totals[name] = token_totals.get(name, 0) + count
    token_totals = dict(sorted(token_totals.items()))
    successful_attempts = tuple(attempt for attempt in attempts if attempt.succeeded)
    token_complete = (
        bool(successful_attempts)
        and not any(attempt.provider_unknown_usage_attempt_count for attempt in attempts)
        and all(attempt.provider_evidence_complete for attempt in attempts)
    )
    if token_section["attempts_with_usage"] != attempts_with_usage:
        issues.append(_issue("/tokens/attempts_with_usage", "token-parity"))
    if token_section["totals"] != token_totals:
        issues.append(_issue("/tokens/totals", "token-parity"))
    if token_section["complete"] is not token_complete:
        issues.append(_issue("/tokens/complete", "token-completeness"))
    token_unknown = cast(list[object], token_section["unknown_reasons"])
    if (token_complete and token_unknown) or (not token_complete and not token_unknown):
        issues.append(_issue("/tokens/unknown_reasons", "token-completeness"))
    expected_token_unknown = (
        []
        if token_complete
        else [
            (
                "provider-attempt-evidence-missing"
                if any(not attempt.provider_evidence_complete for attempt in attempts)
                else (
                    "provider-usage-unknown"
                    if any(attempt.provider_unknown_usage_attempt_count for attempt in attempts)
                    else "attempt-token-usage-missing"
                )
            )
        ]
    )
    if token_unknown != expected_token_unknown:
        issues.append(_issue("/tokens/unknown_reasons", "token-reason-parity"))

    cost = cast(dict[str, object], bundle["cost"])
    cost_required = (
        cost["currency"],
        cost["input_tokens"],
        cost["output_tokens"],
        cost["known_cost"],
        cost["estimated_cost"],
        cost["cost_per_1000_calls"],
    )
    cost_unknown = cast(list[object], cost["unknown_reasons"])
    if cost["complete"] is True and (any(value is None for value in cost_required) or cost_unknown):
        issues.append(_issue("/cost/complete", "cost-completeness"))
    if cost["complete"] is False and not cost_unknown:
        issues.append(_issue("/cost/unknown_reasons", "cost-completeness"))
    input_tokens, output_tokens, unclassified_tokens = _directional_token_totals(attempts)
    if cost["input_tokens"] is not None and cost["input_tokens"] != input_tokens:
        issues.append(_issue("/cost/input_tokens", "cost-token-parity"))
    if cost["output_tokens"] is not None and cost["output_tokens"] != output_tokens:
        issues.append(_issue("/cost/output_tokens", "cost-token-parity"))
    measured_provider_attempts = sum(attempt.provider_attempt_count for attempt in attempts)
    if cast(int, cost["provider_attempt_count"]) != measured_provider_attempts:
        issues.append(_issue("/cost/provider_attempt_count", "provider-attempt-parity"))
    raw_rate_card = cast(list[object], cost["rate_card"])
    rates: tuple[PerformanceRateEvidence, ...] = ()
    try:
        rates = tuple(PerformanceRateEvidence.model_validate(rate) for rate in raw_rate_card)
    except (TypeError, ValueError):
        issues.append(_issue("/cost/rate_card", "rate-card-contract"))
    rate_identities = {(rate.role, rate.provider, rate.model) for rate in rates}
    if len(rate_identities) != len(rates):
        issues.append(_issue("/cost/rate_card", "unique-rate-card-identity"))
    if cost["complete"] is True and not rates:
        issues.append(_issue("/cost/rate_card", "rate-card-required"))
    pricing_digest = cast(str | None, cost["pricing_evidence_digest"])
    source_references = cast(list[str], cost["source_references"])
    identity = cast(dict[str, object], bundle["identity"])
    identity_pricing_digest = cast(str | None, identity["pricing_evidence_digest"])
    has_pricing_provenance = bool(pricing_digest or rates or source_references)
    if cost["complete"] is True and (pricing_digest is None or not source_references):
        issues.append(_issue("/cost/pricing_evidence_digest", "pricing-provenance-required"))
    if has_pricing_provenance:
        try:
            expected_pricing_digest = canonical_pricing_evidence_digest(
                pricing_version=cast(str, cost["pricing_version"]),
                currency=cast(str, cost["currency"]),
                rate_card=rates,
                source_references=source_references,
            )
        except (TypeError, ValueError):
            issues.append(_issue("/cost/source_references", "pricing-provenance-contract"))
        else:
            if pricing_digest != expected_pricing_digest:
                issues.append(
                    _issue(
                        "/cost/pricing_evidence_digest",
                        "pricing-digest-parity",
                    )
                )
    if identity_pricing_digest != pricing_digest:
        issues.append(
            _issue(
                "/identity/pricing_evidence_digest",
                "pricing-digest-binding",
            )
        )
    if rates:
        recalculated = _recalculate_cost(
            attempts,
            rates=rates,
            success_count=success_count,
            currency=cast(str | None, cost["currency"]),
        )
        if "zero-price-with-nonzero-usage" in recalculated.unknown_reasons:
            issues.append(
                _issue(
                    "/cost/rate_card",
                    "zero-price-with-nonzero-usage",
                )
            )
        expected_cost_values: tuple[tuple[str, object], ...] = (
            ("complete", recalculated.complete),
            ("input_tokens", recalculated.input_tokens),
            ("output_tokens", recalculated.output_tokens),
            ("known_cost", recalculated.known_cost),
            ("estimated_cost", recalculated.estimated_cost),
            ("cost_per_1000_calls", recalculated.cost_per_1000_calls),
            ("provider_attempt_count", recalculated.provider_attempt_count),
            ("unknown_reasons", list(recalculated.unknown_reasons)),
        )
        for cost_field, expected_value in expected_cost_values:
            actual = cost[cost_field]
            if isinstance(expected_value, Decimal):
                matches = _decimal(actual) == expected_value
            elif expected_value is None and cost_field in {
                "known_cost",
                "estimated_cost",
                "cost_per_1000_calls",
            }:
                matches = actual is None
            else:
                matches = actual == expected_value
            if not matches:
                issues.append(_issue(f"/cost/{cost_field}", "cost-recalculation-parity"))
    if cost["complete"] is True and any(
        attempt.provider_unknown_usage_attempt_count for attempt in attempts
    ):
        issues.append(_issue("/cost/complete", "cost-usage-completeness"))
    if cost["complete"] is True and unclassified_tokens:
        issues.append(_issue("/cost/complete", "cost-token-classification"))

    issues.extend(
        _decision_issues(
            bundle,
            attempts=attempts,
            warmup_records=warmup_records,
            expected_latency=expected_latency,
        )
    )
    return tuple(issues)


def _decision_issues(
    bundle: dict[str, object],
    *,
    attempts: tuple[LoadAttempt, ...],
    warmup_records: tuple[LoadAttempt, ...],
    expected_latency: object,
) -> tuple[EvidenceIssue, ...]:
    issues: list[EvidenceIssue] = []
    thresholds = cast(dict[str, dict[str, object]], bundle["thresholds"])
    fixed_thresholds: dict[str, tuple[str, float]] = {
        "instance_count": ("==", 1),
        "concurrency": ("==", ACCEPTANCE_CONCURRENCY),
        "successful_requests": (">=", MINIMUM_SUCCESSFUL_REQUESTS),
        "p90_latency_ms": ("<=", P90_LATENCY_THRESHOLD_MS),
        "error_rate": ("<", ERROR_RATE_THRESHOLD),
    }
    for name, (operator, value) in fixed_thresholds.items():
        threshold = thresholds[name]
        if threshold["operator"] != operator or not _float_equal(threshold["value"], value):
            issues.append(_issue(f"/thresholds/{name}", "acceptance-threshold"))

    execution = cast(dict[str, object], bundle["execution"])
    cache = cast(dict[str, object], execution["cache_policy"])
    warmup = cast(dict[str, object], bundle["warm_up"])
    attempt_summary = cast(dict[str, object], bundle["attempts"])
    latency = cast(dict[str, object], bundle["latency_ms"])
    references = cast(dict[str, list[object]], bundle["evidence_references"])
    identity = cast(dict[str, object], bundle["identity"])
    invalid_reasons: list[str] = []
    if warmup["readiness_passed"] is not True:
        invalid_reasons.append("readiness-not-passed")
    if (
        cast(int, warmup["configured_attempts"]) <= 0
        or len(warmup_records) != warmup["configured_attempts"]
        or warmup["errors"] != 0
    ):
        invalid_reasons.append("warmup-incomplete")
    if execution["configured_concurrency"] != ACCEPTANCE_CONCURRENCY:
        invalid_reasons.append("concurrency-not-fixed-five")
    if cast(int, execution["observed_peak_concurrency"]) < ACCEPTANCE_CONCURRENCY:
        invalid_reasons.append("five-concurrent-requests-not-observed")
    if execution["instance_count"] != 1:
        invalid_reasons.append("instance-count-not-one")
    invalid_reasons.extend(
        _instance_identity_invalid_reasons(
            (*warmup_records, *attempts),
            expected_instance_identity=cast(str | None, identity["instance_identity"]),
            declared_instance_count=cast(int, execution["instance_count"]),
        )
    )
    configured_scenario_ids = cast(list[str], execution["scenario_ids"])
    configured_scenarios = set(configured_scenario_ids)
    successful_scenario_ids = {
        attempt.scenario_id
        for attempt in attempts
        if attempt.succeeded and attempt.scenario_id is not None
    }
    if not configured_scenario_ids:
        invalid_reasons.append("workload-scenario-ids-missing")
    elif not configured_scenarios.issubset(successful_scenario_ids):
        invalid_reasons.append("scenario-success-coverage-missing")
    if any(
        attempt.scenario_id is not None and attempt.scenario_id not in configured_scenarios
        for attempt in attempts
    ):
        invalid_reasons.append("scenario-identity-mismatch")
    if identity["instance_identity"] is None:
        invalid_reasons.append("instance-identity-missing")
    if cast(dict[str, object], bundle["cost"])["complete"] is not True:
        invalid_reasons.append("cost-evidence-incomplete")
    if cache["policy"] != CACHE_BYPASS_POLICY or cache["request_value"] != CACHE_BYPASS_POLICY:
        invalid_reasons.append("cache-policy-not-bypass")
    if cast(str, cache["request_header"]).casefold() != CACHE_POLICY_HEADER.casefold():
        invalid_reasons.append("cache-policy-not-bypass")
    if any(
        attempt.succeeded and attempt.cache_status.get("request-policy") != CACHE_BYPASS_POLICY
        for attempt in attempts
    ):
        invalid_reasons.append("cache-policy-evidence-missing")
    observed_cache_hits = sum(attempt.cache_satisfied for attempt in attempts)
    if cache["satisfied_attempts"] != observed_cache_hits:
        issues.append(_issue("/execution/cache_policy/satisfied_attempts", "cache-parity"))
    if observed_cache_hits:
        invalid_reasons.append("cache-satisfied-request-observed")
    if cast(int, attempt_summary["successful"]) < MINIMUM_SUCCESSFUL_REQUESTS:
        invalid_reasons.append("minimum-success-count-not-met")
    if (
        expected_latency is None
        or cast(Any, expected_latency).count != attempt_summary["successful"]
    ):
        invalid_reasons.append("complete-latency-evidence-missing")
    if not cast(dict[str, object], latency["stages"]) or any(
        not _successful_attempt_stage_evidence_complete(attempt)
        for attempt in attempts
        if attempt.succeeded
    ):
        invalid_reasons.append("stage-evidence-missing")
    if not references["metrics"]:
        invalid_reasons.append("metric-reference-missing")
    if not references["logs"]:
        invalid_reasons.append("log-reference-missing")
    trace_references = cast(list[str], references["representative_traces"])
    measured_traces = {attempt.trace_id for attempt in attempts if attempt.trace_id is not None}
    if any(reference not in measured_traces for reference in trace_references):
        issues.append(
            _issue(
                "/evidence_references/representative_traces",
                "trace-reference-parity",
            )
        )
    invalid_reasons.extend(_trace_reference_invalid_reasons(attempts, trace_references))

    configured_models = cast(dict[str, str], identity["model_identities"])
    if _model_identity_mismatch(attempts, configured_models):
        invalid_reasons.append("model-identity-mismatch")

    decision = cast(dict[str, object], bundle["decision"])
    declared_invalid = cast(list[str], decision["invalid_reasons"])
    expected_invalid = list(dict.fromkeys(invalid_reasons))
    if declared_invalid != expected_invalid:
        issues.append(_issue("/decision/invalid_reasons", "decision-parity"))
    valid = not expected_invalid
    if decision["valid"] is not valid:
        issues.append(_issue("/decision/valid", "decision-parity"))

    expected_failures = list(expected_invalid)
    p90 = None if expected_latency is None else cast(Any, expected_latency).p90
    if p90 is None or p90 > P90_LATENCY_THRESHOLD_MS:
        expected_failures.append("p90-latency-threshold-not-met")
    if cast(float, attempt_summary["error_rate"]) >= ERROR_RATE_THRESHOLD:
        expected_failures.append("error-rate-threshold-not-met")
    expected_failures = list(dict.fromkeys(expected_failures))
    if decision["failure_reasons"] != expected_failures:
        issues.append(_issue("/decision/failure_reasons", "decision-parity"))
    passed = valid and not expected_failures
    if decision["passed"] is not passed:
        issues.append(_issue("/decision/passed", "decision-parity"))
    return tuple(issues)


def _validated_attempts(
    records: list[object],
    path: str,
    issues: list[EvidenceIssue],
) -> tuple[LoadAttempt, ...]:
    attempts: list[LoadAttempt] = []
    for index, raw in enumerate(records):
        try:
            attempts.append(LoadAttempt.model_validate(raw))
        except (TypeError, ValueError):
            issues.append(_issue(f"{path}/{index}", "attempt-contract"))
    return tuple(attempts)


def _successful_attempt_stage_evidence_complete(attempt: LoadAttempt) -> bool:
    """Require outcome-specific stage evidence for one successful request."""

    observed = frozenset(attempt.stage_timings_ms)
    required = set(_SUCCESS_REQUIRED_STAGES)
    if attempt.terminal_kind == "answer":
        required.update(_ANSWER_REQUIRED_STAGES)
    elif attempt.terminal_kind == "refusal" and (
        attempt.provider_attempt_count > 0 or bool(observed & _RETRIEVAL_EVIDENCE_STAGES)
    ):
        required.add("retrieval")
    return required.issubset(observed)


def _instance_identity_invalid_reasons(
    attempts: Sequence[LoadAttempt],
    *,
    expected_instance_identity: str | None,
    declared_instance_count: int,
) -> tuple[str, ...]:
    successful = tuple(attempt for attempt in attempts if attempt.succeeded)
    observed = {
        attempt.instance_identity for attempt in successful if attempt.instance_identity is not None
    }
    reasons: list[str] = []
    if any(attempt.instance_identity is None for attempt in successful):
        reasons.append("instance-identity-evidence-missing")
    if (
        len(observed) > 1
        or (bool(observed) and len(observed) != declared_instance_count)
        or (
            expected_instance_identity is not None
            and any(identity != expected_instance_identity for identity in observed)
        )
    ):
        reasons.append("instance-identity-mismatch")
    return tuple(reasons)


def _trace_reference_invalid_reasons(
    attempts: Sequence[LoadAttempt],
    references: Sequence[str],
) -> tuple[str, ...]:
    selected = set(references)
    reasons: list[str] = []
    if not selected:
        reasons.append("trace-reference-missing")

    successful = tuple(attempt for attempt in attempts if attempt.succeeded)
    if successful:
        summary = summarize_latency(attempt.latency_ms for attempt in successful)
        if summary is not None and not any(
            attempt.latency_ms == summary.p90 and attempt.trace_id in selected
            for attempt in successful
        ):
            reasons.append("p90-trace-reference-missing")
        maximum = max(attempt.latency_ms for attempt in successful)
        if not any(
            attempt.latency_ms == maximum and attempt.trace_id in selected for attempt in successful
        ):
            reasons.append("max-latency-trace-reference-missing")
    if any(not attempt.succeeded for attempt in attempts) and not any(
        not attempt.succeeded and attempt.trace_id in selected for attempt in attempts
    ):
        reasons.append("failure-trace-reference-missing")
    if any(attempt.attempt_number > 1 for attempt in attempts) and not any(
        attempt.attempt_number > 1 and attempt.trace_id in selected for attempt in attempts
    ):
        reasons.append("retry-trace-reference-missing")
    return tuple(reasons)


def _directional_token_totals(
    attempts: Sequence[LoadAttempt],
) -> tuple[int, int, tuple[str, ...]]:
    input_tokens = 0
    output_tokens = 0
    unclassified: set[str] = set()
    for attempt in attempts:
        for name, count in attempt.token_counts.items():
            direction = _token_direction(name)
            if direction == "input":
                input_tokens += count
            elif direction == "output":
                output_tokens += count
            else:
                unclassified.add(name)
    return input_tokens, output_tokens, tuple(sorted(unclassified))


@dataclass(frozen=True, slots=True)
class _RecalculatedCost:
    complete: bool
    input_tokens: int
    output_tokens: int
    known_cost: Decimal
    estimated_cost: Decimal | None
    cost_per_1000_calls: Decimal | None
    provider_attempt_count: int
    unknown_reasons: tuple[str, ...]


def _recalculate_cost(
    attempts: Sequence[LoadAttempt],
    *,
    rates: Sequence[PerformanceRateEvidence],
    success_count: int,
    currency: str | None,
) -> _RecalculatedCost:
    input_tokens, output_tokens, unclassified = _directional_token_totals(attempts)
    reasons: list[str] = []
    if unclassified:
        reasons.append("token-direction-unknown")
    rate_index = {(rate.role, rate.provider, rate.model): rate for rate in rates}
    known_parts: list[Decimal] = []
    provider_attempt_count = 0
    for http_attempt in attempts:
        provider_attempt_count += len(http_attempt.provider_attempts)
        for provider_attempt in http_attempt.provider_attempts:
            rate = rate_index.get(
                (provider_attempt.role, provider_attempt.provider, provider_attempt.model)
            )
            if rate is None:
                reasons.append("pricing-rate-missing")
            if provider_attempt.usage.input_tokens is None:
                reasons.append("provider-usage-unknown")
            elif rate is not None:
                if rate.input_per_million is None:
                    reasons.append("input-price-missing")
                else:
                    if provider_attempt.usage.input_tokens > 0 and rate.input_per_million == 0:
                        reasons.append("zero-price-with-nonzero-usage")
                    known_parts.append(
                        Decimal(provider_attempt.usage.input_tokens)
                        * rate.input_per_million
                        / Decimal(1_000_000)
                    )
            if provider_attempt.role is ModelRole.EMBEDDING:
                continue
            if provider_attempt.usage.output_tokens is None:
                reasons.append("provider-usage-unknown")
            elif rate is not None:
                if rate.output_per_million is None:
                    reasons.append("output-price-missing")
                else:
                    if provider_attempt.usage.output_tokens > 0 and rate.output_per_million == 0:
                        reasons.append("zero-price-with-nonzero-usage")
                    known_parts.append(
                        Decimal(provider_attempt.usage.output_tokens)
                        * rate.output_per_million
                        / Decimal(1_000_000)
                    )
    if any(attempt.provider_unknown_usage_attempt_count for attempt in attempts):
        reasons.append("provider-usage-unknown")
    if any(not attempt.provider_evidence_complete for attempt in attempts):
        reasons.append("provider-attempt-evidence-missing")
    if success_count > 0 and provider_attempt_count == 0:
        reasons.append("provider-attempt-evidence-missing")
    if success_count == 0:
        reasons.append("successful-calls-missing")
    if currency is None:
        reasons.append("currency-missing")
    unknown_reasons = tuple(dict.fromkeys(reasons))
    known_cost = sum(known_parts, start=Decimal(0))
    estimated_cost = known_cost if not unknown_reasons else None
    cost_per_1000_calls = (
        estimated_cost * Decimal(1_000) / Decimal(success_count)
        if estimated_cost is not None and success_count > 0
        else None
    )
    return _RecalculatedCost(
        complete=not unknown_reasons,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        known_cost=known_cost,
        estimated_cost=estimated_cost,
        cost_per_1000_calls=cost_per_1000_calls,
        provider_attempt_count=provider_attempt_count,
        unknown_reasons=unknown_reasons,
    )


def _token_direction(name: str) -> str | None:
    normalized = name.strip().casefold().replace("_", "-")
    if normalized in {"input", "output"}:
        return normalized
    if normalized.endswith("-input"):
        return "input"
    if normalized.endswith("-output"):
        return "output"
    return None


def _token_roles(token_counts: Mapping[str, int]) -> frozenset[str]:
    roles: set[str] = set()
    for name in token_counts:
        normalized = name.strip().casefold().replace("_", "-")
        direction = _token_direction(normalized)
        if direction is None or normalized == direction:
            continue
        roles.add(normalized[: -(len(direction) + 1)])
    return frozenset(roles)


def _model_identity_mismatch(
    attempts: Sequence[LoadAttempt],
    configured_models: Mapping[str, str],
) -> bool:
    observed_models: dict[str, str] = {}
    inconsistent_roles: set[str] = set()
    token_roles_missing_models = False
    for attempt in attempts:
        if any(role not in attempt.model_identities for role in _token_roles(attempt.token_counts)):
            token_roles_missing_models = True
        for role, model in attempt.model_identities.items():
            previous = observed_models.get(role)
            if previous is not None and previous != model:
                inconsistent_roles.add(role)
            else:
                observed_models[role] = model
    return bool(
        inconsistent_roles
        or token_roles_missing_models
        or dict(configured_models) != observed_models
    )


def _canonical_decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    normalized = value.normalize()
    return "0" if normalized == 0 else format(normalized, "f")


def _decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (ValueError, ArithmeticError):
        return None
    return parsed if parsed.is_finite() else None


def _check_unique_attempts(
    attempts: tuple[LoadAttempt, ...],
    path: str,
    issues: list[EvidenceIssue],
) -> None:
    identifiers = [attempt.attempt_id for attempt in attempts]
    if len(identifiers) != len(set(identifiers)):
        issues.append(_issue(path, "unique-attempt-id"))


def _check_retry_chains(
    attempts: tuple[LoadAttempt, ...],
    issues: list[EvidenceIssue],
) -> None:
    seen: dict[str, LoadAttempt] = {}
    child_counts: dict[str, int] = {}
    logical_attempt_numbers: dict[str, set[int]] = {}
    for index, attempt in enumerate(attempts):
        numbers = logical_attempt_numbers.setdefault(attempt.logical_request_id, set())
        if attempt.attempt_number in numbers:
            issues.append(
                _issue(
                    f"/attempts/records/{index}/attempt_number",
                    "retry-chain",
                )
            )
        numbers.add(attempt.attempt_number)
        predecessor_id = attempt.retry_of_attempt_id
        if predecessor_id is not None:
            predecessor = seen.get(predecessor_id)
            child_counts[predecessor_id] = child_counts.get(predecessor_id, 0) + 1
            if (
                predecessor is None
                or predecessor.succeeded
                or not predecessor.retryable
                or predecessor.logical_request_id != attempt.logical_request_id
                or predecessor.scenario_id != attempt.scenario_id
                or predecessor.attempt_number + 1 != attempt.attempt_number
                or child_counts[predecessor_id] > 1
            ):
                issues.append(
                    _issue(
                        f"/attempts/records/{index}/retry_of_attempt_id",
                        "retry-chain",
                    )
                )
        seen[attempt.attempt_id] = attempt

    for logical_request_id, numbers in logical_attempt_numbers.items():
        expected = set(range(1, max(numbers, default=0) + 1))
        if numbers != expected:
            first_index = next(
                index
                for index, attempt in enumerate(attempts)
                if attempt.logical_request_id == logical_request_id
            )
            issues.append(
                _issue(
                    f"/attempts/records/{first_index}/attempt_number",
                    "retry-chain",
                )
            )


def _attempt_value(attempt: LoadAttempt) -> dict[str, object]:
    return cast(dict[str, object], attempt.model_dump(mode="json"))


def _model_value(model: BaseModel | None) -> object:
    return None if model is None else model.model_dump(mode="json")


def _normalize(bundle: Mapping[str, object] | BaseModel) -> dict[str, object]:
    try:
        candidate: object = (
            bundle.model_dump(mode="json") if isinstance(bundle, BaseModel) else bundle
        )
        serialized = json.dumps(
            candidate,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        )
        normalized = json.loads(
            serialized,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError, OverflowError, json.JSONDecodeError) as error:
        raise EvidenceSerializationError("performance evidence is not serializable") from error
    if not isinstance(normalized, dict):
        raise EvidenceSerializationError("performance evidence must be an object")
    return cast(dict[str, object], normalized)


def _json_default(value: object) -> object:
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise TypeError("non-finite decimal")
        return format(value, "f")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError("unsupported evidence value")


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceSerializationError("performance evidence contains a duplicate key")
        result[key] = value
    return result


def _reject_constant(value: str) -> Never:
    del value
    raise EvidenceSerializationError("performance evidence contains a non-finite number")


def _float_equal(value: object, expected: float) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isclose(float(value), expected, rel_tol=0, abs_tol=1e-15)
    )


def _issue(path: str, keyword: str) -> EvidenceIssue:
    return EvidenceIssue(path, keyword, "performance evidence semantic invariant failed")


def _json_pointer(path: tuple[object, ...]) -> str:
    if not path:
        return "/"
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in path)


def _atomic_write_text(target: Path, payload: str) -> Path:
    expanded = target.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    if absolute.is_symlink():
        raise EvidenceWriteError("performance evidence target must not be a symbolic link")
    resolved = absolute.parent.resolve() / absolute.name
    resolved.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if resolved.exists():
        raise EvidenceWriteError("immutable performance evidence already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, resolved)
        except FileExistsError as error:
            raise EvidenceWriteError("immutable performance evidence already exists") from error
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return resolved


__all__ = [
    "EVIDENCE_SCHEMA_URI",
    "EVIDENCE_SCHEMA_VERSION",
    "EvidenceIssue",
    "EvidenceSerializationError",
    "EvidenceValidationError",
    "EvidenceWriteError",
    "PerformanceCostEvidence",
    "PerformanceEvidenceIdentity",
    "PerformanceEvidenceReferences",
    "PerformanceRateEvidence",
    "build_performance_evidence_bundle",
    "canonical_performance_evidence_json",
    "canonical_pricing_evidence_digest",
    "load_performance_evidence_bundle",
    "load_performance_evidence_schema",
    "prepare_performance_evidence_bundle",
    "validate_performance_evidence_bundle",
    "write_performance_evidence_bundle",
]
