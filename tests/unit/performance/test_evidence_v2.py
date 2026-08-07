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
    CostUnknownReason,
    EvidenceAvailability,
    ExactPricingRateV2,
    PricingProvenanceV2,
)
from rag_mvp.performance.evidence_v2 import (
    OFFICIAL_P90_SCOPE,
    PerformanceEvidenceV2,
    build_performance_evidence_v2,
)
from rag_mvp.performance.load_report import (
    LoadAcceptanceThresholds,
    LoadAttempt,
    LoadAttemptStatus,
    build_load_report,
    build_warmup_summary,
)

_START = datetime(2026, 8, 7, tzinfo=UTC)


def _pricing() -> PricingProvenanceV2:
    return PricingProvenanceV2.create(
        pricing_version="pricing-2026-08",
        currency="USD",
        rates=(
            ExactPricingRateV2(
                role=ModelRole.GENERATION,
                provider="primary",
                model="chat-v2",
                input_per_million=Decimal("2"),
                output_per_million=Decimal("8"),
            ),
        ),
        source_references=("https://pricing.example/provider/model-card-v2",),
    )


def _attempt(
    identifier: str,
    latency_ms: float,
    *,
    logical_request_id: str | None = None,
    failed: bool = False,
) -> LoadAttempt:
    provider = ProviderAttemptEvidence(
        operation_id="qa-generation",
        attempt_number=1,
        route_id="primary",
        role=ModelRole.GENERATION,
        provider="primary",
        model="chat-v2",
        status=(ModelAttemptStatus.FAILED if failed else ModelAttemptStatus.SUCCEEDED),
        usage=TokenUsage(input_tokens=1_000, output_tokens=500),
    )
    started_at = _START + timedelta(milliseconds=latency_ms)
    return LoadAttempt(
        attempt_id=identifier,
        logical_request_id=logical_request_id or f"logical-{identifier}",
        scenario_id="policy",
        status=(LoadAttemptStatus.TERMINAL_ERROR if failed else LoadAttemptStatus.SUCCEEDED),
        started_at=started_at,
        completed_at=started_at + timedelta(milliseconds=latency_ms),
        latency_ms=latency_ms,
        http_status_code=200,
        request_id=f"request-{identifier}",
        trace_id=f"trace-{identifier}",
        instance_identity="instance-v2",
        terminal_kind="error" if failed else "refusal",
        safe_error_code="capacity" if failed else None,
        retryable=failed,
        provider_attempt_count=1,
        provider_failed_attempt_count=1 if failed else 0,
        provider_attempts=(provider,),
        stage_timings_ms={
            "validation": 0.5,
            "retrieval": latency_ms / 2,
            "total": latency_ms,
        },
        token_counts={"generation-input": 1_000, "generation-output": 500},
        model_identities={"generation": "chat-v2"},
        cache_status={"request-policy": "bypass", "retrieval": "bypass"},
    )


def _evidence() -> PerformanceEvidenceV2:
    warmup = _attempt("warmup-1", 999)
    measured = (
        _attempt("measured-1", 1),
        _attempt("measured-2", 2),
        _attempt("measured-3", 3),
        _attempt("measured-4", 4),
        _attempt("measured-failed", 100, failed=True),
    )
    return build_performance_evidence_v2(
        run_id="performance-run-v2",
        configuration_id="configuration-v2",
        warmup_attempts=(warmup,),
        measured_attempts=measured,
        pricing=_pricing(),
        official_p90_threshold_ms=50,
        created_at=_START,
    )


def test_separates_warmup_and_recomputes_both_nearest_rank_scopes() -> None:
    evidence = _evidence()

    assert evidence.warmup.attempt_count == 1
    assert evidence.warmup.latency_ms.all_attempts is not None
    assert evidence.warmup.latency_ms.all_attempts.p90_ms == 999
    assert evidence.measured.http_attempt_count == 5
    assert evidence.measured.successful_http_attempt_count == 4
    all_attempts = evidence.measured.latency_ms.all_attempts
    successful = evidence.measured.latency_ms.successful_attempts
    assert all_attempts is not None
    assert successful is not None
    assert (
        all_attempts.p50_ms,
        all_attempts.p90_ms,
        all_attempts.p95_ms,
        all_attempts.p99_ms,
    ) == (3, 100, 100, 100)
    assert (
        successful.p50_ms,
        successful.p90_ms,
        successful.p95_ms,
        successful.p99_ms,
    ) == (2, 4, 4, 4)
    assert evidence.measured.error_rate.denominator == 5
    assert evidence.measured.error_rate.value == pytest.approx(0.2)


def test_official_p90_uses_all_measured_http_attempts_and_excludes_warmup_cost() -> None:
    evidence = _evidence()

    gate = evidence.measured.official_p90_gate
    assert gate.scope == OFFICIAL_P90_SCOPE
    assert gate.denominator == 5
    assert gate.observed_ms == 100
    assert gate.threshold_ms == 50
    assert gate.passed is False
    assert evidence.cost.provider_attempt_count == 5
    assert evidence.cost.total_cost == Decimal("0.030")
    assert evidence.cost.cost_per_1000_logical_attempts.per_1000 == Decimal("6")
    assert evidence.cost.cost_per_1000_successes.per_1000 == Decimal("7.5")


def test_v1_successful_only_latency_contract_remains_unchanged() -> None:
    evidence = _evidence()
    warmup_attempt = evidence.warmup.attempts[0]
    report = build_load_report(
        run_id="v1-regression-run",
        started_at=_START,
        completed_at=_START + timedelta(seconds=2),
        duration_ms=2_000,
        instance_count=1,
        configured_concurrency=1,
        observed_peak_concurrency=1,
        cache_policy="bypass",
        workload_scenario_ids=("policy",),
        warmup=build_warmup_summary(
            (warmup_attempt,),
            readiness_passed=True,
            configured_attempts=1,
            started_at=warmup_attempt.started_at,
            completed_at=warmup_attempt.completed_at,
            duration_ms=warmup_attempt.latency_ms,
        ),
        attempts=evidence.measured.attempts,
        thresholds=LoadAcceptanceThresholds(
            required_concurrency=1,
            minimum_successes=4,
            maximum_p90_latency_ms=50,
            maximum_error_rate_exclusive=0.5,
        ),
    )

    assert report.report_version == "http-load-report-v1"
    assert report.complete_latency_ms is not None
    assert report.complete_latency_ms.p90 == 4
    assert report.passed is True
    assert evidence.measured.official_p90_gate.observed_ms == 100
    assert evidence.measured.official_p90_gate.passed is False


def test_derived_latency_and_attempt_ledger_are_tamper_evident() -> None:
    evidence = _evidence()
    latency_payload = evidence.model_dump()
    measured = dict(latency_payload["measured"])
    latency = dict(measured["latency_ms"])
    all_attempts = dict(latency["all_attempts"])
    all_attempts["p50_ms"] = 4
    latency["all_attempts"] = all_attempts
    measured["latency_ms"] = latency
    latency_payload["measured"] = measured
    with pytest.raises(ValidationError, match="measured latency"):
        PerformanceEvidenceV2.model_validate(latency_payload)

    ledger_payload = evidence.model_dump()
    measured = dict(ledger_payload["measured"])
    attempts = measured["attempts"]
    assert isinstance(attempts, tuple)
    first = dict(attempts[0])
    first["latency_ms"] = 1.5
    measured["attempts"] = (first, *attempts[1:])
    ledger_payload["measured"] = measured
    with pytest.raises(ValidationError, match="attempt-ledger digest mismatch"):
        PerformanceEvidenceV2.model_validate(ledger_payload)


def test_empty_measured_denominators_are_explicitly_unavailable() -> None:
    warmup = _attempt("warmup-only", 5)

    evidence = build_performance_evidence_v2(
        run_id="empty-measured-run",
        configuration_id="configuration-v2",
        warmup_attempts=(warmup,),
        measured_attempts=(),
        pricing=_pricing(),
        created_at=_START,
    )

    assert evidence.measured.latency_ms.all_attempts is None
    assert evidence.measured.latency_ms.successful_attempts is None
    assert evidence.measured.error_rate.status is EvidenceAvailability.UNAVAILABLE
    assert evidence.measured.official_p90_gate.status is EvidenceAvailability.UNAVAILABLE
    assert evidence.measured.official_p90_gate.passed is None
    assert evidence.cost.total_cost == Decimal(0)
    assert evidence.cost.cost_per_1000_logical_attempts.per_1000 is None
    assert evidence.cost.cost_per_1000_successes.per_1000 is None
    assert CostUnknownReason.LOGICAL_ATTEMPT_DENOMINATOR_ZERO in evidence.cost.unknown_reasons
    assert CostUnknownReason.SUCCESS_DENOMINATOR_ZERO in evidence.cost.unknown_reasons


def test_warmup_and_measured_logical_requests_cannot_overlap() -> None:
    warmup = _attempt("warmup-overlap", 5, logical_request_id="same-logical")
    measured = _attempt("measured-overlap", 5, logical_request_id="same-logical")

    with pytest.raises(ValueError, match="logical requests overlap"):
        build_performance_evidence_v2(
            run_id="overlap-run",
            configuration_id="configuration-v2",
            warmup_attempts=(warmup,),
            measured_attempts=(measured,),
            pricing=_pricing(),
            created_at=_START,
        )


def test_v2_evidence_round_trips_as_a_frozen_versioned_contract() -> None:
    evidence = _evidence()

    restored = PerformanceEvidenceV2.model_validate_json(evidence.model_dump_json())

    assert restored == evidence
    assert restored.schema_version == "performance-evidence-v2"
    assert restored.cost.schema_version == "provider-cost-evidence-v2"
