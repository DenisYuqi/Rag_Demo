from __future__ import annotations

import copy
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
from rag_mvp.performance.evidence_bundle import (
    EVIDENCE_SCHEMA_URI,
    EvidenceValidationError,
    EvidenceWriteError,
    PerformanceCostEvidence,
    PerformanceEvidenceIdentity,
    PerformanceEvidenceReferences,
    build_performance_evidence_bundle,
    canonical_pricing_evidence_digest,
    load_performance_evidence_bundle,
    load_performance_evidence_schema,
    validate_performance_evidence_bundle,
    write_performance_evidence_bundle,
)
from rag_mvp.performance.load_report import (
    LoadAttempt,
    LoadAttemptStatus,
    LoadReport,
    build_load_report,
    build_warmup_summary,
)
from rag_mvp.performance.pricing import (
    PerformancePricingEvidence,
    PerformanceRolePricing,
    calculate_performance_cost,
)

_START = datetime(2026, 8, 7, 1, 2, 3, tzinfo=UTC)


def _attempt(
    index: int,
    *,
    scope: str = "measured",
    status: LoadAttemptStatus = LoadAttemptStatus.SUCCEEDED,
    logical_request_id: str | None = None,
    attempt_number: int = 1,
    retry_of_attempt_id: str | None = None,
) -> LoadAttempt:
    started = _START + timedelta(milliseconds=index * 20)
    succeeded = status is LoadAttemptStatus.SUCCEEDED
    provider_attempt = ProviderAttemptEvidence(
        operation_id="qa-generation",
        route_id="generation-primary",
        role=ModelRole.GENERATION,
        provider="test",
        model="model-v1",
        status=(ModelAttemptStatus.SUCCEEDED if succeeded else ModelAttemptStatus.FAILED),
        usage=TokenUsage(input_tokens=2, output_tokens=1),
    )
    return LoadAttempt(
        attempt_id=f"{scope}-attempt-{index:06d}",
        logical_request_id=logical_request_id or f"{scope}-request-{index:06d}",
        attempt_number=attempt_number,
        retry_of_attempt_id=retry_of_attempt_id,
        status=status,
        started_at=started,
        completed_at=started + timedelta(milliseconds=index),
        latency_ms=float(index),
        http_status_code=200,
        request_id=f"request-{scope}-{index:06d}",
        trace_id=f"trace-{scope}-{index:06d}",
        instance_identity="container-sha256-abc123",
        scenario_id="scenario-basic",
        terminal_kind="refusal" if succeeded else "error",
        safe_error_code=None if succeeded else "capacity",
        retryable=not succeeded,
        provider_attempt_count=1,
        provider_failed_attempt_count=0 if succeeded else 1,
        provider_unknown_usage_attempt_count=0,
        provider_attempts=(provider_attempt,),
        stage_timings_ms={
            "validation": index / 10,
            "retrieval": index / 4,
            "evidence_assessment": index / 5,
            "generation": index / 2,
            "finalization": index / 10,
            "total": float(index),
        },
        token_counts={"generation-input": 2, "generation-output": 1},
        model_identities={"generation": "model-v1"},
        cache_status={"request-policy": "bypass", "retrieval": "bypass"},
    )


def _accepted_report() -> LoadReport:
    warmup_records = tuple(_attempt(index, scope="warmup") for index in range(1, 6))
    warmup = build_warmup_summary(
        warmup_records,
        readiness_passed=True,
        configured_attempts=5,
        started_at=_START,
        completed_at=_START + timedelta(milliseconds=100),
        duration_ms=100,
    )
    attempts = tuple(_attempt(index) for index in range(1, 501))
    return build_load_report(
        run_id="load-accepted-001",
        started_at=_START + timedelta(seconds=1),
        completed_at=_START + timedelta(seconds=51),
        duration_ms=50_000,
        instance_count=1,
        configured_concurrency=5,
        observed_peak_concurrency=5,
        cache_policy="bypass",
        workload_scenario_ids=("scenario-basic",),
        warmup=warmup,
        attempts=attempts,
    )


def _accepted_retry_report() -> LoadReport:
    warmup_records = tuple(_attempt(index, scope="warmup") for index in range(1, 6))
    warmup = build_warmup_summary(
        warmup_records,
        readiness_passed=True,
        configured_attempts=5,
        started_at=_START,
        completed_at=_START + timedelta(milliseconds=100),
        duration_ms=100,
    )
    attempts = (
        _attempt(
            1,
            status=LoadAttemptStatus.TERMINAL_ERROR,
            logical_request_id="measured-retry-request",
        ),
        _attempt(
            2,
            logical_request_id="measured-retry-request",
            attempt_number=2,
            retry_of_attempt_id="measured-attempt-000001",
        ),
        *tuple(_attempt(index) for index in range(3, 502)),
    )
    return build_load_report(
        run_id="load-accepted-retry-001",
        started_at=_START + timedelta(seconds=1),
        completed_at=_START + timedelta(seconds=51),
        duration_ms=50_000,
        instance_count=1,
        configured_concurrency=5,
        observed_peak_concurrency=5,
        cache_policy="bypass",
        workload_scenario_ids=("scenario-basic",),
        warmup=warmup,
        attempts=attempts,
    )


def _identity() -> PerformanceEvidenceIdentity:
    pricing = _pricing()
    return PerformanceEvidenceIdentity(
        code_revision="sha256:" + "a" * 64,
        configuration_id="config-v1",
        service_version="0.1.0",
        model_identities={"generation": "model-v1"},
        instance_identity="container-sha256-abc123",
        pricing_evidence_digest=canonical_pricing_evidence_digest(
            pricing_version=pricing.pricing_version,
            currency=pricing.currency,
            rate_card=pricing.rates,
            source_references=pricing.source_references,
        ),
    )


def _references() -> PerformanceEvidenceReferences:
    return PerformanceEvidenceReferences(
        metrics=("metrics-snapshot-001",),
        logs=("structured-log-001",),
        representative_traces=("trace-measured-000001",),
    )


def _pricing() -> PerformancePricingEvidence:
    return PerformancePricingEvidence(
        pricing_version="pricing-v1",
        currency="USD",
        rates=(
            PerformanceRolePricing(
                role=ModelRole.GENERATION,
                provider="test",
                model="model-v1",
                input_per_million=Decimal("5"),
                output_per_million=Decimal("10"),
            ),
        ),
        source_references=("https://example.test/pricing-v1",),
        assumptions=("all provider attempts including retries are counted",),
    )


def _cost(report: LoadReport | None = None) -> PerformanceCostEvidence:
    return calculate_performance_cost(report or _accepted_report(), _pricing())


def _bundle() -> dict[str, object]:
    report = _accepted_report()
    return build_performance_evidence_bundle(
        report,
        identity=_identity(),
        references=_references(),
        cost=_cost(report),
        generated_at=_START,
    )


def _retry_bundle() -> dict[str, object]:
    report = _accepted_retry_report()
    return build_performance_evidence_bundle(
        report,
        identity=_identity(),
        references=_references(),
        cost=_cost(report),
        generated_at=_START,
    )


def _with_attempts(report: LoadReport, attempts: tuple[LoadAttempt, ...]) -> LoadReport:
    return build_load_report(
        run_id=report.run_id,
        started_at=report.started_at,
        completed_at=report.completed_at,
        duration_ms=report.duration_ms,
        instance_count=report.instance_count,
        configured_concurrency=report.configured_concurrency,
        observed_peak_concurrency=report.observed_peak_concurrency,
        cache_policy=report.cache_policy,
        workload_digest=report.workload_digest,
        workload_scenario_ids=report.workload_scenario_ids,
        warmup=report.warmup,
        attempts=attempts,
        thresholds=report.thresholds,
    )


def _with_warmup_attempts(
    report: LoadReport,
    attempts: tuple[LoadAttempt, ...],
) -> LoadReport:
    warmup = build_warmup_summary(
        attempts,
        readiness_passed=report.warmup.readiness_passed,
        configured_attempts=report.warmup.configured_attempts,
        started_at=report.warmup.started_at,
        completed_at=report.warmup.completed_at,
        duration_ms=report.warmup.duration_ms,
    )
    return build_load_report(
        run_id=report.run_id,
        started_at=report.started_at,
        completed_at=report.completed_at,
        duration_ms=report.duration_ms,
        instance_count=report.instance_count,
        configured_concurrency=report.configured_concurrency,
        observed_peak_concurrency=report.observed_peak_concurrency,
        cache_policy=report.cache_policy,
        workload_digest=report.workload_digest,
        workload_scenario_ids=report.workload_scenario_ids,
        warmup=warmup,
        attempts=report.attempts,
        thresholds=report.thresholds,
    )


def test_schema_and_semantic_validation_accept_complete_bundle() -> None:
    schema = load_performance_evidence_schema()
    bundle = _bundle()

    assert schema["$id"] == EVIDENCE_SCHEMA_URI
    assert bundle["$schema"] == EVIDENCE_SCHEMA_URI
    assert bundle["decision"] == {
        "valid": True,
        "passed": True,
        "invalid_reasons": [],
        "failure_reasons": [],
    }
    attempts = bundle["attempts"]
    assert isinstance(attempts, dict)
    assert attempts["total"] == 500
    assert attempts["denominator"] == "all-http-attempts-including-retries"
    latency = bundle["latency_ms"]
    assert isinstance(latency, dict)
    complete = latency["complete"]
    assert isinstance(complete, dict)
    assert complete["p90"] == 450.0
    execution = bundle["execution"]
    assert isinstance(execution, dict)
    assert execution["scenario_ids"] == ["scenario-basic"]


def test_passing_bundle_requires_a_correlated_instance_identity() -> None:
    identity = _identity().model_copy(update={"instance_identity": None})

    bundle = build_performance_evidence_bundle(
        _accepted_report(),
        identity=identity,
        references=_references(),
        cost=_cost(),
        generated_at=_START,
    )

    assert bundle["decision"] == {
        "valid": False,
        "passed": False,
        "invalid_reasons": ["instance-identity-missing"],
        "failure_reasons": ["instance-identity-missing"],
    }


@pytest.mark.parametrize("scope", ["measured", "warmup"])
@pytest.mark.parametrize(
    ("instance_identity", "expected_reason"),
    [
        (None, "instance-identity-evidence-missing"),
        ("container-sha256-unexpected", "instance-identity-mismatch"),
    ],
)
def test_successful_attempt_instance_identity_is_bound_to_bundle_identity(
    scope: str,
    instance_identity: str | None,
    expected_reason: str,
) -> None:
    report = _accepted_report()
    source = report.attempts if scope == "measured" else report.warmup.attempts
    values = source[0].model_dump()
    values["instance_identity"] = instance_identity
    changed = LoadAttempt.model_validate(values)
    changed_records = (changed, *source[1:])
    report = (
        _with_attempts(report, changed_records)
        if scope == "measured"
        else _with_warmup_attempts(report, changed_records)
    )

    bundle = build_performance_evidence_bundle(
        report,
        identity=_identity(),
        references=_references(),
        cost=_cost(report),
        generated_at=_START,
    )

    decision = bundle["decision"]
    assert isinstance(decision, dict)
    assert expected_reason in decision["invalid_reasons"]
    validate_performance_evidence_bundle(bundle)


def test_execution_instance_count_matches_distinct_successful_instance_ids() -> None:
    report = _accepted_report()
    report = build_load_report(
        run_id=report.run_id,
        started_at=report.started_at,
        completed_at=report.completed_at,
        duration_ms=report.duration_ms,
        instance_count=2,
        configured_concurrency=report.configured_concurrency,
        observed_peak_concurrency=report.observed_peak_concurrency,
        cache_policy=report.cache_policy,
        workload_digest=report.workload_digest,
        workload_scenario_ids=report.workload_scenario_ids,
        warmup=report.warmup,
        attempts=report.attempts,
        thresholds=report.thresholds,
    )

    bundle = build_performance_evidence_bundle(
        report,
        identity=_identity(),
        references=_references(),
        cost=_cost(report),
        generated_at=_START,
    )

    decision = bundle["decision"]
    assert isinstance(decision, dict)
    assert "instance-count-not-one" in decision["invalid_reasons"]
    assert "instance-identity-mismatch" in decision["invalid_reasons"]
    validate_performance_evidence_bundle(bundle)


def test_configured_scenario_requires_a_successful_measured_attempt() -> None:
    report = _accepted_report()
    failed = _attempt(501, status=LoadAttemptStatus.TERMINAL_ERROR).model_copy(
        update={"scenario_id": "scenario-uncovered"}
    )
    report = build_load_report(
        run_id=report.run_id,
        started_at=report.started_at,
        completed_at=report.completed_at,
        duration_ms=report.duration_ms,
        instance_count=report.instance_count,
        configured_concurrency=report.configured_concurrency,
        observed_peak_concurrency=report.observed_peak_concurrency,
        cache_policy=report.cache_policy,
        workload_digest=report.workload_digest,
        workload_scenario_ids=("scenario-basic", "scenario-uncovered"),
        warmup=report.warmup,
        attempts=(*report.attempts, failed),
        thresholds=report.thresholds,
    )

    bundle = build_performance_evidence_bundle(
        report,
        identity=_identity(),
        references=_references(),
        cost=_cost(report),
        generated_at=_START,
    )

    decision = bundle["decision"]
    assert isinstance(decision, dict)
    assert "scenario-success-coverage-missing" in decision["invalid_reasons"]
    validate_performance_evidence_bundle(bundle)


def test_validation_rejects_duplicate_or_uncovered_raw_scenario_ids() -> None:
    duplicate = copy.deepcopy(_bundle())
    execution = duplicate["execution"]
    assert isinstance(execution, dict)
    execution["scenario_ids"] = ["scenario-basic", "scenario-basic"]

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(duplicate)

    assert any(issue.keyword == "uniqueItems" for issue in captured.value.issues)

    uncovered = copy.deepcopy(_bundle())
    execution = uncovered["execution"]
    assert isinstance(execution, dict)
    execution["scenario_ids"] = ["scenario-uncovered"]

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(uncovered)

    assert any(issue.keyword == "decision-parity" for issue in captured.value.issues)


def test_validation_recomputes_attempt_counts_and_nearest_rank_percentiles() -> None:
    hidden_failure = copy.deepcopy(_bundle())
    attempts = hidden_failure["attempts"]
    assert isinstance(attempts, dict)
    records = attempts["records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    record.update(
        {
            "status": "terminal-error",
            "terminal_kind": "error",
            "safe_error_code": "capacity",
            "retryable": True,
        }
    )

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(hidden_failure)

    assert any(issue.keyword == "attempt-parity" for issue in captured.value.issues)

    wrong_percentile = copy.deepcopy(_bundle())
    latency = wrong_percentile["latency_ms"]
    assert isinstance(latency, dict)
    complete = latency["complete"]
    assert isinstance(complete, dict)
    complete["p90"] = 449.9

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(wrong_percentile)

    assert any(issue.keyword == "nearest-rank-parity" for issue in captured.value.issues)


@pytest.mark.parametrize("missing_stage", ["validation", "total"])
def test_successful_attempt_requires_common_stage_evidence(missing_stage: str) -> None:
    report = _accepted_report()
    values = report.attempts[0].model_dump()
    timings = dict(values["stage_timings_ms"])
    timings.pop(missing_stage)
    values["stage_timings_ms"] = timings
    incomplete = LoadAttempt.model_validate(values)
    report = _with_attempts(report, (incomplete, *report.attempts[1:]))

    bundle = build_performance_evidence_bundle(
        report,
        identity=_identity(),
        references=_references(),
        cost=_cost(report),
        generated_at=_START,
    )

    decision = bundle["decision"]
    assert isinstance(decision, dict)
    assert "stage-evidence-missing" in decision["invalid_reasons"]
    validate_performance_evidence_bundle(bundle)


def test_successful_answer_requires_all_answer_stage_evidence() -> None:
    report = _accepted_report()
    values = report.attempts[0].model_dump()
    values["terminal_kind"] = "answer"
    timings = dict(values["stage_timings_ms"])
    timings.pop("finalization")
    values["stage_timings_ms"] = timings
    incomplete = LoadAttempt.model_validate(values)
    report = _with_attempts(report, (incomplete, *report.attempts[1:]))

    bundle = build_performance_evidence_bundle(
        report,
        identity=_identity(),
        references=_references(),
        cost=_cost(report),
        generated_at=_START,
    )

    decision = bundle["decision"]
    assert isinstance(decision, dict)
    assert "stage-evidence-missing" in decision["invalid_reasons"]
    validate_performance_evidence_bundle(bundle)


def test_provider_backed_refusal_requires_retrieval_stage_evidence() -> None:
    report = _accepted_report()
    values = report.attempts[0].model_dump()
    timings = dict(values["stage_timings_ms"])
    timings.pop("retrieval")
    values["stage_timings_ms"] = timings
    incomplete = LoadAttempt.model_validate(values)
    report = _with_attempts(report, (incomplete, *report.attempts[1:]))

    bundle = build_performance_evidence_bundle(
        report,
        identity=_identity(),
        references=_references(),
        cost=_cost(report),
        generated_at=_START,
    )

    decision = bundle["decision"]
    assert isinstance(decision, dict)
    assert "stage-evidence-missing" in decision["invalid_reasons"]
    validate_performance_evidence_bundle(bundle)


def test_failed_attempt_stage_timings_do_not_enter_success_latency_summaries() -> None:
    report = _accepted_retry_report()
    values = report.attempts[0].model_dump()
    timings = dict(values["stage_timings_ms"])
    timings["failed-only-stage"] = 0.5
    values["stage_timings_ms"] = timings
    failed = LoadAttempt.model_validate(values)
    report = _with_attempts(report, (failed, *report.attempts[1:]))

    bundle = build_performance_evidence_bundle(
        report,
        identity=_identity(),
        references=_references(),
        cost=_cost(report),
        generated_at=_START,
    )

    latency = bundle["latency_ms"]
    assert isinstance(latency, dict)
    stages = latency["stages"]
    assert isinstance(stages, dict)
    assert "failed-only-stage" not in stages
    validate_performance_evidence_bundle(bundle)


def test_validation_rejects_broken_retry_chain_even_when_counts_match() -> None:
    bundle = _bundle()
    attempts = bundle["attempts"]
    assert isinstance(attempts, dict)
    records = attempts["records"]
    assert isinstance(records, list)
    second = records[1]
    assert isinstance(second, dict)
    second["attempt_number"] = 2
    second["retry_of_attempt_id"] = "measured-attempt-000001"

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(bundle)

    assert any(issue.keyword == "retry-chain" for issue in captured.value.issues)


def test_validation_rejects_retry_of_non_retryable_predecessor() -> None:
    bundle = _retry_bundle()
    attempts = bundle["attempts"]
    assert isinstance(attempts, dict)
    records = attempts["records"]
    assert isinstance(records, list)
    predecessor = records[0]
    assert isinstance(predecessor, dict)
    predecessor["retryable"] = False

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(bundle)

    assert any(issue.keyword == "retry-chain" for issue in captured.value.issues)


def test_validation_rejects_retry_with_a_different_scenario() -> None:
    bundle = _retry_bundle()
    attempts = bundle["attempts"]
    assert isinstance(attempts, dict)
    records = attempts["records"]
    assert isinstance(records, list)
    retry = records[1]
    assert isinstance(retry, dict)
    retry["scenario_id"] = "different-scenario"

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(bundle)

    assert any(issue.keyword == "retry-chain" for issue in captured.value.issues)


def test_validation_rejects_duplicate_or_non_contiguous_logical_attempt_numbers() -> None:
    duplicate = _retry_bundle()
    attempts = duplicate["attempts"]
    assert isinstance(attempts, dict)
    records = attempts["records"]
    assert isinstance(records, list)
    independent = records[2]
    assert isinstance(independent, dict)
    independent["logical_request_id"] = "measured-retry-request"

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(duplicate)

    assert any(issue.keyword == "retry-chain" for issue in captured.value.issues)

    gap = _retry_bundle()
    attempts = gap["attempts"]
    assert isinstance(attempts, dict)
    records = attempts["records"]
    assert isinstance(records, list)
    retry = records[1]
    assert isinstance(retry, dict)
    retry["attempt_number"] = 3

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(gap)

    assert any(issue.keyword == "retry-chain" for issue in captured.value.issues)


def test_validation_rejects_multiple_retry_children_for_one_predecessor() -> None:
    bundle = _retry_bundle()
    attempts = bundle["attempts"]
    assert isinstance(attempts, dict)
    records = attempts["records"]
    assert isinstance(records, list)
    second_child = records[2]
    assert isinstance(second_child, dict)
    second_child["logical_request_id"] = "measured-retry-request"
    second_child["attempt_number"] = 2
    second_child["retry_of_attempt_id"] = "measured-attempt-000001"
    attempts["retry_attempts"] = 2

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(bundle)

    assert any(issue.keyword == "retry-chain" for issue in captured.value.issues)


def test_validation_rejects_unmeasured_or_missing_latency_trace_references() -> None:
    forged = copy.deepcopy(_bundle())
    references = forged["evidence_references"]
    assert isinstance(references, dict)
    traces = references["representative_traces"]
    assert isinstance(traces, list)
    traces.append("trace-not-in-measured-attempts")

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(forged)

    assert any(issue.keyword == "trace-reference-parity" for issue in captured.value.issues)

    uncovered = copy.deepcopy(_bundle())
    references = uncovered["evidence_references"]
    assert isinstance(references, dict)
    traces = references["representative_traces"]
    assert isinstance(traces, list)
    traces.remove("trace-measured-000450")

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(uncovered)

    assert any(issue.keyword == "decision-parity" for issue in captured.value.issues)


def test_validation_recomputes_missing_cache_policy_evidence() -> None:
    bundle = copy.deepcopy(_bundle())
    attempts = bundle["attempts"]
    assert isinstance(attempts, dict)
    records = attempts["records"]
    assert isinstance(records, list)
    first = records[0]
    assert isinstance(first, dict)
    cache_status = first["cache_status"]
    assert isinstance(cache_status, dict)
    cache_status.pop("request-policy")

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(bundle)

    assert any(issue.keyword == "decision-parity" for issue in captured.value.issues)

    decision = bundle["decision"]
    assert isinstance(decision, dict)
    decision["invalid_reasons"] = ["cache-policy-evidence-missing"]
    decision["failure_reasons"] = ["cache-policy-evidence-missing"]
    decision["valid"] = False
    decision["passed"] = False
    validated = validate_performance_evidence_bundle(bundle)
    assert validated["decision"] == decision


@pytest.mark.parametrize(
    ("removed_trace", "missing_reason"),
    [
        ("trace-measured-000001", "failure-trace-reference-missing"),
        ("trace-measured-000002", "retry-trace-reference-missing"),
    ],
)
def test_validation_requires_failure_and_retry_trace_coverage(
    removed_trace: str,
    missing_reason: str,
) -> None:
    report = _accepted_retry_report()
    cost = _cost(report)
    bundle = build_performance_evidence_bundle(
        report,
        identity=_identity(),
        references=PerformanceEvidenceReferences(
            metrics=("metrics-snapshot-001",),
            logs=("structured-log-001",),
        ),
        cost=cost,
        generated_at=_START,
    )
    references = bundle["evidence_references"]
    assert isinstance(references, dict)
    traces = references["representative_traces"]
    assert isinstance(traces, list)
    traces.remove(removed_trace)

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(bundle)

    decision = bundle["decision"]
    assert isinstance(decision, dict)
    decision["invalid_reasons"] = [missing_reason]
    decision["failure_reasons"] = [missing_reason]
    decision["valid"] = False
    decision["passed"] = False
    validated = validate_performance_evidence_bundle(bundle)
    assert validated["decision"] == decision
    assert any(issue.keyword == "decision-parity" for issue in captured.value.issues)


@pytest.mark.parametrize(
    ("field", "value", "keyword"),
    [
        ("input_tokens", 999, "cost-token-parity"),
        ("output_tokens", 499, "cost-token-parity"),
        ("estimated_cost", "0.02", "cost-recalculation-parity"),
        ("cost_per_1000_calls", "0.03", "cost-recalculation-parity"),
        ("provider_attempt_count", 501, "provider-attempt-parity"),
    ],
)
def test_validation_recomputes_cost_evidence(
    field: str,
    value: object,
    keyword: str,
) -> None:
    tampered = copy.deepcopy(_bundle())
    cost = tampered["cost"]
    assert isinstance(cost, dict)
    cost[field] = value

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(tampered)

    assert any(issue.keyword == keyword for issue in captured.value.issues)


def test_validation_rejects_complete_zero_cost_against_nonzero_rate_card() -> None:
    tampered = copy.deepcopy(_bundle())
    cost = tampered["cost"]
    assert isinstance(cost, dict)
    cost["known_cost"] = "0"
    cost["estimated_cost"] = "0"
    cost["cost_per_1000_calls"] = "0"

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(tampered)

    assert any(issue.keyword == "cost-recalculation-parity" for issue in captured.value.issues)


def test_validation_rejects_zero_cost_with_simultaneously_forged_zero_rate_card() -> None:
    tampered = copy.deepcopy(_bundle())
    cost = tampered["cost"]
    assert isinstance(cost, dict)
    rate_card = cost["rate_card"]
    assert isinstance(rate_card, list)
    for rate in rate_card:
        assert isinstance(rate, dict)
        if rate["input_per_million"] is not None:
            rate["input_per_million"] = "0"
        if rate["output_per_million"] is not None:
            rate["output_per_million"] = "0"
    cost["complete"] = True
    cost["known_cost"] = "0"
    cost["estimated_cost"] = "0"
    cost["cost_per_1000_calls"] = "0"
    cost["unknown_reasons"] = []

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(tampered)

    assert any(issue.keyword == "zero-price-with-nonzero-usage" for issue in captured.value.issues)


def test_validation_rejects_duplicate_raw_rate_card_identity() -> None:
    tampered = copy.deepcopy(_bundle())
    cost = tampered["cost"]
    assert isinstance(cost, dict)
    rate_card = cost["rate_card"]
    assert isinstance(rate_card, list)
    rate_card.append(copy.deepcopy(rate_card[0]))

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(tampered)

    assert any(issue.keyword == "unique-rate-card-identity" for issue in captured.value.issues)


def test_validation_binds_simultaneously_changed_rate_and_cost_to_trusted_identity() -> None:
    tampered = copy.deepcopy(_bundle())
    cost = tampered["cost"]
    assert isinstance(cost, dict)
    rate_card = cost["rate_card"]
    assert isinstance(rate_card, list)
    rate = rate_card[0]
    assert isinstance(rate, dict)
    rate["input_per_million"] = "0.000001"
    rate["output_per_million"] = "0.000001"
    cost["known_cost"] = "0.000000001500"
    cost["estimated_cost"] = "0.000000001500"
    cost["cost_per_1000_calls"] = "0.000000003000"

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(tampered)

    assert any(issue.keyword == "pricing-digest-parity" for issue in captured.value.issues)

    sources = cost["source_references"]
    assert isinstance(sources, list)
    cost["pricing_evidence_digest"] = canonical_pricing_evidence_digest(
        pricing_version=str(cost["pricing_version"]),
        currency=str(cost["currency"]),
        rate_card=rate_card,
        source_references=sources,
    )
    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(tampered)

    assert any(issue.keyword == "pricing-digest-binding" for issue in captured.value.issues)


def test_validation_binds_pricing_sources_into_the_digest() -> None:
    tampered = copy.deepcopy(_bundle())
    cost = tampered["cost"]
    assert isinstance(cost, dict)
    sources = cost["source_references"]
    rate_card = cost["rate_card"]
    assert isinstance(sources, list)
    assert isinstance(rate_card, list)
    sources[0] = "https://example.test/changed-pricing-source"

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(tampered)

    assert any(issue.keyword == "pricing-digest-parity" for issue in captured.value.issues)

    cost["pricing_evidence_digest"] = canonical_pricing_evidence_digest(
        pricing_version=str(cost["pricing_version"]),
        currency=str(cost["currency"]),
        rate_card=rate_card,
        source_references=sources,
    )
    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(tampered)

    assert any(issue.keyword == "pricing-digest-binding" for issue in captured.value.issues)


def test_unknown_provider_usage_forces_token_and_cost_incompleteness() -> None:
    report = _accepted_report()
    first = report.attempts[0]
    provider_attempt = first.provider_attempts[0].model_copy(
        update={"usage": TokenUsage(input_tokens=2)}
    )
    values = first.model_dump()
    values.update(
        {
            "provider_unknown_usage_attempt_count": 1,
            "provider_attempts": (provider_attempt,),
            "token_counts": {"generation-input": 2},
        }
    )
    attempts = (LoadAttempt.model_validate(values), *report.attempts[1:])
    report = _with_attempts(report, attempts)
    bundle = build_performance_evidence_bundle(
        report,
        identity=_identity(),
        references=_references(),
        cost=_cost(report),
        generated_at=_START,
    )

    tokens = bundle["tokens"]
    cost = bundle["cost"]
    decision = bundle["decision"]
    assert isinstance(tokens, dict)
    assert isinstance(cost, dict)
    assert isinstance(decision, dict)
    assert tokens["complete"] is False
    assert tokens["unknown_reasons"] == ["provider-usage-unknown"]
    assert cost["complete"] is False
    assert "provider-usage-unknown" in cost["unknown_reasons"]
    assert "cost-evidence-incomplete" in decision["invalid_reasons"]

    forged = copy.deepcopy(bundle)
    forged_cost = forged["cost"]
    assert isinstance(forged_cost, dict)
    forged_cost["complete"] = True
    forged_cost["estimated_cost"] = forged_cost["known_cost"]
    forged_cost["cost_per_1000_calls"] = Decimal(str(forged_cost["known_cost"])) * Decimal(2)
    forged_cost["unknown_reasons"] = []
    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(forged)
    assert any(issue.keyword == "cost-usage-completeness" for issue in captured.value.issues)


def test_unverifiable_http_attempt_forces_token_and_cost_incompleteness() -> None:
    report = _accepted_report()
    failed = _attempt(501, status=LoadAttemptStatus.TERMINAL_ERROR).model_copy(
        update={"provider_evidence_complete": False}
    )
    attempts = (*report.attempts, failed)
    report = _with_attempts(report, attempts)

    bundle = build_performance_evidence_bundle(
        report,
        identity=_identity(),
        references=_references(),
        cost=_cost(report),
        generated_at=_START,
    )

    tokens = bundle["tokens"]
    cost = bundle["cost"]
    assert isinstance(tokens, dict)
    assert isinstance(cost, dict)
    assert tokens["complete"] is False
    assert tokens["unknown_reasons"] == ["provider-attempt-evidence-missing"]
    assert cost["complete"] is False
    assert "provider-attempt-evidence-missing" in cost["unknown_reasons"]
    validate_performance_evidence_bundle(bundle)


def test_threshold_failure_with_incomplete_cost_evidence_builds_bundle() -> None:
    report = _accepted_report()
    failed_attempts = tuple(
        _attempt(index, status=LoadAttemptStatus.TERMINAL_ERROR).model_copy(
            update={"provider_evidence_complete": False}
        )
        for index in range(501, 510)
    )
    report = _with_attempts(report, (*report.attempts, *failed_attempts))

    bundle = build_performance_evidence_bundle(
        report,
        identity=_identity(),
        references=_references(),
        cost=_cost(report),
        generated_at=_START,
    )

    decision = bundle["decision"]
    assert isinstance(decision, dict)
    assert report.success_count == 500
    assert report.error_count == 9
    assert decision["invalid_reasons"] == ["cost-evidence-incomplete"]
    assert decision["failure_reasons"] == [
        "cost-evidence-incomplete",
        "error-rate-threshold-not-met",
    ]
    validate_performance_evidence_bundle(bundle)


def test_provider_free_success_keeps_complete_evidence_when_other_calls_are_priced() -> None:
    report = _accepted_report()
    values = report.attempts[0].model_dump()
    values.update(
        {
            "terminal_kind": "refusal",
            "provider_attempt_count": 0,
            "provider_failed_attempt_count": 0,
            "provider_unknown_usage_attempt_count": 0,
            "provider_attempts": (),
            "token_counts": {},
            "model_identities": {},
            "stage_timings_ms": {
                "validation": values["stage_timings_ms"]["validation"],
                "total": values["stage_timings_ms"]["total"],
            },
        }
    )
    provider_free = LoadAttempt.model_validate(values)
    attempts = (provider_free, *report.attempts[1:])
    report = _with_attempts(report, attempts)
    bundle = build_performance_evidence_bundle(
        report,
        identity=_identity(),
        references=_references(),
        cost=_cost(report),
        generated_at=_START,
    )

    tokens = bundle["tokens"]
    cost = bundle["cost"]
    assert isinstance(tokens, dict)
    assert isinstance(cost, dict)
    assert tokens["complete"] is True
    assert tokens["unknown_reasons"] == []
    assert cost["complete"] is True
    validate_performance_evidence_bundle(bundle)


def test_validation_requires_token_roles_to_match_exact_model_identity_evidence() -> None:
    tampered = copy.deepcopy(_bundle())
    attempts = tampered["attempts"]
    assert isinstance(attempts, dict)
    records = attempts["records"]
    assert isinstance(records, list)
    first = records[0]
    assert isinstance(first, dict)
    models = first["model_identities"]
    assert isinstance(models, dict)
    models.clear()

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(tampered)

    assert any(issue.keyword == "decision-parity" for issue in captured.value.issues)


def test_validation_rejects_attempt_and_cost_provider_counts_forged_to_zero() -> None:
    tampered = copy.deepcopy(_bundle())
    attempts = tampered["attempts"]
    cost = tampered["cost"]
    assert isinstance(attempts, dict)
    assert isinstance(cost, dict)
    records = attempts["records"]
    assert isinstance(records, list)
    for record in records:
        assert isinstance(record, dict)
        record["provider_attempt_count"] = 0
        record["provider_failed_attempt_count"] = 0
    cost["provider_attempt_count"] = 0

    with pytest.raises(EvidenceValidationError) as captured:
        validate_performance_evidence_bundle(tampered)

    assert any(issue.keyword == "attempt-contract" for issue in captured.value.issues)


def test_writer_redacts_and_keeps_bundle_immutable(tmp_path: Path) -> None:
    bundle = _bundle()
    identity = bundle["identity"]
    attempts = bundle["attempts"]
    assert isinstance(identity, dict)
    assert isinstance(attempts, dict)
    model_identities = identity["model_identities"]
    records = attempts["records"]
    assert isinstance(model_identities, dict)
    assert isinstance(records, list)
    model_identities["generation"] = "private@example.com"
    for record in records:
        assert isinstance(record, dict)
        observed = record["model_identities"]
        assert isinstance(observed, dict)
        observed["generation"] = "private@example.com"
    output = tmp_path / "performance-evidence.json"

    written = write_performance_evidence_bundle(bundle, output)
    loaded = load_performance_evidence_bundle(written)

    assert "private@example.com" not in output.read_text(encoding="utf-8")
    assert loaded["decision"] == bundle["decision"]
    with pytest.raises(EvidenceWriteError, match="already exists"):
        write_performance_evidence_bundle(bundle, output)
