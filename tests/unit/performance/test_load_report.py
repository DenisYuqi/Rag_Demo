from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from rag_mvp.domain.evaluation import (
    ModelAttemptStatus,
    ModelRole,
    ProviderAttemptEvidence,
    TokenUsage,
)
from rag_mvp.performance.load_report import (
    CACHE_POLICY_HEADER,
    LoadAcceptanceThresholds,
    LoadAttempt,
    LoadAttemptStatus,
    WarmupSummary,
    build_load_report,
    build_warmup_summary,
    nearest_rank_percentile,
)
from rag_mvp.performance.load_test import (
    HttpLoadTestConfig,
    HttpLoadTestHarness,
    LoadScenario,
    parse_qa_response,
)


def _event(
    request_id: str,
    *,
    kind: str = "refusal",
    retryable: bool = False,
) -> dict[str, object]:
    event: dict[str, object] = {
        "request_id": request_id,
        "session_id": "session-1",
        "sequence": 0,
        "kind": kind,
        "response_language": "en",
        "content": "Safe response.",
        "diagnostics": {
            "trace_id": f"trace-{request_id}",
            "stage_timings_ms": {
                "validation": 0.5,
                "retrieval": 2.5,
                "evidence_assessment": 1.5,
                "generation": 4.5,
                "finalization": 0.5,
                "total": 9.5,
            },
            "cache_status": {"retrieval": "bypass"},
            "model_identities": {"generation": "model-v1"},
            "token_counts": {"generation-input": 3, "generation-output": 2},
            "provider_attempts": [
                ProviderAttemptEvidence(
                    operation_id="qa-generation",
                    attempt_number=1,
                    route_id="primary",
                    role=ModelRole.GENERATION,
                    provider="test",
                    model="model-v1",
                    status=ModelAttemptStatus.SUCCEEDED,
                    usage=TokenUsage(input_tokens=1, output_tokens=1),
                ).model_dump(mode="json"),
                ProviderAttemptEvidence(
                    operation_id="qa-generation",
                    attempt_number=2,
                    route_id="fallback",
                    role=ModelRole.GENERATION,
                    provider="test-fallback",
                    model="model-v1",
                    status=ModelAttemptStatus.SUCCEEDED,
                    fallback=True,
                    usage=TokenUsage(input_tokens=2, output_tokens=1),
                ).model_dump(mode="json"),
            ],
            "metadata": {
                "provider_attempt_count": 2,
                "provider_failed_attempt_count": 0,
                "provider_unknown_usage_attempt_count": 0,
            },
        },
        "terminal": True,
    }
    if kind == "refusal":
        event["reason"] = "insufficient-evidence"
    elif kind == "answer":
        event["claims"] = [
            {
                "text": "Supported answer.",
                "citation_chunk_ids": ["chunk-1"],
            }
        ]
        event["citations"] = [
            {
                "source_title": "Handbook",
                "document_version": 1,
                "chunk_id": "chunk-1",
                "locator": {
                    "pages": [1],
                    "section_path": [],
                },
            }
        ]
    elif kind == "error":
        event["error_code"] = "capacity"
        event["retryable"] = retryable
    return event


def _attempt(
    attempt_id: str,
    latency_ms: float,
    *,
    logical_request_id: str | None = None,
    attempt_number: int = 1,
    retry_of_attempt_id: str | None = None,
    failed: bool = False,
    cache_status: str = "bypass",
) -> LoadAttempt:
    started = datetime(2026, 8, 7, tzinfo=UTC) + timedelta(milliseconds=latency_ms)
    provider_attempts = (
        ProviderAttemptEvidence(
            operation_id="qa-generation",
            attempt_number=1,
            route_id="primary",
            role=ModelRole.GENERATION,
            provider="test",
            model="model-v1",
            status=(ModelAttemptStatus.FAILED if failed else ModelAttemptStatus.SUCCEEDED),
            usage=(TokenUsage() if failed else TokenUsage(input_tokens=1, output_tokens=1)),
        ),
        ProviderAttemptEvidence(
            operation_id="qa-retrieval",
            attempt_number=1,
            route_id="embedding-primary",
            role=ModelRole.EMBEDDING,
            provider="test",
            model="embedding-v1",
            status=ModelAttemptStatus.SUCCEEDED,
            usage=TokenUsage(input_tokens=0 if failed else 1),
        ),
    )
    return LoadAttempt(
        attempt_id=attempt_id,
        logical_request_id=logical_request_id or f"logical-{attempt_id}",
        scenario_id="policy",
        attempt_number=attempt_number,
        retry_of_attempt_id=retry_of_attempt_id,
        status=(LoadAttemptStatus.TERMINAL_ERROR if failed else LoadAttemptStatus.SUCCEEDED),
        started_at=started,
        completed_at=started + timedelta(milliseconds=latency_ms),
        latency_ms=latency_ms,
        http_status_code=200,
        request_id=f"request-{attempt_id}",
        trace_id=f"trace-{attempt_id}",
        instance_identity="instance-test-1",
        terminal_kind="error" if failed else "refusal",
        safe_error_code="capacity" if failed else None,
        retryable=failed,
        provider_attempt_count=2,
        provider_failed_attempt_count=1 if failed else 0,
        provider_unknown_usage_attempt_count=1 if failed else 0,
        provider_attempts=provider_attempts,
        stage_timings_ms={
            "validation": 0.5,
            "retrieval": latency_ms / 2,
            "generation": latency_ms / 3,
            "total": latency_ms,
        },
        token_counts=(
            {"embedding-input": 0}
            if failed
            else {
                "embedding-input": 1,
                "generation-input": 1,
                "generation-output": 1,
            }
        ),
        model_identities={"generation": "model-v1"},
        cache_status={"request-policy": "bypass", "retrieval": cache_status},
    )


def _warmup() -> WarmupSummary:
    attempt = _attempt("warmup-1", 5)
    return build_warmup_summary(
        (attempt,),
        readiness_passed=True,
        configured_attempts=1,
        started_at=attempt.started_at,
        completed_at=attempt.completed_at,
        duration_ms=5,
    )


def test_nearest_rank_percentile_does_not_interpolate() -> None:
    samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    assert nearest_rank_percentile(samples, 50) == 5
    assert nearest_rank_percentile(samples, 90) == 9
    assert nearest_rank_percentile([1, 2, 3, 4], 90) == 4
    with pytest.raises(ValueError, match="at least one"):
        nearest_rank_percentile([], 90)


def test_load_config_rejects_duplicate_scenario_ids() -> None:
    scenario = LoadScenario(
        scenario_id="policy",
        owner_id="load-owner",
        question="What is the policy?",
    )

    with pytest.raises(ValueError, match="scenario IDs must be unique"):
        HttpLoadTestConfig(
            run_id="duplicate-scenarios",
            expected_configuration_id="config-v1",
            base_url="http://testserver",
            scenarios=(scenario, scenario),
            concurrency=5,
            target_successes=5,
        )


def test_load_attempt_requires_provider_attempt_for_each_token_role() -> None:
    values = _attempt("attempt-provider-parity", 5).model_dump()
    values["token_counts"] = {
        "embedding-input": 3,
        "generation-input": 2,
        "generation-output": 1,
    }

    with pytest.raises(ValueError, match="token counts disagree"):
        LoadAttempt.model_validate(values)


def test_load_attempt_rejects_duplicate_provider_ledger_records() -> None:
    values = _attempt("attempt-provider-duplicate", 5).model_dump()
    provider_attempts = values["provider_attempts"]
    assert isinstance(provider_attempts, tuple)
    duplicate = provider_attempts[0]
    values.update(
        {
            "provider_attempt_count": 3,
            "provider_attempts": (*provider_attempts, duplicate),
            "token_counts": {
                "embedding-input": 2,
                "generation-input": 1,
                "generation-output": 1,
            },
        }
    )

    with pytest.raises(ValueError, match="duplicate records"):
        LoadAttempt.model_validate(values)


def test_successful_answer_requires_complete_successful_generation_attempt() -> None:
    values = _attempt("attempt-answer-without-generation", 5).model_dump()
    provider_attempts = values["provider_attempts"]
    assert isinstance(provider_attempts, tuple)
    embedding = provider_attempts[1]
    values.update(
        {
            "terminal_kind": "answer",
            "provider_attempt_count": 1,
            "provider_attempts": (embedding,),
            "token_counts": {"embedding-input": 1},
        }
    )

    with pytest.raises(ValueError, match="successful generation attempt"):
        LoadAttempt.model_validate(values)


def test_parser_accepts_one_complete_terminal_event_and_discards_content() -> None:
    raw = _event("request-1")

    parsed = parse_qa_response(
        status_code=200,
        body=json.dumps(raw),
        headers={
            "X-Request-ID": "request-1",
            "X-Trace-ID": "trace-request-1",
        },
    )

    assert parsed.status is LoadAttemptStatus.SUCCEEDED
    assert parsed.terminal_kind == "refusal"
    assert parsed.trace_id == "trace-request-1"
    assert parsed.provider_attempt_count == 2
    assert parsed.provider_failed_attempt_count == 0
    assert parsed.provider_evidence_complete is True
    assert parsed.stage_timings_ms == {
        "validation": 0.5,
        "retrieval": 2.5,
        "evidence_assessment": 1.5,
        "generation": 4.5,
        "finalization": 0.5,
        "total": 9.5,
    }
    assert parsed.token_counts["generation-input"] == 3
    assert not hasattr(parsed, "content")


def test_parser_counts_terminal_error_incomplete_and_http_failure() -> None:
    terminal_error = parse_qa_response(
        status_code=200,
        body=json.dumps(_event("request-error", kind="error", retryable=True)),
    )
    incomplete = parse_qa_response(status_code=200, body="")
    http_error = parse_qa_response(
        status_code=503,
        body='{"error":{"code":"qa_unavailable"}}',
        headers={"X-Request-ID": "request-http-error", "X-Trace-ID": "trace-http-error"},
    )

    assert terminal_error.status is LoadAttemptStatus.TERMINAL_ERROR
    assert terminal_error.retryable is True
    assert terminal_error.provider_attempt_count == 2
    assert incomplete.status is LoadAttemptStatus.INCOMPLETE
    assert incomplete.provider_evidence_complete is False
    assert http_error.status is LoadAttemptStatus.HTTP_ERROR
    assert http_error.safe_error_code == "qa_unavailable"
    assert http_error.retryable is True
    assert http_error.request_id == "request-http-error"
    assert http_error.trace_id == "trace-http-error"
    assert http_error.provider_evidence_complete is False


def test_parser_uses_trace_header_and_rejects_diagnostics_mismatch() -> None:
    header_only = _event("request-header-only")
    diagnostics = header_only["diagnostics"]
    assert isinstance(diagnostics, dict)
    diagnostics["trace_id"] = None

    parsed = parse_qa_response(
        status_code=200,
        body=json.dumps(header_only),
        headers={"X-Trace-ID": "trace-from-header"},
    )
    mismatched = parse_qa_response(
        status_code=200,
        body=json.dumps(_event("request-mismatch")),
        headers={"X-Trace-ID": "trace-from-header"},
    )
    unsafe_http_header = parse_qa_response(
        status_code=503,
        body="{}",
        headers={"X-Trace-ID": "not a safe identifier"},
    )

    assert parsed.status is LoadAttemptStatus.SUCCEEDED
    assert parsed.trace_id == "trace-from-header"
    assert mismatched.status is LoadAttemptStatus.MALFORMED
    assert mismatched.safe_error_code == "trace-correlation-mismatch"
    assert unsafe_http_header.trace_id is None


def test_malformed_and_http_failures_retain_preallocated_correlation_ids() -> None:
    trace_id = "a" * 32
    malformed = parse_qa_response(
        status_code=200,
        body="not-json",
        expected_request_id="load-request-1",
        expected_trace_id=trace_id,
    )
    http_error = parse_qa_response(
        status_code=503,
        body="{}",
        expected_request_id="load-request-2",
        expected_trace_id=trace_id,
    )

    assert (malformed.request_id, malformed.trace_id) == ("load-request-1", trace_id)
    assert (http_error.request_id, http_error.trace_id) == ("load-request-2", trace_id)


def test_parser_rejects_missing_or_inconsistent_provider_attempt_diagnostics() -> None:
    missing = _event("request-missing-provider-count")
    diagnostics = missing["diagnostics"]
    assert isinstance(diagnostics, dict)
    metadata = diagnostics["metadata"]
    assert isinstance(metadata, dict)
    metadata.pop("provider_attempt_count")
    inconsistent = _event("request-inconsistent-provider-count")
    diagnostics = inconsistent["diagnostics"]
    assert isinstance(diagnostics, dict)
    metadata = diagnostics["metadata"]
    assert isinstance(metadata, dict)
    metadata["provider_failed_attempt_count"] = 3
    unknown = _event("request-inconsistent-provider-usage")
    diagnostics = unknown["diagnostics"]
    assert isinstance(diagnostics, dict)
    metadata = diagnostics["metadata"]
    assert isinstance(metadata, dict)
    metadata["provider_unknown_usage_attempt_count"] = 3

    parsed_missing = parse_qa_response(status_code=200, body=json.dumps(missing))
    parsed_inconsistent = parse_qa_response(status_code=200, body=json.dumps(inconsistent))
    parsed_unknown = parse_qa_response(status_code=200, body=json.dumps(unknown))

    assert parsed_missing.status is LoadAttemptStatus.MALFORMED
    assert parsed_missing.safe_error_code == "provider-attempt-diagnostics-invalid"
    assert parsed_inconsistent.status is LoadAttemptStatus.MALFORMED
    assert parsed_unknown.status is LoadAttemptStatus.MALFORMED


def test_parser_rejects_answer_without_complete_generation_attempt() -> None:
    raw = _event("request-answer-without-generation", kind="answer")
    diagnostics = raw["diagnostics"]
    assert isinstance(diagnostics, dict)
    diagnostics["provider_attempts"] = []
    diagnostics["token_counts"] = {}
    diagnostics["model_identities"] = {}
    metadata = diagnostics["metadata"]
    assert isinstance(metadata, dict)
    metadata.update(
        {
            "provider_attempt_count": 0,
            "provider_failed_attempt_count": 0,
            "provider_unknown_usage_attempt_count": 0,
        }
    )

    parsed = parse_qa_response(status_code=200, body=json.dumps(raw))

    assert parsed.status is LoadAttemptStatus.MALFORMED
    assert parsed.safe_error_code == "provider-attempt-diagnostics-invalid"


def test_parser_requires_server_trace_echo_for_correlated_success() -> None:
    trace_id = "b" * 32
    raw = _event("request-trace-not-echoed")
    diagnostics = raw["diagnostics"]
    assert isinstance(diagnostics, dict)
    diagnostics["trace_id"] = None

    parsed = parse_qa_response(
        status_code=200,
        body=json.dumps(raw),
        headers={"X-Request-ID": "request-trace-not-echoed"},
        expected_request_id="request-trace-not-echoed",
        expected_trace_id=trace_id,
    )

    assert parsed.status is LoadAttemptStatus.MALFORMED
    assert parsed.safe_error_code == "trace-correlation-missing"


def test_parser_requires_ready_instance_echo_for_success() -> None:
    raw = _event("request-instance-not-echoed")

    parsed = parse_qa_response(
        status_code=200,
        body=json.dumps(raw),
        expected_instance_identity="instance-test-1",
    )

    assert parsed.status is LoadAttemptStatus.MALFORMED
    assert parsed.safe_error_code == "instance-identity-correlation-mismatch"


@pytest.mark.asyncio
async def test_http_harness_rejects_ready_configuration_mismatch() -> None:
    qa_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal qa_calls
        if request.url.path == "/readyz":
            return httpx.Response(
                200,
                json={"status": "ready", "configuration_id": "different-config"},
            )
        qa_calls += 1
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await HttpLoadTestHarness(
            HttpLoadTestConfig(
                run_id="load-config-mismatch",
                expected_configuration_id="expected-config",
                base_url="http://testserver",
                scenarios=(
                    LoadScenario(
                        scenario_id="policy",
                        owner_id="load-owner",
                        question="What is the policy?",
                    ),
                ),
                warmup_attempts=1,
                concurrency=5,
                target_successes=5,
            ),
            client=client,
            thresholds=LoadAcceptanceThresholds(minimum_successes=5),
        ).run()

    assert report.warmup.readiness_passed is False
    assert report.warmup.attempts == ()
    assert report.attempts == ()
    assert qa_calls == 0


def test_report_keeps_failed_predecessor_in_retry_denominator() -> None:
    first = _attempt("attempt-1", 11, logical_request_id="logical-1", failed=True)
    retry = _attempt(
        "attempt-2",
        9,
        logical_request_id="logical-1",
        attempt_number=2,
        retry_of_attempt_id="attempt-1",
    )
    second = _attempt("attempt-3", 10, logical_request_id="logical-2")
    started = first.started_at
    completed = second.completed_at
    report = build_load_report(
        run_id="load-run-1",
        started_at=started,
        completed_at=completed,
        duration_ms=30,
        instance_count=1,
        configured_concurrency=5,
        observed_peak_concurrency=5,
        cache_policy="bypass",
        workload_scenario_ids=("policy",),
        warmup=_warmup(),
        attempts=(first, retry, second),
        thresholds=LoadAcceptanceThresholds(
            minimum_successes=2,
            maximum_error_rate_exclusive=0.5,
        ),
    )

    assert report.attempt_count == 3
    assert report.success_count == 2
    assert report.error_count == 1
    assert report.retry_attempt_count == 1
    assert report.error_rate == pytest.approx(1 / 3)
    assert report.complete_latency_ms is not None
    assert report.complete_latency_ms.p90 == 10
    assert report.stage_latency_ms["retrieval"].count == 2
    assert report.representative_trace_references[:3] == (
        "trace-attempt-3",
        "trace-attempt-1",
        "trace-attempt-2",
    )
    assert report.passed is True


@pytest.mark.parametrize(
    ("field", "missing_key", "reason"),
    [
        ("cache_status", "request-policy", "cache-policy-evidence-missing"),
        ("stage_timings_ms", "total", "stage-evidence-missing"),
    ],
)
def test_report_rejects_incomplete_success_execution_evidence(
    field: str,
    missing_key: str,
    reason: str,
) -> None:
    attempt = _attempt("attempt-incomplete-execution", 10)
    values = attempt.model_dump()
    evidence = values[field]
    assert isinstance(evidence, dict)
    evidence.pop(missing_key)
    incomplete = LoadAttempt.model_validate(values)

    report = build_load_report(
        run_id="load-incomplete-execution-evidence",
        started_at=incomplete.started_at,
        completed_at=incomplete.completed_at,
        duration_ms=incomplete.latency_ms,
        instance_count=1,
        configured_concurrency=5,
        observed_peak_concurrency=5,
        cache_policy="bypass",
        workload_scenario_ids=("policy",),
        warmup=_warmup(),
        attempts=(incomplete,),
        thresholds=LoadAcceptanceThresholds(minimum_successes=1),
    )

    assert reason in report.invalid_reasons
    assert report.valid is False
    assert report.passed is False


def test_report_rejects_missing_success_instance_identity() -> None:
    attempt = _attempt("attempt-instance-missing", 10).model_copy(
        update={"instance_identity": None}
    )

    report = build_load_report(
        run_id="load-instance-missing",
        started_at=attempt.started_at,
        completed_at=attempt.completed_at,
        duration_ms=attempt.latency_ms,
        instance_count=1,
        configured_concurrency=5,
        observed_peak_concurrency=5,
        cache_policy="bypass",
        workload_scenario_ids=("policy",),
        warmup=_warmup(),
        attempts=(attempt,),
        thresholds=LoadAcceptanceThresholds(minimum_successes=1),
    )

    assert "instance-identity-evidence-missing" in report.invalid_reasons
    assert report.passed is False


def test_report_requires_success_coverage_for_every_pinned_scenario() -> None:
    attempt = _attempt("attempt-only-policy", 10)

    report = build_load_report(
        run_id="load-scenario-coverage",
        started_at=attempt.started_at,
        completed_at=attempt.completed_at,
        duration_ms=attempt.latency_ms,
        instance_count=1,
        configured_concurrency=5,
        observed_peak_concurrency=5,
        cache_policy="bypass",
        workload_scenario_ids=("policy", "policy-exception"),
        warmup=_warmup(),
        attempts=(attempt,),
        thresholds=LoadAcceptanceThresholds(minimum_successes=1),
    )

    assert "scenario-success-coverage-missing" in report.invalid_reasons
    assert report.passed is False


@pytest.mark.asyncio
async def test_http_harness_warms_then_starts_five_cache_bypassed_users() -> None:
    active = 0
    peak = 0
    qa_calls = 0
    observed_cache_headers: list[str | None] = []
    lock = asyncio.Lock()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak, qa_calls
        if request.url.path == "/readyz":
            return httpx.Response(
                200,
                json={
                    "status": "ready",
                    "configuration_id": "config-test",
                    "instance_identity": "instance-test-1",
                },
                headers={"X-RAG-Instance-ID": "instance-test-1"},
            )
        observed_cache_headers.append(request.headers.get(CACHE_POLICY_HEADER))
        async with lock:
            qa_calls += 1
            call_number = qa_calls
            active += 1
            peak = max(peak, active)
        await asyncio.sleep(0.002)
        async with lock:
            active -= 1
        request_id = request.headers["x-request-id"]
        trace_id = request.headers["x-trace-id"]
        # Call one is the excluded warm-up; the first measured call is retryable.
        event = (
            _event(request_id, kind="error", retryable=True)
            if call_number == 2
            else _event(request_id)
        )
        event_diagnostics = event["diagnostics"]
        assert isinstance(event_diagnostics, dict)
        event_diagnostics["trace_id"] = trace_id
        return httpx.Response(
            200,
            content=json.dumps(event).encode(),
            headers={
                "content-type": "application/x-ndjson",
                "x-request-id": request_id,
                "x-trace-id": trace_id,
                CACHE_POLICY_HEADER: "bypass",
                "X-RAG-Instance-ID": "instance-test-1",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        harness = HttpLoadTestHarness(
            HttpLoadTestConfig(
                run_id="load-harness-test",
                expected_configuration_id="config-test",
                base_url="http://testserver",
                scenarios=(
                    LoadScenario(
                        scenario_id="policy",
                        owner_id="load-owner",
                        question="What is the policy?",
                    ),
                ),
                warmup_attempts=1,
                concurrency=5,
                target_successes=5,
                max_attempts=10,
                retry_limit=1,
            ),
            client=client,
            thresholds=LoadAcceptanceThresholds(
                minimum_successes=5,
                maximum_p90_latency_ms=100,
                maximum_error_rate_exclusive=0.5,
            ),
        )
        report = await harness.run()

    assert report.warmup.success_count == 1
    assert report.observed_peak_concurrency == 5
    assert peak == 5
    assert report.success_count >= 5
    assert report.error_count == 1
    assert report.retry_attempt_count == 1
    assert all(value == "bypass" for value in observed_cache_headers)
    assert report.passed is True


@pytest.mark.asyncio
async def test_transport_failure_retains_preallocated_correlation_ids() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/readyz":
            return httpx.Response(
                200,
                json={
                    "status": "ready",
                    "configuration_id": "config-v1",
                    "instance_identity": "instance-test-1",
                },
                headers={"X-RAG-Instance-ID": "instance-test-1"},
            )
        raise httpx.ConnectError("unavailable", request=request)

    config = HttpLoadTestConfig(
        run_id="load-transport-correlation",
        expected_configuration_id="config-v1",
        base_url="http://testserver",
        scenarios=(
            LoadScenario(
                scenario_id="scenario-1",
                owner_id="owner-1",
                question="What is the policy?",
            ),
        ),
        warmup_attempts=1,
        target_successes=5,
        max_attempts=5,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await HttpLoadTestHarness(config, client=client).run()

    attempt = report.warmup.attempts[0]
    assert attempt.status is LoadAttemptStatus.TRANSPORT_ERROR
    assert attempt.request_id is not None
    assert attempt.request_id.startswith("load-")
    assert attempt.trace_id is not None
    assert len(attempt.trace_id) == 32
