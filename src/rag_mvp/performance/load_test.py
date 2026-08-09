"""Async HTTP load harness for the single-instance QA acceptance workload."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Literal, Self, cast
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from pydantic import Field, field_validator, model_validator

from rag_mvp.domain._base import DomainModel, Identifier
from rag_mvp.domain.evaluation import ModelAttemptStatus, ModelRole, ProviderAttemptEvidence
from rag_mvp.domain.qa import StreamEventKind, ValidatedStreamEvent
from rag_mvp.performance.load_report import (
    ACCEPTANCE_CONCURRENCY,
    CACHE_POLICY_HEADER,
    INSTANCE_ID_HEADER,
    MINIMUM_SUCCESSFUL_REQUESTS,
    LoadAcceptanceThresholds,
    LoadAttempt,
    LoadAttemptStatus,
    LoadReport,
    build_load_report,
    build_warmup_summary,
)

MAX_RESPONSE_BYTES = 2 * 1024 * 1024
DEFAULT_WARMUP_ATTEMPTS = 5
DEFAULT_REQUEST_TIMEOUT_SECONDS = 15.0
DEFAULT_RETRY_LIMIT = 1
_SAFE_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_SAFE_HEADER_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,254}$")
_RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class LoadScenario(DomainModel):
    """One request shape; only ``scenario_id`` is retained in run metadata."""

    scenario_id: Identifier
    owner_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$")]
    question: Annotated[str, Field(min_length=1, max_length=4_000)]
    mode: Literal["dense", "hybrid", "hybrid-rerank"] = "hybrid"
    requested_language: Literal["en", "zh-CN"] | None = None

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("load scenario question must not be blank")
        return value


class HttpLoadTestConfig(DomainModel):
    """Explicit workload configuration with cost-bounding attempt limits."""

    run_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$")]
    expected_configuration_id: Annotated[
        str,
        Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,254}$"),
    ]
    base_url: str
    scenarios: Annotated[tuple[LoadScenario, ...], Field(min_length=1)]
    qa_path: Annotated[str, Field(pattern=r"^/[A-Za-z0-9_./-]+$")] = "/api/v1/qa"
    readiness_path: Annotated[str, Field(pattern=r"^/[A-Za-z0-9_./-]+$")] = "/readyz"
    warmup_attempts: Annotated[int, Field(gt=0)] = DEFAULT_WARMUP_ATTEMPTS
    concurrency: Annotated[int, Field(gt=0, le=100)] = ACCEPTANCE_CONCURRENCY
    target_successes: Annotated[int, Field(gt=0)] = MINIMUM_SUCCESSFUL_REQUESTS
    max_attempts: Annotated[int, Field(gt=0)] | None = None
    exact_measured_attempts: Annotated[int, Field(gt=0)] | None = None
    retry_limit: Annotated[int, Field(ge=0, le=20)] = DEFAULT_RETRY_LIMIT
    request_timeout_seconds: Annotated[float, Field(gt=0, le=120)] = DEFAULT_REQUEST_TIMEOUT_SECONDS
    instance_count: Annotated[int, Field(gt=0)] = 1
    cache_policy: Literal["bypass"] = "bypass"

    @field_validator("base_url")
    @classmethod
    def require_absolute_http_url(cls, value: str) -> str:
        normalized = value.rstrip("/")
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base URL must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base URL must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base URL must not contain a query or fragment")
        return normalized

    @model_validator(mode="after")
    def validate_workload_bounds(self) -> Self:
        scenario_ids = tuple(scenario.scenario_id for scenario in self.scenarios)
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("scenario IDs must be unique")
        if self.target_successes < self.concurrency:
            raise ValueError("target successes must cover the configured concurrent users")
        if self.resolved_max_attempts < self.target_successes:
            raise ValueError("maximum attempts cannot be below the success target")
        if self.resolved_max_attempts < self.concurrency:
            raise ValueError("maximum attempts must allow the initial concurrent burst")
        if self.exact_measured_attempts is not None:
            if self.resolved_max_attempts != self.exact_measured_attempts:
                raise ValueError("fixed-sample runs require maximum attempts to equal the sample")
            if self.retry_limit != 0:
                raise ValueError("fixed-sample runs do not permit transport retries")
        return self

    @property
    def resolved_max_attempts(self) -> int:
        if self.max_attempts is not None:
            return self.max_attempts
        return self.target_successes + max(100, self.concurrency)

    @property
    def workload_digest(self) -> str:
        payload = json.dumps(
            [scenario.model_dump(mode="json") for scenario in self.scenarios],
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True, slots=True)
class ParsedQAResponse:
    status: LoadAttemptStatus
    terminal_kind: str | None
    request_id: str | None
    trace_id: str | None
    instance_identity: str | None
    safe_error_code: str | None
    retryable: bool
    provider_attempt_count: int
    provider_failed_attempt_count: int
    provider_unknown_usage_attempt_count: int
    provider_evidence_complete: bool
    provider_attempts: tuple[ProviderAttemptEvidence, ...]
    stage_timings_ms: dict[str, float]
    token_counts: dict[str, int]
    model_identities: dict[str, str]
    cache_status: dict[str, str]


def parse_qa_response(
    *,
    status_code: int,
    body: str | bytes,
    headers: Mapping[str, str] | None = None,
    expected_request_id: str | None = None,
    expected_trace_id: str | None = None,
    expected_cache_policy: str | None = None,
    expected_instance_identity: str | None = None,
) -> ParsedQAResponse:
    """Parse exactly one terminal NDJSON event without retaining its content."""

    if not 100 <= status_code <= 599:
        raise ValueError("HTTP status is outside the valid range")
    header_values = headers or {}
    response_request_id = _safe_header_value(header_values, "x-request-id")
    response_trace_id = _safe_header_value(header_values, "x-trace-id")
    response_cache_policy = _safe_header_value(
        header_values,
        CACHE_POLICY_HEADER.casefold(),
    )
    response_instance_identity = _safe_header_value(
        header_values,
        INSTANCE_ID_HEADER.casefold(),
    )
    if expected_request_id is not None and not _SAFE_HEADER_IDENTIFIER.fullmatch(
        expected_request_id
    ):
        raise ValueError("expected request ID is invalid")
    if expected_trace_id is not None and (
        re.fullmatch(r"[0-9a-fA-F]{32}", expected_trace_id) is None
        or int(expected_trace_id, 16) == 0
    ):
        raise ValueError("expected trace ID is invalid")
    retained_request_id = expected_request_id or response_request_id
    retained_trace_id = expected_trace_id or response_trace_id
    retained_instance_identity = expected_instance_identity or response_instance_identity

    def failed(
        status: LoadAttemptStatus,
        code: str,
        retryable: bool = False,
    ) -> ParsedQAResponse:
        return _failed_parse(
            status,
            code,
            retryable,
            request_id=retained_request_id,
            trace_id=retained_trace_id,
            instance_identity=retained_instance_identity,
        )

    if expected_request_id is not None and response_request_id not in {
        None,
        expected_request_id,
    }:
        return failed(LoadAttemptStatus.MALFORMED, "request-correlation-mismatch")
    if expected_trace_id is not None and response_trace_id not in {
        None,
        expected_trace_id,
    }:
        return failed(LoadAttemptStatus.MALFORMED, "trace-correlation-mismatch")
    if not 200 <= status_code < 300:
        return ParsedQAResponse(
            status=LoadAttemptStatus.HTTP_ERROR,
            terminal_kind=None,
            request_id=retained_request_id,
            trace_id=retained_trace_id,
            instance_identity=retained_instance_identity,
            safe_error_code=_http_error_code(status_code, body),
            retryable=status_code in _RETRYABLE_HTTP_STATUSES,
            provider_attempt_count=0,
            provider_failed_attempt_count=0,
            provider_unknown_usage_attempt_count=0,
            provider_evidence_complete=False,
            provider_attempts=(),
            stage_timings_ms={},
            token_counts={},
            model_identities={},
            cache_status={},
        )
    if expected_cache_policy is not None and response_cache_policy != expected_cache_policy:
        return failed(LoadAttemptStatus.MALFORMED, "cache-policy-correlation-mismatch")
    if (
        expected_instance_identity is not None
        and response_instance_identity != expected_instance_identity
    ):
        return failed(LoadAttemptStatus.MALFORMED, "instance-identity-correlation-mismatch")

    try:
        text = body.decode("utf-8", errors="strict") if isinstance(body, bytes) else body
    except UnicodeDecodeError:
        return failed(LoadAttemptStatus.MALFORMED, "response-not-utf8")
    if len(text.encode("utf-8")) > MAX_RESPONSE_BYTES:
        return failed(LoadAttemptStatus.MALFORMED, "response-too-large")
    lines = tuple(line for line in text.splitlines() if line.strip())
    if not lines:
        return failed(LoadAttemptStatus.INCOMPLETE, "terminal-event-missing")
    if len(lines) != 1:
        return failed(LoadAttemptStatus.MALFORMED, "terminal-event-count-invalid")
    try:
        raw = json.loads(lines[0], object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, ValueError, TypeError):
        return failed(LoadAttemptStatus.MALFORMED, "terminal-event-malformed")
    if not isinstance(raw, dict):
        return failed(LoadAttemptStatus.MALFORMED, "terminal-event-malformed")
    if raw.get("terminal") is not True:
        return failed(LoadAttemptStatus.INCOMPLETE, "terminal-event-incomplete")
    try:
        event = ValidatedStreamEvent.model_validate(raw)
    except (TypeError, ValueError):
        return failed(LoadAttemptStatus.MALFORMED, "terminal-event-invalid")

    if retained_request_id is not None and retained_request_id != event.request_id:
        return failed(LoadAttemptStatus.MALFORMED, "request-correlation-mismatch")
    diagnostics = event.diagnostics
    if expected_trace_id is not None and response_trace_id is None and diagnostics.trace_id is None:
        return failed(LoadAttemptStatus.MALFORMED, "trace-correlation-missing")
    if (
        response_trace_id is not None
        and diagnostics.trace_id is not None
        and response_trace_id != diagnostics.trace_id
    ):
        return failed(LoadAttemptStatus.MALFORMED, "trace-correlation-mismatch")
    if retained_trace_id is not None and diagnostics.trace_id not in {
        None,
        retained_trace_id,
    }:
        return failed(LoadAttemptStatus.MALFORMED, "trace-correlation-mismatch")
    trace_id = retained_trace_id or diagnostics.trace_id
    provider_attempt_count = _safe_metadata_count(
        diagnostics.metadata,
        "provider_attempt_count",
    )
    provider_failed_attempt_count = _safe_metadata_count(
        diagnostics.metadata,
        "provider_failed_attempt_count",
    )
    provider_unknown_usage_attempt_count = _safe_metadata_count(
        diagnostics.metadata,
        "provider_unknown_usage_attempt_count",
    )
    provider_attempts = tuple(diagnostics.provider_attempts)
    if (
        provider_attempt_count is None
        or provider_failed_attempt_count is None
        or provider_unknown_usage_attempt_count is None
        or provider_failed_attempt_count > provider_attempt_count
        or provider_unknown_usage_attempt_count > provider_attempt_count
        or len(provider_attempts) != provider_attempt_count
        or sum(attempt.status is not ModelAttemptStatus.SUCCEEDED for attempt in provider_attempts)
        != provider_failed_attempt_count
        or sum(_provider_usage_unknown(attempt) for attempt in provider_attempts)
        != provider_unknown_usage_attempt_count
        or _provider_token_counts(provider_attempts) != diagnostics.token_counts
        or (
            event.kind is StreamEventKind.ANSWER
            and not _has_complete_generation_attempt(provider_attempts)
        )
    ):
        return failed(
            LoadAttemptStatus.MALFORMED,
            "provider-attempt-diagnostics-invalid",
        )
    if event.kind in {StreamEventKind.ANSWER, StreamEventKind.REFUSAL}:
        return ParsedQAResponse(
            status=LoadAttemptStatus.SUCCEEDED,
            terminal_kind=event.kind.value,
            request_id=event.request_id,
            trace_id=trace_id,
            instance_identity=retained_instance_identity,
            safe_error_code=None,
            retryable=False,
            provider_attempt_count=provider_attempt_count,
            provider_failed_attempt_count=provider_failed_attempt_count,
            provider_unknown_usage_attempt_count=(provider_unknown_usage_attempt_count),
            provider_evidence_complete=True,
            provider_attempts=provider_attempts,
            stage_timings_ms=dict(diagnostics.stage_timings_ms),
            token_counts=dict(diagnostics.token_counts),
            model_identities=dict(diagnostics.model_identities),
            cache_status=_cache_status_with_policy(
                diagnostics.cache_status,
                response_cache_policy,
            ),
        )
    if event.kind is StreamEventKind.ERROR:
        return ParsedQAResponse(
            status=LoadAttemptStatus.TERMINAL_ERROR,
            terminal_kind=event.kind.value,
            request_id=event.request_id,
            trace_id=trace_id,
            instance_identity=retained_instance_identity,
            safe_error_code=(
                event.error_code.value if event.error_code is not None else "terminal-error"
            ),
            retryable=event.retryable is True,
            provider_attempt_count=provider_attempt_count,
            provider_failed_attempt_count=provider_failed_attempt_count,
            provider_unknown_usage_attempt_count=(provider_unknown_usage_attempt_count),
            provider_evidence_complete=True,
            provider_attempts=provider_attempts,
            stage_timings_ms=dict(diagnostics.stage_timings_ms),
            token_counts=dict(diagnostics.token_counts),
            model_identities=dict(diagnostics.model_identities),
            cache_status=_cache_status_with_policy(
                diagnostics.cache_status,
                response_cache_policy,
            ),
        )
    return failed(LoadAttemptStatus.MALFORMED, "terminal-kind-invalid")


class HttpLoadTestHarness:
    """Drive one warm instance with a fixed number of asynchronous users."""

    def __init__(
        self,
        config: HttpLoadTestConfig,
        *,
        client: httpx.AsyncClient | None = None,
        thresholds: LoadAcceptanceThresholds | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._thresholds = thresholds or LoadAcceptanceThresholds()
        self._ready_instance_identity: str | None = None

    async def run(self) -> LoadReport:
        owned_client = self._client is None
        client = self._client or httpx.AsyncClient(
            timeout=httpx.Timeout(self._config.request_timeout_seconds),
            limits=httpx.Limits(
                max_connections=self._config.concurrency,
                max_keepalive_connections=self._config.concurrency,
            ),
            follow_redirects=False,
        )
        try:
            warmup_started_at = datetime.now(UTC)
            warmup_clock = time.perf_counter()
            readiness_passed = await self._readiness_passed(client)
            warmup_records: list[LoadAttempt] = []
            if readiness_passed:
                for index in range(self._config.warmup_attempts):
                    scenario = self._config.scenarios[index % len(self._config.scenarios)]
                    warmup_records.append(
                        await self._perform_attempt(
                            client,
                            attempt_id=f"{self._config.run_id}-warmup-{index + 1:06d}",
                            logical_request_id=(
                                f"{self._config.run_id}-warmup-request-{index + 1:06d}"
                            ),
                            attempt_number=1,
                            retry_of_attempt_id=None,
                            scenario=scenario,
                        )
                    )
            warmup_completed_at = datetime.now(UTC)
            warmup = build_warmup_summary(
                warmup_records,
                readiness_passed=readiness_passed,
                configured_attempts=self._config.warmup_attempts,
                started_at=warmup_started_at,
                completed_at=warmup_completed_at,
                duration_ms=max(0.0, (time.perf_counter() - warmup_clock) * 1_000),
            )

            measured_started_at = datetime.now(UTC)
            measured_clock = time.perf_counter()
            records: tuple[LoadAttempt, ...] = ()
            observed_peak = 0
            if readiness_passed and warmup.error_count == 0:
                records, observed_peak = await self._run_measured(client)
            measured_completed_at = datetime.now(UTC)
            observed_instance_identities = {
                attempt.instance_identity
                for attempt in (*warmup_records, *records)
                if attempt.succeeded and attempt.instance_identity is not None
            }
            return build_load_report(
                run_id=self._config.run_id,
                started_at=measured_started_at,
                completed_at=measured_completed_at,
                duration_ms=max(0.0, (time.perf_counter() - measured_clock) * 1_000),
                instance_count=(
                    len(observed_instance_identities)
                    if observed_instance_identities
                    else self._config.instance_count
                ),
                configured_concurrency=self._config.concurrency,
                observed_peak_concurrency=observed_peak,
                cache_policy=self._config.cache_policy,
                workload_digest=self._config.workload_digest,
                workload_scenario_ids=tuple(
                    scenario.scenario_id for scenario in self._config.scenarios
                ),
                warmup=warmup,
                attempts=records,
                thresholds=self._thresholds,
            )
        finally:
            if owned_client:
                await client.aclose()

    async def _readiness_passed(self, client: httpx.AsyncClient) -> bool:
        try:
            response = await client.get(
                self._url(self._config.readiness_path),
                headers={"Cache-Control": "no-cache, no-store"},
                timeout=self._config.request_timeout_seconds,
            )
        except (httpx.TimeoutException, httpx.TransportError):
            return False
        if response.status_code != 200 or len(response.content) > 64 * 1024:
            return False
        try:
            raw = json.loads(
                response.content.decode("utf-8", errors="strict"),
                object_pairs_hook=_unique_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return False
        if not isinstance(raw, dict):
            return False
        instance_identity = raw.get("instance_identity")
        response_instance_identity = _safe_header_value(
            response.headers,
            INSTANCE_ID_HEADER,
        )
        passed = bool(
            raw.get("status") == "ready"
            and raw.get("configuration_id") == self._config.expected_configuration_id
            and isinstance(instance_identity, str)
            and _SAFE_HEADER_IDENTIFIER.fullmatch(instance_identity)
            and response_instance_identity == instance_identity
        )
        self._ready_instance_identity = instance_identity if passed else None
        return passed

    async def _run_measured(
        self,
        client: httpx.AsyncClient,
    ) -> tuple[tuple[LoadAttempt, ...], int]:
        lock = asyncio.Lock()
        initial_barrier = asyncio.Barrier(self._config.concurrency)
        sequenced_records: list[tuple[int, LoadAttempt]] = []
        reserved_attempts = 0
        successes = 0
        active = 0
        observed_peak = 0

        async def worker(worker_index: int) -> None:
            nonlocal reserved_attempts, successes, active, observed_peak
            logical_sequence = worker_index + 1
            attempt_number = 1
            retry_of_attempt_id: str | None = None
            first_attempt = True
            while True:
                async with lock:
                    if (
                        (
                            self._config.exact_measured_attempts is None
                            and successes >= self._config.target_successes
                        )
                        or reserved_attempts >= self._config.resolved_max_attempts
                    ):
                        return
                    reserved_attempts += 1
                    sequence = reserved_attempts
                    active += 1
                    observed_peak = max(observed_peak, active)
                if first_attempt:
                    await initial_barrier.wait()
                    first_attempt = False
                logical_request_id = f"{self._config.run_id}-request-{logical_sequence:06d}"
                scenario = self._config.scenarios[
                    (logical_sequence - 1) % len(self._config.scenarios)
                ]
                attempt = await self._perform_attempt(
                    client,
                    attempt_id=f"{self._config.run_id}-attempt-{sequence:06d}",
                    logical_request_id=logical_request_id,
                    attempt_number=attempt_number,
                    retry_of_attempt_id=retry_of_attempt_id,
                    scenario=scenario,
                )
                async with lock:
                    active -= 1
                    sequenced_records.append((sequence, attempt))
                    if attempt.succeeded:
                        successes += 1
                if (
                    not attempt.succeeded
                    and attempt.retryable
                    and attempt_number <= self._config.retry_limit
                ):
                    attempt_number += 1
                    retry_of_attempt_id = attempt.attempt_id
                else:
                    logical_sequence += self._config.concurrency
                    attempt_number = 1
                    retry_of_attempt_id = None

        await asyncio.gather(
            *(worker(worker_index) for worker_index in range(self._config.concurrency))
        )
        records = tuple(
            attempt for _, attempt in sorted(sequenced_records, key=lambda item: item[0])
        )
        return records, observed_peak

    async def _perform_attempt(
        self,
        client: httpx.AsyncClient,
        *,
        attempt_id: str,
        logical_request_id: str,
        attempt_number: int,
        retry_of_attempt_id: str | None,
        scenario: LoadScenario,
    ) -> LoadAttempt:
        started_at = datetime.now(UTC)
        started_clock = time.perf_counter()
        status_code: int | None = None
        request_id = f"load-{uuid4().hex}"
        trace_id = uuid4().hex
        try:
            response = await client.post(
                self._url(self._config.qa_path),
                headers={
                    CACHE_POLICY_HEADER: self._config.cache_policy,
                    "Cache-Control": "no-cache, no-store",
                    "Accept": "application/x-ndjson",
                    "X-Request-ID": request_id,
                    "X-Trace-ID": trace_id,
                },
                json=_request_payload(scenario),
                timeout=self._config.request_timeout_seconds,
            )
            status_code = response.status_code
            parsed = parse_qa_response(
                status_code=response.status_code,
                body=response.content,
                headers=response.headers,
                expected_request_id=request_id,
                expected_trace_id=trace_id,
                expected_cache_policy=self._config.cache_policy,
                expected_instance_identity=self._ready_instance_identity,
            )
        except (TimeoutError, httpx.TimeoutException):
            parsed = _failed_parse(
                LoadAttemptStatus.TIMEOUT,
                "http-attempt-timeout",
                True,
                request_id=request_id,
                trace_id=trace_id,
                instance_identity=self._ready_instance_identity,
            )
        except httpx.TransportError:
            parsed = _failed_parse(
                LoadAttemptStatus.TRANSPORT_ERROR,
                "http-transport-error",
                True,
                request_id=request_id,
                trace_id=trace_id,
                instance_identity=self._ready_instance_identity,
            )
        completed_at = datetime.now(UTC)
        return LoadAttempt(
            attempt_id=attempt_id,
            logical_request_id=logical_request_id,
            scenario_id=scenario.scenario_id,
            attempt_number=attempt_number,
            retry_of_attempt_id=retry_of_attempt_id,
            status=parsed.status,
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=max(0.0, (time.perf_counter() - started_clock) * 1_000),
            http_status_code=status_code,
            request_id=parsed.request_id,
            trace_id=parsed.trace_id,
            instance_identity=parsed.instance_identity,
            terminal_kind=parsed.terminal_kind,
            safe_error_code=parsed.safe_error_code,
            retryable=parsed.retryable,
            provider_attempt_count=parsed.provider_attempt_count,
            provider_failed_attempt_count=parsed.provider_failed_attempt_count,
            provider_unknown_usage_attempt_count=(parsed.provider_unknown_usage_attempt_count),
            provider_evidence_complete=parsed.provider_evidence_complete,
            provider_attempts=parsed.provider_attempts,
            stage_timings_ms=parsed.stage_timings_ms,
            token_counts=parsed.token_counts,
            model_identities=parsed.model_identities,
            cache_status={**parsed.cache_status, "request-policy": self._config.cache_policy},
        )

    def _url(self, path: str) -> str:
        return f"{self._config.base_url}{path}"


def _request_payload(scenario: LoadScenario) -> dict[str, str]:
    payload = {
        "owner_id": scenario.owner_id,
        "question": scenario.question,
        "mode": scenario.mode,
    }
    if scenario.requested_language is not None:
        payload["requested_language"] = scenario.requested_language
    return payload


def _failed_parse(
    status: LoadAttemptStatus,
    safe_error_code: str,
    retryable: bool = False,
    *,
    request_id: str | None = None,
    trace_id: str | None = None,
    instance_identity: str | None = None,
) -> ParsedQAResponse:
    return ParsedQAResponse(
        status=status,
        terminal_kind=None,
        request_id=request_id,
        trace_id=trace_id,
        instance_identity=instance_identity,
        safe_error_code=safe_error_code,
        retryable=retryable,
        provider_attempt_count=0,
        provider_failed_attempt_count=0,
        provider_unknown_usage_attempt_count=0,
        provider_evidence_complete=False,
        provider_attempts=(),
        stage_timings_ms={},
        token_counts={},
        model_identities={},
        cache_status={},
    )


def _safe_header_value(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value is None:
        value = next(
            (raw for key, raw in headers.items() if key.casefold() == name.casefold()),
            None,
        )
    if value is None:
        return None
    normalized = value.strip()
    if not _SAFE_HEADER_IDENTIFIER.fullmatch(normalized):
        return None
    return normalized


def _safe_metadata_count(metadata: Mapping[str, object], name: str) -> int | None:
    value = metadata.get(name)
    if type(value) is not int or value < 0:
        return None
    return value


def _provider_usage_unknown(attempt: ProviderAttemptEvidence) -> bool:
    if attempt.usage.input_tokens is None:
        return True
    return attempt.role is not ModelRole.EMBEDDING and attempt.usage.output_tokens is None


def _provider_token_counts(
    attempts: Sequence[ProviderAttemptEvidence],
) -> dict[str, int]:
    totals: dict[str, int] = {}
    for attempt in attempts:
        role = attempt.role.value
        for direction, count in (
            ("input", attempt.usage.input_tokens),
            ("output", attempt.usage.output_tokens),
        ):
            if count is not None:
                key = f"{role}-{direction}"
                totals[key] = totals.get(key, 0) + count
    return dict(sorted(totals.items()))


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


def _cache_status_with_policy(
    values: Mapping[str, str],
    response_cache_policy: str | None,
) -> dict[str, str]:
    resolved = dict(values)
    if response_cache_policy is not None:
        resolved["request-policy"] = response_cache_policy
    return resolved


def _http_error_code(status_code: int, body: str | bytes) -> str:
    try:
        text = body.decode("utf-8", errors="strict") if isinstance(body, bytes) else body
        raw = cast(object, json.loads(text, object_pairs_hook=_unique_object))
        if isinstance(raw, dict):
            error = raw.get("error")
            if isinstance(error, dict):
                code = error.get("code")
                if isinstance(code, str) and _SAFE_ERROR_CODE.fullmatch(code):
                    return code
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return f"http-status-{status_code}"


def _unique_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


__all__ = [
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "DEFAULT_RETRY_LIMIT",
    "DEFAULT_WARMUP_ATTEMPTS",
    "MAX_RESPONSE_BYTES",
    "HttpLoadTestConfig",
    "HttpLoadTestHarness",
    "LoadScenario",
    "ParsedQAResponse",
    "parse_qa_response",
]
