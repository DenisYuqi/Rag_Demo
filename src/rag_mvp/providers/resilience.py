"""Deadline-aware provider attempts, retries, cancellation, and accounting."""

from __future__ import annotations

import asyncio
import math
import threading
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from rag_mvp.providers.errors import (
    ProviderError,
    ProviderOperationError,
    classify_provider_exception,
)
from rag_mvp.providers.models import (
    AttemptedResult,
    AttemptStatus,
    ModelAttempt,
    ProviderCallContext,
    ProviderErrorCategory,
    RouteMetadata,
    TokenUsage,
)
from rag_mvp.providers.protocols import AttemptRecorder, NullAttemptRecorder


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Explicit retries; total attempts are at most ``max_retries + 1``."""

    attempt_timeout_seconds: float
    max_retries: int = 0
    initial_backoff_seconds: float = 0.05
    max_backoff_seconds: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.attempt_timeout_seconds) or self.attempt_timeout_seconds <= 0:
            raise ValueError("attempt timeout must be positive and finite")
        if isinstance(self.max_retries, bool) or self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if not math.isfinite(self.initial_backoff_seconds) or self.initial_backoff_seconds < 0:
            raise ValueError("initial backoff must be finite and non-negative")
        if not math.isfinite(self.max_backoff_seconds) or self.max_backoff_seconds < 0:
            raise ValueError("maximum backoff must be finite and non-negative")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("maximum backoff cannot be below initial backoff")

    def backoff_seconds(self, retry_number: int) -> float:
        """Return deterministic exponential backoff for a one-based retry number."""

        if retry_number <= 0:
            raise ValueError("retry_number must be positive")
        return min(
            self.max_backoff_seconds,
            self.initial_backoff_seconds * (2.0 ** (retry_number - 1)),
        )


class InMemoryAttemptRecorder:
    """Thread-safe recorder useful for composition tests and local diagnostics."""

    def __init__(self) -> None:
        self._attempts: list[ModelAttempt] = []
        self._lock = threading.Lock()

    def record(self, attempt: ModelAttempt) -> None:
        with self._lock:
            self._attempts.append(attempt)

    @property
    def attempts(self) -> tuple[ModelAttempt, ...]:
        with self._lock:
            return tuple(self._attempts)


_REQUEST_ATTEMPT_RECORDER: ContextVar[InMemoryAttemptRecorder | None] = ContextVar(
    "rag_mvp_request_provider_attempt_recorder",
    default=None,
)


@contextmanager
def capture_provider_attempts() -> Iterator[InMemoryAttemptRecorder]:
    """Capture every resilient provider attempt in the current async context."""

    recorder = InMemoryAttemptRecorder()
    token = _REQUEST_ATTEMPT_RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _REQUEST_ATTEMPT_RECORDER.reset(token)


def current_provider_attempts() -> tuple[ModelAttempt, ...]:
    recorder = _REQUEST_ATTEMPT_RECORDER.get()
    return () if recorder is None else recorder.attempts


def _record_attempt(recorder: AttemptRecorder, attempt: ModelAttempt) -> None:
    recorder.record(attempt)
    request_recorder = _REQUEST_ATTEMPT_RECORDER.get()
    if request_recorder is not None and request_recorder is not recorder:
        request_recorder.record(attempt)


def _result_usage(value: object) -> TokenUsage:
    usage = getattr(value, "usage", None)
    return usage if isinstance(usage, TokenUsage) else TokenUsage()


async def execute_with_resilience[T](
    operation: Callable[[], Awaitable[T]],
    *,
    context: ProviderCallContext,
    route: RouteMetadata,
    policy: RetryPolicy,
    is_fallback: bool,
    recorder: AttemptRecorder | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> AttemptedResult[T]:
    """Execute one logical operation on one route within the shared deadline."""

    resolved_recorder = recorder or NullAttemptRecorder()
    attempts: list[ModelAttempt] = []
    attempt_number = 0

    while attempt_number <= policy.max_retries:
        remaining = context.deadline.remaining_seconds
        if remaining <= 0:
            raise ProviderOperationError(
                ProviderErrorCategory.DEADLINE_EXCEEDED,
                tuple(attempts),
                retryable=False,
                fallback_eligible=False,
            )
        attempt_number += 1
        attempt_timeout = min(policy.attempt_timeout_seconds, remaining)
        deadline_limited = remaining <= policy.attempt_timeout_seconds
        started_at = context.deadline.clock()
        timeout_scope = asyncio.timeout(attempt_timeout)
        try:
            async with timeout_scope:
                value = await operation()
        except asyncio.CancelledError:
            attempt = ModelAttempt(
                request_id=context.request_id,
                operation_id=context.operation_id,
                attempt_number=attempt_number,
                route_id=route.route_id,
                role=route.role,
                provider=route.identity.provider,
                model=route.identity.model,
                latency_ms=max(0.0, (context.deadline.clock() - started_at) * 1000),
                status=AttemptStatus.CANCELLED,
                is_fallback=is_fallback,
                error_category=ProviderErrorCategory.CANCELLED,
            )
            _record_attempt(resolved_recorder, attempt)
            raise
        except Exception as raw_error:
            error = classify_provider_exception(raw_error)
            if error.category is ProviderErrorCategory.TIMEOUT and (
                deadline_limited and timeout_scope.expired()
            ):
                error = ProviderError(
                    ProviderErrorCategory.DEADLINE_EXCEEDED,
                    retryable=False,
                    fallback_eligible=False,
                )
            attempt = ModelAttempt(
                request_id=context.request_id,
                operation_id=context.operation_id,
                attempt_number=attempt_number,
                route_id=route.route_id,
                role=route.role,
                provider=route.identity.provider,
                model=route.identity.model,
                latency_ms=max(0.0, (context.deadline.clock() - started_at) * 1000),
                status=AttemptStatus.FAILED,
                is_fallback=is_fallback,
                usage=error.usage,
                error_category=error.category,
            )
            attempts.append(attempt)
            _record_attempt(resolved_recorder, attempt)
            if not error.retryable or attempt_number > policy.max_retries:
                raise ProviderOperationError(
                    error.category,
                    tuple(attempts),
                    retryable=error.retryable,
                    fallback_eligible=error.fallback_eligible,
                ) from None

            delay = policy.backoff_seconds(attempt_number)
            remaining = context.deadline.remaining_seconds
            if remaining <= delay:
                raise ProviderOperationError(
                    ProviderErrorCategory.DEADLINE_EXCEEDED,
                    tuple(attempts),
                    retryable=False,
                    fallback_eligible=False,
                ) from None
            try:
                async with asyncio.timeout(remaining):
                    await sleep(delay)
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                raise ProviderOperationError(
                    ProviderErrorCategory.DEADLINE_EXCEEDED,
                    tuple(attempts),
                    retryable=False,
                    fallback_eligible=False,
                ) from None
            continue

        attempt = ModelAttempt(
            request_id=context.request_id,
            operation_id=context.operation_id,
            attempt_number=attempt_number,
            route_id=route.route_id,
            role=route.role,
            provider=route.identity.provider,
            model=route.identity.model,
            latency_ms=max(0.0, (context.deadline.clock() - started_at) * 1000),
            status=AttemptStatus.SUCCEEDED,
            is_fallback=is_fallback,
            usage=_result_usage(value),
        )
        attempts.append(attempt)
        _record_attempt(resolved_recorder, attempt)
        return AttemptedResult(value=value, attempts=tuple(attempts))

    raise AssertionError("retry loop exhausted without returning")
