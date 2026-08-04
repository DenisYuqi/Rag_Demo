from __future__ import annotations

import asyncio

import pytest

from rag_mvp.providers.errors import ProviderError, ProviderOperationError
from rag_mvp.providers.models import (
    AttemptStatus,
    Deadline,
    GenerationResult,
    ModelIdentity,
    ProviderCallContext,
    ProviderErrorCategory,
    ProviderRole,
    RouteMetadata,
    TokenUsage,
)
from rag_mvp.providers.resilience import (
    InMemoryAttemptRecorder,
    RetryPolicy,
    execute_with_resilience,
)


IDENTITY = ModelIdentity("provider", "model", "v1")
ROUTE = RouteMetadata("primary", ProviderRole.GENERATION, IDENTITY)


def context(seconds: float = 2) -> ProviderCallContext:
    return ProviderCallContext("request", "operation", Deadline.after(seconds))


async def test_transient_failures_retry_with_all_attempts_accounted() -> None:
    calls = 0
    recorder = InMemoryAttemptRecorder()

    async def operation() -> GenerationResult:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ProviderError(ProviderErrorCategory.NETWORK)
        return GenerationResult(
            "ok", IDENTITY, finish_reason="stop", usage=TokenUsage(7, 2)  # type: ignore[arg-type]
        )

    result = await execute_with_resilience(
        operation,
        context=context(),
        route=ROUTE,
        policy=RetryPolicy(1, max_retries=2, initial_backoff_seconds=0),
        is_fallback=False,
        recorder=recorder,
    )

    assert calls == 3
    assert [attempt.status for attempt in result.attempts] == [
        AttemptStatus.FAILED,
        AttemptStatus.FAILED,
        AttemptStatus.SUCCEEDED,
    ]
    assert result.attempts[-1].usage == TokenUsage(7, 2)
    assert recorder.attempts == result.attempts


async def test_authentication_and_invalid_request_are_not_retried() -> None:
    calls = 0

    async def operation() -> object:
        nonlocal calls
        calls += 1
        raise ProviderError(ProviderErrorCategory.AUTHENTICATION)

    with pytest.raises(ProviderOperationError) as caught:
        await execute_with_resilience(
            operation,
            context=context(),
            route=ROUTE,
            policy=RetryPolicy(1, max_retries=5, initial_backoff_seconds=0),
            is_fallback=False,
        )

    assert calls == 1
    assert caught.value.category is ProviderErrorCategory.AUTHENTICATION


async def test_expired_deadline_starts_no_attempt() -> None:
    calls = 0

    async def operation() -> object:
        nonlocal calls
        calls += 1
        return object()

    expired = Deadline(0, clock=lambda: 1)
    with pytest.raises(ProviderOperationError) as caught:
        await execute_with_resilience(
            operation,
            context=ProviderCallContext("request", "operation", expired),
            route=ROUTE,
            policy=RetryPolicy(1),
            is_fallback=False,
        )

    assert calls == 0
    assert caught.value.category is ProviderErrorCategory.DEADLINE_EXCEEDED


async def test_attempt_timeout_is_bounded_by_policy() -> None:
    async def operation() -> object:
        await asyncio.Event().wait()
        return object()

    with pytest.raises(ProviderOperationError) as caught:
        await execute_with_resilience(
            operation,
            context=context(),
            route=ROUTE,
            policy=RetryPolicy(0.01),
            is_fallback=False,
        )

    assert caught.value.category is ProviderErrorCategory.TIMEOUT
    assert len(caught.value.attempts) == 1


async def test_cancellation_stops_active_operation_and_records_it() -> None:
    started = asyncio.Event()
    recorder = InMemoryAttemptRecorder()

    async def operation() -> object:
        started.set()
        await asyncio.Event().wait()
        return object()

    task = asyncio.create_task(
        execute_with_resilience(
            operation,
            context=context(),
            route=ROUTE,
            policy=RetryPolicy(1, max_retries=3),
            is_fallback=False,
            recorder=recorder,
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(recorder.attempts) == 1
    assert recorder.attempts[0].status is AttemptStatus.CANCELLED
    assert recorder.attempts[0].error_category is ProviderErrorCategory.CANCELLED


async def test_retry_is_not_started_when_backoff_cannot_fit_deadline() -> None:
    calls = 0

    async def operation() -> object:
        nonlocal calls
        calls += 1
        raise ProviderError(ProviderErrorCategory.NETWORK)

    with pytest.raises(ProviderOperationError) as caught:
        await execute_with_resilience(
            operation,
            context=context(0.05),
            route=ROUTE,
            policy=RetryPolicy(
                1,
                max_retries=3,
                initial_backoff_seconds=0.1,
                max_backoff_seconds=0.1,
            ),
            is_fallback=False,
        )

    assert calls == 1
    assert caught.value.category is ProviderErrorCategory.DEADLINE_EXCEEDED

