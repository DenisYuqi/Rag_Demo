"""Reusable total/stage deadline budgets with cancellation-safe execution."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass

from rag_mvp.providers.models import Deadline


@dataclass(frozen=True, slots=True)
class QAStageBudgets:
    total_seconds: float = 9.5
    validation_seconds: float = 0.2
    retrieval_seconds: float = 0.8
    rerank_seconds: float = 1.2
    evidence_assessment_seconds: float = 0.3
    generation_seconds: float = 5.3
    finalization_seconds: float = 0.6

    def __post_init__(self) -> None:
        for name in (
            "total_seconds",
            "validation_seconds",
            "retrieval_seconds",
            "rerank_seconds",
            "evidence_assessment_seconds",
            "generation_seconds",
            "finalization_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
            if name != "total_seconds" and value > self.total_seconds:
                raise ValueError(f"{name} cannot exceed total_seconds")


class DeadlineRunner:
    """Race one operation against a deadline and await cancellation cleanup."""

    def __init__(
        self,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not callable(sleep):
            raise TypeError("sleep must be callable")
        self._sleep = sleep

    async def run[T](
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        deadline: Deadline,
    ) -> T:
        if not isinstance(deadline, Deadline) or deadline.expired:
            raise TimeoutError
        operation_task: asyncio.Future[T] = asyncio.ensure_future(operation())
        timeout_task: asyncio.Future[None] = asyncio.ensure_future(
            self._sleep(deadline.remaining_seconds)
        )
        try:
            done, _ = await asyncio.wait(
                (operation_task, timeout_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if operation_task in done:
                timeout_task.cancel()
                with suppress(asyncio.CancelledError):
                    await timeout_task
                result = await operation_task
                if deadline.expired:
                    raise TimeoutError
                return result
            operation_task.cancel()
            with suppress(asyncio.CancelledError):
                await operation_task
            raise TimeoutError
        except asyncio.CancelledError:
            operation_task.cancel()
            timeout_task.cancel()
            with suppress(asyncio.CancelledError):
                await operation_task
            with suppress(asyncio.CancelledError):
                await timeout_task
            raise
        finally:
            if not timeout_task.done():
                timeout_task.cancel()
                with suppress(asyncio.CancelledError):
                    await timeout_task
