"""FIFO admission control for bounded QA pipeline concurrency."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass


class AdmissionRejectedError(RuntimeError):
    """A safe, retryable rejection when active and queued capacity is exhausted."""

    code = "capacity"
    retryable = True

    def __init__(self) -> None:
        super().__init__(self.code)


class AdmissionClosedError(RuntimeError):
    """Raised when new work reaches a controller that is shutting down."""

    code = "admission_closed"
    retryable = True

    def __init__(self) -> None:
        super().__init__(self.code)


@dataclass(frozen=True, slots=True)
class AdmissionSnapshot:
    max_active: int
    max_queue: int
    active: int
    queued: int
    admitted_total: int
    rejected_total: int
    closed: bool


@dataclass(slots=True)
class _Waiter:
    future: asyncio.Future[None]
    granted: bool = False


class AdmissionLease:
    """One idempotently releasable active-pipeline slot."""

    def __init__(self, controller: QAAdmissionController) -> None:
        self._controller = controller
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._controller._release_slot()

    async def __aenter__(self) -> AdmissionLease:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exc_type, exc, traceback
        await self.release()


class QAAdmissionController:
    """Admit active work immediately, queue fairly, and reject beyond the bound.

    Waiting calls are granted in strict FIFO order. Cancellation removes a queued
    waiter, while cancellation after a grant returns the slot to the next waiter.
    """

    def __init__(self, max_active: int = 5, max_queue: int = 10) -> None:
        if type(max_active) is not int or max_active < 1:
            raise ValueError("max_active must be a positive integer")
        if type(max_queue) is not int or max_queue < 0:
            raise ValueError("max_queue must be a non-negative integer")
        self.max_active = max_active
        self.max_queue = max_queue
        self._lock = asyncio.Lock()
        self._waiters: deque[_Waiter] = deque()
        self._active = 0
        self._admitted_total = 0
        self._rejected_total = 0
        self._closed = False

    @property
    def active_count(self) -> int:
        return self._active

    @property
    def queued_count(self) -> int:
        return len(self._waiters)

    @property
    def closed(self) -> bool:
        return self._closed

    async def acquire(self) -> AdmissionLease:
        waiter: _Waiter | None = None
        async with self._lock:
            if self._closed:
                self._rejected_total += 1
                raise AdmissionClosedError
            if self._active < self.max_active and not self._waiters:
                self._active += 1
                self._admitted_total += 1
                return AdmissionLease(self)
            if len(self._waiters) >= self.max_queue:
                self._rejected_total += 1
                raise AdmissionRejectedError
            waiter = _Waiter(asyncio.get_running_loop().create_future())
            self._waiters.append(waiter)

        try:
            await waiter.future
        except asyncio.CancelledError:
            release_grant = False
            async with self._lock:
                if waiter.granted:
                    release_grant = True
                else:
                    with suppress(ValueError):
                        self._waiters.remove(waiter)
            if release_grant:
                await self._release_slot()
            raise
        return AdmissionLease(self)

    @asynccontextmanager
    async def admit(self) -> AsyncIterator[AdmissionLease]:
        lease = await self.acquire()
        try:
            yield lease
        finally:
            await lease.release()

    async def run[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        if not callable(operation):
            raise TypeError("operation must be callable")
        async with self.admit():
            return await operation()

    async def snapshot(self) -> AdmissionSnapshot:
        async with self._lock:
            return AdmissionSnapshot(
                max_active=self.max_active,
                max_queue=self.max_queue,
                active=self._active,
                queued=len(self._waiters),
                admitted_total=self._admitted_total,
                rejected_total=self._rejected_total,
                closed=self._closed,
            )

    async def close(self) -> None:
        """Reject new work and wake queued callers without cancelling active work."""

        async with self._lock:
            self._closed = True
            waiters = tuple(self._waiters)
            self._waiters.clear()
            self._rejected_total += len(waiters)
            for waiter in waiters:
                if not waiter.future.done():
                    waiter.future.set_exception(AdmissionClosedError())

    async def _release_slot(self) -> None:
        async with self._lock:
            if self._active < 1:
                raise RuntimeError("admission slot released without an active lease")
            self._active -= 1
            while self._waiters:
                waiter = self._waiters.popleft()
                if waiter.future.done():
                    continue
                waiter.granted = True
                self._active += 1
                self._admitted_total += 1
                waiter.future.set_result(None)
                break


# Concise aliases for composition code that does not need the QA-specific name.
AdmissionController = QAAdmissionController
CapacityExceededError = AdmissionRejectedError
