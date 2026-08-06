from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from rag_mvp.performance.admission import (
    AdmissionClosedError,
    AdmissionRejectedError,
    QAAdmissionController,
)


@pytest.mark.asyncio
async def test_five_pipelines_overlap_and_excess_is_retryably_rejected() -> None:
    controller = QAAdmissionController(max_active=5, max_queue=1)
    release = asyncio.Event()
    started: list[int] = []

    async def pipeline(index: int) -> int:
        async with controller.admit():
            started.append(index)
            await release.wait()
            return index

    active = [asyncio.create_task(pipeline(index)) for index in range(5)]
    await _wait_until(lambda: len(started) == 5)
    queued = asyncio.create_task(pipeline(5))
    await _wait_until(lambda: controller.queued_count == 1)

    with pytest.raises(AdmissionRejectedError) as rejected:
        await controller.acquire()

    assert rejected.value.code == "capacity"
    assert rejected.value.retryable
    assert controller.active_count == 5

    release.set()
    assert await asyncio.gather(*active, queued) == [0, 1, 2, 3, 4, 5]
    snapshot = await controller.snapshot()
    assert snapshot.active == 0
    assert snapshot.queued == 0
    assert snapshot.admitted_total == 6
    assert snapshot.rejected_total == 1


@pytest.mark.asyncio
async def test_waiters_are_fifo_and_cancelled_waiter_does_not_consume_a_slot() -> None:
    controller = QAAdmissionController(max_active=1, max_queue=3)
    first = await controller.acquire()
    order: list[str] = []

    async def wait_for_slot(name: str) -> None:
        async with controller.admit():
            order.append(name)
            await asyncio.sleep(0)

    cancelled = asyncio.create_task(wait_for_slot("cancelled"))
    second = asyncio.create_task(wait_for_slot("second"))
    third = asyncio.create_task(wait_for_slot("third"))
    await _wait_until(lambda: controller.queued_count == 3)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    await first.release()
    await asyncio.gather(second, third)

    assert order == ["second", "third"]
    assert controller.active_count == 0


@pytest.mark.asyncio
async def test_close_wakes_queue_but_allows_active_lease_to_drain() -> None:
    controller = QAAdmissionController(max_active=1, max_queue=1)
    active = await controller.acquire()
    queued = asyncio.create_task(controller.acquire())
    await _wait_until(lambda: controller.queued_count == 1)

    await controller.close()

    with pytest.raises(AdmissionClosedError):
        await queued
    with pytest.raises(AdmissionClosedError):
        await controller.acquire()
    assert controller.active_count == 1
    await active.release()
    assert controller.active_count == 0


async def _wait_until(predicate: Callable[[], bool]) -> None:
    async def wait() -> None:
        for _ in range(100_000):
            if predicate():
                return
            await asyncio.sleep(0)
        raise AssertionError("condition did not become true")

    await asyncio.wait_for(wait(), timeout=1)
