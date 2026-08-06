from __future__ import annotations

import asyncio
import contextvars
import threading
import time
from collections.abc import Callable

import pytest

from rag_mvp.performance.worker_pools import (
    BoundedWorkerPool,
    RagWorkerPools,
    WorkerPoolLimits,
    WorkerPoolSaturatedError,
)


@pytest.mark.asyncio
async def test_blocking_work_is_bounded_and_event_loop_remains_responsive() -> None:
    pool = BoundedWorkerPool("test", max_workers=2, max_queue=1)
    gate = threading.Event()
    starts = 0
    starts_lock = threading.Lock()

    def blocking() -> str:
        nonlocal starts
        with starts_lock:
            starts += 1
        gate.wait(timeout=2)
        return threading.current_thread().name

    async def started_twice() -> None:
        while True:
            with starts_lock:
                if starts == 2:
                    return
            await asyncio.sleep(0)

    first = asyncio.create_task(pool.run(blocking))
    second = asyncio.create_task(pool.run(blocking))
    await asyncio.wait_for(started_twice(), timeout=1)
    queued = asyncio.create_task(pool.run(blocking))
    await _wait_until(lambda: pool._admission.queued_count == 1)

    with pytest.raises(WorkerPoolSaturatedError) as rejected:
        await pool.run(blocking)
    assert rejected.value.retryable

    ticks = 0
    deadline = time.monotonic() + 0.03
    while time.monotonic() < deadline:
        ticks += 1
        await asyncio.sleep(0)
    assert ticks > 1

    gate.set()
    names = await asyncio.gather(first, second, queued)
    assert all(name.startswith("rag-test") for name in names)
    await pool.aclose()


@pytest.mark.asyncio
async def test_named_pools_isolate_slow_ocr_from_retrieval_and_reports() -> None:
    pools = RagWorkerPools(
        WorkerPoolLimits(
            chroma_workers=1,
            bm25_workers=1,
            ocr_workers=1,
            report_workers=1,
            queue_per_pool=1,
        )
    )
    gate = threading.Event()
    ocr_started = threading.Event()

    def slow_ocr() -> str:
        ocr_started.set()
        gate.wait(timeout=2)
        return "ocr"

    ocr = asyncio.create_task(pools.run_ocr(slow_ocr))
    await _wait_until(ocr_started.is_set)
    try:
        bm25, chroma, report = await asyncio.wait_for(
            asyncio.gather(
                pools.run_bm25(lambda: "bm25"),
                pools.run_chroma(lambda: "chroma"),
                pools.run_report(lambda: "report"),
            ),
            timeout=0.5,
        )
        assert (bm25, chroma, report) == ("bm25", "chroma", "report")
    finally:
        gate.set()
    assert await ocr == "ocr"
    await pools.aclose()


@pytest.mark.asyncio
async def test_worker_call_preserves_async_context_and_cancellation_keeps_bound() -> None:
    marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker")
    marker.set("trace-safe")
    pool = BoundedWorkerPool("context", max_workers=1, max_queue=0)
    assert await pool.run(marker.get) == "trace-safe"

    gate = threading.Event()
    started = threading.Event()

    def blocking() -> None:
        started.set()
        gate.wait(timeout=2)

    task = asyncio.create_task(pool.run(blocking))
    await _wait_until(started.is_set)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(WorkerPoolSaturatedError):
        await pool.run(lambda: None)
    gate.set()
    await _wait_until(lambda: pool._admission.active_count == 0)
    await pool.aclose()


@pytest.mark.asyncio
async def test_cancel_safe_call_finishes_before_resource_cleanup_and_hides_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pool = BoundedWorkerPool("resource", max_workers=1, max_queue=0)
    gate = threading.Event()
    started = threading.Event()
    ordering: list[str] = []

    def blocking_query() -> None:
        started.set()
        gate.wait(timeout=2)
        ordering.append("query-finished")
        raise RuntimeError("private document text must never reach stderr")

    task = asyncio.create_task(pool.run_cancel_safe(blocking_query))
    await _wait_until(started.is_set)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    ordering.append("resource-closed")

    assert ordering == ["query-finished", "resource-closed"]
    assert "private document text" not in capsys.readouterr().err
    assert (await pool.snapshot()).active == 0
    await pool.aclose()


async def _wait_until(predicate: Callable[[], bool]) -> None:
    async def wait() -> None:
        for _ in range(100_000):
            if predicate():
                return
            await asyncio.sleep(0)
        raise AssertionError("condition did not become true")

    await asyncio.wait_for(wait(), timeout=1)
