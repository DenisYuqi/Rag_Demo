"""Bounded thread pools for synchronous storage, retrieval, OCR, and reports."""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from functools import cache, partial
from typing import Any, cast

from rag_mvp.performance.admission import (
    AdmissionClosedError,
    AdmissionRejectedError,
    AdmissionSnapshot,
    QAAdmissionController,
)


class WorkerPoolSaturatedError(RuntimeError):
    """A safe, retryable rejection from a full bounded worker pool."""

    code = "worker_pool_capacity"
    retryable = True

    def __init__(self, pool_name: str) -> None:
        self.pool_name = pool_name
        super().__init__(self.code)


class WorkerPoolClosedError(RuntimeError):
    code = "worker_pool_closed"
    retryable = True

    def __init__(self, pool_name: str) -> None:
        self.pool_name = pool_name
        super().__init__(self.code)


class BoundedWorkerPool:
    """Run blocking callables off-loop with bounded active and queued work.

    ``run`` accepts a callable rather than an already-computed value, which keeps
    Chroma, BM25, Tesseract, and report rendering entirely inside worker threads.
    Async cancellation returns promptly; an already-running thread retains its slot
    until the callable actually exits, preventing hidden oversubscription.
    """

    def __init__(self, name: str, *, max_workers: int, max_queue: int) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("worker pool name must be non-empty")
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        if type(max_queue) is not int or max_queue < 0:
            raise ValueError("max_queue must be a non-negative integer")
        self.name = name.strip()
        self.max_workers = max_workers
        self.max_queue = max_queue
        self._admission = QAAdmissionController(max_workers, max_queue)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"rag-{self.name}",
        )
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._release_tasks: set[asyncio.Task[None]] = set()

    @property
    def closed(self) -> bool:
        return self._closed

    async def run[T](
        self,
        function: Callable[..., T],
        /,
        *args: object,
        **kwargs: object,
    ) -> T:
        return await self._run(function, args, kwargs, wait_on_cancel=False)

    async def run_cancel_safe[T](
        self,
        function: Callable[..., T],
        /,
        *args: object,
        **kwargs: object,
    ) -> T:
        """Wait for a cancelled blocking call before its bound resource is closed."""

        return await self._run(function, args, kwargs, wait_on_cancel=True)

    async def _run[T](
        self,
        function: Callable[..., T],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        *,
        wait_on_cancel: bool,
    ) -> T:
        if not callable(function):
            raise TypeError("worker operation must be callable")
        if self._closed:
            raise WorkerPoolClosedError(self.name)
        try:
            lease = await self._admission.acquire()
        except AdmissionRejectedError:
            raise WorkerPoolSaturatedError(self.name) from None
        except AdmissionClosedError:
            raise WorkerPoolClosedError(self.name) from None

        loop = asyncio.get_running_loop()
        context = contextvars.copy_context()
        invocation = partial(function, *args, **kwargs)
        try:
            future = cast(
                asyncio.Future[T],
                loop.run_in_executor(self._executor, context.run, invocation),
            )
        except RuntimeError:
            await lease.release()
            raise WorkerPoolClosedError(self.name) from None
        except BaseException:
            await lease.release()
            raise
        try:
            result = await asyncio.shield(future)
        except asyncio.CancelledError:
            if wait_on_cancel:
                with suppress(BaseException):
                    await asyncio.shield(future)
                await lease.release()
            else:
                future.add_done_callback(self._release_after_completion)
            raise
        except BaseException:
            await lease.release()
            raise
        await lease.release()
        return result

    async def snapshot(self) -> AdmissionSnapshot:
        return await self._admission.snapshot()

    async def aclose(self, *, wait: bool = True) -> None:
        if type(wait) is not bool:
            raise TypeError("wait must be a boolean")
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            await self._admission.close()
            if wait:
                await asyncio.to_thread(
                    self._executor.shutdown,
                    wait=True,
                    cancel_futures=False,
                )
                if self._release_tasks:
                    await asyncio.gather(*tuple(self._release_tasks), return_exceptions=True)
            else:
                self._executor.shutdown(wait=False, cancel_futures=True)

    async def __aenter__(self) -> BoundedWorkerPool:
        if self._closed:
            raise WorkerPoolClosedError(self.name)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exc_type, exc, traceback
        await self.aclose()

    def _release_after_completion(self, future: asyncio.Future[Any]) -> None:
        # Retrieve failures so asyncio never writes unrestricted worker exception
        # text to stderr after the awaiting request has already been cancelled.
        with suppress(BaseException):
            future.exception()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._release_cancelled_slot())
        self._release_tasks.add(task)
        task.add_done_callback(self._release_tasks.discard)

    async def _release_cancelled_slot(self) -> None:
        await self._admission._release_slot()


@dataclass(frozen=True, slots=True)
class WorkerPoolLimits:
    """Independent limits prevent slow OCR/report work from starving retrieval."""

    chroma_workers: int = 5
    bm25_workers: int = 5
    ocr_workers: int = 2
    report_workers: int = 2
    queue_per_pool: int = 10

    def __post_init__(self) -> None:
        for name in ("chroma_workers", "bm25_workers", "ocr_workers", "report_workers"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.queue_per_pool) is not int or self.queue_per_pool < 0:
            raise ValueError("queue_per_pool must be a non-negative integer")


class RagWorkerPools:
    """Named execution hooks for every synchronous MVP workload."""

    def __init__(self, limits: WorkerPoolLimits | None = None) -> None:
        resolved = limits or WorkerPoolLimits()
        self.limits = resolved
        queue = resolved.queue_per_pool
        self.chroma = BoundedWorkerPool(
            "chroma",
            max_workers=resolved.chroma_workers,
            max_queue=queue,
        )
        self.bm25 = BoundedWorkerPool(
            "bm25",
            max_workers=resolved.bm25_workers,
            max_queue=queue,
        )
        self.ocr = BoundedWorkerPool(
            "ocr",
            max_workers=resolved.ocr_workers,
            max_queue=queue,
        )
        self.report = BoundedWorkerPool(
            "report",
            max_workers=resolved.report_workers,
            max_queue=queue,
        )

    async def run_chroma[T](
        self,
        function: Callable[..., T],
        /,
        *args: object,
        **kwargs: object,
    ) -> T:
        return await self.chroma.run(function, *args, **kwargs)

    async def run_bm25[T](
        self,
        function: Callable[..., T],
        /,
        *args: object,
        **kwargs: object,
    ) -> T:
        return await self.bm25.run(function, *args, **kwargs)

    async def run_ocr[T](
        self,
        function: Callable[..., T],
        /,
        *args: object,
        **kwargs: object,
    ) -> T:
        return await self.ocr.run(function, *args, **kwargs)

    async def run_report[T](
        self,
        function: Callable[..., T],
        /,
        *args: object,
        **kwargs: object,
    ) -> T:
        return await self.report.run(function, *args, **kwargs)

    async def aclose(self, *, wait: bool = True) -> None:
        await asyncio.gather(
            self.chroma.aclose(wait=wait),
            self.bm25.aclose(wait=wait),
            self.ocr.aclose(wait=wait),
            self.report.aclose(wait=wait),
        )

    async def __aenter__(self) -> RagWorkerPools:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exc_type, exc, traceback
        await self.aclose()


SyncWorkerPool = BoundedWorkerPool
WorkerPools = RagWorkerPools


@cache
def default_worker_pools() -> RagWorkerPools:
    """Return the lazy process fallback used by directly composed test services."""

    return RagWorkerPools()
