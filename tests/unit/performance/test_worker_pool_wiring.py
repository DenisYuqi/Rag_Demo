from __future__ import annotations

import asyncio
import gc
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import fitz
import pytest

import rag_mvp.retrieval.binding as binding_module
import rag_mvp.retrieval.query_dense as query_dense_module
from rag_mvp.domain.ingestion import DocumentKind, EmbeddingSpaceIdentity
from rag_mvp.ingestion.extractors import PageUsabilityPolicy
from rag_mvp.ingestion.service import IngestionService
from rag_mvp.ingestion.validation import ValidatedUpload
from rag_mvp.performance.worker_pools import (
    BoundedWorkerPool,
    RagWorkerPools,
    WorkerPoolLimits,
)
from rag_mvp.providers.fakes import DeterministicEmbeddingProvider
from rag_mvp.providers.models import Deadline, ProviderCallContext
from rag_mvp.retrieval.binding import BoundRetrievalSnapshotFactory
from rag_mvp.retrieval.collection import BoundBm25Retriever
from rag_mvp.retrieval.identity import provider_embedding_identity
from rag_mvp.retrieval.query_dense import BoundDenseRetriever


@dataclass
class _BlockingDense:
    gate: threading.Event
    started: threading.Event
    finished: threading.Event
    worker_thread: list[int]

    def query(self, *_args: object, **_kwargs: object) -> tuple[object, ...]:
        self.worker_thread.append(threading.get_ident())
        self.started.set()
        self.gate.wait(timeout=2)
        self.finished.set()
        return ()


@dataclass
class _FakeSnapshot:
    dense: object
    bm25: object | None = None
    close_order: list[str] | None = None
    is_closed: bool = False
    revision_id: str = "revision-worker-test"

    def __post_init__(self) -> None:
        self.revision = SimpleNamespace(embedding_space=_embedding_identity())

    def close(self) -> None:
        if self.close_order is not None:
            self.close_order.append("close")
        self.is_closed = True


def _embedding_identity() -> EmbeddingSpaceIdentity:
    return EmbeddingSpaceIdentity(
        provider_alias="test",
        model="embedding-model",
        dimension=3,
        normalization="none",
        adapter_version="v1",
    )


def _provider_context() -> ProviderCallContext:
    return ProviderCallContext(
        request_id="request-worker-test",
        operation_id="dense-worker-test",
        deadline=Deadline.after(5),
    )


@pytest.mark.asyncio
async def test_cancelled_dense_query_finishes_before_snapshot_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = threading.Event()
    query_started = threading.Event()
    query_finished = threading.Event()
    worker_thread: list[int] = []
    order: list[str] = []
    dense = _BlockingDense(gate, query_started, query_finished, worker_thread)
    snapshot = _FakeSnapshot(dense, close_order=order)
    pool = BoundedWorkerPool("chroma-order", max_workers=1, max_queue=1)
    provider = DeterministicEmbeddingProvider(provider_embedding_identity(_embedding_identity()))

    monkeypatch.setattr(query_dense_module, "BoundRetrievalSnapshot", _FakeSnapshot)
    monkeypatch.setattr(
        binding_module.BoundRetrievalSnapshotFactory,
        "bind",
        lambda _self: snapshot,
    )
    factory = BoundRetrievalSnapshotFactory(cast(Any, None), cast(Any, None))

    async def request() -> None:
        async with factory.bind_async(pool) as bound:
            retriever = BoundDenseRetriever(
                bound,  # type: ignore[arg-type]
                provider,
                _provider_context(),
                worker_pool=pool,
            )
            query = asyncio.create_task(retriever.search("policy", 1))
            await _wait_until(query_started.is_set)
            query.cancel()
            with pytest.raises(asyncio.CancelledError):
                await query

    request_task = asyncio.create_task(request())
    await _wait_until(query_started.is_set)
    await asyncio.sleep(0)
    await asyncio.sleep(0.02)
    closed_before_query_finished = snapshot.is_closed
    gate.set()
    await asyncio.wait_for(request_task, timeout=1)

    assert not closed_before_query_finished
    assert query_finished.is_set()
    assert snapshot.is_closed
    assert worker_thread and worker_thread[0] != threading.get_ident()
    await pool.aclose()


@pytest.mark.asyncio
async def test_cancelled_worker_exception_is_consumed_without_sensitive_loop_output() -> None:
    pool = BoundedWorkerPool("cancel-error", max_workers=1, max_queue=0)
    gate = threading.Event()
    started = threading.Event()
    loop = asyncio.get_running_loop()
    captured: list[dict[str, Any]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: captured.append(context))

    def explode() -> None:
        started.set()
        gate.wait(timeout=2)
        raise RuntimeError("sensitive person@example.com")

    task = asyncio.create_task(pool.run(explode))
    await _wait_until(started.is_set)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gate.set()
    await _wait_until(lambda: pool._admission.active_count == 0)
    await pool.aclose()
    gc.collect()
    for _ in range(3):
        await asyncio.sleep(0)
    loop.set_exception_handler(previous_handler)

    rendered = repr(captured)
    assert "Future exception was never retrieved" not in rendered
    assert "person@example.com" not in rendered
    assert captured == []


@pytest.mark.asyncio
async def test_injected_bm25_scoring_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = threading.Event()
    started = threading.Event()
    worker_threads: list[int] = []

    class BlockingBm25:
        worker_pool: BoundedWorkerPool | None = None

        def configure_worker_pool(self, worker_pool: BoundedWorkerPool) -> None:
            self.worker_pool = worker_pool

        async def search(self, query: str, limit: int) -> tuple[object, ...]:
            assert self.worker_pool is not None
            return await self.worker_pool.run_cancel_safe(self.search_sync, query, limit)

        def search_sync(self, _query: str, _limit: int) -> tuple[object, ...]:
            worker_threads.append(threading.get_ident())
            started.set()
            gate.wait(timeout=2)
            return ()

    snapshot = _FakeSnapshot(object(), bm25=BlockingBm25())
    monkeypatch.setattr(binding_module, "BoundRetrievalSnapshot", _FakeSnapshot)
    pool = BoundedWorkerPool("bm25-wiring", max_workers=1, max_queue=1)
    retriever = BoundBm25Retriever(
        snapshot,  # type: ignore[arg-type]
        worker_pool=pool,
    )
    safety_release = threading.Timer(0.75, gate.set)
    safety_release.start()
    began = time.monotonic()
    search = asyncio.create_task(retriever.search("policy", 1))
    try:
        await _wait_until(started.is_set)
        await asyncio.sleep(0.01)
        assert time.monotonic() - began < 0.2
    finally:
        gate.set()
        safety_release.cancel()
    assert await search == ()
    assert worker_threads and worker_threads[0] != threading.get_ident()
    await pool.aclose()


@pytest.mark.asyncio
async def test_pdf_and_ocr_extraction_does_not_block_event_loop() -> None:
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
    started = threading.Event()
    worker_threads: list[int] = []

    class BlockingOcr:
        version = "blocking-ocr-v1"

        def recognize(self, _png_bytes: bytes, *, languages: str) -> str:
            assert languages == "chi_sim+eng"
            worker_threads.append(threading.get_ident())
            started.set()
            gate.wait(timeout=2)
            return "scanned policy text with enough useful characters"

    document = fitz.open()
    document.new_page()
    content = document.tobytes()
    document.close()
    upload = ValidatedUpload(
        filename="scan.pdf",
        media_type="application/pdf",
        kind=DocumentKind.PDF,
        content=content,
    )
    service = object.__new__(IngestionService)
    service._ocr_worker_pool = pools.ocr
    service._ocr = BlockingOcr()
    service._ocr_languages = "chi_sim+eng"
    service._page_usability = PageUsabilityPolicy()
    safety_release = threading.Timer(0.75, gate.set)
    safety_release.start()
    began = time.monotonic()
    extraction = asyncio.create_task(service._extract(upload))
    try:
        await _wait_until(started.is_set)
        await asyncio.sleep(0.01)
        assert time.monotonic() - began < 0.2
    finally:
        gate.set()
        safety_release.cancel()
    result = await extraction

    assert result.ocr_page_count == 1
    assert worker_threads and worker_threads[0] != threading.get_ident()
    await pools.aclose()


async def _wait_until(predicate: Any) -> None:
    async def wait() -> None:
        for _ in range(100_000):
            if predicate():
                return
            await asyncio.sleep(0)
        raise AssertionError("condition did not become true")

    await asyncio.wait_for(wait(), timeout=1)
