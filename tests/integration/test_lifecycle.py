from __future__ import annotations

import asyncio
import threading
from contextlib import suppress
from pathlib import Path
from typing import cast

import httpx
import pytest
from fastapi import FastAPI

import rag_mvp.api.app as app_module
from rag_mvp.api.app import create_app
from rag_mvp.api.qa import QARuntimeServices
from rag_mvp.config.settings import Settings
from rag_mvp.ingestion.service import IngestionService
from rag_mvp.performance.admission import QAAdmissionController
from rag_mvp.qa.sessions import ConversationService
from rag_mvp.storage.writer_lock import DataRootWriterLockError


class _ReadyEmitter:
    ready = True

    def emit(self, outcome: object, *, owner_id: str) -> object:
        del outcome, owner_id
        raise AssertionError("emitter must not be called")


class _UnusedOrchestrator:
    async def run(self, **kwargs: object) -> object:
        del kwargs
        raise AssertionError("orchestrator must not be called")


class _CloseTrackedIngestion:
    def __init__(self, root: Path, upload_max_bytes: int, events: list[str]) -> None:
        self.data_root = root.resolve()
        self.upload_max_bytes = upload_max_bytes
        self._events = events

    async def recover_startup(self) -> None:
        return None

    def close(self) -> None:
        self._events.append("ingestion_close")


def _settings(root: Path, *, shutdown_grace_seconds: float = 1.0) -> Settings:
    return Settings(
        data_root=root,
        workbench_enabled=False,
        shutdown_grace_seconds=shutdown_grace_seconds,
        _env_file=None,
    )


class _CapturedLogger:
    def __init__(self, events: list[tuple[str, dict[str, object]]]) -> None:
        self._events = events

    def info(self, event: str, **values: object) -> None:
        self._events.append((event, values))


@pytest.mark.asyncio
async def test_competing_writer_is_rejected_and_abnormal_exit_releases_lock(
    tmp_path: Path,
) -> None:
    first = create_app(_settings(tmp_path))
    competitor = create_app(_settings(tmp_path))

    with pytest.raises(RuntimeError, match="abort_running_app"):
        async with first.router.lifespan_context(first):
            assert first.state.runtime.writer_lock.acquired
            with pytest.raises(DataRootWriterLockError, match="data_root_writer_locked"):
                async with competitor.router.lifespan_context(competitor):
                    raise AssertionError("competing app must not start")
            raise RuntimeError("abort_running_app")

    assert not first.state.runtime.writer_lock.acquired
    async with competitor.router.lifespan_context(competitor):
        assert competitor.state.runtime.writer_lock.acquired
    assert not competitor.state.runtime.writer_lock.acquired


@pytest.mark.asyncio
async def test_shutdown_stops_admission_cancels_work_and_closes_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    request_started = asyncio.Event()
    request_cancelled = asyncio.Event()
    admission = QAAdmissionController(max_active=5, max_queue=1)
    client: httpx.AsyncClient | None = None
    app: FastAPI

    async def close_qa() -> None:
        assert app.state.runtime.accepting_traffic is False
        assert admission.closed
        assert request_cancelled.is_set()
        assert client is not None
        readiness = await client.get("/readyz")
        rejected = await client.get("/late")
        assert readiness.status_code == 503
        assert rejected.status_code == 503
        assert rejected.json() == {"error": {"code": "service_unavailable"}}
        events.append("qa_close")

    qa = QARuntimeServices(
        conversations=cast(ConversationService, object()),
        orchestrator=_UnusedOrchestrator(),  # type: ignore[arg-type]
        emitter=_ReadyEmitter(),
        close_callback=close_qa,
        admission=admission,
    )
    settings = _settings(tmp_path, shutdown_grace_seconds=0.1)
    ingestion = _CloseTrackedIngestion(tmp_path, settings.upload_max_bytes, events)
    app = create_app(
        settings,
        ingestion_service=cast(IngestionService, ingestion),
        owns_ingestion_service=True,
        qa_services=qa,
        owns_qa_services=True,
    )

    @app.get("/blocking")
    async def blocking() -> dict[str, str]:
        request_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            request_cancelled.set()
            raise
        return {"status": "unexpected"}

    @app.get("/late")
    async def late() -> dict[str, str]:
        return {"status": "unexpected"}

    async def flush_telemetry(timeout_seconds: float) -> bool:
        assert timeout_seconds > 0
        assert request_cancelled.is_set()
        events.append("telemetry_flush")
        return True

    telemetry = app.state.runtime.telemetry
    assert telemetry is not None
    monkeypatch.setattr(telemetry, "flush", flush_telemetry)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as open_client:
        client = open_client
        async with app.router.lifespan_context(app):
            request = asyncio.create_task(client.get("/blocking"))
            await asyncio.wait_for(request_started.wait(), timeout=1)
        with suppress(asyncio.CancelledError):
            await request

    assert sorted(events) == ["ingestion_close", "qa_close", "telemetry_flush"]
    release_task = app.state.runtime.writer_lock_release_task
    if release_task is not None:
        await asyncio.wait_for(asyncio.shield(release_task), timeout=1)
    assert not app.state.runtime.writer_lock.acquired


@pytest.mark.asyncio
@pytest.mark.parametrize("last_blocker", ["request", "qa", "ingestion"])
async def test_shutdown_hard_deadline_retains_lock_until_stubborn_work_really_finishes(
    tmp_path: Path,
    last_blocker: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grace = 0.1
    request_started = asyncio.Event()
    request_cancelled = asyncio.Event()
    request_release = asyncio.Event()
    qa_close_called = asyncio.Event()
    qa_close_release = asyncio.Event()
    ingestion_close_called = asyncio.Event()
    ingestion_close_release = threading.Event()
    event_loop = asyncio.get_running_loop()

    class StubbornIngestion(_CloseTrackedIngestion):
        def close(self) -> None:
            event_loop.call_soon_threadsafe(ingestion_close_called.set)
            ingestion_close_release.wait(timeout=2)

    async def close_qa() -> None:
        qa_close_called.set()
        try:
            await qa_close_release.wait()
        except asyncio.CancelledError:
            await qa_close_release.wait()

    qa = QARuntimeServices(
        conversations=cast(ConversationService, object()),
        orchestrator=_UnusedOrchestrator(),  # type: ignore[arg-type]
        emitter=_ReadyEmitter(),
        close_callback=close_qa,
    )
    settings = _settings(tmp_path, shutdown_grace_seconds=grace)
    ingestion = StubbornIngestion(tmp_path, settings.upload_max_bytes, [])
    app = create_app(
        settings,
        ingestion_service=cast(IngestionService, ingestion),
        owns_ingestion_service=True,
        qa_services=qa,
        owns_qa_services=True,
    )
    telemetry = app.state.runtime.telemetry
    assert telemetry is not None

    async def flush_telemetry(_timeout_seconds: float) -> bool:
        return True

    monkeypatch.setattr(telemetry, "flush", flush_telemetry)

    async def stubborn_request() -> None:
        request_started.set()
        while not request_release.is_set():
            try:
                await request_release.wait()
            except asyncio.CancelledError:
                request_cancelled.set()

    request: asyncio.Task[None] | None = None
    try:
        started = asyncio.get_running_loop().time()
        async with app.router.lifespan_context(app):
            request = asyncio.create_task(stubborn_request())
            app.state.runtime.request_started(request)
            request.add_done_callback(app.state.runtime.request_finished)
            await request_started.wait()
            started = asyncio.get_running_loop().time()
        elapsed = asyncio.get_running_loop().time() - started

        assert elapsed <= grace
        assert request_cancelled.is_set()
        await asyncio.wait_for(ingestion_close_called.wait(), timeout=1)
        assert ingestion_close_called.is_set()
        assert qa_close_called.is_set()
        assert app.state.runtime.writer_lock.acquired

        assert request is not None
        background = app.state.runtime.shutdown_background_tasks
        blocker_tasks = {
            "request": request,
            "qa": next(task for task in background if task.get_name() == "shutdown-qa-close"),
            "ingestion": next(
                task for task in background if task.get_name() == "shutdown-ingestion-close"
            ),
        }
        protected_task = blocker_tasks[last_blocker]
        if last_blocker != "request":
            request_release.set()
        if last_blocker != "qa":
            qa_close_release.set()
        if last_blocker != "ingestion":
            ingestion_close_release.set()
        await asyncio.gather(
            *(asyncio.shield(task) for task in background if task is not protected_task),
            return_exceptions=True,
        )
        assert not protected_task.done()
        assert app.state.runtime.writer_lock.acquired

        competitor = create_app(_settings(tmp_path))
        with pytest.raises(DataRootWriterLockError, match="data_root_writer_locked"):
            async with competitor.router.lifespan_context(competitor):
                raise AssertionError("lock must remain held while shutdown work is active")

        request_release.set()
        qa_close_release.set()
        ingestion_close_release.set()
        await asyncio.wait_for(asyncio.shield(protected_task), timeout=1)
        release_task = app.state.runtime.writer_lock_release_task
        assert release_task is not None
        await asyncio.wait_for(asyncio.shield(release_task), timeout=1)
        assert not app.state.runtime.writer_lock.acquired, app.state.runtime.shutdown_failed_tasks

        async with competitor.router.lifespan_context(competitor):
            assert competitor.state.runtime.writer_lock.acquired
        assert not competitor.state.runtime.writer_lock.acquired
    finally:
        request_release.set()
        qa_close_release.set()
        ingestion_close_release.set()
        if request is not None:
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(request, timeout=1)


@pytest.mark.asyncio
async def test_shutdown_cleanup_exception_is_failed_and_retains_writer_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    logger = _CapturedLogger(events)
    monkeypatch.setattr(app_module, "get_logger", lambda _operation: logger)

    async def failed_close() -> None:
        raise RuntimeError("qa_close_failed")

    qa = QARuntimeServices(
        conversations=cast(ConversationService, object()),
        orchestrator=_UnusedOrchestrator(),  # type: ignore[arg-type]
        emitter=_ReadyEmitter(),
        close_callback=failed_close,
    )
    app = create_app(
        _settings(tmp_path),
        qa_services=qa,
        owns_qa_services=True,
    )

    try:
        async with app.router.lifespan_context(app):
            assert app.state.runtime.writer_lock.acquired

        runtime = app.state.runtime
        assert runtime.shutdown_failed_tasks == {"shutdown-qa-close"}
        assert runtime.writer_lock.acquired
        shutdown_event = next(
            values for event, values in events if event == "runtime.shutdown.sequence.completed"
        )
        assert shutdown_event["outcome"] == "failed"
        assert shutdown_event["counts"] == {"pending_tasks": 0, "failed_tasks": 1}
    finally:
        app.state.runtime.writer_lock.release()


@pytest.mark.asyncio
async def test_shutdown_telemetry_flush_false_is_a_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    logger = _CapturedLogger(events)
    monkeypatch.setattr(app_module, "get_logger", lambda _operation: logger)
    app = create_app(_settings(tmp_path))
    telemetry = app.state.runtime.telemetry
    assert telemetry is not None

    async def failed_flush(_timeout_seconds: float) -> bool:
        return False

    monkeypatch.setattr(telemetry, "flush", failed_flush)
    try:
        async with app.router.lifespan_context(app):
            assert app.state.runtime.writer_lock.acquired

        runtime = app.state.runtime
        assert runtime.shutdown_failed_tasks == {"shutdown-telemetry-flush"}
        assert runtime.writer_lock.acquired
        shutdown_event = next(
            values for event, values in events if event == "runtime.shutdown.sequence.completed"
        )
        assert shutdown_event["outcome"] == "failed"
        assert shutdown_event["counts"] == {"pending_tasks": 0, "failed_tasks": 1}
    finally:
        app.state.runtime.writer_lock.release()
