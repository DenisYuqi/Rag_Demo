"""FastAPI application factory and operational health endpoints."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST

from rag_mvp.api.comparisons import router as comparisons_router
from rag_mvp.api.documents import router as documents_router
from rag_mvp.api.errors import install_error_handlers
from rag_mvp.api.evaluation_diagnostics import (
    DiagnosticOperations,
    EvaluationOperations,
)
from rag_mvp.api.evaluation_diagnostics import router as evaluation_diagnostics_router
from rag_mvp.api.lifecycle import TrafficLifecycleMiddleware
from rag_mvp.api.qa import (
    QARuntimeReadinessCheck,
    QARuntimeServices,
    install_qa_openapi_contract,
)
from rag_mvp.api.qa import router as qa_router
from rag_mvp.api.readiness import ReadinessRegistry, StaticReadinessCheck
from rag_mvp.config.settings import Settings, get_settings
from rag_mvp.domain.ingestion import IngestionJob
from rag_mvp.ingestion.service import IngestionService
from rag_mvp.observability.logging import (
    RequestTraceContextMiddleware,
    configure_logging,
    get_logger,
    is_safe_identifier,
)
from rag_mvp.observability.metrics import RAGMetrics
from rag_mvp.observability.runtime import DiagnosticSink, PipelineTelemetry
from rag_mvp.observability.tracing import (
    TelemetryConfigurationError,
    create_rag_tracer,
    tracing_readiness_errors,
)
from rag_mvp.performance.admission import QAAdmissionController
from rag_mvp.performance.deadlines import QALatencyBudgets
from rag_mvp.performance.load_report import INSTANCE_ID_HEADER
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, Redactor
from rag_mvp.storage.layout import DataLayout
from rag_mvp.storage.writer_lock import DataRootWriterLock

if TYPE_CHECKING:
    from rag_mvp.ui.services import EvaluationGateway, WorkbenchServices


@dataclass(slots=True)
class RuntimeState:
    settings: Settings
    layout: DataLayout
    writer_lock: DataRootWriterLock
    readiness: ReadinessRegistry
    instance_identity: str = field(default_factory=lambda: f"instance-{uuid4().hex}")
    ingestion_service: IngestionService | None = None
    owns_ingestion_service: bool = False
    qa_services: QARuntimeServices | None = None
    owns_qa_services: bool = False
    evaluation_service: EvaluationOperations | None = None
    owns_evaluation_service: bool = False
    diagnostics_service: DiagnosticOperations | None = None
    workbench_services: WorkbenchServices | None = None
    metrics: RAGMetrics = field(default_factory=RAGMetrics)
    telemetry: PipelineTelemetry | None = None
    owns_qa_admission: bool = False
    redactor: Redactor | None = DEFAULT_REDACTOR
    accepting_traffic: bool = False
    ingestion_tasks: set[asyncio.Task[IngestionJob]] = field(default_factory=set)
    request_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    shutdown_started: bool = False
    shutdown_sequence_complete: bool = False
    shutdown_background_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    shutdown_failed_tasks: set[str] = field(default_factory=set)
    writer_lock_release_task: asyncio.Task[None] | None = None

    def request_started(self, task: asyncio.Task[Any]) -> None:
        self.request_tasks.add(task)

    def request_finished(self, task: asyncio.Task[Any]) -> None:
        self.request_tasks.discard(task)

    async def _run_ingestion(self, job_id: str) -> IngestionJob:
        service = self.ingestion_service
        if service is None:
            raise RuntimeError("ingestion_service_unavailable")
        if self.telemetry is None:
            return await service.run(job_id)
        async with self.telemetry.stage("ingestion"):
            return await service.run(job_id)

    async def schedule_ingestion(self, job_id: str) -> None:
        if not self.accepting_traffic or self.ingestion_service is None:
            return
        task = asyncio.create_task(
            self._run_ingestion(job_id),
            name=f"ingestion-{job_id}",
        )
        self.ingestion_tasks.add(task)
        task.add_done_callback(self._ingestion_finished)

    def _ingestion_finished(self, task: asyncio.Task[IngestionJob]) -> None:
        self.ingestion_tasks.discard(task)
        if not task.cancelled():
            with suppress(Exception):
                job = task.result()
                if self.telemetry is not None:
                    self.telemetry.record_ingestion(job)

    async def shutdown(self) -> None:
        """Stop admission, drain or cancel work, then flush and close resources."""

        if self.shutdown_started:
            return
        self.shutdown_started = True
        self.shutdown_sequence_complete = False
        self.shutdown_failed_tasks.clear()
        self.accepting_traffic = False
        grace = self.settings.app_shutdown_grace_seconds
        loop = asyncio.get_running_loop()
        started = loop.time()
        hard_deadline = started + grace

        admission = self.qa_services.admission if self.qa_services is not None else None
        if admission is not None:
            await self._attempt_async(
                admission.close(),
                deadline=min(hard_deadline, started + (grace * 0.1)),
                task_name="shutdown-admission-close",
            )

        if self.owns_evaluation_service and self.evaluation_service is not None:
            await self._attempt_async(
                self._close_evaluation_service(),
                deadline=min(hard_deadline, started + (grace * 0.3)),
                task_name="shutdown-evaluation-close",
            )

        await self._drain_or_cancel(
            drain_deadline=min(hard_deadline, started + (grace * 0.45)),
            cancel_deadline=min(hard_deadline, started + (grace * 0.55)),
        )

        # Start every resource cleanup before waiting for any one of them. A
        # blocking synchronous close therefore cannot prevent the async QA
        # close from being attempted. The small margin keeps shutdown itself
        # inside the configured hard deadline despite scheduler overhead.
        cleanup_deadline = min(hard_deadline, started + (grace * 0.75))
        cleanup_tasks: set[asyncio.Task[Any]] = set()
        cleanup_budget = max(0.001, cleanup_deadline - loop.time())
        if self.telemetry is not None:
            cleanup_tasks.add(
                asyncio.create_task(
                    self._close_telemetry(cleanup_budget),
                    name="shutdown-telemetry-flush",
                )
            )

        if self.owns_ingestion_service and self.ingestion_service is not None:
            cleanup_tasks.add(
                asyncio.create_task(
                    asyncio.to_thread(self.ingestion_service.close),
                    name="shutdown-ingestion-close",
                )
            )

        if self.owns_qa_services and self.qa_services is not None:
            cleanup_tasks.add(
                asyncio.create_task(
                    self.qa_services.close(),
                    name="shutdown-qa-close",
                )
            )

        await self._wait_until(
            cleanup_tasks,
            deadline=cleanup_deadline,
            cancel_pending=False,
        )
        self.shutdown_sequence_complete = True
        pending_count = len(self._shutdown_pending_tasks())
        failed_count = len(self.shutdown_failed_tasks)
        outcome = "failed" if failed_count else ("deferred" if pending_count else "succeeded")
        get_logger("lifecycle").info(
            "runtime.shutdown.sequence.completed",
            outcome=outcome,
            counts={"pending_tasks": pending_count, "failed_tasks": failed_count},
        )

    def release_writer_lock_when_safe(self) -> None:
        """Release ownership now or after every shutdown operation really ends."""

        if not self.writer_lock.acquired or not self.shutdown_sequence_complete:
            return
        if self.shutdown_failed_tasks:
            return
        pending = self._shutdown_pending_tasks()
        if not pending:
            self.writer_lock.release()
            return
        release_task = self.writer_lock_release_task
        if release_task is not None and not release_task.done():
            return
        release_task = asyncio.create_task(
            self._release_writer_lock_after(pending),
            name="shutdown-writer-lock-release",
        )
        release_task.add_done_callback(self._consume_background_result)
        self.writer_lock_release_task = release_task

    async def _release_writer_lock_after(self, pending: set[asyncio.Task[Any]]) -> None:
        # asyncio.wait does not propagate cancellation to the operations whose
        # completion protects the data root. If the loop exits first, this
        # waiter is cancelled and the process-held OS lock remains authoritative.
        await asyncio.wait(pending)
        for task in pending:
            self._consume_background_result(task)
        if not self._shutdown_pending_tasks() and not self.shutdown_failed_tasks:
            self.writer_lock.release()

    async def _close_telemetry(self, timeout_seconds: float) -> None:
        telemetry = self.telemetry
        if telemetry is None:
            return
        try:
            if not await telemetry.flush(timeout_seconds):
                raise RuntimeError("telemetry_flush_failed")
        finally:
            # Exporter shutdown is isolated from the event loop and remains
            # bounded by the RuntimeState cleanup deadline that owns this task.
            await asyncio.to_thread(telemetry.tracer.shutdown)

    async def _close_evaluation_service(self) -> None:
        service = self.evaluation_service
        close = None if service is None else getattr(service, "close", None)
        if not callable(close):
            raise RuntimeError("evaluation_close_unavailable")
        operation = close()
        if not inspect.isawaitable(operation):
            raise RuntimeError("evaluation_close_invalid")
        await operation

    async def _drain_or_cancel(
        self,
        *,
        drain_deadline: float,
        cancel_deadline: float,
    ) -> None:
        loop = asyncio.get_running_loop()
        current = asyncio.current_task()

        while True:
            tasks = self._unfinished_tasks(current)
            if not tasks:
                return
            remaining = drain_deadline - loop.time()
            if remaining <= 0:
                break
            _, pending = await asyncio.wait(tasks, timeout=remaining)
            if pending:
                break

        pending = self._unfinished_tasks(current)
        for task in pending:
            task.cancel()
        if pending:
            _, stubborn = await asyncio.wait(
                pending, timeout=max(0.0, cancel_deadline - loop.time())
            )
            for task in stubborn:
                task.cancel()
                self._track_shutdown_task(task)

    async def _attempt_async(
        self,
        operation: Coroutine[Any, Any, Any],
        *,
        deadline: float,
        task_name: str,
    ) -> None:
        task: asyncio.Task[Any] = asyncio.create_task(operation, name=task_name)
        await self._wait_until({task}, deadline=deadline)

    async def _wait_until(
        self,
        tasks: set[asyncio.Task[Any]],
        *,
        deadline: float,
        cancel_pending: bool = True,
    ) -> None:
        if not tasks:
            return
        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        done, pending = await asyncio.wait(tasks, timeout=remaining)
        for task in done:
            self._consume_background_result(task)
        for task in pending:
            if cancel_pending:
                task.cancel()
                self._record_shutdown_failure(task)
            self._track_shutdown_task(task)

    def _track_shutdown_task(self, task: asyncio.Task[Any]) -> None:
        self.shutdown_background_tasks.add(task)
        task.add_done_callback(self._consume_background_result)

    def _consume_background_result(self, task: asyncio.Task[Any]) -> None:
        if not task.done():
            return
        if task.cancelled():
            self._record_shutdown_failure(task)
            return
        try:
            task.result()
        except Exception:
            self._record_shutdown_failure(task)

    def _record_shutdown_failure(self, task: asyncio.Task[Any]) -> None:
        task_name = task.get_name()
        if task_name.startswith("shutdown-") and task_name != "shutdown-writer-lock-release":
            self.shutdown_failed_tasks.add(task_name)

    def _unfinished_tasks(
        self,
        current: asyncio.Task[Any] | None,
    ) -> set[asyncio.Task[Any]]:
        return {
            task
            for task in (*self.request_tasks, *self.ingestion_tasks)
            if task is not current and not task.done()
        }

    def _shutdown_pending_tasks(self) -> set[asyncio.Task[Any]]:
        return {
            task
            for task in (
                *self.request_tasks,
                *self.ingestion_tasks,
                *self.shutdown_background_tasks,
            )
            if not task.done()
        }


def _build_runtime(
    settings: Settings,
    *,
    ingestion_service: IngestionService | None,
    owns_ingestion_service: bool,
    qa_services: QARuntimeServices | None,
    owns_qa_services: bool,
    evaluation_service: EvaluationOperations | None,
    owns_evaluation_service: bool,
    diagnostics_service: DiagnosticOperations | None,
    workbench_services: WorkbenchServices | None,
    redactor: Redactor | None,
    writer_lock: DataRootWriterLock | None,
) -> RuntimeState:
    layout = DataLayout.from_root(settings.data_root)
    if ingestion_service is not None and (
        ingestion_service.data_root != layout.root
        or ingestion_service.upload_max_bytes != settings.upload_max_bytes
    ):
        raise ValueError("ingestion_service_configuration_mismatch")
    provider_errors = settings.provider_readiness_errors()
    identity_error = _runtime_identity_error(settings)
    telemetry_errors = list(tracing_readiness_errors(settings))
    try:
        tracer = create_rag_tracer(settings)
    except TelemetryConfigurationError as error:
        if error.reason not in telemetry_errors:
            telemetry_errors.append(error.reason)
        tracer = create_rag_tracer(
            settings.model_copy(
                update={
                    "telemetry_exporter": "none",
                    "telemetry_otlp_traces_endpoint": None,
                }
            )
        )
    safety_ready = redactor is not None and redactor.fully_configured
    registry = ReadinessRegistry(
        [
            StaticReadinessCheck(
                "configuration",
                ready=identity_error is None,
                reason=identity_error,
            ),
            StaticReadinessCheck(
                "providers",
                ready=not provider_errors,
                reason=provider_errors[0] if provider_errors else None,
            ),
            StaticReadinessCheck(
                "safety",
                ready=safety_ready,
                reason=None if safety_ready else "safety_unavailable",
            ),
            StaticReadinessCheck(
                "telemetry",
                ready=not telemetry_errors,
                reason=telemetry_errors[0] if telemetry_errors else None,
            ),
            StaticReadinessCheck("storage", ready=False, reason="storage_initializing"),
        ]
    )
    if ingestion_service is not None:
        registry.register(
            StaticReadinessCheck("ingestion", ready=False, reason="ingestion_initializing")
        )
    if qa_services is not None:
        registry.register(QARuntimeReadinessCheck(qa_services))
    metrics = RAGMetrics()
    diagnostic_sink = (
        cast(DiagnosticSink, diagnostics_service)
        if callable(getattr(diagnostics_service, "save", None))
        else None
    )
    telemetry = PipelineTelemetry(
        settings,
        metrics=metrics,
        tracer=tracer,
        diagnostics=diagnostic_sink,
    )
    owns_qa_admission = False
    if qa_services is not None:
        admission = qa_services.admission
        if admission is None:
            admission = QAAdmissionController(settings.qa_max_active, settings.qa_max_queue)
            owns_qa_admission = True
        effective_telemetry = qa_services.telemetry or telemetry
        telemetry = effective_telemetry
        metrics = effective_telemetry.metrics
        qa_services = replace(
            qa_services,
            admission=admission,
            telemetry=effective_telemetry,
            latency_budgets=(
                qa_services.latency_budgets or QALatencyBudgets.from_settings(settings)
            ),
        )
        install_observer = getattr(qa_services.orchestrator, "set_stage_observer", None)
        if callable(install_observer):
            install_observer(effective_telemetry)
    return RuntimeState(
        settings=settings,
        layout=layout,
        writer_lock=writer_lock or DataRootWriterLock(layout.writer_lock),
        readiness=registry,
        ingestion_service=ingestion_service,
        owns_ingestion_service=owns_ingestion_service,
        qa_services=qa_services,
        owns_qa_services=owns_qa_services,
        evaluation_service=evaluation_service,
        owns_evaluation_service=owns_evaluation_service,
        diagnostics_service=diagnostics_service,
        workbench_services=workbench_services,
        metrics=metrics,
        telemetry=telemetry,
        owns_qa_admission=owns_qa_admission,
        redactor=redactor,
    )


def create_app(
    settings: Settings | None = None,
    *,
    ingestion_service: IngestionService | None = None,
    owns_ingestion_service: bool = False,
    qa_services: QARuntimeServices | None = None,
    owns_qa_services: bool = False,
    evaluation_service: EvaluationOperations | None = None,
    owns_evaluation_service: bool = False,
    diagnostics_service: DiagnosticOperations | None = None,
    workbench_services: WorkbenchServices | None = None,
    redactor: Redactor | None = DEFAULT_REDACTOR,
    writer_lock: DataRootWriterLock | None = None,
) -> FastAPI:
    """Compose one single-process service from explicit or environment settings."""
    if owns_ingestion_service and ingestion_service is None:
        raise ValueError("owned_ingestion_service_missing")
    if owns_qa_services and qa_services is None:
        raise ValueError("owned_qa_services_missing")
    if owns_evaluation_service and evaluation_service is None:
        raise ValueError("owned_evaluation_service_missing")
    resolved_settings = settings or get_settings()
    identity_error = _runtime_identity_error(resolved_settings)
    configure_logging(
        service=(resolved_settings.service_name if identity_error is None else "rag-mvp"),
        service_version=(
            resolved_settings.service_version if identity_error is None else "invalid"
        ),
        config_version=resolved_settings.configuration_identity,
        level=resolved_settings.log_level,
    )
    runtime = _build_runtime(
        resolved_settings,
        ingestion_service=ingestion_service,
        owns_ingestion_service=owns_ingestion_service,
        qa_services=qa_services,
        owns_qa_services=owns_qa_services,
        evaluation_service=evaluation_service,
        owns_evaluation_service=owns_evaluation_service,
        diagnostics_service=diagnostics_service,
        workbench_services=workbench_services,
        redactor=redactor,
        writer_lock=writer_lock,
    )
    if resolved_settings.workbench_enabled and runtime.workbench_services is None:
        from rag_mvp.ui.services import (
            RuntimeDiagnosticsGateway,
            configured_workbench_services,
        )

        diagnostics_gateway = (
            RuntimeDiagnosticsGateway(diagnostics_service, runtime.readiness.report)
            if diagnostics_service is not None
            else None
        )
        runtime.workbench_services = configured_workbench_services(
            settings=resolved_settings,
            qa=runtime.qa_services,
            ingestion=ingestion_service,
            diagnostics=diagnostics_gateway,
            evaluations=cast("EvaluationGateway", evaluation_service),
            redactor=redactor,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        storage_check = runtime.readiness.get("storage")
        runtime.accepting_traffic = False
        runtime.shutdown_started = False
        runtime.shutdown_sequence_complete = False
        runtime.shutdown_failed_tasks.clear()
        storage_writable = False
        try:
            runtime.layout.initialize()
            storage_writable = True
            if isinstance(storage_check, StaticReadinessCheck):
                storage_check.ready = True
                storage_check.reason = None
        except OSError:
            if isinstance(storage_check, StaticReadinessCheck):
                storage_check.ready = False
                storage_check.reason = "storage_not_writable"
        lifecycle_started = False
        try:
            if storage_writable:
                runtime.writer_lock.acquire()
            lifecycle_started = True
            app.state.runtime = runtime
            if runtime.ingestion_service is not None:
                ingestion_check = runtime.readiness.get("ingestion")
                try:
                    await runtime.ingestion_service.recover_startup()
                    if isinstance(ingestion_check, StaticReadinessCheck):
                        ingestion_check.ready = True
                        ingestion_check.reason = None
                except Exception:
                    if isinstance(ingestion_check, StaticReadinessCheck):
                        ingestion_check.ready = False
                        ingestion_check.reason = "ingestion_recovery_failed"
            if runtime.evaluation_service is not None:
                start_evaluations = getattr(runtime.evaluation_service, "startup", None)
                if callable(start_evaluations):
                    try:
                        operation = start_evaluations()
                        if not inspect.isawaitable(operation):
                            raise RuntimeError("evaluation_startup_invalid")
                        await operation
                    except Exception:
                        get_logger("lifecycle").warning(
                            "evaluation.startup.failed",
                            error_category="evaluation_startup_failed",
                        )
            runtime.accepting_traffic = storage_writable
            yield
        finally:
            if lifecycle_started:
                try:
                    await runtime.shutdown()
                finally:
                    runtime.release_writer_lock_when_safe()

    app = FastAPI(
        title="RAG Assistant MVP",
        version=resolved_settings.service_version,
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    app.add_middleware(RequestTraceContextMiddleware)
    app.add_middleware(TrafficLifecycleMiddleware)
    install_error_handlers(app)
    app.include_router(documents_router)
    app.include_router(qa_router)
    app.include_router(evaluation_diagnostics_router)
    app.include_router(comparisons_router)
    install_qa_openapi_contract(app)

    @app.get("/healthz", tags=["operations"])
    async def healthz() -> dict[str, str]:
        return {"status": "alive", "service": resolved_settings.service_name}

    @app.get("/readyz", tags=["operations"])
    async def readyz(request: Request) -> JSONResponse:
        state: RuntimeState = request.app.state.runtime
        ready, components = state.readiness.report()
        ready = ready and state.accepting_traffic
        payload = {
            "status": "ready" if ready else "not_ready",
            "configuration_id": state.settings.configuration_identity,
            "instance_identity": state.instance_identity,
            "components": [
                {"name": item.name, "ready": item.ready, "reason": item.reason}
                for item in components
            ],
        }
        return JSONResponse(
            payload,
            status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={INSTANCE_ID_HEADER: state.instance_identity},
        )

    @app.get("/metrics", tags=["operations"], include_in_schema=False)
    async def metrics() -> Response:
        return Response(runtime.metrics.render(), media_type=CONTENT_TYPE_LATEST)

    if resolved_settings.workbench_enabled:
        from rag_mvp.ui.workbench import mount_workbench

        workbench_check = StaticReadinessCheck(
            "workbench", ready=False, reason="workbench_initializing"
        )
        runtime.readiness.register(workbench_check)
        try:
            if runtime.workbench_services is None:
                raise RuntimeError("workbench_services_missing")
            app = mount_workbench(
                app,
                settings=resolved_settings,
                services=runtime.workbench_services,
            )
            app.state.runtime = runtime
            workbench_check.ready = True
            workbench_check.reason = None
        except Exception:
            workbench_check.ready = False
            workbench_check.reason = "workbench_initialization_failed"

    return app


def _runtime_identity_error(settings: Settings) -> str | None:
    if not is_safe_identifier(settings.service_name) or not is_safe_identifier(
        settings.service_version
    ):
        return "configuration_identity_invalid"
    return None


def create_executable_app(settings: Settings | None = None) -> FastAPI:
    """Build configured production services or remain honestly unready."""

    resolved_settings = settings or get_settings()
    if (
        resolved_settings.provider_backend == "openai"
        and not resolved_settings.provider_readiness_errors()
    ):
        from rag_mvp.api.composition import compose_openai_services

        layout = DataLayout.from_root(resolved_settings.data_root)
        try:
            layout.initialize()
        except OSError:
            return _create_uncomposed_executable_app(resolved_settings)
        writer_lock = DataRootWriterLock(layout.writer_lock)
        writer_lock.acquire()
        try:
            composition = compose_openai_services(resolved_settings, DEFAULT_REDACTOR)
            return create_app(
                resolved_settings,
                ingestion_service=composition.ingestion,
                owns_ingestion_service=True,
                qa_services=composition.qa,
                owns_qa_services=True,
                evaluation_service=composition.evaluation,
                owns_evaluation_service=composition.evaluation is not None,
                diagnostics_service=composition.diagnostics,
                redactor=DEFAULT_REDACTOR,
                writer_lock=writer_lock,
            )
        except BaseException:
            writer_lock.release()
            raise

    return _create_uncomposed_executable_app(resolved_settings)


def _create_uncomposed_executable_app(settings: Settings) -> FastAPI:
    app = create_app(settings)
    runtime: RuntimeState = app.state.runtime
    runtime.readiness.register(StaticReadinessCheck("index", False, "index_not_ready"))
    runtime.readiness.register(StaticReadinessCheck("qa", False, "qa_not_composed"))
    return app
