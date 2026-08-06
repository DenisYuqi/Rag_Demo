"""FastAPI application factory and operational health endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, cast

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST

from rag_mvp.api.documents import router as documents_router
from rag_mvp.api.errors import install_error_handlers
from rag_mvp.api.evaluation_diagnostics import (
    DiagnosticOperations,
    EvaluationOperations,
)
from rag_mvp.api.evaluation_diagnostics import router as evaluation_diagnostics_router
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
from rag_mvp.observability.logging import RequestTraceContextMiddleware, configure_logging
from rag_mvp.observability.metrics import RAGMetrics
from rag_mvp.observability.runtime import DiagnosticSink, PipelineTelemetry
from rag_mvp.observability.tracing import RAGTracer
from rag_mvp.performance.admission import QAAdmissionController
from rag_mvp.performance.deadlines import QALatencyBudgets
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, Redactor
from rag_mvp.storage.layout import DataLayout

if TYPE_CHECKING:
    from rag_mvp.ui.services import WorkbenchServices


@dataclass(slots=True)
class RuntimeState:
    settings: Settings
    layout: DataLayout
    readiness: ReadinessRegistry
    ingestion_service: IngestionService | None = None
    owns_ingestion_service: bool = False
    qa_services: QARuntimeServices | None = None
    owns_qa_services: bool = False
    evaluation_service: EvaluationOperations | None = None
    diagnostics_service: DiagnosticOperations | None = None
    workbench_services: WorkbenchServices | None = None
    metrics: RAGMetrics = field(default_factory=RAGMetrics)
    telemetry: PipelineTelemetry | None = None
    owns_qa_admission: bool = False
    redactor: Redactor | None = DEFAULT_REDACTOR
    accepting_traffic: bool = True
    ingestion_tasks: set[asyncio.Task[IngestionJob]] = field(default_factory=set)

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

    async def stop_ingestion(self) -> None:
        tasks = tuple(self.ingestion_tasks)
        if not tasks:
            return
        _, pending = await asyncio.wait(
            tasks,
            timeout=self.settings.shutdown_grace_seconds,
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def _build_runtime(
    settings: Settings,
    *,
    ingestion_service: IngestionService | None,
    owns_ingestion_service: bool,
    qa_services: QARuntimeServices | None,
    owns_qa_services: bool,
    evaluation_service: EvaluationOperations | None,
    diagnostics_service: DiagnosticOperations | None,
    workbench_services: WorkbenchServices | None,
    redactor: Redactor | None,
) -> RuntimeState:
    layout = DataLayout.from_root(settings.data_root)
    if ingestion_service is not None and (
        ingestion_service.data_root != layout.root
        or ingestion_service.upload_max_bytes != settings.upload_max_bytes
    ):
        raise ValueError("ingestion_service_configuration_mismatch")
    provider_errors = settings.provider_readiness_errors()
    safety_ready = redactor is not None and redactor.fully_configured
    registry = ReadinessRegistry(
        [
            StaticReadinessCheck("configuration"),
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
    tracer = RAGTracer()
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
        readiness=registry,
        ingestion_service=ingestion_service,
        owns_ingestion_service=owns_ingestion_service,
        qa_services=qa_services,
        owns_qa_services=owns_qa_services,
        evaluation_service=evaluation_service,
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
    diagnostics_service: DiagnosticOperations | None = None,
    workbench_services: WorkbenchServices | None = None,
    redactor: Redactor | None = DEFAULT_REDACTOR,
) -> FastAPI:
    """Compose one single-process service from explicit or environment settings."""
    if owns_ingestion_service and ingestion_service is None:
        raise ValueError("owned_ingestion_service_missing")
    if owns_qa_services and qa_services is None:
        raise ValueError("owned_qa_services_missing")
    resolved_settings = settings or get_settings()
    configure_logging(
        service=resolved_settings.service_name,
        service_version=resolved_settings.service_version,
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
        diagnostics_service=diagnostics_service,
        workbench_services=workbench_services,
        redactor=redactor,
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
            redactor=redactor,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        storage_check = runtime.readiness.get("storage")
        runtime.accepting_traffic = True
        try:
            runtime.layout.initialize()
            if isinstance(storage_check, StaticReadinessCheck):
                storage_check.ready = True
                storage_check.reason = None
        except OSError:
            if isinstance(storage_check, StaticReadinessCheck):
                storage_check.ready = False
                storage_check.reason = "storage_not_writable"
        app.state.runtime = runtime
        try:
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
            yield
        finally:
            runtime.accepting_traffic = False
            try:
                await runtime.stop_ingestion()
            finally:
                try:
                    if (
                        runtime.owns_qa_admission
                        and runtime.qa_services is not None
                        and runtime.qa_services.admission is not None
                    ):
                        await runtime.qa_services.admission.close()
                    if runtime.owns_ingestion_service and runtime.ingestion_service is not None:
                        runtime.ingestion_service.close()
                finally:
                    if runtime.owns_qa_services and runtime.qa_services is not None:
                        await runtime.qa_services.close()

    app = FastAPI(
        title="RAG Assistant MVP",
        version=resolved_settings.service_version,
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    app.add_middleware(RequestTraceContextMiddleware)
    install_error_handlers(app)
    app.include_router(documents_router)
    app.include_router(qa_router)
    app.include_router(evaluation_diagnostics_router)
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
            "components": [
                {"name": item.name, "ready": item.ready, "reason": item.reason}
                for item in components
            ],
        }
        return JSONResponse(
            payload,
            status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
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


def create_executable_app(settings: Settings | None = None) -> FastAPI:
    """Build configured production services or remain honestly unready."""

    resolved_settings = settings or get_settings()
    if (
        resolved_settings.provider_backend == "openai"
        and not resolved_settings.provider_readiness_errors()
    ):
        from rag_mvp.api.composition import compose_openai_services

        composition = compose_openai_services(resolved_settings, DEFAULT_REDACTOR)
        return create_app(
            resolved_settings,
            ingestion_service=composition.ingestion,
            owns_ingestion_service=True,
            qa_services=composition.qa,
            owns_qa_services=True,
            diagnostics_service=composition.diagnostics,
            redactor=DEFAULT_REDACTOR,
        )

    app = create_app(resolved_settings)
    runtime: RuntimeState = app.state.runtime
    runtime.readiness.register(StaticReadinessCheck("index", False, "index_not_ready"))
    runtime.readiness.register(StaticReadinessCheck("qa", False, "qa_not_composed"))
    return app
