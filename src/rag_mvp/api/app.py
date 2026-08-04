"""FastAPI application factory and operational health endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from rag_mvp.api.readiness import ReadinessRegistry, StaticReadinessCheck
from rag_mvp.config.settings import Settings, get_settings
from rag_mvp.storage.layout import DataLayout


@dataclass(slots=True)
class RuntimeState:
    settings: Settings
    layout: DataLayout
    readiness: ReadinessRegistry
    accepting_traffic: bool = True


def _build_runtime(settings: Settings) -> RuntimeState:
    layout = DataLayout.from_root(settings.data_root)
    provider_errors = settings.provider_readiness_errors()
    registry = ReadinessRegistry(
        [
            StaticReadinessCheck("configuration"),
            StaticReadinessCheck(
                "providers",
                ready=not provider_errors,
                reason=provider_errors[0] if provider_errors else None,
            ),
            StaticReadinessCheck("safety"),
            StaticReadinessCheck("storage", ready=False, reason="storage_initializing"),
        ]
    )
    return RuntimeState(settings=settings, layout=layout, readiness=registry)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Compose one single-process service from explicit or environment settings."""
    resolved_settings = settings or get_settings()
    runtime = _build_runtime(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        storage_check = runtime.readiness.get("storage")
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
        yield
        runtime.accepting_traffic = False

    app = FastAPI(
        title="RAG Assistant MVP",
        version=resolved_settings.service_version,
        lifespan=lifespan,
    )
    app.state.runtime = runtime

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
            status_code=status.HTTP_200_OK
            if ready
            else status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    return app
