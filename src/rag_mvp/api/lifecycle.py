"""ASGI traffic admission and request tracking for graceful shutdown."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, cast

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_OPERATIONAL_PATHS = frozenset({"/healthz", "/readyz", "/metrics"})


class LifecycleRuntime(Protocol):
    accepting_traffic: bool

    def request_started(self, task: asyncio.Task[Any]) -> None: ...

    def request_finished(self, task: asyncio.Task[Any]) -> None: ...


class TrafficLifecycleMiddleware:
    """Reject new product traffic after shutdown starts and track in-flight work."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope_type = scope["type"]
        if scope_type not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        runtime = cast(LifecycleRuntime, scope["app"].state.runtime)
        path = str(scope.get("path", ""))
        operational = scope_type == "http" and path in _OPERATIONAL_PATHS
        if not operational and not runtime.accepting_traffic:
            if scope_type == "websocket":
                await send(cast(Message, {"type": "websocket.close", "code": 1012}))
            else:
                await JSONResponse(
                    {"error": {"code": "service_unavailable"}},
                    status_code=503,
                )(scope, receive, send)
            return

        if operational:
            await self.app(scope, receive, send)
            return

        task = asyncio.current_task()
        if task is None:
            await self.app(scope, receive, send)
            return
        runtime.request_started(task)
        try:
            await self.app(scope, receive, send)
        finally:
            runtime.request_finished(task)
