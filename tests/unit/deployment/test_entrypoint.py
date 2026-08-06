"""Command entrypoint shutdown-budget wiring tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rag_mvp import __main__ as entrypoint
from rag_mvp.config.settings import Settings


def test_main_passes_server_drain_budget_to_uvicorn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        data_root=tmp_path,
        server_shutdown_grace_seconds=3,
        shutdown_grace_seconds=12,
        _env_file=None,
    )
    captured: dict[str, Any] = {}

    def run(app: str, **kwargs: Any) -> None:
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr(entrypoint, "get_settings", lambda: settings)
    monkeypatch.setattr(entrypoint.uvicorn, "run", run)

    entrypoint.main()

    assert captured["app"] == "rag_mvp.api.app:create_executable_app"
    assert captured["factory"] is True
    assert captured["workers"] == 1
    assert captured["timeout_graceful_shutdown"] == 3
    assert captured["access_log"] is False
    assert captured["log_level"] == "warning"
    assert settings.total_shutdown_budget_seconds == 15
