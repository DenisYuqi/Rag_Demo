from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from rag_mvp.api.app import create_app
from rag_mvp.api.readiness import StaticReadinessCheck
from rag_mvp.config.settings import Settings


def test_healthz_is_live_even_when_provider_is_unready(tmp_path: Path) -> None:
    app = create_app(Settings(data_root=tmp_path, provider_backend="openai", _env_file=None))

    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_readyz_aggregates_ready_components(tmp_path: Path) -> None:
    app = create_app(Settings(data_root=tmp_path, _env_file=None))

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert all(item["ready"] for item in response.json()["components"])


def test_readyz_returns_safe_component_reason(tmp_path: Path) -> None:
    app = create_app(Settings(data_root=tmp_path, _env_file=None))
    app.state.runtime.readiness.register(
        StaticReadinessCheck("test_dependency", False, "dependency_unavailable")
    )

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert {
        "name": "test_dependency",
        "ready": False,
        "reason": "dependency_unavailable",
    } in response.json()["components"]
