from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_mvp.api.app import create_app, create_executable_app
from rag_mvp.api.readiness import StaticReadinessCheck
from rag_mvp.config.settings import Settings
from rag_mvp.safety.redactor import Redactor
from rag_mvp.storage.layout import DataLayout
from rag_mvp.storage.writer_lock import DataRootWriterLock, DataRootWriterLockError


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
    assert response.json()["instance_identity"] == response.headers["x-rag-instance-id"]
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


def test_readyz_fails_closed_when_safety_is_unavailable(tmp_path: Path) -> None:
    app = create_app(
        Settings(data_root=tmp_path, _env_file=None),
        redactor=Redactor(()),
    )

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    assert {
        "name": "safety",
        "ready": False,
        "reason": "safety_unavailable",
    } in response.json()["components"]


def test_executable_is_unready_until_index_and_qa_are_composed(tmp_path: Path) -> None:
    app = create_executable_app(Settings(data_root=tmp_path, _env_file=None))

    with TestClient(app) as client:
        liveness = client.get("/healthz")
        readiness = client.get("/readyz")

    assert liveness.status_code == 200
    assert readiness.status_code == 503
    assert {
        "name": "index",
        "ready": False,
        "reason": "index_not_ready",
    } in readiness.json()["components"]
    assert {
        "name": "qa",
        "ready": False,
        "reason": "qa_not_composed",
    } in readiness.json()["components"]


def test_configured_executable_composes_services_and_waits_for_an_index(tmp_path: Path) -> None:
    app = create_executable_app(
        Settings(
            data_root=tmp_path,
            provider_backend="openai",
            openai_api_key="test-key",
            openai_send_dimensions=False,
            openai_max_tokens_parameter="max_tokens",
            _env_file=None,
        )
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        readiness = client.get("/readyz")
        documents = client.get("/api/v1/documents")
        qa = client.post(
            "/api/v1/qa",
            json={"owner_id": "owner-1", "question": "Question"},
        )

    assert app.state.runtime.ingestion_service is not None
    assert app.state.runtime.qa_services is not None
    assert readiness.status_code == 503
    assert {
        "name": "qa",
        "ready": False,
        "reason": "index_not_ready",
    } in readiness.json()["components"]
    assert documents.status_code == 200
    assert documents.json() == {"active_index_revision": None, "documents": []}
    assert qa.status_code == 503
    assert qa.json() == {"error": {"code": "qa_unavailable"}}


def test_configured_executable_stays_live_when_storage_is_not_writable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_initialize(_layout: DataLayout) -> None:
        raise OSError("synthetic permission failure")

    monkeypatch.setattr(DataLayout, "initialize", fail_initialize)
    app = create_executable_app(
        Settings(
            data_root=tmp_path,
            provider_backend="openai",
            openai_api_key="test-key",
            workbench_enabled=False,
            _env_file=None,
        )
    )

    with TestClient(app) as client:
        liveness = client.get("/healthz")
        readiness = client.get("/readyz")

    assert liveness.status_code == 200
    assert readiness.status_code == 503
    assert {
        "name": "storage",
        "ready": False,
        "reason": "storage_not_writable",
    } in readiness.json()["components"]


def test_healthz_stays_live_when_otlp_configuration_is_incomplete(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            data_root=tmp_path,
            telemetry_exporter="otlp",
            telemetry_otlp_traces_endpoint=None,
            workbench_enabled=False,
            _env_file=None,
        )
    )

    with TestClient(app) as client:
        liveness = client.get("/healthz")
        readiness = client.get("/readyz")

    assert liveness.status_code == 200
    assert readiness.status_code == 503
    assert {
        "name": "telemetry",
        "ready": False,
        "reason": "telemetry_otlp_endpoint_missing",
    } in readiness.json()["components"]


def test_healthz_stays_live_when_telemetry_resource_identity_is_unsafe(
    tmp_path: Path,
) -> None:
    app = create_app(
        Settings(
            data_root=tmp_path,
            telemetry_exporter="console",
            service_name="unsafe service name",
            workbench_enabled=False,
            _env_file=None,
        )
    )

    with TestClient(app) as client:
        liveness = client.get("/healthz")
        readiness = client.get("/readyz")

    assert liveness.status_code == 200
    assert readiness.status_code == 503
    assert {
        "name": "telemetry",
        "ready": False,
        "reason": "telemetry_resource_identity_invalid",
    } in readiness.json()["components"]


def test_configured_executable_still_fails_fast_for_competing_writer(
    tmp_path: Path,
) -> None:
    layout = DataLayout.from_root(tmp_path)
    layout.initialize()
    owner = DataRootWriterLock(layout.writer_lock)
    owner.acquire()
    try:
        with pytest.raises(DataRootWriterLockError, match="data_root_writer_locked"):
            create_executable_app(
                Settings(
                    data_root=tmp_path,
                    provider_backend="openai",
                    openai_api_key="test-key",
                    workbench_enabled=False,
                    _env_file=None,
                )
            )
    finally:
        owner.release()
