from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_mvp.api.app import create_app
from rag_mvp.config.settings import Settings
from rag_mvp.ui.services import WorkbenchServices
from rag_mvp.ui.workbench import create_workbench

pytestmark = pytest.mark.ui


def _settings(root: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_root=root,
        workbench_enabled=True,
        workbench_path="/assistant",
    )


def test_workbench_mount_coexists_with_health_and_api_routes(tmp_path: Path) -> None:
    app = create_app(
        _settings(tmp_path),
        workbench_services=WorkbenchServices(),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        workbench = client.get("/assistant/")
        health = client.get("/healthz")
        api = client.get("/api/v1/documents")

    assert workbench.status_code == 200
    assert "RAG Assistant Workbench" in workbench.text
    assert health.status_code == 200
    assert health.json()["status"] == "alive"
    assert api.status_code == 503
    assert api.json() == {"error": {"code": "ingestion_unavailable"}}


def test_workbench_exposes_four_primary_tabs_and_bilingual_controls(tmp_path: Path) -> None:
    blocks = create_workbench(_settings(tmp_path), WorkbenchServices())
    config = blocks.get_config_file()
    components = config["components"]

    tabs = [
        component["props"]["label"] for component in components if component["type"] == "tabitem"
    ]
    labels = {component["props"].get("label") for component in components if component.get("props")}
    button_values = {
        component["props"].get("value") for component in components if component["type"] == "button"
    }
    api_names = {
        dependency["api_name"]
        for dependency in config["dependencies"]
        if dependency.get("api_name")
    }

    assert tabs == [
        "Chat",
        "Documents",
        "Evaluation",
        "Run / 运行",
        "Overview / 结果总览",
        "Operations / 运维",
        "Artifacts / 报告下载",
        "Diagnostics",
    ]
    assert sum(component["type"] == "chatbot" for component in components) == 1
    assert labels >= {
        "Question / 问题",
        "Retrieval mode / 检索模式",
        "Active documents / 活跃文档",
        "Run history / 运行历史",
        "Canonical operations measures / 规范运维指标",
        "TXT/CSV download status / TXT/CSV 下载状态",
        "Same-origin API downloads / 同源 API 下载",
        "Request trace / 请求跟踪",
    }
    assert button_values >= {
        "Ask / 提问",
        "Reset / 重置",
        "Cancel / 取消",
        "Upload and index / 上传并索引",
        "Start evaluation / 启动评估",
        "Refresh health / 刷新健康状态",
    }
    assert api_names >= {
        "chat_submit",
        "documents_upload",
        "evaluation_start",
        "diagnostics_request",
    }
