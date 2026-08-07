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
        workbench_config = client.get("/assistant/config")
        health = client.get("/healthz")
        api = client.get("/api/v1/documents")

    assert workbench.status_code == 200
    assert "RAG Assistant Workbench" in workbench.text
    assert workbench_config.status_code == 200
    assert workbench_config.json()["footer_links"] == []
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
        "Compare / 对比",
        "Retrieval Comparison / 检索策略对比",
        "Model Comparison / 模型对比",
        "Quality Analysis / 质量分析",
        "Performance & Cost / 性能与成本",
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
        "Executive KPI cards / 核心 KPI 卡片",
        "Registered experiment plan / 已注册实验计划",
        "Vector, hybrid, and reranker comparison / 向量、混合与重排对比",
        "Generation model quality, latency, and cost / 生成模型质量、延迟和成本",
        "Five independent quality metrics / 五项独立质量指标",
        "Quality visualization / 质量可视化",
        "Latency distribution / 延迟分布",
        "Token and cost breakdown / Token 与成本明细",
        "Cache statistics / 缓存统计",
        "Refusal and compliance statistics / 拒答与合规统计",
        "System evidence / 系统证据",
        "Compact baseline-delta plot / 紧凑基线差值图",
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
        "Start comparison / 启动对比",
        "Refresh health / 刷新健康状态",
    }
    assert api_names >= {
        "chat_submit",
        "documents_upload",
        "evaluation_start",
        "comparison_start",
        "diagnostics_request",
    }
