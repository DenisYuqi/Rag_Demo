"""Gradio composition and FastAPI mounting for the four-view workbench."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

import gradio as gr
from fastapi import FastAPI

from rag_mvp.config.settings import Settings

from .callbacks import WorkbenchCallbacks
from .models import (
    BrowserSessionState,
    ChatRender,
    ComparisonRender,
    DiagnosticsRender,
    DocumentsRender,
    EvaluationRender,
)
from .services import WorkbenchServices

_WORKBENCH_CSS = """
.rag-kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 12px;
  margin: 4px 0 16px;
}
.rag-kpi-card {
  border: 1px solid var(--border-color-primary);
  border-radius: 12px;
  background: var(--background-fill-secondary);
  padding: 14px 16px;
  min-height: 116px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.rag-kpi-card span, .rag-kpi-card small { color: var(--body-text-color-subdued); }
.rag-kpi-card strong { font-size: 1.75rem; line-height: 1.1; }
.rag-kpi-passed { border-color: #16a34a; }
.rag-kpi-failed { border-color: #dc2626; }
.rag-kpi-unavailable { border-style: dashed; }
"""

def _chat_outputs(
    render: ChatRender,
) -> tuple[list[dict[str, str]], BrowserSessionState, str, str, str, str]:
    return (
        list(render.history),
        render.state,
        render.citations_markdown,
        render.previews_markdown,
        render.status_markdown,
        "",
    )


def _document_outputs(
    render: DocumentsRender,
) -> tuple[list[list[Any]], list[list[Any]], str]:
    return (
        [list(row) for row in render.document_rows],
        [list(row) for row in render.job_rows],
        render.status_markdown,
    )


def _evaluation_outputs(
    render: EvaluationRender,
) -> tuple[Any, ...]:
    return (
        gr.update(
            choices=list(render.dataset_choices),
            value=render.selected_dataset_key,
            interactive=bool(render.dataset_choices),
        ),
        gr.update(
            choices=list(render.plan_choices),
            value=render.selected_plan_key,
            interactive=bool(render.plan_choices),
        ),
        gr.update(
            choices=list(render.run_choices),
            value=render.selected_run_id,
            interactive=bool(render.run_choices),
        ),
        [list(row) for row in render.identity_rows],
        [list(row) for row in render.plan_rows],
        [list(row) for row in render.run_rows],
        render.progress_markdown,
        render.gate_markdown,
        render.kpi_html,
        [list(row) for row in render.overview_rows],
        [list(row) for row in render.quality_rows],
        _quality_plot(render),
        [list(row) for row in render.category_rows],
        [list(row) for row in render.failure_rows],
        [list(row) for row in render.performance_rows],
        _latency_plot(render),
        [list(row) for row in render.cost_rows],
        [list(row) for row in render.cache_rows],
        [list(row) for row in render.refusal_rows],
        [list(row) for row in render.system_rows],
        [list(row) for row in render.operations_rows],
        render.operations_preview,
        render.operations_links_markdown,
        [list(row) for row in render.artifact_rows],
        render.artifact_links_markdown,
        render.status_markdown,
        gr.update(interactive=render.start_enabled),
        gr.update(active=render.poll_active),
        render.state,
    )


def _diagnostic_outputs(
    render: DiagnosticsRender,
) -> tuple[list[list[Any]], list[list[Any]], str]:
    return (
        [list(row) for row in render.health_rows],
        [list(row) for row in render.request_rows],
        render.status_markdown,
    )


def _comparison_plot(render: ComparisonRender) -> dict[str, object]:
    """Build Gradio's native plot payload without a transitive pandas import."""

    return {
        "columns": ["candidate", "metric", "delta"],
        "data": [list(row) for row in render.plot_rows],
        "datatypes": {
            "candidate": "nominal",
            "metric": "nominal",
            "delta": "quantitative",
        },
        "mark": "bar",
    }


def _focused_comparison_plot(rows: Sequence[tuple[str, str, float]]) -> dict[str, object]:
    return {
        "columns": ["candidate", "metric", "delta"],
        "data": [list(row) for row in rows],
        "datatypes": {
            "candidate": "nominal",
            "metric": "nominal",
            "delta": "quantitative",
        },
        "mark": "bar",
    }


def _quality_plot(render: EvaluationRender) -> dict[str, object]:
    return {
        "columns": ["metric", "score"],
        "data": [list(row) for row in render.quality_plot_rows],
        "datatypes": {"metric": "nominal", "score": "quantitative"},
        "mark": "bar",
    }


def _latency_plot(render: EvaluationRender) -> dict[str, object]:
    return {
        "columns": ["scope", "percentile", "latency_ms"],
        "data": [list(row) for row in render.latency_plot_rows],
        "datatypes": {
            "scope": "nominal",
            "percentile": "quantitative",
            "latency_ms": "quantitative",
        },
        "mark": "line",
    }


def _comparison_outputs(render: ComparisonRender) -> tuple[Any, ...]:
    return (
        gr.update(
            choices=list(render.plan_choices),
            value=render.selected_plan_id,
            interactive=bool(render.plan_choices),
        ),
        gr.update(
            choices=list(render.comparison_choices),
            value=render.selected_comparison_id,
            interactive=bool(render.comparison_choices),
        ),
        [list(row) for row in render.plan_rows],
        [list(row) for row in render.history_rows],
        render.progress_markdown,
        render.gate_markdown,
        [list(row) for row in render.retrieval_rows],
        _focused_comparison_plot(render.retrieval_plot_rows),
        render.retrieval_recommendation_markdown,
        [list(row) for row in render.model_rows],
        _focused_comparison_plot(render.model_plot_rows),
        render.model_recommendation_markdown,
        [list(row) for row in render.controlled_rows],
        [list(row) for row in render.shared_setup_rows],
        [list(row) for row in render.comparison_metric_rows],
        render.cache_conclusion_markdown,
        [list(row) for row in render.candidate_rows],
        _comparison_plot(render),
        [list(row) for row in render.category_rows],
        render.recommendation_markdown,
        [list(row) for row in render.artifact_rows],
        render.artifact_links_markdown,
        render.status_markdown,
        gr.update(interactive=render.start_enabled),
        gr.update(active=render.poll_active),
        render.state,
    )


def create_workbench(
    settings: Settings,
    services: WorkbenchServices,
    *,
    callbacks: WorkbenchCallbacks | None = None,
) -> gr.Blocks:
    """Create one workbench using the same in-process services as the HTTP API."""

    controller = callbacks or WorkbenchCallbacks(services)
    initial_documents = controller.refresh_documents(services.default_retrieval_profile)
    initial_evaluation = controller.refresh_evaluations(
        None,
        profile_id=services.default_retrieval_profile,
    )
    initial_comparison = controller.refresh_comparisons(
        None,
        profile_id=services.default_retrieval_profile,
    )
    with gr.Blocks(title="RAG Assistant Workbench") as demo:
        # Gradio deep-copies the state factory.  A controller-bound method would
        # recursively copy the production service graph, including non-copyable
        # locks and HTTP clients.  The model factory is stateless and still gives
        # every browser session a distinct owner identifier.
        session_state = gr.State(value=BrowserSessionState.create)
        retrieval_profile = gr.Dropdown(
            choices=list(services.retrieval_profile_ids),
            value=services.default_retrieval_profile,
            label="Retrieval profile / 检索模型",
            interactive=len(services.retrieval_profile_ids) > 1,
        )
        gr.Markdown("# RAG Assistant Workbench / RAG 助手工作台")

        with gr.Tabs(selected="chat-tab"):
            with gr.Tab("Chat", id="chat-tab"):
                chatbot = gr.Chatbot(label="Conversation / 对话")
                question = gr.Textbox(label="Question / 问题", lines=3)
                mode = gr.Radio(
                    choices=["dense", "hybrid", "hybrid-rerank"],
                    value=settings.default_retrieval_mode,
                    label="Retrieval mode / 检索模式",
                )
                with gr.Row():
                    ask = gr.Button("Ask / 提问", variant="primary")
                    reset = gr.Button("Reset / 重置")
                    cancel = gr.Button("Cancel / 取消", variant="stop")
                chat_status = gr.Markdown(label="Chat status / 对话状态")
                citations = gr.Markdown(label="Citations / 引用")
                previews = gr.Markdown(label="Source previews / 来源预览")

            with gr.Tab("Documents", id="documents-tab"):
                upload_file = gr.File(
                    label="Upload document / 上传文档",
                    file_types=[".pdf", ".md", ".markdown", ".txt"],
                    type="filepath",
                )
                source_key = gr.Textbox(label="Source key / 来源键")
                display_title = gr.Textbox(label="Display title / 显示标题")
                with gr.Row():
                    upload = gr.Button("Upload and index / 上传并索引", variant="primary")
                    reindex = gr.Button("Reindex / 重建索引")
                    refresh_documents = gr.Button("Refresh / 刷新")
                documents_table = gr.Dataframe(
                    headers=["source_id", "title", "version", "type", "media_type"],
                    value=[list(row) for row in initial_documents.document_rows],
                    label="Active documents / 活跃文档",
                    interactive=False,
                    type="array",
                )
                jobs_table = gr.Dataframe(
                    headers=[
                        "job_id",
                        "operation",
                        "status",
                        "stage",
                        "source_id",
                        "version",
                        "ocr_pages",
                        "chunks",
                        "revision",
                    ],
                    value=[list(row) for row in initial_documents.job_rows],
                    label="Ingestion progress / 摄取进度",
                    interactive=False,
                    type="array",
                )
                delete_source = gr.Textbox(label="Source ID to delete / 待删除来源 ID")
                confirm_delete = gr.Checkbox(label="Confirm deletion / 确认删除")
                delete = gr.Button("Delete / 删除", variant="stop")
                document_status = gr.Markdown(
                    value=initial_documents.status_markdown,
                    label="Document status / 文档状态",
                )

            with gr.Tab("Evaluation", id="evaluation-tab"):
                evaluation_timer = gr.Timer(
                    value=2.0,
                    active=initial_evaluation.poll_active,
                )
                with gr.Tabs(selected="evaluation-run-tab"):
                    with gr.Tab("Run / 运行", id="evaluation-run-tab"):
                        with gr.Row():
                            dataset_select = gr.Dropdown(
                                choices=list(initial_evaluation.dataset_choices),
                                value=initial_evaluation.selected_dataset_key,
                                label="Registered dataset / 已注册数据集",
                                interactive=bool(initial_evaluation.dataset_choices),
                            )
                            plan_select = gr.Dropdown(
                                choices=list(initial_evaluation.plan_choices),
                                value=initial_evaluation.selected_plan_key,
                                label="Registered run type / 已注册运行类型",
                                interactive=bool(initial_evaluation.plan_choices),
                            )
                        identity_table = gr.Dataframe(
                            headers=["identity", "immutable value"],
                            value=[list(row) for row in initial_evaluation.identity_rows],
                            label="Immutable identity / 不可变身份",
                            interactive=False,
                            type="array",
                        )
                        plan_table = gr.Dataframe(
                            headers=[
                                "plan_id",
                                "type",
                                "cases",
                                "candidates",
                                "max_logical_calls",
                                "max_provider_calls",
                                "cache_policy",
                                "cost_estimate",
                                "cost_cap",
                                "max_active_jobs",
                            ],
                            value=[list(row) for row in initial_evaluation.plan_rows],
                            label="Plan preview / 计划预览",
                            interactive=False,
                            type="array",
                        )
                        with gr.Row():
                            start_evaluation = gr.Button(
                                "Start evaluation / 启动评估",
                                variant="primary",
                                interactive=initial_evaluation.start_enabled,
                            )
                            refresh_evaluation = gr.Button(
                                "Refresh persisted evidence / 刷新持久化证据"
                            )
                        run_select = gr.Dropdown(
                            choices=list(initial_evaluation.run_choices),
                            value=initial_evaluation.selected_run_id,
                            label="Persisted run history / 持久化运行历史",
                            interactive=bool(initial_evaluation.run_choices),
                        )
                        runs_table = gr.Dataframe(
                            headers=[
                                "run_id",
                                "type",
                                "status",
                                "completed",
                                "failed",
                                "remaining",
                                "total",
                                "dataset",
                                "version",
                                "corpus",
                                "configuration",
                                "started_at",
                                "completed_at",
                                "gate",
                            ],
                            value=[list(row) for row in initial_evaluation.run_rows],
                            label="Run history / 运行历史",
                            interactive=False,
                            type="array",
                        )
                        evaluation_progress = gr.Markdown(
                            initial_evaluation.progress_markdown,
                            label="Background progress / 后台进度",
                        )
                        evaluation_status = gr.Markdown(
                            initial_evaluation.status_markdown,
                            label="Evaluation status / 评估状态",
                        )

                    with gr.Tab("Overview / 结果总览", id="evaluation-overview-tab"):
                        dashboard_kpis = gr.HTML(
                            initial_evaluation.kpi_html,
                            label="Executive KPI cards / 核心 KPI 卡片",
                        )
                        gate_banner = gr.Markdown(
                            initial_evaluation.gate_markdown,
                            label="Overall gate / 总体门槛",
                        )
                        overview_table = gr.Dataframe(
                            headers=[
                                "metric",
                                "value",
                                "unit",
                                "threshold",
                                "numerator",
                                "denominator",
                                "state",
                                "scorer",
                            ],
                            value=[list(row) for row in initial_evaluation.overview_rows],
                            label="Quality, performance, cost / 质量、性能和成本",
                            interactive=False,
                            type="array",
                        )
                    with gr.Tab("Compare / 对比", id="evaluation-compare-tab"):
                        comparison_timer = gr.Timer(
                            value=2.0,
                            active=initial_comparison.poll_active,
                        )
                        comparison_plan_select = gr.Dropdown(
                            choices=list(initial_comparison.plan_choices),
                            value=initial_comparison.selected_plan_id,
                            label="Registered experiment plan / 已注册实验计划",
                            interactive=bool(initial_comparison.plan_choices),
                        )
                        comparison_plan_table = gr.Dataframe(
                            headers=[
                                "plan_id",
                                "axis",
                                "dataset",
                                "version",
                                "candidates",
                                "baseline",
                                "cases",
                                "repeats",
                                "max_logical_calls",
                                "max_provider_calls",
                                "cache_policy",
                                "cost_estimate",
                                "cost_cap",
                                "launchable",
                                "blocking_codes",
                                "corpus",
                                "corpus_version",
                                "candidate_configurations",
                            ],
                            value=[list(row) for row in initial_comparison.plan_rows],
                            label="Immutable comparison plan / 不可变对比计划",
                            interactive=False,
                            type="array",
                        )
                        with gr.Row():
                            start_comparison = gr.Button(
                                "Start comparison / 启动对比",
                                variant="primary",
                                interactive=initial_comparison.start_enabled,
                            )
                            refresh_comparison = gr.Button(
                                "Refresh comparison evidence / 刷新对比证据"
                            )
                        comparison_select = gr.Dropdown(
                            choices=list(initial_comparison.comparison_choices),
                            value=initial_comparison.selected_comparison_id,
                            label="Persisted comparison history / 持久化对比历史",
                            interactive=bool(initial_comparison.comparison_choices),
                        )
                        comparison_history_table = gr.Dataframe(
                            headers=[
                                "comparison_id",
                                "plan_id",
                                "status",
                                "completed",
                                "failed",
                                "remaining",
                                "total",
                                "started_at",
                                "completed_at",
                                "safe_error",
                                "dataset",
                                "dataset_version",
                                "corpus",
                                "corpus_version",
                                "candidate_configurations",
                                "active",
                                "completed_cases",
                                "failed_cases",
                                "provider_calls",
                                "incurred_cost",
                                "evidence",
                                "gate",
                            ],
                            value=[list(row) for row in initial_comparison.history_rows],
                            label="Comparison status and progress / 对比状态和进度",
                            interactive=False,
                            type="array",
                        )
                        comparison_progress = gr.Markdown(
                            initial_comparison.progress_markdown,
                            label="Background comparison progress / 后台对比进度",
                        )
                        comparison_gate = gr.Markdown(
                            initial_comparison.gate_markdown,
                            label="Comparison gate / 对比门槛",
                        )
                        with gr.Tabs(selected="retrieval-comparison-view"):
                            with gr.Tab(
                                "Retrieval Comparison / 检索策略对比",
                                id="retrieval-comparison-view",
                            ):
                                retrieval_comparison_table = gr.Dataframe(
                                    headers=[
                                        "candidate_id",
                                        "candidate",
                                        "strategy",
                                        "status",
                                        "baseline",
                                        "faithfulness",
                                        "context_precision",
                                        "answer_compliance",
                                        "p95_latency",
                                        "cost_per_1000",
                                        "error_rate",
                                        "degradation_rate",
                                        "gate",
                                        "selected",
                                    ],
                                    value=[list(row) for row in initial_comparison.retrieval_rows],
                                    label=(
                                        "Vector, hybrid, and reranker comparison / "
                                        "向量、混合与重排对比"
                                    ),
                                    interactive=False,
                                    type="array",
                                )
                                retrieval_comparison_plot = gr.BarPlot(
                                    value=cast(
                                        Any,
                                        _focused_comparison_plot(
                                            initial_comparison.retrieval_plot_rows
                                        ),
                                    ),
                                    x="metric",
                                    y="delta",
                                    color="candidate",
                                    title="Retrieval baseline deltas",
                                    x_title="Metric / 指标",
                                    y_title="Baseline delta / 基线差值",
                                    height=300,
                                    label="Retrieval comparison plot / 检索对比图",
                                )
                                retrieval_recommendation = gr.Markdown(
                                    initial_comparison.retrieval_recommendation_markdown,
                                    label="Retrieval recommendation / 检索推荐",
                                )
                            with gr.Tab(
                                "Model Comparison / 模型对比",
                                id="model-comparison-view",
                            ):
                                model_comparison_table = gr.Dataframe(
                                    headers=[
                                        "candidate_id",
                                        "candidate",
                                        "model",
                                        "status",
                                        "baseline",
                                        "faithfulness",
                                        "context_precision",
                                        "answer_compliance",
                                        "p95_latency",
                                        "cost_per_1000",
                                        "error_rate",
                                        "degradation_rate",
                                        "gate",
                                        "selected",
                                    ],
                                    value=[list(row) for row in initial_comparison.model_rows],
                                    label=(
                                        "Generation model quality, latency, and cost / "
                                        "生成模型质量、延迟和成本"
                                    ),
                                    interactive=False,
                                    type="array",
                                )
                                model_comparison_plot = gr.BarPlot(
                                    value=cast(
                                        Any,
                                        _focused_comparison_plot(
                                            initial_comparison.model_plot_rows
                                        ),
                                    ),
                                    x="metric",
                                    y="delta",
                                    color="candidate",
                                    title="Model baseline deltas",
                                    x_title="Metric / 指标",
                                    y_title="Baseline delta / 基线差值",
                                    height=300,
                                    label="Model comparison plot / 模型对比图",
                                )
                                model_recommendation = gr.Markdown(
                                    initial_comparison.model_recommendation_markdown,
                                    label="Model recommendation / 模型推荐",
                                )
                        with gr.Accordion(
                            "Detailed comparison evidence / 详细对比证据", open=False
                        ):
                            gr.Markdown(
                                "All values below remain the authoritative, "
                                "denominator-bearing evidence. / "
                                "以下数值保留为带分母的权威证据。"
                            )
                        controlled_dimensions = gr.Dataframe(
                            headers=["controlled dimension", "fixed value or state"],
                            value=[list(row) for row in initial_comparison.controlled_rows],
                            label="Controlled dimensions and compatibility / 控制维度和兼容性",
                            interactive=False,
                            type="array",
                        )
                        comparison_shared_setup = gr.Dataframe(
                            headers=["scope", "field", "value or unavailable reason"],
                            value=[list(row) for row in initial_comparison.shared_setup_rows],
                            label="Shared setup and inclusive totals / 共享设置与包含总计",
                            interactive=False,
                            type="array",
                        )
                        comparison_level_metrics = gr.Dataframe(
                            headers=[
                                "metric",
                                "value",
                                "unit",
                                "numerator",
                                "denominator",
                                "state",
                                "gate",
                            ],
                            value=[list(row) for row in initial_comparison.comparison_metric_rows],
                            label="Comparison-level evidence / 对比级证据",
                            interactive=False,
                            type="array",
                        )
                        cache_conclusion = gr.Markdown(
                            initial_comparison.cache_conclusion_markdown,
                            label="Cache revision and equivalence / 缓存版本和等价性",
                        )
                        comparison_candidates = gr.Dataframe(
                            headers=[
                                "candidate_id",
                                "candidate",
                                "status",
                                "baseline",
                                "metric",
                                "absolute_value",
                                "unit",
                                "numerator",
                                "denominator",
                                "gate",
                                "baseline_delta",
                                "safe_error",
                                "axis_value",
                                "configuration_id",
                                "evaluation_run_id",
                                "evidence_status",
                                "failed_cases",
                                "provider_calls",
                                "known_partial_cost",
                                "total_cost",
                                "cost_complete",
                                "cost_unknown_reasons",
                            ],
                            value=[list(row) for row in initial_comparison.candidate_rows],
                            label="Authoritative candidate and delta table / 权威候选和差值表",
                            interactive=False,
                            type="array",
                        )
                        comparison_plot = gr.BarPlot(
                            value=cast(Any, _comparison_plot(initial_comparison)),
                            x="metric",
                            y="delta",
                            color="candidate",
                            title="Baseline deltas in metric-native units",
                            x_title="Metric / 指标",
                            y_title="Baseline delta / 基线差值",
                            height=320,
                            label="Compact baseline-delta plot / 紧凑基线差值图",
                        )
                        comparison_categories = gr.Dataframe(
                            headers=[
                                "candidate_id",
                                "category",
                                "cases",
                                "metric",
                                "value",
                                "unit",
                                "denominator",
                                "baseline_delta",
                                "state",
                            ],
                            value=[list(row) for row in initial_comparison.category_rows],
                            label="Challenge-category drill-down / 挑战分类下钻",
                            interactive=False,
                            type="array",
                        )
                        comparison_recommendation = gr.Markdown(
                            initial_comparison.recommendation_markdown,
                            label="Measured recommendation rationale / 测量推荐理由",
                        )
                        comparison_artifacts = gr.Dataframe(
                            headers=[
                                "artifact_id",
                                "format",
                                "schema",
                                "media_type",
                                "digest",
                                "bytes",
                                "created_at",
                            ],
                            value=[list(row) for row in initial_comparison.artifact_rows],
                            label="Comparison artifact manifest / 对比制品清单",
                            interactive=False,
                            type="array",
                        )
                        comparison_artifact_links = gr.Markdown(
                            initial_comparison.artifact_links_markdown,
                            label="Validated comparison downloads / 已验证对比下载",
                        )
                        comparison_status = gr.Markdown(
                            initial_comparison.status_markdown,
                            label="Comparison status / 对比状态",
                        )

                    with gr.Tab("Quality Analysis / 质量分析", id="evaluation-quality-tab"):
                        quality_table = gr.Dataframe(
                            headers=[
                                "metric",
                                "value",
                                "unit",
                                "threshold",
                                "numerator",
                                "denominator",
                                "state",
                                "scorer",
                            ],
                            value=[list(row) for row in initial_evaluation.quality_rows],
                            label="Five independent quality metrics / 五项独立质量指标",
                            interactive=False,
                            type="array",
                        )
                        quality_plot = gr.BarPlot(
                            value=cast(Any, _quality_plot(initial_evaluation)),
                            x="metric",
                            y="score",
                            title="Quality metrics (percent)",
                            x_title="Metric / 指标",
                            y_title="Score / 得分 (%)",
                            height=320,
                            label="Quality visualization / 质量可视化",
                        )
                        category_table = gr.Dataframe(
                            headers=[
                                "category",
                                "cases",
                                "metric",
                                "value",
                                "denominator",
                                "state",
                            ],
                            value=[list(row) for row in initial_evaluation.category_rows],
                            label="Category results / 分类结果",
                            interactive=False,
                            type="array",
                        )
                        failures_table = gr.Dataframe(
                            headers=[
                                "case_id",
                                "safe_error",
                                "tags",
                                "metric_contributions",
                                "refusal_reason",
                                "citation_ids",
                                "request_id",
                                "trace_id",
                                "outcome",
                            ],
                            value=[list(row) for row in initial_evaluation.failure_rows],
                            label="Privacy-safe failed cases / 隐私安全的失败用例",
                            interactive=False,
                            type="array",
                        )

                    with gr.Tab(
                        "Performance & Cost / 性能与成本",
                        id="evaluation-performance-tab",
                    ):
                        performance_table = gr.Dataframe(
                            headers=[
                                "metric",
                                "value",
                                "unit",
                                "threshold",
                                "numerator",
                                "denominator",
                                "state",
                                "scorer",
                            ],
                            value=[list(row) for row in initial_evaluation.performance_rows],
                            label="Latency, success, and concurrency / 延迟、成功率和并发",
                            interactive=False,
                            type="array",
                        )
                        latency_plot = gr.LinePlot(
                            value=cast(Any, _latency_plot(initial_evaluation)),
                            x="percentile",
                            y="latency_ms",
                            color="scope",
                            title="Latency percentiles",
                            x_title="Percentile / 分位数",
                            y_title="Latency / 延迟 (ms)",
                            height=320,
                            label="Latency distribution / 延迟分布",
                        )
                        cost_table = gr.Dataframe(
                            headers=[
                                "metric",
                                "value",
                                "unit",
                                "threshold",
                                "numerator",
                                "denominator",
                                "state",
                                "scorer",
                            ],
                            value=[list(row) for row in initial_evaluation.cost_rows],
                            label="Token and cost breakdown / Token 与成本明细",
                            interactive=False,
                            type="array",
                        )

                    with gr.Tab("Operations / 运维", id="evaluation-operations-tab"):
                        with gr.Row():
                            cache_table = gr.Dataframe(
                                headers=[
                                    "metric",
                                    "value",
                                    "unit",
                                    "threshold",
                                    "numerator",
                                    "denominator",
                                    "state",
                                    "scorer",
                                ],
                                value=[list(row) for row in initial_evaluation.cache_rows],
                                label="Cache statistics / 缓存统计",
                                interactive=False,
                                type="array",
                            )
                            refusal_table = gr.Dataframe(
                                headers=[
                                    "metric",
                                    "value",
                                    "unit",
                                    "threshold",
                                    "numerator",
                                    "denominator",
                                    "state",
                                    "scorer",
                                ],
                                value=[list(row) for row in initial_evaluation.refusal_rows],
                                label="Refusal and compliance statistics / 拒答与合规统计",
                                interactive=False,
                                type="array",
                            )
                        system_table = gr.Dataframe(
                            headers=["metric", "value", "unit", "state", "evidence"],
                            value=[list(row) for row in initial_evaluation.system_rows],
                            label="System evidence / 系统证据",
                            interactive=False,
                            type="array",
                        )
                        operations_table = gr.Dataframe(
                            headers=[
                                "metric",
                                "value",
                                "unit",
                                "threshold",
                                "numerator",
                                "denominator",
                                "state",
                                "scorer",
                            ],
                            value=[list(row) for row in initial_evaluation.operations_rows],
                            label="Canonical operations measures / 规范运维指标",
                            interactive=False,
                            type="array",
                        )
                        operations_preview = gr.Textbox(
                            value=initial_evaluation.operations_preview,
                            label="Validated text preview / 已验证文本预览",
                            lines=14,
                            interactive=False,
                        )
                        operations_links = gr.Markdown(
                            initial_evaluation.operations_links_markdown,
                            label="TXT/CSV download status / TXT/CSV 下载状态",
                        )

                    with gr.Tab("Artifacts / 报告下载", id="evaluation-artifacts-tab"):
                        artifacts_table = gr.Dataframe(
                            headers=[
                                "artifact_id",
                                "format",
                                "schema",
                                "media_type",
                                "digest",
                                "bytes",
                                "created_at",
                            ],
                            value=[list(row) for row in initial_evaluation.artifact_rows],
                            label="Validated artifact manifest / 已验证制品清单",
                            interactive=False,
                            type="array",
                        )
                        artifact_links = gr.Markdown(
                            initial_evaluation.artifact_links_markdown,
                            label="Same-origin API downloads / 同源 API 下载",
                        )

            with gr.Tab("Diagnostics", id="diagnostics-tab"):
                refresh_health = gr.Button("Refresh health / 刷新健康状态")
                health_table = gr.Dataframe(
                    headers=["component", "ready", "reason"],
                    label="Health / 健康状态",
                    interactive=False,
                    type="array",
                )
                request_id = gr.Textbox(label="Request or trace ID / 请求或跟踪 ID")
                inspect = gr.Button("Inspect request / 检查请求")
                request_table = gr.Dataframe(
                    headers=["field", "safe value"],
                    label="Request trace / 请求跟踪",
                    interactive=False,
                    type="array",
                )
                diagnostics_status = gr.Markdown(label="Diagnostics status / 诊断状态")

        async def on_ask(
            raw_question: str,
            raw_mode: str,
            raw_history: Sequence[Mapping[str, object]] | None,
            raw_state: BrowserSessionState | None,
            raw_profile: str | None,
        ) -> tuple[list[dict[str, str]], BrowserSessionState, str, str, str, str]:
            return _chat_outputs(
                await controller.submit_chat(
                    raw_question,
                    raw_mode,
                    raw_history,
                    raw_state,
                    raw_profile,
                )
            )

        def on_reset(
            raw_state: BrowserSessionState | None,
            raw_profile: str | None = None,
        ) -> tuple[list[dict[str, str]], BrowserSessionState, str, str, str, str]:
            return _chat_outputs(controller.reset_chat(raw_state, raw_profile))

        def on_cancel(
            raw_history: Sequence[Mapping[str, object]] | None,
            raw_state: BrowserSessionState | None,
        ) -> tuple[list[dict[str, str]], BrowserSessionState, str, str, str, str]:
            return _chat_outputs(controller.cancel_chat(raw_history, raw_state))

        def on_documents(
            raw_profile: str | None = None,
        ) -> tuple[list[list[Any]], list[list[Any]], str]:
            return _document_outputs(controller.refresh_documents(raw_profile))

        async def on_upload(
            path: str | None,
            key: str,
            title: str,
            raw_profile: str | None,
        ) -> tuple[list[list[Any]], list[list[Any]], str]:
            return _document_outputs(
                await controller.upload_document(path, key, title, raw_profile)
            )

        async def on_reindex(
            raw_profile: str | None,
        ) -> tuple[list[list[Any]], list[list[Any]], str]:
            return _document_outputs(await controller.reindex_documents(raw_profile))

        async def on_delete(
            source: str,
            confirmed: bool,
            raw_profile: str | None,
        ) -> tuple[list[list[Any]], list[list[Any]], str]:
            return _document_outputs(
                await controller.delete_document(source, confirmed, raw_profile)
            )

        async def on_start_evaluation(
            selected_dataset: str | None,
            selected_plan: str | None,
            raw_state: BrowserSessionState | None,
            raw_profile: str | None,
        ) -> tuple[Any, ...]:
            return _evaluation_outputs(
                await controller.start_registered_evaluation(
                    selected_dataset,
                    selected_plan,
                    raw_state,
                    raw_profile,
                )
            )

        def on_refresh_evaluation(
            selected_dataset: str | None,
            selected_plan: str | None,
            selected_run: str | None,
            raw_state: BrowserSessionState | None,
            raw_profile: str | None,
        ) -> tuple[Any, ...]:
            return _evaluation_outputs(
                controller.preview_evaluation_plan(
                    selected_dataset,
                    selected_plan,
                    selected_run,
                    raw_state,
                    raw_profile,
                )
            )

        async def on_start_comparison(
            selected_plan: str | None,
            raw_state: BrowserSessionState | None,
            raw_profile: str | None,
        ) -> tuple[Any, ...]:
            return _comparison_outputs(
                await controller.start_registered_comparison(
                    selected_plan,
                    raw_state,
                    raw_profile,
                )
            )

        def on_refresh_comparison(
            selected_plan: str | None,
            selected_comparison: str | None,
            raw_state: BrowserSessionState | None,
            raw_profile: str | None,
        ) -> tuple[Any, ...]:
            return _comparison_outputs(
                controller.preview_comparison(
                    selected_plan,
                    selected_comparison,
                    raw_state,
                    raw_profile,
                )
            )

        def on_health() -> tuple[list[list[Any]], list[list[Any]], str]:
            return _diagnostic_outputs(controller.refresh_health())

        def on_inspect(value: str) -> tuple[list[list[Any]], list[list[Any]], str]:
            return _diagnostic_outputs(controller.inspect_request(value))

        chat_outputs = [chatbot, session_state, citations, previews, chat_status, question]
        ask_event = ask.click(
            on_ask,
            inputs=[question, mode, chatbot, session_state, retrieval_profile],
            outputs=chat_outputs,
            api_name="chat_submit",
        )
        question.submit(
            on_ask,
            inputs=[question, mode, chatbot, session_state, retrieval_profile],
            outputs=chat_outputs,
            api_name=None,
        )
        reset.click(
            on_reset,
            inputs=[session_state, retrieval_profile],
            outputs=chat_outputs,
            api_name="chat_reset",
        )
        cancel.click(
            on_cancel,
            inputs=[chatbot, session_state],
            outputs=chat_outputs,
            cancels=[ask_event],
            api_name="chat_cancel",
        )
        chat_load = demo.load(
            on_reset,
            inputs=[session_state],
            outputs=chat_outputs,
            api_name=None,
            show_progress="hidden",
        )

        document_outputs = [documents_table, jobs_table, document_status]
        refresh_documents.click(
            on_documents,
            inputs=[retrieval_profile],
            outputs=document_outputs,
            api_name="documents_refresh",
        )
        upload.click(
            on_upload,
            inputs=[upload_file, source_key, display_title, retrieval_profile],
            outputs=document_outputs,
            api_name="documents_upload",
        )
        reindex.click(
            on_reindex,
            inputs=[retrieval_profile],
            outputs=document_outputs,
            api_name="documents_reindex",
        )
        delete.click(
            on_delete,
            inputs=[delete_source, confirm_delete, retrieval_profile],
            outputs=document_outputs,
            api_name="documents_delete",
        )
        demo.load(
            on_documents,
            outputs=document_outputs,
            api_name=None,
            show_progress="hidden",
        )
        evaluation_outputs = [
            dataset_select,
            plan_select,
            run_select,
            identity_table,
            plan_table,
            runs_table,
            evaluation_progress,
            gate_banner,
            dashboard_kpis,
            overview_table,
            quality_table,
            quality_plot,
            category_table,
            failures_table,
            performance_table,
            latency_plot,
            cost_table,
            cache_table,
            refusal_table,
            system_table,
            operations_table,
            operations_preview,
            operations_links,
            artifacts_table,
            artifact_links,
            evaluation_status,
            start_evaluation,
            evaluation_timer,
            session_state,
        ]
        evaluation_inputs = [
            dataset_select,
            plan_select,
            run_select,
            session_state,
            retrieval_profile,
        ]
        start_evaluation.click(
            on_start_evaluation,
            inputs=[dataset_select, plan_select, session_state, retrieval_profile],
            outputs=evaluation_outputs,
            api_name="evaluation_start",
        )
        refresh_evaluation.click(
            on_refresh_evaluation,
            inputs=evaluation_inputs,
            outputs=evaluation_outputs,
            api_name="evaluation_refresh",
        )
        dataset_select.input(
            on_refresh_evaluation,
            inputs=evaluation_inputs,
            outputs=evaluation_outputs,
            api_name=None,
        )
        plan_select.input(
            on_refresh_evaluation,
            inputs=evaluation_inputs,
            outputs=evaluation_outputs,
            api_name=None,
        )
        run_select.input(
            on_refresh_evaluation,
            inputs=evaluation_inputs,
            outputs=evaluation_outputs,
            api_name=None,
        )
        evaluation_timer.tick(
            on_refresh_evaluation,
            inputs=evaluation_inputs,
            outputs=evaluation_outputs,
            api_name=None,
            show_progress="hidden",
        )
        chat_load.then(
            on_refresh_evaluation,
            inputs=evaluation_inputs,
            outputs=evaluation_outputs,
            api_name=None,
            show_progress="hidden",
        )

        comparison_outputs = [
            comparison_plan_select,
            comparison_select,
            comparison_plan_table,
            comparison_history_table,
            comparison_progress,
            comparison_gate,
            retrieval_comparison_table,
            retrieval_comparison_plot,
            retrieval_recommendation,
            model_comparison_table,
            model_comparison_plot,
            model_recommendation,
            controlled_dimensions,
            comparison_shared_setup,
            comparison_level_metrics,
            cache_conclusion,
            comparison_candidates,
            comparison_plot,
            comparison_categories,
            comparison_recommendation,
            comparison_artifacts,
            comparison_artifact_links,
            comparison_status,
            start_comparison,
            comparison_timer,
            session_state,
        ]
        comparison_inputs = [
            comparison_plan_select,
            comparison_select,
            session_state,
            retrieval_profile,
        ]
        start_comparison.click(
            on_start_comparison,
            inputs=[comparison_plan_select, session_state, retrieval_profile],
            outputs=comparison_outputs,
            api_name="comparison_start",
        )
        refresh_comparison.click(
            on_refresh_comparison,
            inputs=comparison_inputs,
            outputs=comparison_outputs,
            api_name="comparison_refresh",
        )
        comparison_plan_select.input(
            on_refresh_comparison,
            inputs=comparison_inputs,
            outputs=comparison_outputs,
            api_name=None,
        )
        comparison_select.input(
            on_refresh_comparison,
            inputs=comparison_inputs,
            outputs=comparison_outputs,
            api_name=None,
        )
        comparison_timer.tick(
            on_refresh_comparison,
            inputs=comparison_inputs,
            outputs=comparison_outputs,
            api_name=None,
            show_progress="hidden",
        )
        chat_load.then(
            on_refresh_comparison,
            inputs=comparison_inputs,
            outputs=comparison_outputs,
            api_name=None,
            show_progress="hidden",
        )

        def on_switch_profile(
            selected: str | None,
            raw_state: BrowserSessionState | None,
        ) -> tuple[Any, ...]:
            chat_render = controller.switch_profile(selected, raw_state)
            reset_state = chat_render.state.with_evaluation(None).with_comparison(None)
            evaluation_render = controller.refresh_evaluations(
                reset_state,
                profile_id=selected,
            )
            comparison_render = controller.refresh_comparisons(
                evaluation_render.state.with_comparison(None),
                profile_id=selected,
            )
            final_state = comparison_render.state
            return (
                *_chat_outputs(replace(chat_render, state=final_state)),
                *_document_outputs(controller.refresh_documents(selected)),
                *_evaluation_outputs(evaluation_render)[:-1],
                *_comparison_outputs(comparison_render)[:-1],
            )

        retrieval_profile.change(
            on_switch_profile,
            inputs=[retrieval_profile, session_state],
            outputs=[
                *chat_outputs,
                *document_outputs,
                *evaluation_outputs[:-1],
                *comparison_outputs[:-1],
            ],
            api_name="retrieval_profile_select",
        )

        diagnostic_outputs = [health_table, request_table, diagnostics_status]
        refresh_health.click(on_health, outputs=diagnostic_outputs, api_name="diagnostics_health")
        inspect.click(
            on_inspect,
            inputs=[request_id],
            outputs=diagnostic_outputs,
            api_name="diagnostics_request",
        )

    return cast(gr.Blocks, demo)


def mount_workbench(
    app: FastAPI,
    *,
    settings: Settings,
    services: WorkbenchServices,
    blocks: gr.Blocks | None = None,
) -> FastAPI:
    """Mount Gradio without exposing tracebacks or unrestricted filesystem paths."""

    demo = blocks or create_workbench(settings, services)
    return cast(
        FastAPI,
        gr.mount_gradio_app(
            app,
            demo,
            path=settings.workbench_path,
            footer_links=[],
            show_error=False,
            max_file_size=settings.upload_max_bytes,
            allowed_paths=[],
            css=_WORKBENCH_CSS,
        ),
    )
