"""Gradio composition and FastAPI mounting for the four-view workbench."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import gradio as gr
from fastapi import FastAPI

from rag_mvp.config.settings import Settings

from .callbacks import WorkbenchCallbacks
from .models import (
    BrowserSessionState,
    ChatRender,
    DiagnosticsRender,
    DocumentsRender,
    EvaluationRender,
)
from .services import WorkbenchServices


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
) -> tuple[list[list[Any]], list[list[Any]], str, str, BrowserSessionState]:
    return (
        [list(row) for row in render.run_rows],
        [list(row) for row in render.failure_rows],
        render.metrics_markdown,
        render.status_markdown,
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


def create_workbench(
    settings: Settings,
    services: WorkbenchServices,
    *,
    callbacks: WorkbenchCallbacks | None = None,
) -> gr.Blocks:
    """Create one workbench using the same in-process services as the HTTP API."""

    controller = callbacks or WorkbenchCallbacks(services)
    with gr.Blocks(title="RAG Assistant Workbench") as demo:
        # Gradio deep-copies the state factory.  A controller-bound method would
        # recursively copy the production service graph, including non-copyable
        # locks and HTTP clients.  The model factory is stateless and still gives
        # every browser session a distinct owner identifier.
        session_state = gr.State(value=BrowserSessionState.create)
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
                    label="Ingestion progress / 摄取进度",
                    interactive=False,
                    type="array",
                )
                delete_source = gr.Textbox(label="Source ID to delete / 待删除来源 ID")
                confirm_delete = gr.Checkbox(label="Confirm deletion / 确认删除")
                delete = gr.Button("Delete / 删除", variant="stop")
                document_status = gr.Markdown(label="Document status / 文档状态")

            with gr.Tab("Evaluation", id="evaluation-tab"):
                dataset_id = gr.Textbox(label="Dataset ID / 数据集 ID", value="mvp-v1")
                dataset_version = gr.Textbox(label="Dataset version / 数据集版本", value="1.0.0")
                with gr.Row():
                    start_evaluation = gr.Button("Run evaluation / 运行评估", variant="primary")
                    refresh_evaluation = gr.Button("Refresh runs / 刷新运行")
                runs_table = gr.Dataframe(
                    headers=[
                        "run_id",
                        "status",
                        "dataset",
                        "version",
                        "complete",
                        "failed",
                        "total",
                    ],
                    label="Evaluation runs / 评估运行",
                    interactive=False,
                    type="array",
                )
                failures_table = gr.Dataframe(
                    label="Failed cases / 失败用例", interactive=False, type="array"
                )
                baseline_run = gr.Textbox(label="Baseline run ID / 基线运行 ID")
                candidate_run = gr.Textbox(label="Candidate run ID / 候选运行 ID")
                compare = gr.Button("Compare compatible runs / 比较兼容运行")
                metrics = gr.Markdown(label="Metrics / 指标")
                evaluation_status = gr.Markdown(label="Evaluation status / 评估状态")
                report_run = gr.Textbox(label="Report run ID / 报告运行 ID")
                with gr.Row():
                    download_json = gr.DownloadButton("Download JSON / 下载 JSON")
                    download_html = gr.DownloadButton("Download HTML / 下载 HTML")

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
        ) -> tuple[list[dict[str, str]], BrowserSessionState, str, str, str, str]:
            return _chat_outputs(
                await controller.submit_chat(raw_question, raw_mode, raw_history, raw_state)
            )

        def on_reset(
            raw_state: BrowserSessionState | None,
        ) -> tuple[list[dict[str, str]], BrowserSessionState, str, str, str, str]:
            return _chat_outputs(controller.reset_chat(raw_state))

        def on_cancel(
            raw_history: Sequence[Mapping[str, object]] | None,
            raw_state: BrowserSessionState | None,
        ) -> tuple[list[dict[str, str]], BrowserSessionState, str, str, str, str]:
            return _chat_outputs(controller.cancel_chat(raw_history, raw_state))

        def on_documents() -> tuple[list[list[Any]], list[list[Any]], str]:
            return _document_outputs(controller.refresh_documents())

        async def on_upload(
            path: str | None, key: str, title: str
        ) -> tuple[list[list[Any]], list[list[Any]], str]:
            return _document_outputs(await controller.upload_document(path, key, title))

        async def on_reindex() -> tuple[list[list[Any]], list[list[Any]], str]:
            return _document_outputs(await controller.reindex_documents())

        async def on_delete(
            source: str, confirmed: bool
        ) -> tuple[list[list[Any]], list[list[Any]], str]:
            return _document_outputs(await controller.delete_document(source, confirmed))

        async def on_start_evaluation(
            selected_dataset: str,
            selected_version: str,
            raw_state: BrowserSessionState | None,
        ) -> tuple[list[list[Any]], list[list[Any]], str, str, BrowserSessionState]:
            return _evaluation_outputs(
                await controller.start_evaluation(selected_dataset, selected_version, raw_state)
            )

        def on_refresh_evaluation(
            raw_state: BrowserSessionState | None,
        ) -> tuple[list[list[Any]], list[list[Any]], str, str, BrowserSessionState]:
            return _evaluation_outputs(controller.refresh_evaluations(raw_state))

        def on_compare(
            baseline: str,
            candidate: str,
            raw_state: BrowserSessionState | None,
        ) -> tuple[list[list[Any]], list[list[Any]], str, str, BrowserSessionState]:
            return _evaluation_outputs(
                controller.compare_evaluations(baseline, candidate, raw_state)
            )

        def on_health() -> tuple[list[list[Any]], list[list[Any]], str]:
            return _diagnostic_outputs(controller.refresh_health())

        def on_inspect(value: str) -> tuple[list[list[Any]], list[list[Any]], str]:
            return _diagnostic_outputs(controller.inspect_request(value))

        chat_outputs = [chatbot, session_state, citations, previews, chat_status, question]
        ask_event = ask.click(
            on_ask,
            inputs=[question, mode, chatbot, session_state],
            outputs=chat_outputs,
            api_name="chat_submit",
        )
        question.submit(
            on_ask,
            inputs=[question, mode, chatbot, session_state],
            outputs=chat_outputs,
            api_name=None,
        )
        reset.click(on_reset, inputs=[session_state], outputs=chat_outputs, api_name="chat_reset")
        cancel.click(
            on_cancel,
            inputs=[chatbot, session_state],
            outputs=chat_outputs,
            cancels=[ask_event],
            api_name="chat_cancel",
        )

        document_outputs = [documents_table, jobs_table, document_status]
        refresh_documents.click(
            on_documents, outputs=document_outputs, api_name="documents_refresh"
        )
        upload.click(
            on_upload,
            inputs=[upload_file, source_key, display_title],
            outputs=document_outputs,
            api_name="documents_upload",
        )
        reindex.click(on_reindex, outputs=document_outputs, api_name="documents_reindex")
        delete.click(
            on_delete,
            inputs=[delete_source, confirm_delete],
            outputs=document_outputs,
            api_name="documents_delete",
        )

        evaluation_outputs = [
            runs_table,
            failures_table,
            metrics,
            evaluation_status,
            session_state,
        ]
        start_evaluation.click(
            on_start_evaluation,
            inputs=[dataset_id, dataset_version, session_state],
            outputs=evaluation_outputs,
            api_name="evaluation_start",
        )
        refresh_evaluation.click(
            on_refresh_evaluation,
            inputs=[session_state],
            outputs=evaluation_outputs,
            api_name="evaluation_refresh",
        )
        compare.click(
            on_compare,
            inputs=[baseline_run, candidate_run, session_state],
            outputs=evaluation_outputs,
            api_name="evaluation_compare",
        )
        download_json.click(
            lambda value: controller.report_path(value, "json"),
            inputs=[report_run],
            outputs=[download_json],
            api_name="evaluation_report_json",
        )
        download_html.click(
            lambda value: controller.report_path(value, "html"),
            inputs=[report_run],
            outputs=[download_html],
            api_name="evaluation_report_html",
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
            show_error=False,
            max_file_size=settings.upload_max_bytes,
            allowed_paths=[],
        ),
    )
