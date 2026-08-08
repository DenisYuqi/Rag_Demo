from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import gradio as gr
import pytest

from rag_mvp.config.settings import Settings
from rag_mvp.domain.qa import RefusalReason, StreamEventKind, ValidatedStreamEvent
from rag_mvp.domain.retrieval import RetrievalMode
from rag_mvp.ui.callbacks import SAFE_UI_ERROR, WorkbenchCallbacks
from rag_mvp.ui.models import BrowserSessionState, ChatServiceResult
from rag_mvp.ui.services import WorkbenchServices
from rag_mvp.ui.workbench import create_workbench

pytestmark = pytest.mark.ui


@dataclass
class IsolatingChatGateway:
    sessions: dict[str, str] = field(default_factory=dict)
    submissions: list[tuple[str, str | None, str]] = field(default_factory=list)

    async def submit(
        self,
        *,
        owner_id: str,
        session_id: str | None,
        question: str,
        mode: RetrievalMode,
    ) -> ChatServiceResult:
        del mode
        expected_session = self.sessions.setdefault(
            owner_id,
            f"session_{len(self.sessions) + 1}",
        )
        if session_id is not None and session_id != expected_session:
            raise AssertionError("cross_session_state")
        self.submissions.append((owner_id, session_id, question))
        return ChatServiceResult(
            event=ValidatedStreamEvent(
                request_id=f"request_{len(self.submissions)}",
                session_id=expected_session,
                sequence=0,
                kind=StreamEventKind.REFUSAL,
                response_language="en",
                content="No indexed evidence supports the request.",
                reason=RefusalReason.INSUFFICIENT_EVIDENCE,
                terminal=True,
            )
        )

    def reset(self, *, owner_id: str, session_id: str | None) -> str:
        del session_id
        replacement = f"session_reset_{len(self.sessions) + 1}"
        self.sessions[owner_id] = replacement
        return replacement


@dataclass
class ExplodingChatGateway:
    async def submit(
        self,
        *,
        owner_id: str,
        session_id: str | None,
        question: str,
        mode: RetrievalMode,
    ) -> ChatServiceResult:
        del owner_id, session_id, question, mode
        raise RuntimeError("private stack person@example.com token sk-live-secret")

    def reset(self, *, owner_id: str, session_id: str | None) -> str:
        del owner_id, session_id
        raise RuntimeError("private reset failure")


def test_page_load_clears_conversation_without_loading_hidden_views(tmp_path: Path) -> None:
    blocks = create_workbench(
        Settings(_env_file=None, data_root=tmp_path),
        WorkbenchServices(chat=IsolatingChatGateway()),
    )
    config = blocks.get_config_file()
    labels = {
        component["id"]: component.get("props", {}).get("label")
        for component in config["components"]
    }
    chat_load = next(
        dependency
        for dependency in config["dependencies"]
        if any(event == "load" for _, event in dependency["targets"])
        and labels.get(dependency["outputs"][0]) == "Conversation / 对话"
    )

    assert len(chat_load["inputs"]) == 1
    assert [labels.get(output) for output in chat_load["outputs"]] == [
        "Conversation / 对话",
        None,
        "Citations / 引用",
        "Source previews / 来源预览",
        "Chat status / 对话状态",
        "Question / 问题",
    ]
    assert chat_load["inputs"] == [chat_load["outputs"][1]]
    page_load_successors = [
        dependency
        for dependency in config["dependencies"]
        if any(event == "then" for _, event in dependency["targets"])
    ]
    assert page_load_successors == []


def test_inactive_views_refresh_only_when_their_tabs_are_selected(tmp_path: Path) -> None:
    blocks = create_workbench(
        Settings(_env_file=None, data_root=tmp_path),
        WorkbenchServices(chat=IsolatingChatGateway()),
    )
    config = blocks.get_config_file()
    labels = {
        component["id"]: component.get("props", {}).get("label")
        for component in config["components"]
    }
    selected_view_outputs = {
        labels.get(dependency["outputs"][0])
        for dependency in config["dependencies"]
        if any(event == "select" for _, event in dependency["targets"])
    }

    assert selected_view_outputs >= {
        "Active documents / 活跃文档",
        "Registered dataset / 已注册数据集",
        "Registered experiment plan / 已注册实验计划",
    }


def test_chat_submission_hides_gradio_processing_indicator(tmp_path: Path) -> None:
    blocks = create_workbench(
        Settings(_env_file=None, data_root=tmp_path),
        WorkbenchServices(chat=IsolatingChatGateway()),
    )
    config = blocks.get_config_file()
    labels = {
        component["id"]: component.get("props", {}).get("label")
        for component in config["components"]
    }
    chat_submissions = [
        dependency
        for dependency in config["dependencies"]
        if dependency.get("api_name") == "chat_submit"
        or (
            any(event == "submit" for _, event in dependency["targets"])
            and labels.get(dependency["inputs"][0]) == "Question / 问题"
        )
    ]

    assert len(chat_submissions) == 2
    assert {dependency["show_progress"] for dependency in chat_submissions} == {"hidden"}


@pytest.mark.asyncio
async def test_gradio_state_factory_keeps_parallel_browser_sessions_isolated(
    tmp_path: Path,
) -> None:
    gateway = IsolatingChatGateway()
    services = WorkbenchServices(chat=gateway)
    blocks = create_workbench(
        Settings(_env_file=None, data_root=tmp_path),
        services,
    )
    state_component = next(
        component for component in blocks.blocks.values() if isinstance(component, gr.State)
    )
    state_factory = cast("object", state_component.value)

    assert callable(state_factory)
    first_state = state_factory()
    second_state = state_factory()
    assert isinstance(first_state, BrowserSessionState)
    assert isinstance(second_state, BrowserSessionState)
    assert first_state.owner_id != second_state.owner_id

    callbacks = WorkbenchCallbacks(services)
    first, second = await asyncio.gather(
        callbacks.submit_chat("First browser", "hybrid", None, first_state),
        callbacks.submit_chat("Second browser", "hybrid", None, second_state),
    )

    assert first.state.owner_id == first_state.owner_id
    assert second.state.owner_id == second_state.owner_id
    assert first.state.session_id != second.state.session_id
    assert gateway.submissions == [
        (first_state.owner_id, None, "First browser"),
        (second_state.owner_id, None, "Second browser"),
    ]


@pytest.mark.asyncio
async def test_ui_exception_state_is_fixed_content_and_contains_no_stack_or_secret() -> None:
    callbacks = WorkbenchCallbacks(WorkbenchServices(chat=ExplodingChatGateway()))
    state = BrowserSessionState.create().with_active_request("request_correlation")
    prior = ({"role": "assistant", "content": "Previously validated."},)

    rendered = await callbacks.submit_chat("Next question", "dense", prior, state)
    serialized = repr(rendered)

    assert rendered.history == prior
    assert rendered.citations_markdown == ""
    assert rendered.previews_markdown == ""
    assert SAFE_UI_ERROR in rendered.status_markdown
    assert "request_correlation" in rendered.status_markdown
    assert rendered.state.active_request_id is None
    assert "person@example.com" not in serialized
    assert "sk-live-secret" not in serialized
    assert "RuntimeError" not in serialized
    assert "Traceback" not in serialized
