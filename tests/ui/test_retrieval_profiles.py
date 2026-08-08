from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from rag_mvp.config.settings import Settings
from rag_mvp.domain.evaluation import EvaluationRun
from rag_mvp.domain.qa import RefusalReason, StreamEventKind, ValidatedStreamEvent
from rag_mvp.domain.retrieval import RetrievalMode
from rag_mvp.ui.callbacks import SAFE_UNAVAILABLE, WorkbenchCallbacks
from rag_mvp.ui.models import BrowserSessionState, ChatServiceResult
from rag_mvp.ui.services import RetrievalProfileGateways, WorkbenchServices
from rag_mvp.ui.workbench import create_workbench

pytestmark = pytest.mark.ui


@dataclass
class ProfileChatGateway:
    profile_id: str
    submissions: list[str] = field(default_factory=list)
    resets: list[str | None] = field(default_factory=list)

    async def submit(
        self,
        *,
        owner_id: str,
        session_id: str | None,
        question: str,
        mode: RetrievalMode,
    ) -> ChatServiceResult:
        del owner_id, session_id, mode
        self.submissions.append(question)
        return ChatServiceResult(
            event=ValidatedStreamEvent(
                request_id=f"request-{self.profile_id}",
                session_id=f"session-{self.profile_id}",
                sequence=0,
                kind=StreamEventKind.REFUSAL,
                response_language="en",
                content="No indexed evidence.",
                reason=RefusalReason.INSUFFICIENT_EVIDENCE,
                terminal=True,
            )
        )

    def reset(self, *, owner_id: str, session_id: str | None) -> str:
        del owner_id
        self.resets.append(session_id)
        return f"session-{self.profile_id}"


@dataclass
class ProfileDocumentGateway:
    profile_id: str
    reads: int = 0

    def submit_upload(self, payload: object) -> object:
        del payload
        raise AssertionError("not used")

    def submit_reindex(self) -> object:
        raise AssertionError("not used")

    def submit_delete(self, source_id: str) -> object:
        del source_id
        raise AssertionError("not used")

    async def run_job(self, job_id: str) -> object:
        del job_id
        raise AssertionError("not used")

    def get_job(self, job_id: str) -> object:
        del job_id
        return None

    def list_active_documents(self) -> tuple[str, tuple[()]]:
        self.reads += 1
        return f"revision-{self.profile_id}", ()

    def list_jobs(self) -> tuple[()]:
        return ()


@dataclass
class ProfileEvaluationGateway:
    profile_id: str
    starts: list[tuple[str, str | None]] = field(default_factory=list)
    reads: int = 0

    async def start(
        self,
        dataset_id: str,
        dataset_version: str | None = None,
    ) -> EvaluationRun:
        self.starts.append((dataset_id, dataset_version))
        return self.list_runs()[0]

    def list_runs(self) -> tuple[EvaluationRun, ...]:
        self.reads += 1
        return (
            EvaluationRun(
                run_id=f"run-{self.profile_id}",
                dataset_id="mvp-v1",
                dataset_version="1.0.0",
                dataset_hash="dataset-hash",
                corpus_version="corpus-v1",
                configuration_id=f"config-{self.profile_id}",
                code_revision="revision-test",
                scorer_versions={"faithfulness": "v1"},
                cache_policy="bypass-final",
                total_cases=1,
            ),
        )

    def failed_cases(self, run_id: str) -> tuple[()]:
        del run_id
        return ()


def profile_services() -> tuple[WorkbenchServices, ProfileChatGateway, ProfileChatGateway]:
    openai = ProfileChatGateway("openai-api")
    bge = ProfileChatGateway("bge-local")
    services = WorkbenchServices(
        retrieval_profiles={
            "openai-api": RetrievalProfileGateways(
                chat=openai,
                documents=ProfileDocumentGateway("openai-api"),  # type: ignore[arg-type]
            ),
            "bge-local": RetrievalProfileGateways(
                chat=bge,
                documents=ProfileDocumentGateway("bge-local"),  # type: ignore[arg-type]
            ),
        },
        evaluation_profiles={
            "openai-api": ProfileEvaluationGateway("openai-api"),  # type: ignore[dict-item]
            "bge-local": ProfileEvaluationGateway("bge-local"),  # type: ignore[dict-item]
        },
        default_retrieval_profile="openai-api",
    )
    return services, openai, bge


@pytest.mark.asyncio
async def test_callbacks_route_explicit_profile_and_reject_unknown_profile() -> None:
    services, openai, bge = profile_services()
    callbacks = WorkbenchCallbacks(services)

    rendered = await callbacks.submit_chat(
        "Question",
        "hybrid-rerank",
        None,
        BrowserSessionState.create(),
        "bge-local",
    )
    unknown = await callbacks.submit_chat("Question", "hybrid", None, None, "unknown-profile")

    assert bge.submissions == ["Question"]
    assert openai.submissions == []
    assert rendered.state.session_id == "session-bge-local"
    assert unknown.status_markdown == SAFE_UNAVAILABLE


def test_switching_profile_starts_a_fresh_profile_session_and_refreshes_documents() -> None:
    services, openai, bge = profile_services()
    callbacks = WorkbenchCallbacks(services)
    state = BrowserSessionState.create().with_session("session-openai-api")

    switched = callbacks.switch_profile("bge-local", state)
    documents = callbacks.refresh_documents("bge-local")

    assert switched.state.session_id == "session-bge-local"
    assert bge.resets == [None]
    assert openai.resets == []
    assert "revision-bge-local" in documents.status_markdown


def test_evaluation_data_is_isolated_by_selected_retrieval_profile() -> None:
    services, _, _ = profile_services()
    callbacks = WorkbenchCallbacks(services)

    openai = callbacks.refresh_evaluations(None, profile_id="openai-api")
    bge = callbacks.refresh_evaluations(None, profile_id="bge-local")
    unknown = callbacks.refresh_evaluations(None, profile_id="unknown-profile")

    assert openai.run_rows[0][0] == "run-openai-api"
    assert bge.run_rows[0][0] == "run-bge-local"
    assert unknown.status_markdown == SAFE_UNAVAILABLE


@pytest.mark.asyncio
async def test_evaluation_start_is_routed_to_the_selected_retrieval_profile() -> None:
    services, _, _ = profile_services()
    callbacks = WorkbenchCallbacks(services)
    openai = services.evaluations_for("openai-api")
    bge = services.evaluations_for("bge-local")
    assert isinstance(openai, ProfileEvaluationGateway)
    assert isinstance(bge, ProfileEvaluationGateway)

    rendered = await callbacks.start_evaluation(
        "mvp-v1",
        "1.0.0",
        BrowserSessionState.create(),
        "bge-local",
    )

    assert bge.starts == [("mvp-v1", "1.0.0")]
    assert openai.starts == []
    assert rendered.run_rows[0][0] == "run-bge-local"


def test_workbench_profile_selector_is_wired_to_chat_and_documents() -> None:
    services, _, _ = profile_services()
    blocks = create_workbench(Settings(_env_file=None), services)
    config = blocks.get_config_file()
    profile = next(
        component
        for component in config["components"]
        if component.get("props", {}).get("label") == "Retrieval profile / 检索模型"
    )

    assert profile["props"]["choices"] == [
        ("openai-api", "openai-api"),
        ("bge-local", "bge-local"),
    ]
    assert profile["props"]["value"] == "openai-api"
    by_api_name = {dependency.get("api_name"): dependency for dependency in config["dependencies"]}
    for api_name in (
        "chat_submit",
        "documents_refresh",
        "documents_upload",
        "documents_reindex",
        "documents_delete",
        "evaluation_start",
        "evaluation_refresh",
        "comparison_start",
        "comparison_refresh",
    ):
        assert profile["id"] in by_api_name[api_name]["inputs"]
    assert by_api_name["retrieval_profile_select"]["inputs"][0] == profile["id"]
    profile_outputs = by_api_name["retrieval_profile_select"]["outputs"]
    chat_outputs = by_api_name["chat_submit"]["outputs"]
    assert profile_outputs[: len(chat_outputs)] == chat_outputs
    assert len(profile_outputs) == len(chat_outputs) + 1
    assert by_api_name["retrieval_profile_select"]["show_progress"] == "hidden"


def test_evaluation_tab_reuses_loaded_results_for_the_same_profile() -> None:
    services, _, _ = profile_services()
    blocks = create_workbench(Settings(_env_file=None), services)
    config = blocks.get_config_file()
    labels = {
        component["id"]: component.get("props", {}).get("label")
        for component in config["components"]
    }
    dependency = next(
        item
        for item in config["dependencies"]
        if any(
            event == "select" and labels.get(component_id) == "Evaluation"
            for component_id, event in item["targets"]
        )
    )
    handler = blocks.fns[dependency["id"]].fn
    bge = services.evaluations_for("bge-local")
    assert callable(handler)
    assert isinstance(bge, ProfileEvaluationGateway)
    baseline_reads = bge.reads

    first = handler(
        None,
        None,
        None,
        BrowserSessionState.create(),
        "bge-local",
        None,
    )
    first_state = first[-2]
    loaded_profile = first[-1]
    assert bge.reads == baseline_reads + 1
    assert loaded_profile == "bge-local"

    second = handler(
        None,
        None,
        None,
        first_state,
        "bge-local",
        loaded_profile,
    )

    assert bge.reads == baseline_reads + 1
    assert second[-1] == "bge-local"
    assert dependency["show_progress"] == "minimal"
