from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rag_mvp.api.qa import QARuntimeServices
from rag_mvp.domain.evaluation import EvaluationRun, EvaluationRunStatus
from rag_mvp.domain.qa import (
    ConversationRole,
    QARefusal,
    RefusalReason,
    StreamEventKind,
    ValidatedStreamEvent,
)
from rag_mvp.domain.retrieval import CachePolicy, RetrievalMode
from rag_mvp.evaluation.runner import (
    EvaluationCaseExecution,
    EvaluationCaseInput,
    EvaluationConversationTurn,
    EvaluationEnvironment,
    EvaluationRunIdentity,
    EvaluationRunner,
    EvaluationRunPlan,
    ImmutableRunError,
    ProductionQAExecutor,
)
from rag_mvp.qa.orchestrator import OrchestratedResponse
from rag_mvp.qa.query_rewrite import select_response_language
from rag_mvp.qa.sessions import ConversationService
from rag_mvp.qa.streaming import CompleteResponseEmitter
from rag_mvp.safety.redactor import DEFAULT_REDACTOR
from rag_mvp.storage.database import Database
from rag_mvp.storage.repositories import SessionRepository


@dataclass
class MemoryRunRepository:
    value: EvaluationRun | None = None
    history: list[EvaluationRun] = field(default_factory=list)

    def create(self, run: EvaluationRun) -> None:
        if self.value is not None:
            raise ValueError("duplicate")
        self.value = run
        self.history.append(run)

    def get(self, run_id: str) -> EvaluationRun | None:
        if self.value is None or self.value.run_id != run_id:
            return None
        return self.value

    def update(self, run: EvaluationRun) -> None:
        self.value = run
        self.history.append(run)


@dataclass
class FakeExecutor:
    fail_case_id: str | None = None
    calls: list[tuple[str, str, CachePolicy]] = field(default_factory=list)

    async def execute(
        self,
        case: EvaluationCaseInput,
        *,
        owner_id: str,
        cache_policy: CachePolicy,
    ) -> EvaluationCaseExecution:
        self.calls.append((case.case_id, owner_id, cache_policy))
        if case.case_id == self.fail_case_id:
            raise RuntimeError("sensitive provider detail")
        session_id = f"session_{case.case_id}"
        request_id = f"request_{case.case_id}"
        event = ValidatedStreamEvent(
            request_id=request_id,
            session_id=session_id,
            sequence=0,
            kind=StreamEventKind.REFUSAL,
            response_language=case.language,
            content="I do not have enough evidence.",
            reason=RefusalReason.INSUFFICIENT_EVIDENCE,
            terminal=True,
        )
        return EvaluationCaseExecution(
            case_id=case.case_id,
            owner_id=owner_id,
            session_id=session_id,
            request_id=request_id,
            event=event,
            latency_ms=1.0,
        )


class IncrementingClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 7, tzinfo=UTC)

    def __call__(self) -> datetime:
        current = self.value
        self.value += timedelta(milliseconds=1)
        return current


def _identity() -> EvaluationRunIdentity:
    return EvaluationRunIdentity(
        dataset_id="mvp",
        dataset_version="1.0.0",
        dataset_hash="dataset-sha256",
        corpus_version="corpus-v1",
        corpus_hash="corpus-sha256",
        configuration_id="config-v1",
        code_revision="abcdef1",
        prompt_versions={"generation": "prompt-v1", "grounding": "grounding-v1"},
        provider_identities={"generation": "provider-v1"},
        model_identities={"generation": "model-v1", "embedding": "embed-v1"},
        generation_settings={"temperature": 0, "seed": 7},
        embedding_identity={"dimension": 3, "normalization": "none"},
        chunking_identity={"target_tokens": 500, "overlap_tokens": 80},
        retrieval_configuration={"mode": "hybrid", "rrf_k": 60},
        scorer_versions={
            "faithfulness": "v1",
            "context_precision": "v1",
            "answer_completeness": "v1",
            "style_consistency": "v1",
            "refusal_appropriateness": "v1",
        },
        pricing_version="pricing-v1",
        random_seeds={"generation": 7},
        environment=EvaluationEnvironment(
            python_version="3.12.11",
            platform="windows-amd64",
            deployment="test",
        ),
    )


def _plan() -> EvaluationRunPlan:
    return EvaluationRunPlan(
        run_id="run_001",
        identity=_identity(),
        cases=(
            EvaluationCaseInput(case_id="case_en", question="Question?", language="en"),
            EvaluationCaseInput(case_id="case_zh", question="问题?", language="zh"),
        ),
    )


@pytest.mark.asyncio
async def test_runner_bypasses_caches_persists_progress_and_write_once_artifacts(
    tmp_path: Path,
) -> None:
    repository = MemoryRunRepository()
    executor = FakeExecutor(fail_case_id="case_zh")
    runner = EvaluationRunner(
        repository,
        tmp_path / "evaluations",
        executor,
        clock=IncrementingClock(),
    )
    plan = _plan()

    queued = runner.queue(plan)
    original_manifest = (tmp_path / "evaluations" / plan.run_id / "manifest.json").read_bytes()
    completed = await runner.execute(plan)

    assert queued.status is EvaluationRunStatus.QUEUED
    assert completed.status is EvaluationRunStatus.COMPLETED
    assert (completed.completed_cases, completed.failed_cases) == (1, 1)
    assert [state.status for state in repository.history] == [
        EvaluationRunStatus.QUEUED,
        EvaluationRunStatus.RUNNING,
        EvaluationRunStatus.RUNNING,
        EvaluationRunStatus.RUNNING,
        EvaluationRunStatus.COMPLETED,
    ]
    assert [(state.completed_cases, state.failed_cases) for state in repository.history[2:4]] == [
        (1, 0),
        (1, 1),
    ]
    owners = {owner for _, owner, _ in executor.calls}
    assert len(owners) == 2
    assert all(owner.startswith("eval_owner_") and len(owner) == 75 for owner in owners)
    assert all(policy is CachePolicy.BYPASS for _, _, policy in executor.calls)
    manifest = runner.load_manifest(plan.run_id)
    assert manifest.case_ids == ("case_en", "case_zh")
    assert manifest.identity.cache_policy == "bypass"
    results = runner.load_case_results(plan.run_id)
    assert [(result.case_id, result.succeeded) for result in results] == [
        ("case_en", True),
        ("case_zh", False),
    ]
    assert results[1].safe_error_code == "case_execution_failed"
    assert "sensitive provider detail" not in results[1].model_dump_json()

    with pytest.raises(ImmutableRunError, match="evaluation_run_already_exists"):
        runner.queue(plan)
    assert (tmp_path / "evaluations" / plan.run_id / "manifest.json").read_bytes() == (
        original_manifest
    )


@dataclass
class RecordingOrchestrator:
    cache_policies: list[CachePolicy] = field(default_factory=list)

    async def run(
        self,
        *,
        request_id: str,
        session_id: str,
        owner_id: str,
        question: str,
        mode: RetrievalMode | str,
        requested_language: str | None = None,
        cache_policy: CachePolicy | str = CachePolicy.USE,
    ) -> OrchestratedResponse:
        del owner_id, mode
        self.cache_policies.append(CachePolicy(cache_policy))
        response_language = select_response_language(
            question,
            requested_language=requested_language,
        )
        return OrchestratedResponse._create(
            QARefusal(
                request_id=request_id,
                session_id=session_id,
                response_language=response_language,
                message="I do not have enough evidence.",
                reason=RefusalReason.INSUFFICIENT_EVIDENCE,
            ),
            retrieved_chunk_ids=("chunk_1", "chunk_2"),
            context_chunk_ids=("chunk_2",),
        )


@pytest.mark.asyncio
async def test_production_executor_uses_isolated_session_and_validated_qa_boundary(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "metadata.sqlite3")
    database.initialize()
    conversations = ConversationService(SessionRepository(database))
    orchestrator = RecordingOrchestrator()
    services = QARuntimeServices(
        conversations=conversations,
        orchestrator=orchestrator,
        emitter=CompleteResponseEmitter(conversations),
    )
    executor = ProductionQAExecutor(services, DEFAULT_REDACTOR)
    case = EvaluationCaseInput(
        case_id="multi_turn",
        question="What is the final answer?",
        language="en",
        history=(
            EvaluationConversationTurn(role=ConversationRole.USER, content="Earlier question"),
            EvaluationConversationTurn(role=ConversationRole.ASSISTANT, content="Earlier answer"),
        ),
    )

    first = await executor.execute(case, owner_id="owner_one", cache_policy=CachePolicy.BYPASS)
    second = await executor.execute(case, owner_id="owner_two", cache_policy=CachePolicy.BYPASS)

    assert first.session_id != second.session_id
    assert first.event.kind is StreamEventKind.REFUSAL
    assert first.retrieved_chunk_ids == ("chunk_1", "chunk_2")
    assert first.context_chunk_ids == ("chunk_2",)
    assert orchestrator.cache_policies == [CachePolicy.BYPASS, CachePolicy.BYPASS]
    turns = conversations.list_turns(first.session_id, "owner_one")
    assert [(turn.role, turn.content) for turn in turns] == [
        (ConversationRole.USER, "Earlier question"),
        (ConversationRole.ASSISTANT, "Earlier answer"),
        (ConversationRole.ASSISTANT, "I do not have enough evidence."),
    ]


@pytest.mark.asyncio
async def test_production_executor_normalizes_dataset_language_alias(tmp_path: Path) -> None:
    database = Database(tmp_path / "metadata.sqlite3")
    database.initialize()
    conversations = ConversationService(SessionRepository(database))
    services = QARuntimeServices(
        conversations=conversations,
        orchestrator=RecordingOrchestrator(),
        emitter=CompleteResponseEmitter(conversations),
    )
    executor = ProductionQAExecutor(services, DEFAULT_REDACTOR)

    execution = await executor.execute(
        EvaluationCaseInput(case_id="case_zh", question="年假有多少天?", language="zh"),
        owner_id="owner_zh",
        cache_policy=CachePolicy.BYPASS,
    )

    assert execution.event.kind is StreamEventKind.REFUSAL
    assert execution.event.response_language == "zh-CN"
