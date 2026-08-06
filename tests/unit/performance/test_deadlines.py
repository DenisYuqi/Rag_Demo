from __future__ import annotations

import asyncio
import time

import pytest

from rag_mvp.config.settings import Settings
from rag_mvp.performance.deadlines import (
    DeadlineController,
    DeadlineExceededError,
    QALatencyBudgets,
    StageDeadlineExceededError,
)
from rag_mvp.qa.deadlines import QAStageBudgets


def test_default_total_and_settings_driven_stage_budgets(tmp_path: object) -> None:
    defaults = QALatencyBudgets()
    qa_defaults = QAStageBudgets()
    assert defaults.total_seconds == 9.5
    assert qa_defaults.retrieval_seconds == 1.8
    assert defaults.for_stage("reranking") == 1.2
    assert defaults.for_stage("dense") == 0.8
    assert defaults.for_stage("evidence") == 2.0

    settings = Settings(
        data_root=tmp_path,
        qa_deadline_seconds=19,
        rerank_deadline_seconds=2,
        _env_file=None,
    )
    configured = QALatencyBudgets.from_settings(settings)
    assert configured.total_seconds == 19
    assert configured.validation_seconds == pytest.approx(0.4)
    assert configured.evidence_assessment_seconds == pytest.approx(4.0)
    assert configured.rerank_seconds == 2


@pytest.mark.asyncio
async def test_total_deadline_cancels_unfinished_required_work() -> None:
    cancelled = asyncio.Event()

    async def blocking() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    controller = DeadlineController(started_at=time.monotonic() - 9.48)
    with pytest.raises(DeadlineExceededError):
        await controller.run_required("generation", blocking)
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_stage_timeout_is_distinct_from_total_timeout() -> None:
    budgets = QALatencyBudgets(rerank_seconds=0.01)
    controller = DeadlineController(budgets)

    with pytest.raises(StageDeadlineExceededError) as timed_out:
        await controller.run_required("rerank", lambda: asyncio.sleep(1))

    assert timed_out.value.stage == "rerank"
    assert controller.remaining_seconds > 1


@pytest.mark.asyncio
async def test_optional_reranker_cancels_and_degrades_to_exact_base_ranking() -> None:
    budgets = QALatencyBudgets(rerank_seconds=0.01)
    controller = DeadlineController(budgets)
    cancelled = asyncio.Event()
    base = ("chunk-a", "chunk-b")

    async def slow_reranker() -> tuple[str, ...]:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    result = await controller.run_optional_reranker(
        slow_reranker,
        base_ranking=base,
    )

    assert result.value is base
    assert result.degraded
    assert result.degradation_reason == "rerank_timeout"
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_generation_does_not_start_without_full_budget_and_finalization_reserve() -> None:
    controller = DeadlineController(started_at=time.monotonic() - 4.3)
    called = False

    async def generation() -> str:
        nonlocal called
        called = True
        return "unsafe partial"

    with pytest.raises(DeadlineExceededError):
        await controller.run_generation(generation)
    assert not called


@pytest.mark.asyncio
async def test_parent_cancellation_is_never_converted_to_optional_degradation() -> None:
    controller = DeadlineController()
    operation_started = asyncio.Event()

    async def operation() -> tuple[str, ...]:
        operation_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    task = asyncio.create_task(controller.run_optional_reranker(operation, base_ranking=("base",)))
    await operation_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
