from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace

import pytest
from retrieval_test_helpers import candidate

from rag_mvp.domain.retrieval import RetrievalCandidate, RetrievalMode
from rag_mvp.providers.errors import ProviderError
from rag_mvp.providers.fakes import DeterministicRerankingProvider
from rag_mvp.providers.models import (
    AttemptStatus,
    Deadline,
    ModelIdentity,
    ProviderCallContext,
    ProviderErrorCategory,
    RerankRequest,
    RerankResult,
    TokenUsage,
)
from rag_mvp.providers.resilience import RetryPolicy, capture_provider_attempts
from rag_mvp.providers.routing import ModelProviderRouter, ProviderRoute
from rag_mvp.retrieval.request import RetrievalRequestContext
from rag_mvp.retrieval.rerank import (
    RerankIntegrityError,
    RerankStage,
    RerankTruncationPolicy,
    validate_rerank_stage_result,
)
from rag_mvp.retrieval.service import RetrievalService


def _context(seconds: float = 5) -> ProviderCallContext:
    return ProviderCallContext("request", "rerank", Deadline.after(seconds))


class CapturingReranker:
    identity = ModelIdentity("test", "reranker", "adapter-v1")

    def __init__(self, order: tuple[str, ...] | None = None) -> None:
        self.order = order
        self.requests: list[RerankRequest] = []
        self.contexts: list[ProviderCallContext] = []

    async def rerank(
        self,
        request: RerankRequest,
        context: ProviderCallContext,
    ) -> RerankResult:
        self.requests.append(request)
        self.contexts.append(context)
        return RerankResult(
            self.order or tuple(reversed(request.candidate_ids)),
            self.identity,
            request.prompt_version,
            TokenUsage(input_tokens=7, output_tokens=3),
        )


class FailingReranker(CapturingReranker):
    async def rerank(
        self,
        request: RerankRequest,
        context: ProviderCallContext,
    ) -> RerankResult:
        self.requests.append(request)
        self.contexts.append(context)
        raise ProviderError(ProviderErrorCategory.SERVER)


class BlockingReranker(CapturingReranker):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = False

    async def rerank(
        self,
        request: RerankRequest,
        context: ProviderCallContext,
    ) -> RerankResult:
        self.requests.append(request)
        self.contexts.append(context)
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class StaticRetriever:
    def __init__(self, results: tuple[RetrievalCandidate, ...]) -> None:
        self.results = results

    async def search(self, query: str, limit: int) -> tuple[RetrievalCandidate, ...]:
        del query, limit
        return self.results


class InvalidLegacyReranker:
    identity = "invalid-v1"

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
    ) -> tuple[str, ...]:
        del query
        return (candidates[0].chunk_id, "invented")


class SlowLegacyReranker:
    identity = "slow-v1"

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
    ) -> tuple[str, ...]:
        del query
        await asyncio.sleep(1)
        return tuple(candidate.chunk_id for candidate in candidates)


async def test_stage_bounds_registry_and_text_before_provider_call() -> None:
    provider = CapturingReranker()
    stage = RerankStage(
        provider,
        candidate_limit=2,
        budget_seconds=1,
        truncation=RerankTruncationPolicy(
            maximum_query_characters=4,
            maximum_query_tokens=3,
            maximum_candidate_characters=5,
            maximum_candidate_tokens=4,
        ),
    )
    base = tuple(candidate(chunk_id, dense_rank=rank) for rank, chunk_id in enumerate("abc", 1))

    result = await stage.run("123456", base, _context())

    assert result.applied and not result.degraded
    assert result.submitted_count == 2
    assert [item.chunk_id for item in result.ordered_candidates] == ["b", "a", "c"]
    assert [item.reranking_rank for item in result.ordered_candidates] == [1, 2, None]
    assert provider.requests[0].query == "123"
    assert provider.requests[0].candidate_ids == ("a", "b")
    assert [item.text for item in provider.requests[0].candidates] == ["Evid", "Evid"]
    assert result.identity == provider.identity
    assert result.usage == TokenUsage(input_tokens=7, output_tokens=3)
    assert result.prompt_version == RerankStage.PROMPT_VERSION
    assert result.elapsed_ms >= 0


@pytest.mark.parametrize(
    "order",
    [("a", "a"), ("a", "invented"), ("a",)],
)
async def test_unknown_duplicate_and_missing_ids_degrade_to_exact_base_order(
    order: tuple[str, ...],
) -> None:
    base = (candidate("a", dense_rank=1), candidate("b", dense_rank=2))
    result = await RerankStage(CapturingReranker(order)).run("query", base, _context())

    assert not result.applied and result.degraded
    assert result.reason == "rerank_invalid_output"
    assert result.ordered_candidates == base
    assert result.identity is not None
    assert result.usage == TokenUsage(input_tokens=7, output_tokens=3)


async def test_provider_error_degrades_without_stage_retry() -> None:
    provider = FailingReranker()
    base = (candidate("a", dense_rank=1),)

    result = await RerankStage(provider).run("query", base, _context())

    assert result.ordered_candidates == base
    assert result.reason == "rerank_provider_server"
    assert len(provider.requests) == 1


async def test_router_degraded_result_remains_degradation() -> None:
    router = ModelProviderRouter()
    base = (candidate("a", dense_rank=1),)

    result = await RerankStage(router).run("query", base, _context())

    assert not result.applied and result.degraded
    assert result.reason == "rerank_provider_unavailable"
    assert result.ordered_candidates == base
    assert result.identity is None and result.usage is None


async def test_router_success_preserves_route_identity_attempts_and_usage() -> None:
    provider = DeterministicRerankingProvider()
    router = ModelProviderRouter(
        reranking_routes=(ProviderRoute("primary-rerank", provider, RetryPolicy(1, max_retries=0)),)
    )
    base = (candidate("a", dense_rank=1), candidate("b", dense_rank=2))

    result = await RerankStage(router).run("query", base, _context())

    assert result.applied
    assert result.route_id == "primary-rerank"
    assert result.identity == provider.identity
    assert result.usage is not None
    assert len(result.attempts) == 1


async def test_router_stage_timeout_preserves_one_real_attempt_in_both_ledgers() -> None:
    provider = BlockingReranker()
    router = ModelProviderRouter(
        reranking_routes=(ProviderRoute("primary-rerank", provider, RetryPolicy(1)),)
    )
    base = (candidate("a", dense_rank=1),)

    with capture_provider_attempts() as request_ledger:
        result = await RerankStage(router, budget_seconds=0.01).run(
            "query",
            base,
            _context(),
        )

    assert result.reason == "rerank_provider_deadline_exceeded"
    assert result.ordered_candidates == base
    assert provider.cancelled
    assert len(result.attempts) == 1
    assert result.attempts == request_ledger.attempts
    assert result.attempts[0].status is AttemptStatus.FAILED
    assert result.attempts[0].error_category is ProviderErrorCategory.DEADLINE_EXCEEDED


async def test_stage_timeout_cancels_provider_and_falls_back() -> None:
    provider = BlockingReranker()
    base = (candidate("a", dense_rank=1),)
    result = await RerankStage(provider, budget_seconds=0.01).run("query", base, _context())

    assert result.reason == "rerank_timeout"
    assert result.ordered_candidates == base
    assert provider.cancelled


async def test_parent_cancellation_propagates_and_cancels_provider() -> None:
    provider = BlockingReranker()
    task = asyncio.create_task(
        RerankStage(provider, budget_seconds=5).run(
            "query",
            (candidate("a", dense_rank=1),),
            _context(),
        )
    )
    await provider.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.cancelled


async def test_subdeadline_is_minimum_of_budget_and_total_remaining() -> None:
    provider = CapturingReranker()
    context = _context(0.2)
    before = context.deadline.remaining_seconds

    await RerankStage(provider, budget_seconds=5).run(
        "query",
        (candidate("a", dense_rank=1),),
        context,
    )

    subdeadline = provider.contexts[0].deadline
    assert 0 < subdeadline.expires_at - subdeadline.clock() <= before
    assert subdeadline.expires_at <= context.deadline.expires_at


async def test_service_boundary_rejects_mutated_stage_candidate() -> None:
    base = (candidate("a", dense_rank=1),)
    valid = await RerankStage(CapturingReranker()).run("query", base, _context())
    polluted = RetrievalCandidate.model_validate(
        {**valid.ordered_candidates[0].model_dump(), "display_title": "Invented"}
    )
    malformed = replace(valid, ordered_candidates=(polluted,))

    with pytest.raises(RerankIntegrityError, match="rerank_candidate_mutated"):
        validate_rerank_stage_result(base, malformed)


async def test_invalid_rerank_falls_back_to_rrf_legacy_compatibility() -> None:
    dense = StaticRetriever((candidate("a", dense_rank=1), candidate("b", dense_rank=2)))
    service = RetrievalService(
        dense=dense,
        lexical=StaticRetriever(()),
        reranker=InvalidLegacyReranker(),
    )

    result = await service.retrieve(
        RetrievalRequestContext("req", "query", RetrievalMode.HYBRID_RERANK, "rev")
    )

    assert [item.chunk_id for item in result.evidence] == ["a", "b"]
    assert "rerank_degraded" in result.diagnostics.degradation_reasons


async def test_rerank_timeout_falls_back_without_waiting_past_budget() -> None:
    dense = StaticRetriever((candidate("a", dense_rank=1),))
    service = RetrievalService(
        dense=dense,
        lexical=StaticRetriever(()),
        reranker=SlowLegacyReranker(),
        rerank_deadline_seconds=0.01,
    )

    result = await service.retrieve(
        RetrievalRequestContext("req", "query", RetrievalMode.HYBRID_RERANK, "rev")
    )

    assert result.evidence[0].chunk_id == "a"
    assert "rerank_degraded" in result.diagnostics.degradation_reasons
