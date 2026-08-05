from __future__ import annotations

from dataclasses import dataclass

import pytest

from rag_mvp.providers.errors import ProviderError, ProviderOperationError
from rag_mvp.providers.fakes import (
    DeterministicEmbeddingProvider,
    DeterministicGenerationProvider,
    DeterministicRerankingProvider,
)
from rag_mvp.providers.models import (
    ChatMessage,
    ChatRole,
    EmbeddingRequest,
    EmbeddingSpaceIdentity,
    GenerationRequest,
    GenerationResult,
    ModelIdentity,
    NormalizationPolicy,
    ProviderCallContext,
    ProviderErrorCategory,
    ProviderRole,
    RerankCandidate,
    RerankRequest,
    RerankResult,
)
from rag_mvp.providers.resilience import InMemoryAttemptRecorder, RetryPolicy
from rag_mvp.providers.routing import ModelProviderRouter, ProviderRoute

POLICY = RetryPolicy(1, max_retries=0)


class FailingGenerationProvider:
    def __init__(self, category: ProviderErrorCategory, name: str) -> None:
        self._identity = ModelIdentity("fake", name, "v1")
        self.category = category
        self.call_count = 0

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    async def generate(
        self, request: GenerationRequest, context: ProviderCallContext
    ) -> GenerationResult:
        del request, context
        self.call_count += 1
        raise ProviderError(self.category)


@dataclass
class InvalidReranker:
    _identity: ModelIdentity

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    async def rerank(self, request: RerankRequest, context: ProviderCallContext) -> RerankResult:
        del context
        return RerankResult(
            (request.candidate_ids[0], request.candidate_ids[0]),
            self.identity,
            request.prompt_version,
        )


async def test_generation_primary_failure_uses_ordered_fallback(
    provider_context: ProviderCallContext,
) -> None:
    primary = FailingGenerationProvider(ProviderErrorCategory.SERVER, "primary")
    fallback = DeterministicGenerationProvider(ModelIdentity("fake", "fallback", "v1"))
    recorder = InMemoryAttemptRecorder()
    router = ModelProviderRouter(
        generation_routes=(
            ProviderRoute("primary", primary, POLICY),
            ProviderRoute("fallback", fallback, POLICY),
        ),
        recorder=recorder,
    )

    result = await router.generate(
        GenerationRequest((ChatMessage(ChatRole.USER, "question"),)), provider_context
    )

    assert result.used_fallback
    assert primary.call_count == 1
    assert fallback.call_count == 1
    assert [attempt.route_id for attempt in result.attempts] == ["primary", "fallback"]
    assert result.attempts[-1].usage == result.value.usage
    assert recorder.attempts == result.attempts


async def test_invalid_generation_request_does_not_fallback(
    provider_context: ProviderCallContext,
) -> None:
    primary = FailingGenerationProvider(ProviderErrorCategory.INVALID_REQUEST, "primary")
    fallback = DeterministicGenerationProvider()
    router = ModelProviderRouter(
        generation_routes=(
            ProviderRoute("primary", primary, POLICY),
            ProviderRoute("fallback", fallback, POLICY),
        )
    )

    with pytest.raises(ProviderOperationError) as caught:
        await router.generate(
            GenerationRequest((ChatMessage(ChatRole.USER, "question"),)), provider_context
        )

    assert caught.value.category is ProviderErrorCategory.INVALID_REQUEST
    assert fallback.call_count == 0


async def test_embedding_fallback_must_match_exact_active_space(
    provider_context: ProviderCallContext,
) -> None:
    incompatible = DeterministicEmbeddingProvider(
        EmbeddingSpaceIdentity("fake", "other", 4, NormalizationPolicy.NONE, "v1")
    )
    required = EmbeddingSpaceIdentity("fake", "active", 4, NormalizationPolicy.NONE, "v1")
    compatible = DeterministicEmbeddingProvider(required)
    router = ModelProviderRouter(
        embedding_routes=(
            ProviderRoute("incompatible-primary", incompatible, POLICY),
            ProviderRoute("compatible-fallback", compatible, POLICY),
        )
    )

    result = await router.embed(
        EmbeddingRequest(("query",)), provider_context, required_space=required
    )

    assert result.used_fallback
    assert incompatible.call_count == 0
    assert compatible.call_count == 1
    assert result.value.identity == required


async def test_no_compatible_embedding_route_makes_no_provider_call(
    provider_context: ProviderCallContext,
) -> None:
    provider = DeterministicEmbeddingProvider()
    router = ModelProviderRouter(embedding_routes=(ProviderRoute("embedding", provider, POLICY),))
    other_space = EmbeddingSpaceIdentity(
        "other", "model", provider.identity.dimension, NormalizationPolicy.L2, "v1"
    )

    with pytest.raises(ProviderOperationError) as caught:
        await router.embed(
            EmbeddingRequest(("query",)), provider_context, required_space=other_space
        )

    assert caught.value.category is ProviderErrorCategory.INCOMPATIBLE_RESPONSE
    assert provider.call_count == 0
    assert caught.value.attempts == ()


async def test_all_reranking_routes_fail_closed_to_base_order(
    provider_context: ProviderCallContext,
) -> None:
    identity = ModelIdentity("fake", "reranker", "v1")
    invalid = InvalidReranker(identity)
    router = ModelProviderRouter(reranking_routes=(ProviderRoute("invalid", invalid, POLICY),))
    request = RerankRequest(
        "query",
        (RerankCandidate("c1", "one"), RerankCandidate("c2", "two")),
    )

    result = await router.rerank(request, provider_context)

    assert result.ordered_ids == ("c1", "c2")
    assert not result.applied
    assert result.degraded
    assert result.degradation_reason is ProviderErrorCategory.INCOMPATIBLE_RESPONSE
    assert len(result.attempts) == 1


async def test_reranking_fallback_can_succeed(
    provider_context: ProviderCallContext,
) -> None:
    invalid = InvalidReranker(ModelIdentity("fake", "invalid", "v1"))
    fallback = DeterministicRerankingProvider()
    router = ModelProviderRouter(
        reranking_routes=(
            ProviderRoute("invalid", invalid, POLICY),
            ProviderRoute("fallback", fallback, POLICY),
        )
    )
    request = RerankRequest(
        "query",
        (RerankCandidate("c1", "one"), RerankCandidate("c2", "two")),
    )

    result = await router.rerank(request, provider_context)

    assert result.applied
    assert not result.degraded
    assert len(result.attempts) == 2
    assert result.attempts[-1].is_fallback


def test_optional_reranking_does_not_block_qa_readiness() -> None:
    router = ModelProviderRouter(
        embedding_routes=(ProviderRoute("embedding", DeterministicEmbeddingProvider(), POLICY),),
        generation_routes=(ProviderRoute("generation", DeterministicGenerationProvider(), POLICY),),
    )

    readiness = {status.role: status for status in router.readiness}
    assert router.qa_ready
    assert not readiness[ProviderRole.RERANKING].ready
    assert readiness[ProviderRole.RERANKING].reason == "reranking_provider_unavailable"


def test_missing_required_roles_make_qa_unready() -> None:
    router = ModelProviderRouter()

    readiness = {status.role: status for status in router.readiness}
    assert not router.qa_ready
    assert readiness[ProviderRole.EMBEDDING].reason == "embedding_provider_unavailable"
    assert readiness[ProviderRole.GENERATION].reason == "generation_provider_unavailable"
