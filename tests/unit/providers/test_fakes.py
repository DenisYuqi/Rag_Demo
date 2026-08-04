from __future__ import annotations

import math

import pytest

from rag_mvp.providers.fakes import (
    DeterministicEmbeddingProvider,
    DeterministicGenerationProvider,
    DeterministicRerankingProvider,
)
from rag_mvp.providers.models import (
    ChatMessage,
    ChatRole,
    EmbeddingRequest,
    GenerationRequest,
    ProviderCallContext,
    RerankCandidate,
    RerankRequest,
)


async def test_fake_embeddings_are_reproducible_and_ordered(
    provider_context: ProviderCallContext,
) -> None:
    first_provider = DeterministicEmbeddingProvider()
    second_provider = DeterministicEmbeddingProvider()
    request = EmbeddingRequest(("中文", "English", "中文"))

    first = await first_provider.embed(request, provider_context)
    second = await second_provider.embed(request, provider_context)

    assert first == second
    assert first.vectors[0] == first.vectors[2]
    assert first.vectors[0] != first.vectors[1]
    assert all(
        math.isclose(math.sqrt(sum(value**2 for value in vector)), 1)
        for vector in first.vectors
    )
    assert first_provider.call_count == 1


async def test_fake_generation_is_reproducible(
    provider_context: ProviderCallContext,
) -> None:
    request = GenerationRequest((ChatMessage(ChatRole.USER, "What is the policy?"),))
    provider = DeterministicGenerationProvider()

    first = await provider.generate(request, provider_context)
    second = await provider.generate(request, provider_context)

    assert first == second
    assert first.usage.input_tokens is not None
    assert provider.call_count == 2


async def test_fake_reranking_is_a_stable_permutation(
    provider_context: ProviderCallContext,
) -> None:
    request = RerankRequest(
        "query",
        (
            RerankCandidate("c1", "one"),
            RerankCandidate("c2", "two"),
            RerankCandidate("c3", "three"),
        ),
    )
    provider = DeterministicRerankingProvider()

    first = await provider.rerank(request, provider_context)
    second = await provider.rerank(request, provider_context)

    assert first == second
    assert set(first.ordered_ids) == set(request.candidate_ids)
    assert len(first.ordered_ids) == len(request.candidate_ids)


def test_fake_request_content_is_not_in_repr() -> None:
    secret_text = "sensitive-document-content"

    assert secret_text not in repr(EmbeddingRequest((secret_text,)))
    assert secret_text not in repr(ChatMessage(ChatRole.USER, secret_text))


def test_duplicate_rerank_input_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        RerankRequest(
            "query",
            (RerankCandidate("same", "one"), RerankCandidate("same", "two")),
        )
