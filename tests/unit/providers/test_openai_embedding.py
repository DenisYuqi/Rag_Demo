from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest

from rag_mvp.providers.errors import ProviderError
from rag_mvp.providers.models import (
    EmbeddingRequest,
    EmbeddingSpaceIdentity,
    NormalizationPolicy,
    ProviderCallContext,
    ProviderErrorCategory,
)
from rag_mvp.providers.openai_adapters import OpenAIEmbeddingProvider


class FakeCreateResource:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


@dataclass
class FakeClient:
    embeddings: FakeCreateResource


def identity(
    normalization: NormalizationPolicy = NormalizationPolicy.NONE,
) -> EmbeddingSpaceIdentity:
    return EmbeddingSpaceIdentity("openai", "embedding-model", 2, normalization, "v1")


async def test_embedding_restores_input_order_and_records_usage(
    provider_context: ProviderCallContext,
) -> None:
    resource = FakeCreateResource(
        {
            "data": [
                {"index": 1, "embedding": [3.0, 4.0]},
                {"index": 0, "embedding": [1.0, 2.0]},
            ],
            "usage": {"prompt_tokens": 7},
        }
    )
    provider = OpenAIEmbeddingProvider(FakeClient(resource), identity())

    result = await provider.embed(EmbeddingRequest(("first", "second")), provider_context)

    assert result.vectors == ((1.0, 2.0), (3.0, 4.0))
    assert result.usage.input_tokens == 7
    assert result.usage.output_tokens is None
    assert resource.calls[0]["dimensions"] == 2


async def test_embedding_applies_declared_l2_normalization(
    provider_context: ProviderCallContext,
) -> None:
    resource = FakeCreateResource({"data": [{"index": 0, "embedding": [3.0, 4.0]}]})
    provider = OpenAIEmbeddingProvider(
        FakeClient(resource), identity(NormalizationPolicy.L2)
    )

    result = await provider.embed(EmbeddingRequest(("text",)), provider_context)

    assert result.vectors[0] == pytest.approx((0.6, 0.8))
    assert math.isclose(sum(value**2 for value in result.vectors[0]), 1)


@pytest.mark.parametrize(
    "response",
    [
        {"data": []},
        {"data": [{"index": 0, "embedding": [1.0]}]},
        {"data": [{"index": 0, "embedding": [float("nan"), 1.0]}]},
        {"data": [{"index": 0, "embedding": [float("inf"), 1.0]}]},
        {
            "data": [
                {"index": 0, "embedding": [1.0, 2.0]},
                {"index": 0, "embedding": [3.0, 4.0]},
            ]
        },
    ],
)
async def test_incompatible_embedding_responses_are_rejected(
    provider_context: ProviderCallContext,
    response: dict[str, object],
) -> None:
    provider = OpenAIEmbeddingProvider(FakeClient(FakeCreateResource(response)), identity())
    data = response["data"]
    assert isinstance(data, list)
    input_texts = ("one", "two") if len(data) == 2 else ("one",)

    with pytest.raises(ProviderError) as caught:
        await provider.embed(EmbeddingRequest(input_texts), provider_context)

    assert caught.value.category is ProviderErrorCategory.INCOMPATIBLE_RESPONSE


async def test_zero_vector_cannot_be_l2_normalized(
    provider_context: ProviderCallContext,
) -> None:
    provider = OpenAIEmbeddingProvider(
        FakeClient(FakeCreateResource({"data": [{"index": 0, "embedding": [0.0, 0.0]}]})),
        identity(NormalizationPolicy.L2),
    )

    with pytest.raises(ProviderError, match="provider_incompatible_response"):
        await provider.embed(EmbeddingRequest(("one",)), provider_context)


async def test_raw_provider_error_text_is_not_exposed(
    provider_context: ProviderCallContext,
) -> None:
    raw_secret = "authorization=Bearer-secret-value"
    resource = FakeCreateResource(error=RuntimeError(raw_secret))
    provider = OpenAIEmbeddingProvider(FakeClient(resource), identity())

    with pytest.raises(ProviderError) as caught:
        await provider.embed(EmbeddingRequest(("one",)), provider_context)

    assert raw_secret not in str(caught.value)
    assert caught.value.category is ProviderErrorCategory.UNAVAILABLE
