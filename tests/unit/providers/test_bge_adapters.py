from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from typing import Any

import pytest

from rag_mvp.providers.bge_adapters import (
    LocalBgeEmbeddingProvider,
    LocalBgeRerankingProvider,
)
from rag_mvp.providers.errors import ProviderError
from rag_mvp.providers.models import (
    EmbeddingRequest,
    EmbeddingSpaceIdentity,
    ModelIdentity,
    NormalizationPolicy,
    ProviderCallContext,
    ProviderErrorCategory,
    RerankCandidate,
    RerankRequest,
)


class FakeEmbeddingModel:
    def __init__(self, vectors: object) -> None:
        self.vectors = vectors
        self.calls: list[dict[str, Any]] = []
        self.thread_id: int | None = None

    def encode(self, texts: Sequence[str], **kwargs: Any) -> dict[str, object]:
        self.thread_id = threading.get_ident()
        self.calls.append({"texts": list(texts), **kwargs})
        return {"dense_vecs": self.vectors}


class FakeRerankerModel:
    def __init__(self, scores: object) -> None:
        self.scores = scores
        self.calls: list[dict[str, Any]] = []
        self.thread_id: int | None = None

    def compute_score(self, pairs: object, **kwargs: Any) -> object:
        self.thread_id = threading.get_ident()
        self.calls.append({"pairs": pairs, **kwargs})
        return self.scores


def embedding_identity() -> EmbeddingSpaceIdentity:
    return EmbeddingSpaceIdentity(
        provider="bge-local",
        model="BAAI/bge-m3",
        dimension=3,
        normalization=NormalizationPolicy.L2,
        adapter_version="flag-embedding-v1",
    )


def reranking_identity() -> ModelIdentity:
    return ModelIdentity(
        provider="bge-local",
        model="BAAI/bge-reranker-v2-m3",
        adapter_version="flag-embedding-v1",
    )


async def test_embedding_is_lazy_ordered_normalized_and_off_event_loop(
    provider_context: ProviderCallContext,
) -> None:
    model = FakeEmbeddingModel([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    loads = 0

    def factory() -> object:
        nonlocal loads
        loads += 1
        return model

    provider = LocalBgeEmbeddingProvider(
        embedding_identity(),
        batch_size=2,
        max_length=128,
        model_factory=factory,
    )
    event_loop_thread = threading.get_ident()

    first = await provider.embed(EmbeddingRequest(("first", "second")), provider_context)
    model.vectors = [[3.0, 4.0, 0.0]]
    second = await provider.embed(EmbeddingRequest(("third",)), provider_context)

    assert loads == 1
    assert first.identity == embedding_identity()
    assert first.vectors[0] == pytest.approx((0.6, 0.8, 0.0))
    assert first.vectors[1] == pytest.approx((0.0, 0.0, 1.0))
    assert second.vectors[0] == pytest.approx((0.6, 0.8, 0.0))
    assert model.thread_id != event_loop_thread
    assert model.calls[0] == {
        "texts": ["first", "second"],
        "batch_size": 2,
        "max_length": 128,
        "return_dense": True,
        "return_sparse": False,
        "return_colbert_vecs": False,
    }


@pytest.mark.parametrize(
    "vectors",
    [
        [],
        [[1.0, 2.0]],
        [[0.0, 0.0, 0.0]],
        [[math.nan, 0.0, 1.0]],
        [[math.inf, 0.0, 1.0]],
        "invalid",
    ],
)
async def test_embedding_rejects_incompatible_output(
    provider_context: ProviderCallContext,
    vectors: object,
) -> None:
    provider = LocalBgeEmbeddingProvider(
        embedding_identity(),
        model_factory=lambda: FakeEmbeddingModel(vectors),
    )

    with pytest.raises(ProviderError) as caught:
        await provider.embed(EmbeddingRequest(("text",)), provider_context)

    assert caught.value.category is ProviderErrorCategory.INCOMPATIBLE_RESPONSE


async def test_embedding_failure_is_safely_classified(
    provider_context: ProviderCallContext,
) -> None:
    secret = "private-model-loader-detail"
    provider = LocalBgeEmbeddingProvider(
        embedding_identity(),
        model_factory=lambda: (_ for _ in ()).throw(RuntimeError(secret)),
    )

    with pytest.raises(ProviderError) as caught:
        await provider.embed(EmbeddingRequest(("text",)), provider_context)

    assert caught.value.category is ProviderErrorCategory.UNAVAILABLE
    assert secret not in str(caught.value)


async def test_reranker_orders_scores_stably_and_runs_off_event_loop(
    provider_context: ProviderCallContext,
) -> None:
    model = FakeRerankerModel([0.5, 0.9, 0.5])
    provider = LocalBgeRerankingProvider(
        reranking_identity(),
        batch_size=3,
        max_length=256,
        max_candidates=3,
        model_factory=lambda: model,
    )
    request = RerankRequest(
        "query",
        (
            RerankCandidate("c1", "one"),
            RerankCandidate("c2", "two"),
            RerankCandidate("c3", "three"),
        ),
        prompt_version="bge-rerank-v1",
    )
    event_loop_thread = threading.get_ident()

    result = await provider.rerank(request, provider_context)

    assert result.ordered_ids == ("c2", "c1", "c3")
    assert result.identity == reranking_identity()
    assert result.prompt_version == "bge-rerank-v1"
    assert model.thread_id != event_loop_thread
    assert model.calls == [
        {
            "pairs": [["query", "one"], ["query", "two"], ["query", "three"]],
            "batch_size": 3,
            "max_length": 256,
            "normalize": False,
        }
    ]


@pytest.mark.parametrize("scores", [[], [1.0], [math.nan, 1.0], "invalid"])
async def test_reranker_rejects_incompatible_scores(
    provider_context: ProviderCallContext,
    scores: object,
) -> None:
    provider = LocalBgeRerankingProvider(
        reranking_identity(),
        model_factory=lambda: FakeRerankerModel(scores),
    )
    request = RerankRequest(
        "query",
        (RerankCandidate("c1", "one"), RerankCandidate("c2", "two")),
    )

    with pytest.raises(ProviderError) as caught:
        await provider.rerank(request, provider_context)

    assert caught.value.category is ProviderErrorCategory.INCOMPATIBLE_RESPONSE


async def test_reranker_rejects_candidate_overflow_before_loading(
    provider_context: ProviderCallContext,
) -> None:
    loaded = False

    def factory() -> object:
        nonlocal loaded
        loaded = True
        return FakeRerankerModel([1.0, 0.0])

    provider = LocalBgeRerankingProvider(
        reranking_identity(), max_candidates=1, model_factory=factory
    )
    request = RerankRequest(
        "query",
        (RerankCandidate("c1", "one"), RerankCandidate("c2", "two")),
    )

    with pytest.raises(ProviderError) as caught:
        await provider.rerank(request, provider_context)

    assert caught.value.category is ProviderErrorCategory.INVALID_REQUEST
    assert not loaded
