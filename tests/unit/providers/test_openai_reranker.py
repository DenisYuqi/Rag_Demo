from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from rag_mvp.providers.errors import ProviderError
from rag_mvp.providers.models import (
    ModelIdentity,
    ProviderCallContext,
    ProviderErrorCategory,
    RerankCandidate,
    RerankRequest,
)
from rag_mvp.providers.openai_adapters import (
    OpenAIChatGenerationProvider,
    OpenAIListwiseRerankingProvider,
    validate_listwise_json,
)


class FakeCreateResource:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return {
            "choices": [{"message": {"content": self.content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }


@dataclass
class FakeChat:
    completions: FakeCreateResource


@dataclass
class FakeClient:
    chat: FakeChat


def reranker_for(content: str) -> tuple[OpenAIListwiseRerankingProvider, FakeCreateResource]:
    resource = FakeCreateResource(content)
    generation = OpenAIChatGenerationProvider(
        FakeClient(FakeChat(resource)), ModelIdentity("openai", "rerank-chat", "v1")
    )
    return OpenAIListwiseRerankingProvider(generation, max_candidates=3), resource


async def test_valid_listwise_ranking_is_accepted_and_bounded(
    provider_context: ProviderCallContext,
) -> None:
    reranker, resource = reranker_for('{"ordered_ids":["c2","c1"]}')
    request = RerankRequest(
        "query-too-long",
        (RerankCandidate("c1", "abcdef"), RerankCandidate("c2", "uvwxyz")),
        prompt_version="rerank-prompt-v7",
        max_query_tokens=5,
        max_candidate_tokens=3,
    )

    result = await reranker.rerank(request, provider_context)

    assert result.ordered_ids == ("c2", "c1")
    assert result.prompt_version == "rerank-prompt-v7"
    messages = resource.calls[0]["messages"]
    payload = json.loads(messages[1]["content"])
    assert payload["query"] == "query"
    assert [item["text"] for item in payload["candidates"]] == ["abc", "uvw"]
    assert resource.calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    "content",
    [
        '{"ordered_ids":["c1","unknown"]}',
        '{"ordered_ids":["c1","c1"]}',
        '{"ordered_ids":["c1"]}',
        '{"ordered_ids":["c1","c2"],"score":1}',
        '["c1","c2"]',
        "not-json",
    ],
)
def test_unknown_duplicate_missing_or_non_strict_output_is_invalid(content: str) -> None:
    with pytest.raises(ProviderError) as caught:
        validate_listwise_json(content, ("c1", "c2"))

    assert caught.value.category is ProviderErrorCategory.INCOMPATIBLE_RESPONSE


async def test_candidate_count_is_bounded_before_provider_call(
    provider_context: ProviderCallContext,
) -> None:
    reranker, resource = reranker_for('{"ordered_ids":[]}')
    request = RerankRequest(
        "query",
        tuple(RerankCandidate(f"c{index}", "text") for index in range(4)),
    )

    with pytest.raises(ProviderError) as caught:
        await reranker.rerank(request, provider_context)

    assert caught.value.category is ProviderErrorCategory.INVALID_REQUEST
    assert resource.calls == []
