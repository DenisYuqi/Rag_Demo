from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from rag_mvp.providers.errors import ProviderError
from rag_mvp.providers.models import (
    ChatMessage,
    ChatRole,
    FinishReason,
    GenerationFormat,
    GenerationRequest,
    ModelIdentity,
    ProviderCallContext,
    ProviderErrorCategory,
)
from rag_mvp.providers.openai_adapters import OpenAIChatGenerationProvider


class FakeCreateResource:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return self.response


@dataclass
class FakeChat:
    completions: FakeCreateResource


@dataclass
class FakeClient:
    chat: FakeChat


def provider_for(response: object) -> tuple[OpenAIChatGenerationProvider, FakeCreateResource]:
    resource = FakeCreateResource(response)
    provider = OpenAIChatGenerationProvider(
        FakeClient(FakeChat(resource)),
        ModelIdentity("openai", "chat-model", "v1"),
    )
    return provider, resource


async def test_generation_normalizes_content_finish_and_usage(
    provider_context: ProviderCallContext,
) -> None:
    provider, resource = provider_for(
        {
            "choices": [{"message": {"content": "grounded answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4},
        }
    )
    request = GenerationRequest(
        (ChatMessage(ChatRole.USER, "question"),),
        max_output_tokens=100,
        response_format=GenerationFormat.JSON_OBJECT,
    )

    result = await provider.generate(request, provider_context)

    assert result.content == "grounded answer"
    assert result.finish_reason is FinishReason.STOP
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 4
    assert resource.calls[0]["response_format"] == {"type": "json_object"}
    assert resource.calls[0]["max_tokens"] == 100
    assert resource.calls[0]["n"] == 1
    assert resource.calls[0]["stream"] is False


async def test_missing_usage_remains_unknown(
    provider_context: ProviderCallContext,
) -> None:
    provider, _ = provider_for(
        {"choices": [{"message": {"content": "answer"}, "finish_reason": "novel"}]}
    )

    result = await provider.generate(
        GenerationRequest((ChatMessage(ChatRole.USER, "question"),)), provider_context
    )

    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None
    assert result.finish_reason is FinishReason.UNKNOWN


@pytest.mark.parametrize(
    "response",
    [
        {"choices": []},
        {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
        {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]},
        {
            "choices": [
                {"message": {"content": "one"}},
                {"message": {"content": "two"}},
            ]
        },
        {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": -1}},
    ],
)
async def test_empty_and_malformed_generation_is_rejected(
    provider_context: ProviderCallContext,
    response: object,
) -> None:
    provider, _ = provider_for(response)

    with pytest.raises(ProviderError) as caught:
        await provider.generate(
            GenerationRequest((ChatMessage(ChatRole.USER, "question"),)),
            provider_context,
        )

    assert caught.value.category is ProviderErrorCategory.INCOMPATIBLE_RESPONSE
