from __future__ import annotations

import pytest

from rag_mvp.providers.openai_client import (
    OpenAIClientConfig,
    create_async_openai_client,
)


def test_client_configuration_masks_secret_and_supports_custom_url() -> None:
    raw_secret = "sk-do-not-log-this-value"
    config = OpenAIClientConfig(
        base_url="https://compatible.example/v1/",
        api_key=raw_secret,
        secret_reference="env:RAG_MVP_OPENAI_API_KEY",
        timeout_seconds=2,
    )

    assert raw_secret not in repr(config)
    assert config.secret_reference in repr(config)


async def test_factory_disables_hidden_sdk_retries() -> None:
    config = OpenAIClientConfig(
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        secret_reference="env:OPENAI_API_KEY",
        timeout_seconds=2,
    )

    client = create_async_openai_client(config)
    try:
        assert client.max_retries == 0
        assert str(client.base_url).rstrip("/") == config.base_url
    finally:
        await client.close()


@pytest.mark.parametrize(
    "url",
    ["file:///tmp/provider", "api.openai.com/v1", "https://user:pass@example.test/v1"],
)
def test_client_rejects_unsafe_base_urls(url: str) -> None:
    with pytest.raises(ValueError):
        OpenAIClientConfig(url, "sk-test", "env:KEY", 1)


def test_client_requires_resolved_secret_and_reference() -> None:
    with pytest.raises(ValueError):
        OpenAIClientConfig("https://example.test/v1", "", "env:KEY", 1)
    with pytest.raises(ValueError):
        OpenAIClientConfig("https://example.test/v1", "sk-test", "", 1)

