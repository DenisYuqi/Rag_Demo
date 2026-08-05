from __future__ import annotations

import httpx
import pytest

from rag_mvp.providers.openai_client import (
    OpenAIClientConfig,
    create_async_openai_client,
)


def test_client_configuration_masks_secret_and_supports_custom_url() -> None:
    raw_secret = "sk-do-not-log-this-value"
    raw_proxy = "http://proxy-user:proxy-password@127.0.0.1:7890"
    config = OpenAIClientConfig(
        base_url="https://compatible.example/v1/",
        api_key=raw_secret,
        secret_reference="env:RAG_MVP_OPENAI_API_KEY",
        timeout_seconds=2,
        proxy_url=raw_proxy,
    )

    assert raw_secret not in repr(config)
    assert raw_proxy not in repr(config)
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


async def test_factory_applies_configuration_to_mocked_http() -> None:
    requests: list[httpx.Request] = []

    async def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"object": "list", "data": []})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
    client = create_async_openai_client(
        OpenAIClientConfig(
            base_url="https://compatible.example/v1/",
            api_key="sk-mocked",
            secret_reference="env:OPENAI_API_KEY",
            timeout_seconds=2,
        ),
        http_client=http_client,
    )
    try:
        await client.models.list()
    finally:
        await client.close()

    assert len(requests) == 1
    assert str(requests[0].url) == "https://compatible.example/v1/models"
    assert requests[0].headers["authorization"] == "Bearer sk-mocked"


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


@pytest.mark.parametrize("proxy_url", ["127.0.0.1:7890", "file:///tmp/proxy"])
def test_client_rejects_unsafe_proxy_urls(proxy_url: str) -> None:
    with pytest.raises(ValueError, match="proxy URL"):
        OpenAIClientConfig(
            "https://example.test/v1",
            "sk-test",
            "env:KEY",
            1,
            proxy_url,
        )
