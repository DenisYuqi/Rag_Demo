from __future__ import annotations

import os

import pytest

from rag_mvp.config.settings import Settings
from rag_mvp.providers.models import (
    ChatMessage,
    ChatRole,
    Deadline,
    EmbeddingRequest,
    EmbeddingSpaceIdentity,
    GenerationRequest,
    ModelIdentity,
    NormalizationPolicy,
    ProviderCallContext,
)
from rag_mvp.providers.openai_adapters import (
    OpenAIChatGenerationProvider,
    OpenAIEmbeddingProvider,
)
from rag_mvp.providers.openai_client import (
    OpenAIClientConfig,
    create_async_openai_client,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("RAG_MVP_RUN_LIVE_OPENAI") != "1",
        reason="set RAG_MVP_RUN_LIVE_OPENAI=1 to call the paid OpenAI API",
    ),
]


async def test_openai_embedding_and_generation_smoke() -> None:
    settings = Settings()
    if settings.openai_api_key is None:
        pytest.fail("RAG_MVP_OPENAI_API_KEY is required")

    proxy_url = (
        settings.openai_proxy_url.get_secret_value()
        if settings.openai_proxy_url is not None
        else None
    )
    client = create_async_openai_client(
        OpenAIClientConfig(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key.get_secret_value(),
            secret_reference="env:RAG_MVP_OPENAI_API_KEY",
            timeout_seconds=settings.provider_timeout_seconds,
            proxy_url=proxy_url,
        )
    )
    try:
        context = ProviderCallContext(
            request_id="live-openai-smoke",
            operation_id="live-openai-smoke",
            deadline=Deadline.after(60),
        )
        embedding_provider = OpenAIEmbeddingProvider(
            client,
            EmbeddingSpaceIdentity(
                provider="openai",
                model=settings.embedding_model,
                dimension=settings.embedding_dimension,
                normalization=NormalizationPolicy.NONE,
                adapter_version="openai-compatible-v1",
            ),
        )
        embedding = await embedding_provider.embed(
            EmbeddingRequest(("OpenAI provider smoke test",)),
            context,
        )

        generation_provider = OpenAIChatGenerationProvider(
            client,
            ModelIdentity(
                provider="openai",
                model=settings.generation_model,
                adapter_version="openai-compatible-v1",
            ),
            max_tokens_parameter="max_completion_tokens",
        )
        generation = await generation_provider.generate(
            GenerationRequest(
                messages=(
                    ChatMessage(
                        ChatRole.USER,
                        "Reply with exactly: provider smoke ok",
                    ),
                ),
                max_output_tokens=64,
                temperature=0.0,
            ),
            context,
        )
    finally:
        await client.close()

    assert len(embedding.vectors) == 1
    assert len(embedding.vectors[0]) == settings.embedding_dimension
    assert embedding.identity.model == settings.embedding_model
    assert generation.content.strip()
    assert generation.identity.model == settings.generation_model
