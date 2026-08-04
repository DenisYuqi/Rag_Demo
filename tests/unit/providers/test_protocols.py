from __future__ import annotations

from typing import cast

import pytest

from rag_mvp.providers.errors import ProviderConfigurationError
from rag_mvp.providers.fakes import (
    DeterministicEmbeddingProvider,
    DeterministicGenerationProvider,
    DeterministicRerankingProvider,
)
from rag_mvp.providers.protocols import (
    EmbeddingProvider,
    GenerationProvider,
    RerankingProvider,
)
from rag_mvp.providers.resilience import RetryPolicy
from rag_mvp.providers.routing import ModelProviderRouter, ProviderRoute


def test_fakes_satisfy_independent_runtime_contracts() -> None:
    assert isinstance(DeterministicEmbeddingProvider(), EmbeddingProvider)
    assert isinstance(DeterministicGenerationProvider(), GenerationProvider)
    assert isinstance(DeterministicRerankingProvider(), RerankingProvider)


def test_provider_assigned_to_unsupported_role_fails_configuration() -> None:
    generation_only = DeterministicGenerationProvider()
    invalid_route = ProviderRoute(
        "wrong-role",
        cast(EmbeddingProvider, generation_only),
        RetryPolicy(attempt_timeout_seconds=1),
    )

    with pytest.raises(ProviderConfigurationError, match="embedding_capability_missing"):
        ModelProviderRouter(embedding_routes=(invalid_route,))

