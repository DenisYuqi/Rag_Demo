"""Replaceable, provider-neutral model contracts and adapters."""

from rag_mvp.providers.bge_adapters import (
    LocalBgeEmbeddingProvider,
    LocalBgeRerankingProvider,
)
from rag_mvp.providers.errors import (
    ProviderConfigurationError,
    ProviderError,
    ProviderOperationError,
)
from rag_mvp.providers.models import (
    ChatMessage,
    Deadline,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingSpaceIdentity,
    GenerationRequest,
    GenerationResult,
    ModelIdentity,
    ProviderCallContext,
    ProviderErrorCategory,
    ProviderRole,
    RerankCandidate,
    RerankRequest,
    RerankResult,
    TokenUsage,
)
from rag_mvp.providers.protocols import (
    EmbeddingProvider,
    GenerationProvider,
    RerankingProvider,
)

__all__ = [
    "ChatMessage",
    "Deadline",
    "EmbeddingProvider",
    "EmbeddingRequest",
    "EmbeddingResult",
    "EmbeddingSpaceIdentity",
    "GenerationProvider",
    "GenerationRequest",
    "GenerationResult",
    "LocalBgeEmbeddingProvider",
    "LocalBgeRerankingProvider",
    "ModelIdentity",
    "ProviderCallContext",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderErrorCategory",
    "ProviderOperationError",
    "ProviderRole",
    "RerankCandidate",
    "RerankRequest",
    "RerankResult",
    "RerankingProvider",
    "TokenUsage",
]
