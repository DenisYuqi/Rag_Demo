"""Fail-closed conversion between persisted and provider embedding identities."""

from __future__ import annotations

from rag_mvp.domain.ingestion import (
    EmbeddingSpaceIdentity as DomainEmbeddingSpaceIdentity,
)
from rag_mvp.providers.models import (
    EmbeddingSpaceIdentity as ProviderEmbeddingSpaceIdentity,
)
from rag_mvp.providers.models import NormalizationPolicy


class EmbeddingIdentityError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def provider_embedding_identity(
    identity: DomainEmbeddingSpaceIdentity,
) -> ProviderEmbeddingSpaceIdentity:
    """Convert all five persisted identity fields or reject the identity."""

    if not isinstance(identity, DomainEmbeddingSpaceIdentity):
        raise EmbeddingIdentityError("domain_embedding_identity_invalid")
    try:
        normalization = NormalizationPolicy(identity.normalization)
        return ProviderEmbeddingSpaceIdentity(
            provider=identity.provider_alias,
            model=identity.model,
            dimension=identity.dimension,
            normalization=normalization,
            adapter_version=identity.adapter_version,
        )
    except (TypeError, ValueError):
        raise EmbeddingIdentityError("domain_embedding_identity_invalid") from None


def domain_embedding_identity(
    identity: ProviderEmbeddingSpaceIdentity,
) -> DomainEmbeddingSpaceIdentity:
    """Convert all five provider identity fields or reject the identity."""

    if not isinstance(identity, ProviderEmbeddingSpaceIdentity):
        raise EmbeddingIdentityError("provider_embedding_identity_invalid")
    try:
        return DomainEmbeddingSpaceIdentity(
            provider_alias=identity.provider,
            model=identity.model,
            dimension=identity.dimension,
            normalization=identity.normalization.value,
            adapter_version=identity.adapter_version,
        )
    except (AttributeError, TypeError, ValueError):
        raise EmbeddingIdentityError("provider_embedding_identity_invalid") from None
