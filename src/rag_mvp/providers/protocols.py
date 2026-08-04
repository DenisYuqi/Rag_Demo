"""Asynchronous provider contracts used by ingestion, retrieval, and QA."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_mvp.providers.models import (
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingSpaceIdentity,
    GenerationRequest,
    GenerationResult,
    ModelAttempt,
    ModelIdentity,
    ProviderCallContext,
    RerankRequest,
    RerankResult,
)


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def identity(self) -> EmbeddingSpaceIdentity: ...

    async def embed(
        self, request: EmbeddingRequest, context: ProviderCallContext
    ) -> EmbeddingResult: ...


@runtime_checkable
class GenerationProvider(Protocol):
    @property
    def identity(self) -> ModelIdentity: ...

    async def generate(
        self, request: GenerationRequest, context: ProviderCallContext
    ) -> GenerationResult: ...


@runtime_checkable
class RerankingProvider(Protocol):
    @property
    def identity(self) -> ModelIdentity: ...

    async def rerank(
        self, request: RerankRequest, context: ProviderCallContext
    ) -> RerankResult: ...


class AttemptRecorder(Protocol):
    def record(self, attempt: ModelAttempt) -> None: ...


class NullAttemptRecorder:
    """Default recorder for applications that have not composed persistence yet."""

    def record(self, attempt: ModelAttempt) -> None:
        del attempt

