"""Deterministic offline providers used by unit and integration tests."""

from __future__ import annotations

import hashlib
import math

from rag_mvp.providers.models import (
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingSpaceIdentity,
    FinishReason,
    GenerationRequest,
    GenerationResult,
    ModelIdentity,
    NormalizationPolicy,
    ProviderCallContext,
    RerankRequest,
    RerankResult,
    TokenUsage,
)


def _estimated_tokens(text: str) -> int:
    """Return a stable non-zero estimate suitable only for fake usage records."""

    return max(1, (len(text) + 3) // 4)


def _deterministic_vector(seed: str, text: str, dimension: int) -> tuple[float, ...]:
    values: list[float] = []
    block = 0
    while len(values) < dimension:
        digest = hashlib.sha256(f"{seed}\0{block}\0{text}".encode()).digest()
        values.extend((byte - 127.5) / 127.5 for byte in digest)
        block += 1
    return tuple(values[:dimension])


class DeterministicEmbeddingProvider:
    """Stable hash-based vectors; never suitable for semantic production search."""

    def __init__(
        self,
        identity: EmbeddingSpaceIdentity | None = None,
        *,
        seed: str = "rag-mvp-fake-embedding-v1",
    ) -> None:
        self._identity = identity or EmbeddingSpaceIdentity(
            provider="deterministic-fake",
            model="hash-vector-v1",
            dimension=16,
            normalization=NormalizationPolicy.L2,
            adapter_version="fake-v1",
        )
        self._seed = seed
        self.call_count = 0

    @property
    def identity(self) -> EmbeddingSpaceIdentity:
        return self._identity

    async def embed(
        self, request: EmbeddingRequest, context: ProviderCallContext
    ) -> EmbeddingResult:
        del context
        self.call_count += 1
        vectors: list[tuple[float, ...]] = []
        for text in request.texts:
            vector = _deterministic_vector(self._seed, text, self.identity.dimension)
            if self.identity.normalization is NormalizationPolicy.L2:
                norm = math.sqrt(sum(value * value for value in vector))
                vector = tuple(value / norm for value in vector)
            vectors.append(vector)
        usage = TokenUsage(input_tokens=sum(_estimated_tokens(text) for text in request.texts))
        return EmbeddingResult(tuple(vectors), self.identity, usage)


class DeterministicGenerationProvider:
    """A repeatable generator whose result depends on the complete normalized request."""

    def __init__(
        self,
        identity: ModelIdentity | None = None,
        *,
        prefix: str = "deterministic-answer",
    ) -> None:
        self._identity = identity or ModelIdentity(
            provider="deterministic-fake",
            model="digest-generation-v1",
            adapter_version="fake-v1",
        )
        self._prefix = prefix
        self.call_count = 0

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    async def generate(
        self, request: GenerationRequest, context: ProviderCallContext
    ) -> GenerationResult:
        del context
        self.call_count += 1
        canonical = "\n".join(f"{message.role}:{message.content}" for message in request.messages)
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        content = f"{self._prefix}:{digest}"
        input_tokens = sum(_estimated_tokens(message.content) for message in request.messages)
        return GenerationResult(
            content=content,
            identity=self.identity,
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=_estimated_tokens(content),
            ),
        )


class DeterministicRerankingProvider:
    """Orders supplied IDs with a stable query/content digest."""

    def __init__(self, identity: ModelIdentity | None = None) -> None:
        self._identity = identity or ModelIdentity(
            provider="deterministic-fake",
            model="digest-reranker-v1",
            adapter_version="fake-v1",
        )
        self.call_count = 0

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    async def rerank(
        self, request: RerankRequest, context: ProviderCallContext
    ) -> RerankResult:
        del context
        self.call_count += 1

        def score(item_id: str, text: str) -> tuple[str, str]:
            payload = f"{request.query}\0{item_id}\0{text}".encode()
            return hashlib.sha256(payload).hexdigest(), item_id

        ordered = tuple(
            candidate.candidate_id
            for candidate in sorted(
                request.candidates,
                key=lambda candidate: score(candidate.candidate_id, candidate.text),
            )
        )
        input_text = request.query + "".join(candidate.text for candidate in request.candidates)
        return RerankResult(
            ordered_ids=ordered,
            identity=self.identity,
            prompt_version=request.prompt_version,
            usage=TokenUsage(
                input_tokens=_estimated_tokens(input_text),
                output_tokens=_estimated_tokens("".join(ordered)),
            ),
        )
