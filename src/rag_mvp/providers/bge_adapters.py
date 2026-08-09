"""Local BGE embedding and reranking adapters with lazy bounded inference."""

from __future__ import annotations

import asyncio
import importlib
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from rag_mvp.providers.errors import ProviderError, classify_provider_exception
from rag_mvp.providers.models import (
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingSpaceIdentity,
    ModelIdentity,
    ProviderCallContext,
    ProviderErrorCategory,
    RerankRequest,
    RerankResult,
)

ModelFactory = Callable[[], object]


def _flag_embedding_class(name: str) -> type[Any]:
    module = importlib.import_module("FlagEmbedding")
    model_class = getattr(module, name, None)
    if not isinstance(model_class, type):
        raise RuntimeError("bge_model_class_unavailable")
    return model_class


def _model_kwargs(
    *,
    device: str,
    use_fp16: bool,
    cache_dir: Path | None,
) -> dict[str, object]:
    values: dict[str, object] = {"use_fp16": use_fp16}
    if device != "auto":
        values["devices"] = device
    if cache_dir is not None:
        values["cache_dir"] = str(cache_dir)
    return values


class LocalBgeEmbeddingProvider:
    """Run BGE-M3 dense embedding locally without blocking the event loop."""

    def __init__(
        self,
        identity: EmbeddingSpaceIdentity,
        *,
        device: str = "auto",
        use_fp16: bool = False,
        batch_size: int = 8,
        max_length: int = 8192,
        cache_dir: Path | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        if identity.normalization.value != "l2":
            raise ValueError("BGE embedding identity must declare L2 normalization")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must not be empty")
        if isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if isinstance(max_length, bool) or max_length <= 0:
            raise ValueError("max_length must be positive")
        self._identity = identity
        self._batch_size = batch_size
        self._max_length = max_length
        self._model: object | None = None
        self._inference_lock = threading.Lock()
        self._model_factory = model_factory or (
            lambda: _flag_embedding_class("BGEM3FlagModel")(
                identity.model,
                normalize_embeddings=True,
                **_model_kwargs(device=device, use_fp16=use_fp16, cache_dir=cache_dir),
            )
        )

    @property
    def identity(self) -> EmbeddingSpaceIdentity:
        return self._identity

    async def embed(
        self,
        request: EmbeddingRequest,
        context: ProviderCallContext,
    ) -> EmbeddingResult:
        if context.deadline.expired:
            raise ProviderError(
                ProviderErrorCategory.DEADLINE_EXCEEDED,
                retryable=False,
                fallback_eligible=False,
            )
        try:
            vectors = await asyncio.to_thread(self._embed_sync, request.texts)
            return EmbeddingResult(vectors=vectors, identity=self.identity)
        except ProviderError:
            raise
        except Exception as error:
            raise classify_provider_exception(error) from None

    def _embed_sync(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        with self._inference_lock:
            model = self._load_model()
            encode = getattr(model, "encode", None)
            if not callable(encode):
                raise ProviderError(ProviderErrorCategory.INCOMPATIBLE_RESPONSE)
            output = encode(
                list(texts),
                batch_size=self._batch_size,
                max_length=self._max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        if not isinstance(output, Mapping) or "dense_vecs" not in output:
            raise ProviderError(ProviderErrorCategory.INCOMPATIBLE_RESPONSE)
        raw_vectors = _as_sequence(output["dense_vecs"])
        if len(raw_vectors) != len(texts):
            raise ProviderError(ProviderErrorCategory.INCOMPATIBLE_RESPONSE)
        return tuple(self._validated_vector(raw) for raw in raw_vectors)

    def _validated_vector(self, raw: object) -> tuple[float, ...]:
        values = _as_sequence(raw)
        if len(values) != self.identity.dimension:
            raise ProviderError(ProviderErrorCategory.INCOMPATIBLE_RESPONSE)
        try:
            vector = tuple(float(cast(Any, value)) for value in values)
        except (TypeError, ValueError):
            raise ProviderError(ProviderErrorCategory.INCOMPATIBLE_RESPONSE) from None
        if any(not math.isfinite(value) for value in vector):
            raise ProviderError(ProviderErrorCategory.INCOMPATIBLE_RESPONSE)
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or norm <= 0:
            raise ProviderError(ProviderErrorCategory.INCOMPATIBLE_RESPONSE)
        return tuple(value / norm for value in vector)

    def _load_model(self) -> object:
        if self._model is None:
            self._model = self._model_factory()
        return self._model

    def warmup(self) -> None:
        """Load model assets and weights before the profile accepts chat traffic."""

        with self._inference_lock:
            self._load_model()

    def close(self) -> None:
        with self._inference_lock:
            self._model = None


class LocalBgeRerankingProvider:
    """Run a local BGE cross-encoder and return one stable exact permutation."""

    def __init__(
        self,
        identity: ModelIdentity,
        *,
        device: str = "auto",
        use_fp16: bool = False,
        batch_size: int = 8,
        max_length: int = 1024,
        max_candidates: int = 10,
        cache_dir: Path | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must not be empty")
        for name, value in (
            ("batch_size", batch_size),
            ("max_length", max_length),
            ("max_candidates", max_candidates),
        ):
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        self._identity = identity
        self._batch_size = batch_size
        self._max_length = max_length
        self._max_candidates = max_candidates
        self._model: object | None = None
        self._inference_lock = threading.Lock()
        self._model_factory = model_factory or (
            lambda: _flag_embedding_class("FlagReranker")(
                identity.model,
                **_model_kwargs(device=device, use_fp16=use_fp16, cache_dir=cache_dir),
            )
        )

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    async def rerank(
        self,
        request: RerankRequest,
        context: ProviderCallContext,
    ) -> RerankResult:
        if len(request.candidates) > self._max_candidates:
            raise ProviderError(
                ProviderErrorCategory.INVALID_REQUEST,
                retryable=False,
                fallback_eligible=False,
            )
        if context.deadline.expired:
            raise ProviderError(
                ProviderErrorCategory.DEADLINE_EXCEEDED,
                retryable=False,
                fallback_eligible=False,
            )
        try:
            scores = await asyncio.to_thread(self._score_sync, request)
            ranked = sorted(
                enumerate(request.candidates),
                key=lambda item: (-scores[item[0]], item[0]),
            )
            return RerankResult(
                ordered_ids=tuple(candidate.candidate_id for _, candidate in ranked),
                identity=self.identity,
                prompt_version=request.prompt_version,
            )
        except ProviderError:
            raise
        except Exception as error:
            raise classify_provider_exception(error) from None

    def _score_sync(self, request: RerankRequest) -> tuple[float, ...]:
        pairs = [[request.query, candidate.text] for candidate in request.candidates]
        with self._inference_lock:
            model = self._load_model()
            compute_score = getattr(model, "compute_score", None)
            if not callable(compute_score):
                raise ProviderError(ProviderErrorCategory.INCOMPATIBLE_RESPONSE)
            output = compute_score(
                pairs,
                batch_size=self._batch_size,
                max_length=self._max_length,
                normalize=False,
            )
        raw_scores = [output] if isinstance(output, int | float) else _as_sequence(output)
        if len(raw_scores) != len(request.candidates):
            raise ProviderError(ProviderErrorCategory.INCOMPATIBLE_RESPONSE)
        try:
            scores = tuple(float(cast(Any, score)) for score in raw_scores)
        except (TypeError, ValueError):
            raise ProviderError(ProviderErrorCategory.INCOMPATIBLE_RESPONSE) from None
        if any(not math.isfinite(score) for score in scores):
            raise ProviderError(ProviderErrorCategory.INCOMPATIBLE_RESPONSE)
        return scores

    def _load_model(self) -> object:
        if self._model is None:
            self._model = self._model_factory()
        return self._model

    def warmup(self) -> None:
        """Load model assets and weights before the profile accepts chat traffic."""

        with self._inference_lock:
            self._load_model()

    def close(self) -> None:
        with self._inference_lock:
            self._model = None


def _as_sequence(value: object) -> Sequence[object]:
    if isinstance(value, str | bytes | bytearray):
        raise ProviderError(ProviderErrorCategory.INCOMPATIBLE_RESPONSE)
    if isinstance(value, Sequence):
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if isinstance(converted, Sequence) and not isinstance(converted, str | bytes | bytearray):
            return converted
    raise ProviderError(ProviderErrorCategory.INCOMPATIBLE_RESPONSE)
