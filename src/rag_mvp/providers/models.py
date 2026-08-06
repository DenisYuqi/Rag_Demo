"""Provider-neutral request, result, identity, and diagnostic models.

The types in this module deliberately do not expose any vendor SDK objects.  Content
fields are excluded from ``repr`` so an innocent debug representation cannot disclose
prompts, document text, questions, or answers.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum


def _require_text(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_non_negative_int(value: int | None, field_name: str) -> None:
    if value is not None and (isinstance(value, bool) or value < 0):
        raise ValueError(f"{field_name} must be a non-negative integer or unknown")


class ProviderRole(StrEnum):
    """Independent model roles understood by the RAG domain."""

    EMBEDDING = "embedding"
    GENERATION = "generation"
    RERANKING = "reranking"


class ProviderErrorCategory(StrEnum):
    """Safe, stable provider error taxonomy.

    The categories are suitable for APIs and telemetry.  Raw exception messages and
    provider response bodies are intentionally absent.
    """

    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SERVER = "server"
    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    INCOMPATIBLE_RESPONSE = "incompatible_response"
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    UNAVAILABLE = "unavailable"


class AttemptStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NormalizationPolicy(StrEnum):
    NONE = "none"
    L2 = "l2"


class ChatRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class GenerationFormat(StrEnum):
    TEXT = "text"
    JSON_OBJECT = "json_object"


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    provider: str
    model: str
    adapter_version: str

    def __post_init__(self) -> None:
        _require_text(self.provider, "provider")
        _require_text(self.model, "model")
        _require_text(self.adapter_version, "adapter_version")


@dataclass(frozen=True, slots=True)
class EmbeddingSpaceIdentity:
    """Complete identity of a vector space, independent of a concrete route."""

    provider: str
    model: str
    dimension: int
    normalization: NormalizationPolicy
    adapter_version: str

    def __post_init__(self) -> None:
        _require_text(self.provider, "provider")
        _require_text(self.model, "model")
        _require_text(self.adapter_version, "adapter_version")
        if isinstance(self.dimension, bool) or self.dimension <= 0:
            raise ValueError("embedding dimension must be positive")

    @property
    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(self.provider, self.model, self.adapter_version)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Provider-reported usage; ``None`` means unknown and is never coerced to zero."""

    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int(self.input_tokens, "input_tokens")
        _require_non_negative_int(self.output_tokens, "output_tokens")

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


UNKNOWN_USAGE = TokenUsage()


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    texts: tuple[str, ...] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "texts", tuple(self.texts))
        if not self.texts:
            raise ValueError("embedding request must contain at least one text")
        if any(not isinstance(text, str) for text in self.texts):
            raise TypeError("embedding inputs must be strings")


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vectors: tuple[tuple[float, ...], ...] = field(repr=False)
    identity: EmbeddingSpaceIdentity
    usage: TokenUsage = UNKNOWN_USAGE

    def __post_init__(self) -> None:
        vectors = tuple(tuple(vector) for vector in self.vectors)
        object.__setattr__(self, "vectors", vectors)
        if not vectors:
            raise ValueError("embedding result must contain vectors")
        for vector in vectors:
            if len(vector) != self.identity.dimension:
                raise ValueError("embedding vector dimension is incompatible")
            if any(not math.isfinite(value) for value in vector):
                raise ValueError("embedding vector contains a non-finite number")


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: ChatRole
    content: str = field(repr=False)

    def __post_init__(self) -> None:
        _require_text(self.content, "message content")


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    messages: tuple[ChatMessage, ...] = field(repr=False)
    max_output_tokens: int = 512
    temperature: float = 0.0
    response_format: GenerationFormat = GenerationFormat.TEXT
    prompt_version: str = "generation-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        if not self.messages:
            raise ValueError("generation request must contain messages")
        if isinstance(self.max_output_tokens, bool) or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be finite and between 0 and 2")
        _require_text(self.prompt_version, "prompt_version")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    content: str = field(repr=False)
    identity: ModelIdentity
    finish_reason: FinishReason
    usage: TokenUsage = UNKNOWN_USAGE

    def __post_init__(self) -> None:
        _require_text(self.content, "generation content")


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    candidate_id: str
    text: str = field(repr=False)

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        _require_text(self.text, "candidate text")


@dataclass(frozen=True, slots=True)
class RerankRequest:
    query: str = field(repr=False)
    candidates: tuple[RerankCandidate, ...] = field(repr=False)
    prompt_version: str = "listwise-rerank-v1"
    max_query_tokens: int = 256
    max_candidate_tokens: int = 512

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        _require_text(self.query, "rerank query")
        _require_text(self.prompt_version, "prompt_version")
        if not self.candidates:
            raise ValueError("reranking requires at least one candidate")
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("rerank candidate IDs must be unique")
        if isinstance(self.max_query_tokens, bool) or self.max_query_tokens <= 0:
            raise ValueError("max_query_tokens must be positive")
        if isinstance(self.max_candidate_tokens, bool) or self.max_candidate_tokens <= 0:
            raise ValueError("max_candidate_tokens must be positive")

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)


@dataclass(frozen=True, slots=True)
class RerankResult:
    ordered_ids: tuple[str, ...]
    identity: ModelIdentity
    prompt_version: str
    usage: TokenUsage = UNKNOWN_USAGE

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordered_ids", tuple(self.ordered_ids))
        _require_text(self.prompt_version, "prompt_version")
        if not self.ordered_ids:
            raise ValueError("rerank result must not be empty")
        if any(
            not isinstance(candidate_id, str) or not candidate_id.strip()
            for candidate_id in self.ordered_ids
        ):
            raise ValueError("rerank result IDs must be non-empty strings")
        if not isinstance(self.identity, ModelIdentity):
            raise TypeError("rerank identity must be a ModelIdentity")
        if not isinstance(self.usage, TokenUsage):
            raise TypeError("rerank usage must be TokenUsage")


@dataclass(frozen=True, slots=True)
class Deadline:
    """Monotonic absolute deadline shared across nested provider calls."""

    expires_at: float
    clock: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.expires_at):
            raise ValueError("deadline must be finite")

    @classmethod
    def after(cls, seconds: float, *, clock: Callable[[], float] = time.monotonic) -> Deadline:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("deadline duration must be positive and finite")
        return cls(clock() + seconds, clock)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.expires_at - self.clock())

    @property
    def expired(self) -> bool:
        return self.remaining_seconds <= 0


@dataclass(frozen=True, slots=True)
class ProviderCallContext:
    request_id: str
    operation_id: str
    deadline: Deadline

    def __post_init__(self) -> None:
        _require_text(self.request_id, "request_id")
        _require_text(self.operation_id, "operation_id")


@dataclass(frozen=True, slots=True)
class RouteMetadata:
    route_id: str
    role: ProviderRole
    identity: ModelIdentity

    def __post_init__(self) -> None:
        _require_text(self.route_id, "route_id")


@dataclass(frozen=True, slots=True)
class ModelAttempt:
    request_id: str
    operation_id: str
    attempt_number: int
    route_id: str
    role: ProviderRole
    provider: str
    model: str
    latency_ms: float
    status: AttemptStatus
    is_fallback: bool
    usage: TokenUsage = UNKNOWN_USAGE
    error_category: ProviderErrorCategory | None = None

    def __post_init__(self) -> None:
        if isinstance(self.attempt_number, bool) or self.attempt_number <= 0:
            raise ValueError("attempt_number must be positive")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency_ms must be finite and non-negative")
        if self.status is AttemptStatus.SUCCEEDED and self.error_category is not None:
            raise ValueError("successful attempt cannot contain an error category")
        if self.status is not AttemptStatus.SUCCEEDED and self.error_category is None:
            raise ValueError("failed or cancelled attempt requires an error category")


@dataclass(frozen=True, slots=True)
class RoleReadiness:
    role: ProviderRole
    ready: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.ready and self.reason is not None:
            raise ValueError("ready role cannot contain a failure reason")
        if not self.ready and not self.reason:
            raise ValueError("unready role requires a safe reason")


@dataclass(frozen=True, slots=True)
class AttemptedResult[T]:
    value: T
    attempts: tuple[ModelAttempt, ...]


@dataclass(frozen=True, slots=True)
class RoutedResult[T]:
    value: T
    attempts: tuple[ModelAttempt, ...]
    used_fallback: bool


@dataclass(frozen=True, slots=True)
class RoutedRerankResult:
    ordered_ids: tuple[str, ...]
    attempts: tuple[ModelAttempt, ...]
    applied: bool
    degraded: bool
    degradation_reason: ProviderErrorCategory | None = None
    route_id: str | None = None
    identity: ModelIdentity | None = None
    prompt_version: str | None = None
    usage: TokenUsage | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordered_ids", tuple(self.ordered_ids))
        object.__setattr__(self, "attempts", tuple(self.attempts))
        if self.applied and self.degraded:
            raise ValueError("an applied rerank cannot be degraded")
        if self.degraded and self.degradation_reason is None:
            raise ValueError("degraded reranking requires a reason")
        if any(not isinstance(attempt, ModelAttempt) for attempt in self.attempts):
            raise TypeError("reranking attempts must be ModelAttempt values")
        if self.route_id is not None:
            _require_text(self.route_id, "route_id")
        if self.identity is not None and not isinstance(self.identity, ModelIdentity):
            raise TypeError("reranking identity must be a ModelIdentity")
        if self.prompt_version is not None:
            _require_text(self.prompt_version, "prompt_version")
        if self.usage is not None and not isinstance(self.usage, TokenUsage):
            raise TypeError("reranking usage must be TokenUsage")
