"""Grounded-QA domain contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from rag_mvp.domain._base import (
    DomainModel,
    Identifier,
    NonEmptyText,
    NonNegativeFiniteFloat,
    SafeScalar,
    utc_now,
)
from rag_mvp.domain.evaluation import ProviderAttemptEvidence
from rag_mvp.domain.ingestion import ChunkLocator


class ConversationRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class SessionStatus(StrEnum):
    ACTIVE = "active"
    RESET = "reset"


class RefusalReason(StrEnum):
    INSUFFICIENT_EVIDENCE = "insufficient-evidence"
    CONFLICTING_EVIDENCE = "conflicting-evidence"
    UNSAFE_REQUEST = "unsafe-request"


class QAErrorCode(StrEnum):
    CAPACITY = "capacity"
    INDEX_NOT_READY = "index-not-ready"
    RETRIEVAL_UNAVAILABLE = "retrieval-unavailable"
    DEPENDENCY_FAILURE = "dependency-failure"
    DEADLINE_EXPIRED = "deadline-expired"
    SAFETY_UNAVAILABLE = "safety-unavailable"
    INTERNAL = "internal"


class StreamEventKind(StrEnum):
    ANSWER = "answer"
    SENTENCE = "sentence"
    REFUSAL = "refusal"
    ERROR = "error"
    DONE = "done"


class Citation(DomainModel):
    source_title: Identifier
    document_version: Annotated[int, Field(gt=0)]
    chunk_id: Identifier
    locator: ChunkLocator


class AnswerClaim(DomainModel):
    text: NonEmptyText
    citation_chunk_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def require_citation(self) -> AnswerClaim:
        if not self.citation_chunk_ids:
            raise ValueError("a substantive claim requires at least one citation")
        return self


class SafeQADiagnostics(DomainModel):
    trace_id: str | None = None
    stage_timings_ms: dict[str, NonNegativeFiniteFloat] = Field(default_factory=dict)
    cache_status: dict[str, str] = Field(default_factory=dict)
    model_identities: dict[str, str] = Field(default_factory=dict)
    token_counts: dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    provider_attempts: tuple[ProviderAttemptEvidence, ...] = ()
    degradation_reasons: tuple[str, ...] = ()
    metadata: dict[str, SafeScalar] = Field(default_factory=dict)


class RequestDiagnostic(DomainModel):
    """Privacy-safe persisted request evidence used by the diagnostics API."""

    request_id: Identifier
    session_id: str | None = None
    trace_id: str | None = None
    outcome: Identifier
    safe_error_category: str | None = None
    stage_timings_ms: dict[str, NonNegativeFiniteFloat] = Field(default_factory=dict)
    cache_status: dict[str, str] = Field(default_factory=dict)
    model_identities: dict[str, str] = Field(default_factory=dict)
    token_counts: dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    metadata: dict[str, SafeScalar] = Field(default_factory=dict)
    created_at: AwareDatetime = Field(default_factory=utc_now)
    expires_at: AwareDatetime | None = None


class QAAnswer(DomainModel):
    outcome: Literal["answer"] = "answer"
    request_id: Identifier
    session_id: Identifier
    response_language: Identifier
    answer: NonEmptyText
    claims: tuple[AnswerClaim, ...]
    citations: tuple[Citation, ...]
    diagnostics: SafeQADiagnostics = Field(default_factory=SafeQADiagnostics)

    @model_validator(mode="after")
    def citations_cover_claims(self) -> QAAnswer:
        available = {citation.chunk_id for citation in self.citations}
        referenced = {chunk_id for claim in self.claims for chunk_id in claim.citation_chunk_ids}
        if not referenced.issubset(available):
            raise ValueError("every claim citation must exist in citations")
        return self


class QARefusal(DomainModel):
    outcome: Literal["refusal"] = "refusal"
    request_id: Identifier
    session_id: Identifier
    response_language: Identifier
    message: NonEmptyText
    reason: RefusalReason
    citations: tuple[Citation, ...] = ()
    diagnostics: SafeQADiagnostics = Field(default_factory=SafeQADiagnostics)


class QAError(DomainModel):
    outcome: Literal["error"] = "error"
    request_id: Identifier
    session_id: Identifier
    response_language: Identifier
    message: NonEmptyText
    code: QAErrorCode
    retryable: bool = False
    citations: tuple[Citation, ...] = ()
    diagnostics: SafeQADiagnostics = Field(default_factory=SafeQADiagnostics)


type QAResponse = QAAnswer | QARefusal | QAError


class ConversationSession(DomainModel):
    session_id: Identifier
    owner_id: Identifier
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)
    reset_at: AwareDatetime | None = None


class ConversationTurn(DomainModel):
    turn_id: Identifier
    session_id: Identifier
    ordinal: Annotated[int, Field(ge=0)]
    role: ConversationRole
    content: NonEmptyText
    created_at: AwareDatetime = Field(default_factory=utc_now)


class ValidatedStreamEvent(DomainModel):
    request_id: Identifier
    session_id: Identifier
    sequence: Annotated[int, Field(ge=0)]
    kind: StreamEventKind
    response_language: Identifier
    content: str | None = None
    claims: tuple[AnswerClaim, ...] = ()
    citations: tuple[Citation, ...] = ()
    reason: RefusalReason | None = None
    error_code: QAErrorCode | None = None
    retryable: bool | None = None
    diagnostics: SafeQADiagnostics = Field(default_factory=SafeQADiagnostics)
    terminal: bool = False

    @model_validator(mode="after")
    def validate_event_shape(self) -> ValidatedStreamEvent:
        if self.kind in {StreamEventKind.ANSWER, StreamEventKind.SENTENCE}:
            if not self.content or not self.claims or not self.citations:
                raise ValueError("an answer unit requires content, claims, and citations")
            available = {citation.chunk_id for citation in self.citations}
            referenced = {
                chunk_id for claim in self.claims for chunk_id in claim.citation_chunk_ids
            }
            if not referenced.issubset(available):
                raise ValueError("every event claim citation must exist in citations")
        elif self.claims:
            raise ValueError("only an answer unit can include claims")
        if self.kind is StreamEventKind.ANSWER and not self.terminal:
            raise ValueError("an answer event must be terminal")
        if self.kind is StreamEventKind.SENTENCE and self.terminal:
            raise ValueError("a sentence event cannot be terminal")
        if self.kind is StreamEventKind.DONE and self.content is not None:
            raise ValueError("a done event cannot contain dynamic content")
        if (
            self.kind in {StreamEventKind.REFUSAL, StreamEventKind.ERROR, StreamEventKind.DONE}
            and not self.terminal
        ):
            raise ValueError("refusal, error, and done events must be terminal")
        if self.kind is StreamEventKind.REFUSAL:
            if not self.content or self.reason is None:
                raise ValueError("a refusal event requires content and a reason")
        elif self.reason is not None:
            raise ValueError("only a refusal event can include a refusal reason")
        if self.kind is StreamEventKind.ERROR:
            if not self.content or self.error_code is None or self.retryable is None:
                raise ValueError("an error event requires content, a code, and retryability")
        elif self.error_code is not None or self.retryable is not None:
            raise ValueError("only an error event can include error details")
        return self
