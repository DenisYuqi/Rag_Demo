"""Atomic complete-response validation, redaction, persistence, and emission."""

from __future__ import annotations

from pydantic import ValidationError

from rag_mvp.domain.qa import (
    ConversationRole,
    QAAnswer,
    QAError,
    QAErrorCode,
    QARefusal,
    QAResponse,
    RefusalReason,
    StreamEventKind,
    ValidatedStreamEvent,
)
from rag_mvp.qa.orchestrator import OrchestratedResponse
from rag_mvp.qa.sessions import ConversationService
from rag_mvp.safety.injection import InjectionPolicy
from rag_mvp.safety.output import SAFE_UNAVAILABLE_MESSAGE, redact_output
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, Redactor
from rag_mvp.safety.streaming import SafeStream


class ResponseReleaseError(ValueError):
    """A content-free reason that an internal response was not releasable."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class CompleteResponseEmitter:
    """Emit exactly one event only after every complete-response gate succeeds."""

    def __init__(
        self,
        conversations: ConversationService,
        *,
        injection_policy: InjectionPolicy | None = None,
        redactor: Redactor | None = DEFAULT_REDACTOR,
        maximum_buffer_characters: int = 65_536,
    ) -> None:
        if type(maximum_buffer_characters) is not int or maximum_buffer_characters < 1:
            raise ValueError("maximum_buffer_characters must be a positive integer")
        self._conversations = conversations
        self._injection_policy = injection_policy or InjectionPolicy()
        self._redactor = redactor
        self._maximum_buffer_characters = maximum_buffer_characters

    @property
    def ready(self) -> bool:
        return self._redactor is not None and self._redactor.fully_configured

    def emit(
        self,
        outcome: OrchestratedResponse,
        *,
        owner_id: str,
    ) -> tuple[ValidatedStreamEvent, ...]:
        if not isinstance(outcome, OrchestratedResponse):
            raise ResponseReleaseError("orchestrated_response_required")
        response = outcome.response
        try:
            self._validate_pipeline_proof(outcome)
            self._validate_output_injection(response)
            redacted = self._redacted_response(response)
            event = self._event(redacted)
            self._persist_if_conversational(redacted, event.content, owner_id)
            return (event,)
        # This service-boundary gate never lets dynamic exception content cross it.
        except Exception:
            return (self._failure_event(response),)

    @staticmethod
    def _validate_pipeline_proof(outcome: OrchestratedResponse) -> None:
        if not outcome.trusted_pipeline_result:
            raise ResponseReleaseError("orchestration_proof_missing")
        response = outcome.response
        if isinstance(response, QAAnswer):
            grounded = outcome.grounded_answer
            if grounded is None:
                raise ResponseReleaseError("grounding_proof_missing")
            expected_answer = grounded.answer
            if outcome.application_suffix is not None:
                expected_answer = f"{expected_answer.rstrip()}\n\n{outcome.application_suffix}"
            if (
                response.request_id != grounded.request_id
                or response.answer != expected_answer
                or response.claims != grounded.claims
                or response.citations != grounded.citations
            ):
                raise ResponseReleaseError("grounding_proof_mismatch")
        elif outcome.grounded_answer is not None or outcome.application_suffix is not None:
            raise ResponseReleaseError("unexpected_grounding_proof")

    def _validate_output_injection(self, response: QAResponse) -> None:
        content = _response_content(response)
        visible_text = [content]
        for citation in response.citations:
            visible_text.extend(
                (
                    citation.source_title,
                    citation.chunk_id,
                    *citation.locator.section_path,
                )
            )
        if isinstance(response, QAAnswer):
            visible_text.extend(claim.text for claim in response.claims)
        for text in visible_text:
            if self._injection_policy.assess_generated_output(text).requires_refusal:
                raise ResponseReleaseError("generated_output_injection")

    def _redacted_response(self, response: QAResponse) -> QAResponse:
        if self._redactor is None or not self._redactor.fully_configured:
            raise ResponseReleaseError("redactor_unavailable")
        payload = redact_output(response, redactor=self._redactor)
        if not isinstance(payload, dict):
            raise ResponseReleaseError("redacted_response_invalid")
        content_field = "answer" if isinstance(response, QAAnswer) else "message"
        payload[content_field] = self._safe_complete_text(_response_content(response))
        if isinstance(response, QAAnswer):
            raw_claims = payload.get("claims")
            if not isinstance(raw_claims, (list, tuple)) or len(raw_claims) != len(response.claims):
                raise ResponseReleaseError("redacted_claims_invalid")
            safe_claim_texts = self._safe_claim_texts(response)
            for raw_claim, safe_text in zip(raw_claims, safe_claim_texts, strict=True):
                if not isinstance(raw_claim, dict):
                    raise ResponseReleaseError("redacted_claims_invalid")
                raw_claim["text"] = safe_text
        raw_citations = payload.get("citations")
        if not isinstance(raw_citations, (list, tuple)) or len(raw_citations) != len(
            response.citations
        ):
            raise ResponseReleaseError("redacted_citations_invalid")
        for raw, original in zip(raw_citations, response.citations, strict=True):
            if not isinstance(raw, dict):
                raise ResponseReleaseError("redacted_citations_invalid")
            raw["source_title"] = self._safe_complete_text(original.source_title)
            raw["chunk_id"] = self._safe_complete_text(original.chunk_id)
            locator = raw.get("locator")
            if not isinstance(locator, dict):
                raise ResponseReleaseError("redacted_citations_invalid")
            locator["section_path"] = [
                self._safe_complete_text(part) for part in original.locator.section_path
            ]
        model_type = type(response)
        try:
            redacted = model_type.model_validate(payload)
        except (TypeError, ValueError, ValidationError):
            raise ResponseReleaseError("redacted_response_invalid") from None
        if not isinstance(redacted, (QAAnswer, QARefusal, QAError)):
            raise ResponseReleaseError("redacted_response_invalid")
        return redacted

    def _safe_claim_texts(self, response: QAAnswer) -> tuple[str, ...]:
        safe_claims = tuple(self._safe_complete_text(claim.text) for claim in response.claims)
        safe_combined = self._safe_complete_text("".join(claim.text for claim in response.claims))
        if "".join(safe_claims) != safe_combined:
            raise ResponseReleaseError("cross_claim_sensitive_value")
        return safe_claims

    def _safe_complete_text(self, text: str) -> str:
        stream = SafeStream(
            redactor=self._redactor,
            max_buffer_chars=self._maximum_buffer_characters,
        )
        pieces = (*stream.push(text), *stream.finish())
        if stream.failed:
            raise ResponseReleaseError(stream.failure_reason or "safe_stream_failed")
        return "".join(pieces)

    @staticmethod
    def _event(response: QAResponse) -> ValidatedStreamEvent:
        if isinstance(response, QAAnswer):
            kind = StreamEventKind.ANSWER
            content = response.answer
            claims = response.claims
            reason = None
            error_code = None
            retryable = None
        elif isinstance(response, QARefusal):
            kind = StreamEventKind.REFUSAL
            content = response.message
            claims = ()
            reason = response.reason
            error_code = None
            retryable = None
        else:
            kind = StreamEventKind.ERROR
            content = response.message
            claims = ()
            reason = None
            error_code = response.code
            retryable = response.retryable
        return ValidatedStreamEvent(
            request_id=response.request_id,
            session_id=response.session_id,
            sequence=0,
            kind=kind,
            response_language=response.response_language,
            content=content,
            claims=claims,
            citations=response.citations,
            reason=reason,
            error_code=error_code,
            retryable=retryable,
            diagnostics=response.diagnostics,
            terminal=True,
        )

    def _persist_if_conversational(
        self,
        response: QAResponse,
        content: str | None,
        owner_id: str,
    ) -> None:
        should_persist = isinstance(response, QAAnswer) or (
            isinstance(response, QARefusal) and response.reason is not RefusalReason.UNSAFE_REQUEST
        )
        if not should_persist:
            return
        if not content:
            raise ResponseReleaseError("released_content_missing")
        self._conversations.append_turn(
            response.session_id,
            owner_id,
            ConversationRole.ASSISTANT,
            content,
        )

    @staticmethod
    def _failure_event(response: QAResponse) -> ValidatedStreamEvent:
        return ValidatedStreamEvent(
            request_id=response.request_id,
            session_id=response.session_id,
            sequence=0,
            kind=StreamEventKind.ERROR,
            response_language=response.response_language,
            content=SAFE_UNAVAILABLE_MESSAGE,
            error_code=QAErrorCode.SAFETY_UNAVAILABLE,
            retryable=True,
            terminal=True,
        )


def _response_content(response: QAResponse) -> str:
    return response.answer if isinstance(response, QAAnswer) else response.message
