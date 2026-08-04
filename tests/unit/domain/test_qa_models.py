from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_mvp.domain.ingestion import ChunkLocator
from rag_mvp.domain.qa import (
    AnswerClaim,
    Citation,
    ConversationRole,
    ConversationSession,
    ConversationTurn,
    QAAnswer,
    QAError,
    QAErrorCode,
    QARefusal,
    RefusalReason,
    StreamEventKind,
    ValidatedStreamEvent,
)


def _citation() -> Citation:
    return Citation(
        source_title="Employee Handbook",
        document_version=2,
        chunk_id="chunk-1",
        locator=ChunkLocator(pages=(7,)),
    )


def test_answer_requires_resolvable_claim_citations() -> None:
    with pytest.raises(ValidationError):
        QAAnswer(
            request_id="request-1",
            session_id="session-1",
            response_language="zh-CN",
            answer="年假为十天。",
            claims=(AnswerClaim(text="年假为十天。", citation_chunk_ids=("invented",)),),
            citations=(_citation(),),
        )

    answer = QAAnswer(
        request_id="request-1",
        session_id="session-1",
        response_language="zh-CN",
        answer="年假为十天。",
        claims=(AnswerClaim(text="年假为十天。", citation_chunk_ids=("chunk-1",)),),
        citations=(_citation(),),
    )
    assert QAAnswer.model_validate_json(answer.model_dump_json()) == answer


def test_answer_refusal_and_error_have_stable_discriminators() -> None:
    refusal = QARefusal(
        request_id="request-1",
        session_id="session-1",
        response_language="en",
        message="The corpus does not contain that information.",
        reason=RefusalReason.INSUFFICIENT_EVIDENCE,
    )
    error = QAError(
        request_id="request-2",
        session_id="session-1",
        response_language="en",
        message="The request reached its deadline.",
        code=QAErrorCode.DEADLINE_EXPIRED,
        retryable=True,
    )

    assert refusal.outcome == "refusal"
    assert error.outcome == "error"


def test_conversation_models_serialize_without_cross_session_state() -> None:
    session = ConversationSession(session_id="session-1", owner_id="owner-a")
    turn = ConversationTurn(
        turn_id="turn-1",
        session_id=session.session_id,
        ordinal=0,
        role=ConversationRole.USER,
        content="What is the leave policy?",
    )

    assert ConversationSession.model_validate_json(session.model_dump_json()) == session
    assert ConversationTurn.model_validate_json(turn.model_dump_json()) == turn


def test_validated_stream_event_rejects_unsafe_shapes() -> None:
    with pytest.raises(ValidationError):
        ValidatedStreamEvent(
            request_id="request-1",
            session_id="session-1",
            sequence=0,
            kind=StreamEventKind.SENTENCE,
            response_language="en",
        )

    with pytest.raises(ValidationError):
        ValidatedStreamEvent(
            request_id="request-1",
            session_id="session-1",
            sequence=0,
            kind=StreamEventKind.ERROR,
            response_language="en",
            content="safe error",
            terminal=False,
        )

    event = ValidatedStreamEvent(
        request_id="request-1",
        session_id="session-1",
        sequence=0,
        kind=StreamEventKind.SENTENCE,
        response_language="en",
        content="A validated sentence.",
        citations=(_citation(),),
    )
    assert event.terminal is False
