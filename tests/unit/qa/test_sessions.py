from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag_mvp.domain.qa import ConversationRole, SessionStatus
from rag_mvp.qa.sessions import ConversationService
from rag_mvp.storage.database import Database
from rag_mvp.storage.repositories import (
    RepositoryConflict,
    SessionOwnershipError,
    SessionRepository,
)

NOW = datetime(2026, 8, 6, 12, tzinfo=UTC)


@pytest.fixture
def conversations(tmp_path: Path) -> tuple[ConversationService, Database]:
    database = Database(tmp_path / "metadata.sqlite3")
    database.initialize()
    session_ids = iter(("session-a", "session-b"))
    turn_ids = iter(("turn-a-1", "turn-a-2", "turn-b-1", "turn-reset"))
    service = ConversationService(
        SessionRepository(database),
        session_id_factory=lambda: next(session_ids),
        turn_id_factory=lambda: next(turn_ids),
        clock=lambda: NOW,
    )
    return service, database


def test_sessions_store_ordered_turns_without_cross_session_state(
    conversations: tuple[ConversationService, Database],
) -> None:
    service, database = conversations
    first = service.create_session("owner-a")
    second = service.create_session("owner-b")

    first_user = service.append_turn(
        first.session_id,
        "owner-a",
        ConversationRole.USER,
        "What is the leave policy?",
    )
    first_assistant = service.append_turn(
        first.session_id,
        "owner-a",
        ConversationRole.ASSISTANT,
        "I will retrieve fresh evidence.",
    )
    second_user = service.append_turn(
        second.session_id,
        "owner-b",
        ConversationRole.USER,
        "年假政策是什么?",
    )

    assert (first_user.ordinal, first_assistant.ordinal) == (0, 1)
    assert service.list_turns(first.session_id, "owner-a") == (first_user, first_assistant)
    assert service.list_turns(second.session_id, "owner-b") == (second_user,)

    reopened = ConversationService(SessionRepository(Database(database.path)))
    assert reopened.get_session(first.session_id, "owner-a") == first
    assert reopened.list_turns(first.session_id, "owner-a") == (first_user, first_assistant)


def test_every_session_operation_enforces_ownership(
    conversations: tuple[ConversationService, Database],
) -> None:
    service, _ = conversations
    session = service.create_session("owner-a")

    with pytest.raises(SessionOwnershipError):
        service.get_session(session.session_id, "owner-b")
    with pytest.raises(SessionOwnershipError):
        service.list_turns(session.session_id, "owner-b")
    with pytest.raises(SessionOwnershipError):
        service.append_turn(
            session.session_id,
            "owner-b",
            ConversationRole.USER,
            "Cross-session access",
        )
    with pytest.raises(SessionOwnershipError):
        service.reset_session(session.session_id, "owner-b")

    assert service.list_turns(session.session_id, "owner-a") == ()


def test_reset_hides_history_and_makes_the_session_terminal(
    conversations: tuple[ConversationService, Database],
) -> None:
    service, _ = conversations
    session = service.create_session("owner-a")
    service.append_turn(
        session.session_id,
        "owner-a",
        ConversationRole.USER,
        "Remember this turn",
    )

    reset = service.reset_session(session.session_id, "owner-a")

    assert reset.status is SessionStatus.RESET
    assert service.list_turns(session.session_id, "owner-a") == ()
    with pytest.raises(RepositoryConflict, match="reset session"):
        service.append_turn(
            session.session_id,
            "owner-a",
            ConversationRole.USER,
            "This must start in a new session",
        )
