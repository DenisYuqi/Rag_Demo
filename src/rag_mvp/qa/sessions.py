"""Owner-scoped conversation session management."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from uuid import uuid4

from rag_mvp.domain._base import utc_now
from rag_mvp.domain.qa import (
    ConversationRole,
    ConversationSession,
    ConversationTurn,
)
from rag_mvp.storage.repositories import SessionRepository


class ConversationService:
    """Persist conversations while enforcing ownership at every entry point."""

    def __init__(
        self,
        sessions: SessionRepository,
        *,
        session_id_factory: Callable[[], str] | None = None,
        turn_id_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._sessions = sessions
        self._session_id_factory = session_id_factory or (lambda: f"session_{uuid4().hex}")
        self._turn_id_factory = turn_id_factory or (lambda: f"turn_{uuid4().hex}")
        self._clock = clock

    def create_session(self, owner_id: str) -> ConversationSession:
        timestamp = self._clock()
        session = ConversationSession(
            session_id=self._session_id_factory(),
            owner_id=owner_id,
            created_at=timestamp,
            updated_at=timestamp,
        )
        self._sessions.create(session)
        return session

    def get_session(self, session_id: str, owner_id: str) -> ConversationSession:
        return self._sessions.require_owned(session_id, owner_id)

    def list_turns(self, session_id: str, owner_id: str) -> tuple[ConversationTurn, ...]:
        self._sessions.require_owned(session_id, owner_id)
        return tuple(self._sessions.list_turns(session_id))

    def append_turn(
        self,
        session_id: str,
        owner_id: str,
        role: ConversationRole,
        content: str,
    ) -> ConversationTurn:
        turn_id = self._turn_id_factory()
        timestamp = self._clock()
        with self._sessions.database.transaction() as connection:
            self._sessions.require_owned(session_id, owner_id, connection=connection)
            ordinal = len(self._sessions.list_turns(session_id, connection=connection))
            turn = ConversationTurn(
                turn_id=turn_id,
                session_id=session_id,
                ordinal=ordinal,
                role=role,
                content=content,
                created_at=timestamp,
            )
            self._sessions.append_turn(turn, connection=connection)
        return turn

    def reset_session(self, session_id: str, owner_id: str) -> ConversationSession:
        with self._sessions.database.transaction() as connection:
            self._sessions.require_owned(session_id, owner_id, connection=connection)
            return self._sessions.reset(session_id, connection=connection)
