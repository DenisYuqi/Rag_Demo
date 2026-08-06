from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from rag_mvp.domain.ingestion import ChunkLocator
from rag_mvp.domain.qa import (
    AnswerClaim,
    Citation,
    RefusalReason,
    StreamEventKind,
    ValidatedStreamEvent,
)
from rag_mvp.domain.retrieval import RetrievalMode
from rag_mvp.ui.callbacks import SAFE_CANCELLED, SAFE_UI_ERROR, WorkbenchCallbacks
from rag_mvp.ui.models import BrowserSessionState, ChatServiceResult, SourcePreview
from rag_mvp.ui.services import WorkbenchServices

pytestmark = pytest.mark.ui


@dataclass
class FakeChatGateway:
    result: ChatServiceResult
    submissions: list[tuple[str, str | None, str, RetrievalMode]] = field(default_factory=list)
    resets: list[tuple[str, str | None]] = field(default_factory=list)

    async def submit(
        self,
        *,
        owner_id: str,
        session_id: str | None,
        question: str,
        mode: RetrievalMode,
    ) -> ChatServiceResult:
        self.submissions.append((owner_id, session_id, question, mode))
        return self.result

    def reset(self, *, owner_id: str, session_id: str | None) -> str:
        self.resets.append((owner_id, session_id))
        return "session_reset"


def _citation() -> Citation:
    return Citation(
        source_title="Handbook person@example.com",
        document_version=3,
        chunk_id="chunk_policy",
        locator=ChunkLocator(pages=(4, 5)),
    )


def _answer_result() -> ChatServiceResult:
    citation = _citation()
    event = ValidatedStreamEvent(
        request_id="request_answer",
        session_id="session_answer",
        sequence=0,
        kind=StreamEventKind.ANSWER,
        response_language="en",
        content="Employees receive twelve days. person@example.com",
        claims=(
            AnswerClaim(
                text="Employees receive twelve days.",
                citation_chunk_ids=(citation.chunk_id,),
            ),
        ),
        citations=(citation,),
        terminal=True,
    )
    return ChatServiceResult(
        event=event,
        previews=(
            SourcePreview(
                citation=citation,
                preview="The policy contact is person@example.com.",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_chat_renders_only_terminal_validated_answer_with_citations_and_previews() -> None:
    gateway = FakeChatGateway(_answer_result())
    callbacks = WorkbenchCallbacks(WorkbenchServices(chat=gateway))
    state = BrowserSessionState.create().with_active_request("request_pending")

    rendered = await callbacks.submit_chat(
        "What does person@example.com receive?",
        "hybrid-rerank",
        None,
        state,
    )

    assert gateway.submissions == [
        (state.owner_id, None, "What does person@example.com receive?", RetrievalMode.HYBRID_RERANK)
    ]
    assert [turn["role"] for turn in rendered.history] == ["user", "assistant"]
    assert rendered.history[0]["content"] == "What does [REDACTED_EMAIL] receive?"
    assert rendered.history[1]["content"].endswith("[1]")
    assert "Employees receive twelve days." in rendered.history[1]["content"]
    assert "**Handbook [REDACTED_EMAIL]**, v3" in rendered.citations_markdown
    assert "pages 4, 5" in rendered.citations_markdown
    assert "`chunk_policy`" in rendered.citations_markdown
    assert "<details>" in rendered.previews_markdown
    assert "[REDACTED_EMAIL]" in rendered.previews_markdown
    assert "person@example.com" not in repr(rendered)
    assert rendered.state.owner_id == state.owner_id
    assert rendered.state.session_id == "session_answer"
    assert rendered.state.active_request_id is None
    assert "Answer validated" in rendered.status_markdown


@pytest.mark.asyncio
async def test_refusal_is_visually_distinct_from_an_answer() -> None:
    event = ValidatedStreamEvent(
        request_id="request_refusal",
        session_id="session_refusal",
        sequence=0,
        kind=StreamEventKind.REFUSAL,
        response_language="en",
        content="The indexed corpus does not support that answer.",
        reason=RefusalReason.INSUFFICIENT_EVIDENCE,
        terminal=True,
    )
    callbacks = WorkbenchCallbacks(
        WorkbenchServices(chat=FakeChatGateway(ChatServiceResult(event=event)))
    )

    rendered = await callbacks.submit_chat("Unsupported question", "dense", None, None)

    assert rendered.history[-1]["content"] == event.content
    assert rendered.citations_markdown == ""
    assert rendered.previews_markdown == ""
    assert "Refusal / 拒绝" in rendered.status_markdown
    assert RefusalReason.INSUFFICIENT_EVIDENCE.value in rendered.status_markdown
    assert "Answer validated" not in rendered.status_markdown


def test_reset_and_cancel_do_not_share_or_reveal_pending_text() -> None:
    gateway = FakeChatGateway(_answer_result())
    callbacks = WorkbenchCallbacks(WorkbenchServices(chat=gateway))
    state = (
        BrowserSessionState.create()
        .with_session("session_old")
        .with_active_request("request_pending")
    )

    reset = callbacks.reset_chat(state)
    cancelled = callbacks.cancel_chat(
        ({"role": "assistant", "content": "Previously validated."},),
        state,
    )

    assert gateway.resets == [(state.owner_id, "session_old")]
    assert reset.history == ()
    assert reset.state.owner_id == state.owner_id
    assert reset.state.session_id == "session_reset"
    assert "Session reset" in reset.status_markdown
    assert cancelled.history == ({"role": "assistant", "content": "Previously validated."},)
    assert cancelled.state.active_request_id is None
    assert cancelled.citations_markdown == ""
    assert cancelled.previews_markdown == ""
    assert cancelled.status_markdown == SAFE_CANCELLED


@pytest.mark.asyncio
async def test_nonterminal_event_fails_closed_without_rendering_pending_content() -> None:
    citation = _citation()
    pending = ValidatedStreamEvent(
        request_id="request_pending",
        session_id="session_pending",
        sequence=0,
        kind=StreamEventKind.SENTENCE,
        response_language="en",
        content="Unvalidated secret person@example.com",
        claims=(
            AnswerClaim(
                text="Unvalidated secret person@example.com",
                citation_chunk_ids=(citation.chunk_id,),
            ),
        ),
        citations=(citation,),
    )
    callbacks = WorkbenchCallbacks(
        WorkbenchServices(chat=FakeChatGateway(ChatServiceResult(event=pending)))
    )
    prior = ({"role": "assistant", "content": "Previously validated."},)

    rendered = await callbacks.submit_chat("Next question", "hybrid", prior, None)

    assert rendered.history == prior
    assert rendered.citations_markdown == ""
    assert rendered.previews_markdown == ""
    assert SAFE_UI_ERROR in rendered.status_markdown
    assert "person@example.com" not in repr(rendered)
