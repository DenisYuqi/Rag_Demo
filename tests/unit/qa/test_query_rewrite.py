from __future__ import annotations

import pytest

from rag_mvp.domain.qa import ConversationRole, ConversationTurn
from rag_mvp.qa.query_rewrite import (
    QueryRewriteError,
    QueryRewriter,
    is_contextual_follow_up,
    select_response_language,
)


def _turn(
    turn_id: str,
    ordinal: int,
    role: ConversationRole,
    content: str,
    *,
    session_id: str = "session-1",
) -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id,
        session_id=session_id,
        ordinal=ordinal,
        role=role,
        content=content,
    )


def test_latest_user_turn_controls_language_not_assistant_history() -> None:
    rewriter = QueryRewriter()
    chinese_user = _turn("user-1", 0, ConversationRole.USER, "年假政策是什么?")
    english_assistant = _turn(
        "assistant-1",
        1,
        ConversationRole.ASSISTANT,
        "This assistant response must not choose the language.",
    )

    chinese = rewriter.prepare((english_assistant, chinese_user))
    english = rewriter.prepare(
        (
            chinese_user,
            english_assistant,
            _turn("user-2", 2, ConversationRole.USER, "Where is the policy published?"),
        )
    )

    assert chinese.response_language == "zh-CN"
    assert english.response_language == "en"


def test_explicit_supported_language_overrides_latest_turn() -> None:
    assert select_response_language("Where is the policy?", requested_language="zh_hans") == "zh-CN"
    assert select_response_language("年假有多少天?", requested_language="en-US") == "en"
    with pytest.raises(QueryRewriteError, match="unsupported_response_language"):
        select_response_language("Question", requested_language="fr")


def test_any_han_character_selects_chinese_for_mixed_language_questions() -> None:
    assert select_response_language("What does 保 mean in this policy?") == "zh-CN"
    assert select_response_language("What does this policy cover?") == "en"


def test_standalone_question_does_not_include_conversation_history() -> None:
    turns = (
        _turn("user-1", 0, ConversationRole.USER, "Tell me about annual leave."),
        _turn("assistant-1", 1, ConversationRole.ASSISTANT, "Annual leave details."),
        _turn("user-2", 2, ConversationRole.USER, "Where is the expense policy published?"),
    )

    result = QueryRewriter().prepare(turns)

    assert result.query == "Where is the expense policy published?"
    assert result.rewritten is False
    assert result.source_user_turn_ids == ("user-2",)


def test_contextual_follow_up_uses_user_questions_and_excludes_assistant_answers() -> None:
    turns = (
        _turn("user-topic", 0, ConversationRole.USER, "What is the annual leave policy?"),
        _turn(
            "assistant-untrusted",
            1,
            ConversationRole.ASSISTANT,
            "It grants ninety-nine days. assistant-only-marker",
        ),
        _turn("user-follow-up", 2, ConversationRole.USER, "How many days does it provide?"),
    )

    result = QueryRewriter().prepare(turns)

    assert result.query == ("What is the annual leave policy? How many days does it provide?")
    assert "ninety-nine" not in result.query
    assert "assistant-only-marker" not in result.query
    assert result.rewritten is True
    assert result.source_user_turn_ids == ("user-topic", "user-follow-up")


def test_follow_up_without_prior_user_context_remains_unchanged() -> None:
    result = QueryRewriter().prepare(
        (_turn("user-1", 0, ConversationRole.USER, "What about contractors?"),)
    )

    assert is_contextual_follow_up(result.query)
    assert result.query == "What about contractors?"
    assert result.rewritten is False


def test_rewrite_is_bounded_and_rejects_cross_session_history() -> None:
    rewriter = QueryRewriter(maximum_query_characters=50)
    result = rewriter.prepare(
        (
            _turn("user-1", 0, ConversationRole.USER, "Annual leave " * 10),
            _turn("user-2", 1, ConversationRole.USER, "How does it work?"),
        )
    )

    assert len(result.query) <= 50
    assert result.query.endswith("How does it work?")
    with pytest.raises(QueryRewriteError, match="mixed_session_history"):
        rewriter.prepare(
            (
                _turn("user-a", 0, ConversationRole.USER, "Question A"),
                _turn(
                    "user-b",
                    1,
                    ConversationRole.USER,
                    "What about it?",
                    session_id="session-2",
                ),
            )
        )
