"""Deterministic language selection and user-history-only query rewriting."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field

from rag_mvp.domain.qa import ConversationRole, ConversationTurn
from rag_mvp.retrieval.request import RetrievalRequestError, canonicalize_query

ENGLISH_RESPONSE_LANGUAGE = "en"
CHINESE_RESPONSE_LANGUAGE = "zh-CN"
QUERY_REWRITE_VERSION = "user-history-expansion-v1"

_LANGUAGE_ALIASES = {
    "en": ENGLISH_RESPONSE_LANGUAGE,
    "en-gb": ENGLISH_RESPONSE_LANGUAGE,
    "en-us": ENGLISH_RESPONSE_LANGUAGE,
    "english": ENGLISH_RESPONSE_LANGUAGE,
    "zh": CHINESE_RESPONSE_LANGUAGE,
    "zh-cn": CHINESE_RESPONSE_LANGUAGE,
    "zh-hans": CHINESE_RESPONSE_LANGUAGE,
    "chinese": CHINESE_RESPONSE_LANGUAGE,
}
_ENGLISH_CONTEXT_REFERENCE = re.compile(
    r"(?i)(?:\b(?:it|its|they|them|their|those|these|former|latter|above)\b|"
    r"^\s*(?:(?:what|how)\s+about|and|also|then|so)\b|"
    r"\b(?:this|that)\s+(?:one|policy|rule|requirement|document|section|process|benefit)\b)"
)
_CHINESE_CONTEXT_REFERENCE = re.compile(
    r"(?:它|它们|他们|她们|其|这个|这项|这份|这条|这些|那个|那项|那份|那条|那些|"
    r"上述|前者|后者|同样|怎么办|呢\s*\??$)"
)


class QueryRewriteError(ValueError):
    """A stable validation failure for query preparation."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class QueryRewriteResult:
    query: str = field(repr=False)
    response_language: str
    rewritten: bool
    source_user_turn_ids: tuple[str, ...]
    version: str = QUERY_REWRITE_VERSION


def select_response_language(
    latest_user_text: str,
    *,
    requested_language: str | None = None,
) -> str:
    """Select Chinese or English from an explicit choice or the latest user text."""

    if requested_language is not None:
        if not isinstance(requested_language, str):
            raise QueryRewriteError("unsupported_response_language")
        normalized = requested_language.strip().replace("_", "-").casefold()
        try:
            return _LANGUAGE_ALIASES[normalized]
        except KeyError:
            raise QueryRewriteError("unsupported_response_language") from None

    text = _canonicalize(latest_user_text, maximum_characters=max(1, len(latest_user_text)))
    for character in text:
        name = unicodedata.name(character, "")
        if "CJK UNIFIED IDEOGRAPH" in name or "CJK COMPATIBILITY IDEOGRAPH" in name:
            return CHINESE_RESPONSE_LANGUAGE
    return ENGLISH_RESPONSE_LANGUAGE


def is_contextual_follow_up(query: str) -> bool:
    """Return whether a question contains a bounded bilingual context reference."""

    canonical = _canonicalize(query, maximum_characters=max(1, len(query)))
    return bool(
        _ENGLISH_CONTEXT_REFERENCE.search(canonical) or _CHINESE_CONTEXT_REFERENCE.search(canonical)
    )


class QueryRewriter:
    """Expand contextual follow-ups using prior user questions and no assistant content."""

    def __init__(
        self,
        *,
        maximum_history_turns: int = 4,
        maximum_query_characters: int = 4096,
    ) -> None:
        if type(maximum_history_turns) is not int or maximum_history_turns < 1:
            raise ValueError("maximum_history_turns must be a positive integer")
        if type(maximum_query_characters) is not int or maximum_query_characters < 1:
            raise ValueError("maximum_query_characters must be a positive integer")
        self.maximum_history_turns = maximum_history_turns
        self.maximum_query_characters = maximum_query_characters

    def prepare(
        self,
        turns: Sequence[ConversationTurn],
        *,
        requested_language: str | None = None,
    ) -> QueryRewriteResult:
        ordered = self._validated_history(turns)
        user_turns = tuple(turn for turn in ordered if turn.role is ConversationRole.USER)
        if not user_turns:
            raise QueryRewriteError("user_turn_missing")

        latest = user_turns[-1]
        latest_query = _canonicalize(
            latest.content,
            maximum_characters=self.maximum_query_characters,
        )
        response_language = select_response_language(
            latest_query,
            requested_language=requested_language,
        )
        prior_users = user_turns[:-1]
        if not prior_users or not is_contextual_follow_up(latest_query):
            return QueryRewriteResult(
                query=latest_query,
                response_language=response_language,
                rewritten=False,
                source_user_turn_ids=(latest.turn_id,),
            )

        selected: list[tuple[ConversationTurn, str]] = []
        remaining = self.maximum_query_characters - len(latest_query)
        for turn in reversed(prior_users[-self.maximum_history_turns :]):
            if remaining <= 1:
                break
            text = _canonicalize(turn.content, maximum_characters=max(1, len(turn.content)))
            available = remaining - 1
            bounded = text[:available].rstrip()
            if not bounded:
                continue
            selected.append((turn, bounded))
            remaining -= len(bounded) + 1

        if not selected:
            return QueryRewriteResult(
                query=latest_query,
                response_language=response_language,
                rewritten=False,
                source_user_turn_ids=(latest.turn_id,),
            )

        selected.reverse()
        query = " ".join((*[text for _, text in selected], latest_query))
        return QueryRewriteResult(
            query=query,
            response_language=response_language,
            rewritten=True,
            source_user_turn_ids=(
                *(turn.turn_id for turn, _ in selected),
                latest.turn_id,
            ),
        )

    @staticmethod
    def _validated_history(
        turns: Sequence[ConversationTurn],
    ) -> tuple[ConversationTurn, ...]:
        if not turns or any(not isinstance(turn, ConversationTurn) for turn in turns):
            raise QueryRewriteError("turn_history_invalid")
        session_ids = {turn.session_id for turn in turns}
        ordinals = [turn.ordinal for turn in turns]
        turn_ids = [turn.turn_id for turn in turns]
        if len(session_ids) != 1:
            raise QueryRewriteError("mixed_session_history")
        if len(ordinals) != len(set(ordinals)) or len(turn_ids) != len(set(turn_ids)):
            raise QueryRewriteError("turn_history_invalid")
        return tuple(sorted(turns, key=lambda turn: turn.ordinal))


def _canonicalize(text: str, *, maximum_characters: int) -> str:
    try:
        return canonicalize_query(text, maximum_characters=maximum_characters)
    except (RetrievalRequestError, TypeError, ValueError) as error:
        code = error.code if isinstance(error, RetrievalRequestError) else "invalid_query"
        raise QueryRewriteError(code) from None
