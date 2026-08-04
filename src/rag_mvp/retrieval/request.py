"""Retrieval request validation and immutable revision binding."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from rag_mvp.domain.retrieval import RetrievalMode


class RetrievalRequestError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def canonicalize_query(query: str, *, maximum_characters: int = 4096) -> str:
    canonical = unicodedata.normalize("NFC", query).strip()
    canonical = " ".join(canonical.split())
    if not canonical:
        raise RetrievalRequestError("empty_query")
    if len(canonical) > maximum_characters:
        raise RetrievalRequestError("query_too_long")
    if any(ord(character) < 32 for character in canonical):
        raise RetrievalRequestError("invalid_query")
    return canonical


@dataclass(frozen=True, slots=True)
class RetrievalRequestContext:
    request_id: str
    query: str
    mode: RetrievalMode
    revision_id: str

    @classmethod
    def bind(
        cls,
        *,
        request_id: str,
        query: str,
        mode: RetrievalMode | str,
        active_revision_id: str | None,
    ) -> RetrievalRequestContext:
        if not request_id:
            raise RetrievalRequestError("invalid_request_id")
        if not active_revision_id:
            raise RetrievalRequestError("index_not_ready")
        try:
            resolved_mode = RetrievalMode(mode)
        except ValueError as error:
            raise RetrievalRequestError("invalid_retrieval_mode") from error
        return cls(
            request_id=request_id,
            query=canonicalize_query(query),
            mode=resolved_mode,
            revision_id=active_revision_id,
        )
