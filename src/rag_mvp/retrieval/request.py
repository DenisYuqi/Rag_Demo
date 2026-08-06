"""Retrieval request validation and immutable revision binding."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rag_mvp.domain.ingestion import IndexRevision, IndexRevisionStatus
from rag_mvp.domain.retrieval import CachePolicy, RetrievalMode

if TYPE_CHECKING:
    from rag_mvp.retrieval.binding import BoundRetrievalSnapshot


DEFAULT_MAXIMUM_QUERY_CHARACTERS = 4096
QUERY_CANONICALIZATION_VERSION = "unicode-nfc-whitespace-v1"
_OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$")
_UNICODE_QUERY_WHITESPACE = frozenset(
    "\t\n\v\f\r \u0085\u00a0\u1680\u2028\u2029\u202f\u205f\u3000"
) | frozenset(chr(codepoint) for codepoint in range(0x2000, 0x200B))


class RetrievalRequestError(ValueError):
    def __init__(self, code: str, *, detail_code: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.detail_code = detail_code


def canonicalize_query(
    query: str,
    *,
    maximum_characters: int = DEFAULT_MAXIMUM_QUERY_CHARACTERS,
) -> str:
    if isinstance(maximum_characters, bool) or not isinstance(maximum_characters, int):
        raise ValueError("maximum_characters must be a positive integer")
    if maximum_characters < 1:
        raise ValueError("maximum_characters must be a positive integer")
    if not isinstance(query, str):
        raise RetrievalRequestError("invalid_query")
    if len(query) > maximum_characters:
        raise RetrievalRequestError("query_too_long")
    if not query:
        raise RetrievalRequestError("empty_query")
    if any(_is_disallowed_query_character(character) for character in query):
        raise RetrievalRequestError("invalid_query")
    try:
        canonical = unicodedata.normalize("NFC", query)
    except (TypeError, ValueError):
        raise RetrievalRequestError("invalid_query") from None
    canonical = " ".join(canonical.split())
    if not canonical:
        raise RetrievalRequestError("empty_query")
    if len(canonical) > maximum_characters:
        raise RetrievalRequestError("query_too_long")
    if any(_is_disallowed_query_character(character) for character in canonical):
        raise RetrievalRequestError("invalid_query")
    return canonical


def _is_disallowed_query_character(character: str) -> bool:
    category = unicodedata.category(character)
    if category == "Cc":
        return character not in _UNICODE_QUERY_WHITESPACE
    return category in {"Cf", "Cs"}


def _validate_opaque_id(value: object, code: str) -> str:
    if not isinstance(value, str) or _OPAQUE_ID.fullmatch(value) is None:
        raise RetrievalRequestError(code)
    return value


@dataclass(frozen=True, slots=True)
class RetrievalRequestContext:
    request_id: str
    query: str
    mode: RetrievalMode | str
    revision_id: str
    maximum_query_characters: int = field(
        default=DEFAULT_MAXIMUM_QUERY_CHARACTERS,
        kw_only=True,
        repr=False,
        compare=False,
    )
    cache_policy: CachePolicy | str = field(default=CachePolicy.USE, kw_only=True)
    revision: IndexRevision | None = field(default=None, init=False, repr=False)
    _binding_token: object | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        request_id = _validate_opaque_id(self.request_id, "invalid_request_id")
        revision_id = _validate_opaque_id(self.revision_id, "invalid_revision_id")
        try:
            mode = RetrievalMode(self.mode)
        except (TypeError, ValueError):
            raise RetrievalRequestError("invalid_retrieval_mode") from None
        try:
            cache_policy = CachePolicy(self.cache_policy)
        except (TypeError, ValueError):
            raise RetrievalRequestError("invalid_cache_policy") from None
        query = canonicalize_query(
            self.query,
            maximum_characters=self.maximum_query_characters,
        )
        if self.revision is not None:
            if not isinstance(self.revision, IndexRevision):
                raise RetrievalRequestError("invalid_revision_identity")
            if self.revision.revision_id != revision_id:
                raise RetrievalRequestError("revision_identity_mismatch")
            if self.revision.status not in {
                IndexRevisionStatus.ACTIVE,
                IndexRevisionStatus.SUPERSEDED,
            }:
                raise RetrievalRequestError("revision_not_committed")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "cache_policy", cache_policy)
        object.__setattr__(self, "revision_id", revision_id)

    @classmethod
    def bind(
        cls,
        *,
        request_id: str,
        query: str,
        mode: RetrievalMode | str,
        active_revision_id: str | None = None,
        snapshot: BoundRetrievalSnapshot | None = None,
        maximum_characters: int = DEFAULT_MAXIMUM_QUERY_CHARACTERS,
        cache_policy: CachePolicy | str = CachePolicy.USE,
    ) -> RetrievalRequestContext:
        _validate_opaque_id(request_id, "invalid_request_id")
        try:
            resolved_mode = RetrievalMode(mode)
        except (TypeError, ValueError):
            raise RetrievalRequestError("invalid_retrieval_mode") from None
        canonical_query = canonicalize_query(query, maximum_characters=maximum_characters)
        if snapshot is not None:
            if active_revision_id is not None:
                raise RetrievalRequestError("ambiguous_revision_binding")
            return cls._from_bound_snapshot(
                request_id=request_id,
                query=canonical_query,
                mode=resolved_mode,
                snapshot=snapshot,
                maximum_characters=maximum_characters,
                cache_policy=cache_policy,
            )
        if active_revision_id is None:
            raise RetrievalRequestError("index_not_ready")
        _validate_opaque_id(active_revision_id, "invalid_revision_id")
        raise RetrievalRequestError("untrusted_revision_binding")

    @classmethod
    def from_snapshot(
        cls,
        *,
        request_id: str,
        query: str,
        mode: RetrievalMode | str,
        snapshot: BoundRetrievalSnapshot,
        maximum_characters: int = DEFAULT_MAXIMUM_QUERY_CHARACTERS,
        cache_policy: CachePolicy | str = CachePolicy.USE,
    ) -> RetrievalRequestContext:
        return cls._from_bound_snapshot(
            request_id=request_id,
            query=canonicalize_query(query, maximum_characters=maximum_characters),
            mode=mode,
            snapshot=snapshot,
            maximum_characters=maximum_characters,
            cache_policy=cache_policy,
        )

    @classmethod
    def _from_bound_snapshot(
        cls,
        *,
        request_id: str,
        query: str,
        mode: RetrievalMode | str,
        snapshot: BoundRetrievalSnapshot,
        maximum_characters: int,
        cache_policy: CachePolicy | str,
    ) -> RetrievalRequestContext:
        from rag_mvp.retrieval.binding import BoundRetrievalSnapshot

        if not isinstance(snapshot, BoundRetrievalSnapshot) or snapshot.is_closed:
            raise RetrievalRequestError("invalid_snapshot_binding")
        context = cls(
            request_id=request_id,
            query=query,
            mode=mode,
            revision_id=snapshot.revision_id,
            maximum_query_characters=maximum_characters,
            cache_policy=cache_policy,
        )
        object.__setattr__(context, "revision", snapshot.revision)
        object.__setattr__(context, "_binding_token", snapshot.binding_token)
        return context

    @property
    def is_snapshot_bound(self) -> bool:
        return self.revision is not None and self._binding_token is not None

    def assert_matches_snapshot(self, snapshot: BoundRetrievalSnapshot) -> None:
        if (
            snapshot.is_closed
            or not self.is_snapshot_bound
            or self._binding_token is not snapshot.binding_token
            or self.revision != snapshot.revision
            or self.revision_id != snapshot.revision_id
        ):
            raise RetrievalRequestError("snapshot_context_mismatch")
