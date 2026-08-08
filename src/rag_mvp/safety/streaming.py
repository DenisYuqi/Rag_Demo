"""Bounded, sentence-level safe streaming across arbitrary model deltas."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from rag_mvp.safety.detectors import has_unclosed_private_key
from rag_mvp.safety.models import DetectionSpan, SensitiveKind
from rag_mvp.safety.output import SAFE_UNAVAILABLE_MESSAGE
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, RedactionError, Redactor


class SafeStream:
    """Hold unvalidated deltas and emit only complete redacted sentence units."""

    _private_key_block: Final[re.Pattern[str]] = re.compile(
        r"-----BEGIN (?P<label>[A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?)-----"
        r"[\s\S]*?-----END (?P=label)-----",
        re.IGNORECASE,
    )
    _private_key_begin: Final[re.Pattern[str]] = re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----", re.IGNORECASE
    )
    _ambiguous_email_tail: Final[re.Pattern[str]] = re.compile(
        r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]*\Z"
    )
    _ambiguous_tails: Final[tuple[re.Pattern[str], ...]] = (
        re.compile(r"(?:\d[ .-]?){6,}\Z"),
        re.compile(r"(?<![\d.,])(?:\d{1,3}\.){1,3}\d{0,3}\Z"),
        re.compile(
            r"(?i)(?:bearer|basic|api[_-]?key|access[_-]?token|password|passwd|pwd)"
            r"\s*[:=]?\s*\S*\Z"
        ),
    )

    def __init__(
        self,
        *,
        redactor: Redactor | None = DEFAULT_REDACTOR,
        max_buffer_chars: int = 32_768,
        unavailable_message: str = SAFE_UNAVAILABLE_MESSAGE,
    ) -> None:
        if max_buffer_chars <= 0:
            raise ValueError("max_buffer_chars must be positive")
        if unavailable_message != SAFE_UNAVAILABLE_MESSAGE:
            raise ValueError("unavailable_message must be a pre-vetted fixed message")
        self._redactor = redactor
        self._max_buffer_chars = max_buffer_chars
        self._unavailable_message = unavailable_message
        self._buffer = ""
        self._closed = False
        self._failed = False
        self._failure_reason: str | None = None
        self._counts: Counter[SensitiveKind] = Counter()

    @property
    def pending_length(self) -> int:
        """Number of private, not-yet-validated characters."""

        return len(self._buffer)

    @property
    def closed(self) -> bool:
        """Whether the stream can no longer consume model deltas."""

        return self._closed

    @property
    def failed(self) -> bool:
        """Whether the stream ended through a fail-closed path."""

        return self._failed

    @property
    def failure_reason(self) -> str | None:
        """Stable content-free reason for a fail-closed outcome."""

        return self._failure_reason

    @property
    def redaction_counts(self) -> Mapping[SensitiveKind, int]:
        """Immutable category counts for safely emitted text."""

        return MappingProxyType(dict(self._counts))

    def push(self, delta: object) -> tuple[str, ...]:
        """Append a model delta and return newly validated sentence events."""

        if self._closed:
            return ()
        if not isinstance(delta, str):
            return self._fail("malformed_delta")
        self._buffer += delta
        if len(self._buffer) > self._max_buffer_chars:
            return self._fail("buffer_limit")

        cutoff = self._last_safe_sentence_boundary(self._buffer)
        if cutoff == 0:
            return ()
        candidate = self._buffer[:cutoff]
        self._buffer = self._buffer[cutoff:]
        return self._emit(candidate)

    def finish(self) -> tuple[str, ...]:
        """Scan and emit the complete pending tail, or discard it if uncertain."""

        if self._closed:
            return ()
        self._closed = True
        candidate = self._buffer
        self._buffer = ""
        if not candidate:
            return ()
        if has_unclosed_private_key(candidate):
            return self._fail_after_close("incomplete_private_key")
        return self._emit_final(candidate)

    def abort(self, reason: str = "cancelled") -> tuple[str, ...]:
        """Discard pending content without exposing it."""

        if self._closed:
            return ()
        self._buffer = ""
        self._closed = True
        self._failure_reason = reason
        return ()

    def _emit(self, candidate: str) -> tuple[str, ...]:
        if self._redactor is None or not self._redactor.initialized:
            return self._fail("redactor_unavailable")
        try:
            result = self._redactor.redact(candidate)
        except (RedactionError, TypeError, ValueError, RecursionError):
            return self._fail("redaction_failed")
        if self._has_uncovered_ambiguous_tail(candidate, result.spans):
            return self._fail("ambiguous_boundary")
        self._counts.update(span.kind for span in result.spans)
        return (result.redacted_text,)

    def _emit_final(self, candidate: str) -> tuple[str, ...]:
        if self._redactor is None or not self._redactor.initialized:
            return self._fail_after_close("redactor_unavailable")
        try:
            result = self._redactor.redact(candidate)
        except (RedactionError, TypeError, ValueError, RecursionError):
            return self._fail_after_close("redaction_failed")

        if self._has_uncovered_ambiguous_tail(candidate, result.spans):
            return self._fail_after_close("ambiguous_tail")
        self._counts.update(span.kind for span in result.spans)
        return (result.redacted_text,)

    def _fail(self, reason: str) -> tuple[str, ...]:
        self._buffer = ""
        self._closed = True
        self._failed = True
        self._failure_reason = reason
        return (self._unavailable_message,)

    def _fail_after_close(self, reason: str) -> tuple[str, ...]:
        self._buffer = ""
        self._closed = True
        self._failed = True
        self._failure_reason = reason
        return (self._unavailable_message,)

    @classmethod
    def _has_uncovered_ambiguous_tail(
        cls, candidate: str, spans: tuple[DetectionSpan, ...]
    ) -> bool:
        scan_text = candidate.rstrip()
        email_match = cls._ambiguous_email_tail.search(scan_text)
        if (
            email_match is not None
            and not cls._is_tail_covered(email_match.start(), email_match.end(), spans)
            and not cls._is_terminal_email_punctuation_covered(email_match, spans)
        ):
            return True

        for pattern in cls._ambiguous_tails:
            match = pattern.search(scan_text)
            if match is None:
                continue
            if not cls._is_tail_covered(match.start(), match.end(), spans):
                return True
        return False

    @staticmethod
    def _is_tail_covered(start: int, end: int, spans: tuple[DetectionSpan, ...]) -> bool:
        return any(span.start <= start and span.end >= end for span in spans)

    @classmethod
    def _is_terminal_email_punctuation_covered(
        cls, match: re.Match[str], spans: tuple[DetectionSpan, ...]
    ) -> bool:
        if match.group(0)[-1:] not in ".!?":
            return False
        return cls._is_tail_covered(match.start(), match.end() - 1, spans)

    @classmethod
    def _protected_private_key_ranges(cls, text: str) -> tuple[tuple[int, int], ...]:
        ranges = [(match.start(), match.end()) for match in cls._private_key_block.finditer(text)]
        for begin in cls._private_key_begin.finditer(text):
            if not any(start <= begin.start() < end for start, end in ranges):
                ranges.append((begin.start(), len(text) + 1))
        return tuple(sorted(ranges))

    @classmethod
    def _last_safe_sentence_boundary(cls, text: str) -> int:
        protected = cls._protected_private_key_ranges(text)
        cutoff = 0
        for index, character in enumerate(text):
            boundary = 0
            if character in "\u3002\uff01\uff1f\n" or (
                character in ".!?" and index + 1 < len(text) and text[index + 1].isspace()
            ):
                boundary = index + 1
            if boundary and not any(start < boundary < end for start, end in protected):
                cutoff = boundary
        return cutoff


# A more explicit name for adapters that prefer the buffering terminology.
SafeStreamBuffer = SafeStream
