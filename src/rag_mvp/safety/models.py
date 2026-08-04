"""Typed privacy detection and redaction results.

The result types deliberately retain offsets and categories, but never retain the
matched source value.  This makes the objects safe to attach to diagnostics as
long as callers do not separately log the original input.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol


class SensitiveKind(StrEnum):
    """Sensitive value categories supported by the deterministic redactor."""

    EMAIL = "email"
    PHONE = "phone"
    CHINESE_ID = "chinese_id"
    SSN = "ssn"
    PAYMENT_CARD = "payment_card"
    IPV4 = "ipv4"
    IPV6 = "ipv6"
    SECRET = "secret"  # noqa: S105 - category label, not a credential

    @property
    def placeholder(self) -> str:
        """Return the stable user-visible placeholder for this category."""

        return f"[REDACTED_{self.value.upper()}]"


@dataclass(frozen=True, slots=True)
class DetectionSpan:
    """A half-open sensitive span in the original input."""

    start: int
    end: int
    kind: SensitiveKind
    detector: str
    priority: int = 0

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("span start must be non-negative")
        if self.end <= self.start:
            raise ValueError("span end must be greater than span start")
        if not self.detector:
            raise ValueError("detector name must not be empty")

    @property
    def length(self) -> int:
        """Length of the match in the original input."""

        return self.end - self.start

    @property
    def placeholder(self) -> str:
        """Typed replacement text for the match."""

        return self.kind.placeholder


# A descriptive alias for callers that prefer domain terminology.
SensitiveSpan = DetectionSpan


class Detector(Protocol):
    """Protocol implemented by all deterministic sensitive-value detectors."""

    name: str

    def detect(self, text: str) -> Sequence[DetectionSpan]:
        """Return sensitive spans using offsets into ``text``."""


@dataclass(frozen=True, slots=True)
class RedactionResult:
    """A redacted string plus content-free detection metadata."""

    redacted_text: str
    spans: tuple[DetectionSpan, ...]
    original_length: int

    @property
    def text(self) -> str:
        """Convenience alias for adapters that expect a text result."""

        return self.redacted_text

    @property
    def detected(self) -> bool:
        """Whether at least one sensitive span was replaced."""

        return bool(self.spans)

    @property
    def counts(self) -> Mapping[SensitiveKind, int]:
        """Return immutable per-category counts suitable for diagnostics."""

        counts = Counter(span.kind for span in self.spans)
        return MappingProxyType(dict(counts))
