from __future__ import annotations

from collections.abc import Sequence

import pytest

from rag_mvp.safety import (
    DetectionSpan,
    RedactionError,
    Redactor,
    SensitiveKind,
    resolve_overlaps,
)


class StaticDetector:
    name = "static"

    def __init__(self, spans: Sequence[DetectionSpan]) -> None:
        self._spans = spans

    def detect(self, text: str) -> Sequence[DetectionSpan]:
        del text
        return self._spans


class BrokenDetector:
    name = "broken"

    def detect(self, text: str) -> Sequence[DetectionSpan]:
        del text
        raise RuntimeError("the raw dependency error must not escape")


def test_typed_result_contains_only_offsets_and_category_counts() -> None:
    result = Redactor().redact("Contact person@example.com or 13800138000")

    assert result.detected
    assert result.original_length == len("Contact person@example.com or 13800138000")
    assert result.counts[SensitiveKind.EMAIL] == 1
    assert result.counts[SensitiveKind.PHONE] == 1
    assert "person@example.com" not in repr(result)
    assert "13800138000" not in repr(result)


def test_overlap_resolution_uses_priority_and_redacts_the_full_union() -> None:
    spans = (
        DetectionSpan(2, 8, SensitiveKind.EMAIL, "low", priority=10),
        DetectionSpan(5, 12, SensitiveKind.SECRET, "high", priority=100),
    )

    assert resolve_overlaps(spans) == (
        DetectionSpan(2, 12, SensitiveKind.SECRET, "high", priority=100),
    )
    result = Redactor((StaticDetector(spans),)).redact("__abcdefghij__")
    assert result.redacted_text == "__[REDACTED_SECRET]__"


def test_overlap_ties_are_deterministic_independent_of_detector_order() -> None:
    left = DetectionSpan(0, 5, SensitiveKind.PHONE, "z", priority=10)
    right = DetectionSpan(1, 6, SensitiveKind.EMAIL, "a", priority=10)

    assert resolve_overlaps((left, right)) == resolve_overlaps((right, left))
    assert resolve_overlaps((left, right))[0].kind is SensitiveKind.EMAIL


def test_uninitialized_or_broken_redactor_fails_closed() -> None:
    with pytest.raises(RedactionError, match="not initialized"):
        Redactor(()).redact("ordinary text")
    with pytest.raises(RedactionError, match="detector failed"):
        Redactor((BrokenDetector(),)).redact("ordinary text")


def test_invalid_detector_span_is_rejected() -> None:
    detector = StaticDetector((DetectionSpan(0, 100, SensitiveKind.SECRET, "bad"),))
    with pytest.raises(RedactionError, match="out-of-bounds"):
        Redactor((detector,)).redact("short")
