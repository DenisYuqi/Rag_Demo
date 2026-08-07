from __future__ import annotations

import pytest

from rag_mvp.safety import EmailDetector, PhoneDetector, redact_text


@pytest.mark.parametrize(
    ("value", "placeholder"),
    [
        ("person@example.com", "[REDACTED_EMAIL]"),
        ("first.last+tag@sub.example.co.uk", "[REDACTED_EMAIL]"),
        ("13800138000", "[REDACTED_PHONE]"),
        ("+86 138 0013 8000", "[REDACTED_PHONE]"),
        ("010-12345678", "[REDACTED_PHONE]"),
        ("+1 (415) 555-2671", "[REDACTED_PHONE]"),
        ("+44 20 7946 0958", "[REDACTED_PHONE]"),
    ],
)
def test_contact_values_are_completely_replaced(value: str, placeholder: str) -> None:
    output = redact_text(f"中英 contact: {value}, done")
    assert value not in output
    assert placeholder in output


@pytest.mark.parametrize(
    "value",
    [
        "person@example",
        "@example.com",
        "2026-08-04",
        "123-45-6789",
        "version 1234567",
    ],
)
def test_contact_false_positives_are_not_redacted_by_default(value: str) -> None:
    assert EmailDetector().detect(value) == ()
    assert PhoneDetector().detect(value) == ()


def test_dot_separated_phone_number_remains_fail_safe() -> None:
    value = "415.5552671"

    spans = PhoneDetector().detect(value)

    assert len(spans) == 1
    assert value[spans[0].start : spans[0].end] == value
