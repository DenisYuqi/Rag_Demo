from __future__ import annotations

import pytest

from rag_mvp.safety import PaymentCardDetector, redact_text


@pytest.mark.parametrize(
    "value",
    [
        "4111111111111111",
        "4111 1111 1111 1111",
        "4012-8888-8888-1881",
        "5555555555554444",
    ],
)
def test_luhn_valid_card_is_fully_redacted(value: str) -> None:
    output = redact_text(f"card={value}")
    assert value not in output
    assert output == "card=[REDACTED_PAYMENT_CARD]"


@pytest.mark.parametrize("value", ["4111111111111112", "0000000000000000", "1234 5678 9012"])
def test_invalid_or_implausible_card_is_not_redacted(value: str) -> None:
    assert PaymentCardDetector().detect(value) == ()


def test_luhn_helper_ignores_visual_separators() -> None:
    assert PaymentCardDetector.passes_luhn("4111-1111 1111-1111")
