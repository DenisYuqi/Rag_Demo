from __future__ import annotations

from collections.abc import Sequence

import pytest

from rag_mvp.safety import (
    SAFE_UNAVAILABLE_MESSAGE,
    DetectionSpan,
    Redactor,
    SafeStream,
)


class BrokenDetector:
    name = "broken"

    def detect(self, text: str) -> Sequence[DetectionSpan]:
        del text
        raise RuntimeError("unsafe provider error")


def test_email_split_across_deltas_never_emits_a_raw_prefix() -> None:
    stream = SafeStream()
    assert stream.push("Contact person@") == ()
    events = stream.push("example.com. Next ")
    assert events == ("Contact [REDACTED_EMAIL].",)
    assert "person@" not in "".join(events)
    assert stream.finish() == (" Next ",)


@pytest.mark.parametrize(
    "sensitive_value",
    [
        "person@example.com",
        "+86 138 0013 8000",
        "4111 1111 1111 1111",
        "192.168.1.1",
        "2001:db8::1",
        (
            "-----BEGIN PRIVATE KEY-----\n"
            "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n"
            "-----END PRIVATE KEY-----"
        ),
    ],
)
def test_sensitive_value_is_safe_at_every_delta_split(sensitive_value: str) -> None:
    for split_at in range(1, len(sensitive_value)):
        stream = SafeStream()
        first_events = stream.push(f"Value: {sensitive_value[:split_at]}")
        second_events = stream.push(f"{sensitive_value[split_at:]}\n")
        final_events = stream.finish()
        rendered = "".join((*first_events, *second_events, *final_events))
        assert sensitive_value not in rendered
        assert "[REDACTED_" in rendered
        assert not stream.failed


def test_generation_without_sentence_boundary_scans_and_redacts_tail() -> None:
    stream = SafeStream()
    assert stream.push("Email person@example.com") == ()
    assert stream.finish() == ("Email [REDACTED_EMAIL]",)


def test_terminal_currency_amount_is_not_misclassified_as_partial_ip() -> None:
    stream = SafeStream()
    answer = "The current airfare reimbursement cap is RMB 1,800."

    assert stream.push(answer) == ()
    assert stream.finish() == (answer,)
    assert not stream.failed


def test_actual_partial_ip_tail_remains_fail_closed() -> None:
    stream = SafeStream()
    stream.push("Internal address 192.168.1.")

    assert stream.finish() == (SAFE_UNAVAILABLE_MESSAGE,)
    assert stream.failed
    assert stream.failure_reason == "ambiguous_tail"


def test_ambiguous_or_incomplete_tail_is_discarded() -> None:
    stream = SafeStream()
    stream.push("Email person@")
    assert stream.finish() == (SAFE_UNAVAILABLE_MESSAGE,)
    assert stream.failed
    assert stream.failure_reason == "ambiguous_tail"


def test_ambiguous_email_prefix_before_newline_is_never_emitted() -> None:
    stream = SafeStream()
    assert stream.push("Email person@\n") == (SAFE_UNAVAILABLE_MESSAGE,)
    assert stream.failed
    assert stream.failure_reason == "ambiguous_boundary"


def test_incomplete_private_key_is_discarded() -> None:
    stream = SafeStream()
    stream.push("-----BEGIN PRIVATE KEY-----\nsecret-material")
    assert stream.finish() == (SAFE_UNAVAILABLE_MESSAGE,)
    assert stream.failure_reason == "incomplete_private_key"


def test_buffer_overflow_and_redactor_failure_are_fail_closed() -> None:
    overflow = SafeStream(max_buffer_chars=5)
    assert overflow.push("123456") == (SAFE_UNAVAILABLE_MESSAGE,)
    assert overflow.failed
    assert overflow.pending_length == 0

    broken = SafeStream(redactor=Redactor((BrokenDetector(),)))
    assert broken.push("ordinary sentence. ") == (SAFE_UNAVAILABLE_MESSAGE,)
    assert broken.failure_reason == "redaction_failed"


def test_failure_message_cannot_be_replaced_with_unvetted_dynamic_text() -> None:
    with pytest.raises(ValueError, match="pre-vetted"):
        SafeStream(unavailable_message="contact person@example.com")


def test_abort_discards_pending_content_without_an_error_message() -> None:
    stream = SafeStream()
    stream.push("person@")
    assert stream.abort() == ()
    assert stream.closed
    assert stream.pending_length == 0
    assert stream.push("example.com") == ()
