from __future__ import annotations

import pytest

from rag_mvp.safety import IPAddressDetector, redact_text


@pytest.mark.parametrize(
    ("value", "placeholder"),
    [
        ("192.168.1.1", "[REDACTED_IPV4]"),
        ("8.8.8.8", "[REDACTED_IPV4]"),
        ("2001:db8::1", "[REDACTED_IPV6]"),
        ("::1", "[REDACTED_IPV6]"),
        ("fe80::abcd:1", "[REDACTED_IPV6]"),
    ],
)
def test_valid_ip_address_is_redacted(value: str, placeholder: str) -> None:
    output = redact_text(f"ip={value}")
    assert value not in output
    assert placeholder in output


def test_sentence_punctuation_after_ipv4_is_preserved() -> None:
    assert redact_text("Host 192.168.1.1. Next") == "Host [REDACTED_IPV4]. Next"


@pytest.mark.parametrize("value", ["999.1.1.1", "1.2.3", "2001:::1", "version:1"])
def test_invalid_ip_address_is_not_detected(value: str) -> None:
    assert IPAddressDetector().detect(value) == ()


@pytest.mark.parametrize(
    ("source", "secret"),
    [
        ("Authorization: Bearer abcdefghijklmnop", "abcdefghijklmnop"),
        ("Bearer tiny", "tiny"),
        ("api_key=abcdefghijklmnopqrstuvwxyz123456", "abcdefghijklmnopqrstuvwxyz123456"),
        ("OPENAI_API_KEY=short", "short"),
        ("password=x", "x"),
        ("password='correct-horse-battery-staple'", "correct-horse-battery-staple"),
        ("sk-abcdefghijklmnopqrstuvwxyz1234", "sk-abcdefghijklmnopqrstuvwxyz1234"),
        ("AKIAABCDEFGHIJKLMNOP", "AKIAABCDEFGHIJKLMNOP"),
        ("ghp_abcdefghijklmnopqrstuvwxyz123456", "ghp_abcdefghijklmnopqrstuvwxyz123456"),
    ],
)
def test_common_secret_values_are_replaced(source: str, secret: str) -> None:
    output = redact_text(source)
    assert secret not in output
    assert "[REDACTED_SECRET]" in output


def test_complete_private_key_block_is_replaced_as_one_secret() -> None:
    private_key = (
        "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n-----END PRIVATE KEY-----"
    )
    output = redact_text(f"before\n{private_key}\nafter")
    assert private_key not in output
    assert output == "before\n[REDACTED_SECRET]\nafter"
