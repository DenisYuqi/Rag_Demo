from __future__ import annotations

import pytest

from rag_mvp.safety import ChineseNationalIdDetector, SSNDetector, redact_text

VALID_CHINESE_ID = "11010519491231002X"


def test_checksum_valid_chinese_identity_number_is_redacted() -> None:
    output = redact_text(f"身份证: {VALID_CHINESE_ID}")
    assert output == "身份证: [REDACTED_CHINESE_ID]"
    assert ChineseNationalIdDetector.is_valid(VALID_CHINESE_ID)


@pytest.mark.parametrize(
    "value",
    [
        "110105194912310021",  # incorrect checksum
        "11010519990230002X",  # impossible date
        "00000019491231002X",  # invalid region prefix
        "11010519491231000X",  # invalid sequence and checksum
    ],
)
def test_invalid_chinese_identity_candidates_are_not_classified(value: str) -> None:
    assert ChineseNationalIdDetector().detect(value) == ()


def test_valid_ssn_is_redacted() -> None:
    value = "078-05-1120"
    assert redact_text(f"SSN {value}") == "SSN [REDACTED_SSN]"
    assert len(SSNDetector().detect(value)) == 1


@pytest.mark.parametrize(
    "value",
    ["000-12-3456", "666-12-3456", "901-12-3456", "123-00-6789", "123-45-0000"],
)
def test_invalid_ssn_groups_are_rejected(value: str) -> None:
    assert SSNDetector().detect(value) == ()
