from __future__ import annotations

import math

import pytest

from rag_mvp.ingestion.extractors import PageUsabilityPolicy


def test_page_usability_is_versioned_and_bilingual() -> None:
    policy = PageUsabilityPolicy(minimum_alphanumeric_characters=4)

    assert policy.version == "page-usability-v1"
    assert policy.is_usable("公司 Policy")
    assert not policy.is_usable(" -- ")


def test_page_usability_threshold_is_deterministic() -> None:
    policy = PageUsabilityPolicy(minimum_alphanumeric_characters=5)

    assert not policy.is_usable("abcd")
    assert policy.is_usable("abcde")


@pytest.mark.parametrize(
    "values",
    [
        {"version": ""},
        {"minimum_alphanumeric_characters": 0},
        {"minimum_alphanumeric_characters": 1.5},
        {"minimum_alphanumeric_characters": True},
        {"minimum_printable_ratio": -0.1},
        {"minimum_printable_ratio": 1.1},
        {"minimum_printable_ratio": math.inf},
        {"minimum_printable_ratio": math.nan},
        {"minimum_printable_ratio": True},
    ],
)
def test_page_usability_rejects_invalid_configuration(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PageUsabilityPolicy(**values)  # type: ignore[arg-type]
