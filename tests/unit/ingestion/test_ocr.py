from __future__ import annotations

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
