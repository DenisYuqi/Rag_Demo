from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_mvp.domain.qa import RefusalReason
from rag_mvp.qa.refusal_guidance import (
    DEFAULT_REFUSAL_GUIDANCE_CATALOG,
    MAX_REFUSAL_GUIDANCE_MESSAGE_CHARACTERS,
    REFUSAL_GUIDANCE_CATALOG_VERSION,
    RefusalGuidanceCatalog,
    RefusalGuidanceLanguage,
    RefusalGuidanceReason,
    canonical_guidance_reason,
)


def test_default_catalog_is_complete_versioned_bilingual_and_bounded() -> None:
    catalog = DEFAULT_REFUSAL_GUIDANCE_CATALOG

    assert catalog.catalog_version == REFUSAL_GUIDANCE_CATALOG_VERSION
    assert len(catalog.templates) == 10
    assert {
        (template.reason_code, template.response_language) for template in catalog.templates
    } == {
        (reason, language)
        for reason in RefusalGuidanceReason
        for language in RefusalGuidanceLanguage
    }
    for template in catalog.templates:
        assert template.next_steps
        assert len(template.message) <= MAX_REFUSAL_GUIDANCE_MESSAGE_CHARACTERS
        assert "{" not in template.message
        assert "}" not in template.message


@pytest.mark.parametrize(
    ("legacy_reason", "canonical_reason"),
    [
        (RefusalReason.INSUFFICIENT_EVIDENCE, RefusalGuidanceReason.LOW_CONFIDENCE),
        (RefusalReason.CONFLICTING_EVIDENCE, RefusalGuidanceReason.CONFLICTING_EVIDENCE),
        (RefusalReason.UNSAFE_REQUEST, RefusalGuidanceReason.SAFETY),
        (RefusalReason.LOW_CONFIDENCE, RefusalGuidanceReason.LOW_CONFIDENCE),
        (RefusalReason.OUT_OF_SCOPE, RefusalGuidanceReason.OUT_OF_SCOPE),
        (RefusalReason.PROMPT_INJECTION, RefusalGuidanceReason.PROMPT_INJECTION),
        (RefusalReason.SAFETY, RefusalGuidanceReason.SAFETY),
    ],
)
def test_legacy_and_extended_reason_taxonomies_map_deterministically(
    legacy_reason: RefusalReason,
    canonical_reason: RefusalGuidanceReason,
) -> None:
    selected = DEFAULT_REFUSAL_GUIDANCE_CATALOG.select(legacy_reason, "en")

    assert canonical_guidance_reason(legacy_reason) is canonical_reason
    assert selected.reason_code is canonical_reason
    assert selected.template_id == (
        f"{REFUSAL_GUIDANCE_CATALOG_VERSION}.{canonical_reason.value}.en"
    )


def test_chinese_and_english_templates_are_distinct_static_messages() -> None:
    english = DEFAULT_REFUSAL_GUIDANCE_CATALOG.select(
        RefusalGuidanceReason.OUT_OF_SCOPE,
        "en",
    )
    chinese = DEFAULT_REFUSAL_GUIDANCE_CATALOG.select(
        RefusalGuidanceReason.OUT_OF_SCOPE,
        "zh-CN",
    )

    assert english.message != chinese.message
    assert "indexed documents" in english.message
    assert "已索引文档" in chinese.message
    assert english.response_language is RefusalGuidanceLanguage.ENGLISH
    assert chinese.response_language is RefusalGuidanceLanguage.CHINESE


def test_catalog_rejects_incomplete_or_duplicate_reason_language_coverage() -> None:
    templates = DEFAULT_REFUSAL_GUIDANCE_CATALOG.templates
    payload = DEFAULT_REFUSAL_GUIDANCE_CATALOG.model_dump()
    payload["templates"] = (*templates[:-1], templates[0])

    with pytest.raises(ValidationError, match="cover every reason and language"):
        RefusalGuidanceCatalog.model_validate(payload)


def test_catalog_rejects_dynamic_interpolation_placeholders() -> None:
    template = DEFAULT_REFUSAL_GUIDANCE_CATALOG.templates[0]
    payload = template.model_dump()
    payload["explanation"] = "Unsafe dynamic value: {raw_user_input}"

    with pytest.raises(ValidationError, match="interpolation placeholders"):
        type(template).model_validate(payload)


def test_catalog_selection_never_accepts_an_unbounded_language() -> None:
    with pytest.raises(ValueError, match="unsupported refusal guidance language"):
        DEFAULT_REFUSAL_GUIDANCE_CATALOG.select(
            RefusalGuidanceReason.SAFETY,
            "fr",
        )
