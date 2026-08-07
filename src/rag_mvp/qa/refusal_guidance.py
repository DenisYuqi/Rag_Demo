"""Versioned, bounded, bilingual guidance for policy-decided refusals."""

from __future__ import annotations

from enum import StrEnum
from itertools import product
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from rag_mvp.domain._base import DomainModel, NonEmptyText
from rag_mvp.domain.qa import RefusalReason

REFUSAL_GUIDANCE_CATALOG_VERSION: Literal["refusal-guidance-v1"] = "refusal-guidance-v1"
MAX_REFUSAL_GUIDANCE_MESSAGE_CHARACTERS = 640

type GuidanceText = Annotated[NonEmptyText, Field(max_length=320)]
type GuidanceTemplateId = Annotated[
    str,
    Field(pattern=r"^refusal-guidance-v1\.[a-z][a-z0-9-]*\.(?:en|zh-CN)$"),
]


class RefusalGuidanceReason(StrEnum):
    """Canonical v2 guidance reasons, separate from retained v1 aliases."""

    LOW_CONFIDENCE = "low-confidence"
    OUT_OF_SCOPE = "out-of-scope"
    CONFLICTING_EVIDENCE = "conflicting-evidence"
    PROMPT_INJECTION = "prompt-injection"
    SAFETY = "safety"


class RefusalGuidanceLanguage(StrEnum):
    ENGLISH = "en"
    CHINESE = "zh-CN"


class RefusalGuidanceTemplate(DomainModel):
    """One static explanation plus at least one static next action."""

    catalog_version: Literal["refusal-guidance-v1"] = REFUSAL_GUIDANCE_CATALOG_VERSION
    template_id: GuidanceTemplateId
    reason_code: RefusalGuidanceReason
    response_language: RefusalGuidanceLanguage
    explanation: GuidanceText
    next_steps: Annotated[tuple[GuidanceText, ...], Field(min_length=1, max_length=2)]

    @property
    def message(self) -> str:
        return " ".join((self.explanation, *self.next_steps))

    @model_validator(mode="after")
    def validate_static_bounded_content(self) -> Self:
        expected_template_id = (
            f"{self.catalog_version}.{self.reason_code.value}.{self.response_language.value}"
        )
        if self.template_id != expected_template_id:
            raise ValueError("refusal guidance template identity is inconsistent")
        fragments = (self.explanation, *self.next_steps)
        if any("{" in fragment or "}" in fragment for fragment in fragments):
            raise ValueError("refusal guidance cannot contain interpolation placeholders")
        if len(self.message) > MAX_REFUSAL_GUIDANCE_MESSAGE_CHARACTERS:
            raise ValueError("refusal guidance message exceeds its release bound")
        return self


class RefusalGuidanceCatalog(DomainModel):
    """A complete immutable five-reason by two-language template registry."""

    catalog_version: Literal["refusal-guidance-v1"] = REFUSAL_GUIDANCE_CATALOG_VERSION
    templates: Annotated[
        tuple[RefusalGuidanceTemplate, ...],
        Field(min_length=10, max_length=10),
    ]

    @model_validator(mode="after")
    def validate_complete_registry(self) -> Self:
        expected = set(product(RefusalGuidanceReason, RefusalGuidanceLanguage))
        actual = {(template.reason_code, template.response_language) for template in self.templates}
        if actual != expected or len(actual) != len(self.templates):
            raise ValueError("refusal guidance catalog must cover every reason and language")
        if len({template.template_id for template in self.templates}) != len(self.templates):
            raise ValueError("refusal guidance template IDs must be unique")
        if any(template.catalog_version != self.catalog_version for template in self.templates):
            raise ValueError("refusal guidance catalog versions disagree")
        return self

    def select(
        self,
        reason: RefusalGuidanceReason | RefusalReason,
        language: RefusalGuidanceLanguage | str,
    ) -> RefusalGuidanceTemplate:
        resolved_reason = canonical_guidance_reason(reason)
        try:
            resolved_language = RefusalGuidanceLanguage(language)
        except (TypeError, ValueError):
            raise ValueError("unsupported refusal guidance language") from None
        return next(
            template
            for template in self.templates
            if template.reason_code is resolved_reason
            and template.response_language is resolved_language
        )


def canonical_guidance_reason(
    reason: RefusalGuidanceReason | RefusalReason,
) -> RefusalGuidanceReason:
    """Map legacy terminal reason codes to their canonical guidance taxonomy."""

    if isinstance(reason, RefusalGuidanceReason):
        return reason
    try:
        return _REFUSAL_REASON_COMPATIBILITY[reason]
    except (KeyError, TypeError):
        raise ValueError("unsupported refusal guidance reason") from None


def _template(
    reason: RefusalGuidanceReason,
    language: RefusalGuidanceLanguage,
    explanation: str,
    *next_steps: str,
) -> RefusalGuidanceTemplate:
    return RefusalGuidanceTemplate(
        template_id=(f"{REFUSAL_GUIDANCE_CATALOG_VERSION}.{reason.value}.{language.value}"),
        reason_code=reason,
        response_language=language,
        explanation=explanation,
        next_steps=next_steps,
    )


_REFUSAL_REASON_COMPATIBILITY: dict[RefusalReason, RefusalGuidanceReason] = {
    RefusalReason.INSUFFICIENT_EVIDENCE: RefusalGuidanceReason.LOW_CONFIDENCE,
    RefusalReason.CONFLICTING_EVIDENCE: RefusalGuidanceReason.CONFLICTING_EVIDENCE,
    RefusalReason.UNSAFE_REQUEST: RefusalGuidanceReason.SAFETY,
    RefusalReason.LOW_CONFIDENCE: RefusalGuidanceReason.LOW_CONFIDENCE,
    RefusalReason.OUT_OF_SCOPE: RefusalGuidanceReason.OUT_OF_SCOPE,
    RefusalReason.PROMPT_INJECTION: RefusalGuidanceReason.PROMPT_INJECTION,
    RefusalReason.SAFETY: RefusalGuidanceReason.SAFETY,
}

DEFAULT_REFUSAL_GUIDANCE_CATALOG = RefusalGuidanceCatalog(
    templates=(
        _template(
            RefusalGuidanceReason.LOW_CONFIDENCE,
            RefusalGuidanceLanguage.ENGLISH,
            "The available evidence is not strong enough to support a reliable answer.",
            "Please narrow the question or identify a relevant document or section.",
        ),
        _template(
            RefusalGuidanceReason.LOW_CONFIDENCE,
            RefusalGuidanceLanguage.CHINESE,
            "现有证据不足以支持可靠回答。",
            "请缩小问题范围。也可指出相关文档或章节。",
        ),
        _template(
            RefusalGuidanceReason.OUT_OF_SCOPE,
            RefusalGuidanceLanguage.ENGLISH,
            "This request is outside the knowledge currently available to me.",
            (
                "Please reframe it around the indexed documents or provide a relevant "
                "document or section."
            ),
        ),
        _template(
            RefusalGuidanceReason.OUT_OF_SCOPE,
            RefusalGuidanceLanguage.CHINESE,
            "此请求超出了当前可用知识的范围。",
            "请围绕已索引文档重新提问。也可提供相关文档或章节。",
        ),
        _template(
            RefusalGuidanceReason.CONFLICTING_EVIDENCE,
            RefusalGuidanceLanguage.ENGLISH,
            "The available evidence conflicts, so I cannot choose an unsupported interpretation.",
            (
                "Please clarify the applicable version, date, department, or responsible "
                "document owner."
            ),
        ),
        _template(
            RefusalGuidanceReason.CONFLICTING_EVIDENCE,
            RefusalGuidanceLanguage.CHINESE,
            "现有证据存在冲突。因此无法选择缺乏支持的解释。",
            "请明确适用的版本、日期、部门或文档负责人。",
        ),
        _template(
            RefusalGuidanceReason.PROMPT_INJECTION,
            RefusalGuidanceLanguage.ENGLISH,
            "I cannot complete this request safely.",
            (
                "Please ask a knowledge-base question that can be answered from the "
                "available documents."
            ),
        ),
        _template(
            RefusalGuidanceReason.PROMPT_INJECTION,
            RefusalGuidanceLanguage.CHINESE,
            "我无法安全地完成此请求。",
            "请提出能够依据现有文档回答的知识库问题。",
        ),
        _template(
            RefusalGuidanceReason.SAFETY,
            RefusalGuidanceLanguage.ENGLISH,
            "I cannot help with this request safely.",
            "Please reframe it as a benign question about the available documents.",
        ),
        _template(
            RefusalGuidanceReason.SAFETY,
            RefusalGuidanceLanguage.CHINESE,
            "我无法安全地协助此请求。",
            "请将其改为关于现有文档的安全问题。",
        ),
    )
)


__all__ = [
    "DEFAULT_REFUSAL_GUIDANCE_CATALOG",
    "MAX_REFUSAL_GUIDANCE_MESSAGE_CHARACTERS",
    "REFUSAL_GUIDANCE_CATALOG_VERSION",
    "RefusalGuidanceCatalog",
    "RefusalGuidanceLanguage",
    "RefusalGuidanceReason",
    "RefusalGuidanceTemplate",
    "canonical_guidance_reason",
]
