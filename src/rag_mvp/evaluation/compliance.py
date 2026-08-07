"""Deterministic, content-safe assessment of dataset-v2 compliance obligations."""

from __future__ import annotations

import unicodedata
from collections.abc import Callable

from rag_mvp.domain.qa import StreamEventKind, ValidatedStreamEvent
from rag_mvp.evaluation.answer_metrics import ComplianceAssessment
from rag_mvp.evaluation.dataset import (
    ComplianceObligation,
    ComplianceObligationKind,
    EvaluationCaseV2,
    EvaluationLanguage,
    RefusalGuidanceExpectation,
)
from rag_mvp.qa.refusal_guidance import DEFAULT_REFUSAL_GUIDANCE_CATALOG
from rag_mvp.safety.models import SensitiveKind
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, RedactionError, Redactor

_PROHIBITED_KIND = {
    "email-address": SensitiveKind.EMAIL,
    "phone-number": SensitiveKind.PHONE,
    "chinese-id": SensitiveKind.CHINESE_ID,
    "ssn": SensitiveKind.SSN,
    "payment-card": SensitiveKind.PAYMENT_CARD,
    "ipv4-address": SensitiveKind.IPV4,
    "ipv6-address": SensitiveKind.IPV6,
    "secret": SensitiveKind.SECRET,
}


class ComplianceAssessmentError(ValueError):
    """A stable input error that never includes response or obligation content."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def assess_compliance_obligations(
    case: EvaluationCaseV2,
    event: ValidatedStreamEvent,
    *,
    redactor: Redactor = DEFAULT_REDACTOR,
) -> tuple[ComplianceAssessment, ...]:
    """Assess every declared obligation independently and fail closed on uncertainty."""

    if not isinstance(case, EvaluationCaseV2):
        raise ComplianceAssessmentError("compliance_case_v2_required")
    if not isinstance(event, ValidatedStreamEvent):
        raise ComplianceAssessmentError("compliance_event_invalid")
    if not isinstance(redactor, Redactor) or not redactor.fully_configured:
        raise ComplianceAssessmentError("compliance_redactor_unavailable")
    assessors: dict[
        ComplianceObligationKind,
        Callable[[ComplianceObligation], ComplianceAssessment],
    ] = {
        ComplianceObligationKind.REQUIRED_CONTENT: lambda obligation: _required_content(
            obligation,
            event,
        ),
        ComplianceObligationKind.PROHIBITED_CONTENT: lambda obligation: _prohibited_content(
            obligation,
            event,
            redactor,
        ),
        ComplianceObligationKind.RESPONSE_LANGUAGE: lambda obligation: _response_language(
            obligation,
            case.language,
            event,
        ),
        ComplianceObligationKind.RESPONSE_FORMAT: lambda obligation: _response_format(
            obligation,
            event,
        ),
        ComplianceObligationKind.CITATION_BEHAVIOR: lambda obligation: _citation_behavior(
            obligation,
            event,
        ),
        ComplianceObligationKind.REFUSAL_GUIDANCE: lambda obligation: _refusal_guidance(
            obligation,
            case,
            event,
        ),
    }
    return tuple(
        assessors[obligation.kind](obligation) for obligation in case.compliance_obligations
    )


def _required_content(
    obligation: ComplianceObligation,
    event: ValidatedStreamEvent,
) -> ComplianceAssessment:
    content = _canonical_text(event.content or "")
    expected = tuple(_canonical_text(value) for value in obligation.expected_values)
    satisfied = bool(content) and all(value in content for value in expected)
    return _assessment(
        obligation,
        satisfied,
        "required-content-present" if satisfied else "required-content-missing",
    )


def _prohibited_content(
    obligation: ComplianceObligation,
    event: ValidatedStreamEvent,
    redactor: Redactor,
) -> ComplianceAssessment:
    expected_kinds = tuple(_PROHIBITED_KIND.get(value) for value in obligation.expected_values)
    if any(kind is None for kind in expected_kinds):
        return _assessment(obligation, False, "prohibited-content-kind-unsupported")
    try:
        detected = {span.kind for span in redactor.detect(event.content or "")}
    except RedactionError:
        return _assessment(obligation, False, "prohibited-content-evidence-unavailable")
    satisfied = detected.isdisjoint(expected_kinds)
    return _assessment(
        obligation,
        satisfied,
        "prohibited-content-absent" if satisfied else "prohibited-content-detected",
    )


def _response_language(
    obligation: ComplianceObligation,
    case_language: EvaluationLanguage,
    event: ValidatedStreamEvent,
) -> ComplianceAssessment:
    expected = set(obligation.expected_values)
    declared = event.response_language.casefold()
    actual = (
        "zh"
        if declared == "zh" or declared.startswith("zh-")
        else "en"
        if declared == "en" or declared.startswith("en-")
        else "mixed"
    )
    satisfied = len(expected) == 1 and actual in expected and case_language.value in expected
    return _assessment(
        obligation,
        satisfied,
        "response-language-matched" if satisfied else "response-language-mismatch",
        ("response-language",),
    )


def _response_format(
    obligation: ComplianceObligation,
    event: ValidatedStreamEvent,
) -> ComplianceAssessment:
    expected = set(obligation.expected_values)
    supported = {"answer", "refusal", "terminal", "structured"}
    if not expected.issubset(supported):
        return _assessment(obligation, False, "response-format-unsupported")
    satisfied = all(
        (
            value == event.kind.value
            if value in {"answer", "refusal"}
            else event.terminal
            if value == "terminal"
            else event.kind in {StreamEventKind.ANSWER, StreamEventKind.REFUSAL}
        )
        for value in expected
    )
    return _assessment(
        obligation,
        satisfied,
        "response-format-matched" if satisfied else "response-format-mismatch",
    )


def _citation_behavior(
    obligation: ComplianceObligation,
    event: ValidatedStreamEvent,
) -> ComplianceAssessment:
    expected = set(obligation.expected_values)
    if expected == {"required"}:
        satisfied = (
            event.kind is StreamEventKind.ANSWER and bool(event.claims) and bool(event.citations)
        )
    elif expected == {"forbidden"}:
        satisfied = not event.claims and not event.citations
    else:
        return _assessment(obligation, False, "citation-behavior-unsupported")
    return _assessment(
        obligation,
        satisfied,
        "citation-behavior-matched" if satisfied else "citation-behavior-mismatch",
        tuple(citation.chunk_id for citation in event.citations),
    )


def _refusal_guidance(
    obligation: ComplianceObligation,
    case: EvaluationCaseV2,
    event: ValidatedStreamEvent,
) -> ComplianceAssessment:
    refusal_expectation = getattr(case, "refusal_expectation", None)
    if not isinstance(refusal_expectation, RefusalGuidanceExpectation):
        raise ComplianceAssessmentError("compliance_refusal_expectation_missing")
    expected = set(obligation.expected_values)
    if expected not in ({"present"}, {"absent"}):
        return _assessment(obligation, False, "refusal-guidance-unsupported")
    metadata = event.diagnostics.metadata
    template_id = metadata.get("refusal_guidance_template_id")
    present = False
    if event.kind is StreamEventKind.REFUSAL and event.reason is not None:
        response_language = (
            "zh-CN" if case.language is EvaluationLanguage.CHINESE else case.language.value
        )
        try:
            template = DEFAULT_REFUSAL_GUIDANCE_CATALOG.select(
                event.reason,
                response_language,
            )
        except (StopIteration, TypeError, ValueError):
            template = None
        present = bool(
            template is not None
            and refusal_expectation.guidance_required
            and event.reason.value in set(refusal_expectation.reason_codes)
            and event.response_language.casefold()
            in {case.language.value.casefold(), response_language.casefold()}
            and event.content == template.message
            and metadata.get("refusal_guidance_present") is True
            and metadata.get("refusal_reason_code") == event.reason.value
            and metadata.get("refusal_guidance_reason_code") == template.reason_code.value
            and template_id == template.template_id
            and metadata.get("refusal_guidance_catalog_version") == template.catalog_version
            and metadata.get("refusal_guidance_language") == template.response_language.value
        )
    satisfied = present if expected == {"present"} else not present
    references = (template_id,) if isinstance(template_id, str) and template_id else ()
    return _assessment(
        obligation,
        satisfied,
        "refusal-guidance-matched" if satisfied else "refusal-guidance-mismatch",
        references,
    )


def _assessment(
    obligation: ComplianceObligation,
    satisfied: bool,
    rationale: str,
    evidence_references: tuple[str, ...] = (),
) -> ComplianceAssessment:
    return ComplianceAssessment(
        obligation_id=obligation.obligation_id,
        satisfied=satisfied,
        rationale=rationale,
        evidence_references=tuple(dict.fromkeys(evidence_references)),
    )


def _canonical_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


__all__ = [
    "ComplianceAssessmentError",
    "assess_compliance_obligations",
]
