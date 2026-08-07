import pytest

from rag_mvp.domain.ingestion import ChunkLocator
from rag_mvp.domain.qa import (
    AnswerClaim,
    Citation,
    RefusalReason,
    SafeQADiagnostics,
    StreamEventKind,
    ValidatedStreamEvent,
)
from rag_mvp.evaluation.compliance import (
    ComplianceAssessmentError,
    assess_compliance_obligations,
)
from rag_mvp.evaluation.dataset import (
    Answerability,
    ComplianceObligation,
    ComplianceObligationKind,
    EvaluationCaseV2,
    EvaluationCategory,
    EvaluationLanguage,
    ExpectedFactV2,
    RefusalGuidanceExpectation,
    ResponseInstruction,
    StyleExpectation,
    SupportAnchorGroup,
)
from rag_mvp.qa.refusal_guidance import DEFAULT_REFUSAL_GUIDANCE_CATALOG


def _obligation(
    obligation_id: str,
    kind: ComplianceObligationKind,
    *expected: str,
) -> ComplianceObligation:
    return ComplianceObligation(
        obligation_id=obligation_id,
        version="2.0.0",
        instruction_id=f"instruction-{obligation_id}",
        kind=kind,
        description="Machine-checkable response requirement.",
        expected_values=expected,
    )


def _case(
    *obligations: ComplianceObligation,
    language: EvaluationLanguage = EvaluationLanguage.ENGLISH,
) -> EvaluationCaseV2:
    refusal = any(
        obligation.kind is ComplianceObligationKind.REFUSAL_GUIDANCE for obligation in obligations
    )
    return EvaluationCaseV2(
        case_id="case-compliance",
        question="What does the policy say?",
        language=language,
        answerability=(Answerability.UNANSWERABLE if refusal else Answerability.ANSWERABLE),
        category=(
            EvaluationCategory.UNANSWERABLE
            if refusal
            else EvaluationCategory.ANSWERABLE_CHINESE
            if language is EvaluationLanguage.CHINESE
            else EvaluationCategory.ANSWERABLE_ENGLISH
        ),
        expected_facts=(
            ()
            if refusal
            else (
                ExpectedFactV2(
                    fact_id="fact-compliance",
                    text="The required code is OPS-RAG-7421.",
                    evidence_ids=("chunk-authoritative",),
                    support_anchor_groups=(
                        SupportAnchorGroup(
                            group_id="authoritative-code",
                            alternatives=("OPS-RAG-7421",),
                        ),
                    ),
                    approved_propositions=("The required code is OPS-RAG-7421.",),
                ),
            )
        ),
        authoritative_evidence_ids=() if refusal else ("chunk-authoritative",),
        permitted_source_ids=() if refusal else ("source-authoritative",),
        response_instructions=tuple(
            ResponseInstruction(
                instruction_id=obligation.instruction_id,
                text=obligation.description,
            )
            for obligation in obligations
        ),
        compliance_obligations=obligations,
        refusal_expectation=RefusalGuidanceExpectation(
            expected=refusal,
            reason_codes=("out-of-scope",) if refusal else (),
            guidance_required=refusal,
            language=language,
        ),
        challenge_tags=(),
        style_expectations=(
            StyleExpectation.ANSWER_IN_REQUEST_LANGUAGE,
            StyleExpectation.REFUSAL_CONCISE if refusal else StyleExpectation.CITATIONS_REQUIRED,
        ),
    )


def _answer(content: str = "The required code is OPS-RAG-7421.") -> ValidatedStreamEvent:
    return ValidatedStreamEvent(
        request_id="request-compliance",
        session_id="session-compliance",
        sequence=0,
        kind=StreamEventKind.ANSWER,
        response_language="en",
        content=content,
        claims=(AnswerClaim(text=content, citation_chunk_ids=("chunk-authoritative",)),),
        citations=(
            Citation(
                source_title="Policy",
                document_version=1,
                chunk_id="chunk-authoritative",
                locator=ChunkLocator(pages=(1,)),
            ),
        ),
        terminal=True,
    )


def _refusal(*, guidance: bool) -> ValidatedStreamEvent:
    metadata: dict[str, str | bool] = {}
    if guidance:
        template = DEFAULT_REFUSAL_GUIDANCE_CATALOG.select(
            RefusalReason.OUT_OF_SCOPE,
            "en",
        )
        metadata = {
            "refusal_guidance_present": True,
            "refusal_reason_code": RefusalReason.OUT_OF_SCOPE.value,
            "refusal_guidance_reason_code": template.reason_code.value,
            "refusal_guidance_template_id": template.template_id,
            "refusal_guidance_catalog_version": template.catalog_version,
            "refusal_guidance_language": template.response_language.value,
        }
        content = template.message
    else:
        content = "Please provide the relevant internal document or narrow the question."
    return ValidatedStreamEvent(
        request_id="request-refusal",
        session_id="session-refusal",
        sequence=0,
        kind=StreamEventKind.REFUSAL,
        response_language="en",
        content=content,
        reason=RefusalReason.OUT_OF_SCOPE,
        diagnostics=SafeQADiagnostics(metadata=metadata),
        terminal=True,
    )


def test_deterministic_assessor_checks_every_answer_obligation() -> None:
    case = _case(
        _obligation("language", ComplianceObligationKind.RESPONSE_LANGUAGE, "en"),
        _obligation("citations", ComplianceObligationKind.CITATION_BEHAVIOR, "required"),
        _obligation(
            "required-code",
            ComplianceObligationKind.REQUIRED_CONTENT,
            "OPS-RAG-7421",
        ),
        _obligation(
            "no-pii",
            ComplianceObligationKind.PROHIBITED_CONTENT,
            "email-address",
            "phone-number",
        ),
    )

    assessments = assess_compliance_obligations(case, _answer())

    assert tuple(item.obligation_id for item in assessments) == (
        "language",
        "citations",
        "required-code",
        "no-pii",
    )
    assert all(item.satisfied for item in assessments)
    assert assessments[1].evidence_references == ("chunk-authoritative",)


def test_prohibited_content_is_content_safe_and_fails_the_obligation() -> None:
    case = _case(
        _obligation(
            "no-pii",
            ComplianceObligationKind.PROHIBITED_CONTENT,
            "email-address",
            "phone-number",
        )
    )

    assessment = assess_compliance_obligations(
        case,
        _answer("Contact person@example.com for the required code."),
    )[0]

    assert not assessment.satisfied
    assert assessment.rationale == "prohibited-content-detected"
    assert "person@example.com" not in repr(assessment)


def test_refusal_guidance_requires_versioned_metadata_not_merely_refusal_text() -> None:
    case = _case(_obligation("guidance", ComplianceObligationKind.REFUSAL_GUIDANCE, "present"))

    missing = assess_compliance_obligations(case, _refusal(guidance=False))[0]
    present = assess_compliance_obligations(case, _refusal(guidance=True))[0]

    assert not missing.satisfied
    assert missing.rationale == "refusal-guidance-mismatch"
    assert present.satisfied
    assert present.evidence_references == ("refusal-guidance-v1.out-of-scope.en",)


def test_refusal_guidance_rejects_mismatched_catalog_identity_and_static_message() -> None:
    case = _case(_obligation("guidance", ComplianceObligationKind.REFUSAL_GUIDANCE, "present"))
    valid = _refusal(guidance=True)
    wrong_metadata = valid.diagnostics.model_copy(
        update={
            "metadata": {
                **valid.diagnostics.metadata,
                "refusal_guidance_reason_code": "safety",
            }
        }
    )

    mismatched_identity = assess_compliance_obligations(
        case,
        valid.model_copy(update={"diagnostics": wrong_metadata}),
    )[0]
    non_static_message = assess_compliance_obligations(
        case,
        valid.model_copy(update={"content": "A different, unaudited next step."}),
    )[0]

    assert not mismatched_identity.satisfied
    assert not non_static_message.satisfied
    assert mismatched_identity.rationale == "refusal-guidance-mismatch"
    assert non_static_message.rationale == "refusal-guidance-mismatch"


def test_refusal_guidance_fails_closed_for_legacy_constructed_case() -> None:
    valid = _case(_obligation("guidance", ComplianceObligationKind.REFUSAL_GUIDANCE, "present"))
    fields = {
        name: getattr(valid, name)
        for name in type(valid).model_fields
        if name != "refusal_expectation"
    }
    invalid = EvaluationCaseV2.model_construct(**fields)

    with pytest.raises(
        ComplianceAssessmentError,
        match="compliance_refusal_expectation_missing",
    ):
        assess_compliance_obligations(invalid, _refusal(guidance=True))


def test_unsupported_machine_check_fails_closed() -> None:
    case = _case(
        _obligation("unknown-pii", ComplianceObligationKind.PROHIBITED_CONTENT, "unknown-kind")
    )

    assessment = assess_compliance_obligations(case, _answer())[0]

    assert not assessment.satisfied
    assert assessment.rationale == "prohibited-content-kind-unsupported"
