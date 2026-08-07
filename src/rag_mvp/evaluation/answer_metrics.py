"""Versioned deterministic answer and refusal metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from rag_mvp.domain.evaluation import (
    MetricObservation,
    MetricObservationStatus,
    UnavailableValue,
)
from rag_mvp.evaluation.grounding_metrics import (
    EvidenceVerdict,
    MetricEvidence,
    MetricInputError,
    MetricName,
    MetricResult,
)

ANSWER_COMPLETENESS_SCORER_VERSION = "answer-completeness-expected-facts-v1"
ANSWER_COMPLIANCE_METRIC_ID = "answer-compliance"
ANSWER_COMPLIANCE_SCORER_VERSION = "answer-compliance-all-obligations-v1"
STYLE_CONSISTENCY_SCORER_VERSION = "style-consistency-applicable-checks-v1"
REFUSAL_APPROPRIATENESS_SCORER_VERSION = "refusal-appropriateness-outcome-v1"
GUIDED_REFUSAL_APPROPRIATENESS_SCORER_VERSION = "refusal-appropriateness-reason-guidance-v2"


@dataclass(frozen=True, slots=True)
class StyleAssessment:
    """One deterministic check for an applicable dataset style expectation."""

    expectation_id: str
    satisfied: bool
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expectation_id",
            _identifier(self.expectation_id, "expectation_id"),
        )
        if type(self.satisfied) is not bool:
            raise ValueError("style_verdict_invalid")
        object.__setattr__(self, "rationale", _text(self.rationale, "style_rationale"))


@dataclass(frozen=True, slots=True)
class ComplianceAssessment:
    """Deterministic verdict for one case-local compliance obligation ID."""

    obligation_id: str
    satisfied: bool
    rationale: str
    evidence_references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "obligation_id",
            _identifier(self.obligation_id, "obligation_id"),
        )
        if type(self.satisfied) is not bool:
            raise ValueError("compliance_verdict_invalid")
        object.__setattr__(
            self,
            "rationale",
            _text(self.rationale, "compliance_rationale"),
        )
        object.__setattr__(
            self,
            "evidence_references",
            _identifiers(
                self.evidence_references,
                "compliance_evidence_references",
                allow_empty=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class AnswerComplianceResult:
    """Binary all-obligations result for one eligible answerable case."""

    case_id: str
    scorer_version: str
    scored: bool
    eligible: bool
    score: float | None
    numerator: int | None
    denominator: int | None
    rationale: str
    assessments: tuple[ComplianceAssessment, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _identifier(self.case_id, "case_id"))
        object.__setattr__(
            self,
            "scorer_version",
            _identifier(self.scorer_version, "scorer_version"),
        )
        object.__setattr__(self, "rationale", _text(self.rationale, "compliance_rationale"))
        assessments = _compliance_assessments(self.assessments)
        object.__setattr__(self, "assessments", assessments)
        if type(self.eligible) is not bool:
            raise ValueError("compliance_eligibility_invalid")
        if type(self.scored) is not bool:
            raise ValueError("compliance_scored_invalid")
        if not self.scored:
            if self.eligible:
                raise ValueError("unscored_compliance_cannot_be_aggregate_eligible")
            if any(value is not None for value in (self.score, self.numerator, self.denominator)):
                raise ValueError("unscored_compliance_must_not_have_score")
            if assessments:
                raise ValueError("unscored_compliance_must_not_have_assessments")
            return
        if not assessments:
            raise ValueError("scored_compliance_requires_obligations")
        if self.denominator != 1 or self.numerator not in {0, 1}:
            raise ValueError("compliance_case_denominator_invalid")
        if self.score != float(self.numerator):
            raise ValueError("compliance_case_formula_mismatch")

    def to_metric_observation(self) -> MetricObservation:
        """Expose this case result through the same explicit schema-v2 value contract."""

        if not self.scored:
            missing = UnavailableValue(reason="case-not-answerable")
            return MetricObservation(
                metric_id=ANSWER_COMPLIANCE_METRIC_ID,
                unit="ratio",
                value=missing,
                numerator=missing,
                denominator=missing,
                eligible=False,
                scorer_version=self.scorer_version,
                status=MetricObservationStatus.UNAVAILABLE,
            )
        references = tuple(
            dict.fromkeys(
                reference
                for assessment in self.assessments
                for reference in (assessment.obligation_id, *assessment.evidence_references)
            )
        )
        return MetricObservation(
            metric_id=ANSWER_COMPLIANCE_METRIC_ID,
            unit="ratio",
            value=cast(float, self.score),
            numerator=float(cast(int, self.numerator)),
            denominator=cast(int, self.denominator),
            eligible=True,
            scorer_version=self.scorer_version,
            status=MetricObservationStatus.OBSERVED,
            evidence_references=references,
        )


@dataclass(frozen=True, slots=True)
class AnswerComplianceAggregate:
    """Compliant eligible cases divided by all eligible answerable cases."""

    scorer_version: str
    score: float | None
    numerator: int
    denominator: int
    total_cases: int
    case_results: tuple[AnswerComplianceResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scorer_version",
            _identifier(self.scorer_version, "scorer_version"),
        )
        results = tuple(self.case_results)
        if any(not isinstance(result, AnswerComplianceResult) for result in results):
            raise ValueError("compliance_results_invalid")
        if len({result.case_id for result in results}) != len(results):
            raise ValueError("duplicate_compliance_case")
        if any(result.scorer_version != self.scorer_version for result in results):
            raise ValueError("compliance_scorer_version_mismatch")
        if self.total_cases != len(results):
            raise ValueError("compliance_total_case_count_invalid")
        if self.denominator != sum(result.eligible for result in results):
            raise ValueError("compliance_denominator_mismatch")
        if self.numerator != sum(
            cast(int, result.numerator) for result in results if result.eligible
        ):
            raise ValueError("compliance_numerator_mismatch")
        if not 0 <= self.numerator <= self.denominator:
            raise ValueError("compliance_aggregate_counts_invalid")
        if self.denominator == 0:
            if self.score is not None:
                raise ValueError("compliance_without_denominator_must_be_unavailable")
        elif self.score != self.numerator / self.denominator:
            raise ValueError("compliance_aggregate_formula_mismatch")

    def to_metric_observation(self) -> MetricObservation:
        """Return canonical schema-v2 evidence without converting unavailable to zero."""

        if self.denominator == 0:
            missing = UnavailableValue(reason="no-eligible-cases")
            return MetricObservation(
                metric_id=ANSWER_COMPLIANCE_METRIC_ID,
                unit="ratio",
                value=missing,
                numerator=missing,
                denominator=missing,
                eligible=False,
                scorer_version=self.scorer_version,
                status=MetricObservationStatus.UNAVAILABLE,
            )
        return MetricObservation(
            metric_id=ANSWER_COMPLIANCE_METRIC_ID,
            unit="ratio",
            value=cast(float, self.score),
            numerator=float(self.numerator),
            denominator=self.denominator,
            eligible=True,
            scorer_version=self.scorer_version,
            status=MetricObservationStatus.OBSERVED,
            evidence_references=tuple(
                result.case_id for result in self.case_results if result.eligible
            ),
        )


class AnswerComplianceScorer:
    """Require every declared case obligation to pass; no weighted compensation."""

    version = ANSWER_COMPLIANCE_SCORER_VERSION

    def score(
        self,
        *,
        case_id: str,
        answerable: bool,
        response_outcome: str,
        assessments: Sequence[ComplianceAssessment],
    ) -> AnswerComplianceResult:
        resolved_case_id = _identifier(case_id, "case_id")
        if type(answerable) is not bool:
            raise MetricInputError("answerability_invalid")
        outcome = _response_outcome(response_outcome)
        obligations = _compliance_assessments(assessments)
        if not obligations:
            raise MetricInputError("compliance_obligations_missing")

        expected_outcome = "answer" if answerable else "refusal"
        compliant = outcome == expected_outcome and all(item.satisfied for item in obligations)
        if outcome != expected_outcome:
            rationale = f"{expected_outcome}_required_for_compliance"
        elif compliant:
            rationale = f"satisfied_obligations={len(obligations)}; obligations={len(obligations)}"
        else:
            satisfied = sum(item.satisfied for item in obligations)
            rationale = f"satisfied_obligations={satisfied}; obligations={len(obligations)}"
        return AnswerComplianceResult(
            case_id=resolved_case_id,
            scorer_version=self.version,
            scored=True,
            eligible=answerable,
            score=float(compliant),
            numerator=int(compliant),
            denominator=1,
            rationale=rationale,
            assessments=obligations,
        )


def aggregate_answer_compliance(
    results: Sequence[AnswerComplianceResult],
    *,
    scorer_version: str = ANSWER_COMPLIANCE_SCORER_VERSION,
) -> AnswerComplianceAggregate:
    """Aggregate binary case outcomes with explicit compliant/eligible counts."""

    raw_results: object = results
    if isinstance(raw_results, (str, bytes, bytearray)) or not isinstance(raw_results, Sequence):
        raise MetricInputError("compliance_results_invalid")
    values = tuple(cast(Sequence[object], raw_results))
    if any(not isinstance(result, AnswerComplianceResult) for result in values):
        raise MetricInputError("compliance_results_invalid")
    typed = cast(tuple[AnswerComplianceResult, ...], values)
    if len({result.case_id for result in typed}) != len(typed):
        raise MetricInputError("duplicate_compliance_case")
    if any(result.scorer_version != scorer_version for result in typed):
        raise MetricInputError("compliance_scorer_version_mismatch")
    eligible = tuple(result for result in typed if result.eligible)
    numerator = sum(cast(int, result.numerator) for result in eligible)
    denominator = len(eligible)
    return AnswerComplianceAggregate(
        scorer_version=scorer_version,
        score=numerator / denominator if denominator else None,
        numerator=numerator,
        denominator=denominator,
        total_cases=len(typed),
        case_results=typed,
    )


class AnswerCompletenessScorer:
    """Score covered expected facts divided by all expected facts."""

    version = ANSWER_COMPLETENESS_SCORER_VERSION

    def score(
        self,
        *,
        case_id: str,
        answerable: bool,
        response_outcome: str,
        expected_fact_ids: Sequence[str],
        covered_fact_ids: Sequence[str],
    ) -> MetricResult:
        resolved_case_id = _identifier(case_id, "case_id")
        if type(answerable) is not bool:
            raise MetricInputError("answerability_invalid")
        outcome = _response_outcome(response_outcome)
        expected = _identifiers(expected_fact_ids, "expected_fact_ids", allow_empty=True)
        covered = _identifiers(covered_fact_ids, "covered_fact_ids", allow_empty=True)
        if not set(covered).issubset(expected):
            raise MetricInputError("covered_fact_not_expected")
        if outcome != "answer" and covered:
            raise MetricInputError("covered_fact_without_answer")

        if not answerable:
            return _ineligible(
                resolved_case_id,
                MetricName.ANSWER_COMPLETENESS,
                self.version,
                "case_not_answerable",
            )
        if not expected:
            return _ineligible(
                resolved_case_id,
                MetricName.ANSWER_COMPLETENESS,
                self.version,
                "no_expected_facts",
            )
        if outcome == "error":
            return _ineligible(
                resolved_case_id,
                MetricName.ANSWER_COMPLETENESS,
                self.version,
                "response_error",
            )

        covered_set = set(covered)
        numerator = len(covered)
        denominator = len(expected)
        evidence = tuple(
            MetricEvidence(
                reference_id=fact_id,
                verdict=(
                    EvidenceVerdict.COVERED if fact_id in covered_set else EvidenceVerdict.MISSING
                ),
                rationale=(
                    "expected_fact_covered"
                    if fact_id in covered_set
                    else "expected_fact_not_covered"
                ),
            )
            for fact_id in expected
        )
        return MetricResult(
            case_id=resolved_case_id,
            metric=MetricName.ANSWER_COMPLETENESS,
            scorer_version=self.version,
            eligible=True,
            score=numerator / denominator,
            numerator=float(numerator),
            denominator=denominator,
            rationale=f"covered_expected_facts={numerator}; expected_facts={denominator}",
            evidence=evidence,
        )


class StyleConsistencyScorer:
    """Score satisfied checks across only the style expectations applicable to a case."""

    version = STYLE_CONSISTENCY_SCORER_VERSION

    def score(
        self,
        *,
        case_id: str,
        response_outcome: str,
        assessments: Sequence[StyleAssessment],
    ) -> MetricResult:
        resolved_case_id = _identifier(case_id, "case_id")
        outcome = _response_outcome(response_outcome)
        checks = _style_assessments(assessments)
        if outcome == "error":
            return _ineligible(
                resolved_case_id,
                MetricName.STYLE_CONSISTENCY,
                self.version,
                "response_error",
            )
        if not checks:
            return _ineligible(
                resolved_case_id,
                MetricName.STYLE_CONSISTENCY,
                self.version,
                "no_applicable_style_expectations",
            )

        satisfied = sum(check.satisfied for check in checks)
        denominator = len(checks)
        evidence = tuple(
            MetricEvidence(
                reference_id=check.expectation_id,
                verdict=(
                    EvidenceVerdict.SATISFIED if check.satisfied else EvidenceVerdict.VIOLATED
                ),
                rationale=check.rationale,
            )
            for check in checks
        )
        return MetricResult(
            case_id=resolved_case_id,
            metric=MetricName.STYLE_CONSISTENCY,
            scorer_version=self.version,
            eligible=True,
            score=satisfied / denominator,
            numerator=float(satisfied),
            denominator=denominator,
            rationale=f"satisfied_style_checks={satisfied}; applicable_checks={denominator}",
            evidence=evidence,
        )


class RefusalAppropriatenessScorer:
    """Score whether the answer/refusal outcome matches the dataset label."""

    version = REFUSAL_APPROPRIATENESS_SCORER_VERSION

    def score(
        self,
        *,
        case_id: str,
        expected_refusal: bool,
        response_outcome: str,
        expected_reason: str | None = None,
        expected_reasons: Sequence[str] | None = None,
        actual_reason: str | None = None,
        guidance_compliant: bool | None = None,
        guidance_evidence_references: Sequence[str] = (),
    ) -> MetricResult:
        resolved_case_id = _identifier(case_id, "case_id")
        if type(expected_refusal) is not bool:
            raise MetricInputError("expected_refusal_invalid")
        outcome = _response_outcome(response_outcome)
        resolved_expected_reason = _optional_identifier(expected_reason, "expected_reason")
        if expected_reasons is not None and resolved_expected_reason is not None:
            raise MetricInputError("expected_refusal_reasons_ambiguous")
        resolved_expected_reasons = (
            _identifiers(
                expected_reasons,
                "expected_reasons",
                allow_empty=False,
            )
            if expected_reasons is not None
            else ((resolved_expected_reason,) if resolved_expected_reason is not None else ())
        )
        resolved_actual_reason = _optional_identifier(actual_reason, "actual_reason")
        if resolved_expected_reasons and not expected_refusal:
            raise MetricInputError("reason_without_expected_refusal")
        if guidance_compliant is not None and type(guidance_compliant) is not bool:
            raise MetricInputError("guidance_compliance_invalid")
        guidance_references = _identifiers(
            guidance_evidence_references,
            "guidance_evidence_references",
            allow_empty=True,
        )
        if guidance_references and guidance_compliant is None:
            raise MetricInputError("guidance_evidence_without_assessment")
        if outcome == "answer" and resolved_actual_reason is not None:
            raise MetricInputError("answer_has_refusal_reason")
        if outcome == "refusal" and resolved_actual_reason is None:
            raise MetricInputError("refusal_reason_missing")
        if outcome == "error" and resolved_actual_reason is not None:
            raise MetricInputError("error_has_refusal_reason")
        if outcome == "error":
            return _ineligible(
                resolved_case_id,
                MetricName.REFUSAL_APPROPRIATENESS,
                self.version,
                "response_error",
            )

        actual_refusal = outcome == "refusal"
        reason_matches = not resolved_expected_reasons or resolved_actual_reason in set(
            resolved_expected_reasons
        )
        guidance_matches = guidance_compliant is not False
        appropriate = expected_refusal == actual_refusal and reason_matches and guidance_matches
        rationale = _refusal_rationale(
            expected_refusal=expected_refusal,
            actual_refusal=actual_refusal,
            reason_matches=reason_matches,
            guidance_matches=guidance_matches,
        )
        return MetricResult(
            case_id=resolved_case_id,
            metric=MetricName.REFUSAL_APPROPRIATENESS,
            scorer_version=self.version,
            eligible=True,
            score=float(appropriate),
            numerator=float(appropriate),
            denominator=1,
            rationale=rationale,
            evidence=(
                MetricEvidence(
                    reference_id=resolved_case_id,
                    verdict=(
                        EvidenceVerdict.APPROPRIATE
                        if appropriate
                        else EvidenceVerdict.INAPPROPRIATE
                    ),
                    rationale=rationale,
                    evidence_references=tuple(
                        dict.fromkeys(
                            value
                            for value in (
                                *resolved_expected_reasons,
                                resolved_actual_reason,
                                *guidance_references,
                            )
                            if value is not None
                        )
                    ),
                ),
            ),
        )


class GuidedRefusalAppropriatenessScorer(RefusalAppropriatenessScorer):
    """V2 refusal scorer bound to allowed reasons and validated guidance."""

    version = GUIDED_REFUSAL_APPROPRIATENESS_SCORER_VERSION


def _refusal_rationale(
    *,
    expected_refusal: bool,
    actual_refusal: bool,
    reason_matches: bool,
    guidance_matches: bool = True,
) -> str:
    if expected_refusal and not actual_refusal:
        return "refusal_expected_but_answer_emitted"
    if not expected_refusal and actual_refusal:
        return "answer_expected_but_refusal_emitted"
    if expected_refusal and not reason_matches:
        return "refusal_reason_mismatch"
    if expected_refusal and not guidance_matches:
        return "refusal_guidance_invalid"
    if expected_refusal:
        return "expected_refusal_emitted"
    return "expected_answer_emitted"


def _ineligible(
    case_id: str,
    metric: MetricName,
    scorer_version: str,
    rationale: str,
) -> MetricResult:
    return MetricResult(
        case_id=case_id,
        metric=metric,
        scorer_version=scorer_version,
        eligible=False,
        score=None,
        numerator=None,
        denominator=None,
        rationale=rationale,
    )


def _style_assessments(values: object) -> tuple[StyleAssessment, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise MetricInputError("style_assessments_invalid")
    items = tuple(cast(Sequence[object], values))
    if any(not isinstance(item, StyleAssessment) for item in items):
        raise MetricInputError("style_assessments_invalid")
    assessments = cast(tuple[StyleAssessment, ...], items)
    if len({item.expectation_id for item in assessments}) != len(assessments):
        raise MetricInputError("duplicate_style_expectation")
    return assessments


def _compliance_assessments(values: object) -> tuple[ComplianceAssessment, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise MetricInputError("compliance_assessments_invalid")
    items = tuple(cast(Sequence[object], values))
    if any(not isinstance(item, ComplianceAssessment) for item in items):
        raise MetricInputError("compliance_assessments_invalid")
    assessments = cast(tuple[ComplianceAssessment, ...], items)
    if len({item.obligation_id for item in assessments}) != len(assessments):
        raise MetricInputError("duplicate_compliance_obligation")
    return assessments


def _response_outcome(value: object) -> str:
    if not isinstance(value, str) or value not in {"answer", "refusal", "error"}:
        raise MetricInputError("response_outcome_invalid")
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field}_invalid")
    resolved = value.strip()
    if not resolved or len(resolved) > 255:
        raise ValueError(f"{field}_invalid")
    return resolved


def _optional_identifier(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, field)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}_invalid")
    return value.strip()


def _identifiers(
    values: object,
    field: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise MetricInputError(f"{field}_invalid")
    try:
        resolved = tuple(_identifier(item, field) for item in cast(Sequence[object], values))
    except ValueError:
        raise MetricInputError(f"{field}_invalid") from None
    if not allow_empty and not resolved:
        raise MetricInputError(f"{field}_invalid")
    if len(set(resolved)) != len(resolved):
        raise MetricInputError(f"{field}_duplicate")
    return resolved


__all__ = [
    "ANSWER_COMPLETENESS_SCORER_VERSION",
    "ANSWER_COMPLIANCE_METRIC_ID",
    "ANSWER_COMPLIANCE_SCORER_VERSION",
    "GUIDED_REFUSAL_APPROPRIATENESS_SCORER_VERSION",
    "REFUSAL_APPROPRIATENESS_SCORER_VERSION",
    "STYLE_CONSISTENCY_SCORER_VERSION",
    "AnswerCompletenessScorer",
    "AnswerComplianceAggregate",
    "AnswerComplianceResult",
    "AnswerComplianceScorer",
    "ComplianceAssessment",
    "GuidedRefusalAppropriatenessScorer",
    "RefusalAppropriatenessScorer",
    "StyleAssessment",
    "StyleConsistencyScorer",
    "aggregate_answer_compliance",
]
