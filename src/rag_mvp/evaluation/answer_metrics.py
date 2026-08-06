"""Versioned deterministic answer and refusal metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from rag_mvp.evaluation.grounding_metrics import (
    EvidenceVerdict,
    MetricEvidence,
    MetricInputError,
    MetricName,
    MetricResult,
)

ANSWER_COMPLETENESS_SCORER_VERSION = "answer-completeness-expected-facts-v1"
STYLE_CONSISTENCY_SCORER_VERSION = "style-consistency-applicable-checks-v1"
REFUSAL_APPROPRIATENESS_SCORER_VERSION = "refusal-appropriateness-outcome-v1"


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
        actual_reason: str | None = None,
    ) -> MetricResult:
        resolved_case_id = _identifier(case_id, "case_id")
        if type(expected_refusal) is not bool:
            raise MetricInputError("expected_refusal_invalid")
        outcome = _response_outcome(response_outcome)
        resolved_expected_reason = _optional_identifier(expected_reason, "expected_reason")
        resolved_actual_reason = _optional_identifier(actual_reason, "actual_reason")
        if resolved_expected_reason is not None and not expected_refusal:
            raise MetricInputError("reason_without_expected_refusal")
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
        reason_matches = (
            resolved_expected_reason is None or resolved_expected_reason == resolved_actual_reason
        )
        appropriate = expected_refusal == actual_refusal and reason_matches
        rationale = _refusal_rationale(
            expected_refusal=expected_refusal,
            actual_refusal=actual_refusal,
            reason_matches=reason_matches,
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
                            for value in (resolved_expected_reason, resolved_actual_reason)
                            if value is not None
                        )
                    ),
                ),
            ),
        )


def _refusal_rationale(
    *,
    expected_refusal: bool,
    actual_refusal: bool,
    reason_matches: bool,
) -> str:
    if expected_refusal and not actual_refusal:
        return "refusal_expected_but_answer_emitted"
    if not expected_refusal and actual_refusal:
        return "answer_expected_but_refusal_emitted"
    if expected_refusal and not reason_matches:
        return "refusal_reason_mismatch"
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
