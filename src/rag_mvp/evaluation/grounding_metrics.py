"""Versioned deterministic grounding metrics for RAG evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

FAITHFULNESS_SCORER_VERSION = "faithfulness-factual-unit-support-v1"
CONTEXT_PRECISION_SCORER_VERSION = "context-precision-average-precision-v1"


class MetricInputError(ValueError):
    """A stable, content-free evaluation input error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class MetricName(StrEnum):
    FAITHFULNESS = "faithfulness"
    CONTEXT_PRECISION = "context-precision"
    ANSWER_COMPLETENESS = "answer-completeness"
    STYLE_CONSISTENCY = "style-consistency"
    REFUSAL_APPROPRIATENESS = "refusal-appropriateness"


class EvidenceVerdict(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    COVERED = "covered"
    MISSING = "missing"
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    APPROPRIATE = "appropriate"
    INAPPROPRIATE = "inappropriate"


@dataclass(frozen=True, slots=True)
class MetricEvidence:
    """One auditable evidence item contributing to a case score."""

    reference_id: str
    verdict: EvidenceVerdict
    rationale: str
    evidence_references: tuple[str, ...] = ()
    rank: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_id", _identifier(self.reference_id, "reference_id"))
        object.__setattr__(self, "verdict", _verdict(self.verdict))
        object.__setattr__(self, "rationale", _text(self.rationale, "evidence_rationale"))
        object.__setattr__(
            self,
            "evidence_references",
            _identifiers(
                self.evidence_references,
                "evidence_references",
                allow_empty=True,
            ),
        )
        if self.rank is not None and (type(self.rank) is not int or self.rank < 1):
            raise ValueError("evidence_rank_invalid")


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Auditable score for one case and one versioned metric."""

    case_id: str
    metric: MetricName
    scorer_version: str
    eligible: bool
    score: float | None
    numerator: float | None
    denominator: int | None
    rationale: str
    evidence: tuple[MetricEvidence, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _identifier(self.case_id, "case_id"))
        object.__setattr__(self, "metric", _metric(self.metric))
        object.__setattr__(
            self,
            "scorer_version",
            _identifier(self.scorer_version, "scorer_version"),
        )
        object.__setattr__(self, "rationale", _text(self.rationale, "metric_rationale"))
        if type(self.eligible) is not bool:
            raise ValueError("metric_eligibility_invalid")

        raw_evidence = tuple(self.evidence)
        if any(not isinstance(item, MetricEvidence) for item in raw_evidence):
            raise ValueError("metric_evidence_invalid")
        object.__setattr__(self, "evidence", raw_evidence)

        if not self.eligible:
            if self.score is not None or self.numerator is not None or self.denominator is not None:
                raise ValueError("ineligible_metric_must_not_have_score")
            if raw_evidence:
                raise ValueError("ineligible_metric_must_not_have_evidence")
            return

        score = _unit_score(self.score, "metric_score")
        numerator = _non_negative_number(self.numerator, "metric_numerator")
        denominator = self.denominator
        if type(denominator) is not int or denominator < 1:
            raise ValueError("metric_denominator_invalid")
        if numerator > denominator:
            raise ValueError("metric_numerator_invalid")
        if not math.isclose(score, numerator / denominator, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("metric_formula_mismatch")
        if not raw_evidence:
            raise ValueError("eligible_metric_requires_evidence")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "numerator", numerator)

    @property
    def value(self) -> float | None:
        """Report-friendly alias; the value is never rounded by the scorer."""

        return self.score


@dataclass(frozen=True, slots=True)
class MetricAggregate:
    """Unweighted mean of eligible case scores for one scorer version."""

    metric: MetricName
    scorer_version: str
    score: float | None
    eligible_cases: int
    total_cases: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", _metric(self.metric))
        object.__setattr__(
            self,
            "scorer_version",
            _identifier(self.scorer_version, "scorer_version"),
        )
        if type(self.eligible_cases) is not int or self.eligible_cases < 0:
            raise ValueError("eligible_case_count_invalid")
        if type(self.total_cases) is not int or self.total_cases < self.eligible_cases:
            raise ValueError("total_case_count_invalid")
        if self.eligible_cases == 0:
            if self.score is not None:
                raise ValueError("aggregate_without_denominator_must_not_have_score")
            return
        object.__setattr__(self, "score", _unit_score(self.score, "aggregate_score"))

    @property
    def value(self) -> float | None:
        """Report-friendly alias; the value is never rounded by the aggregator."""

        return self.score


@dataclass(frozen=True, slots=True)
class FactSupportAssessment:
    """Versioned adjudication for one factual unit emitted by the QA pipeline."""

    fact_id: str
    supported: bool
    rationale: str
    evidence_chunk_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact_id", _identifier(self.fact_id, "fact_id"))
        if type(self.supported) is not bool:
            raise ValueError("fact_support_verdict_invalid")
        object.__setattr__(self, "rationale", _text(self.rationale, "fact_rationale"))
        evidence_ids = _identifiers(
            self.evidence_chunk_ids,
            "fact_evidence_chunk_ids",
            allow_empty=True,
        )
        if self.supported and not evidence_ids:
            raise ValueError("supported_fact_requires_evidence")
        object.__setattr__(self, "evidence_chunk_ids", evidence_ids)


class FaithfulnessScorer:
    """Score supported factual units divided by all emitted factual units."""

    version = FAITHFULNESS_SCORER_VERSION

    def score(
        self,
        *,
        case_id: str,
        answerable: bool,
        response_outcome: str,
        factual_units: Sequence[FactSupportAssessment],
    ) -> MetricResult:
        resolved_case_id = _identifier(case_id, "case_id")
        if type(answerable) is not bool:
            raise MetricInputError("answerability_invalid")
        outcome = _response_outcome(response_outcome)
        facts = _fact_assessments(factual_units)

        if not answerable:
            return _ineligible(
                resolved_case_id,
                MetricName.FAITHFULNESS,
                self.version,
                "case_not_answerable",
            )
        if outcome != "answer":
            return _ineligible(
                resolved_case_id,
                MetricName.FAITHFULNESS,
                self.version,
                "response_not_answer",
            )
        if not facts:
            return _ineligible(
                resolved_case_id,
                MetricName.FAITHFULNESS,
                self.version,
                "no_factual_units",
            )

        supported = sum(fact.supported for fact in facts)
        denominator = len(facts)
        evidence = tuple(
            MetricEvidence(
                reference_id=fact.fact_id,
                verdict=(
                    EvidenceVerdict.SUPPORTED if fact.supported else EvidenceVerdict.UNSUPPORTED
                ),
                rationale=fact.rationale,
                evidence_references=fact.evidence_chunk_ids,
            )
            for fact in facts
        )
        return MetricResult(
            case_id=resolved_case_id,
            metric=MetricName.FAITHFULNESS,
            scorer_version=self.version,
            eligible=True,
            score=supported / denominator,
            numerator=float(supported),
            denominator=denominator,
            rationale=f"supported_factual_units={supported}; factual_units={denominator}",
            evidence=evidence,
        )


class ContextPrecisionScorer:
    """Compute rank-aware average precision against authoritative chunk IDs.

    The numerator is the sum of precision-at-rank for every authoritative hit.
    The denominator is the complete authoritative evidence set, so missing every
    authoritative chunk is an eligible score of zero rather than a missing value.
    """

    version = CONTEXT_PRECISION_SCORER_VERSION

    def score(
        self,
        *,
        case_id: str,
        answerable: bool,
        retrieved_evidence_ids: Sequence[str],
        authoritative_evidence_ids: Sequence[str],
    ) -> MetricResult:
        resolved_case_id = _identifier(case_id, "case_id")
        if type(answerable) is not bool:
            raise MetricInputError("answerability_invalid")
        retrieved = _identifiers(
            retrieved_evidence_ids,
            "retrieved_evidence_ids",
            allow_empty=True,
        )
        authoritative = _identifiers(
            authoritative_evidence_ids,
            "authoritative_evidence_ids",
            allow_empty=True,
        )

        if not answerable:
            return _ineligible(
                resolved_case_id,
                MetricName.CONTEXT_PRECISION,
                self.version,
                "case_not_answerable",
            )
        if not authoritative:
            return _ineligible(
                resolved_case_id,
                MetricName.CONTEXT_PRECISION,
                self.version,
                "no_authoritative_evidence",
            )

        authoritative_set = set(authoritative)
        relevant_seen = 0
        contributions: list[float] = []
        evidence: list[MetricEvidence] = []
        for rank, reference_id in enumerate(retrieved, start=1):
            relevant = reference_id in authoritative_set
            if relevant:
                relevant_seen += 1
                contributions.append(relevant_seen / rank)
            evidence.append(
                MetricEvidence(
                    reference_id=reference_id,
                    verdict=(EvidenceVerdict.RELEVANT if relevant else EvidenceVerdict.IRRELEVANT),
                    rationale=(
                        "authoritative_evidence_retrieved"
                        if relevant
                        else "non_authoritative_context_retrieved"
                    ),
                    rank=rank,
                )
            )

        retrieved_set = set(retrieved)
        evidence.extend(
            MetricEvidence(
                reference_id=reference_id,
                verdict=EvidenceVerdict.MISSING,
                rationale="authoritative_evidence_not_retrieved",
            )
            for reference_id in authoritative
            if reference_id not in retrieved_set
        )
        numerator = math.fsum(contributions)
        denominator = len(authoritative)
        return MetricResult(
            case_id=resolved_case_id,
            metric=MetricName.CONTEXT_PRECISION,
            scorer_version=self.version,
            eligible=True,
            score=numerator / denominator,
            numerator=numerator,
            denominator=denominator,
            rationale=(
                f"average_precision_contribution={numerator!r}; "
                f"authoritative_evidence={denominator}"
            ),
            evidence=tuple(evidence),
        )


def aggregate_metric(
    results: Sequence[MetricResult],
    *,
    metric: MetricName,
    scorer_version: str,
) -> MetricAggregate:
    """Aggregate only eligible cases without rounding or denominator substitution."""

    resolved_metric = _metric(metric)
    resolved_version = _identifier(scorer_version, "scorer_version")
    raw_results: object = results
    if isinstance(raw_results, (str, bytes, bytearray)) or not isinstance(raw_results, Sequence):
        raise MetricInputError("metric_results_invalid")
    values = tuple(cast(Sequence[object], raw_results))
    if any(not isinstance(result, MetricResult) for result in values):
        raise MetricInputError("metric_results_invalid")
    typed_values = cast(tuple[MetricResult, ...], values)
    if len({result.case_id for result in typed_values}) != len(typed_values):
        raise MetricInputError("duplicate_metric_case")
    if any(
        result.metric is not resolved_metric or result.scorer_version != resolved_version
        for result in typed_values
    ):
        raise MetricInputError("metric_result_identity_mismatch")

    eligible = tuple(result for result in typed_values if result.eligible)
    score = (
        math.fsum(cast(float, result.score) for result in eligible) / len(eligible)
        if eligible
        else None
    )
    return MetricAggregate(
        metric=resolved_metric,
        scorer_version=resolved_version,
        score=score,
        eligible_cases=len(eligible),
        total_cases=len(typed_values),
    )


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


def _fact_assessments(values: object) -> tuple[FactSupportAssessment, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise MetricInputError("factual_units_invalid")
    items = tuple(cast(Sequence[object], values))
    if any(not isinstance(item, FactSupportAssessment) for item in items):
        raise MetricInputError("factual_units_invalid")
    assessments = cast(tuple[FactSupportAssessment, ...], items)
    if len({item.fact_id for item in assessments}) != len(assessments):
        raise MetricInputError("duplicate_factual_unit")
    return assessments


def _response_outcome(value: object) -> str:
    if not isinstance(value, str) or value not in {"answer", "refusal", "error"}:
        raise MetricInputError("response_outcome_invalid")
    return value


def _metric(value: object) -> MetricName:
    if isinstance(value, MetricName):
        return value
    if not isinstance(value, str):
        raise ValueError("metric_name_invalid")
    try:
        return MetricName(value)
    except (TypeError, ValueError):
        raise ValueError("metric_name_invalid") from None


def _verdict(value: object) -> EvidenceVerdict:
    if isinstance(value, EvidenceVerdict):
        return value
    if not isinstance(value, str):
        raise ValueError("evidence_verdict_invalid")
    try:
        return EvidenceVerdict(value)
    except (TypeError, ValueError):
        raise ValueError("evidence_verdict_invalid") from None


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field}_invalid")
    resolved = value.strip()
    if not resolved or len(resolved) > 255:
        raise ValueError(f"{field}_invalid")
    return resolved


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
        raise ValueError(f"{field}_invalid")
    resolved = tuple(_identifier(item, field) for item in cast(Sequence[object], values))
    if not allow_empty and not resolved:
        raise ValueError(f"{field}_invalid")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{field}_duplicate")
    return resolved


def _unit_score(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{field}_invalid")
    return float(value)


def _non_negative_number(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{field}_invalid")
    return float(value)
