"""Provider-usage and RAG-evaluation domain contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from decimal import Decimal
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal, cast

from pydantic import AwareDatetime, Field, model_validator

from rag_mvp.domain._base import (
    DomainModel,
    FiniteFloat,
    Identifier,
    NonEmptyText,
    NonNegativeFiniteFloat,
    SafeScalar,
    utc_now,
)

EVIDENCE_SCHEMA_VERSION_V2: Literal["2.0.0"] = "2.0.0"
ACCEPTANCE_CONTRACT_SCHEMA_VERSION_V2: Literal["rag-acceptance-contract-v2"] = (
    "rag-acceptance-contract-v2"
)
OPERATIONS_SUMMARY_SCHEMA_VERSION_V2: Literal["rag-operations-summary-v2"] = (
    "rag-operations-summary-v2"
)

type UnavailableReason = Annotated[
    str,
    Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_.-]*$"),
]
type Sha256Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
type MediaType = Annotated[
    str,
    Field(pattern=r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$"),
]


class EvidenceComparisonOperator(StrEnum):
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="
    EQUAL = "=="


class MetricObservationStatus(StrEnum):
    OBSERVED = "observed"
    PASSED = "passed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class GateStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class UnavailableValue(DomainModel):
    """An explicit, machine-readable replacement for an unknown numeric value."""

    status: Literal["unavailable"] = "unavailable"
    reason: UnavailableReason


type EvidenceValue = FiniteFloat | UnavailableValue
type EvidenceNumerator = NonNegativeFiniteFloat | UnavailableValue
type EvidenceDenominator = Annotated[int, Field(ge=0)] | UnavailableValue
type EvidenceScorerVersion = Identifier | UnavailableValue


def _compare_evidence(value: float, operator: EvidenceComparisonOperator, threshold: float) -> bool:
    if operator is EvidenceComparisonOperator.GREATER_THAN:
        return value > threshold
    if operator is EvidenceComparisonOperator.GREATER_THAN_OR_EQUAL:
        return value >= threshold
    if operator is EvidenceComparisonOperator.LESS_THAN:
        return value < threshold
    if operator is EvidenceComparisonOperator.LESS_THAN_OR_EQUAL:
        return value <= threshold
    return math.isclose(value, threshold, rel_tol=0, abs_tol=0)


class MetricObservation(DomainModel):
    """Canonical schema-v2 metric evidence with explicit denominator semantics."""

    schema_version: Literal["2.0.0"] = EVIDENCE_SCHEMA_VERSION_V2
    metric_id: Identifier
    unit: Identifier
    value: EvidenceValue
    numerator: EvidenceNumerator
    denominator: EvidenceDenominator
    eligible: bool
    threshold: FiniteFloat | None = None
    operator: EvidenceComparisonOperator | None = None
    scorer_version: EvidenceScorerVersion
    status: MetricObservationStatus
    evidence_references: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_observation(self) -> MetricObservation:
        if (self.threshold is None) != (self.operator is None):
            raise ValueError("metric threshold and operator must be supplied together")
        if len(self.evidence_references) != len(set(self.evidence_references)):
            raise ValueError("metric evidence references must be unique")

        unavailable_fields = tuple(
            isinstance(value, UnavailableValue)
            for value in (self.value, self.numerator, self.denominator, self.scorer_version)
        )
        if not self.eligible:
            if (
                not all(unavailable_fields[:3])
                or self.status is not MetricObservationStatus.UNAVAILABLE
            ):
                raise ValueError("an ineligible metric must expose unavailable numeric evidence")
            return self

        if isinstance(self.denominator, int) and self.denominator == 0:
            raise ValueError("an eligible metric denominator must be non-zero")
        if any(unavailable_fields):
            if self.status is not MetricObservationStatus.UNAVAILABLE:
                raise ValueError("incomplete metric evidence must be unavailable")
            return self
        if self.status is MetricObservationStatus.UNAVAILABLE:
            raise ValueError("complete metric evidence cannot be unavailable")

        if self.unit == "ratio":
            observed_value = cast(float, self.value)
            observed_numerator = cast(float, self.numerator)
            observed_denominator = cast(int, self.denominator)
            if (
                not 0 <= observed_value <= 1
                or observed_numerator > observed_denominator
                or not math.isclose(
                    observed_value,
                    observed_numerator / observed_denominator,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("ratio metric value disagrees with its numerator and denominator")

        if self.status in {MetricObservationStatus.PASSED, MetricObservationStatus.FAILED}:
            if self.threshold is None or self.operator is None:
                raise ValueError("a threshold decision requires a threshold and operator")
            observed_pass = _compare_evidence(
                cast(float, self.value),
                self.operator,
                self.threshold,
            )
            if (self.status is MetricObservationStatus.PASSED) is not observed_pass:
                raise ValueError("metric status disagrees with its unrounded threshold decision")
        return self


class GateResult(DomainModel):
    """Canonical schema-v2 gate decision over independently evaluated observations."""

    schema_version: Literal["2.0.0"] = EVIDENCE_SCHEMA_VERSION_V2
    gate_id: Identifier
    profile_version: Identifier
    status: GateStatus
    valid: bool
    passed: bool
    case_executions_complete: bool
    observations: Annotated[tuple[MetricObservation, ...], Field(min_length=1)]
    failure_reasons: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_gate(self) -> GateResult:
        metric_ids = tuple(observation.metric_id for observation in self.observations)
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("gate metric observations must be unique")
        if len(self.failure_reasons) != len(set(self.failure_reasons)):
            raise ValueError("gate failure reasons must be unique")
        expected_status = (
            GateStatus.UNAVAILABLE
            if not self.valid
            else GateStatus.PASSED
            if self.passed
            else GateStatus.FAILED
        )
        if self.status is not expected_status:
            raise ValueError("gate status disagrees with validity and pass state")
        if self.passed and self.failure_reasons:
            raise ValueError("a passing gate cannot contain failure reasons")
        if not self.passed and not self.failure_reasons:
            raise ValueError("a non-passing gate requires a failure reason")
        return self


class AcceptanceMetricRequirement(DomainModel):
    metric_id: Identifier
    unit: Identifier = "ratio"
    threshold: FiniteFloat
    operator: EvidenceComparisonOperator
    minimum_denominator: Annotated[int, Field(gt=0)] = 1


class AcceptanceContract(DomainModel):
    """Versioned acceptance inputs; runtime observations never redefine this policy."""

    schema_version: Literal["rag-acceptance-contract-v2"] = ACCEPTANCE_CONTRACT_SCHEMA_VERSION_V2
    contract_id: Identifier
    version: Identifier
    gate_profile_version: Identifier
    dataset_schema_version: Identifier
    performance_schema_version: Identifier
    cost_schema_version: Identifier
    metric_requirements: Annotated[
        tuple[AcceptanceMetricRequirement, ...],
        Field(min_length=1),
    ]

    @model_validator(mode="after")
    def require_unique_metrics(self) -> AcceptanceContract:
        metric_ids = tuple(item.metric_id for item in self.metric_requirements)
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("acceptance metric requirements must be unique")
        return self


class OperationsSummary(DomainModel):
    """Canonical operations evidence consumed by deterministic projections."""

    schema_version: Literal["rag-operations-summary-v2"] = OPERATIONS_SUMMARY_SCHEMA_VERSION_V2
    run_id: Identifier
    configuration_id: Identifier
    observations: Annotated[tuple[MetricObservation, ...], Field(min_length=1)]
    source_artifact_ids: tuple[Identifier, ...] = ()
    generated_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def require_unique_observations(self) -> OperationsSummary:
        metric_ids = tuple(item.metric_id for item in self.observations)
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("operations observations must be unique")
        if len(self.source_artifact_ids) != len(set(self.source_artifact_ids)):
            raise ValueError("operations source artifacts must be unique")
        return self


class ArtifactDescriptor(DomainModel):
    """Safe immutable descriptor for a published evidence artifact."""

    schema_version: Identifier
    artifact_id: Identifier
    format: Identifier
    media_type: MediaType
    relative_path: NonEmptyText
    sha256_digest: Sha256Digest
    byte_size: Annotated[int, Field(ge=0)]
    created_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_relative_path(self) -> ArtifactDescriptor:
        path = PurePosixPath(self.relative_path)
        if (
            "\\" in self.relative_path
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("artifact path must be a safe relative POSIX path")
        return self


def adapt_v1_metric_observation(
    metric_id: str,
    aggregate: Mapping[str, object],
    *,
    scorer_version: str | None = None,
) -> MetricObservation:
    """Read legacy aggregate evidence without mutating or upgrading the v1 artifact."""

    value = aggregate.get("value")
    eligible_cases = aggregate.get("eligible_cases")
    threshold = aggregate.get("threshold")
    operator = aggregate.get("operator")
    if value is not None and (isinstance(value, bool) or not isinstance(value, int | float)):
        raise ValueError("legacy metric value is invalid")
    if type(eligible_cases) is not int or eligible_cases < 0:
        raise ValueError("legacy metric denominator is invalid")
    resolved_operator = None if operator is None else EvidenceComparisonOperator(str(operator))
    if threshold is not None and (
        isinstance(threshold, bool) or not isinstance(threshold, int | float)
    ):
        raise ValueError("legacy metric threshold is invalid")
    missing = UnavailableValue(reason="not-recorded-in-v1")
    denominator: EvidenceDenominator = eligible_cases if eligible_cases else missing
    return MetricObservation(
        metric_id=metric_id,
        unit="ratio",
        value=float(value) if value is not None else missing,
        numerator=missing,
        denominator=denominator,
        eligible=bool(eligible_cases),
        threshold=float(threshold) if threshold is not None else None,
        operator=resolved_operator,
        scorer_version=scorer_version or missing,
        status=MetricObservationStatus.UNAVAILABLE,
    )


def adapt_v1_metric_aggregates(
    aggregates: Mapping[str, object],
    *,
    scorer_versions: Mapping[str, str] | None = None,
) -> tuple[MetricObservation, ...]:
    """Adapt a legacy report's aggregate mapping while leaving its source read-only."""

    versions = scorer_versions or {}
    observations: list[MetricObservation] = []
    for metric_id, value in aggregates.items():
        if not isinstance(metric_id, str) or not isinstance(value, Mapping):
            raise ValueError("legacy aggregate mapping is invalid")
        observations.append(
            adapt_v1_metric_observation(
                metric_id,
                cast(Mapping[str, object], value),
                scorer_version=versions.get(metric_id),
            )
        )
    return tuple(observations)


class ModelRole(StrEnum):
    EMBEDDING = "embedding"
    GENERATION = "generation"
    RERANKING = "reranking"
    EVALUATION = "evaluation"


class ModelAttemptStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed-out"


class EvaluationRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALID = "invalid"


class IssueClassification(StrEnum):
    GENUINE = "genuine"
    CONTROLLED = "controlled"


class TokenUsage(DomainModel):
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    total_tokens_reported: Annotated[int, Field(ge=0)] | None = None

    @property
    def known_total(self) -> int | None:
        if self.total_tokens_reported is not None:
            return self.total_tokens_reported
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens

    @model_validator(mode="after")
    def validate_reported_total(self) -> TokenUsage:
        if (
            self.total_tokens_reported is not None
            and self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens_reported < self.input_tokens + self.output_tokens
        ):
            raise ValueError("reported total tokens cannot be smaller than input plus output")
        return self


class ModelPricing(DomainModel):
    pricing_version: Identifier
    provider: Identifier
    model: Identifier
    currency: Identifier
    input_per_million: Annotated[Decimal, Field(ge=0)] | None = None
    output_per_million: Annotated[Decimal, Field(ge=0)] | None = None


class ProviderAttemptEvidence(DomainModel):
    """Content-free, per-call provider ledger entry used by runtime evidence."""

    operation_id: Identifier
    attempt_number: Annotated[int, Field(gt=0)] = 1
    route_id: Identifier | None = None
    role: ModelRole
    provider: Identifier
    model: Identifier
    status: ModelAttemptStatus
    fallback: bool = False
    latency_ms: NonNegativeFiniteFloat | None = None
    safe_error_category: (
        Annotated[
            str,
            Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_.-]*$"),
        ]
        | None
    ) = None
    usage: TokenUsage = Field(default_factory=TokenUsage)


class ModelAttempt(DomainModel):
    attempt_id: Identifier
    operation_id: Identifier
    request_id: str | None = None
    run_id: str | None = None
    role: ModelRole
    provider: Identifier
    model: Identifier
    status: ModelAttemptStatus
    attempt_number: Annotated[int, Field(gt=0)] = 1
    fallback: bool = False
    latency_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    usage: TokenUsage = Field(default_factory=TokenUsage)
    estimated_cost: Decimal | None = None
    currency: str | None = None
    safe_error_category: str | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_cost(self) -> ModelAttempt:
        if self.estimated_cost is not None and self.estimated_cost < 0:
            raise ValueError("estimated cost cannot be negative")
        if self.estimated_cost is not None and not self.currency:
            raise ValueError("currency is required when estimated cost is known")
        return self


class EvaluationRun(DomainModel):
    run_id: Identifier
    status: EvaluationRunStatus = EvaluationRunStatus.QUEUED
    dataset_id: Identifier
    dataset_version: Identifier
    dataset_hash: Identifier
    corpus_version: Identifier
    configuration_id: Identifier
    code_revision: Identifier
    scorer_versions: dict[str, str]
    cache_policy: Identifier
    total_cases: Annotated[int, Field(ge=0)]
    completed_cases: Annotated[int, Field(ge=0)] = 0
    failed_cases: Annotated[int, Field(ge=0)] = 0
    safe_error_code: str | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_progress(self) -> EvaluationRun:
        if self.completed_cases + self.failed_cases > self.total_cases:
            raise ValueError("completed and failed cases cannot exceed total cases")
        if self.status is EvaluationRunStatus.FAILED and not self.safe_error_code:
            raise ValueError("a failed run requires safe_error_code")
        return self


class IssueEvidence(DomainModel):
    issue_id: Identifier
    classification: IssueClassification
    affected_case_ids: tuple[Identifier, ...]
    symptom: NonEmptyText
    metric_references: tuple[Identifier, ...]
    log_references: tuple[Identifier, ...]
    run_references: tuple[Identifier, ...]
    trace_references: tuple[Identifier, ...]
    root_cause: NonEmptyText
    exact_fix: NonEmptyText
    fix_rationale: NonEmptyText
    primary_metric: Identifier
    baseline_value: FiniteFloat
    post_fix_value: FiniteFloat
    relative_improvement_percent: FiniteFloat

    @model_validator(mode="after")
    def require_evidence(self) -> IssueEvidence:
        if not self.affected_case_ids:
            raise ValueError("issue evidence requires affected cases")
        if not all(
            (
                self.metric_references,
                self.log_references,
                self.run_references,
                self.trace_references,
            )
        ):
            raise ValueError("issue evidence requires metric, log, run, and trace references")
        return self


class ReportManifest(DomainModel):
    run_id: Identifier
    schema_version: Identifier
    json_report_path: NonEmptyText
    html_report_path: NonEmptyText
    content_hash: Identifier
    metadata: dict[str, SafeScalar] = Field(default_factory=dict)
    created_at: AwareDatetime = Field(default_factory=utc_now)
