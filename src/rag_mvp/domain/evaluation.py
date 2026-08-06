"""Provider-usage and RAG-evaluation domain contracts."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated

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
