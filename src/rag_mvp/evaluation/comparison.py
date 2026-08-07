"""Immutable state and evidence contracts for controlled comparisons."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self, cast

from pydantic import AwareDatetime, ConfigDict, Field, field_validator, model_validator

from rag_mvp.domain import (
    ArtifactDescriptor,
    EvidenceComparisonOperator,
    GateResult,
    GateStatus,
    MetricObservation,
    MetricObservationStatus,
    ModelAttemptStatus,
    ModelRole,
    ProviderAttemptEvidence,
    UnavailableValue,
)
from rag_mvp.domain._base import DomainModel, Identifier, utc_now
from rag_mvp.domain.retrieval import CacheOutcome, CachePolicy
from rag_mvp.evaluation.experiment import (
    ExperimentAxis,
    ExperimentPlan,
    FinalTieBreak,
    FixedIdentity,
    SelectionDirection,
)
from rag_mvp.evaluation.json_report import canonical_json_value, decode_json_report
from rag_mvp.evaluation.report_builder import case_ids_content_hash
from rag_mvp.evaluation.report_v2 import (
    REPORT_SCHEMA_VERSION_V2,
    CategoryResultV2,
    EvaluationReportV2,
    canonical_report_document_v2,
    parse_report_v2,
)
from rag_mvp.evaluation.runner import EvaluationRunIdentity
from rag_mvp.performance.load_report import LoadAttempt, nearest_rank_percentile

COMPARISON_SUITE_SCHEMA_VERSION: Literal["comparison-suite-v1"] = "comparison-suite-v1"
COMPARISON_CANDIDATE_EVIDENCE_SCHEMA_VERSION: Literal["comparison-candidate-evidence-v1"] = (
    "comparison-candidate-evidence-v1"
)
COMPARISON_SHARED_SETUP_EVIDENCE_SCHEMA_VERSION: Literal["comparison-shared-setup-evidence-v1"] = (
    "comparison-shared-setup-evidence-v1"
)
COMPARISON_SHARED_SETUP_AGGREGATE_UNAVAILABLE_REASON = "setup-ledger-integrity-unavailable"
COMPARISON_LEGACY_COST_UNAVAILABLE_REASON = "legacy-cost-completeness-unavailable"
COMPARISON_PENDING_COST_REASON = "candidate-cost-pending"
COMPARISON_SELECTION_ELIGIBILITY_GATE_ID = "comparison-selection-eligibility"
COMPARISON_SELECTION_ELIGIBILITY_GATE_VERSION = "comparison-selection-eligibility-v2"
_LEGACY_COMPARISON_SELECTION_ELIGIBILITY_GATE_VERSION = "comparison-selection-eligibility-v1"
COMPARISON_SELECTION_MAX_P90_MS = 10_000.0
COMPARISON_RERANKER_MIN_QUALITY_BENEFIT = 0.0
COMPARISON_RERANKER_BENEFIT_PROFILE_ID = "registered-retrieval-strategy-v1-selection-eligibility-v2"
COMPARISON_RESULT_SCHEMA_VERSION: Literal["comparison-result-v1"] = "comparison-result-v1"
COMPARISON_ARTIFACT_MANIFEST_SCHEMA_VERSION: Literal["comparison-artifact-manifest-v1"] = (
    "comparison-artifact-manifest-v1"
)

_SAFE_CODE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_IDENTITY_TEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:/+@=-]{0,4095}$")
_SECRET_VALUE = re.compile(
    r"(?i)(?:\b(?:api[-_ ]?key|authorization|bearer|password|secret)\b\s*[:=]|"
    r"\bsk-[A-Za-z0-9_-]{8,}\b|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
)
_PATH_VALUE = re.compile(r"(?:^[A-Za-z]:[\\/]|^/|\\\\|(?:^|[/\\])\.\.(?:[/\\]|$))")
_DISPLAY_PATH_VALUE = re.compile(
    r"(?:\b[A-Za-z]:[\\/]|\\\\|(?:^|\s)/(?!\s)[^\s]+|(?:^|[/\\])\.\.(?:[/\\]|$))"
)
_TERMINAL_CANDIDATE_STATUSES: frozenset[ComparisonCandidateStatus]
_FINALIZATION_FAILURE_PREFIXES = (
    "aggregation-",
    "artifact-",
    "integrity-",
    "publication-",
    "result-",
)
_POST_EXECUTION_FAILURE_CODES = frozenset({"comparison-interrupted"})

_IDENTITY_KEYS: Mapping[str, frozenset[str]] = {
    "prompt": frozenset(
        {
            "generation",
            "generation-output-schema",
            "grounding",
            "query-rewrite",
            "reranking",
            "reranking-parser",
        }
    ),
    "provider": frozenset({"adapter", "backend", "embedding", "generation", "reranking"}),
    "generation": frozenset(
        {
            "max_tokens_parameter",
            "maximum_output_tokens",
            "maximum_question_characters",
            "provider_retry_limit",
            "provider_timeout_seconds",
            "qa_deadline_seconds",
            "qa_finalization_budget_seconds",
            "qa_generation_budget_seconds",
            "response_format",
            "seed",
            "temperature",
        }
    ),
    "embedding": frozenset(
        {
            "adapter_version",
            "dimension",
            "dimensions",
            "model",
            "normalization",
            "provider",
            "send_dimensions",
        }
    ),
    "chunking": frozenset(
        {
            "chunking_version",
            "extraction_version",
            "max_tokens",
            "normalization_version",
            "ocr_enabled",
            "ocr_languages",
            "overlap_tokens",
            "parent_target_tokens",
            "target_tokens",
            "tokenizer_version",
            "version",
        }
    ),
    "retrieval": frozenset(
        {
            "allow_single_retriever_degradation",
            "context_chunk_limit",
            "context_selection_version",
            "context_tokenizer_version",
            "degradation_policy_version",
            "dense_candidate_limit",
            "dense_weight",
            "fact_evidence_assessor_version",
            "grounding_validator_version",
            "identity_version",
            "lexical_candidate_limit",
            "lexical_weight",
            "minimum_support_score",
            "mode",
            "qa_bm25_budget_seconds",
            "qa_dense_retrieval_budget_seconds",
            "qa_embedding_budget_seconds",
            "qa_evidence_assessment_budget_seconds",
            "qa_fusion_budget_seconds",
            "qa_retrieval_budget_seconds",
            "refusal_policy_version",
            "rerank_candidate_limit",
            "rerank_deadline_seconds",
            "reranking_enabled",
            "retrieval_cache_enabled",
            "retrieval_cache_max_entries",
            "retrieval_cache_ttl_seconds",
            "rrf_k",
            "rrf_tie_policy_version",
            "rrf_version",
            "top_k",
        }
    ),
    "scorer": frozenset(
        {
            "advanced",
            "advanced-quality-gate",
            "answer-compliance",
            "answer-completeness",
            "context-precision",
            "faithfulness",
            "faithfulness-text-matcher",
            "faithfulness-text-normalization",
            "quality-gate",
            "refusal-appropriateness",
            "scoring-pipeline",
            "style-consistency",
        }
    ),
    "seed": frozenset({"case-order", "generation", "runner", "scoring"}),
}

_COMPARISON_REQUIRED_ARTIFACTS: Mapping[str, tuple[str, str, str]] = {
    "comparison-plan-json": ("experiment-plan-v1", "application/json", "comparison-plan.json"),
    "comparison-report-json": (
        COMPARISON_RESULT_SCHEMA_VERSION,
        "application/json",
        "comparison-report.json",
    ),
    "comparison-report-html": (
        "comparison-report-html-v1",
        "text/html",
        "comparison-report.html",
    ),
    "comparison-report-txt": (
        "comparison-report-text-v1",
        "text/plain",
        "comparison-report.txt",
    ),
    "comparison-report-csv": (
        "comparison-report-csv-v1",
        "text/csv",
        "comparison-report.csv",
    ),
}

type NonNegativeInteger = Annotated[int, Field(ge=0)]
type NonNegativeCost = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
type Sha256Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
type ComparisonValue = float | UnavailableValue
type ComparisonNumerator = float | UnavailableValue
type ComparisonDenominator = int | UnavailableValue


class ComparisonStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALID = "invalid"


class ComparisonCandidateStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


_TERMINAL_CANDIDATE_STATUSES = frozenset(
    {
        ComparisonCandidateStatus.COMPLETED,
        ComparisonCandidateStatus.FAILED,
        ComparisonCandidateStatus.INTERRUPTED,
    }
)


class ComparisonEvidenceStatus(StrEnum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class ComparisonRecommendationState(StrEnum):
    RECOMMENDED = "recommended"
    NO_RECOMMENDATION = "no-recommendation"


class ComparisonLogicalAttemptStatus(StrEnum):
    SUCCEEDED = "succeeded"
    ERROR = "error"
    TIMEOUT = "timeout"


class ComparisonSharedSetupStatus(StrEnum):
    REUSED = "reused"
    COMPLETED = "completed"
    FAILED = "failed"


class CompatibilityIssueCode(StrEnum):
    AXIS_VALUE_MISMATCH = "axis-value-mismatch"
    CONTROLLED_IDENTITY_MISMATCH = "controlled-identity-mismatch"
    CONTROLLED_IDENTITY_MISSING = "controlled-identity-missing"
    CONTROLLED_IDENTITY_UNDECLARED = "controlled-identity-undeclared"
    DATASET_IDENTITY_MISMATCH = "dataset-identity-mismatch"
    CORPUS_IDENTITY_MISMATCH = "corpus-identity-mismatch"
    CASE_SET_IDENTITY_MISMATCH = "case-set-identity-mismatch"
    CONFIGURATION_IDENTITY_MISMATCH = "configuration-identity-mismatch"


class ComparisonSharedSetupAttempt(DomainModel):
    """One persisted, unbound provider call made to prepare the shared corpus index."""

    model_config = ConfigDict(hide_input_in_errors=True)

    attempt_reference: Identifier
    setup_id: Identifier
    request_id: Identifier
    index_revision_id: Identifier
    source_run_id: None = None
    evidence: ProviderAttemptEvidence
    latency_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    pricing_version: Identifier
    pricing_hash: Sha256Digest
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
    input_per_million: NonNegativeCost | None = None
    output_per_million: NonNegativeCost | None = None
    pricing_source_reference: str = Field(min_length=1, max_length=2048)
    known_partial_cost: NonNegativeCost
    total_cost: NonNegativeCost | None
    cost_complete: bool
    unknown_reasons: tuple[Identifier, ...]
    recorded_at: AwareDatetime

    @classmethod
    def create(
        cls,
        *,
        attempt_reference: str,
        setup_id: str,
        request_id: str,
        index_revision_id: str,
        source_run_id: str | None,
        evidence: ProviderAttemptEvidence,
        latency_ms: float,
        pricing_version: str,
        pricing_hash: str,
        currency: str,
        input_per_million: Decimal | None,
        output_per_million: Decimal | None,
        pricing_source_reference: str,
        recorded_at: AwareDatetime,
    ) -> ComparisonSharedSetupAttempt:
        if source_run_id is not None:
            raise ValueError("comparison_shared_setup_source_run_must_be_unbound")
        known_partial, total, reasons = _derive_provider_attempt_cost(
            evidence,
            input_per_million,
            output_per_million,
        )
        return cls(
            attempt_reference=attempt_reference,
            setup_id=setup_id,
            request_id=request_id,
            index_revision_id=index_revision_id,
            source_run_id=None,
            evidence=evidence,
            latency_ms=latency_ms,
            pricing_version=pricing_version,
            pricing_hash=pricing_hash,
            currency=currency,
            input_per_million=input_per_million,
            output_per_million=output_per_million,
            pricing_source_reference=pricing_source_reference,
            known_partial_cost=known_partial,
            total_cost=total,
            cost_complete=not reasons,
            unknown_reasons=reasons,
            recorded_at=recorded_at,
        )

    @model_validator(mode="after")
    def validate_setup_attempt(self) -> Self:
        if self.evidence.role is not ModelRole.EMBEDDING:
            raise ValueError("comparison_shared_setup_role_invalid")
        if self.evidence.operation_id != self.index_revision_id:
            raise ValueError("comparison_shared_setup_revision_mismatch")
        if self.evidence.latency_ms is None or self.evidence.latency_ms != self.latency_ms:
            raise ValueError("comparison_shared_setup_latency_mismatch")
        if (self.evidence.status is ModelAttemptStatus.SUCCEEDED) == (
            self.evidence.safe_error_category is not None
        ):
            raise ValueError("comparison_shared_setup_attempt_status_error_mismatch")
        known_partial, expected_total, expected_reasons = _derive_provider_attempt_cost(
            self.evidence,
            self.input_per_million,
            self.output_per_million,
        )
        if (
            self.known_partial_cost != known_partial
            or self.total_cost != expected_total
            or self.cost_complete is bool(expected_reasons)
            or self.unknown_reasons != expected_reasons
        ):
            raise ValueError("comparison_shared_setup_cost_derivation_mismatch")
        return self


class ComparisonSharedSetupEvidence(DomainModel):
    """Actual shared index preparation calls, separate from candidate evidence and deltas."""

    model_config = ConfigDict(hide_input_in_errors=True)

    schema_version: Literal["comparison-shared-setup-evidence-v1"] = (
        COMPARISON_SHARED_SETUP_EVIDENCE_SCHEMA_VERSION
    )
    comparison_id: Identifier
    plan_id: Identifier
    plan_content_hash: Sha256Digest
    setup_id: Identifier
    request_id: Identifier
    corpus_id: Identifier
    corpus_version: Identifier
    corpus_hash: Sha256Digest
    index_revision_id: Identifier
    status: ComparisonSharedSetupStatus
    safe_error_code: Identifier | None = None
    attempts: tuple[ComparisonSharedSetupAttempt, ...]
    pricing_version: Identifier
    pricing_hash: Sha256Digest
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
    provider_calls_complete: bool = True
    provider_call_count: NonNegativeInteger | UnavailableValue
    known_partial_cost: NonNegativeCost
    total_cost: NonNegativeCost | None
    cost_complete: bool
    unknown_reasons: tuple[Identifier, ...]
    recorded_at: AwareDatetime = Field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        comparison_id: str,
        plan: ExperimentPlan,
        status: ComparisonSharedSetupStatus,
        attempts: Sequence[ComparisonSharedSetupAttempt],
        safe_error_code: str | None = None,
        provider_calls_complete: bool = True,
        recorded_at: AwareDatetime | None = None,
    ) -> ComparisonSharedSetupEvidence:
        corpus_hash = plan.fixed_identities.corpus_hash
        digest = corpus_hash.removeprefix("sha256:")
        attempt_tuple = tuple(attempts)
        known_partial = sum(
            (item.known_partial_cost for item in attempt_tuple),
            start=Decimal(0),
        )
        total = (
            sum(
                (cast(Decimal, item.total_cost) for item in attempt_tuple),
                start=Decimal(0),
            )
            if provider_calls_complete
            and all(item.total_cost is not None for item in attempt_tuple)
            else None
        )
        reasons = {reason for item in attempt_tuple for reason in item.unknown_reasons}
        if not provider_calls_complete:
            reasons.add(COMPARISON_SHARED_SETUP_AGGREGATE_UNAVAILABLE_REASON)
        reason_tuple = tuple(sorted(reasons))
        return cls(
            comparison_id=comparison_id,
            plan_id=plan.plan_id,
            plan_content_hash=plan.content_hash,
            setup_id=comparison_shared_setup_id(comparison_id),
            request_id=f"eval_corpus_{digest}",
            corpus_id=plan.fixed_identities.corpus_id,
            corpus_version=plan.fixed_identities.corpus_version,
            corpus_hash=corpus_hash,
            index_revision_id=f"rev_eval_{digest}",
            status=status,
            safe_error_code=safe_error_code,
            attempts=attempt_tuple,
            pricing_version=plan.pricing.pricing_version,
            pricing_hash=plan.pricing.pricing_hash,
            currency=plan.pricing.currency,
            provider_calls_complete=provider_calls_complete,
            provider_call_count=(
                len(attempt_tuple)
                if provider_calls_complete
                else UnavailableValue(reason=COMPARISON_SHARED_SETUP_AGGREGATE_UNAVAILABLE_REASON)
            ),
            known_partial_cost=known_partial,
            total_cost=total,
            cost_complete=provider_calls_complete and not reason_tuple,
            unknown_reasons=reason_tuple,
            recorded_at=recorded_at or utc_now(),
        )

    @model_validator(mode="after")
    def validate_setup_evidence(self) -> Self:
        digest = self.corpus_hash.removeprefix("sha256:")
        if (
            self.setup_id != comparison_shared_setup_id(self.comparison_id)
            or self.request_id != f"eval_corpus_{digest}"
            or self.index_revision_id != f"rev_eval_{digest}"
        ):
            raise ValueError("comparison_shared_setup_identity_mismatch")
        references = tuple(item.attempt_reference for item in self.attempts)
        numbers = tuple(item.evidence.attempt_number for item in self.attempts)
        if (
            len(references) != len(set(references))
            or numbers != tuple(range(1, len(self.attempts) + 1))
            or tuple(item.recorded_at for item in self.attempts)
            != tuple(sorted(item.recorded_at for item in self.attempts))
            or any(item.recorded_at > self.recorded_at for item in self.attempts)
        ):
            raise ValueError("comparison_shared_setup_attempt_history_invalid")
        if any(
            item.setup_id != self.setup_id
            or item.request_id != self.request_id
            or item.index_revision_id != self.index_revision_id
            or item.pricing_version != self.pricing_version
            or item.pricing_hash != self.pricing_hash
            or item.currency != self.currency
            for item in self.attempts
        ):
            raise ValueError("comparison_shared_setup_attempt_binding_mismatch")
        if (
            not self.provider_calls_complete
            and self.status is not ComparisonSharedSetupStatus.FAILED
        ):
            raise ValueError("comparison_shared_setup_aggregate_unavailable_not_failed")
        succeeded = tuple(
            item for item in self.attempts if item.evidence.status is ModelAttemptStatus.SUCCEEDED
        )
        if self.status is ComparisonSharedSetupStatus.REUSED:
            if self.attempts or self.safe_error_code is not None:
                raise ValueError("comparison_shared_setup_reuse_has_provider_attempts")
        elif self.status is ComparisonSharedSetupStatus.COMPLETED:
            if (
                self.safe_error_code is not None
                or len(succeeded) != 1
                or succeeded[0] is not self.attempts[-1]
            ):
                raise ValueError("comparison_shared_setup_completion_invalid")
        elif self.safe_error_code is None:
            raise ValueError("comparison_shared_setup_failure_invalid")
        _validate_safe_code(self.safe_error_code)
        expected_partial = sum(
            (item.known_partial_cost for item in self.attempts),
            start=Decimal(0),
        )
        expected_total = (
            sum(
                (cast(Decimal, item.total_cost) for item in self.attempts),
                start=Decimal(0),
            )
            if self.provider_calls_complete
            and all(item.total_cost is not None for item in self.attempts)
            else None
        )
        reasons = {reason for item in self.attempts for reason in item.unknown_reasons}
        if not self.provider_calls_complete:
            reasons.add(COMPARISON_SHARED_SETUP_AGGREGATE_UNAVAILABLE_REASON)
        expected_reasons = tuple(sorted(reasons))
        expected_count: int | UnavailableValue = (
            len(self.attempts)
            if self.provider_calls_complete
            else UnavailableValue(reason=COMPARISON_SHARED_SETUP_AGGREGATE_UNAVAILABLE_REASON)
        )
        if (
            self.provider_call_count != expected_count
            or self.known_partial_cost != expected_partial
            or self.total_cost != expected_total
            or self.cost_complete is not (self.provider_calls_complete and not expected_reasons)
            or self.unknown_reasons != expected_reasons
        ):
            raise ValueError("comparison_shared_setup_aggregate_mismatch")
        return self


class ComparisonCandidateReference(DomainModel):
    """Immutable binding from a predeclared variant to one normal evaluation run."""

    variant_id: Identifier
    axis_value: str = Field(min_length=1, max_length=4096)
    configuration_id: Identifier
    evaluation_run_id: Identifier


class ComparisonCandidateSnapshot(DomainModel):
    sequence: NonNegativeInteger
    status: ComparisonCandidateStatus
    total_cases: NonNegativeInteger
    completed_cases: NonNegativeInteger = 0
    failed_cases: NonNegativeInteger = 0
    provider_calls: NonNegativeInteger = 0
    incurred_cost: NonNegativeCost | None = None
    known_partial_cost: NonNegativeCost = Decimal(0)
    cost_complete: bool = True
    cost_unknown_reasons: tuple[Identifier, ...] = ()
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")] | None = None
    safe_error_code: Identifier | None = None
    recorded_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def hydrate_legacy_cost_state(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        values = dict(value)
        if {
            "known_partial_cost",
            "cost_complete",
            "cost_unknown_reasons",
        }.issubset(values):
            return values
        incurred = values.get("incurred_cost")
        provider_calls = int(values.get("provider_calls", 0))
        complete = provider_calls == 0 and incurred is None
        values.setdefault("known_partial_cost", incurred if incurred is not None else Decimal(0))
        values.setdefault("cost_complete", complete)
        values.setdefault(
            "cost_unknown_reasons",
            () if complete else (COMPARISON_LEGACY_COST_UNAVAILABLE_REASON,),
        )
        if not complete:
            values["incurred_cost"] = None
        return values

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        if self.completed_cases + self.failed_cases > self.total_cases:
            raise ValueError("comparison_candidate_progress_exceeds_total")
        reasons = tuple(sorted(set(self.cost_unknown_reasons)))
        if self.cost_unknown_reasons != reasons:
            raise ValueError("comparison_candidate_cost_reason_invalid")
        zero_call_exact = self.provider_calls == 0 and self.incurred_cost is None
        if self.cost_complete:
            if self.cost_unknown_reasons or (
                not zero_call_exact
                and (
                    self.incurred_cost is None
                    or self.currency is None
                    or self.known_partial_cost != self.incurred_cost
                )
            ):
                raise ValueError("comparison_candidate_cost_state_mismatch")
            if zero_call_exact and (self.known_partial_cost != 0 or self.currency is not None):
                raise ValueError("comparison_candidate_cost_state_mismatch")
        elif (
            self.incurred_cost is not None
            or not self.cost_unknown_reasons
            or (self.provider_calls > 0 and self.currency is None)
        ):
            raise ValueError("comparison_candidate_cost_state_mismatch")
        if self.status is ComparisonCandidateStatus.COMPLETED and (
            self.completed_cases + self.failed_cases != self.total_cases
        ):
            raise ValueError("comparison_candidate_completed_progress_incomplete")
        failed = self.status in {
            ComparisonCandidateStatus.FAILED,
            ComparisonCandidateStatus.INTERRUPTED,
        }
        if failed != (self.safe_error_code is not None):
            raise ValueError("comparison_candidate_terminal_error_mismatch")
        _validate_safe_code(self.safe_error_code)
        return self


class ComparisonCandidateHistory(DomainModel):
    reference: ComparisonCandidateReference
    snapshots: Annotated[tuple[ComparisonCandidateSnapshot, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_history(self) -> Self:
        expected_sequence = tuple(range(len(self.snapshots)))
        if tuple(item.sequence for item in self.snapshots) != expected_sequence:
            raise ValueError("comparison_candidate_history_sequence_invalid")
        total_cases = self.snapshots[0].total_cases
        previous = self.snapshots[0]
        if previous.status is not ComparisonCandidateStatus.QUEUED:
            raise ValueError("comparison_candidate_history_must_start_queued")
        for current in self.snapshots[1:]:
            if previous.status in _TERMINAL_CANDIDATE_STATUSES:
                raise ValueError("comparison_candidate_terminal_history_extended")
            if not _candidate_transition_allowed(previous.status, current.status):
                raise ValueError("comparison_candidate_transition_invalid")
            if (
                current.total_cases != total_cases
                or current.completed_cases < previous.completed_cases
                or current.failed_cases < previous.failed_cases
                or current.provider_calls < previous.provider_calls
                or current.recorded_at < previous.recorded_at
            ):
                raise ValueError("comparison_candidate_history_not_monotonic")
            if (
                current.known_partial_cost < previous.known_partial_cost
                or (not previous.cost_complete and current.cost_complete)
                or not set(previous.cost_unknown_reasons).issubset(current.cost_unknown_reasons)
                or (
                    previous.currency is not None
                    and current.currency is not None
                    and current.currency != previous.currency
                )
            ):
                raise ValueError("comparison_candidate_cost_history_not_monotonic")
            previous = current
        return self

    @property
    def latest(self) -> ComparisonCandidateSnapshot:
        return self.snapshots[-1]


class ComparisonProgressSnapshot(DomainModel):
    sequence: NonNegativeInteger
    status: ComparisonStatus
    total_candidates: Annotated[int, Field(gt=0)]
    completed_candidates: NonNegativeInteger = 0
    failed_candidates: NonNegativeInteger = 0
    active_candidates: NonNegativeInteger = 0
    remaining_candidates: NonNegativeInteger
    completed_cases: NonNegativeInteger = 0
    failed_cases: NonNegativeInteger = 0
    provider_calls: NonNegativeInteger = 0
    incurred_cost: NonNegativeCost | None = None
    known_partial_cost: NonNegativeCost = Decimal(0)
    cost_complete: bool = True
    cost_unknown_reasons: tuple[Identifier, ...] = ()
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")] | None = None
    recorded_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def hydrate_legacy_cost_state(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        values = dict(value)
        if {
            "known_partial_cost",
            "cost_complete",
            "cost_unknown_reasons",
        }.issubset(values):
            return values
        incurred = values.get("incurred_cost")
        provider_calls = int(values.get("provider_calls", 0))
        complete = provider_calls == 0 and incurred is None
        values.setdefault("known_partial_cost", incurred if incurred is not None else Decimal(0))
        values.setdefault("cost_complete", complete)
        values.setdefault(
            "cost_unknown_reasons",
            () if complete else (COMPARISON_LEGACY_COST_UNAVAILABLE_REASON,),
        )
        if not complete:
            values["incurred_cost"] = None
        return values

    @model_validator(mode="after")
    def validate_progress(self) -> Self:
        if (
            self.completed_candidates
            + self.failed_candidates
            + self.active_candidates
            + self.remaining_candidates
            != self.total_candidates
        ):
            raise ValueError("comparison_progress_candidate_counts_invalid")
        reasons = tuple(sorted(set(self.cost_unknown_reasons)))
        if self.cost_unknown_reasons != reasons:
            raise ValueError("comparison_progress_cost_reason_invalid")
        if self.cost_complete:
            if self.cost_unknown_reasons or (
                self.provider_calls > 0
                and (
                    self.incurred_cost is None
                    or self.currency is None
                    or self.known_partial_cost != self.incurred_cost
                )
            ):
                raise ValueError("comparison_progress_cost_state_mismatch")
        elif (
            self.incurred_cost is not None
            or not self.cost_unknown_reasons
            or (self.provider_calls > 0 and self.currency is None)
        ):
            raise ValueError("comparison_progress_cost_state_mismatch")
        return self


class ComparisonSuite(DomainModel):
    """Append-only orchestration state with its full immutable plan and histories."""

    model_config = ConfigDict(hide_input_in_errors=True)

    schema_version: Literal["comparison-suite-v1"] = COMPARISON_SUITE_SCHEMA_VERSION
    comparison_id: Identifier
    plan: ExperimentPlan
    plan_content_hash: Identifier
    status: ComparisonStatus
    candidates: Annotated[tuple[ComparisonCandidateHistory, ...], Field(min_length=2)]
    progress_history: Annotated[tuple[ComparisonProgressSnapshot, ...], Field(min_length=1)]
    safe_error_code: Identifier | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def hydrate_legacy_progress_cost_state(cls, value: object) -> object:
        """Adapt v1 suite rows without mutating their immutable stored payloads."""

        if not isinstance(value, Mapping):
            return value
        values = dict(value)
        raw_progress = values.get("progress_history")
        raw_candidates = values.get("candidates")
        if not isinstance(raw_progress, (tuple, list)) or not isinstance(
            raw_candidates, (tuple, list)
        ):
            return values
        if all(
            isinstance(item, Mapping)
            and {
                "known_partial_cost",
                "cost_complete",
                "cost_unknown_reasons",
            }.issubset(item)
            for item in raw_progress
        ):
            return values
        try:
            candidates = tuple(
                ComparisonCandidateHistory.model_validate(item) for item in raw_candidates
            )
            enriched_progress: list[dict[str, object]] = []
            for raw in raw_progress:
                if not isinstance(raw, Mapping):
                    return values
                recorded_value = raw.get("recorded_at")
                recorded_at = (
                    recorded_value
                    if isinstance(recorded_value, datetime)
                    else datetime.fromisoformat(str(recorded_value).replace("Z", "+00:00"))
                )
                sequence = int(raw.get("sequence", -1))
                status = ComparisonStatus(str(raw.get("status")))
                prefixes = tuple(
                    ComparisonCandidateHistory(
                        reference=history.reference,
                        snapshots=tuple(
                            snapshot
                            for snapshot in history.snapshots
                            if snapshot.recorded_at <= recorded_at
                        ),
                    )
                    for history in candidates
                )
                expected = _progress_from_candidates(
                    prefixes,
                    sequence=sequence,
                    status=status,
                    recorded_at=recorded_at,
                )
                item = dict(raw)
                item.update(
                    {
                        "incurred_cost": expected.incurred_cost,
                        "known_partial_cost": expected.known_partial_cost,
                        "cost_complete": expected.cost_complete,
                        "cost_unknown_reasons": expected.cost_unknown_reasons,
                        "currency": expected.currency,
                    }
                )
                enriched_progress.append(item)
            values["progress_history"] = tuple(enriched_progress)
        except (TypeError, ValueError):
            return values
        return values

    @model_validator(mode="after")
    def validate_suite(self) -> Self:
        validate_comparison_plan_safe_values(self.plan)
        self.plan.verify_hash()
        if self.plan_content_hash != self.plan.content_hash:
            raise ValueError("comparison_plan_content_hash_mismatch")
        variants = {item.variant_id: item for item in self.plan.variants}
        if tuple(item.reference.variant_id for item in self.candidates) != tuple(variants):
            raise ValueError("comparison_candidate_order_mismatch")
        run_ids = tuple(item.reference.evaluation_run_id for item in self.candidates)
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("comparison_candidate_evaluation_run_duplicate")
        for candidate in self.candidates:
            variant = variants[candidate.reference.variant_id]
            if (
                candidate.reference.axis_value != variant.axis_value
                or candidate.reference.configuration_id != variant.configuration_id
            ):
                raise ValueError("comparison_candidate_reference_plan_mismatch")
        if tuple(item.sequence for item in self.progress_history) != tuple(
            range(len(self.progress_history))
        ):
            raise ValueError("comparison_progress_history_sequence_invalid")
        progress_times = {item.recorded_at for item in self.progress_history}
        noninitial_snapshot_times = tuple(
            snapshot.recorded_at
            for history in self.candidates
            for snapshot in history.snapshots[1:]
        )
        ordinary_progress_times = {
            item.recorded_at
            for item in self.progress_history[1:]
            if item.status not in {ComparisonStatus.FAILED, ComparisonStatus.INVALID}
        }
        if (
            len(noninitial_snapshot_times) != len(set(noninitial_snapshot_times))
            or any(item > self.updated_at for item in noninitial_snapshot_times)
            or set(noninitial_snapshot_times) - progress_times
            or set(noninitial_snapshot_times) != ordinary_progress_times
        ):
            raise ValueError("comparison_candidate_progress_timestamp_binding_invalid")
        for previous, current in zip(
            self.progress_history,
            self.progress_history[1:],
            strict=False,
        ):
            if (
                current.recorded_at <= previous.recorded_at
                or current.completed_candidates < previous.completed_candidates
                or current.failed_candidates < previous.failed_candidates
                or current.completed_cases < previous.completed_cases
                or current.failed_cases < previous.failed_cases
                or current.provider_calls < previous.provider_calls
            ):
                raise ValueError("comparison_progress_history_not_monotonic")
            if not _suite_transition_allowed(previous.status, current.status):
                raise ValueError("comparison_progress_status_transition_invalid")
        for progress in self.progress_history:
            prefixes = tuple(
                ComparisonCandidateHistory(
                    reference=history.reference,
                    snapshots=tuple(
                        item
                        for item in history.snapshots
                        if item.recorded_at <= progress.recorded_at
                    ),
                )
                for history in self.candidates
            )
            if any(not item.snapshots for item in prefixes):
                raise ValueError("comparison_progress_precedes_candidate_history")
            expected = _progress_from_candidates(
                prefixes,
                sequence=progress.sequence,
                status=progress.status,
                recorded_at=progress.recorded_at,
            )
            if progress != expected:
                raise ValueError("comparison_progress_history_candidate_mismatch")
            derived_status = _derived_suite_status(prefixes)
            if (
                progress.status not in {ComparisonStatus.FAILED, ComparisonStatus.INVALID}
                and progress.status is not derived_status
            ):
                raise ValueError("comparison_progress_status_not_derived")
        latest = self.progress_history[-1]
        if latest.status is not self.status or latest.total_candidates != len(self.candidates):
            raise ValueError("comparison_status_progress_mismatch")
        if self.updated_at != latest.recorded_at or self.updated_at < self.created_at:
            raise ValueError("comparison_timestamp_progress_mismatch")
        failed = self.status in {ComparisonStatus.FAILED, ComparisonStatus.INVALID}
        if failed != (self.safe_error_code is not None):
            raise ValueError("comparison_terminal_error_mismatch")
        _validate_safe_code(self.safe_error_code)
        return self

    @property
    def latest_progress(self) -> ComparisonProgressSnapshot:
        return self.progress_history[-1]

    @property
    def partial_failure(self) -> bool:
        return any(
            item.latest.failed_cases > 0
            or item.latest.status
            in {ComparisonCandidateStatus.FAILED, ComparisonCandidateStatus.INTERRUPTED}
            for item in self.candidates
        )

    def transition_candidate(
        self,
        variant_id: str,
        *,
        status: ComparisonCandidateStatus,
        completed_cases: int,
        failed_cases: int,
        provider_calls: int,
        incurred_cost: Decimal | None = None,
        known_partial_cost: Decimal | None = None,
        cost_complete: bool | None = None,
        cost_unknown_reasons: tuple[str, ...] | None = None,
        currency: str | None = None,
        safe_error_code: str | None = None,
        recorded_at: AwareDatetime | None = None,
    ) -> ComparisonSuite:
        timestamp = recorded_at or utc_now()
        updated: list[ComparisonCandidateHistory] = []
        found = False
        for history in self.candidates:
            if history.reference.variant_id != variant_id:
                updated.append(history)
                continue
            found = True
            snapshot_values: dict[str, object] = {
                "sequence": len(history.snapshots),
                "status": status,
                "total_cases": history.latest.total_cases,
                "completed_cases": completed_cases,
                "failed_cases": failed_cases,
                "provider_calls": provider_calls,
                "incurred_cost": incurred_cost,
                "known_partial_cost": (
                    known_partial_cost
                    if known_partial_cost is not None
                    else incurred_cost
                    if incurred_cost is not None
                    else Decimal(0)
                ),
                "cost_complete": (
                    cost_complete
                    if cost_complete is not None
                    else incurred_cost is not None or provider_calls == 0
                ),
                "cost_unknown_reasons": (
                    cost_unknown_reasons
                    if cost_unknown_reasons is not None
                    else ()
                    if incurred_cost is not None or provider_calls == 0
                    else (COMPARISON_LEGACY_COST_UNAVAILABLE_REASON,)
                ),
                "currency": currency,
                "safe_error_code": safe_error_code,
                "recorded_at": timestamp,
            }
            snapshot = ComparisonCandidateSnapshot.model_validate(snapshot_values)
            updated.append(
                ComparisonCandidateHistory(
                    reference=history.reference,
                    snapshots=(*history.snapshots, snapshot),
                )
            )
        if not found:
            raise ValueError("comparison_candidate_not_found")
        suite_status = _derived_suite_status(tuple(updated))
        progress = _progress_from_candidates(
            tuple(updated),
            sequence=len(self.progress_history),
            status=suite_status,
            recorded_at=timestamp,
        )
        return type(self).model_validate(
            self.model_copy(
                update={
                    "status": suite_status,
                    "candidates": tuple(updated),
                    "progress_history": (*self.progress_history, progress),
                    "updated_at": timestamp,
                }
            )
        )

    def fail(self, safe_error_code: str, *, recorded_at: AwareDatetime | None = None) -> Self:
        if self.status in {ComparisonStatus.FAILED, ComparisonStatus.INVALID}:
            raise ValueError("comparison_terminal_history_extended")
        _validate_safe_code(safe_error_code)
        if (
            self.status is ComparisonStatus.COMPLETED
            and safe_error_code not in _POST_EXECUTION_FAILURE_CODES
            and not safe_error_code.startswith(_FINALIZATION_FAILURE_PREFIXES)
        ):
            raise ValueError("comparison_completed_failure_not_finalization")
        timestamp = recorded_at or utc_now()
        if timestamp <= self.updated_at:
            raise ValueError("comparison_progress_timestamp_not_monotonic")
        progress = _progress_from_candidates(
            self.candidates,
            sequence=len(self.progress_history),
            status=ComparisonStatus.FAILED,
            recorded_at=timestamp,
        )
        return type(self).model_validate(
            self.model_copy(
                update={
                    "status": ComparisonStatus.FAILED,
                    "safe_error_code": safe_error_code,
                    "progress_history": (*self.progress_history, progress),
                    "updated_at": timestamp,
                }
            )
        )


class ComparisonIdentityProjection(DomainModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    variant_id: Identifier
    configuration_id: Identifier
    dataset_id: Identifier
    dataset_version: Identifier
    dataset_hash: Identifier
    corpus_id: Identifier
    corpus_version: Identifier
    corpus_hash: Identifier
    case_set_hash: Identifier
    identities: tuple[FixedIdentity, ...]

    @field_validator("identities")
    @classmethod
    def canonicalize_identities(
        cls,
        values: tuple[FixedIdentity, ...],
    ) -> tuple[FixedIdentity, ...]:
        names = tuple(item.name for item in values)
        if len(names) != len(set(names)):
            raise ValueError("comparison_identity_projection_duplicate")
        for item in values:
            _safe_identity_text(item.value)
        return tuple(sorted(values, key=lambda item: item.name))

    @model_validator(mode="after")
    def validate_safe_projection_fields(self) -> Self:
        for value in (
            self.variant_id,
            self.configuration_id,
            self.dataset_id,
            self.dataset_version,
            self.dataset_hash,
            self.corpus_id,
            self.corpus_version,
            self.corpus_hash,
            self.case_set_hash,
        ):
            _safe_identity_text(value)
        return self

    def identity_map(self) -> Mapping[str, str]:
        return {item.name: item.value for item in self.identities}


class ComparisonCompatibilityIssue(DomainModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    variant_id: Identifier
    code: CompatibilityIssueCode
    identity_name: Identifier
    expected: str | UnavailableValue
    actual: str | UnavailableValue

    @model_validator(mode="after")
    def validate_safe_issue(self) -> Self:
        _safe_identity_text(self.variant_id)
        _safe_identity_text(self.identity_name)
        for value in (self.expected, self.actual):
            _safe_identity_text(value.reason if isinstance(value, UnavailableValue) else value)
        return self


class ComparisonCompatibility(DomainModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    compatible: bool
    axis: ExperimentAxis
    controlled_dimensions: tuple[FixedIdentity, ...]
    issues: tuple[ComparisonCompatibilityIssue, ...] = ()

    @model_validator(mode="after")
    def validate_compatibility(self) -> Self:
        if self.compatible == bool(self.issues):
            raise ValueError("comparison_compatibility_issue_state_mismatch")
        for identity in self.controlled_dimensions:
            _safe_identity_text(identity.name)
            _safe_identity_text(identity.value)
        return self


class RerankerCaseEvidence(DomainModel):
    candidate_variant_id: Identifier
    case_id: Identifier
    logical_attempt_id: Identifier
    rerank_sensitive: bool
    pre_rerank_chunk_ids: tuple[Identifier, ...]
    post_rerank_chunk_ids: tuple[Identifier, ...]
    pre_rerank_context_chunk_ids: tuple[Identifier, ...]
    selected_context_chunk_ids: tuple[Identifier, ...]
    reranking_attempt_references: tuple[Identifier, ...]
    successful_reranking_attempt_references: tuple[Identifier, ...]

    @model_validator(mode="after")
    def validate_reranker_evidence(self) -> Self:
        for values in (
            self.pre_rerank_chunk_ids,
            self.post_rerank_chunk_ids,
            self.pre_rerank_context_chunk_ids,
            self.selected_context_chunk_ids,
            self.reranking_attempt_references,
            self.successful_reranking_attempt_references,
        ):
            if len(values) != len(set(values)):
                raise ValueError("reranker_evidence_identifier_duplicate")
        if set(self.pre_rerank_chunk_ids) != set(self.post_rerank_chunk_ids):
            raise ValueError("reranker_candidate_set_mismatch")
        if not set(self.pre_rerank_context_chunk_ids).issubset(self.pre_rerank_chunk_ids):
            raise ValueError("reranker_pre_context_not_candidate_subset")
        if not set(self.selected_context_chunk_ids).issubset(self.post_rerank_chunk_ids):
            raise ValueError("reranker_selected_context_not_candidate_subset")
        if not set(self.successful_reranking_attempt_references).issubset(
            self.reranking_attempt_references
        ):
            raise ValueError("reranker_successful_attempt_not_invoked")
        return self

    @property
    def invoked(self) -> bool:
        return bool(self.reranking_attempt_references)

    @property
    def discriminating(self) -> bool:
        return (
            self.rerank_sensitive
            and bool(self.successful_reranking_attempt_references)
            and (
                self.pre_rerank_chunk_ids != self.post_rerank_chunk_ids
                or (
                    bool(self.pre_rerank_context_chunk_ids)
                    and self.pre_rerank_context_chunk_ids != self.selected_context_chunk_ids
                )
            )
        )


class ComparisonProviderAttempt(DomainModel):
    """Privacy-safe provider evidence with exact role-specific pricing derivation."""

    attempt_reference: Identifier
    logical_attempt_id: Identifier
    evaluation_run_id: Identifier
    evidence: ProviderAttemptEvidence
    latency_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    pricing_version: Identifier
    pricing_hash: Sha256Digest
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
    input_per_million: NonNegativeCost | None = None
    output_per_million: NonNegativeCost | None = None
    pricing_source_reference: str = Field(min_length=1, max_length=2048)
    known_partial_cost: NonNegativeCost
    total_cost: NonNegativeCost | None
    complete: bool
    unknown_reasons: tuple[Identifier, ...]

    @classmethod
    def create(
        cls,
        *,
        attempt_reference: str,
        logical_attempt_id: str,
        evaluation_run_id: str,
        evidence: ProviderAttemptEvidence,
        latency_ms: float,
        pricing_version: str,
        pricing_hash: str,
        currency: str,
        input_per_million: Decimal | None,
        output_per_million: Decimal | None,
        pricing_source_reference: str,
    ) -> ComparisonProviderAttempt:
        known_partial, total, reasons = _derive_provider_attempt_cost(
            evidence,
            input_per_million,
            output_per_million,
        )
        return cls(
            attempt_reference=attempt_reference,
            logical_attempt_id=logical_attempt_id,
            evaluation_run_id=evaluation_run_id,
            evidence=evidence,
            latency_ms=latency_ms,
            pricing_version=pricing_version,
            pricing_hash=pricing_hash,
            currency=currency,
            input_per_million=input_per_million,
            output_per_million=output_per_million,
            pricing_source_reference=pricing_source_reference,
            known_partial_cost=known_partial,
            total_cost=total,
            complete=not reasons,
            unknown_reasons=reasons,
        )

    @model_validator(mode="after")
    def validate_priced_attempt(self) -> Self:
        if self.evidence.latency_ms is None or self.evidence.latency_ms != self.latency_ms:
            raise ValueError("comparison_provider_latency_mismatch")
        known_partial, expected_total, expected_reasons = _derive_provider_attempt_cost(
            self.evidence,
            self.input_per_million,
            self.output_per_million,
        )
        expected_complete = not expected_reasons
        if (
            self.known_partial_cost != known_partial
            or self.total_cost != expected_total
            or self.complete is not expected_complete
            or self.unknown_reasons != expected_reasons
        ):
            raise ValueError("comparison_provider_cost_derivation_mismatch")
        return self


class ComparisonProviderRoleCount(DomainModel):
    role: ModelRole
    attempt_count: NonNegativeInteger
    successful_count: NonNegativeInteger
    failed_count: NonNegativeInteger

    @model_validator(mode="after")
    def validate_count(self) -> Self:
        if self.successful_count + self.failed_count != self.attempt_count:
            raise ValueError("comparison_provider_role_count_mismatch")
        return self


class ComparisonLogicalAttempt(DomainModel):
    """One real case attempt; this is not HTTP or service-level evidence."""

    scope: Literal["comparison-logical-case-attempts"] = "comparison-logical-case-attempts"
    attempt_id: Identifier
    case_id: Identifier
    repeat_index: NonNegativeInteger
    order_index: NonNegativeInteger
    status: ComparisonLogicalAttemptStatus
    latency_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    terminal_kind: Literal["answer", "refusal", "error"]
    cache_policy: CachePolicy
    cache_outcome: CacheOutcome
    index_revision_id: Identifier | UnavailableValue
    retrieved_chunk_ids: Annotated[tuple[Identifier, ...], Field(max_length=512)]
    context_chunk_ids: Annotated[tuple[Identifier, ...], Field(max_length=128)]
    retrieval_evidence_digest: Sha256Digest | UnavailableValue
    safe_error_code: Identifier | None = None
    provider_attempt_references: tuple[Identifier, ...]
    provider_failed_attempt_count: NonNegativeInteger = 0
    input_tokens: NonNegativeInteger | None = None
    output_tokens: NonNegativeInteger | None = None
    estimated_cost: NonNegativeCost | None = None
    known_partial_cost: NonNegativeCost = Decimal(0)
    cost_complete: bool = True
    cost_unknown_reasons: tuple[Identifier, ...] = ()
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")] | None = None
    degradation_codes: tuple[Identifier, ...] = ()
    degradation_evidence_complete: bool = True
    completed_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def hydrate_legacy_cost_state(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        values = dict(value)
        if {
            "known_partial_cost",
            "cost_complete",
            "cost_unknown_reasons",
        }.issubset(values):
            return values
        estimated = values.get("estimated_cost")
        complete = estimated is not None
        values.setdefault("known_partial_cost", estimated if estimated is not None else Decimal(0))
        values.setdefault("cost_complete", complete)
        values.setdefault(
            "cost_unknown_reasons",
            () if complete else (COMPARISON_LEGACY_COST_UNAVAILABLE_REASON,),
        )
        return values

    @model_validator(mode="after")
    def validate_attempt(self) -> Self:
        if len(self.provider_attempt_references) != len(set(self.provider_attempt_references)):
            raise ValueError("comparison_provider_attempt_reference_duplicate")
        if self.provider_failed_attempt_count > len(self.provider_attempt_references):
            raise ValueError("comparison_provider_failure_count_invalid")
        if self.cost_unknown_reasons != tuple(sorted(set(self.cost_unknown_reasons))):
            raise ValueError("comparison_attempt_cost_reason_invalid")
        if self.cost_complete:
            if (
                self.estimated_cost is None
                or self.currency is None
                or self.known_partial_cost != self.estimated_cost
                or self.cost_unknown_reasons
            ):
                raise ValueError("comparison_attempt_cost_state_mismatch")
        elif (
            self.estimated_cost is not None
            or self.currency is None
            or not self.cost_unknown_reasons
        ):
            raise ValueError("comparison_attempt_cost_state_mismatch")
        if len(self.retrieved_chunk_ids) != len(set(self.retrieved_chunk_ids)) or len(
            self.context_chunk_ids
        ) != len(set(self.context_chunk_ids)):
            raise ValueError("comparison_attempt_chunk_id_duplicate")
        if not set(self.context_chunk_ids).issubset(self.retrieved_chunk_ids):
            raise ValueError("comparison_attempt_context_not_retrieved")
        if self.cache_policy is CachePolicy.BYPASS:
            if self.cache_outcome is not CacheOutcome.BYPASS:
                raise ValueError("comparison_attempt_cache_policy_outcome_mismatch")
        elif self.cache_outcome is CacheOutcome.BYPASS:
            raise ValueError("comparison_attempt_cache_policy_outcome_mismatch")
        succeeded = self.status is ComparisonLogicalAttemptStatus.SUCCEEDED
        if succeeded:
            if self.terminal_kind not in {"answer", "refusal"} or self.safe_error_code is not None:
                raise ValueError("comparison_successful_attempt_terminal_invalid")
            if self.terminal_kind == "answer" and not self.provider_attempt_references:
                raise ValueError("comparison_answer_generation_evidence_missing")
        elif self.terminal_kind != "error" or self.safe_error_code is None:
            raise ValueError("comparison_failed_attempt_terminal_invalid")
        if (
            succeeded
            and (
                self.cache_outcome in {CacheOutcome.HIT, CacheOutcome.MISS}
                or self.terminal_kind == "answer"
            )
            and (
                isinstance(self.index_revision_id, UnavailableValue)
                or isinstance(self.retrieval_evidence_digest, UnavailableValue)
            )
        ):
            raise ValueError("comparison_retrieval_equivalence_evidence_missing")
        _validate_safe_code(self.safe_error_code)
        return self

    @property
    def provider_attempt_count(self) -> int:
        return len(self.provider_attempt_references)

    @property
    def degraded(self) -> bool:
        return bool(self.degradation_codes)


class ComparisonCandidateEvidence(DomainModel):
    """Canonical candidate evidence from normal evaluation work, never fabricated SLA data."""

    schema_version: Literal["comparison-candidate-evidence-v1"] = (
        COMPARISON_CANDIDATE_EVIDENCE_SCHEMA_VERSION
    )
    comparison_id: Identifier
    plan_id: Identifier
    plan_content_hash: Identifier
    variant_id: Identifier
    evaluation_run_id: Identifier
    configuration_id: Identifier
    identity_projection: ComparisonIdentityProjection
    source_artifact_id: Identifier
    source_report_descriptor: ArtifactDescriptor | None = None
    attempt_scope: Literal["comparison-logical-case-attempts"] = "comparison-logical-case-attempts"
    cache_policy: CachePolicy
    case_ids: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    logical_attempts: Annotated[tuple[ComparisonLogicalAttempt, ...], Field(min_length=1)]
    provider_attempts: tuple[ComparisonProviderAttempt, ...]
    provider_role_counts: tuple[ComparisonProviderRoleCount, ...]
    pricing_version: Identifier
    pricing_hash: Sha256Digest
    pricing_currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
    metrics: Annotated[tuple[MetricObservation, ...], Field(min_length=1)]
    gates: Annotated[tuple[GateResult, ...], Field(min_length=1)]
    category_results: tuple[CategoryResultV2, ...] = ()
    failed_case_count: NonNegativeInteger = 0
    provider_call_count: NonNegativeInteger
    known_partial_cost: NonNegativeCost = Decimal(0)
    total_cost: NonNegativeCost | None
    cost_complete: bool = True
    cost_unknown_reasons: tuple[Identifier, ...] = ()
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")] | None
    reranker_evidence: tuple[RerankerCaseEvidence, ...] = ()
    generated_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def hydrate_legacy_cost_state(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        values = dict(value)
        total = values.get("total_cost")
        if "known_partial_cost" not in values:
            values["known_partial_cost"] = total if total is not None else Decimal(0)
        if "cost_complete" not in values:
            values["cost_complete"] = total is not None
        if "cost_unknown_reasons" not in values:
            values["cost_unknown_reasons"] = (
                () if total is not None else (COMPARISON_LEGACY_COST_UNAVAILABLE_REASON,)
            )
        if values.get("currency") is None and values.get("pricing_currency") is not None:
            values["currency"] = values["pricing_currency"]
        return values

    @model_validator(mode="after")
    def validate_candidate_evidence(self) -> Self:
        metric_ids = tuple(item.metric_id for item in self.metrics)
        gate_ids = tuple(item.gate_id for item in self.gates)
        category_ids = tuple(item.category_id for item in self.category_results)
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("comparison_candidate_metric_duplicate")
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("comparison_candidate_gate_duplicate")
        if len(category_ids) != len(set(category_ids)):
            raise ValueError("comparison_candidate_category_duplicate")
        if self.currency != self.pricing_currency:
            raise ValueError("comparison_candidate_total_cost_currency_mismatch")
        if self.cost_unknown_reasons != tuple(sorted(set(self.cost_unknown_reasons))):
            raise ValueError("comparison_candidate_cost_reason_invalid")
        if any(item.candidate_variant_id != self.variant_id for item in self.reranker_evidence):
            raise ValueError("reranker_candidate_identity_mismatch")
        reranker_logical_ids = tuple(item.logical_attempt_id for item in self.reranker_evidence)
        if len(reranker_logical_ids) != len(set(reranker_logical_ids)):
            raise ValueError("reranker_logical_attempt_evidence_duplicate")
        attempt_ids = tuple(item.attempt_id for item in self.logical_attempts)
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("comparison_candidate_case_id_duplicate")
        order = tuple(item.order_index for item in self.logical_attempts)
        provider_references = tuple(
            reference
            for attempt in self.logical_attempts
            for reference in attempt.provider_attempt_references
        )
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("comparison_logical_attempt_duplicate")
        if order != tuple(range(len(self.logical_attempts))):
            raise ValueError("comparison_logical_attempt_order_invalid")
        if len(provider_references) != len(set(provider_references)):
            raise ValueError("comparison_provider_attempt_reference_reused")
        provider_by_reference = {item.attempt_reference: item for item in self.provider_attempts}
        if len(provider_by_reference) != len(self.provider_attempts) or set(
            provider_references
        ) != set(provider_by_reference):
            raise ValueError("comparison_provider_attempt_ledger_binding_mismatch")
        logical_by_id = {item.attempt_id: item for item in self.logical_attempts}
        for reranker in self.reranker_evidence:
            logical = logical_by_id.get(reranker.logical_attempt_id)
            if logical is None or logical.case_id != reranker.case_id:
                raise ValueError("reranker_logical_attempt_binding_mismatch")
            try:
                referenced = tuple(
                    provider_by_reference[item] for item in reranker.reranking_attempt_references
                )
            except KeyError:
                raise ValueError("reranker_provider_attempt_reference_missing") from None
            if any(
                item.logical_attempt_id != logical.attempt_id
                or item.evidence.role is not ModelRole.RERANKING
                for item in referenced
            ):
                raise ValueError("reranker_provider_attempt_binding_invalid")
            actual_successful = tuple(
                item.attempt_reference
                for item in referenced
                if item.evidence.status is ModelAttemptStatus.SUCCEEDED
                and item.evidence.safe_error_category is None
            )
            if actual_successful != reranker.successful_reranking_attempt_references:
                raise ValueError("reranker_successful_attempt_evidence_mismatch")
        for logical_attempt in self.logical_attempts:
            bound = tuple(
                provider_by_reference[reference]
                for reference in logical_attempt.provider_attempt_references
            )
            if any(item.logical_attempt_id != logical_attempt.attempt_id for item in bound):
                raise ValueError("comparison_provider_logical_attempt_binding_mismatch")
            if any(item.evaluation_run_id != self.evaluation_run_id for item in bound):
                raise ValueError("comparison_provider_evaluation_run_mismatch")
            if logical_attempt.terminal_kind == "answer" and not any(
                item.evidence.role is ModelRole.GENERATION
                and item.evidence.operation_id == "qa-generation"
                and item.evidence.status is ModelAttemptStatus.SUCCEEDED
                and item.evidence.safe_error_category is None
                for item in bound
            ):
                raise ValueError("comparison_answer_generation_evidence_incomplete")
            failed_providers = sum(
                item.evidence.status is not ModelAttemptStatus.SUCCEEDED for item in bound
            )
            input_applicable = tuple(item for item in bound if item.input_per_million is not None)
            output_applicable = tuple(item for item in bound if item.output_per_million is not None)
            input_values = tuple(item.evidence.usage.input_tokens for item in input_applicable)
            output_values = tuple(item.evidence.usage.output_tokens for item in output_applicable)
            expected_input = (
                sum(cast(int, item) for item in input_values)
                if all(item is not None for item in input_values)
                else None
            )
            expected_output = (
                sum(cast(int, item) for item in output_values)
                if all(item is not None for item in output_values)
                else None
            )
            known_cost = all(item.total_cost is not None for item in bound)
            expected_partial_cost = sum(
                (item.known_partial_cost for item in bound),
                start=Decimal(0),
            )
            expected_attempt_cost = (
                sum((cast(Decimal, item.total_cost) for item in bound), start=Decimal(0))
                if known_cost
                else None
            )
            if (
                logical_attempt.provider_failed_attempt_count != failed_providers
                or logical_attempt.input_tokens != expected_input
                or logical_attempt.output_tokens != expected_output
                or logical_attempt.estimated_cost != expected_attempt_cost
                or logical_attempt.known_partial_cost != expected_partial_cost
                or logical_attempt.cost_complete is not known_cost
                or logical_attempt.cost_unknown_reasons
                != tuple(sorted({reason for item in bound for reason in item.unknown_reasons}))
                or logical_attempt.currency != self.pricing_currency
            ):
                raise ValueError("comparison_logical_provider_aggregate_mismatch")
        if any(
            item.pricing_version != self.pricing_version
            or item.pricing_hash != self.pricing_hash
            or item.currency != self.pricing_currency
            for item in self.provider_attempts
        ):
            raise ValueError("comparison_provider_pricing_provenance_mismatch")
        expected_role_counts = _provider_role_counts(self.provider_attempts)
        if self.provider_role_counts != expected_role_counts:
            raise ValueError("comparison_provider_role_counts_mismatch")
        if any(item.cache_policy is not self.cache_policy for item in self.logical_attempts):
            raise ValueError("comparison_candidate_cache_policy_mismatch")
        if (
            self.identity_projection.variant_id != self.variant_id
            or self.identity_projection.configuration_id != self.configuration_id
        ):
            raise ValueError("comparison_candidate_identity_projection_mismatch")
        expected_failed = sum(
            item.status is not ComparisonLogicalAttemptStatus.SUCCEEDED
            for item in self.logical_attempts
        )
        if self.failed_case_count != expected_failed:
            raise ValueError("comparison_failed_case_count_mismatch")
        if self.provider_call_count != len(self.provider_attempts):
            raise ValueError("comparison_provider_call_count_mismatch")
        expected_partial = sum(
            (item.known_partial_cost for item in self.provider_attempts),
            start=Decimal(0),
        )
        expected_complete = all(item.complete for item in self.provider_attempts)
        expected_cost = expected_partial if expected_complete else None
        expected_reasons = tuple(
            sorted({reason for item in self.provider_attempts for reason in item.unknown_reasons})
        )
        if (
            self.known_partial_cost != expected_partial
            or self.total_cost != expected_cost
            or self.cost_complete is not expected_complete
            or self.cost_unknown_reasons != expected_reasons
        ):
            raise ValueError("comparison_total_cost_ledger_mismatch")
        _validate_attempt_metric_parity(
            self.metrics,
            self.logical_attempts,
            self.currency,
            self.source_artifact_id,
        )
        return self


class VerifiedCandidateReport(DomainModel):
    reference: ComparisonCandidateReference
    descriptor: ArtifactDescriptor
    evidence: ComparisonCandidateEvidence

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        content = canonical_candidate_evidence(self.evidence)
        if (
            self.evidence.variant_id != self.reference.variant_id
            or self.evidence.evaluation_run_id != self.reference.evaluation_run_id
            or self.evidence.configuration_id != self.reference.configuration_id
            or self.evidence.source_artifact_id != self.descriptor.artifact_id
            or self.descriptor.artifact_id != f"comparison-candidate-{self.reference.variant_id}"
            or self.descriptor.schema_version != COMPARISON_CANDIDATE_EVIDENCE_SCHEMA_VERSION
            or self.descriptor.format != "json"
            or self.descriptor.media_type != "application/json"
            or self.descriptor.relative_path != f"candidates/{self.reference.variant_id}.json"
            or self.descriptor.byte_size != len(content)
            or self.descriptor.sha256_digest != f"sha256:{hashlib.sha256(content).hexdigest()}"
        ):
            raise ValueError("comparison_candidate_artifact_identity_mismatch")
        return self


class ComparisonMetricResult(DomainModel):
    metric_id: Identifier
    unit: Identifier
    value: ComparisonValue
    numerator: ComparisonNumerator
    denominator: ComparisonDenominator
    status: MetricObservationStatus
    gate_status: GateStatus
    baseline_delta: ComparisonValue
    scorer_version: Identifier | UnavailableValue
    evidence_references: tuple[Identifier, ...] = ()


class ComparisonCategoryResult(DomainModel):
    category_id: Identifier
    candidate_variant_id: Identifier
    case_count: Annotated[int, Field(gt=0)]
    metrics: tuple[ComparisonMetricResult, ...]


class ComparisonGateDecision(DomainModel):
    gate_id: Identifier
    candidate_variant_id: Identifier | None = None
    status: GateStatus
    reason_codes: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_gate_decision(self) -> Self:
        if (self.status is GateStatus.PASSED) == bool(self.reason_codes):
            raise ValueError("comparison_gate_reason_state_mismatch")
        return self


class ComparisonCandidateResult(DomainModel):
    reference: ComparisonCandidateReference
    status: ComparisonCandidateStatus
    evidence_status: ComparisonEvidenceStatus
    source_descriptor: ArtifactDescriptor | None = None
    source_evidence: ComparisonCandidateEvidence | None = None
    metrics: tuple[ComparisonMetricResult, ...]
    category_results: tuple[ComparisonCategoryResult, ...] = ()
    gates: tuple[ComparisonGateDecision, ...] = ()
    failed_case_count: NonNegativeInteger = 0
    provider_call_count: NonNegativeInteger = 0
    known_partial_cost: NonNegativeCost = Decimal(0)
    total_cost: NonNegativeCost | None = None
    cost_complete: bool = True
    cost_unknown_reasons: tuple[Identifier, ...] = ()
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")] | None = None
    safe_error_code: Identifier | None = None

    @model_validator(mode="before")
    @classmethod
    def hydrate_legacy_cost_state(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        values = dict(value)
        total = values.get("total_cost")
        provider_calls = int(values.get("provider_call_count", 0))
        complete = total is not None or provider_calls == 0
        values.setdefault("known_partial_cost", total if total is not None else Decimal(0))
        values.setdefault("cost_complete", complete)
        values.setdefault(
            "cost_unknown_reasons",
            () if complete else (COMPARISON_LEGACY_COST_UNAVAILABLE_REASON,),
        )
        source = values.get("source_evidence")
        if values.get("currency") is None and isinstance(source, Mapping):
            values["currency"] = source.get("pricing_currency")
        return values

    @model_validator(mode="after")
    def validate_candidate_result(self) -> Self:
        available_source = self.source_descriptor is not None and self.source_evidence is not None
        if (self.source_descriptor is None) != (self.source_evidence is None):
            raise ValueError("comparison_candidate_result_source_incomplete")
        if (self.evidence_status is ComparisonEvidenceStatus.AVAILABLE) is not available_source:
            raise ValueError("comparison_candidate_result_source_status_mismatch")
        if self.cost_unknown_reasons != tuple(sorted(set(self.cost_unknown_reasons))):
            raise ValueError("comparison_candidate_result_cost_reason_invalid")
        zero_call_exact = self.provider_call_count == 0 and self.total_cost is None
        if self.cost_complete:
            if self.cost_unknown_reasons or (
                not zero_call_exact
                and (
                    self.total_cost is None
                    or self.currency is None
                    or self.known_partial_cost != self.total_cost
                )
            ):
                raise ValueError("comparison_candidate_result_cost_state_mismatch")
        elif self.total_cost is not None or self.currency is None or not self.cost_unknown_reasons:
            raise ValueError("comparison_candidate_result_cost_state_mismatch")
        failed = self.status in {
            ComparisonCandidateStatus.FAILED,
            ComparisonCandidateStatus.INTERRUPTED,
        }
        if failed != (self.safe_error_code is not None):
            raise ValueError("comparison_candidate_result_error_state_mismatch")
        if self.evidence_status is ComparisonEvidenceStatus.AVAILABLE and (
            self.status is not ComparisonCandidateStatus.COMPLETED
        ):
            raise ValueError("comparison_candidate_available_evidence_not_completed")
        if self.source_evidence is not None and self.source_descriptor is not None:
            content = canonical_candidate_evidence(self.source_evidence)
            if (
                self.source_evidence.variant_id != self.reference.variant_id
                or self.source_evidence.evaluation_run_id != self.reference.evaluation_run_id
                or self.source_evidence.configuration_id != self.reference.configuration_id
                or self.source_descriptor.artifact_id != self.source_evidence.source_artifact_id
                or self.source_descriptor.byte_size != len(content)
                or self.source_descriptor.sha256_digest
                != f"sha256:{hashlib.sha256(content).hexdigest()}"
                or self.provider_call_count != self.source_evidence.provider_call_count
                or self.failed_case_count != self.source_evidence.failed_case_count
                or self.known_partial_cost != self.source_evidence.known_partial_cost
                or self.total_cost != self.source_evidence.total_cost
                or self.cost_complete is not self.source_evidence.cost_complete
                or self.cost_unknown_reasons != self.source_evidence.cost_unknown_reasons
                or self.currency != self.source_evidence.currency
            ):
                raise ValueError("comparison_candidate_result_evidence_mismatch")
            _validate_metric_projection(self.metrics, self.source_evidence.metrics)
            _validate_category_projection(
                self.category_results,
                self.source_evidence.category_results,
                self.reference.variant_id,
            )
        metric_ids = tuple(item.metric_id for item in self.metrics)
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("comparison_candidate_result_metric_duplicate")
        if any(
            item.candidate_variant_id != self.reference.variant_id for item in self.category_results
        ):
            raise ValueError("comparison_candidate_category_identity_mismatch")
        gate_ids = tuple(item.gate_id for item in self.gates)
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("comparison_candidate_result_gate_duplicate")
        if any(item.candidate_variant_id != self.reference.variant_id for item in self.gates):
            raise ValueError("comparison_candidate_gate_identity_mismatch")
        _validate_safe_code(self.safe_error_code)
        return self


class ComparisonRecommendation(DomainModel):
    state: ComparisonRecommendationState
    selected_variant_id: Identifier | None = None
    rationale_codes: Annotated[tuple[Identifier, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_recommendation(self) -> Self:
        recommended = self.state is ComparisonRecommendationState.RECOMMENDED
        if recommended != (self.selected_variant_id is not None):
            raise ValueError("comparison_recommendation_selection_mismatch")
        return self


class ComparisonResult(DomainModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    schema_version: Literal["comparison-result-v1"] = COMPARISON_RESULT_SCHEMA_VERSION
    comparison_id: Identifier
    plan: ExperimentPlan
    plan_id: Identifier
    plan_content_hash: Identifier
    axis: ExperimentAxis
    baseline_variant_id: Identifier
    compatibility: ComparisonCompatibility
    shared_setup: ComparisonSharedSetupEvidence
    candidates: Annotated[tuple[ComparisonCandidateResult, ...], Field(min_length=2)]
    category_results: tuple[ComparisonCategoryResult, ...]
    gates: tuple[ComparisonGateDecision, ...]
    cache_observations: tuple[MetricObservation, ...] = ()
    recommendation: ComparisonRecommendation
    provider_call_count: NonNegativeInteger
    known_partial_cost: NonNegativeCost = Decimal(0)
    total_cost: NonNegativeCost | None
    cost_complete: bool = True
    cost_unknown_reasons: tuple[Identifier, ...] = ()
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")] | None
    completed_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def hydrate_legacy_cost_state(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        values = dict(value)
        total = values.get("total_cost")
        values.setdefault("known_partial_cost", total if total is not None else Decimal(0))
        values.setdefault("cost_complete", total is not None)
        values.setdefault(
            "cost_unknown_reasons",
            () if total is not None else (COMPARISON_LEGACY_COST_UNAVAILABLE_REASON,),
        )
        plan = values.get("plan")
        if values.get("currency") is None and isinstance(plan, Mapping):
            pricing = plan.get("pricing")
            if isinstance(pricing, Mapping):
                values["currency"] = pricing.get("currency")
        return values

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        validate_comparison_plan_safe_values(self.plan)
        self.plan.verify_hash()
        if (
            self.plan.plan_id != self.plan_id
            or self.plan.content_hash != self.plan_content_hash
            or self.plan.axis is not self.axis
            or self.plan.baseline_variant_id != self.baseline_variant_id
        ):
            raise ValueError("comparison_result_plan_identity_mismatch")
        variant_order = tuple(item.variant_id for item in self.plan.variants)
        if tuple(item.reference.variant_id for item in self.candidates) != variant_order:
            raise ValueError("comparison_result_candidate_order_mismatch")
        if self.compatibility.axis is not self.axis:
            raise ValueError("comparison_result_compatibility_axis_mismatch")
        if self.compatibility.controlled_dimensions != self.plan.fixed_identities.controlled:
            raise ValueError("comparison_result_controlled_dimensions_mismatch")
        _validate_shared_setup_against_plan(
            self.shared_setup,
            comparison_id=self.comparison_id,
            plan=self.plan,
            require_ready=True,
        )
        candidate_ids = set(variant_order)
        if any(item.variant_id not in candidate_ids for item in self.compatibility.issues):
            raise ValueError("comparison_result_compatibility_candidate_unknown")
        variants = {item.variant_id: item for item in self.plan.variants}
        run_ids = tuple(item.reference.evaluation_run_id for item in self.candidates)
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("comparison_result_evaluation_run_duplicate")
        for candidate in self.candidates:
            variant = variants[candidate.reference.variant_id]
            if (
                candidate.reference.axis_value != variant.axis_value
                or candidate.reference.configuration_id != variant.configuration_id
            ):
                raise ValueError("comparison_result_candidate_reference_mismatch")
            if candidate.source_evidence is not None:
                _validate_candidate_evidence_against_plan(
                    candidate.source_evidence,
                    candidate.reference,
                    self.plan,
                )
                if candidate.gates != _gate_decisions(self.plan, candidate.source_evidence):
                    raise ValueError("comparison_candidate_result_gate_evidence_mismatch")
        if any(item.candidate_variant_id not in candidate_ids for item in self.category_results):
            raise ValueError("comparison_result_category_candidate_unknown")
        if any(
            item.candidate_variant_id is not None and item.candidate_variant_id not in candidate_ids
            for item in self.gates
        ):
            raise ValueError("comparison_result_gate_candidate_unknown")
        required_gates = set(self.plan.selection_policy.required_gate_ids)
        for candidate in self.candidates:
            if not required_gates.issubset({item.gate_id for item in candidate.gates}):
                raise ValueError("comparison_candidate_required_gates_mismatch")
        baseline = next(
            item
            for item in self.candidates
            if item.reference.variant_id == self.baseline_variant_id
        )
        baseline_metrics = {item.metric_id: item for item in baseline.metrics}
        expected_metric_order = tuple(baseline_metrics)
        for candidate in self.candidates:
            if tuple(item.metric_id for item in candidate.metrics) != expected_metric_order:
                raise ValueError("comparison_result_metric_alignment_mismatch")
            for metric in candidate.metrics:
                baseline_metric = baseline_metrics[metric.metric_id]
                expected_delta = _baseline_delta(metric, baseline_metric)
                if metric.baseline_delta != expected_delta:
                    raise ValueError("comparison_result_baseline_delta_mismatch")
        expected_categories = tuple(
            category for candidate in self.candidates for category in candidate.category_results
        )
        if self.category_results != expected_categories:
            raise ValueError("comparison_result_category_projection_mismatch")
        expected_gates = tuple(gate for candidate in self.candidates for gate in candidate.gates)
        if self.gates != expected_gates:
            raise ValueError("comparison_result_gate_projection_mismatch")
        expected_cache_observations, _ = _cache_axis_observations(self.plan, self.candidates)
        if self.cache_observations != expected_cache_observations:
            raise ValueError("comparison_result_cache_observation_mismatch")
        setup_provider_calls = self.shared_setup.provider_call_count
        if not isinstance(setup_provider_calls, int):
            raise ValueError("comparison_result_setup_provider_calls_unavailable")
        if self.provider_call_count != setup_provider_calls + sum(
            item.provider_call_count for item in self.candidates
        ):
            raise ValueError("comparison_result_provider_call_total_mismatch")
        expected_partial = self.shared_setup.known_partial_cost + sum(
            (item.known_partial_cost for item in self.candidates),
            start=Decimal(0),
        )
        expected_complete = self.shared_setup.cost_complete and all(
            item.cost_complete for item in self.candidates
        )
        expected_cost = expected_partial if expected_complete else None
        expected_reasons = tuple(
            sorted(
                {
                    *self.shared_setup.unknown_reasons,
                    *(reason for item in self.candidates for reason in item.cost_unknown_reasons),
                }
            )
        )
        if (
            self.known_partial_cost != expected_partial
            or self.total_cost != expected_cost
            or self.cost_complete is not expected_complete
            or self.cost_unknown_reasons != expected_reasons
            or self.currency != self.plan.pricing.currency
        ):
            raise ValueError("comparison_result_cost_total_mismatch")
        expected_recommendation = _deterministic_recommendation(
            self.plan,
            self.candidates,
            self.compatibility,
            provider_call_count=self.provider_call_count,
            known_partial_cost=self.known_partial_cost,
            total_cost=self.total_cost,
        )
        if self.recommendation != expected_recommendation:
            raise ValueError("comparison_result_recommendation_not_deterministic")
        return self


class ComparisonArtifactManifest(DomainModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    schema_version: Literal["comparison-artifact-manifest-v1"] = (
        COMPARISON_ARTIFACT_MANIFEST_SCHEMA_VERSION
    )
    comparison_id: Identifier
    plan_id: Identifier
    plan_content_hash: Sha256Digest
    candidate_variant_ids: Annotated[tuple[Identifier, ...], Field(min_length=2)]
    artifacts: Annotated[tuple[ArtifactDescriptor, ...], Field(min_length=1)]
    manifest_content_hash: Sha256Digest
    created_at: AwareDatetime = Field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        comparison_id: str,
        plan: ExperimentPlan,
        artifacts: Sequence[ArtifactDescriptor],
        created_at: AwareDatetime | None = None,
    ) -> ComparisonArtifactManifest:
        timestamp = created_at or utc_now()
        validate_comparison_plan_safe_values(plan)
        values: dict[str, object] = {
            "schema_version": COMPARISON_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "comparison_id": comparison_id,
            "plan_id": plan.plan_id,
            "plan_content_hash": plan.content_hash,
            "candidate_variant_ids": tuple(item.variant_id for item in plan.variants),
            "artifacts": tuple(artifacts),
            "created_at": timestamp,
        }
        values["manifest_content_hash"] = _comparison_manifest_hash(values)
        return cls.model_validate(values)

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        ids = tuple(item.artifact_id for item in self.artifacts)
        paths = tuple(item.relative_path for item in self.artifacts)
        if (
            len(ids) != len(set(ids))
            or len(paths) != len(set(paths))
            or len(self.candidate_variant_ids) != len(set(self.candidate_variant_ids))
        ):
            raise ValueError("comparison_manifest_artifact_duplicate")
        if not set(_COMPARISON_REQUIRED_ARTIFACTS).issubset(ids):
            raise ValueError("comparison_manifest_required_artifact_missing")
        expected_candidates = {
            f"comparison-candidate-{variant_id}" for variant_id in self.candidate_variant_ids
        }
        actual_candidates = {
            artifact_id for artifact_id in ids if artifact_id.startswith("comparison-candidate-")
        }
        if not actual_candidates.issubset(expected_candidates):
            raise ValueError("comparison_manifest_candidate_artifact_set_mismatch")
        for descriptor in self.artifacts:
            contract = _COMPARISON_REQUIRED_ARTIFACTS.get(descriptor.artifact_id)
            if contract is None and descriptor.artifact_id.startswith("comparison-candidate-"):
                variant_id = descriptor.artifact_id.removeprefix("comparison-candidate-")
                contract = (
                    COMPARISON_CANDIDATE_EVIDENCE_SCHEMA_VERSION,
                    "application/json",
                    f"candidates/{variant_id}.json",
                )
            if (
                contract is None
                or (
                    descriptor.schema_version,
                    descriptor.media_type,
                    descriptor.relative_path,
                )
                != contract
            ):
                raise ValueError("comparison_manifest_artifact_contract_invalid")
            expected_format = PurePosixPath(contract[2]).suffix.removeprefix(".")
            if descriptor.format != expected_format or descriptor.created_at > self.created_at:
                raise ValueError("comparison_manifest_artifact_contract_invalid")
        values = self.model_dump(mode="python", exclude={"manifest_content_hash"})
        if self.manifest_content_hash != _comparison_manifest_hash(values):
            raise ValueError("comparison_manifest_content_hash_mismatch")
        return self


@dataclass(frozen=True, slots=True)
class ResolvedComparisonArtifact:
    descriptor: ArtifactDescriptor
    content: bytes

    def __post_init__(self) -> None:
        if (
            not self.content
            or self.descriptor.byte_size != len(self.content)
            or self.descriptor.sha256_digest != f"sha256:{hashlib.sha256(self.content).hexdigest()}"
        ):
            raise ValueError("comparison_artifact_integrity_failed")


class ComparisonDomainError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def create_comparison_suite(
    comparison_id: str,
    plan: ExperimentPlan,
    candidate_run_ids: Mapping[str, str],
    *,
    created_at: AwareDatetime | None = None,
) -> ComparisonSuite:
    """Persistable initial state containing the plan and every candidate reference."""

    timestamp = created_at or utc_now()
    if set(candidate_run_ids) != {item.variant_id for item in plan.variants}:
        raise ComparisonDomainError("comparison_candidate_run_ids_mismatch")
    total_cases = plan.fixed_identities.case_count * plan.repeat_order_policy.repeats_per_case
    candidates = tuple(
        ComparisonCandidateHistory(
            reference=ComparisonCandidateReference(
                variant_id=variant.variant_id,
                axis_value=variant.axis_value,
                configuration_id=variant.configuration_id,
                evaluation_run_id=candidate_run_ids[variant.variant_id],
            ),
            snapshots=(
                ComparisonCandidateSnapshot(
                    sequence=0,
                    status=ComparisonCandidateStatus.QUEUED,
                    total_cases=total_cases,
                    recorded_at=timestamp,
                ),
            ),
        )
        for variant in plan.variants
    )
    progress = _progress_from_candidates(
        candidates,
        sequence=0,
        status=ComparisonStatus.QUEUED,
        recorded_at=timestamp,
    )
    return ComparisonSuite(
        comparison_id=comparison_id,
        plan=plan,
        plan_content_hash=plan.content_hash,
        status=ComparisonStatus.QUEUED,
        candidates=candidates,
        progress_history=(progress,),
        created_at=timestamp,
        updated_at=timestamp,
    )


def seal_comparison_candidate_evidence(
    reference: ComparisonCandidateReference,
    evidence: ComparisonCandidateEvidence,
) -> VerifiedCandidateReport:
    """Create the only canonical descriptor accepted for native candidate evidence."""

    content = canonical_candidate_evidence(evidence)
    descriptor = ArtifactDescriptor(
        schema_version=COMPARISON_CANDIDATE_EVIDENCE_SCHEMA_VERSION,
        artifact_id=f"comparison-candidate-{reference.variant_id}",
        format="json",
        media_type="application/json",
        relative_path=f"candidates/{reference.variant_id}.json",
        sha256_digest=f"sha256:{hashlib.sha256(content).hexdigest()}",
        byte_size=len(content),
        created_at=evidence.generated_at,
    )
    return VerifiedCandidateReport(reference=reference, descriptor=descriptor, evidence=evidence)


def aggregate_comparison_result(
    suite: ComparisonSuite,
    compatibility: ComparisonCompatibility,
    verified_reports: Mapping[str, VerifiedCandidateReport | None],
    *,
    shared_setup: ComparisonSharedSetupEvidence,
    completed_at: AwareDatetime | None = None,
) -> ComparisonResult:
    """Build aligned evidence and apply the plan's complete deterministic decision policy."""

    plan = suite.plan
    _validate_shared_setup_against_plan(
        shared_setup,
        comparison_id=suite.comparison_id,
        plan=plan,
        require_ready=True,
    )
    variant_ids = tuple(item.variant_id for item in plan.variants)
    if set(verified_reports) != set(variant_ids):
        raise ComparisonDomainError("comparison_result_candidate_evidence_set_mismatch")
    reports: dict[str, VerifiedCandidateReport] = {}
    for history in suite.candidates:
        report = verified_reports[history.reference.variant_id]
        if report is None:
            continue
        if report.reference != history.reference:
            raise ComparisonDomainError("comparison_result_candidate_reference_mismatch")
        _validate_candidate_evidence_against_plan(report.evidence, report.reference, plan)
        if report.evidence.comparison_id != suite.comparison_id:
            raise ComparisonDomainError("comparison_result_candidate_comparison_mismatch")
        expected_completed = sum(
            item.status is ComparisonLogicalAttemptStatus.SUCCEEDED
            for item in report.evidence.logical_attempts
        )
        latest = history.latest
        if (
            latest.status is not ComparisonCandidateStatus.COMPLETED
            or latest.completed_cases != expected_completed
            or latest.failed_cases != report.evidence.failed_case_count
            or latest.provider_calls != report.evidence.provider_call_count
            or latest.known_partial_cost != report.evidence.known_partial_cost
            or latest.incurred_cost != report.evidence.total_cost
            or latest.cost_complete is not report.evidence.cost_complete
            or latest.cost_unknown_reasons != report.evidence.cost_unknown_reasons
            or latest.currency != report.evidence.currency
        ):
            raise ComparisonDomainError("comparison_result_candidate_history_evidence_mismatch")
        reports[history.reference.variant_id] = report

    observations = {
        variant_id: {item.metric_id: item for item in report.evidence.metrics}
        for variant_id, report in reports.items()
    }
    metric_order = _aligned_metric_order(plan, observations)
    metric_templates = {
        metric_id: next(
            values[metric_id] for values in observations.values() if metric_id in values
        )
        for metric_id in metric_order
    }
    baseline_observations = observations.get(plan.baseline_variant_id, {})

    category_evidence = {
        variant_id: {item.category_id: item for item in report.evidence.category_results}
        for variant_id, report in reports.items()
    }
    category_order = tuple(
        sorted({name for values in category_evidence.values() for name in values})
    )
    category_metric_order = {
        category_id: tuple(
            sorted(
                {
                    observation.metric_id
                    for values in category_evidence.values()
                    for category in (values.get(category_id),)
                    if category is not None
                    for observation in category.observations
                }
            )
        )
        for category_id in category_order
    }
    candidates: list[ComparisonCandidateResult] = []
    for history in suite.candidates:
        variant_id = history.reference.variant_id
        report = reports.get(variant_id)
        latest = history.latest
        if report is not None:
            candidate_observations = observations[variant_id]
            metrics = tuple(
                _metric_result(
                    candidate_observations[metric_id],
                    _delta_from_observations(
                        candidate_observations[metric_id],
                        baseline_observations.get(metric_id),
                    ),
                )
                for metric_id in metric_order
            )
            categories = _build_category_results(
                variant_id,
                category_order,
                category_metric_order,
                category_evidence,
                plan.baseline_variant_id,
            )
            gates = _gate_decisions(plan, report.evidence)
            evidence_status = ComparisonEvidenceStatus.AVAILABLE
            source_descriptor = report.descriptor
            source_evidence = report.evidence
            total_cost = report.evidence.total_cost
            known_partial_cost = report.evidence.known_partial_cost
            cost_complete = report.evidence.cost_complete
            cost_unknown_reasons = report.evidence.cost_unknown_reasons
            currency = report.evidence.currency
            failed_case_count = report.evidence.failed_case_count
            provider_calls = report.evidence.provider_call_count
        else:
            metrics = tuple(
                _unavailable_metric_result(
                    metric_id,
                    metric_templates[metric_id].unit,
                    "candidate-evidence-unavailable",
                )
                for metric_id in metric_order
            )
            categories = _unavailable_category_results(
                variant_id,
                category_order,
                category_metric_order,
                category_evidence,
            )
            gates = tuple(
                ComparisonGateDecision(
                    gate_id=gate_id,
                    candidate_variant_id=variant_id,
                    status=GateStatus.UNAVAILABLE,
                    reason_codes=("candidate-evidence-unavailable",),
                )
                for gate_id in plan.selection_policy.required_gate_ids
            )
            evidence_status = (
                ComparisonEvidenceStatus.PARTIAL
                if latest.completed_cases or latest.failed_cases or latest.provider_calls
                else ComparisonEvidenceStatus.UNAVAILABLE
            )
            source_descriptor = None
            source_evidence = None
            total_cost = latest.incurred_cost
            known_partial_cost = latest.known_partial_cost
            cost_complete = latest.cost_complete
            cost_unknown_reasons = latest.cost_unknown_reasons
            currency = latest.currency
            failed_case_count = latest.failed_cases
            provider_calls = latest.provider_calls
        candidates.append(
            ComparisonCandidateResult(
                reference=history.reference,
                status=latest.status,
                evidence_status=evidence_status,
                source_descriptor=source_descriptor,
                source_evidence=source_evidence,
                metrics=metrics,
                category_results=categories,
                gates=gates,
                failed_case_count=failed_case_count,
                provider_call_count=provider_calls,
                known_partial_cost=known_partial_cost,
                total_cost=total_cost,
                cost_complete=cost_complete,
                cost_unknown_reasons=cost_unknown_reasons,
                currency=currency,
                safe_error_code=latest.safe_error_code,
            )
        )
    candidate_tuple = tuple(candidates)
    setup_provider_calls = shared_setup.provider_call_count
    if not isinstance(setup_provider_calls, int):
        raise ComparisonDomainError("comparison_shared_setup_not_ready")
    provider_call_count = setup_provider_calls + sum(
        item.provider_call_count for item in candidate_tuple
    )
    known_partial_cost = shared_setup.known_partial_cost + sum(
        (item.known_partial_cost for item in candidate_tuple),
        start=Decimal(0),
    )
    cost_complete = shared_setup.cost_complete and all(
        item.cost_complete for item in candidate_tuple
    )
    total_cost = known_partial_cost if cost_complete else None
    cost_unknown_reasons = tuple(
        sorted(
            {
                *shared_setup.unknown_reasons,
                *(reason for item in candidate_tuple for reason in item.cost_unknown_reasons),
            }
        )
    )
    currency = plan.pricing.currency
    recommendation = _deterministic_recommendation(
        plan,
        candidate_tuple,
        compatibility,
        provider_call_count=provider_call_count,
        known_partial_cost=known_partial_cost,
        total_cost=total_cost,
    )
    cache_observations, _ = _cache_axis_observations(plan, candidate_tuple)
    return ComparisonResult(
        comparison_id=suite.comparison_id,
        plan=plan,
        plan_id=plan.plan_id,
        plan_content_hash=plan.content_hash,
        axis=plan.axis,
        baseline_variant_id=plan.baseline_variant_id,
        compatibility=compatibility,
        shared_setup=shared_setup,
        candidates=candidate_tuple,
        category_results=tuple(
            category for candidate in candidate_tuple for category in candidate.category_results
        ),
        gates=tuple(gate for candidate in candidate_tuple for gate in candidate.gates),
        cache_observations=cache_observations,
        recommendation=recommendation,
        provider_call_count=provider_call_count,
        known_partial_cost=known_partial_cost,
        total_cost=total_cost,
        cost_complete=cost_complete,
        cost_unknown_reasons=cost_unknown_reasons,
        currency=currency,
        completed_at=completed_at or utc_now(),
    )


def _aligned_metric_order(
    plan: ExperimentPlan,
    observations: Mapping[str, Mapping[str, MetricObservation]],
) -> tuple[str, ...]:
    if not observations:
        return ()
    metric_sets = {frozenset(values) for values in observations.values()}
    if len(metric_sets) != 1:
        raise ComparisonDomainError("comparison_candidate_metric_alignment_mismatch")
    metric_ids = set(next(iter(metric_sets)))
    tie_breakers = tuple(item.metric for item in plan.selection_policy.tie_breakers)
    if not set(tie_breakers).issubset(metric_ids):
        raise ComparisonDomainError("comparison_selection_metric_missing")
    return (*tie_breakers, *tuple(sorted(metric_ids - set(tie_breakers))))


def _metric_gate_status(status: MetricObservationStatus) -> GateStatus:
    if status is MetricObservationStatus.PASSED:
        return GateStatus.PASSED
    if status is MetricObservationStatus.FAILED:
        return GateStatus.FAILED
    return GateStatus.UNAVAILABLE


def _metric_result(
    observation: MetricObservation,
    baseline_delta: ComparisonValue,
) -> ComparisonMetricResult:
    return ComparisonMetricResult(
        metric_id=observation.metric_id,
        unit=observation.unit,
        value=observation.value,
        numerator=observation.numerator,
        denominator=observation.denominator,
        status=observation.status,
        gate_status=_metric_gate_status(observation.status),
        baseline_delta=baseline_delta,
        scorer_version=observation.scorer_version,
        evidence_references=observation.evidence_references,
    )


def _unavailable_metric_result(
    metric_id: str,
    unit: str,
    reason: str,
) -> ComparisonMetricResult:
    unavailable = UnavailableValue(reason=reason)
    unavailable_delta = UnavailableValue(reason="baseline-delta-unavailable")
    return ComparisonMetricResult(
        metric_id=metric_id,
        unit=unit,
        value=unavailable,
        numerator=unavailable,
        denominator=unavailable,
        status=MetricObservationStatus.UNAVAILABLE,
        gate_status=GateStatus.UNAVAILABLE,
        baseline_delta=unavailable_delta,
        scorer_version=unavailable,
    )


def _delta_from_observations(
    observation: MetricObservation,
    baseline: MetricObservation | None,
) -> ComparisonValue:
    if (
        baseline is None
        or observation.unit != baseline.unit
        or not isinstance(observation.value, float)
        or not isinstance(baseline.value, float)
    ):
        return UnavailableValue(reason="baseline-delta-unavailable")
    return observation.value - baseline.value


def _baseline_delta(
    metric: ComparisonMetricResult,
    baseline: ComparisonMetricResult,
) -> ComparisonValue:
    if (
        metric.unit != baseline.unit
        or not isinstance(metric.value, float)
        or not isinstance(baseline.value, float)
    ):
        return UnavailableValue(reason="baseline-delta-unavailable")
    return metric.value - baseline.value


def _validate_metric_projection(
    projected: Sequence[ComparisonMetricResult],
    observations: Sequence[MetricObservation],
) -> None:
    by_id = {item.metric_id: item for item in observations}
    if len(by_id) != len(observations) or {item.metric_id for item in projected} != set(by_id):
        raise ValueError("comparison_candidate_result_metric_evidence_mismatch")
    for item in projected:
        expected = _metric_result(by_id[item.metric_id], item.baseline_delta)
        if item != expected:
            raise ValueError("comparison_candidate_result_metric_evidence_mismatch")


def _validate_category_projection(
    projected: Sequence[ComparisonCategoryResult],
    categories: Sequence[CategoryResultV2],
    variant_id: str,
) -> None:
    by_id = {item.category_id: item for item in categories}
    if len(by_id) != len(categories) or {item.category_id for item in projected} != set(by_id):
        raise ValueError("comparison_candidate_result_category_evidence_mismatch")
    for item in projected:
        source = by_id[item.category_id]
        if item.candidate_variant_id != variant_id or item.case_count != source.case_count:
            raise ValueError("comparison_candidate_result_category_evidence_mismatch")
        _validate_metric_projection(item.metrics, source.observations)


def _build_category_results(
    variant_id: str,
    category_order: Sequence[str],
    metric_order: Mapping[str, tuple[str, ...]],
    evidence: Mapping[str, Mapping[str, CategoryResultV2]],
    baseline_variant_id: str,
) -> tuple[ComparisonCategoryResult, ...]:
    candidate = evidence.get(variant_id, {})
    baseline = evidence.get(baseline_variant_id, {})
    values: list[ComparisonCategoryResult] = []
    for category_id in category_order:
        category = candidate.get(category_id)
        if category is None:
            raise ComparisonDomainError("comparison_category_alignment_mismatch")
        observations = {item.metric_id: item for item in category.observations}
        baseline_observations = {
            item.metric_id: item for item in baseline.get(category_id, category).observations
        }
        if set(observations) != set(metric_order[category_id]):
            raise ComparisonDomainError("comparison_category_metric_alignment_mismatch")
        values.append(
            ComparisonCategoryResult(
                category_id=category_id,
                candidate_variant_id=variant_id,
                case_count=category.case_count,
                metrics=tuple(
                    _metric_result(
                        observations[metric_id],
                        _delta_from_observations(
                            observations[metric_id],
                            baseline_observations.get(metric_id),
                        ),
                    )
                    for metric_id in metric_order[category_id]
                ),
            )
        )
    return tuple(values)


def _unavailable_category_results(
    variant_id: str,
    category_order: Sequence[str],
    metric_order: Mapping[str, tuple[str, ...]],
    evidence: Mapping[str, Mapping[str, CategoryResultV2]],
) -> tuple[ComparisonCategoryResult, ...]:
    values: list[ComparisonCategoryResult] = []
    for category_id in category_order:
        template = next(items[category_id] for items in evidence.values() if category_id in items)
        observations = {item.metric_id: item for item in template.observations}
        values.append(
            ComparisonCategoryResult(
                category_id=category_id,
                candidate_variant_id=variant_id,
                case_count=template.case_count,
                metrics=tuple(
                    _unavailable_metric_result(
                        metric_id,
                        observations[metric_id].unit,
                        "candidate-evidence-unavailable",
                    )
                    for metric_id in metric_order[category_id]
                ),
            )
        )
    return tuple(values)


def _gate_decisions(
    plan: ExperimentPlan,
    evidence: ComparisonCandidateEvidence,
) -> tuple[ComparisonGateDecision, ...]:
    gates = {item.gate_id: item for item in evidence.gates}
    required = set(plan.selection_policy.required_gate_ids)
    values: list[ComparisonGateDecision] = []
    for source in evidence.gates:
        gate_id = source.gate_id
        reasons = (
            ()
            if source.status is GateStatus.PASSED
            else source.failure_reasons
            or (
                "mandatory-gate-failed"
                if source.status is GateStatus.FAILED
                else "mandatory-gate-unavailable",
            )
        )
        values.append(
            ComparisonGateDecision(
                gate_id=gate_id,
                candidate_variant_id=evidence.variant_id,
                status=source.status,
                reason_codes=reasons,
            )
        )
    for gate_id in sorted(required - set(gates)):
        values.append(
            ComparisonGateDecision(
                gate_id=gate_id,
                candidate_variant_id=evidence.variant_id,
                status=GateStatus.UNAVAILABLE,
                reason_codes=("mandatory-gate-evidence-missing",),
            )
        )
    return tuple(values)


def _cache_axis_observations(
    plan: ExperimentPlan,
    candidates: Sequence[ComparisonCandidateResult],
) -> tuple[tuple[MetricObservation, ...], tuple[str, ...]]:
    if plan.axis is not ExperimentAxis.CACHE_BEHAVIOR:
        return (), ()
    reasons: list[str] = []
    if plan.repeat_order_policy.repeats_per_case != 1:
        reasons.append("cache-repeat-policy-invalid")
    by_axis = {item.reference.axis_value: item for item in candidates}
    cold = by_axis.get("cold")
    warm = by_axis.get("warm")
    if cold is None or warm is None or cold.source_evidence is None or warm.source_evidence is None:
        reasons.append("cache-paired-evidence-incomplete")
        return _unavailable_cache_observations(reasons[0], 0), tuple(dict.fromkeys(reasons))
    cold_evidence = cold.source_evidence
    warm_evidence = warm.source_evidence
    cold_attempts = {
        (item.case_id, item.repeat_index): item
        for item in cold_evidence.logical_attempts
        if item.cache_outcome in {CacheOutcome.HIT, CacheOutcome.MISS}
    }
    warm_attempts = {
        (item.case_id, item.repeat_index): item
        for item in warm_evidence.logical_attempts
        if item.cache_outcome in {CacheOutcome.HIT, CacheOutcome.MISS}
    }
    pair_keys = set(cold_attempts) & set(warm_attempts)
    pair_sets_match = bool(cold_attempts) and set(cold_attempts) == set(warm_attempts)
    if not pair_sets_match:
        reasons.append("cache-eligible-pair-set-mismatch")
    if any(
        item.cache_outcome is CacheOutcome.ERROR
        for candidate in (cold, warm)
        for item in cast(ComparisonCandidateEvidence, candidate.source_evidence).logical_attempts
    ):
        reasons.append("cache-error-observed")
    if any(
        item.cache_outcome not in {CacheOutcome.HIT, CacheOutcome.MISS}
        for candidate in (cold, warm)
        for item in cast(ComparisonCandidateEvidence, candidate.source_evidence).logical_attempts
    ):
        reasons.append("cache-ineligible-attempt-scheduled")
    total_reduction = 0
    retrieval_reduction = 0
    embedding_reduction = 0
    reranking_reduction = 0
    latency_delta = 0.0
    equivalent_pairs = 0
    warm_hit_count = sum(item.cache_outcome is CacheOutcome.HIT for item in warm_attempts.values())
    for key in sorted(pair_keys):
        cold_attempt = cold_attempts[key]
        warm_attempt = warm_attempts[key]
        if cold_attempt.cache_outcome is not CacheOutcome.MISS:
            reasons.append("cache-cold-not-miss")
        if warm_attempt.cache_outcome is not CacheOutcome.HIT:
            reasons.append("cache-warm-not-hit")
        equivalent = (
            cold_attempt.index_revision_id == warm_attempt.index_revision_id
            and cold_attempt.retrieved_chunk_ids == warm_attempt.retrieved_chunk_ids
            and cold_attempt.context_chunk_ids == warm_attempt.context_chunk_ids
            and cold_attempt.retrieval_evidence_digest == warm_attempt.retrieval_evidence_digest
        )
        if not equivalent:
            reasons.append("cache-retrieval-equivalence-mismatch")
        else:
            equivalent_pairs += 1
        cold_bound = _providers_for_logical(cold_evidence, cold_attempt.attempt_id)
        warm_bound = _providers_for_logical(warm_evidence, warm_attempt.attempt_id)
        cold_total = len(cold_bound)
        warm_total = len(warm_bound)
        cold_retrieval = sum(item.evidence.operation_id == "qa-retrieval" for item in cold_bound)
        warm_retrieval = sum(item.evidence.operation_id == "qa-retrieval" for item in warm_bound)
        cold_embedding = sum(
            item.evidence.operation_id == "qa-retrieval"
            and item.evidence.role is ModelRole.EMBEDDING
            for item in cold_bound
        )
        warm_embedding = sum(
            item.evidence.operation_id == "qa-retrieval"
            and item.evidence.role is ModelRole.EMBEDDING
            for item in warm_bound
        )
        cold_reranking = sum(
            item.evidence.operation_id == "qa-retrieval"
            and item.evidence.role is ModelRole.RERANKING
            for item in cold_bound
        )
        warm_reranking = sum(
            item.evidence.operation_id == "qa-retrieval"
            and item.evidence.role is ModelRole.RERANKING
            for item in warm_bound
        )
        if warm_retrieval != 0 or cold_retrieval <= warm_retrieval:
            reasons.append("cache-retrieval-provider-call-reduction-missing")
        if warm_embedding != 0 or cold_embedding <= warm_embedding:
            reasons.append("cache-embedding-call-reduction-missing")
        if cold_reranking and warm_reranking != 0:
            reasons.append("cache-reranking-call-reduction-missing")
        total_reduction += cold_total - warm_total
        retrieval_reduction += cold_retrieval - warm_retrieval
        embedding_reduction += cold_embedding - warm_embedding
        reranking_reduction += cold_reranking - warm_reranking
        latency_delta += warm_attempt.latency_ms - cold_attempt.latency_ms
    pair_count = len(pair_keys)
    normalized_reasons = tuple(dict.fromkeys(reasons))
    references = (cold_evidence.source_artifact_id, warm_evidence.source_artifact_id)
    if not pair_sets_match and pair_count == 0:
        warm_count = len(warm_attempts)
        if warm_count:
            warm_observations = (
                _observed_metric(
                    "comparison-cache-warm-hits",
                    "hits",
                    warm_hit_count,
                    warm_hit_count,
                    warm_count,
                    references,
                ),
                _observed_metric(
                    "comparison-cache-hit-rate",
                    "ratio-per-eligible-pair",
                    warm_hit_count / warm_count,
                    warm_hit_count,
                    warm_count,
                    references,
                ),
            )
        else:
            warm_observations = (
                _unavailable_metric(
                    "comparison-cache-warm-hits",
                    "hits",
                    "cache-warm-eligible-observations-missing",
                ),
                _unavailable_metric(
                    "comparison-cache-hit-rate",
                    "ratio-per-eligible-pair",
                    "cache-warm-eligible-observations-missing",
                ),
            )
        return (
            (
                _unavailable_metric(
                    "comparison-cache-eligible-pairs",
                    "eligible-pairs",
                    "cache-eligible-pair-set-mismatch",
                    denominator=pair_count or None,
                ),
                *warm_observations,
                *_unavailable_cache_pair_observations(
                    "cache-eligible-pair-set-mismatch",
                    pair_count,
                ),
            ),
            normalized_reasons,
        )
    return (
        (
            _observed_metric(
                "comparison-cache-eligible-pairs",
                "eligible-pairs",
                pair_count,
                pair_count,
                pair_count,
                references,
            ),
            _observed_metric(
                "comparison-cache-warm-hits",
                "hits",
                warm_hit_count,
                warm_hit_count,
                pair_count,
                references,
            ),
            _observed_metric(
                "comparison-cache-hit-rate",
                "ratio-per-eligible-pair",
                warm_hit_count / pair_count,
                warm_hit_count,
                pair_count,
                references,
            ),
            _observed_metric(
                "comparison-cache-total-provider-call-reduction",
                "calls-reduced-per-eligible-pair",
                total_reduction / pair_count,
                total_reduction,
                pair_count,
                references,
            ),
            _observed_metric(
                "comparison-cache-retrieval-provider-call-reduction",
                "retrieval-calls-reduced-per-eligible-pair",
                retrieval_reduction / pair_count,
                retrieval_reduction,
                pair_count,
                references,
            ),
            _observed_metric(
                "comparison-cache-embedding-call-reduction",
                "embedding-calls-reduced-per-eligible-pair",
                embedding_reduction / pair_count,
                embedding_reduction,
                pair_count,
                references,
            ),
            _observed_metric(
                "comparison-cache-reranking-call-reduction",
                "reranking-calls-reduced-per-eligible-pair",
                reranking_reduction / pair_count,
                reranking_reduction,
                pair_count,
                references,
            ),
            _observed_metric(
                "comparison-cache-latency-delta",
                "warm-minus-cold-milliseconds-per-eligible-pair",
                latency_delta / pair_count,
                latency_delta,
                pair_count,
                references,
            ),
            _observed_metric(
                "comparison-cache-retrieval-equivalence-rate",
                "ratio-per-eligible-pair",
                equivalent_pairs / pair_count,
                equivalent_pairs,
                pair_count,
                references,
            ),
        ),
        normalized_reasons,
    )


def _providers_for_logical(
    evidence: ComparisonCandidateEvidence,
    logical_attempt_id: str,
) -> tuple[ComparisonProviderAttempt, ...]:
    return tuple(
        item for item in evidence.provider_attempts if item.logical_attempt_id == logical_attempt_id
    )


def _unavailable_cache_observations(
    reason: str,
    denominator: int,
) -> tuple[MetricObservation, ...]:
    contracts = (
        ("comparison-cache-eligible-pairs", "eligible-pairs"),
        ("comparison-cache-warm-hits", "hits"),
        ("comparison-cache-hit-rate", "ratio-per-eligible-pair"),
        (
            "comparison-cache-total-provider-call-reduction",
            "calls-reduced-per-eligible-pair",
        ),
        (
            "comparison-cache-retrieval-provider-call-reduction",
            "retrieval-calls-reduced-per-eligible-pair",
        ),
        (
            "comparison-cache-embedding-call-reduction",
            "embedding-calls-reduced-per-eligible-pair",
        ),
        (
            "comparison-cache-reranking-call-reduction",
            "reranking-calls-reduced-per-eligible-pair",
        ),
        (
            "comparison-cache-latency-delta",
            "warm-minus-cold-milliseconds-per-eligible-pair",
        ),
        (
            "comparison-cache-retrieval-equivalence-rate",
            "ratio-per-eligible-pair",
        ),
    )
    return tuple(
        _unavailable_metric(
            metric_id,
            unit,
            reason,
            denominator=denominator or None,
        )
        for metric_id, unit in contracts
    )


def _unavailable_cache_pair_observations(
    reason: str,
    denominator: int,
) -> tuple[MetricObservation, ...]:
    contracts = (
        (
            "comparison-cache-total-provider-call-reduction",
            "calls-reduced-per-eligible-pair",
        ),
        (
            "comparison-cache-retrieval-provider-call-reduction",
            "retrieval-calls-reduced-per-eligible-pair",
        ),
        (
            "comparison-cache-embedding-call-reduction",
            "embedding-calls-reduced-per-eligible-pair",
        ),
        (
            "comparison-cache-reranking-call-reduction",
            "reranking-calls-reduced-per-eligible-pair",
        ),
        (
            "comparison-cache-latency-delta",
            "warm-minus-cold-milliseconds-per-eligible-pair",
        ),
        (
            "comparison-cache-retrieval-equivalence-rate",
            "ratio-per-eligible-pair",
        ),
    )
    return tuple(
        _unavailable_metric(
            metric_id,
            unit,
            reason,
            denominator=denominator or None,
        )
        for metric_id, unit in contracts
    )


def _deterministic_recommendation(
    plan: ExperimentPlan,
    candidates: Sequence[ComparisonCandidateResult],
    compatibility: ComparisonCompatibility,
    *,
    provider_call_count: int,
    known_partial_cost: Decimal,
    total_cost: Decimal | None,
) -> ComparisonRecommendation:
    reasons: list[str] = []
    _, cache_reasons = _cache_axis_observations(plan, candidates)
    reasons.extend(cache_reasons)
    if not compatibility.compatible:
        reasons.append("comparison-incompatible")
    if any(
        item.status is not ComparisonCandidateStatus.COMPLETED
        or item.evidence_status is not ComparisonEvidenceStatus.AVAILABLE
        for item in candidates
    ):
        reasons.append("candidate-evidence-incomplete")
    if provider_call_count > plan.maximum_provider_calls:
        reasons.append("provider-call-cap-exceeded")
    if known_partial_cost > plan.maximum_cost or (
        total_cost is not None and total_cost > plan.maximum_cost
    ):
        reasons.append("comparison-cost-cap-exceeded")
    if reasons:
        return ComparisonRecommendation(
            state=ComparisonRecommendationState.NO_RECOMMENDATION,
            rationale_codes=tuple(reasons),
        )
    directions = {item.metric: item.direction for item in plan.selection_policy.tie_breakers}
    variant_order = {item.variant_id: index for index, item in enumerate(plan.variants)}
    candidates_by_axis = {item.reference.axis_value: item for item in candidates}
    required_gate_ids = set(plan.selection_policy.required_gate_ids)
    eligible: list[ComparisonCandidateResult] = []
    reranker_excluded = False
    reranker_benefit_excluded = False
    for candidate in candidates:
        metrics = {item.metric_id: item for item in candidate.metrics}
        if any(
            item.gate_id in required_gate_ids and item.status is not GateStatus.PASSED
            for item in candidate.gates
        ):
            continue
        if any(
            metric_id not in metrics or not isinstance(metrics[metric_id].value, float)
            for metric_id in directions
        ):
            continue
        if (
            plan.axis is ExperimentAxis.RETRIEVAL_STRATEGY
            and candidate.reference.axis_value == "hybrid-rerank"
        ):
            if candidate.source_evidence is None or not any(
                item.discriminating for item in candidate.source_evidence.reranker_evidence
            ):
                reranker_excluded = True
                continue
            if plan.gate_profile.profile_id == COMPARISON_RERANKER_BENEFIT_PROFILE_ID:
                benefit_metric_id = next(
                    (
                        item.metric
                        for item in plan.selection_policy.tie_breakers
                        if item.direction is SelectionDirection.MAXIMIZE
                    ),
                    None,
                )
                hybrid = candidates_by_axis.get("hybrid")
                hybrid_metric = (
                    None
                    if hybrid is None or benefit_metric_id is None
                    else next(
                        (item for item in hybrid.metrics if item.metric_id == benefit_metric_id),
                        None,
                    )
                )
                rerank_metric = (
                    None if benefit_metric_id is None else metrics.get(benefit_metric_id)
                )
                if (
                    hybrid_metric is None
                    or rerank_metric is None
                    or not isinstance(hybrid_metric.value, float)
                    or not isinstance(rerank_metric.value, float)
                    or rerank_metric.value - hybrid_metric.value
                    <= COMPARISON_RERANKER_MIN_QUALITY_BENEFIT
                ):
                    reranker_benefit_excluded = True
                    continue
        eligible.append(candidate)
    if not eligible:
        return ComparisonRecommendation(
            state=ComparisonRecommendationState.NO_RECOMMENDATION,
            rationale_codes=(
                "reranker-non-discriminating"
                if reranker_excluded
                else "reranker-minimum-quality-benefit-not-met"
                if reranker_benefit_excluded
                else "no-candidate-passed-gates",
            ),
        )

    def key(candidate: ComparisonCandidateResult) -> tuple[float | int, ...]:
        metrics = {item.metric_id: item for item in candidate.metrics}
        criteria = tuple(
            -cast(float, metrics[metric_id].value)
            if direction is SelectionDirection.MAXIMIZE
            else cast(float, metrics[metric_id].value)
            for metric_id, direction in directions.items()
        )
        final: tuple[int, ...]
        if plan.selection_policy.final_tie_break is FinalTieBreak.BASELINE_FIRST:
            final = (
                0 if candidate.reference.variant_id == plan.baseline_variant_id else 1,
                variant_order[candidate.reference.variant_id],
            )
        else:
            final = (variant_order[candidate.reference.variant_id],)
        return (*criteria, *final)

    selected = min(eligible, key=key)
    rationale = [
        (
            "comparison-selection-eligibility-passed"
            if COMPARISON_SELECTION_ELIGIBILITY_GATE_ID in plan.selection_policy.required_gate_ids
            else "mandatory-gates-passed"
        ),
        f"selected-by-{plan.selection_policy.policy_id}",
    ]
    if reranker_excluded:
        rationale.append("reranker-non-discriminating-excluded")
    if reranker_benefit_excluded:
        rationale.append("reranker-minimum-quality-benefit-not-met")
    if total_cost is None:
        rationale.append("comparison-cost-lower-bound-only")
    return ComparisonRecommendation(
        state=ComparisonRecommendationState.RECOMMENDED,
        selected_variant_id=selected.reference.variant_id,
        rationale_codes=tuple(rationale),
    )


def project_evaluation_identity(
    variant_id: str,
    identity: EvaluationRunIdentity,
    *,
    corpus_id: str,
    case_set_hash: str,
    cache_behavior: str | None = None,
) -> ComparisonIdentityProjection:
    """Project only allowlisted semantic run identities; configuration ID is diagnostic only."""

    values: dict[str, str] = {
        "code.revision": _safe_identity_text(identity.code_revision),
        "pricing.version": _safe_identity_text(identity.pricing_version),
        "cache.behavior": _safe_identity_text(cache_behavior or identity.cache_policy),
        "environment.python-version": _safe_identity_text(identity.environment.python_version),
        "environment.platform": _safe_identity_text(identity.environment.platform),
        "environment.deployment": _safe_identity_text(identity.environment.deployment),
    }
    for prefix, mapping in (
        ("prompt", identity.prompt_versions),
        ("provider", identity.provider_identities),
        ("generation", identity.generation_settings),
        ("embedding", identity.embedding_identity),
        ("chunking", identity.chunking_identity),
        ("retrieval", identity.retrieval_configuration),
        ("scorer", identity.scorer_versions),
        ("seed", identity.random_seeds),
    ):
        allowed = _IDENTITY_KEYS[prefix]
        if any(name not in allowed for name in mapping):
            raise ComparisonDomainError("comparison_identity_key_not_allowlisted")
        for name, value in mapping.items():
            values[f"{prefix}.{name}"] = _safe_identity_text(_canonical_identity_value(value))
    if any(
        name not in {"embedding", "generation", "reranking"} for name in identity.model_identities
    ):
        raise ComparisonDomainError("comparison_identity_key_not_allowlisted")
    for name, value in identity.model_identities.items():
        identity_name = "generation.model" if name == "generation" else f"model.{name}"
        values[identity_name] = _safe_identity_text(_canonical_identity_value(value))
    retrieval_mode = values.get("retrieval.mode")
    reranking = values.get("retrieval.reranking_enabled") == "true"
    if retrieval_mode == "hybrid" and reranking:
        values["retrieval.mode"] = "hybrid-rerank"
    return ComparisonIdentityProjection(
        variant_id=variant_id,
        configuration_id=identity.configuration_id,
        dataset_id=identity.dataset_id,
        dataset_version=identity.dataset_version,
        dataset_hash=identity.dataset_hash,
        corpus_id=corpus_id,
        corpus_version=identity.corpus_version,
        corpus_hash=identity.corpus_hash,
        case_set_hash=case_set_hash,
        identities=tuple(FixedIdentity(name=name, value=value) for name, value in values.items()),
    )


def validate_comparison_compatibility(
    plan: ExperimentPlan,
    projections: Sequence[ComparisonIdentityProjection],
) -> ComparisonCompatibility:
    """Require exact controlled identities while allowing only the declared axis bundle."""

    by_variant = {item.variant_id: item for item in projections}
    if len(by_variant) != len(projections) or set(by_variant) != {
        item.variant_id for item in plan.variants
    }:
        raise ComparisonDomainError("comparison_identity_candidates_mismatch")
    expected_controlled = {item.name: item.value for item in plan.fixed_identities.controlled}
    allowed_axis = _axis_identity_names(plan.axis)
    issues: list[ComparisonCompatibilityIssue] = []
    unavailable = UnavailableValue(reason="identity-missing")
    for variant in plan.variants:
        projection = by_variant[variant.variant_id]
        actual = projection.identity_map()
        if projection.configuration_id != variant.configuration_id:
            issues.append(
                ComparisonCompatibilityIssue(
                    variant_id=variant.variant_id,
                    code=CompatibilityIssueCode.CONFIGURATION_IDENTITY_MISMATCH,
                    identity_name="configuration.id",
                    expected=variant.configuration_id,
                    actual=projection.configuration_id,
                )
            )
        fixed_checks = (
            ("dataset.id", plan.fixed_identities.dataset_id, projection.dataset_id),
            ("dataset.version", plan.fixed_identities.dataset_version, projection.dataset_version),
            ("dataset.hash", plan.fixed_identities.dataset_hash, projection.dataset_hash),
            ("corpus.id", plan.fixed_identities.corpus_id, projection.corpus_id),
            ("corpus.version", plan.fixed_identities.corpus_version, projection.corpus_version),
            ("corpus.hash", plan.fixed_identities.corpus_hash, projection.corpus_hash),
            ("case-set.hash", plan.fixed_identities.case_set_hash, projection.case_set_hash),
        )
        for name, expected, observed in fixed_checks:
            if expected != observed:
                code = (
                    CompatibilityIssueCode.DATASET_IDENTITY_MISMATCH
                    if name.startswith("dataset")
                    else CompatibilityIssueCode.CORPUS_IDENTITY_MISMATCH
                    if name.startswith("corpus")
                    else CompatibilityIssueCode.CASE_SET_IDENTITY_MISMATCH
                )
                issues.append(
                    ComparisonCompatibilityIssue(
                        variant_id=variant.variant_id,
                        code=code,
                        identity_name=name,
                        expected=expected,
                        actual=observed,
                    )
                )
        axis_actual = actual.get(plan.axis.identity_name)
        if axis_actual != variant.axis_value:
            issues.append(
                ComparisonCompatibilityIssue(
                    variant_id=variant.variant_id,
                    code=CompatibilityIssueCode.AXIS_VALUE_MISMATCH,
                    identity_name=plan.axis.identity_name,
                    expected=variant.axis_value,
                    actual=unavailable if axis_actual is None else axis_actual,
                )
            )
        for name, expected in expected_controlled.items():
            controlled_actual = actual.get(name)
            if controlled_actual is None:
                issues.append(
                    ComparisonCompatibilityIssue(
                        variant_id=variant.variant_id,
                        code=CompatibilityIssueCode.CONTROLLED_IDENTITY_MISSING,
                        identity_name=name,
                        expected=expected,
                        actual=unavailable,
                    )
                )
            elif controlled_actual != expected:
                issues.append(
                    ComparisonCompatibilityIssue(
                        variant_id=variant.variant_id,
                        code=CompatibilityIssueCode.CONTROLLED_IDENTITY_MISMATCH,
                        identity_name=name,
                        expected=expected,
                        actual=controlled_actual,
                    )
                )
        undeclared = set(actual) - set(expected_controlled) - allowed_axis
        for name in sorted(undeclared):
            issues.append(
                ComparisonCompatibilityIssue(
                    variant_id=variant.variant_id,
                    code=CompatibilityIssueCode.CONTROLLED_IDENTITY_UNDECLARED,
                    identity_name=name,
                    expected=UnavailableValue(reason="identity-undeclared"),
                    actual=actual[name],
                )
            )
    return ComparisonCompatibility(
        compatible=not issues,
        axis=plan.axis,
        controlled_dimensions=plan.fixed_identities.controlled,
        issues=tuple(issues),
    )


def canonical_candidate_evidence(evidence: ComparisonCandidateEvidence) -> bytes:
    return (canonical_json_value(evidence.model_dump(mode="json")) + "\n").encode("utf-8")


def canonical_comparison_manifest(manifest: ComparisonArtifactManifest) -> bytes:
    return (canonical_json_value(manifest.model_dump(mode="json")) + "\n").encode("utf-8")


def resolve_comparison_artifact(
    manifest: ComparisonArtifactManifest,
    artifact_id: str,
    content: bytes,
) -> ResolvedComparisonArtifact:
    descriptor = next(
        (item for item in manifest.artifacts if item.artifact_id == artifact_id),
        None,
    )
    if descriptor is None:
        raise ComparisonDomainError("comparison_artifact_not_manifested")
    try:
        return ResolvedComparisonArtifact(descriptor=descriptor, content=content)
    except ValueError:
        raise ComparisonDomainError("comparison_artifact_integrity_failed") from None


def load_verified_candidate_report(
    reference: ComparisonCandidateReference,
    descriptor: ArtifactDescriptor,
    content: bytes,
    *,
    comparison_id: str,
    plan: ExperimentPlan,
) -> VerifiedCandidateReport:
    """Verify canonical bytes and identity before evidence can affect a recommendation."""

    if (
        descriptor.artifact_id != f"comparison-candidate-{reference.variant_id}"
        or descriptor.schema_version != COMPARISON_CANDIDATE_EVIDENCE_SCHEMA_VERSION
        or descriptor.format != "json"
        or descriptor.media_type != "application/json"
        or descriptor.relative_path != f"candidates/{reference.variant_id}.json"
        or descriptor.byte_size != len(content)
        or descriptor.sha256_digest != f"sha256:{hashlib.sha256(content).hexdigest()}"
    ):
        raise ComparisonDomainError("comparison_candidate_artifact_integrity_failed")
    try:
        raw = decode_json_report(content.decode("utf-8"))
        evidence = ComparisonCandidateEvidence.model_validate(raw)
    except (TypeError, UnicodeError, ValueError):
        raise ComparisonDomainError("comparison_candidate_artifact_invalid") from None
    if content != canonical_candidate_evidence(evidence):
        raise ComparisonDomainError("comparison_candidate_artifact_noncanonical")
    if (
        evidence.comparison_id != comparison_id
        or evidence.plan_id != plan.plan_id
        or evidence.plan_content_hash != plan.content_hash
    ):
        raise ComparisonDomainError("comparison_candidate_artifact_plan_mismatch")
    _validate_provider_pricing_against_plan(evidence, plan)
    _validate_candidate_evidence_against_plan(evidence, reference, plan)
    return VerifiedCandidateReport(
        reference=reference,
        descriptor=descriptor,
        evidence=evidence,
    )


def adapt_verified_evaluation_report(
    reference: ComparisonCandidateReference,
    descriptor: ArtifactDescriptor,
    content: bytes,
    *,
    comparison_id: str,
    plan: ExperimentPlan,
    run_identity: EvaluationRunIdentity,
    corpus_id: str,
    expected_case_ids: Sequence[str],
    cache_behavior: str | None = None,
) -> VerifiedCandidateReport:
    """Adapt a genuine full v2 report without inventing missing candidate evidence."""

    if (
        descriptor.artifact_id != "evaluation-report-json"
        or descriptor.schema_version != REPORT_SCHEMA_VERSION_V2
        or descriptor.format != "json"
        or descriptor.media_type != "application/json"
        or descriptor.relative_path != "evaluation-report.json"
        or descriptor.byte_size != len(content)
        or descriptor.sha256_digest != f"sha256:{hashlib.sha256(content).hexdigest()}"
    ):
        raise ComparisonDomainError("comparison_candidate_artifact_integrity_failed")
    try:
        raw = decode_json_report(content.decode("utf-8"))
        report = parse_report_v2(cast(Mapping[str, object], raw))
    except (TypeError, UnicodeError, ValueError):
        raise ComparisonDomainError("comparison_candidate_artifact_invalid") from None
    if content != canonical_report_document_v2(report):
        raise ComparisonDomainError("comparison_candidate_artifact_noncanonical")
    projection = project_evaluation_identity(
        reference.variant_id,
        run_identity,
        corpus_id=corpus_id,
        case_set_hash=case_ids_content_hash(expected_case_ids),
        cache_behavior=cache_behavior,
    )
    _validate_source_report_binding(
        report,
        reference=reference,
        plan=plan,
        identity=run_identity,
        projection=projection,
        expected_case_ids=expected_case_ids,
    )
    evidence = _candidate_evidence_from_report(
        report,
        comparison_id=comparison_id,
        plan=plan,
        variant=reference,
        identity_projection=projection,
        expected_case_ids=expected_case_ids,
        source_report_descriptor=descriptor,
    )
    candidate_content = canonical_candidate_evidence(evidence)
    candidate_descriptor = ArtifactDescriptor(
        schema_version=COMPARISON_CANDIDATE_EVIDENCE_SCHEMA_VERSION,
        artifact_id=f"comparison-candidate-{reference.variant_id}",
        format="json",
        media_type="application/json",
        relative_path=f"candidates/{reference.variant_id}.json",
        sha256_digest=f"sha256:{hashlib.sha256(candidate_content).hexdigest()}",
        byte_size=len(candidate_content),
        created_at=report.generated_at,
    )
    return VerifiedCandidateReport(
        reference=reference,
        descriptor=candidate_descriptor,
        evidence=evidence,
    )


def _candidate_evidence_from_report(
    report: EvaluationReportV2,
    *,
    comparison_id: str,
    plan: ExperimentPlan,
    variant: ComparisonCandidateReference,
    identity_projection: ComparisonIdentityProjection,
    expected_case_ids: Sequence[str],
    source_report_descriptor: ArtifactDescriptor,
) -> ComparisonCandidateEvidence:
    gate = next(item for item in report.gates if item.gate_id == report.acceptance_gate_id)
    performance = report.performance_evidence
    provider_attempts = _provider_attempts_from_report(report, plan)
    logical_attempts = tuple(
        _logical_attempt_from_load_attempt(
            report,
            plan,
            attempt,
            order_index,
            tuple(
                item for item in provider_attempts if item.logical_attempt_id == attempt.attempt_id
            ),
        )
        for order_index, attempt in enumerate(performance.measured.attempts)
    )
    selection_gate = build_comparison_selection_eligibility_gate(
        logical_attempts,
        provider_attempts,
        expected_logical_attempt_count=(
            len(expected_case_ids) * plan.repeat_order_policy.repeats_per_case
        ),
    )
    gates = (
        *(
            item
            for item in report.gates
            if item.gate_id != COMPARISON_SELECTION_ELIGIBILITY_GATE_ID
        ),
        selection_gate,
    )
    return build_comparison_candidate_evidence(
        comparison_id=comparison_id,
        plan=plan,
        reference=variant,
        identity_projection=identity_projection,
        expected_case_ids=expected_case_ids,
        logical_attempts=logical_attempts,
        provider_attempts=provider_attempts,
        quality_metrics=gate.observations,
        gates=gates,
        category_results=report.category_results,
        source_report_descriptor=source_report_descriptor,
        generated_at=report.generated_at,
    )


def build_comparison_candidate_evidence(
    *,
    comparison_id: str,
    plan: ExperimentPlan,
    reference: ComparisonCandidateReference,
    identity_projection: ComparisonIdentityProjection,
    expected_case_ids: Sequence[str],
    logical_attempts: Sequence[ComparisonLogicalAttempt],
    provider_attempts: Sequence[ComparisonProviderAttempt],
    quality_metrics: Sequence[MetricObservation],
    gates: Sequence[GateResult],
    category_results: Sequence[CategoryResultV2] = (),
    reranker_evidence: Sequence[RerankerCaseEvidence] = (),
    source_report_descriptor: ArtifactDescriptor | None = None,
    generated_at: AwareDatetime | None = None,
) -> ComparisonCandidateEvidence:
    """Build exact comparison metrics solely from the immutable logical-attempt ledger."""

    attempts = tuple(logical_attempts)
    providers = tuple(provider_attempts)
    if not attempts:
        raise ComparisonDomainError("comparison_logical_attempts_empty")
    _validate_reference_against_plan(reference, plan)
    _validate_projection_against_plan(identity_projection, reference, plan)
    _validate_logical_attempt_matrix(attempts, expected_case_ids, plan)
    _validate_provider_pricing_values_against_plan(providers, plan)
    _validate_variant_provider_identity(providers, reference, identity_projection, plan)
    _validate_required_provider_roles(
        attempts,
        providers,
        reference,
        identity_projection,
        plan,
    )
    known_partial_cost = sum(
        (item.known_partial_cost for item in providers),
        start=Decimal(0),
    )
    cost_complete = all(item.complete for item in providers)
    cost_unknown_reasons = tuple(
        sorted({reason for item in providers for reason in item.unknown_reasons})
    )
    currency = plan.pricing.currency
    total_cost = known_partial_cost if cost_complete else None
    source_id = f"comparison-candidate-{reference.variant_id}"
    attempt_metrics = _logical_attempt_metric_observations(attempts, currency, source_id)
    quality = tuple(quality_metrics)
    if set(item.metric_id for item in quality) & set(item.metric_id for item in attempt_metrics):
        raise ComparisonDomainError("comparison_metric_id_collision")
    return ComparisonCandidateEvidence(
        comparison_id=comparison_id,
        plan_id=plan.plan_id,
        plan_content_hash=plan.content_hash,
        variant_id=reference.variant_id,
        evaluation_run_id=reference.evaluation_run_id,
        configuration_id=reference.configuration_id,
        identity_projection=identity_projection,
        source_artifact_id=source_id,
        source_report_descriptor=source_report_descriptor,
        cache_policy=plan.cache_policy,
        case_ids=tuple(expected_case_ids),
        logical_attempts=attempts,
        provider_attempts=providers,
        provider_role_counts=_provider_role_counts(providers),
        pricing_version=plan.pricing.pricing_version,
        pricing_hash=plan.pricing.pricing_hash,
        pricing_currency=plan.pricing.currency,
        metrics=(*quality, *attempt_metrics),
        gates=tuple(gates),
        category_results=tuple(category_results),
        failed_case_count=sum(
            item.status is not ComparisonLogicalAttemptStatus.SUCCEEDED for item in attempts
        ),
        provider_call_count=sum(item.provider_attempt_count for item in attempts),
        known_partial_cost=known_partial_cost,
        total_cost=total_cost,
        cost_complete=cost_complete,
        cost_unknown_reasons=cost_unknown_reasons,
        currency=currency,
        reranker_evidence=tuple(reranker_evidence),
        generated_at=generated_at or utc_now(),
    )


def build_comparison_selection_eligibility_gate(
    logical_attempts: Sequence[ComparisonLogicalAttempt],
    provider_attempts: Sequence[ComparisonProviderAttempt],
    *,
    expected_logical_attempt_count: int,
) -> GateResult:
    """Derive the Phase 15 relative-selection gate from immutable ledgers."""

    attempts = tuple(logical_attempts)
    providers = tuple(provider_attempts)
    observed_count = len(attempts)
    if expected_logical_attempt_count <= 0 or observed_count <= 0:
        raise ComparisonDomainError("comparison_selection_evidence_empty")
    references = tuple(item.attempt_id for item in attempts)
    coverage = observed_count / expected_logical_attempt_count
    error_count = sum(item.status is ComparisonLogicalAttemptStatus.ERROR for item in attempts)
    timeout_count = sum(item.status is ComparisonLogicalAttemptStatus.TIMEOUT for item in attempts)
    p90_ms = nearest_rank_percentile(tuple(item.latency_ms for item in attempts), 90)
    cost_complete = all(item.complete for item in providers)
    unavailable_cost = UnavailableValue(reason="provider-cost-evidence-incomplete")
    observations = (
        MetricObservation(
            metric_id="comparison-terminal-case-coverage",
            unit="ratio",
            value=coverage,
            numerator=float(observed_count),
            denominator=expected_logical_attempt_count,
            eligible=True,
            threshold=1.0,
            operator=EvidenceComparisonOperator.EQUAL,
            scorer_version=COMPARISON_SELECTION_ELIGIBILITY_GATE_VERSION,
            status=(
                MetricObservationStatus.PASSED
                if observed_count == expected_logical_attempt_count
                else MetricObservationStatus.FAILED
            ),
            evidence_references=references,
        ),
        MetricObservation(
            metric_id="comparison-logical-error-rate",
            unit="ratio",
            value=error_count / observed_count,
            numerator=float(error_count),
            denominator=observed_count,
            eligible=True,
            threshold=0.0,
            operator=EvidenceComparisonOperator.LESS_THAN_OR_EQUAL,
            scorer_version=COMPARISON_SELECTION_ELIGIBILITY_GATE_VERSION,
            status=(
                MetricObservationStatus.PASSED
                if error_count == 0
                else MetricObservationStatus.FAILED
            ),
            evidence_references=references,
        ),
        MetricObservation(
            metric_id="comparison-logical-timeout-rate",
            unit="ratio",
            value=timeout_count / observed_count,
            numerator=float(timeout_count),
            denominator=observed_count,
            eligible=True,
            threshold=0.0,
            operator=EvidenceComparisonOperator.LESS_THAN_OR_EQUAL,
            scorer_version=COMPARISON_SELECTION_ELIGIBILITY_GATE_VERSION,
            status=(
                MetricObservationStatus.PASSED
                if timeout_count == 0
                else MetricObservationStatus.FAILED
            ),
            evidence_references=references,
        ),
        MetricObservation(
            metric_id="comparison-provider-cost-evidence-completeness",
            unit="ratio",
            value=1.0 if cost_complete else unavailable_cost,
            numerator=1.0 if cost_complete else unavailable_cost,
            denominator=1,
            eligible=True,
            threshold=1.0,
            operator=EvidenceComparisonOperator.GREATER_THAN_OR_EQUAL,
            scorer_version=COMPARISON_SELECTION_ELIGIBILITY_GATE_VERSION,
            status=(
                MetricObservationStatus.PASSED
                if cost_complete
                else MetricObservationStatus.UNAVAILABLE
            ),
            evidence_references=references,
        ),
        MetricObservation(
            metric_id="comparison-logical-all-p90-ms",
            unit="milliseconds",
            value=p90_ms,
            numerator=p90_ms,
            denominator=observed_count,
            eligible=True,
            threshold=COMPARISON_SELECTION_MAX_P90_MS,
            operator=EvidenceComparisonOperator.LESS_THAN_OR_EQUAL,
            scorer_version=COMPARISON_SELECTION_ELIGIBILITY_GATE_VERSION,
            status=(
                MetricObservationStatus.PASSED
                if p90_ms <= COMPARISON_SELECTION_MAX_P90_MS
                else MetricObservationStatus.FAILED
            ),
            evidence_references=references,
        ),
    )
    failure_reasons = tuple(
        reason
        for failed, reason in (
            (
                observed_count != expected_logical_attempt_count,
                "comparison-terminal-case-coverage-incomplete",
            ),
            (
                p90_ms > COMPARISON_SELECTION_MAX_P90_MS,
                "comparison-logical-p90-limit-exceeded",
            ),
        )
        if failed
    )
    valid = True
    passed = not failure_reasons
    return GateResult(
        gate_id=COMPARISON_SELECTION_ELIGIBILITY_GATE_ID,
        profile_version=COMPARISON_SELECTION_ELIGIBILITY_GATE_VERSION,
        status=(
            GateStatus.PASSED if passed else GateStatus.FAILED if valid else GateStatus.UNAVAILABLE
        ),
        valid=valid,
        passed=passed,
        case_executions_complete=(observed_count == expected_logical_attempt_count),
        observations=observations,
        failure_reasons=failure_reasons,
    )


def _logical_attempt_metric_observations(
    attempts: Sequence[ComparisonLogicalAttempt],
    currency: str | None,
    source_artifact_id: str,
) -> tuple[MetricObservation, ...]:
    references = (source_artifact_id,)
    values: list[MetricObservation] = []
    successful = tuple(
        item for item in attempts if item.status is ComparisonLogicalAttemptStatus.SUCCEEDED
    )
    for scope_name, scoped_attempts in (
        ("comparison-logical-all", tuple(attempts)),
        ("comparison-logical-successful", successful),
    ):
        for percentile in ("p50", "p90", "p95"):
            metric_id = f"{scope_name}-{percentile}-ms"
            if not scoped_attempts:
                values.append(_unavailable_metric(metric_id, "milliseconds", "no-attempts"))
            else:
                observed = nearest_rank_percentile(
                    tuple(item.latency_ms for item in scoped_attempts),
                    int(percentile[1:]),
                )
                values.append(
                    _observed_metric(
                        metric_id,
                        "milliseconds",
                        observed,
                        observed,
                        len(scoped_attempts),
                        references,
                    )
                )
    denominator = len(attempts)
    for metric_id, numerator in (
        (
            "comparison-logical-error-rate",
            sum(item.status is ComparisonLogicalAttemptStatus.ERROR for item in attempts),
        ),
        (
            "comparison-logical-timeout-rate",
            sum(item.status is ComparisonLogicalAttemptStatus.TIMEOUT for item in attempts),
        ),
    ):
        values.append(
            _observed_metric(
                metric_id,
                "ratio",
                numerator / denominator,
                numerator,
                denominator,
                references,
            )
        )
    degradation_metric_id = "comparison-logical-degradation-rate"
    if all(item.degradation_evidence_complete for item in attempts):
        degradation_count = sum(item.degraded for item in attempts)
        values.append(
            _observed_metric(
                degradation_metric_id,
                "ratio",
                degradation_count / denominator,
                degradation_count,
                denominator,
                references,
            )
        )
    else:
        values.append(
            _unavailable_metric(
                degradation_metric_id,
                "ratio",
                "degradation-evidence-unavailable",
                denominator=denominator,
            )
        )
    for direction in ("input", "output"):
        tokens = tuple(
            item.input_tokens if direction == "input" else item.output_tokens for item in attempts
        )
        provider_denominator = sum(item.provider_attempt_count for item in attempts)
        known_total = sum(item for item in tokens if item is not None)
        if provider_denominator and all(item is not None for item in tokens):
            values.append(
                _observed_metric(
                    f"comparison-{direction}-tokens",
                    "tokens",
                    known_total,
                    known_total,
                    provider_denominator,
                    references,
                )
            )
        else:
            values.append(
                _unavailable_metric(
                    f"comparison-{direction}-tokens",
                    "tokens",
                    f"{direction}-tokens-unavailable",
                    denominator=provider_denominator or None,
                )
            )
    total_cost = (
        sum((cast(Decimal, item.estimated_cost) for item in attempts), start=Decimal(0))
        if currency is not None and all(item.estimated_cost is not None for item in attempts)
        else None
    )
    for metric_id, cost_denominator, zero_reason, unit_scope in (
        (
            "comparison-cost-per-1000-logical-attempts",
            len(attempts),
            "no-attempts",
            "logical-attempts",
        ),
        (
            "comparison-cost-per-1000-successes",
            len(successful),
            "no-successes",
            "successes",
        ),
    ):
        unit = f"{currency or 'unknown'}-per-1000-{unit_scope}"
        if cost_denominator == 0:
            values.append(_unavailable_metric(metric_id, unit, zero_reason))
        elif total_cost is None:
            values.append(
                _unavailable_metric(
                    metric_id,
                    unit,
                    "cost-unavailable",
                    denominator=cost_denominator,
                )
            )
        else:
            values.append(
                _observed_metric(
                    metric_id,
                    unit,
                    float(total_cost * Decimal(1_000) / Decimal(cost_denominator)),
                    float(total_cost),
                    cost_denominator,
                    references,
                )
            )
    return tuple(values)


def _validate_attempt_metric_parity(
    metrics: Sequence[MetricObservation],
    attempts: Sequence[ComparisonLogicalAttempt],
    currency: str | None,
    source_artifact_id: str,
) -> None:
    by_id = {item.metric_id: item for item in metrics}
    expected = _logical_attempt_metric_observations(
        attempts,
        currency,
        source_artifact_id,
    )
    if any(by_id.get(item.metric_id) != item for item in expected):
        raise ValueError("comparison_attempt_metric_ledger_mismatch")


def _logical_attempt_from_load_attempt(
    report: EvaluationReportV2,
    plan: ExperimentPlan,
    measured_attempt: LoadAttempt,
    order_index: int,
    provider_attempts: tuple[ComparisonProviderAttempt, ...],
) -> ComparisonLogicalAttempt:
    cost_known = all(item.total_cost is not None for item in provider_attempts)
    estimated_cost = (
        sum((cast(Decimal, item.total_cost) for item in provider_attempts), start=Decimal(0))
        if cost_known
        else None
    )
    known_partial_cost = sum(
        (item.known_partial_cost for item in provider_attempts),
        start=Decimal(0),
    )
    cost_unknown_reasons = tuple(
        sorted({reason for item in provider_attempts for reason in item.unknown_reasons})
    )
    input_applicable = tuple(
        item for item in provider_attempts if item.input_per_million is not None
    )
    input_values = tuple(item.evidence.usage.input_tokens for item in input_applicable)
    output_applicable = tuple(
        item for item in provider_attempts if item.output_per_million is not None
    )
    output_values = tuple(item.evidence.usage.output_tokens for item in output_applicable)
    status = (
        ComparisonLogicalAttemptStatus.SUCCEEDED
        if measured_attempt.succeeded
        else ComparisonLogicalAttemptStatus.TIMEOUT
        if measured_attempt.status.value == "timeout"
        else ComparisonLogicalAttemptStatus.ERROR
    )
    return ComparisonLogicalAttempt(
        attempt_id=measured_attempt.attempt_id,
        case_id=measured_attempt.scenario_id or measured_attempt.logical_request_id,
        repeat_index=measured_attempt.attempt_number - 1,
        order_index=order_index,
        status=status,
        latency_ms=measured_attempt.latency_ms,
        terminal_kind=(measured_attempt.terminal_kind if measured_attempt.succeeded else "error"),
        cache_policy=plan.cache_policy,
        cache_outcome=_load_attempt_cache_outcome(measured_attempt),
        index_revision_id=UnavailableValue(reason="not-recorded-in-evaluation-report-v2"),
        retrieved_chunk_ids=(),
        context_chunk_ids=(),
        retrieval_evidence_digest=UnavailableValue(reason="not-recorded-in-evaluation-report-v2"),
        safe_error_code=measured_attempt.safe_error_code,
        provider_attempt_references=tuple(item.attempt_reference for item in provider_attempts),
        provider_failed_attempt_count=measured_attempt.provider_failed_attempt_count,
        input_tokens=(
            sum(cast(int, item) for item in input_values)
            if all(item is not None for item in input_values)
            else None
        ),
        output_tokens=(
            sum(cast(int, item) for item in output_values)
            if all(item is not None for item in output_values)
            else None
        ),
        estimated_cost=estimated_cost,
        known_partial_cost=known_partial_cost,
        cost_complete=cost_known,
        cost_unknown_reasons=cost_unknown_reasons,
        currency=report.performance_evidence.cost.pricing.currency,
        degradation_evidence_complete=False,
        completed_at=measured_attempt.completed_at,
    )


def _provider_attempts_from_report(
    report: EvaluationReportV2,
    plan: ExperimentPlan,
) -> tuple[ComparisonProviderAttempt, ...]:
    values: list[ComparisonProviderAttempt] = []
    for logical_attempt in report.performance_evidence.measured.attempts:
        for ordinal, provider in enumerate(logical_attempt.provider_attempts, start=1):
            rate = next(
                (
                    item
                    for item in plan.pricing.rate_card
                    if item.role.value == provider.role.value
                    and item.provider == provider.provider
                    and item.model == provider.model
                ),
                None,
            )
            if rate is None or provider.latency_ms is None:
                raise ComparisonDomainError("comparison_provider_pricing_or_latency_missing")
            values.append(
                ComparisonProviderAttempt.create(
                    attempt_reference=f"{logical_attempt.attempt_id}-provider-{ordinal}",
                    logical_attempt_id=logical_attempt.attempt_id,
                    evaluation_run_id=report.run_id,
                    evidence=provider,
                    latency_ms=provider.latency_ms,
                    pricing_version=plan.pricing.pricing_version,
                    pricing_hash=plan.pricing.pricing_hash,
                    currency=plan.pricing.currency,
                    input_per_million=rate.input_per_million,
                    output_per_million=rate.output_per_million,
                    pricing_source_reference=rate.source_reference,
                )
            )
    return tuple(values)


def _load_attempt_cache_outcome(attempt: LoadAttempt) -> CacheOutcome:
    value = attempt.cache_status.get("retrieval")
    if value is None:
        return CacheOutcome.NOT_APPLICABLE
    try:
        return CacheOutcome(str(value).strip().casefold())
    except ValueError:
        raise ComparisonDomainError("comparison_source_report_cache_outcome_invalid") from None


def _observed_metric(
    metric_id: str,
    unit: str,
    value: float | int,
    numerator: float | int,
    denominator: int,
    references: tuple[str, ...],
) -> MetricObservation:
    return MetricObservation(
        metric_id=metric_id,
        unit=unit,
        value=float(value),
        numerator=float(numerator),
        denominator=denominator,
        eligible=True,
        scorer_version="comparison-evidence-v1",
        status=MetricObservationStatus.OBSERVED,
        evidence_references=references,
    )


def _derive_provider_attempt_cost(
    evidence: ProviderAttemptEvidence,
    input_per_million: Decimal | None,
    output_per_million: Decimal | None,
) -> tuple[Decimal, Decimal | None, tuple[str, ...]]:
    usage = evidence.usage
    known_partial = Decimal(0)
    reasons: list[str] = []
    if input_per_million is not None:
        if usage.input_tokens is None:
            reasons.append("input-usage-unknown")
        else:
            known_partial += Decimal(usage.input_tokens) * input_per_million / Decimal(1_000_000)
    if output_per_million is not None:
        if usage.output_tokens is None:
            reasons.append("output-usage-unknown")
        else:
            known_partial += Decimal(usage.output_tokens) * output_per_million / Decimal(1_000_000)
    normalized_reasons = tuple(sorted(reasons))
    return (
        known_partial,
        None if normalized_reasons else known_partial,
        normalized_reasons,
    )


def comparison_shared_setup_id(comparison_id: str) -> str:
    """Return the bounded deterministic scope identifier for unbound setup attempts."""

    digest = hashlib.sha256(comparison_id.encode("utf-8")).hexdigest()
    return f"comparison-setup-{digest}"


def _provider_role_counts(
    attempts: Sequence[ComparisonProviderAttempt],
) -> tuple[ComparisonProviderRoleCount, ...]:
    return tuple(
        ComparisonProviderRoleCount(
            role=role,
            attempt_count=sum(item.evidence.role is role for item in attempts),
            successful_count=sum(
                item.evidence.role is role and item.evidence.status is ModelAttemptStatus.SUCCEEDED
                for item in attempts
            ),
            failed_count=sum(
                item.evidence.role is role
                and item.evidence.status is not ModelAttemptStatus.SUCCEEDED
                for item in attempts
            ),
        )
        for role in ModelRole
    )


def _validate_provider_pricing_against_plan(
    evidence: ComparisonCandidateEvidence,
    plan: ExperimentPlan,
) -> None:
    if (
        evidence.pricing_version != plan.pricing.pricing_version
        or evidence.pricing_hash != plan.pricing.pricing_hash
        or evidence.pricing_currency != plan.pricing.currency
    ):
        raise ComparisonDomainError("comparison_candidate_pricing_plan_mismatch")
    _validate_provider_pricing_values_against_plan(evidence.provider_attempts, plan)


def _validate_reference_against_plan(
    reference: ComparisonCandidateReference,
    plan: ExperimentPlan,
) -> None:
    validate_comparison_plan_safe_values(plan)
    plan.verify_hash()
    variant = next(
        (item for item in plan.variants if item.variant_id == reference.variant_id),
        None,
    )
    if variant is None or (
        reference.axis_value != variant.axis_value
        or reference.configuration_id != variant.configuration_id
    ):
        raise ComparisonDomainError("comparison_candidate_reference_plan_mismatch")


def _validate_projection_against_plan(
    projection: ComparisonIdentityProjection,
    reference: ComparisonCandidateReference,
    plan: ExperimentPlan,
) -> None:
    if (
        projection.variant_id != reference.variant_id
        or projection.configuration_id != reference.configuration_id
    ):
        raise ComparisonDomainError("comparison_candidate_projection_reference_mismatch")
    fixed = plan.fixed_identities
    if (
        projection.dataset_id,
        projection.dataset_version,
        projection.dataset_hash,
        projection.corpus_id,
        projection.corpus_version,
        projection.corpus_hash,
        projection.case_set_hash,
    ) != (
        fixed.dataset_id,
        fixed.dataset_version,
        fixed.dataset_hash,
        fixed.corpus_id,
        fixed.corpus_version,
        fixed.corpus_hash,
        fixed.case_set_hash,
    ):
        raise ComparisonDomainError("comparison_candidate_projection_fixed_identity_mismatch")
    actual = projection.identity_map()
    controlled = {item.name: item.value for item in fixed.controlled}
    allowed_axis = _axis_identity_names(plan.axis)
    if actual.get(plan.axis.identity_name) != reference.axis_value:
        raise ComparisonDomainError("comparison_candidate_projection_axis_mismatch")
    if any(actual.get(name) != value for name, value in controlled.items()):
        raise ComparisonDomainError("comparison_candidate_projection_controlled_mismatch")
    if set(actual) - set(controlled) - allowed_axis:
        raise ComparisonDomainError("comparison_candidate_projection_identity_undeclared")


def _validate_logical_attempt_matrix(
    attempts: Sequence[ComparisonLogicalAttempt],
    expected_case_ids: Sequence[str],
    plan: ExperimentPlan,
) -> None:
    case_ids = tuple(expected_case_ids)
    fixed = plan.fixed_identities
    repeats = plan.repeat_order_policy.repeats_per_case
    if (
        len(case_ids) != fixed.case_count
        or len(case_ids) != len(set(case_ids))
        or case_ids_content_hash(case_ids) != fixed.case_set_hash
        or len(attempts) != len(case_ids) * repeats
    ):
        raise ComparisonDomainError("comparison_candidate_case_matrix_mismatch")
    observed = {(item.case_id, item.repeat_index) for item in attempts}
    expected = {(case_id, repeat) for case_id in case_ids for repeat in range(repeats)}
    if len(observed) != len(attempts) or observed != expected:
        raise ComparisonDomainError("comparison_candidate_case_matrix_mismatch")


def _validate_variant_provider_identity(
    attempts: Sequence[ComparisonProviderAttempt],
    reference: ComparisonCandidateReference,
    projection: ComparisonIdentityProjection,
    plan: ExperimentPlan,
) -> None:
    actual = projection.identity_map()
    if any(item.evaluation_run_id != reference.evaluation_run_id for item in attempts):
        raise ComparisonDomainError("comparison_provider_evaluation_run_mismatch")
    for attempt in attempts:
        role = attempt.evidence.role.value
        model_name = "generation.model" if role == "generation" else f"model.{role}"
        expected_model = actual.get(model_name)
        expected_provider = actual.get(f"provider.{role}")
        if (
            expected_model is None
            or expected_provider is None
            or attempt.evidence.model != expected_model
            or attempt.evidence.provider != expected_provider
        ):
            raise ComparisonDomainError("comparison_provider_variant_identity_mismatch")
    if plan.axis is ExperimentAxis.GENERATION_MODEL and any(
        item.evidence.role is ModelRole.GENERATION and item.evidence.model != reference.axis_value
        for item in attempts
    ):
        raise ComparisonDomainError("comparison_provider_variant_identity_mismatch")


def _validate_candidate_evidence_against_plan(
    evidence: ComparisonCandidateEvidence,
    reference: ComparisonCandidateReference,
    plan: ExperimentPlan,
) -> None:
    if (
        evidence.variant_id != reference.variant_id
        or evidence.evaluation_run_id != reference.evaluation_run_id
        or evidence.configuration_id != reference.configuration_id
        or evidence.cache_policy is not plan.cache_policy
    ):
        raise ComparisonDomainError("comparison_candidate_artifact_identity_mismatch")
    _validate_reference_against_plan(reference, plan)
    _validate_projection_against_plan(evidence.identity_projection, reference, plan)
    _validate_logical_attempt_matrix(evidence.logical_attempts, evidence.case_ids, plan)
    _validate_variant_provider_identity(
        evidence.provider_attempts,
        reference,
        evidence.identity_projection,
        plan,
    )
    _validate_required_provider_roles(
        evidence.logical_attempts,
        evidence.provider_attempts,
        reference,
        evidence.identity_projection,
        plan,
    )
    stored_gate = next(
        (
            item
            for item in evidence.gates
            if item.gate_id == COMPARISON_SELECTION_ELIGIBILITY_GATE_ID
        ),
        None,
    )
    gate_required = (
        COMPARISON_SELECTION_ELIGIBILITY_GATE_ID in plan.selection_policy.required_gate_ids
    )
    if stored_gate is None and gate_required:
        raise ComparisonDomainError("comparison_selection_eligibility_gate_mismatch")
    if stored_gate is not None and stored_gate.profile_version not in {
        _LEGACY_COMPARISON_SELECTION_ELIGIBILITY_GATE_VERSION,
        COMPARISON_SELECTION_ELIGIBILITY_GATE_VERSION,
    }:
        raise ComparisonDomainError("comparison_selection_eligibility_gate_mismatch")
    if (
        stored_gate is not None
        and stored_gate.profile_version == COMPARISON_SELECTION_ELIGIBILITY_GATE_VERSION
    ):
        derived_gate = build_comparison_selection_eligibility_gate(
            evidence.logical_attempts,
            evidence.provider_attempts,
            expected_logical_attempt_count=(
                len(evidence.case_ids) * plan.repeat_order_policy.repeats_per_case
            ),
        )
        if stored_gate != derived_gate:
            raise ComparisonDomainError("comparison_selection_eligibility_gate_mismatch")


def _validate_required_provider_roles(
    logical_attempts: Sequence[ComparisonLogicalAttempt],
    provider_attempts: Sequence[ComparisonProviderAttempt],
    reference: ComparisonCandidateReference,
    projection: ComparisonIdentityProjection,
    plan: ExperimentPlan,
) -> None:
    by_logical: dict[str, tuple[ComparisonProviderAttempt, ...]] = {
        logical.attempt_id: tuple(
            item for item in provider_attempts if item.logical_attempt_id == logical.attempt_id
        )
        for logical in logical_attempts
    }
    retrieval_mode = projection.identity_map().get("retrieval.mode")
    for logical in logical_attempts:
        bound = by_logical[logical.attempt_id]
        if logical.cache_outcome is CacheOutcome.HIT and any(
            item.evidence.operation_id == "qa-retrieval" for item in bound
        ):
            raise ComparisonDomainError("comparison_cache_hit_provider_attempt_present")
        if (
            logical.status is not ComparisonLogicalAttemptStatus.SUCCEEDED
            or logical.terminal_kind != "answer"
        ):
            continue
        if plan.cache_policy is CachePolicy.BYPASS:
            if logical.cache_outcome is not CacheOutcome.BYPASS:
                raise ComparisonDomainError("comparison_cache_outcome_plan_mismatch")
            embedding_required = True
        else:
            embedding_required = logical.cache_outcome is not CacheOutcome.HIT
        if embedding_required and not any(
            item.evidence.role is ModelRole.EMBEDDING
            and item.evidence.operation_id == "qa-retrieval"
            and item.evidence.status is ModelAttemptStatus.SUCCEEDED
            and item.evidence.safe_error_category is None
            for item in bound
        ):
            raise ComparisonDomainError("comparison_answer_embedding_evidence_missing")
        if (
            retrieval_mode == "hybrid-rerank"
            and logical.cache_outcome in {CacheOutcome.BYPASS, CacheOutcome.MISS}
            and not any(
                item.evidence.role is ModelRole.RERANKING
                and item.evidence.operation_id == "qa-retrieval"
                for item in bound
            )
        ):
            raise ComparisonDomainError("comparison_answer_reranking_evidence_missing")


def _validate_source_report_binding(
    report: EvaluationReportV2,
    *,
    reference: ComparisonCandidateReference,
    plan: ExperimentPlan,
    identity: EvaluationRunIdentity,
    projection: ComparisonIdentityProjection,
    expected_case_ids: Sequence[str],
) -> None:
    _validate_reference_against_plan(reference, plan)
    fixed = plan.fixed_identities
    provenance = report.provenance
    if (
        report.run_id != reference.evaluation_run_id
        or report.configuration_id != reference.configuration_id
        or provenance.configuration_id != reference.configuration_id
        or provenance.experiment_plan_id != plan.plan_id
        or provenance.experiment_plan_content_hash != plan.content_hash
        or (
            provenance.dataset_id,
            provenance.dataset_version,
            provenance.dataset_content_hash,
        )
        != (fixed.dataset_id, fixed.dataset_version, fixed.dataset_hash)
        or (
            provenance.corpus_id,
            provenance.corpus_version,
            provenance.corpus_content_hash,
        )
        != (fixed.corpus_id, fixed.corpus_version, fixed.corpus_hash)
        or provenance.case_set_content_hash != fixed.case_set_hash
        or provenance.pricing_version != plan.pricing.pricing_version
        or provenance.pricing_content_hash != plan.pricing.pricing_hash
        or identity.configuration_id != reference.configuration_id
        or (
            identity.dataset_id,
            identity.dataset_version,
            identity.dataset_hash,
            identity.corpus_version,
            identity.corpus_hash,
            identity.pricing_version,
            identity.cache_policy,
        )
        != (
            fixed.dataset_id,
            fixed.dataset_version,
            fixed.dataset_hash,
            fixed.corpus_version,
            fixed.corpus_hash,
            plan.pricing.pricing_version,
            plan.cache_policy,
        )
        or case_ids_content_hash(expected_case_ids) != fixed.case_set_hash
    ):
        raise ComparisonDomainError("comparison_source_report_plan_binding_mismatch")
    _validate_projection_against_plan(projection, reference, plan)
    statuses = tuple(
        str(value).strip().casefold()
        for attempt in report.performance_evidence.measured.attempts
        for value in (attempt.cache_status.get("retrieval"),)
        if value is not None
    )
    if not statuses:
        raise ComparisonDomainError("comparison_source_report_cache_evidence_missing")
    if plan.cache_policy is CachePolicy.BYPASS:
        if any(value != "bypass" for value in statuses):
            raise ComparisonDomainError("comparison_source_report_cache_policy_mismatch")
    elif plan.axis is ExperimentAxis.CACHE_BEHAVIOR:
        if reference.axis_value == "cold" and any(value == "hit" for value in statuses):
            raise ComparisonDomainError("comparison_source_report_cache_policy_mismatch")
        if reference.axis_value == "warm" and not any(value == "hit" for value in statuses):
            raise ComparisonDomainError("comparison_source_report_cache_policy_mismatch")


def _validate_provider_pricing_values_against_plan(
    attempts: Sequence[ComparisonProviderAttempt],
    plan: ExperimentPlan,
) -> None:
    rates = {(item.role.value, item.provider, item.model): item for item in plan.pricing.rate_card}
    for attempt in attempts:
        key = (
            attempt.evidence.role.value,
            attempt.evidence.provider,
            attempt.evidence.model,
        )
        rate = rates.get(key)
        if rate is None or (
            attempt.pricing_version != plan.pricing.pricing_version
            or attempt.pricing_hash != plan.pricing.pricing_hash
            or attempt.currency != plan.pricing.currency
            or attempt.input_per_million != rate.input_per_million
            or attempt.output_per_million != rate.output_per_million
            or attempt.pricing_source_reference != rate.source_reference
        ):
            raise ComparisonDomainError("comparison_provider_rate_plan_mismatch")


def _validate_shared_setup_against_plan(
    evidence: ComparisonSharedSetupEvidence,
    *,
    comparison_id: str,
    plan: ExperimentPlan,
    require_ready: bool,
) -> None:
    fixed = plan.fixed_identities
    if (
        evidence.comparison_id != comparison_id
        or evidence.plan_id != plan.plan_id
        or evidence.plan_content_hash != plan.content_hash
        or (evidence.corpus_id, evidence.corpus_version, evidence.corpus_hash)
        != (fixed.corpus_id, fixed.corpus_version, fixed.corpus_hash)
        or evidence.pricing_version != plan.pricing.pricing_version
        or evidence.pricing_hash != plan.pricing.pricing_hash
        or evidence.currency != plan.pricing.currency
    ):
        raise ComparisonDomainError("comparison_shared_setup_plan_binding_mismatch")
    if require_ready and (
        evidence.status is ComparisonSharedSetupStatus.FAILED
        or not evidence.provider_calls_complete
        or not isinstance(evidence.provider_call_count, int)
    ):
        raise ComparisonDomainError("comparison_shared_setup_not_ready")
    controlled = {item.name: item.value for item in fixed.controlled}
    expected_provider = controlled.get("provider.embedding")
    expected_model = controlled.get("model.embedding")
    if evidence.attempts and (expected_provider is None or expected_model is None):
        raise ComparisonDomainError("comparison_shared_setup_embedding_identity_missing")
    rates = {(item.role.value, item.provider, item.model): item for item in plan.pricing.rate_card}
    for attempt in evidence.attempts:
        key = (
            attempt.evidence.role.value,
            attempt.evidence.provider,
            attempt.evidence.model,
        )
        rate = rates.get(key)
        if (
            attempt.evidence.provider != expected_provider
            or attempt.evidence.model != expected_model
            or rate is None
            or attempt.pricing_version != plan.pricing.pricing_version
            or attempt.pricing_hash != plan.pricing.pricing_hash
            or attempt.currency != plan.pricing.currency
            or attempt.input_per_million != rate.input_per_million
            or attempt.output_per_million != rate.output_per_million
            or attempt.pricing_source_reference != rate.source_reference
        ):
            raise ComparisonDomainError("comparison_shared_setup_rate_plan_mismatch")


def validate_comparison_shared_setup(
    evidence: ComparisonSharedSetupEvidence,
    *,
    comparison_id: str,
    plan: ExperimentPlan,
    require_ready: bool = False,
) -> None:
    """Validate setup identity and exact rate-card provenance at a persistence boundary."""

    _validate_shared_setup_against_plan(
        evidence,
        comparison_id=comparison_id,
        plan=plan,
        require_ready=require_ready,
    )


def _unavailable_metric(
    metric_id: str,
    unit: str,
    reason: str,
    *,
    denominator: int | None = None,
) -> MetricObservation:
    unavailable = UnavailableValue(reason=reason)
    return MetricObservation(
        metric_id=metric_id,
        unit=unit,
        value=unavailable,
        numerator=unavailable,
        denominator=unavailable if denominator is None else denominator,
        eligible=denominator is not None,
        scorer_version="comparison-evidence-v1",
        status=MetricObservationStatus.UNAVAILABLE,
    )


def _progress_from_candidates(
    candidates: tuple[ComparisonCandidateHistory, ...],
    *,
    sequence: int,
    status: ComparisonStatus,
    recorded_at: AwareDatetime,
) -> ComparisonProgressSnapshot:
    latest = tuple(item.latest for item in candidates)
    completed = sum(item.status is ComparisonCandidateStatus.COMPLETED for item in latest)
    failed = sum(
        item.status in {ComparisonCandidateStatus.FAILED, ComparisonCandidateStatus.INTERRUPTED}
        for item in latest
    )
    active = sum(item.status is ComparisonCandidateStatus.RUNNING for item in latest)
    remaining = len(latest) - completed - failed - active
    exact_costs = tuple(item for item in latest if item.incurred_cost is not None)
    currencies = {item.currency for item in latest if item.currency is not None}
    total_cost = (
        sum((cast(Decimal, item.incurred_cost) for item in exact_costs), start=Decimal(0))
        if len(exact_costs) == len(latest) and len(currencies) == 1
        else None
    )
    provider_calls = sum(item.provider_calls for item in latest)
    known_partial_cost = sum(
        (item.known_partial_cost for item in latest),
        start=Decimal(0),
    )
    cost_complete = total_cost is not None or provider_calls == 0
    reasons = {reason for item in latest for reason in item.cost_unknown_reasons}
    if not cost_complete and not reasons:
        reasons.add(COMPARISON_PENDING_COST_REASON)
    return ComparisonProgressSnapshot(
        sequence=sequence,
        status=status,
        total_candidates=len(latest),
        completed_candidates=completed,
        failed_candidates=failed,
        active_candidates=active,
        remaining_candidates=remaining,
        completed_cases=sum(item.completed_cases for item in latest),
        failed_cases=sum(item.failed_cases for item in latest),
        provider_calls=provider_calls,
        incurred_cost=total_cost,
        known_partial_cost=known_partial_cost,
        cost_complete=cost_complete,
        cost_unknown_reasons=tuple(sorted(reasons)),
        currency=(next(iter(currencies)) if len(currencies) == 1 else None),
        recorded_at=recorded_at,
    )


def _derived_suite_status(
    candidates: tuple[ComparisonCandidateHistory, ...],
) -> ComparisonStatus:
    statuses = tuple(item.latest.status for item in candidates)
    if all(item in _TERMINAL_CANDIDATE_STATUSES for item in statuses):
        return ComparisonStatus.COMPLETED
    if any(item is not ComparisonCandidateStatus.QUEUED for item in statuses):
        return ComparisonStatus.RUNNING
    return ComparisonStatus.QUEUED


def _candidate_transition_allowed(
    previous: ComparisonCandidateStatus,
    current: ComparisonCandidateStatus,
) -> bool:
    if previous is ComparisonCandidateStatus.QUEUED:
        return current in {
            ComparisonCandidateStatus.RUNNING,
            ComparisonCandidateStatus.FAILED,
            ComparisonCandidateStatus.INTERRUPTED,
        }
    return previous is ComparisonCandidateStatus.RUNNING and current in {
        ComparisonCandidateStatus.RUNNING,
        ComparisonCandidateStatus.COMPLETED,
        ComparisonCandidateStatus.FAILED,
        ComparisonCandidateStatus.INTERRUPTED,
    }


def _suite_transition_allowed(previous: ComparisonStatus, current: ComparisonStatus) -> bool:
    if previous is ComparisonStatus.QUEUED:
        return current in {
            ComparisonStatus.QUEUED,
            ComparisonStatus.RUNNING,
            ComparisonStatus.FAILED,
            ComparisonStatus.INVALID,
        }
    if previous is ComparisonStatus.RUNNING:
        return current in {
            ComparisonStatus.RUNNING,
            ComparisonStatus.COMPLETED,
            ComparisonStatus.FAILED,
            ComparisonStatus.INVALID,
        }
    if previous is ComparisonStatus.COMPLETED:
        return current is ComparisonStatus.FAILED
    return False


def _axis_identity_names(axis: ExperimentAxis) -> set[str]:
    if axis is ExperimentAxis.GENERATION_MODEL:
        return {"generation.model"}
    if axis is ExperimentAxis.RETRIEVAL_STRATEGY:
        return {
            "retrieval.mode",
            "retrieval.reranking_enabled",
            "model.reranking",
            "provider.reranking",
        }
    return {"cache.behavior"}


def _canonical_identity_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _safe_identity_text(value: object) -> str:
    text = str(value)
    if (
        _SAFE_IDENTITY_TEXT.fullmatch(text) is None
        or _SECRET_VALUE.search(text) is not None
        or _PATH_VALUE.search(text) is not None
        or "\n" in text
        or "\r" in text
    ):
        raise ComparisonDomainError("comparison_identity_value_not_safe")
    return text


def _safe_display_text(value: object) -> str:
    text = str(value)
    if (
        not 1 <= len(text) <= 4096
        or text != text.strip()
        or not text.isprintable()
        or _SECRET_VALUE.search(text) is not None
        or _DISPLAY_PATH_VALUE.search(text) is not None
    ):
        raise ComparisonDomainError("comparison_display_value_not_safe")
    return text


def validate_comparison_plan_safe_values(plan: ExperimentPlan) -> ExperimentPlan:
    """Reject secret-, PII-, and path-shaped values before a plan crosses trust boundaries."""

    plan.verify_hash()
    fixed = plan.fixed_identities
    values: list[object] = [
        plan.plan_id,
        plan.baseline_variant_id,
        fixed.dataset_id,
        fixed.dataset_version,
        fixed.dataset_hash,
        fixed.corpus_id,
        fixed.corpus_version,
        fixed.corpus_hash,
        fixed.case_set_hash,
        plan.pricing.pricing_version,
        plan.pricing.pricing_hash,
        plan.pricing.currency,
        plan.gate_profile.profile_id,
        plan.gate_profile.profile_version,
        plan.gate_profile.profile_hash,
        plan.selection_policy.policy_id,
        plan.selection_policy.policy_version,
    ]
    values.extend(plan.pricing.source_references)
    values.extend(plan.gate_profile.mandatory_gate_ids)
    values.extend(plan.selection_policy.required_gate_ids)
    for identity in fixed.controlled:
        values.extend((identity.name, identity.value))
    for variant in plan.variants:
        values.extend(
            (
                variant.variant_id,
                variant.axis_value,
                variant.configuration_id,
            )
        )
    for rate in plan.pricing.rate_card:
        values.extend((rate.role.value, rate.provider, rate.model, rate.source_reference))
    for criterion in plan.selection_policy.tie_breakers:
        values.extend((criterion.metric, criterion.direction.value))
    for value in values:
        _safe_identity_text(value)
    _safe_display_text(plan.display_name)
    for variant in plan.variants:
        _safe_display_text(variant.display_name)
    return plan


def _validate_safe_code(value: str | None) -> None:
    if value is not None and _SAFE_CODE.fullmatch(value) is None:
        raise ValueError("comparison_safe_error_code_invalid")


def _comparison_manifest_hash(values: Mapping[str, object]) -> str:
    raw_artifacts = cast(Sequence[ArtifactDescriptor | Mapping[str, object]], values["artifacts"])
    created_at = cast(datetime, values["created_at"])
    payload = {
        "schema_version": str(values["schema_version"]),
        "comparison_id": str(values["comparison_id"]),
        "plan_id": str(values["plan_id"]),
        "plan_content_hash": str(values["plan_content_hash"]),
        "candidate_variant_ids": list(cast(Sequence[str], values["candidate_variant_ids"])),
        "artifacts": [
            item.model_dump(mode="json")
            if isinstance(item, ArtifactDescriptor)
            else ArtifactDescriptor.model_validate(item).model_dump(mode="json")
            for item in raw_artifacts
        ],
        "created_at": created_at.isoformat(),
    }
    encoded = canonical_json_value(payload).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


__all__ = [
    "COMPARISON_ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "COMPARISON_CANDIDATE_EVIDENCE_SCHEMA_VERSION",
    "COMPARISON_RERANKER_BENEFIT_PROFILE_ID",
    "COMPARISON_RERANKER_MIN_QUALITY_BENEFIT",
    "COMPARISON_RESULT_SCHEMA_VERSION",
    "COMPARISON_SELECTION_ELIGIBILITY_GATE_ID",
    "COMPARISON_SELECTION_ELIGIBILITY_GATE_VERSION",
    "COMPARISON_SELECTION_MAX_P90_MS",
    "COMPARISON_SHARED_SETUP_AGGREGATE_UNAVAILABLE_REASON",
    "COMPARISON_SHARED_SETUP_EVIDENCE_SCHEMA_VERSION",
    "COMPARISON_SUITE_SCHEMA_VERSION",
    "ComparisonArtifactManifest",
    "ComparisonCandidateEvidence",
    "ComparisonCandidateHistory",
    "ComparisonCandidateReference",
    "ComparisonCandidateResult",
    "ComparisonCandidateSnapshot",
    "ComparisonCandidateStatus",
    "ComparisonCategoryResult",
    "ComparisonCompatibility",
    "ComparisonCompatibilityIssue",
    "ComparisonDomainError",
    "ComparisonEvidenceStatus",
    "ComparisonGateDecision",
    "ComparisonIdentityProjection",
    "ComparisonLogicalAttempt",
    "ComparisonLogicalAttemptStatus",
    "ComparisonMetricResult",
    "ComparisonProgressSnapshot",
    "ComparisonProviderAttempt",
    "ComparisonProviderRoleCount",
    "ComparisonRecommendation",
    "ComparisonRecommendationState",
    "ComparisonResult",
    "ComparisonSharedSetupAttempt",
    "ComparisonSharedSetupEvidence",
    "ComparisonSharedSetupStatus",
    "ComparisonStatus",
    "ComparisonSuite",
    "CompatibilityIssueCode",
    "RerankerCaseEvidence",
    "ResolvedComparisonArtifact",
    "VerifiedCandidateReport",
    "adapt_verified_evaluation_report",
    "aggregate_comparison_result",
    "build_comparison_selection_eligibility_gate",
    "canonical_candidate_evidence",
    "canonical_comparison_manifest",
    "comparison_shared_setup_id",
    "create_comparison_suite",
    "load_verified_candidate_report",
    "project_evaluation_identity",
    "resolve_comparison_artifact",
    "seal_comparison_candidate_evidence",
    "validate_comparison_compatibility",
    "validate_comparison_plan_safe_values",
    "validate_comparison_shared_setup",
]
