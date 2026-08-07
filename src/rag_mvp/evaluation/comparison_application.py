"""Path-free application DTOs for registered controlled comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import PurePosixPath
from typing import Annotated, Literal, Protocol

from pydantic import AwareDatetime, Field

from rag_mvp.domain import (
    ArtifactDescriptor,
    EvaluationRun,
    GateStatus,
    MetricObservation,
    UnavailableValue,
)
from rag_mvp.domain._base import DomainModel, Identifier
from rag_mvp.evaluation.comparison import (
    ComparisonArtifactManifest,
    ComparisonCandidateResult,
    ComparisonMetricResult,
    ComparisonResult,
    ComparisonSharedSetupEvidence,
    ComparisonStatus,
    ComparisonSuite,
    ResolvedComparisonArtifact,
)
from rag_mvp.evaluation.experiment import ExperimentPlan


class ComparisonApplicationError(RuntimeError):
    """Stable content-free comparison application failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ComparisonConflictError(ComparisonApplicationError):
    """The same immutable registered plan is already active."""


class ComparisonCapacityError(ComparisonApplicationError):
    """The one shared paid-job supervisor has no available slot."""


class ComparisonNotFoundError(ComparisonApplicationError):
    """A requested registered comparison plan does not exist."""


class ComparisonUnavailableError(ComparisonApplicationError):
    """Comparison prerequisites or the production service are unavailable."""


class ComparisonValidationError(ComparisonApplicationError):
    """A registered comparison cannot pass its fail-closed preflight."""


class ComparisonPlanVariantEntry(DomainModel):
    variant_id: Identifier
    display_name: str = Field(min_length=1, max_length=255)
    axis_value: str = Field(min_length=1, max_length=4096)
    configuration_id: Identifier | UnavailableValue


class ComparisonPlanCatalogEntry(DomainModel):
    experiment_plan_id: Identifier
    plan_content_hash: Identifier | UnavailableValue
    display_name: str = Field(min_length=1, max_length=255)
    axis: Identifier
    dataset_id: Identifier
    dataset_version: Identifier
    dataset_hash: Identifier
    corpus_id: Identifier
    corpus_version: Identifier
    corpus_hash: Identifier
    case_set_hash: Identifier
    planned_case_count: int = Field(gt=0)
    variants: Annotated[tuple[ComparisonPlanVariantEntry, ...], Field(min_length=2)]
    baseline_variant_id: Identifier
    repeats_per_case: int = Field(gt=0)
    maximum_logical_calls: int = Field(gt=0)
    maximum_provider_calls: int = Field(gt=0)
    cache_policy: Identifier
    cost_estimate_status: Literal["available", "unavailable"]
    cost_estimate: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    cost_cap: Decimal = Field(ge=0, allow_inf_nan=False)
    currency: Identifier
    launchable: bool
    blocking_codes: tuple[Identifier, ...] = ()

    @classmethod
    def from_plan(
        cls,
        plan: ExperimentPlan,
        *,
        launchable: bool,
        blocking_codes: tuple[str, ...] = (),
        conservative_cost_estimate: Decimal | None = None,
    ) -> ComparisonPlanCatalogEntry:
        fixed = plan.fixed_identities
        return cls(
            experiment_plan_id=plan.plan_id,
            plan_content_hash=plan.content_hash,
            display_name=plan.display_name,
            axis=plan.axis.value,
            dataset_id=fixed.dataset_id,
            dataset_version=fixed.dataset_version,
            dataset_hash=fixed.dataset_hash,
            corpus_id=fixed.corpus_id,
            corpus_version=fixed.corpus_version,
            corpus_hash=fixed.corpus_hash,
            case_set_hash=fixed.case_set_hash,
            planned_case_count=fixed.case_count,
            variants=tuple(
                ComparisonPlanVariantEntry(
                    variant_id=item.variant_id,
                    display_name=item.display_name,
                    axis_value=item.axis_value,
                    configuration_id=item.configuration_id,
                )
                for item in plan.variants
            ),
            baseline_variant_id=plan.baseline_variant_id,
            repeats_per_case=plan.repeat_order_policy.repeats_per_case,
            maximum_logical_calls=(
                fixed.case_count * plan.repeat_order_policy.repeats_per_case * len(plan.variants)
            ),
            maximum_provider_calls=plan.maximum_provider_calls,
            cache_policy=plan.cache_policy.value,
            cost_estimate_status=(
                "available" if conservative_cost_estimate is not None else "unavailable"
            ),
            cost_estimate=conservative_cost_estimate,
            cost_cap=plan.maximum_cost,
            currency=plan.pricing.currency,
            launchable=launchable and not blocking_codes,
            blocking_codes=blocking_codes,
        )


class ComparisonRunEntry(DomainModel):
    comparison_id: Identifier
    experiment_plan_id: Identifier
    plan_content_hash: Identifier
    status: ComparisonStatus
    total_candidates: int = Field(gt=0)
    completed_candidates: int = Field(ge=0)
    failed_candidates: int = Field(ge=0)
    active_candidates: int = Field(ge=0)
    remaining_candidates: int = Field(ge=0)
    completed_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    provider_calls: int | UnavailableValue
    incurred_cost: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    known_partial_cost: Decimal = Field(default=Decimal(0), ge=0, allow_inf_nan=False)
    cost_complete: bool = True
    cost_unknown_reasons: tuple[Identifier, ...] = ()
    currency: Identifier | None = None
    safe_error_code: Identifier | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime
    completed_at: AwareDatetime | None = None

    @classmethod
    def from_suite(
        cls,
        suite: ComparisonSuite,
        shared_setup: ComparisonSharedSetupEvidence | None = None,
    ) -> ComparisonRunEntry:
        progress = suite.latest_progress
        setup_calls = (
            shared_setup.provider_call_count
            if shared_setup is not None
            else UnavailableValue(reason="setup-evidence-not-recorded")
        )
        provider_calls: int | UnavailableValue = (
            setup_calls + progress.provider_calls if isinstance(setup_calls, int) else setup_calls
        )
        known_partial_cost = progress.known_partial_cost + (
            shared_setup.known_partial_cost if shared_setup is not None else Decimal(0)
        )
        cost_complete = (
            shared_setup is not None and shared_setup.cost_complete and progress.cost_complete
        )
        exact_cost = known_partial_cost if cost_complete else None
        cost_unknown_reasons = tuple(
            sorted(
                {
                    *(
                        shared_setup.unknown_reasons
                        if shared_setup is not None
                        else ("setup-evidence-not-recorded",)
                    ),
                    *progress.cost_unknown_reasons,
                }
            )
        )
        terminal = suite.status in {
            ComparisonStatus.COMPLETED,
            ComparisonStatus.FAILED,
            ComparisonStatus.INVALID,
        }
        return cls(
            comparison_id=suite.comparison_id,
            experiment_plan_id=suite.plan.plan_id,
            plan_content_hash=suite.plan_content_hash,
            status=suite.status,
            total_candidates=progress.total_candidates,
            completed_candidates=progress.completed_candidates,
            failed_candidates=progress.failed_candidates,
            active_candidates=progress.active_candidates,
            remaining_candidates=progress.remaining_candidates,
            completed_cases=progress.completed_cases,
            failed_cases=progress.failed_cases,
            provider_calls=provider_calls,
            incurred_cost=exact_cost,
            known_partial_cost=known_partial_cost,
            cost_complete=cost_complete,
            cost_unknown_reasons=cost_unknown_reasons,
            currency=suite.plan.pricing.currency,
            safe_error_code=suite.safe_error_code,
            created_at=suite.created_at,
            updated_at=suite.updated_at,
            completed_at=suite.updated_at if terminal else None,
        )


class ComparisonControlledDimension(DomainModel):
    name: Identifier
    value: str = Field(min_length=1, max_length=4096)


class ComparisonMetricSummary(DomainModel):
    metric_id: Identifier
    value: float | UnavailableValue
    unit: Identifier
    numerator: float | UnavailableValue
    denominator: int | UnavailableValue
    state: Identifier
    gate_state: Identifier
    baseline_delta: float | UnavailableValue

    @classmethod
    def from_result(cls, metric: ComparisonMetricResult) -> ComparisonMetricSummary:
        return cls(
            metric_id=metric.metric_id,
            value=metric.value,
            unit=metric.unit,
            numerator=metric.numerator,
            denominator=metric.denominator,
            state=metric.status.value,
            gate_state=metric.gate_status.value,
            baseline_delta=metric.baseline_delta,
        )

    @classmethod
    def from_observation(cls, metric: MetricObservation) -> ComparisonMetricSummary:
        unavailable = UnavailableValue(reason="comparison-level-metric")
        return cls(
            metric_id=metric.metric_id,
            value=metric.value,
            unit=metric.unit,
            numerator=metric.numerator,
            denominator=metric.denominator,
            state=metric.status.value,
            gate_state=metric.status.value,
            baseline_delta=unavailable,
        )


class ComparisonGateSummary(DomainModel):
    gate_id: Identifier
    status: Identifier
    required_for_selection: bool
    reason_codes: tuple[Identifier, ...] = ()


class ComparisonCandidateSummary(DomainModel):
    candidate_id: Identifier
    display_name: str = Field(min_length=1, max_length=255)
    axis_value: str = Field(min_length=1, max_length=4096)
    evaluation_run_id: Identifier
    configuration_id: Identifier
    status: Identifier
    evidence_status: Identifier
    is_baseline: bool
    safe_error_code: Identifier | None = None
    failed_case_count: int = Field(ge=0)
    provider_call_count: int = Field(ge=0)
    known_partial_cost: Decimal = Field(default=Decimal(0), ge=0, allow_inf_nan=False)
    total_cost: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    cost_complete: bool = True
    cost_unknown_reasons: tuple[Identifier, ...] = ()
    currency: Identifier | None = None
    metrics: tuple[ComparisonMetricSummary, ...] = ()
    gates: tuple[ComparisonGateSummary, ...] = ()


class ComparisonCategorySummary(DomainModel):
    candidate_id: Identifier
    category_id: Identifier
    case_count: int = Field(gt=0)
    metrics: tuple[ComparisonMetricSummary, ...]


class ComparisonRecommendationSummary(DomainModel):
    state: Identifier
    selected_candidate_id: Identifier | None = None
    rationale_codes: Annotated[tuple[Identifier, ...], Field(min_length=1)]


class ComparisonSharedSetupSummary(DomainModel):
    """Privacy-safe comparison-level index setup accounting."""

    status: Literal["reused", "completed", "failed", "unavailable"]
    safe_error_code: Identifier | None = None
    provider_call_count: int | UnavailableValue
    known_partial_cost: Decimal | UnavailableValue
    total_cost: Decimal | UnavailableValue
    currency: Identifier | UnavailableValue
    provider_calls_complete: bool
    cost_complete: bool
    unknown_reasons: tuple[Identifier, ...]

    @classmethod
    def from_evidence(
        cls,
        evidence: ComparisonSharedSetupEvidence | None,
    ) -> ComparisonSharedSetupSummary:
        if evidence is None:
            unavailable = UnavailableValue(reason="setup-evidence-not-recorded")
            return cls(
                status="unavailable",
                provider_call_count=unavailable,
                known_partial_cost=unavailable,
                total_cost=unavailable,
                currency=unavailable,
                provider_calls_complete=False,
                cost_complete=False,
                unknown_reasons=("setup-evidence-not-recorded",),
            )
        return cls(
            status=evidence.status.value,
            safe_error_code=evidence.safe_error_code,
            provider_call_count=evidence.provider_call_count,
            known_partial_cost=evidence.known_partial_cost,
            total_cost=(
                evidence.total_cost
                if evidence.total_cost is not None
                else UnavailableValue(reason="setup-cost-incomplete")
            ),
            currency=evidence.currency,
            provider_calls_complete=evidence.provider_calls_complete,
            cost_complete=evidence.cost_complete,
            unknown_reasons=evidence.unknown_reasons,
        )


class ComparisonSummary(DomainModel):
    comparison_id: Identifier
    experiment_plan_id: Identifier
    status: ComparisonStatus
    evidence_status: Literal["available", "incomplete", "unavailable"]
    gate_status: Literal["passed", "failed", "unavailable"]
    compatibility_state: Literal["compatible", "incompatible", "unavailable"]
    compatibility_issues: tuple[Identifier, ...] = ()
    controlled_dimensions: tuple[ComparisonControlledDimension, ...] = ()
    candidates: tuple[ComparisonCandidateSummary, ...]
    categories: tuple[ComparisonCategorySummary, ...] = ()
    comparison_metrics: tuple[ComparisonMetricSummary, ...] = ()
    recommendation: ComparisonRecommendationSummary
    shared_setup: ComparisonSharedSetupSummary
    provider_call_count: int | UnavailableValue
    known_partial_cost: Decimal | UnavailableValue
    total_cost: Decimal | UnavailableValue
    cost_complete: bool
    cost_unknown_reasons: tuple[Identifier, ...]
    currency: Identifier | UnavailableValue
    completed_at: AwareDatetime | None = None

    @classmethod
    def from_evidence(
        cls,
        suite: ComparisonSuite,
        result: ComparisonResult | None,
        shared_setup: ComparisonSharedSetupEvidence | None = None,
    ) -> ComparisonSummary:
        if result is not None and (
            result.comparison_id != suite.comparison_id
            or result.plan_id != suite.plan.plan_id
            or result.plan_content_hash != suite.plan_content_hash
            or tuple(item.reference for item in result.candidates)
            != tuple(item.reference for item in suite.candidates)
        ):
            raise ComparisonApplicationError("comparison_result_identity_mismatch")
        if result is not None:
            if shared_setup is not None and shared_setup != result.shared_setup:
                raise ComparisonApplicationError("comparison_setup_result_mismatch")
            shared_setup = result.shared_setup
        setup_summary = ComparisonSharedSetupSummary.from_evidence(shared_setup)
        if result is None:
            progressed = any(
                history.latest.completed_cases
                or history.latest.failed_cases
                or history.latest.provider_calls
                for history in suite.candidates
            )
            evidence_status: Literal["available", "incomplete", "unavailable"] = (
                "incomplete" if progressed else "unavailable"
            )
            variants = {item.variant_id: item for item in suite.plan.variants}
            candidates = tuple(
                ComparisonCandidateSummary(
                    candidate_id=history.reference.variant_id,
                    display_name=variants[history.reference.variant_id].display_name,
                    axis_value=history.reference.axis_value,
                    evaluation_run_id=history.reference.evaluation_run_id,
                    configuration_id=history.reference.configuration_id,
                    status=history.latest.status.value,
                    evidence_status="unavailable",
                    is_baseline=(history.reference.variant_id == suite.plan.baseline_variant_id),
                    safe_error_code=history.latest.safe_error_code,
                    failed_case_count=history.latest.failed_cases,
                    provider_call_count=history.latest.provider_calls,
                    known_partial_cost=history.latest.known_partial_cost,
                    total_cost=(
                        history.latest.incurred_cost
                        if history.latest.incurred_cost is not None
                        else history.latest.known_partial_cost
                        if history.latest.cost_complete
                        else None
                    ),
                    cost_complete=history.latest.cost_complete,
                    cost_unknown_reasons=history.latest.cost_unknown_reasons,
                    currency=(
                        history.latest.currency
                        or (suite.plan.pricing.currency if history.latest.cost_complete else None)
                    ),
                )
                for history in suite.candidates
            )
            return cls(
                comparison_id=suite.comparison_id,
                experiment_plan_id=suite.plan.plan_id,
                status=suite.status,
                evidence_status=evidence_status,
                gate_status="unavailable",
                compatibility_state="unavailable",
                candidates=candidates,
                recommendation=ComparisonRecommendationSummary(
                    state="unavailable",
                    rationale_codes=("recommendation-not-recorded",),
                ),
                shared_setup=setup_summary,
                provider_call_count=(
                    shared_setup.provider_call_count + suite.latest_progress.provider_calls
                    if shared_setup is not None
                    and isinstance(shared_setup.provider_call_count, int)
                    else UnavailableValue(reason="comparison-provider-calls-incomplete")
                ),
                known_partial_cost=(
                    shared_setup.known_partial_cost + suite.latest_progress.known_partial_cost
                    if shared_setup is not None
                    else suite.latest_progress.known_partial_cost
                ),
                total_cost=(
                    shared_setup.total_cost + suite.latest_progress.known_partial_cost
                    if shared_setup is not None
                    and shared_setup.cost_complete
                    and shared_setup.total_cost is not None
                    and suite.latest_progress.cost_complete
                    else UnavailableValue(reason="comparison-cost-incomplete")
                ),
                cost_complete=(
                    shared_setup is not None
                    and shared_setup.cost_complete
                    and suite.latest_progress.cost_complete
                ),
                cost_unknown_reasons=tuple(
                    sorted(
                        {
                            *(
                                shared_setup.unknown_reasons
                                if shared_setup is not None
                                else ("setup-evidence-not-recorded",)
                            ),
                            *suite.latest_progress.cost_unknown_reasons,
                        }
                    )
                ),
                currency=suite.plan.pricing.currency,
            )
        variants = {item.variant_id: item for item in result.plan.variants}
        cache_metrics = tuple(
            ComparisonMetricSummary.from_observation(item) for item in result.cache_observations
        )
        candidates = tuple(
            _candidate_summary(
                item,
                display_name=variants[item.reference.variant_id].display_name,
                baseline_variant_id=result.baseline_variant_id,
                comparison_metrics=(cache_metrics if item.reference.axis_value == "warm" else ()),
                required_gate_ids=result.plan.selection_policy.required_gate_ids,
            )
            for item in result.candidates
        )
        gate_status = _aggregate_gate_status(result)
        compatibility_state: Literal["compatible", "incompatible"] = (
            "compatible" if result.compatibility.compatible else "incompatible"
        )
        return cls(
            comparison_id=suite.comparison_id,
            experiment_plan_id=result.plan_id,
            status=suite.status,
            evidence_status=(
                "available" if suite.status is ComparisonStatus.COMPLETED else "incomplete"
            ),
            gate_status=(
                gate_status if suite.status is ComparisonStatus.COMPLETED else "unavailable"
            ),
            compatibility_state=compatibility_state,
            compatibility_issues=tuple(item.code.value for item in result.compatibility.issues),
            controlled_dimensions=tuple(
                ComparisonControlledDimension(name=item.name, value=item.value)
                for item in result.compatibility.controlled_dimensions
            ),
            candidates=candidates,
            categories=tuple(
                ComparisonCategorySummary(
                    candidate_id=item.candidate_variant_id,
                    category_id=item.category_id,
                    case_count=item.case_count,
                    metrics=tuple(
                        ComparisonMetricSummary.from_result(metric) for metric in item.metrics
                    ),
                )
                for item in result.category_results
            ),
            comparison_metrics=cache_metrics,
            recommendation=ComparisonRecommendationSummary(
                state=result.recommendation.state.value,
                selected_candidate_id=result.recommendation.selected_variant_id,
                rationale_codes=result.recommendation.rationale_codes,
            ),
            shared_setup=setup_summary,
            provider_call_count=result.provider_call_count,
            known_partial_cost=result.known_partial_cost,
            total_cost=(
                result.total_cost
                if result.total_cost is not None
                else UnavailableValue(reason="comparison-cost-incomplete")
            ),
            cost_complete=result.cost_complete,
            cost_unknown_reasons=result.cost_unknown_reasons,
            currency=result.currency or result.plan.pricing.currency,
            completed_at=result.completed_at,
        )


class ComparisonArtifactDescriptor(DomainModel):
    artifact_id: Identifier
    schema_version: Identifier
    format: Identifier
    media_type: Identifier
    sha256_digest: Identifier
    byte_size: int = Field(ge=0)
    created_at: AwareDatetime

    @classmethod
    def from_descriptor(cls, descriptor: ArtifactDescriptor) -> ComparisonArtifactDescriptor:
        return cls(
            artifact_id=descriptor.artifact_id,
            schema_version=descriptor.schema_version,
            format=descriptor.format,
            media_type=descriptor.media_type,
            sha256_digest=descriptor.sha256_digest,
            byte_size=descriptor.byte_size,
            created_at=descriptor.created_at,
        )


class ComparisonArtifactManifestView(DomainModel):
    comparison_id: Identifier
    experiment_plan_id: Identifier
    plan_content_hash: Identifier
    manifest_content_hash: Identifier
    artifacts: tuple[ComparisonArtifactDescriptor, ...]
    created_at: AwareDatetime

    @classmethod
    def from_manifest(
        cls,
        manifest: ComparisonArtifactManifest,
    ) -> ComparisonArtifactManifestView:
        return cls(
            comparison_id=manifest.comparison_id,
            experiment_plan_id=manifest.plan_id,
            plan_content_hash=manifest.plan_content_hash,
            manifest_content_hash=manifest.manifest_content_hash,
            artifacts=tuple(
                ComparisonArtifactDescriptor.from_descriptor(item) for item in manifest.artifacts
            ),
            created_at=manifest.created_at,
        )


@dataclass(frozen=True, slots=True)
class ResolvedComparisonDownload:
    artifact_id: str
    content: bytes
    media_type: str
    filename: str

    @classmethod
    def from_resolved(
        cls,
        resolved: ResolvedComparisonArtifact,
    ) -> ResolvedComparisonDownload:
        filename = PurePosixPath(resolved.descriptor.relative_path).name
        return cls(
            artifact_id=resolved.descriptor.artifact_id,
            content=resolved.content,
            media_type=resolved.descriptor.media_type,
            filename=filename,
        )

    def __post_init__(self) -> None:
        if (
            not self.content
            or not self.filename
            or any(character in self.filename for character in '\\/\r\n"')
        ):
            raise ValueError("comparison_download_invalid")


@dataclass(frozen=True, slots=True)
class PreparedComparisonLaunch:
    """All immutable rows and executor context prepared before shared admission."""

    suite: ComparisonSuite
    evaluation_runs: tuple[EvaluationRun, ...]
    execution_context: object

    def __post_init__(self) -> None:
        expected = {item.reference.evaluation_run_id for item in self.suite.candidates}
        if (
            len(self.evaluation_runs) != len(expected)
            or {item.run_id for item in self.evaluation_runs} != expected
        ):
            raise ValueError("comparison_prepared_run_set_mismatch")


class ComparisonLaunchCatalog(Protocol):
    def list(self) -> Sequence[ComparisonPlanCatalogEntry]: ...

    def prepare(self, comparison_id: str, plan_id: str) -> PreparedComparisonLaunch: ...


class ComparisonRunStore(Protocol):
    def create(
        self,
        suite: ComparisonSuite,
        evaluation_runs: Sequence[EvaluationRun],
    ) -> None: ...

    def append(self, suite: ComparisonSuite) -> None: ...

    def get(self, comparison_id: str) -> ComparisonSuite | None: ...

    def list(self) -> Sequence[ComparisonSuite]: ...

    def get_result(self, comparison_id: str) -> ComparisonResult | None: ...

    def get_shared_setup(
        self,
        comparison_id: str,
    ) -> ComparisonSharedSetupEvidence | None: ...


class ComparisonJobExecutor(Protocol):
    async def execute(self, launch: PreparedComparisonLaunch) -> None: ...


class ComparisonArtifactStore(Protocol):
    def manifest(self, comparison_id: str) -> ComparisonArtifactManifest | None: ...

    def resolve(
        self,
        comparison_id: str,
        artifact_id: str,
    ) -> ResolvedComparisonArtifact | None: ...


def _candidate_summary(
    candidate: ComparisonCandidateResult,
    *,
    display_name: str,
    baseline_variant_id: str,
    comparison_metrics: tuple[ComparisonMetricSummary, ...],
    required_gate_ids: tuple[str, ...],
) -> ComparisonCandidateSummary:
    required = set(required_gate_ids)
    return ComparisonCandidateSummary(
        candidate_id=candidate.reference.variant_id,
        display_name=display_name,
        axis_value=candidate.reference.axis_value,
        evaluation_run_id=candidate.reference.evaluation_run_id,
        configuration_id=candidate.reference.configuration_id,
        status=candidate.status.value,
        evidence_status=candidate.evidence_status.value,
        is_baseline=candidate.reference.variant_id == baseline_variant_id,
        safe_error_code=candidate.safe_error_code,
        failed_case_count=candidate.failed_case_count,
        provider_call_count=candidate.provider_call_count,
        known_partial_cost=candidate.known_partial_cost,
        total_cost=candidate.total_cost,
        cost_complete=candidate.cost_complete,
        cost_unknown_reasons=candidate.cost_unknown_reasons,
        currency=candidate.currency,
        metrics=(
            *(ComparisonMetricSummary.from_result(item) for item in candidate.metrics),
            *comparison_metrics,
        ),
        gates=tuple(
            ComparisonGateSummary(
                gate_id=item.gate_id,
                status=item.status.value,
                required_for_selection=item.gate_id in required,
                reason_codes=item.reason_codes,
            )
            for item in candidate.gates
        ),
    )


def _aggregate_gate_status(
    result: ComparisonResult,
) -> Literal["passed", "failed", "unavailable"]:
    if result.known_partial_cost > result.plan.maximum_cost:
        return "failed"
    if result.recommendation.state.value == "recommended":
        return "passed"
    required = set(result.plan.selection_policy.required_gate_ids)
    statuses = tuple(item.status for item in result.gates if item.gate_id in required)
    if not statuses or any(item is GateStatus.UNAVAILABLE for item in statuses):
        return "unavailable"
    if any(item is GateStatus.FAILED for item in statuses):
        return "failed"
    return "passed"


__all__ = [
    "ComparisonApplicationError",
    "ComparisonArtifactDescriptor",
    "ComparisonArtifactManifestView",
    "ComparisonArtifactStore",
    "ComparisonCandidateSummary",
    "ComparisonCapacityError",
    "ComparisonCategorySummary",
    "ComparisonConflictError",
    "ComparisonControlledDimension",
    "ComparisonGateSummary",
    "ComparisonJobExecutor",
    "ComparisonLaunchCatalog",
    "ComparisonMetricSummary",
    "ComparisonNotFoundError",
    "ComparisonPlanCatalogEntry",
    "ComparisonPlanVariantEntry",
    "ComparisonRecommendationSummary",
    "ComparisonRunEntry",
    "ComparisonRunStore",
    "ComparisonSharedSetupSummary",
    "ComparisonSummary",
    "ComparisonUnavailableError",
    "ComparisonValidationError",
    "PreparedComparisonLaunch",
    "ResolvedComparisonDownload",
]
