"""Registered, materialized controlled-comparison plans for production execution."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol, Self, cast

from pydantic import Field, field_validator, model_validator

from rag_mvp.config.settings import Settings
from rag_mvp.domain._base import DomainModel, Identifier
from rag_mvp.domain.retrieval import CachePolicy
from rag_mvp.evaluation.comparison import (
    COMPARISON_RERANKER_BENEFIT_PROFILE_ID,
    COMPARISON_RERANKER_MIN_QUALITY_BENEFIT,
    COMPARISON_SELECTION_ELIGIBILITY_GATE_ID,
    COMPARISON_SELECTION_ELIGIBILITY_GATE_VERSION,
    COMPARISON_SELECTION_MAX_P90_MS,
    ComparisonDomainError,
    ComparisonIdentityProjection,
    project_evaluation_identity,
    validate_comparison_plan_safe_values,
)
from rag_mvp.evaluation.comparison_preflight import (
    ComparisonPreflightError,
    ComparisonWorkPreflight,
    minimum_cache_experiment_ttl_seconds,
    preflight_comparison_work,
    validate_registered_comparison_dataset,
)
from rag_mvp.evaluation.comparison_schedule import (
    ComparisonExecutionSchedule,
    ComparisonScheduleError,
    build_comparison_schedule,
    cache_eligible_case_ids,
    materialize_variant_cases,
)
from rag_mvp.evaluation.dataset import EvaluationDataset
from rag_mvp.evaluation.experiment import (
    DeterministicSelectionPolicy,
    ExperimentAxis,
    ExperimentFixedIdentities,
    ExperimentGateProfile,
    ExperimentOrderPolicy,
    ExperimentPlan,
    ExperimentPricingProvenance,
    ExperimentPricingRate,
    ExperimentVariant,
    FinalTieBreak,
    FixedIdentity,
    PricingRole,
    RepeatOrderPolicy,
    SelectionCriterion,
    SelectionDirection,
)
from rag_mvp.evaluation.json_report import canonical_json_value
from rag_mvp.evaluation.plan import EvaluationPlanError, build_evaluation_plan
from rag_mvp.evaluation.report_builder import case_ids_content_hash
from rag_mvp.evaluation.runner import (
    EvaluationRunIdentity,
    EvaluationRunPlan,
)
from rag_mvp.performance.pricing import (
    OPENAI_COMPARISON_PRICING_PROVIDER,
    OPENAI_COMPARISON_PRICING_VERSION,
    PerformancePricingEvidence,
    PricingPreflightError,
    preflight_openai_comparison_pricing,
)

REGISTERED_GENERATION_PLAN_ID = "registered-generation-model-v1"
REGISTERED_RETRIEVAL_PLAN_ID = "registered-retrieval-strategy-v1"
REGISTERED_CACHE_PLAN_ID = "registered-cache-behavior-v1"
REGISTERED_COMPARISON_PLAN_IDS = (
    REGISTERED_GENERATION_PLAN_ID,
    REGISTERED_RETRIEVAL_PLAN_ID,
    REGISTERED_CACHE_PLAN_ID,
)

_GENERATION_MODELS = ("gpt-4.1-mini", "gpt-5.4")
_GENERATION_VARIANT_IDS: Mapping[str, str] = MappingProxyType(
    {
        "gpt-4.1-mini": "generation-gpt-4-1-mini",
        "gpt-5.4": "generation-gpt-5-4",
    }
)
_RETRIEVAL_MODES = ("dense", "hybrid", "hybrid-rerank")
_CACHE_BEHAVIORS = ("cold", "warm")
_RERANKING_MODEL = "gpt-4.1-mini"
_EMBEDDING_MODEL = "text-embedding-3-small"
_DEFAULT_PRICING_PATH = (
    Path(__file__).resolve().parents[3]
    / "evaluations"
    / "pricing"
    / "openai-comparison-standard-2026-08-07-v1.json"
)

_PLAN_LABELS: Mapping[str, str] = MappingProxyType(
    {
        REGISTERED_GENERATION_PLAN_ID: "Generation model comparison / 生成模型对比",
        REGISTERED_RETRIEVAL_PLAN_ID: "Retrieval strategy comparison / 检索策略对比",
        REGISTERED_CACHE_PLAN_ID: "Cold/warm cache comparison / 冷暖缓存对比",
    }
)
_PLAN_AXES: Mapping[str, ExperimentAxis] = MappingProxyType(
    {
        REGISTERED_GENERATION_PLAN_ID: ExperimentAxis.GENERATION_MODEL,
        REGISTERED_RETRIEVAL_PLAN_ID: ExperimentAxis.RETRIEVAL_STRATEGY,
        REGISTERED_CACHE_PLAN_ID: ExperimentAxis.CACHE_BEHAVIOR,
    }
)
_PLAN_REPEATS: Mapping[str, int] = MappingProxyType(
    {
        REGISTERED_GENERATION_PLAN_ID: 2,
        REGISTERED_RETRIEVAL_PLAN_ID: 2,
        REGISTERED_CACHE_PLAN_ID: 1,
    }
)
_PLAN_CALL_CAPS: Mapping[str, int] = MappingProxyType(
    {
        REGISTERED_GENERATION_PLAN_ID: 2_000,
        REGISTERED_RETRIEVAL_PLAN_ID: 4_000,
        REGISTERED_CACHE_PLAN_ID: 2_000,
    }
)
_PLAN_COST_CAPS: Mapping[str, Decimal] = MappingProxyType(
    {
        REGISTERED_GENERATION_PLAN_ID: Decimal("25"),
        REGISTERED_RETRIEVAL_PLAN_ID: Decimal("40"),
        REGISTERED_CACHE_PLAN_ID: Decimal("20"),
    }
)


class RegisteredComparisonPlanError(RuntimeError):
    """Stable, privacy-safe registry/materialization failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _SelectionRecordLike(Protocol):
    axis: object
    comparison_id: object
    plan_id: object
    plan_content_hash: object
    result_content_hash: object
    selected_variant_id: object
    selected_axis_value: object
    selected_configuration_id: object
    selected_evaluation_run_id: object
    upstream_identities: object


class _SelectionIdentityLike(Protocol):
    name: object
    value: object


class UpstreamComparisonSelection(DomainModel):
    """Path-free semantic selection accepted by downstream registered plans."""

    axis: Literal["generation-model", "retrieval-strategy"]
    comparison_id: Identifier
    plan_id: Identifier
    plan_content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    result_content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    selected_variant_id: Identifier
    selected_axis_value: str = Field(min_length=1, max_length=4096)
    selected_configuration_id: Identifier
    selected_evaluation_run_id: Identifier
    upstream_identities: tuple[FixedIdentity, ...] = ()

    @field_validator("upstream_identities")
    @classmethod
    def validate_upstream_identities(
        cls,
        values: tuple[FixedIdentity, ...],
    ) -> tuple[FixedIdentity, ...]:
        names = tuple(item.name for item in values)
        if len(names) != len(set(names)) or any(not name.startswith("upstream.") for name in names):
            raise ValueError("comparison_upstream_selection_identity_invalid")
        return tuple(sorted(values, key=lambda item: item.name))

    @classmethod
    def from_record(cls, record: object) -> UpstreamComparisonSelection:
        """Adapt the storage selection DTO without importing storage into evaluation."""

        try:
            value = cast(_SelectionRecordLike, record)
            return cls(
                axis=str(value.axis),
                comparison_id=str(value.comparison_id),
                plan_id=str(value.plan_id),
                plan_content_hash=str(value.plan_content_hash),
                result_content_hash=str(value.result_content_hash),
                selected_variant_id=str(value.selected_variant_id),
                selected_axis_value=str(value.selected_axis_value),
                selected_configuration_id=str(value.selected_configuration_id),
                selected_evaluation_run_id=str(value.selected_evaluation_run_id),
                upstream_identities=tuple(
                    FixedIdentity(name=str(item.name), value=str(item.value))
                    for item in cast(Sequence[_SelectionIdentityLike], value.upstream_identities)
                ),
            )
        except (AttributeError, TypeError, ValueError):
            raise RegisteredComparisonPlanError("comparison-upstream-selection-invalid") from None


class UpstreamSelections(DomainModel):
    generation_model: UpstreamComparisonSelection | None = None
    retrieval_strategy: UpstreamComparisonSelection | None = None

    @model_validator(mode="after")
    def validate_axes(self) -> Self:
        if self.generation_model is not None and self.generation_model.axis != "generation-model":
            raise ValueError("comparison_generation_selection_axis_invalid")
        if (
            self.retrieval_strategy is not None
            and self.retrieval_strategy.axis != "retrieval-strategy"
        ):
            raise ValueError("comparison_retrieval_selection_axis_invalid")
        return self


@dataclass(frozen=True, slots=True, kw_only=True)
class ComparisonPlanCatalogContext:
    dataset: EvaluationDataset
    settings: Settings
    upstream_selections: UpstreamSelections = field(default_factory=UpstreamSelections)


@dataclass(frozen=True, slots=True, kw_only=True)
class ComparisonPlanMaterializationContext(ComparisonPlanCatalogContext):
    comparison_id: str
    candidate_run_ids: Mapping[str, str]


class RegisteredComparisonCandidateSummary(DomainModel):
    variant_id: Identifier
    display_name: str = Field(min_length=1, max_length=255)
    axis_value: str = Field(min_length=1, max_length=4096)


class RegisteredComparisonPlanSummary(DomainModel):
    schema_version: Literal["registered-comparison-plan-summary-v1"] = (
        "registered-comparison-plan-summary-v1"
    )
    plan_id: Identifier
    display_name: str = Field(min_length=1, max_length=255)
    axis: ExperimentAxis
    dataset_id: Identifier
    dataset_version: Identifier
    dataset_hash: str
    case_set_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    candidates: tuple[RegisteredComparisonCandidateSummary, ...]
    baseline_variant_id: Identifier
    case_count: int = Field(ge=1)
    repeats_per_case: int = Field(ge=1)
    candidate_count: int = Field(ge=2)
    planned_logical_attempts: int = Field(ge=1)
    maximum_provider_calls: int = Field(ge=1)
    cache_policy: CachePolicy
    cache_eligible_case_count: int = Field(ge=0)
    cache_max_entries: int | None = Field(default=None, ge=1)
    cache_ttl_seconds: float | None = Field(default=None, gt=0)
    pricing_version: Identifier
    pricing_currency: str = Field(pattern=r"^[A-Z]{3}$")
    conservative_cost_estimate: Decimal | None = Field(default=None, ge=0)
    maximum_cost: Decimal = Field(ge=0)
    selection_policy_id: Identifier
    plan_content_hash: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    launchable: bool
    blocking_codes: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if self.candidate_count != len(self.candidates):
            raise ValueError("comparison_plan_summary_candidate_count_mismatch")
        if self.planned_logical_attempts != (
            self.case_count * self.repeats_per_case * self.candidate_count
        ):
            raise ValueError("comparison_plan_summary_logical_count_mismatch")
        if self.launchable == bool(self.blocking_codes):
            raise ValueError("comparison_plan_summary_launch_state_mismatch")
        if self.launchable != (
            self.plan_content_hash is not None and self.conservative_cost_estimate is not None
        ):
            raise ValueError("comparison_plan_summary_estimate_state_mismatch")
        if (
            self.conservative_cost_estimate is not None
            and self.conservative_cost_estimate > self.maximum_cost
        ):
            raise ValueError("comparison_plan_summary_cost_cap_exceeded")
        return self


@dataclass(frozen=True, slots=True)
class MaterializedComparisonCandidate:
    variant_id: str
    settings: Settings
    evaluation_plan: EvaluationRunPlan
    identity_projection: ComparisonIdentityProjection


@dataclass(frozen=True, slots=True)
class MaterializedComparisonPlan:
    plan: ExperimentPlan
    candidates: tuple[MaterializedComparisonCandidate, ...]
    schedule: ComparisonExecutionSchedule
    selected_case_ids: tuple[str, ...]
    preflight: ComparisonWorkPreflight

    @property
    def candidate_plans(self) -> Mapping[str, EvaluationRunPlan]:
        return MappingProxyType({item.variant_id: item.evaluation_plan for item in self.candidates})


class RegisteredComparisonPlanRegistry:
    """Resolve only the three audited comparison definitions at explicit start time."""

    def __init__(self, pricing_path: str | Path = _DEFAULT_PRICING_PATH) -> None:
        self._pricing = _load_pricing(pricing_path)
        self._experiment_pricing = _experiment_pricing(self._pricing)

    def list(
        self,
        context: ComparisonPlanCatalogContext,
    ) -> tuple[RegisteredComparisonPlanSummary, ...]:
        return tuple(self._summary(plan_id, context) for plan_id in REGISTERED_COMPARISON_PLAN_IDS)

    def resolve(
        self,
        plan_id: str,
        context: ComparisonPlanMaterializationContext,
    ) -> MaterializedComparisonPlan:
        if plan_id not in REGISTERED_COMPARISON_PLAN_IDS:
            raise RegisteredComparisonPlanError("comparison-plan-not-registered")
        blocking = self._blocking_codes(plan_id, context)
        if blocking:
            raise RegisteredComparisonPlanError(blocking[0])
        return self._materialize(plan_id, context)

    def _summary(
        self,
        plan_id: str,
        context: ComparisonPlanCatalogContext,
    ) -> RegisteredComparisonPlanSummary:
        blocking = self._blocking_codes(plan_id, context)
        specs = _variant_specs(plan_id, context)
        selected_cases = _selected_case_ids(plan_id, context.dataset)
        case_set_hash = case_ids_content_hash(selected_cases)
        cache_settings: Settings | None = None
        if plan_id == REGISTERED_CACHE_PLAN_ID and not blocking:
            try:
                cache_settings = _candidate_settings(plan_id, specs[0].axis_value, context)
            except RegisteredComparisonPlanError as error:
                blocking = (error.code,)
        content_hash: str | None = None
        conservative_cost_estimate: Decimal | None = None
        if not blocking:
            preview = ComparisonPlanMaterializationContext(
                dataset=context.dataset,
                settings=context.settings,
                upstream_selections=context.upstream_selections,
                comparison_id=f"preview-{plan_id}",
                candidate_run_ids={
                    item.variant_id: f"preview-candidate-{index + 1}"
                    for index, item in enumerate(specs)
                },
            )
            try:
                materialized = self._materialize(plan_id, preview)
                content_hash = materialized.plan.content_hash
                conservative_cost_estimate = materialized.preflight.snapshot.reserved_cost
            except RegisteredComparisonPlanError as error:
                blocking = (error.code,)
        return RegisteredComparisonPlanSummary(
            plan_id=plan_id,
            display_name=_PLAN_LABELS[plan_id],
            axis=_PLAN_AXES[plan_id],
            dataset_id=context.dataset.manifest.dataset_id,
            dataset_version=context.dataset.manifest.version,
            dataset_hash=context.dataset.manifest.content_hash,
            case_set_hash=case_set_hash,
            candidates=specs,
            baseline_variant_id=specs[0].variant_id,
            case_count=len(selected_cases),
            repeats_per_case=_PLAN_REPEATS[plan_id],
            candidate_count=len(specs),
            planned_logical_attempts=(len(selected_cases) * _PLAN_REPEATS[plan_id] * len(specs)),
            maximum_provider_calls=_PLAN_CALL_CAPS[plan_id],
            cache_policy=(
                CachePolicy.USE if plan_id == REGISTERED_CACHE_PLAN_ID else CachePolicy.BYPASS
            ),
            cache_eligible_case_count=(
                len(selected_cases) if plan_id == REGISTERED_CACHE_PLAN_ID else 0
            ),
            cache_max_entries=(
                cache_settings.retrieval_cache_max_entries if cache_settings is not None else None
            ),
            cache_ttl_seconds=(
                cache_settings.retrieval_cache_ttl_seconds if cache_settings is not None else None
            ),
            pricing_version=self._experiment_pricing.pricing_version,
            pricing_currency=self._experiment_pricing.currency,
            conservative_cost_estimate=conservative_cost_estimate,
            maximum_cost=_PLAN_COST_CAPS[plan_id],
            selection_policy_id=_selection_policy(plan_id).policy_id,
            plan_content_hash=content_hash,
            launchable=not blocking,
            blocking_codes=blocking,
        )

    def _blocking_codes(
        self,
        plan_id: str,
        context: ComparisonPlanCatalogContext,
    ) -> tuple[str, ...]:
        if plan_id not in REGISTERED_COMPARISON_PLAN_IDS:
            return ("comparison-plan-not-registered",)
        values: list[str] = []
        try:
            validate_registered_comparison_dataset(context.dataset)
        except ComparisonPreflightError as error:
            values.append(error.code)
        if context.settings.provider_backend != "openai":
            values.append("comparison-provider-backend-unsupported")
        values.extend(f"comparison-{item}" for item in context.settings.provider_readiness_errors())
        if context.settings.embedding_model != _EMBEDDING_MODEL:
            values.append("comparison-embedding-model-not-priced")
        if context.settings.generation_model not in _GENERATION_MODELS:
            values.append("comparison-generation-baseline-not-registered")
        upstream = context.upstream_selections
        if plan_id != REGISTERED_GENERATION_PLAN_ID and upstream.generation_model is None:
            values.append("comparison-generation-selection-required")
        if plan_id == REGISTERED_CACHE_PLAN_ID and upstream.retrieval_strategy is None:
            values.append("comparison-retrieval-selection-required")
        if upstream.generation_model is not None and (
            upstream.generation_model.selected_axis_value not in _GENERATION_MODELS
        ):
            values.append("comparison-generation-selection-not-registered")
        if upstream.generation_model is not None and (
            upstream.generation_model.plan_id != REGISTERED_GENERATION_PLAN_ID
        ):
            values.append("comparison-generation-selection-plan-mismatch")
        if upstream.generation_model is not None and (
            upstream.generation_model.selected_variant_id
            != _GENERATION_VARIANT_IDS.get(
                upstream.generation_model.selected_axis_value,
                "unregistered",
            )
        ):
            values.append("comparison-generation-selection-variant-mismatch")
        if upstream.retrieval_strategy is not None and (
            upstream.retrieval_strategy.selected_axis_value not in _RETRIEVAL_MODES
        ):
            values.append("comparison-retrieval-selection-not-registered")
        if upstream.retrieval_strategy is not None and (
            upstream.retrieval_strategy.plan_id != REGISTERED_RETRIEVAL_PLAN_ID
        ):
            values.append("comparison-retrieval-selection-plan-mismatch")
        if upstream.retrieval_strategy is not None and (
            upstream.retrieval_strategy.selected_variant_id
            != f"retrieval-{upstream.retrieval_strategy.selected_axis_value}"
        ):
            values.append("comparison-retrieval-selection-variant-mismatch")
        if (
            plan_id == REGISTERED_CACHE_PLAN_ID
            and upstream.generation_model is not None
            and upstream.retrieval_strategy is not None
            and _fixed_identity_map(upstream.retrieval_strategy.upstream_identities)
            != _fixed_identity_map(
                _selection_provenance_identities(
                    "generation-model",
                    upstream.generation_model,
                )
            )
        ):
            values.append("comparison-retrieval-selection-generation-chain-mismatch")
        try:
            _selected_case_ids(plan_id, context.dataset)
            specs = _variant_specs(plan_id, context)
            baseline_settings = _candidate_settings(plan_id, specs[0].axis_value, context)
            preview = build_evaluation_plan(
                context.dataset,
                baseline_settings,
                "readiness-candidate",
            )
            if (
                preview.identity.provider_identities.get("generation")
                != OPENAI_COMPARISON_PRICING_PROVIDER
                or preview.identity.provider_identities.get("embedding")
                != OPENAI_COMPARISON_PRICING_PROVIDER
            ):
                values.append("comparison-pricing-provider-mismatch")
        except (
            ComparisonScheduleError,
            EvaluationPlanError,
            RegisteredComparisonPlanError,
            ValueError,
        ) as error:
            values.append(getattr(error, "code", "comparison-plan-preflight-invalid"))
        return tuple(dict.fromkeys(values))

    def _materialize(
        self,
        plan_id: str,
        context: ComparisonPlanMaterializationContext,
    ) -> MaterializedComparisonPlan:
        specs = _variant_specs(plan_id, context)
        expected_variants = {item.variant_id for item in specs}
        if set(context.candidate_run_ids) != expected_variants or len(
            set(context.candidate_run_ids.values())
        ) != len(specs):
            raise RegisteredComparisonPlanError("comparison-candidate-run-id-set-invalid")
        selected_case_ids = _selected_case_ids(plan_id, context.dataset)
        case_set_hash = case_ids_content_hash(selected_case_ids)
        candidate_settings: list[Settings] = []
        base_plans: list[EvaluationRunPlan] = []
        projections: list[ComparisonIdentityProjection] = []
        try:
            for spec in specs:
                settings = _candidate_settings(plan_id, spec.axis_value, context)
                candidate = build_evaluation_plan(
                    context.dataset,
                    settings,
                    context.candidate_run_ids[spec.variant_id],
                )
                if candidate.identity.runtime_configuration_id != (
                    settings.runtime_configuration_identity
                ):
                    raise RegisteredComparisonPlanError(
                        "comparison-runtime-configuration-identity-mismatch"
                    )
                if plan_id == REGISTERED_CACHE_PLAN_ID:
                    identity = EvaluationRunIdentity.model_validate(
                        {
                            **candidate.identity.model_dump(mode="python"),
                            "cache_policy": CachePolicy.USE,
                        }
                    )
                    candidate = EvaluationRunPlan(
                        run_id=candidate.run_id,
                        identity=identity,
                        cases=candidate.cases,
                    )
                projection = project_evaluation_identity(
                    spec.variant_id,
                    candidate.identity,
                    corpus_id=context.dataset.corpus.manifest.snapshot_id,
                    case_set_hash=case_set_hash,
                    cache_behavior=(
                        spec.axis_value if plan_id == REGISTERED_CACHE_PLAN_ID else None
                    ),
                )
                projection = _with_upstream_identities(
                    projection,
                    _upstream_fixed_identities(plan_id, context.upstream_selections),
                )
                candidate_settings.append(settings)
                base_plans.append(candidate)
                projections.append(projection)
        except (
            ComparisonDomainError,
            EvaluationPlanError,
            RegisteredComparisonPlanError,
            TypeError,
            ValueError,
        ) as error:
            raise RegisteredComparisonPlanError(
                getattr(error, "code", "comparison-plan-materialization-invalid")
            ) from None
        axis = _PLAN_AXES[plan_id]
        controlled = _controlled_identities(axis, projections)
        variants = tuple(
            ExperimentVariant(
                variant_id=spec.variant_id,
                display_name=spec.display_name,
                axis_value=spec.axis_value,
                configuration_id=base_plans[index].identity.configuration_id,
            )
            for index, spec in enumerate(specs)
        )
        fixed = ExperimentFixedIdentities(
            dataset_id=context.dataset.manifest.dataset_id,
            dataset_version=context.dataset.manifest.version,
            dataset_hash=context.dataset.manifest.content_hash,
            corpus_id=context.dataset.corpus.manifest.snapshot_id,
            corpus_version=context.dataset.corpus.manifest.version,
            corpus_hash=context.dataset.corpus.manifest.content_hash,
            case_set_hash=case_set_hash,
            case_count=len(selected_case_ids),
            controlled=controlled,
        )
        plan = ExperimentPlan.create(
            plan_id=plan_id,
            display_name=_PLAN_LABELS[plan_id],
            axis=axis,
            fixed_identities=fixed,
            variants=variants,
            baseline_variant_id=variants[0].variant_id,
            repeat_order_policy=RepeatOrderPolicy(
                repeats_per_case=_PLAN_REPEATS[plan_id],
                order_policy=(
                    ExperimentOrderPolicy.SEEDED_SHUFFLE
                    if axis is ExperimentAxis.CACHE_BEHAVIOR
                    else ExperimentOrderPolicy.SEEDED_INTERLEAVED
                ),
                seed=15_2026,
            ),
            cache_policy=(
                CachePolicy.USE if axis is ExperimentAxis.CACHE_BEHAVIOR else CachePolicy.BYPASS
            ),
            pricing=self._experiment_pricing,
            maximum_provider_calls=_PLAN_CALL_CAPS[plan_id],
            maximum_cost=_PLAN_COST_CAPS[plan_id],
            gate_profile=_gate_profile(plan_id),
            selection_policy=_selection_policy(plan_id),
        )
        try:
            validate_comparison_plan_safe_values(plan)
            schedule = build_comparison_schedule(
                plan,
                context.dataset,
                selected_case_ids=(
                    selected_case_ids if plan_id == REGISTERED_CACHE_PLAN_ID else None
                ),
            )
            candidates = tuple(
                MaterializedComparisonCandidate(
                    variant_id=spec.variant_id,
                    settings=candidate_settings[index],
                    evaluation_plan=EvaluationRunPlan(
                        run_id=base_plans[index].run_id,
                        identity=base_plans[index].identity,
                        cases=materialize_variant_cases(
                            schedule,
                            plan,
                            context.dataset,
                            spec.variant_id,
                            base_plans[index].cases,
                        ),
                    ),
                    identity_projection=projections[index],
                )
                for index, spec in enumerate(specs)
            )
            candidate_mapping = {item.variant_id: item.evaluation_plan for item in candidates}
            preflight = preflight_comparison_work(
                context.comparison_id,
                plan,
                context.dataset,
                candidate_mapping,
            )
        except (
            ComparisonDomainError,
            ComparisonPreflightError,
            ComparisonScheduleError,
            TypeError,
            ValueError,
        ) as error:
            raise RegisteredComparisonPlanError(
                getattr(error, "code", "comparison-plan-materialization-invalid")
            ) from None
        return MaterializedComparisonPlan(
            plan=plan,
            candidates=candidates,
            schedule=schedule,
            selected_case_ids=selected_case_ids,
            preflight=preflight,
        )


def _load_pricing(path: str | Path) -> PerformancePricingEvidence:
    candidate = Path(path)
    if candidate.is_symlink():
        raise RegisteredComparisonPlanError("comparison-pricing-path-unsafe")
    try:
        resolved = candidate.resolve(strict=True)
        content = resolved.read_bytes()
    except OSError:
        raise RegisteredComparisonPlanError("comparison-pricing-unavailable") from None
    if not resolved.is_file() or len(content) > 1_000_000:
        raise RegisteredComparisonPlanError("comparison-pricing-invalid")
    try:
        pricing = PerformancePricingEvidence.model_validate_json(content)
        preflight_openai_comparison_pricing(pricing)
    except (PricingPreflightError, TypeError, ValueError) as error:
        raise RegisteredComparisonPlanError(
            getattr(error, "code", "comparison-pricing-invalid")
        ) from None
    return pricing


def _experiment_pricing(
    pricing: PerformancePricingEvidence,
) -> ExperimentPricingProvenance:
    digest = preflight_openai_comparison_pricing(pricing)
    sources_by_model = {
        model: next(source for source in pricing.source_references if model in source)
        for model in {_EMBEDDING_MODEL, *_GENERATION_MODELS}
    }
    return ExperimentPricingProvenance(
        pricing_version=pricing.pricing_version,
        pricing_hash=digest,
        currency=pricing.currency,
        source_references=pricing.source_references,
        rate_card=tuple(
            ExperimentPricingRate(
                role=PricingRole(rate.role.value),
                provider=rate.provider,
                model=rate.model,
                input_per_million=rate.input_per_million,
                output_per_million=rate.output_per_million,
                source_reference=sources_by_model[rate.model],
            )
            for rate in pricing.rates
        ),
    )


def _variant_specs(
    plan_id: str,
    context: ComparisonPlanCatalogContext,
) -> tuple[RegisteredComparisonCandidateSummary, ...]:
    if plan_id == REGISTERED_GENERATION_PLAN_ID:
        current = context.settings.generation_model
        models = (
            (current, *tuple(item for item in _GENERATION_MODELS if item != current))
            if current in _GENERATION_MODELS
            else _GENERATION_MODELS
        )
        return tuple(
            RegisteredComparisonCandidateSummary(
                variant_id=_GENERATION_VARIANT_IDS[model],
                display_name=model,
                axis_value=model,
            )
            for model in models
        )
    if plan_id == REGISTERED_RETRIEVAL_PLAN_ID:
        return tuple(
            RegisteredComparisonCandidateSummary(
                variant_id=f"retrieval-{mode}",
                display_name=mode,
                axis_value=mode,
            )
            for mode in _RETRIEVAL_MODES
        )
    if plan_id == REGISTERED_CACHE_PLAN_ID:
        return tuple(
            RegisteredComparisonCandidateSummary(
                variant_id=f"cache-{behavior}",
                display_name=behavior,
                axis_value=behavior,
            )
            for behavior in _CACHE_BEHAVIORS
        )
    raise RegisteredComparisonPlanError("comparison-plan-not-registered")


def _candidate_settings(
    plan_id: str,
    axis_value: str,
    context: ComparisonPlanCatalogContext,
) -> Settings:
    updates: dict[str, object] = {
        "pricing_version": OPENAI_COMPARISON_PRICING_VERSION,
        "retrieval_cache_enabled": False,
    }
    if plan_id == REGISTERED_GENERATION_PLAN_ID:
        updates.update(
            generation_model=axis_value,
            default_retrieval_mode="hybrid",
            reranking_model=None,
        )
    elif plan_id == REGISTERED_RETRIEVAL_PLAN_ID:
        selected = context.upstream_selections.generation_model
        if selected is None:
            raise RegisteredComparisonPlanError("comparison-generation-selection-required")
        updates.update(
            generation_model=selected.selected_axis_value,
            default_retrieval_mode=axis_value,
            reranking_model=(_RERANKING_MODEL if axis_value == "hybrid-rerank" else None),
        )
    elif plan_id == REGISTERED_CACHE_PLAN_ID:
        generation = context.upstream_selections.generation_model
        retrieval = context.upstream_selections.retrieval_strategy
        if generation is None:
            raise RegisteredComparisonPlanError("comparison-generation-selection-required")
        if retrieval is None:
            raise RegisteredComparisonPlanError("comparison-retrieval-selection-required")
        eligible_count = len(cache_eligible_case_ids(context.dataset))
        try:
            minimum_ttl = minimum_cache_experiment_ttl_seconds(
                eligible_count,
                context.settings.qa_deadline_seconds,
            )
        except ComparisonPreflightError as error:
            raise RegisteredComparisonPlanError(error.code) from None
        updates.update(
            generation_model=generation.selected_axis_value,
            default_retrieval_mode=retrieval.selected_axis_value,
            reranking_model=(
                _RERANKING_MODEL if retrieval.selected_axis_value == "hybrid-rerank" else None
            ),
            retrieval_cache_enabled=True,
            retrieval_cache_max_entries=max(
                context.settings.retrieval_cache_max_entries,
                eligible_count,
            ),
            retrieval_cache_ttl_seconds=max(
                context.settings.retrieval_cache_ttl_seconds,
                minimum_ttl,
            ),
        )
    else:
        raise RegisteredComparisonPlanError("comparison-plan-not-registered")
    try:
        return Settings.model_validate({**context.settings.model_dump(mode="python"), **updates})
    except ValueError:
        raise RegisteredComparisonPlanError("comparison-candidate-settings-invalid") from None


def _selected_case_ids(plan_id: str, dataset: EvaluationDataset) -> tuple[str, ...]:
    if plan_id == REGISTERED_CACHE_PLAN_ID:
        try:
            return cache_eligible_case_ids(dataset)
        except ComparisonScheduleError as error:
            raise RegisteredComparisonPlanError(error.code) from None
    values = tuple(item.case_id for item in dataset.cases)
    if not values:
        raise RegisteredComparisonPlanError("comparison-dataset-empty")
    return values


def _controlled_identities(
    axis: ExperimentAxis,
    projections: Sequence[ComparisonIdentityProjection],
) -> tuple[FixedIdentity, ...]:
    if not projections:
        raise RegisteredComparisonPlanError("comparison-identity-projections-empty")
    allowed = {
        ExperimentAxis.GENERATION_MODEL: {"generation.model"},
        ExperimentAxis.RETRIEVAL_STRATEGY: {
            "retrieval.mode",
            "retrieval.reranking_enabled",
            "model.reranking",
            "provider.reranking",
        },
        ExperimentAxis.CACHE_BEHAVIOR: {"cache.behavior"},
    }[axis]
    first = projections[0]
    first_map = first.identity_map()
    controlled = tuple(item for item in first.identities if item.name not in allowed)
    controlled_map = {item.name: item.value for item in controlled}
    if any(
        {name: value for name, value in projection.identity_map().items() if name not in allowed}
        != controlled_map
        for projection in projections[1:]
    ):
        raise RegisteredComparisonPlanError("comparison-controlled-identity-mismatch")
    if axis.identity_name not in first_map:
        raise RegisteredComparisonPlanError("comparison-axis-identity-missing")
    return controlled


def _upstream_fixed_identities(
    plan_id: str,
    selections: UpstreamSelections,
) -> tuple[FixedIdentity, ...]:
    values: list[FixedIdentity] = []
    required: tuple[tuple[str, UpstreamComparisonSelection | None], ...] = ()
    if plan_id == REGISTERED_RETRIEVAL_PLAN_ID:
        required = (("generation-model", selections.generation_model),)
    elif plan_id == REGISTERED_CACHE_PLAN_ID:
        required = (
            ("generation-model", selections.generation_model),
            ("retrieval-strategy", selections.retrieval_strategy),
        )
    for expected_axis, selection in required:
        if selection is None:
            raise RegisteredComparisonPlanError(
                f"comparison-{expected_axis.removesuffix('-model').removesuffix('-strategy')}-selection-required"
            )
        values.extend(_selection_provenance_identities(expected_axis, selection))
    return tuple(values)


def _selection_provenance_identities(
    axis: str,
    selection: UpstreamComparisonSelection,
) -> tuple[FixedIdentity, ...]:
    prefix = f"upstream.{axis}"
    return (
        FixedIdentity(name=f"{prefix}.plan-id", value=selection.plan_id),
        FixedIdentity(name=f"{prefix}.plan-hash", value=selection.plan_content_hash),
        FixedIdentity(name=f"{prefix}.result-hash", value=selection.result_content_hash),
        FixedIdentity(name=f"{prefix}.variant-id", value=selection.selected_variant_id),
        FixedIdentity(name=f"{prefix}.axis-value", value=selection.selected_axis_value),
        FixedIdentity(
            name=f"{prefix}.configuration-id",
            value=selection.selected_configuration_id,
        ),
        FixedIdentity(
            name=f"{prefix}.evaluation-run-id",
            value=selection.selected_evaluation_run_id,
        ),
    )


def _fixed_identity_map(values: Sequence[FixedIdentity]) -> dict[str, str]:
    return {item.name: item.value for item in values}


def _with_upstream_identities(
    projection: ComparisonIdentityProjection,
    upstream: tuple[FixedIdentity, ...],
) -> ComparisonIdentityProjection:
    if not upstream:
        return projection
    existing = {item.name for item in projection.identities}
    if any(item.name in existing for item in upstream):
        raise RegisteredComparisonPlanError("comparison-upstream-identity-collision")
    return ComparisonIdentityProjection.model_validate(
        {
            **projection.model_dump(mode="python"),
            "identities": (*projection.identities, *upstream),
        }
    )


def _gate_profile(plan_id: str) -> ExperimentGateProfile:
    payload = {
        "gate_id": COMPARISON_SELECTION_ELIGIBILITY_GATE_ID,
        "version": COMPARISON_SELECTION_ELIGIBILITY_GATE_VERSION,
        "thresholds": {
            "comparison-terminal-case-coverage": {"operator": "==", "value": 1.0},
            "comparison-logical-all-p90-ms": {
                "operator": "<=",
                "value": COMPARISON_SELECTION_MAX_P90_MS,
            },
        },
        "diagnostics": {
            "comparison-provider-cost-evidence-completeness": "non-blocking",
            "comparison-logical-error-rate": "non-blocking",
            "comparison-logical-timeout-rate": "non-blocking",
        },
    }
    if plan_id == REGISTERED_RETRIEVAL_PLAN_ID:
        payload["reranker_minimum_quality_benefit"] = {
            "metric": "primary-maximize",
            "operator": ">",
            "value": COMPARISON_RERANKER_MIN_QUALITY_BENEFIT,
        }
    digest = hashlib.sha256(canonical_json_value(payload).encode("utf-8")).hexdigest()
    return ExperimentGateProfile(
        profile_id=(
            COMPARISON_RERANKER_BENEFIT_PROFILE_ID
            if plan_id == REGISTERED_RETRIEVAL_PLAN_ID
            else f"{plan_id}-selection-eligibility-v2"
        ),
        profile_version=COMPARISON_SELECTION_ELIGIBILITY_GATE_VERSION,
        profile_hash=f"sha256:{digest}",
        mandatory_gate_ids=(COMPARISON_SELECTION_ELIGIBILITY_GATE_ID,),
    )


def _selection_policy(plan_id: str) -> DeterministicSelectionPolicy:
    criteria: tuple[SelectionCriterion, ...]
    if plan_id == REGISTERED_GENERATION_PLAN_ID:
        criteria = (
            SelectionCriterion(
                metric="answer-compliance",
                direction=SelectionDirection.MAXIMIZE,
            ),
            SelectionCriterion(
                metric="faithfulness",
                direction=SelectionDirection.MAXIMIZE,
            ),
            SelectionCriterion(
                metric="comparison-logical-all-p90-ms",
                direction=SelectionDirection.MINIMIZE,
            ),
        )
    elif plan_id == REGISTERED_RETRIEVAL_PLAN_ID:
        criteria = (
            SelectionCriterion(
                metric="context-precision",
                direction=SelectionDirection.MAXIMIZE,
            ),
            SelectionCriterion(
                metric="answer-compliance",
                direction=SelectionDirection.MAXIMIZE,
            ),
            SelectionCriterion(
                metric="comparison-logical-all-p90-ms",
                direction=SelectionDirection.MINIMIZE,
            ),
        )
    else:
        criteria = (
            SelectionCriterion(
                metric="comparison-logical-all-p90-ms",
                direction=SelectionDirection.MINIMIZE,
            ),
        )
    return DeterministicSelectionPolicy(
        policy_id=f"{plan_id}-selection-v3",
        policy_version="3.0.0",
        required_gate_ids=(COMPARISON_SELECTION_ELIGIBILITY_GATE_ID,),
        tie_breakers=criteria,
        final_tie_break=FinalTieBreak.BASELINE_FIRST,
    )


__all__ = [
    "REGISTERED_CACHE_PLAN_ID",
    "REGISTERED_COMPARISON_PLAN_IDS",
    "REGISTERED_GENERATION_PLAN_ID",
    "REGISTERED_RETRIEVAL_PLAN_ID",
    "ComparisonPlanCatalogContext",
    "ComparisonPlanMaterializationContext",
    "MaterializationContext",
    "MaterializedComparisonCandidate",
    "MaterializedComparisonPlan",
    "RegisteredComparisonCandidateSummary",
    "RegisteredComparisonPlanError",
    "RegisteredComparisonPlanRegistry",
    "RegisteredComparisonPlanSummary",
    "UpstreamComparisonSelection",
    "UpstreamSelections",
]


MaterializationContext = ComparisonPlanMaterializationContext
