"""Immutable registered experiment plans for controlled evaluation comparisons."""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import Field, StringConstraints, field_validator, model_validator

from rag_mvp.domain._base import DomainModel
from rag_mvp.domain.retrieval import CachePolicy

EXPERIMENT_PLAN_SCHEMA_VERSION: Final = "experiment-plan-v1"
EXPERIMENT_PLAN_REGISTRY_SCHEMA_VERSION: Final = "experiment-plan-registry-v1"

type Sha256Digest = Annotated[
    str,
    StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$"),
]
type PlanIdentifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    ),
]
type IdentityName = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=255,
        pattern=r"^[a-z][a-z0-9_.-]*$",
    ),
]
type VersionIdentifier = Annotated[str, StringConstraints(min_length=1, max_length=255)]
type DisplayText = Annotated[str, StringConstraints(min_length=1, max_length=255)]
type CanonicalIdentityValue = Annotated[str, StringConstraints(min_length=1, max_length=4096)]
type SourceReference = Annotated[str, StringConstraints(min_length=1, max_length=2048)]
type CurrencyCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
type NonNegativeCost = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
type NonNegativeRate = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]

_PLAN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")


class ExperimentAxis(StrEnum):
    """The sole identity dimension permitted to vary among candidates."""

    GENERATION_MODEL = "generation-model"
    RETRIEVAL_STRATEGY = "retrieval-strategy"
    CACHE_BEHAVIOR = "cache-behavior"

    @property
    def identity_name(self) -> str:
        return {
            ExperimentAxis.GENERATION_MODEL: "generation.model",
            ExperimentAxis.RETRIEVAL_STRATEGY: "retrieval.mode",
            ExperimentAxis.CACHE_BEHAVIOR: "cache.behavior",
        }[self]


class ExperimentOrderPolicy(StrEnum):
    """Predeclared deterministic ordering policy for repeated candidate work."""

    DECLARED = "declared"
    SEEDED_SHUFFLE = "seeded-shuffle"
    SEEDED_INTERLEAVED = "seeded-interleaved"


class PricingRole(StrEnum):
    EMBEDDING = "embedding"
    GENERATION = "generation"
    RERANKING = "reranking"
    EVALUATION = "evaluation"


class SelectionDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class FinalTieBreak(StrEnum):
    """A total, deterministic final ordering after all metric tie-breaks."""

    BASELINE_FIRST = "baseline-first"
    VARIANT_ORDER = "variant-order"


class FixedIdentity(DomainModel):
    """One canonical controlled identity used for compatibility diagnostics."""

    name: IdentityName
    value: CanonicalIdentityValue


class ExperimentFixedIdentities(DomainModel):
    """Dataset/corpus/case identities plus every non-experimental dimension."""

    dataset_id: PlanIdentifier
    dataset_version: VersionIdentifier
    dataset_hash: Sha256Digest
    corpus_id: PlanIdentifier
    corpus_version: VersionIdentifier
    corpus_hash: Sha256Digest
    case_set_hash: Sha256Digest
    case_count: Annotated[int, Field(gt=0)]
    controlled: tuple[FixedIdentity, ...]

    @field_validator("controlled")
    @classmethod
    def canonicalize_controlled(
        cls,
        values: tuple[FixedIdentity, ...],
    ) -> tuple[FixedIdentity, ...]:
        if not values:
            raise ValueError("experiment_controlled_identities_empty")
        names = tuple(value.name for value in values)
        if len(set(names)) != len(names):
            raise ValueError("experiment_controlled_identity_duplicate")
        return tuple(sorted(values, key=lambda value: value.name))


class ExperimentVariant(DomainModel):
    """One ordered candidate and its exact value along the declared axis."""

    variant_id: PlanIdentifier
    display_name: DisplayText
    axis_value: CanonicalIdentityValue
    configuration_id: PlanIdentifier

    @field_validator("display_name", "axis_value")
    @classmethod
    def require_single_line(cls, value: str) -> str:
        if "\n" in value or "\r" in value:
            raise ValueError("experiment_variant_text_multiline")
        return value


class RepeatOrderPolicy(DomainModel):
    """Bounded repetition and deterministic ordering declared before execution."""

    repeats_per_case: Annotated[int, Field(ge=1, le=10_000)]
    order_policy: ExperimentOrderPolicy
    seed: Annotated[int, Field(ge=0, le=9_223_372_036_854_775_807)]


class ExperimentPricingRate(DomainModel):
    """Exact role/provider/model rates captured by the plan's pricing evidence."""

    role: PricingRole
    provider: PlanIdentifier
    model: VersionIdentifier
    input_per_million: NonNegativeRate | None = None
    output_per_million: NonNegativeRate | None = None
    source_reference: SourceReference

    @model_validator(mode="after")
    def require_known_rate(self) -> Self:
        if self.input_per_million is None and self.output_per_million is None:
            raise ValueError("experiment_pricing_rate_unknown")
        return self

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.role.value, self.provider, self.model


class ExperimentPricingProvenance(DomainModel):
    """Versioned, hashed pricing provenance and the exact applicable rate card."""

    pricing_version: VersionIdentifier
    pricing_hash: Sha256Digest
    currency: CurrencyCode
    source_references: tuple[SourceReference, ...]
    rate_card: tuple[ExperimentPricingRate, ...]

    @field_validator("source_references")
    @classmethod
    def canonicalize_sources(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("experiment_pricing_sources_empty")
        if len(set(values)) != len(values):
            raise ValueError("experiment_pricing_source_duplicate")
        return tuple(sorted(values))

    @field_validator("rate_card")
    @classmethod
    def canonicalize_rates(
        cls,
        values: tuple[ExperimentPricingRate, ...],
    ) -> tuple[ExperimentPricingRate, ...]:
        if not values:
            raise ValueError("experiment_pricing_rate_card_empty")
        identities = tuple(value.identity for value in values)
        if len(set(identities)) != len(identities):
            raise ValueError("experiment_pricing_rate_duplicate")
        return tuple(sorted(values, key=lambda value: value.identity))

    @model_validator(mode="after")
    def require_rate_sources(self) -> Self:
        sources = set(self.source_references)
        if any(rate.source_reference not in sources for rate in self.rate_card):
            raise ValueError("experiment_pricing_rate_source_unregistered")
        return self


class ExperimentGateProfile(DomainModel):
    """Immutable reference to the predeclared gates used by a comparison."""

    profile_id: PlanIdentifier
    profile_version: VersionIdentifier
    profile_hash: Sha256Digest
    mandatory_gate_ids: tuple[PlanIdentifier, ...]

    @field_validator("mandatory_gate_ids")
    @classmethod
    def canonicalize_gates(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("experiment_gate_profile_empty")
        if len(set(values)) != len(values):
            raise ValueError("experiment_gate_duplicate")
        return tuple(sorted(values))


class SelectionCriterion(DomainModel):
    """One ordered metric tie-break in a deterministic selection policy."""

    metric: PlanIdentifier
    direction: SelectionDirection


class DeterministicSelectionPolicy(DomainModel):
    """Mandatory gates and total tie-break order fixed before provider calls."""

    policy_id: PlanIdentifier
    policy_version: VersionIdentifier
    required_gate_ids: tuple[PlanIdentifier, ...]
    tie_breakers: tuple[SelectionCriterion, ...]
    final_tie_break: FinalTieBreak
    no_recommendation_if_incomplete: Literal[True] = True
    no_recommendation_if_incompatible: Literal[True] = True
    no_recommendation_if_cost_cap_exceeded: Literal[True] = True

    @field_validator("required_gate_ids")
    @classmethod
    def canonicalize_required_gates(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("experiment_selection_gates_empty")
        if len(set(values)) != len(values):
            raise ValueError("experiment_selection_gate_duplicate")
        return tuple(sorted(values))

    @field_validator("tie_breakers")
    @classmethod
    def require_unique_tie_breakers(
        cls,
        values: tuple[SelectionCriterion, ...],
    ) -> tuple[SelectionCriterion, ...]:
        if not values:
            raise ValueError("experiment_selection_tie_breakers_empty")
        metrics = tuple(value.metric for value in values)
        if len(set(metrics)) != len(metrics):
            raise ValueError("experiment_selection_tie_breaker_duplicate")
        return values


class _ExperimentPlanContent(DomainModel):
    schema_version: Literal["experiment-plan-v1"] = EXPERIMENT_PLAN_SCHEMA_VERSION
    plan_id: PlanIdentifier
    display_name: DisplayText
    axis: ExperimentAxis
    fixed_identities: ExperimentFixedIdentities
    variants: tuple[ExperimentVariant, ...]
    baseline_variant_id: PlanIdentifier
    repeat_order_policy: RepeatOrderPolicy
    cache_policy: CachePolicy
    pricing: ExperimentPricingProvenance
    maximum_provider_calls: Annotated[int, Field(gt=0)]
    maximum_cost: NonNegativeCost
    gate_profile: ExperimentGateProfile
    selection_policy: DeterministicSelectionPolicy

    @field_validator("display_name")
    @classmethod
    def require_single_line_display_name(cls, value: str) -> str:
        if "\n" in value or "\r" in value:
            raise ValueError("experiment_plan_display_name_multiline")
        return value

    @model_validator(mode="after")
    def validate_plan_contract(self) -> Self:
        if len(self.variants) < 2:
            raise ValueError("experiment_variants_insufficient")
        variant_ids = tuple(variant.variant_id for variant in self.variants)
        if len(set(variant_ids)) != len(variant_ids):
            raise ValueError("experiment_variant_id_duplicate")
        axis_values = tuple(variant.axis_value for variant in self.variants)
        if len(set(axis_values)) != len(axis_values):
            raise ValueError("experiment_axis_value_duplicate")
        if self.baseline_variant_id not in set(variant_ids):
            raise ValueError("experiment_baseline_not_registered")

        controlled_names = {identity.name for identity in self.fixed_identities.controlled}
        if self.axis.identity_name in controlled_names:
            raise ValueError("experiment_axis_declared_fixed")

        if self.axis is ExperimentAxis.CACHE_BEHAVIOR:
            if self.cache_policy is not CachePolicy.USE:
                raise ValueError("experiment_cache_axis_requires_use")
        elif self.cache_policy is not CachePolicy.BYPASS:
            raise ValueError("experiment_official_comparison_requires_cache_bypass")

        if set(self.selection_policy.required_gate_ids) != set(
            self.gate_profile.mandatory_gate_ids
        ):
            raise ValueError("experiment_selection_gate_profile_mismatch")
        return self


class ExperimentPlan(_ExperimentPlanContent):
    """A deeply immutable plan whose canonical content hash is self-verifying."""

    content_hash: Sha256Digest

    @classmethod
    def create(cls, **values: object) -> ExperimentPlan:
        content = _ExperimentPlanContent.model_validate(values)
        payload = content.model_dump(mode="python")
        payload["content_hash"] = _canonical_content_hash(content)
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def verify_content_hash(self) -> Self:
        if self.content_hash != _canonical_content_hash(self):
            raise ValueError("experiment_plan_content_hash_mismatch")
        return self

    def verify_hash(self) -> Self:
        """Re-validate the plan hash at an explicit trust boundary."""

        if self.content_hash != _canonical_content_hash(self):
            raise ValueError("experiment_plan_content_hash_mismatch")
        return self


class ExperimentPlanRegistryError(LookupError):
    """Stable registry lookup error that contains no plan internals."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ExperimentPlanRegistry(DomainModel):
    """Validated immutable catalog of predeclared experiment plans."""

    schema_version: Literal["experiment-plan-registry-v1"] = EXPERIMENT_PLAN_REGISTRY_SCHEMA_VERSION
    plans: tuple[ExperimentPlan, ...]

    @field_validator("plans")
    @classmethod
    def validate_plans(cls, plans: tuple[ExperimentPlan, ...]) -> tuple[ExperimentPlan, ...]:
        if not plans:
            raise ValueError("experiment_plan_registry_empty")
        plan_ids = tuple(plan.plan_id for plan in plans)
        if len(set(plan_ids)) != len(plan_ids):
            raise ValueError("experiment_plan_registry_duplicate_id")
        hashes = tuple(plan.content_hash for plan in plans)
        if len(set(hashes)) != len(hashes):
            raise ValueError("experiment_plan_registry_duplicate_hash")
        for plan in plans:
            plan.verify_hash()
        return plans

    def get(self, plan_id: str) -> ExperimentPlan | None:
        if not isinstance(plan_id, str) or _PLAN_ID.fullmatch(plan_id) is None:
            raise ExperimentPlanRegistryError("experiment_plan_id_invalid")
        return next((plan for plan in self.plans if plan.plan_id == plan_id), None)

    def resolve(self, plan_id: str) -> ExperimentPlan:
        plan = self.get(plan_id)
        if plan is None:
            raise ExperimentPlanRegistryError("experiment_plan_not_found")
        return plan

    def list(self, *, axis: ExperimentAxis | str | None = None) -> tuple[ExperimentPlan, ...]:
        if axis is None:
            return self.plans
        try:
            resolved = ExperimentAxis(axis)
        except (TypeError, ValueError):
            raise ExperimentPlanRegistryError("experiment_axis_invalid") from None
        return tuple(plan for plan in self.plans if plan.axis is resolved)


def experiment_plan_content_hash(plan: ExperimentPlan) -> str:
    """Return the verified canonical SHA-256 identity for a plan."""

    plan.verify_hash()
    return plan.content_hash


def _canonical_content_hash(plan: _ExperimentPlanContent) -> str:
    payload = plan.model_dump(mode="json", exclude={"content_hash"})
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


__all__ = [
    "EXPERIMENT_PLAN_REGISTRY_SCHEMA_VERSION",
    "EXPERIMENT_PLAN_SCHEMA_VERSION",
    "DeterministicSelectionPolicy",
    "ExperimentAxis",
    "ExperimentFixedIdentities",
    "ExperimentGateProfile",
    "ExperimentOrderPolicy",
    "ExperimentPlan",
    "ExperimentPlanRegistry",
    "ExperimentPlanRegistryError",
    "ExperimentPricingProvenance",
    "ExperimentPricingRate",
    "ExperimentVariant",
    "FinalTieBreak",
    "FixedIdentity",
    "PricingRole",
    "RepeatOrderPolicy",
    "SelectionCriterion",
    "SelectionDirection",
    "experiment_plan_content_hash",
]
