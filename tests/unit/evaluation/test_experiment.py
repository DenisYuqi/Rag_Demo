from __future__ import annotations

import re
from decimal import Decimal

import pytest
from pydantic import ValidationError

from rag_mvp.domain.retrieval import CachePolicy
from rag_mvp.evaluation.experiment import (
    DeterministicSelectionPolicy,
    ExperimentAxis,
    ExperimentFixedIdentities,
    ExperimentGateProfile,
    ExperimentOrderPolicy,
    ExperimentPlan,
    ExperimentPlanRegistry,
    ExperimentPlanRegistryError,
    ExperimentPricingProvenance,
    ExperimentPricingRate,
    ExperimentVariant,
    FinalTieBreak,
    FixedIdentity,
    PricingRole,
    RepeatOrderPolicy,
    SelectionCriterion,
    SelectionDirection,
    experiment_plan_content_hash,
)

_HASH_A = "sha256:" + ("a" * 64)
_HASH_B = "sha256:" + ("b" * 64)
_HASH_C = "sha256:" + ("c" * 64)
_MODEL_SOURCE = "https://pricing.example.test/models"
_EMBEDDING_SOURCE = "https://pricing.example.test/embeddings"


def _fixed_identities(
    controlled: tuple[FixedIdentity, ...] | None = None,
) -> ExperimentFixedIdentities:
    return ExperimentFixedIdentities(
        dataset_id="acceptance-v2",
        dataset_version="2.0.0",
        dataset_hash=_HASH_A,
        corpus_id="acceptance-corpus-v2",
        corpus_version="2.0.0",
        corpus_hash=_HASH_B,
        case_set_hash=_HASH_C,
        case_count=24,
        controlled=controlled
        or (
            FixedIdentity(name="retrieval.mode", value="hybrid"),
            FixedIdentity(name="prompt.generation", value="grounded-claims-json-v2"),
            FixedIdentity(name="scorer.answer-compliance", value="all-obligations-v1"),
        ),
    )


def _pricing(
    *,
    reverse_sources: bool = False,
    reverse_rates: bool = False,
) -> ExperimentPricingProvenance:
    sources = (_MODEL_SOURCE, _EMBEDDING_SOURCE)
    rates = (
        ExperimentPricingRate(
            role=PricingRole.GENERATION,
            provider="provider-a",
            model="generation-v1",
            input_per_million=Decimal("0.40"),
            output_per_million=Decimal("1.60"),
            source_reference=_MODEL_SOURCE,
        ),
        ExperimentPricingRate(
            role=PricingRole.GENERATION,
            provider="provider-a",
            model="generation-v2",
            input_per_million=Decimal("2.50"),
            output_per_million=Decimal("15.00"),
            source_reference=_MODEL_SOURCE,
        ),
        ExperimentPricingRate(
            role=PricingRole.EMBEDDING,
            provider="provider-a",
            model="embedding-v1",
            input_per_million=Decimal("0.02"),
            source_reference=_EMBEDDING_SOURCE,
        ),
    )
    return ExperimentPricingProvenance(
        pricing_version="pricing-2026-08-07",
        pricing_hash=_HASH_A,
        currency="USD",
        source_references=tuple(reversed(sources)) if reverse_sources else sources,
        rate_card=tuple(reversed(rates)) if reverse_rates else rates,
    )


def _gate_profile(*, reverse: bool = False) -> ExperimentGateProfile:
    gates = ("advanced-quality", "all-attempt-p90", "cost-cap")
    return ExperimentGateProfile(
        profile_id="comparison-gates-v1",
        profile_version="1.0.0",
        profile_hash=_HASH_B,
        mandatory_gate_ids=tuple(reversed(gates)) if reverse else gates,
    )


def _selection(*, reverse_gates: bool = False) -> DeterministicSelectionPolicy:
    gates = ("advanced-quality", "all-attempt-p90", "cost-cap")
    return DeterministicSelectionPolicy(
        policy_id="quality-cost-latency-v1",
        policy_version="1.0.0",
        required_gate_ids=tuple(reversed(gates)) if reverse_gates else gates,
        tie_breakers=(
            SelectionCriterion(
                metric="answer-compliance",
                direction=SelectionDirection.MAXIMIZE,
            ),
            SelectionCriterion(
                metric="cost-per-1000-attempts",
                direction=SelectionDirection.MINIMIZE,
            ),
            SelectionCriterion(
                metric="all-attempt-p90-ms",
                direction=SelectionDirection.MINIMIZE,
            ),
        ),
        final_tie_break=FinalTieBreak.BASELINE_FIRST,
    )


def _variants() -> tuple[ExperimentVariant, ...]:
    return (
        ExperimentVariant(
            variant_id="model-v1",
            display_name="Generation v1",
            axis_value="provider-a/generation-v1",
            configuration_id="configuration-v1",
        ),
        ExperimentVariant(
            variant_id="model-v2",
            display_name="Generation v2",
            axis_value="provider-a/generation-v2",
            configuration_id="configuration-v2",
        ),
    )


def _plan_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "plan_id": "generation-model-comparison-v1",
        "display_name": "Generation model comparison / 生成模型对比",
        "axis": ExperimentAxis.GENERATION_MODEL,
        "fixed_identities": _fixed_identities(),
        "variants": _variants(),
        "baseline_variant_id": "model-v1",
        "repeat_order_policy": RepeatOrderPolicy(
            repeats_per_case=2,
            order_policy=ExperimentOrderPolicy.SEEDED_INTERLEAVED,
            seed=20260807,
        ),
        "cache_policy": CachePolicy.BYPASS,
        "pricing": _pricing(),
        "maximum_provider_calls": 500,
        "maximum_cost": Decimal("25.00"),
        "gate_profile": _gate_profile(),
        "selection_policy": _selection(),
    }
    values.update(overrides)
    return values


def _plan(**overrides: object) -> ExperimentPlan:
    return ExperimentPlan.create(**_plan_values(**overrides))


def test_plan_captures_the_complete_precommitted_experiment_contract() -> None:
    plan = _plan()

    assert plan.axis is ExperimentAxis.GENERATION_MODEL
    assert tuple(variant.variant_id for variant in plan.variants) == ("model-v1", "model-v2")
    assert plan.baseline_variant_id == "model-v1"
    assert plan.repeat_order_policy == RepeatOrderPolicy(
        repeats_per_case=2,
        order_policy=ExperimentOrderPolicy.SEEDED_INTERLEAVED,
        seed=20260807,
    )
    assert plan.cache_policy is CachePolicy.BYPASS
    assert plan.maximum_provider_calls == 500
    assert plan.maximum_cost == Decimal("25.00")
    assert plan.pricing.currency == "USD"
    assert plan.selection_policy.no_recommendation_if_incomplete is True
    assert plan.selection_policy.no_recommendation_if_incompatible is True
    assert plan.selection_policy.no_recommendation_if_cost_cap_exceeded is True
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", plan.content_hash)
    assert experiment_plan_content_hash(plan) == plan.content_hash
    assert ExperimentPlan.model_validate_json(plan.model_dump_json()) == plan


def test_hash_is_canonical_for_set_like_fields_but_preserves_variant_order() -> None:
    first = _plan()
    reversed_controlled = tuple(reversed(_fixed_identities().controlled))
    equivalent = _plan(
        fixed_identities=_fixed_identities(reversed_controlled),
        pricing=_pricing(reverse_sources=True, reverse_rates=True),
        gate_profile=_gate_profile(reverse=True),
        selection_policy=_selection(reverse_gates=True),
    )
    reordered_variants = _plan(variants=tuple(reversed(_variants())))
    changed_seed = _plan(
        repeat_order_policy=RepeatOrderPolicy(
            repeats_per_case=2,
            order_policy=ExperimentOrderPolicy.SEEDED_INTERLEAVED,
            seed=20260808,
        )
    )

    assert equivalent.content_hash == first.content_hash
    assert reordered_variants.content_hash != first.content_hash
    assert changed_seed.content_hash != first.content_hash


def test_plan_hash_rejects_tampering_and_models_are_deeply_immutable() -> None:
    plan = _plan()
    tampered = plan.model_dump(mode="python")
    tampered["maximum_provider_calls"] = 501

    with pytest.raises(ValidationError, match="experiment_plan_content_hash_mismatch"):
        ExperimentPlan.model_validate(tampered)
    with pytest.raises(ValidationError, match="frozen"):
        plan.plan_id = "replacement"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        plan.variants[0].axis_value = "replacement"  # type: ignore[misc]
    assert isinstance(plan.variants, tuple)
    assert isinstance(plan.fixed_identities.controlled, tuple)


@pytest.mark.parametrize(
    ("overrides", "error_code"),
    (
        ({"variants": _variants()[:1]}, "experiment_variants_insufficient"),
        ({"baseline_variant_id": "missing"}, "experiment_baseline_not_registered"),
        (
            {"variants": (_variants()[0], _variants()[0])},
            "experiment_variant_id_duplicate",
        ),
        (
            {
                "variants": (
                    _variants()[0],
                    _variants()[1].model_copy(update={"axis_value": _variants()[0].axis_value}),
                )
            },
            "experiment_axis_value_duplicate",
        ),
        (
            {
                "fixed_identities": _fixed_identities(
                    (
                        FixedIdentity(name="generation.model", value="generation-v1"),
                        FixedIdentity(name="retrieval.mode", value="hybrid"),
                    )
                )
            },
            "experiment_axis_declared_fixed",
        ),
        ({"cache_policy": CachePolicy.USE}, "experiment_official_comparison_requires_cache_bypass"),
        (
            {
                "selection_policy": DeterministicSelectionPolicy(
                    policy_id="mismatched-policy",
                    policy_version="1.0.0",
                    required_gate_ids=("advanced-quality",),
                    tie_breakers=(
                        SelectionCriterion(
                            metric="answer-compliance",
                            direction=SelectionDirection.MAXIMIZE,
                        ),
                    ),
                    final_tie_break=FinalTieBreak.VARIANT_ORDER,
                )
            },
            "experiment_selection_gate_profile_mismatch",
        ),
    ),
)
def test_plan_rejects_ambiguous_or_incompatible_contracts(
    overrides: dict[str, object],
    error_code: str,
) -> None:
    with pytest.raises(ValidationError, match=error_code):
        ExperimentPlan.create(**_plan_values(**overrides))


def test_cache_experiment_requires_use_while_official_axes_require_bypass() -> None:
    cache_variants = (
        ExperimentVariant(
            variant_id="cold",
            display_name="Cold cache",
            axis_value="cold",
            configuration_id="cache-configuration",
        ),
        ExperimentVariant(
            variant_id="warm",
            display_name="Warm cache",
            axis_value="warm",
            configuration_id="cache-configuration",
        ),
    )
    cache_plan = _plan(
        plan_id="cache-comparison-v1",
        axis=ExperimentAxis.CACHE_BEHAVIOR,
        variants=cache_variants,
        baseline_variant_id="cold",
        cache_policy=CachePolicy.USE,
        fixed_identities=_fixed_identities(
            (
                FixedIdentity(name="generation.model", value="generation-v2"),
                FixedIdentity(name="retrieval.mode", value="hybrid"),
            )
        ),
    )

    assert cache_plan.cache_policy is CachePolicy.USE
    with pytest.raises(ValidationError, match="experiment_cache_axis_requires_use"):
        ExperimentPlan.create(
            **_plan_values(
                plan_id="cache-comparison-v1",
                axis=ExperimentAxis.CACHE_BEHAVIOR,
                variants=cache_variants,
                baseline_variant_id="cold",
                cache_policy=CachePolicy.BYPASS,
                fixed_identities=_fixed_identities(
                    (
                        FixedIdentity(name="generation.model", value="generation-v2"),
                        FixedIdentity(name="retrieval.mode", value="hybrid"),
                    )
                ),
            )
        )


def test_pricing_and_selection_contracts_fail_closed() -> None:
    with pytest.raises(ValidationError, match="experiment_pricing_rate_unknown"):
        ExperimentPricingRate(
            role=PricingRole.GENERATION,
            provider="provider-a",
            model="generation-v1",
            source_reference=_MODEL_SOURCE,
        )
    with pytest.raises(ValidationError, match="experiment_pricing_rate_source_unregistered"):
        ExperimentPricingProvenance(
            pricing_version="pricing-v1",
            pricing_hash=_HASH_A,
            currency="USD",
            source_references=(_MODEL_SOURCE,),
            rate_card=(
                ExperimentPricingRate(
                    role=PricingRole.EMBEDDING,
                    provider="provider-a",
                    model="embedding-v1",
                    input_per_million=Decimal("0.02"),
                    source_reference=_EMBEDDING_SOURCE,
                ),
            ),
        )
    with pytest.raises(ValidationError, match="experiment_selection_tie_breaker_duplicate"):
        DeterministicSelectionPolicy(
            policy_id="duplicate-tie-breaker",
            policy_version="1.0.0",
            required_gate_ids=("quality",),
            tie_breakers=(
                SelectionCriterion(
                    metric="faithfulness",
                    direction=SelectionDirection.MAXIMIZE,
                ),
                SelectionCriterion(
                    metric="faithfulness",
                    direction=SelectionDirection.MINIMIZE,
                ),
            ),
            final_tie_break=FinalTieBreak.VARIANT_ORDER,
        )


def test_registry_validates_unique_plans_and_provides_safe_lookup() -> None:
    model_plan = _plan()
    retrieval_plan = _plan(
        plan_id="retrieval-comparison-v1",
        display_name="Retrieval comparison",
        axis=ExperimentAxis.RETRIEVAL_STRATEGY,
        fixed_identities=_fixed_identities(
            (
                FixedIdentity(name="generation.model", value="generation-v2"),
                FixedIdentity(name="prompt.generation", value="grounded-claims-json-v2"),
            )
        ),
        variants=(
            ExperimentVariant(
                variant_id="dense",
                display_name="Dense",
                axis_value="dense",
                configuration_id="configuration-dense",
            ),
            ExperimentVariant(
                variant_id="hybrid",
                display_name="Hybrid",
                axis_value="hybrid",
                configuration_id="configuration-hybrid",
            ),
            ExperimentVariant(
                variant_id="hybrid-rerank",
                display_name="Hybrid plus rerank",
                axis_value="hybrid-rerank",
                configuration_id="configuration-hybrid-rerank",
            ),
        ),
        baseline_variant_id="dense",
    )
    registry = ExperimentPlanRegistry(plans=(model_plan, retrieval_plan))

    assert registry.resolve(model_plan.plan_id) is model_plan
    assert registry.get("missing") is None
    assert registry.list() == (model_plan, retrieval_plan)
    assert registry.list(axis="retrieval-strategy") == (retrieval_plan,)
    with pytest.raises(ExperimentPlanRegistryError, match="experiment_plan_not_found"):
        registry.resolve("missing")
    with pytest.raises(ExperimentPlanRegistryError, match="experiment_plan_id_invalid"):
        registry.resolve("../unsafe")
    with pytest.raises(ExperimentPlanRegistryError, match="experiment_axis_invalid"):
        registry.list(axis="unknown")
    with pytest.raises(ValidationError, match="experiment_plan_registry_duplicate_id"):
        ExperimentPlanRegistry(plans=(model_plan, model_plan))
    with pytest.raises(ValidationError, match="experiment_plan_registry_empty"):
        ExperimentPlanRegistry(plans=())
