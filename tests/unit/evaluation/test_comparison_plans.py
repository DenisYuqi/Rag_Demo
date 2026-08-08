from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic import SecretStr

from rag_mvp.config.settings import Settings
from rag_mvp.domain.retrieval import CachePolicy
from rag_mvp.evaluation.comparison import (
    COMPARISON_SELECTION_ELIGIBILITY_GATE_ID,
)
from rag_mvp.evaluation.comparison_plans import (
    REGISTERED_CACHE_PLAN_ID,
    REGISTERED_GENERATION_PLAN_ID,
    REGISTERED_RETRIEVAL_PLAN_ID,
    ComparisonPlanCatalogContext,
    ComparisonPlanMaterializationContext,
    RegisteredComparisonPlanError,
    RegisteredComparisonPlanRegistry,
    UpstreamComparisonSelection,
    UpstreamSelections,
)
from rag_mvp.evaluation.comparison_preflight import minimum_cache_experiment_ttl_seconds
from rag_mvp.evaluation.comparison_schedule import cache_eligible_case_ids
from rag_mvp.evaluation.experiment import ExperimentAxis, FixedIdentity, PricingRole
from rag_mvp.evaluation.plan import EvaluationDatasetRegistry
from rag_mvp.evaluation.report_builder import case_ids_content_hash

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DATASETS_ROOT = _REPOSITORY_ROOT / "evaluations" / "datasets"
_PRICING_PATH = (
    _REPOSITORY_ROOT / "evaluations" / "pricing" / "openai-comparison-standard-2026-08-07-v1.json"
)
_SHA_A = "sha256:" + ("a" * 64)
_SHA_B = "sha256:" + ("b" * 64)
_OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$")


def _dataset():
    return EvaluationDatasetRegistry(_DATASETS_ROOT).resolve(
        "original-pdf-acceptance",
        "2.0.0",
    )


def _settings(data_root: Path, **updates: object) -> Settings:
    return Settings(
        _env_file=None,
        provider_backend="openai",
        openai_api_key=SecretStr("unit-test-key"),
        environment="test",
        data_root=data_root,
        **updates,
    )


def _selection(
    axis: str,
    *,
    plan_id: str,
    axis_value: str,
    suffix: str,
    selected_variant_id: str,
    plan_content_hash: str = _SHA_A,
    result_content_hash: str = _SHA_B,
    upstream_identities: tuple[FixedIdentity, ...] = (),
) -> UpstreamComparisonSelection:
    return UpstreamComparisonSelection(
        axis=axis,
        comparison_id=f"comparison-{suffix}",
        plan_id=plan_id,
        plan_content_hash=plan_content_hash,
        result_content_hash=result_content_hash,
        selected_variant_id=selected_variant_id,
        selected_axis_value=axis_value,
        selected_configuration_id=f"configuration-{suffix}",
        selected_evaluation_run_id=f"evaluation-{suffix}",
        upstream_identities=upstream_identities,
    )


def _generation_selection(
    *,
    plan_id: str = REGISTERED_GENERATION_PLAN_ID,
    axis_value: str = "gpt-4.1-mini",
    selected_variant_id: str = "generation-gpt-4-1-mini",
    result_content_hash: str = _SHA_B,
):
    return _selection(
        "generation-model",
        plan_id=plan_id,
        axis_value=axis_value,
        suffix="generation",
        selected_variant_id=selected_variant_id,
        result_content_hash=result_content_hash,
    )


def _retrieval_selection(
    *,
    plan_id: str = REGISTERED_RETRIEVAL_PLAN_ID,
    generation: UpstreamComparisonSelection | None = None,
):
    selected_generation = generation or _generation_selection()
    prefix = "upstream.generation-model"
    return _selection(
        "retrieval-strategy",
        plan_id=plan_id,
        axis_value="hybrid-rerank",
        suffix="retrieval",
        selected_variant_id="retrieval-hybrid-rerank",
        upstream_identities=(
            FixedIdentity(name=f"{prefix}.plan-id", value=selected_generation.plan_id),
            FixedIdentity(
                name=f"{prefix}.plan-hash",
                value=selected_generation.plan_content_hash,
            ),
            FixedIdentity(
                name=f"{prefix}.result-hash",
                value=selected_generation.result_content_hash,
            ),
            FixedIdentity(
                name=f"{prefix}.variant-id",
                value=selected_generation.selected_variant_id,
            ),
            FixedIdentity(
                name=f"{prefix}.axis-value",
                value=selected_generation.selected_axis_value,
            ),
            FixedIdentity(
                name=f"{prefix}.configuration-id",
                value=selected_generation.selected_configuration_id,
            ),
            FixedIdentity(
                name=f"{prefix}.evaluation-run-id",
                value=selected_generation.selected_evaluation_run_id,
            ),
        ),
    )


def _run_ids(summary) -> dict[str, str]:
    return {
        item.variant_id: f"candidate-{index + 1}" for index, item in enumerate(summary.candidates)
    }


def test_catalog_lists_generation_and_blocks_missing_upstream_without_raising(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    registry = RegisteredComparisonPlanRegistry()

    summaries = registry.list(
        ComparisonPlanCatalogContext(
            dataset=dataset,
            settings=_settings(tmp_path / "catalog"),
        )
    )

    generation, retrieval, cache = summaries
    assert generation.plan_id == REGISTERED_GENERATION_PLAN_ID
    assert generation.launchable is True
    assert generation.blocking_codes == ()
    assert generation.case_count == len(dataset.cases)
    assert generation.repeats_per_case == 2
    assert generation.planned_logical_attempts == len(dataset.cases) * 2 * 2
    assert generation.plan_content_hash is not None
    assert generation.conservative_cost_estimate is not None
    assert generation.conservative_cost_estimate <= generation.maximum_cost
    assert retrieval.launchable is False
    assert retrieval.conservative_cost_estimate is None
    assert retrieval.blocking_codes == ("comparison-generation-selection-required",)
    assert cache.launchable is False
    assert cache.blocking_codes == (
        "comparison-generation-selection-required",
        "comparison-retrieval-selection-required",
    )
    for summary in summaries:
        assert _OPAQUE_ID.fullmatch(summary.plan_id)
        assert all(_OPAQUE_ID.fullmatch(item.variant_id) for item in summary.candidates)
    materialized = registry.resolve(
        REGISTERED_GENERATION_PLAN_ID,
        ComparisonPlanMaterializationContext(
            dataset=dataset,
            settings=_settings(tmp_path / "materialized"),
            comparison_id="comparison-selection-gate",
            candidate_run_ids=_run_ids(generation),
        ),
    )
    assert materialized.plan.gate_profile.mandatory_gate_ids == (
        COMPARISON_SELECTION_ELIGIBILITY_GATE_ID,
    )
    assert materialized.plan.selection_policy.required_gate_ids == (
        COMPARISON_SELECTION_ELIGIBILITY_GATE_ID,
    )


def test_catalog_blocks_manifest_valid_but_unpinned_acceptance_dataset(
    tmp_path: Path,
) -> None:
    source = _dataset()
    dataset = source.model_copy(
        update={
            "manifest": source.manifest.model_copy(update={"content_hash": "sha256:" + "f" * 64})
        }
    )
    registry = RegisteredComparisonPlanRegistry()
    settings = _settings(tmp_path / "unpinned-dataset")
    catalog_context = ComparisonPlanCatalogContext(
        dataset=dataset,
        settings=settings,
    )

    summaries = registry.list(catalog_context)

    assert all(not item.launchable for item in summaries)
    assert all(
        "comparison_registered_dataset_identity_mismatch" in item.blocking_codes
        for item in summaries
    )
    generation = summaries[0]
    with pytest.raises(
        RegisteredComparisonPlanError,
        match="comparison_registered_dataset_identity_mismatch",
    ):
        registry.resolve(
            REGISTERED_GENERATION_PLAN_ID,
            ComparisonPlanMaterializationContext(
                dataset=dataset,
                settings=settings,
                comparison_id="comparison-unpinned-dataset",
                candidate_run_ids=_run_ids(generation),
            ),
        )


def test_generation_plan_materializes_exact_models_and_bounded_work(tmp_path: Path) -> None:
    dataset = _dataset()
    settings = _settings(tmp_path / "suite-generation")
    registry = RegisteredComparisonPlanRegistry()
    summary = registry.list(ComparisonPlanCatalogContext(dataset=dataset, settings=settings))[0]

    materialized = registry.resolve(
        REGISTERED_GENERATION_PLAN_ID,
        ComparisonPlanMaterializationContext(
            dataset=dataset,
            settings=settings,
            comparison_id="comparison-generation",
            candidate_run_ids=_run_ids(summary),
        ),
    )

    assert materialized.plan.axis is ExperimentAxis.GENERATION_MODEL
    assert materialized.plan.cache_policy is CachePolicy.BYPASS
    assert tuple(item.axis_value for item in materialized.plan.variants) == (
        "gpt-4.1-mini",
        "gpt-5.4",
    )
    assert materialized.plan.gate_profile.profile_id == (
        f"{REGISTERED_GENERATION_PLAN_ID}-selection-eligibility-v2"
    )
    assert tuple(item.variant_id for item in materialized.plan.variants) == (
        "generation-gpt-4-1-mini",
        "generation-gpt-5-4",
    )
    assert materialized.selected_case_ids == tuple(item.case_id for item in dataset.cases)
    assert materialized.preflight.logical_attempt_count == len(dataset.cases) * 2 * 2
    assert (
        materialized.preflight.snapshot.reserved_provider_calls
        <= materialized.plan.maximum_provider_calls
    )
    assert materialized.preflight.snapshot.reserved_cost <= materialized.plan.maximum_cost
    for candidate, variant in zip(
        materialized.candidates,
        materialized.plan.variants,
        strict=True,
    ):
        assert candidate.settings.generation_model == variant.axis_value
        assert candidate.settings.data_root == settings.data_root
        assert candidate.evaluation_plan.identity.runtime_configuration_id == (
            candidate.settings.runtime_configuration_identity
        )
        assert candidate.identity_projection.identity_map()["generation.model"] == (
            variant.axis_value
        )
        assert len(candidate.evaluation_plan.cases) == len(dataset.cases) * 2


def test_semantic_plan_hash_is_path_independent_but_runtime_identity_is_not(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rag_mvp.evaluation.plan.source_code_revision",
        lambda: "sha256:" + ("c" * 64),
    )
    dataset = _dataset()
    registry = RegisteredComparisonPlanRegistry()
    settings_a = _settings(tmp_path / "suite-a")
    settings_b = _settings(tmp_path / "suite-b")
    summary = registry.list(ComparisonPlanCatalogContext(dataset=dataset, settings=settings_a))[0]
    run_ids = _run_ids(summary)

    first = registry.resolve(
        REGISTERED_GENERATION_PLAN_ID,
        ComparisonPlanMaterializationContext(
            dataset=dataset,
            settings=settings_a,
            comparison_id="comparison-path-a",
            candidate_run_ids=run_ids,
        ),
    )
    second = registry.resolve(
        REGISTERED_GENERATION_PLAN_ID,
        ComparisonPlanMaterializationContext(
            dataset=dataset,
            settings=settings_b,
            comparison_id="comparison-path-b",
            candidate_run_ids=run_ids,
        ),
    )

    assert first.plan.content_hash == second.plan.content_hash
    assert tuple(item.configuration_id for item in first.plan.variants) == tuple(
        item.configuration_id for item in second.plan.variants
    )
    first_runtime = tuple(
        item.evaluation_plan.identity.runtime_configuration_id for item in first.candidates
    )
    second_runtime = tuple(
        item.evaluation_plan.identity.runtime_configuration_id for item in second.candidates
    )
    assert first_runtime != second_runtime
    assert first_runtime == tuple(
        item.settings.runtime_configuration_identity for item in first.candidates
    )
    assert second_runtime == tuple(
        item.settings.runtime_configuration_identity for item in second.candidates
    )


def test_retrieval_plan_pins_upstream_selection_and_role_exact_pricing(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    settings = _settings(tmp_path / "suite-retrieval")
    registry = RegisteredComparisonPlanRegistry()
    upstream = UpstreamSelections(generation_model=_generation_selection())
    context = ComparisonPlanCatalogContext(
        dataset=dataset,
        settings=settings,
        upstream_selections=upstream,
    )
    summary = registry.list(context)[1]
    assert summary.launchable is True
    assert summary.conservative_cost_estimate is not None
    assert summary.conservative_cost_estimate <= summary.maximum_cost

    materialized = registry.resolve(
        REGISTERED_RETRIEVAL_PLAN_ID,
        ComparisonPlanMaterializationContext(
            dataset=dataset,
            settings=settings,
            upstream_selections=upstream,
            comparison_id="comparison-retrieval",
            candidate_run_ids=_run_ids(summary),
        ),
    )

    assert tuple(item.axis_value for item in materialized.plan.variants) == (
        "dense",
        "hybrid",
        "hybrid-rerank",
    )
    controlled = {item.name: item.value for item in materialized.plan.fixed_identities.controlled}
    assert controlled["upstream.generation-model.plan-id"] == (REGISTERED_GENERATION_PLAN_ID)
    assert controlled["upstream.generation-model.plan-hash"] == _SHA_A
    assert controlled["upstream.generation-model.result-hash"] == _SHA_B
    assert all(
        candidate.identity_projection.identity_map()["upstream.generation-model.result-hash"]
        == _SHA_B
        for candidate in materialized.candidates
    )
    assert tuple(
        item.evaluation_plan.identity.retrieval_configuration["mode"]
        for item in materialized.candidates
    ) == ("dense", "hybrid", "hybrid-rerank")
    rerank = materialized.candidates[-1]
    assert rerank.settings.reranking_model == "gpt-4.1-mini"
    assert rerank.evaluation_plan.identity.model_identities["reranking"] == ("gpt-4.1-mini")
    assert any(
        rate.role is PricingRole.RERANKING
        and rate.model == "gpt-4.1-mini"
        and rate.provider == rerank.evaluation_plan.identity.provider_identities["reranking"]
        for rate in materialized.plan.pricing.rate_card
    )


def test_foreign_same_axis_selection_is_nonlaunchable_and_rejected(tmp_path: Path) -> None:
    dataset = _dataset()
    settings = _settings(tmp_path / "foreign-selection")
    registry = RegisteredComparisonPlanRegistry()
    upstream = UpstreamSelections(
        generation_model=_generation_selection(plan_id="foreign-generation-plan")
    )
    context = ComparisonPlanCatalogContext(
        dataset=dataset,
        settings=settings,
        upstream_selections=upstream,
    )

    summary = registry.list(context)[1]
    assert summary.launchable is False
    assert "comparison-generation-selection-plan-mismatch" in summary.blocking_codes
    with pytest.raises(
        RegisteredComparisonPlanError,
        match="comparison-generation-selection-plan-mismatch",
    ):
        registry.resolve(
            REGISTERED_RETRIEVAL_PLAN_ID,
            ComparisonPlanMaterializationContext(
                dataset=dataset,
                settings=settings,
                upstream_selections=upstream,
                comparison_id="comparison-foreign",
                candidate_run_ids={
                    "retrieval-dense": "candidate-1",
                    "retrieval-hybrid": "candidate-2",
                    "retrieval-hybrid-rerank": "candidate-3",
                },
            ),
        )


def test_registered_selection_variant_must_match_its_axis_value(tmp_path: Path) -> None:
    dataset = _dataset()
    settings = _settings(tmp_path / "variant-mismatch")
    registry = RegisteredComparisonPlanRegistry()
    upstream = UpstreamSelections(
        generation_model=_generation_selection(
            selected_variant_id="generation-gpt-5-4",
        )
    )

    summary = registry.list(
        ComparisonPlanCatalogContext(
            dataset=dataset,
            settings=settings,
            upstream_selections=upstream,
        )
    )[1]

    assert summary.launchable is False
    assert "comparison-generation-selection-variant-mismatch" in summary.blocking_codes


def test_cache_plan_rejects_stale_retrieval_generation_chain(tmp_path: Path) -> None:
    dataset = _dataset()
    settings = _settings(tmp_path / "stale-chain")
    registry = RegisteredComparisonPlanRegistry()
    old_generation = _generation_selection(result_content_hash="sha256:" + ("c" * 64))
    latest_generation = _generation_selection()
    upstream = UpstreamSelections(
        generation_model=latest_generation,
        retrieval_strategy=_retrieval_selection(generation=old_generation),
    )

    summary = registry.list(
        ComparisonPlanCatalogContext(
            dataset=dataset,
            settings=settings,
            upstream_selections=upstream,
        )
    )[2]

    assert summary.launchable is False
    assert "comparison-retrieval-selection-generation-chain-mismatch" in summary.blocking_codes


def test_cache_plan_commits_eligible_subset_capacity_ttl_and_upstream_provenance(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    settings = _settings(
        tmp_path / "suite-cache",
        retrieval_cache_max_entries=1,
        retrieval_cache_ttl_seconds=1,
    )
    registry = RegisteredComparisonPlanRegistry()
    upstream = UpstreamSelections(
        generation_model=_generation_selection(),
        retrieval_strategy=_retrieval_selection(),
    )
    context = ComparisonPlanCatalogContext(
        dataset=dataset,
        settings=settings,
        upstream_selections=upstream,
    )
    summary = registry.list(context)[2]
    eligible = cache_eligible_case_ids(dataset)
    expected_ttl = minimum_cache_experiment_ttl_seconds(
        len(eligible),
        settings.qa_deadline_seconds,
    )
    assert summary.launchable is True
    assert summary.conservative_cost_estimate is not None
    assert summary.conservative_cost_estimate <= summary.maximum_cost
    assert summary.case_count == len(eligible)
    assert summary.case_set_hash == case_ids_content_hash(eligible)
    assert summary.cache_eligible_case_count == len(eligible)
    assert summary.cache_max_entries == len(eligible)
    assert summary.cache_ttl_seconds == expected_ttl

    materialized = registry.resolve(
        REGISTERED_CACHE_PLAN_ID,
        ComparisonPlanMaterializationContext(
            dataset=dataset,
            settings=settings,
            upstream_selections=upstream,
            comparison_id="comparison-cache",
            candidate_run_ids=_run_ids(summary),
        ),
    )

    assert materialized.selected_case_ids == eligible
    assert materialized.plan.fixed_identities.case_set_hash == case_ids_content_hash(eligible)
    assert materialized.plan.repeat_order_policy.repeats_per_case == 1
    assert materialized.plan.cache_policy is CachePolicy.USE
    phase_size = len(eligible)
    assert {item.variant_id for item in materialized.schedule.steps[:phase_size]} == {"cache-cold"}
    assert {item.variant_id for item in materialized.schedule.steps[phase_size:]} == {"cache-warm"}
    assert tuple(
        item.dataset_case_id for item in materialized.schedule.steps[:phase_size]
    ) == tuple(item.dataset_case_id for item in materialized.schedule.steps[phase_size:])
    runtime_ids = {
        item.evaluation_plan.identity.runtime_configuration_id for item in materialized.candidates
    }
    assert len(runtime_ids) == 1
    for candidate in materialized.candidates:
        assert candidate.settings.retrieval_cache_enabled is True
        assert candidate.settings.retrieval_cache_max_entries == len(eligible)
        assert candidate.settings.retrieval_cache_ttl_seconds == expected_ttl
        assert candidate.evaluation_plan.identity.cache_policy is CachePolicy.USE
        assert candidate.evaluation_plan.identity.retrieval_configuration[
            "retrieval_cache_max_entries"
        ] == len(eligible)
    controlled = {item.name: item.value for item in materialized.plan.fixed_identities.controlled}
    assert controlled["upstream.generation-model.result-hash"] == _SHA_B
    assert controlled["upstream.retrieval-strategy.result-hash"] == _SHA_B
    assert controlled["retrieval.retrieval_cache_max_entries"] == str(len(eligible))
    assert controlled["retrieval.retrieval_cache_ttl_seconds"] == str(expected_ttl)
    assert materialized.preflight.cache_eligible_case_count == len(eligible)
    assert materialized.preflight.minimum_cache_ttl_seconds == expected_ttl


def test_registry_rejects_candidate_run_set_and_modified_price_card(tmp_path: Path) -> None:
    dataset = _dataset()
    settings = _settings(tmp_path / "invalid")
    registry = RegisteredComparisonPlanRegistry()
    with pytest.raises(
        RegisteredComparisonPlanError,
        match="comparison-candidate-run-id-set-invalid",
    ):
        registry.resolve(
            REGISTERED_GENERATION_PLAN_ID,
            ComparisonPlanMaterializationContext(
                dataset=dataset,
                settings=settings,
                comparison_id="comparison-invalid-runs",
                candidate_run_ids={"generation-gpt-4-1-mini": "candidate-1"},
            ),
        )

    payload = json.loads(_PRICING_PATH.read_text(encoding="utf-8"))
    payload["rates"][0]["input_per_million"] = "999"
    modified = tmp_path / "modified-pricing.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RegisteredComparisonPlanError):
        RegisteredComparisonPlanRegistry(modified)
