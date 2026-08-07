from __future__ import annotations

from pathlib import Path

import pytest
from test_experiment import _plan

from rag_mvp.config.settings import Settings
from rag_mvp.domain.retrieval import CachePolicy, RetrievalMode
from rag_mvp.evaluation.comparison_schedule import (
    ComparisonScheduleError,
    build_comparison_schedule,
    materialize_variant_cases,
)
from rag_mvp.evaluation.experiment import (
    ExperimentAxis,
    ExperimentOrderPolicy,
    ExperimentVariant,
)
from rag_mvp.evaluation.plan import EvaluationDatasetRegistry, build_evaluation_plan
from rag_mvp.evaluation.report_builder import case_ids_content_hash

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DATASETS_ROOT = _REPOSITORY_ROOT / "evaluations" / "datasets"


def _dataset():
    return EvaluationDatasetRegistry(_DATASETS_ROOT).resolve(
        "original-pdf-acceptance",
        "2.0.0",
    )


def _bound_plan(dataset, **updates: object):
    source = _plan()
    case_ids = tuple(item.case_id for item in dataset.cases)
    fixed = source.fixed_identities.model_copy(
        update={
            "dataset_id": dataset.manifest.dataset_id,
            "dataset_version": dataset.manifest.version,
            "dataset_hash": dataset.manifest.content_hash,
            "corpus_id": dataset.corpus.manifest.snapshot_id,
            "corpus_version": dataset.corpus.manifest.version,
            "corpus_hash": dataset.corpus.manifest.content_hash,
            "case_set_hash": case_ids_content_hash(case_ids),
            "case_count": len(case_ids),
        }
    )
    return type(source).create(
        **{
            **source.model_dump(exclude={"content_hash"}),
            "fixed_identities": fixed,
            **updates,
        }
    )


def test_seeded_interleaving_is_repeatable_and_complete() -> None:
    dataset = _dataset()
    source = _bound_plan(dataset)
    plan = type(source).create(
        **{
            **source.model_dump(exclude={"content_hash"}),
            "repeat_order_policy": source.repeat_order_policy.model_copy(
                update={
                    "repeats_per_case": 2,
                    "order_policy": ExperimentOrderPolicy.SEEDED_INTERLEAVED,
                    "seed": 781,
                }
            ),
        }
    )

    first = build_comparison_schedule(plan, dataset)
    second = build_comparison_schedule(plan, dataset)

    assert first == second
    assert len(first.steps) == len(dataset.cases) * 2 * len(plan.variants)
    assert {item.variant_id for item in first.steps[:4]} == {"model-v1", "model-v2"}
    assert len({item.execution_case_id for item in first.steps}) == len(dataset.cases) * 2
    assert first.case_set_hash == case_ids_content_hash(
        tuple(item.case_id for item in dataset.cases)
    )


def test_cache_schedule_keeps_cold_then_warm_on_one_equivalent_order() -> None:
    dataset = _dataset()
    source = _bound_plan(dataset)
    variants = (
        source.variants[0].model_copy(update={"variant_id": "cold", "axis_value": "cold"}),
        source.variants[1].model_copy(update={"variant_id": "warm", "axis_value": "warm"}),
    )
    plan = type(source).create(
        **{
            **source.model_dump(exclude={"content_hash"}),
            "axis": ExperimentAxis.CACHE_BEHAVIOR,
            "variants": variants,
            "baseline_variant_id": "cold",
            "cache_policy": CachePolicy.USE,
            "repeat_order_policy": source.repeat_order_policy.model_copy(
                update={
                    "order_policy": ExperimentOrderPolicy.SEEDED_INTERLEAVED,
                    "seed": 45,
                }
            ),
        }
    )

    schedule = build_comparison_schedule(plan, dataset)
    phase_size = len(dataset.cases) * plan.repeat_order_policy.repeats_per_case
    cold = tuple(item.dataset_case_id for item in schedule.steps[:phase_size])
    warm = tuple(item.dataset_case_id for item in schedule.steps[phase_size:])

    assert {item.variant_id for item in schedule.steps[:phase_size]} == {"cold"}
    assert {item.variant_id for item in schedule.steps[phase_size:]} == {"warm"}
    assert cold == warm


def test_repeat_materialization_uses_unique_persisted_ids_with_source_identity() -> None:
    dataset = _dataset()
    source = _bound_plan(dataset)
    plan = type(source).create(
        **{
            **source.model_dump(exclude={"content_hash"}),
            "repeat_order_policy": source.repeat_order_policy.model_copy(
                update={"repeats_per_case": 2}
            ),
        }
    )
    schedule = build_comparison_schedule(plan, dataset)
    base = build_evaluation_plan(dataset, Settings(_env_file=None), "base-evaluation-run")

    materialized = materialize_variant_cases(
        schedule,
        plan,
        dataset,
        "model-v1",
        base.cases,
    )

    assert len(materialized) == len(dataset.cases) * 2
    assert len({item.case_id for item in materialized}) == len(materialized)
    assert tuple(item.source_case_id for item in materialized).count(
        dataset.cases[0].case_id
    ) == 2
    assert {item.repeat_index for item in materialized} == {0, 1}
    expected = tuple(
        (item.repetition, item.dataset_case_id, item.execution_case_id)
        for item in schedule.steps
        if item.variant_id == "model-v1"
    )
    assert tuple(
        (item.repeat_index, item.source_case_id, item.case_id) for item in materialized
    ) == expected


def test_retrieval_variant_is_applied_to_every_materialized_case() -> None:
    dataset = _dataset()
    source = _bound_plan(dataset)
    controlled = tuple(
        item for item in source.fixed_identities.controlled if item.name != "retrieval.mode"
    )
    variants = (
        ExperimentVariant(
            variant_id="dense",
            display_name="Dense",
            axis_value="dense",
            configuration_id="dense-configuration",
        ),
        ExperimentVariant(
            variant_id="hybrid-rerank",
            display_name="Hybrid rerank",
            axis_value="hybrid-rerank",
            configuration_id="hybrid-rerank-configuration",
        ),
    )
    plan = type(source).create(
        **{
            **source.model_dump(exclude={"content_hash"}),
            "axis": ExperimentAxis.RETRIEVAL_STRATEGY,
            "fixed_identities": source.fixed_identities.model_copy(
                update={"controlled": controlled}
            ),
            "variants": variants,
            "baseline_variant_id": "dense",
        }
    )
    schedule = build_comparison_schedule(plan, dataset)
    base = build_evaluation_plan(dataset, Settings(_env_file=None), "retrieval-base-run")

    dense = materialize_variant_cases(schedule, plan, dataset, "dense", base.cases)
    reranked = materialize_variant_cases(
        schedule,
        plan,
        dataset,
        "hybrid-rerank",
        base.cases,
    )

    assert {item.retrieval_mode for item in dense} == {RetrievalMode.DENSE}
    assert {item.retrieval_mode for item in reranked} == {RetrievalMode.HYBRID_RERANK}


def test_materialization_rejects_tampered_schedule_or_dataset_binding() -> None:
    dataset = _dataset()
    plan = _bound_plan(dataset)
    schedule = build_comparison_schedule(plan, dataset)
    base = build_evaluation_plan(dataset, Settings(_env_file=None), "tamper-base-run")
    tampered = schedule.model_copy(update={"seed": schedule.seed + 1})

    with pytest.raises(ComparisonScheduleError, match="comparison_schedule_plan_mismatch"):
        materialize_variant_cases(
            tampered,
            plan,
            dataset,
            plan.variants[0].variant_id,
            base.cases,
        )

    foreign = plan.fixed_identities.model_copy(
        update={"dataset_hash": "sha256:" + ("f" * 64)}
    )
    foreign_plan = type(plan).create(
        **{
            **plan.model_dump(exclude={"content_hash"}),
            "fixed_identities": foreign,
        }
    )
    with pytest.raises(ComparisonScheduleError, match="comparison_schedule_case_set_mismatch"):
        build_comparison_schedule(foreign_plan, dataset)


def test_declared_subset_keeps_full_dataset_identity_and_materializes_only_selected() -> None:
    dataset = _dataset()
    selected = tuple(item.case_id for item in dataset.cases[:3])
    source = _bound_plan(dataset)
    plan = type(source).create(
        **{
            **source.model_dump(exclude={"content_hash"}),
            "fixed_identities": source.fixed_identities.model_copy(
                update={
                    "case_count": len(selected),
                    "case_set_hash": case_ids_content_hash(selected),
                }
            ),
        }
    )

    with pytest.raises(
        ComparisonScheduleError,
        match="comparison_schedule_case_set_mismatch",
    ):
        build_comparison_schedule(plan, dataset)

    schedule = build_comparison_schedule(
        plan,
        dataset,
        selected_case_ids=selected,
    )
    base = build_evaluation_plan(dataset, Settings(_env_file=None), "subset-base-run")
    materialized = materialize_variant_cases(
        schedule,
        plan,
        dataset,
        plan.variants[0].variant_id,
        base.cases,
    )

    assert schedule.dataset_hash == dataset.manifest.content_hash
    assert schedule.dataset_case_ids == selected
    assert tuple(item.source_case_id for item in materialized) == tuple(
        item.dataset_case_id
        for item in schedule.steps
        if item.variant_id == plan.variants[0].variant_id
    )
    assert set(item.source_case_id for item in materialized) == set(selected)


def test_declared_subset_rejects_noncanonical_or_foreign_case_ids() -> None:
    dataset = _dataset()
    selected = tuple(item.case_id for item in dataset.cases[:2])
    source = _bound_plan(dataset)
    plan = type(source).create(
        **{
            **source.model_dump(exclude={"content_hash"}),
            "fixed_identities": source.fixed_identities.model_copy(
                update={
                    "case_count": len(selected),
                    "case_set_hash": case_ids_content_hash(selected),
                }
            ),
        }
    )

    for invalid in ((selected[1], selected[0]), (*selected, "foreign-case")):
        with pytest.raises(
            ComparisonScheduleError,
            match="comparison_schedule_case_subset_invalid",
        ):
            build_comparison_schedule(
                plan,
                dataset,
                selected_case_ids=invalid,
            )
