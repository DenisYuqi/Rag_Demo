from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pytest
from test_comparison_evidence import _experiment_reference, _run_fixture

from rag_mvp.domain.evaluation import EvaluationRun, EvaluationRunStatus
from rag_mvp.domain.retrieval import CachePolicy, RetrievalMode
from rag_mvp.evaluation.comparison_preflight import (
    MAXIMUM_CACHE_TTL_SECONDS,
    ComparisonPreflightError,
    _require_unique_cache_queries,
    _rerank_input_byte_bound,
    minimum_cache_experiment_ttl_seconds,
    preflight_comparison_work,
)
from rag_mvp.evaluation.comparison_schedule import (
    build_comparison_schedule,
    cache_eligible_case_ids,
    materialize_variant_cases,
)
from rag_mvp.evaluation.report_builder import case_ids_content_hash
from rag_mvp.evaluation.runner import EvaluationRunner
from rag_mvp.retrieval.cache import BoundedCacheLookupStatus, BoundedTtlCache


def _candidate_plans(
    *,
    reranking: bool = False,
    unknown_prompt: bool = False,
    forged_scorer: str | None = None,
):
    dataset, first, _ = _run_fixture()
    identity = first.identity
    retrieval = dict(identity.retrieval_configuration)
    providers = dict(identity.provider_identities)
    models = dict(identity.model_identities)
    prompts = dict(identity.prompt_versions)
    scorers = dict(identity.scorer_versions)
    if forged_scorer is not None:
        scorers[forged_scorer] = "forged-scorer-v999"
    cases = first.cases
    if reranking:
        retrieval.update({"mode": "hybrid-rerank", "reranking_enabled": True})
        providers["reranking"] = "provider-a"
        models["reranking"] = "reranking-v1"
        cases = tuple(
            item.model_copy(update={"retrieval_mode": RetrievalMode.HYBRID_RERANK})
            for item in cases
        )
    if unknown_prompt:
        prompts["generation"] = "future-generation-prompt"
    first = first.model_copy(
        update={
            "cases": cases,
            "identity": identity.model_copy(
                update={
                    "retrieval_configuration": retrieval,
                    "provider_identities": providers,
                    "model_identities": models,
                    "prompt_versions": prompts,
                    "scorer_versions": scorers,
                }
            ),
        }
    )
    experiment, _ = _experiment_reference(dataset, first)
    second_variant = experiment.variants[1]
    second = first.model_copy(
        update={
            "run_id": "candidate-run-2",
            "identity": first.identity.model_copy(
                update={
                    "configuration_id": second_variant.configuration_id,
                    "model_identities": {
                        **first.identity.model_identities,
                        "generation": second_variant.axis_value,
                    },
                }
            ),
        }
    )
    return dataset, experiment, {
        experiment.variants[0].variant_id: first,
        second_variant.variant_id: second,
    }


def _cache_candidate_plans(
    *,
    maximum_entries: int = 64,
    ttl_seconds: float = 1_000.0,
    runtime_ids: tuple[str, str] = ("shared-runtime", "shared-runtime"),
):
    dataset, first, _ = _run_fixture()
    eligible = cache_eligible_case_ids(dataset)
    identity = first.identity.model_copy(
        update={
            "cache_policy": CachePolicy.USE,
            "runtime_configuration_id": runtime_ids[0],
            "retrieval_configuration": {
                **first.identity.retrieval_configuration,
                "retrieval_cache_enabled": True,
                "retrieval_cache_max_entries": maximum_entries,
                "retrieval_cache_ttl_seconds": ttl_seconds,
            },
        }
    )
    first = first.model_copy(update={"identity": identity})
    experiment, _ = _experiment_reference(
        dataset,
        first,
        cache_behavior="cold",
    )
    experiment = type(experiment).create(
        **{
            **experiment.model_dump(exclude={"content_hash"}),
            "fixed_identities": experiment.fixed_identities.model_copy(
                update={
                    "case_count": len(eligible),
                    "case_set_hash": case_ids_content_hash(eligible),
                }
            ),
        }
    )
    schedule = build_comparison_schedule(
        experiment,
        dataset,
        selected_case_ids=eligible,
    )
    candidates = {}
    for index, variant in enumerate(experiment.variants):
        run_id = f"cache-candidate-{variant.axis_value}"
        cases = materialize_variant_cases(
            schedule,
            experiment,
            dataset,
            variant.variant_id,
            first.cases,
        )
        candidates[variant.variant_id] = first.model_copy(
            update={
                "run_id": run_id,
                "cases": cases,
                "identity": first.identity.model_copy(
                    update={
                        "configuration_id": variant.configuration_id,
                        "runtime_configuration_id": runtime_ids[index],
                    }
                ),
            }
        )
    return dataset, experiment, candidates, eligible


@dataclass
class _SuiteRunRepository:
    values: dict[str, EvaluationRun] = field(default_factory=dict)

    def create(self, run: EvaluationRun) -> None:
        if run.run_id in self.values:
            raise ValueError("duplicate")
        self.values[run.run_id] = run

    def get(self, run_id: str) -> EvaluationRun | None:
        return self.values.get(run_id)

    def update(self, run: EvaluationRun) -> None:
        self.values[run.run_id] = run


@dataclass
class _NoCallExecutor:
    calls: int = 0

    async def execute(self, *args: object, **kwargs: object) -> object:
        self.calls += 1
        raise AssertionError("provider work must not start during admission")


def test_preflight_reserves_complete_suite_and_reuses_one_exact_index() -> None:
    dataset, experiment, candidates = _candidate_plans()

    preflight = preflight_comparison_work(
        "comparison-preflight",
        experiment,
        dataset,
        candidates,
    )

    assert preflight.logical_attempt_count == len(dataset.cases) * len(candidates)
    assert preflight.index_build_count == 1
    assert preflight.snapshot.reservation_count == preflight.logical_attempt_count + 1
    assert preflight.snapshot.reserved_provider_calls > preflight.logical_attempt_count
    assert Decimal(0) < preflight.snapshot.reserved_cost <= experiment.maximum_cost


@pytest.mark.parametrize(
    "scorer_name",
    [
        "faithfulness",
        "faithfulness-text-matcher",
        "faithfulness-text-normalization",
        "context-precision",
        "answer-completeness",
        "style-consistency",
        "refusal-appropriateness",
        "answer-compliance",
        "scoring-pipeline",
        "quality-gate",
        "advanced-quality-gate",
    ],
)
def test_preflight_rejects_self_consistent_forged_scorer_before_reservation(
    scorer_name: str,
) -> None:
    dataset, experiment, candidates = _candidate_plans(forged_scorer=scorer_name)

    with pytest.raises(
        ComparisonPreflightError,
        match="comparison_scorer_version_mismatch",
    ):
        preflight_comparison_work(
            "comparison-forged-scorer",
            experiment,
            dataset,
            candidates,
        )


@pytest.mark.parametrize(
    "identity",
    [
        "dataset-id",
        "dataset-version",
        "dataset-hash",
        "corpus-id",
        "corpus-version",
        "corpus-hash",
    ],
)
def test_preflight_rejects_manifest_valid_but_unpinned_dataset_identity(
    identity: str,
) -> None:
    dataset, experiment, candidates = _candidate_plans()
    if identity.startswith("dataset-"):
        field_name = {
            "dataset-id": "dataset_id",
            "dataset-version": "version",
            "dataset-hash": "content_hash",
        }[identity]
        dataset = dataset.model_copy(
            update={
                "manifest": dataset.manifest.model_copy(
                    update={
                        field_name: (
                            "sha256:" + "f" * 64
                            if field_name == "content_hash"
                            else "foreign"
                        )
                    }
                )
            }
        )
    else:
        field_name = {
            "corpus-id": "snapshot_id",
            "corpus-version": "version",
            "corpus-hash": "content_hash",
        }[identity]
        corpus = dataset.corpus.model_copy(
            update={
                "manifest": dataset.corpus.manifest.model_copy(
                    update={
                        field_name: (
                            "sha256:" + "e" * 64
                            if field_name == "content_hash"
                            else "foreign"
                        )
                    }
                )
            }
        )
        dataset = dataset.model_copy(update={"corpus": corpus})

    with pytest.raises(
        ComparisonPreflightError,
        match="comparison_registered_dataset_identity_mismatch",
    ):
        preflight_comparison_work(
            "comparison-unpinned-dataset",
            experiment,
            dataset,
            candidates,
        )


@pytest.mark.parametrize("cap", ["calls", "cost"])
def test_preflight_rejects_whole_suite_cap_before_work(cap: str) -> None:
    dataset, experiment, candidates = _candidate_plans()
    accepted = preflight_comparison_work(
        "comparison-cap-baseline",
        experiment,
        dataset,
        candidates,
    )
    updates: dict[str, object]
    expected: str
    if cap == "calls":
        updates = {
            "maximum_provider_calls": accepted.snapshot.reserved_provider_calls - 1
        }
        expected = "provider_call_cap_exceeded"
    else:
        updates = {"maximum_cost": accepted.snapshot.reserved_cost - Decimal("0.00000001")}
        expected = "provider_cost_cap_exceeded"
    limited = type(experiment).create(
        **{
            **experiment.model_dump(exclude={"content_hash"}),
            **updates,
        }
    )

    with pytest.raises(ComparisonPreflightError, match=expected):
        preflight_comparison_work(
            "comparison-cap-rejected",
            limited,
            dataset,
            candidates,
        )


def test_preflight_fails_closed_without_exact_reranking_rate() -> None:
    dataset, experiment, candidates = _candidate_plans(reranking=True)

    with pytest.raises(
        ComparisonPreflightError,
        match="comparison_exact_pricing_missing",
    ):
        preflight_comparison_work(
            "comparison-reranking-price",
            experiment,
            dataset,
            candidates,
        )


def test_preflight_fails_closed_when_prompt_overhead_version_is_unknown() -> None:
    dataset, experiment, candidates = _candidate_plans(unknown_prompt=True)

    with pytest.raises(
        ComparisonPreflightError,
        match="comparison_prompt_overhead_unknown",
    ):
        preflight_comparison_work(
            "comparison-prompt-version",
            experiment,
            dataset,
            candidates,
        )


def test_rerank_preflight_uses_truncated_utf8_byte_bound_for_max_cjk_candidates() -> None:
    dataset, _, candidates = _candidate_plans(reranking=True)
    plan = next(iter(candidates.values()))
    submitted = int(plan.identity.retrieval_configuration["rerank_candidate_limit"])

    def with_chunk_text(text: str):
        chunks = tuple(item.model_copy(update={"text": text}) for item in dataset.corpus.chunks)
        corpus = dataset.corpus.model_copy(update={"chunks": chunks})
        return dataset.model_copy(update={"corpus": corpus})

    ascii_bound = _rerank_input_byte_bound(
        plan,
        with_chunk_text("a" * 512),
        submitted,
    )
    cjk_bound = _rerank_input_byte_bound(
        plan,
        with_chunk_text("界" * 512),
        submitted,
    )
    included = min(submitted, len(dataset.corpus.chunks))

    assert cjk_bound - ascii_bound == included * 512 * 2
    assert cjk_bound == 8_192 + (256 * 4) + (included * 512 * 3)


def test_cache_preflight_binds_eligible_subset_capacity_ttl_and_shared_runtime() -> None:
    dataset, experiment, candidates, eligible = _cache_candidate_plans()

    preflight = preflight_comparison_work(
        "comparison-cache",
        experiment,
        dataset,
        candidates,
    )

    assert preflight.cache_eligible_case_count == len(eligible)
    assert preflight.logical_attempt_count == len(eligible) * 2
    assert preflight.minimum_cache_ttl_seconds == minimum_cache_experiment_ttl_seconds(
        len(eligible),
        9.5,
    )


def test_cache_window_survives_delayed_warm_phase_and_expires_at_declared_bound() -> None:
    eligible_count = 24
    deadline = 9.5
    old_provider_only_bound = eligible_count * deadline
    minimum = minimum_cache_experiment_ttl_seconds(eligible_count, deadline)
    assert minimum > old_provider_only_bound

    now = [0.0]
    old_cache: BoundedTtlCache[str] = BoundedTtlCache(
        maximum_entries=eligible_count,
        ttl_seconds=old_provider_only_bound,
        clock=lambda: now[0],
    )
    guarded_cache: BoundedTtlCache[str] = BoundedTtlCache(
        maximum_entries=eligible_count,
        ttl_seconds=minimum,
        clock=lambda: now[0],
    )
    old_cache.put("first-cold", "retrieval-evidence")
    guarded_cache.put("first-cold", "retrieval-evidence")

    now[0] = old_provider_only_bound + 1.0
    assert old_cache.lookup("first-cold").status is BoundedCacheLookupStatus.EXPIRED
    assert guarded_cache.lookup("first-cold").status is BoundedCacheLookupStatus.HIT
    now[0] = minimum - 0.001
    assert guarded_cache.lookup("first-cold").status is BoundedCacheLookupStatus.HIT
    now[0] = minimum
    assert guarded_cache.lookup("first-cold").status is BoundedCacheLookupStatus.EXPIRED


def test_cache_window_rejects_unrepresentable_ttl() -> None:
    with pytest.raises(
        ComparisonPreflightError,
        match="comparison_cache_ttl_window_unrepresentable",
    ):
        minimum_cache_experiment_ttl_seconds(10_000, MAXIMUM_CACHE_TTL_SECONDS)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("capacity", "comparison_cache_capacity_insufficient"),
        ("ttl", "comparison_cache_ttl_insufficient"),
        ("runtime", "comparison_cache_runtime_identity_mismatch"),
    ),
)
def test_cache_preflight_fails_closed_before_work(
    mutation: str,
    expected: str,
) -> None:
    kwargs: dict[str, object] = {}
    if mutation == "capacity":
        kwargs["maximum_entries"] = 1
    elif mutation == "ttl":
        kwargs["ttl_seconds"] = 1.0
    else:
        kwargs["runtime_ids"] = ("cold-runtime", "warm-runtime")
    dataset, experiment, candidates, _ = _cache_candidate_plans(**kwargs)

    with pytest.raises(ComparisonPreflightError, match=expected):
        preflight_comparison_work(
            "comparison-cache-invalid",
            experiment,
            dataset,
            candidates,
        )


def test_cache_preflight_rejects_ineligible_full_acceptance_case_set() -> None:
    dataset, first, _ = _run_fixture()
    first = first.model_copy(
        update={
            "identity": first.identity.model_copy(
                update={
                    "cache_policy": CachePolicy.USE,
                    "runtime_configuration_id": "shared-runtime",
                    "retrieval_configuration": {
                        **first.identity.retrieval_configuration,
                        "retrieval_cache_enabled": True,
                        "retrieval_cache_max_entries": 64,
                        "retrieval_cache_ttl_seconds": 1_000.0,
                    },
                }
            )
        }
    )
    experiment, _ = _experiment_reference(
        dataset,
        first,
        cache_behavior="cold",
    )
    warm = first.model_copy(
        update={
            "run_id": "cache-full-warm",
            "identity": first.identity.model_copy(
                update={"configuration_id": experiment.variants[1].configuration_id}
            ),
        }
    )
    candidates = {
        experiment.variants[0].variant_id: first,
        experiment.variants[1].variant_id: warm,
    }

    with pytest.raises(
        ComparisonPreflightError,
        match="comparison_cache_case_set_ineligible",
    ):
        preflight_comparison_work(
            "comparison-cache-ineligible",
            experiment,
            dataset,
            candidates,
        )


def test_cache_query_uniqueness_uses_exact_history_sensitive_rewrite() -> None:
    dataset, _, _ = _run_fixture()
    all_multi_turn = tuple(case for case in dataset.cases if case.history)
    first = all_multi_turn[0]
    second = next(case for case in all_multi_turn[1:] if case.history != first.history)
    multi_turn = (first, second)
    assert len(multi_turn) == 2
    same_follow_up = tuple(
        case.model_copy(update={"question": "What about this policy?"})
        for case in multi_turn
    )
    history_sensitive = dataset.model_copy(update={"cases": same_follow_up})
    case_ids = tuple(case.case_id for case in same_follow_up)

    _require_unique_cache_queries(history_sensitive, case_ids)

    duplicate_history = same_follow_up[1].model_copy(
        update={"history": same_follow_up[0].history}
    )
    converged = dataset.model_copy(
        update={"cases": (same_follow_up[0], duplicate_history)}
    )
    with pytest.raises(
        ComparisonPreflightError,
        match="comparison_cache_query_duplicate",
    ):
        _require_unique_cache_queries(converged, case_ids)


def test_multi_candidate_runners_consume_one_atomic_pre_reserved_suite_ledger(
    tmp_path: Path,
) -> None:
    dataset, experiment, candidates = _candidate_plans()
    preflight = preflight_comparison_work(
        "comparison-shared-ledger",
        experiment,
        dataset,
        candidates,
    )
    reserved = preflight.budget.snapshot()
    repository = _SuiteRunRepository()
    executor = _NoCallExecutor()
    runners = []
    offset = 0
    for variant in experiment.variants:
        plan = candidates[variant.variant_id]
        count = len(plan.cases)
        estimates = preflight.estimates[offset : offset + count]
        offset += count
        by_case = dict(zip((item.case_id for item in plan.cases), estimates, strict=True))
        runner = EvaluationRunner(
            repository,
            tmp_path / "evaluations",
            executor,
            work_budget=preflight.budget,
            case_work_estimator=lambda case, values=by_case: values[case.case_id],
            work_reservations_prepared=True,
        )
        runner.queue(plan)
        runners.append((runner, plan))

    assert set(repository.values) == {item.run_id for item in candidates.values()}
    assert {item.status for item in repository.values.values()} == {
        EvaluationRunStatus.QUEUED
    }
    for runner, plan in runners:
        runner.start(plan)

    assert preflight.budget.snapshot() == reserved
    assert reserved.reservation_count == preflight.logical_attempt_count + 1
    assert executor.calls == 0
