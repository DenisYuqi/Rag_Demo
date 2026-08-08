"""Deterministic candidate/case schedules for registered comparison plans."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence

from pydantic import Field, model_validator

from rag_mvp.domain._base import DomainModel, Identifier
from rag_mvp.domain.retrieval import RetrievalMode
from rag_mvp.evaluation.dataset import EvaluationCaseV2, EvaluationDataset
from rag_mvp.evaluation.experiment import (
    ExperimentAxis,
    ExperimentOrderPolicy,
    ExperimentPlan,
)
from rag_mvp.evaluation.report_builder import case_ids_content_hash
from rag_mvp.evaluation.runner import EvaluationCaseInput


class ComparisonScheduleError(ValueError):
    """Stable fail-closed schedule/materialization error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ComparisonExecutionStep(DomainModel):
    sequence: int = Field(ge=0)
    variant_id: Identifier
    repetition: int = Field(ge=0)
    dataset_case_id: Identifier
    execution_case_id: Identifier


class ComparisonExecutionSchedule(DomainModel):
    plan_id: Identifier
    plan_content_hash: Identifier
    dataset_id: Identifier
    dataset_version: Identifier
    dataset_hash: Identifier
    case_set_hash: Identifier
    axis: ExperimentAxis
    variant_ids: tuple[Identifier, ...]
    dataset_case_ids: tuple[Identifier, ...]
    repeats_per_case: int = Field(ge=1)
    order_policy: ExperimentOrderPolicy
    seed: int = Field(ge=0)
    steps: tuple[ComparisonExecutionStep, ...]

    @model_validator(mode="after")
    def validate_schedule(self) -> ComparisonExecutionSchedule:
        if not self.steps:
            raise ValueError("comparison_schedule_empty")
        if tuple(item.sequence for item in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("comparison_schedule_sequence_invalid")
        if (
            not self.variant_ids
            or len(self.variant_ids) != len(set(self.variant_ids))
            or not self.dataset_case_ids
            or len(self.dataset_case_ids) != len(set(self.dataset_case_ids))
            or self.case_set_hash != case_ids_content_hash(self.dataset_case_ids)
        ):
            raise ValueError("comparison_schedule_identity_invalid")
        identities = tuple(
            (item.variant_id, item.repetition, item.dataset_case_id) for item in self.steps
        )
        expected = _ordered_work(
            axis=self.axis,
            variant_ids=self.variant_ids,
            case_ids=self.dataset_case_ids,
            repeats=self.repeats_per_case,
            policy=self.order_policy,
            seed=self.seed,
        )
        if identities != expected or len(identities) != len(set(identities)):
            raise ValueError("comparison_schedule_work_duplicate")
        if any(
            item.execution_case_id
            != _execution_case_id(
                item.dataset_case_id,
                item.repetition,
                self.repeats_per_case,
            )
            for item in self.steps
        ):
            raise ValueError("comparison_schedule_execution_identity_invalid")
        return self


def cache_eligible_case_ids(dataset: EvaluationDataset) -> tuple[str, ...]:
    """Select the deterministic v2 subset that always reaches retrieval."""

    values: list[str] = []
    for case in dataset.cases:
        if not isinstance(case, EvaluationCaseV2):
            raise ComparisonScheduleError("comparison_cache_dataset_v2_required")
        if set(case.refusal_expectation.reason_codes).intersection(
            {"prompt-injection", "safety", "unsafe-request"}
        ):
            continue
        values.append(case.case_id)
    if not values:
        raise ComparisonScheduleError("comparison_cache_case_set_empty")
    return tuple(values)


def build_comparison_schedule(
    plan: ExperimentPlan,
    dataset: EvaluationDataset,
    *,
    selected_case_ids: Sequence[str] | None = None,
) -> ComparisonExecutionSchedule:
    """Materialize a plan after binding a declared case subset to the full dataset."""

    plan.verify_hash()
    full_case_ids = tuple(item.case_id for item in dataset.cases)
    fixed = plan.fixed_identities
    corpus = dataset.corpus.manifest
    if (
        not full_case_ids
        or len(full_case_ids) != len(set(full_case_ids))
        or (
            fixed.dataset_id,
            fixed.dataset_version,
            fixed.dataset_hash,
        )
        != (
            dataset.manifest.dataset_id,
            dataset.manifest.version,
            dataset.manifest.content_hash,
        )
        or (fixed.corpus_id, fixed.corpus_version, fixed.corpus_hash)
        != (corpus.snapshot_id, corpus.version, corpus.content_hash)
    ):
        raise ComparisonScheduleError("comparison_schedule_case_set_mismatch")
    canonical_cases = _validated_selected_case_ids(
        full_case_ids,
        selected_case_ids,
    )
    if len(canonical_cases) != fixed.case_count or fixed.case_set_hash != case_ids_content_hash(
        canonical_cases
    ):
        raise ComparisonScheduleError("comparison_schedule_case_set_mismatch")
    variants = tuple(item.variant_id for item in plan.variants)
    repeats = plan.repeat_order_policy.repeats_per_case
    policy = plan.repeat_order_policy.order_policy
    work = _ordered_work(
        axis=plan.axis,
        variant_ids=variants,
        case_ids=canonical_cases,
        repeats=repeats,
        policy=policy,
        seed=plan.repeat_order_policy.seed,
    )
    steps = tuple(
        ComparisonExecutionStep(
            sequence=sequence,
            variant_id=variant_id,
            repetition=repetition,
            dataset_case_id=case_id,
            execution_case_id=_execution_case_id(case_id, repetition, repeats),
        )
        for sequence, (variant_id, repetition, case_id) in enumerate(work)
    )
    return ComparisonExecutionSchedule(
        plan_id=plan.plan_id,
        plan_content_hash=plan.content_hash,
        dataset_id=dataset.manifest.dataset_id,
        dataset_version=dataset.manifest.version,
        dataset_hash=dataset.manifest.content_hash,
        case_set_hash=fixed.case_set_hash,
        axis=plan.axis,
        variant_ids=variants,
        dataset_case_ids=canonical_cases,
        repeats_per_case=repeats,
        order_policy=policy,
        seed=plan.repeat_order_policy.seed,
        steps=steps,
    )


def materialize_variant_cases(
    schedule: ComparisonExecutionSchedule,
    plan: ExperimentPlan,
    dataset: EvaluationDataset,
    variant_id: str,
    cases: Sequence[EvaluationCaseInput],
) -> tuple[EvaluationCaseInput, ...]:
    """Expand one verified variant and apply its declared retrieval behavior."""

    if schedule != build_comparison_schedule(
        plan,
        dataset,
        selected_case_ids=schedule.dataset_case_ids,
    ):
        raise ComparisonScheduleError("comparison_schedule_plan_mismatch")
    by_id: Mapping[str, EvaluationCaseInput] = {item.case_id: item for item in cases}
    canonical_case_ids = tuple(item.case_id for item in dataset.cases)
    if len(by_id) != len(cases) or tuple(by_id) != canonical_case_ids:
        raise ComparisonScheduleError("comparison_schedule_case_set_mismatch")
    variant = next((item for item in plan.variants if item.variant_id == variant_id), None)
    if variant is None:
        raise ComparisonScheduleError("comparison_schedule_variant_mismatch")
    selected = tuple(item for item in schedule.steps if item.variant_id == variant_id)
    expected_count = len(schedule.dataset_case_ids) * plan.repeat_order_policy.repeats_per_case
    if len(selected) != expected_count or {item.dataset_case_id for item in selected} != set(
        schedule.dataset_case_ids
    ):
        raise ComparisonScheduleError("comparison_schedule_variant_mismatch")
    retrieval_mode: RetrievalMode | None = None
    if plan.axis is ExperimentAxis.RETRIEVAL_STRATEGY:
        try:
            retrieval_mode = RetrievalMode(variant.axis_value)
        except ValueError:
            raise ComparisonScheduleError("comparison_schedule_retrieval_mode_invalid") from None
    try:
        return tuple(
            EvaluationCaseInput.model_validate(
                by_id[step.dataset_case_id].model_copy(
                    update={
                        "case_id": step.execution_case_id,
                        "source_case_id": step.dataset_case_id,
                        "repeat_index": step.repetition,
                        "retrieval_mode": (
                            retrieval_mode or by_id[step.dataset_case_id].retrieval_mode
                        ),
                    }
                )
            )
            for step in selected
        )
    except KeyError:
        raise ComparisonScheduleError("comparison_schedule_case_set_mismatch") from None


def _validated_selected_case_ids(
    full_case_ids: tuple[str, ...],
    selected_case_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    if selected_case_ids is None:
        return full_case_ids
    selected = tuple(selected_case_ids)
    selected_set = set(selected)
    canonical = tuple(case_id for case_id in full_case_ids if case_id in selected_set)
    if (
        not selected
        or len(selected) != len(selected_set)
        or selected != canonical
        or not selected_set.issubset(full_case_ids)
    ):
        raise ComparisonScheduleError("comparison_schedule_case_subset_invalid")
    return selected


def _ordered_work(
    *,
    axis: ExperimentAxis,
    variant_ids: tuple[str, ...],
    case_ids: tuple[str, ...],
    repeats: int,
    policy: ExperimentOrderPolicy,
    seed: int,
) -> tuple[tuple[str, int, str], ...]:
    rng = random.Random(seed)  # noqa: S311 - deterministic registered plan order
    work: list[tuple[str, int, str]] = []
    if axis is ExperimentAxis.CACHE_BEHAVIOR:
        ordered_cases: list[tuple[int, tuple[str, ...]]] = []
        for repetition in range(repeats):
            cases = list(case_ids)
            if policy is not ExperimentOrderPolicy.DECLARED:
                rng.shuffle(cases)
            ordered_cases.append((repetition, tuple(cases)))
        for variant_id in variant_ids:
            for repetition, case_order in ordered_cases:
                work.extend((variant_id, repetition, case_id) for case_id in case_order)
    elif policy is ExperimentOrderPolicy.DECLARED:
        for repetition in range(repeats):
            for variant_id in variant_ids:
                work.extend((variant_id, repetition, case_id) for case_id in case_ids)
    elif policy is ExperimentOrderPolicy.SEEDED_SHUFFLE:
        work = [
            (variant_id, repetition, case_id)
            for repetition in range(repeats)
            for variant_id in variant_ids
            for case_id in case_ids
        ]
        rng.shuffle(work)
    else:
        for repetition in range(repeats):
            cases = list(case_ids)
            rng.shuffle(cases)
            for index, case_id in enumerate(cases):
                offset = (rng.randrange(len(variant_ids)) + index) % len(variant_ids)
                interleaved = list(variant_ids[offset:] + variant_ids[:offset])
                work.extend((variant_id, repetition, case_id) for variant_id in interleaved)
    return tuple(work)


def _execution_case_id(case_id: str, repetition: int, repeats: int) -> str:
    if repeats == 1:
        return case_id
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:12]
    return f"repeat-{repetition + 1}-{digest}"


__all__ = [
    "ComparisonExecutionSchedule",
    "ComparisonExecutionStep",
    "ComparisonScheduleError",
    "build_comparison_schedule",
    "cache_eligible_case_ids",
    "materialize_variant_cases",
]
