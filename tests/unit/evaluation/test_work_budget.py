from decimal import Decimal

import pytest

from rag_mvp.evaluation.runner import EvaluationCaseInput
from rag_mvp.evaluation.work_budget import (
    ProviderWorkBudget,
    ProviderWorkBudgetError,
    ProviderWorkEstimate,
    ProviderWorkItem,
    collect_work_estimates,
    scoped_work_estimator,
)


def _estimate(work_id: str, calls: int, cost: str) -> ProviderWorkEstimate:
    return ProviderWorkEstimate(
        work_id=work_id,
        provider_calls=calls,
        conservative_cost=Decimal(cost),
        currency="USD",
    )


def test_budget_reserves_a_complete_plan_atomically_before_work() -> None:
    budget = ProviderWorkBudget(6, Decimal("1.20"), "USD")

    budget.reserve_many(
        (
            _estimate("candidate-a-case-1", 2, "0.20"),
            _estimate("candidate-a-case-2", 2, "0.30"),
        )
    )

    assert budget.snapshot().model_dump() == {
        "maximum_provider_calls": 6,
        "maximum_cost": Decimal("1.20"),
        "currency": "USD",
        "reserved_provider_calls": 4,
        "reserved_cost": Decimal("0.50"),
        "reservation_count": 2,
    }


def test_call_cap_fails_before_mutating_any_reservation() -> None:
    budget = ProviderWorkBudget(3, Decimal("10"), "USD")

    with pytest.raises(ProviderWorkBudgetError, match="provider_call_cap_exceeded"):
        budget.reserve_many(
            (
                _estimate("case-1", 2, "0.10"),
                _estimate("case-2", 2, "0.10"),
            )
        )

    assert budget.snapshot().reservation_count == 0


def test_cost_cap_fails_before_mutating_any_reservation() -> None:
    budget = ProviderWorkBudget(10, Decimal("0.50"), "USD")

    with pytest.raises(ProviderWorkBudgetError, match="provider_cost_cap_exceeded"):
        budget.reserve_many(
            (
                _estimate("case-1", 1, "0.30"),
                _estimate("case-2", 1, "0.21"),
            )
        )

    assert budget.snapshot().reserved_cost == Decimal(0)


@pytest.mark.parametrize(
    "estimates",
    [
        (_estimate("case-1", 1, "0.10"), _estimate("case-1", 1, "0.10")),
        (
            ProviderWorkEstimate(
                work_id="case-1",
                provider_calls=1,
                conservative_cost=Decimal("0.10"),
                currency="EUR",
            ),
        ),
    ],
)
def test_duplicate_or_wrong_currency_reservations_fail_closed(
    estimates: tuple[ProviderWorkEstimate, ...],
) -> None:
    budget = ProviderWorkBudget(10, Decimal("1"), "USD")

    with pytest.raises(ProviderWorkBudgetError):
        budget.reserve_many(estimates)

    assert budget.snapshot().reservation_count == 0


def test_reservations_are_not_recycled_after_lower_actual_work() -> None:
    budget = ProviderWorkBudget(2, Decimal("0.20"), "USD")
    budget.reserve(_estimate("case-1", 2, "0.20"))

    with pytest.raises(ProviderWorkBudgetError, match="provider_call_cap_exceeded"):
        budget.reserve(_estimate("case-2", 1, "0.01"))


def test_identical_cases_across_candidates_use_one_atomic_scoped_suite_ledger() -> None:
    cases = (
        EvaluationCaseInput(case_id="case-1", question="One?", language="en"),
        EvaluationCaseInput(case_id="case-2", question="Two?", language="en"),
    )

    def base(item: ProviderWorkItem) -> ProviderWorkEstimate:
        return _estimate(item.case_id, 2, "0.10")

    estimates = tuple(
        estimate
        for variant, run_id in (("dense", "run-a"), ("hybrid", "run-b"))
        for estimate in collect_work_estimates(
            cases,
            scoped_work_estimator(f"comparison-1.{variant}.{run_id}.repeat-0", base),
        )
    )
    budget = ProviderWorkBudget(8, Decimal("0.40"), "USD")

    budget.reserve_many(estimates)

    snapshot = budget.snapshot()
    assert snapshot.reservation_count == 4
    assert snapshot.reserved_provider_calls == 8
    assert snapshot.reserved_cost == Decimal("0.40")
    assert len({item.work_id for item in estimates}) == 4


def test_multi_candidate_suite_cap_failure_is_atomic_before_any_reservation() -> None:
    cases = (
        EvaluationCaseInput(case_id="same-case", question="One?", language="en"),
    )

    def base(item: ProviderWorkItem) -> ProviderWorkEstimate:
        return _estimate(item.case_id, 2, "0.10")

    estimates = tuple(
        collect_work_estimates(
            cases,
            scoped_work_estimator(f"comparison-1.{variant}.{run_id}", base),
        )[0]
        for variant, run_id in (("a", "run-a"), ("b", "run-b"))
    )
    budget = ProviderWorkBudget(3, Decimal("1.00"), "USD")

    with pytest.raises(ProviderWorkBudgetError, match="provider_call_cap_exceeded"):
        budget.reserve_many(estimates)

    assert budget.snapshot().reservation_count == 0


@pytest.mark.parametrize("scope", ["../private", "x" * 256, "contains space"])
def test_scoped_estimator_rejects_unsafe_or_overlong_scope(scope: str) -> None:
    with pytest.raises(ValueError, match="provider work scope is invalid"):
        scoped_work_estimator(scope, lambda item: _estimate(item.case_id, 1, "0.01"))


def test_scoped_estimator_revalidates_combined_identifier_length() -> None:
    estimator = scoped_work_estimator(
        "s" * 250,
        lambda item: _estimate(item.case_id, 1, "0.01"),
    )
    item = EvaluationCaseInput(case_id="case-1", question="One?", language="en")

    with pytest.raises(ValueError, match="provider work identifier is unsafe"):
        estimator(item)
