"""Conservative pre-work provider-call and monetary budget reservations."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from threading import Lock
from typing import Annotated, Protocol, Self

from pydantic import Field, field_validator, model_validator

from rag_mvp.domain._base import DomainModel, Identifier

_SAFE_WORK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")


class ProviderWorkBudgetError(RuntimeError):
    """A privacy-safe hard-budget failure raised before provider work starts."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ProviderWorkItem(Protocol):
    """The minimal case-like identity accepted by scoped estimators."""

    @property
    def case_id(self) -> str: ...


class ProviderWorkEstimate(DomainModel):
    """Worst-case paid work reserved for one case or other indivisible unit."""

    work_id: Identifier
    provider_calls: Annotated[int, Field(ge=0)]
    conservative_cost: Annotated[Decimal, Field(ge=0)]
    currency: Identifier

    @field_validator("work_id")
    @classmethod
    def safe_work_id(cls, value: str) -> str:
        if _SAFE_WORK_ID.fullmatch(value) is None:
            raise ValueError("provider work identifier is unsafe")
        return value

    @field_validator("conservative_cost")
    @classmethod
    def finite_cost(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("provider work cost must be finite")
        return value

    @model_validator(mode="after")
    def require_bounded_work(self) -> Self:
        if self.provider_calls == 0 and self.conservative_cost == 0:
            raise ValueError("provider work estimate cannot be empty")
        return self


class ProviderWorkBudgetSnapshot(DomainModel):
    maximum_provider_calls: Annotated[int, Field(gt=0)]
    maximum_cost: Annotated[Decimal, Field(ge=0)]
    currency: Identifier
    reserved_provider_calls: Annotated[int, Field(ge=0)]
    reserved_cost: Annotated[Decimal, Field(ge=0)]
    reservation_count: Annotated[int, Field(ge=0)]


@dataclass(slots=True)
class ProviderWorkBudget:
    """Atomic fail-closed reservations that are never recycled after execution.

    Reserving worst-case work before an indivisible provider operation makes the
    configured limit hard: callers cannot begin work unless the complete reservation
    fits. Unused reservation is intentionally not released, avoiding a later call from
    consuming budget that may already have been spent but not yet reconciled.
    """

    maximum_provider_calls: int
    maximum_cost: Decimal
    currency: str
    _reservations: dict[str, ProviderWorkEstimate] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.maximum_provider_calls) is not int or self.maximum_provider_calls < 1:
            raise ValueError("provider work call cap must be positive")
        try:
            maximum_cost = Decimal(self.maximum_cost)
        except (InvalidOperation, TypeError, ValueError):
            raise ValueError("provider work cost cap is invalid") from None
        if not maximum_cost.is_finite() or maximum_cost < 0:
            raise ValueError("provider work cost cap must be non-negative and finite")
        if not isinstance(self.currency, str) or not self.currency.strip():
            raise ValueError("provider work currency is required")
        self.maximum_cost = maximum_cost
        self.currency = self.currency.strip()

    def reserve(self, estimate: ProviderWorkEstimate) -> ProviderWorkEstimate:
        """Reserve one work unit before it begins."""

        return self.reserve_many((estimate,))[0]

    def reserve_many(
        self,
        estimates: tuple[ProviderWorkEstimate, ...],
    ) -> tuple[ProviderWorkEstimate, ...]:
        """Atomically reserve a complete plan; on failure nothing is consumed."""

        if not estimates:
            raise ProviderWorkBudgetError("provider_work_estimates_missing")
        if any(not isinstance(estimate, ProviderWorkEstimate) for estimate in estimates):
            raise ProviderWorkBudgetError("provider_work_estimate_invalid")
        work_ids = tuple(estimate.work_id for estimate in estimates)
        if len(work_ids) != len(set(work_ids)):
            raise ProviderWorkBudgetError("provider_work_reservation_duplicate")
        if any(estimate.currency != self.currency for estimate in estimates):
            raise ProviderWorkBudgetError("provider_work_currency_mismatch")

        with self._lock:
            if any(work_id in self._reservations for work_id in work_ids):
                raise ProviderWorkBudgetError("provider_work_reservation_duplicate")
            current_calls = sum(item.provider_calls for item in self._reservations.values())
            current_cost = sum(
                (item.conservative_cost for item in self._reservations.values()),
                start=Decimal(0),
            )
            next_calls = current_calls + sum(item.provider_calls for item in estimates)
            next_cost = current_cost + sum(
                (item.conservative_cost for item in estimates),
                start=Decimal(0),
            )
            if next_calls > self.maximum_provider_calls:
                raise ProviderWorkBudgetError("provider_call_cap_exceeded")
            if next_cost > self.maximum_cost:
                raise ProviderWorkBudgetError("provider_cost_cap_exceeded")
            self._reservations.update((estimate.work_id, estimate) for estimate in estimates)
        return estimates

    def snapshot(self) -> ProviderWorkBudgetSnapshot:
        with self._lock:
            values = tuple(self._reservations.values())
        return ProviderWorkBudgetSnapshot(
            maximum_provider_calls=self.maximum_provider_calls,
            maximum_cost=self.maximum_cost,
            currency=self.currency,
            reserved_provider_calls=sum(item.provider_calls for item in values),
            reserved_cost=sum(
                (item.conservative_cost for item in values),
                start=Decimal(0),
            ),
            reservation_count=len(values),
        )

    def require_reserved(
        self,
        estimates: tuple[ProviderWorkEstimate, ...],
    ) -> tuple[ProviderWorkEstimate, ...]:
        """Prove work was already atomically reserved without reserving it twice."""

        if not estimates or any(
            not isinstance(estimate, ProviderWorkEstimate) for estimate in estimates
        ):
            raise ProviderWorkBudgetError("provider_work_estimate_invalid")
        work_ids = tuple(item.work_id for item in estimates)
        if len(work_ids) != len(set(work_ids)):
            raise ProviderWorkBudgetError("provider_work_reservation_duplicate")
        with self._lock:
            if any(self._reservations.get(item.work_id) != item for item in estimates):
                raise ProviderWorkBudgetError("provider_work_reservation_missing")
        return estimates


def scoped_work_estimator(
    scope_id: str,
    estimator: Callable[[ProviderWorkItem], ProviderWorkEstimate],
) -> Callable[[ProviderWorkItem], ProviderWorkEstimate]:
    """Bind identical dataset case IDs to one immutable comparison/run scope."""

    if not isinstance(scope_id, str) or _SAFE_WORK_ID.fullmatch(scope_id.strip()) is None:
        raise ValueError("provider work scope is invalid")
    normalized_scope = scope_id.strip()

    def estimate(item: ProviderWorkItem) -> ProviderWorkEstimate:
        value = estimator(item)
        if not isinstance(value, ProviderWorkEstimate):
            raise ProviderWorkBudgetError("provider_work_estimate_invalid")
        combined_id = f"{normalized_scope}.{value.work_id}"
        if _SAFE_WORK_ID.fullmatch(combined_id) is None:
            raise ValueError("provider work identifier is unsafe")
        return ProviderWorkEstimate.model_validate(
            {
                **value.model_dump(),
                "work_id": combined_id,
            }
        )

    return estimate


def collect_work_estimates(
    items: Iterable[ProviderWorkItem],
    estimator: Callable[[ProviderWorkItem], ProviderWorkEstimate],
) -> tuple[ProviderWorkEstimate, ...]:
    """Materialize every scoped reservation before any provider work can start."""

    values: list[ProviderWorkEstimate] = []
    try:
        for item in items:
            estimate = estimator(item)
            if not isinstance(estimate, ProviderWorkEstimate):
                raise ProviderWorkBudgetError("provider_work_estimate_invalid")
            values.append(estimate)
    except ProviderWorkBudgetError:
        raise
    except Exception:
        raise ProviderWorkBudgetError("provider_work_estimate_unavailable") from None
    if not values:
        raise ProviderWorkBudgetError("provider_work_estimates_missing")
    return tuple(values)


__all__ = [
    "ProviderWorkBudget",
    "ProviderWorkBudgetError",
    "ProviderWorkBudgetSnapshot",
    "ProviderWorkEstimate",
    "ProviderWorkItem",
    "collect_work_estimates",
    "scoped_work_estimator",
]
