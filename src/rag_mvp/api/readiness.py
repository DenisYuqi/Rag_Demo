"""Component readiness aggregation with safe reason codes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ComponentStatus:
    name: str
    ready: bool
    reason: str | None = None


class ReadinessCheck(Protocol):
    @property
    def name(self) -> str: ...

    def check(self) -> ComponentStatus: ...


@dataclass(slots=True)
class StaticReadinessCheck:
    name: str
    ready: bool = True
    reason: str | None = None

    def check(self) -> ComponentStatus:
        return ComponentStatus(self.name, self.ready, self.reason if not self.ready else None)


class ReadinessRegistry:
    def __init__(self, checks: list[ReadinessCheck] | None = None) -> None:
        self._checks: list[ReadinessCheck] = list(checks or [])

    def register(self, check: ReadinessCheck) -> None:
        if any(existing.name == check.name for existing in self._checks):
            raise ValueError(f"duplicate readiness component: {check.name}")
        self._checks.append(check)

    def get(self, name: str) -> ReadinessCheck:
        """Return a registered component by its stable name."""
        try:
            return next(check for check in self._checks if check.name == name)
        except StopIteration as error:
            raise KeyError(name) from error

    def report(self) -> tuple[bool, list[ComponentStatus]]:
        statuses: list[ComponentStatus] = []
        for check in self._checks:
            try:
                statuses.append(check.check())
            except Exception:
                statuses.append(ComponentStatus(check.name, False, "component_check_failed"))
        return all(status.ready for status in statuses), statuses
