"""Shared primitives for provider-neutral domain contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

type Identifier = Annotated[str, Field(min_length=1, max_length=255)]
type Digest = Annotated[str, Field(min_length=8, max_length=255)]
type NonEmptyText = Annotated[str, Field(min_length=1)]
type FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
type NonNegativeFiniteFloat = Annotated[
    float,
    Field(ge=0, allow_inf_nan=False),
]
type SafeScalar = str | int | float | bool | None


def utc_now() -> datetime:
    """Return an aware UTC timestamp for persistence-friendly defaults."""

    return datetime.now(UTC)


class DomainModel(BaseModel):
    """Immutable base model shared by all persisted domain values."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )
