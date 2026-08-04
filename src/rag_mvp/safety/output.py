"""Fail-closed recursive redaction for JSON-shaped output objects."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from uuid import UUID

from pydantic import BaseModel

from rag_mvp.safety.redactor import DEFAULT_REDACTOR, RedactionError, Redactor

type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | tuple[JsonValue, ...] | dict[str, JsonValue]

SAFE_UNAVAILABLE_MESSAGE = "内容暂时无法安全显示。Content is temporarily unavailable."


class OutputRedactionError(RedactionError):
    """Raised when an output object cannot be safely serialized and redacted."""


def redact_output(
    value: object,
    *,
    redactor: Redactor = DEFAULT_REDACTOR,
    max_depth: int = 32,
) -> JsonValue:
    """Recursively redact a JSON-shaped value.

    Unsupported object types, excessive nesting, and key collisions after
    redaction fail closed instead of passing an unknown object's contents
    through unchanged.
    """

    if max_depth < 0:
        raise OutputRedactionError("output exceeds maximum nesting depth")
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return redactor.redact(value).redacted_text
    if isinstance(value, BaseModel):
        return redact_output(
            value.model_dump(mode="json"), redactor=redactor, max_depth=max_depth - 1
        )
    if isinstance(value, Enum):
        return redact_output(value.value, redactor=redactor, max_depth=max_depth - 1)
    if isinstance(value, datetime | date | time | Decimal | UUID):
        return redactor.redact(str(value)).redacted_text
    if isinstance(value, Mapping):
        redacted_mapping: dict[str, JsonValue] = {}
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise OutputRedactionError("output mappings require string keys")
            safe_key = redactor.redact(raw_key).redacted_text
            if safe_key in redacted_mapping:
                raise OutputRedactionError("redaction produced duplicate mapping keys")
            redacted_mapping[safe_key] = redact_output(
                child, redactor=redactor, max_depth=max_depth - 1
            )
        return redacted_mapping
    if isinstance(value, list):
        return [redact_output(item, redactor=redactor, max_depth=max_depth - 1) for item in value]
    if isinstance(value, tuple):
        return tuple(
            redact_output(item, redactor=redactor, max_depth=max_depth - 1) for item in value
        )
    raise OutputRedactionError("unsupported dynamic output type")


def safe_redact_output(
    value: object,
    *,
    redactor: Redactor | None = DEFAULT_REDACTOR,
    unavailable_message: str = SAFE_UNAVAILABLE_MESSAGE,
    max_depth: int = 32,
) -> JsonValue:
    """Redact output or return only a fixed pre-vetted message on failure."""

    if redactor is None:
        return unavailable_message
    try:
        return redact_output(value, redactor=redactor, max_depth=max_depth)
    except (RedactionError, TypeError, ValueError, RecursionError):
        return unavailable_message
