"""Content-minimized, allowlisted telemetry filtering."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Final

from rag_mvp.safety.output import JsonValue, redact_output
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, RedactionError, Redactor

DEFAULT_TELEMETRY_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {
        "timestamp",
        "level",
        "service",
        "service_version",
        "config_version",
        "event",
        "event_name",
        "request_id",
        "trace_id",
        "span_id",
        "run_id",
        "operation",
        "stage",
        "outcome",
        "status",
        "error_category",
        "safe_error_category",
        "duration_ms",
        "stage_duration_ms",
        "cache_status",
        "cache_outcome",
        "counts",
        "model",
        "model_identity",
        "provider",
        "token_usage",
        "tokens",
        "estimated_cost",
        "cost",
        "currency",
        "redaction",
        "redaction_metadata",
        "degraded_reason",
        "attempt",
        "retry",
        "fallback",
        "in_flight",
        "queue_depth",
        "metadata",
    }
)

_PROHIBITED_FIELD_PARTS: Final[tuple[str, ...]] = (
    "question",
    "answer",
    "prompt",
    "conversation",
    "history",
    "retrieved_text",
    "document_text",
    "document_content",
    "uploaded_content",
    "raw_content",
    "raw_payload",
    "source_preview",
    "citation_text",
    "authorization",
    "credential",
    "api_key",
    "password",
    "private_key",
)

_PROHIBITED_FIELDS: Final[frozenset[str]] = frozenset(
    {"body", "content", "context", "text", "source", "citation"}
)


def _normalized_field(field: str) -> str:
    return field.casefold().replace("-", "_").replace(" ", "_")


def _is_prohibited(field: str) -> bool:
    normalized = _normalized_field(field)
    return normalized in _PROHIBITED_FIELDS or any(
        part in normalized for part in _PROHIBITED_FIELD_PARTS
    )


class TelemetryFilter:
    """Drop unknown/content fields and redact all retained dynamic strings."""

    def __init__(
        self,
        *,
        allowlist: Iterable[str] = DEFAULT_TELEMETRY_ALLOWLIST,
        redactor: Redactor | None = DEFAULT_REDACTOR,
        max_depth: int = 12,
    ) -> None:
        self._allowlist = frozenset(allowlist)
        self._redactor = redactor
        self._max_depth = max_depth
        self._dropped_events = 0

    @property
    def dropped_events(self) -> int:
        """Content-free count of events dropped because filtering was uncertain."""

        return self._dropped_events

    def filter(self, event: Mapping[str, object]) -> dict[str, JsonValue] | None:
        """Return an allowlisted event, or ``None`` when safe filtering fails."""

        if self._redactor is None or not self._redactor.initialized:
            self._dropped_events += 1
            return None

        retained: dict[str, JsonValue] = {}
        try:
            for field, value in event.items():
                if field not in self._allowlist or _is_prohibited(field):
                    continue
                retained[field] = redact_output(
                    value,
                    redactor=self._redactor,
                    max_depth=self._max_depth,
                )
        except (RedactionError, TypeError, ValueError, RecursionError):
            self._dropped_events += 1
            return None
        return retained


def filter_telemetry_event(
    event: Mapping[str, object],
    *,
    redactor: Redactor | None = DEFAULT_REDACTOR,
) -> dict[str, JsonValue] | None:
    """Filter one event with the standard telemetry policy."""

    return TelemetryFilter(redactor=redactor).filter(event)
