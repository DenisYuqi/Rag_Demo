"""Versioned structured-log dictionary and privacy-safe JSONL sample contract."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import Field, model_validator

from rag_mvp.domain._base import DomainModel
from rag_mvp.evaluation.json_report import (
    MAX_REPORT_BYTES,
    ReportSerializationError,
    ReportWriteError,
    _atomic_write_text,
    canonical_json_value,
    decode_json_report,
)
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, RedactionError, Redactor
from rag_mvp.safety.telemetry import DEFAULT_TELEMETRY_ALLOWLIST, TelemetryFilter

LOG_DICTIONARY_SCHEMA_VERSION: Literal["structured-log-field-dictionary-v1"] = (
    "structured-log-field-dictionary-v1"
)
LOG_EVENT_SCHEMA_VERSION: Literal["rag-structured-log-v1"] = "rag-structured-log-v1"
LOG_DICTIONARY_FILENAME = "structured-log-field-dictionary-v1.json"
LOG_SAMPLE_FILENAME = "privacy-safe-sample-v1.jsonl"
_PROHIBITED_FIELD_PARTS = (
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
_PROHIBITED_FIELDS = frozenset({"body", "content", "context", "text", "source", "citation"})
_ABSOLUTE_PATH = re.compile(r"(?i)(?:[A-Z]:[\\/]|/(?:home|users|tmp|var|etc)/)")
_CONTENT_FREE_STRING = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,254}$")


class LogValueType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    STRING_OR_INTEGER = "string-or-integer"
    NUMBER_OR_OBJECT = "number-or-object"


class LogCardinality(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNIQUE = "unique-per-event"


class LogRedactionRule(StrEnum):
    STATIC_ALLOWLIST = "static-allowlist"
    ENUM_ALLOWLIST = "enum-allowlist"
    OPAQUE_IDENTIFIER = "opaque-identifier-only"
    NONNEGATIVE_NUMBER = "nonnegative-number-only"
    RECURSIVE_REDACTION = "recursive-supported-sensitive-redaction"


class LogFieldDefinition(DomainModel):
    """User-facing contract for one allowlisted structured-log field."""

    name: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    meaning: Annotated[str, Field(min_length=1, max_length=500)]
    value_type: LogValueType
    unit: Annotated[str, Field(min_length=1, max_length=64)] | None
    cardinality: LogCardinality
    presence_condition: Annotated[str, Field(min_length=1, max_length=300)]
    redaction_rule: LogRedactionRule
    sensitive: bool

    @model_validator(mode="after")
    def validate_redaction(self) -> LogFieldDefinition:
        if self.sensitive and self.redaction_rule in {
            LogRedactionRule.STATIC_ALLOWLIST,
            LogRedactionRule.NONNEGATIVE_NUMBER,
        }:
            raise ValueError("sensitive log fields require an active redaction rule")
        if self.name not in DEFAULT_TELEMETRY_ALLOWLIST:
            raise ValueError("log dictionary field is not telemetry-allowlisted")
        return self


class StructuredLogFieldDictionaryV1(DomainModel):
    schema_version: Literal["structured-log-field-dictionary-v1"] = LOG_DICTIONARY_SCHEMA_VERSION
    log_schema_version: Literal["rag-structured-log-v1"] = LOG_EVENT_SCHEMA_VERSION
    fields: Annotated[tuple[LogFieldDefinition, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_coverage(self) -> StructuredLogFieldDictionaryV1:
        names = tuple(item.name for item in self.fields)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("log dictionary fields must be unique and sorted")
        if set(names) != set(DEFAULT_TELEMETRY_ALLOWLIST):
            raise ValueError("log dictionary must cover the complete telemetry allowlist")
        return self


class LogDocumentationError(ValueError):
    """Raised with content-free diagnostics when log documentation is unsafe."""


def build_log_field_dictionary_v1() -> StructuredLogFieldDictionaryV1:
    """Build the complete deterministic dictionary from the runtime allowlist."""

    return StructuredLogFieldDictionaryV1(
        fields=tuple(_field_definition(name) for name in sorted(DEFAULT_TELEMETRY_ALLOWLIST))
    )


def sample_log_events_v1() -> tuple[dict[str, object], ...]:
    """Return representative content-free events using only documented fields."""

    common: dict[str, object] = {
        "timestamp": "2026-08-07T03:04:05Z",
        "level": "info",
        "service": "rag-mvp",
        "service_version": "0.1.0",
        "config_version": "configuration-v2",
    }
    return (
        {
            **common,
            "event": "http.request.completed",
            "request_id": "request-sample-001",
            "trace_id": "0123456789abcdef0123456789abcdef",
            "operation": "http.request",
            "outcome": "succeeded",
            "status": 200,
            "duration_ms": 184.5,
            "metadata": {"log_schema_version": LOG_EVENT_SCHEMA_VERSION},
        },
        {
            **common,
            "event": "retrieval.cache.lookup",
            "request_id": "request-sample-002",
            "trace_id": "1123456789abcdef0123456789abcdef",
            "operation": "retrieval.cache",
            "outcome": "succeeded",
            "cache_outcome": "hit",
            "counts": {"eligible": 1, "hits": 1, "misses": 0},
            "duration_ms": 2.75,
            "metadata": {"log_schema_version": LOG_EVENT_SCHEMA_VERSION},
        },
        {
            **common,
            "event": "qa.refusal.completed",
            "request_id": "request-sample-003",
            "trace_id": "2123456789abcdef0123456789abcdef",
            "operation": "qa.refusal",
            "outcome": "refused",
            "degraded_reason": "low-confidence",
            "metadata": {
                "guidance_present": True,
                "language": "zh",
                "log_schema_version": LOG_EVENT_SCHEMA_VERSION,
            },
        },
        {
            **common,
            "event": "provider.attempt.completed",
            "request_id": "request-sample-004",
            "trace_id": "3123456789abcdef0123456789abcdef",
            "operation": "generation",
            "outcome": "succeeded",
            "provider": "primary",
            "model_identity": "chat-v2",
            "attempt": 1,
            "fallback": False,
            "token_usage": {"input": 900, "output": 180},
            "estimated_cost": 0.00324,
            "currency": "USD",
            "metadata": {"log_schema_version": LOG_EVENT_SCHEMA_VERSION},
        },
        {
            **common,
            "event": "evaluation.run.completed",
            "run_id": "acceptance-run-v2",
            "operation": "evaluation.run",
            "outcome": "succeeded",
            "counts": {"completed": 24, "failed": 0, "total": 24},
            "metadata": {"log_schema_version": LOG_EVENT_SCHEMA_VERSION},
        },
    )


def canonical_log_dictionary_json(
    dictionary: StructuredLogFieldDictionaryV1 | Mapping[str, object] | None = None,
) -> str:
    parsed = _parse_dictionary(dictionary or build_log_field_dictionary_v1())
    return canonical_json_value(parsed.model_dump(mode="json"))


def parse_log_dictionary_json(text: str) -> StructuredLogFieldDictionaryV1:
    if not isinstance(text, str) or not text or len(text.encode("utf-8")) > MAX_REPORT_BYTES:
        raise LogDocumentationError("log dictionary size is outside allowed bounds")
    try:
        raw = decode_json_report(text)
        if not isinstance(raw, dict):
            raise ValueError
        return _parse_dictionary(cast(Mapping[str, object], raw))
    except (ReportSerializationError, TypeError, ValueError) as error:
        raise LogDocumentationError("log dictionary JSON is invalid") from error


def render_log_sample_jsonl(
    events: Sequence[Mapping[str, object]] | None = None,
    *,
    dictionary: StructuredLogFieldDictionaryV1 | None = None,
) -> str:
    active_dictionary = dictionary or build_log_field_dictionary_v1()
    values = tuple(events or sample_log_events_v1())
    validated = validate_log_documentation(active_dictionary, values)
    return "".join(canonical_json_value(event) + "\n" for event in validated)


def parse_log_sample_jsonl(text: str) -> tuple[dict[str, object], ...]:
    if not isinstance(text, str) or not text or len(text.encode("utf-8")) > MAX_REPORT_BYTES:
        raise LogDocumentationError("log sample size is outside allowed bounds")
    lines = text.splitlines()
    if not lines or len(lines) > 1_000:
        raise LogDocumentationError("log sample line count is outside allowed bounds")
    events: list[dict[str, object]] = []
    try:
        for line in lines:
            raw = decode_json_report(line)
            if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
                raise ValueError
            events.append(cast(dict[str, object], raw))
    except (ReportSerializationError, TypeError, ValueError) as error:
        raise LogDocumentationError("log sample JSONL is invalid") from error
    return tuple(events)


def validate_log_documentation(
    dictionary: StructuredLogFieldDictionaryV1 | Mapping[str, object],
    events: Sequence[Mapping[str, object]],
    *,
    redactor: Redactor = DEFAULT_REDACTOR,
) -> tuple[dict[str, object], ...]:
    """Validate coverage, types, redaction rules, paths, secrets, and supported PII."""

    parsed = _parse_dictionary(dictionary)
    definitions = {item.name: item for item in parsed.fields}
    if not events:
        raise LogDocumentationError("log sample must contain at least one event")
    telemetry_filter = TelemetryFilter(redactor=redactor)
    normalized: list[dict[str, object]] = []
    required = {"timestamp", "level", "service", "service_version", "config_version", "event"}
    for event in events:
        if not isinstance(event, Mapping) or not all(isinstance(key, str) for key in event):
            raise LogDocumentationError("log sample event must be an object")
        if not required.issubset(event):
            raise LogDocumentationError("log sample event is missing required identity fields")
        unknown = set(event).difference(definitions)
        if unknown:
            raise LogDocumentationError("log sample contains an undocumented field")
        if any(_is_prohibited_field(field) for field in event):
            raise LogDocumentationError("log sample contains a prohibited field")
        for field, value in event.items():
            _validate_value_type(definitions[field], value)
            _validate_safe_value(field, value, redactor)
        filtered = telemetry_filter.filter(event)
        if filtered is None or set(filtered) != set(event):
            raise LogDocumentationError("log sample does not survive the runtime telemetry filter")
        normalized_event = cast(dict[str, object], filtered)
        if canonical_json_value(normalized_event) != canonical_json_value(event):
            raise LogDocumentationError("runtime telemetry filtering changes the sample")
        normalized.append(normalized_event)

    documentation_text = canonical_log_dictionary_json(parsed)
    sample_text = "".join(canonical_json_value(event) + "\n" for event in normalized)
    if _ABSOLUTE_PATH.search(documentation_text) or _ABSOLUTE_PATH.search(sample_text):
        raise LogDocumentationError("log documentation contains an absolute filesystem path")
    return tuple(normalized)


def validate_log_documentation_files(
    dictionary_path: Path | str,
    sample_path: Path | str,
) -> None:
    try:
        dictionary_text = Path(dictionary_path).read_bytes().decode("utf-8")
        sample_text = Path(sample_path).read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise LogDocumentationError("log documentation file is unavailable") from error
    dictionary = parse_log_dictionary_json(dictionary_text)
    events = parse_log_sample_jsonl(sample_text)
    validate_log_documentation(dictionary, events)
    if dictionary_text != canonical_log_dictionary_json(dictionary) + "\n":
        raise LogDocumentationError("log dictionary file is not canonical JSON")
    if sample_text != render_log_sample_jsonl(events, dictionary=dictionary):
        raise LogDocumentationError("log sample file is not canonical JSONL")


def write_log_documentation_v1(
    directory: Path | str,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Atomically create both versioned documentation files without replacing by default."""

    target = Path(directory)
    dictionary = build_log_field_dictionary_v1()
    events = sample_log_events_v1()
    validate_log_documentation(dictionary, events)
    dictionary_path = target / LOG_DICTIONARY_FILENAME
    sample_path = target / LOG_SAMPLE_FILENAME
    if not overwrite and any(
        path.expanduser().exists() or path.expanduser().is_symlink()
        for path in (dictionary_path, sample_path)
    ):
        raise ReportWriteError("immutable log documentation target already exists")
    written_dictionary: Path | None = None
    written_sample: Path | None = None
    try:
        written_dictionary = _atomic_write_text(
            dictionary_path,
            canonical_log_dictionary_json(dictionary) + "\n",
            overwrite=overwrite,
        )
        written_sample = _atomic_write_text(
            sample_path,
            render_log_sample_jsonl(events, dictionary=dictionary),
            overwrite=overwrite,
        )
    except BaseException:
        if not overwrite:
            if written_dictionary is not None:
                written_dictionary.unlink(missing_ok=True)
            if written_sample is not None:
                written_sample.unlink(missing_ok=True)
        raise
    return written_dictionary, written_sample


def _parse_dictionary(
    value: StructuredLogFieldDictionaryV1 | Mapping[str, object],
) -> StructuredLogFieldDictionaryV1:
    try:
        return (
            value
            if isinstance(value, StructuredLogFieldDictionaryV1)
            else StructuredLogFieldDictionaryV1.model_validate(value)
        )
    except (TypeError, ValueError) as error:
        raise LogDocumentationError("log dictionary violates its schema") from error


def _field_definition(name: str) -> LogFieldDefinition:
    value_type = _field_value_type(name)
    unit = _field_unit(name)
    cardinality = _field_cardinality(name)
    sensitive = name in {
        "request_id",
        "trace_id",
        "span_id",
        "run_id",
        "model",
        "model_identity",
        "provider",
        "metadata",
        "redaction_metadata",
    }
    if sensitive and value_type is LogValueType.STRING:
        rule = LogRedactionRule.OPAQUE_IDENTIFIER
    elif sensitive:
        rule = LogRedactionRule.RECURSIVE_REDACTION
    elif value_type in {LogValueType.NUMBER, LogValueType.INTEGER}:
        rule = LogRedactionRule.NONNEGATIVE_NUMBER
    elif value_type in {LogValueType.OBJECT, LogValueType.NUMBER_OR_OBJECT}:
        rule = LogRedactionRule.RECURSIVE_REDACTION
    elif name in _ENUM_FIELDS:
        rule = LogRedactionRule.ENUM_ALLOWLIST
    else:
        rule = LogRedactionRule.STATIC_ALLOWLIST
    return LogFieldDefinition(
        name=name,
        meaning=_FIELD_MEANINGS.get(
            name,
            f"Allowlisted operational attribute for {name.replace('_', ' ')}.",
        ),
        value_type=value_type,
        unit=unit,
        cardinality=cardinality,
        presence_condition=(
            "Present on every emitted event."
            if name in {"timestamp", "level", "service", "service_version", "config_version"}
            else "Present only when the named operation produces this evidence."
        ),
        redaction_rule=rule,
        sensitive=sensitive,
    )


_OBJECT_FIELDS = {
    "counts",
    "token_usage",
    "tokens",
    "redaction",
    "redaction_metadata",
    "metadata",
}
_INTEGER_FIELDS = {"attempt", "retry", "in_flight", "queue_depth"}
_NUMBER_FIELDS = {"duration_ms", "stage_duration_ms"}
_NUMBER_OR_OBJECT_FIELDS = {"estimated_cost", "cost"}
_BOOLEAN_FIELDS = {"fallback"}
_STRING_OR_INTEGER_FIELDS = {"status"}
_ENUM_FIELDS = {
    "level",
    "outcome",
    "stage",
    "cache_status",
    "cache_outcome",
    "safe_error_category",
    "error_category",
    "currency",
    "degraded_reason",
}


def _field_value_type(name: str) -> LogValueType:
    if name in _OBJECT_FIELDS:
        return LogValueType.OBJECT
    if name in _INTEGER_FIELDS:
        return LogValueType.INTEGER
    if name in _NUMBER_FIELDS:
        return LogValueType.NUMBER
    if name in _NUMBER_OR_OBJECT_FIELDS:
        return LogValueType.NUMBER_OR_OBJECT
    if name in _BOOLEAN_FIELDS:
        return LogValueType.BOOLEAN
    if name in _STRING_OR_INTEGER_FIELDS:
        return LogValueType.STRING_OR_INTEGER
    return LogValueType.STRING


def _field_unit(name: str) -> str | None:
    if name in {"duration_ms", "stage_duration_ms"}:
        return "milliseconds"
    if name in {"token_usage", "tokens"}:
        return "tokens"
    if name in {"estimated_cost", "cost"}:
        return "declared-currency"
    if name in {"attempt", "retry", "in_flight", "queue_depth", "counts"}:
        return "count"
    if name == "timestamp":
        return "rfc3339-utc"
    return None


def _field_cardinality(name: str) -> LogCardinality:
    if name in {"request_id", "trace_id", "span_id"}:
        return LogCardinality.UNIQUE
    if name in {"timestamp", "run_id", "duration_ms", "stage_duration_ms"}:
        return LogCardinality.HIGH
    if name in {"model", "model_identity", "provider", "operation", "event", "event_name"}:
        return LogCardinality.MEDIUM
    return LogCardinality.LOW


_FIELD_MEANINGS = {
    "timestamp": "UTC timestamp assigned by the logging processor.",
    "level": "Allowlisted severity label for the event.",
    "service": "Stable service identity emitting the event.",
    "service_version": "Deployed application version.",
    "config_version": "Opaque effective configuration version used by the operation.",
    "event": "Stable structured event name emitted by structlog.",
    "event_name": "Stable event name used by pre-rendered event mappings.",
    "request_id": "Opaque request correlation identifier; never request content.",
    "trace_id": "Opaque distributed-trace correlation identifier.",
    "span_id": "Opaque trace-span correlation identifier.",
    "run_id": "Opaque evaluation or load-run identifier.",
    "operation": "Stable bounded operation or component name.",
    "stage": "Allowlisted lifecycle stage label.",
    "outcome": "Allowlisted content-free operation outcome.",
    "status": "Protocol status code or stable lifecycle status label.",
    "error_category": "Stable content-free error category.",
    "safe_error_category": "Allowlisted error category derived without exception text.",
    "duration_ms": "Complete measured operation duration in milliseconds.",
    "stage_duration_ms": "Measured stage duration in milliseconds.",
    "cache_status": "Stable cache lifecycle status.",
    "cache_outcome": "Stable hit, miss, bypass, expiry, eviction, or error outcome.",
    "counts": "Content-free named counters for the event.",
    "model": "Opaque configured model identifier.",
    "model_identity": "Opaque exact model/version identity.",
    "provider": "Opaque provider identity with no endpoint or credential.",
    "token_usage": "Content-free token totals by declared direction.",
    "tokens": "Content-free token totals.",
    "estimated_cost": "Nonnegative estimated provider cost in the declared currency.",
    "cost": "Nonnegative cost evidence or a recursively redacted cost summary.",
    "currency": "Allowlisted ISO-style currency label.",
    "redaction": "Content-free redaction counters and categories.",
    "redaction_metadata": "Recursively redacted metadata about applied privacy filtering.",
    "degraded_reason": "Stable content-free reason for controlled degradation.",
    "attempt": "One-based attempt ordinal.",
    "retry": "Retry ordinal or count.",
    "fallback": "Whether the operation used a configured fallback route.",
    "in_flight": "Current bounded in-flight operation count.",
    "queue_depth": "Current bounded supervisor queue depth.",
    "metadata": "Allowlisted recursively redacted operational metadata only.",
}


def _validate_value_type(definition: LogFieldDefinition, value: object) -> None:
    valid = {
        LogValueType.STRING: isinstance(value, str),
        LogValueType.INTEGER: type(value) is int,
        LogValueType.NUMBER: not isinstance(value, bool) and isinstance(value, int | float),
        LogValueType.BOOLEAN: isinstance(value, bool),
        LogValueType.OBJECT: isinstance(value, Mapping),
        LogValueType.STRING_OR_INTEGER: isinstance(value, str) or type(value) is int,
        LogValueType.NUMBER_OR_OBJECT: (
            (not isinstance(value, bool) and isinstance(value, int | float))
            or isinstance(value, Mapping)
        ),
    }[definition.value_type]
    if not valid:
        raise LogDocumentationError("log sample field has the wrong documented type")
    if (
        definition.redaction_rule is LogRedactionRule.NONNEGATIVE_NUMBER
        and cast(int | float, value) < 0
    ):
        raise LogDocumentationError("log sample numeric field is negative")


def _validate_safe_value(field: str, value: object, redactor: Redactor) -> None:
    if isinstance(value, str):
        if _ABSOLUTE_PATH.search(value):
            raise LogDocumentationError("log sample contains an absolute filesystem path")
        if _CONTENT_FREE_STRING.fullmatch(value) is None:
            raise LogDocumentationError("log sample contains a content-bearing string")
        try:
            if redactor.detect(value):
                raise LogDocumentationError("log sample contains supported PII or a secret")
        except RedactionError as error:
            raise LogDocumentationError("log sample privacy validation failed closed") from error
    elif isinstance(value, Mapping):
        for nested_field, nested_value in value.items():
            if not isinstance(nested_field, str) or _is_prohibited_field(nested_field):
                raise LogDocumentationError("log sample metadata contains a prohibited field")
            _validate_safe_value(f"{field}.{nested_field}", nested_value, redactor)
    elif isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        for item in value:
            _validate_safe_value(field, item, redactor)


def _is_prohibited_field(field: str) -> bool:
    normalized = field.casefold().replace("-", "_").replace(" ", "_")
    return normalized in _PROHIBITED_FIELDS or any(
        part in normalized for part in _PROHIBITED_FIELD_PARTS
    )


__all__ = [
    "LOG_DICTIONARY_FILENAME",
    "LOG_DICTIONARY_SCHEMA_VERSION",
    "LOG_EVENT_SCHEMA_VERSION",
    "LOG_SAMPLE_FILENAME",
    "LogCardinality",
    "LogDocumentationError",
    "LogFieldDefinition",
    "LogRedactionRule",
    "LogValueType",
    "StructuredLogFieldDictionaryV1",
    "build_log_field_dictionary_v1",
    "canonical_log_dictionary_json",
    "parse_log_dictionary_json",
    "parse_log_sample_jsonl",
    "render_log_sample_jsonl",
    "sample_log_events_v1",
    "validate_log_documentation",
    "validate_log_documentation_files",
    "write_log_documentation_v1",
]
