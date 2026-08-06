"""Versioned, privacy-safe JSON evaluation reports.

The persisted JSON document is the report source of truth.  Serialization is
canonical so its content hash, the HTML embedding, and offline verification do
not depend on mapping insertion order or pretty-printing choices.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from functools import cache
from hashlib import sha256
from importlib.resources import files
from itertools import pairwise
from pathlib import Path
from typing import Any, Never, cast

from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]
from pydantic import BaseModel

from rag_mvp.safety.output import OutputRedactionError, redact_output
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, Redactor

REPORT_SCHEMA_VERSION = "1.0.0"
REPORT_SCHEMA_URI = "https://rag-mvp.local/schemas/evaluation-report-v1.schema.json"
REPORT_SCHEMA_RESOURCE = "evaluation-report-v1.schema.json"
MAX_REPORT_BYTES = 16 * 1024 * 1024
_SAFE_ERROR_PATH_TOKEN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class SchemaIssue:
    """A content-free schema/semantic failure suitable for logs and APIs."""

    path: str
    keyword: str
    message: str


class ReportValidationError(ValueError):
    """Raised when a report cannot satisfy the frozen v1 contract."""

    def __init__(self, issues: Sequence[SchemaIssue]) -> None:
        normalized = tuple(issues)
        if not normalized:
            raise ValueError("report validation error requires at least one issue")
        self.issues = normalized
        first = normalized[0]
        super().__init__(
            f"report validation failed at {first.path} ({first.keyword}); "
            f"{len(normalized)} issue(s)"
        )


class ReportSerializationError(ValueError):
    """Raised when an input is not safely representable as JSON."""


class ReportWriteError(OSError):
    """Raised when an immutable report target cannot be safely written."""


def load_report_schema() -> dict[str, Any]:
    """Load a defensive copy of the meta-validated packaged report schema."""

    return deepcopy(_packaged_report_schema())


@cache
def _packaged_report_schema() -> dict[str, Any]:
    """Load and meta-validate the packaged Draft 2020-12 report schema once."""

    resource = files("rag_mvp.evaluation").joinpath("schemas", REPORT_SCHEMA_RESOURCE)
    raw = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("packaged report schema is not an object")
    schema = cast(dict[str, Any], raw)
    Draft202012Validator.check_schema(schema)
    return schema


@cache
def _report_validator() -> Draft202012Validator:
    return Draft202012Validator(_packaged_report_schema(), format_checker=FormatChecker())


def validate_report(report: Mapping[str, object] | BaseModel) -> JsonObject:
    """Normalize and validate a report against schema v1 and gate invariants.

    Error objects deliberately contain only JSON pointers and validation
    keywords.  They never echo report values, which may contain sensitive input
    when this function is used to reject a malformed caller payload.
    """

    normalized = _normalize_report(report)
    issues = tuple(
        _schema_issue(error)
        for error in sorted(
            _report_validator().iter_errors(normalized),
            key=lambda item: (tuple(str(part) for part in item.absolute_path), item.validator),
        )
    )
    if issues:
        raise ReportValidationError(issues)

    semantic_issues = _semantic_issues(normalized)
    if semantic_issues:
        raise ReportValidationError(semantic_issues)
    return normalized


def canonical_report_json(report: Mapping[str, object] | BaseModel) -> str:
    """Return the validated canonical JSON representation without a final newline."""

    normalized = validate_report(report)
    return _canonical_json_value(normalized)


def report_content_hash(report: Mapping[str, object] | BaseModel) -> str:
    """Hash exactly the canonical UTF-8 bytes persisted by :func:`write_json_report`."""

    payload = (canonical_report_json(report) + "\n").encode("utf-8")
    return f"sha256:{sha256(payload).hexdigest()}"


def prepare_report(
    report: Mapping[str, object] | BaseModel,
    *,
    redactor: Redactor = DEFAULT_REDACTOR,
) -> JsonObject:
    """Normalize, recursively redact, and validate a report before publication."""

    normalized = _normalize_report(report)
    try:
        redacted = redact_output(normalized, redactor=redactor)
    except (OutputRedactionError, TypeError, ValueError, RecursionError) as error:
        raise ReportSerializationError("report redaction failed closed") from error
    if not isinstance(redacted, dict):
        raise ReportSerializationError("redacted report is not an object")
    return validate_report(cast(Mapping[str, object], redacted))


def write_json_report(
    report: Mapping[str, object] | BaseModel,
    output_path: Path | str,
    *,
    overwrite: bool = False,
    redactor: Redactor = DEFAULT_REDACTOR,
) -> Path:
    """Safely publish one canonical JSON report.

    The default is immutable: an existing run artifact is never replaced.  A
    deliberate ``overwrite=True`` is retained for controlled test fixtures and
    local report regeneration, and rejects symbolic-link targets.
    """

    target = Path(output_path)
    if target.suffix.casefold() != ".json":
        raise ReportWriteError("JSON report path must end in .json")
    prepared = prepare_report(report, redactor=redactor)
    payload = canonical_report_json(prepared) + "\n"
    return _atomic_write_text(target, payload, overwrite=overwrite)


def load_json_report(path: Path | str) -> JsonObject:
    """Load a bounded UTF-8 JSON report, rejecting duplicate keys and constants."""

    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as error:
        raise ReportSerializationError("report file is unavailable") from error
    if size <= 0 or size > MAX_REPORT_BYTES:
        raise ReportSerializationError("report file size is outside allowed bounds")
    try:
        text = source.read_text(encoding="utf-8")
        raw = decode_json_report(text)
    except (OSError, UnicodeError, json.JSONDecodeError, ReportSerializationError) as error:
        if isinstance(error, ReportSerializationError):
            raise
        raise ReportSerializationError("report file is not valid UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise ReportSerializationError("report document is not an object")
    return validate_report(cast(Mapping[str, object], raw))


def decode_json_report(text: str) -> object:
    """Strictly decode JSON text without accepting duplicate keys or NaN values."""

    if not isinstance(text, str):
        raise TypeError("JSON report text must be a string")
    return _strict_json_loads(text)


def canonical_report_document(report: Mapping[str, object] | BaseModel) -> bytes:
    """Return the exact canonical bytes used for an on-disk JSON report."""

    return (canonical_report_json(report) + "\n").encode("utf-8")


def canonical_json_value(value: object) -> str:
    """Serialize a JSON-shaped sub-value with the report's canonical encoding."""

    try:
        serialized = json.dumps(
            value,
            default=_json_default,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ReportSerializationError("value is not safely JSON serializable") from error
    return _escape_html_sensitive_json(serialized)


def _normalize_report(report: Mapping[str, object] | BaseModel) -> JsonObject:
    try:
        if isinstance(report, BaseModel):
            candidate: object = report.model_dump(mode="json")
        else:
            candidate = report
        serialized = json.dumps(
            candidate,
            default=_json_default,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        normalized = _strict_json_loads(serialized)
    except (TypeError, ValueError, OverflowError, json.JSONDecodeError) as error:
        raise ReportSerializationError("report is not safely JSON serializable") from error
    if not isinstance(normalized, dict):
        raise ReportSerializationError("report document must be an object")
    return cast(JsonObject, normalized)


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise TypeError("non-finite decimal")
        return str(value)
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError("unsupported report value type")


def _strict_json_loads(text: str) -> object:
    return json.loads(
        text,
        object_pairs_hook=_object_without_duplicates,
        parse_constant=_reject_json_constant,
    )


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReportSerializationError("report JSON contains a duplicate object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Never:
    del value
    raise ReportSerializationError("report JSON contains a non-finite number")


def _canonical_json_value(value: object) -> str:
    return canonical_json_value(value)


def _escape_html_sensitive_json(serialized: str) -> str:
    # Keep the exact same safe JSON bytes in the source report and HTML script.
    # Escaping these characters prevents a value from terminating the script tag.
    return (
        serialized.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _schema_issue(error: Any) -> SchemaIssue:
    path = _json_pointer(tuple(error.absolute_path))
    keyword = str(error.validator)
    messages = {
        "required": "required property is missing",
        "additionalProperties": "unexpected property is present",
        "type": "value has the wrong JSON type",
        "const": "value does not match the report schema version",
        "enum": "value is outside the allowed enumeration",
        "format": "value has an invalid format",
        "pattern": "value does not match the required pattern",
        "minimum": "numeric value is below its minimum",
        "maximum": "numeric value is above its maximum",
        "minItems": "array has too few items",
        "minProperties": "object has too few properties",
        "uniqueItems": "array values must be unique",
        "oneOf": "value does not match exactly one allowed shape",
    }
    return SchemaIssue(
        path=path,
        keyword=keyword,
        message=messages.get(keyword, "value does not satisfy the report schema"),
    )


def _semantic_issues(report: JsonObject) -> tuple[SchemaIssue, ...]:
    issues: list[SchemaIssue] = []
    thresholds = cast(dict[str, dict[str, object]], report["thresholds"])
    metrics = cast(dict[str, object], report["metrics"])
    aggregate = cast(dict[str, dict[str, object]], metrics["aggregate"])

    for name, metric in aggregate.items():
        threshold = thresholds[name]
        if metric["operator"] != threshold["operator"] or metric["threshold"] != threshold["value"]:
            issues.append(
                SchemaIssue(
                    f"/metrics/aggregate/{_pointer_token(name)}",
                    "threshold-parity",
                    "aggregate decision does not use the declared threshold",
                )
            )
            continue
        value = cast(float | None, metric["value"])
        expected = (
            False
            if value is None
            else _compare(
                value,
                cast(str, metric["operator"]),
                cast(float, metric["threshold"]),
            )
        )
        if metric["passed"] is not expected:
            issues.append(
                SchemaIssue(
                    f"/metrics/aggregate/{_pointer_token(name)}/passed",
                    "decision",
                    "metric decision disagrees with its unrounded value",
                )
            )

    gate = cast(dict[str, object], report["gate"])
    performance = cast(dict[str, object], report["performance"])
    case_executions_complete = performance["errors"] == 0
    quality_passed = case_executions_complete and all(
        cast(bool, metric["passed"]) for metric in aggregate.values()
    )
    quality_valid = case_executions_complete and all(
        cast(int, metric["eligible_cases"]) > 0 and metric["value"] is not None
        for metric in aggregate.values()
    )
    if gate["quality_passed"] is not quality_passed:
        issues.append(
            SchemaIssue(
                "/gate/quality_passed",
                "gate-parity",
                "quality gate disagrees with aggregate metric decisions",
            )
        )

    privacy = cast(dict[str, object], report["privacy"])
    privacy_checks = cast(list[dict[str, object]], privacy["checks"])
    privacy_passed = (
        privacy["raw_supported_pii_matches"] == 0
        and privacy["raw_secret_matches"] == 0
        and all(check["passed"] is True and check["matches"] == 0 for check in privacy_checks)
    )
    if privacy["passed"] is not privacy_passed:
        issues.append(
            SchemaIssue(
                "/privacy/passed",
                "privacy-parity",
                "privacy decision disagrees with recorded checks",
            )
        )
    if gate["privacy_passed"] is not privacy_passed:
        issues.append(
            SchemaIssue(
                "/gate/privacy_passed",
                "gate-parity",
                "privacy gate disagrees with the privacy section",
            )
        )

    provenance = cast(dict[str, object], report["provenance"])
    cost = cast(dict[str, object], report["cost"])
    if cost["pricing_version"] != provenance["pricing_version"]:
        issues.append(
            SchemaIssue(
                "/cost/pricing_version",
                "provenance-parity",
                "cost and provenance pricing versions differ",
            )
        )
    if cost["complete"] is True and (
        cost["currency"] is None
        or cost["estimated_cost"] is None
        or cost["cost_per_1000_calls"] is None
        or cast(list[object], cost["unknown_reasons"])
    ):
        issues.append(
            SchemaIssue(
                "/cost/complete",
                "cost-completeness",
                "complete cost evidence contains unknown values",
            )
        )

    performance = cast(dict[str, object], report["performance"])
    complete_latency = performance["complete_latency_ms"]
    if isinstance(complete_latency, dict):
        _check_latency_summary(
            cast(dict[str, object], complete_latency),
            "/performance/complete_latency_ms",
            issues,
        )
    for stage, summary in cast(
        dict[str, dict[str, object]], performance["stage_latency_ms"]
    ).items():
        _check_latency_summary(
            summary,
            f"/performance/stage_latency_ms/{_safe_pointer_token(stage)}",
            issues,
        )
    latency_evidence_count = cast(
        int,
        performance.get("latency_evidence_count", performance["case_count"]),
    )
    performance_unknown = cast(list[object], performance.get("unknown_reasons", []))
    performance_complete = (
        isinstance(complete_latency, dict)
        and latency_evidence_count == performance["case_count"]
        and not performance_unknown
    )
    run_valid = quality_valid and cast(bool, cost["complete"]) and performance_complete
    if gate["valid"] is not run_valid:
        issues.append(
            SchemaIssue(
                "/gate/valid",
                "gate-parity",
                "run validity disagrees with quality, cost, or performance evidence",
            )
        )

    issue_records = cast(list[dict[str, object]], report["issues"])
    seen_issue_ids: set[str] = set()
    for index, issue in enumerate(issue_records):
        path = f"/issues/{index}"
        issue_id = cast(str, issue["issue_id"])
        if issue_id in seen_issue_ids:
            issues.append(SchemaIssue(f"{path}/issue_id", "unique", "issue IDs must be unique"))
        seen_issue_ids.add(issue_id)
        issues.extend(_issue_semantic_issues(issue, path))

    issues_passed = len(issue_records) >= 2 and all(
        cast(bool, issue["passed"]) for issue in issue_records
    )
    if gate["issues_passed"] is not issues_passed:
        issues.append(
            SchemaIssue(
                "/gate/issues_passed",
                "gate-parity",
                "issue gate disagrees with the issue investigations",
            )
        )

    final_passed = all(
        gate[name] is True
        for name in (
            "valid",
            "quality_passed",
            "privacy_passed",
            "reporting_passed",
            "issues_passed",
        )
    )
    if gate["final_passed"] is not final_passed:
        issues.append(
            SchemaIssue(
                "/gate/final_passed",
                "gate-parity",
                "final gate disagrees with its component gates",
            )
        )
    if final_passed and cast(list[object], gate["failures"]):
        issues.append(
            SchemaIssue(
                "/gate/failures",
                "gate-parity",
                "a passing final gate cannot contain failures",
            )
        )
    return tuple(issues)


def _check_latency_summary(
    summary: dict[str, object],
    path: str,
    issues: list[SchemaIssue],
) -> None:
    values = [cast(float, summary["p50"]), cast(float, summary["p90"])]
    if "p99" in summary:
        values.append(cast(float, summary["p99"]))
    values.append(cast(float, summary["max"]))
    if any(left > right for left, right in pairwise(values)):
        issues.append(
            SchemaIssue(
                path,
                "percentile-order",
                "latency percentiles must be monotonically non-decreasing",
            )
        )


def _issue_semantic_issues(issue: dict[str, object], path: str) -> tuple[SchemaIssue, ...]:
    issues: list[SchemaIssue] = []
    comparison = cast(dict[str, dict[str, object]], issue["comparison"])
    baseline_identity = comparison["baseline"]
    post_identity = comparison["post_fix"]
    for field in (
        "dataset_version",
        "corpus_version",
        "case_ids_hash",
        "scorer_version",
        "eligible_cases",
    ):
        if baseline_identity[field] != post_identity[field]:
            issues.append(
                SchemaIssue(
                    f"{path}/comparison/post_fix/{field}",
                    "comparison-compatibility",
                    "pre-fix and post-fix comparison identities differ",
                )
            )

    run_references = cast(list[str], issue["run_references"])
    if (
        cast(str, baseline_identity["run_id"]) not in run_references
        or cast(str, post_identity["run_id"]) not in run_references
    ):
        issues.append(
            SchemaIssue(
                f"{path}/run_references",
                "comparison-evidence",
                "comparison run IDs must appear in issue evidence",
            )
        )

    baseline = cast(float, issue["baseline_value"])
    post_fix = cast(float, issue["post_fix_value"])
    if baseline == 0:
        calculated = None
    elif issue["direction"] == "higher-is-better":
        calculated = (post_fix - baseline) / baseline * 100
    else:
        calculated = (baseline - post_fix) / baseline * 100
    declared = cast(float, issue["relative_improvement_percent"])
    if calculated is None or not math.isclose(
        declared,
        calculated,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        issues.append(
            SchemaIssue(
                f"{path}/relative_improvement_percent",
                "improvement-calculation",
                "declared relative improvement is not reproducible",
            )
        )
    passed = calculated is not None and calculated >= 10
    if issue["passed"] is not passed:
        issues.append(
            SchemaIssue(
                f"{path}/passed",
                "improvement-gate",
                "issue decision disagrees with the unrounded improvement",
            )
        )
    return tuple(issues)


def _compare(value: float, operator: str, threshold: float) -> bool:
    match operator:
        case ">":
            return value > threshold
        case ">=":
            return value >= threshold
        case "<":
            return value < threshold
        case "<=":
            return value <= threshold
        case "==":
            return value == threshold
        case _:
            raise RuntimeError("validated report contains an unknown threshold operator")


def _json_pointer(path: tuple[object, ...]) -> str:
    if not path:
        return "/"
    return "/" + "/".join(_safe_pointer_token(str(part)) for part in path)


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _safe_pointer_token(value: str) -> str:
    if _SAFE_ERROR_PATH_TOKEN.fullmatch(value) is None:
        return "*"
    return _pointer_token(value)


def _atomic_write_text(target: Path, payload: str, *, overwrite: bool) -> Path:
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")
    expanded = target.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    if absolute.is_symlink():
        raise ReportWriteError("report target must not be a symbolic link")
    target = absolute.parent.resolve() / absolute.name
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise ReportWriteError("immutable report target already exists")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            try:
                os.link(temporary, target)
            except FileExistsError as error:
                raise ReportWriteError("immutable report target already exists") from error
            temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


class JsonReportGenerator:
    """Small injectable façade used by evaluation services and worker pools."""

    def __init__(self, *, redactor: Redactor = DEFAULT_REDACTOR) -> None:
        self._redactor = redactor

    def generate(
        self,
        report: Mapping[str, object] | BaseModel,
        output_path: Path | str,
        *,
        overwrite: bool = False,
    ) -> Path:
        return write_json_report(
            report,
            output_path,
            overwrite=overwrite,
            redactor=self._redactor,
        )
