"""Scan release artifacts for supported privacy fixtures and sensitive values."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Final, cast

from rag_mvp.safety.detectors import has_unclosed_private_key
from rag_mvp.safety.models import SensitiveKind
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, RedactionError

_SCANNER_VERSION: Final[str] = "artifact-scan-v1"
_SAFE_LABEL: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PROMETHEUS_SAMPLE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<prefix>[A-Za-z_:][A-Za-z0-9_:]*(?:\{.*\})?)\s+"
    r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|[+-]?Inf)"
    r"(?:\s+\d+)?\s*$"
)
_REDACTED_FIXTURE: Final[str] = "[REDACTED_FIXTURE]"
_REDACTED_FILE: Final[str] = "[REDACTED_FILE]"
_RAW_FILESYSTEM_PATH: Final[re.Pattern[str]] = re.compile(
    r"(?:(?<![A-Za-z])[A-Za-z]:[\\/]|\\\\|"
    r"/(?:home|Users|tmp|var|etc|opt|root|mnt|workspace)/)"
)
_COMPARISON_CSV_FIELD_LIMIT: Final[int] = 64 * 1024 * 1024
_COMPARISON_PROJECTION_NAMES: Final[frozenset[str]] = frozenset(
    {
        "comparison-report.html",
        "comparison-report.txt",
        "comparison-report.csv",
    }
)
_COMPARISON_HTML_VISIBLE: Final[re.Pattern[str]] = re.compile(
    r'(?P<prefix><pre id="comparison-result-visible">)'
    r"(?P<canonical>.*?)"
    r"(?P<suffix></pre>)",
    re.DOTALL,
)
_COMPARISON_HTML_EMBEDDED: Final[re.Pattern[str]] = re.compile(
    r'(?P<prefix><script id="comparison-result-json" type="application/json" '
    r'data-encoding="base64">)'
    r"(?P<canonical>[A-Za-z0-9+/=]+)"
    r"(?P<suffix></script>)"
)
_COMPARISON_HTML_BODY: Final[re.Pattern[str]] = re.compile(
    r"(?P<prefix><tbody>)(?P<rows>.*?)(?P<suffix></tbody>)",
    re.DOTALL,
)
_COMPARISON_HTML_ROW: Final[re.Pattern[str]] = re.compile(r"<tr>(.*?)</tr>", re.DOTALL)
_COMPARISON_HTML_CELL: Final[re.Pattern[str]] = re.compile(r"<td>(.*?)</td>", re.DOTALL)
_EVALUATION_HTML_EMBEDDED: Final[re.Pattern[str]] = re.compile(
    r'<script id="evaluation-report-json" type="application/json">'
    r"(?P<canonical>.*?)"
    r"</script>",
    re.DOTALL,
)
_COMPARISON_HTML_NUMERIC_COLUMNS: Final[frozenset[int]] = frozenset({3, 4, 10, 11, 12})
_COMPARISON_TEXT_NUMERIC_COLUMNS: Final[frozenset[int]] = frozenset({5, 6, 12, 13, 14, 15})
_COMPARISON_CSV_NUMERIC_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "known_partial_cost",
        "total_cost",
        "value",
        "numerator",
        "denominator",
        "baseline_delta",
    }
)


@dataclass(frozen=True, slots=True)
class PrivacyFixture:
    """One synthetic value used to prove privacy behavior."""

    fixture_id: str
    category: str
    value: str


@dataclass(frozen=True, slots=True)
class FixtureSet:
    """Validated fixture metadata without any reporting behavior."""

    version: str
    digest: str
    fixtures: tuple[PrivacyFixture, ...]


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be an object with string keys")
    return cast(dict[str, object], value)


def _safe_label(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_LABEL.fullmatch(value) is None:
        raise ValueError(f"{field} must be a safe label")
    return value


def load_fixture_set(path: Path) -> FixtureSet:
    """Load a synthetic-only fixture file and retain only validated values in memory."""

    raw = path.read_bytes()
    digest = f"sha256:{hashlib.sha256(raw).hexdigest()}"
    decoded: object = json.loads(raw.decode("utf-8"))
    payload = _mapping(decoded, field="fixture document")

    version = payload.get("fixture_version")
    if version != "supported-fixtures-v1":
        raise ValueError("unsupported fixture version")
    if payload.get("synthetic_only") is not True:
        raise ValueError("fixture document must declare synthetic-only values")
    raw_fixtures = payload.get("fixtures")
    if not isinstance(raw_fixtures, list) or not raw_fixtures:
        raise ValueError("fixture document must contain a non-empty fixture list")

    fixtures: list[PrivacyFixture] = []
    fixture_ids: set[str] = set()
    fixture_values: set[str] = set()
    for index, raw_fixture in enumerate(raw_fixtures):
        fixture = _mapping(raw_fixture, field=f"fixtures[{index}]")
        fixture_id = _safe_label(fixture.get("id"), field=f"fixtures[{index}].id")
        category = _safe_label(fixture.get("category"), field=f"fixtures[{index}].category")
        value = fixture.get("value")
        if fixture.get("synthetic") is not True:
            raise ValueError("every fixture must be explicitly synthetic")
        if not isinstance(value, str) or not value or len(value) > 65_536:
            raise ValueError("fixture values must be bounded non-empty strings")
        if fixture_id in fixture_ids or value in fixture_values:
            raise ValueError("fixture identifiers and values must be unique")
        fixture_ids.add(fixture_id)
        fixture_values.add(value)
        fixtures.append(PrivacyFixture(fixture_id, category, value))

    return FixtureSet(version, digest, tuple(fixtures))


def _resolved(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except OSError:
        return Path(os.path.abspath(path))


def _is_excluded(path: Path, exclusions: tuple[Path, ...]) -> bool:
    resolved = _resolved(path)
    return any(resolved == excluded or resolved.is_relative_to(excluded) for excluded in exclusions)


def _collect_files(
    targets: Sequence[Path], exclusions: tuple[Path, ...]
) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    files: set[Path] = set()
    excluded_paths: set[Path] = set()
    errors: set[Path] = set()

    for requested_target in targets:
        target = _resolved(requested_target)
        if _is_excluded(target, exclusions):
            excluded_paths.add(target)
            continue
        if target.is_symlink():
            errors.add(target)
            continue
        if target.is_file():
            files.add(target)
            continue
        if not target.is_dir():
            errors.add(target)
            continue

        walk_target = target

        def record_walk_error(error: OSError, fallback: Path = walk_target) -> None:
            error_path = Path(error.filename) if error.filename is not None else fallback
            errors.add(_resolved(error_path))

        for root_text, directory_names, file_names in os.walk(
            target,
            topdown=True,
            onerror=record_walk_error,
            followlinks=False,
        ):
            root = Path(root_text)
            retained_directories: list[str] = []
            for directory_name in sorted(directory_names):
                directory = root / directory_name
                if _is_excluded(directory, exclusions):
                    excluded_paths.add(_resolved(directory))
                elif directory.is_symlink():
                    errors.add(_resolved(directory))
                else:
                    retained_directories.append(directory_name)
            directory_names[:] = retained_directories

            for file_name in sorted(file_names):
                artifact = root / file_name
                if _is_excluded(artifact, exclusions):
                    excluded_paths.add(_resolved(artifact))
                elif artifact.is_symlink():
                    errors.add(_resolved(artifact))
                else:
                    files.add(_resolved(artifact))

    return (
        tuple(sorted(files, key=str)),
        tuple(sorted(excluded_paths, key=str)),
        tuple(sorted(errors, key=str)),
    )


def _decode_artifact(raw: bytes) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="replace")

    sample = raw[:4096]
    threshold = max(2, len(sample) // 8)
    if sample[1::2].count(0) > threshold:
        return raw.decode("utf-16-le", errors="replace")
    if sample[0::2].count(0) > threshold:
        return raw.decode("utf-16-be", errors="replace")
    return raw.decode("utf-8", errors="replace")


def _scan_text(
    text: str,
    fixtures: tuple[PrivacyFixture, ...],
    *,
    detector_text: str | None = None,
) -> tuple[Counter[str], Counter[str]]:
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_detector_text = (
        normalized_text
        if detector_text is None
        else detector_text.replace("\r\n", "\n").replace("\r", "\n")
    )
    exact_counts: Counter[str] = Counter()
    for fixture in fixtures:
        count = normalized_text.count(fixture.value)
        if count == 0 and detector_text is not None:
            count = normalized_detector_text.count(fixture.value)
        if count:
            exact_counts[fixture.category] += count

    placeholder_ranges: list[tuple[int, int]] = []
    for kind in SensitiveKind:
        placeholder_ranges.extend(
            (match.start(), match.end())
            for match in re.finditer(re.escape(kind.placeholder), normalized_detector_text)
        )
    detector_counts: Counter[str] = Counter(
        span.kind.value
        for span in DEFAULT_REDACTOR.detect(normalized_detector_text)
        if not any(start <= span.start and span.end <= end for start, end in placeholder_ranges)
    )
    if has_unclosed_private_key(normalized_detector_text):
        detector_counts[SensitiveKind.SECRET.value] += 1
    detector_counts[SensitiveKind.SECRET.value] += len(
        tuple(_RAW_FILESYSTEM_PATH.finditer(normalized_detector_text))
    )
    return exact_counts, detector_counts


def _without_json_numbers(value: object) -> object:
    """Preserve JSON strings and structure while removing typed numeric values."""

    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int | float):
        return None
    if isinstance(value, list):
        return [_without_json_numbers(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _without_json_numbers(item)
            for key, item in cast(dict[str, object], value).items()
        }
    raise TypeError("unsupported JSON value")


def _comparison_display(value: object) -> str:
    if isinstance(value, dict) and value.get("status") == "unavailable":
        reason = value.get("reason")
        if isinstance(reason, str):
            return f"unavailable:{reason}"
    return str(value)


def _json_string_nodes(value: object) -> tuple[str, ...]:
    strings: list[str] = []

    def collect(node: object) -> None:
        if isinstance(node, str):
            strings.append(node)
            return
        if isinstance(node, list):
            for item in node:
                collect(item)
            return
        if isinstance(node, dict):
            for key, item in cast(dict[str, object], node).items():
                strings.append(str(key))
                collect(item)

    collect(value)
    return tuple(strings)


def _comparison_projection_rows(
    payload: Mapping[str, object],
    *,
    html_projection: bool,
) -> tuple[tuple[str, ...], ...]:
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("comparison candidates are unavailable")
    rows: list[tuple[str, ...]] = []
    for candidate_index, raw_candidate in enumerate(raw_candidates):
        candidate = _mapping(raw_candidate, field=f"candidates[{candidate_index}]")
        reference = _mapping(
            candidate.get("reference"),
            field=f"candidates[{candidate_index}].reference",
        )
        raw_metrics = candidate.get("metrics")
        raw_reasons = candidate.get("cost_unknown_reasons", [])
        if not isinstance(raw_metrics, list) or not isinstance(raw_reasons, list):
            raise ValueError("comparison candidate projection is invalid")
        reasons = tuple(str(item) for item in raw_reasons)
        common = (
            str(reference.get("variant_id")),
            str(candidate.get("status")),
            str(candidate.get("evidence_status")),
        )
        cost = (
            _comparison_display(candidate.get("known_partial_cost")),
            _comparison_display(candidate.get("total_cost")),
            str(candidate.get("cost_complete")).lower(),
            ",".join(reasons),
            "" if candidate.get("currency") is None else str(candidate.get("currency")),
        )
        for metric_index, raw_metric in enumerate(raw_metrics):
            metric = _mapping(
                raw_metric,
                field=f"candidates[{candidate_index}].metrics[{metric_index}]",
            )
            metric_tail = (
                str(metric.get("metric_id")),
                str(metric.get("unit")),
                _comparison_display(metric.get("value")),
            )
            if html_projection:
                rows.append(
                    (
                        *common,
                        *cost,
                        *metric_tail,
                        _comparison_display(metric.get("denominator")),
                        _comparison_display(metric.get("baseline_delta")),
                        str(metric.get("status")),
                        str(metric.get("gate_status")),
                    )
                )
                continue
            rows.append(
                (
                    *common,
                    ""
                    if candidate.get("safe_error_code") is None
                    else str(candidate.get("safe_error_code")),
                    str(reference.get("axis_value")),
                    *cost,
                    *metric_tail,
                    _comparison_display(metric.get("numerator")),
                    _comparison_display(metric.get("denominator")),
                    _comparison_display(metric.get("baseline_delta")),
                    str(metric.get("status")),
                    str(metric.get("gate_status")),
                )
            )
    return tuple(rows)


def _comparison_payload(canonical: str) -> tuple[Mapping[str, object], str]:
    decoded = json.loads(canonical)
    payload = _mapping(decoded, field="comparison result")
    if not isinstance(payload.get("comparison_id"), str):
        raise ValueError("comparison result identity is unavailable")
    semantic_json = json.dumps(
        _without_json_numbers(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    semantic = "\n".join((semantic_json, *_json_string_nodes(payload)))
    return payload, semantic


def _mask_html_comparison_row(
    row: str,
    expected: tuple[str, ...],
) -> str:
    cells = tuple(_COMPARISON_HTML_CELL.finditer(row))
    if len(cells) != len(expected):
        return row
    parts: list[str] = []
    cursor = 0
    for index, cell in enumerate(cells):
        parts.append(row[cursor : cell.start(1)])
        value = cell.group(1)
        parts.append(
            "0"
            if index in _COMPARISON_HTML_NUMERIC_COLUMNS and html.unescape(value) == expected[index]
            else value
        )
        cursor = cell.end(1)
    parts.append(row[cursor:])
    return "".join(parts)


def _comparison_html_detector_text(content: str) -> str | None:
    visible = _COMPARISON_HTML_VISIBLE.search(content)
    embedded = _COMPARISON_HTML_EMBEDDED.search(content)
    body = _COMPARISON_HTML_BODY.search(content)
    if visible is None or embedded is None or body is None:
        return None
    try:
        visible_canonical = html.unescape(visible.group("canonical"))
        embedded_canonical = base64.b64decode(
            embedded.group("canonical"),
            validate=True,
        ).decode("utf-8")
        if visible_canonical != embedded_canonical:
            return None
        payload, semantic = _comparison_payload(visible_canonical)
        expected_rows = _comparison_projection_rows(payload, html_projection=True)
    except (UnicodeError, ValueError, TypeError):
        return None

    normalized = _COMPARISON_HTML_VISIBLE.sub(
        lambda match: f"{match.group('prefix')}{html.escape(semantic)}{match.group('suffix')}",
        content,
        count=1,
    )
    normalized = _COMPARISON_HTML_EMBEDDED.sub(
        lambda match: f"{match.group('prefix')}[CANONICAL_JSON_SCANNED]{match.group('suffix')}",
        normalized,
        count=1,
    )

    def normalize_body(match: re.Match[str]) -> str:
        rows_text = match.group("rows")
        parts: list[str] = []
        cursor = 0
        for index, row_match in enumerate(_COMPARISON_HTML_ROW.finditer(rows_text)):
            parts.append(rows_text[cursor : row_match.start()])
            row = row_match.group(0)
            parts.append(
                _mask_html_comparison_row(row, expected_rows[index])
                if index < len(expected_rows)
                else row
            )
            cursor = row_match.end()
        parts.append(rows_text[cursor:])
        return f"{match.group('prefix')}{''.join(parts)}{match.group('suffix')}"

    return _COMPARISON_HTML_BODY.sub(normalize_body, normalized, count=1)


def _mask_delimited_comparison_row(
    actual: list[str],
    expected: tuple[str, ...],
    numeric_columns: frozenset[int],
) -> list[str]:
    if len(actual) != len(expected):
        return actual
    return [
        "0" if index in numeric_columns and value == expected[index] else value
        for index, value in enumerate(actual)
    ]


def _comparison_text_detector_text(content: str) -> str | None:
    lines = content.splitlines()
    if (
        len(lines) < 4
        or lines[0] != "comparison-report-text-v1"
        or not lines[2].startswith("canonical-json=")
        or lines[3] != "candidate-metrics:"
    ):
        return None
    canonical = lines[2].removeprefix("canonical-json=")
    try:
        payload, semantic = _comparison_payload(canonical)
        expected_rows = _comparison_projection_rows(payload, html_projection=False)
    except (ValueError, TypeError):
        return None
    normalized = list(lines)
    normalized[2] = f"canonical-json={semantic}"
    for index, expected in enumerate(expected_rows, start=4):
        if index >= len(normalized):
            break
        normalized[index] = "|".join(
            _mask_delimited_comparison_row(
                normalized[index].split("|"),
                expected,
                _COMPARISON_TEXT_NUMERIC_COLUMNS,
            )
        )
    return "\n".join(normalized)


def _comparison_csv_detector_text(content: str) -> str | None:
    csv.field_size_limit(max(csv.field_size_limit(), _COMPARISON_CSV_FIELD_LIMIT))
    try:
        rows = [list(row) for row in csv.reader(StringIO(content, newline=""))]
        if (
            len(rows) < 2
            or not rows[0]
            or rows[0][0] != "record_type"
            or len(rows[1]) < 3
            or rows[1][0] != "comparison-result"
        ):
            return None
        payload, semantic = _comparison_payload(rows[1][2])
        expected_values = _comparison_projection_rows(payload, html_projection=False)
        header = rows[0]
        expected_rows = tuple(("candidate-metric", "", "", *item) for item in expected_values)
        numeric_columns = frozenset(
            index for index, name in enumerate(header) if name in _COMPARISON_CSV_NUMERIC_COLUMNS
        )
        rows[1][2] = semantic
        for index, expected in enumerate(expected_rows, start=2):
            if index >= len(rows):
                break
            rows[index] = _mask_delimited_comparison_row(
                rows[index],
                expected,
                numeric_columns,
            )
    except (csv.Error, ValueError, TypeError):
        return None
    return "\n".join("\x1f".join(row) for row in rows)


def _comparison_projection_detector_text(artifact: Path, content: str) -> str | None:
    if artifact.name == "comparison-report.html":
        return _comparison_html_detector_text(content)
    if artifact.name == "comparison-report.txt":
        return _comparison_text_detector_text(content)
    if artifact.name == "comparison-report.csv":
        return _comparison_csv_detector_text(content)
    return None


def _evaluation_html_detector_text(content: str) -> str | None:
    """Scan report string nodes while excluding typed numeric measurements."""

    embedded = _EVALUATION_HTML_EMBEDDED.search(content)
    if embedded is None:
        return None
    try:
        payload = json.loads(embedded.group("canonical"))
        if not isinstance(payload, dict) or payload.get("schema_version") != "2.0.0":
            return None
        semantic = _without_json_numbers(payload)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return "\n".join(
        (
            json.dumps(semantic, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
            *_json_string_nodes(payload),
        )
    )


def _mixed_json_lines_without_numbers(content: str) -> str | None:
    decoder = json.JSONDecoder()
    normalized_lines: list[str] = []
    structured_lines = 0
    for line in content.splitlines():
        if not line.strip():
            normalized_lines.append(line)
            continue
        cursor = 0
        parsed_values: list[dict[str, object] | list[object]] = []
        while cursor < len(line):
            while cursor < len(line) and line[cursor].isspace():
                cursor += 1
            if cursor == len(line):
                break
            try:
                parsed, end = decoder.raw_decode(line, cursor)
            except (json.JSONDecodeError, TypeError, ValueError):
                parsed_values.clear()
                break
            if not isinstance(parsed, dict | list):
                parsed_values.clear()
                break
            parsed_values.append(parsed)
            cursor = end
        if not parsed_values:
            normalized_lines.append(line)
            continue
        structured_lines += 1
        normalized_lines.append(
            "".join(
                json.dumps(
                    _without_json_numbers(parsed),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                for parsed in parsed_values
            )
        )
    return "\n".join(normalized_lines) if structured_lines else None


def _prometheus_without_samples(content: str) -> str:
    normalized_lines: list[str] = []
    for line in content.splitlines():
        match = _PROMETHEUS_SAMPLE.fullmatch(line)
        normalized_lines.append(line if match is None else f"{match.group('prefix')} 0")
    return "\n".join(normalized_lines)


def _json_detector_text(artifact: Path, content: str) -> str | None:
    """Return semantic machine output without numeric nodes or sample values."""

    suffix = artifact.suffix.lower()
    try:
        if suffix == ".json":
            payload = _without_json_numbers(json.loads(content))
        elif suffix == ".prom":
            return _prometheus_without_samples(content)
        else:
            return _mixed_json_lines_without_numbers(content)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _safe_path(path: Path, fixtures: tuple[PrivacyFixture, ...]) -> str:
    resolved = _resolved(path)
    try:
        label = resolved.relative_to(_resolved(Path.cwd())).as_posix()
    except ValueError:
        parent_digest = hashlib.sha256(resolved.parent.as_posix().encode("utf-8")).hexdigest()[:12]
        label = f"[EXTERNAL-{parent_digest}]/{resolved.name}"
    for fixture in sorted(fixtures, key=lambda item: len(item.value), reverse=True):
        label = label.replace(fixture.value, _REDACTED_FIXTURE)
    try:
        return DEFAULT_REDACTOR.redact(label).redacted_text
    except RedactionError:
        return _REDACTED_FILE


def scan_artifacts(
    targets: Sequence[Path],
    *,
    fixture_path: Path,
    excludes: Sequence[Path] = (),
) -> dict[str, object]:
    """Return a content-free report; any finding, scan error, or empty scan fails."""

    fixture_set = load_fixture_set(fixture_path)
    exclusions = tuple(_resolved(path) for path in excludes)
    files, excluded_paths, collection_errors = _collect_files(targets, exclusions)

    exact_counts: Counter[str] = Counter({fixture.category: 0 for fixture in fixture_set.fixtures})
    detector_counts: Counter[str] = Counter({kind.value: 0 for kind in SensitiveKind})
    matched_files: set[Path] = set()
    scan_errors: set[Path] = set(collection_errors)
    scanned_files: list[Path] = []

    for artifact in files:
        try:
            content = _decode_artifact(artifact.read_bytes())
            detector_content = _comparison_projection_detector_text(artifact, content)
            if artifact.name in _COMPARISON_PROJECTION_NAMES and detector_content is None:
                raise ValueError("comparison projection cannot be normalized safely")
            if detector_content is None and artifact.name == "evaluation-report.html":
                detector_content = _evaluation_html_detector_text(content)
                if detector_content is None:
                    raise ValueError("evaluation HTML projection cannot be normalized safely")
            if detector_content is None:
                detector_content = _json_detector_text(artifact, content)
            artifact_exact, artifact_detector = _scan_text(
                f"{artifact.name}\n{content}",
                fixture_set.fixtures,
                detector_text=(
                    None if detector_content is None else f"{artifact.name}\n{detector_content}"
                ),
            )
        except (OSError, RedactionError, TypeError, ValueError, RecursionError):
            scan_errors.add(artifact)
            continue
        scanned_files.append(artifact)
        exact_counts.update(artifact_exact)
        detector_counts.update(artifact_detector)
        if sum(artifact_exact.values()) + sum(artifact_detector.values()) > 0:
            matched_files.add(artifact)

    exact_total = sum(exact_counts.values())
    detector_total = sum(detector_counts.values())
    passed = bool(scanned_files) and not scan_errors and exact_total == 0 and detector_total == 0

    def safe_path(path: Path) -> str:
        return _safe_path(path, fixture_set.fixtures)

    return {
        "version": {
            "scanner": _SCANNER_VERSION,
            "fixtures": fixture_set.version,
        },
        "hash": {"fixtures": fixture_set.digest},
        "categories": {
            "fixture": dict(sorted(exact_counts.items())),
            "detector": dict(sorted(detector_counts.items())),
        },
        "files": {
            "scanned": [safe_path(path) for path in scanned_files],
            "matched": [safe_path(path) for path in sorted(matched_files, key=str)],
            "excluded": [safe_path(path) for path in excluded_paths],
            "errors": [safe_path(path) for path in sorted(scan_errors, key=str)],
        },
        "counts": {
            "targets": len(targets),
            "scanned": len(scanned_files),
            "matched": len(matched_files),
            "excluded": len(excluded_paths),
            "fixture_matches": exact_total,
            "detector_matches": detector_total,
            "errors": len(scan_errors),
        },
        "passed": passed,
    }


def _failure_report(target_count: int) -> dict[str, object]:
    return {
        "version": {"scanner": _SCANNER_VERSION, "fixtures": "unavailable"},
        "hash": {"fixtures": "unavailable"},
        "categories": {"fixture": {}, "detector": {}},
        "files": {"scanned": [], "matched": [], "excluded": [], "errors": []},
        "counts": {
            "targets": target_count,
            "scanned": 0,
            "matched": 0,
            "excluded": 0,
            "fixture_matches": 0,
            "detector_matches": 0,
            "errors": 1,
        },
        "passed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", type=Path, help="files or directories to scan")
    parser.add_argument("--fixtures", required=True, type=Path, help="synthetic fixture JSON")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        type=Path,
        help="explicit file or directory exclusion; may be repeated",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    targets = cast(list[Path], args.targets)
    fixture_path = cast(Path, args.fixtures)
    excludes = cast(list[Path], args.exclude)
    try:
        report = scan_artifacts(targets, fixture_path=fixture_path, excludes=excludes)
    except Exception:
        report = _failure_report(len(targets))
        exit_code = 2
    else:
        exit_code = 0 if report["passed"] is True else 1
    print(json.dumps(report, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
