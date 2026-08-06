"""Scan release artifacts for supported privacy fixtures and sensitive values."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from rag_mvp.safety.detectors import has_unclosed_private_key
from rag_mvp.safety.models import SensitiveKind
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, RedactionError

_SCANNER_VERSION: Final[str] = "artifact-scan-v1"
_SAFE_LABEL: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_REDACTED_FIXTURE: Final[str] = "[REDACTED_FIXTURE]"
_REDACTED_FILE: Final[str] = "[REDACTED_FILE]"


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
    text: str, fixtures: tuple[PrivacyFixture, ...]
) -> tuple[Counter[str], Counter[str]]:
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    exact_counts: Counter[str] = Counter()
    for fixture in fixtures:
        count = normalized_text.count(fixture.value)
        if count:
            exact_counts[fixture.category] += count

    placeholder_ranges: list[tuple[int, int]] = []
    for kind in SensitiveKind:
        placeholder_ranges.extend(
            (match.start(), match.end())
            for match in re.finditer(re.escape(kind.placeholder), normalized_text)
        )
    detector_counts: Counter[str] = Counter(
        span.kind.value
        for span in DEFAULT_REDACTOR.detect(normalized_text)
        if not any(start <= span.start and span.end <= end for start, end in placeholder_ranges)
    )
    if has_unclosed_private_key(normalized_text):
        detector_counts[SensitiveKind.SECRET.value] += 1
    return exact_counts, detector_counts


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
            artifact_exact, artifact_detector = _scan_text(
                f"{artifact.name}\n{content}", fixture_set.fixtures
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
