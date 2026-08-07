"""Immutable, semantically verified multi-format artifacts for one comparison."""

from __future__ import annotations

import base64
import csv
import hashlib
import html
import os
import re
import shutil
import stat
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from filelock import FileLock

from rag_mvp.domain import ArtifactDescriptor, UnavailableValue
from rag_mvp.domain._base import utc_now
from rag_mvp.evaluation.comparison import (
    COMPARISON_RESULT_SCHEMA_VERSION,
    ComparisonArtifactManifest,
    ComparisonCandidateResult,
    ComparisonDomainError,
    ComparisonEvidenceStatus,
    ComparisonResult,
    ComparisonSuite,
    ResolvedComparisonArtifact,
    VerifiedCandidateReport,
    canonical_candidate_evidence,
    canonical_comparison_manifest,
    load_verified_candidate_report,
)
from rag_mvp.evaluation.experiment import ExperimentPlan
from rag_mvp.evaluation.json_report import canonical_json_value, decode_json_report
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, Redactor

COMPARISON_MANIFEST_FILENAME = "comparison-artifact-manifest.json"
MAX_COMPARISON_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_COMPARISON_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_COMPARISON_BUNDLE_BYTES = 64 * 1024 * 1024
MAX_COMPARISON_BUNDLE_ENTRIES = 128

csv.field_size_limit(max(csv.field_size_limit(), MAX_COMPARISON_ARTIFACT_BYTES))

_CORE_CONTRACTS: Mapping[str, tuple[str, str, str]] = MappingProxyType(
    {
        "comparison-plan-json": (
            "experiment-plan-v1",
            "application/json",
            "comparison-plan.json",
        ),
        "comparison-report-json": (
            COMPARISON_RESULT_SCHEMA_VERSION,
            "application/json",
            "comparison-report.json",
        ),
        "comparison-report-html": (
            "comparison-report-html-v1",
            "text/html",
            "comparison-report.html",
        ),
        "comparison-report-txt": (
            "comparison-report-text-v1",
            "text/plain",
            "comparison-report.txt",
        ),
        "comparison-report-csv": (
            "comparison-report-csv-v1",
            "text/csv",
            "comparison-report.csv",
        ),
    }
)
_HTML_EMBEDDED = re.compile(
    r'<script id="comparison-result-json" type="application/json" '
    r'data-encoding="base64">([A-Za-z0-9+/=]+)</script>'
)
_HTML_VISIBLE = re.compile(
    r'<pre id="comparison-result-visible">(.*?)</pre>',
    re.DOTALL,
)
_HTML_DIGEST = re.compile(r'<p id="comparison-result-digest">(sha256:[0-9a-f]{64})</p>')
_RAW_FILESYSTEM_PATH = re.compile(
    r"(?:(?<![A-Za-z])[A-Za-z]:[\\/]|\\\\|"
    r"/(?:home|Users|tmp|var|etc|opt|root|mnt|workspace)/)"
)
_CSV_HEADER = (
    "record_type",
    "canonical_json_sha256",
    "canonical_json",
    "candidate_id",
    "candidate_status",
    "evidence_status",
    "safe_error_code",
    "axis_value",
    "known_partial_cost",
    "total_cost",
    "cost_complete",
    "cost_unknown_reasons",
    "currency",
    "metric_id",
    "unit",
    "value",
    "numerator",
    "denominator",
    "baseline_delta",
    "status",
    "gate_status",
)
_STATIC_PROJECTION_TEXT = (
    "Comparison report",
    "Canonical comparison result",
    "comparison-report-text-v1",
    "canonical-json-sha256",
    "canonical-json",
    "candidate-metrics",
    *_CSV_HEADER,
)


class ComparisonArtifactError(RuntimeError):
    """Stable immutable-comparison artifact failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class PublishedComparisonArtifacts:
    manifest: ComparisonArtifactManifest


@dataclass(frozen=True, slots=True)
class ComparisonArtifactCatalog:
    """Publish and resolve semantic comparison bytes without exposing local paths."""

    root: Path
    redactor: Redactor = DEFAULT_REDACTOR

    def __post_init__(self) -> None:
        declared = _absolute_path(Path(self.root).expanduser())
        _safe_catalog_root(declared)
        object.__setattr__(self, "root", declared)

    def publish(
        self,
        suite: ComparisonSuite,
        result: ComparisonResult,
        reports: Mapping[str, VerifiedCandidateReport],
    ) -> PublishedComparisonArtifacts:
        ordered_reports = _validate_publication(suite, result, reports)
        payloads = _comparison_payloads(result, ordered_reports)
        _validate_payloads(
            payloads,
            self.redactor,
            privacy_values=(result.plan, result),
        )
        root = _safe_catalog_root(self.root)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root = _safe_catalog_root(self.root)
        target = _bundle_root(root, suite.comparison_id)
        lock_path = root / ".publication.lock"
        if _is_link(lock_path) or (lock_path.exists() and not lock_path.is_file()):
            raise ComparisonArtifactError("comparison-artifact-lock-unsafe")
        lock = FileLock(str(lock_path))
        with lock:
            root = _safe_catalog_root(self.root)
            target = _bundle_root(root, suite.comparison_id)
            if target.exists() or _is_link(target):
                raise ComparisonArtifactError("comparison-artifact-bundle-exists")
            temporary = _absolute_path(
                Path(tempfile.mkdtemp(prefix=f".{suite.comparison_id}.", dir=root))
            )
            if temporary.parent != root or not temporary.name.startswith(
                f".{suite.comparison_id}."
            ):
                raise ComparisonArtifactError("comparison-artifact-temporary-path-invalid")
            try:
                timestamp = utc_now()
                descriptors = _publication_descriptors(
                    suite.plan,
                    result,
                    ordered_reports,
                    payloads,
                    timestamp,
                )
                manifest = ComparisonArtifactManifest.create(
                    comparison_id=suite.comparison_id,
                    plan=suite.plan,
                    artifacts=descriptors,
                    created_at=timestamp,
                )
                for descriptor in descriptors:
                    path = temporary.joinpath(*descriptor.relative_path.split("/"))
                    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    _write_new(path, payloads[descriptor.artifact_id])
                _write_new(
                    temporary / COMPARISON_MANIFEST_FILENAME,
                    canonical_comparison_manifest(manifest),
                )
                if target.exists() or _is_link(target):
                    raise ComparisonArtifactError("comparison-artifact-bundle-exists")
                _atomic_publish_directory(temporary, target)
            except BaseException:
                _remove_temporary(temporary, root)
                raise
        verified = self.manifest(suite.comparison_id)
        if verified is None:
            raise ComparisonArtifactError("comparison-artifact-publication-failed")
        return PublishedComparisonArtifacts(manifest=verified)

    def manifest(self, comparison_id: str) -> ComparisonArtifactManifest | None:
        bundle = _bundle_root(_safe_catalog_root(self.root), comparison_id)
        if not bundle.exists():
            return None
        manifest, _ = _verify_bundle(bundle, self.redactor)
        return manifest

    def resolve(
        self,
        comparison_id: str,
        artifact_id: str,
    ) -> ResolvedComparisonArtifact | None:
        bundle = _bundle_root(_safe_catalog_root(self.root), comparison_id)
        if not bundle.exists():
            return None
        manifest, contents = _verify_bundle(bundle, self.redactor)
        descriptor = next(
            (item for item in manifest.artifacts if item.artifact_id == artifact_id),
            None,
        )
        if descriptor is None:
            return None
        return ResolvedComparisonArtifact(
            descriptor=descriptor,
            content=contents[artifact_id],
        )


def _validate_publication(
    suite: ComparisonSuite,
    result: ComparisonResult,
    reports: Mapping[str, VerifiedCandidateReport],
) -> tuple[VerifiedCandidateReport, ...]:
    expected_references = tuple(item.reference for item in suite.candidates)
    if (
        suite.status.value != "completed"
        or result.comparison_id != suite.comparison_id
        or result.plan != suite.plan
        or tuple(item.reference for item in result.candidates) != expected_references
        or tuple(item.status for item in result.candidates)
        != tuple(item.latest.status for item in suite.candidates)
    ):
        raise ComparisonArtifactError("comparison-publication-identity-invalid")
    available = tuple(
        item
        for item in result.candidates
        if item.evidence_status is ComparisonEvidenceStatus.AVAILABLE
    )
    expected_ids = tuple(item.reference.variant_id for item in available)
    if set(reports) != set(expected_ids) or len(reports) != len(expected_ids):
        raise ComparisonArtifactError("comparison-publication-candidate-set-invalid")
    ordered: list[VerifiedCandidateReport] = []
    for candidate in available:
        report = reports[candidate.reference.variant_id]
        try:
            verified = load_verified_candidate_report(
                candidate.reference,
                report.descriptor,
                canonical_candidate_evidence(report.evidence),
                comparison_id=suite.comparison_id,
                plan=suite.plan,
            )
        except (ComparisonDomainError, TypeError, ValueError):
            raise ComparisonArtifactError("comparison-publication-candidate-invalid") from None
        if (
            verified != report
            or candidate.source_descriptor != verified.descriptor
            or candidate.source_evidence != verified.evidence
        ):
            raise ComparisonArtifactError("comparison-publication-candidate-mismatch")
        ordered.append(verified)
    return tuple(ordered)


def _comparison_payloads(
    result: ComparisonResult,
    reports: Sequence[VerifiedCandidateReport],
) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {
        "comparison-plan-json": _canonical_model(result.plan),
        "comparison-report-json": _canonical_model(result),
        "comparison-report-html": _render_html(result),
        "comparison-report-txt": _render_text(result),
        "comparison-report-csv": _render_csv(result),
    }
    payloads.update(
        (report.descriptor.artifact_id, canonical_candidate_evidence(report.evidence))
        for report in reports
    )
    return payloads


def _canonical_model(model: ExperimentPlan | ComparisonResult) -> bytes:
    return (canonical_json_value(model.model_dump(mode="json")) + "\n").encode("utf-8")


def _canonical_result_text(result: ComparisonResult) -> str:
    return canonical_json_value(result.model_dump(mode="json"))


def _result_digest(canonical: str) -> str:
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _render_html(result: ComparisonResult) -> bytes:
    canonical = _canonical_result_text(result)
    digest = _result_digest(canonical)
    encoded = base64.b64encode(canonical.encode("utf-8")).decode("ascii")
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(candidate.reference.variant_id)}</td>"
        f"<td>{html.escape(candidate.status.value)}</td>"
        f"<td>{html.escape(candidate.evidence_status.value)}</td>"
        f"<td>{html.escape(_display(candidate.known_partial_cost))}</td>"
        f"<td>{html.escape(_display(candidate.total_cost))}</td>"
        f"<td>{str(candidate.cost_complete).lower()}</td>"
        f"<td>{html.escape(','.join(candidate.cost_unknown_reasons))}</td>"
        f"<td>{html.escape(candidate.currency or '')}</td>"
        f"<td>{html.escape(metric.metric_id)}</td>"
        f"<td>{html.escape(metric.unit)}</td>"
        f"<td>{html.escape(_display(metric.value))}</td>"
        f"<td>{html.escape(_display(metric.denominator))}</td>"
        f"<td>{html.escape(_display(metric.baseline_delta))}</td>"
        f"<td>{html.escape(metric.status.value)}</td>"
        f"<td>{html.escape(metric.gate_status.value)}</td>"
        "</tr>"
        for candidate in result.candidates
        for metric in candidate.metrics
    )
    document = (
        '<!doctype html><html><head><meta charset="utf-8"><title>Comparison report</title>'
        "</head><body>"
        f"<h1>{html.escape(result.plan.display_name)}</h1>"
        f'<p id="comparison-result-digest">{digest}</p>'
        '<section aria-label="Canonical comparison result">'
        f'<pre id="comparison-result-visible">{html.escape(canonical)}</pre>'
        "</section>"
        '<script id="comparison-result-json" type="application/json" '
        f'data-encoding="base64">{encoded}</script>'
        "<table><thead><tr><th>candidate</th><th>candidate status</th>"
        "<th>evidence status</th><th>known partial cost</th><th>total cost</th>"
        "<th>cost complete</th><th>cost unknown reasons</th><th>currency</th>"
        "<th>metric</th><th>unit</th><th>value</th>"
        "<th>denominator</th><th>baseline delta</th><th>status</th><th>gate</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
        "</body></html>\n"
    )
    return document.encode("utf-8")


def _render_text(result: ComparisonResult) -> bytes:
    canonical = _canonical_result_text(result)
    lines = (
        "comparison-report-text-v1",
        f"canonical-json-sha256={_result_digest(canonical)}",
        f"canonical-json={canonical}",
        "candidate-metrics:",
        *(row for candidate in result.candidates for row in _text_metric_rows(candidate)),
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _text_metric_rows(candidate: ComparisonCandidateResult) -> tuple[str, ...]:
    reference = candidate.reference
    return tuple(
        "|".join(
            (
                str(reference.variant_id),
                candidate.status.value,
                candidate.evidence_status.value,
                candidate.safe_error_code or "",
                str(reference.axis_value),
                _display(candidate.known_partial_cost),
                _display(candidate.total_cost),
                str(candidate.cost_complete).lower(),
                ",".join(candidate.cost_unknown_reasons),
                candidate.currency or "",
                metric.metric_id,
                metric.unit,
                _display(metric.value),
                _display(metric.numerator),
                _display(metric.denominator),
                _display(metric.baseline_delta),
                metric.status.value,
                metric.gate_status.value,
            )
        )
        for metric in candidate.metrics
    )


def _render_csv(result: ComparisonResult) -> bytes:
    canonical = _canonical_result_text(result)
    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(_CSV_HEADER)
    writer.writerow(
        (
            "comparison-result",
            _result_digest(canonical),
            canonical,
            *("",) * (len(_CSV_HEADER) - 3),
        )
    )
    for candidate in result.candidates:
        for metric in candidate.metrics:
            writer.writerow(
                (
                    "candidate-metric",
                    "",
                    "",
                    candidate.reference.variant_id,
                    candidate.status.value,
                    candidate.evidence_status.value,
                    candidate.safe_error_code or "",
                    candidate.reference.axis_value,
                    _display(candidate.known_partial_cost),
                    _display(candidate.total_cost),
                    str(candidate.cost_complete).lower(),
                    ",".join(candidate.cost_unknown_reasons),
                    candidate.currency or "",
                    metric.metric_id,
                    metric.unit,
                    _display(metric.value),
                    _display(metric.numerator),
                    _display(metric.denominator),
                    _display(metric.baseline_delta),
                    metric.status.value,
                    metric.gate_status.value,
                )
            )
    return stream.getvalue().encode("utf-8")


def _display(value: object) -> str:
    if isinstance(value, UnavailableValue):
        return f"unavailable:{value.reason}"
    return str(value)


def _parse_html_projection(content: bytes) -> ComparisonResult:
    try:
        text = content.decode("utf-8")
        embedded_match = _HTML_EMBEDDED.search(text)
        visible_match = _HTML_VISIBLE.search(text)
        digest_match = _HTML_DIGEST.search(text)
        if embedded_match is None or visible_match is None or digest_match is None:
            raise ValueError
        canonical = base64.b64decode(
            embedded_match.group(1),
            validate=True,
        ).decode("utf-8")
        if html.unescape(visible_match.group(1)) != canonical:
            raise ValueError
        result = ComparisonResult.model_validate(decode_json_report(canonical))
    except (UnicodeError, TypeError, ValueError):
        raise ComparisonArtifactError("comparison-artifact-html-invalid") from None
    if (
        canonical != _canonical_result_text(result)
        or digest_match.group(1) != _result_digest(canonical)
        or content != _render_html(result)
    ):
        raise ComparisonArtifactError("comparison-artifact-html-parity-failed")
    return result


def _parse_text_projection(content: bytes) -> ComparisonResult:
    try:
        text = content.decode("utf-8")
        lines = text.splitlines()
        if (
            len(lines) < 4
            or lines[0] != "comparison-report-text-v1"
            or not lines[1].startswith("canonical-json-sha256=")
            or not lines[2].startswith("canonical-json=")
        ):
            raise ValueError
        digest = lines[1].removeprefix("canonical-json-sha256=")
        canonical = lines[2].removeprefix("canonical-json=")
        result = ComparisonResult.model_validate(decode_json_report(canonical))
    except (UnicodeError, TypeError, ValueError):
        raise ComparisonArtifactError("comparison-artifact-text-invalid") from None
    if (
        canonical != _canonical_result_text(result)
        or digest != _result_digest(canonical)
        or content != _render_text(result)
    ):
        raise ComparisonArtifactError("comparison-artifact-text-parity-failed")
    return result


def _parse_csv_projection(content: bytes) -> ComparisonResult:
    try:
        text = content.decode("utf-8")
        rows = tuple(csv.reader(StringIO(text, newline="")))
        if (
            len(rows) < 2
            or tuple(rows[0]) != _CSV_HEADER
            or len(rows[1]) != len(_CSV_HEADER)
            or rows[1][0] != "comparison-result"
        ):
            raise ValueError
        digest = rows[1][1]
        canonical = rows[1][2]
        result = ComparisonResult.model_validate(decode_json_report(canonical))
    except (csv.Error, UnicodeError, TypeError, ValueError):
        raise ComparisonArtifactError("comparison-artifact-csv-invalid") from None
    if (
        canonical != _canonical_result_text(result)
        or digest != _result_digest(canonical)
        or content != _render_csv(result)
    ):
        raise ComparisonArtifactError("comparison-artifact-csv-parity-failed")
    return result


def _publication_descriptors(
    plan: ExperimentPlan,
    result: ComparisonResult,
    reports: Sequence[VerifiedCandidateReport],
    payloads: Mapping[str, bytes],
    created_at: datetime,
) -> tuple[ArtifactDescriptor, ...]:
    descriptors = tuple(
        _descriptor(artifact_id, payloads[artifact_id], created_at)
        for artifact_id in _CORE_CONTRACTS
    )
    candidate_descriptors = tuple(report.descriptor for report in reports)
    expected_candidate_ids = tuple(
        item.source_descriptor.artifact_id
        for item in result.candidates
        if item.source_descriptor is not None
    )
    if (
        tuple(item.artifact_id for item in candidate_descriptors) != expected_candidate_ids
        or any(item.created_at > created_at for item in candidate_descriptors)
        or tuple(item.variant_id for item in plan.variants)
        != tuple(item.reference.variant_id for item in result.candidates)
    ):
        raise ComparisonArtifactError("comparison-publication-descriptor-invalid")
    return (*descriptors, *candidate_descriptors)


def _descriptor(artifact_id: str, content: bytes, created_at: datetime) -> ArtifactDescriptor:
    try:
        schema_version, media_type, relative_path = _CORE_CONTRACTS[artifact_id]
    except KeyError:
        raise ComparisonArtifactError("comparison-artifact-id-invalid") from None
    return ArtifactDescriptor(
        schema_version=schema_version,
        artifact_id=artifact_id,
        format=relative_path.rsplit(".", 1)[-1],
        media_type=media_type,
        relative_path=relative_path,
        sha256_digest=f"sha256:{hashlib.sha256(content).hexdigest()}",
        byte_size=len(content),
        created_at=created_at,
    )


def _validate_payloads(
    payloads: Mapping[str, bytes],
    redactor: Redactor,
    *,
    privacy_values: Iterable[object],
) -> None:
    if any(
        not content or len(content) > MAX_COMPARISON_ARTIFACT_BYTES for content in payloads.values()
    ):
        raise ComparisonArtifactError("comparison-artifact-size-invalid")
    if sum(map(len, payloads.values())) > MAX_COMPARISON_BUNDLE_BYTES:
        raise ComparisonArtifactError("comparison-artifact-bundle-too-large")
    _require_path_safe_payloads(payloads.values())
    _require_typed_privacy_safe(privacy_values, redactor)
    _require_typed_privacy_safe(_STATIC_PROJECTION_TEXT, redactor)


def _require_path_safe_payloads(contents: Iterable[bytes]) -> None:
    try:
        for content in contents:
            text = content.decode("utf-8")
            if _RAW_FILESYSTEM_PATH.search(text) is not None:
                raise ComparisonArtifactError("comparison-artifact-privacy-failed")
    except UnicodeError:
        raise ComparisonArtifactError("comparison-artifact-encoding-invalid") from None


def _require_typed_privacy_safe(values: Iterable[object], redactor: Redactor) -> None:
    def visit(value: object) -> None:
        dump = getattr(value, "model_dump", None)
        if callable(dump):
            visit(dump(mode="python"))
            return
        if isinstance(value, str):
            if redactor.detect(value) or _RAW_FILESYSTEM_PATH.search(value) is not None:
                raise ComparisonArtifactError("comparison-artifact-privacy-failed")
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(key)
                visit(item)
            return
        if isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                visit(item)

    for value in values:
        visit(value)


def _require_projection_delta_privacy_safe(
    content: bytes,
    expected: bytes,
    redactor: Redactor,
) -> None:
    """Scan only noncanonical projection text, never typed numeric serialization."""

    try:
        actual_text = content.decode("utf-8")
        expected_text = expected.decode("utf-8")
    except UnicodeError:
        raise ComparisonArtifactError("comparison-artifact-encoding-invalid") from None
    if actual_text == expected_text:
        return
    prefix = 0
    maximum_prefix = min(len(actual_text), len(expected_text))
    while prefix < maximum_prefix and actual_text[prefix] == expected_text[prefix]:
        prefix += 1
    suffix = 0
    maximum_suffix = maximum_prefix - prefix
    while suffix < maximum_suffix and actual_text[-(suffix + 1)] == expected_text[-(suffix + 1)]:
        suffix += 1
    actual_end = len(actual_text) - suffix if suffix else len(actual_text)
    unexpected = actual_text[prefix:actual_end]
    if redactor.detect(unexpected) or _RAW_FILESYSTEM_PATH.search(unexpected) is not None:
        raise ComparisonArtifactError("comparison-artifact-privacy-failed")


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _is_link(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(is_junction is not None and is_junction())


def _reject_link_ancestors(path: Path) -> None:
    absolute = _absolute_path(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if _is_link(current):
            raise ComparisonArtifactError("comparison-artifact-path-unsafe")


def _safe_catalog_root(root: Path) -> Path:
    candidate = _absolute_path(Path(root))
    _reject_link_ancestors(candidate)
    if candidate.exists() and not candidate.is_dir():
        raise ComparisonArtifactError("comparison-artifact-root-unsafe")
    resolved = candidate.resolve()
    if _absolute_path(resolved) != candidate:
        raise ComparisonArtifactError("comparison-artifact-root-unsafe")
    return resolved


def _bundle_root(root: Path, comparison_id: str) -> Path:
    if not comparison_id or any(
        character not in "-_" and not character.isalnum() for character in comparison_id
    ):
        raise ComparisonArtifactError("comparison-artifact-id-invalid")
    candidate = root / comparison_id
    if _is_link(candidate):
        raise ComparisonArtifactError("comparison-artifact-path-unsafe")
    resolved = candidate.resolve()
    if (
        not resolved.is_relative_to(root)
        or resolved.parent != root
        or resolved.name != comparison_id
    ):
        raise ComparisonArtifactError("comparison-artifact-path-unsafe")
    return resolved


def _write_new(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(content)
    except FileExistsError:
        raise ComparisonArtifactError("comparison-artifact-bundle-exists") from None


def _atomic_publish_directory(temporary: Path, target: Path) -> None:
    """Rename once, with a bounded Windows scanner-sharing retry."""

    for delay in (0.0, 0.01, 0.05, 0.1, 0.2):
        if delay:
            time.sleep(delay)
        if target.exists() or _is_link(target):
            raise ComparisonArtifactError("comparison-artifact-bundle-exists")
        try:
            os.rename(temporary, target)
            return
        except PermissionError:
            continue
        except FileExistsError:
            raise ComparisonArtifactError("comparison-artifact-bundle-exists") from None
    raise ComparisonArtifactError("comparison-artifact-publication-failed")


def _remove_temporary(temporary: Path, root: Path) -> None:
    if (
        temporary.exists()
        and _absolute_path(temporary).parent == _absolute_path(root)
        and temporary.name.startswith(".")
    ):
        shutil.rmtree(temporary)


def _read_regular_file(
    path: Path,
    *,
    bundle: Path,
    maximum_bytes: int,
    expected_size: int | None = None,
) -> bytes:
    if _is_link(path):
        raise ComparisonArtifactError("comparison-artifact-path-unsafe")
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError:
        raise ComparisonArtifactError("comparison-artifact-unavailable") from None
    if (
        not stat.S_ISREG(before.st_mode)
        or not resolved.is_relative_to(bundle)
        or before.st_size <= 0
        or before.st_size > maximum_bytes
    ):
        raise ComparisonArtifactError("comparison-artifact-size-invalid")
    if expected_size is not None and before.st_size != expected_size:
        raise ComparisonArtifactError("comparison-artifact-integrity-failed")
    try:
        content = path.read_bytes()
        after = path.lstat()
    except OSError:
        raise ComparisonArtifactError("comparison-artifact-unavailable") from None
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if (
        identity_before != identity_after
        or _is_link(path)
        or path.resolve(strict=True) != resolved
        or len(content) != before.st_size
    ):
        raise ComparisonArtifactError("comparison-artifact-integrity-failed")
    return content


def _inventory_bundle(
    bundle: Path,
    expected_files: set[str],
) -> None:
    try:
        entries = tuple(bundle.rglob("*"))
    except OSError:
        raise ComparisonArtifactError("comparison-artifact-unavailable") from None
    if len(entries) > MAX_COMPARISON_BUNDLE_ENTRIES:
        raise ComparisonArtifactError("comparison-artifact-entry-set-invalid")
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    total_size = 0
    for entry in entries:
        if _is_link(entry):
            raise ComparisonArtifactError("comparison-artifact-path-unsafe")
        try:
            metadata = entry.lstat()
        except OSError:
            raise ComparisonArtifactError("comparison-artifact-unavailable") from None
        relative = entry.relative_to(bundle).as_posix()
        if stat.S_ISREG(metadata.st_mode):
            actual_files.add(relative)
            total_size += metadata.st_size
        elif stat.S_ISDIR(metadata.st_mode):
            actual_directories.add(relative)
        else:
            raise ComparisonArtifactError("comparison-artifact-entry-set-invalid")
    expected_directories = {
        parent.as_posix()
        for value in expected_files
        for parent in PurePosixPath(value).parents
        if parent.as_posix() != "."
    }
    if (
        actual_files != expected_files
        or actual_directories != expected_directories
        or total_size > MAX_COMPARISON_BUNDLE_BYTES
    ):
        raise ComparisonArtifactError("comparison-artifact-entry-set-invalid")


def _verify_bundle(
    bundle: Path,
    redactor: Redactor,
) -> tuple[ComparisonArtifactManifest, Mapping[str, bytes]]:
    if _is_link(bundle) or not bundle.is_dir():
        raise ComparisonArtifactError("comparison-artifact-path-unsafe")
    manifest_path = bundle / COMPARISON_MANIFEST_FILENAME
    raw_manifest = _read_regular_file(
        manifest_path,
        bundle=bundle,
        maximum_bytes=MAX_COMPARISON_MANIFEST_BYTES,
    )
    try:
        manifest = ComparisonArtifactManifest.model_validate(
            decode_json_report(raw_manifest.decode("utf-8"))
        )
    except (UnicodeError, TypeError, ValueError):
        raise ComparisonArtifactError("comparison-artifact-manifest-invalid") from None
    if (
        bundle.name != manifest.comparison_id
        or raw_manifest != canonical_comparison_manifest(manifest)
        or len(manifest.artifacts) + 1 > MAX_COMPARISON_BUNDLE_ENTRIES
        or any(
            item.byte_size <= 0 or item.byte_size > MAX_COMPARISON_ARTIFACT_BYTES
            for item in manifest.artifacts
        )
        or sum(item.byte_size for item in manifest.artifacts) > MAX_COMPARISON_BUNDLE_BYTES
    ):
        raise ComparisonArtifactError("comparison-artifact-manifest-invalid")
    expected_files = {
        COMPARISON_MANIFEST_FILENAME,
        *(item.relative_path for item in manifest.artifacts),
    }
    _inventory_bundle(bundle, expected_files)
    contents: dict[str, bytes] = {}
    for descriptor in manifest.artifacts:
        path = bundle.joinpath(*descriptor.relative_path.split("/"))
        content = _read_regular_file(
            path,
            bundle=bundle,
            maximum_bytes=MAX_COMPARISON_ARTIFACT_BYTES,
            expected_size=descriptor.byte_size,
        )
        if f"sha256:{hashlib.sha256(content).hexdigest()}" != descriptor.sha256_digest:
            raise ComparisonArtifactError("comparison-artifact-integrity-failed")
        contents[descriptor.artifact_id] = content
    _require_path_safe_payloads(tuple(contents.values()))
    _verify_semantic_contents(manifest, contents, redactor)
    return manifest, MappingProxyType(contents)


def _verify_semantic_contents(
    manifest: ComparisonArtifactManifest,
    contents: Mapping[str, bytes],
    redactor: Redactor,
) -> None:
    try:
        plan = ExperimentPlan.model_validate(
            decode_json_report(contents["comparison-plan-json"].decode("utf-8"))
        )
        result = ComparisonResult.model_validate(
            decode_json_report(contents["comparison-report-json"].decode("utf-8"))
        )
        projections = (
            (contents["comparison-report-html"], _render_html(result)),
            (contents["comparison-report-txt"], _render_text(result)),
            (contents["comparison-report-csv"], _render_csv(result)),
        )
    except (KeyError, UnicodeError, TypeError, ValueError):
        raise ComparisonArtifactError("comparison-artifact-content-invalid") from None
    _require_typed_privacy_safe((plan, result), redactor)
    _require_typed_privacy_safe(_STATIC_PROJECTION_TEXT, redactor)
    for content, expected in projections:
        _require_projection_delta_privacy_safe(content, expected, redactor)
    try:
        html_result = _parse_html_projection(contents["comparison-report-html"])
        text_result = _parse_text_projection(contents["comparison-report-txt"])
        csv_result = _parse_csv_projection(contents["comparison-report-csv"])
    except (KeyError, UnicodeError, TypeError, ValueError):
        raise ComparisonArtifactError("comparison-artifact-content-invalid") from None
    if (
        plan.plan_id != manifest.plan_id
        or plan.content_hash != manifest.plan_content_hash
        or tuple(item.variant_id for item in plan.variants) != manifest.candidate_variant_ids
        or result.comparison_id != manifest.comparison_id
        or result.plan != plan
        or result.completed_at > manifest.created_at
        or contents["comparison-plan-json"] != _canonical_model(plan)
        or contents["comparison-report-json"] != _canonical_model(result)
        or html_result != result
        or text_result != result
        or csv_result != result
    ):
        raise ComparisonArtifactError("comparison-artifact-parity-failed")
    descriptors = {item.artifact_id: item for item in manifest.artifacts}
    expected_candidate_ids = {
        candidate.source_descriptor.artifact_id
        for candidate in result.candidates
        if candidate.source_descriptor is not None
    }
    actual_candidate_ids = {
        artifact_id
        for artifact_id in descriptors
        if artifact_id.startswith("comparison-candidate-")
    }
    if expected_candidate_ids != actual_candidate_ids:
        raise ComparisonArtifactError("comparison-artifact-candidate-set-mismatch")
    for candidate in result.candidates:
        if candidate.source_descriptor is None or candidate.source_evidence is None:
            continue
        artifact_id = candidate.source_descriptor.artifact_id
        try:
            verified = load_verified_candidate_report(
                candidate.reference,
                descriptors[artifact_id],
                contents[artifact_id],
                comparison_id=result.comparison_id,
                plan=plan,
            )
        except (ComparisonDomainError, KeyError, TypeError, ValueError):
            raise ComparisonArtifactError("comparison-artifact-candidate-invalid") from None
        _require_typed_privacy_safe((verified.evidence,), redactor)
        if (
            verified.descriptor != candidate.source_descriptor
            or verified.evidence != candidate.source_evidence
        ):
            raise ComparisonArtifactError("comparison-artifact-candidate-mismatch")


__all__ = [
    "COMPARISON_MANIFEST_FILENAME",
    "MAX_COMPARISON_ARTIFACT_BYTES",
    "MAX_COMPARISON_BUNDLE_BYTES",
    "MAX_COMPARISON_MANIFEST_BYTES",
    "ComparisonArtifactCatalog",
    "ComparisonArtifactError",
    "PublishedComparisonArtifacts",
]
