"""Read-only, fail-closed access to sealed evaluation releases."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import cast

from rag_mvp.domain.evaluation import EvaluationRun, EvaluationRunStatus
from rag_mvp.evaluation.application import (
    EvaluationArtifactDescriptor,
    EvaluationArtifactManifest,
    ReleaseEvidenceSnapshot,
    ReleaseMetricEvidence,
    ReleasePerformanceEvidence,
    ResolvedEvaluationArtifact,
)
from rag_mvp.evaluation.json_report import decode_json_report
from rag_mvp.evaluation.report_dispatch import (
    ReportSchemaVersion,
    canonical_versioned_report_document,
    validate_versioned_report,
    verify_versioned_html_parity,
)
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, Redactor

_RELEASE_MANIFEST = "release-manifest.json"
_MAX_MANIFEST_BYTES = 1_000_000
_EXPOSED_REPORTS = {
    "evaluation_json": (
        "evaluation-report-json",
        "json",
        "application/json",
        "evaluation-report.json",
    ),
    "evaluation_html": (
        "evaluation-report-html",
        "html",
        "text/html",
        "evaluation-report.html",
    ),
}
_QUALITY_NAMES = {
    "faithfulness": "faithfulness",
    "context_precision": "context-precision",
    "answer_completeness": "answer-completeness",
    "style_consistency": "style",
    "refusal_appropriateness": "refusal-appropriateness",
}


class ReleaseEvidenceError(ValueError):
    """A sealed release is unavailable or fails an integrity boundary."""


@dataclass(frozen=True, slots=True)
class _VerifiedRelease:
    snapshot: ReleaseEvidenceSnapshot
    artifacts: Mapping[str, ResolvedEvaluationArtifact]


@dataclass(slots=True)
class VerifiedReleaseEvidenceStore:
    """Expose only fully verified accepted releases below one configured root."""

    releases_root: Path
    redactor: Redactor = DEFAULT_REDACTOR
    _records: dict[str, _VerifiedRelease] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.releases_root = Path(self.releases_root).expanduser().resolve()

    def list(self) -> tuple[ReleaseEvidenceSnapshot, ...]:
        records = self._load_records()
        return tuple(
            record.snapshot
            for record in sorted(
                records.values(),
                key=lambda item: item.snapshot.run.updated_at,
                reverse=True,
            )
        )

    def get(self, run_id: str) -> ReleaseEvidenceSnapshot | None:
        record = self._load_records().get(run_id)
        return None if record is None else record.snapshot

    def resolve(
        self,
        run_id: str,
        artifact_id: str,
    ) -> ResolvedEvaluationArtifact | None:
        record = self._load_records().get(run_id)
        if record is None:
            return None
        return record.artifacts.get(artifact_id)

    def _load_records(self) -> dict[str, _VerifiedRelease]:
        if self._records is not None:
            return self._records
        records: dict[str, _VerifiedRelease] = {}
        root = self.releases_root
        if root.is_symlink() or not root.is_dir():
            self._records = records
            return records
        for directory in root.iterdir():
            if not directory.is_dir() or directory.is_symlink():
                continue
            try:
                record = self._load_release(directory)
            except (OSError, UnicodeError, ValueError, TypeError):
                continue
            run_id = record.snapshot.run.run_id
            if run_id in records:
                continue
            records[run_id] = record
        self._records = records
        return records

    def _load_release(self, directory: Path) -> _VerifiedRelease:
        root = self.releases_root
        resolved_directory = directory.resolve(strict=True)
        if not resolved_directory.is_relative_to(root):
            raise ReleaseEvidenceError("release_directory_outside_root")
        manifest_path = resolved_directory / _RELEASE_MANIFEST
        manifest_bytes = _read_regular_file(manifest_path, root, _MAX_MANIFEST_BYTES)
        manifest = _json_object(manifest_bytes)
        if (
            _text(manifest, "schema_version") != "rag-mvp-release-manifest-v1"
            or _text(manifest, "status") != "accepted"
        ):
            raise ReleaseEvidenceError("release_manifest_not_accepted")

        generated_at = _timestamp(_text(manifest, "generated_at_utc"))
        artifacts_raw = _mapping(manifest, "artifacts")
        verified_content: dict[str, bytes] = {}
        for name, value in artifacts_raw.items():
            if not isinstance(name, str) or not isinstance(value, Mapping):
                raise ReleaseEvidenceError("release_artifact_descriptor_invalid")
            descriptor = cast(Mapping[str, object], value)
            relative_path = _text(descriptor, "path")
            artifact_path = _safe_artifact_path(resolved_directory, root, relative_path)
            content = _read_regular_file(artifact_path, root, 10_000_000)
            if len(content) != _integer(descriptor, "bytes"):
                raise ReleaseEvidenceError("release_artifact_size_mismatch")
            if f"sha256:{sha256(content).hexdigest()}" != _text(descriptor, "sha256"):
                raise ReleaseEvidenceError("release_artifact_digest_mismatch")
            verified_content[name] = content

        report_content = verified_content.get("evaluation_json")
        html_content = verified_content.get("evaluation_html")
        if report_content is None or html_content is None:
            raise ReleaseEvidenceError("release_report_missing")
        report = _validated_legacy_report(report_content)
        try:
            html = html_content.decode("utf-8")
            verify_versioned_html_parity(report, html, redactor=self.redactor)
        except (UnicodeError, TypeError, ValueError):
            raise ReleaseEvidenceError("release_report_parity_failed") from None

        snapshot = _release_snapshot(
            manifest,
            report,
            generated_at=generated_at,
            manifest_digest=f"sha256:{sha256(manifest_bytes).hexdigest()}",
            artifact_content=verified_content,
        )
        downloads: dict[str, ResolvedEvaluationArtifact] = {}
        for manifest_name, (artifact_id, _format, media_type, filename) in _EXPOSED_REPORTS.items():
            downloads[artifact_id] = ResolvedEvaluationArtifact(
                artifact_id=artifact_id,
                content=verified_content[manifest_name],
                media_type=media_type,
                filename=filename,
            )
        return _VerifiedRelease(snapshot=snapshot, artifacts=downloads)


def _release_snapshot(
    manifest: Mapping[str, object],
    report: Mapping[str, object],
    *,
    generated_at: datetime,
    manifest_digest: str,
    artifact_content: Mapping[str, bytes],
) -> ReleaseEvidenceSnapshot:
    release_id = _text(manifest, "release_id")
    configuration = _mapping(manifest, "configuration")
    dataset_and_corpus = _mapping(manifest, "dataset_and_corpus")
    dataset = _mapping(dataset_and_corpus, "dataset")
    corpus = _mapping(dataset_and_corpus, "corpus")
    runtime = _mapping(manifest, "accepted_runtime_source")
    quality = _mapping(manifest, "quality")
    performance = _mapping(manifest, "performance")
    security = _mapping(manifest, "security")
    privacy = _mapping(manifest, "privacy")
    report_configuration = _mapping(report, "configuration")
    report_provenance = _mapping(report, "provenance")
    scorer_versions_value = report_provenance.get("scorer_versions", {})
    scorer_versions = (
        cast(Mapping[str, object], scorer_versions_value)
        if isinstance(scorer_versions_value, Mapping)
        else {}
    )
    case_results = _sequence(report, "case_results")
    run_id = _text(report, "run_id")
    configuration_id = _text(configuration, "configuration_id")
    if _text(report_configuration, "configuration_id") != configuration_id:
        raise ReleaseEvidenceError("release_configuration_mismatch")
    if _text(quality, "run_id") != run_id:
        raise ReleaseEvidenceError("release_run_identity_mismatch")
    gate = _mapping(report, "gate")
    gate_passed = _boolean(gate, "final_passed")
    if not (
        gate_passed
        and _boolean(gate, "valid")
        and _boolean(quality, "final_gate_passed")
        and _boolean(performance, "decision_valid")
        and _boolean(performance, "decision_passed")
    ):
        raise ReleaseEvidenceError("release_acceptance_gate_failed")

    report_metrics = _mapping(_mapping(report, "metrics"), "aggregate")
    metrics: list[ReleaseMetricEvidence] = []
    for legacy_name, metric_id in _QUALITY_NAMES.items():
        aggregate = _mapping(report_metrics, legacy_name)
        scorer = scorer_versions.get(metric_id, scorer_versions.get(legacy_name))
        metrics.append(
            ReleaseMetricEvidence(
                metric_id=metric_id,
                value=_optional_float(aggregate, "value"),
                threshold=_optional_float(aggregate, "threshold"),
                operator=_optional_text(aggregate, "operator"),
                denominator=_integer(aggregate, "eligible_cases"),
                passed=_boolean(aggregate, "passed"),
                scorer_version=str(scorer) if isinstance(scorer, str) else None,
            )
        )

    latency = _mapping(performance, "latency_ms")
    tokens = _mapping(performance, "tokens")
    attempts = _integer(performance, "attempts")
    successes = _integer(performance, "successes")
    errors = _integer(performance, "errors")
    if successes + errors != attempts:
        raise ReleaseEvidenceError("release_performance_denominator_mismatch")
    total_cost = _decimal(performance, "estimated_cost_usd")
    cost_per_attempts = total_cost * Decimal(1000) / Decimal(attempts)
    report_cost = _mapping(report, "cost")
    outcomes = [
        _optional_text(cast(Mapping[str, object], item), "outcome")
        for item in case_results
        if isinstance(item, Mapping)
    ]
    answered = sum(item == "answer" for item in outcomes)
    refusals = sum(item == "refusal" for item in outcomes)
    security_passed = (
        _boolean(security, "secret_gate_passed")
        and _boolean(security, "critical_policy_gate_passed")
        and _boolean(privacy, "passed")
    )
    release_performance = ReleasePerformanceEvidence(
        attempts=attempts,
        successes=successes,
        errors=errors,
        configured_concurrency=_integer(performance, "configured_concurrency"),
        observed_peak_concurrency=_integer(performance, "observed_peak_concurrency"),
        p50_ms=_number(latency, "p50"),
        p90_ms=_number(latency, "p90"),
        p95_ms=_number(latency, "p95"),
        p99_ms=_number(latency, "p99"),
        provider_attempt_count=_integer(performance, "provider_attempt_count"),
        input_tokens=_integer(tokens, "embedding_input") + _integer(tokens, "generation_input"),
        output_tokens=_integer(tokens, "generation_output"),
        total_cost=total_cost,
        cost_per_1000_attempts=cost_per_attempts,
        cost_per_1000_successes=_decimal(performance, "cost_per_1000_successful_calls_usd"),
        currency=_text(report_cost, "currency"),
        refusals=refusals,
        answered_requests=answered,
        security_passed=security_passed,
    )
    artifacts_raw = _mapping(manifest, "artifacts")
    descriptors: list[EvaluationArtifactDescriptor] = []
    for manifest_name, (
        artifact_id,
        artifact_format,
        media_type,
        _filename,
    ) in _EXPOSED_REPORTS.items():
        source = _mapping(artifacts_raw, manifest_name)
        content = artifact_content[manifest_name]
        descriptors.append(
            EvaluationArtifactDescriptor(
                artifact_id=artifact_id,
                schema_version="1.0.0",
                format=artifact_format,
                media_type=media_type,
                sha256_digest=f"sha256:{sha256(content).hexdigest()}",
                byte_size=len(content),
                created_at=generated_at,
            )
        )
        if _text(source, "path") != _filename:
            raise ReleaseEvidenceError("release_report_filename_mismatch")
    artifact_manifest = EvaluationArtifactManifest(
        run_id=run_id,
        configuration_id=configuration_id,
        manifest_content_hash=manifest_digest,
        artifacts=tuple(descriptors),
    )
    run = EvaluationRun(
        run_id=run_id,
        status=EvaluationRunStatus.COMPLETED,
        dataset_id=_text(dataset, "id"),
        dataset_version=_text(dataset, "version"),
        dataset_hash=_text(dataset, "content_hash"),
        corpus_version=_text(corpus, "version"),
        configuration_id=configuration_id,
        code_revision=_text(runtime, "git_revision"),
        scorer_versions={
            item.metric_id: item.scorer_version or "not-recorded-in-v1-release" for item in metrics
        },
        cache_policy=_text(performance, "cache_policy"),
        total_cases=len(case_results),
        completed_cases=len(case_results),
        failed_cases=0,
        created_at=generated_at,
        updated_at=generated_at,
    )
    return ReleaseEvidenceSnapshot(
        release_id=release_id,
        source_schema_version=_text(manifest, "schema_version"),
        run=run,
        corpus_hash=_text(corpus, "content_hash"),
        gate_passed=gate_passed,
        quality_metrics=tuple(metrics),
        performance=release_performance,
        artifact_manifest=artifact_manifest,
    )


def _validated_legacy_report(content: bytes) -> dict[str, object]:
    try:
        decoded = decode_json_report(content.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ReleaseEvidenceError("release_report_invalid")
        loaded = validate_versioned_report(cast(Mapping[str, object], decoded))
    except (UnicodeError, TypeError, ValueError):
        raise ReleaseEvidenceError("release_report_invalid") from None
    if loaded.schema_version is not ReportSchemaVersion.V1:
        raise ReleaseEvidenceError("release_report_schema_mismatch")
    if canonical_versioned_report_document(loaded.document) != content:
        raise ReleaseEvidenceError("release_report_not_canonical")
    return cast(dict[str, object], loaded.document)


def _safe_artifact_path(directory: Path, root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        "\\" in relative
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ReleaseEvidenceError("release_artifact_path_invalid")
    expected = directory.joinpath(*pure.parts)
    if expected.is_symlink():
        raise ReleaseEvidenceError("release_artifact_symlink_rejected")
    resolved = expected.resolve(strict=True)
    if not resolved.is_relative_to(directory) or not resolved.is_relative_to(root):
        raise ReleaseEvidenceError("release_artifact_outside_root")
    return resolved


def _read_regular_file(path: Path, root: Path, maximum_bytes: int) -> bytes:
    if path.is_symlink():
        raise ReleaseEvidenceError("release_symlink_rejected")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ReleaseEvidenceError("release_file_outside_root")
    size = resolved.stat().st_size
    if not 0 < size <= maximum_bytes:
        raise ReleaseEvidenceError("release_file_size_invalid")
    return resolved.read_bytes()


def _json_object(content: bytes) -> dict[str, object]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise ReleaseEvidenceError("release_manifest_invalid") from None
    if not isinstance(value, dict):
        raise ReleaseEvidenceError("release_manifest_invalid")
    return cast(dict[str, object], value)


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ReleaseEvidenceError("release_field_invalid")
    return cast(Mapping[str, object], item)


def _sequence(value: Mapping[str, object], key: str) -> tuple[object, ...]:
    item = value.get(key)
    if not isinstance(item, list):
        raise ReleaseEvidenceError("release_field_invalid")
    return tuple(item)


def _text(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ReleaseEvidenceError("release_field_invalid")
    return item


def _optional_text(value: Mapping[str, object], key: str) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str) or not item:
        raise ReleaseEvidenceError("release_field_invalid")
    return item


def _boolean(value: Mapping[str, object], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise ReleaseEvidenceError("release_field_invalid")
    return item


def _integer(value: Mapping[str, object], key: str) -> int:
    item = value.get(key)
    if type(item) is not int or item < 0:
        raise ReleaseEvidenceError("release_field_invalid")
    return item


def _number(value: Mapping[str, object], key: str) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int | float) or item < 0:
        raise ReleaseEvidenceError("release_field_invalid")
    return float(item)


def _optional_float(value: Mapping[str, object], key: str) -> float | None:
    item = value.get(key)
    if item is None:
        return None
    if isinstance(item, bool) or not isinstance(item, int | float):
        raise ReleaseEvidenceError("release_field_invalid")
    return float(item)


def _decimal(value: Mapping[str, object], key: str) -> Decimal:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, str | int | float):
        raise ReleaseEvidenceError("release_field_invalid")
    try:
        result = Decimal(str(item))
    except InvalidOperation:
        raise ReleaseEvidenceError("release_field_invalid") from None
    if not result.is_finite() or result < 0:
        raise ReleaseEvidenceError("release_field_invalid")
    return result


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ReleaseEvidenceError("release_timestamp_invalid") from None
    if parsed.tzinfo is None:
        raise ReleaseEvidenceError("release_timestamp_invalid")
    return parsed


__all__ = ["ReleaseEvidenceError", "VerifiedReleaseEvidenceStore"]
