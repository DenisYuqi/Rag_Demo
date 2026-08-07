"""Immutable schema-v2 multi-format evaluation artifact catalog."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Annotated, Literal, cast

from filelock import FileLock
from pydantic import AwareDatetime, Field, model_validator

from rag_mvp.domain import (
    ArtifactDescriptor,
    MetricObservation,
    MetricObservationStatus,
    OperationsSummary,
    UnavailableValue,
)
from rag_mvp.domain._base import DomainModel, Identifier, utc_now
from rag_mvp.observability.costs_v2 import (
    EvidenceAvailability,
    NormalizedCostV2,
    TokenDirection,
)
from rag_mvp.observability.log_contract_v1 import (
    LOG_DICTIONARY_SCHEMA_VERSION,
    LOG_EVENT_SCHEMA_VERSION,
    canonical_log_dictionary_json,
    parse_log_dictionary_json,
    parse_log_sample_jsonl,
    render_log_sample_jsonl,
    validate_log_documentation,
)
from rag_mvp.performance.evidence_v2 import PerformanceEvidenceV2
from rag_mvp.safety.scan_artifacts import scan_artifacts

from .html_report_v2 import render_html_report_v2, verify_html_parity_v2
from .json_report import MAX_REPORT_BYTES, canonical_json_value, decode_json_report
from .operations_v2 import (
    OperationsMetricId,
    parse_operations_csv,
    parse_operations_text,
    validate_operations_summary_v2,
    verify_operations_parity,
)
from .report_dispatch import (
    ReportSchemaVersion,
    canonical_versioned_report_document,
    validate_versioned_report,
)

ARTIFACT_MANIFEST_SCHEMA_VERSION: Literal["evaluation-artifact-manifest-v2"] = (
    "evaluation-artifact-manifest-v2"
)
ARTIFACT_MANIFEST_FILENAME = "artifact-manifest.json"
MAX_PUBLISHED_ARTIFACT_BYTES = MAX_REPORT_BYTES * 2
MAX_PUBLISHED_BUNDLE_BYTES = MAX_PUBLISHED_ARTIFACT_BYTES * 6
_SAFE_OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
type Sha256Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class ArtifactFormatV2(StrEnum):
    JSON = "json"
    HTML = "html"
    TXT = "txt"
    CSV = "csv"
    JSONL = "jsonl"


@dataclass(frozen=True, slots=True)
class _ArtifactContract:
    artifact_id: str
    format: ArtifactFormatV2
    filename: str
    media_type: str


_ARTIFACT_CONTRACT_ORDER = (
    _ArtifactContract(
        "evaluation-report-json",
        ArtifactFormatV2.JSON,
        "evaluation-report.json",
        "application/json",
    ),
    _ArtifactContract(
        "evaluation-report-html",
        ArtifactFormatV2.HTML,
        "evaluation-report.html",
        "text/html",
    ),
    _ArtifactContract(
        "operations-summary-txt",
        ArtifactFormatV2.TXT,
        "operations-summary.txt",
        "text/plain",
    ),
    _ArtifactContract(
        "operations-summary-csv",
        ArtifactFormatV2.CSV,
        "operations-summary.csv",
        "text/csv",
    ),
    _ArtifactContract(
        "structured-log-field-dictionary",
        ArtifactFormatV2.JSON,
        "structured-log-field-dictionary-v1.json",
        "application/json",
    ),
    _ArtifactContract(
        "privacy-safe-log-sample",
        ArtifactFormatV2.JSONL,
        "privacy-safe-sample-v1.jsonl",
        "application/x-ndjson",
    ),
)
_ARTIFACT_CONTRACTS = {contract.artifact_id: contract for contract in _ARTIFACT_CONTRACT_ORDER}
_CORE_ARTIFACT_IDS = frozenset(contract.artifact_id for contract in _ARTIFACT_CONTRACT_ORDER[:4])
_DOCUMENTATION_ARTIFACT_IDS = frozenset(
    contract.artifact_id for contract in _ARTIFACT_CONTRACT_ORDER[4:]
)
ARTIFACT_MEDIA_TYPES_V2: Mapping[ArtifactFormatV2, str] = MappingProxyType(
    {
        ArtifactFormatV2.JSON: "application/json",
        ArtifactFormatV2.HTML: "text/html",
        ArtifactFormatV2.TXT: "text/plain",
        ArtifactFormatV2.CSV: "text/csv",
        ArtifactFormatV2.JSONL: "application/x-ndjson",
    }
)
ARTIFACT_DOWNLOAD_FILENAMES_V2: Mapping[str, str] = MappingProxyType(
    {contract.artifact_id: contract.filename for contract in _ARTIFACT_CONTRACT_ORDER}
)
ARTIFACT_MEDIA_TYPES_BY_ID_V2: Mapping[str, str] = MappingProxyType(
    {contract.artifact_id: contract.media_type for contract in _ARTIFACT_CONTRACT_ORDER}
)


def allowed_artifact_media_types_v2() -> frozenset[str]:
    """Return the complete immutable media-type allowlist for v2 artifacts."""

    return frozenset(ARTIFACT_MEDIA_TYPES_V2.values())


@dataclass(frozen=True, slots=True)
class ArtifactPayloadV2:
    """One validated publication input; callers never choose its filename."""

    artifact_id: str
    schema_version: str
    format: ArtifactFormatV2
    content: bytes

    def __post_init__(self) -> None:
        contract = _ARTIFACT_CONTRACTS.get(self.artifact_id)
        if contract is None or self.format is not contract.format:
            raise ValueError("artifact identifier does not match its registered format")
        if not _safe_identifier(self.schema_version):
            raise ValueError("artifact schema version must be an opaque identifier")
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("artifact content must be non-empty bytes")
        if len(self.content) > MAX_PUBLISHED_ARTIFACT_BYTES:
            raise ValueError("artifact content exceeds the publication bound")


class ArtifactManifestV2(DomainModel):
    """Safe searchable metadata for one immutable published artifact bundle."""

    schema_version: Literal["evaluation-artifact-manifest-v2"] = ARTIFACT_MANIFEST_SCHEMA_VERSION
    run_id: Identifier
    configuration_id: Identifier
    artifacts: Annotated[tuple[ArtifactDescriptor, ...], Field(min_length=4, max_length=6)]
    created_at: AwareDatetime = Field(default_factory=utc_now)
    manifest_content_hash: Sha256Digest

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        configuration_id: str,
        artifacts: Sequence[ArtifactDescriptor],
        created_at: datetime,
    ) -> ArtifactManifestV2:
        values = tuple(artifacts)
        provisional = cls.model_construct(
            run_id=run_id,
            configuration_id=configuration_id,
            artifacts=values,
            created_at=created_at,
            manifest_content_hash="sha256:" + "0" * 64,
        )
        digest = _manifest_content_hash(provisional)
        return cls(
            run_id=run_id,
            configuration_id=configuration_id,
            artifacts=values,
            created_at=created_at,
            manifest_content_hash=digest,
        )

    @model_validator(mode="after")
    def validate_manifest(self) -> ArtifactManifestV2:
        if not _safe_identifier(self.run_id) or not _safe_identifier(self.configuration_id):
            raise ValueError("artifact manifest identities must be opaque identifiers")
        by_id: dict[str, ArtifactDescriptor] = {}
        for descriptor in self.artifacts:
            try:
                artifact_format = ArtifactFormatV2(descriptor.format)
            except ValueError as error:
                raise ValueError("artifact manifest contains an unsupported format") from error
            contract = _ARTIFACT_CONTRACTS.get(descriptor.artifact_id)
            if (
                contract is None
                or descriptor.artifact_id in by_id
                or artifact_format is not contract.format
                or descriptor.relative_path != contract.filename
                or descriptor.media_type != contract.media_type
                or descriptor.created_at != self.created_at
            ):
                raise ValueError("artifact descriptor differs from its registered contract")
            by_id[descriptor.artifact_id] = descriptor
        artifact_ids = frozenset(by_id)
        if artifact_ids not in {
            _CORE_ARTIFACT_IDS,
            _CORE_ARTIFACT_IDS | _DOCUMENTATION_ARTIFACT_IDS,
        }:
            raise ValueError("artifact manifest requires core formats and an optional log pair")
        expected_order = tuple(
            contract.artifact_id
            for contract in _ARTIFACT_CONTRACT_ORDER
            if contract.artifact_id in artifact_ids
        )
        if tuple(item.artifact_id for item in self.artifacts) != expected_order:
            raise ValueError("artifact descriptors must use canonical artifact order")
        if self.manifest_content_hash != _manifest_content_hash(self):
            raise ValueError("artifact manifest content hash mismatch")
        return self


@dataclass(frozen=True, slots=True)
class PublishedArtifactBundleV2:
    """Internal publication result; only the manifest should cross API boundaries."""

    bundle_root: Path
    manifest: ArtifactManifestV2


@dataclass(frozen=True, slots=True)
class ResolvedArtifactV2:
    """Verified download material with no backing filesystem path exposure."""

    descriptor: ArtifactDescriptor
    content: bytes


class ArtifactPublicationError(RuntimeError):
    """Base class for safe schema-v2 publication failures."""


class ArtifactBundleExistsError(ArtifactPublicationError):
    """Raised instead of replacing any prior run bundle."""


class ArtifactIntegrityError(ArtifactPublicationError):
    """Raised when manifest, parity, privacy, or content integrity fails."""


class ArtifactNotFoundError(ArtifactPublicationError):
    """Raised with a stable code when a manifest or artifact is absent."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def artifact_download_filename_v2(artifact_id: str) -> str:
    """Resolve a registered artifact ID to its server-owned download filename."""

    try:
        return ARTIFACT_DOWNLOAD_FILENAMES_V2[artifact_id]
    except (KeyError, TypeError):
        raise ArtifactNotFoundError("artifact-not-found") from None


def artifact_media_type_v2(artifact_id: str) -> str:
    """Resolve a registered artifact ID to its exact authoritative media type."""

    try:
        return ARTIFACT_MEDIA_TYPES_BY_ID_V2[artifact_id]
    except (KeyError, TypeError):
        raise ArtifactNotFoundError("artifact-not-found") from None


class ArtifactCatalogV2:
    """File-backed immutable catalog rooted outside runner plan/case storage."""

    def __init__(
        self,
        root: Path | str,
        *,
        privacy_fixture_path: Path | str | None = None,
    ) -> None:
        self._root = _safe_catalog_root(root)
        self._privacy_fixture_path = _privacy_fixture(privacy_fixture_path)

    @property
    def root(self) -> Path:
        """Internal storage root; API adapters must not serialize this property."""

        return self._root

    def publish(
        self,
        *,
        run_id: str,
        configuration_id: str,
        payloads: Sequence[ArtifactPayloadV2],
        created_at: datetime | None = None,
    ) -> PublishedArtifactBundleV2:
        return publish_artifact_bundle(
            self._root,
            run_id=run_id,
            configuration_id=configuration_id,
            payloads=payloads,
            created_at=created_at,
            privacy_fixture_path=self._privacy_fixture_path,
        )

    def get(self, run_id: str) -> ArtifactManifestV2 | None:
        bundle = _bundle_root(self._root, run_id)
        if not bundle.exists():
            return None
        return verify_artifact_manifest(
            bundle,
            privacy_fixture_path=self._privacy_fixture_path,
        )

    def require(self, run_id: str) -> ArtifactManifestV2:
        manifest = self.get(run_id)
        if manifest is None:
            raise ArtifactNotFoundError("artifact-manifest-not-found")
        return manifest

    def list(self) -> tuple[ArtifactManifestV2, ...]:
        if not self._root.exists():
            return ()
        manifests: list[ArtifactManifestV2] = []
        for path in sorted(self._root.iterdir(), key=lambda item: item.name):
            if path.name.startswith("."):
                continue
            if path.is_symlink() or not path.is_dir() or not _safe_identifier(path.name):
                raise ArtifactIntegrityError("artifact catalog contains an unsafe entry")
            manifests.append(
                verify_artifact_manifest(
                    path,
                    privacy_fixture_path=self._privacy_fixture_path,
                )
            )
        return tuple(sorted(manifests, key=lambda item: (item.created_at, item.run_id)))

    def resolve(self, run_id: str, artifact_id: str) -> ResolvedArtifactV2:
        bundle = _bundle_root(self._root, run_id)
        return resolve_verified_artifact(
            bundle,
            artifact_id,
            privacy_fixture_path=self._privacy_fixture_path,
        )


def publish_artifact_bundle(
    artifacts_root: Path | str,
    *,
    run_id: str,
    configuration_id: str,
    payloads: Sequence[ArtifactPayloadV2],
    created_at: datetime | None = None,
    privacy_fixture_path: Path | str | None = None,
) -> PublishedArtifactBundleV2:
    """Validate, stage, privacy-scan, and atomically publish one v2 bundle."""

    root = _safe_catalog_root(artifacts_root)
    if not _safe_identifier(run_id) or not _safe_identifier(configuration_id):
        raise ArtifactPublicationError("artifact publication identity is invalid")
    normalized = _canonical_payloads(payloads)
    _validate_cross_format_payloads(normalized, run_id, configuration_id)
    fixture_path = _privacy_fixture(privacy_fixture_path)
    timestamp = created_at or utc_now()
    target = _bundle_root(root, run_id)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = FileLock(str(root / ".publication.lock"))
    with lock:
        if target.exists() or target.is_symlink():
            raise ArtifactBundleExistsError("immutable artifact bundle already exists")
        temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=root))
        try:
            descriptors: list[ArtifactDescriptor] = []
            for payload in normalized:
                contract = _ARTIFACT_CONTRACTS[payload.artifact_id]
                artifact_path = temporary / contract.filename
                _write_new_file(artifact_path, payload.content)
                descriptors.append(
                    ArtifactDescriptor(
                        schema_version=payload.schema_version,
                        artifact_id=payload.artifact_id,
                        format=payload.format,
                        media_type=contract.media_type,
                        relative_path=contract.filename,
                        sha256_digest=_content_digest(payload.content),
                        byte_size=len(payload.content),
                        created_at=timestamp,
                    )
                )
            manifest = ArtifactManifestV2.create(
                run_id=run_id,
                configuration_id=configuration_id,
                artifacts=descriptors,
                created_at=timestamp,
            )
            _write_new_file(
                temporary / ARTIFACT_MANIFEST_FILENAME,
                canonical_artifact_manifest_document(manifest),
            )
            _require_privacy_scan(temporary, fixture_path)
            if target.exists() or target.is_symlink():
                raise ArtifactBundleExistsError("immutable artifact bundle already exists")
            try:
                os.rename(temporary, target)
            except FileExistsError as error:
                raise ArtifactBundleExistsError(
                    "immutable artifact bundle already exists"
                ) from error
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    verified = verify_artifact_manifest(target, privacy_fixture_path=fixture_path)
    return PublishedArtifactBundleV2(bundle_root=target, manifest=verified)


def canonical_artifact_manifest_document(manifest: ArtifactManifestV2) -> bytes:
    """Return deterministic UTF-8 manifest bytes including a final newline."""

    validated = ArtifactManifestV2.model_validate(manifest)
    return (canonical_json_value(validated.model_dump(mode="json")) + "\n").encode("utf-8")


def load_artifact_manifest(bundle_root: Path | str) -> ArtifactManifestV2:
    """Strictly load one canonical manifest without resolving arbitrary paths."""

    root = _safe_existing_bundle_root(bundle_root)
    path = root / ARTIFACT_MANIFEST_FILENAME
    if path.is_symlink():
        raise ArtifactIntegrityError("artifact manifest must not be a symbolic link")
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise ArtifactNotFoundError("artifact-manifest-not-found") from None
    except OSError as error:
        raise ArtifactIntegrityError("artifact manifest is unreadable") from error
    if not raw or len(raw) > MAX_REPORT_BYTES:
        raise ArtifactIntegrityError("artifact manifest size is outside allowed bounds")
    try:
        decoded = decode_json_report(raw.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError
        manifest = ArtifactManifestV2.model_validate(decoded)
    except (UnicodeError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError("artifact manifest is invalid") from error
    if raw != canonical_artifact_manifest_document(manifest):
        raise ArtifactIntegrityError("artifact manifest is not canonical JSON")
    return manifest


def verify_artifact_manifest(
    bundle_root: Path | str,
    manifest: ArtifactManifestV2 | None = None,
    *,
    privacy_fixture_path: Path | str | None = None,
) -> ArtifactManifestV2:
    """Verify path, media, size, digest, cross-format parity, and privacy."""

    active, _ = _verify_artifact_bundle(
        bundle_root,
        manifest,
        privacy_fixture_path=privacy_fixture_path,
    )
    return active


def _verify_artifact_bundle(
    bundle_root: Path | str,
    manifest: ArtifactManifestV2 | None = None,
    *,
    privacy_fixture_path: Path | str | None = None,
) -> tuple[ArtifactManifestV2, Mapping[str, bytes]]:
    """Verify a bundle and retain the exact bytes whose digests were checked."""

    root = _safe_existing_bundle_root(bundle_root)
    persisted = load_artifact_manifest(root)
    if manifest is not None:
        supplied = ArtifactManifestV2.model_validate(manifest)
        if supplied != persisted:
            raise ArtifactIntegrityError("supplied manifest differs from published manifest")
    active = persisted
    if root.name != active.run_id:
        raise ArtifactIntegrityError("artifact bundle path and run identity differ")
    expected_names = {ARTIFACT_MANIFEST_FILENAME}
    payloads: list[ArtifactPayloadV2] = []
    verified_contents: dict[str, bytes] = {}
    total_size = 0
    for descriptor in active.artifacts:
        artifact_format = ArtifactFormatV2(descriptor.format)
        contract = _ARTIFACT_CONTRACTS[descriptor.artifact_id]
        path = _resolve_descriptor_path(root, descriptor)
        expected_names.add(contract.filename)
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            raise ArtifactNotFoundError("artifact-content-not-found") from None
        except OSError as error:
            raise ArtifactIntegrityError("artifact content is unreadable") from error
        total_size += len(content)
        if (
            len(content) != descriptor.byte_size
            or _content_digest(content) != descriptor.sha256_digest
        ):
            raise ArtifactIntegrityError("artifact content digest or size mismatch")
        payloads.append(
            ArtifactPayloadV2(
                artifact_id=descriptor.artifact_id,
                schema_version=descriptor.schema_version,
                format=artifact_format,
                content=content,
            )
        )
        verified_contents[descriptor.artifact_id] = content
    if total_size > MAX_PUBLISHED_BUNDLE_BYTES:
        raise ArtifactIntegrityError("artifact bundle exceeds the publication bound")
    actual_names: set[str] = set()
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ArtifactIntegrityError("artifact bundle contains an unsafe entry")
        actual_names.add(path.name)
    if actual_names != expected_names:
        raise ArtifactIntegrityError("artifact bundle contains missing or unexpected files")
    normalized = _canonical_payloads(payloads)
    _validate_cross_format_payloads(normalized, active.run_id, active.configuration_id)
    _require_privacy_scan(root, _privacy_fixture(privacy_fixture_path))
    return active, MappingProxyType(verified_contents)


def resolve_verified_artifact(
    bundle_root: Path | str,
    artifact_id: str,
    *,
    privacy_fixture_path: Path | str | None = None,
) -> ResolvedArtifactV2:
    """Return verified bytes and descriptor, never a backing path."""

    if not _safe_identifier(artifact_id):
        raise ArtifactNotFoundError("artifact-not-found")
    root = _safe_existing_bundle_root(bundle_root)
    manifest, verified_contents = _verify_artifact_bundle(
        root,
        privacy_fixture_path=privacy_fixture_path,
    )
    descriptor = next(
        (item for item in manifest.artifacts if item.artifact_id == artifact_id),
        None,
    )
    if descriptor is None:
        raise ArtifactNotFoundError("artifact-not-found")
    return ResolvedArtifactV2(
        descriptor=descriptor,
        content=verified_contents[descriptor.artifact_id],
    )


def _canonical_payloads(payloads: Sequence[ArtifactPayloadV2]) -> tuple[ArtifactPayloadV2, ...]:
    values = tuple(payloads)
    if sum(len(item.content) for item in values) > MAX_PUBLISHED_BUNDLE_BYTES:
        raise ArtifactPublicationError("artifact bundle exceeds the publication bound")
    by_id = {item.artifact_id: item for item in values}
    artifact_ids = frozenset(by_id)
    if len(by_id) != len(values) or artifact_ids not in {
        _CORE_ARTIFACT_IDS,
        _CORE_ARTIFACT_IDS | _DOCUMENTATION_ARTIFACT_IDS,
    }:
        raise ArtifactPublicationError(
            "publication requires the four core artifacts and an optional log pair"
        )
    return tuple(
        by_id[contract.artifact_id]
        for contract in _ARTIFACT_CONTRACT_ORDER
        if contract.artifact_id in artifact_ids
    )


def _validate_cross_format_payloads(
    payloads: Sequence[ArtifactPayloadV2],
    run_id: str,
    configuration_id: str,
) -> None:
    by_id = {item.artifact_id: item for item in payloads}
    try:
        report_json = by_id["evaluation-report-json"]
        report_html = by_id["evaluation-report-html"]
        operations_text = by_id["operations-summary-txt"]
        operations_csv = by_id["operations-summary-csv"]
        report_raw = decode_json_report(report_json.content.decode("utf-8"))
        if not isinstance(report_raw, dict):
            raise ValueError
        loaded_report = validate_versioned_report(cast(Mapping[str, object], report_raw))
        if loaded_report.schema_version is not ReportSchemaVersion.V2:
            raise ValueError
        report = loaded_report.document
        if (
            report.get("run_id") != run_id
            or report.get("configuration_id") != configuration_id
            or report_json.schema_version != report.get("schema_version")
            or report_html.schema_version != report.get("schema_version")
            or report_json.content != canonical_versioned_report_document(report)
        ):
            raise ValueError
        html = report_html.content.decode("utf-8")
        verify_html_parity_v2(report, html)
        if html != render_html_report_v2(report) + "\n":
            raise ValueError
        text = operations_text.content.decode("utf-8")
        csv_text = operations_csv.content.decode("utf-8")
        text_summary = parse_operations_text(text)
        csv_summary = parse_operations_csv(csv_text)
        verify_operations_parity(text_summary, text, csv_text)
        report_summary = validate_operations_summary_v2(
            cast(Mapping[str, object], report["operations_summary"])
        )
        _validate_operations_performance_parity(
            report_summary,
            report["performance_evidence"],
        )
        if (
            text_summary != csv_summary
            or text_summary != report_summary
            or text_summary.run_id != run_id
            or text_summary.configuration_id != configuration_id
            or operations_text.schema_version != text_summary.schema_version
            or operations_csv.schema_version != text_summary.schema_version
        ):
            raise ValueError
        if _DOCUMENTATION_ARTIFACT_IDS.issubset(by_id):
            dictionary_payload = by_id["structured-log-field-dictionary"]
            sample_payload = by_id["privacy-safe-log-sample"]
            dictionary_text = dictionary_payload.content.decode("utf-8")
            sample_text = sample_payload.content.decode("utf-8")
            dictionary = parse_log_dictionary_json(dictionary_text)
            events = parse_log_sample_jsonl(sample_text)
            validate_log_documentation(dictionary, events)
            if (
                dictionary_payload.schema_version != LOG_DICTIONARY_SCHEMA_VERSION
                or sample_payload.schema_version != LOG_EVENT_SCHEMA_VERSION
                or dictionary_text != canonical_log_dictionary_json(dictionary) + "\n"
                or sample_text != render_log_sample_jsonl(events, dictionary=dictionary)
            ):
                raise ValueError
    except (KeyError, TypeError, UnicodeError, ValueError) as error:
        raise ArtifactIntegrityError("artifact cross-format validation failed") from error


def _validate_operations_performance_parity(
    summary: OperationsSummary,
    performance_payload: object,
) -> None:
    """Bind the human-facing operations projection to its canonical evidence."""

    performance = PerformanceEvidenceV2.model_validate(performance_payload)
    measured = performance.measured
    cost = performance.cost
    by_id = {observation.metric_id: observation for observation in summary.observations}

    total = by_id[OperationsMetricId.TOTAL_LOGICAL_REQUESTS]
    successful = by_id[OperationsMetricId.SUCCESSFUL_LOGICAL_REQUESTS]
    if (
        _observed_value(total) != measured.logical_attempt_count
        or _observed_value(total) != cost.logical_attempt_count
        or _observed_value(successful) != measured.successful_logical_attempt_count
        or _observed_value(successful) != cost.successful_logical_attempt_count
    ):
        raise ValueError("operations logical-attempt counts disagree with performance evidence")

    _validate_latency_evidence_parity(by_id, performance)
    _validate_token_evidence_parity(by_id, performance)
    _validate_cost_evidence_parity(by_id, performance)


def _validate_latency_evidence_parity(
    by_id: Mapping[str, MetricObservation],
    performance: PerformanceEvidenceV2,
) -> None:
    measured = performance.measured
    all_attempts = measured.latency_ms.all_attempts
    p50 = by_id[OperationsMetricId.LATENCY_P50_MS]
    p95 = by_id[OperationsMetricId.LATENCY_P95_MS]
    if all_attempts is None:
        if measured.http_attempt_count != 0:
            raise ValueError("missing all-attempt latency evidence has a nonzero denominator")
        _require_unavailable_observation(
            p50,
            reason="no-latency-attempts",
            denominator=None,
        )
        _require_unavailable_observation(
            p95,
            reason="no-latency-attempts",
            denominator=None,
        )
        return
    if all_attempts.count != measured.http_attempt_count:
        raise ValueError("all-attempt latency denominator disagrees with measured attempts")
    _require_observed_observation(
        p50,
        value=all_attempts.p50_ms,
        numerator=all_attempts.p50_ms,
        denominator=all_attempts.count,
    )
    _require_observed_observation(
        p95,
        value=all_attempts.p95_ms,
        numerator=all_attempts.p95_ms,
        denominator=all_attempts.count,
    )


def _validate_token_evidence_parity(
    by_id: Mapping[str, MetricObservation],
    performance: PerformanceEvidenceV2,
) -> None:
    cost = performance.cost
    for direction, metric_id in (
        (TokenDirection.INPUT, OperationsMetricId.INPUT_TOKENS),
        (TokenDirection.OUTPUT, OperationsMetricId.OUTPUT_TOKENS),
    ):
        entries = tuple(item for item in cost.role_direction_tokens if item.direction is direction)
        provider_attempt_count = sum(item.provider_attempt_count for item in entries)
        if provider_attempt_count != cost.provider_attempt_count:
            raise ValueError("token denominator disagrees with provider-attempt evidence")
        observation = by_id[metric_id]
        if provider_attempt_count == 0:
            _require_unavailable_observation(
                observation,
                reason="no-provider-attempts",
                denominator=None,
            )
            continue

        known_tokens = sum(item.known_tokens for item in entries)
        unknown_usage_count = sum(item.unknown_usage_attempt_count for item in entries)
        complete = unknown_usage_count == 0 and all(
            item.status
            in {
                EvidenceAvailability.AVAILABLE,
                EvidenceAvailability.NOT_APPLICABLE,
            }
            for item in entries
        )
        if complete:
            total_tokens = sum(item.total_tokens or 0 for item in entries)
            if total_tokens != known_tokens:
                raise ValueError("complete token totals disagree with known-token evidence")
            _require_observed_observation(
                observation,
                value=total_tokens,
                numerator=total_tokens,
                denominator=provider_attempt_count,
            )
            continue

        reason = f"{metric_id.value}-usage-incomplete"
        _require_unavailable_observation(
            observation,
            reason=reason,
            denominator=provider_attempt_count,
            known_numerator=known_tokens,
        )


def _validate_cost_evidence_parity(
    by_id: Mapping[str, MetricObservation],
    performance: PerformanceEvidenceV2,
) -> None:
    cost = performance.cost
    currency = cost.pricing.currency
    _validate_normalized_cost_parity(
        by_id[OperationsMetricId.COST_PER_1000_LOGICAL_ATTEMPTS],
        normalized=cost.cost_per_1000_logical_attempts,
        expected_denominator=cost.logical_attempt_count,
        expected_unit=f"{currency}-per-1000-logical-attempts",
        zero_reason="no-logical-attempts",
        total_cost=cost.total_cost,
    )
    _validate_normalized_cost_parity(
        by_id[OperationsMetricId.COST_PER_1000_SUCCESSES],
        normalized=cost.cost_per_1000_successes,
        expected_denominator=cost.successful_logical_attempt_count,
        expected_unit=f"{currency}-per-1000-successes",
        zero_reason="no-successful-attempts",
        total_cost=cost.total_cost,
    )


def _validate_normalized_cost_parity(
    observation: MetricObservation,
    *,
    normalized: NormalizedCostV2,
    expected_denominator: int,
    expected_unit: str,
    zero_reason: str,
    total_cost: Decimal | None,
) -> None:
    if observation.unit != expected_unit or normalized.denominator != expected_denominator:
        raise ValueError("normalized cost unit or denominator disagrees with pricing evidence")
    if expected_denominator == 0:
        _require_unavailable_observation(
            observation,
            reason=zero_reason,
            denominator=None,
        )
        return
    if normalized.status is EvidenceAvailability.AVAILABLE:
        if normalized.per_1000 is None or total_cost is None:
            raise ValueError("available normalized cost is missing its exact value")
        _require_observed_observation(
            observation,
            value=float(normalized.per_1000),
            numerator=float(total_cost),
            denominator=expected_denominator,
        )
        return
    if normalized.status is not EvidenceAvailability.UNAVAILABLE or normalized.per_1000 is not None:
        raise ValueError("normalized cost availability is inconsistent")
    _require_unavailable_observation(
        observation,
        reason="cost-incomplete",
        denominator=expected_denominator,
    )


def _observed_value(observation: MetricObservation) -> float:
    if observation.status is not MetricObservationStatus.OBSERVED or isinstance(
        observation.value, UnavailableValue
    ):
        raise ValueError("expected an observed operations value")
    return observation.value


def _require_observed_observation(
    observation: MetricObservation,
    *,
    value: float | int,
    numerator: float | int,
    denominator: int,
) -> None:
    if (
        observation.status is not MetricObservationStatus.OBSERVED
        or not observation.eligible
        or isinstance(observation.value, UnavailableValue)
        or isinstance(observation.numerator, UnavailableValue)
        or observation.value != float(value)
        or observation.numerator != float(numerator)
        or observation.denominator != denominator
    ):
        raise ValueError("operations observation disagrees with canonical evidence")


def _require_unavailable_observation(
    observation: MetricObservation,
    *,
    reason: str,
    denominator: int | None,
    known_numerator: int | None = None,
) -> None:
    expected_denominator = (
        denominator is None
        and isinstance(observation.denominator, UnavailableValue)
        and observation.denominator.reason == reason
    ) or observation.denominator == denominator
    expected_numerator = (
        isinstance(observation.numerator, UnavailableValue)
        and observation.numerator.reason == reason
    ) or (
        known_numerator is not None
        and not isinstance(observation.numerator, UnavailableValue)
        and observation.numerator == float(known_numerator)
    )
    expected_eligibility = denominator is not None
    if (
        observation.status is not MetricObservationStatus.UNAVAILABLE
        or observation.eligible is not expected_eligibility
        or not isinstance(observation.value, UnavailableValue)
        or observation.value.reason != reason
        or not expected_denominator
        or not expected_numerator
    ):
        raise ValueError("operations unavailable observation disagrees with canonical evidence")


def _resolve_descriptor_path(root: Path, descriptor: ArtifactDescriptor) -> Path:
    contract = _ARTIFACT_CONTRACTS.get(descriptor.artifact_id)
    if contract is None or ArtifactFormatV2(descriptor.format) is not contract.format:
        raise ArtifactIntegrityError("artifact descriptor contract is unknown")
    pure = PurePosixPath(descriptor.relative_path)
    if (
        descriptor.relative_path != contract.filename
        or pure.is_absolute()
        or len(pure.parts) != 1
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ArtifactIntegrityError("artifact descriptor path is unsafe")
    path = root / contract.filename
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ArtifactIntegrityError("artifact descriptor resolves outside its bundle")
    return path


def _manifest_content_hash(manifest: ArtifactManifestV2) -> str:
    payload = manifest.model_dump(mode="json", exclude={"manifest_content_hash"})
    return _content_digest(canonical_json_value(payload).encode("utf-8"))


def _content_digest(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _write_new_file(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise ArtifactBundleExistsError("immutable artifact already exists") from error


def _require_privacy_scan(root: Path, fixture_path: Path) -> None:
    try:
        report = scan_artifacts((root,), fixture_path=fixture_path)
    except Exception as error:
        raise ArtifactIntegrityError("artifact privacy validation failed closed") from error
    if report.get("passed") is not True:
        raise ArtifactIntegrityError("artifact privacy validation failed")


def _privacy_fixture(path: Path | str | None) -> Path:
    candidate = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[3]
        / "evaluations"
        / "privacy"
        / "supported-fixtures-v1.json"
    )
    expanded = candidate.expanduser()
    if expanded.is_symlink():
        raise ArtifactIntegrityError("privacy fixture must not be a symbolic link")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise ArtifactIntegrityError("privacy fixture file is unavailable")
    return resolved


def _safe_catalog_root(root: Path | str) -> Path:
    candidate = Path(root).expanduser()
    if candidate.is_symlink():
        raise ArtifactIntegrityError("artifact catalog root must not be a symbolic link")
    return candidate.resolve()


def _safe_existing_bundle_root(root: Path | str) -> Path:
    candidate = Path(root).expanduser()
    if candidate.is_symlink():
        raise ArtifactIntegrityError("artifact bundle must not be a symbolic link")
    resolved = candidate.resolve()
    if not resolved.is_dir():
        raise ArtifactNotFoundError("artifact-bundle-not-found")
    return resolved


def _bundle_root(catalog_root: Path, run_id: str) -> Path:
    if not _safe_identifier(run_id):
        raise ArtifactNotFoundError("artifact-manifest-not-found")
    unresolved = catalog_root / run_id
    if unresolved.is_symlink():
        raise ArtifactIntegrityError("artifact bundle must not be a symbolic link")
    candidate = unresolved.resolve()
    if (
        not candidate.is_relative_to(catalog_root)
        or candidate.parent != catalog_root
        or candidate.name != run_id
    ):
        raise ArtifactIntegrityError("artifact bundle path escapes its catalog")
    return candidate


def _safe_identifier(value: str) -> bool:
    return isinstance(value, str) and _SAFE_OPAQUE_ID.fullmatch(value) is not None


__all__ = [
    "ARTIFACT_DOWNLOAD_FILENAMES_V2",
    "ARTIFACT_MANIFEST_FILENAME",
    "ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "ARTIFACT_MEDIA_TYPES_BY_ID_V2",
    "ARTIFACT_MEDIA_TYPES_V2",
    "ArtifactBundleExistsError",
    "ArtifactCatalogV2",
    "ArtifactFormatV2",
    "ArtifactIntegrityError",
    "ArtifactManifestV2",
    "ArtifactNotFoundError",
    "ArtifactPayloadV2",
    "ArtifactPublicationError",
    "PublishedArtifactBundleV2",
    "ResolvedArtifactV2",
    "allowed_artifact_media_types_v2",
    "artifact_download_filename_v2",
    "artifact_media_type_v2",
    "canonical_artifact_manifest_document",
    "load_artifact_manifest",
    "publish_artifact_bundle",
    "resolve_verified_artifact",
    "verify_artifact_manifest",
]
