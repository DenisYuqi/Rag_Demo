"""Read-only version dispatch for sealed v1 and schema-v2 reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from pydantic import BaseModel

from rag_mvp.safety.redactor import DEFAULT_REDACTOR, Redactor

from .html_report import render_html_report, verify_html_parity
from .html_report_v2 import render_html_report_v2, verify_html_parity_v2
from .json_report import (
    MAX_REPORT_BYTES,
    REPORT_SCHEMA_VERSION,
    JsonObject,
    ReportSerializationError,
    canonical_report_document,
    decode_json_report,
    validate_report,
)
from .report_v2 import (
    REPORT_SCHEMA_VERSION_V2,
    canonical_report_document_v2,
    validate_report_v2,
)


class ReportSchemaVersion(StrEnum):
    V1 = REPORT_SCHEMA_VERSION
    V2 = REPORT_SCHEMA_VERSION_V2


class UnsupportedReportVersionError(ValueError):
    """Raised without echoing input when a report version is unsupported."""


@dataclass(frozen=True, slots=True)
class LoadedEvaluationReport:
    """Validated report plus the dispatch decision used to read it."""

    schema_version: ReportSchemaVersion
    document: JsonObject

    @property
    def is_legacy(self) -> bool:
        return self.schema_version is ReportSchemaVersion.V1


def detect_report_schema_version(
    report: Mapping[str, object] | BaseModel,
) -> ReportSchemaVersion:
    """Read only the explicit version discriminator without upgrading content."""

    if isinstance(report, BaseModel):
        raw: object = report.model_dump(mode="json", by_alias=True)
    else:
        raw = report
    if not isinstance(raw, Mapping):
        raise ReportSerializationError("versioned report document must be an object")
    version = raw.get("schema_version")
    if not isinstance(version, str):
        raise UnsupportedReportVersionError("report schema version is missing")
    try:
        return ReportSchemaVersion(version)
    except ValueError as error:
        raise UnsupportedReportVersionError("report schema version is unsupported") from error


def validate_versioned_report(
    report: Mapping[str, object] | BaseModel,
) -> LoadedEvaluationReport:
    """Dispatch validation while preserving the source report's schema version."""

    version = detect_report_schema_version(report)
    document = (
        validate_report(report) if version is ReportSchemaVersion.V1 else validate_report_v2(report)
    )
    return LoadedEvaluationReport(schema_version=version, document=document)


def load_evaluation_report(path: Path | str) -> LoadedEvaluationReport:
    """Strictly read and validate a v1 or v2 JSON report without writing it."""

    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as error:
        raise ReportSerializationError("versioned report file is unavailable") from error
    if size <= 0 or size > MAX_REPORT_BYTES:
        raise ReportSerializationError("versioned report size is outside allowed bounds")
    try:
        raw = decode_json_report(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ReportSerializationError) as error:
        if isinstance(error, ReportSerializationError):
            raise
        raise ReportSerializationError("versioned report is not valid UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise ReportSerializationError("versioned report document is not an object")
    return validate_versioned_report(cast(Mapping[str, object], raw))


def canonical_versioned_report_document(
    report: Mapping[str, object] | BaseModel,
) -> bytes:
    """Return the version-specific canonical JSON document bytes."""

    version = detect_report_schema_version(report)
    if version is ReportSchemaVersion.V1:
        return canonical_report_document(report)
    return canonical_report_document_v2(report)


def render_versioned_html_report(
    report: Mapping[str, object] | BaseModel,
    *,
    redactor: Redactor = DEFAULT_REDACTOR,
) -> str:
    """Dispatch to the matching immutable HTML contract."""

    version = detect_report_schema_version(report)
    if version is ReportSchemaVersion.V1:
        return render_html_report(report, redactor=redactor)
    return render_html_report_v2(report, redactor=redactor)


def verify_versioned_html_parity(
    report: Mapping[str, object] | BaseModel,
    html: str,
    *,
    redactor: Redactor = DEFAULT_REDACTOR,
) -> None:
    """Apply the exact parity validator corresponding to the report version."""

    version = detect_report_schema_version(report)
    if version is ReportSchemaVersion.V1:
        verify_html_parity(report, html, redactor=redactor)
    else:
        verify_html_parity_v2(report, html, redactor=redactor)
