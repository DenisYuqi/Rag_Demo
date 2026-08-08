"""Canonical schema-v2 evaluation report contract and JSON persistence.

This module is intentionally additive.  The sealed Phase 12 report remains on
the v1 implementation in :mod:`rag_mvp.evaluation.json_report`; schema-v2
artifacts opt in to the stricter typed contract defined here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import cache
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from rag_mvp.domain import (
    AcceptanceContract,
    ArtifactDescriptor,
    GateResult,
    GateStatus,
    MetricObservation,
    MetricObservationStatus,
    OperationsSummary,
    UnavailableValue,
)
from rag_mvp.domain._base import DomainModel, Identifier, utc_now
from rag_mvp.observability.costs_v2 import COST_EVIDENCE_SCHEMA_VERSION
from rag_mvp.performance.evidence_v2 import (
    PERFORMANCE_EVIDENCE_SCHEMA_VERSION,
    PerformanceEvidenceV2,
)
from rag_mvp.safety.output import OutputRedactionError, redact_output
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, Redactor

from .json_report import (
    MAX_REPORT_BYTES,
    JsonObject,
    ReportSerializationError,
    ReportWriteError,
    SchemaIssue,
    _atomic_write_text,
    _json_pointer,
    _schema_issue,
    canonical_json_value,
    decode_json_report,
)

REPORT_SCHEMA_VERSION_V2: Literal["2.0.0"] = "2.0.0"
REPORT_SCHEMA_URI_V2: Literal["https://rag-mvp.local/schemas/evaluation-report-v2.schema.json"] = (
    "https://rag-mvp.local/schemas/evaluation-report-v2.schema.json"
)
REPORT_SCHEMA_RESOURCE_V2 = "evaluation-report-v2.schema.json"

type Sha256DigestV2 = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class EvaluationReportProvenanceV2(DomainModel):
    """Immutable identities required to reproduce or compare a v2 report."""

    dataset_id: Identifier
    dataset_version: Identifier
    dataset_content_hash: Sha256DigestV2
    corpus_id: Identifier
    corpus_version: Identifier
    corpus_content_hash: Sha256DigestV2
    case_set_content_hash: Sha256DigestV2
    experiment_plan_id: Identifier
    experiment_plan_content_hash: Sha256DigestV2
    configuration_id: Identifier
    code_revision: Identifier
    pricing_version: Identifier
    pricing_content_hash: Sha256DigestV2
    evaluation_scorer_backend: Literal["legacy", "ragas"] = Field(
        default="legacy",
        exclude_if=lambda value: value == "legacy",
    )
    evaluation_judge_model: Annotated[
        str,
        Field(min_length=1, max_length=255),
    ] | None = Field(default=None, exclude_if=lambda value: value is None)


class CategoryResultV2(DomainModel):
    """Denominator-bearing metric observations for one declared category."""

    category_id: Identifier
    case_count: Annotated[int, Field(gt=0)]
    observations: Annotated[tuple[MetricObservation, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_category_evidence(self) -> CategoryResultV2:
        metric_ids = tuple(item.metric_id for item in self.observations)
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("category metric observations must be unique")
        if any(
            isinstance(item.denominator, int) and item.denominator > self.case_count
            for item in self.observations
        ):
            raise ValueError("category metric denominator exceeds its case count")
        return self


class FailedCaseEvidenceV2(DomainModel):
    """Allowlisted case-level diagnostics with no prompt or document content."""

    case_id: Identifier
    category_ids: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    failed_metric_ids: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    safe_reason_code: Identifier
    trace_id: Identifier | UnavailableValue

    @model_validator(mode="after")
    def validate_case_diagnostics(self) -> FailedCaseEvidenceV2:
        if len(self.category_ids) != len(set(self.category_ids)):
            raise ValueError("failed-case categories must be unique")
        if len(self.failed_metric_ids) != len(set(self.failed_metric_ids)):
            raise ValueError("failed-case metrics must be unique")
        return self


class EvaluationReportV2(DomainModel):
    """Concrete schema-v2 source of truth for JSON and HTML projections."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    schema_uri: Literal["https://rag-mvp.local/schemas/evaluation-report-v2.schema.json"] = Field(
        default=REPORT_SCHEMA_URI_V2, alias="$schema"
    )
    schema_version: Literal["2.0.0"] = REPORT_SCHEMA_VERSION_V2
    report_id: Identifier
    run_id: Identifier
    configuration_id: Identifier
    generated_at: AwareDatetime = Field(default_factory=utc_now)
    provenance: EvaluationReportProvenanceV2
    acceptance_contract: AcceptanceContract
    acceptance_gate_id: Identifier
    gates: Annotated[tuple[GateResult, ...], Field(min_length=1)]
    performance_evidence: PerformanceEvidenceV2
    operations_summary: OperationsSummary
    category_results: tuple[CategoryResultV2, ...] = ()
    failed_cases: tuple[FailedCaseEvidenceV2, ...] = ()
    artifacts: Annotated[tuple[ArtifactDescriptor, ...], Field(min_length=1)]
    status: GateStatus
    accepted: bool
    limitations: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_report_evidence(self) -> EvaluationReportV2:
        if self.configuration_id != self.provenance.configuration_id:
            raise ValueError("report and provenance configuration identities differ")
        if self.configuration_id != self.operations_summary.configuration_id:
            raise ValueError("report and operations configuration identities differ")
        if self.run_id != self.operations_summary.run_id:
            raise ValueError("report and operations run identities differ")
        if self.run_id != self.performance_evidence.run_id:
            raise ValueError("report and performance run identities differ")
        if self.configuration_id != self.performance_evidence.configuration_id:
            raise ValueError("report and performance configuration identities differ")
        if (
            self.acceptance_contract.performance_schema_version
            != PERFORMANCE_EVIDENCE_SCHEMA_VERSION
            or self.acceptance_contract.cost_schema_version != COST_EVIDENCE_SCHEMA_VERSION
        ):
            raise ValueError("acceptance contract references the wrong evidence schema")
        pricing = self.performance_evidence.cost.pricing
        if (
            pricing.pricing_version != self.provenance.pricing_version
            or pricing.digest != self.provenance.pricing_content_hash
        ):
            raise ValueError("performance pricing differs from report provenance")

        gate_by_id = {gate.gate_id: gate for gate in self.gates}
        if len(gate_by_id) != len(self.gates):
            raise ValueError("report gate identifiers must be unique")
        acceptance_gate = gate_by_id.get(self.acceptance_gate_id)
        if acceptance_gate is None:
            raise ValueError("declared acceptance gate is missing")
        if acceptance_gate.profile_version != self.acceptance_contract.gate_profile_version:
            raise ValueError("acceptance gate profile differs from its contract")

        requirement_by_metric = {
            requirement.metric_id: requirement
            for requirement in self.acceptance_contract.metric_requirements
        }
        observation_by_metric = {
            observation.metric_id: observation for observation in acceptance_gate.observations
        }
        if set(requirement_by_metric) != set(observation_by_metric):
            raise ValueError("acceptance gate observations differ from contract metrics")
        for metric_id, requirement in requirement_by_metric.items():
            observation = observation_by_metric[metric_id]
            if (
                observation.unit != requirement.unit
                or observation.threshold != requirement.threshold
                or observation.operator is not requirement.operator
            ):
                raise ValueError("acceptance observation policy differs from its contract")
            if (
                not observation.eligible
                or isinstance(observation.denominator, UnavailableValue)
                or observation.denominator < requirement.minimum_denominator
                or observation.status
                not in {MetricObservationStatus.PASSED, MetricObservationStatus.FAILED}
            ):
                raise ValueError("acceptance observation lacks an eligible denominator")

        expected_accepted = acceptance_gate.valid and acceptance_gate.passed
        if self.accepted is not expected_accepted or self.status is not acceptance_gate.status:
            raise ValueError("report decision differs from its declared acceptance gate")

        category_ids = tuple(item.category_id for item in self.category_results)
        if len(category_ids) != len(set(category_ids)):
            raise ValueError("report category identifiers must be unique")
        failed_case_ids = tuple(item.case_id for item in self.failed_cases)
        if len(failed_case_ids) != len(set(failed_case_ids)):
            raise ValueError("report failed-case identifiers must be unique")

        artifact_ids = tuple(item.artifact_id for item in self.artifacts)
        artifact_paths = tuple(item.relative_path for item in self.artifacts)
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("report artifact identifiers must be unique")
        if len(artifact_paths) != len(set(artifact_paths)):
            raise ValueError("report artifact paths must be unique")
        if not set(self.operations_summary.source_artifact_ids).issubset(artifact_ids):
            raise ValueError("operations summary references an unknown artifact")
        if len(self.limitations) != len(set(self.limitations)):
            raise ValueError("report limitation codes must be unique")
        return self


class ReportV2ValidationError(ValueError):
    """Raised with content-free issues when schema-v2 validation fails."""

    def __init__(self, issues: Sequence[SchemaIssue]) -> None:
        normalized = tuple(issues)
        if not normalized:
            raise ValueError("schema-v2 report validation requires at least one issue")
        self.issues = normalized
        first = normalized[0]
        super().__init__(
            f"schema-v2 report validation failed at {first.path} ({first.keyword}); "
            f"{len(normalized)} issue(s)"
        )


def load_report_schema_v2() -> dict[str, Any]:
    """Load a defensive copy of the meta-validated packaged v2 schema."""

    return deepcopy(_packaged_report_schema_v2())


@cache
def _packaged_report_schema_v2() -> dict[str, Any]:
    resource = files("rag_mvp.evaluation").joinpath("schemas", REPORT_SCHEMA_RESOURCE_V2)
    raw = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("packaged schema-v2 report schema is not an object")
    schema = cast(dict[str, Any], raw)
    Draft202012Validator.check_schema(schema)
    return schema


@cache
def _report_validator_v2() -> Draft202012Validator:
    return Draft202012Validator(
        _packaged_report_schema_v2(),
        format_checker=FormatChecker(),
    )


def validate_report_v2(report: Mapping[str, object] | BaseModel) -> JsonObject:
    """Validate JSON shape and the cross-model schema-v2 evidence contract."""

    normalized = _normalize_report_v2(report)
    issues = tuple(
        _schema_issue(error)
        for error in sorted(
            _report_validator_v2().iter_errors(normalized),
            key=lambda item: (
                tuple(str(part) for part in item.absolute_path),
                str(item.validator),
            ),
        )
    )
    if issues:
        raise ReportV2ValidationError(issues)
    try:
        validated = EvaluationReportV2.model_validate(normalized)
    except ValidationError as error:
        raise ReportV2ValidationError(
            tuple(_pydantic_issue(item) for item in error.errors(include_input=False))
        ) from error
    return cast(JsonObject, validated.model_dump(mode="json", by_alias=True))


def parse_report_v2(report: Mapping[str, object] | BaseModel) -> EvaluationReportV2:
    """Return the immutable typed representation of a valid schema-v2 report."""

    return EvaluationReportV2.model_validate(validate_report_v2(report))


def canonical_report_json_v2(report: Mapping[str, object] | BaseModel) -> str:
    """Return validated, deterministic schema-v2 JSON without a final newline."""

    return canonical_json_value(validate_report_v2(report))


def canonical_report_document_v2(report: Mapping[str, object] | BaseModel) -> bytes:
    """Return the exact canonical UTF-8 bytes persisted for a v2 JSON report."""

    return (canonical_report_json_v2(report) + "\n").encode("utf-8")


def report_content_hash_v2(report: Mapping[str, object] | BaseModel) -> str:
    """Hash the exact canonical document, including its final newline."""

    return f"sha256:{sha256(canonical_report_document_v2(report)).hexdigest()}"


def prepare_report_v2(
    report: Mapping[str, object] | BaseModel,
    *,
    redactor: Redactor = DEFAULT_REDACTOR,
) -> JsonObject:
    """Normalize, redact, and validate a v2 report before publication."""

    normalized = _normalize_report_v2(report)
    try:
        redacted = redact_output(normalized, redactor=redactor)
    except (OutputRedactionError, TypeError, ValueError, RecursionError) as error:
        raise ReportSerializationError("schema-v2 report redaction failed closed") from error
    if not isinstance(redacted, dict):
        raise ReportSerializationError("redacted schema-v2 report is not an object")
    return validate_report_v2(cast(Mapping[str, object], redacted))


def write_json_report_v2(
    report: Mapping[str, object] | BaseModel,
    output_path: Path | str,
    *,
    overwrite: bool = False,
    redactor: Redactor = DEFAULT_REDACTOR,
) -> Path:
    """Atomically publish canonical v2 JSON without replacing by default."""

    target = Path(output_path)
    if target.suffix.casefold() != ".json":
        raise ReportWriteError("schema-v2 JSON report path must end in .json")
    prepared = prepare_report_v2(report, redactor=redactor)
    payload = canonical_report_json_v2(prepared) + "\n"
    return _atomic_write_text(target, payload, overwrite=overwrite)


def load_json_report_v2(path: Path | str) -> JsonObject:
    """Read a bounded, strict UTF-8 JSON document and validate schema v2."""

    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as error:
        raise ReportSerializationError("schema-v2 report file is unavailable") from error
    if size <= 0 or size > MAX_REPORT_BYTES:
        raise ReportSerializationError("schema-v2 report size is outside allowed bounds")
    try:
        raw = decode_json_report(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ReportSerializationError) as error:
        if isinstance(error, ReportSerializationError):
            raise
        raise ReportSerializationError("schema-v2 report is not valid UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise ReportSerializationError("schema-v2 report document is not an object")
    return validate_report_v2(cast(Mapping[str, object], raw))


def _normalize_report_v2(report: Mapping[str, object] | BaseModel) -> JsonObject:
    try:
        candidate: object
        if isinstance(report, BaseModel):
            candidate = report.model_dump(mode="json", by_alias=True)
        else:
            candidate = report
        normalized = decode_json_report(canonical_json_value(candidate))
    except (TypeError, ValueError, OverflowError, json.JSONDecodeError) as error:
        if isinstance(error, ReportSerializationError):
            raise
        raise ReportSerializationError(
            "schema-v2 report is not safely JSON serializable"
        ) from error
    if not isinstance(normalized, dict):
        raise ReportSerializationError("schema-v2 report document must be an object")
    return cast(JsonObject, normalized)


def _pydantic_issue(error: Mapping[str, Any]) -> SchemaIssue:
    location = tuple(error.get("loc", ()))
    error_type = str(error.get("type", "semantic"))
    return SchemaIssue(
        path=_json_pointer(location),
        keyword="semantic",
        message=(
            "value violates the schema-v2 semantic contract"
            if error_type
            else "schema-v2 semantic validation failed"
        ),
    )
