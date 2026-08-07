"""Canonical operations evidence and deterministic TXT/CSV projections."""

from __future__ import annotations

import csv
import io
import math
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, cast

from pydantic import AwareDatetime, BaseModel, Field, model_validator

from rag_mvp.domain import (
    EvidenceComparisonOperator,
    MetricObservation,
    MetricObservationStatus,
    OperationsSummary,
    UnavailableValue,
)
from rag_mvp.domain._base import DomainModel, Identifier, NonNegativeFiniteFloat, utc_now
from rag_mvp.performance.load_report import nearest_rank_percentile
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, RedactionError, Redactor

from .json_report import (
    MAX_REPORT_BYTES,
    ReportSerializationError,
    ReportWriteError,
    _atomic_write_text,
    canonical_json_value,
    decode_json_report,
)

OPERATIONS_RENDERER_VERSION = "operations-summary-projection-v2"
OPERATIONS_SCORER_VERSION = "operations-summary-v2"
OPERATIONS_TEXT_HEADER = "RAG Operations Summary v2"
OPERATIONS_TEXT_COLUMNS = (
    "metric_id",
    "unit",
    "status",
    "eligible",
    "value",
    "numerator",
    "denominator",
    "threshold",
    "operator",
    "scorer_version",
    "evidence_references",
)
_SAFE_PUBLIC_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,254}$")
_PROHIBITED_OUTPUT_FIELD = re.compile(
    r"(?i)(raw[_-]?(?:prompt|question|answer|content|text)|retrieved[_-]?text|"
    r"document[_-]?(?:text|content)|authorization|api[_-]?key|password|private[_-]?key)"
)
_ABSOLUTE_PATH = re.compile(r"(?i)(?:[A-Z]:[\\/]|/(?:home|users|tmp|var|etc)/)")


class OperationsMetricId(StrEnum):
    TOTAL_LOGICAL_REQUESTS = "total-logical-requests"
    SUCCESSFUL_LOGICAL_REQUESTS = "successful-logical-requests"
    LATENCY_P50_MS = "all-attempt-latency-p50-ms"
    LATENCY_P95_MS = "all-attempt-latency-p95-ms"
    INPUT_TOKENS = "input-tokens"
    OUTPUT_TOKENS = "output-tokens"
    CACHE_HITS = "cache-hits"
    CACHE_ELIGIBLE_LOOKUPS = "cache-eligible-lookups"
    CACHE_HIT_RATE = "cache-hit-rate"
    REFUSALS = "refusals"
    ANSWERED_REQUESTS = "answered-requests"
    REFUSAL_RATE = "refusal-rate"
    COMPLIANT_ANSWERS = "compliant-answers"
    SCORED_ANSWERS = "scored-answers"
    ANSWER_COMPLIANCE_RATE = "answer-compliance-rate"
    COST_PER_1000_LOGICAL_ATTEMPTS = "cost-per-1000-logical-attempts"
    COST_PER_1000_SUCCESSES = "cost-per-1000-successes"


OPERATIONS_METRIC_ORDER = tuple(OperationsMetricId)


class OperationsEvidenceInputV2(DomainModel):
    """Raw counts used to derive the canonical operations record once."""

    run_id: Identifier
    configuration_id: Identifier
    total_logical_requests: Annotated[int, Field(ge=0)]
    successful_logical_requests: Annotated[int, Field(ge=0)]
    all_attempt_latency_ms: tuple[NonNegativeFiniteFloat, ...]
    provider_attempt_count: Annotated[int, Field(ge=0)]
    input_tokens: Annotated[int, Field(ge=0)] | None
    output_tokens: Annotated[int, Field(ge=0)] | None
    unknown_input_token_usage_attempt_count: Annotated[int, Field(ge=0)] = 0
    unknown_output_token_usage_attempt_count: Annotated[int, Field(ge=0)] = 0
    cache_hits: Annotated[int, Field(ge=0)]
    cache_eligible_lookups: Annotated[int, Field(ge=0)]
    refusals: Annotated[int, Field(ge=0)]
    answered_requests: Annotated[int, Field(ge=0)]
    compliant_answers: Annotated[int, Field(ge=0)]
    scored_answers: Annotated[int, Field(ge=0)]
    total_cost: Annotated[Decimal, Field(ge=0)] | None
    currency: Identifier
    source_artifact_ids: tuple[Identifier, ...] = ()
    generated_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_counts(self) -> OperationsEvidenceInputV2:
        if self.successful_logical_requests > self.total_logical_requests:
            raise ValueError("successful requests exceed total logical requests")
        if self.refusals + self.answered_requests != self.successful_logical_requests:
            raise ValueError("answered and refused requests must equal successful requests")
        if self.compliant_answers > self.scored_answers:
            raise ValueError("compliant answers exceed scored answers")
        if self.scored_answers > self.answered_requests:
            raise ValueError("scored answers exceed answered requests")
        if self.cache_hits > self.cache_eligible_lookups:
            raise ValueError("cache hits exceed eligible lookups")
        unknown_usage_counts = (
            self.unknown_input_token_usage_attempt_count,
            self.unknown_output_token_usage_attempt_count,
        )
        if any(count > self.provider_attempt_count for count in unknown_usage_counts):
            raise ValueError("unknown token usage attempts exceed provider attempts")
        if self.provider_attempt_count == 0 and (
            self.input_tokens not in {None, 0}
            or self.output_tokens not in {None, 0}
            or any(unknown_usage_counts)
        ):
            raise ValueError("token totals require a provider-attempt denominator")
        if self.provider_attempt_count > 0:
            for token_count, unknown_count in zip(
                (self.input_tokens, self.output_tokens), unknown_usage_counts, strict=True
            ):
                if token_count is None and unknown_count == 0:
                    raise ValueError("missing token totals require an unknown-usage count")
        if len(self.source_artifact_ids) != len(set(self.source_artifact_ids)):
            raise ValueError("operations source artifact identifiers must be unique")
        return self


class OperationsProjectionError(ValueError):
    """Raised when an operations record or projection is malformed or divergent."""


def build_operations_summary_v2(evidence: OperationsEvidenceInputV2) -> OperationsSummary:
    """Build the sole canonical record used by JSON, TXT, CSV, API, and UI."""

    references = evidence.source_artifact_ids
    observations = (
        _count_observation(
            OperationsMetricId.TOTAL_LOGICAL_REQUESTS,
            evidence.total_logical_requests,
            references,
        ),
        _count_observation(
            OperationsMetricId.SUCCESSFUL_LOGICAL_REQUESTS,
            evidence.successful_logical_requests,
            references,
        ),
        _latency_observation(
            OperationsMetricId.LATENCY_P50_MS,
            evidence.all_attempt_latency_ms,
            50,
            references,
        ),
        _latency_observation(
            OperationsMetricId.LATENCY_P95_MS,
            evidence.all_attempt_latency_ms,
            95,
            references,
        ),
        _token_observation(
            OperationsMetricId.INPUT_TOKENS,
            evidence.input_tokens,
            evidence.provider_attempt_count,
            evidence.unknown_input_token_usage_attempt_count,
            references,
        ),
        _token_observation(
            OperationsMetricId.OUTPUT_TOKENS,
            evidence.output_tokens,
            evidence.provider_attempt_count,
            evidence.unknown_output_token_usage_attempt_count,
            references,
        ),
        _count_observation(OperationsMetricId.CACHE_HITS, evidence.cache_hits, references),
        _count_observation(
            OperationsMetricId.CACHE_ELIGIBLE_LOOKUPS,
            evidence.cache_eligible_lookups,
            references,
        ),
        _rate_observation(
            OperationsMetricId.CACHE_HIT_RATE,
            evidence.cache_hits,
            evidence.cache_eligible_lookups,
            "no-cache-eligible-lookups",
            references,
        ),
        _count_observation(OperationsMetricId.REFUSALS, evidence.refusals, references),
        _count_observation(
            OperationsMetricId.ANSWERED_REQUESTS,
            evidence.answered_requests,
            references,
        ),
        _rate_observation(
            OperationsMetricId.REFUSAL_RATE,
            evidence.refusals,
            evidence.refusals + evidence.answered_requests,
            "no-terminal-responses",
            references,
        ),
        _count_observation(
            OperationsMetricId.COMPLIANT_ANSWERS,
            evidence.compliant_answers,
            references,
        ),
        _count_observation(
            OperationsMetricId.SCORED_ANSWERS,
            evidence.scored_answers,
            references,
        ),
        _rate_observation(
            OperationsMetricId.ANSWER_COMPLIANCE_RATE,
            evidence.compliant_answers,
            evidence.scored_answers,
            "no-scored-answers",
            references,
        ),
        _cost_observation(
            OperationsMetricId.COST_PER_1000_LOGICAL_ATTEMPTS,
            evidence.total_cost,
            evidence.total_logical_requests,
            evidence.currency,
            "no-logical-attempts",
            references,
        ),
        _cost_observation(
            OperationsMetricId.COST_PER_1000_SUCCESSES,
            evidence.total_cost,
            evidence.successful_logical_requests,
            evidence.currency,
            "no-successful-attempts",
            references,
        ),
    )
    summary = OperationsSummary(
        run_id=evidence.run_id,
        configuration_id=evidence.configuration_id,
        observations=observations,
        source_artifact_ids=references,
        generated_at=evidence.generated_at,
    )
    return validate_operations_summary_v2(summary)


def validate_operations_summary_v2(
    summary: OperationsSummary | Mapping[str, object] | BaseModel,
) -> OperationsSummary:
    """Validate the fixed metric set and all cross-metric denominator semantics."""

    try:
        if isinstance(summary, OperationsSummary):
            parsed = summary
        elif isinstance(summary, BaseModel):
            parsed = OperationsSummary.model_validate(summary.model_dump(mode="json"))
        else:
            parsed = OperationsSummary.model_validate(summary)
    except (TypeError, ValueError) as error:
        raise OperationsProjectionError("operations summary violates schema v2") from error
    if not _safe_identifier(parsed.run_id) or not _safe_identifier(parsed.configuration_id):
        raise OperationsProjectionError("operations identity is not safe for publication")
    if tuple(item.metric_id for item in parsed.observations) != tuple(OPERATIONS_METRIC_ORDER):
        raise OperationsProjectionError("operations metrics are missing, extra, or out of order")
    if any(not _safe_identifier(item) for item in parsed.source_artifact_ids):
        raise OperationsProjectionError("operations source artifact identifier is unsafe")

    by_id = {item.metric_id: item for item in parsed.observations}
    for metric_id in (
        OperationsMetricId.TOTAL_LOGICAL_REQUESTS,
        OperationsMetricId.SUCCESSFUL_LOGICAL_REQUESTS,
        OperationsMetricId.CACHE_HITS,
        OperationsMetricId.CACHE_ELIGIBLE_LOOKUPS,
        OperationsMetricId.REFUSALS,
        OperationsMetricId.ANSWERED_REQUESTS,
        OperationsMetricId.COMPLIANT_ANSWERS,
        OperationsMetricId.SCORED_ANSWERS,
    ):
        _validate_count_observation(by_id[metric_id])

    _validate_token_observation(by_id[OperationsMetricId.INPUT_TOKENS])
    _validate_token_observation(by_id[OperationsMetricId.OUTPUT_TOKENS])
    if (
        by_id[OperationsMetricId.INPUT_TOKENS].denominator
        != by_id[OperationsMetricId.OUTPUT_TOKENS].denominator
    ):
        raise OperationsProjectionError("input and output token denominators differ")

    _validate_latency_pair(by_id)
    _validate_rate_from_counts(
        by_id[OperationsMetricId.CACHE_HIT_RATE],
        by_id[OperationsMetricId.CACHE_HITS],
        by_id[OperationsMetricId.CACHE_ELIGIBLE_LOOKUPS],
        "no-cache-eligible-lookups",
    )
    refusal_denominator = _count_value(by_id[OperationsMetricId.REFUSALS]) + _count_value(
        by_id[OperationsMetricId.ANSWERED_REQUESTS]
    )
    _validate_rate_values(
        by_id[OperationsMetricId.REFUSAL_RATE],
        _count_value(by_id[OperationsMetricId.REFUSALS]),
        refusal_denominator,
        "no-terminal-responses",
    )
    _validate_rate_from_counts(
        by_id[OperationsMetricId.ANSWER_COMPLIANCE_RATE],
        by_id[OperationsMetricId.COMPLIANT_ANSWERS],
        by_id[OperationsMetricId.SCORED_ANSWERS],
        "no-scored-answers",
    )
    if refusal_denominator != _count_value(by_id[OperationsMetricId.SUCCESSFUL_LOGICAL_REQUESTS]):
        raise OperationsProjectionError("terminal response counts disagree with successes")
    if _count_value(by_id[OperationsMetricId.COMPLIANT_ANSWERS]) > _count_value(
        by_id[OperationsMetricId.SCORED_ANSWERS]
    ):
        raise OperationsProjectionError("compliant answers exceed scored answers")
    if _count_value(by_id[OperationsMetricId.SCORED_ANSWERS]) > _count_value(
        by_id[OperationsMetricId.ANSWERED_REQUESTS]
    ):
        raise OperationsProjectionError("scored answers exceed answered requests")
    _validate_cost_observation(
        by_id[OperationsMetricId.COST_PER_1000_LOGICAL_ATTEMPTS],
        _count_value(by_id[OperationsMetricId.TOTAL_LOGICAL_REQUESTS]),
        "no-logical-attempts",
    )
    _validate_cost_observation(
        by_id[OperationsMetricId.COST_PER_1000_SUCCESSES],
        _count_value(by_id[OperationsMetricId.SUCCESSFUL_LOGICAL_REQUESTS]),
        "no-successful-attempts",
    )
    return parsed


def canonical_operations_json(summary: OperationsSummary | Mapping[str, object]) -> str:
    """Serialize the validated canonical record deterministically."""

    parsed = validate_operations_summary_v2(summary)
    return canonical_json_value(parsed.model_dump(mode="json"))


def render_operations_text(summary: OperationsSummary | Mapping[str, object]) -> str:
    """Render a deterministic human-readable line format from the canonical record."""

    parsed = validate_operations_summary_v2(summary)
    metadata = parsed.model_dump(mode="json", exclude={"observations"})
    lines = [
        OPERATIONS_TEXT_HEADER,
        f"renderer_version={OPERATIONS_RENDERER_VERSION}",
        f"schema_version={metadata['schema_version']}",
        f"run_id={metadata['run_id']}",
        f"configuration_id={metadata['configuration_id']}",
        f"generated_at={metadata['generated_at']}",
        f"source_artifact_ids={canonical_json_value(metadata['source_artifact_ids'])}",
        f"columns={'|'.join(OPERATIONS_TEXT_COLUMNS)}",
    ]
    lines.extend(
        "metric=" + "|".join(_observation_cells(observation)) for observation in parsed.observations
    )
    rendered = "\n".join(lines) + "\n"
    validate_operations_privacy(rendered, parsed)
    return rendered


def parse_operations_text(text: str) -> OperationsSummary:
    """Parse a TXT projection back to the canonical typed record."""

    if not isinstance(text, str) or not text or len(text.encode("utf-8")) > MAX_REPORT_BYTES:
        raise OperationsProjectionError("operations TXT size is outside allowed bounds")
    lines = text.splitlines()
    if len(lines) != 8 + len(OPERATIONS_METRIC_ORDER) or lines[0] != OPERATIONS_TEXT_HEADER:
        raise OperationsProjectionError("operations TXT structure is invalid")
    expected_keys = (
        "renderer_version",
        "schema_version",
        "run_id",
        "configuration_id",
        "generated_at",
        "source_artifact_ids",
        "columns",
    )
    metadata: dict[str, str] = {}
    for key, line in zip(expected_keys, lines[1:8], strict=True):
        prefix = f"{key}="
        if not line.startswith(prefix):
            raise OperationsProjectionError("operations TXT metadata is invalid")
        metadata[key] = line[len(prefix) :]
    if metadata["renderer_version"] != OPERATIONS_RENDERER_VERSION:
        raise OperationsProjectionError("operations TXT renderer version is unsupported")
    if metadata["columns"] != "|".join(OPERATIONS_TEXT_COLUMNS):
        raise OperationsProjectionError("operations TXT columns are invalid")
    observations = tuple(_parse_text_metric(line) for line in lines[8:])
    return _summary_from_projection(metadata, observations)


def render_operations_csv(summary: OperationsSummary | Mapping[str, object]) -> str:
    """Render RFC-4180-compatible rows with repeated immutable identity."""

    parsed = validate_operations_summary_v2(summary)
    serialized = parsed.model_dump(mode="json", exclude={"observations"})
    columns = (
        "renderer_version",
        "schema_version",
        "run_id",
        "configuration_id",
        "generated_at",
        "source_artifact_ids",
        *OPERATIONS_TEXT_COLUMNS,
    )
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    for observation in parsed.observations:
        writer.writerow(
            (
                OPERATIONS_RENDERER_VERSION,
                serialized["schema_version"],
                serialized["run_id"],
                serialized["configuration_id"],
                serialized["generated_at"],
                canonical_json_value(serialized["source_artifact_ids"]),
                *_observation_cells(observation),
            )
        )
    rendered = stream.getvalue()
    validate_operations_privacy(rendered, parsed)
    return rendered


def parse_operations_csv(text: str) -> OperationsSummary:
    """Parse a CSV projection and reject identity or row-level divergence."""

    if not isinstance(text, str) or not text or len(text.encode("utf-8")) > MAX_REPORT_BYTES:
        raise OperationsProjectionError("operations CSV size is outside allowed bounds")
    try:
        rows = list(csv.DictReader(io.StringIO(text, newline=""), strict=True))
    except (csv.Error, UnicodeError) as error:
        raise OperationsProjectionError("operations CSV cannot be parsed") from error
    expected_columns = [
        "renderer_version",
        "schema_version",
        "run_id",
        "configuration_id",
        "generated_at",
        "source_artifact_ids",
        *OPERATIONS_TEXT_COLUMNS,
    ]
    if len(rows) != len(OPERATIONS_METRIC_ORDER):
        raise OperationsProjectionError("operations CSV row count is invalid")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != expected_columns:
        raise OperationsProjectionError("operations CSV columns are invalid")
    first = rows[0]
    identity_keys = expected_columns[:6]
    if any(row.get(key) != first.get(key) for row in rows for key in identity_keys):
        raise OperationsProjectionError("operations CSV identities differ between rows")
    if first["renderer_version"] != OPERATIONS_RENDERER_VERSION:
        raise OperationsProjectionError("operations CSV renderer version is unsupported")
    observations = tuple(_observation_from_cells(row) for row in rows)
    metadata = {key: cast(str, first[key]) for key in identity_keys}
    return _summary_from_projection(metadata, observations)


def verify_operations_parity(
    summary: OperationsSummary | Mapping[str, object],
    text_projection: str,
    csv_projection: str,
) -> None:
    """Prove both projections reconstruct the exact same canonical record."""

    canonical = validate_operations_summary_v2(summary)
    if parse_operations_text(text_projection) != canonical:
        raise OperationsProjectionError("operations TXT differs from canonical evidence")
    if parse_operations_csv(csv_projection) != canonical:
        raise OperationsProjectionError("operations CSV differs from canonical evidence")
    if text_projection != render_operations_text(canonical):
        raise OperationsProjectionError("operations TXT bytes are not canonical")
    if csv_projection != render_operations_csv(canonical):
        raise OperationsProjectionError("operations CSV bytes are not canonical")


def validate_operations_privacy(
    projection: str,
    summary: OperationsSummary,
    *,
    redactor: Redactor = DEFAULT_REDACTOR,
) -> None:
    """Fail closed on content fields, paths, secrets, or supported PII in safe strings."""

    if _PROHIBITED_OUTPUT_FIELD.search(projection) or _ABSOLUTE_PATH.search(projection):
        raise OperationsProjectionError("operations projection contains prohibited content")
    values = (
        summary.run_id,
        summary.configuration_id,
        *summary.source_artifact_ids,
        *(item.metric_id for item in summary.observations),
        *(item.unit for item in summary.observations),
        *(reference for item in summary.observations for reference in item.evidence_references),
    )
    try:
        if any(redactor.detect(value) for value in values):
            raise OperationsProjectionError("operations projection contains sensitive evidence")
    except RedactionError as error:
        raise OperationsProjectionError("operations privacy validation failed closed") from error


def write_operations_text(
    summary: OperationsSummary | Mapping[str, object],
    path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    target = Path(path)
    if target.suffix.casefold() != ".txt":
        raise ReportWriteError("operations text path must end in .txt")
    return _atomic_write_text(target, render_operations_text(summary), overwrite=overwrite)


def write_operations_csv(
    summary: OperationsSummary | Mapping[str, object],
    path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    target = Path(path)
    if target.suffix.casefold() != ".csv":
        raise ReportWriteError("operations CSV path must end in .csv")
    return _atomic_write_text(target, render_operations_csv(summary), overwrite=overwrite)


def _count_observation(
    metric_id: OperationsMetricId,
    count: int,
    references: tuple[str, ...],
    *,
    unit: str = "count",
) -> MetricObservation:
    return MetricObservation(
        metric_id=metric_id,
        unit=unit,
        value=float(count),
        numerator=float(count),
        denominator=1,
        eligible=True,
        scorer_version=OPERATIONS_SCORER_VERSION,
        status=MetricObservationStatus.OBSERVED,
        evidence_references=references,
    )


def _token_observation(
    metric_id: OperationsMetricId,
    count: int | None,
    provider_attempt_count: int,
    unknown_usage_attempt_count: int,
    references: tuple[str, ...],
) -> MetricObservation:
    if provider_attempt_count == 0:
        return _unavailable_observation(
            metric_id,
            "tokens",
            "no-provider-attempts",
            references,
        )
    if unknown_usage_attempt_count:
        reason = f"{metric_id.value}-usage-incomplete"
        return MetricObservation(
            metric_id=metric_id,
            unit="tokens",
            value=UnavailableValue(reason=reason),
            numerator=(UnavailableValue(reason=reason) if count is None else float(count)),
            denominator=provider_attempt_count,
            eligible=True,
            scorer_version=OPERATIONS_SCORER_VERSION,
            status=MetricObservationStatus.UNAVAILABLE,
            evidence_references=references,
        )
    if count is None:
        raise OperationsProjectionError("complete token evidence is missing its total")
    return MetricObservation(
        metric_id=metric_id,
        unit="tokens",
        value=float(count),
        numerator=float(count),
        denominator=provider_attempt_count,
        eligible=True,
        scorer_version=OPERATIONS_SCORER_VERSION,
        status=MetricObservationStatus.OBSERVED,
        evidence_references=references,
    )


def _latency_observation(
    metric_id: OperationsMetricId,
    values: Sequence[float],
    percentile: int,
    references: tuple[str, ...],
) -> MetricObservation:
    if not values:
        return _unavailable_observation(
            metric_id,
            "milliseconds",
            "no-latency-attempts",
            references,
        )
    observed = nearest_rank_percentile(values, percentile)
    return MetricObservation(
        metric_id=metric_id,
        unit="milliseconds",
        value=observed,
        numerator=observed,
        denominator=len(values),
        eligible=True,
        scorer_version=OPERATIONS_SCORER_VERSION,
        status=MetricObservationStatus.OBSERVED,
        evidence_references=references,
    )


def _rate_observation(
    metric_id: OperationsMetricId,
    numerator: int,
    denominator: int,
    unavailable_reason: str,
    references: tuple[str, ...],
) -> MetricObservation:
    if denominator == 0:
        return _unavailable_observation(metric_id, "ratio", unavailable_reason, references)
    return MetricObservation(
        metric_id=metric_id,
        unit="ratio",
        value=numerator / denominator,
        numerator=float(numerator),
        denominator=denominator,
        eligible=True,
        scorer_version=OPERATIONS_SCORER_VERSION,
        status=MetricObservationStatus.OBSERVED,
        evidence_references=references,
    )


def _cost_observation(
    metric_id: OperationsMetricId,
    total_cost: Decimal | None,
    denominator: int,
    currency: str,
    zero_reason: str,
    references: tuple[str, ...],
) -> MetricObservation:
    denominator_label = (
        "logical-attempts"
        if metric_id is OperationsMetricId.COST_PER_1000_LOGICAL_ATTEMPTS
        else "successes"
    )
    unit = f"{currency}-per-1000-{denominator_label}"
    if denominator == 0:
        return _unavailable_observation(metric_id, unit, zero_reason, references)
    if total_cost is None:
        unavailable = UnavailableValue(reason="cost-incomplete")
        return MetricObservation(
            metric_id=metric_id,
            unit=unit,
            value=unavailable,
            numerator=unavailable,
            denominator=denominator,
            eligible=True,
            scorer_version=OPERATIONS_SCORER_VERSION,
            status=MetricObservationStatus.UNAVAILABLE,
            evidence_references=references,
        )
    return MetricObservation(
        metric_id=metric_id,
        unit=unit,
        value=float(total_cost * Decimal(1_000) / Decimal(denominator)),
        numerator=float(total_cost),
        denominator=denominator,
        eligible=True,
        scorer_version=OPERATIONS_SCORER_VERSION,
        status=MetricObservationStatus.OBSERVED,
        evidence_references=references,
    )


def _unavailable_observation(
    metric_id: OperationsMetricId,
    unit: str,
    reason: str,
    references: tuple[str, ...],
) -> MetricObservation:
    unavailable = UnavailableValue(reason=reason)
    return MetricObservation(
        metric_id=metric_id,
        unit=unit,
        value=unavailable,
        numerator=unavailable,
        denominator=unavailable,
        eligible=False,
        scorer_version=OPERATIONS_SCORER_VERSION,
        status=MetricObservationStatus.UNAVAILABLE,
        evidence_references=references,
    )


def _validate_count_observation(observation: MetricObservation) -> None:
    if (
        observation.unit != "count"
        or not observation.eligible
        or observation.status is not MetricObservationStatus.OBSERVED
        or observation.threshold is not None
        or observation.operator is not None
        or observation.scorer_version != OPERATIONS_SCORER_VERSION
        or observation.denominator != 1
        or isinstance(observation.value, UnavailableValue)
        or isinstance(observation.numerator, UnavailableValue)
        or not observation.value.is_integer()
        or observation.value < 0
        or observation.numerator != observation.value
    ):
        raise OperationsProjectionError("operations count observation is inconsistent")


def _validate_token_observation(observation: MetricObservation) -> None:
    if isinstance(observation.value, UnavailableValue):
        no_attempts = (
            observation.value.reason == "no-provider-attempts"
            and isinstance(observation.denominator, UnavailableValue)
            and not observation.eligible
        )
        incomplete_usage = (
            observation.value.reason == f"{observation.metric_id}-usage-incomplete"
            and isinstance(observation.denominator, int)
            and observation.denominator > 0
            and observation.eligible
            and (
                isinstance(observation.numerator, UnavailableValue)
                or (observation.numerator >= 0 and observation.numerator.is_integer())
            )
        )
        if not (
            observation.unit == "tokens"
            and observation.status is MetricObservationStatus.UNAVAILABLE
            and observation.scorer_version == OPERATIONS_SCORER_VERSION
            and observation.threshold is None
            and observation.operator is None
            and (no_attempts or incomplete_usage)
        ):
            raise OperationsProjectionError("token availability is inconsistent")
        return
    if (
        observation.unit != "tokens"
        or not observation.eligible
        or observation.status is not MetricObservationStatus.OBSERVED
        or observation.threshold is not None
        or observation.operator is not None
        or observation.scorer_version != OPERATIONS_SCORER_VERSION
        or not isinstance(observation.denominator, int)
        or observation.denominator <= 0
        or isinstance(observation.numerator, UnavailableValue)
        or not observation.value.is_integer()
        or observation.value < 0
        or observation.numerator != observation.value
    ):
        raise OperationsProjectionError("token observation is inconsistent")


def _count_value(observation: MetricObservation) -> int:
    _validate_count_observation(observation)
    return int(cast(float, observation.value))


def _validate_latency_pair(by_id: Mapping[str, MetricObservation]) -> None:
    p50 = by_id[OperationsMetricId.LATENCY_P50_MS]
    p95 = by_id[OperationsMetricId.LATENCY_P95_MS]
    if isinstance(p50.value, UnavailableValue) or isinstance(p95.value, UnavailableValue):
        if not (
            isinstance(p50.value, UnavailableValue)
            and isinstance(p95.value, UnavailableValue)
            and p50.value.reason == p95.value.reason == "no-latency-attempts"
        ):
            raise OperationsProjectionError("latency availability differs between percentiles")
        return
    if (
        p50.unit != p95.unit
        or p50.unit != "milliseconds"
        or p50.denominator != p95.denominator
        or not isinstance(p50.denominator, int)
        or p50.denominator <= 0
        or p50.value > p95.value
    ):
        raise OperationsProjectionError("operations latency observations are inconsistent")


def _validate_rate_from_counts(
    observation: MetricObservation,
    numerator_observation: MetricObservation,
    denominator_observation: MetricObservation,
    unavailable_reason: str,
) -> None:
    _validate_rate_values(
        observation,
        _count_value(numerator_observation),
        _count_value(denominator_observation),
        unavailable_reason,
    )


def _validate_rate_values(
    observation: MetricObservation,
    numerator: int,
    denominator: int,
    unavailable_reason: str,
) -> None:
    if denominator == 0:
        if not (
            isinstance(observation.value, UnavailableValue)
            and observation.value.reason == unavailable_reason
            and not observation.eligible
            and observation.status is MetricObservationStatus.UNAVAILABLE
        ):
            raise OperationsProjectionError("zero-denominator rate is not unavailable")
        return
    if (
        observation.unit != "ratio"
        or not observation.eligible
        or observation.status is not MetricObservationStatus.OBSERVED
        or isinstance(observation.value, UnavailableValue)
        or isinstance(observation.numerator, UnavailableValue)
        or observation.denominator != denominator
        or observation.numerator != numerator
        or not math.isclose(observation.value, numerator / denominator, rel_tol=0, abs_tol=1e-15)
    ):
        raise OperationsProjectionError("operations rate disagrees with its counts")


def _validate_cost_observation(
    observation: MetricObservation,
    denominator: int,
    zero_reason: str,
) -> None:
    expected_suffix = (
        "-per-1000-logical-attempts"
        if observation.metric_id == OperationsMetricId.COST_PER_1000_LOGICAL_ATTEMPTS
        else "-per-1000-successes"
    )
    if not observation.unit.endswith(expected_suffix):
        raise OperationsProjectionError("normalized cost unit is inconsistent")
    if denominator == 0:
        if not (
            isinstance(observation.value, UnavailableValue)
            and observation.value.reason == zero_reason
            and not observation.eligible
        ):
            raise OperationsProjectionError("zero-denominator cost is not unavailable")
        return
    if isinstance(observation.value, UnavailableValue):
        if not (
            observation.value.reason == "cost-incomplete"
            and observation.eligible
            and observation.status is MetricObservationStatus.UNAVAILABLE
            and observation.denominator == denominator
            and isinstance(observation.numerator, UnavailableValue)
            and observation.numerator.reason == "cost-incomplete"
        ):
            raise OperationsProjectionError("unknown cost is not explicitly unavailable")
        return
    if (
        observation.denominator != denominator
        or isinstance(observation.numerator, UnavailableValue)
        or not math.isclose(
            observation.value,
            observation.numerator * 1_000 / denominator,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise OperationsProjectionError("normalized cost disagrees with its denominator")


def _observation_cells(observation: MetricObservation) -> tuple[str, ...]:
    payload = observation.model_dump(mode="json")
    return (
        cast(str, payload["metric_id"]),
        cast(str, payload["unit"]),
        cast(str, payload["status"]),
        "true" if payload["eligible"] is True else "false",
        _projection_value(payload["value"]),
        _projection_value(payload["numerator"]),
        _projection_value(payload["denominator"]),
        _projection_value(payload["threshold"]),
        _projection_value(payload["operator"]),
        _projection_value(payload["scorer_version"]),
        canonical_json_value(payload["evidence_references"]),
    )


def _projection_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, dict) and value.get("status") == "unavailable":
        return f"unavailable:{value.get('reason', '')}"
    if isinstance(value, str):
        return value
    return canonical_json_value(value)


def _parse_text_metric(line: str) -> MetricObservation:
    if not line.startswith("metric="):
        raise OperationsProjectionError("operations TXT metric row is invalid")
    cells = line[len("metric=") :].split("|")
    if len(cells) != len(OPERATIONS_TEXT_COLUMNS):
        raise OperationsProjectionError("operations TXT metric column count is invalid")
    return _observation_from_cells(dict(zip(OPERATIONS_TEXT_COLUMNS, cells, strict=True)))


def _observation_from_cells(cells: Mapping[str, str | None]) -> MetricObservation:
    try:
        eligible = {"true": True, "false": False}[cast(str, cells["eligible"])]
        evidence = decode_json_report(cast(str, cells["evidence_references"]))
        if not isinstance(evidence, list) or any(not isinstance(item, str) for item in evidence):
            raise ValueError
        return MetricObservation(
            metric_id=cast(str, cells["metric_id"]),
            unit=cast(str, cells["unit"]),
            status=MetricObservationStatus(cast(str, cells["status"])),
            eligible=eligible,
            value=_parse_projection_value(cast(str, cells["value"]), numeric_kind="float"),
            numerator=_parse_projection_value(cast(str, cells["numerator"]), numeric_kind="float"),
            denominator=_parse_projection_value(
                cast(str, cells["denominator"]), numeric_kind="int"
            ),
            threshold=_parse_nullable_float(cast(str, cells["threshold"])),
            operator=_parse_operator(cast(str, cells["operator"])),
            scorer_version=_parse_projection_value(
                cast(str, cells["scorer_version"]), numeric_kind="string"
            ),
            evidence_references=tuple(cast(list[str], evidence)),
        )
    except (KeyError, TypeError, ValueError, ReportSerializationError) as error:
        raise OperationsProjectionError("operations metric row is invalid") from error


def _parse_projection_value(
    value: str,
    *,
    numeric_kind: str,
) -> float | int | str | UnavailableValue:
    if value.startswith("unavailable:"):
        return UnavailableValue(reason=value.removeprefix("unavailable:"))
    if numeric_kind == "string":
        return value
    decoded = decode_json_report(value)
    if numeric_kind == "int":
        if type(decoded) is not int:
            raise ValueError
        return decoded
    if isinstance(decoded, bool) or not isinstance(decoded, int | float):
        raise ValueError
    return float(decoded)


def _parse_nullable_float(value: str) -> float | None:
    if value == "null":
        return None
    decoded = decode_json_report(value)
    if isinstance(decoded, bool) or not isinstance(decoded, int | float):
        raise ValueError
    return float(decoded)


def _parse_operator(value: str) -> EvidenceComparisonOperator | None:
    return None if value == "null" else EvidenceComparisonOperator(value)


def _summary_from_projection(
    metadata: Mapping[str, str],
    observations: tuple[MetricObservation, ...],
) -> OperationsSummary:
    try:
        raw_references = decode_json_report(metadata["source_artifact_ids"])
        if not isinstance(raw_references, list) or any(
            not isinstance(item, str) for item in raw_references
        ):
            raise ValueError
        summary = OperationsSummary(
            schema_version=metadata["schema_version"],
            run_id=metadata["run_id"],
            configuration_id=metadata["configuration_id"],
            observations=observations,
            source_artifact_ids=tuple(cast(list[str], raw_references)),
            generated_at=metadata["generated_at"],
        )
    except (KeyError, TypeError, ValueError, ReportSerializationError) as error:
        raise OperationsProjectionError("operations projection metadata is invalid") from error
    return validate_operations_summary_v2(summary)


def _safe_identifier(value: str) -> bool:
    return _SAFE_PUBLIC_IDENTIFIER.fullmatch(value) is not None


__all__ = [
    "OPERATIONS_METRIC_ORDER",
    "OPERATIONS_RENDERER_VERSION",
    "OPERATIONS_SCORER_VERSION",
    "OperationsEvidenceInputV2",
    "OperationsMetricId",
    "OperationsProjectionError",
    "build_operations_summary_v2",
    "canonical_operations_json",
    "parse_operations_csv",
    "parse_operations_text",
    "render_operations_csv",
    "render_operations_text",
    "validate_operations_privacy",
    "validate_operations_summary_v2",
    "verify_operations_parity",
    "write_operations_csv",
    "write_operations_text",
]
