from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from rag_mvp.domain import MetricObservation, OperationsSummary, UnavailableValue
from rag_mvp.evaluation.json_report import ReportWriteError
from rag_mvp.evaluation.operations_v2 import (
    OPERATIONS_METRIC_ORDER,
    OperationsEvidenceInputV2,
    OperationsMetricId,
    OperationsProjectionError,
    build_operations_summary_v2,
    parse_operations_csv,
    parse_operations_text,
    render_operations_csv,
    render_operations_text,
    validate_operations_summary_v2,
    verify_operations_parity,
    write_operations_csv,
    write_operations_text,
)

_NOW = datetime(2026, 8, 7, 3, 4, 5, tzinfo=UTC)


def _evidence(**updates: object) -> OperationsEvidenceInputV2:
    values: dict[str, object] = {
        "run_id": "acceptance-run-v2",
        "configuration_id": "configuration-v2",
        "total_logical_requests": 4,
        "successful_logical_requests": 3,
        "all_attempt_latency_ms": (10.0, 20.0, 30.0, 40.0),
        "provider_attempt_count": 3,
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_hits": 2,
        "cache_eligible_lookups": 3,
        "refusals": 1,
        "answered_requests": 2,
        "compliant_answers": 2,
        "scored_answers": 2,
        "total_cost": Decimal("0.03"),
        "currency": "USD",
        "source_artifact_ids": ("attempt-ledger",),
        "generated_at": _NOW,
    }
    values.update(updates)
    return OperationsEvidenceInputV2.model_validate(values)


def _observation(summary: OperationsSummary, metric_id: OperationsMetricId) -> MetricObservation:
    parsed = validate_operations_summary_v2(summary)
    return next(item for item in parsed.observations if item.metric_id == metric_id)


def test_builder_preserves_declared_denominators_units_and_nearest_rank_latency() -> None:
    summary = build_operations_summary_v2(_evidence())

    assert tuple(item.metric_id for item in summary.observations) == OPERATIONS_METRIC_ORDER
    assert _observation(summary, OperationsMetricId.LATENCY_P50_MS).value == 20.0
    assert _observation(summary, OperationsMetricId.LATENCY_P95_MS).value == 40.0
    assert _observation(summary, OperationsMetricId.INPUT_TOKENS).denominator == 3
    assert _observation(summary, OperationsMetricId.OUTPUT_TOKENS).denominator == 3
    assert _observation(summary, OperationsMetricId.CACHE_HIT_RATE).value == pytest.approx(2 / 3)
    assert _observation(summary, OperationsMetricId.REFUSAL_RATE).value == pytest.approx(1 / 3)
    logical_cost = _observation(summary, OperationsMetricId.COST_PER_1000_LOGICAL_ATTEMPTS)
    success_cost = _observation(summary, OperationsMetricId.COST_PER_1000_SUCCESSES)
    assert (logical_cost.unit, logical_cost.denominator, logical_cost.value) == (
        "USD-per-1000-logical-attempts",
        4,
        7.5,
    )
    assert (success_cost.unit, success_cost.denominator, success_cost.value) == (
        "USD-per-1000-successes",
        3,
        10.0,
    )


def test_zero_denominators_are_explicitly_unavailable_without_losing_counts() -> None:
    summary = build_operations_summary_v2(
        _evidence(
            total_logical_requests=0,
            successful_logical_requests=0,
            all_attempt_latency_ms=(),
            provider_attempt_count=0,
            input_tokens=0,
            output_tokens=0,
            cache_hits=0,
            cache_eligible_lookups=0,
            refusals=0,
            answered_requests=0,
            compliant_answers=0,
            scored_answers=0,
            total_cost=None,
        )
    )

    for metric_id in (
        OperationsMetricId.LATENCY_P50_MS,
        OperationsMetricId.LATENCY_P95_MS,
        OperationsMetricId.INPUT_TOKENS,
        OperationsMetricId.OUTPUT_TOKENS,
        OperationsMetricId.CACHE_HIT_RATE,
        OperationsMetricId.REFUSAL_RATE,
        OperationsMetricId.ANSWER_COMPLIANCE_RATE,
        OperationsMetricId.COST_PER_1000_LOGICAL_ATTEMPTS,
        OperationsMetricId.COST_PER_1000_SUCCESSES,
    ):
        observation = _observation(summary, metric_id)
        assert observation.eligible is False
        assert isinstance(observation.value, UnavailableValue)
    assert _observation(summary, OperationsMetricId.CACHE_HITS).value == 0
    assert _observation(summary, OperationsMetricId.CACHE_ELIGIBLE_LOOKUPS).value == 0


def test_unknown_cost_is_not_coerced_to_zero() -> None:
    summary = build_operations_summary_v2(_evidence(total_cost=None))

    for metric_id, denominator in (
        (OperationsMetricId.COST_PER_1000_LOGICAL_ATTEMPTS, 4),
        (OperationsMetricId.COST_PER_1000_SUCCESSES, 3),
    ):
        observation = _observation(summary, metric_id)
        assert isinstance(observation.value, UnavailableValue)
        assert observation.value.reason == "cost-incomplete"
        assert observation.eligible is True
        assert observation.denominator == denominator


def test_incomplete_provider_usage_is_unavailable_with_real_attempt_denominator() -> None:
    summary = build_operations_summary_v2(
        _evidence(
            input_tokens=100,
            output_tokens=None,
            unknown_input_token_usage_attempt_count=1,
            unknown_output_token_usage_attempt_count=2,
        )
    )

    input_observation = _observation(summary, OperationsMetricId.INPUT_TOKENS)
    output_observation = _observation(summary, OperationsMetricId.OUTPUT_TOKENS)
    assert isinstance(input_observation.value, UnavailableValue)
    assert input_observation.numerator == 100
    assert input_observation.denominator == 3
    assert input_observation.eligible is True
    assert isinstance(output_observation.value, UnavailableValue)
    assert isinstance(output_observation.numerator, UnavailableValue)
    assert output_observation.denominator == 3
    assert output_observation.eligible is True


def test_txt_and_csv_round_trip_to_one_canonical_record() -> None:
    summary = build_operations_summary_v2(_evidence())
    text = render_operations_text(summary)
    csv_text = render_operations_csv(summary)

    assert parse_operations_text(text) == summary
    assert parse_operations_csv(csv_text) == summary
    verify_operations_parity(summary, text, csv_text)


@pytest.mark.parametrize("projection", ["text", "csv"])
def test_parseable_noncanonical_projection_bytes_fail_parity(projection: str) -> None:
    summary = build_operations_summary_v2(_evidence())
    text = render_operations_text(summary)
    csv_text = render_operations_csv(summary)
    if projection == "text":
        text = text.removesuffix("\n")
        assert parse_operations_text(text) == summary
    else:
        csv_text = csv_text.replace("\n", "\r\n")
        assert parse_operations_csv(csv_text) == summary

    with pytest.raises(OperationsProjectionError, match="bytes are not canonical"):
        verify_operations_parity(summary, text, csv_text)


def test_semantic_validation_rejects_false_token_denominator() -> None:
    raw = build_operations_summary_v2(_evidence()).model_dump(mode="json")
    output = next(
        item
        for item in raw["observations"]
        if item["metric_id"] == OperationsMetricId.OUTPUT_TOKENS
    )
    output["denominator"] = 4

    with pytest.raises(OperationsProjectionError, match="token denominators differ"):
        validate_operations_summary_v2(raw)


def test_semantic_validation_rejects_success_cost_with_attempt_unit() -> None:
    raw = build_operations_summary_v2(_evidence()).model_dump(mode="json")
    success_cost = next(
        item
        for item in raw["observations"]
        if item["metric_id"] == OperationsMetricId.COST_PER_1000_SUCCESSES
    )
    success_cost["unit"] = "USD-per-1000-logical-attempts"

    with pytest.raises(OperationsProjectionError, match="cost unit"):
        validate_operations_summary_v2(raw)


def test_provider_tokens_require_a_real_attempt_denominator() -> None:
    with pytest.raises(ValueError, match="provider-attempt denominator"):
        _evidence(provider_attempt_count=0, input_tokens=1)


def test_privacy_validation_rejects_a_secret_in_evidence_identity() -> None:
    summary = build_operations_summary_v2(_evidence(source_artifact_ids=("AKIAABCDEFGHIJKLMNOP",)))

    with pytest.raises(OperationsProjectionError, match="sensitive evidence"):
        render_operations_text(summary)


def test_writers_are_deterministic_and_no_overwrite(tmp_path: Path) -> None:
    directory = tmp_path
    summary = build_operations_summary_v2(_evidence())
    text_path = directory / "operations.txt"
    csv_path = directory / "operations.csv"

    assert write_operations_text(summary, text_path).read_text(encoding="utf-8") == (
        render_operations_text(summary)
    )
    assert write_operations_csv(summary, csv_path).read_text(encoding="utf-8") == (
        render_operations_csv(summary)
    )
    with pytest.raises(ReportWriteError):
        write_operations_text(summary, text_path)
    with pytest.raises(ReportWriteError):
        write_operations_csv(summary, csv_path)
