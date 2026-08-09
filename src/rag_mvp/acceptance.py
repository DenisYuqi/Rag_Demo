"""Build and verify the lightweight original-PDF Release v2 submission."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import cast

import httpx

from rag_mvp.performance.load_report import LoadAcceptanceThresholds
from rag_mvp.performance.load_test import (
    HttpLoadTestConfig,
    HttpLoadTestHarness,
    LoadScenario,
)
from rag_mvp.safety.models import SensitiveKind
from rag_mvp.safety.redactor import DEFAULT_REDACTOR

SCHEMA_VERSION = "rag-mvp-release-v2"
CONFIG_SCHEMA_VERSION = "mvp-selected-configuration-v1"
PERFORMANCE_SAMPLE_SIZE = 100
REQUIRED_CONCURRENCY = 5
MINIMUM_WITHIN_TEN_SECONDS = 90
LATENCY_LIMIT_MS = 10_000.0
_MAX_INPUT_BYTES = 20_000_000
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_RELEASE_FILE_NAME = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
_ABSOLUTE_PATH = re.compile(
    r"(?:(?<![A-Za-z])[A-Za-z]:[\\/]|\\\\|"
    r"/(?:home|Users|tmp|var|etc|opt|root|mnt|workspace)/)"
)
_DECIMAL_LITERAL = re.compile(r"(?<![\w])[-+]?\d+\.\d+(?![\w])")
_QUALITY_THRESHOLDS = {
    "faithfulness": Decimal("0.85"),
    "context-precision": Decimal("0.70"),
    "answer-compliance": Decimal("0.90"),
    "style-consistency": Decimal("0.85"),
    "refusal-appropriateness": Decimal("0.90"),
}
_REQUIRED_FILES = (
    "summary.md",
    "summary.json",
    "selected-configuration.json",
    "operations-cost.md",
    "operations-cost.json",
    "pdf-crosswalk.md",
    "pdf-crosswalk.json",
    "limitations.md",
    "implementation-evidence.md",
    "reproduce.md",
    "evidence/quality-report.json",
    "evidence/model-comparison.json",
    "evidence/retrieval-reranker-comparison.json",
    "evidence/cache-comparison.json",
    "evidence/performance-100.json",
    "evidence/issue-diagnosis.json",
    "evidence/issue-diagnosis.md",
    "evidence/structured-log-field-dictionary.json",
    "evidence/structured-log-sample.jsonl",
)


class AcceptanceError(ValueError):
    """Privacy-safe acceptance failure suitable for terminal output."""


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise AcceptanceError(f"{label}_file_missing")
    if path.stat().st_size <= 0 or path.stat().st_size > _MAX_INPUT_BYTES:
        raise AcceptanceError(f"{label}_file_size_invalid")
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise AcceptanceError(f"{label}_json_invalid") from None
    if not isinstance(decoded, dict):
        raise AcceptanceError(f"{label}_json_invalid")
    return cast(dict[str, object], decoded)


def _load_json_list(path: Path, *, label: str) -> list[object]:
    if path.is_symlink() or not path.is_file():
        raise AcceptanceError(f"{label}_file_missing")
    if path.stat().st_size <= 0 or path.stat().st_size > _MAX_INPUT_BYTES:
        raise AcceptanceError(f"{label}_file_size_invalid")
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise AcceptanceError(f"{label}_json_invalid") from None
    if not isinstance(decoded, list):
        raise AcceptanceError(f"{label}_json_invalid")
    return cast(list[object], decoded)


def _validate_support_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise AcceptanceError(f"{label}_file_missing")
    if path.stat().st_size <= 0 or path.stat().st_size > _MAX_INPUT_BYTES:
        raise AcceptanceError(f"{label}_file_size_invalid")
    return path


def _validate_release_relative_name(relative: str) -> None:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or any(_RELEASE_FILE_NAME.fullmatch(part) is None for part in pure.parts)
    ):
        raise AcceptanceError(f"packaged_filename_invalid:{relative}")


def _mapping(value: object, *, error: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise AcceptanceError(error)
    return cast(dict[str, object], value)


def _text(value: object, *, error: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AcceptanceError(error)
    return value.strip()


def _decimal(value: object, *, error: str, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise AcceptanceError(error)
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise AcceptanceError(error) from None
    if not result.is_finite() or result < 0 or (positive and result <= 0):
        raise AcceptanceError(error)
    return result


def _integer(value: object, *, error: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise AcceptanceError(error)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _canonical_json(value: object) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical_json(value), encoding="utf-8", newline="\n")


def _nested(value: Mapping[str, object], *keys: str) -> object | None:
    current: object = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _first(value: Mapping[str, object], paths: Sequence[tuple[str, ...]]) -> object | None:
    for path in paths:
        found = _nested(value, *path)
        if found is not None:
            return found
    return None


def _validate_selected_configuration(payload: Mapping[str, object]) -> None:
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise AcceptanceError("selected_configuration_schema_invalid")
    for key in ("configuration_id", "profile", "retrieval_mode", "dataset_id"):
        _text(payload.get(key), error=f"selected_configuration_{key}_invalid")
    runtime_configuration_id = payload.get("runtime_configuration_id")
    if runtime_configuration_id is not None:
        _text(
            runtime_configuration_id,
            error="selected_configuration_runtime_configuration_id_invalid",
        )
    source_configuration_ids = payload.get("source_configuration_ids")
    if source_configuration_ids is not None:
        source_ids = _mapping(
            source_configuration_ids,
            error="selected_configuration_source_ids_invalid",
        )
        for label, value in source_ids.items():
            if label not in {"quality", "model", "retrieval", "cache"}:
                raise AcceptanceError("selected_configuration_source_ids_invalid")
            _text(value, error=f"selected_configuration_{label}_source_id_invalid")
    comparison_compatibility = payload.get("comparison_compatibility")
    if comparison_compatibility is not None:
        declarations = _mapping(
            comparison_compatibility,
            error="selected_configuration_comparison_compatibility_invalid",
        )
        for label, value in declarations.items():
            if label not in {"model", "retrieval", "cache"}:
                raise AcceptanceError("selected_configuration_comparison_compatibility_invalid")
            declaration = _mapping(
                value,
                error=f"selected_configuration_{label}_compatibility_invalid",
            )
            status = _text(
                declaration.get("status"),
                error=f"selected_configuration_{label}_compatibility_invalid",
            )
            if status not in {"exact", "partial"}:
                raise AcceptanceError(f"selected_configuration_{label}_compatibility_invalid")
            if status == "partial":
                _text(
                    declaration.get("limitation"),
                    error=f"selected_configuration_{label}_limitation_invalid",
                )
    _text(payload.get("scorer_contract"), error="selected_configuration_scorer_invalid")
    models = _mapping(payload.get("models"), error="selected_configuration_models_invalid")
    for key in ("embedding", "reranking", "generation"):
        _text(models.get(key), error=f"selected_configuration_{key}_model_invalid")
    _text(
        payload.get("parent_child_strategy"),
        error="selected_configuration_parent_child_invalid",
    )
    rationale = _mapping(
        payload.get("selection_rationale"),
        error="selected_configuration_rationale_invalid",
    )
    for key in ("model", "retrieval", "reranker", "cache"):
        _text(rationale.get(key), error=f"selected_configuration_{key}_rationale_invalid")
    cost = _mapping(payload.get("cost"), error="selected_configuration_cost_invalid")
    remote = _mapping(cost.get("remote"), error="selected_configuration_remote_cost_invalid")
    local = _mapping(cost.get("local"), error="selected_configuration_local_cost_invalid")
    if _text(remote.get("currency"), error="remote_currency_invalid") != _text(
        local.get("currency"), error="local_currency_invalid"
    ):
        raise AcceptanceError("cost_currency_mismatch")
    for key in ("input_per_million", "output_per_million"):
        _decimal(remote.get(key), error=f"remote_{key}_invalid", positive=True)
    for key in ("maximum_input_tokens", "maximum_output_tokens"):
        _integer(remote.get(key), error=f"remote_{key}_invalid", minimum=1)
    _text(remote.get("pricing_source"), error="remote_pricing_source_invalid")
    _text(local.get("hardware"), error="local_hardware_invalid")
    _decimal(local.get("hourly_rate"), error="local_hourly_rate_invalid", positive=True)
    duration_source = local.get("allocation_duration_source", "declared")
    if duration_source not in {"declared", "measured-load-duration"}:
        raise AcceptanceError("local_duration_source_invalid")
    _decimal(
        local.get("allocation_duration_seconds"),
        error="local_duration_invalid",
        positive=True,
    )
    limitations = payload.get("limitations")
    if (
        not isinstance(limitations, list)
        or not limitations
        or not all(isinstance(item, str) and item.strip() for item in limitations)
    ):
        raise AcceptanceError("selected_configuration_limitations_invalid")


def _source_identity(payload: Mapping[str, object]) -> str:
    value = _first(
        payload,
        (("report_id",), ("comparison_id",), ("run_id",), ("metadata", "report_id")),
    )
    return _text(value, error="source_report_identity_missing")


def _source_metadata(payload: Mapping[str, object]) -> dict[str, str]:
    candidates = {
        "configuration_id": (
            ("configuration_id",),
            ("metadata", "configuration_id"),
            ("configuration", "configuration_id"),
            ("provenance", "configuration_id"),
        ),
        "profile": (
            ("profile",),
            ("metadata", "profile"),
            ("selected_profile",),
            ("provenance", "retrieval_profile"),
        ),
        "retrieval_mode": (
            ("retrieval_mode",),
            ("metadata", "retrieval_mode"),
            ("provenance", "retrieval_configuration", "mode"),
        ),
        "dataset_id": (
            ("dataset_id",),
            ("metadata", "dataset_id"),
            ("provenance", "dataset_id"),
            ("provenance", "dataset", "id"),
            ("plan", "fixed_identities", "dataset_id"),
        ),
        "scorer_contract": (
            ("scorer_contract",),
            ("metadata", "scorer_contract"),
            ("provenance", "scoring_version"),
            ("acceptance_contract", "gate_profile_version"),
        ),
    }
    result: dict[str, str] = {}
    for name, paths in candidates.items():
        found = _first(payload, paths)
        if found is not None:
            result[name] = _text(found, error=f"source_{name}_invalid")
    return result


def _source_complete(payload: Mapping[str, object]) -> bool:
    status = _first(payload, (("status",), ("metadata", "status"), ("selection", "status")))
    if isinstance(status, str) and status.casefold() in {
        "accepted",
        "complete",
        "completed",
        "passed",
    }:
        return True
    if payload.get("accepted") is True:
        return True
    if _nested(payload, "recommendation", "state") == "recommended":
        return True
    return _nested(payload, "gate", "final_passed") is True


def _comparison_declaration(selected: Mapping[str, object], label: str) -> tuple[str, str | None]:
    raw = selected.get("comparison_compatibility")
    if not isinstance(raw, Mapping):
        return "exact", None
    declaration = raw.get(label)
    if not isinstance(declaration, Mapping):
        return "exact", None
    status = declaration.get("status")
    limitation = declaration.get("limitation")
    return (
        status if isinstance(status, str) else "exact",
        limitation if isinstance(limitation, str) else None,
    )


def _partial_comparison_complete(payload: Mapping[str, object]) -> bool:
    candidates = payload.get("candidates")
    return (
        payload.get("schema_version") == "comparison-result-v1"
        and isinstance(payload.get("completed_at"), str)
        and isinstance(candidates, list)
        and bool(candidates)
        and all(
            isinstance(candidate, Mapping) and candidate.get("status") in {"completed", "failed"}
            for candidate in candidates
        )
    )


def _source_configuration_id(selected: Mapping[str, object], label: str) -> str:
    source_ids = selected.get("source_configuration_ids")
    if isinstance(source_ids, Mapping) and source_ids.get(label) is not None:
        return _text(source_ids[label], error=f"{label}_source_configuration_id_invalid")
    return _text(selected.get("configuration_id"), error="config_invalid")


def _validate_compatibility(
    payload: Mapping[str, object],
    selected: Mapping[str, object],
    *,
    label: str,
    require_scorer: bool,
) -> dict[str, str]:
    compatibility_status, limitation = _comparison_declaration(selected, label)
    complete = _source_complete(payload) or (
        compatibility_status == "partial" and _partial_comparison_complete(payload)
    )
    if not complete:
        raise AcceptanceError(f"{label}_report_incomplete")
    actual = _source_metadata(payload)
    if compatibility_status == "partial":
        expected_dataset = _text(selected.get("dataset_id"), error="dataset_invalid")
        if actual.get("dataset_id") != expected_dataset:
            raise AcceptanceError(f"{label}_dataset_id_incompatible")
        actual["compatibility_status"] = "partial"
        actual["limitation"] = _text(
            limitation,
            error=f"selected_configuration_{label}_limitation_invalid",
        )
        return actual
    expected = {
        "configuration_id": _source_configuration_id(selected, label),
        "profile": _text(selected.get("profile"), error="profile_invalid"),
        "retrieval_mode": _text(selected.get("retrieval_mode"), error="mode_invalid"),
        "dataset_id": _text(selected.get("dataset_id"), error="dataset_invalid"),
    }
    if require_scorer:
        expected["scorer_contract"] = _text(selected.get("scorer_contract"), error="scorer_invalid")
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value is None and key in {"profile", "retrieval_mode"}:
            continue
        if actual_value != expected_value:
            raise AcceptanceError(f"{label}_{key}_incompatible")
    actual["compatibility_status"] = "exact"
    return actual


def _quality_metrics(payload: Mapping[str, object]) -> dict[str, dict[str, object]]:
    metrics: dict[str, dict[str, object]] = {}
    raw = payload.get("metrics")
    if isinstance(raw, Mapping):
        aggregate = raw.get("aggregate", raw)
        if isinstance(aggregate, Mapping):
            aliases = {
                "context_precision": "context-precision",
                "answer_compliance": "answer-compliance",
                "style": "style-consistency",
                "style_consistency": "style-consistency",
                "refusal_appropriateness": "refusal-appropriateness",
            }
            for raw_name, item in aggregate.items():
                if isinstance(raw_name, str) and isinstance(item, Mapping):
                    name = aliases.get(raw_name, raw_name.replace("_", "-"))
                    metrics[name] = dict(item)
    gates = payload.get("gates")
    if isinstance(gates, list):
        for gate in gates:
            if not isinstance(gate, Mapping):
                continue
            observations = gate.get("observations")
            if not isinstance(observations, list):
                continue
            for item in observations:
                if isinstance(item, Mapping) and isinstance(item.get("metric_id"), str):
                    raw_name = cast(str, item["metric_id"])
                    name = {
                        "style": "style-consistency",
                    }.get(raw_name, raw_name)
                    metrics[name] = dict(item)
    return metrics


def _evaluate_quality(payload: Mapping[str, object]) -> dict[str, object]:
    observations = _quality_metrics(payload)
    results: dict[str, object] = {}
    passed = True
    for metric, threshold in _QUALITY_THRESHOLDS.items():
        observation = observations.get(metric)
        if observation is None:
            results[metric] = {"passed": False, "reason": "metric-missing"}
            passed = False
            continue
        value = _decimal(observation.get("value"), error=f"quality_{metric}_value_invalid")
        denominator = _integer(
            observation.get("denominator", observation.get("eligible_cases")),
            error=f"quality_{metric}_denominator_invalid",
            minimum=1,
        )
        metric_passed = value >= threshold
        results[metric] = {
            "denominator": denominator,
            "passed": metric_passed,
            "threshold": str(threshold),
            "value": str(value),
        }
        passed = passed and metric_passed
    return {"passed": passed, "metrics": results}


def _nearest_rank(values: Sequence[float], percentile: int) -> float:
    if not values:
        raise AcceptanceError("performance_latency_samples_missing")
    ordered = sorted(values)
    return ordered[math.ceil(percentile / 100 * len(ordered)) - 1]


def _evaluate_performance(payload: Mapping[str, object]) -> dict[str, object]:
    attempts = payload.get("attempts")
    if not isinstance(attempts, list):
        raise AcceptanceError("performance_attempts_missing")
    latencies: list[float] = []
    successes = 0
    within = 0
    failures = 0
    timeouts = 0
    identities: set[str] = set()
    logical_request_ids: set[str] = set()
    identity_evidence_count = 0
    cache_bypass_count = 0
    retry_count = 0
    for index, raw in enumerate(attempts):
        item = _mapping(raw, error=f"performance_attempt_{index}_invalid")
        latency_value = item.get("latency_ms")
        if isinstance(latency_value, bool) or not isinstance(latency_value, int | float):
            raise AcceptanceError(f"performance_attempt_{index}_latency_invalid")
        latency = float(latency_value)
        if not math.isfinite(latency) or latency < 0:
            raise AcceptanceError(f"performance_attempt_{index}_latency_invalid")
        latencies.append(latency)
        status = item.get("status")
        if status == "succeeded":
            successes += 1
            within += int(latency <= LATENCY_LIMIT_MS)
        else:
            failures += 1
            timeouts += int(status == "timeout")
        identity = item.get("instance_identity")
        if isinstance(identity, str) and identity:
            identities.add(identity)
            identity_evidence_count += 1
        logical_request_id = item.get("logical_request_id")
        if isinstance(logical_request_id, str) and logical_request_id:
            logical_request_ids.add(logical_request_id)
        retry_count += int(item.get("attempt_number", 1) != 1)
        cache_status = item.get("cache_status")
        if isinstance(cache_status, Mapping) and cache_status.get("request-policy") == "bypass":
            cache_bypass_count += 1
    p50 = _nearest_rank(latencies, 50)
    p90 = _nearest_rank(latencies, 90)
    p95 = _nearest_rank(latencies, 95)
    observed = _integer(
        payload.get("observed_peak_concurrency"),
        error="performance_observed_concurrency_invalid",
    )
    configured = _integer(
        payload.get("configured_concurrency"),
        error="performance_configured_concurrency_invalid",
    )
    instance_count = _integer(
        payload.get("instance_count"), error="performance_instance_count_invalid", minimum=1
    )
    gates = {
        "exactly_100_measured": len(attempts) == PERFORMANCE_SAMPLE_SIZE,
        "cache_bypassed": (
            payload.get("cache_policy") == "bypass"
            and cache_bypass_count == PERFORMANCE_SAMPLE_SIZE
        ),
        "exactly_100_logical_requests": (
            len(logical_request_ids) == PERFORMANCE_SAMPLE_SIZE and retry_count == 0
        ),
        "configured_concurrency_five": configured == REQUIRED_CONCURRENCY,
        "observed_concurrency_at_least_five": observed >= REQUIRED_CONCURRENCY,
        "one_instance": (
            instance_count == 1
            and len(identities) == 1
            and identity_evidence_count == PERFORMANCE_SAMPLE_SIZE
        ),
        "at_least_90_successful_within_10s": within >= MINIMUM_WITHIN_TEN_SECONDS,
        "all_attempt_p90_at_most_10s": p90 <= LATENCY_LIMIT_MS,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "measured_requests": len(attempts),
        "successes": successes,
        "failures": failures,
        "timeouts": timeouts,
        "successful_within_10s": within,
        "configured_concurrency": configured,
        "observed_concurrency": observed,
        "instance_count": instance_count,
        "instance_identity_count": len(identities),
        "latency_ms": {"p50": p50, "p90": p90, "p95": p95},
    }


def _token_total(payload: Mapping[str, object], direction: str) -> int:
    totals = payload.get("token_totals", {})
    if not isinstance(totals, Mapping):
        raise AcceptanceError("performance_token_totals_invalid")
    normalized_direction = direction.casefold()
    result = sum(
        value
        for key, value in totals.items()
        if isinstance(key, str)
        and key.casefold().replace("_", "-")
        in {normalized_direction, f"generation-{normalized_direction}"}
        and type(value) is int
        and value >= 0
    )
    if result <= 0:
        raise AcceptanceError(f"performance_{direction}_tokens_missing")
    return result


def _cost_summary(
    selected: Mapping[str, object], performance: Mapping[str, object]
) -> dict[str, object]:
    cost = _mapping(selected.get("cost"), error="selected_configuration_cost_invalid")
    remote = _mapping(cost.get("remote"), error="selected_configuration_remote_cost_invalid")
    local = _mapping(cost.get("local"), error="selected_configuration_local_cost_invalid")
    measured = len(cast(list[object], performance["attempts"]))
    if measured <= 0:
        raise AcceptanceError("cost_denominator_invalid")
    input_tokens = _token_total(performance, "input")
    output_tokens = _token_total(performance, "output")
    input_rate = _decimal(remote.get("input_per_million"), error="remote_input_rate_invalid")
    output_rate = _decimal(remote.get("output_per_million"), error="remote_output_rate_invalid")
    hourly_rate = _decimal(local.get("hourly_rate"), error="local_hourly_rate_invalid")
    declared_duration = _decimal(
        local.get("allocation_duration_seconds"), error="local_duration_invalid"
    )
    duration_source = local.get("allocation_duration_source", "declared")
    duration = declared_duration
    if duration_source == "measured-load-duration":
        duration = _decimal(
            performance.get("duration_ms"),
            error="performance_duration_invalid",
            positive=True,
        ) / Decimal(1000)
    remote_observed = (
        Decimal(input_tokens) * input_rate + Decimal(output_tokens) * output_rate
    ) / Decimal(1_000_000)
    local_observed = hourly_rate * duration / Decimal(3600)
    factor = Decimal(1000) / Decimal(measured)
    remote_per_1000 = remote_observed * factor
    local_per_1000 = local_observed * factor
    combined = remote_per_1000 + local_per_1000
    return {
        "complete": True,
        "currency": remote["currency"],
        "calculated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "denominator": measured,
        "remote": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_per_million": str(input_rate),
            "output_per_million": str(output_rate),
            "pricing_source": remote["pricing_source"],
            "observed_cost": str(remote_observed),
            "per_1000_requests": str(remote_per_1000),
        },
        "local": {
            "hardware": local["hardware"],
            "hourly_rate": str(hourly_rate),
            "allocation_duration_seconds": str(duration),
            "allocation_duration_source": duration_source,
            "declared_maximum_duration_seconds": str(declared_duration),
            "observed_cost": str(local_observed),
            "per_1000_requests": str(local_per_1000),
        },
        "combined_per_1000_requests": str(combined),
        "formula": (
            "remote=(input_tokens*input_rate+output_tokens*output_rate)/1e6; "
            "local=hourly_rate*duration_seconds/3600; each scaled by 1000/requests"
        ),
        "evaluation_judge_cost": {
            "status": "unavailable",
            "reason": "judge usage is not part of serving performance evidence",
        },
    }


def _maximum_run_cost(selected: Mapping[str, object]) -> Decimal:
    cost = _mapping(selected.get("cost"), error="selected_configuration_cost_invalid")
    remote = _mapping(cost.get("remote"), error="selected_configuration_remote_cost_invalid")
    local = _mapping(cost.get("local"), error="selected_configuration_local_cost_invalid")
    remote_max = (
        Decimal(PERFORMANCE_SAMPLE_SIZE)
        * (
            Decimal(_integer(remote.get("maximum_input_tokens"), error="max_input_invalid"))
            * _decimal(remote.get("input_per_million"), error="input_rate_invalid")
            + Decimal(_integer(remote.get("maximum_output_tokens"), error="max_output_invalid"))
            * _decimal(remote.get("output_per_million"), error="output_rate_invalid")
        )
        / Decimal(1_000_000)
    )
    local_max = (
        _decimal(local.get("hourly_rate"), error="hourly_rate_invalid")
        * _decimal(local.get("allocation_duration_seconds"), error="duration_invalid")
        / Decimal(3600)
    )
    return remote_max + local_max


def _load_scenarios(path: Path) -> tuple[LoadScenario, ...]:
    payload = _load_json(path, label="scenario")
    raw = payload.get("scenarios")
    if not isinstance(raw, list) or not raw:
        raise AcceptanceError("scenario_list_invalid")
    try:
        return tuple(LoadScenario.model_validate(item) for item in raw)
    except ValueError:
        raise AcceptanceError("scenario_list_invalid") from None


def _load_config(args: argparse.Namespace, selected: Mapping[str, object]) -> HttpLoadTestConfig:
    if args.max_run_cost is None:
        raise AcceptanceError("max_run_cost_required")
    allowed = _decimal(args.max_run_cost, error="max_run_cost_invalid", positive=True)
    if _maximum_run_cost(selected) > allowed:
        raise AcceptanceError("maximum_run_cost_exceeded")
    if args.base_url is None or args.scenario_file is None:
        raise AcceptanceError("load_service_and_scenarios_required")
    return HttpLoadTestConfig(
        run_id=args.run_id,
        expected_configuration_id=cast(
            str,
            selected.get("runtime_configuration_id", selected["configuration_id"]),
        ),
        base_url=args.base_url,
        scenarios=_load_scenarios(args.scenario_file),
        warmup_attempts=args.warmup_attempts,
        concurrency=REQUIRED_CONCURRENCY,
        target_successes=MINIMUM_WITHIN_TEN_SECONDS,
        max_attempts=PERFORMANCE_SAMPLE_SIZE,
        exact_measured_attempts=PERFORMANCE_SAMPLE_SIZE,
        retry_limit=0,
        request_timeout_seconds=args.request_timeout_seconds,
        instance_count=1,
    )


async def _preflight_load_service(config: HttpLoadTestConfig) -> None:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            response = await client.get(f"{config.base_url}{config.readiness_path}")
            payload = response.json()
    except (httpx.HTTPError, ValueError):
        raise AcceptanceError("load_service_unavailable") from None
    if not isinstance(payload, dict) or response.status_code != 200:
        raise AcceptanceError("load_service_unavailable")
    if payload.get("status") != "ready":
        raise AcceptanceError("load_service_not_ready")
    if payload.get("configuration_id") != config.expected_configuration_id:
        raise AcceptanceError("load_service_configuration_mismatch")
    if not isinstance(payload.get("instance_identity"), str):
        raise AcceptanceError("load_service_instance_identity_missing")


async def _run_load(args: argparse.Namespace, selected: Mapping[str, object]) -> dict[str, object]:
    config = _load_config(args, selected)
    thresholds = LoadAcceptanceThresholds(
        minimum_successes=MINIMUM_WITHIN_TEN_SECONDS,
        maximum_error_rate_exclusive=1,
    )
    report = await HttpLoadTestHarness(config, thresholds=thresholds).run()
    return cast(dict[str, object], report.model_dump(mode="json"))


def _evidence_record(
    label: str,
    path: Path,
    payload: Mapping[str, object],
    selected: Mapping[str, object],
) -> dict[str, object]:
    compatibility_status, limitation = _comparison_declaration(selected, label)
    record: dict[str, object] = {
        "kind": label,
        "source_id": _source_identity(payload),
        "sha256": _sha256(path),
        "compatibility_status": "exact" if label == "quality" else compatibility_status,
    }
    if limitation is not None:
        record["limitation"] = limitation
    return record


def _validate_issue_evidence(items: Sequence[object]) -> list[Mapping[str, object]]:
    if len(items) < 2:
        raise AcceptanceError("issue_evidence_requires_two_items")
    validated: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for index, raw in enumerate(items):
        item = _mapping(raw, error=f"issue_evidence_{index}_invalid")
        issue_id = _text(item.get("issue_id"), error=f"issue_evidence_{index}_id_invalid")
        if issue_id in seen:
            raise AcceptanceError("issue_evidence_duplicate_id")
        seen.add(issue_id)
        for key in ("symptom", "root_cause", "exact_fix", "primary_metric"):
            _text(item.get(key), error=f"issue_evidence_{index}_{key}_invalid")
        baseline = _decimal(
            item.get("baseline_value"), error=f"issue_evidence_{index}_baseline_invalid"
        )
        post_fix = _decimal(
            item.get("post_fix_value"), error=f"issue_evidence_{index}_post_fix_invalid"
        )
        improvement = _decimal(
            item.get("relative_improvement_percent"),
            error=f"issue_evidence_{index}_improvement_invalid",
        )
        if item.get("passed") is not True or post_fix >= baseline or improvement < 10:
            raise AcceptanceError(f"issue_evidence_{index}_not_passing")
        validated.append(item)
    return validated


def _validate_log_evidence(dictionary: Mapping[str, object], sample_path: Path) -> dict[str, int]:
    fields = dictionary.get("fields")
    if not isinstance(fields, list) or not fields:
        raise AcceptanceError("log_dictionary_fields_missing")
    field_names = {
        item.get("name")
        for item in fields
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    required_fields = {"event", "level", "service", "timestamp"}
    if not required_fields.issubset(field_names):
        raise AcceptanceError("log_dictionary_required_fields_missing")
    sample_count = 0
    try:
        for line in sample_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            decoded = json.loads(line)
            if not isinstance(decoded, dict) or not required_fields.issubset(decoded):
                raise AcceptanceError("log_sample_record_invalid")
            sample_count += 1
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise AcceptanceError("log_sample_jsonl_invalid") from None
    if sample_count == 0:
        raise AcceptanceError("log_sample_empty")
    return {"field_count": len(field_names), "sample_count": sample_count}


def _issue_markdown(items: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Two-Issue Diagnosis and Before/After Evidence",
        "",
        "This report is rendered from `issue-diagnosis.json`; no provider was called.",
        "",
        "| Issue | Root cause | Fix | Metric | Before | After | Improvement |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for item in items:
        lines.append(
            "| `{issue}` | {cause} | {fix} | `{metric}` | {before} | {after} | {change}% |".format(
                issue=item["issue_id"],
                cause=item["root_cause"],
                fix=item["exact_fix"],
                metric=item["primary_metric"],
                before=item["baseline_value"],
                after=item["post_fix_value"],
                change=item["relative_improvement_percent"],
            )
        )
    lines.extend(
        (
            "",
            "Every listed issue has a passing machine-readable record and at least "
            "10% relative improvement.",
            "",
        )
    )
    return "\n".join(lines)


def _crosswalk(summary: Mapping[str, object]) -> list[dict[str, object]]:
    quality_passed = _nested(summary, "gates", "quality", "passed") is True
    performance_passed = _nested(summary, "gates", "performance", "passed") is True
    cost_passed = _nested(summary, "gates", "cost", "passed") is True
    content_safety_passed = summary.get("content_safety_passed") is True
    supporting = _mapping(summary.get("supporting_evidence"), error="supporting_evidence_missing")
    retrieval_ready = supporting.get("retrieval_comparison") is True
    issue_ready = supporting.get("two_issue_diagnosis") is True
    logs_ready = supporting.get("structured_logs") is True
    implementation_ready = supporting.get("implementation_documentation") is True
    one_command_ready = supporting.get("one_command_reproduction") is True
    operations_ready = quality_passed and performance_passed and cost_passed
    rows = [
        ("performance", performance_passed, "summary.json#/gates/performance", "100 requests"),
        ("cost", cost_passed, "operations-cost.json", "per 1,000 requests"),
        ("rag-quality", quality_passed, "summary.json#/gates/quality", "quality denominators"),
        (
            "logging-tracing",
            logs_ready and performance_passed,
            "evidence/structured-log-sample.jsonl",
            "packaged samples and 100 attempts",
        ),
        (
            "security",
            quality_passed and logs_ready,
            "evidence/quality-report.json",
            "quality safety cases and packaged content scan",
        ),
        (
            "vector-and-hybrid",
            retrieval_ready,
            "evidence/retrieval-reranker-comparison.json",
            "comparison",
        ),
        ("reranker-toggle", retrieval_ready, "selected-configuration.json", "configuration"),
        ("refusal-safety", quality_passed, "summary.json#/gates/quality", "refusal denominator"),
        (
            "privacy",
            content_safety_passed,
            "manifest.json#/content_checks",
            "all packaged text",
        ),
        ("operations-report", operations_ready, "operations-cost.md", "100 requests"),
        (
            "three-retrieval-configurations",
            retrieval_ready,
            "evidence/retrieval-reranker-comparison.json",
            "comparison",
        ),
        ("evolvability", implementation_ready, "implementation-evidence.md", "repository seams"),
        (
            "advanced-generative-quality",
            quality_passed,
            "summary.json#/gates/quality",
            "quality denominators",
        ),
        ("two-issue-diagnosis", issue_ready, "evidence/issue-diagnosis.json", "two issues"),
        ("complete-code-configs", implementation_ready, "implementation-evidence.md", "repository"),
        ("one-click-evaluation", one_command_ready, "reproduce.md", "one command"),
        ("before-after-report", issue_ready, "evidence/issue-diagnosis.md", "two comparisons"),
        (
            "log-dictionary-samples",
            logs_ready,
            "evidence/structured-log-field-dictionary.json",
            "field dictionary and samples",
        ),
    ]
    source_limitations: dict[str, str] = {}
    source_evidence = summary.get("source_evidence")
    if isinstance(source_evidence, list):
        filenames = {
            "model": "evidence/model-comparison.json",
            "retrieval": "evidence/retrieval-reranker-comparison.json",
            "cache": "evidence/cache-comparison.json",
        }
        for item in source_evidence:
            if not isinstance(item, Mapping):
                continue
            kind = item.get("kind")
            limitation = item.get("limitation")
            if isinstance(kind, str) and isinstance(limitation, str) and kind in filenames:
                source_limitations[filenames[kind]] = limitation
    return [
        {
            "requirement_id": name,
            "status": "complete" if status else "incomplete",
            "evidence": evidence,
            "denominator": denominator,
            "limitation": (
                source_limitations.get(evidence)
                if status
                else "mandatory supporting gate is not passing"
            ),
        }
        for name, status, evidence, denominator in rows
    ]


def _markdown_table(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| Requirement | Status | Evidence | Denominator | Limitation |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {requirement_id} | {status} | `{evidence}` | {denominator} | {limitation} |".format(
                **{key: "" if value is None else value for key, value in row.items()}
            )
        )
    return "\n".join(lines) + "\n"


def _scan_text_files(root: Path) -> tuple[bool, list[str]]:
    issues: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            issues.append(f"non_utf8:{relative}")
            continue
        if _ABSOLUTE_PATH.search(text):
            issues.append(f"absolute_path:{relative}")
        candidates: Sequence[str] = (text,)
        if path.suffix.casefold() == ".json":
            try:
                candidates = tuple(_json_text_values(json.loads(text)))
            except json.JSONDecodeError:
                issues.append(f"invalid_json:{relative}")
                continue
        else:
            candidates = (_DECIMAL_LITERAL.sub("[decimal]", _DIGEST.sub("[digest]", text)),)
        kinds = {
            span.kind
            for candidate in candidates
            if not _safe_machine_string(candidate)
            for span in DEFAULT_REDACTOR.detect(candidate)
        }
        prohibited = kinds & {
            SensitiveKind.EMAIL,
            SensitiveKind.PHONE,
            SensitiveKind.CHINESE_ID,
            SensitiveKind.SSN,
            SensitiveKind.PAYMENT_CARD,
            SensitiveKind.SECRET,
        }
        if prohibited:
            issues.append(f"sensitive_content:{relative}")
    return not issues, issues


def _json_text_values(value: object) -> Sequence[str]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(text for item in value for text in _json_text_values(item))
    if isinstance(value, dict):
        return tuple(text for item in value.values() for text in _json_text_values(item))
    return ()


def _safe_machine_string(value: str) -> bool:
    return bool(_DIGEST.fullmatch(value) or re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value))


def _manifest(root: Path, *, accepted: bool) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        _validate_release_relative_name(relative)
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    safe, issues = _scan_text_files(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "accepted" if accepted and safe else "rejected",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "files": files,
        "content_checks": {"passed": safe, "issues": issues},
    }


def verify_release(root: Path) -> dict[str, object]:
    resolved = root.resolve(strict=True)
    manifest = _load_json(resolved / "manifest.json", label="manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise AcceptanceError("manifest_schema_invalid")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise AcceptanceError("manifest_files_invalid")
    recorded: set[str] = set()
    for index, raw in enumerate(files):
        item = _mapping(raw, error=f"manifest_file_{index}_invalid")
        relative = _text(item.get("path"), error=f"manifest_file_{index}_path_invalid")
        _validate_release_relative_name(relative)
        pure = PurePosixPath(relative)
        if (
            "\\" in relative
            or pure.is_absolute()
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise AcceptanceError(f"manifest_file_{index}_path_invalid")
        path = resolved.joinpath(*pure.parts)
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(resolved):
            raise AcceptanceError(f"manifest_file_missing:{relative}")
        size = _integer(item.get("bytes"), error=f"manifest_size_invalid:{relative}")
        digest = _text(item.get("sha256"), error=f"manifest_digest_invalid:{relative}")
        if not _DIGEST.fullmatch(digest):
            raise AcceptanceError(f"manifest_digest_invalid:{relative}")
        if path.stat().st_size != size:
            raise AcceptanceError(f"manifest_size_mismatch:{relative}")
        if _sha256(path) != digest:
            raise AcceptanceError(f"manifest_digest_mismatch:{relative}")
        recorded.add(relative)
    missing = sorted(set(_REQUIRED_FILES) - recorded)
    if missing:
        raise AcceptanceError(f"required_content_missing:{missing[0]}")
    actual = {
        path.relative_to(resolved).as_posix()
        for path in resolved.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual != recorded:
        raise AcceptanceError("manifest_file_set_mismatch")
    summary = _load_json(resolved / "summary.json", label="summary")
    operations = _load_json(resolved / "operations-cost.json", label="operations")
    crosswalk = _load_json(resolved / "pdf-crosswalk.json", label="crosswalk")
    requirements = crosswalk.get("requirements")
    expected_requirements = _crosswalk(summary)
    if requirements != expected_requirements:
        raise AcceptanceError("crosswalk_summary_mismatch")
    for index, requirement in enumerate(expected_requirements):
        evidence_reference = _text(
            requirement.get("evidence"), error=f"crosswalk_{index}_evidence_invalid"
        )
        evidence_path = evidence_reference.split("#", maxsplit=1)[0]
        _validate_release_relative_name(evidence_path)
        if evidence_path != "manifest.json" and evidence_path not in recorded:
            raise AcceptanceError(f"crosswalk_evidence_missing:{evidence_path}")
    if summary.get("accepted") is True and any(
        requirement.get("status") != "complete" for requirement in expected_requirements
    ):
        raise AcceptanceError("accepted_crosswalk_incomplete")
    summary_performance = _nested(summary, "gates", "performance")
    if summary_performance != operations.get("performance"):
        raise AcceptanceError("summary_operations_performance_mismatch")
    if _nested(summary, "gates", "cost", "passed") is not _nested(operations, "cost", "complete"):
        raise AcceptanceError("summary_operations_cost_mismatch")
    expected_status = "accepted" if summary.get("accepted") is True else "rejected"
    if manifest.get("status") != expected_status:
        raise AcceptanceError("summary_manifest_status_mismatch")
    checks = _mapping(manifest.get("content_checks"), error="manifest_checks_invalid")
    if expected_status == "accepted" and checks.get("passed") is not True:
        raise AcceptanceError("accepted_manifest_checks_failed")
    return {
        "verified": True,
        "status": manifest["status"],
        "file_count": len(recorded),
        "manifest_sha256": _sha256(resolved / "manifest.json"),
    }


def _build_release(
    args: argparse.Namespace,
    selected: Mapping[str, object],
    sources: Mapping[str, tuple[Path, Mapping[str, object]]],
    performance: Mapping[str, object],
    issue_records: Sequence[Mapping[str, object]],
    log_dictionary_path: Path,
    log_sample_path: Path,
    log_evidence: Mapping[str, int],
) -> dict[str, object]:
    output = args.output
    if output.exists():
        raise AcceptanceError("output_path_already_exists")
    quality = _evaluate_quality(sources["quality"][1])
    performance_gate = _evaluate_performance(performance)
    cost = _cost_summary(selected, performance)
    source_records = [
        _evidence_record(label, path, payload, selected)
        for label, (path, payload) in sources.items()
    ]
    incomplete_non_blocking = [
        {
            "kind": record["kind"],
            "status": "partial",
            "limitation": record["limitation"],
        }
        for record in source_records
        if record.get("compatibility_status") == "partial"
    ]
    quality_metrics = cast(Mapping[str, object], quality["metrics"])
    answer_compliance = quality_metrics["answer-compliance"]
    refusal_count = sum(
        1
        for item in cast(list[object], performance["attempts"])
        if isinstance(item, Mapping) and item.get("terminal_kind") == "refusal"
    )
    terminal_count = sum(
        1
        for item in cast(list[object], performance["attempts"])
        if isinstance(item, Mapping) and item.get("terminal_kind") in {"answer", "refusal"}
    )
    operations = {
        "performance": performance_gate,
        "cost": cost,
        "token_usage": {
            "generation_input": _token_total(performance, "input"),
            "generation_output": _token_total(performance, "output"),
            "source": "evidence/performance-100.json#/token_totals",
        },
        "cache_hit_rate": {
            "status": "unavailable",
            "reason": "official performance traffic is cache bypassed",
            "denominator": PERFORMANCE_SAMPLE_SIZE,
        },
        "refusal_rate": (
            {
                "status": "available",
                "value": refusal_count / terminal_count,
                "numerator": refusal_count,
                "denominator": terminal_count,
            }
            if terminal_count
            else {
                "status": "unavailable",
                "reason": "terminal answer/refusal kinds are absent from the reused evidence",
                "denominator": 0,
            }
        ),
        "answer_compliance": {
            **cast(dict[str, object], answer_compliance),
            "source": "evidence/quality-report.json",
        },
        "selection_rationale": selected["selection_rationale"],
        "source_reports": source_records,
    }
    gates = {
        "quality": quality,
        "performance": performance_gate,
        "cost": {"passed": cost["complete"] is True},
        "source_compatibility": {"passed": True},
    }
    source_kinds = {record["kind"] for record in source_records}
    supporting_evidence = {
        "model_comparison": "model" in source_kinds,
        "retrieval_comparison": "retrieval" in source_kinds,
        "cache_comparison": "cache" in source_kinds,
        "two_issue_diagnosis": len(issue_records) >= 2,
        "structured_logs": (
            log_evidence.get("field_count", 0) > 0 and log_evidence.get("sample_count", 0) > 0
        ),
        "implementation_documentation": True,
        "one_command_reproduction": True,
    }
    gates["supporting_evidence"] = {"passed": all(supporting_evidence.values())}
    accepted = all(
        cast(Mapping[str, object], gate).get("passed") is True for gate in gates.values()
    )
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "accepted": accepted,
        "content_safety_passed": True,
        "configuration_id": selected["configuration_id"],
        "gates": gates,
        "source_evidence": source_records,
        "supporting_evidence": supporting_evidence,
        "incomplete_non_blocking": incomplete_non_blocking,
        "limitations": selected["limitations"],
    }
    output.mkdir(parents=True)
    evidence = output / "evidence"
    evidence.mkdir()
    copies = {
        "quality": "quality-report.json",
        "model": "model-comparison.json",
        "retrieval": "retrieval-reranker-comparison.json",
        "cache": "cache-comparison.json",
    }
    for label, filename in copies.items():
        shutil.copyfile(sources[label][0], evidence / filename)
    _write_json(evidence / "performance-100.json", performance)
    _write_json(evidence / "issue-diagnosis.json", list(issue_records))
    (evidence / "issue-diagnosis.md").write_text(
        _issue_markdown(issue_records), encoding="utf-8", newline="\n"
    )
    shutil.copyfile(log_dictionary_path, evidence / "structured-log-field-dictionary.json")
    shutil.copyfile(log_sample_path, evidence / "structured-log-sample.jsonl")
    _write_json(output / "selected-configuration.json", selected)
    _write_json(output / "operations-cost.json", operations)
    operations_md = (
        "# Operations and Cost\n\n"
        f"- Requests: {performance_gate['measured_requests']}\n"
        f"- Successes: {performance_gate['successes']}\n"
        f"- Failures: {performance_gate['failures']}\n"
        f"- Successful within 10 seconds: {performance_gate['successful_within_10s']}/100\n"
        f"- Observed concurrency: {performance_gate['observed_concurrency']}\n"
        f"- Latency p50/p90/p95 ms: {performance_gate['latency_ms']}\n"
        f"- Generation input/output tokens: {operations['token_usage']}\n"
        "- Cache hit rate: unavailable (official traffic bypasses cache)\n"
        f"- Refusal rate: {operations['refusal_rate']}\n"
        f"- Answer compliance: {operations['answer_compliance']}\n"
        f"- Serving cost per 1,000: {cost['combined_per_1000_requests']} {cost['currency']}\n"
        "- Evaluation judge expense: unavailable; not represented as zero\n"
        f"- Selection rationale: {operations['selection_rationale']}\n"
    )
    (output / "operations-cost.md").write_text(operations_md, encoding="utf-8", newline="\n")
    _write_json(output / "summary.json", summary)
    summary_md = (
        "# Release v2 Acceptance Summary\n\n"
        f"Overall status: **{'ACCEPTED' if accepted else 'REJECTED'}**\n\n"
        f"Configuration: `{selected['configuration_id']}`\n\n"
        f"Quality gate: {quality['passed']}\n\n"
        f"Performance gate: {performance_gate['passed']}\n\n"
        f"Serving-cost gate: {cost['complete']}\n"
    )
    (output / "summary.md").write_text(summary_md, encoding="utf-8", newline="\n")
    rows = _crosswalk(summary)
    _write_json(output / "pdf-crosswalk.json", {"requirements": rows})
    (output / "pdf-crosswalk.md").write_text(
        "# Original PDF Requirement Crosswalk\n\n" + _markdown_table(rows),
        encoding="utf-8",
        newline="\n",
    )
    limitations = cast(list[str], selected["limitations"])
    (output / "limitations.md").write_text(
        "# Known MVP Limitations\n\n" + "".join(f"- {item}\n" for item in limitations),
        encoding="utf-8",
        newline="\n",
    )
    (output / "implementation-evidence.md").write_text(
        "# Implementation Evidence\n\n"
        "| Area | Repository evidence |\n"
        "|---|---|\n"
        "| Architecture and evolvability | `docs/architecture.md`, `src/rag_mvp/` |\n"
        "| Configuration | `docs/configuration.md`, `.env.example`, "
        "`src/rag_mvp/config/settings.py` |\n"
        "| Acceptance entry point | `src/rag_mvp/acceptance.py`, `pyproject.toml` |\n"
        "| Retrieval and reranker toggle | `src/rag_mvp/retrieval/`, "
        "`selected-configuration.json` |\n"
        "| Safety and privacy | `src/rag_mvp/safety/`, packaged quality and log evidence |\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / "reproduce.md").write_text(
        "# Reproduction\n\n"
        "Rebuild the package from its copied evidence without provider calls:\n\n"
        "```powershell\n"
        "uv run rag-mvp-acceptance `\n"
        "  --selected-config <release-directory>/selected-configuration.json `\n"
        "  --quality-report <release-directory>/evidence/quality-report.json `\n"
        "  --model-report <release-directory>/evidence/model-comparison.json `\n"
        "  --retrieval-report <release-directory>/evidence/retrieval-reranker-comparison.json `\n"
        "  --cache-report <release-directory>/evidence/cache-comparison.json `\n"
        "  --issue-data <release-directory>/evidence/issue-diagnosis.json `\n"
        "  --log-dictionary <release-directory>/evidence/structured-log-field-dictionary.json `\n"
        "  --log-sample <release-directory>/evidence/structured-log-sample.jsonl `\n"
        "  --performance-report <release-directory>/evidence/performance-100.json `\n"
        "  --output data/releases/<new-release-id>\n"
        "```\n\n"
        "Verify this directory without credentials:\n\n"
        "```powershell\n"
        "uv run rag-mvp-acceptance --offline-verify <release-directory>\n"
        "```\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = _manifest(output, accepted=accepted)
    content_safe = manifest["content_checks"]["passed"] is True
    if not content_safe or manifest["status"] != ("accepted" if accepted else "rejected"):
        summary["content_safety_passed"] = content_safe
        summary["accepted"] = accepted and content_safe
        _write_json(output / "summary.json", summary)
        (output / "summary.md").write_text(
            "# Release v2 Acceptance Summary\n\n"
            f"Overall status: **{'ACCEPTED' if summary['accepted'] else 'REJECTED'}**\n",
            encoding="utf-8",
            newline="\n",
        )
        rows = _crosswalk(summary)
        _write_json(output / "pdf-crosswalk.json", {"requirements": rows})
        (output / "pdf-crosswalk.md").write_text(
            "# Original PDF Requirement Crosswalk\n\n" + _markdown_table(rows),
            encoding="utf-8",
            newline="\n",
        )
        manifest = _manifest(output, accepted=summary["accepted"] is True)
    _write_json(output / "manifest.json", manifest)
    verified = verify_release(output)
    return {"output": str(output), "summary": summary, "verification": verified}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-config", type=Path)
    parser.add_argument("--quality-report", type=Path)
    parser.add_argument("--model-report", type=Path)
    parser.add_argument("--retrieval-report", type=Path)
    parser.add_argument("--cache-report", type=Path)
    parser.add_argument(
        "--issue-data",
        type=Path,
        default=Path("evaluations/two-issue-diagnosis.json"),
    )
    parser.add_argument(
        "--log-dictionary",
        type=Path,
        default=Path("evaluations/logging/structured-log-field-dictionary-v1.json"),
    )
    parser.add_argument(
        "--log-sample",
        type=Path,
        default=Path("evaluations/logging/privacy-safe-sample-v1.jsonl"),
    )
    parser.add_argument("--performance-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-load", action="store_true")
    parser.add_argument("--offline-verify", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--scenario-file", type=Path)
    parser.add_argument("--run-id", default="mvp-acceptance-v2")
    parser.add_argument("--warmup-attempts", type=int, default=5)
    parser.add_argument("--request-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--max-requests", type=int, default=PERFORMANCE_SAMPLE_SIZE)
    parser.add_argument("--max-run-cost")
    return parser


def _required_path(args: argparse.Namespace, name: str) -> Path:
    value = getattr(args, name)
    if not isinstance(value, Path):
        raise AcceptanceError(f"{name}_required")
    return value


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.offline_verify is not None:
        return verify_release(args.offline_verify)
    if args.max_requests != PERFORMANCE_SAMPLE_SIZE:
        raise AcceptanceError("max_requests_must_equal_100")
    if args.run_load and args.performance_report is not None:
        raise AcceptanceError("performance_source_is_ambiguous")
    if not args.run_load and args.performance_report is None:
        raise AcceptanceError("performance_report_or_run_load_required")
    output = _required_path(args, "output")
    if output.exists():
        raise AcceptanceError("output_path_already_exists")
    selected_path = _required_path(args, "selected_config")
    selected = _load_json(selected_path, label="selected_configuration")
    _validate_selected_configuration(selected)
    sources: dict[str, tuple[Path, Mapping[str, object]]] = {}
    for label, argument in (
        ("quality", "quality_report"),
        ("model", "model_report"),
        ("retrieval", "retrieval_report"),
        ("cache", "cache_report"),
    ):
        path = _required_path(args, argument)
        payload = _load_json(path, label=label)
        _validate_compatibility(
            payload,
            selected,
            label=label,
            require_scorer=label == "quality",
        )
        sources[label] = (path, payload)
    quality_gate = _evaluate_quality(sources["quality"][1])
    issue_path = _required_path(args, "issue_data")
    issue_records = _validate_issue_evidence(_load_json_list(issue_path, label="issue_evidence"))
    log_dictionary_path = _required_path(args, "log_dictionary")
    log_dictionary = _load_json(log_dictionary_path, label="log_dictionary")
    log_sample_path = _validate_support_file(_required_path(args, "log_sample"), label="log_sample")
    log_evidence = _validate_log_evidence(log_dictionary, log_sample_path)
    if args.run_load and quality_gate["passed"] is not True:
        raise AcceptanceError("quality_gate_failed_before_paid_work")
    load_config = _load_config(args, selected) if args.run_load else None
    if args.run_load:
        asyncio.run(_preflight_load_service(cast(HttpLoadTestConfig, load_config)))
    if args.dry_run:
        return {
            "dry_run": True,
            "preflight_passed": True,
            "quality_gate_passed": quality_gate["passed"],
            "maximum_run_cost": str(_maximum_run_cost(selected)),
            "would_run_load": args.run_load,
        }
    performance = (
        asyncio.run(_run_load(args, selected))
        if args.run_load
        else _load_json(cast(Path, args.performance_report), label="performance")
    )
    return _build_release(
        args,
        selected,
        sources,
        performance,
        issue_records,
        log_dictionary_path,
        log_sample_path,
        log_evidence,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        result = run(args)
    except (AcceptanceError, OSError, ValueError) as error:
        print(
            json.dumps(
                {"accepted": False, "error": str(error)},
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    summary = result.get("summary")
    if isinstance(summary, Mapping) and summary.get("accepted") is not True:
        return 1
    if result.get("status") == "rejected":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["AcceptanceError", "main", "run", "verify_release"]
