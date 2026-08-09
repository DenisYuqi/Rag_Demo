from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_mvp import acceptance
from rag_mvp.performance.load_test import HttpLoadTestConfig, LoadScenario


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _selected() -> dict[str, object]:
    return {
        "schema_version": "mvp-selected-configuration-v1",
        "configuration_id": "bge-parent-child-v1",
        "profile": "bge-local",
        "retrieval_mode": "hybrid-rerank",
        "dataset_id": "original-pdf-acceptance",
        "scorer_contract": "advanced-v2",
        "models": {
            "embedding": "BAAI-bge-m3",
            "reranking": "BAAI-bge-reranker-v2-m3",
            "generation": "gpt-5-mini",
        },
        "parent_child_strategy": "structure-page-parent-child-v1",
        "selection_rationale": {
            "model": "Best passing quality and serving-cost trade-off.",
            "retrieval": "Hybrid improves bilingual exact-term recall.",
            "reranker": "Reranking improves top-context precision.",
            "cache": "Normal serving uses cache; official measurement bypasses it.",
        },
        "cost": {
            "remote": {
                "currency": "USD",
                "input_per_million": "0.25",
                "output_per_million": "2.00",
                "maximum_input_tokens": 4000,
                "maximum_output_tokens": 500,
                "pricing_source": "pinned-price-card-2026-08",
            },
            "local": {
                "currency": "USD",
                "hardware": "single-gpu-workstation",
                "hourly_rate": "0.50",
                "allocation_duration_seconds": "60",
            },
        },
        "limitations": [
            "The 100-request sample is MVP evidence, not a production capacity guarantee."
        ],
    }


def _metadata(*, scorer: bool = False) -> dict[str, object]:
    metadata: dict[str, object] = {
        "status": "completed",
        "configuration_id": "bge-parent-child-v1",
        "profile": "bge-local",
        "retrieval_mode": "hybrid-rerank",
        "dataset_id": "original-pdf-acceptance",
    }
    if scorer:
        metadata["scorer_contract"] = "advanced-v2"
    return metadata


def _quality() -> dict[str, object]:
    metrics = {
        "faithfulness": {"value": 0.91, "denominator": 24},
        "context-precision": {"value": 0.79, "denominator": 24},
        "answer-compliance": {"value": 0.95, "denominator": 24},
        "style-consistency": {"value": 0.88, "denominator": 24},
        "refusal-appropriateness": {"value": 0.94, "denominator": 24},
    }
    return {
        "report_id": "quality-24-bge",
        "metadata": _metadata(scorer=True),
        "metrics": metrics,
    }


def _comparison(report_id: str) -> dict[str, object]:
    return {"report_id": report_id, "metadata": _metadata()}


def _performance(*, slow_attempts: int = 0) -> dict[str, object]:
    attempts = []
    for index in range(100):
        slow = index >= 100 - slow_attempts
        attempts.append(
            {
                "attempt_id": f"attempt-{index + 1:03d}",
                "logical_request_id": f"request-{index + 1:03d}",
                "attempt_number": 1,
                "status": "succeeded",
                "latency_ms": 12_000 if slow else 500 + index,
                "instance_identity": "instance-one",
                "terminal_kind": "refusal" if index % 10 == 0 else "answer",
                "cache_status": {"request-policy": "bypass"},
            }
        )
    return {
        "report_version": "http-load-report-v1",
        "run_id": "mvp-load-100",
        "configured_concurrency": 5,
        "observed_peak_concurrency": 5,
        "instance_count": 1,
        "cache_policy": "bypass",
        "attempts": attempts,
        "token_totals": {"generation-input": 100_000, "generation-output": 20_000},
    }


def _issues() -> list[dict[str, object]]:
    return [
        {
            "issue_id": f"issue-{index}",
            "symptom": "A bounded stage timed out.",
            "root_cause": "The original deadline was below measured provider latency.",
            "exact_fix": "Calibrate the stage deadline while retaining the total deadline.",
            "primary_metric": f"stage-{index}-error-rate",
            "baseline_value": 1.0,
            "post_fix_value": 0.0,
            "relative_improvement_percent": 100.0,
            "passed": True,
        }
        for index in range(1, 3)
    ]


def _arguments(tmp_path: Path, *, slow_attempts: int = 0) -> list[str]:
    selected = _write_json(tmp_path / "selected.json", _selected())
    quality = _write_json(tmp_path / "quality.json", _quality())
    model = _write_json(tmp_path / "model.json", _comparison("model-comparison"))
    retrieval = _write_json(tmp_path / "retrieval.json", _comparison("retrieval-comparison"))
    cache = _write_json(tmp_path / "cache.json", _comparison("cache-comparison"))
    performance = _write_json(
        tmp_path / "performance.json", _performance(slow_attempts=slow_attempts)
    )
    issues = _write_json(tmp_path / "issues.json", _issues())
    log_dictionary = _write_json(
        tmp_path / "log-dictionary.json",
        {"fields": [{"name": name} for name in ("event", "level", "service", "timestamp")]},
    )
    log_sample = tmp_path / "log-sample.jsonl"
    log_sample.write_text(
        json.dumps(
            {
                "event": "request.completed",
                "level": "info",
                "service": "rag-mvp",
                "timestamp": "2026-08-09T00:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return [
        "--selected-config",
        str(selected),
        "--quality-report",
        str(quality),
        "--model-report",
        str(model),
        "--retrieval-report",
        str(retrieval),
        "--cache-report",
        str(cache),
        "--issue-data",
        str(issues),
        "--log-dictionary",
        str(log_dictionary),
        "--log-sample",
        str(log_sample),
        "--performance-report",
        str(performance),
        "--output",
        str(tmp_path / "release-v2"),
    ]


def test_builds_non_overwriting_release_and_verifies_offline(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)

    assert acceptance.main(arguments) == 0

    release = tmp_path / "release-v2"
    summary = json.loads((release / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    operations = json.loads((release / "operations-cost.json").read_text(encoding="utf-8"))
    crosswalk = json.loads((release / "pdf-crosswalk.json").read_text(encoding="utf-8"))
    assert summary["accepted"] is True
    assert summary["gates"]["performance"]["measured_requests"] == 100
    assert summary["gates"]["performance"]["successful_within_10s"] == 100
    assert operations["performance"]["latency_ms"]["p90"] <= 10_000
    assert manifest["status"] == "accepted"
    assert (release / "reproduce.md").is_file()
    assert not (release / "REPRODUCE.md").exists()
    assert (release / "evidence" / "issue-diagnosis.json").is_file()
    assert (release / "evidence" / "structured-log-sample.jsonl").is_file()
    assert all(item["status"] == "complete" for item in crosswalk["requirements"])
    assert all(
        part == part.casefold() for item in manifest["files"] for part in item["path"].split("/")
    )
    assert acceptance.main(["--offline-verify", str(release)]) == 0
    assert acceptance.main(arguments) == 2


def test_offline_verification_detects_tampering(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    assert acceptance.main(arguments) == 0
    release = tmp_path / "release-v2"
    (release / "summary.md").write_text("changed", encoding="utf-8")

    assert acceptance.main(["--offline-verify", str(release)]) == 2


def test_offline_verification_rejects_unmanifested_file(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    assert acceptance.main(arguments) == 0
    release = tmp_path / "release-v2"
    (release / "unexpected.txt").write_text("not recorded", encoding="utf-8")

    assert acceptance.main(["--offline-verify", str(release)]) == 2


@pytest.mark.parametrize(
    "relative",
    ("REPRODUCE.md", "evidence/Issue Report.json", "evidence\\report.json"),
)
def test_release_filename_policy_rejects_nonportable_names(relative: str) -> None:
    with pytest.raises(acceptance.AcceptanceError, match="packaged_filename_invalid"):
        acceptance._validate_release_relative_name(relative)


def test_content_scan_ignores_machine_decimals_but_rejects_string_pii(
    tmp_path: Path,
) -> None:
    machine = tmp_path / "machine.json"
    _write_json(
        machine,
        {
            "latency_ms": 2748.0203000013717,
            "cost": "0.00344382",
            "note": "No private data is present.",
        },
    )
    assert acceptance._scan_text_files(tmp_path) == (True, [])

    _write_json(machine, {"contact": "person@example.com"})
    passed, issues = acceptance._scan_text_files(tmp_path)
    assert passed is False
    assert issues == ["sensitive_content:machine.json"]


def test_failed_performance_gate_is_packaged_but_returns_nonzero(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path, slow_attempts=11)

    assert acceptance.main(arguments) == 1

    summary = json.loads((tmp_path / "release-v2" / "summary.json").read_text(encoding="utf-8"))
    assert summary["accepted"] is False
    assert summary["gates"]["performance"]["passed"] is False


def test_dry_run_preflights_without_creating_output(tmp_path: Path) -> None:
    arguments = [*_arguments(tmp_path), "--dry-run"]

    assert acceptance.main(arguments) == 0
    assert not (tmp_path / "release-v2").exists()


def test_load_preflight_rejects_cost_cap_before_service_call(tmp_path: Path) -> None:
    arguments = [
        *_arguments(tmp_path),
        "--run-load",
        "--base-url",
        "http://127.0.0.1:1",
        "--scenario-file",
        str(tmp_path / "missing-scenarios.json"),
        "--max-run-cost",
        "0.0001",
        "--dry-run",
    ]
    performance_index = arguments.index("--performance-report")
    del arguments[performance_index : performance_index + 2]

    assert acceptance.main(arguments) == 2
    assert not (tmp_path / "release-v2").exists()


def test_incompatible_quality_report_fails_before_output(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    quality_path = tmp_path / "quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["metadata"]["dataset_id"] = "different-dataset"
    _write_json(quality_path, quality)

    assert acceptance.main(arguments) == 2
    assert not (tmp_path / "release-v2").exists()


def test_partial_historical_comparison_is_reused_with_explicit_limitation(
    tmp_path: Path,
) -> None:
    selected = _selected()
    selected["comparison_compatibility"] = {
        "cache": {
            "status": "partial",
            "limitation": "Historical OpenAI-profile cache evidence is directional only.",
        }
    }
    payload = {
        "schema_version": "comparison-result-v1",
        "comparison_id": "historical-cache",
        "axis": "cache-behavior",
        "completed_at": "2026-08-07T00:00:00Z",
        "plan": {"fixed_identities": {"dataset_id": "original-pdf-acceptance"}},
        "candidates": [
            {"candidate_variant_id": "cache-cold", "status": "completed"},
            {"candidate_variant_id": "cache-warm", "status": "failed"},
        ],
        "recommendation": {"state": "no-recommendation"},
    }

    metadata = acceptance._validate_compatibility(
        payload,
        selected,
        label="cache",
        require_scorer=False,
    )

    assert metadata["dataset_id"] == "original-pdf-acceptance"
    assert metadata["compatibility_status"] == "partial"
    assert "directional only" in metadata["limitation"]


def test_runtime_configuration_identity_is_distinct_from_quality_source_identity() -> None:
    selected = _selected()
    selected["runtime_configuration_id"] = "runtime-bge-current"
    selected["source_configuration_ids"] = {"quality": "quality-bge-historical"}
    quality = _quality()
    quality["metadata"]["configuration_id"] = "quality-bge-historical"

    metadata = acceptance._validate_compatibility(
        quality,
        selected,
        label="quality",
        require_scorer=True,
    )

    assert metadata["configuration_id"] == "quality-bge-historical"


def test_fixed_sample_load_configuration_requires_no_retries() -> None:
    scenario = LoadScenario(
        scenario_id="policy",
        owner_id="acceptance-owner",
        question="What is the policy?",
    )
    config = HttpLoadTestConfig(
        run_id="fixed-sample",
        expected_configuration_id="bge-parent-child-v1",
        base_url="http://testserver",
        scenarios=(scenario,),
        concurrency=5,
        target_successes=90,
        max_attempts=100,
        exact_measured_attempts=100,
        retry_limit=0,
    )

    assert config.resolved_max_attempts == 100
    with pytest.raises(ValueError, match="do not permit transport retries"):
        config.model_copy(update={"retry_limit": 1}).model_validate(
            {**config.model_dump(), "retry_limit": 1}
        )
