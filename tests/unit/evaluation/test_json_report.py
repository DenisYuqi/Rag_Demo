from __future__ import annotations

import copy
from pathlib import Path

import pytest

from rag_mvp.evaluation.json_report import (
    REPORT_SCHEMA_URI,
    REPORT_SCHEMA_VERSION,
    ReportValidationError,
    ReportWriteError,
    canonical_report_document,
    canonical_report_json,
    load_json_report,
    load_report_schema,
    report_content_hash,
    validate_report,
    write_json_report,
)


def _metric(value: float, operator: str, threshold: float) -> dict[str, object]:
    passed = value > threshold if operator == ">" else value >= threshold
    return {
        "value": value,
        "eligible_cases": 4,
        "operator": operator,
        "threshold": threshold,
        "passed": passed,
    }


def valid_report() -> dict[str, object]:
    aggregate = {
        "faithfulness": _metric(0.9000001, ">", 0.85),
        "context_precision": _metric(0.7100001, ">", 0.70),
        "answer_completeness": _metric(0.8, ">=", 0.8),
        "style_consistency": _metric(0.81, ">=", 0.8),
        "refusal_appropriateness": _metric(0.82, ">=", 0.8),
    }
    return {
        "$schema": REPORT_SCHEMA_URI,
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": "run-baseline-001",
        "generated_at": "2026-08-07T01:02:03Z",
        "provenance": {
            "code_revision": "abc12345",
            "dataset": {
                "id": "mvp-v1",
                "version": "1.0.0",
                "content_hash": "sha256:" + "1" * 64,
            },
            "corpus": {
                "version": "corpus-v1",
                "content_hash": "sha256:" + "2" * 64,
                "index_revision": "revision-1",
            },
            "configuration_id": "config-001",
            "prompt_versions": {"generation": "generation-v1", "judge": "judge-v1"},
            "provider_models": {
                "generation": {"provider": "test-provider", "model": "model-v1"},
                "evaluation": {"provider": "test-provider", "model": "judge-v1"},
            },
            "embedding_identity": {"model": "embedding-v1", "dimensions": 8},
            "chunking_identity": {"version": "chunk-v1", "max_tokens": 256},
            "retrieval_configuration": {"mode": "hybrid", "top_k": 8},
            "scorer_versions": {
                "faithfulness": "faithfulness-v1",
                "context_precision": "context-precision-v1",
                "answer_completeness": "completeness-v1",
                "style_consistency": "style-v1",
                "refusal_appropriateness": "refusal-v1",
            },
            "pricing_version": "pricing-v1",
            "random_seeds": {"runner": 7, "judge": 0},
            "cache_policy": "bypass",
            "environment": {
                "python_version": "3.12.11",
                "platform": "test",
                "deployment": "single-process",
            },
        },
        "configuration": {"id": "config-001", "retrieval_mode": "hybrid"},
        "thresholds": {
            "faithfulness": {"operator": ">", "value": 0.85},
            "context_precision": {"operator": ">", "value": 0.70},
            "answer_completeness": {"operator": ">=", "value": 0.8},
            "style_consistency": {"operator": ">=", "value": 0.8},
            "refusal_appropriateness": {"operator": ">=", "value": 0.8},
        },
        "metrics": {
            "aggregate": aggregate,
            "categories": {
                "answerable-en": {
                    "case_count": 4,
                    "metrics": {"faithfulness": copy.deepcopy(aggregate["faithfulness"])},
                }
            },
        },
        "failed_cases": [],
        "performance": {
            "case_count": 4,
            "complete_latency_ms": {
                "count": 4,
                "p50": 100.0,
                "p90": 200.0,
                "p99": 250.0,
                "max": 300.0,
            },
            "stage_latency_ms": {"retrieval": {"count": 4, "p50": 20.0, "p90": 30.0, "max": 40.0}},
            "attempts": 4,
            "successes": 4,
            "errors": 0,
            "error_rate": 0.0,
            "concurrency": 1,
            "instance_count": 1,
            "representative_trace_references": ["trace-001"],
        },
        "cost": {
            "pricing_version": "pricing-v1",
            "currency": "USD",
            "complete": True,
            "input_tokens": 1000,
            "output_tokens": 400,
            "known_cost": "0.0042",
            "estimated_cost": "0.0042",
            "cost_per_1000_calls": "1.05",
            "unknown_reasons": [],
            "attempt_count": 4,
            "successful_calls": 4,
            "assumptions": ["all attempts are included"],
        },
        "privacy": {
            "passed": True,
            "raw_supported_pii_matches": 0,
            "raw_secret_matches": 0,
            "checks": [
                {
                    "check_id": "captured-output-scan",
                    "passed": True,
                    "matches": 0,
                    "evidence_references": ["privacy-artifact-001"],
                }
            ],
        },
        "issues": [],
        "gate": {
            "valid": True,
            "quality_passed": True,
            "privacy_passed": True,
            "reporting_passed": True,
            "issues_passed": False,
            "final_passed": False,
            "failures": ["issues-incomplete"],
        },
    }


def test_schema_v1_validates_all_required_report_sections() -> None:
    schema = load_report_schema()
    validated = validate_report(valid_report())

    assert schema["$id"] == REPORT_SCHEMA_URI
    assert validated["schema_version"] == REPORT_SCHEMA_VERSION
    assert validated["provenance"]
    assert validated["metrics"]
    assert validated["performance"]
    assert validated["cost"]
    assert validated["privacy"]
    assert validated["issues"] == []


@pytest.mark.parametrize(
    "section",
    ["provenance", "metrics", "performance", "cost", "privacy", "issues"],
)
def test_schema_rejects_each_missing_evidence_section(section: str) -> None:
    report = valid_report()
    del report[section]

    with pytest.raises(ReportValidationError) as captured:
        validate_report(report)

    assert captured.value.issues[0].keyword == "required"
    assert section not in str(captured.value)


def test_semantic_validation_uses_unrounded_values_and_rejects_gate_disagreement() -> None:
    report = valid_report()
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    aggregate = metrics["aggregate"]
    assert isinstance(aggregate, dict)
    faithfulness = aggregate["faithfulness"]
    assert isinstance(faithfulness, dict)
    faithfulness["value"] = 0.85
    faithfulness["passed"] = True

    with pytest.raises(ReportValidationError) as captured:
        validate_report(report)

    assert any(issue.keyword == "decision" for issue in captured.value.issues)


def test_semantic_validation_rejects_false_privacy_and_cost_claims() -> None:
    report = valid_report()
    privacy = report["privacy"]
    cost = report["cost"]
    assert isinstance(privacy, dict)
    assert isinstance(cost, dict)
    privacy["raw_secret_matches"] = 1
    cost["unknown_reasons"] = ["pricing-not-found"]

    with pytest.raises(ReportValidationError) as captured:
        validate_report(report)

    keywords = {issue.keyword for issue in captured.value.issues}
    assert {"privacy-parity", "cost-completeness"}.issubset(keywords)


def test_canonical_json_is_stable_and_safe_for_html_embedding() -> None:
    report = valid_report()
    configuration = report["configuration"]
    assert isinstance(configuration, dict)
    configuration["display_note"] = "</script><script>unsafe</script>"

    first = canonical_report_json(report)
    second = canonical_report_json(dict(reversed(tuple(report.items()))))

    assert first == second
    assert "</script>" not in first
    assert "\\u003c/script\\u003e" in first


def test_writer_redacts_then_publishes_immutable_canonical_json(tmp_path: Path) -> None:
    report = valid_report()
    configuration = report["configuration"]
    assert isinstance(configuration, dict)
    configuration["operator_note"] = "contact alice@example.com"
    output = tmp_path / "run-baseline-001.json"

    written = write_json_report(report, output)
    loaded = load_json_report(written)

    assert written == output.resolve()
    assert "alice@example.com" not in output.read_text(encoding="utf-8")
    assert output.read_bytes() == canonical_report_document(loaded)
    assert report_content_hash(loaded).removeprefix("sha256:")
    with pytest.raises(ReportWriteError, match="already exists"):
        write_json_report(report, output)
