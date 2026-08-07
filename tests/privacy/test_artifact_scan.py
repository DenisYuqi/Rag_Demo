from __future__ import annotations

import base64
import csv
import hashlib
import html
import json
import shutil
from io import StringIO
from pathlib import Path

import pytest

from rag_mvp.safety.redactor import DEFAULT_REDACTOR
from rag_mvp.safety.scan_artifacts import FixtureSet, load_fixture_set, main

pytestmark = pytest.mark.privacy

_REPOSITORY_ROOT = Path(__file__).parents[2]
_FIXTURE_PATH = _REPOSITORY_ROOT / "evaluations" / "privacy" / "supported-fixtures-v1.json"
_EXPECTED_REPORT_FIELDS = {"version", "hash", "categories", "files", "counts", "passed"}
_COMPARISON_CSV_HEADER = (
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


@pytest.fixture(scope="module")
def fixture_set() -> FixtureSet:
    return load_fixture_set(_FIXTURE_PATH)


def _run_scan(
    capsys: pytest.CaptureFixture[str], *arguments: str
) -> tuple[int, dict[str, object], str]:
    exit_code = main(["--fixtures", str(_FIXTURE_PATH), *arguments])
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert captured.err == ""
    assert set(report) == _EXPECTED_REPORT_FIELDS
    return exit_code, report, captured.out


def _raw_fixture_artifact(fixture_set: FixtureSet) -> str:
    return "\n".join(f"acceptance_fixture={fixture.value}" for fixture in fixture_set.fixtures)


def _write_real_float_comparison_reports(
    root: Path,
    *,
    canonical_note: str | None = None,
) -> None:
    root.mkdir()
    metric = {
        "baseline_delta": -3.45700000123456,
        "denominator": 500,
        "gate_status": "passed",
        "metric_id": "latency-p90-ms",
        "numerator": 17.07899999746587,
        "status": "available",
        "unit": "ms",
        "value": 17.07899999746587,
    }
    candidate: dict[str, object] = {
        "cost_complete": True,
        "cost_unknown_reasons": [],
        "currency": "USD",
        "evidence_status": "available",
        "known_partial_cost": "0.02149750",
        "metrics": [metric],
        "reference": {"axis_value": "gpt-5.4", "variant_id": "variant-gpt"},
        "safe_error_code": None,
        "status": "completed",
        "total_cost": "0.02149750",
    }
    payload: dict[str, object] = {
        "candidates": [candidate],
        "comparison_id": "comparison-real-float",
    }
    if canonical_note is not None:
        payload["safe_note"] = canonical_note
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    digest = f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
    (root / "comparison-report.json").write_text(f"{canonical}\n", encoding="utf-8")

    html_row = (
        "<tr><td>variant-gpt</td><td>completed</td><td>available</td>"
        "<td>0.02149750</td><td>0.02149750</td><td>true</td><td></td>"
        "<td>USD</td><td>latency-p90-ms</td><td>ms</td>"
        "<td>17.07899999746587</td><td>500</td><td>-3.45700000123456</td>"
        "<td>available</td><td>passed</td></tr>"
    )
    encoded = base64.b64encode(canonical.encode("utf-8")).decode("ascii")
    html_report = (
        '<!doctype html><html><body><pre id="comparison-result-visible">'
        f"{html.escape(canonical)}</pre>"
        '<script id="comparison-result-json" type="application/json" '
        f'data-encoding="base64">{encoded}</script>'
        f"<table><tbody>{html_row}</tbody></table></body></html>\n"
    )
    (root / "comparison-report.html").write_text(html_report, encoding="utf-8")

    text_row = (
        "variant-gpt|completed|available||gpt-5.4|0.02149750|0.02149750|"
        "true||USD|latency-p90-ms|ms|17.07899999746587|17.07899999746587|"
        "500|-3.45700000123456|available|passed"
    )
    text_report = (
        "comparison-report-text-v1\n"
        f"canonical-json-sha256={digest}\ncanonical-json={canonical}\n"
        f"candidate-metrics:\n{text_row}\n"
    )
    (root / "comparison-report.txt").write_text(text_report, encoding="utf-8")

    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(_COMPARISON_CSV_HEADER)
    writer.writerow(("comparison-result", digest, canonical, *("",) * 18))
    writer.writerow(("candidate-metric", "", "", *text_row.split("|")))
    (root / "comparison-report.csv").write_text(stream.getvalue(), encoding="utf-8")


def test_raw_fixture_artifact_fails_without_disclosing_matches(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fixture_set: FixtureSet,
) -> None:
    artifact = tmp_path / "raw.json"
    artifact.write_text(_raw_fixture_artifact(fixture_set), encoding="utf-8")

    exit_code, report, output = _run_scan(capsys, str(artifact))

    assert exit_code != 0
    assert report["passed"] is False
    counts = report["counts"]
    assert isinstance(counts, dict)
    assert counts["fixture_matches"] >= len(fixture_set.fixtures)
    assert counts["detector_matches"] >= len(fixture_set.fixtures)
    assert all(fixture.value not in output for fixture in fixture_set.fixtures)
    assert str(tmp_path.parent) not in output


def test_fully_redacted_artifact_passes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fixture_set: FixtureSet,
) -> None:
    raw = _raw_fixture_artifact(fixture_set)
    artifact = tmp_path / "redacted.json"
    artifact.write_text(DEFAULT_REDACTOR.redact(raw).redacted_text, encoding="utf-8")

    exit_code, report, output = _run_scan(capsys, str(artifact))

    assert exit_code == 0
    assert report["passed"] is True
    assert all(fixture.value not in output for fixture in fixture_set.fixtures)


def test_fixture_artifact_is_not_implicitly_excluded(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fixture_set: FixtureSet,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    fixture_copy = artifact_root / "fixtures.json"
    shutil.copyfile(_FIXTURE_PATH, fixture_copy)
    (artifact_root / "safe.log").write_text("privacy scan completed", encoding="utf-8")

    rejected_code, rejected, rejected_output = _run_scan(capsys, str(artifact_root))
    accepted_code, accepted, accepted_output = _run_scan(
        capsys,
        "--exclude",
        str(fixture_copy),
        str(artifact_root),
    )

    assert rejected_code != 0
    assert rejected["passed"] is False
    assert accepted_code == 0
    assert accepted["passed"] is True
    assert all(
        fixture.value not in rejected_output + accepted_output for fixture in fixture_set.fixtures
    )


def test_data_volume_requires_an_explicit_exclusion(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fixture_set: FixtureSet,
) -> None:
    artifact_root = tmp_path / "release"
    data_volume = artifact_root / "data"
    data_volume.mkdir(parents=True)
    (artifact_root / "safe.log").write_text("release completed", encoding="utf-8")
    (data_volume / "raw.txt").write_text(_raw_fixture_artifact(fixture_set), encoding="utf-8")

    rejected_code, rejected, _ = _run_scan(capsys, str(artifact_root))
    accepted_code, accepted, _ = _run_scan(
        capsys,
        "--exclude",
        str(data_volume),
        str(artifact_root),
    )

    assert rejected_code != 0
    assert rejected["passed"] is False
    assert accepted_code == 0
    assert accepted["passed"] is True


def test_non_fixture_sensitive_value_is_found_by_the_installed_detectors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    non_fixture_email = "".join(("unlisted", "@", "example", ".", "invalid"))
    artifact = tmp_path / "diagnostics.json"
    artifact.write_text(non_fixture_email, encoding="utf-8")

    exit_code, report, output = _run_scan(capsys, str(artifact))

    assert exit_code != 0
    categories = report["categories"]
    assert isinstance(categories, dict)
    fixture_categories = categories["fixture"]
    detector_categories = categories["detector"]
    assert isinstance(fixture_categories, dict)
    assert isinstance(detector_categories, dict)
    assert sum(fixture_categories.values()) == 0
    assert detector_categories["email"] == 1
    assert non_fixture_email not in output


def test_json_numeric_metrics_do_not_trigger_privacy_detectors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    json_artifact = tmp_path / "metrics.json"
    json_artifact.write_text(
        '{"latency_ms":848.5567909999645,"queue_ms":0.9999999999763531,\n'
        '"stage_ms":749.556456999926}',
        encoding="utf-8",
    )
    jsonl_artifact = tmp_path / "attempts.jsonl"
    jsonl_artifact.write_text(
        '{"score":0.5835019999267388}\n{"latency_ms":1318.758312}\n',
        encoding="utf-8",
    )

    exit_code, report, _ = _run_scan(capsys, str(json_artifact), str(jsonl_artifact))

    assert exit_code == 0
    assert report["passed"] is True
    counts = report["counts"]
    categories = report["categories"]
    assert isinstance(counts, dict)
    assert isinstance(categories, dict)
    detector_categories = categories["detector"]
    assert isinstance(detector_categories, dict)
    assert counts["scanned"] == 2
    assert counts["detector_matches"] == 0
    assert all(count == 0 for count in detector_categories.values())


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_json_string_values_still_use_fail_safe_detectors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    suffix: str,
) -> None:
    artifact = tmp_path / f"unsafe{suffix}"
    artifact.write_text('{"contact":"415.5552671"}\n', encoding="utf-8")

    exit_code, report, _ = _run_scan(capsys, str(artifact))

    assert exit_code == 1
    assert report["passed"] is False
    categories = report["categories"]
    assert isinstance(categories, dict)
    detector_categories = categories["detector"]
    assert isinstance(detector_categories, dict)
    assert detector_categories["phone"] == 1


def test_mixed_json_log_and_prometheus_samples_ignore_only_typed_numbers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_artifact = tmp_path / "application.log"
    log_artifact.write_text(
        "runtime initialization complete\n"
        '{"duration_ms":848.5567909999645,"status":200}'
        '{"queue_ms":0.9999999999763531}\n',
        encoding="utf-8",
    )
    metrics_artifact = tmp_path / "metrics.prom"
    metrics_artifact.write_text(
        '# TYPE rag_latency_seconds gauge\nrag_latency_seconds{route="qa"} 0.9999999999763531\n',
        encoding="utf-8",
    )

    exit_code, report, _ = _run_scan(
        capsys,
        str(log_artifact),
        str(metrics_artifact),
    )

    assert exit_code == 0
    assert report["passed"] is True
    counts = report["counts"]
    assert isinstance(counts, dict)
    assert counts["detector_matches"] == 0


def test_prometheus_label_values_remain_fail_safe(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "unsafe.prom"
    artifact.write_text(
        'rag_contact_info{contact="415.5552671"} 1\n',
        encoding="utf-8",
    )

    exit_code, report, _ = _run_scan(capsys, str(artifact))

    assert exit_code == 1
    assert report["passed"] is False
    categories = report["categories"]
    assert isinstance(categories, dict)
    detector_categories = categories["detector"]
    assert isinstance(detector_categories, dict)
    assert detector_categories["phone"] == 1


def test_comparison_projections_ignore_only_real_typed_numeric_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reports = tmp_path / "comparison"
    _write_real_float_comparison_reports(reports)

    exit_code, report, _ = _run_scan(capsys, str(reports))

    assert exit_code == 0
    assert report["passed"] is True
    counts = report["counts"]
    assert isinstance(counts, dict)
    assert counts == {
        "targets": 1,
        "scanned": 4,
        "matched": 0,
        "excluded": 0,
        "fixture_matches": 0,
        "detector_matches": 0,
        "errors": 0,
    }


@pytest.mark.parametrize(
    ("artifact_name", "unsafe_value", "in_canonical"),
    [
        ("comparison-report.html", "projection.audit@example.invalid", True),
        ("comparison-report.html", "sk-projection-unsafe-0123456789abcdef", False),
        ("comparison-report.txt", r"D:\private\comparison-report.json", False),
        ("comparison-report.csv", "4242424242424242", False),
    ],
)
def test_comparison_projection_strings_and_noncanonical_text_remain_fail_safe(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    artifact_name: str,
    unsafe_value: str,
    in_canonical: bool,
) -> None:
    reports = tmp_path / "comparison"
    _write_real_float_comparison_reports(
        reports,
        canonical_note=unsafe_value if in_canonical else None,
    )
    if not in_canonical:
        with (reports / artifact_name).open("a", encoding="utf-8", newline="") as stream:
            stream.write(f"\n{unsafe_value}\n")

    exit_code, report, output = _run_scan(capsys, str(reports))

    assert exit_code == 1
    assert report["passed"] is False
    counts = report["counts"]
    assert isinstance(counts, dict)
    assert counts["matched"] >= 1
    assert counts["fixture_matches"] + counts["detector_matches"] >= 1
    assert unsafe_value not in output
