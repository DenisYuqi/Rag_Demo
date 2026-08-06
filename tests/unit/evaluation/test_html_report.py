from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_json_report import valid_report

from rag_mvp.evaluation.html_report import (
    HtmlReportParityError,
    extract_embedded_report,
    render_html_report,
    verify_html_parity,
    write_html_report,
)
from rag_mvp.evaluation.json_report import prepare_report, write_json_report
from rag_mvp.evaluation.verify_report import ReportVerificationError, main, verify_report


def test_jinja_html_embeds_canonical_json_and_all_visible_values_match() -> None:
    report = valid_report()

    html = render_html_report(report)
    embedded = extract_embedded_report(html)

    assert html.startswith("<!doctype html>")
    assert embedded == prepare_report(report)
    assert 'id="evaluation-report-json"' in html
    assert 'data-report-pointer="/metrics"' in html
    verify_html_parity(report, html)


def test_html_parity_rejects_tampered_embedded_json() -> None:
    report = valid_report()
    html = render_html_report(report)
    tampered = html.replace(
        '"run_id":"run-baseline-001"',
        '"run_id":"run-baseline-002"',
        1,
    )

    with pytest.raises(HtmlReportParityError, match="embedded JSON differs"):
        verify_html_parity(report, tampered)


def test_html_parity_rejects_tampered_visible_value() -> None:
    report = valid_report()
    html = render_html_report(report)
    needle = 'data-report-pointer="/performance/complete_latency_ms/p90">200.0</span>'
    assert needle in html
    tampered = html.replace(needle, needle.replace("200.0", "199.0"), 1)

    with pytest.raises(HtmlReportParityError, match="visible HTML value differs"):
        verify_html_parity(report, tampered)


def test_html_rendering_redacts_content_and_blocks_script_termination() -> None:
    report = valid_report()
    configuration = report["configuration"]
    assert isinstance(configuration, dict)
    configuration["operator_note"] = "alice@example.com </script><script>unsafe()</script>"

    html = render_html_report(report)
    embedded = extract_embedded_report(html)

    assert "alice@example.com" not in html
    assert "</script><script>unsafe()" not in html
    embedded_configuration = embedded["configuration"]
    assert isinstance(embedded_configuration, dict)
    assert "alice@example.com" not in str(embedded_configuration["operator_note"])


def test_report_verifier_checks_canonical_json_and_adjacent_html(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = valid_report()
    json_path = tmp_path / "run-baseline-001.json"
    html_path = tmp_path / "run-baseline-001.html"
    write_json_report(report, json_path)
    write_html_report(report, html_path)

    result = verify_report(json_path)

    assert result.valid
    assert result.html_verified
    assert result.html_path == html_path.resolve()
    assert result.run_id == "run-baseline-001"
    assert not result.final_gate_passed
    assert main([str(json_path)]) == 0
    assert "report verified" in capsys.readouterr().out


def test_report_verifier_rejects_noncanonical_source_json(tmp_path: Path) -> None:
    report = valid_report()
    json_path = tmp_path / "run-baseline-001.json"
    html_path = tmp_path / "run-baseline-001.html"
    write_json_report(report, json_path)
    write_html_report(report, html_path)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    json_path.write_text(json.dumps(loaded, indent=2), encoding="utf-8")

    with pytest.raises(ReportVerificationError, match="not canonically serialized"):
        verify_report(json_path)
