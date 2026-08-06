from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from rag_mvp.safety.redactor import DEFAULT_REDACTOR
from rag_mvp.safety.scan_artifacts import FixtureSet, load_fixture_set, main

pytestmark = pytest.mark.privacy

_REPOSITORY_ROOT = Path(__file__).parents[2]
_FIXTURE_PATH = _REPOSITORY_ROOT / "evaluations" / "privacy" / "supported-fixtures-v1.json"
_EXPECTED_REPORT_FIELDS = {"version", "hash", "categories", "files", "counts", "passed"}


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
