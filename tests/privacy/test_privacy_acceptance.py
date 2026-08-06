from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_mvp.safety.output import redact_output
from rag_mvp.safety.redactor import DEFAULT_REDACTOR
from rag_mvp.safety.scan_artifacts import PrivacyFixture, load_fixture_set
from rag_mvp.safety.streaming import SafeStream
from rag_mvp.safety.telemetry import TelemetryFilter

pytestmark = pytest.mark.privacy

_REPOSITORY_ROOT = Path(__file__).parents[2]
_FIXTURE_PATH = _REPOSITORY_ROOT / "evaluations" / "privacy" / "supported-fixtures-v1.json"
_FIXTURE_SET = load_fixture_set(_FIXTURE_PATH)
_SURFACES = ("answer", "citation", "diagnostics", "report", "log", "trace")


def _render_surface(surface: str, fixture: PrivacyFixture) -> str:
    if surface in {"log", "trace"}:
        rendered = TelemetryFilter().filter(
            {
                "event_name": f"privacy.{surface}.acceptance",
                "metadata": {"fixture": fixture.value},
            }
        )
        assert rendered is not None
    else:
        rendered = redact_output({surface: {"fixture": fixture.value}})
    return json.dumps(rendered, ensure_ascii=False, sort_keys=True)


def test_fixture_manifest_is_synthetic_and_covers_every_supported_class() -> None:
    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))

    assert payload["synthetic_only"] is True
    assert all(fixture["synthetic"] is True for fixture in payload["fixtures"])
    assert {fixture.fixture_id for fixture in _FIXTURE_SET.fixtures} == {
        "email",
        "phone_china",
        "phone_international",
        "chinese_national_id",
        "us_ssn",
        "luhn_payment_card",
        "ipv4_rfc5737",
        "ipv6_rfc3849",
        "api_key",
        "bearer_authorization",
        "password_assignment",
        "private_key_block",
    }


@pytest.mark.parametrize("fixture", _FIXTURE_SET.fixtures, ids=lambda item: item.fixture_id)
def test_every_fixture_is_detected_and_completely_replaced(fixture: PrivacyFixture) -> None:
    result = DEFAULT_REDACTOR.redact(fixture.value)

    assert result.detected
    assert fixture.value not in result.redacted_text
    assert "[REDACTED_" in result.redacted_text


@pytest.mark.parametrize("surface", _SURFACES)
@pytest.mark.parametrize("fixture", _FIXTURE_SET.fixtures, ids=lambda item: item.fixture_id)
def test_supported_values_never_escape_any_output_surface(
    fixture: PrivacyFixture,
    surface: str,
) -> None:
    rendered = _render_surface(surface, fixture)

    assert fixture.value not in rendered
    assert "[REDACTED_" in rendered


@pytest.mark.parametrize("fixture", _FIXTURE_SET.fixtures, ids=lambda item: item.fixture_id)
def test_supported_values_are_safe_across_every_two_delta_split(
    fixture: PrivacyFixture,
) -> None:
    source = f"Privacy acceptance value: {fixture.value}. Complete."

    for split_at in range(1, len(source)):
        stream = SafeStream()
        emitted = [
            *stream.push(source[:split_at]),
            *stream.push(source[split_at:]),
            *stream.finish(),
        ]
        rendered = "".join(emitted)

        assert fixture.value not in rendered, (fixture.fixture_id, split_at)
        assert rendered, (fixture.fixture_id, split_at, stream.failure_reason)
