from __future__ import annotations

from collections.abc import Sequence

from rag_mvp.safety import DetectionSpan, Redactor, TelemetryFilter


class BrokenDetector:
    name = "broken"

    def detect(self, text: str) -> Sequence[DetectionSpan]:
        del text
        raise RuntimeError("dependency leaked request content")


def test_filter_keeps_only_allowlisted_content_minimized_fields() -> None:
    fixture = "person@example.com"
    event = {
        "event_name": "qa.completed",
        "request_id": "request-1",
        "trace_id": "trace-1",
        "stage_duration_ms": 12.5,
        "counts": {"pii": 1},
        "question": f"tell me about {fixture}",
        "answer": f"contact {fixture}",
        "prompt": "raw prompt",
        "retrieved_text": "raw chunk",
        "authorization": "Bearer abcdefghijklmnop",
        "unknown": "not allowed",
    }

    filtered = TelemetryFilter().filter(event)
    assert filtered == {
        "event_name": "qa.completed",
        "request_id": "request-1",
        "trace_id": "trace-1",
        "stage_duration_ms": 12.5,
        "counts": {"pii": 1},
    }
    assert fixture not in repr(filtered)


def test_permitted_free_text_metadata_is_redacted_before_export() -> None:
    filtered = TelemetryFilter().filter(
        {
            "error_category": "provider_error person@example.com",
            "metadata": {"safe_note": "client 192.168.1.1"},
        }
    )
    assert filtered == {
        "error_category": "provider_error [REDACTED_EMAIL]",
        "metadata": {"safe_note": "client [REDACTED_IPV4]"},
    }


def test_custom_allowlist_cannot_enable_prohibited_content_fields() -> None:
    filtered = TelemetryFilter(allowlist={"event_name", "question"}).filter(
        {"event_name": "qa", "question": "person@example.com"}
    )
    assert filtered == {"event_name": "qa"}


def test_filter_drops_whole_event_when_redaction_is_uncertain() -> None:
    telemetry_filter = TelemetryFilter(redactor=Redactor((BrokenDetector(),)))
    assert telemetry_filter.filter({"event_name": "qa person@example.com"}) is None
    assert telemetry_filter.dropped_events == 1


def test_filter_drops_event_when_redactor_is_unavailable() -> None:
    telemetry_filter = TelemetryFilter(redactor=None)
    assert telemetry_filter.filter({"event_name": "qa"}) is None
    assert telemetry_filter.dropped_events == 1
