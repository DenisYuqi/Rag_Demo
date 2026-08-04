from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from rag_mvp.safety import (
    SAFE_UNAVAILABLE_MESSAGE,
    DetectionSpan,
    Redactor,
    SensitiveKind,
    redact_output,
    safe_redact_output,
)


class BrokenDetector:
    name = "broken"

    def detect(self, text: str) -> Sequence[DetectionSpan]:
        del text
        raise RuntimeError("unsafe source details")


class CitationModel(BaseModel):
    title: str
    preview: str


def test_nested_answer_citations_diagnostics_and_report_are_redacted() -> None:
    value = {
        "answer": "Email person@example.com, call 13800138000",
        "citations": [{"title": "owner person@example.com", "preview": "host 192.168.1.1"}],
        "diagnostics": {"error": "password=correct-horse-battery-staple"},
        "report": ("card 4111111111111111",),
    }

    output = redact_output(value)
    serialized = repr(output)
    for raw_value in (
        "person@example.com",
        "13800138000",
        "192.168.1.1",
        "correct-horse-battery-staple",
        "4111111111111111",
    ):
        assert raw_value not in serialized
    assert "[REDACTED_EMAIL]" in serialized
    assert "[REDACTED_SECRET]" in serialized


def test_dynamic_mapping_keys_are_redacted() -> None:
    assert redact_output({"person@example.com": "safe"}) == {"[REDACTED_EMAIL]": "safe"}


def test_pydantic_output_model_is_dumped_and_recursively_redacted() -> None:
    output = redact_output(CitationModel(title="person@example.com", preview="server 192.168.1.1"))
    assert output == {
        "title": "[REDACTED_EMAIL]",
        "preview": "server [REDACTED_IPV4]",
    }


def test_redaction_failure_withholds_the_entire_dynamic_object() -> None:
    value = {"safe": "would otherwise be visible", "unsafe": object()}
    assert safe_redact_output(value) == SAFE_UNAVAILABLE_MESSAGE
    assert safe_redact_output(value, redactor=None) == SAFE_UNAVAILABLE_MESSAGE
    assert (
        safe_redact_output("content", redactor=Redactor((BrokenDetector(),)))
        == SAFE_UNAVAILABLE_MESSAGE
    )


def test_excessive_nesting_fails_closed() -> None:
    assert safe_redact_output([[["content"]]], max_depth=1) == SAFE_UNAVAILABLE_MESSAGE


def test_redacted_result_does_not_need_raw_value_for_metadata() -> None:
    redactor = Redactor()
    result = redactor.redact("person@example.com")
    assert result.spans == (DetectionSpan(0, 18, SensitiveKind.EMAIL, "email", 70),)
