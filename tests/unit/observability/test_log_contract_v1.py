from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from rag_mvp.evaluation.json_report import ReportWriteError
from rag_mvp.observability.log_contract_v1 import (
    LOG_DICTIONARY_FILENAME,
    LOG_EVENT_SCHEMA_VERSION,
    LOG_SAMPLE_FILENAME,
    LogDocumentationError,
    StructuredLogFieldDictionaryV1,
    build_log_field_dictionary_v1,
    canonical_log_dictionary_json,
    parse_log_dictionary_json,
    parse_log_sample_jsonl,
    render_log_sample_jsonl,
    sample_log_events_v1,
    validate_log_documentation,
    validate_log_documentation_files,
    write_log_documentation_v1,
)
from rag_mvp.safety.telemetry import DEFAULT_TELEMETRY_ALLOWLIST

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DICTIONARY_FIXTURE = _REPOSITORY_ROOT / "evaluations" / "logging" / LOG_DICTIONARY_FILENAME
_SAMPLE_FIXTURE = _REPOSITORY_ROOT / "evaluations" / "logging" / LOG_SAMPLE_FILENAME


def test_dictionary_exactly_documents_the_runtime_allowlist() -> None:
    dictionary = build_log_field_dictionary_v1()

    assert tuple(item.name for item in dictionary.fields) == tuple(
        sorted(DEFAULT_TELEMETRY_ALLOWLIST)
    )
    for field in dictionary.fields:
        assert field.meaning
        assert field.value_type
        assert field.cardinality
        assert field.presence_condition
        assert field.redaction_rule
    assert next(item for item in dictionary.fields if item.name == "duration_ms").unit == (
        "milliseconds"
    )
    assert next(item for item in dictionary.fields if item.name == "token_usage").unit == ("tokens")


def test_sample_is_canonical_filter_surviving_jsonl() -> None:
    dictionary = build_log_field_dictionary_v1()
    events = sample_log_events_v1()

    assert validate_log_documentation(dictionary, events) == events
    rendered = render_log_sample_jsonl(events, dictionary=dictionary)
    assert rendered.endswith("\n")
    assert parse_log_sample_jsonl(rendered) == events
    assert all(
        event["metadata"] == {"log_schema_version": LOG_EVENT_SCHEMA_VERSION}
        or event["event"] == "qa.refusal.completed"
        for event in events
    )


def test_committed_dictionary_and_sample_are_exact_canonical_fixtures() -> None:
    validate_log_documentation_files(_DICTIONARY_FIXTURE, _SAMPLE_FIXTURE)

    dictionary_text = _DICTIONARY_FIXTURE.read_text(encoding="utf-8")
    sample_text = _SAMPLE_FIXTURE.read_text(encoding="utf-8")
    dictionary = parse_log_dictionary_json(dictionary_text)
    events = parse_log_sample_jsonl(sample_text)
    assert dictionary_text == canonical_log_dictionary_json(dictionary) + "\n"
    assert sample_text == render_log_sample_jsonl(events, dictionary=dictionary)


def test_dictionary_rejects_missing_runtime_field() -> None:
    raw = build_log_field_dictionary_v1().model_dump(mode="json")
    raw["fields"] = raw["fields"][:-1]

    with pytest.raises(ValueError, match="complete telemetry allowlist"):
        StructuredLogFieldDictionaryV1.model_validate(raw)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("raw_prompt", "ignore prior instructions", "undocumented field"),
        ("operation", r"C:\\private\\document.pdf", "filesystem path"),
        ("request_id", "alice@example.com", "supported PII"),
        ("request_id", "AKIAABCDEFGHIJKLMNOP", "supported PII or a secret"),
        ("duration_ms", "184.5", "wrong documented type"),
    ],
)
def test_sample_rejects_content_paths_pii_secrets_and_wrong_types(
    field: str,
    value: object,
    message: str,
) -> None:
    events = list(deepcopy(sample_log_events_v1()))
    events[0][field] = value

    with pytest.raises(LogDocumentationError, match=message):
        validate_log_documentation(build_log_field_dictionary_v1(), events)


def test_nested_prohibited_content_field_is_rejected() -> None:
    events = list(deepcopy(sample_log_events_v1()))
    events[0]["metadata"] = {"retrieved_text": "not publishable"}

    with pytest.raises(LogDocumentationError, match="prohibited field"):
        validate_log_documentation(build_log_field_dictionary_v1(), events)


def test_raw_prose_cannot_hide_under_an_innocuous_metadata_key() -> None:
    events = list(deepcopy(sample_log_events_v1()))
    events[0]["metadata"] = {"note": "What is the policy text?"}

    with pytest.raises(LogDocumentationError, match="content-bearing string"):
        validate_log_documentation(build_log_field_dictionary_v1(), events)


def test_file_validation_rejects_noncanonical_but_parseable_jsonl(tmp_path: Path) -> None:
    dictionary_path, sample_path = write_log_documentation_v1(tmp_path)
    canonical = sample_path.read_text(encoding="utf-8")
    sample_path.write_text(canonical.replace("\n", "\r\n"), encoding="utf-8", newline="")

    with pytest.raises(LogDocumentationError, match="not canonical JSONL"):
        validate_log_documentation_files(dictionary_path, sample_path)


def test_writer_creates_pair_without_overwriting_prior_bytes(tmp_path: Path) -> None:
    dictionary_path, sample_path = write_log_documentation_v1(tmp_path)
    prior_dictionary = dictionary_path.read_bytes()
    prior_sample = sample_path.read_bytes()

    with pytest.raises(ReportWriteError, match="already exists"):
        write_log_documentation_v1(tmp_path)

    assert dictionary_path.read_bytes() == prior_dictionary
    assert sample_path.read_bytes() == prior_sample
