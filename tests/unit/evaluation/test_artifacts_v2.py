from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from rag_mvp.evaluation.artifacts_v2 import (
    ARTIFACT_DOWNLOAD_FILENAMES_V2,
    ARTIFACT_MEDIA_TYPES_BY_ID_V2,
    ARTIFACT_MEDIA_TYPES_V2,
    ArtifactBundleExistsError,
    ArtifactCatalogV2,
    ArtifactFormatV2,
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ArtifactPayloadV2,
    ArtifactPublicationError,
    allowed_artifact_media_types_v2,
    artifact_download_filename_v2,
    artifact_media_type_v2,
    verify_artifact_manifest,
)
from rag_mvp.evaluation.html_report_v2 import render_html_report_v2
from rag_mvp.evaluation.operations_v2 import (
    OperationsEvidenceInputV2,
    build_operations_summary_v2,
    render_operations_csv,
    render_operations_text,
)
from rag_mvp.evaluation.report_v2 import (
    EvaluationReportV2,
    canonical_report_document_v2,
    parse_report_v2,
)
from rag_mvp.observability.log_contract_v1 import (
    LOG_DICTIONARY_SCHEMA_VERSION,
    LOG_EVENT_SCHEMA_VERSION,
    build_log_field_dictionary_v1,
    canonical_log_dictionary_json,
    render_log_sample_jsonl,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_REPORT_FIXTURE = _REPOSITORY_ROOT / "tests" / "fixtures" / "evaluation-report-v2.json"
_PRIVACY_FIXTURE = _REPOSITORY_ROOT / "evaluations" / "privacy" / "supported-fixtures-v1.json"


def _report() -> EvaluationReportV2:
    base = parse_report_v2(json.loads(_REPORT_FIXTURE.read_text(encoding="utf-8")))
    summary = build_operations_summary_v2(
        OperationsEvidenceInputV2(
            run_id=base.run_id,
            configuration_id=base.configuration_id,
            total_logical_requests=2,
            successful_logical_requests=2,
            all_attempt_latency_ms=(100.0, 200.0),
            provider_attempt_count=2,
            input_tokens=2000,
            output_tokens=1000,
            cache_hits=2,
            cache_eligible_lookups=2,
            refusals=0,
            answered_requests=2,
            compliant_answers=2,
            scored_answers=2,
            total_cost=Decimal("0.012"),
            currency="USD",
            source_artifact_ids=("attempt-ledger",),
            generated_at=base.generated_at,
        )
    )
    raw = base.model_dump(mode="json", by_alias=True)
    raw["operations_summary"] = summary.model_dump(mode="json")
    return EvaluationReportV2.model_validate(raw)


def _payloads(*, include_documentation: bool = True) -> tuple[ArtifactPayloadV2, ...]:
    report = _report()
    return _payloads_for_report(report, include_documentation=include_documentation)


def _payloads_for_report(
    report: EvaluationReportV2,
    *,
    include_documentation: bool,
) -> tuple[ArtifactPayloadV2, ...]:
    summary = report.operations_summary
    values = [
        ArtifactPayloadV2(
            artifact_id="evaluation-report-json",
            schema_version=report.schema_version,
            format=ArtifactFormatV2.JSON,
            content=canonical_report_document_v2(report),
        ),
        ArtifactPayloadV2(
            artifact_id="evaluation-report-html",
            schema_version=report.schema_version,
            format=ArtifactFormatV2.HTML,
            content=(render_html_report_v2(report) + "\n").encode("utf-8"),
        ),
        ArtifactPayloadV2(
            artifact_id="operations-summary-txt",
            schema_version=summary.schema_version,
            format=ArtifactFormatV2.TXT,
            content=render_operations_text(summary).encode("utf-8"),
        ),
        ArtifactPayloadV2(
            artifact_id="operations-summary-csv",
            schema_version=summary.schema_version,
            format=ArtifactFormatV2.CSV,
            content=render_operations_csv(summary).encode("utf-8"),
        ),
    ]
    if include_documentation:
        dictionary = build_log_field_dictionary_v1()
        values.extend(
            (
                ArtifactPayloadV2(
                    artifact_id="structured-log-field-dictionary",
                    schema_version=LOG_DICTIONARY_SCHEMA_VERSION,
                    format=ArtifactFormatV2.JSON,
                    content=(canonical_log_dictionary_json(dictionary) + "\n").encode("utf-8"),
                ),
                ArtifactPayloadV2(
                    artifact_id="privacy-safe-log-sample",
                    schema_version=LOG_EVENT_SCHEMA_VERSION,
                    format=ArtifactFormatV2.JSONL,
                    content=render_log_sample_jsonl(dictionary=dictionary).encode("utf-8"),
                ),
            )
        )
    return tuple(values)


def _tampered_operations_report(case: str) -> EvaluationReportV2:
    report = _report()
    raw: dict[str, Any] = report.model_dump(mode="json", by_alias=True)
    observations = {item["metric_id"]: item for item in raw["operations_summary"]["observations"]}
    if case == "logical-count-and-cost-denominator":
        observations["total-logical-requests"]["value"] = 3.0
        observations["total-logical-requests"]["numerator"] = 3.0
        logical_cost = observations["cost-per-1000-logical-attempts"]
        logical_cost["denominator"] = 3
        logical_cost["value"] = 4.0
    elif case == "successful-count-and-cost-denominator":
        observations["successful-logical-requests"]["value"] = 1.0
        observations["successful-logical-requests"]["numerator"] = 1.0
        observations["answered-requests"]["value"] = 1.0
        observations["answered-requests"]["numerator"] = 1.0
        observations["scored-answers"]["value"] = 1.0
        observations["scored-answers"]["numerator"] = 1.0
        observations["compliant-answers"]["value"] = 1.0
        observations["compliant-answers"]["numerator"] = 1.0
        observations["refusal-rate"]["denominator"] = 1
        observations["answer-compliance-rate"]["numerator"] = 1.0
        observations["answer-compliance-rate"]["denominator"] = 1
        success_cost = observations["cost-per-1000-successes"]
        success_cost["denominator"] = 1
        success_cost["value"] = 12.0
    elif case == "latency-values":
        observations["all-attempt-latency-p50-ms"]["value"] = 101.0
        observations["all-attempt-latency-p50-ms"]["numerator"] = 101.0
        observations["all-attempt-latency-p95-ms"]["value"] = 201.0
        observations["all-attempt-latency-p95-ms"]["numerator"] = 201.0
    elif case == "latency-denominator":
        observations["all-attempt-latency-p50-ms"]["denominator"] = 3
        observations["all-attempt-latency-p95-ms"]["denominator"] = 3
    elif case == "token-total":
        observations["input-tokens"]["value"] = 2001.0
        observations["input-tokens"]["numerator"] = 2001.0
    elif case == "token-completeness":
        unavailable = {"status": "unavailable", "reason": "input-tokens-usage-incomplete"}
        observations["input-tokens"].update(
            {
                "value": unavailable,
                "numerator": 2000.0,
                "status": "unavailable",
                "eligible": True,
            }
        )
    elif case == "provider-attempt-denominator":
        observations["input-tokens"]["denominator"] = 3
        observations["output-tokens"]["denominator"] = 3
    elif case == "currency":
        observations["cost-per-1000-logical-attempts"]["unit"] = "EUR-per-1000-logical-attempts"
        observations["cost-per-1000-successes"]["unit"] = "EUR-per-1000-successes"
    elif case == "normalized-cost-values":
        for metric_id in (
            "cost-per-1000-logical-attempts",
            "cost-per-1000-successes",
        ):
            observations[metric_id]["value"] = 7.0
            observations[metric_id]["numerator"] = 0.014
    else:  # pragma: no cover - protects the parametrized case table
        raise AssertionError(f"unknown tamper case: {case}")
    return EvaluationReportV2.model_validate(raw)


def _catalog(tmp_path: Path) -> ArtifactCatalogV2:
    return ArtifactCatalogV2(tmp_path / "published")


def test_full_publication_exposes_verified_six_artifact_manifest_and_bytes(
    tmp_path: Path,
) -> None:
    report = _report()
    catalog = _catalog(tmp_path)
    published = catalog.publish(
        run_id=report.run_id,
        configuration_id=report.configuration_id,
        payloads=_payloads(),
        created_at=report.generated_at,
    )

    assert [item.artifact_id for item in published.manifest.artifacts] == list(
        ARTIFACT_DOWNLOAD_FILENAMES_V2
    )
    assert catalog.require(report.run_id) == published.manifest
    assert catalog.list() == (published.manifest,)
    for descriptor in published.manifest.artifacts:
        assert descriptor.relative_path == artifact_download_filename_v2(descriptor.artifact_id)
        assert descriptor.media_type in allowed_artifact_media_types_v2()
        resolved = catalog.resolve(report.run_id, descriptor.artifact_id)
        assert resolved.descriptor == descriptor
        assert not hasattr(resolved, "path")
        assert f"sha256:{hashlib.sha256(resolved.content).hexdigest()}" == (
            descriptor.sha256_digest
        )


def test_exactly_four_core_artifacts_remain_a_valid_strict_bundle(tmp_path: Path) -> None:
    report = _report()
    catalog = _catalog(tmp_path)
    manifest = catalog.publish(
        run_id=report.run_id,
        configuration_id=report.configuration_id,
        payloads=_payloads(include_documentation=False),
        created_at=report.generated_at,
    ).manifest

    assert len(manifest.artifacts) == 4
    assert {item.format for item in manifest.artifacts} == {"json", "html", "txt", "csv"}
    with pytest.raises(ArtifactNotFoundError, match="artifact-not-found"):
        catalog.resolve(report.run_id, "privacy-safe-log-sample")


def test_log_documentation_must_be_an_all_or_none_contract_pair(tmp_path: Path) -> None:
    report = _report()

    with pytest.raises(ArtifactPublicationError, match="optional log pair"):
        _catalog(tmp_path).publish(
            run_id=report.run_id,
            configuration_id=report.configuration_id,
            payloads=_payloads()[:-1],
            created_at=report.generated_at,
        )


def test_public_media_and_filename_contracts_are_immutable_and_server_owned() -> None:
    assert ARTIFACT_MEDIA_TYPES_V2[ArtifactFormatV2.JSONL] == "application/x-ndjson"
    assert artifact_media_type_v2("privacy-safe-log-sample") == "application/x-ndjson"
    with pytest.raises(TypeError):
        ARTIFACT_DOWNLOAD_FILENAMES_V2["new-artifact"] = "unsafe.txt"  # type: ignore[index]
    with pytest.raises(TypeError):
        ARTIFACT_MEDIA_TYPES_BY_ID_V2["privacy-safe-log-sample"] = (  # type: ignore[index]
            "text/html"
        )
    with pytest.raises(ArtifactNotFoundError, match="artifact-not-found"):
        artifact_download_filename_v2("../outside")
    with pytest.raises(ArtifactNotFoundError, match="artifact-not-found"):
        artifact_media_type_v2("../outside")


def test_publication_never_overwrites_a_prior_bundle(tmp_path: Path) -> None:
    report = _report()
    catalog = _catalog(tmp_path)
    first = catalog.publish(
        run_id=report.run_id,
        configuration_id=report.configuration_id,
        payloads=_payloads(),
        created_at=report.generated_at,
    )
    manifest_bytes = (first.bundle_root / "artifact-manifest.json").read_bytes()

    with pytest.raises(ArtifactBundleExistsError, match="already exists"):
        catalog.publish(
            run_id=report.run_id,
            configuration_id=report.configuration_id,
            payloads=_payloads(),
            created_at=report.generated_at,
        )
    assert (first.bundle_root / "artifact-manifest.json").read_bytes() == manifest_bytes


def test_noncanonical_but_parseable_operations_bytes_fail_before_publication(
    tmp_path: Path,
) -> None:
    report = _report()
    payloads = list(_payloads(include_documentation=False))
    original = payloads[3]
    payloads[3] = ArtifactPayloadV2(
        artifact_id=original.artifact_id,
        schema_version=original.schema_version,
        format=original.format,
        content=original.content.replace(b"\n", b"\r\n"),
    )

    with pytest.raises(ArtifactIntegrityError, match="cross-format"):
        _catalog(tmp_path).publish(
            run_id=report.run_id,
            configuration_id=report.configuration_id,
            payloads=payloads,
            created_at=report.generated_at,
        )


@pytest.mark.parametrize(
    "case",
    (
        "logical-count-and-cost-denominator",
        "successful-count-and-cost-denominator",
        "latency-values",
        "latency-denominator",
        "token-total",
        "token-completeness",
        "provider-attempt-denominator",
        "currency",
        "normalized-cost-values",
    ),
)
def test_cross_format_consistent_operations_divergence_from_evidence_fails_closed(
    tmp_path: Path,
    case: str,
) -> None:
    report = _tampered_operations_report(case)

    with pytest.raises(ArtifactIntegrityError, match="cross-format"):
        _catalog(tmp_path).publish(
            run_id=report.run_id,
            configuration_id=report.configuration_id,
            payloads=_payloads_for_report(report, include_documentation=False),
            created_at=report.generated_at,
        )


def test_tampered_content_fails_digest_verification_and_resolution(tmp_path: Path) -> None:
    report = _report()
    catalog = _catalog(tmp_path)
    published = catalog.publish(
        run_id=report.run_id,
        configuration_id=report.configuration_id,
        payloads=_payloads(),
        created_at=report.generated_at,
    )
    target = published.bundle_root / "operations-summary.txt"
    target.write_bytes(target.read_bytes() + b"tamper")

    with pytest.raises(ArtifactIntegrityError, match="digest or size"):
        catalog.require(report.run_id)
    with pytest.raises(ArtifactIntegrityError, match="digest or size"):
        catalog.resolve(report.run_id, "operations-summary-txt")


def test_bundle_rejects_unexpected_files_and_artifact_symlinks(tmp_path: Path) -> None:
    report = _report()
    catalog = _catalog(tmp_path)
    published = catalog.publish(
        run_id=report.run_id,
        configuration_id=report.configuration_id,
        payloads=_payloads(include_documentation=False),
        created_at=report.generated_at,
    )
    (published.bundle_root / "extra.txt").write_text("extra", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="unexpected files"):
        verify_artifact_manifest(published.bundle_root)


def test_in_catalog_run_symlink_alias_is_rejected_before_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report()
    catalog = _catalog(tmp_path)
    catalog.publish(
        run_id=report.run_id,
        configuration_id=report.configuration_id,
        payloads=_payloads(include_documentation=False),
        created_at=report.generated_at,
    )
    alias = catalog.root / "alias-run"
    original_is_symlink = Path.is_symlink

    def simulated_is_symlink(path: Path) -> bool:
        return path == alias or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", simulated_is_symlink)

    with pytest.raises(ArtifactIntegrityError, match="symbolic link"):
        catalog.get("alias-run")
    with pytest.raises(ArtifactIntegrityError, match="symbolic link"):
        catalog.resolve("alias-run", "evaluation-report-json")


def test_privacy_fixture_symlink_is_rejected_before_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_copy = tmp_path / "fixture.json"
    fixture_copy.write_bytes(_PRIVACY_FIXTURE.read_bytes())
    alias = tmp_path / "fixture-alias.json"
    resolved_fixture = fixture_copy.resolve()
    original_is_symlink = Path.is_symlink
    original_resolve = Path.resolve

    def simulated_is_symlink(path: Path) -> bool:
        return path == alias or original_is_symlink(path)

    def simulated_resolve(path: Path, strict: bool = False) -> Path:
        if path == alias:
            return resolved_fixture
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "is_symlink", simulated_is_symlink)
    monkeypatch.setattr(Path, "resolve", simulated_resolve)

    with pytest.raises(ArtifactIntegrityError, match="symbolic link"):
        ArtifactCatalogV2(tmp_path / "published", privacy_fixture_path=alias)


def test_resolver_returns_already_verified_bytes_without_a_second_content_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report()
    catalog = _catalog(tmp_path)
    published = catalog.publish(
        run_id=report.run_id,
        configuration_id=report.configuration_id,
        payloads=_payloads(include_documentation=False),
        created_at=report.generated_at,
    )
    target = (published.bundle_root / "operations-summary.txt").resolve()
    original_read_bytes = Path.read_bytes
    read_count = [0]

    def counted_read_bytes(path: Path) -> bytes:
        if path.resolve() == target:
            read_count[0] += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    resolved = catalog.resolve(report.run_id, "operations-summary-txt")

    # One integrity read plus one independent privacy scan; no third resolver read.
    assert read_count == [2]
    assert f"sha256:{hashlib.sha256(resolved.content).hexdigest()}" == (
        resolved.descriptor.sha256_digest
    )
