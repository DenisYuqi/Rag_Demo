from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import pytest

from rag_mvp.domain import (
    AcceptanceContract,
    AcceptanceMetricRequirement,
    ArtifactDescriptor,
    EvidenceComparisonOperator,
    GateResult,
    GateStatus,
    MetricObservation,
    MetricObservationStatus,
    ModelAttemptStatus,
    ModelRole,
    OperationsSummary,
    ProviderAttemptEvidence,
    TokenUsage,
)
from rag_mvp.evaluation.html_report import HtmlReportParityError
from rag_mvp.evaluation.html_report_v2 import (
    extract_embedded_report_v2,
    render_html_report_v2,
    verify_html_parity_v2,
)
from rag_mvp.evaluation.json_report import (
    REPORT_SCHEMA_URI,
    REPORT_SCHEMA_VERSION,
    ReportWriteError,
    load_json_report,
)
from rag_mvp.evaluation.report_dispatch import (
    ReportSchemaVersion,
    UnsupportedReportVersionError,
    canonical_versioned_report_document,
    load_evaluation_report,
    render_versioned_html_report,
    validate_versioned_report,
    verify_versioned_html_parity,
)
from rag_mvp.evaluation.report_v2 import (
    REPORT_SCHEMA_URI_V2,
    REPORT_SCHEMA_VERSION_V2,
    CategoryResultV2,
    EvaluationReportProvenanceV2,
    EvaluationReportV2,
    ReportV2ValidationError,
    canonical_report_document_v2,
    canonical_report_json_v2,
    load_json_report_v2,
    load_report_schema_v2,
    parse_report_v2,
    report_content_hash_v2,
    validate_report_v2,
    write_json_report_v2,
)
from rag_mvp.observability.costs_v2 import ExactPricingRateV2, PricingProvenanceV2
from rag_mvp.performance.evidence_v2 import build_performance_evidence_v2
from rag_mvp.performance.load_report import LoadAttempt, LoadAttemptStatus

_NOW = datetime(2026, 8, 7, 3, 4, 5, tzinfo=UTC)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_V2_FIXTURE = _REPOSITORY_ROOT / "tests" / "fixtures" / "evaluation-report-v2.json"
_SEALED_V1 = (
    _REPOSITORY_ROOT
    / "evaluations"
    / "releases"
    / "phase12_20260807t030340z-954bb3e2"
    / "evaluation-report.json"
)
_SEALED_V1_SHA256 = "2a0b29f075531a053ae981ba086297555694ae8740b2f2d2bdc3d48a9ce5bbf0"


def _pricing() -> PricingProvenanceV2:
    return PricingProvenanceV2.create(
        pricing_version="pricing-2026-08",
        currency="USD",
        rates=(
            ExactPricingRateV2(
                role=ModelRole.GENERATION,
                provider="primary",
                model="chat-v2",
                input_per_million=Decimal("2"),
                output_per_million=Decimal("8"),
            ),
        ),
        source_references=("https://pricing.example/provider/chat-v2",),
    )


def _attempt(identifier: str, latency_ms: float) -> LoadAttempt:
    provider_attempt = ProviderAttemptEvidence(
        operation_id=f"operation-{identifier}",
        role=ModelRole.GENERATION,
        provider="primary",
        model="chat-v2",
        status=ModelAttemptStatus.SUCCEEDED,
        latency_ms=latency_ms / 2,
        usage=TokenUsage(input_tokens=1_000, output_tokens=500),
    )
    started_at = _NOW + timedelta(milliseconds=latency_ms)
    return LoadAttempt(
        attempt_id=identifier,
        logical_request_id=f"logical-{identifier}",
        scenario_id="acceptance",
        status=LoadAttemptStatus.SUCCEEDED,
        started_at=started_at,
        completed_at=started_at + timedelta(milliseconds=latency_ms),
        latency_ms=latency_ms,
        http_status_code=200,
        request_id=f"request-{identifier}",
        trace_id=f"trace-{identifier}",
        instance_identity="instance-v2",
        terminal_kind="answer",
        provider_attempt_count=1,
        provider_attempts=(provider_attempt,),
        stage_timings_ms={"retrieval": latency_ms / 2, "total": latency_ms},
        token_counts={"generation-input": 1_000, "generation-output": 500},
        model_identities={"generation": "chat-v2"},
        cache_status={"request-policy": "bypass", "retrieval": "bypass"},
    )


def _quality_observations(*, denominator: int = 24) -> tuple[MetricObservation, ...]:
    definitions = (
        ("faithfulness", 23 / 24, 0.85),
        ("context-precision", 22 / 24, 0.70),
        ("answer-compliance", 23 / 24, 0.90),
        ("style", 1.0, 0.85),
        ("refusal-appropriateness", 23 / 24, 0.90),
    )
    return tuple(
        MetricObservation(
            metric_id=metric_id,
            unit="ratio",
            value=value,
            numerator=value * denominator,
            denominator=denominator,
            eligible=True,
            threshold=threshold,
            operator=EvidenceComparisonOperator.GREATER_THAN_OR_EQUAL,
            scorer_version=f"{metric_id}-v2",
            status=MetricObservationStatus.PASSED,
            evidence_references=(f"metric-{metric_id}",),
        )
        for metric_id, value, threshold in definitions
    )


def _report_v2() -> EvaluationReportV2:
    pricing = _pricing()
    performance = build_performance_evidence_v2(
        run_id="acceptance-run-v2",
        configuration_id="configuration-v2",
        warmup_attempts=(_attempt("warmup-1", 50),),
        measured_attempts=(_attempt("measured-1", 100), _attempt("measured-2", 200)),
        pricing=pricing,
        created_at=_NOW,
    )
    quality = _quality_observations()
    contract = AcceptanceContract(
        contract_id="original-pdf-acceptance",
        version="2.0.0",
        gate_profile_version="advanced-v2",
        dataset_schema_version="evaluation-dataset-v2",
        performance_schema_version="performance-evidence-v2",
        cost_schema_version="provider-cost-evidence-v2",
        metric_requirements=tuple(
            AcceptanceMetricRequirement(
                metric_id=item.metric_id,
                threshold=item.threshold,
                operator=item.operator,
                minimum_denominator=24,
            )
            for item in quality
            if item.threshold is not None and item.operator is not None
        ),
    )
    artifact = ArtifactDescriptor(
        schema_version="http-attempt-ledger-v2",
        artifact_id="attempt-ledger",
        format="jsonl",
        media_type="application/jsonl",
        relative_path="evidence/attempt-ledger.jsonl",
        sha256_digest="sha256:" + "a" * 64,
        byte_size=4_096,
        created_at=_NOW,
    )
    return EvaluationReportV2(
        report_id="evaluation-report-v2",
        run_id="acceptance-run-v2",
        configuration_id="configuration-v2",
        generated_at=_NOW,
        provenance=EvaluationReportProvenanceV2(
            dataset_id="acceptance-v2",
            dataset_version="2.0.0",
            dataset_content_hash="sha256:" + "1" * 64,
            corpus_id="original-pdf-corpus",
            corpus_version="2.0.0",
            corpus_content_hash="sha256:" + "2" * 64,
            case_set_content_hash="sha256:" + "3" * 64,
            experiment_plan_id="standard-acceptance-v2",
            experiment_plan_content_hash="sha256:" + "4" * 64,
            configuration_id="configuration-v2",
            code_revision="revision-v2",
            pricing_version=pricing.pricing_version,
            pricing_content_hash=pricing.digest,
        ),
        acceptance_contract=contract,
        acceptance_gate_id="advanced-quality",
        gates=(
            GateResult(
                gate_id="advanced-quality",
                profile_version="advanced-v2",
                status=GateStatus.PASSED,
                valid=True,
                passed=True,
                case_executions_complete=True,
                observations=quality,
            ),
        ),
        performance_evidence=performance,
        operations_summary=OperationsSummary(
            run_id="acceptance-run-v2",
            configuration_id="configuration-v2",
            observations=(
                MetricObservation(
                    metric_id="logical-attempt-count",
                    unit="count",
                    value=2,
                    numerator=2,
                    denominator=2,
                    eligible=True,
                    scorer_version="operations-v2",
                    status=MetricObservationStatus.OBSERVED,
                    evidence_references=("attempt-ledger",),
                ),
            ),
            source_artifact_ids=("attempt-ledger",),
            generated_at=_NOW,
        ),
        category_results=(
            CategoryResultV2(
                category_id="scanned-document",
                case_count=2,
                observations=(_quality_observations(denominator=2)[0],),
            ),
        ),
        artifacts=(artifact,),
        status=GateStatus.PASSED,
        accepted=True,
        limitations=("single-instance-measurement",),
    )


def test_schema_v2_is_packaged_meta_validated_and_defensively_copied() -> None:
    schema = load_report_schema_v2()

    assert schema["$id"] == REPORT_SCHEMA_URI_V2
    assert schema["properties"]["schema_version"]["const"] == REPORT_SCHEMA_VERSION_V2
    schema["title"] = "tampered"
    assert load_report_schema_v2()["title"] == "RAG MVP Evaluation Report v2"


def test_real_v2_fixture_is_typed_canonical_and_deterministic() -> None:
    report = _report_v2()
    document = validate_report_v2(report)

    assert parse_report_v2(document) == report
    assert canonical_report_document_v2(report) == _V2_FIXTURE.read_bytes()
    assert canonical_report_json_v2(dict(reversed(document.items()))) == canonical_report_json_v2(
        document
    )
    assert report_content_hash_v2(report).startswith("sha256:")
    assert document["performance_evidence"]["cost"]["complete"] is True


def test_v2_json_writer_is_immutable_and_round_trips(tmp_path: Path) -> None:
    target = tmp_path / "evaluation-report-v2.json"
    report = _report_v2()

    assert write_json_report_v2(report, target) == target.resolve()
    assert target.read_bytes() == canonical_report_document_v2(report)
    assert load_json_report_v2(target) == validate_report_v2(report)
    with pytest.raises(ReportWriteError, match="already exists"):
        write_json_report_v2(report, target)


def test_v2_semantic_validation_rejects_cross_section_tampering() -> None:
    payload = _report_v2().model_dump(mode="json", by_alias=True)
    payload["accepted"] = False

    with pytest.raises(ReportV2ValidationError) as error:
        validate_report_v2(payload)

    assert error.value.issues[0].keyword == "semantic"
    assert "acceptance-run-v2" not in str(error.value)


def test_v2_html_has_exact_embedded_and_visible_parity() -> None:
    report = _report_v2()
    html = render_html_report_v2(report)

    assert extract_embedded_report_v2(html) == validate_report_v2(report)
    verify_html_parity_v2(report, html)
    assert render_versioned_html_report(report) == html
    verify_versioned_html_parity(report, html)

    tampered = html.replace(
        'data-report-pointer="/accepted">PASS',
        'data-report-pointer="/accepted">FAIL',
        1,
    )
    with pytest.raises(HtmlReportParityError, match="visible HTML value"):
        verify_html_parity_v2(report, tampered)


def test_version_dispatch_reads_v2_and_rejects_unknown_versions() -> None:
    loaded = validate_versioned_report(_report_v2())

    assert loaded.schema_version is ReportSchemaVersion.V2
    assert loaded.is_legacy is False
    assert canonical_versioned_report_document(loaded.document) == _V2_FIXTURE.read_bytes()
    with pytest.raises(UnsupportedReportVersionError, match="unsupported"):
        validate_versioned_report({"schema_version": "99.0.0"})


def test_sealed_phase12_v1_report_remains_byte_identical_and_readable() -> None:
    before = _SEALED_V1.read_bytes()
    assert sha256(before).hexdigest() == _SEALED_V1_SHA256
    assert REPORT_SCHEMA_VERSION == "1.0.0"
    assert REPORT_SCHEMA_URI.endswith("evaluation-report-v1.schema.json")

    loaded = load_evaluation_report(_SEALED_V1)

    assert loaded.schema_version is ReportSchemaVersion.V1
    assert loaded.is_legacy is True
    assert load_json_report(_SEALED_V1) == loaded.document
    assert canonical_versioned_report_document(loaded.document) == before
    verify_versioned_html_parity(
        loaded.document,
        _SEALED_V1.with_suffix(".html").read_text(encoding="utf-8"),
    )
    assert _SEALED_V1.read_bytes() == before


def test_v2_dispatch_loads_committed_fixture_without_modifying_it() -> None:
    before = deepcopy(_V2_FIXTURE.read_bytes())

    loaded = load_evaluation_report(_V2_FIXTURE)

    assert loaded.schema_version is ReportSchemaVersion.V2
    assert loaded.document == validate_report_v2(_report_v2())
    assert _V2_FIXTURE.read_bytes() == before
