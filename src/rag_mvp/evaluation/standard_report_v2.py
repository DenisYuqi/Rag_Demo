"""Build and publish schema-v2 evidence for one standard Evaluation run."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import cast

from rag_mvp.domain import (
    AcceptanceContract,
    AcceptanceMetricRequirement,
    ArtifactDescriptor,
    OperationsSummary,
    UnavailableValue,
)
from rag_mvp.domain.evaluation import (
    ModelAttempt,
    ModelAttemptStatus,
    ModelRole,
    ProviderAttemptEvidence,
)
from rag_mvp.domain.qa import StreamEventKind
from rag_mvp.evaluation.artifacts_v2 import (
    ArtifactCatalogV2,
    ArtifactFormatV2,
    ArtifactManifestV2,
    ArtifactPayloadV2,
)
from rag_mvp.evaluation.dataset import EvaluationCaseV2, EvaluationDataset
from rag_mvp.evaluation.html_report_v2 import render_html_report_v2
from rag_mvp.evaluation.json_report import canonical_json_value
from rag_mvp.evaluation.operations_v2 import (
    OperationsEvidenceInputV2,
    build_operations_summary_v2,
    render_operations_csv,
    render_operations_text,
)
from rag_mvp.evaluation.pricing import OPENAI_PRICING_SOURCES
from rag_mvp.evaluation.quality_gate import AdvancedMetricName
from rag_mvp.evaluation.report_builder import case_ids_content_hash
from rag_mvp.evaluation.report_v2 import (
    CategoryResultV2,
    EvaluationReportProvenanceV2,
    EvaluationReportV2,
    FailedCaseEvidenceV2,
    canonical_report_document_v2,
)
from rag_mvp.evaluation.runner import (
    EvaluationRunManifest,
    PersistedCaseResult,
)
from rag_mvp.evaluation.scoring_v2 import AdvancedEvaluationScorecard
from rag_mvp.observability.costs import PricingCatalog
from rag_mvp.observability.costs_v2 import (
    ExactPricingRateV2,
    PricingProvenanceV2,
    RoleDirectionTokenTotalV2,
    TokenDirection,
)
from rag_mvp.performance.evidence_v2 import (
    PerformanceEvidenceV2,
    build_performance_evidence_v2,
)
from rag_mvp.performance.load_report import LoadAttempt, LoadAttemptStatus
from rag_mvp.safety.redactor import Redactor

ATTEMPT_LEDGER_ARTIFACT_ID = "attempt-ledger"
ATTEMPT_LEDGER_SCHEMA_VERSION = "evaluation-attempt-ledger-v2"
ATTEMPT_LEDGER_FILENAME = "attempt-ledger-v2.jsonl"
STANDARD_EVALUATION_PLAN_ID = "standard-evaluation-parent-child-v2"


class StandardReportV2Error(RuntimeError):
    """Stable, content-free schema-v2 report construction failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class StandardReportV2Evidence:
    report: EvaluationReportV2
    attempt_ledger: bytes


@dataclass(frozen=True, slots=True)
class PublishedStandardReportV2:
    report: EvaluationReportV2
    manifest: ArtifactManifestV2


def build_standard_report_v2(
    *,
    dataset: EvaluationDataset,
    manifest: EvaluationRunManifest,
    results: Sequence[PersistedCaseResult],
    scorecard: AdvancedEvaluationScorecard,
    attempts: Sequence[ModelAttempt],
    pricing_catalog: PricingCatalog,
) -> StandardReportV2Evidence:
    """Derive the complete dashboard report from immutable run evidence."""

    if not dataset.cases or any(not isinstance(case, EvaluationCaseV2) for case in dataset.cases):
        raise StandardReportV2Error("evaluation_v2_dataset_required")
    ordered_results = _ordered_results(manifest, results)
    if scorecard.run_id != manifest.run_id or scorecard.case_ids != manifest.case_ids:
        raise StandardReportV2Error("evaluation_v2_scorecard_identity_mismatch")
    attempts_by_request = _attempts_by_request(manifest.run_id, attempts)
    measured = tuple(
        _load_attempt(
            result,
            ordinal=ordinal,
            attempts=attempts_by_request.get(_request_id(result), ()),
            model_identities=manifest.identity.model_identities,
            configuration_id=manifest.identity.configuration_id,
        )
        for ordinal, result in enumerate(ordered_results, start=1)
    )
    bound_requests = {
        _request_id(result) for result in ordered_results if result.execution is not None
    }
    if set(attempts_by_request) - bound_requests:
        raise StandardReportV2Error("evaluation_v2_provider_attempt_unbound")

    generated_at = max(result.completed_at for result in ordered_results)
    pricing = _pricing_provenance(manifest, pricing_catalog)
    performance = build_performance_evidence_v2(
        run_id=manifest.run_id,
        configuration_id=manifest.identity.configuration_id,
        warmup_attempts=(),
        measured_attempts=measured,
        pricing=pricing,
        created_at=generated_at,
    )
    ledger = _attempt_ledger(measured)
    ledger_descriptor = ArtifactDescriptor(
        schema_version=ATTEMPT_LEDGER_SCHEMA_VERSION,
        artifact_id=ATTEMPT_LEDGER_ARTIFACT_ID,
        format="jsonl",
        media_type="application/x-ndjson",
        relative_path=ATTEMPT_LEDGER_FILENAME,
        sha256_digest=f"sha256:{hashlib.sha256(ledger).hexdigest()}",
        byte_size=len(ledger),
        created_at=generated_at,
    )
    operations = _operations_summary(scorecard, measured, performance)
    contract = AcceptanceContract(
        contract_id=dataset.manifest.dataset_id,
        version=getattr(dataset.manifest, "contract_version", dataset.manifest.version),
        gate_profile_version=scorecard.gate.profile_version,
        dataset_schema_version=str(dataset.manifest.schema_version),
        performance_schema_version=performance.schema_version,
        cost_schema_version=performance.cost.schema_version,
        metric_requirements=tuple(
            AcceptanceMetricRequirement(
                metric_id=observation.metric_id,
                threshold=observation.threshold,
                operator=observation.operator,
                minimum_denominator=1,
            )
            for observation in scorecard.gate.observations
            if observation.threshold is not None and observation.operator is not None
        ),
    )
    report = EvaluationReportV2(
        report_id="standard-evaluation-v2",
        run_id=manifest.run_id,
        configuration_id=manifest.identity.configuration_id,
        generated_at=generated_at,
        provenance=EvaluationReportProvenanceV2(
            dataset_id=dataset.manifest.dataset_id,
            dataset_version=dataset.manifest.version,
            dataset_content_hash=dataset.manifest.content_hash,
            corpus_id=dataset.corpus.manifest.snapshot_id,
            corpus_version=dataset.corpus.manifest.version,
            corpus_content_hash=dataset.corpus.manifest.content_hash,
            case_set_content_hash=case_ids_content_hash(manifest.case_ids),
            experiment_plan_id=STANDARD_EVALUATION_PLAN_ID,
            experiment_plan_content_hash=f"sha256:{manifest.manifest_hash}",
            configuration_id=manifest.identity.configuration_id,
            code_revision=manifest.identity.code_revision,
            pricing_version=pricing.pricing_version,
            pricing_content_hash=pricing.digest,
        ),
        acceptance_contract=contract,
        acceptance_gate_id=scorecard.gate.gate_id,
        gates=(scorecard.gate,),
        performance_evidence=performance,
        operations_summary=operations,
        category_results=tuple(
            CategoryResultV2(
                category_id=category.category_id,
                case_count=len(category.case_ids),
                observations=category.observations,
            )
            for category in scorecard.categories
        ),
        failed_cases=_failed_cases(dataset, ordered_results),
        artifacts=(ledger_descriptor,),
        status=scorecard.gate.status,
        accepted=scorecard.gate.valid and scorecard.gate.passed,
        limitations=(
            "single-evaluation-run",
            *(("local-model-cost-unavailable",) if not performance.cost.complete else ()),
        ),
    )
    return StandardReportV2Evidence(report=report, attempt_ledger=ledger)


def publish_standard_report_v2(
    evidence: StandardReportV2Evidence,
    *,
    run_root: Path,
    artifact_catalog: ArtifactCatalogV2,
    redactor: Redactor,
) -> PublishedStandardReportV2:
    """Persist source evidence and atomically publish the four UI artifacts."""

    ledger_path = run_root / ATTEMPT_LEDGER_FILENAME
    try:
        with ledger_path.open("xb") as stream:
            stream.write(evidence.attempt_ledger)
    except FileExistsError:
        try:
            persisted_ledger = ledger_path.read_bytes()
        except OSError:
            raise StandardReportV2Error("evaluation_artifact_integrity_failed") from None
        if persisted_ledger != evidence.attempt_ledger:
            raise StandardReportV2Error("evaluation_artifact_integrity_failed") from None
    report = evidence.report
    summary = report.operations_summary
    payloads = (
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
            content=(render_html_report_v2(report, redactor=redactor) + "\n").encode("utf-8"),
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
    )
    published = artifact_catalog.publish(
        run_id=report.run_id,
        configuration_id=report.configuration_id,
        payloads=payloads,
        created_at=report.generated_at,
    )
    return PublishedStandardReportV2(report=report, manifest=published.manifest)


def _ordered_results(
    manifest: EvaluationRunManifest,
    results: Sequence[PersistedCaseResult],
) -> tuple[PersistedCaseResult, ...]:
    by_case = {result.case_id: result for result in results}
    if (
        len(by_case) != len(results)
        or set(by_case) != set(manifest.case_ids)
        or any(result.run_id != manifest.run_id for result in results)
    ):
        raise StandardReportV2Error("evaluation_v2_case_identity_mismatch")
    return tuple(by_case[case_id] for case_id in manifest.case_ids)


def _attempts_by_request(
    run_id: str,
    attempts: Sequence[ModelAttempt],
) -> dict[str, tuple[ModelAttempt, ...]]:
    grouped: dict[str, list[ModelAttempt]] = {}
    attempt_ids: set[str] = set()
    for attempt in attempts:
        if (
            attempt.attempt_id in attempt_ids
            or attempt.run_id != run_id
            or attempt.request_id is None
        ):
            raise StandardReportV2Error("evaluation_v2_provider_attempt_identity_invalid")
        attempt_ids.add(attempt.attempt_id)
        grouped.setdefault(attempt.request_id, []).append(attempt)
    return {
        request_id: tuple(sorted(values, key=lambda item: (item.created_at, item.attempt_id)))
        for request_id, values in grouped.items()
    }


def _request_id(result: PersistedCaseResult) -> str:
    if result.execution is None:
        return f"unavailable-{result.case_id}"
    return result.execution.request_id


def _load_attempt(
    result: PersistedCaseResult,
    *,
    ordinal: int,
    attempts: Sequence[ModelAttempt],
    model_identities: Mapping[str, str],
    configuration_id: str,
) -> LoadAttempt:
    execution = result.execution
    latency_ms = result.logical_latency_ms
    if latency_ms is None:
        latency_ms = 0.0 if execution is None else execution.latency_ms
    provider_attempts = tuple(_provider_attempt(item) for item in attempts)
    token_counts: dict[str, int] = {}
    for attempt in provider_attempts:
        for direction, value in (
            ("input", attempt.usage.input_tokens),
            ("output", attempt.usage.output_tokens),
        ):
            if value is not None:
                key = f"{attempt.role.value}-{direction}"
                token_counts[key] = token_counts.get(key, 0) + value
    succeeded = result.succeeded and execution is not None
    terminal_kind = "error" if execution is None else execution.event.kind.value
    return LoadAttempt(
        attempt_id=f"evaluation-attempt-{ordinal}",
        logical_request_id=f"evaluation-request-{ordinal}",
        scenario_id=result.case_id,
        status=(LoadAttemptStatus.SUCCEEDED if succeeded else LoadAttemptStatus.TERMINAL_ERROR),
        started_at=result.completed_at - timedelta(milliseconds=latency_ms),
        completed_at=result.completed_at,
        latency_ms=latency_ms,
        http_status_code=200 if succeeded else 500,
        request_id=None if execution is None else execution.request_id,
        instance_identity=configuration_id,
        terminal_kind=terminal_kind,
        safe_error_code=None if succeeded else (result.safe_error_code or "evaluation-case-failed"),
        provider_attempt_count=len(provider_attempts),
        provider_failed_attempt_count=sum(
            attempt.status is not ModelAttemptStatus.SUCCEEDED for attempt in provider_attempts
        ),
        provider_unknown_usage_attempt_count=sum(
            attempt.usage.input_tokens is None
            or (attempt.role is not ModelRole.EMBEDDING and attempt.usage.output_tokens is None)
            for attempt in provider_attempts
        ),
        provider_evidence_complete=True,
        provider_attempts=provider_attempts,
        stage_timings_ms={"total": latency_ms},
        token_counts=dict(sorted(token_counts.items())),
        model_identities=dict(model_identities),
        cache_status={"request-policy": "bypass", "retrieval": "bypass"},
    )


def _provider_attempt(attempt: ModelAttempt) -> ProviderAttemptEvidence:
    return ProviderAttemptEvidence(
        operation_id=attempt.operation_id,
        attempt_number=attempt.attempt_number,
        role=attempt.role,
        provider=attempt.provider,
        model=attempt.model,
        status=attempt.status,
        fallback=attempt.fallback,
        latency_ms=attempt.latency_ms,
        safe_error_category=attempt.safe_error_category,
        usage=attempt.usage,
    )


def _pricing_provenance(
    manifest: EvaluationRunManifest,
    catalog: PricingCatalog,
) -> PricingProvenanceV2:
    rates: list[ExactPricingRateV2] = []
    for role in ModelRole:
        provider = manifest.identity.provider_identities.get(role.value)
        model = manifest.identity.model_identities.get(role.value)
        if provider is None or model is None:
            continue
        entry = catalog.lookup(provider, model)
        if entry is not None:
            rates.append(
                ExactPricingRateV2(
                    role=role,
                    provider=provider,
                    model=model,
                    input_per_million=entry.input_per_million,
                    output_per_million=entry.output_per_million,
                )
            )
    if not rates:
        raise StandardReportV2Error("evaluation_v2_pricing_provenance_unavailable")
    return PricingProvenanceV2.create(
        pricing_version=catalog.version,
        currency="USD",
        rates=rates,
        source_references=OPENAI_PRICING_SOURCES,
    )


def _attempt_ledger(attempts: Sequence[LoadAttempt]) -> bytes:
    content = "".join(
        canonical_json_value(attempt.model_dump(mode="json")) + "\n" for attempt in attempts
    )
    return content.encode("utf-8")


def _operations_summary(
    scorecard: AdvancedEvaluationScorecard,
    measured: tuple[LoadAttempt, ...],
    performance: PerformanceEvidenceV2,
) -> OperationsSummary:
    cost = performance.cost
    input_known, input_unknown = _token_evidence(cost.role_direction_tokens, TokenDirection.INPUT)
    output_known, output_unknown = _token_evidence(
        cost.role_direction_tokens,
        TokenDirection.OUTPUT,
    )
    refusals = sum(attempt.terminal_kind == StreamEventKind.REFUSAL.value for attempt in measured)
    answers = sum(attempt.terminal_kind == StreamEventKind.ANSWER.value for attempt in measured)
    answered_case_ids = {
        cast(str, attempt.scenario_id)
        for attempt in measured
        if attempt.terminal_kind == StreamEventKind.ANSWER.value
    }
    scored_answers = tuple(
        result
        for result in scorecard.compliance.case_results
        if result.case_id in answered_case_ids and result.eligible
    )
    return build_operations_summary_v2(
        OperationsEvidenceInputV2(
            run_id=scorecard.run_id,
            configuration_id=performance.configuration_id,
            total_logical_requests=len(measured),
            successful_logical_requests=sum(attempt.succeeded for attempt in measured),
            all_attempt_latency_ms=tuple(attempt.latency_ms for attempt in measured),
            provider_attempt_count=cost.provider_attempt_count,
            input_tokens=input_known,
            output_tokens=output_known,
            unknown_input_token_usage_attempt_count=input_unknown,
            unknown_output_token_usage_attempt_count=output_unknown,
            cache_hits=0,
            cache_eligible_lookups=0,
            refusals=refusals,
            answered_requests=answers,
            compliant_answers=sum(cast(int, result.numerator) for result in scored_answers),
            scored_answers=len(scored_answers),
            total_cost=cost.total_cost,
            currency=cost.pricing.currency,
            source_artifact_ids=(ATTEMPT_LEDGER_ARTIFACT_ID,),
            generated_at=performance.created_at,
        )
    )


def _token_evidence(
    values: Sequence[RoleDirectionTokenTotalV2],
    direction: TokenDirection,
) -> tuple[int, int]:
    selected = tuple(item for item in values if item.direction is direction)
    return (
        sum(item.known_tokens for item in selected),
        sum(item.unknown_usage_attempt_count for item in selected),
    )


def _failed_cases(
    dataset: EvaluationDataset,
    results: Sequence[PersistedCaseResult],
) -> tuple[FailedCaseEvidenceV2, ...]:
    case_by_id = {case.case_id: case for case in dataset.cases}
    failed: list[FailedCaseEvidenceV2] = []
    for result in results:
        if result.succeeded:
            continue
        case = cast(EvaluationCaseV2, case_by_id[result.case_id])
        failed.append(
            FailedCaseEvidenceV2(
                case_id=result.case_id,
                category_ids=tuple(
                    dict.fromkeys(
                        (case.category.value, *(tag.value for tag in case.challenge_tags))
                    )
                ),
                failed_metric_ids=tuple(metric.value for metric in AdvancedMetricName),
                safe_reason_code=result.safe_error_code or "evaluation-case-failed",
                trace_id=UnavailableValue(reason="trace-unavailable"),
            )
        )
    return tuple(failed)


__all__ = [
    "PublishedStandardReportV2",
    "StandardReportV2Error",
    "StandardReportV2Evidence",
    "build_standard_report_v2",
    "publish_standard_report_v2",
]
