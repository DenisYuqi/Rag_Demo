"""Assemble schema-v1 evaluation reports from immutable runtime evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal, cast

from rag_mvp.domain._base import utc_now
from rag_mvp.domain.evaluation import (
    IssueEvidence,
    ModelAttemptStatus,
    ModelRole,
)
from rag_mvp.domain.evaluation import (
    ModelAttempt as PersistedModelAttempt,
)
from rag_mvp.domain.evaluation import (
    TokenUsage as PersistedTokenUsage,
)
from rag_mvp.domain.qa import StreamEventKind
from rag_mvp.evaluation.dataset import EvaluationDataset
from rag_mvp.evaluation.grounding_metrics import MetricName, MetricResult
from rag_mvp.evaluation.json_report import (
    REPORT_SCHEMA_URI,
    REPORT_SCHEMA_VERSION,
    JsonObject,
    prepare_report,
)
from rag_mvp.evaluation.runner import EvaluationRunManifest, PersistedCaseResult
from rag_mvp.evaluation.scoring import EvaluationScorecard
from rag_mvp.observability.costs import PricingCatalog, project_per_thousand_calls
from rag_mvp.providers.models import ModelAttempt as ProviderModelAttempt
from rag_mvp.providers.models import ProviderErrorCategory
from rag_mvp.safety.detectors import has_unclosed_private_key
from rag_mvp.safety.models import SensitiveKind
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, RedactionError, Redactor

type ReportAttempt = PersistedModelAttempt | ProviderModelAttempt
type IssueDirection = Literal["higher-is-better", "lower-is-better"]

_METRIC_KEYS: Mapping[MetricName, str] = {
    MetricName.FAITHFULNESS: "faithfulness",
    MetricName.CONTEXT_PRECISION: "context_precision",
    MetricName.ANSWER_COMPLETENESS: "answer_completeness",
    MetricName.STYLE_CONSISTENCY: "style_consistency",
    MetricName.REFUSAL_APPROPRIATENESS: "refusal_appropriateness",
}
_TIMEOUT_CATEGORIES = {
    ProviderErrorCategory.TIMEOUT,
    ProviderErrorCategory.DEADLINE_EXCEEDED,
}
_PROVIDER_METADATA_KEYS = frozenset({"adapter", "backend"})


class ReportBuildError(ValueError):
    """A stable, content-free report-assembly failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class IssueRunIdentity:
    """Compatibility evidence for one side of an issue comparison."""

    run_id: str
    dataset_version: str
    corpus_version: str
    case_ids_hash: str
    scorer_version: str
    eligible_cases: int
    configuration_id: str

    def as_report_value(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "dataset_version": self.dataset_version,
            "corpus_version": self.corpus_version,
            "case_ids_hash": self.case_ids_hash,
            "scorer_version": self.scorer_version,
            "eligible_cases": self.eligible_cases,
            "configuration_id": self.configuration_id,
        }


@dataclass(frozen=True, slots=True)
class IssueComparison:
    baseline: IssueRunIdentity
    post_fix: IssueRunIdentity

    def as_report_value(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.as_report_value(),
            "post_fix": self.post_fix.as_report_value(),
        }


@dataclass(frozen=True, slots=True)
class ReportIssueRecord:
    """Typed adapter around the domain issue evidence and paired-run identities."""

    evidence: IssueEvidence
    direction: IssueDirection
    comparison: IssueComparison

    def as_report_value(self) -> dict[str, object]:
        value = self.evidence.model_dump(mode="json")
        value.update(
            {
                "direction": self.direction,
                "comparison": self.comparison.as_report_value(),
                "passed": self.evidence.relative_improvement_percent >= 10,
            }
        )
        return value


type ReportIssueInput = ReportIssueRecord | Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _PrivacyScan:
    pii_matches: int = 0
    secret_matches: int = 0

    def __add__(self, other: _PrivacyScan) -> _PrivacyScan:
        return _PrivacyScan(
            pii_matches=self.pii_matches + other.pii_matches,
            secret_matches=self.secret_matches + other.secret_matches,
        )


@dataclass(frozen=True, slots=True)
class EvaluationReportBuilder:
    """Build a complete report from only versioned and persisted evidence."""

    pricing_catalog: PricingCatalog
    redactor: Redactor = field(default=DEFAULT_REDACTOR, repr=False)
    clock: Callable[[], datetime] = field(default=utc_now, repr=False)

    def build(
        self,
        *,
        dataset: EvaluationDataset,
        manifest: EvaluationRunManifest,
        results: Sequence[PersistedCaseResult],
        scorecard: EvaluationScorecard,
        attempts: Sequence[ReportAttempt] = (),
        issues: Sequence[ReportIssueInput] = (),
    ) -> JsonObject:
        """Validate cross-artifact identity and produce a redacted schema-v1 object."""

        persisted = _validated_inputs(
            dataset=dataset,
            manifest=manifest,
            results=results,
            scorecard=scorecard,
            pricing_catalog=self.pricing_catalog,
        )
        normalized_attempts = _normalize_attempts(
            attempts,
            run_id=manifest.run_id,
            request_ids={
                result.execution.request_id for result in persisted if result.execution is not None
            },
        )
        issue_values = tuple(_issue_value(issue) for issue in issues)
        identity = manifest.identity
        provider_models = _provider_models(identity.provider_identities, identity.model_identities)
        provider_metadata = {
            key: identity.provider_identities[key]
            for key in sorted(_PROVIDER_METADATA_KEYS & identity.provider_identities.keys())
        }
        configuration: dict[str, object] = {
            "configuration_id": identity.configuration_id,
            "manifest_hash": manifest.manifest_hash,
            "runner_version": manifest.runner_version,
            "scoring_version": scorecard.scoring_version,
            "quality_gate_version": scorecard.quality_gate.version,
            "generation_settings": identity.generation_settings,
            "provider_identities": identity.provider_identities,
            "model_identities": identity.model_identities,
        }

        metrics, thresholds = _metric_sections(dataset, scorecard)
        case_results, failed_cases = _case_sections(dataset, persisted, scorecard)
        performance = _performance_section(persisted)
        cost = _cost_section(
            normalized_attempts,
            pricing_catalog=self.pricing_catalog,
            run_id=manifest.run_id,
            successful_calls=sum(result.succeeded for result in persisted),
        )
        privacy = _privacy_section(
            persisted,
            configuration=configuration,
            issues=issue_values,
            redactor=self.redactor,
            run_id=manifest.run_id,
        )

        issues_passed = len(issue_values) >= 2 and all(
            issue.get("passed") is True for issue in issue_values
        )
        quality_valid = scorecard.quality_gate.valid
        quality_passed = scorecard.quality_gate.passed
        cost_complete = cast(bool, cost["complete"])
        performance_complete = performance["latency_evidence_count"] == len(persisted)
        run_valid = quality_valid and cost_complete and performance_complete
        privacy_passed = cast(bool, privacy["passed"])
        failures = _gate_failures(
            scorecard,
            cost_complete=cost_complete,
            performance_complete=performance_complete,
            privacy_passed=privacy_passed,
            issues_passed=issues_passed,
        )
        final_passed = run_valid and quality_passed and privacy_passed and issues_passed
        generated_at = self.clock()
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise ReportBuildError("report_clock_must_be_timezone_aware")

        report: dict[str, object] = {
            "$schema": REPORT_SCHEMA_URI,
            "schema_version": REPORT_SCHEMA_VERSION,
            "run_id": manifest.run_id,
            "generated_at": generated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "provenance": {
                "code_revision": identity.code_revision,
                "dataset": {
                    "id": identity.dataset_id,
                    "version": identity.dataset_version,
                    "content_hash": identity.dataset_hash,
                },
                "corpus": {
                    "version": identity.corpus_version,
                    "content_hash": identity.corpus_hash,
                },
                "configuration_id": identity.configuration_id,
                "prompt_versions": identity.prompt_versions,
                "provider_models": provider_models,
                **({"provider_metadata": provider_metadata} if provider_metadata else {}),
                "embedding_identity": identity.embedding_identity,
                "chunking_identity": identity.chunking_identity,
                "retrieval_configuration": identity.retrieval_configuration,
                "scorer_versions": identity.scorer_versions,
                "pricing_version": identity.pricing_version,
                "random_seeds": identity.random_seeds,
                "cache_policy": identity.cache_policy,
                "environment": identity.environment.model_dump(mode="json"),
            },
            "configuration": configuration,
            "thresholds": thresholds,
            "metrics": metrics,
            "case_results": case_results,
            "failed_cases": failed_cases,
            "performance": performance,
            "cost": cost,
            "privacy": privacy,
            "issues": list(issue_values),
            "gate": {
                "valid": run_valid,
                "quality_passed": quality_passed,
                "privacy_passed": privacy_passed,
                "reporting_passed": True,
                "issues_passed": issues_passed,
                "final_passed": final_passed,
                "failures": failures,
            },
        }
        try:
            return prepare_report(report, redactor=self.redactor)
        except (TypeError, ValueError, RecursionError) as error:
            raise ReportBuildError("report_contract_invalid") from error


def build_evaluation_report(
    *,
    dataset: EvaluationDataset,
    manifest: EvaluationRunManifest,
    results: Sequence[PersistedCaseResult],
    scorecard: EvaluationScorecard,
    attempts: Sequence[ReportAttempt],
    pricing_catalog: PricingCatalog,
    issues: Sequence[ReportIssueInput] = (),
    redactor: Redactor = DEFAULT_REDACTOR,
    clock: Callable[[], datetime] = utc_now,
) -> JsonObject:
    """Functional façade for service composition and bounded report workers."""

    return EvaluationReportBuilder(
        pricing_catalog=pricing_catalog,
        redactor=redactor,
        clock=clock,
    ).build(
        dataset=dataset,
        manifest=manifest,
        results=results,
        scorecard=scorecard,
        attempts=attempts,
        issues=issues,
    )


def case_ids_content_hash(case_ids: Sequence[str]) -> str:
    """Hash an ordered paired-run denominator for issue compatibility evidence."""

    values = tuple(case_ids)
    if not values or len(set(values)) != len(values) or any(not value for value in values):
        raise ReportBuildError("issue_case_ids_invalid")
    payload = json.dumps(values, ensure_ascii=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _validated_inputs(
    *,
    dataset: EvaluationDataset,
    manifest: EvaluationRunManifest,
    results: object,
    scorecard: EvaluationScorecard,
    pricing_catalog: PricingCatalog,
) -> tuple[PersistedCaseResult, ...]:
    if not isinstance(dataset, EvaluationDataset):
        raise ReportBuildError("evaluation_dataset_invalid")
    if not isinstance(manifest, EvaluationRunManifest):
        raise ReportBuildError("run_manifest_invalid")
    try:
        EvaluationRunManifest.model_validate(manifest.model_dump(mode="json"))
    except ValueError as error:
        raise ReportBuildError("run_manifest_invalid") from error
    if not isinstance(scorecard, EvaluationScorecard):
        raise ReportBuildError("scorecard_invalid")
    if not isinstance(pricing_catalog, PricingCatalog):
        raise ReportBuildError("pricing_catalog_invalid")
    if isinstance(results, str | bytes | bytearray) or not isinstance(results, Sequence):
        raise ReportBuildError("persisted_results_invalid")
    values = tuple(cast(Sequence[object], results))
    if not values or any(not isinstance(result, PersistedCaseResult) for result in values):
        raise ReportBuildError("persisted_results_invalid")
    persisted = cast(tuple[PersistedCaseResult, ...], values)

    dataset_case_ids = tuple(case.case_id for case in dataset.cases)
    result_case_ids = tuple(result.case_id for result in persisted)
    if manifest.run_id != scorecard.run_id or any(
        result.run_id != manifest.run_id for result in persisted
    ):
        raise ReportBuildError("report_run_identity_mismatch")
    if (
        manifest.case_ids != dataset_case_ids
        or scorecard.case_ids != dataset_case_ids
        or len(set(result_case_ids)) != len(result_case_ids)
        or set(result_case_ids) != set(dataset_case_ids)
    ):
        raise ReportBuildError("report_case_set_mismatch")

    identity = manifest.identity
    dataset_manifest = dataset.manifest
    if (
        identity.dataset_id != dataset_manifest.dataset_id
        or identity.dataset_version != dataset_manifest.version
        or identity.dataset_hash != dataset_manifest.content_hash
        or scorecard.dataset_id != dataset_manifest.dataset_id
        or scorecard.dataset_version != dataset_manifest.version
    ):
        raise ReportBuildError("report_dataset_identity_mismatch")
    if (
        identity.corpus_version != dataset_manifest.corpus.version
        or identity.corpus_hash != dataset_manifest.corpus.content_hash
    ):
        raise ReportBuildError("report_corpus_identity_mismatch")
    if identity.cache_policy != "bypass":
        raise ReportBuildError("report_cache_policy_invalid")
    if identity.pricing_version != pricing_catalog.version:
        raise ReportBuildError("report_pricing_identity_mismatch")
    if not scorecard.quality_gate.decisions:
        raise ReportBuildError("report_quality_decisions_missing")
    for aggregate in scorecard.aggregates:
        keys = (aggregate.metric.value, _metric_key(aggregate.metric))
        configured = next(
            (identity.scorer_versions[key] for key in keys if key in identity.scorer_versions),
            None,
        )
        if configured != aggregate.scorer_version:
            raise ReportBuildError("report_scorer_identity_mismatch")
    _provider_models(identity.provider_identities, identity.model_identities)
    registry = {result.case_id: result for result in persisted}
    return tuple(registry[case_id] for case_id in dataset_case_ids)


def _provider_models(
    providers: Mapping[str, str],
    models: Mapping[str, str],
) -> dict[str, object]:
    provider_roles = set(providers).difference(_PROVIDER_METADATA_KEYS)
    if (
        not provider_roles
        or provider_roles != set(models)
        or any(not key or not value for key, value in providers.items())
        or any(not key or not value for key, value in models.items())
    ):
        raise ReportBuildError("report_provider_model_identity_mismatch")
    return {
        role: {"provider": providers[role], "model": models[role]}
        for role in sorted(provider_roles)
    }


def _metric_sections(
    dataset: EvaluationDataset,
    scorecard: EvaluationScorecard,
) -> tuple[dict[str, object], dict[str, object]]:
    decisions = {decision.metric: decision for decision in scorecard.quality_gate.decisions}
    aggregate_values: dict[str, object] = {}
    thresholds: dict[str, object] = {}
    for metric in MetricName:
        decision = decisions[metric]
        key = _metric_key(metric)
        thresholds[key] = {
            "operator": decision.operator.value,
            "value": decision.threshold,
        }
        aggregate_values[key] = {
            "value": decision.value,
            "eligible_cases": decision.eligible_cases,
            "operator": decision.operator.value,
            "threshold": decision.threshold,
            "passed": decision.passed,
            "rationale": decision.rationale,
        }

    by_case = scorecard.per_case_by_id
    categories: dict[str, object] = {}
    for category in sorted({case.category.value for case in dataset.cases}):
        category_cases = tuple(case for case in dataset.cases if case.category.value == category)
        category_metrics: dict[str, object] = {}
        for metric in MetricName:
            decision = decisions[metric]
            results = tuple(by_case[case.case_id][metric] for case in category_cases)
            eligible = tuple(result for result in results if result.eligible)
            value = (
                math.fsum(cast(float, result.score) for result in eligible) / len(eligible)
                if eligible
                else None
            )
            category_metrics[_metric_key(metric)] = {
                "value": value,
                "eligible_cases": len(eligible),
                "operator": decision.operator.value,
                "threshold": decision.threshold,
                "passed": value is not None
                and _threshold_passed(value, decision.operator.value, decision.threshold),
                "rationale": (
                    "category_threshold_evaluated"
                    if value is not None
                    else "category_metric_has_no_eligible_cases"
                ),
            }
        categories[category] = {
            "case_count": len(category_cases),
            "metrics": category_metrics,
        }
    return {"aggregate": aggregate_values, "categories": categories}, thresholds


def _case_sections(
    dataset: EvaluationDataset,
    results: tuple[PersistedCaseResult, ...],
    scorecard: EvaluationScorecard,
) -> tuple[list[object], list[object]]:
    result_registry = {result.case_id: result for result in results}
    score_registry = scorecard.per_case_by_id
    decisions = {decision.metric: decision for decision in scorecard.quality_gate.decisions}
    case_values: list[object] = []
    failed_values: list[object] = []
    for case in dataset.cases:
        persisted = result_registry[case.case_id]
        metrics = score_registry[case.case_id]
        execution = persisted.execution
        outcome = (
            execution.event.kind.value
            if execution is not None
            and execution.event.kind
            in {StreamEventKind.ANSWER, StreamEventKind.REFUSAL, StreamEventKind.ERROR}
            else "error"
        )
        metric_values = {
            _metric_key(metric): _case_metric_value(metrics[metric]) for metric in MetricName
        }
        case_value: dict[str, object] = {
            "case_id": case.case_id,
            "category": case.category.value,
            "outcome": outcome,
            "metrics": metric_values,
        }
        if persisted.safe_error_code is not None:
            case_value["safe_error_code"] = persisted.safe_error_code
        trace_id = execution.event.diagnostics.trace_id if execution is not None else None
        if trace_id:
            case_value["trace_id"] = trace_id
        case_values.append(case_value)

        # Apply the exact per-metric operator instead of assuming every threshold
        # is inclusive.  Execution errors have no eligible scores, so they retain
        # all metric identities as explicit failed evidence.
        failed_metrics = tuple(
            _metric_key(metric)
            for metric in MetricName
            if metrics[metric].eligible
            and not _threshold_passed(
                cast(float, metrics[metric].score),
                decisions[metric].operator.value,
                decisions[metric].threshold,
            )
        )
        if case.case_id in scorecard.failed_case_ids:
            failed_metrics = tuple(_metric_key(metric) for metric in MetricName)
        if failed_metrics:
            failure: dict[str, object] = {
                "case_id": case.case_id,
                "category": case.category.value,
                "outcome": outcome,
                "failed_metrics": list(failed_metrics),
                "rationale": persisted.safe_error_code or "quality_threshold_failed",
            }
            if persisted.safe_error_code is not None:
                failure["safe_error_code"] = persisted.safe_error_code
            if trace_id:
                failure["trace_id"] = trace_id
            failed_values.append(failure)
    return case_values, failed_values


def _case_metric_value(result: MetricResult) -> dict[str, object]:
    references: list[str] = []
    for evidence in result.evidence:
        references.append(evidence.reference_id)
        references.extend(evidence.evidence_references)
    return {
        "eligible": result.eligible,
        "value": result.score,
        "rationale": result.rationale,
        "evidence_references": list(dict.fromkeys(references)),
    }


def _performance_section(results: tuple[PersistedCaseResult, ...]) -> dict[str, object]:
    executions = tuple(result.execution for result in results if result.execution is not None)
    complete_latencies = tuple(execution.latency_ms for execution in executions)
    stage_values: defaultdict[str, list[float]] = defaultdict(list)
    traces: list[str] = []
    for execution in executions:
        diagnostics = execution.event.diagnostics
        for stage, latency in diagnostics.stage_timings_ms.items():
            stage_values[stage].append(latency)
        if diagnostics.trace_id:
            traces.append(diagnostics.trace_id)
    attempts = len(results)
    successes = sum(result.succeeded for result in results)
    unknown_reasons: list[str] = []
    if not executions:
        unknown_reasons.append("complete-latency-unavailable")
    elif len(executions) != len(results):
        unknown_reasons.append("partial-complete-latency")
    return {
        "case_count": len(results),
        "latency_evidence_count": len(executions),
        "complete_latency_ms": (
            _latency_summary(complete_latencies) if complete_latencies else None
        ),
        "stage_latency_ms": {
            stage: _latency_summary(tuple(values)) for stage, values in sorted(stage_values.items())
        },
        "attempts": attempts,
        "successes": successes,
        "errors": attempts - successes,
        "error_rate": (attempts - successes) / attempts,
        "representative_trace_references": list(dict.fromkeys(traces)),
        "unknown_reasons": unknown_reasons,
    }


def _latency_summary(values: tuple[float, ...]) -> dict[str, object]:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ReportBuildError("report_latency_evidence_invalid")
    ordered = tuple(sorted(values))
    return {
        "count": len(ordered),
        "p50": _nearest_rank(ordered, 0.50),
        "p90": _nearest_rank(ordered, 0.90),
        "p99": _nearest_rank(ordered, 0.99),
        "max": ordered[-1],
    }


def _nearest_rank(ordered: tuple[float, ...], quantile: float) -> float:
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _normalize_attempts(
    attempts: object,
    *,
    run_id: str,
    request_ids: set[str],
) -> tuple[PersistedModelAttempt, ...]:
    if isinstance(attempts, str | bytes | bytearray) or not isinstance(attempts, Sequence):
        raise ReportBuildError("provider_attempts_invalid")
    values = tuple(cast(Sequence[object], attempts))
    normalized: list[PersistedModelAttempt] = []
    seen_ids: set[str] = set()
    for index, attempt in enumerate(values):
        if isinstance(attempt, PersistedModelAttempt):
            if attempt.run_id not in {None, run_id}:
                raise ReportBuildError("provider_attempt_run_mismatch")
            if attempt.run_id is None and attempt.request_id is None:
                raise ReportBuildError("provider_attempt_run_mismatch")
            if attempt.request_id is not None and attempt.request_id not in request_ids:
                raise ReportBuildError("provider_attempt_request_mismatch")
            resolved = attempt.model_copy(update={"run_id": run_id})
        elif isinstance(attempt, ProviderModelAttempt):
            if attempt.request_id not in request_ids:
                raise ReportBuildError("provider_attempt_request_mismatch")
            status = ModelAttemptStatus(attempt.status.value)
            if attempt.error_category in _TIMEOUT_CATEGORIES:
                status = ModelAttemptStatus.TIMED_OUT
            digest_input = (
                f"{attempt.request_id}\0{attempt.operation_id}\0{attempt.attempt_number}"
                f"\0{attempt.route_id}\0{index}"
            )
            resolved = PersistedModelAttempt(
                attempt_id=f"attempt_{hashlib.sha256(digest_input.encode()).hexdigest()[:24]}",
                operation_id=attempt.operation_id,
                request_id=attempt.request_id,
                run_id=run_id,
                role=ModelRole(attempt.role.value),
                provider=attempt.provider,
                model=attempt.model,
                status=status,
                attempt_number=attempt.attempt_number,
                fallback=attempt.is_fallback,
                latency_ms=attempt.latency_ms,
                usage=PersistedTokenUsage(
                    input_tokens=attempt.usage.input_tokens,
                    output_tokens=attempt.usage.output_tokens,
                    total_tokens_reported=attempt.usage.total_tokens,
                ),
                safe_error_category=(
                    attempt.error_category.value if attempt.error_category is not None else None
                ),
            )
        else:
            raise ReportBuildError("provider_attempts_invalid")
        if resolved.attempt_id in seen_ids:
            raise ReportBuildError("provider_attempt_id_duplicate")
        seen_ids.add(resolved.attempt_id)
        normalized.append(resolved)
    return tuple(normalized)


def _cost_section(
    attempts: tuple[PersistedModelAttempt, ...],
    *,
    pricing_catalog: PricingCatalog,
    run_id: str,
    successful_calls: int,
) -> dict[str, object]:
    aggregate = pricing_catalog.aggregate_run(attempts, run_id=run_id)
    reasons = [reason.value for reason in aggregate.unknown_reasons]
    per_thousand: Decimal | None = None
    complete = aggregate.complete
    assumptions: list[str] = []
    if successful_calls > 0:
        projection = project_per_thousand_calls(aggregate, successful_calls=successful_calls)
        per_thousand = projection.estimated_cost_per_1000
        assumptions = list(projection.assumptions)
        complete = projection.complete
        reasons = list(
            dict.fromkeys((*reasons, *(item.value for item in projection.unknown_reasons)))
        )
    else:
        complete = False
        reasons.append("no-successful-calls")
    return {
        "pricing_version": aggregate.pricing_version,
        "currency": aggregate.currency,
        "complete": complete,
        "input_tokens": aggregate.input_tokens,
        "output_tokens": aggregate.output_tokens,
        "known_cost": _decimal_value(aggregate.known_cost),
        "estimated_cost": _decimal_value(aggregate.estimated_cost) if complete else None,
        "cost_per_1000_calls": _decimal_value(per_thousand),
        "unknown_reasons": list(dict.fromkeys(reasons)),
        "attempt_count": aggregate.attempt_count,
        "successful_calls": successful_calls,
        "assumptions": assumptions,
    }


def _privacy_section(
    results: tuple[PersistedCaseResult, ...],
    *,
    configuration: Mapping[str, object],
    issues: tuple[dict[str, object], ...],
    redactor: Redactor,
    run_id: str,
) -> dict[str, object]:
    if not redactor.fully_configured:
        raise ReportBuildError("privacy_scanner_unavailable")
    persisted_scan = _scan_values(
        tuple(result.model_dump(mode="json") for result in results),
        redactor=redactor,
    )
    report_input_scan = _scan_values(
        ({"configuration": configuration, "issues": issues},),
        redactor=redactor,
    )
    total = persisted_scan + report_input_scan
    checks = [
        _privacy_check("persisted-results-pii-scan", persisted_scan.pii_matches, run_id),
        _privacy_check("persisted-results-secret-scan", persisted_scan.secret_matches, run_id),
        _privacy_check("report-input-pii-scan", report_input_scan.pii_matches, run_id),
        _privacy_check("report-input-secret-scan", report_input_scan.secret_matches, run_id),
    ]
    return {
        "passed": total.pii_matches == 0 and total.secret_matches == 0,
        "raw_supported_pii_matches": total.pii_matches,
        "raw_secret_matches": total.secret_matches,
        "checks": checks,
    }


def _scan_values(values: Sequence[object], *, redactor: Redactor) -> _PrivacyScan:
    pii_matches = 0
    secret_matches = 0
    try:
        for value in values:
            for text in _iter_string_values(value):
                spans = redactor.detect(text)
                pii_matches += sum(span.kind is not SensitiveKind.SECRET for span in spans)
                secret_matches += sum(span.kind is SensitiveKind.SECRET for span in spans)
                if has_unclosed_private_key(text):
                    secret_matches += 1
    except (RedactionError, TypeError, ValueError, RecursionError) as error:
        raise ReportBuildError("privacy_scan_failed") from error
    return _PrivacyScan(pii_matches=pii_matches, secret_matches=secret_matches)


def _iter_string_values(value: object) -> Sequence[str]:
    if isinstance(value, str):
        return (value,)
    if value is None or isinstance(value, bool | int | float):
        return ()
    if isinstance(value, Mapping):
        strings: list[str] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("privacy scan mappings require string keys")
            strings.append(key)
            strings.extend(_iter_string_values(item))
        return tuple(strings)
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        strings = []
        for item in value:
            strings.extend(_iter_string_values(item))
        return tuple(strings)
    raise TypeError("privacy scan input must be JSON-compatible")


def _privacy_check(check_id: str, matches: int, run_id: str) -> dict[str, object]:
    return {
        "check_id": check_id,
        "passed": matches == 0,
        "matches": matches,
        "evidence_references": [run_id],
    }


def _issue_value(issue: ReportIssueInput) -> dict[str, object]:
    if isinstance(issue, ReportIssueRecord):
        return issue.as_report_value()
    if not isinstance(issue, Mapping):
        raise ReportBuildError("issue_record_invalid")
    return dict(issue)


def _gate_failures(
    scorecard: EvaluationScorecard,
    *,
    cost_complete: bool,
    performance_complete: bool,
    privacy_passed: bool,
    issues_passed: bool,
) -> list[str]:
    failures: list[str] = []
    if not scorecard.quality_gate.valid:
        failures.append("quality-invalid")
    if scorecard.failed_case_ids:
        failures.append("case-execution-failed")
    failures.extend(
        f"quality-{_metric_key(metric)}-failed" for metric in scorecard.quality_gate.failed_metrics
    )
    if not cost_complete:
        failures.append("cost-incomplete")
    if not performance_complete:
        failures.append("performance-incomplete")
    if not privacy_passed:
        failures.append("privacy-failed")
    if not issues_passed:
        failures.append("issues-incomplete")
    return list(dict.fromkeys(failures))


def _metric_key(metric: MetricName) -> str:
    return _METRIC_KEYS[metric]


def _threshold_passed(value: float, operator: str, threshold: float) -> bool:
    return value > threshold if operator == ">" else value >= threshold


def _decimal_value(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")
