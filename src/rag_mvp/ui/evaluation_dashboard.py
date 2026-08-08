"""Typed, path-free rendering for the standard Evaluation workbench views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from html import escape
from urllib.parse import quote

from rag_mvp.domain import MetricObservation, UnavailableValue
from rag_mvp.domain.evaluation import EvaluationRun, EvaluationRunStatus
from rag_mvp.evaluation.application import (
    EvaluationArtifactManifest,
    EvaluationDatasetCatalogEntry,
    EvaluationPlanCatalogEntry,
    EvaluationRunSummary,
    FailedCaseDiagnostic,
    ReleaseEvidenceSnapshot,
    ResolvedEvaluationArtifact,
)
from rag_mvp.evaluation.json_report import decode_json_report
from rag_mvp.evaluation.operations_v2 import (
    OPERATIONS_METRIC_ORDER,
    parse_operations_csv,
    parse_operations_text,
    verify_operations_parity,
)
from rag_mvp.evaluation.report_v2 import EvaluationReportV2, parse_report_v2
from rag_mvp.observability.costs_v2 import EvidenceAvailability, TokenDirection
from rag_mvp.performance.load_report import LoadAttemptStatus
from rag_mvp.safety.output import redact_output
from rag_mvp.safety.redactor import Redactor

from .models import BrowserSessionState, EvaluationRender
from .services import EvaluationGateway

_ACTIVE_RUN_STATUSES = frozenset({EvaluationRunStatus.QUEUED, EvaluationRunStatus.RUNNING})
_REPORT_ARTIFACT_ID = "evaluation-report-json"
_OPERATIONS_TEXT_ARTIFACT_ID = "operations-summary-txt"
_OPERATIONS_CSV_ARTIFACT_ID = "operations-summary-csv"
_API_PREFIX = "/api/v1"
_QUALITY_METRICS = (
    "faithfulness",
    "context-precision",
    "answer-compliance",
    "style",
    "refusal-appropriateness",
)
_PERFORMANCE_METRICS = (
    "total-logical-requests",
    "successful-logical-requests",
    "success-rate",
    "all-attempt-latency-p50",
    "all-attempt-latency-p90",
    "all-attempt-latency-p95",
    "all-attempt-latency-p99",
    "successful-only-latency-p50",
    "successful-only-latency-p90",
    "successful-only-latency-p95",
    "successful-only-latency-p99",
    "observed-peak-concurrency",
    "error-rate",
    "timeout-rate",
)
_COST_METRICS = (
    "provider-attempt-count",
    "input-tokens",
    "output-tokens",
    "total-cost",
    "cost-per-1000-logical-attempts",
    "cost-per-1000-successes",
)
_CACHE_METRICS = ("cache-hits", "cache-eligible-lookups", "cache-hit-rate")
_REFUSAL_METRICS = (
    "refusals",
    "answered-requests",
    "refusal-rate",
    "compliant-answers",
    "scored-answers",
    "answer-compliance-rate",
)


class EvaluationDashboardError(ValueError):
    """A content-free rendering failure safe to collapse at the callback boundary."""


def supports_typed_dashboard(service: object) -> bool:
    """Return whether the gateway exposes every read-only dashboard boundary."""

    return all(
        callable(getattr(service, name, None))
        for name in (
            "datasets",
            "plans",
            "summary",
            "artifact_manifest",
            "artifact",
        )
    )


def dataset_key(item: EvaluationDatasetCatalogEntry) -> str:
    return f"{item.dataset_id}::{item.dataset_version}"


def plan_key(item: EvaluationPlanCatalogEntry) -> str:
    return f"{item.plan_id}::{item.dataset_id}::{item.dataset_version}"


def resolve_registered_plan(
    service: EvaluationGateway,
    selected_dataset_key: str | None,
    selected_plan_key: str | None,
) -> tuple[EvaluationDatasetCatalogEntry, EvaluationPlanCatalogEntry]:
    """Resolve UI tokens against the live allowlisted catalog before launch."""

    datasets = _datasets(service)
    plans = _plans(service)
    dataset = next(
        (item for item in datasets if dataset_key(item) == selected_dataset_key),
        None,
    )
    plan = next((item for item in plans if plan_key(item) == selected_plan_key), None)
    if (
        dataset is None
        or plan is None
        or (plan.dataset_id, plan.dataset_version) != (dataset.dataset_id, dataset.dataset_version)
    ):
        raise EvaluationDashboardError("evaluation_catalog_selection_invalid")
    return dataset, plan


def render_evaluation_dashboard(
    service: EvaluationGateway,
    *,
    redactor: Redactor,
    state: BrowserSessionState,
    selected_dataset_key: str | None = None,
    selected_plan_key: str | None = None,
    selected_run_id: str | None = None,
    retrieval_profile: str | None = None,
) -> EvaluationRender:
    """Build all four standard-evaluation views from typed persisted evidence."""

    if not redactor.fully_configured:
        raise EvaluationDashboardError("evaluation_redaction_unavailable")
    datasets = _datasets(service)
    plans = _plans(service)
    dataset_choices = tuple(
        (
            _safe_text(
                f"{item.dataset_id} {item.dataset_version} ({item.case_count} cases)",
                redactor,
            ),
            dataset_key(item),
        )
        for item in datasets
    )
    selected_dataset = next(
        (item for item in datasets if dataset_key(item) == selected_dataset_key),
        datasets[0] if datasets else None,
    )
    matching_plans = tuple(
        item
        for item in plans
        if selected_dataset is not None
        and (item.dataset_id, item.dataset_version)
        == (selected_dataset.dataset_id, selected_dataset.dataset_version)
    )
    plan_choices = tuple(
        (
            _safe_text(f"{item.kind} ({item.plan_version})", redactor),
            plan_key(item),
        )
        for item in matching_plans
    )
    selected_plan = next(
        (item for item in matching_plans if plan_key(item) == selected_plan_key),
        matching_plans[0] if matching_plans else None,
    )

    runs = _runs(service)
    summaries = {run.run_id: _summary(service, run.run_id) for run in runs}
    releases = {run.run_id: _release_evidence(service, run.run_id) for run in runs}
    run_choices = tuple((_run_choice(run, summaries[run.run_id]), run.run_id) for run in runs)
    requested_run_id = selected_run_id or state.evaluation_run_id
    selected_run = next(
        (run for run in runs if run.run_id == requested_run_id),
        runs[0] if runs else None,
    )
    selected_summary = None if selected_run is None else summaries.get(selected_run.run_id)
    next_state = state.with_evaluation(None if selected_run is None else selected_run.run_id)

    identity_rows = _identity_rows(selected_dataset)
    plan_rows = _plan_rows(selected_plan)
    run_rows = tuple(
        _run_row(run, summaries[run.run_id], is_release=releases[run.run_id] is not None)
        for run in runs
    )
    failure_rows = (
        ()
        if selected_run is None
        else _failed_case_rows(service.failed_cases(selected_run.run_id), redactor)
    )
    progress_markdown = _progress_markdown(selected_run, selected_summary)
    gate_markdown = _gate_markdown(selected_run, selected_summary)
    overview_rows: tuple[tuple[object, ...], ...] = _unavailable_overview_rows()
    quality_plot_rows: tuple[tuple[str, float], ...] = ()
    latency_plot_rows: tuple[tuple[str, int, float], ...] = ()
    kpi_html = _kpi_html(None)
    system_rows = _system_rows(None)
    category_rows: tuple[tuple[object, ...], ...] = ()
    operations_rows: tuple[tuple[object, ...], ...] = ()
    operations_preview = "Evidence unavailable. / 证据不可用。"
    operations_links = "Operations downloads unavailable. / 运维下载不可用。"
    artifact_rows: tuple[tuple[object, ...], ...] = ()
    artifact_links = "Artifacts unavailable. / 报告文件不可用。"
    report: EvaluationReportV2 | None = None

    if selected_run is not None:
        release = releases[selected_run.run_id]
        manifest = _manifest(service, selected_run.run_id)
        if manifest is not None:
            _validate_manifest_identity(selected_run, manifest)
            artifact_rows = tuple(_artifact_row(item) for item in manifest.artifacts)
            artifact_links = _artifact_links(manifest, retrieval_profile)
            operations_links = _operations_links(manifest, retrieval_profile)
        elif (
            selected_summary is not None
            and selected_run.status is EvaluationRunStatus.COMPLETED
            and selected_summary.evidence_status == "available"
        ):
            artifact_links = _compatibility_report_links(
                selected_run.run_id,
                retrieval_profile,
            )
        report = _report(service, selected_run, manifest)
        if report is not None:
            overview_rows = _overview_rows(report)
            category_rows = _category_rows(report)
            quality_plot_rows = _quality_plot_rows(report)
            latency_plot_rows = _latency_plot_rows(report)
            kpi_html = _kpi_html(report)
            system_rows = _system_rows(report)
            judge = report.provenance.evaluation_judge_model or "deterministic"
            gate_markdown = (
                f"{gate_markdown}\n\nScorer backend / 评估后端: "
                f"`{report.provenance.evaluation_scorer_backend}` · judge: `{judge}`"
            )
        elif release is not None:
            overview_rows = _release_overview_rows(release)
            quality_plot_rows = _release_quality_plot_rows(release)
            latency_plot_rows = _release_latency_plot_rows(release)
            kpi_html = _release_kpi_html(release)
            system_rows = _release_system_rows(release)
            gate_markdown = _release_gate_markdown(release)
        else:
            overview_rows = _unavailable_overview_rows()
        if release is not None:
            operations_rows = _release_operations_rows(release)
            operations_preview = (
                "Sealed Phase 12 release (schema v1). V2-only operations fields are "
                "explicitly unavailable. / 已封存的 Phase 12 release (schema v1); "
                "仅 v2 提供的运维字段明确标记为不可用。"
            )
        else:
            operations_rows, operations_preview = _operations_view(
                service,
                selected_run,
                manifest,
                report,
            )
        if (
            selected_summary is not None
            and selected_summary.evidence_status == "available"
            and report is None
            and release is None
        ):
            gate_markdown = (
                "### Evidence unavailable / 证据不可用\n"
                "The persisted report could not be validated; no passing result is shown. / "
                "持久化报告未通过验证, 因此不会显示通过结论。"
            )

    if not datasets:
        status = (
            "No validated datasets are registered; starting is disabled. / "
            "没有已验证的数据集, 无法启动评估。"
        )
    elif not matching_plans:
        status = (
            "No compatible registered plan is available for this dataset. / "
            "所选数据集没有兼容的已注册计划。"
        )
    elif not runs:
        status = (
            "Catalog loaded; no persisted runs yet. Start is explicit. / "
            "目录已加载; 暂无历史运行。评估只会在明确点击启动后开始。"
        )
    else:
        status = (
            "Read-only refresh complete; no provider work was started. / "
            "只读刷新完成; 未启动任何模型调用。"
        )
    cache_rows = _cache_statistics_rows(operations_rows, report)
    return EvaluationRender(
        run_rows=run_rows,
        failure_rows=failure_rows,
        metrics_markdown="",
        status_markdown=status,
        state=next_state,
        dataset_choices=dataset_choices,
        plan_choices=plan_choices,
        run_choices=run_choices,
        selected_dataset_key=(None if selected_dataset is None else dataset_key(selected_dataset)),
        selected_plan_key=None if selected_plan is None else plan_key(selected_plan),
        selected_run_id=None if selected_run is None else selected_run.run_id,
        identity_rows=identity_rows,
        plan_rows=plan_rows,
        overview_rows=overview_rows,
        quality_rows=_metric_subset(overview_rows, _QUALITY_METRICS),
        quality_plot_rows=quality_plot_rows,
        performance_rows=_metric_subset(overview_rows, _PERFORMANCE_METRICS),
        latency_plot_rows=latency_plot_rows,
        cost_rows=_metric_subset(overview_rows, _COST_METRICS),
        category_rows=category_rows,
        operations_rows=operations_rows,
        cache_rows=cache_rows,
        refusal_rows=_metric_subset(operations_rows, _REFUSAL_METRICS),
        system_rows=system_rows,
        artifact_rows=artifact_rows,
        kpi_html=kpi_html,
        gate_markdown=gate_markdown,
        progress_markdown=progress_markdown,
        operations_preview=operations_preview,
        operations_links_markdown=operations_links,
        artifact_links_markdown=artifact_links,
        poll_active=(selected_run is not None and selected_run.status in _ACTIVE_RUN_STATUSES),
        start_enabled=selected_dataset is not None and selected_plan is not None,
    )


def with_status(render: EvaluationRender, status: str) -> EvaluationRender:
    return replace(render, status_markdown=status)


def _datasets(service: EvaluationGateway) -> tuple[EvaluationDatasetCatalogEntry, ...]:
    values = tuple(service.datasets())
    if any(not isinstance(item, EvaluationDatasetCatalogEntry) for item in values):
        raise EvaluationDashboardError("evaluation_dataset_catalog_invalid")
    return values


def _plans(service: EvaluationGateway) -> tuple[EvaluationPlanCatalogEntry, ...]:
    values = tuple(service.plans())
    if any(not isinstance(item, EvaluationPlanCatalogEntry) for item in values):
        raise EvaluationDashboardError("evaluation_plan_catalog_invalid")
    return values


def _runs(service: EvaluationGateway) -> tuple[EvaluationRun, ...]:
    values = tuple(service.list_runs())
    if any(not isinstance(item, EvaluationRun) for item in values):
        raise EvaluationDashboardError("evaluation_run_history_invalid")
    return values


def _summary(service: EvaluationGateway, run_id: str) -> EvaluationRunSummary | None:
    value = service.summary(run_id)
    if value is not None and not isinstance(value, EvaluationRunSummary):
        raise EvaluationDashboardError("evaluation_summary_invalid")
    return value


def _release_evidence(
    service: EvaluationGateway,
    run_id: str,
) -> ReleaseEvidenceSnapshot | None:
    reader = getattr(service, "release_evidence", None)
    if not callable(reader):
        return None
    value = reader(run_id)
    if value is not None and not isinstance(value, ReleaseEvidenceSnapshot):
        raise EvaluationDashboardError("evaluation_release_evidence_invalid")
    return value


def _manifest(
    service: EvaluationGateway,
    run_id: str,
) -> EvaluationArtifactManifest | None:
    value = service.artifact_manifest(run_id)
    if value is not None and not isinstance(value, EvaluationArtifactManifest):
        raise EvaluationDashboardError("evaluation_artifact_manifest_invalid")
    return value


def _artifact(
    service: EvaluationGateway,
    run_id: str,
    artifact_id: str,
) -> ResolvedEvaluationArtifact | None:
    value = service.artifact(run_id, artifact_id)
    if value is not None and not isinstance(value, ResolvedEvaluationArtifact):
        raise EvaluationDashboardError("evaluation_artifact_invalid")
    return value


def _identity_rows(
    dataset: EvaluationDatasetCatalogEntry | None,
) -> tuple[tuple[object, ...], ...]:
    if dataset is None:
        return ()
    return (
        ("dataset_id", dataset.dataset_id),
        ("dataset_version", dataset.dataset_version),
        ("dataset_schema", dataset.schema_version),
        ("dataset_hash", dataset.content_hash),
        ("corpus_version", dataset.corpus_version),
        ("corpus_hash", dataset.corpus_hash),
        ("case_count", dataset.case_count),
        ("languages", ", ".join(dataset.languages)),
    )


def _plan_rows(
    plan: EvaluationPlanCatalogEntry | None,
) -> tuple[tuple[object, ...], ...]:
    if plan is None:
        return ()
    return (
        (
            plan.plan_id,
            plan.kind,
            plan.planned_case_count,
            plan.candidate_count,
            plan.maximum_logical_calls,
            plan.maximum_provider_calls,
            plan.cache_policy,
            _unavailable(plan.cost_estimate_status),
            _unavailable("unavailable") if plan.cost_cap is None else str(plan.cost_cap),
            plan.maximum_active_jobs,
        ),
    )


def _run_choice(run: EvaluationRun, summary: EvaluationRunSummary | None) -> str:
    remaining = (
        max(0, run.total_cases - run.completed_cases - run.failed_cases)
        if summary is None
        else summary.remaining_cases
    )
    return (
        f"{run.run_id} · {run.status.value} · "
        f"{run.completed_cases}/{run.total_cases} complete · {remaining} remaining"
    )


def _run_row(
    run: EvaluationRun,
    summary: EvaluationRunSummary | None,
    *,
    is_release: bool = False,
) -> tuple[object, ...]:
    remaining = (
        max(0, run.total_cases - run.completed_cases - run.failed_cases)
        if summary is None
        else summary.remaining_cases
    )
    return (
        run.run_id,
        "sealed-release-v1" if is_release else "standard-evaluation",
        run.status.value,
        run.completed_cases,
        run.failed_cases,
        remaining,
        run.total_cases,
        run.dataset_id,
        run.dataset_version,
        run.corpus_version,
        run.configuration_id,
        _timestamp(run.created_at),
        "" if summary is None or summary.completed_at is None else _timestamp(summary.completed_at),
        "unavailable" if summary is None else summary.gate_status,
    )


def _progress_markdown(
    run: EvaluationRun | None,
    summary: EvaluationRunSummary | None,
) -> str:
    if run is None:
        return "No run selected. / 未选择运行。"
    remaining = (
        max(0, run.total_cases - run.completed_cases - run.failed_cases)
        if summary is None
        else summary.remaining_cases
    )
    return (
        f"**{run.status.value}** · completed `{run.completed_cases}` · failed "
        f"`{run.failed_cases}` · active/remaining `{remaining}` · total `{run.total_cases}`  \n"
        f"Run: `{run.run_id}` · Configuration: `{run.configuration_id}`"
    )


def _gate_markdown(
    run: EvaluationRun | None,
    summary: EvaluationRunSummary | None,
) -> str:
    if run is None:
        return "### No evidence selected / 未选择证据"
    if run.status in {EvaluationRunStatus.FAILED, EvaluationRunStatus.INVALID}:
        code = run.safe_error_code or "evaluation_failed"
        return (
            f"### Run failed / 运行失败\nSafe reason / 安全原因: `{code}`. "
            "Missing evidence is not treated as success. / 缺失证据不会被视为成功。"
        )
    if summary is None or summary.evidence_status == "unavailable":
        return (
            "### Evidence unavailable / 证据不可用\n"
            "The overall gate is unavailable, not passing. / 总体验收门槛不可用, 并非通过。"
        )
    if summary.evidence_status == "incomplete":
        return (
            "### Evidence incomplete / 证据不完整\n"
            "Partial committed evidence is shown without a passing conclusion. / "
            "仅显示已提交的部分证据, 不给出通过结论。"
        )
    if summary.gate_status == "passed":
        return "### PASS / 通过\nValidated persisted evidence satisfies the declared gate."
    if summary.gate_status == "failed":
        return "### FAIL / 未通过\nOne or more declared evidence gates failed."
    return "### Gate unavailable / 验收门槛不可用"


def _failed_case_rows(
    values: Sequence[FailedCaseDiagnostic | Mapping[str, object]],
    redactor: Redactor,
) -> tuple[tuple[object, ...], ...]:
    rows: list[tuple[object, ...]] = []
    for value in values:
        if isinstance(value, FailedCaseDiagnostic):
            contributions = ", ".join(
                f"{item.metric_id}:{item.status}" for item in value.metric_contributions
            )
            rows.append(
                (
                    _safe_text(value.case_id, redactor),
                    _safe_text(value.safe_error_code, redactor),
                    _safe_text(", ".join(value.tags), redactor),
                    _safe_text(contributions, redactor),
                    _safe_text(value.refusal_reason or "", redactor),
                    _safe_text(", ".join(value.citation_chunk_ids), redactor),
                    _safe_text(value.request_id or "", redactor),
                    _safe_text(value.trace_id or "", redactor),
                    _safe_text(value.outcome or "", redactor),
                )
            )
            continue
        if not isinstance(value, Mapping):
            raise EvaluationDashboardError("evaluation_failed_case_invalid")
        rows.append(
            (
                _safe_text(value.get("case_id", ""), redactor),
                _safe_text(
                    value.get("safe_error_code", value.get("safe_reason", "")),
                    redactor,
                ),
                _safe_join(value.get("tags"), redactor),
                _safe_join(value.get("metric_contributions"), redactor),
                _safe_text(value.get("refusal_reason", ""), redactor),
                _safe_join(value.get("citation_chunk_ids"), redactor),
                _safe_text(value.get("request_id", ""), redactor),
                _safe_text(value.get("trace_id", ""), redactor),
                _safe_text(value.get("outcome", ""), redactor),
            )
        )
    return tuple(rows)


def _validate_manifest_identity(
    run: EvaluationRun,
    manifest: EvaluationArtifactManifest,
) -> None:
    if manifest.run_id != run.run_id or manifest.configuration_id != run.configuration_id:
        raise EvaluationDashboardError("evaluation_artifact_identity_mismatch")


def _artifact_row(item: object) -> tuple[object, ...]:
    artifact_id = getattr(item, "artifact_id", None)
    artifact_format = getattr(item, "format", None)
    schema_version = getattr(item, "schema_version", None)
    media_type = getattr(item, "media_type", None)
    digest = getattr(item, "sha256_digest", None)
    byte_size = getattr(item, "byte_size", None)
    created_at = getattr(item, "created_at", None)
    if (
        not all(
            isinstance(value, str)
            for value in (artifact_id, artifact_format, schema_version, media_type, digest)
        )
        or type(byte_size) is not int
        or not isinstance(created_at, datetime)
    ):
        raise EvaluationDashboardError("evaluation_artifact_descriptor_invalid")
    return (
        artifact_id,
        artifact_format,
        schema_version,
        media_type,
        digest,
        byte_size,
        _timestamp(created_at),
    )


def _artifact_links(
    manifest: EvaluationArtifactManifest,
    retrieval_profile: str | None = None,
) -> str:
    run = quote(manifest.run_id, safe="")
    suffix = _profile_suffix(retrieval_profile)
    lines = [
        "Validated same-origin downloads / 已验证的同源下载:",
        f"- [Integrity manifest / 完整性清单]"
        f"({_API_PREFIX}/evaluations/{run}/artifacts{suffix})",
    ]
    for item in manifest.artifacts:
        artifact = quote(item.artifact_id, safe="")
        lines.append(
            f"- [{item.artifact_id}]"
            f"({_API_PREFIX}/evaluations/{run}/artifacts/{artifact}{suffix}) "
            f"— `{item.format}`, `{item.byte_size}` bytes"
        )
    return "\n".join(lines)


def _operations_links(
    manifest: EvaluationArtifactManifest,
    retrieval_profile: str | None = None,
) -> str:
    run = quote(manifest.run_id, safe="")
    suffix = _profile_suffix(retrieval_profile)
    by_id = {item.artifact_id: item for item in manifest.artifacts}
    lines = ["Operations downloads / 运维下载:"]
    for artifact_id, label in (
        (_OPERATIONS_TEXT_ARTIFACT_ID, "TXT"),
        (_OPERATIONS_CSV_ARTIFACT_ID, "CSV"),
    ):
        descriptor = by_id.get(artifact_id)
        if descriptor is None:
            lines.append(f"- {label}: unavailable / 不可用")
            continue
        artifact = quote(artifact_id, safe="")
        lines.append(
            f"- [{label}]({_API_PREFIX}/evaluations/{run}/artifacts/{artifact}{suffix}) "
            f"- `{descriptor.byte_size}` bytes"
        )
    return "\n".join(lines)


def _compatibility_report_links(
    run_id: str,
    retrieval_profile: str | None = None,
) -> str:
    run = quote(run_id, safe="")
    suffix = _profile_suffix(retrieval_profile)
    return "\n".join(
        (
            "Validated compatibility downloads / 已验证的兼容下载:",
            f"- [evaluation-report.json]({_API_PREFIX}/reports/{run}.json{suffix})",
            f"- [evaluation-report.html]({_API_PREFIX}/reports/{run}.html{suffix})",
        )
    )


def _profile_suffix(retrieval_profile: str | None) -> str:
    if retrieval_profile is None or retrieval_profile == "openai-api":
        return ""
    return f"?retrieval_profile={quote(retrieval_profile, safe='')}"


def _report(
    service: EvaluationGateway,
    run: EvaluationRun,
    manifest: EvaluationArtifactManifest | None,
) -> EvaluationReportV2 | None:
    if manifest is None or not any(
        item.artifact_id == _REPORT_ARTIFACT_ID for item in manifest.artifacts
    ):
        return None
    artifact = _artifact(service, run.run_id, _REPORT_ARTIFACT_ID)
    if (
        artifact is None
        or artifact.artifact_id != _REPORT_ARTIFACT_ID
        or artifact.media_type != "application/json"
    ):
        return None
    try:
        raw = decode_json_report(artifact.content.decode("utf-8"))
        if not isinstance(raw, dict):
            return None
        report = parse_report_v2(raw)
    except (UnicodeError, ValueError):
        return None
    if report.run_id != run.run_id or report.configuration_id != run.configuration_id:
        return None
    return report


def _overview_rows(report: EvaluationReportV2) -> tuple[tuple[object, ...], ...]:
    gate = next(item for item in report.gates if item.gate_id == report.acceptance_gate_id)
    rows = [_observation_row(item) for item in gate.observations]
    performance = report.performance_evidence
    measured = performance.measured
    all_latency = measured.latency_ms.all_attempts
    successful_latency = measured.latency_ms.successful_attempts
    for scope, latency in (
        ("all-attempt", all_latency),
        ("successful-only", successful_latency),
    ):
        for percentile in ("p50", "p90", "p95", "p99"):
            value = None if latency is None else getattr(latency, f"{percentile}_ms")
            denominator = 0 if latency is None else latency.count
            threshold: object = ""
            status = "observed" if latency is not None else "unavailable"
            if scope == "all-attempt" and percentile == "p90":
                threshold = f"<= {measured.official_p90_gate.threshold_ms}"
                status = (
                    measured.official_p90_gate.status.value
                    if measured.official_p90_gate.passed is None
                    else "passed"
                    if measured.official_p90_gate.passed
                    else "failed"
                )
            rows.append(
                _metric_row(
                    f"{scope}-latency-{percentile}",
                    value,
                    "ms",
                    threshold=threshold,
                    denominator=denominator,
                    status=status,
                    scorer="nearest-rank",
                )
            )
    rows.extend(
        (
            _metric_row(
                "total-logical-requests",
                measured.logical_attempt_count,
                "requests",
                numerator=measured.logical_attempt_count,
                denominator=measured.logical_attempt_count,
                status="observed" if measured.logical_attempt_count else "unavailable",
                scorer="all-http-attempts-v2",
            ),
            _metric_row(
                "successful-logical-requests",
                measured.successful_logical_attempt_count,
                "requests",
                numerator=measured.successful_logical_attempt_count,
                denominator=measured.logical_attempt_count,
                status="observed" if measured.logical_attempt_count else "unavailable",
                scorer="all-http-attempts-v2",
            ),
            _metric_row(
                "success-rate",
                (
                    measured.successful_logical_attempt_count / measured.logical_attempt_count
                    if measured.logical_attempt_count
                    else None
                ),
                "ratio",
                numerator=measured.successful_logical_attempt_count,
                denominator=measured.logical_attempt_count,
                status="observed" if measured.logical_attempt_count else "unavailable",
                scorer="all-http-attempts-v2",
            ),
            _metric_row(
                "observed-peak-concurrency",
                _observed_peak_concurrency(measured.attempts),
                "requests",
                denominator=measured.http_attempt_count,
                status="observed" if measured.http_attempt_count else "unavailable",
                scorer="attempt-interval-sweep-v1",
            ),
            _metric_row(
                "error-rate",
                measured.error_rate.value,
                "ratio",
                numerator=measured.error_rate.numerator,
                denominator=measured.error_rate.denominator,
                status=measured.error_rate.status.value,
                scorer="all-http-attempts-v2",
            ),
            _metric_row(
                "timeout-rate",
                (
                    sum(
                        attempt.status is LoadAttemptStatus.TIMEOUT for attempt in measured.attempts
                    )
                    / measured.http_attempt_count
                    if measured.http_attempt_count
                    else None
                ),
                "ratio",
                numerator=sum(
                    attempt.status is LoadAttemptStatus.TIMEOUT for attempt in measured.attempts
                ),
                denominator=measured.http_attempt_count,
                status="available" if measured.http_attempt_count else "unavailable",
                scorer="all-http-attempts-v2",
            ),
        )
    )
    cost = performance.cost
    rows.extend(_token_rows(cost.role_direction_tokens))
    rows.extend(
        (
            _metric_row(
                "provider-attempt-count",
                cost.provider_attempt_count,
                "attempts",
                numerator=cost.provider_attempt_count,
                denominator=measured.http_attempt_count,
                status="observed" if measured.http_attempt_count else "unavailable",
                scorer=cost.schema_version,
            ),
            _metric_row(
                "total-cost",
                cost.total_cost,
                cost.pricing.currency,
                numerator=cost.known_partial_cost,
                denominator=cost.provider_attempt_count,
                status=cost.total_cost_status.value,
                scorer=cost.schema_version,
            ),
            _metric_row(
                "cost-per-1000-logical-attempts",
                cost.cost_per_1000_logical_attempts.per_1000,
                f"{cost.pricing.currency}/1000-attempts",
                denominator=cost.cost_per_1000_logical_attempts.denominator,
                status=cost.cost_per_1000_logical_attempts.status.value,
                scorer=cost.schema_version,
            ),
            _metric_row(
                "cost-per-1000-successes",
                cost.cost_per_1000_successes.per_1000,
                f"{cost.pricing.currency}/1000-successes",
                denominator=cost.cost_per_1000_successes.denominator,
                status=cost.cost_per_1000_successes.status.value,
                scorer=cost.schema_version,
            ),
        )
    )
    operations = {item.metric_id: item for item in report.operations_summary.observations}
    for metric_id in ("cache-hit-rate", "refusal-rate"):
        observation = operations.get(metric_id)
        rows.append(
            _observation_row(observation)
            if observation is not None
            else _metric_row(
                metric_id,
                None,
                "ratio",
                denominator=None,
                status="unavailable",
                scorer="unavailable",
            )
        )
    return tuple(rows)


def _unavailable_overview_rows() -> tuple[tuple[object, ...], ...]:
    metrics = (
        ("faithfulness", "ratio"),
        ("context-precision", "ratio"),
        ("answer-compliance", "ratio"),
        ("style", "ratio"),
        ("refusal-appropriateness", "ratio"),
        *((f"all-attempt-latency-{item}", "ms") for item in ("p50", "p90", "p95", "p99")),
        *((f"successful-only-latency-{item}", "ms") for item in ("p50", "p90", "p95", "p99")),
        ("total-logical-requests", "requests"),
        ("successful-logical-requests", "requests"),
        ("success-rate", "ratio"),
        ("observed-peak-concurrency", "requests"),
        ("provider-attempt-count", "attempts"),
        ("input-tokens", "tokens"),
        ("output-tokens", "tokens"),
        ("total-cost", "currency"),
        ("cost-per-1000-logical-attempts", "currency/1000-attempts"),
        ("cost-per-1000-successes", "currency/1000-successes"),
        ("error-rate", "ratio"),
        ("timeout-rate", "ratio"),
        ("cache-hit-rate", "ratio"),
        ("refusal-rate", "ratio"),
    )
    return tuple(
        _metric_row(
            metric_id,
            None,
            unit,
            denominator=None,
            status="unavailable",
            scorer="unavailable",
        )
        for metric_id, unit in metrics
    )


def _metric_subset(
    rows: Sequence[tuple[object, ...]],
    metric_ids: Sequence[str],
) -> tuple[tuple[object, ...], ...]:
    """Select metric rows in a stable presentation order without recomputing evidence."""

    by_id = {str(row[0]): row for row in rows if row}
    return tuple(by_id[metric_id] for metric_id in metric_ids if metric_id in by_id)


def _cache_statistics_rows(
    operations_rows: Sequence[tuple[object, ...]],
    report: EvaluationReportV2 | None,
) -> tuple[tuple[object, ...], ...]:
    """Expose recorded cache bypass evidence without inventing a hit rate."""

    base_rows = list(_metric_subset(operations_rows, _CACHE_METRICS))
    if report is None:
        return tuple(base_rows)
    attempts = report.performance_evidence.measured.attempts
    if not attempts:
        return tuple(base_rows)
    policies = tuple(attempt.cache_status.get("request-policy") for attempt in attempts)
    if not policies or any(not isinstance(policy, str) or not policy for policy in policies):
        return tuple(base_rows)

    policy = policies[0] if len(set(policies)) == 1 else "mixed"
    bypassed = sum(item == "bypass" for item in policies)
    evidence_rows = [
        _metric_row(
            "cache-policy",
            policy,
            "policy",
            denominator=len(attempts),
            status="observed",
            scorer="attempt-ledger-v2",
        ),
        _metric_row(
            "cache-bypassed-lookups",
            bypassed,
            "requests",
            numerator=bypassed,
            denominator=len(attempts),
            status="observed",
            scorer="attempt-ledger-v2",
        ),
    ]
    if bypassed == len(attempts):
        not_applicable = _metric_row(
            "cache-hit-rate",
            "N/A (cache bypass) / 不适用 (缓存已绕过)",
            "ratio",
            numerator=0,
            denominator=0,
            status="not-applicable",
            scorer="derived-from-attempt-ledger-v2",
        )
        base_rows = [row for row in base_rows if row[0] != "cache-hit-rate"]
        base_rows.append(not_applicable)
    return tuple((*evidence_rows, *base_rows))


def _release_overview_rows(
    release: ReleaseEvidenceSnapshot,
) -> tuple[tuple[object, ...], ...]:
    missing = _unavailable("not-recorded-in-v1-release")
    quality = {item.metric_id: item for item in release.quality_metrics}
    rows: list[tuple[object, ...]] = []
    for metric_id in _QUALITY_METRICS:
        metric = quality.get(metric_id)
        if metric is None:
            rows.append(
                _metric_row(
                    metric_id,
                    missing,
                    "ratio",
                    numerator=missing,
                    denominator=missing,
                    status="unavailable",
                    scorer=missing,
                )
            )
            continue
        threshold = (
            ""
            if metric.threshold is None
            else f"{metric.operator or ''} {metric.threshold}".strip()
        )
        rows.append(
            _metric_row(
                metric_id,
                metric.value,
                "ratio",
                threshold=threshold,
                numerator=missing,
                denominator=metric.denominator if metric.denominator else missing,
                status=(
                    "unavailable"
                    if metric.passed is None
                    else "passed"
                    if metric.passed
                    else "failed"
                ),
                scorer=metric.scorer_version or missing,
            )
        )

    performance = release.performance
    latencies = {
        "p50": performance.p50_ms,
        "p90": performance.p90_ms,
        "p95": performance.p95_ms,
        "p99": performance.p99_ms,
    }
    for percentile, value in latencies.items():
        rows.append(
            _metric_row(
                f"all-attempt-latency-{percentile}",
                value,
                "ms",
                threshold="<= 10000" if percentile == "p90" else "",
                denominator=performance.attempts,
                status="passed" if percentile == "p90" and value <= 10_000 else "observed",
                scorer="sealed-release-manifest-v1",
            )
        )
    for percentile in ("p50", "p90", "p95", "p99"):
        rows.append(
            _metric_row(
                f"successful-only-latency-{percentile}",
                None,
                "ms",
                denominator=missing,
                status="unavailable",
                scorer=missing,
            )
        )
    success_rate = performance.successes / performance.attempts
    error_rate = performance.errors / performance.attempts
    rows.extend(
        (
            _metric_row(
                "total-logical-requests",
                performance.attempts,
                "requests",
                numerator=performance.attempts,
                denominator=performance.attempts,
                status="observed",
                scorer="sealed-release-manifest-v1",
            ),
            _metric_row(
                "successful-logical-requests",
                performance.successes,
                "requests",
                numerator=performance.successes,
                denominator=performance.attempts,
                status="observed",
                scorer="sealed-release-manifest-v1",
            ),
            _metric_row(
                "success-rate",
                success_rate,
                "ratio",
                numerator=performance.successes,
                denominator=performance.attempts,
                status="observed",
                scorer="sealed-release-manifest-v1",
            ),
            _metric_row(
                "observed-peak-concurrency",
                performance.observed_peak_concurrency,
                "requests",
                denominator=performance.attempts,
                status="observed",
                scorer="sealed-release-manifest-v1",
            ),
            _metric_row(
                "error-rate",
                error_rate,
                "ratio",
                numerator=performance.errors,
                denominator=performance.attempts,
                status="passed" if error_rate < 0.01 else "failed",
                scorer="sealed-release-manifest-v1",
            ),
            _metric_row(
                "timeout-rate",
                None,
                "ratio",
                denominator=missing,
                status="unavailable",
                scorer=missing,
            ),
            _metric_row(
                "provider-attempt-count",
                performance.provider_attempt_count,
                "attempts",
                denominator=performance.attempts,
                status="observed",
                scorer="sealed-release-manifest-v1",
            ),
            _metric_row(
                "input-tokens",
                performance.input_tokens,
                "tokens",
                denominator=performance.provider_attempt_count,
                status="observed",
                scorer="sealed-release-manifest-v1",
            ),
            _metric_row(
                "output-tokens",
                performance.output_tokens,
                "tokens",
                denominator=performance.provider_attempt_count,
                status="observed",
                scorer="sealed-release-manifest-v1",
            ),
            _metric_row(
                "total-cost",
                performance.total_cost,
                performance.currency,
                denominator=performance.provider_attempt_count,
                status="observed",
                scorer="sealed-release-manifest-v1",
            ),
            _metric_row(
                "cost-per-1000-logical-attempts",
                performance.cost_per_1000_attempts,
                f"{performance.currency}/1000-attempts",
                denominator=performance.attempts,
                status="observed",
                scorer="derived-from-sealed-release-v1",
            ),
            _metric_row(
                "cost-per-1000-successes",
                performance.cost_per_1000_successes,
                f"{performance.currency}/1000-successes",
                denominator=performance.successes,
                status="observed",
                scorer="sealed-release-manifest-v1",
            ),
        )
    )
    return tuple(rows)


def _release_quality_plot_rows(
    release: ReleaseEvidenceSnapshot,
) -> tuple[tuple[str, float], ...]:
    return tuple(
        (item.metric_id, item.value * 100.0)
        for item in release.quality_metrics
        if item.metric_id in _QUALITY_METRICS and item.value is not None
    )


def _release_latency_plot_rows(
    release: ReleaseEvidenceSnapshot,
) -> tuple[tuple[str, int, float], ...]:
    performance = release.performance
    return tuple(
        ("all attempts", percentile, value)
        for percentile, value in (
            (50, performance.p50_ms),
            (90, performance.p90_ms),
            (95, performance.p95_ms),
            (99, performance.p99_ms),
        )
    )


def _release_system_rows(
    release: ReleaseEvidenceSnapshot,
) -> tuple[tuple[object, ...], ...]:
    performance = release.performance
    missing = _unavailable("not-recorded-in-v1-release")
    return (
        (
            "observed-peak-concurrency",
            performance.observed_peak_concurrency,
            "requests",
            "observed",
            "sealed release manifest",
        ),
        (
            "measured-http-attempts",
            performance.attempts,
            "attempts",
            "observed",
            "sealed release manifest",
        ),
        (
            "successful-logical-requests",
            performance.successes,
            "requests",
            "observed",
            "sealed release manifest",
        ),
        ("active-requests", missing, "requests", "unavailable", "historical release"),
        ("queue-depth", missing, "requests", "unavailable", "historical release"),
        ("cpu-utilization", missing, "percent", "unavailable", "not recorded in v1"),
        ("memory-usage", missing, "bytes", "unavailable", "not recorded in v1"),
    )


def _release_operations_rows(
    release: ReleaseEvidenceSnapshot,
) -> tuple[tuple[object, ...], ...]:
    performance = release.performance
    missing = _unavailable("not-recorded-in-v1-release")
    values: dict[str, tuple[object, object, object, str, str]] = {
        "total-logical-requests": (
            performance.attempts,
            performance.attempts,
            performance.attempts,
            "observed",
            "sealed-release-manifest-v1",
        ),
        "successful-logical-requests": (
            performance.successes,
            performance.successes,
            performance.attempts,
            "observed",
            "sealed-release-manifest-v1",
        ),
        "all-attempt-latency-p50-ms": (
            performance.p50_ms,
            "",
            performance.attempts,
            "observed",
            "sealed-release-manifest-v1",
        ),
        "all-attempt-latency-p95-ms": (
            performance.p95_ms,
            "",
            performance.attempts,
            "observed",
            "sealed-release-manifest-v1",
        ),
        "input-tokens": (
            performance.input_tokens,
            performance.input_tokens,
            performance.provider_attempt_count,
            "observed",
            "sealed-release-manifest-v1",
        ),
        "output-tokens": (
            performance.output_tokens,
            performance.output_tokens,
            performance.provider_attempt_count,
            "observed",
            "sealed-release-manifest-v1",
        ),
        "refusals": (
            performance.refusals,
            performance.refusals,
            release.run.total_cases,
            "observed",
            "evaluation-report-v1",
        ),
        "answered-requests": (
            performance.answered_requests,
            performance.answered_requests,
            release.run.total_cases,
            "observed",
            "evaluation-report-v1",
        ),
        "refusal-rate": (
            performance.refusals / release.run.total_cases,
            performance.refusals,
            release.run.total_cases,
            "observed",
            "derived-from-evaluation-report-v1",
        ),
        "cost-per-1000-logical-attempts": (
            performance.cost_per_1000_attempts,
            "",
            performance.attempts,
            "observed",
            "derived-from-sealed-release-v1",
        ),
        "cost-per-1000-successes": (
            performance.cost_per_1000_successes,
            "",
            performance.successes,
            "observed",
            "sealed-release-manifest-v1",
        ),
    }
    rows: list[tuple[object, ...]] = []
    for metric in OPERATIONS_METRIC_ORDER:
        metric_id = metric.value
        value = values.get(metric_id)
        if value is None:
            rows.append(
                _metric_row(
                    metric_id,
                    None,
                    _operations_unit(metric_id),
                    numerator=missing,
                    denominator=missing,
                    status="unavailable",
                    scorer=missing,
                )
            )
            continue
        observed, numerator, denominator, status, scorer = value
        rows.append(
            _metric_row(
                metric_id,
                observed,
                _operations_unit(metric_id),
                numerator=numerator,
                denominator=denominator,
                status=status,
                scorer=scorer,
            )
        )
    return tuple(rows)


def _quality_plot_rows(report: EvaluationReportV2) -> tuple[tuple[str, float], ...]:
    gate = next(item for item in report.gates if item.gate_id == report.acceptance_gate_id)
    by_id = {item.metric_id: item for item in gate.observations}
    rows: list[tuple[str, float]] = []
    for metric_id in _QUALITY_METRICS:
        observation = by_id.get(metric_id)
        if observation is None or isinstance(observation.value, UnavailableValue):
            continue
        value = observation.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        rows.append((metric_id, float(value) * 100.0))
    return tuple(rows)


def _latency_plot_rows(report: EvaluationReportV2) -> tuple[tuple[str, int, float], ...]:
    measured = report.performance_evidence.measured
    rows: list[tuple[str, int, float]] = []
    for scope, latency in (
        ("all attempts", measured.latency_ms.all_attempts),
        ("successful only", measured.latency_ms.successful_attempts),
    ):
        if latency is None:
            continue
        for percentile in (50, 90, 95, 99):
            rows.append((scope, percentile, float(getattr(latency, f"p{percentile}_ms"))))
    return tuple(rows)


def _system_rows(report: EvaluationReportV2 | None) -> tuple[tuple[object, ...], ...]:
    unavailable = _unavailable("not-recorded-in-acceptance-evidence")
    if report is None:
        peak: object = unavailable
        attempts: object = unavailable
        successes: object = unavailable
        state = "unavailable"
    else:
        measured = report.performance_evidence.measured
        peak = _observed_peak_concurrency(measured.attempts)
        attempts = measured.http_attempt_count
        successes = measured.successful_logical_attempt_count
        state = "observed"
    return (
        ("observed-peak-concurrency", peak, "requests", state, "attempt interval ledger"),
        ("measured-http-attempts", attempts, "attempts", state, "attempt ledger"),
        ("successful-logical-requests", successes, "requests", state, "attempt ledger"),
        ("active-requests", unavailable, "requests", "unavailable", "live-only metric"),
        ("queue-depth", unavailable, "requests", "unavailable", "live-only metric"),
        ("cpu-utilization", unavailable, "percent", "unavailable", "not collected"),
        ("memory-usage", unavailable, "bytes", "unavailable", "not collected"),
    )


def _release_kpi_html(release: ReleaseEvidenceSnapshot) -> str:
    performance = release.performance
    passed = sum(item.passed is True for item in release.quality_metrics)
    quality_total = len(release.quality_metrics)
    success_rate = performance.successes / performance.attempts
    cards = (
        (
            "Quality gates / 质量门槛",
            f"{passed}/{quality_total} v1",
            "Phase 12 PASS" if release.gate_passed else "Phase 12 FAIL",
            "passed" if release.gate_passed else "failed",
        ),
        (
            "All-attempt P95 / 全请求 P95",
            f"{performance.p95_ms / 1000:.2f}s",
            f"{performance.attempts} accepted-load attempts",
            "observed",
        ),
        (
            "Cost / 1k QA",
            f"{performance.currency} {performance.cost_per_1000_attempts:.4f}",
            "all logical attempts",
            "observed",
        ),
        (
            "Success rate / 成功率",
            f"{success_rate * 100:.2f}%",
            f"{performance.successes}/{performance.attempts}",
            "observed",
        ),
        (
            "Security / 安全",
            "PASS" if performance.security_passed else "FAIL",
            "secret + critical + privacy gates",
            "passed" if performance.security_passed else "failed",
        ),
    )
    return (
        '<div class="rag-kpi-grid">'
        + "".join(
            '<section class="rag-kpi-card rag-kpi-'
            + escape(state)
            + '"><span>'
            + escape(label)
            + "</span><strong>"
            + escape(value)
            + "</strong><small>"
            + escape(note)
            + "</small></section>"
            for label, value, note, state in cards
        )
        + "</div>"
    )


def _release_gate_markdown(release: ReleaseEvidenceSnapshot) -> str:
    status = "ACCEPTED" if release.gate_passed else "FAILED"
    return (
        f"### {status} sealed release / {status} 已封存发布\n"
        f"Release: `{release.release_id}` · source schema: `{release.source_schema_version}`. "
        "V2-only metrics remain unavailable and are not inferred. / "
        "仅 v2 提供的指标保持不可用, 不会推断或补零。"
    )


def _kpi_html(report: EvaluationReportV2 | None) -> str:
    if report is None:
        cards = (
            ("Quality gates / 质量门槛", "—", "evidence unavailable", "unavailable"),
            ("All-attempt P95 / 全请求 P95", "—", "evidence unavailable", "unavailable"),
            (
                "Cost / 1k QA",
                "USD 0",
                "cost evidence unavailable; displayed as 0",
                "unavailable",
            ),
            ("Success rate / 成功率", "—", "evidence unavailable", "unavailable"),
        )
    else:
        gate = next(item for item in report.gates if item.gate_id == report.acceptance_gate_id)
        passed = sum(item.status.value == "passed" for item in gate.observations)
        quality_value = f"{passed}/{len(gate.observations)}"
        quality_note = gate.status.value.upper()
        quality_state = gate.status.value
        latency = report.performance_evidence.measured.latency_ms.all_attempts
        latency_value = "—" if latency is None else f"{latency.p95_ms / 1000:.2f}s"
        latency_note = "all measured attempts"
        latency_state = "unavailable" if latency is None else "observed"
        cost = report.performance_evidence.cost.cost_per_1000_logical_attempts
        if cost.per_1000 is None:
            cost_value = f"{report.performance_evidence.cost.pricing.currency} 0"
            cost_note = "cost evidence unavailable; displayed as 0"
            cost_state = "unavailable"
        else:
            cost_value = f"{report.performance_evidence.cost.pricing.currency} {cost.per_1000}"
            cost_note = "logical attempts"
            cost_state = "observed"
        measured = report.performance_evidence.measured
        success_rate = (
            None
            if measured.logical_attempt_count == 0
            else measured.successful_logical_attempt_count / measured.logical_attempt_count
        )
        success_value = "—" if success_rate is None else f"{success_rate * 100:.1f}%"
        success_state = "unavailable" if success_rate is None else "observed"
        cards = (
            ("Quality gates / 质量门槛", quality_value, quality_note, quality_state),
            ("All-attempt P95 / 全请求 P95", latency_value, latency_note, latency_state),
            ("Cost / 1k QA", cost_value, cost_note, cost_state),
            ("Success rate / 成功率", success_value, "logical requests", success_state),
        )
    return (
        '<div class="rag-kpi-grid">'
        + "".join(
            '<section class="rag-kpi-card rag-kpi-'
            + escape(state)
            + '"><span>'
            + escape(label)
            + "</span><strong>"
            + escape(value)
            + "</strong><small>"
            + escape(note)
            + "</small></section>"
            for label, value, note, state in cards
        )
        + "</div>"
    )


def _category_rows(report: EvaluationReportV2) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            category.category_id,
            category.case_count,
            observation.metric_id,
            _display(observation.value),
            _display(observation.denominator),
            observation.status.value,
        )
        for category in report.category_results
        for observation in category.observations
    )


def _operations_view(
    service: EvaluationGateway,
    run: EvaluationRun,
    manifest: EvaluationArtifactManifest | None,
    report: EvaluationReportV2 | None,
) -> tuple[tuple[tuple[object, ...], ...], str]:
    summary = None if report is None else report.operations_summary
    text_artifact = None
    csv_artifact = None
    if manifest is not None:
        artifact_ids = {item.artifact_id for item in manifest.artifacts}
        if _OPERATIONS_TEXT_ARTIFACT_ID in artifact_ids:
            text_artifact = _artifact(service, run.run_id, _OPERATIONS_TEXT_ARTIFACT_ID)
        if _OPERATIONS_CSV_ARTIFACT_ID in artifact_ids:
            csv_artifact = _artifact(service, run.run_id, _OPERATIONS_CSV_ARTIFACT_ID)
    preview = "Evidence unavailable. / 证据不可用。"
    try:
        text = (
            None
            if (
                text_artifact is None
                or text_artifact.artifact_id != _OPERATIONS_TEXT_ARTIFACT_ID
                or text_artifact.media_type != "text/plain"
            )
            else text_artifact.content.decode("utf-8")
        )
        csv_text = (
            None
            if (
                csv_artifact is None
                or csv_artifact.artifact_id != _OPERATIONS_CSV_ARTIFACT_ID
                or csv_artifact.media_type != "text/csv"
            )
            else csv_artifact.content.decode("utf-8")
        )
        if text is not None:
            parsed_text = parse_operations_text(text)
            if (
                parsed_text.run_id != run.run_id
                or parsed_text.configuration_id != run.configuration_id
            ):
                raise EvaluationDashboardError("evaluation_operations_identity_mismatch")
            if summary is not None and parsed_text != summary:
                raise EvaluationDashboardError("evaluation_operations_report_mismatch")
            summary = parsed_text
            preview = text
        if csv_text is not None:
            parsed_csv = parse_operations_csv(csv_text)
            if (
                parsed_csv.run_id != run.run_id
                or parsed_csv.configuration_id != run.configuration_id
            ):
                raise EvaluationDashboardError("evaluation_operations_identity_mismatch")
            if summary is not None and parsed_csv != summary:
                raise EvaluationDashboardError("evaluation_operations_report_mismatch")
            summary = parsed_csv
        if summary is not None and text is not None and csv_text is not None:
            verify_operations_parity(summary, text, csv_text)
    except (UnicodeError, ValueError):
        return (
            _operations_rows(()),
            "Operations evidence failed validation. / 运维证据验证失败。",
        )
    if summary is None:
        return _operations_rows(()), preview
    return _operations_rows(summary.observations), preview


def _operations_rows(
    observations: Sequence[MetricObservation],
) -> tuple[tuple[object, ...], ...]:
    by_id = {item.metric_id: item for item in observations}
    rows: list[tuple[object, ...]] = []
    for metric in OPERATIONS_METRIC_ORDER:
        observation = by_id.pop(metric.value, None)
        rows.append(
            _observation_row(observation)
            if observation is not None
            else _metric_row(
                metric.value,
                None,
                _operations_unit(metric.value),
                denominator=None,
                status="unavailable",
                scorer="unavailable",
            )
        )
    rows.extend(_observation_row(item) for item in by_id.values())
    return tuple(rows)


def _operations_unit(metric_id: str) -> str:
    if metric_id.endswith("-rate"):
        return "ratio"
    if "latency" in metric_id:
        return "ms"
    if metric_id.endswith("-tokens"):
        return "tokens"
    if metric_id.startswith("cost-per-"):
        return "currency/1000"
    return "count"


def _observation_row(observation: MetricObservation) -> tuple[object, ...]:
    threshold = ""
    if observation.threshold is not None:
        operator = "" if observation.operator is None else observation.operator.value
        threshold = f"{operator} {_display(observation.threshold)}".strip()
    return _metric_row(
        observation.metric_id,
        observation.value,
        observation.unit,
        threshold=threshold,
        numerator=observation.numerator,
        denominator=observation.denominator,
        status=observation.status.value,
        scorer=_display(observation.scorer_version),
    )


def _metric_row(
    metric_id: str,
    value: object,
    unit: str,
    *,
    threshold: object = "",
    numerator: object = "",
    denominator: object = "",
    status: str,
    scorer: str,
) -> tuple[object, ...]:
    return (
        metric_id,
        _display(value),
        unit,
        _display(threshold),
        _display(numerator),
        _display(denominator),
        status,
        scorer,
    )


def _token_rows(values: Sequence[object]) -> tuple[tuple[object, ...], ...]:
    rows: list[tuple[object, ...]] = []
    for direction in (TokenDirection.INPUT, TokenDirection.OUTPUT):
        selected = tuple(item for item in values if getattr(item, "direction", None) is direction)
        denominator = sum(getattr(item, "provider_attempt_count", 0) for item in selected)
        known = sum(getattr(item, "known_tokens", 0) for item in selected)
        available = bool(selected) and all(
            getattr(item, "status", None) is EvidenceAvailability.AVAILABLE for item in selected
        )
        total = (
            sum(getattr(item, "total_tokens", 0) or 0 for item in selected) if available else None
        )
        rows.append(
            _metric_row(
                f"{direction.value}-tokens",
                total,
                "tokens",
                numerator=known,
                denominator=denominator,
                status="available" if available else "unavailable",
                scorer="provider-cost-evidence-v2",
            )
        )
    return tuple(rows)


def _observed_peak_concurrency(attempts: Sequence[object]) -> int:
    events: list[tuple[datetime, int]] = []
    for attempt in attempts:
        started = getattr(attempt, "started_at", None)
        completed = getattr(attempt, "completed_at", None)
        if not isinstance(started, datetime) or not isinstance(completed, datetime):
            raise EvaluationDashboardError("evaluation_attempt_interval_invalid")
        events.extend(((started, 1), (completed, -1)))
    active = 0
    peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    return peak


def _safe_text(value: object, redactor: Redactor) -> str:
    redacted = redact_output(value, redactor=redactor)
    if not isinstance(redacted, str):
        raise EvaluationDashboardError("evaluation_safe_text_invalid")
    return redacted


def _safe_join(value: object, redactor: Redactor) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _safe_text(value, redactor)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return ", ".join(_safe_text(item, redactor) for item in value)
    raise EvaluationDashboardError("evaluation_failed_case_field_invalid")


def _display(value: object) -> str:
    if isinstance(value, UnavailableValue):
        return _unavailable(value.reason)
    if value is None:
        return _unavailable("not-recorded")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def _unavailable(reason: str) -> str:
    return f"unavailable ({reason}) / 不可用 ({reason})"


def _timestamp(value: datetime) -> str:
    return value.isoformat()


__all__ = [
    "EvaluationDashboardError",
    "dataset_key",
    "plan_key",
    "render_evaluation_dashboard",
    "resolve_registered_plan",
    "supports_typed_dashboard",
    "with_status",
]
