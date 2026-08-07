"""Typed, privacy-safe rendering for persisted controlled comparisons."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from enum import Enum
from urllib.parse import quote

from rag_mvp.domain import UnavailableValue
from rag_mvp.safety.output import redact_output
from rag_mvp.safety.redactor import Redactor

from .models import BrowserSessionState, ComparisonRender
from .services import EvaluationGateway

_ACTIVE_STATUSES = frozenset({"queued", "running"})
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")
_API_PREFIX = "/api/v1"
_MISSING = object()


class ComparisonDashboardError(ValueError):
    """A stable rendering failure collapsed at the callback boundary."""


@dataclass(frozen=True, slots=True)
class _Variant:
    variant_id: str
    display_name: str
    axis_value: str
    configuration_id: str


@dataclass(frozen=True, slots=True)
class _Plan:
    plan_id: str
    display_name: str
    axis: str
    dataset_id: str
    dataset_version: str
    corpus_id: str
    corpus_version: str
    case_count: int
    variants: tuple[_Variant, ...]
    baseline_variant_id: str
    repeats_per_case: int
    maximum_logical_calls: int
    maximum_provider_calls: int
    cache_policy: str
    cost_estimate_status: str
    cost_estimate: object
    cost_cap: object
    currency: str
    launchable: bool
    blocking_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Run:
    comparison_id: str
    plan_id: str
    status: str
    completed_candidates: int
    failed_candidates: int
    active_candidates: int
    remaining_candidates: int
    total_candidates: int
    completed_cases: int
    failed_cases: int
    provider_calls: object
    incurred_cost: object
    known_partial_cost: object
    cost_complete: bool
    cost_unknown_reasons: tuple[str, ...]
    currency: str | None
    safe_error_code: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


@dataclass(frozen=True, slots=True)
class _Metric:
    metric_id: str
    value: object
    unit: str
    numerator: object
    denominator: object
    state: str
    gate_state: str
    baseline_delta: object


@dataclass(frozen=True, slots=True)
class _Gate:
    gate_id: str
    status: str
    required_for_selection: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Candidate:
    candidate_id: str
    display_name: str
    axis_value: str
    evaluation_run_id: str
    configuration_id: str
    status: str
    evidence_status: str
    baseline: bool
    safe_error_code: str | None
    failed_case_count: int
    provider_call_count: int
    known_partial_cost: object
    total_cost: object
    cost_complete: bool
    cost_unknown_reasons: tuple[str, ...]
    currency: str | None
    metrics: tuple[_Metric, ...]
    gates: tuple[_Gate, ...]


@dataclass(frozen=True, slots=True)
class _Category:
    candidate_id: str
    category_id: str
    case_count: int
    metrics: tuple[_Metric, ...]


@dataclass(frozen=True, slots=True)
class _Recommendation:
    state: str
    selected_candidate_id: str | None
    rationale_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SharedSetup:
    status: str
    safe_error_code: str | None
    provider_call_count: object
    known_partial_cost: object
    total_cost: object
    currency: object
    provider_calls_complete: bool
    cost_complete: bool
    unknown_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Summary:
    comparison_id: str
    status: str
    evidence_status: str
    gate_status: str
    compatibility_state: str
    compatibility_issues: tuple[str, ...]
    controlled_dimensions: tuple[tuple[str, str], ...]
    candidates: tuple[_Candidate, ...]
    categories: tuple[_Category, ...]
    comparison_metrics: tuple[_Metric, ...]
    recommendation: _Recommendation
    shared_setup: _SharedSetup
    provider_call_count: object
    known_partial_cost: object
    total_cost: object
    cost_complete: bool
    cost_unknown_reasons: tuple[str, ...]
    currency: object


@dataclass(frozen=True, slots=True)
class _Descriptor:
    artifact_id: str
    artifact_format: str
    schema_version: str
    media_type: str
    digest: str
    byte_size: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class _Manifest:
    comparison_id: str
    artifacts: tuple[_Descriptor, ...]


def supports_comparison_dashboard(service: object) -> bool:
    return all(
        callable(getattr(service, name, None))
        for name in (
            "comparison_plans",
            "list_comparisons",
            "comparison_summary",
            "comparison_manifest",
            "start_comparison",
        )
    )


def resolve_registered_comparison_plan(service: EvaluationGateway, plan_id: str | None) -> str:
    plans = _plans(service)
    selected = next((item for item in plans if item.plan_id == plan_id), None)
    if selected is None or not selected.launchable:
        raise ComparisonDashboardError("comparison_plan_selection_invalid")
    return selected.plan_id


def comparison_id(value: object) -> str:
    return _identifier(_attribute(value, "comparison_id", "run_id"))


def render_comparison_dashboard(
    service: EvaluationGateway,
    *,
    redactor: Redactor,
    state: BrowserSessionState,
    selected_plan_id: str | None = None,
    selected_comparison_id: str | None = None,
) -> ComparisonRender:
    """Render catalogs and persisted evidence without launching provider work."""

    if not redactor.fully_configured:
        raise ComparisonDashboardError("comparison_redaction_unavailable")
    plans = _plans(service)
    runs = _runs(service)
    plans_by_id = {item.plan_id: item for item in plans}
    plan = next(
        (item for item in plans if item.plan_id == selected_plan_id),
        plans[0] if plans else None,
    )
    requested_id = selected_comparison_id or state.comparison_run_id
    run = next(
        (item for item in runs if item.comparison_id == requested_id),
        runs[0] if runs else None,
    )
    next_state = state.with_comparison(None if run is None else run.comparison_id)
    plan_choices = tuple(
        (
            _safe_text(f"{item.display_name} ({item.axis})", redactor),
            item.plan_id,
        )
        for item in plans
    )
    comparison_choices = tuple(
        (
            _safe_text(
                f"{item.comparison_id} - {item.status} - "
                f"{item.completed_candidates}/{item.total_candidates}",
                redactor,
            ),
            item.comparison_id,
        )
        for item in runs
    )
    summaries = {
        item.comparison_id: _summary(
            service.comparison_summary(item.comparison_id),
            item.comparison_id,
        )
        for item in runs
    }
    summary = None if run is None else summaries[run.comparison_id]
    run_plan = None if run is None else plans_by_id.get(run.plan_id)
    manifest = None
    if (
        run is not None
        and run.status == "completed"
        and summary is not None
        and summary.evidence_status == "available"
    ):
        manifest = _manifest(
            service.comparison_manifest(run.comparison_id),
            run.comparison_id,
        )

    controlled_rows = _controlled_rows(summary, redactor)
    shared_setup_rows = _shared_setup_rows(summary)
    candidate_rows = _candidate_rows(summary, redactor)
    comparison_metric_rows = _comparison_metric_rows(summary)
    category_rows = _category_rows(summary, redactor)
    plot_rows = _plot_rows(summary, redactor)
    artifact_rows = _artifact_rows(manifest)
    artifact_links = _artifact_links(manifest)
    gate = _gate_markdown(run, summary)
    cache_conclusion = _cache_conclusion_markdown(run_plan, summary, redactor)
    recommendation = _recommendation_markdown(summary, redactor)
    progress = _progress_markdown(run)

    if not plans:
        status = (
            "No validated experiment plans are registered; starting is disabled. / "
            "没有已验证的实验计划, 已禁用启动。"
        )
    elif plan is not None and not plan.launchable:
        blockers = ", ".join(plan.blocking_codes) or "comparison-prerequisites-unavailable"
        status = _safe_text(
            "The selected registered plan cannot launch until its safe prerequisites are "
            f"resolved. / 所选已注册计划必须先解决安全先决条件才能启动。 "
            f"Blockers / 阻塞代码: {blockers}",
            redactor,
        )
    elif not runs:
        status = (
            "Comparison catalog loaded; no persisted comparisons yet. Start is explicit. / "
            "对比目录已加载; 暂无持久化对比。只有明确点击才会启动。"
        )
    else:
        status = (
            "Read-only comparison refresh complete; no provider work was started. / "
            "对比只读刷新完成; 未启动任何模型调用。"
        )
    return ComparisonRender(
        state=next_state,
        plan_choices=plan_choices,
        comparison_choices=comparison_choices,
        selected_plan_id=None if plan is None else plan.plan_id,
        selected_comparison_id=None if run is None else run.comparison_id,
        plan_rows=() if plan is None else (_plan_row(plan),),
        history_rows=tuple(
            _history_row(
                item,
                plans_by_id.get(item.plan_id),
                summaries[item.comparison_id],
                redactor,
            )
            for item in runs
        ),
        controlled_rows=controlled_rows,
        shared_setup_rows=shared_setup_rows,
        candidate_rows=candidate_rows,
        comparison_metric_rows=comparison_metric_rows,
        category_rows=category_rows,
        plot_rows=plot_rows,
        artifact_rows=artifact_rows,
        progress_markdown=progress,
        gate_markdown=gate,
        cache_conclusion_markdown=cache_conclusion,
        recommendation_markdown=recommendation,
        artifact_links_markdown=artifact_links,
        status_markdown=status,
        poll_active=run is not None and run.status in _ACTIVE_STATUSES,
        start_enabled=plan is not None and plan.launchable,
    )


def with_comparison_status(render: ComparisonRender, status: str) -> ComparisonRender:
    return replace(render, status_markdown=status)


def _plans(service: EvaluationGateway) -> tuple[_Plan, ...]:
    return tuple(_plan(item) for item in service.comparison_plans())


def _runs(service: EvaluationGateway) -> tuple[_Run, ...]:
    return tuple(_run(item) for item in service.list_comparisons())


def _plan(value: object) -> _Plan:
    plan_id = _identifier(_attribute(value, "experiment_plan_id", "plan_id"))
    display_name = _single_line(_attribute(value, "display_name", "label"))
    axis = _enum_text(_attribute(value, "axis"))
    fixed = _attribute(value, "fixed_identities", default=None)
    raw_dataset_id = _attribute(value, "dataset_id", default=None)
    if raw_dataset_id is None:
        raw_dataset_id = _attribute(fixed, "dataset_id")
    dataset_id = _identifier(raw_dataset_id)
    raw_dataset_version = _attribute(value, "dataset_version", default=None)
    if raw_dataset_version is None:
        raw_dataset_version = _attribute(fixed, "dataset_version")
    dataset_version = _single_line(raw_dataset_version)
    raw_corpus_id = _attribute(value, "corpus_id", default=None)
    if raw_corpus_id is None:
        raw_corpus_id = _attribute(fixed, "corpus_id", default="unavailable")
    corpus_id = _identifier(raw_corpus_id)
    raw_corpus_version = _attribute(value, "corpus_version", default=None)
    if raw_corpus_version is None:
        raw_corpus_version = _attribute(fixed, "corpus_version", default="unavailable")
    corpus_version = _single_line(raw_corpus_version)
    raw_case_count = _attribute(value, "planned_case_count", "case_count", default=None)
    if raw_case_count is None:
        raw_case_count = _attribute(fixed, "case_count")
    case_count = _nonnegative_int(
        raw_case_count,
        positive=True,
    )
    raw_variants = _sequence(_attribute(value, "variants", "candidates"))
    variants = tuple(
        _Variant(
            variant_id=_identifier(_attribute(item, "variant_id", "candidate_id")),
            display_name=_single_line(_attribute(item, "display_name", "label")),
            axis_value=_single_line(_attribute(item, "axis_value", "value")),
            configuration_id=_identifier(
                _attribute(
                    item,
                    "configuration_id",
                    default=_attribute(item, "variant_id", "candidate_id"),
                )
            ),
        )
        for item in raw_variants
    )
    if len(variants) < 2:
        raise ComparisonDashboardError("comparison_plan_candidates_invalid")
    repeat_policy = _attribute(value, "repeat_order_policy", default=None)
    repeats = _nonnegative_int(
        _attribute(
            value,
            "repeat_count",
            "repeats_per_case",
            default=_attribute(repeat_policy, "repeats_per_case", default=1),
        ),
        positive=True,
    )
    logical_calls = _nonnegative_int(
        _attribute(
            value,
            "maximum_logical_calls",
            default=case_count * repeats * len(variants),
        ),
        positive=True,
    )
    pricing = _attribute(value, "pricing", default=None)
    currency = _single_line(
        _attribute(value, "currency", default=_attribute(pricing, "currency", default="unknown"))
    )
    return _Plan(
        plan_id=plan_id,
        display_name=display_name,
        axis=axis,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        case_count=case_count,
        variants=variants,
        baseline_variant_id=_identifier(
            _attribute(value, "baseline_variant_id", "baseline_candidate_id")
        ),
        repeats_per_case=repeats,
        maximum_logical_calls=logical_calls,
        maximum_provider_calls=_nonnegative_int(
            _attribute(value, "maximum_provider_calls"), positive=True
        ),
        cache_policy=_enum_text(_attribute(value, "cache_policy")),
        cost_estimate_status=_enum_text(
            _attribute(value, "cost_estimate_status", default="unavailable")
        ),
        cost_estimate=_scalar(_attribute(value, "cost_estimate", default=None)),
        cost_cap=_scalar(_attribute(value, "cost_cap", "maximum_cost", default=None)),
        currency=currency,
        launchable=_boolean(_attribute(value, "launchable", default=False)),
        blocking_codes=tuple(
            _identifier(item) for item in _sequence(_attribute(value, "blocking_codes", default=()))
        ),
    )


def _run(value: object) -> _Run:
    completed = _nonnegative_int(
        _attribute(value, "completed_candidates", "completed_count", default=0)
    )
    failed = _nonnegative_int(_attribute(value, "failed_candidates", "failed_count", default=0))
    remaining_value = _attribute(value, "remaining_candidates", "remaining_count", default=None)
    total_value = _attribute(value, "total_candidates", "candidate_count", default=None)
    if total_value is None:
        remaining = _nonnegative_int(remaining_value or 0)
        total = completed + failed + remaining
    else:
        total = _nonnegative_int(total_value, positive=True)
        remaining = (
            max(0, total - completed - failed)
            if remaining_value is None
            else _nonnegative_int(remaining_value)
        )
    return _Run(
        comparison_id=_identifier(_attribute(value, "comparison_id", "run_id")),
        plan_id=_identifier(_attribute(value, "experiment_plan_id", "plan_id")),
        status=_enum_text(_attribute(value, "status")),
        completed_candidates=completed,
        failed_candidates=failed,
        active_candidates=_nonnegative_int(_attribute(value, "active_candidates", default=0)),
        remaining_candidates=remaining,
        total_candidates=total,
        completed_cases=_nonnegative_int(_attribute(value, "completed_cases", default=0)),
        failed_cases=_nonnegative_int(_attribute(value, "failed_cases", default=0)),
        provider_calls=_scalar(_attribute(value, "provider_calls", default=0)),
        incurred_cost=_scalar(_attribute(value, "incurred_cost", default=None)),
        known_partial_cost=_scalar(_attribute(value, "known_partial_cost", default=Decimal(0))),
        cost_complete=bool(_attribute(value, "cost_complete", default=False)),
        cost_unknown_reasons=tuple(
            _identifier(item)
            for item in _sequence(_attribute(value, "cost_unknown_reasons", default=()))
        ),
        currency=_optional_identifier(_attribute(value, "currency", default=None)),
        safe_error_code=_optional_identifier(_attribute(value, "safe_error_code", default=None)),
        created_at=_datetime(_attribute(value, "created_at")),
        updated_at=_datetime(_attribute(value, "updated_at")),
        completed_at=_optional_datetime(_attribute(value, "completed_at", default=None)),
    )


def _summary(value: object | None, expected_id: str) -> _Summary | None:
    if value is None:
        return None
    resolved_id = _identifier(_attribute(value, "comparison_id", "run_id"))
    if resolved_id != expected_id:
        raise ComparisonDashboardError("comparison_summary_identity_mismatch")
    raw_controlled = _sequence(_attribute(value, "controlled_dimensions", default=()))
    controlled = tuple(
        (
            _single_line(_attribute(item, "name", "dimension")),
            _single_line(_attribute(item, "value")),
        )
        for item in raw_controlled
    )
    candidates = tuple(_candidate(item) for item in _sequence(_attribute(value, "candidates")))
    categories = tuple(
        _category(item)
        for item in _sequence(_attribute(value, "categories", "category_results", default=()))
    )
    recommendation_value = _attribute(value, "recommendation", default=None)
    recommendation = (
        _Recommendation("unavailable", None, ("recommendation-not-recorded",))
        if recommendation_value is None
        else _recommendation(recommendation_value)
    )
    shared_setup = _shared_setup(_attribute(value, "shared_setup", default=None))
    return _Summary(
        comparison_id=resolved_id,
        status=_enum_text(_attribute(value, "status")),
        evidence_status=_enum_text(_attribute(value, "evidence_status", default="unavailable")),
        gate_status=_enum_text(_attribute(value, "gate_status", default="unavailable")),
        compatibility_state=_enum_text(
            _attribute(value, "compatibility_state", default="unavailable")
        ),
        compatibility_issues=tuple(
            _identifier(item)
            for item in _sequence(_attribute(value, "compatibility_issues", default=()))
        ),
        controlled_dimensions=controlled,
        candidates=candidates,
        categories=categories,
        comparison_metrics=tuple(
            _metric(item) for item in _sequence(_attribute(value, "comparison_metrics", default=()))
        ),
        recommendation=recommendation,
        shared_setup=shared_setup,
        provider_call_count=_scalar(
            _attribute(
                value,
                "provider_call_count",
                default=UnavailableValue(reason="comparison-provider-calls-not-recorded"),
            )
        ),
        known_partial_cost=_scalar(
            _attribute(
                value,
                "known_partial_cost",
                default=UnavailableValue(reason="comparison-cost-not-recorded"),
            )
        ),
        total_cost=_scalar(
            _attribute(
                value,
                "total_cost",
                default=UnavailableValue(reason="comparison-cost-not-recorded"),
            )
        ),
        cost_complete=bool(_attribute(value, "cost_complete", default=False)),
        cost_unknown_reasons=tuple(
            _identifier(item)
            for item in _sequence(_attribute(value, "cost_unknown_reasons", default=()))
        ),
        currency=_scalar(
            _attribute(
                value,
                "currency",
                default=UnavailableValue(reason="comparison-cost-not-recorded"),
            )
        ),
    )


def _candidate(value: object) -> _Candidate:
    metrics = tuple(_metric(item) for item in _sequence(_attribute(value, "metrics", default=())))
    status = _enum_text(_attribute(value, "status"))
    return _Candidate(
        candidate_id=_identifier(_attribute(value, "candidate_id", "variant_id")),
        display_name=_single_line(_attribute(value, "display_name", "label")),
        axis_value=_single_line(
            _attribute(
                value,
                "axis_value",
                default=_attribute(value, "candidate_id", "variant_id"),
            )
        ),
        evaluation_run_id=_identifier(
            _attribute(value, "evaluation_run_id", default="unavailable")
        ),
        configuration_id=_identifier(_attribute(value, "configuration_id", default="unavailable")),
        status=status,
        evidence_status=_enum_text(
            _attribute(
                value,
                "evidence_status",
                default="available" if status == "completed" and metrics else "unavailable",
            )
        ),
        baseline=bool(_attribute(value, "is_baseline", "baseline", default=False)),
        safe_error_code=_optional_identifier(_attribute(value, "safe_error_code", default=None)),
        failed_case_count=_nonnegative_int(_attribute(value, "failed_case_count", default=0)),
        provider_call_count=_nonnegative_int(_attribute(value, "provider_call_count", default=0)),
        known_partial_cost=_scalar(_attribute(value, "known_partial_cost", default=Decimal(0))),
        total_cost=_scalar(_attribute(value, "total_cost", default=None)),
        cost_complete=bool(_attribute(value, "cost_complete", default=False)),
        cost_unknown_reasons=tuple(
            _identifier(item)
            for item in _sequence(_attribute(value, "cost_unknown_reasons", default=()))
        ),
        currency=_optional_identifier(_attribute(value, "currency", default=None)),
        metrics=metrics,
        gates=tuple(
            _gate(item) for item in _sequence(_attribute(value, "gates", default=()))
        ),
    )


def _gate(value: object) -> _Gate:
    return _Gate(
        gate_id=_identifier(_attribute(value, "gate_id")),
        status=_enum_text(_attribute(value, "status")),
        required_for_selection=bool(
            _attribute(value, "required_for_selection", default=True)
        ),
        reason_codes=tuple(
            _identifier(item)
            for item in _sequence(_attribute(value, "reason_codes", default=()))
        ),
    )


def _metric(value: object) -> _Metric:
    state = _enum_text(_attribute(value, "state", "status", default="unavailable"))
    return _Metric(
        metric_id=_identifier(_attribute(value, "metric_id")),
        value=_scalar(_attribute(value, "value", default=None)),
        unit=_single_line(_attribute(value, "unit", default="unavailable")),
        numerator=_scalar(_attribute(value, "numerator", default=None)),
        denominator=_scalar(_attribute(value, "denominator", default=None)),
        state=state,
        gate_state=_enum_text(_attribute(value, "gate_state", "gate_status", default=state)),
        baseline_delta=_scalar(_attribute(value, "baseline_delta", "delta", default=None)),
    )


def _category(value: object) -> _Category:
    return _Category(
        candidate_id=_identifier(_attribute(value, "candidate_id", "variant_id")),
        category_id=_identifier(_attribute(value, "category_id")),
        case_count=_nonnegative_int(_attribute(value, "case_count")),
        metrics=tuple(_metric(item) for item in _sequence(_attribute(value, "metrics"))),
    )


def _recommendation(value: object) -> _Recommendation:
    return _Recommendation(
        state=_enum_text(_attribute(value, "state", "status")),
        selected_candidate_id=_optional_identifier(
            _attribute(
                value,
                "selected_candidate_id",
                "selected_variant_id",
                "candidate_id",
                default=None,
            )
        ),
        rationale_codes=tuple(
            _identifier(item)
            for item in _sequence(_attribute(value, "rationale_codes", "reasons", default=()))
        ),
    )


def _shared_setup(value: object | None) -> _SharedSetup:
    if value is None:
        unavailable = UnavailableValue(reason="setup-evidence-not-recorded")
        return _SharedSetup(
            status="unavailable",
            safe_error_code=None,
            provider_call_count=unavailable,
            known_partial_cost=unavailable,
            total_cost=unavailable,
            currency=unavailable,
            provider_calls_complete=False,
            cost_complete=False,
            unknown_reasons=("setup-evidence-not-recorded",),
        )
    return _SharedSetup(
        status=_enum_text(_attribute(value, "status")),
        safe_error_code=_optional_identifier(_attribute(value, "safe_error_code", default=None)),
        provider_call_count=_scalar(_attribute(value, "provider_call_count")),
        known_partial_cost=_scalar(_attribute(value, "known_partial_cost")),
        total_cost=_scalar(_attribute(value, "total_cost")),
        currency=_scalar(_attribute(value, "currency")),
        provider_calls_complete=_boolean(
            _attribute(value, "provider_calls_complete", default=True)
        ),
        cost_complete=_boolean(_attribute(value, "cost_complete")),
        unknown_reasons=tuple(
            _identifier(item)
            for item in _sequence(_attribute(value, "unknown_reasons", default=()))
        ),
    )


def _manifest(value: object | None, expected_id: str) -> _Manifest | None:
    if value is None:
        return None
    resolved_id = _identifier(_attribute(value, "comparison_id", "run_id"))
    if resolved_id != expected_id:
        raise ComparisonDashboardError("comparison_manifest_identity_mismatch")
    artifacts = tuple(
        _Descriptor(
            artifact_id=_identifier(_attribute(item, "artifact_id")),
            artifact_format=_single_line(_attribute(item, "format")),
            schema_version=_single_line(_attribute(item, "schema_version")),
            media_type=_single_line(_attribute(item, "media_type")),
            digest=_single_line(_attribute(item, "sha256_digest", "digest")),
            byte_size=_nonnegative_int(_attribute(item, "byte_size")),
            created_at=_datetime(_attribute(item, "created_at")),
        )
        for item in _sequence(_attribute(value, "artifacts"))
    )
    return _Manifest(resolved_id, artifacts)


def _plan_row(plan: _Plan) -> tuple[object, ...]:
    estimate = (
        _unavailable(plan.cost_estimate_status)
        if plan.cost_estimate is None
        else _display(plan.cost_estimate)
    )
    cap = _unavailable("not-declared") if plan.cost_cap is None else _display(plan.cost_cap)
    return (
        plan.plan_id,
        plan.axis,
        plan.dataset_id,
        plan.dataset_version,
        ", ".join(item.variant_id for item in plan.variants),
        plan.baseline_variant_id,
        plan.case_count,
        plan.repeats_per_case,
        plan.maximum_logical_calls,
        plan.maximum_provider_calls,
        plan.cache_policy,
        estimate,
        f"{cap} {plan.currency}",
        plan.launchable,
        ", ".join(plan.blocking_codes),
        plan.corpus_id,
        plan.corpus_version,
        ", ".join(f"{item.variant_id}={item.configuration_id}" for item in plan.variants),
    )


def _history_row(
    run: _Run,
    plan: _Plan | None,
    summary: _Summary | None,
    redactor: Redactor,
) -> tuple[object, ...]:
    configurations = (
        _unavailable("registered-plan-unavailable")
        if plan is None
        else _safe_text(
            ", ".join(f"{item.variant_id}={item.configuration_id}" for item in plan.variants),
            redactor,
        )
    )
    provider_calls = summary.provider_call_count if summary is not None else run.provider_calls
    known_partial_cost = (
        summary.known_partial_cost if summary is not None else run.known_partial_cost
    )
    total_cost = summary.total_cost if summary is not None else run.incurred_cost
    cost_complete = summary.cost_complete if summary is not None else run.cost_complete
    cost_unknown_reasons = (
        summary.cost_unknown_reasons if summary is not None else run.cost_unknown_reasons
    )
    currency = summary.currency if summary is not None else run.currency
    if cost_complete:
        exact_cost = (
            known_partial_cost
            if total_cost is None or isinstance(total_cost, UnavailableValue)
            else total_cost
        )
        currency_text = "" if currency is None else _display(currency)
        cost = f"{_display(exact_cost)} {currency_text}".strip()
    else:
        cost = (
            f">= {_display(known_partial_cost)} {_display(currency)} "
            f"(incomplete: {', '.join(cost_unknown_reasons) or 'unknown'})"
        )
    return (
        run.comparison_id,
        run.plan_id,
        run.status,
        run.completed_candidates,
        run.failed_candidates,
        run.remaining_candidates,
        run.total_candidates,
        run.created_at.isoformat(),
        "" if run.completed_at is None else run.completed_at.isoformat(),
        run.safe_error_code or "",
        _unavailable("registered-plan-unavailable") if plan is None else plan.dataset_id,
        _unavailable("registered-plan-unavailable") if plan is None else plan.dataset_version,
        _unavailable("registered-plan-unavailable") if plan is None else plan.corpus_id,
        _unavailable("registered-plan-unavailable") if plan is None else plan.corpus_version,
        configurations,
        run.active_candidates,
        run.completed_cases,
        run.failed_cases,
        (provider_calls if isinstance(provider_calls, int) else _display(provider_calls)),
        cost,
        "unavailable" if summary is None else summary.evidence_status,
        "unavailable" if summary is None else summary.gate_status,
    )


def _controlled_rows(
    summary: _Summary | None,
    redactor: Redactor,
) -> tuple[tuple[object, ...], ...]:
    if summary is None:
        return (("compatibility", _unavailable("not-recorded")),)
    rows: list[tuple[object, ...]] = [("compatibility", summary.compatibility_state)]
    rows.extend(
        (_safe_text(name, redactor), _safe_text(value, redactor))
        for name, value in summary.controlled_dimensions
    )
    rows.extend(("compatibility_issue", item) for item in summary.compatibility_issues)
    return tuple(rows)


def _shared_setup_rows(
    summary: _Summary | None,
) -> tuple[tuple[object, ...], ...]:
    if summary is None:
        return (
            (
                "shared-setup",
                "status",
                _unavailable("setup-evidence-not-recorded"),
            ),
        )
    setup = summary.shared_setup
    currency = _display(setup.currency)
    rows: list[tuple[object, ...]] = [
        ("shared-setup", "status", setup.status),
        ("shared-setup", "provider-call-count", _display(setup.provider_call_count)),
        (
            "shared-setup",
            "provider-calls-complete",
            str(setup.provider_calls_complete).lower(),
        ),
        (
            "shared-setup",
            "known-partial-cost",
            f"{_display(setup.known_partial_cost)} {currency}",
        ),
        (
            "shared-setup",
            "total-cost",
            f"{_display(setup.total_cost)} {currency}",
        ),
        ("shared-setup", "cost-complete", str(setup.cost_complete).lower()),
        (
            "inclusive-comparison",
            "provider-call-count",
            _display(summary.provider_call_count),
        ),
        (
            "inclusive-comparison",
            "known-partial-cost",
            f"{_display(summary.known_partial_cost)} {_display(summary.currency)}",
        ),
        (
            "inclusive-comparison",
            "total-cost",
            f"{_display(summary.total_cost)} {_display(summary.currency)}",
        ),
        (
            "inclusive-comparison",
            "cost-complete",
            str(summary.cost_complete).lower(),
        ),
    ]
    if setup.safe_error_code is not None:
        rows.append(("shared-setup", "safe-error", setup.safe_error_code))
    if setup.unknown_reasons:
        rows.append(
            (
                "shared-setup",
                "unknown-reasons",
                ", ".join(setup.unknown_reasons),
            )
        )
    if summary.cost_unknown_reasons:
        rows.append(
            (
                "inclusive-comparison",
                "unknown-reasons",
                ", ".join(summary.cost_unknown_reasons),
            )
        )
    return tuple(rows)


def _candidate_rows(
    summary: _Summary | None,
    redactor: Redactor,
) -> tuple[tuple[object, ...], ...]:
    if summary is None:
        return ()
    rows: list[tuple[object, ...]] = []
    for candidate in summary.candidates:
        evidence_columns = (
            _safe_text(candidate.axis_value, redactor),
            candidate.configuration_id,
            candidate.evaluation_run_id,
            candidate.evidence_status,
            candidate.failed_case_count,
            candidate.provider_call_count,
            f"{_display(candidate.known_partial_cost)} {candidate.currency or 'unknown'}",
            (
                _unavailable("cost-unavailable")
                if candidate.total_cost is None
                else f"{_display(candidate.total_cost)} {candidate.currency or 'unknown'}"
            ),
            str(candidate.cost_complete).lower(),
            ", ".join(candidate.cost_unknown_reasons),
        )
        for gate in candidate.gates:
            rows.append(
                (
                    candidate.candidate_id,
                    _safe_text(candidate.display_name, redactor),
                    candidate.status,
                    candidate.baseline,
                    f"gate:{gate.gate_id}",
                    gate.status,
                    (
                        "selection-required"
                        if gate.required_for_selection
                        else "diagnostic-phase16"
                    ),
                    _unavailable("gate-not-numeric"),
                    _unavailable("gate-not-numeric"),
                    gate.status,
                    _unavailable("gate-not-delta"),
                    ", ".join(gate.reason_codes),
                    *evidence_columns,
                )
            )
        if not candidate.metrics:
            rows.append(
                (
                    candidate.candidate_id,
                    _safe_text(candidate.display_name, redactor),
                    candidate.status,
                    candidate.baseline,
                    "",
                    _unavailable("candidate-evidence-missing"),
                    "unavailable",
                    _unavailable("candidate-evidence-missing"),
                    _unavailable("candidate-evidence-missing"),
                    "unavailable",
                    _unavailable("candidate-evidence-missing"),
                    candidate.safe_error_code or "",
                    *evidence_columns,
                )
            )
            continue
        for metric in candidate.metrics:
            rows.append(
                (
                    candidate.candidate_id,
                    _safe_text(candidate.display_name, redactor),
                    candidate.status,
                    candidate.baseline,
                    metric.metric_id,
                    _display(metric.value),
                    metric.unit,
                    _display(metric.numerator),
                    _denominator(metric.denominator, metric.state),
                    metric.gate_state,
                    _display(metric.baseline_delta),
                    candidate.safe_error_code or "",
                    *evidence_columns,
                )
            )
    return tuple(rows)


def _comparison_metric_rows(
    summary: _Summary | None,
) -> tuple[tuple[object, ...], ...]:
    if summary is None:
        return ()
    return tuple(
        (
            metric.metric_id,
            _display(metric.value),
            metric.unit,
            _display(metric.numerator),
            _denominator(metric.denominator, metric.state),
            metric.state,
            metric.gate_state,
        )
        for metric in summary.comparison_metrics
    )


def _category_rows(
    summary: _Summary | None,
    redactor: Redactor,
) -> tuple[tuple[object, ...], ...]:
    if summary is None:
        return ()
    return tuple(
        (
            category.candidate_id,
            category.category_id,
            category.case_count,
            metric.metric_id,
            _display(metric.value),
            metric.unit,
            _denominator(metric.denominator, metric.state),
            _display(metric.baseline_delta),
            metric.state,
        )
        for category in summary.categories
        for metric in category.metrics
        if _safe_text(category.candidate_id, redactor)
    )


def _plot_rows(
    summary: _Summary | None,
    redactor: Redactor,
) -> tuple[tuple[str, str, float], ...]:
    if summary is None:
        return ()
    rows: list[tuple[str, str, float]] = []
    for candidate in summary.candidates:
        label = _safe_text(candidate.display_name, redactor)
        for metric in candidate.metrics:
            delta = metric.baseline_delta
            if isinstance(delta, bool) or not isinstance(delta, (int, float, Decimal)):
                continue
            numeric = float(delta)
            if math.isfinite(numeric):
                rows.append((label, metric.metric_id, numeric))
    return tuple(rows)


def _artifact_rows(manifest: _Manifest | None) -> tuple[tuple[object, ...], ...]:
    if manifest is None:
        return ()
    return tuple(
        (
            item.artifact_id,
            item.artifact_format,
            item.schema_version,
            item.media_type,
            item.digest,
            item.byte_size,
            item.created_at.isoformat(),
        )
        for item in manifest.artifacts
    )


def _artifact_links(manifest: _Manifest | None) -> str:
    if manifest is None:
        return "Comparison artifacts unavailable. / 对比制品不可用。"
    run = quote(manifest.comparison_id, safe="")
    lines = [
        "Validated same-origin comparison downloads / 已验证的同源对比下载:",
        f"- [Integrity manifest / 完整性清单]({_API_PREFIX}/comparisons/{run}/artifacts)",
    ]
    lines.extend(
        f"- [{item.artifact_id}]({_API_PREFIX}/comparisons/{run}/artifacts/"
        f"{quote(item.artifact_id, safe='')}) - `{item.artifact_format}`, "
        f"`{item.byte_size}` bytes"
        for item in manifest.artifacts
    )
    return "\n".join(lines)


def _progress_markdown(run: _Run | None) -> str:
    if run is None:
        return "No comparison selected. / 未选择对比。"
    return (
        f"**{run.status}** - completed `{run.completed_candidates}` - failed "
        f"`{run.failed_candidates}` - active `{run.active_candidates}` - remaining "
        f"`{run.remaining_candidates}` - total `{run.total_candidates}`  \n"
        f"Cases completed/failed / 用例完成/失败: `{run.completed_cases}` / "
        f"`{run.failed_cases}`; provider calls / 模型调用: "
        f"`{_display(run.provider_calls)}`  \n"
        f"Comparison: `{run.comparison_id}`"
    )


def _gate_markdown(run: _Run | None, summary: _Summary | None) -> str:
    if run is None:
        return "### No comparison evidence selected / 未选择对比证据"
    if run.status in {"failed", "invalid"}:
        return (
            "### Comparison failed / 对比失败\n"
            f"Safe reason / 安全原因: `{run.safe_error_code or 'comparison-failed'}`. "
            "Missing evidence is not success. / 缺失证据不代表成功。"
        )
    if summary is None or summary.evidence_status == "unavailable":
        return (
            "### Evidence unavailable / 证据不可用\n"
            "The comparison gate is unavailable, not passing. / "
            "对比门槛不可用, 并非通过。"
        )
    if summary.evidence_status == "incomplete":
        return (
            "### Evidence incomplete / 证据不完整\n"
            "Partial candidates remain visible without a final recommendation. / "
            "保留部分候选结果, 但不给出最终推荐。"
        )
    if summary.compatibility_state == "incompatible":
        return "### INCOMPATIBLE / 不兼容\nControlled identities did not validate."
    if summary.gate_status == "passed":
        return (
            "### PASS / 通过\n"
            "All declared Phase 15 selection gates passed; advanced acceptance "
            "remains diagnostic until Phase 16."
        )
    if summary.gate_status == "failed":
        return "### FAIL / 未通过\nOne or more required selection gates failed."
    return "### Gate unavailable / 门槛不可用"


def _cache_conclusion_markdown(
    plan: _Plan | None,
    summary: _Summary | None,
    redactor: Redactor,
) -> str:
    if plan is None or plan.axis != "cache-behavior":
        return (
            "Cache revision/equivalence conclusion is not applicable to this plan. / "
            "缓存版本和等价性结论不适用于此计划。"
        )
    revision = _safe_text(
        f"{plan.corpus_id} version {plan.corpus_version}",
        redactor,
    )
    if summary is None or summary.evidence_status != "available":
        return (
            "### Cache evidence unavailable / 缓存证据不可用\n"
            f"Plan-bound corpus revision / 计划绑定语料版本: `{revision}`.  \n"
            "Retrieval equivalence is not confirmed from incomplete evidence. / "
            "检索等价性尚未由完整证据确认。"
        )
    metric = next(
        (
            item
            for item in summary.comparison_metrics
            if item.metric_id == "comparison-cache-retrieval-equivalence-rate"
        ),
        None,
    )
    if metric is None:
        return (
            "### Cache equivalence unavailable / 缓存等价性不可用\n"
            f"Plan-bound corpus revision / 计划绑定语料版本: `{revision}`.  \n"
            "The authoritative equivalence observation was not recorded; this is not a pass. / "
            "未记录权威等价性观测, 因此不能判定为通过。"
        )
    value = metric.value
    numerator = metric.numerator
    denominator = metric.denominator
    confirmed = (
        metric.state != "unavailable"
        and not isinstance(value, bool)
        and isinstance(value, (int, float, Decimal))
        and not isinstance(numerator, bool)
        and isinstance(numerator, (int, float, Decimal))
        and not isinstance(denominator, bool)
        and isinstance(denominator, int)
        and denominator > 0
        and math.isclose(float(value), 1.0, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(
            float(numerator),
            float(denominator),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    conclusion_en = "CONFIRMED" if confirmed else "NOT CONFIRMED"
    conclusion_zh = "已确认" if confirmed else "未确认"
    return (
        f"### Cache retrieval equivalence: {conclusion_en} / 缓存检索等价性: {conclusion_zh}\n"
        f"Plan-bound corpus revision / 计划绑定语料版本: `{revision}`.  \n"
        f"Equivalent pairs / 等价配对: `{_display(metric.numerator)}` / "
        f"`{_denominator(metric.denominator, metric.state)}`.  \n"
        "This authoritative observation checks index-revision and ordered-retrieval evidence; "
        "no raw query or document content is displayed. / 此权威观测检查索引版本和有序检索证据, "
        "且不显示原始查询或文档内容。"
    )


def _recommendation_markdown(summary: _Summary | None, redactor: Redactor) -> str:
    if summary is None:
        return "Recommendation unavailable. / 推荐不可用。"
    recommendation = summary.recommendation
    selected = recommendation.selected_candidate_id or "none"
    reasons = ", ".join(recommendation.rationale_codes) or "not-recorded"
    return _safe_text(
        f"### Recommendation: {recommendation.state} / 推荐: {recommendation.state}\n"
        f"Selected candidate / 所选候选: `{selected}`  \n"
        f"Measured rationale codes / 测量理由代码: `{reasons}`",
        redactor,
    )


def _attribute(value: object, *names: str, default: object = _MISSING) -> object:
    if value is not None:
        for name in names:
            if hasattr(value, name):
                return getattr(value, name)
    if default is not _MISSING:
        return default
    raise ComparisonDashboardError("comparison_dto_field_missing")


def _sequence(value: object) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ComparisonDashboardError("comparison_dto_sequence_invalid")
    return tuple(value)


def _enum_text(value: object) -> str:
    resolved = value.value if isinstance(value, Enum) else value
    return _single_line(resolved)


def _identifier(value: object) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ComparisonDashboardError("comparison_identifier_invalid")
    return value


def _optional_identifier(value: object) -> str | None:
    return None if value is None else _identifier(value)


def _single_line(value: object) -> str:
    if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
        raise ComparisonDashboardError("comparison_text_invalid")
    return value


def _nonnegative_int(value: object, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise ComparisonDashboardError("comparison_count_invalid")
    return value


def _boolean(value: object) -> bool:
    if type(value) is not bool:
        raise ComparisonDashboardError("comparison_boolean_invalid")
    return value


def _datetime(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ComparisonDashboardError("comparison_timestamp_invalid")
    return value


def _optional_datetime(value: object) -> datetime | None:
    return None if value is None else _datetime(value)


def _scalar(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, Decimal, UnavailableValue)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ComparisonDashboardError("comparison_metric_invalid")
        return value
    raise ComparisonDashboardError("comparison_metric_invalid")


def _safe_text(value: object, redactor: Redactor) -> str:
    redacted = redact_output(value, redactor=redactor)
    if not isinstance(redacted, str):
        raise ComparisonDashboardError("comparison_redaction_invalid")
    return redacted


def _display(value: object) -> str:
    if isinstance(value, UnavailableValue):
        return _unavailable(value.reason)
    if value is None:
        return _unavailable("not-recorded")
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def _denominator(value: object, state: str) -> str:
    if state == "unavailable" and (value is None or value == 0):
        return _unavailable("no-eligible-observations")
    return _display(value)


def _unavailable(reason: str) -> str:
    return f"unavailable ({reason}) / 不可用 ({reason})"


__all__ = [
    "ComparisonDashboardError",
    "comparison_id",
    "render_comparison_dashboard",
    "resolve_registered_comparison_plan",
    "supports_comparison_dashboard",
    "with_comparison_status",
]
