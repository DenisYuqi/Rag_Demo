from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from rag_mvp.config.settings import Settings
from rag_mvp.domain import UnavailableValue
from rag_mvp.ui import comparison_dashboard
from rag_mvp.ui.callbacks import WorkbenchCallbacks
from rag_mvp.ui.models import BrowserSessionState
from rag_mvp.ui.services import WorkbenchServices
from rag_mvp.ui.workbench import create_workbench

pytestmark = pytest.mark.ui

_NOW = datetime(2026, 8, 7, 4, 5, 6, tzinfo=UTC)


@dataclass(frozen=True)
class Variant:
    candidate_id: str
    label: str
    value: str
    configuration_id: str = "semantic-config"


@dataclass(frozen=True)
class Plan:
    experiment_plan_id: str = "retrieval-comparison-v1"
    display_name: str = "Retrieval comparison"
    axis: str = "retrieval-strategy"
    dataset_id: str = "acceptance-v2"
    dataset_version: str = "2.0.0"
    corpus_id: str = "acceptance-corpus"
    corpus_version: str = "2.0.0"
    planned_case_count: int = 24
    variants: tuple[Variant, ...] = (
        Variant("dense", "Dense", "dense", "config-dense"),
        Variant("hybrid", "Hybrid", "hybrid", "config-hybrid"),
        Variant(
            "hybrid-rerank",
            "Hybrid plus rerank",
            "hybrid-rerank",
            "config-hybrid-rerank",
        ),
    )
    baseline_candidate_id: str = "dense"
    repeat_count: int = 1
    maximum_logical_calls: int = 72
    maximum_provider_calls: int = 720
    cache_policy: str = "bypass"
    cost_estimate_status: str = "available"
    cost_estimate: Decimal = Decimal("4.20")
    cost_cap: Decimal = Decimal("10.00")
    currency: str = "USD"
    launchable: bool = True
    blocking_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Run:
    comparison_id: str
    experiment_plan_id: str = "retrieval-comparison-v1"
    status: str = "completed"
    completed_candidates: int = 3
    failed_candidates: int = 0
    active_candidates: int = 0
    remaining_candidates: int = 0
    total_candidates: int = 3
    completed_cases: int = 72
    failed_cases: int = 0
    provider_calls: int = 144
    incurred_cost: Decimal | None = Decimal("3.25")
    known_partial_cost: Decimal = Decimal("3.25")
    cost_complete: bool = True
    cost_unknown_reasons: tuple[str, ...] = ()
    currency: str | None = "USD"
    safe_error_code: str | None = None
    created_at: datetime = _NOW
    updated_at: datetime = _NOW
    completed_at: datetime | None = _NOW


@dataclass(frozen=True)
class Dimension:
    name: str
    value: str


@dataclass(frozen=True)
class Metric:
    metric_id: str
    value: float | None
    unit: str
    numerator: float | None
    denominator: int | None
    status: str
    gate_status: str
    baseline_delta: float | None


@dataclass(frozen=True)
class Gate:
    gate_id: str
    status: str
    required_for_selection: bool
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    display_name: str
    status: str
    is_baseline: bool
    metrics: tuple[Metric, ...]
    safe_error_code: str | None = None
    raw_prompt: str = "must never be rendered person@example.com"
    axis_value: str = "hybrid"
    evaluation_run_id: str = "evaluation-run"
    configuration_id: str = "semantic-config"
    evidence_status: str = "available"
    failed_case_count: int = 0
    provider_call_count: int = 48
    known_partial_cost: Decimal = Decimal("1.00")
    total_cost: Decimal | None = Decimal("1.00")
    cost_complete: bool = True
    cost_unknown_reasons: tuple[str, ...] = ()
    currency: str | None = "USD"
    gates: tuple[Gate, ...] = (
        Gate("comparison-selection-eligibility", "passed", True),
        Gate(
            "advanced-quality-gate",
            "failed",
            False,
            ("context-precision-below-threshold",),
        ),
    )


@dataclass(frozen=True)
class Category:
    candidate_id: str
    category_id: str
    case_count: int
    metrics: tuple[Metric, ...]


@dataclass(frozen=True)
class Recommendation:
    state: str
    selected_variant_id: str | None
    rationale_codes: tuple[str, ...]


@dataclass(frozen=True)
class SharedSetup:
    status: str = "completed"
    safe_error_code: str | None = None
    provider_call_count: object = 2
    known_partial_cost: object = Decimal("0.10")
    total_cost: object = Decimal("0.10")
    currency: object = "USD"
    provider_calls_complete: bool = True
    cost_complete: bool = True
    unknown_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class Summary:
    comparison_id: str
    status: str
    evidence_status: str
    gate_status: str
    compatibility_state: str
    compatibility_issues: tuple[str, ...]
    controlled_dimensions: tuple[Dimension, ...]
    candidates: tuple[Candidate, ...]
    category_results: tuple[Category, ...]
    recommendation: Recommendation
    comparison_metrics: tuple[Metric, ...] = ()
    shared_setup: SharedSetup = field(default_factory=SharedSetup)
    provider_call_count: object = 146
    known_partial_cost: object = Decimal("3.35")
    total_cost: object = Decimal("3.35")
    cost_complete: bool = True
    cost_unknown_reasons: tuple[str, ...] = ()
    currency: object = "USD"


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    format: str
    schema_version: str
    media_type: str
    sha256_digest: str
    byte_size: int
    created_at: datetime


@dataclass(frozen=True)
class Manifest:
    comparison_id: str
    artifacts: tuple[Artifact, ...]
    backing_path: str = r"D:\private\comparison"


def _metric(
    metric_id: str,
    value: float,
    unit: str,
    delta: float,
) -> Metric:
    return Metric(metric_id, value, unit, value * 24, 24, "available", "passed", delta)


def _candidate(
    candidate_id: str,
    display_name: str,
    *,
    baseline: bool,
    multiplier: float,
) -> Candidate:
    return Candidate(
        candidate_id,
        display_name,
        "completed",
        baseline,
        (
            _metric("faithfulness", 0.9 * multiplier, "ratio", 0.02 * (multiplier - 1)),
            _metric("all-attempt-p95", 800 * multiplier, "ms", 50 * (multiplier - 1)),
            _metric("cost-per-1000", 5 * multiplier, "USD/1000", 0.5 * (multiplier - 1)),
            _metric("input-tokens", 2000 * multiplier, "tokens", 100 * (multiplier - 1)),
            _metric("error-rate", 0.01 * multiplier, "ratio", 0.001 * (multiplier - 1)),
            _metric("degradation-rate", 0.02 * multiplier, "ratio", 0.002 * (multiplier - 1)),
        ),
    )


def _completed_summary(run_id: str = "comparison-complete") -> Summary:
    candidates = (
        _candidate("dense", "Dense", baseline=True, multiplier=1),
        _candidate("hybrid", "Hybrid person@example.com", baseline=False, multiplier=1.1),
        _candidate("hybrid-rerank", "Hybrid rerank", baseline=False, multiplier=1.2),
    )
    return Summary(
        comparison_id=run_id,
        status="completed",
        evidence_status="available",
        gate_status="passed",
        compatibility_state="compatible",
        compatibility_issues=(),
        controlled_dimensions=(
            Dimension("generation.model", "model-v2"),
            Dimension("dataset.hash", "sha256:" + "1" * 64),
        ),
        candidates=candidates,
        category_results=(
            Category(
                "hybrid-rerank",
                "rerank-sensitive",
                4,
                (_metric("context-precision", 0.9, "ratio", 0.1),),
            ),
        ),
        recommendation=Recommendation(
            "recommended",
            "hybrid-rerank",
            ("mandatory-gates-passed", "selection-policy-winner"),
        ),
    )


def _manifest(run_id: str = "comparison-complete") -> Manifest:
    return Manifest(
        run_id,
        (
            Artifact(
                "comparison-report-json",
                "json",
                "comparison-report-v1",
                "application/json",
                "sha256:" + "a" * 64,
                2048,
                _NOW,
            ),
            Artifact(
                "comparison-report-html",
                "html",
                "comparison-report-v1",
                "text/html",
                "sha256:" + "b" * 64,
                4096,
                _NOW,
            ),
        ),
    )


@dataclass
class FakeComparisonGateway:
    plans: list[Plan] = field(default_factory=lambda: [Plan()])
    runs: list[Run] = field(default_factory=lambda: [Run("comparison-complete")])
    summaries: dict[str, Summary] = field(
        default_factory=lambda: {"comparison-complete": _completed_summary()}
    )
    manifests: dict[str, Manifest] = field(
        default_factory=lambda: {"comparison-complete": _manifest()}
    )
    starts: list[str] = field(default_factory=list)

    def comparison_plans(self) -> tuple[Plan, ...]:
        return tuple(self.plans)

    def list_comparisons(self) -> tuple[Run, ...]:
        return tuple(self.runs)

    def comparison_summary(self, comparison_id: str) -> Summary | None:
        return self.summaries.get(comparison_id)

    def comparison_manifest(self, comparison_id: str) -> Manifest | None:
        return self.manifests.get(comparison_id)

    async def start_comparison(self, experiment_plan_id: str) -> Run:
        self.starts.append(experiment_plan_id)
        run = Run(
            "comparison-queued",
            status="queued",
            completed_candidates=0,
            failed_candidates=0,
            remaining_candidates=3,
            completed_at=None,
        )
        self.runs.insert(0, run)
        return run


def _callbacks(gateway: FakeComparisonGateway) -> WorkbenchCallbacks:
    return WorkbenchCallbacks(
        WorkbenchServices(evaluations=gateway)  # type: ignore[arg-type]
    )


def test_compare_view_renders_authoritative_tables_plot_and_path_free_artifacts() -> None:
    rendered = _callbacks(FakeComparisonGateway()).refresh_comparisons(BrowserSessionState.create())
    metrics = {row[4]: row for row in rendered.candidate_rows if row[0] == "hybrid"}

    assert rendered.start_enabled
    assert rendered.selected_axis == "retrieval-strategy"
    assert {row[0] for row in rendered.retrieval_rows} == {
        "dense",
        "hybrid",
        "hybrid-rerank",
    }
    assert rendered.model_rows == ()
    assert "hybrid-rerank" in rendered.retrieval_recommendation_markdown
    assert "Select a model comparison plan" in rendered.model_recommendation_markdown
    assert rendered.retrieval_plot_rows
    assert "PASS" in rendered.gate_markdown
    assert ("compatibility", "compatible") in rendered.controlled_rows
    assert metrics["faithfulness"][8] == "24"
    assert metrics["all-attempt-p95"][10] != ""
    assert metrics["gate:comparison-selection-eligibility"][6] == "selection-required"
    assert metrics["gate:advanced-quality-gate"][6] == "diagnostic-phase16"
    assert {row[1] for row in rendered.plot_rows} >= {
        "faithfulness",
        "all-attempt-p95",
        "cost-per-1000",
        "input-tokens",
        "error-rate",
        "degradation-rate",
    }
    assert rendered.category_rows[0][1] == "rerank-sensitive"
    assert "hybrid-rerank" in rendered.recommendation_markdown
    assert "[REDACTED_EMAIL]" in repr(rendered.candidate_rows)
    assert "raw_prompt" not in repr(rendered)
    assert "must never be rendered" not in repr(rendered)
    assert "/api/v1/comparisons/comparison-complete/artifacts/" in (
        rendered.artifact_links_markdown
    )
    assert r"D:\private" not in repr(rendered)
    assert "file:" not in rendered.artifact_links_markdown
    assert not rendered.poll_active
    history = rendered.history_rows[0]
    assert history[10:15] == (
        "acceptance-v2",
        "2.0.0",
        "acceptance-corpus",
        "2.0.0",
        "dense=config-dense, hybrid=config-hybrid, hybrid-rerank=config-hybrid-rerank",
    )
    assert history[16:22] == (72, 0, 146, "3.35 USD", "available", "passed")
    candidate = metrics["faithfulness"]
    assert candidate[13:19] == (
        "semantic-config",
        "evaluation-run",
        "available",
        0,
        48,
        "1.00 USD",
    )
    assert ("shared-setup", "status", "completed") in rendered.shared_setup_rows
    assert (
        "inclusive-comparison",
        "provider-call-count",
        "146",
    ) in rendered.shared_setup_rows


def test_bge_comparison_download_links_preserve_the_selected_profile() -> None:
    gateway = FakeComparisonGateway()
    callbacks = WorkbenchCallbacks(
        WorkbenchServices(
            evaluation_profiles={
                "openai-api": gateway,  # type: ignore[dict-item]
                "bge-local": gateway,  # type: ignore[dict-item]
            }
        )
    )

    rendered = callbacks.refresh_comparisons(
        BrowserSessionState.create(),
        profile_id="bge-local",
    )

    assert "?retrieval_profile=bge-local" in rendered.artifact_links_markdown


def test_model_axis_populates_model_view_without_cross_axis_substitution() -> None:
    model_plan_id = "generation-model-comparison-v1"
    variants = (
        Variant("model-mini", "Model Mini", "gpt-mini", "config-mini"),
        Variant("model-large", "Model Large", "gpt-large", "config-large"),
    )
    plan = replace(
        Plan(),
        experiment_plan_id=model_plan_id,
        display_name="Generation model comparison",
        axis="generation-model",
        variants=variants,
        baseline_candidate_id="model-mini",
        maximum_logical_calls=48,
    )
    candidates = (
        replace(
            _candidate("model-mini", "Model Mini", baseline=True, multiplier=1),
            axis_value="gpt-mini",
        ),
        replace(
            _candidate("model-large", "Model Large", baseline=False, multiplier=1.1),
            axis_value="gpt-large",
        ),
    )
    summary = replace(
        _completed_summary("model-comparison-complete"),
        candidates=candidates,
        category_results=(),
        recommendation=Recommendation(
            "recommended",
            "model-mini",
            ("best-cost-quality-tradeoff",),
        ),
    )
    run = replace(
        Run("model-comparison-complete"),
        experiment_plan_id=model_plan_id,
        completed_candidates=2,
        total_candidates=2,
        completed_cases=48,
    )
    gateway = FakeComparisonGateway(
        plans=[plan],
        runs=[run],
        summaries={run.comparison_id: summary},
        manifests={run.comparison_id: _manifest(run.comparison_id)},
    )

    rendered = _callbacks(gateway).refresh_comparisons(BrowserSessionState.create())

    assert rendered.selected_axis == "generation-model"
    assert {row[0] for row in rendered.model_rows} == {"model-mini", "model-large"}
    assert rendered.retrieval_rows == ()
    assert "model-mini" in rendered.model_recommendation_markdown
    assert "Select a retrieval comparison plan" in (rendered.retrieval_recommendation_markdown)
    assert rendered.model_plot_rows


def test_zero_call_exact_history_cost_is_not_rendered_as_incomplete() -> None:
    gateway = FakeComparisonGateway(
        runs=[
            Run(
                "comparison-complete",
                provider_calls=0,
                incurred_cost=None,
                known_partial_cost=Decimal(0),
                cost_complete=True,
                currency=None,
            )
        ],
        summaries={},
        manifests={},
    )

    rendered = _callbacks(gateway).refresh_comparisons(BrowserSessionState.create())

    assert rendered.history_rows[0][19] == "0"
    assert "incomplete" not in str(rendered.history_rows[0][19])


def test_shared_setup_unknown_totals_remain_explicit_and_path_free() -> None:
    unavailable = UnavailableValue(reason="setup-ledger-integrity-unavailable")
    source = _completed_summary("comparison-setup-unknown")
    summary = Summary(
        comparison_id="comparison-setup-unknown",
        status="failed",
        evidence_status="incomplete",
        gate_status="unavailable",
        compatibility_state=source.compatibility_state,
        compatibility_issues=source.compatibility_issues,
        controlled_dimensions=source.controlled_dimensions,
        candidates=source.candidates,
        category_results=source.category_results,
        recommendation=source.recommendation,
        shared_setup=SharedSetup(
            status="failed",
            safe_error_code="comparison-shared-setup-ledger-mismatch",
            provider_call_count=unavailable,
            known_partial_cost=unavailable,
            total_cost=unavailable,
            currency=unavailable,
            provider_calls_complete=False,
            cost_complete=False,
            unknown_reasons=("setup-ledger-integrity-unavailable",),
        ),
        provider_call_count=unavailable,
        total_cost=unavailable,
        currency=unavailable,
    )
    run = Run(
        "comparison-setup-unknown",
        status="failed",
        failed_candidates=3,
        completed_candidates=0,
        safe_error_code="comparison-shared-setup-ledger-mismatch",
    )
    gateway = FakeComparisonGateway(
        runs=[run],
        summaries={run.comparison_id: summary},
        manifests={},
    )

    rendered = _callbacks(gateway).refresh_comparisons(BrowserSessionState.create())
    rows = {(row[0], row[1]): row[2] for row in rendered.shared_setup_rows}

    assert "setup-ledger-integrity-unavailable" in rows[("shared-setup", "provider-call-count")]
    assert rows[("shared-setup", "cost-complete")] == "false"
    assert rows[("shared-setup", "provider-calls-complete")] == "false"
    assert rows[("shared-setup", "safe-error")] == ("comparison-shared-setup-ledger-mismatch")
    assert "setup-ledger-integrity-unavailable" in rows[("inclusive-comparison", "total-cost")]
    assert "setup_id" not in repr(rendered.shared_setup_rows)
    assert "request_id" not in repr(rendered.shared_setup_rows)
    assert "attempt_reference" not in repr(rendered.shared_setup_rows)
    assert r"D:\private" not in repr(rendered)


def test_candidate_unknown_usage_renders_lower_bound_without_blocking_gate() -> None:
    source = _completed_summary("comparison-cost-incomplete")
    incomplete_candidate = replace(
        source.candidates[0],
        known_partial_cost=Decimal("0.01870706"),
        total_cost=None,
        cost_complete=False,
        cost_unknown_reasons=("input-usage-unknown",),
    )
    summary = replace(
        source,
        gate_status="passed",
        candidates=(incomplete_candidate, *source.candidates[1:]),
        provider_call_count=147,
        known_partial_cost=Decimal("0.02149750"),
        total_cost=UnavailableValue(reason="comparison-cost-incomplete"),
        cost_complete=False,
        cost_unknown_reasons=("input-usage-unknown",),
    )
    run = Run(
        "comparison-cost-incomplete",
        provider_calls=146,
        incurred_cost=None,
        known_partial_cost=Decimal("0.02148220"),
        cost_complete=False,
        cost_unknown_reasons=("input-usage-unknown",),
    )
    gateway = FakeComparisonGateway(
        runs=[run],
        summaries={run.comparison_id: summary},
        manifests={},
    )

    rendered = _callbacks(gateway).refresh_comparisons(BrowserSessionState.create())
    candidate = next(row for row in rendered.candidate_rows if row[0] == "dense")
    setup_rows = {(row[0], row[1]): row[2] for row in rendered.shared_setup_rows}

    assert rendered.history_rows[0][18] == 147
    assert ">= 0.02149750 USD" in str(rendered.history_rows[0][19])
    assert candidate[18] == "0.01870706 USD"
    assert "unavailable" in str(candidate[19])
    assert candidate[20] == "false"
    assert candidate[21] == "input-usage-unknown"
    assert setup_rows[("inclusive-comparison", "known-partial-cost")] == ("0.02149750 USD")
    assert setup_rows[("inclusive-comparison", "cost-complete")] == "false"
    assert "selection" in rendered.gate_markdown.casefold()


def test_cache_metrics_render_from_authoritative_top_level_with_safe_conclusion() -> None:
    plan = Plan(
        experiment_plan_id="cache-comparison-v1",
        display_name="Cache comparison",
        axis="cache-behavior",
        corpus_id="cache-corpus",
        corpus_version="2.1.0",
        variants=(
            Variant("cold", "Cold", "cold", "config-cache"),
            Variant("warm", "Warm", "warm", "config-cache"),
        ),
        baseline_candidate_id="cold",
        maximum_logical_calls=48,
        cache_policy="use",
    )
    cache_metrics = (
        Metric(
            "comparison-cache-hit-rate",
            1.0,
            "ratio-per-eligible-pair",
            24,
            24,
            "observed",
            "observed",
            None,
        ),
        Metric(
            "comparison-cache-retrieval-provider-call-reduction",
            1.0,
            "retrieval-calls-reduced-per-eligible-pair",
            24,
            24,
            "observed",
            "observed",
            None,
        ),
        Metric(
            "comparison-cache-latency-delta",
            -125.0,
            "warm-minus-cold-milliseconds-per-eligible-pair",
            -3000,
            24,
            "observed",
            "observed",
            None,
        ),
        Metric(
            "comparison-cache-retrieval-equivalence-rate",
            1.0,
            "ratio-per-eligible-pair",
            24,
            24,
            "observed",
            "observed",
            None,
        ),
    )
    summary = Summary(
        comparison_id="comparison-cache",
        status="completed",
        evidence_status="available",
        gate_status="passed",
        compatibility_state="compatible",
        compatibility_issues=(),
        controlled_dimensions=(),
        candidates=(
            Candidate("cold", "Cold", "completed", True, ()),
            Candidate("warm", "Warm", "completed", False, ()),
        ),
        category_results=(),
        recommendation=Recommendation(
            "no-recommendation",
            None,
            ("cache-axis-not-production-selection",),
        ),
        comparison_metrics=cache_metrics,
    )
    run = Run(
        "comparison-cache",
        experiment_plan_id=plan.experiment_plan_id,
        completed_candidates=2,
        total_candidates=2,
        completed_cases=48,
        provider_calls=72,
    )
    gateway = FakeComparisonGateway(
        plans=[plan],
        runs=[run],
        summaries={run.comparison_id: summary},
        manifests={},
    )

    rendered = _callbacks(gateway).refresh_comparisons(BrowserSessionState.create())
    metrics = {row[0]: row for row in rendered.comparison_metric_rows}

    assert set(metrics) == {item.metric_id for item in cache_metrics}
    assert metrics["comparison-cache-hit-rate"][4] == "24"
    assert "CONFIRMED" in rendered.cache_conclusion_markdown
    assert "已确认" in rendered.cache_conclusion_markdown
    assert "cache-corpus version 2.1.0" in rendered.cache_conclusion_markdown
    assert "raw" in rendered.cache_conclusion_markdown
    assert "person@example.com" not in repr(rendered)
    assert r"D:\private" not in repr(rendered)


@pytest.mark.asyncio
async def test_comparison_refresh_is_provider_free_and_start_is_explicit_nonblocking() -> None:
    gateway = FakeComparisonGateway()
    callbacks = _callbacks(gateway)
    state = BrowserSessionState.create()
    initial = callbacks.refresh_comparisons(state)

    for _ in range(3):
        callbacks.preview_comparison(
            initial.selected_plan_id,
            initial.selected_comparison_id,
            state,
        )
    assert gateway.starts == []

    started = await callbacks.start_registered_comparison(initial.selected_plan_id, state)

    assert gateway.starts == ["retrieval-comparison-v1"]
    assert started.selected_comparison_id == "comparison-queued"
    assert started.poll_active
    assert "queued" in started.progress_markdown
    assert started.artifact_rows == ()
    assert "unavailable" in started.artifact_links_markdown

    restarted_browser = _callbacks(gateway).refresh_comparisons(BrowserSessionState.create())
    assert any(choice[1] == "comparison-queued" for choice in restarted_browser.comparison_choices)


def test_partial_failure_and_zero_denominator_remain_visible_without_recommendation() -> None:
    failed_candidate = Candidate(
        "hybrid-rerank",
        "Hybrid rerank",
        "failed",
        False,
        (),
        "reranker-provider-failed",
    )
    unavailable = Metric(
        "cache-hit-rate",
        None,
        "ratio",
        0,
        0,
        "unavailable",
        "unavailable",
        None,
    )
    summary = _completed_summary("comparison-partial")
    summary = Summary(
        comparison_id="comparison-partial",
        status="running",
        evidence_status="incomplete",
        gate_status="unavailable",
        compatibility_state="compatible",
        compatibility_issues=(),
        controlled_dimensions=summary.controlled_dimensions,
        candidates=(
            Candidate("dense", "Dense", "completed", True, (unavailable,)),
            failed_candidate,
        ),
        category_results=(),
        recommendation=Recommendation(
            "unavailable",
            None,
            ("candidate-failure", "evidence-incomplete"),
        ),
    )
    run = Run(
        "comparison-partial",
        status="running",
        completed_candidates=1,
        failed_candidates=1,
        remaining_candidates=1,
        completed_at=None,
    )
    gateway = FakeComparisonGateway(
        runs=[run],
        summaries={run.comparison_id: summary},
        manifests={run.comparison_id: _manifest(run.comparison_id)},
    )

    rendered = _callbacks(gateway).refresh_comparisons(BrowserSessionState.create())
    rows = {row[0]: row for row in rendered.candidate_rows}

    assert "incomplete" in rendered.gate_markdown
    assert rows["dense"][8].startswith("unavailable")
    assert rows["hybrid-rerank"][2] == "failed"
    assert rows["hybrid-rerank"][11] == "reranker-provider-failed"
    assert "unavailable" in rendered.recommendation_markdown
    assert rendered.artifact_rows == ()
    assert "unavailable" in rendered.artifact_links_markdown
    assert rendered.poll_active


@pytest.mark.asyncio
async def test_blocked_registered_plan_disables_and_rejects_launch() -> None:
    blocked = Plan(
        launchable=False,
        blocking_codes=("reranker-unavailable", "pricing-unavailable"),
    )
    gateway = FakeComparisonGateway(plans=[blocked], runs=[], summaries={}, manifests={})
    callbacks = _callbacks(gateway)
    rendered = callbacks.refresh_comparisons(BrowserSessionState.create())

    assert not rendered.start_enabled
    assert "reranker-unavailable" in rendered.status_markdown
    rejected = await callbacks.start_registered_comparison(
        blocked.experiment_plan_id,
        BrowserSessionState.create(),
    )
    assert gateway.starts == []
    assert "Select a valid registered experiment plan" in rejected.status_markdown


def test_workbench_contains_real_compact_plot_and_no_legacy_compare_controls(
    tmp_path: Path,
) -> None:
    gateway = FakeComparisonGateway()
    blocks = create_workbench(
        Settings(_env_file=None, data_root=tmp_path),
        WorkbenchServices(evaluations=gateway),  # type: ignore[arg-type]
    )
    config = blocks.get_config_file()
    components = config["components"]
    tab_labels = [
        component["props"].get("label")
        for component in components
        if component["type"] == "tabitem"
    ]
    labels = {component["props"].get("label") for component in components}
    textbox_labels = {
        component["props"].get("label")
        for component in components
        if component["type"] == "textbox"
    }

    assert gateway.starts == []
    assert "Shared setup and inclusive totals / 共享设置与包含总计" in labels
    assert "Compare / 对比" in tab_labels
    assert "Compact baseline-delta plot / 紧凑基线差值图" in labels
    assert "Comparison-level evidence / 对比级证据" in labels
    assert "Cache revision and equivalence / 缓存版本和等价性" in labels
    assert any(
        component["type"] == "nativeplot"
        and component["props"].get("name") == "barplot"
        and component["props"].get("value", {}).get("mark") == "bar"
        for component in components
    )
    assert "Registered experiment plan / 已注册实验计划" in labels
    assert not any("Baseline run ID" in str(label) for label in textbox_labels)
    assert not any("Candidate run ID" in str(label) for label in textbox_labels)


def test_comparison_ui_source_and_rendered_text_are_utf8_without_mojibake() -> None:
    source_path = Path(comparison_dashboard.__file__)
    source = source_path.read_text(encoding="utf-8")
    rendered = _callbacks(FakeComparisonGateway()).refresh_comparisons(BrowserSessionState.create())
    rendered_text = repr(rendered)
    forbidden_markers = ("锟", "娌", "涓", "\ufffd")

    assert not any(marker in source for marker in forbidden_markers)
    assert not any(marker in rendered_text for marker in forbidden_markers)
    assert "对比" in source
    assert "通过" in rendered.gate_markdown
