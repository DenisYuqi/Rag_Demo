from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from rag_mvp.domain.evaluation import (
    IssueClassification,
    IssueEvidence,
    ModelAttemptStatus,
    ModelPricing,
    ModelRole,
)
from rag_mvp.domain.evaluation import (
    ModelAttempt as PersistedModelAttempt,
)
from rag_mvp.domain.evaluation import (
    TokenUsage as PersistedTokenUsage,
)
from rag_mvp.domain.ingestion import ChunkLocator
from rag_mvp.domain.qa import (
    AnswerClaim,
    Citation,
    RefusalReason,
    SafeQADiagnostics,
    StreamEventKind,
    ValidatedStreamEvent,
)
from rag_mvp.evaluation.dataset import (
    Answerability,
    CorpusReference,
    CorpusSnapshot,
    DatasetManifest,
    EvaluationCase,
    EvaluationCategory,
    EvaluationDataset,
    EvaluationLanguage,
    ExpectedFact,
    StyleExpectation,
)
from rag_mvp.evaluation.html_report import render_html_report, verify_html_parity
from rag_mvp.evaluation.json_report import canonical_report_json, validate_report
from rag_mvp.evaluation.report_builder import (
    EvaluationReportBuilder,
    IssueComparison,
    IssueRunIdentity,
    ReportBuildError,
    ReportIssueRecord,
    case_ids_content_hash,
)
from rag_mvp.evaluation.runner import (
    EvaluationCaseExecution,
    EvaluationCaseInput,
    EvaluationEnvironment,
    EvaluationRunIdentity,
    EvaluationRunManifest,
    EvaluationRunPlan,
    PersistedCaseResult,
)
from rag_mvp.evaluation.scoring import EvaluationScorecard, EvaluationScorer
from rag_mvp.observability.costs import PricingCatalog
from rag_mvp.providers.models import (
    AttemptStatus,
    ProviderRole,
)
from rag_mvp.providers.models import (
    ModelAttempt as ProviderModelAttempt,
)
from rag_mvp.providers.models import (
    TokenUsage as ProviderTokenUsage,
)

_NOW = datetime(2026, 8, 7, 2, 3, 4, tzinfo=UTC)
_DATASET_HASH = "sha256:" + "1" * 64
_CORPUS_HASH = "sha256:" + "2" * 64


def _answer_case() -> EvaluationCase:
    return EvaluationCase(
        case_id="case-answer",
        question="What is the policy?",
        language=EvaluationLanguage.ENGLISH,
        answerability=Answerability.ANSWERABLE,
        category=EvaluationCategory.ANSWERABLE_ENGLISH,
        expected_facts=(
            ExpectedFact(
                fact_id="fact-policy",
                text="The policy covers the benefit.",
                evidence_ids=("chunk-policy",),
            ),
        ),
        authoritative_evidence_ids=("chunk-policy",),
        style_expectations=(
            StyleExpectation.ANSWER_IN_REQUEST_LANGUAGE,
            StyleExpectation.CITATIONS_REQUIRED,
            StyleExpectation.CONCISE,
        ),
    )


def _refusal_case() -> EvaluationCase:
    return EvaluationCase(
        case_id="case-refusal",
        question="What is absent?",
        language=EvaluationLanguage.ENGLISH,
        answerability=Answerability.UNANSWERABLE,
        category=EvaluationCategory.UNANSWERABLE,
        style_expectations=(
            StyleExpectation.ANSWER_IN_REQUEST_LANGUAGE,
            StyleExpectation.REFUSAL_CONCISE,
        ),
    )


def _dataset() -> EvaluationDataset:
    cases = (_answer_case(), _refusal_case())
    corpus = CorpusReference.model_construct(
        snapshot_id="corpus-mvp",
        version="1.0.0",
        content_hash=_CORPUS_HASH,
        manifest_file="corpus/manifest.json",
    )
    manifest = DatasetManifest.model_construct(
        dataset_id="mvp-v1",
        version="1.0.0",
        content_hash=_DATASET_HASH,
        corpus=corpus,
    )
    return EvaluationDataset.model_construct(
        root=Path("evaluations/datasets/mvp-v1"),
        manifest=manifest,
        cases=cases,
        corpus=CorpusSnapshot.model_construct(),
        category_counts={
            EvaluationCategory.ANSWERABLE_ENGLISH: 1,
            EvaluationCategory.UNANSWERABLE: 1,
        },
        metric_eligibility_counts={},
    )


def _answer_event(content: str = "The policy covers the benefit.") -> ValidatedStreamEvent:
    return ValidatedStreamEvent(
        request_id="request-answer",
        session_id="session-answer",
        sequence=0,
        kind=StreamEventKind.ANSWER,
        response_language="en",
        content=content,
        claims=(AnswerClaim(text=content, citation_chunk_ids=("chunk-policy",)),),
        citations=(
            Citation(
                source_title="Policy",
                document_version=1,
                chunk_id="chunk-policy",
                locator=ChunkLocator(pages=(1,)),
            ),
        ),
        diagnostics=SafeQADiagnostics(
            trace_id="trace-answer",
            stage_timings_ms={"retrieval": 20.0, "generation": 40.0},
            token_counts={"input": 100, "output": 40},
        ),
        terminal=True,
    )


def _refusal_event() -> ValidatedStreamEvent:
    return ValidatedStreamEvent(
        request_id="request-refusal",
        session_id="session-refusal",
        sequence=0,
        kind=StreamEventKind.REFUSAL,
        response_language="en",
        content="The available evidence does not support an answer.",
        reason=RefusalReason.INSUFFICIENT_EVIDENCE,
        diagnostics=SafeQADiagnostics(
            trace_id="trace-refusal",
            stage_timings_ms={"retrieval": 30.0, "generation": 10.0},
            token_counts={"input": 100, "output": 40},
        ),
        terminal=True,
    )


def _results(*, answer_content: str | None = None) -> tuple[PersistedCaseResult, ...]:
    answer_event = _answer_event(answer_content or "The policy covers the benefit.")
    refusal_event = _refusal_event()
    return (
        PersistedCaseResult(
            run_id="run-candidate",
            case_id="case-answer",
            succeeded=True,
            execution=EvaluationCaseExecution(
                case_id="case-answer",
                owner_id="owner-answer",
                session_id=answer_event.session_id,
                request_id=answer_event.request_id,
                event=answer_event,
                retrieved_chunk_ids=("chunk-policy", "chunk-noise"),
                context_chunk_ids=("chunk-policy",),
                latency_ms=120.0,
            ),
            completed_at=_NOW,
        ),
        PersistedCaseResult(
            run_id="run-candidate",
            case_id="case-refusal",
            succeeded=True,
            execution=EvaluationCaseExecution(
                case_id="case-refusal",
                owner_id="owner-refusal",
                session_id=refusal_event.session_id,
                request_id=refusal_event.request_id,
                event=refusal_event,
                latency_ms=200.0,
            ),
            completed_at=_NOW,
        ),
    )


def _manifest(dataset: EvaluationDataset, scorer: EvaluationScorer) -> EvaluationRunManifest:
    scorer_versions = {metric.value: version for metric, version in scorer.scorer_versions.items()}
    identity = EvaluationRunIdentity(
        dataset_id=dataset.manifest.dataset_id,
        dataset_version=dataset.manifest.version,
        dataset_hash=dataset.manifest.content_hash,
        corpus_version=dataset.manifest.corpus.version,
        corpus_hash=dataset.manifest.corpus.content_hash,
        configuration_id="config-candidate",
        code_revision="abc12345",
        prompt_versions={"generation": "generation-v1"},
        provider_identities={
            "backend": "offline",
            "generation": "test-provider",
            "adapter": "offline-config-v1",
        },
        model_identities={"generation": "test-model"},
        generation_settings={"temperature": 0.0, "max_output_tokens": 256},
        embedding_identity={"model": "embedding-v1", "dimensions": 8},
        chunking_identity={"version": "chunk-v1", "max_tokens": 256},
        retrieval_configuration={"mode": "hybrid", "top_k": 8},
        scorer_versions=scorer_versions,
        pricing_version="pricing-v1",
        random_seeds={"runner": 7},
        environment=EvaluationEnvironment(
            python_version="3.12.11",
            platform="test",
            deployment="single-process",
        ),
    )
    plan = EvaluationRunPlan(
        run_id="run-candidate",
        identity=identity,
        cases=tuple(
            EvaluationCaseInput(
                case_id=case.case_id,
                question=case.question,
                language=case.language.value,
            )
            for case in dataset.cases
        ),
    )
    return EvaluationRunManifest.create(plan, created_at=_NOW)


def _manifest_with_identity(
    dataset: EvaluationDataset,
    identity: EvaluationRunIdentity,
) -> EvaluationRunManifest:
    plan = EvaluationRunPlan(
        run_id="run-candidate",
        identity=identity,
        cases=tuple(
            EvaluationCaseInput(
                case_id=case.case_id,
                question=case.question,
                language=case.language.value,
            )
            for case in dataset.cases
        ),
    )
    return EvaluationRunManifest.create(plan, created_at=_NOW)


def _catalog() -> PricingCatalog:
    return PricingCatalog(
        version="pricing-v1",
        entries=(
            ModelPricing(
                pricing_version="pricing-v1",
                provider="test-provider",
                model="test-model",
                currency="USD",
                input_per_million=Decimal("2"),
                output_per_million=Decimal("10"),
            ),
        ),
    )


def _attempts() -> tuple[ProviderModelAttempt | PersistedModelAttempt, ...]:
    return (
        ProviderModelAttempt(
            request_id="request-answer",
            operation_id="operation-answer",
            attempt_number=1,
            route_id="route-generation",
            role=ProviderRole.GENERATION,
            provider="test-provider",
            model="test-model",
            latency_ms=35,
            status=AttemptStatus.SUCCEEDED,
            is_fallback=False,
            usage=ProviderTokenUsage(input_tokens=100, output_tokens=40),
        ),
        PersistedModelAttempt(
            attempt_id="attempt-refusal",
            operation_id="operation-refusal",
            request_id="request-refusal",
            run_id="run-candidate",
            role=ModelRole.GENERATION,
            provider="test-provider",
            model="test-model",
            status=ModelAttemptStatus.SUCCEEDED,
            latency_ms=8,
            usage=PersistedTokenUsage(
                input_tokens=100,
                output_tokens=40,
                total_tokens_reported=140,
            ),
            created_at=_NOW,
        ),
    )


def _evidence() -> tuple[
    EvaluationDataset,
    EvaluationRunManifest,
    tuple[PersistedCaseResult, ...],
    EvaluationScorecard,
]:
    dataset = _dataset()
    scorer = EvaluationScorer()
    results = _results()
    scorecard = scorer.score(dataset, results)
    return dataset, _manifest(dataset, scorer), results, scorecard


def _builder() -> EvaluationReportBuilder:
    return EvaluationReportBuilder(pricing_catalog=_catalog(), clock=lambda: _NOW)


def test_builder_assembles_schema_valid_metrics_performance_cost_and_provenance() -> None:
    dataset, manifest, results, scorecard = _evidence()

    report = _builder().build(
        dataset=dataset,
        manifest=manifest,
        results=results,
        scorecard=scorecard,
        attempts=_attempts(),
    )

    assert validate_report(report) == report
    assert report["run_id"] == "run-candidate"
    provenance = report["provenance"]
    assert isinstance(provenance, dict)
    assert provenance["dataset"] == {
        "id": "mvp-v1",
        "version": "1.0.0",
        "content_hash": _DATASET_HASH,
    }
    assert provenance["provider_models"] == {
        "generation": {"provider": "test-provider", "model": "test-model"}
    }
    assert provenance["provider_metadata"] == {
        "adapter": "offline-config-v1",
        "backend": "offline",
    }
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    aggregate = metrics["aggregate"]
    assert isinstance(aggregate, dict)
    assert aggregate["faithfulness"]["value"] == 1
    assert metrics["categories"]["unanswerable"]["case_count"] == 1
    assert len(report["case_results"]) == 2
    assert report["failed_cases"] == []
    performance = report["performance"]
    assert performance["complete_latency_ms"] == {
        "count": 2,
        "p50": 120.0,
        "p90": 200.0,
        "p99": 200.0,
        "max": 200.0,
    }
    assert performance["stage_latency_ms"]["retrieval"]["p90"] == 30.0
    cost = report["cost"]
    assert cost["attempt_count"] == 2
    assert cost["input_tokens"] == 200
    assert cost["output_tokens"] == 80
    assert cost["estimated_cost"] == "0.0012"
    assert cost["cost_per_1000_calls"] == "0.6000"
    privacy = report["privacy"]
    assert privacy["passed"] is True
    gate = report["gate"]
    assert gate["quality_passed"] is True
    assert gate["issues_passed"] is False
    assert gate["final_passed"] is False
    assert gate["failures"] == ["issues-incomplete"]


def test_builder_rejects_quality_when_one_case_execution_failed() -> None:
    dataset = _dataset()
    scorer = EvaluationScorer()
    successful = _results()[0]
    failed = PersistedCaseResult(
        run_id="run-candidate",
        case_id="case-refusal",
        succeeded=False,
        safe_error_code="case_execution_failed",
        completed_at=_NOW,
    )
    results = (successful, failed)
    scorecard = scorer.score(dataset, results)

    report = _builder().build(
        dataset=dataset,
        manifest=_manifest(dataset, scorer),
        results=results,
        scorecard=scorecard,
        attempts=_attempts()[:1],
    )

    gate = report["gate"]
    assert isinstance(gate, dict)
    assert gate["valid"] is False
    assert gate["quality_passed"] is False
    assert gate["final_passed"] is False
    assert "case-execution-failed" in gate["failures"]
    assert report["failed_cases"] == [
        {
            "case_id": "case-refusal",
            "category": "unanswerable",
            "outcome": "error",
            "failed_metrics": [
                "faithfulness",
                "context_precision",
                "answer_completeness",
                "style_consistency",
                "refusal_appropriateness",
            ],
            "rationale": "case_execution_failed",
            "safe_error_code": "case_execution_failed",
        }
    ]


@pytest.mark.parametrize(
    ("raw_value", "counter"),
    [
        ("alice@example.com", "raw_supported_pii_matches"),
        ("sk-abcdefghijklmnopqrstuvwx", "raw_secret_matches"),
    ],
)
def test_builder_scans_raw_persisted_sensitive_values_without_copying_them(
    raw_value: str,
    counter: str,
) -> None:
    dataset = _dataset()
    scorer = EvaluationScorer()
    results = _results(answer_content=f"The policy covers the benefit; contact {raw_value}.")
    scorecard = scorer.score(dataset, results)
    manifest = _manifest(dataset, scorer)

    report = _builder().build(
        dataset=dataset,
        manifest=manifest,
        results=results,
        scorecard=scorecard,
        attempts=_attempts(),
    )

    privacy = report["privacy"]
    gate = report["gate"]
    assert privacy[counter] >= 1
    assert privacy["passed"] is False
    assert gate["privacy_passed"] is False
    assert gate["final_passed"] is False
    assert raw_value not in canonical_report_json(report)


def test_builder_does_not_treat_numeric_evidence_as_raw_pii() -> None:
    dataset = _dataset()
    scorer = EvaluationScorer()
    original_results = _results()
    execution = original_results[0].execution
    assert execution is not None
    results = (
        original_results[0].model_copy(
            update={"execution": execution.model_copy(update={"latency_ms": 843.32269999868})}
        ),
        original_results[1],
    )
    scorecard = scorer.score(dataset, results)

    report = _builder().build(
        dataset=dataset,
        manifest=_manifest(dataset, scorer),
        results=results,
        scorecard=scorecard,
        attempts=_attempts(),
    )

    privacy = report["privacy"]
    assert privacy["raw_supported_pii_matches"] == 0
    assert privacy["raw_secret_matches"] == 0
    assert privacy["passed"] is True


def test_builder_marks_unknown_cost_invalid_instead_of_coercing_it_to_zero() -> None:
    dataset, manifest, results, scorecard = _evidence()

    report = _builder().build(
        dataset=dataset,
        manifest=manifest,
        results=results,
        scorecard=scorecard,
        attempts=(),
    )

    cost = report["cost"]
    gate = report["gate"]
    assert cost["complete"] is False
    assert cost["estimated_cost"] is None
    assert cost["cost_per_1000_calls"] is None
    assert cost["unknown_reasons"] == ["no-attempts"]
    assert gate["valid"] is False
    assert "cost-incomplete" in gate["failures"]


def test_builder_reports_missing_latency_and_metric_denominators_without_fabrication() -> None:
    dataset = _dataset()
    scorer = EvaluationScorer()
    results = tuple(
        PersistedCaseResult(
            run_id="run-candidate",
            case_id=case.case_id,
            succeeded=False,
            safe_error_code="case-execution-failed",
            completed_at=_NOW,
        )
        for case in dataset.cases
    )
    scorecard = scorer.score(dataset, results)
    manifest = _manifest(dataset, scorer)

    report = _builder().build(
        dataset=dataset,
        manifest=manifest,
        results=results,
        scorecard=scorecard,
        attempts=(),
    )

    performance = report["performance"]
    metrics = report["metrics"]
    gate = report["gate"]
    assert performance["latency_evidence_count"] == 0
    assert performance["complete_latency_ms"] is None
    assert performance["unknown_reasons"] == ["complete-latency-unavailable"]
    assert metrics["aggregate"]["faithfulness"]["value"] is None
    assert len(report["failed_cases"]) == 2
    assert gate["valid"] is False
    assert "quality-invalid" in gate["failures"]
    assert "performance-incomplete" in gate["failures"]
    assert validate_report(report) == report
    html = render_html_report(report)
    verify_html_parity(report, html)


def _issue(
    issue_id: str,
    *,
    baseline_value: float,
    post_fix_value: float,
    improvement: float,
    manifest: EvaluationRunManifest,
) -> ReportIssueRecord:
    case_hash = case_ids_content_hash(manifest.case_ids)
    scorer_version = manifest.identity.scorer_versions["faithfulness"]
    baseline_run = f"baseline-{issue_id}"
    comparison = IssueComparison(
        baseline=IssueRunIdentity(
            run_id=baseline_run,
            dataset_version=manifest.identity.dataset_version,
            corpus_version=manifest.identity.corpus_version,
            case_ids_hash=case_hash,
            scorer_version=scorer_version,
            eligible_cases=1,
            configuration_id=f"baseline-config-{issue_id}",
        ),
        post_fix=IssueRunIdentity(
            run_id=manifest.run_id,
            dataset_version=manifest.identity.dataset_version,
            corpus_version=manifest.identity.corpus_version,
            case_ids_hash=case_hash,
            scorer_version=scorer_version,
            eligible_cases=1,
            configuration_id=manifest.identity.configuration_id,
        ),
    )
    return ReportIssueRecord(
        evidence=IssueEvidence(
            issue_id=issue_id,
            classification=IssueClassification.GENUINE,
            affected_case_ids=("case-answer",),
            symptom="A measured evaluation regression.",
            metric_references=("metric-faithfulness",),
            log_references=(f"log-{issue_id}",),
            run_references=(baseline_run, manifest.run_id),
            trace_references=(f"trace-{issue_id}",),
            root_cause="A bounded configuration defect.",
            exact_fix="Apply the smallest configuration correction.",
            fix_rationale="The paired evidence isolates this correction.",
            primary_metric="faithfulness",
            baseline_value=baseline_value,
            post_fix_value=post_fix_value,
            relative_improvement_percent=improvement,
        ),
        direction="higher-is-better",
        comparison=comparison,
    )


def test_two_compatible_improving_issues_enable_the_final_gate() -> None:
    dataset, manifest, results, scorecard = _evidence()
    issues = (
        _issue(
            "issue-one",
            baseline_value=0.5,
            post_fix_value=0.6,
            improvement=20.0,
            manifest=manifest,
        ),
        _issue(
            "issue-two",
            baseline_value=0.4,
            post_fix_value=0.5,
            improvement=25.0,
            manifest=manifest,
        ),
    )

    report = _builder().build(
        dataset=dataset,
        manifest=manifest,
        results=results,
        scorecard=scorecard,
        attempts=_attempts(),
        issues=issues,
    )

    gate = report["gate"]
    assert len(report["issues"]) == 2
    assert gate["issues_passed"] is True
    assert gate["final_passed"] is True
    assert gate["failures"] == []


def test_builder_rejects_cross_run_or_dataset_identity_mismatch() -> None:
    dataset, manifest, results, scorecard = _evidence()
    foreign = results[0].model_copy(update={"run_id": "run-foreign"})

    with pytest.raises(ReportBuildError, match="report_run_identity_mismatch"):
        _builder().build(
            dataset=dataset,
            manifest=manifest,
            results=(foreign, results[1]),
            scorecard=scorecard,
            attempts=_attempts(),
        )

    changed_manifest = manifest.model_copy(
        update={
            "identity": manifest.identity.model_copy(update={"dataset_hash": "sha256:" + "9" * 64})
        }
    )
    with pytest.raises(ReportBuildError, match="run_manifest_invalid"):
        _builder().build(
            dataset=dataset,
            manifest=changed_manifest,
            results=results,
            scorecard=scorecard,
            attempts=_attempts(),
        )


@pytest.mark.parametrize(
    ("providers", "models"),
    [
        (
            {
                "backend": "offline",
                "adapter": "offline-config-v1",
                "generation": "test-provider",
                "region": "test-region",
            },
            {"generation": "test-model"},
        ),
        (
            {
                "backend": "offline",
                "adapter": "offline-config-v1",
                "generation": "test-provider",
            },
            {"generation": "test-model", "embedding": "embedding-model"},
        ),
    ],
)
def test_builder_rejects_unknown_provider_metadata_and_role_mismatch(
    providers: dict[str, str],
    models: dict[str, str],
) -> None:
    dataset, manifest, results, scorecard = _evidence()
    identity = manifest.identity.model_copy(
        update={"provider_identities": providers, "model_identities": models}
    )
    incompatible = _manifest_with_identity(dataset, identity)

    with pytest.raises(ReportBuildError, match="report_provider_model_identity_mismatch"):
        _builder().build(
            dataset=dataset,
            manifest=incompatible,
            results=results,
            scorecard=scorecard,
            attempts=_attempts(),
        )
