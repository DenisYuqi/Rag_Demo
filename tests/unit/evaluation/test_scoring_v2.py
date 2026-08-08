from pathlib import Path

import pytest

from rag_mvp.config.settings import Settings
from rag_mvp.domain.evaluation import ModelAttempt, ModelAttemptStatus, ModelRole, TokenUsage
from rag_mvp.domain.ingestion import ChunkLocator
from rag_mvp.domain.qa import (
    AnswerClaim,
    Citation,
    RefusalReason,
    SafeQADiagnostics,
    StreamEventKind,
    ValidatedStreamEvent,
)
from rag_mvp.domain.retrieval import CachePolicy
from rag_mvp.evaluation.artifacts_v2 import ArtifactCatalogV2
from rag_mvp.evaluation.dataset import Answerability, EvaluationCaseV2, EvaluationLanguage
from rag_mvp.evaluation.grounding_metrics import MetricName
from rag_mvp.evaluation.json_report import decode_json_report
from rag_mvp.evaluation.plan import (
    EvaluationDatasetRegistry,
    build_evaluation_plan,
    evaluation_scorer_versions,
)
from rag_mvp.evaluation.pricing import openai_standard_pricing_catalog
from rag_mvp.evaluation.quality_gate import AdvancedMetricName
from rag_mvp.evaluation.report_v2 import parse_report_v2
from rag_mvp.evaluation.runner import (
    EvaluationCaseExecution,
    EvaluationRunManifest,
    PersistedCaseResult,
)
from rag_mvp.evaluation.scoring import (
    ADVANCED_SCORING_PIPELINE_VERSION,
    EvaluationScoringError,
)
from rag_mvp.evaluation.scoring_v2 import AdvancedScoringError, score_evaluation_v2
from rag_mvp.evaluation.standard_report_v2 import (
    build_standard_report_v2,
    publish_standard_report_v2,
)
from rag_mvp.qa.refusal import PARTIAL_EVIDENCE_MESSAGES, EvidenceDecisionCode
from rag_mvp.qa.refusal_guidance import DEFAULT_REFUSAL_GUIDANCE_CATALOG
from rag_mvp.safety.redactor import DEFAULT_REDACTOR

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DATASETS_ROOT = _REPOSITORY_ROOT / "evaluations" / "datasets"


def _event(case: EvaluationCaseV2, *, wrong_language: bool = False) -> ValidatedStreamEvent:
    expected_language = "zh-CN" if case.language is EvaluationLanguage.CHINESE else "en"
    response_language = "en" if wrong_language else expected_language
    if case.answerability is Answerability.ANSWERABLE:
        claims = tuple(
            AnswerClaim(text=fact.text, citation_chunk_ids=fact.evidence_ids)
            for fact in case.expected_facts
        )
        content = " ".join(claim.text for claim in claims)
        citation_ids = case.authoritative_evidence_ids
        return ValidatedStreamEvent(
            request_id=f"request-{case.case_id}",
            session_id=f"session-{case.case_id}",
            sequence=0,
            kind=StreamEventKind.ANSWER,
            response_language=response_language,
            content=content,
            claims=claims,
            citations=tuple(
                Citation(
                    source_title="Internal evidence",
                    document_version=1,
                    chunk_id=chunk_id,
                    locator=ChunkLocator(pages=(1,)),
                )
                for chunk_id in citation_ids
            ),
            terminal=True,
        )
    reason = RefusalReason(case.refusal_expectation.reason_codes[0])
    guidance = DEFAULT_REFUSAL_GUIDANCE_CATALOG.select(
        reason,
        expected_language,
    )
    return ValidatedStreamEvent(
        request_id=f"request-{case.case_id}",
        session_id=f"session-{case.case_id}",
        sequence=0,
        kind=StreamEventKind.REFUSAL,
        response_language=response_language,
        content=guidance.message,
        reason=reason,
        diagnostics=SafeQADiagnostics(
            metadata={
                "refusal_guidance_present": True,
                "refusal_reason_code": reason.value,
                "refusal_guidance_reason_code": guidance.reason_code.value,
                "refusal_guidance_template_id": guidance.template_id,
                "refusal_guidance_catalog_version": guidance.catalog_version,
                "refusal_guidance_language": guidance.response_language.value,
            }
        ),
        terminal=True,
    )


def _result(case: EvaluationCaseV2, *, wrong_language: bool = False) -> PersistedCaseResult:
    event = _event(case, wrong_language=wrong_language)
    retrieved = case.authoritative_evidence_ids
    return PersistedCaseResult(
        run_id="advanced-run-v2",
        case_id=case.case_id,
        succeeded=True,
        execution=EvaluationCaseExecution(
            case_id=case.case_id,
            owner_id=f"owner-{case.case_id}",
            session_id=event.session_id,
            request_id=event.request_id,
            event=event,
            cache_policy=CachePolicy.BYPASS,
            retrieved_chunk_ids=retrieved,
            context_chunk_ids=retrieved,
            latency_ms=10.0,
        ),
    )


def test_advanced_v2_scoring_uses_real_obligations_and_nonzero_denominators() -> None:
    dataset = EvaluationDatasetRegistry(_DATASETS_ROOT).resolve(
        "original-pdf-acceptance",
        "2.0.0",
    )
    cases = tuple(case for case in dataset.cases if isinstance(case, EvaluationCaseV2))

    scorecard = score_evaluation_v2(dataset, tuple(_result(case) for case in cases))

    assert scorecard.gate.valid
    assert scorecard.gate.passed
    observations = {item.metric_id: item for item in scorecard.observations}
    assert observations[AdvancedMetricName.ANSWER_COMPLIANCE.value].denominator == 18
    assert observations[AdvancedMetricName.ANSWER_COMPLIANCE.value].value == 1.0
    assert all(
        isinstance(observation.denominator, int) and observation.denominator > 0
        for observation in scorecard.observations
    )
    assert any(category.category_id == "rerank-sensitive" for category in scorecard.categories)


def test_completeness_can_pass_while_compliance_gate_fails() -> None:
    dataset = EvaluationDatasetRegistry(_DATASETS_ROOT).resolve(
        "original-pdf-acceptance",
        "2.0.0",
    )
    cases = tuple(case for case in dataset.cases if isinstance(case, EvaluationCaseV2))
    answerable_ids = tuple(
        case.case_id for case in cases if case.answerability is Answerability.ANSWERABLE
    )
    results = tuple(
        _result(case, wrong_language=case.case_id in set(answerable_ids[:2])) for case in cases
    )

    scorecard = score_evaluation_v2(dataset, results)

    completeness = next(
        aggregate
        for aggregate in scorecard.legacy.aggregates
        if aggregate.metric is MetricName.ANSWER_COMPLETENESS
    )
    compliance = next(
        observation
        for observation in scorecard.observations
        if observation.metric_id == AdvancedMetricName.ANSWER_COMPLIANCE.value
    )
    assert completeness.score == 1.0
    assert compliance.value == 16 / 18
    assert compliance.status.value == "failed"
    assert not scorecard.gate.passed


@pytest.mark.parametrize("defect", ["wrong-reason", "missing-guidance", "invalid-guidance"])
def test_refusal_reason_and_guidance_defects_lower_refusal_appropriateness(
    defect: str,
) -> None:
    dataset = EvaluationDatasetRegistry(_DATASETS_ROOT).resolve(
        "original-pdf-acceptance",
        "2.0.0",
    )
    cases = tuple(case for case in dataset.cases if isinstance(case, EvaluationCaseV2))
    results: list[PersistedCaseResult] = []
    for case in cases:
        result = _result(case)
        if case.answerability is Answerability.ANSWERABLE:
            results.append(result)
            continue
        execution = result.execution
        assert execution is not None
        event = execution.event
        if defect == "wrong-reason":
            reason = RefusalReason.OUT_OF_SCOPE
            guidance = DEFAULT_REFUSAL_GUIDANCE_CATALOG.select(
                reason,
                "zh-CN" if case.language is EvaluationLanguage.CHINESE else "en",
            )
            event = event.model_copy(
                update={
                    "reason": reason,
                    "content": guidance.message,
                    "diagnostics": SafeQADiagnostics(
                        metadata={
                            "refusal_guidance_present": True,
                            "refusal_reason_code": reason.value,
                            "refusal_guidance_reason_code": guidance.reason_code.value,
                            "refusal_guidance_template_id": guidance.template_id,
                            "refusal_guidance_catalog_version": guidance.catalog_version,
                            "refusal_guidance_language": guidance.response_language.value,
                        }
                    ),
                }
            )
        elif defect == "missing-guidance":
            event = event.model_copy(update={"diagnostics": SafeQADiagnostics()})
        else:
            event = event.model_copy(
                update={
                    "diagnostics": event.diagnostics.model_copy(
                        update={
                            "metadata": {
                                **event.diagnostics.metadata,
                                "refusal_guidance_reason_code": "safety",
                            }
                        }
                    )
                }
            )
        results.append(
            result.model_copy(update={"execution": execution.model_copy(update={"event": event})})
        )

    scorecard = score_evaluation_v2(dataset, tuple(results))
    refusal = next(
        observation
        for observation in scorecard.observations
        if observation.metric_id == AdvancedMetricName.REFUSAL_APPROPRIATENESS.value
    )
    compliance = next(
        observation
        for observation in scorecard.observations
        if observation.metric_id == AdvancedMetricName.ANSWER_COMPLIANCE.value
    )

    assert isinstance(refusal.value, float) and refusal.value < 0.90
    assert refusal.status.value == "failed"
    assert not scorecard.gate.passed
    assert compliance.denominator == 18
    assert compliance.value == 1.0
    refusal_compliance = tuple(
        result
        for result in scorecard.compliance.case_results
        if result.scored and not result.eligible
    )
    assert len(refusal_compliance) == 6
    assert any(result.score == 0.0 for result in refusal_compliance)


def test_adjudicated_claim_support_rejects_false_claim_and_accepts_paraphrase() -> None:
    dataset = EvaluationDatasetRegistry(_DATASETS_ROOT).resolve(
        "original-pdf-acceptance",
        "2.0.0",
    )
    cases = tuple(case for case in dataset.cases if isinstance(case, EvaluationCaseV2))
    target = next(case for case in cases if case.case_id == "accept-en-001")

    def scored_with(content: str):
        results: list[PersistedCaseResult] = []
        for case in cases:
            result = _result(case)
            if case.case_id == target.case_id:
                execution = result.execution
                assert execution is not None
                event = execution.event.model_copy(
                    update={
                        "content": content,
                        "claims": (
                            AnswerClaim(
                                text=content,
                                citation_chunk_ids=target.authoritative_evidence_ids,
                            ),
                        ),
                    }
                )
                result = result.model_copy(
                    update={"execution": execution.model_copy(update={"event": event})}
                )
            results.append(result)
        return score_evaluation_v2(dataset, tuple(results))

    false_scorecard = scored_with("The Moon is made of cheese.")
    paraphrase_scorecard = scored_with("Use OPS-RAG-7421 as the authoritative escalation code.")
    false_metrics = false_scorecard.legacy.per_case_by_id[target.case_id]
    paraphrase_metrics = paraphrase_scorecard.legacy.per_case_by_id[target.case_id]

    assert false_metrics[MetricName.FAITHFULNESS].score == 0.0
    assert false_metrics[MetricName.ANSWER_COMPLETENESS].score == 0.0
    assert paraphrase_metrics[MetricName.FAITHFULNESS].score == 1.0
    assert paraphrase_metrics[MetricName.ANSWER_COMPLETENESS].score == 1.0


def test_real_shaped_formatting_and_audited_source_surfaces_remain_supported() -> None:
    dataset = EvaluationDatasetRegistry(_DATASETS_ROOT).resolve(
        "original-pdf-acceptance",
        "2.0.0",
    )
    cases = tuple(case for case in dataset.cases if isinstance(case, EvaluationCaseV2))
    replacements = {
        "accept-en-001": "The authoritative RAG escalation code is `OPS-RAG-7421`.",
        "accept-zh-003": (
            "当前有效政策规定\N{FULLWIDTH COMMA}境内航班经济舱票价报销上限为人民币1800元。"
        ),
    }
    results: list[PersistedCaseResult] = []
    for case in cases:
        result = _result(case)
        replacement = replacements.get(case.case_id)
        if replacement is not None:
            execution = result.execution
            assert execution is not None
            event = execution.event.model_copy(
                update={
                    "content": replacement,
                    "claims": (
                        AnswerClaim(
                            text=replacement,
                            citation_chunk_ids=case.authoritative_evidence_ids,
                        ),
                    ),
                }
            )
            result = result.model_copy(
                update={"execution": execution.model_copy(update={"event": event})}
            )
        results.append(result)

    scorecard = score_evaluation_v2(dataset, tuple(results))

    for case_id in replacements:
        metrics = scorecard.legacy.per_case_by_id[case_id]
        assert metrics[MetricName.FAITHFULNESS].score == 1.0
        assert metrics[MetricName.ANSWER_COMPLETENESS].score == 1.0
    assert scorecard.gate.passed


@pytest.mark.parametrize(
    ("case_id", "content"),
    [
        ("accept-en-012", "Escalation owns the RAG Operations Desk."),
        ("accept-zh-012", "RAG Operations Desk 由升级事务负责。"),
        ("accept-en-001", "escalation code OPS-RAG-7421"),
        ("accept-zh-012", "升级事务 负责 RAG Operations Desk"),
        (
            "accept-en-012",
            "The RAG Operations Desk owns escalation: Escalation owns the RAG Operations Desk.",
        ),
        (
            "accept-en-012",
            "The RAG Operations Desk owns escalation - Escalation owns the RAG Operations Desk.",
        ),
        (
            "accept-zh-001",
            "权威的 RAG 升级代码是 `OPS-RAG-7421`"
            "\N{FULLWIDTH COMMA}责任团队是 RAG Operations Desk。",
        ),
        (
            "accept-en-003",
            "The withdrawn draft proposed a domestic airfare cap of CNY 2,400.",
        ),
    ],
)
def test_advanced_scoring_rejects_reversals_salads_and_compounds(
    case_id: str,
    content: str,
) -> None:
    dataset = EvaluationDatasetRegistry(_DATASETS_ROOT).resolve(
        "original-pdf-acceptance",
        "2.0.0",
    )
    cases = tuple(case for case in dataset.cases if isinstance(case, EvaluationCaseV2))
    target = next(case for case in cases if case.case_id == case_id)
    results: list[PersistedCaseResult] = []
    for case in cases:
        result = _result(case)
        if case.case_id == target.case_id:
            execution = result.execution
            assert execution is not None
            event = execution.event.model_copy(
                update={
                    "content": content,
                    "claims": (
                        AnswerClaim(
                            text=content,
                            citation_chunk_ids=target.authoritative_evidence_ids,
                        ),
                    ),
                }
            )
            result = result.model_copy(
                update={"execution": execution.model_copy(update={"event": event})}
            )
        results.append(result)

    scorecard = score_evaluation_v2(dataset, tuple(results))
    target_metrics = scorecard.legacy.per_case_by_id[target.case_id]

    assert len(scorecard.observations) == 5
    assert target_metrics[MetricName.FAITHFULNESS].score == 0.0
    assert target_metrics[MetricName.ANSWER_COMPLETENESS].score == 0.0


def test_advanced_five_metric_gate_fails_for_dataset_wide_proposition_attacks() -> None:
    dataset = EvaluationDatasetRegistry(_DATASETS_ROOT).resolve(
        "original-pdf-acceptance",
        "2.0.0",
    )
    cases = tuple(case for case in dataset.cases if isinstance(case, EvaluationCaseV2))
    results: list[PersistedCaseResult] = []
    for case in cases:
        result = _result(case)
        if case.answerability is Answerability.ANSWERABLE:
            execution = result.execution
            assert execution is not None
            claims = tuple(
                AnswerClaim(
                    text=" ".join(group.alternatives[0] for group in fact.support_anchor_groups),
                    citation_chunk_ids=fact.evidence_ids,
                )
                for fact in case.expected_facts
            )
            if case.case_id == "accept-en-012":
                claims = (
                    AnswerClaim(
                        text="Escalation owns the RAG Operations Desk.",
                        citation_chunk_ids=case.expected_facts[0].evidence_ids,
                    ),
                )
            elif case.case_id == "accept-zh-012":
                claims = (
                    AnswerClaim(
                        text="RAG Operations Desk 由升级事务负责。",
                        citation_chunk_ids=case.expected_facts[0].evidence_ids,
                    ),
                )
            content = " ".join(claim.text for claim in claims)
            event = execution.event.model_copy(update={"content": content, "claims": claims})
            result = result.model_copy(
                update={"execution": execution.model_copy(update={"event": event})}
            )
        results.append(result)

    scorecard = score_evaluation_v2(dataset, tuple(results))
    answerable = tuple(case for case in cases if case.answerability is Answerability.ANSWERABLE)

    assert len(scorecard.observations) == 5
    assert all(
        scorecard.legacy.per_case_by_id[case.case_id][MetricName.FAITHFULNESS].score == 0.0
        for case in answerable
    )
    assert all(
        scorecard.legacy.per_case_by_id[case.case_id][MetricName.ANSWER_COMPLETENESS].score == 0.0
        for case in answerable
    )
    assert not scorecard.gate.passed


def test_advanced_v2_scoring_rejects_content_only_persisted_event_tampering() -> None:
    dataset = EvaluationDatasetRegistry(_DATASETS_ROOT).resolve(
        "original-pdf-acceptance",
        "2.0.0",
    )
    cases = tuple(case for case in dataset.cases if isinstance(case, EvaluationCaseV2))
    target = next(case for case in cases if case.case_id == "accept-en-001")
    results = list(_result(case) for case in cases)
    target_index = next(
        index for index, result in enumerate(results) if result.case_id == target.case_id
    )
    persisted = results[target_index]
    execution = persisted.execution
    assert execution is not None
    tampered_event = execution.event.model_copy(update={"content": "The Moon is made of cheese."})
    results[target_index] = persisted.model_copy(
        update={"execution": execution.model_copy(update={"event": tampered_event})}
    )

    with pytest.raises(
        EvaluationScoringError,
        match="case_answer_claim_coverage_invalid",
    ):
        score_evaluation_v2(dataset, tuple(results))


@pytest.mark.parametrize(
    ("case_id", "response_language"),
    [
        ("accept-en-001", "en"),
        ("accept-zh-001", "zh-CN"),
    ],
)
def test_advanced_v2_scoring_accepts_only_the_bounded_partial_evidence_suffix(
    case_id: str,
    response_language: str,
) -> None:
    dataset = EvaluationDatasetRegistry(_DATASETS_ROOT).resolve(
        "original-pdf-acceptance",
        "2.0.0",
    )
    cases = tuple(case for case in dataset.cases if isinstance(case, EvaluationCaseV2))
    results = list(_result(case) for case in cases)
    target_index = next(index for index, result in enumerate(results) if result.case_id == case_id)
    persisted = results[target_index]
    execution = persisted.execution
    assert execution is not None
    event = execution.event
    assert event.content is not None
    partial_event = event.model_copy(
        update={
            "content": (
                f"{event.content.rstrip()}\n\n{PARTIAL_EVIDENCE_MESSAGES[response_language]}"
            ),
            "diagnostics": SafeQADiagnostics(
                metadata={
                    "decision_code": EvidenceDecisionCode.PARTIAL_EVIDENCE.value,
                }
            ),
        }
    )
    results[target_index] = persisted.model_copy(
        update={"execution": execution.model_copy(update={"event": partial_event})}
    )

    scorecard = score_evaluation_v2(dataset, tuple(results))

    assert scorecard.legacy.scoring_version == ADVANCED_SCORING_PIPELINE_VERSION
    assert scorecard.gate.passed


@pytest.mark.parametrize(
    ("decision_code", "response_language", "separator", "before", "after"),
    [
        ("answerable", "en", "\n\n", "", ""),
        ("partial-evidence", "zh-CN", "\n\n", "", ""),
        ("partial-evidence", "en", "\n", "", ""),
        ("partial-evidence", "en", "\n\n", " Unsupported assertion.", ""),
        ("partial-evidence", "en", "\n\n", "", " Additional prose."),
    ],
)
def test_advanced_v2_scoring_rejects_unbound_or_modified_partial_suffix(
    decision_code: str,
    response_language: str,
    separator: str,
    before: str,
    after: str,
) -> None:
    dataset = EvaluationDatasetRegistry(_DATASETS_ROOT).resolve(
        "original-pdf-acceptance",
        "2.0.0",
    )
    cases = tuple(case for case in dataset.cases if isinstance(case, EvaluationCaseV2))
    results = list(_result(case) for case in cases)
    target_index = next(
        index for index, result in enumerate(results) if result.case_id == "accept-en-001"
    )
    persisted = results[target_index]
    execution = persisted.execution
    assert execution is not None
    event = execution.event
    assert event.content is not None
    partial_event = event.model_copy(
        update={
            "content": (
                f"{event.content.rstrip()}{before}{separator}"
                f"{PARTIAL_EVIDENCE_MESSAGES['en']}{after}"
            ),
            "response_language": response_language,
            "diagnostics": SafeQADiagnostics(
                metadata={"decision_code": decision_code},
            ),
        }
    )
    results[target_index] = persisted.model_copy(
        update={"execution": execution.model_copy(update={"event": partial_event})}
    )

    with pytest.raises(
        EvaluationScoringError,
        match="case_answer_claim_coverage_invalid",
    ):
        score_evaluation_v2(dataset, tuple(results))


@pytest.mark.parametrize("variant", ["unknown", "duplicate", "missing"])
def test_advanced_v2_scoring_rejects_non_exact_persisted_case_sets(
    variant: str,
) -> None:
    dataset = EvaluationDatasetRegistry(_DATASETS_ROOT).resolve(
        "original-pdf-acceptance",
        "2.0.0",
    )
    cases = tuple(case for case in dataset.cases if isinstance(case, EvaluationCaseV2))
    results = list(_result(case) for case in cases)
    if variant == "unknown":
        results[-1] = results[-1].model_copy(update={"case_id": "unknown-case"})
    elif variant == "duplicate":
        results[-1] = results[-1].model_copy(update={"case_id": results[0].case_id})
    else:
        results.pop()

    with pytest.raises(AdvancedScoringError, match="persisted_case_set_mismatch"):
        score_evaluation_v2(dataset, tuple(results))


def test_advanced_run_publishes_complete_dashboard_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = EvaluationDatasetRegistry(_DATASETS_ROOT).resolve(
        "original-pdf-acceptance",
        "2.0.0",
    )
    cases = tuple(case for case in dataset.cases if isinstance(case, EvaluationCaseV2))
    results = tuple(_result(case) for case in cases)
    settings = Settings(
        _env_file=None,
        environment="test",
        provider_backend="openai",
        openai_api_key="unit-test-key",
    )
    monkeypatch.setattr(
        "rag_mvp.evaluation.plan.source_code_revision",
        lambda: "sha256:" + "1" * 64,
    )
    plan = build_evaluation_plan(dataset, settings, "advanced-run-v2")
    manifest = EvaluationRunManifest.create(plan, created_at=results[0].completed_at)
    provider = plan.identity.provider_identities[ModelRole.GENERATION.value]
    model = plan.identity.model_identities[ModelRole.GENERATION.value]
    attempts = tuple(
        ModelAttempt(
            attempt_id=f"provider-{ordinal}",
            operation_id=f"generation-{ordinal}",
            request_id=result.execution.request_id if result.execution is not None else None,
            run_id=plan.run_id,
            role=ModelRole.GENERATION,
            provider=provider,
            model=model,
            status=ModelAttemptStatus.SUCCEEDED,
            latency_ms=5.0,
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            created_at=result.completed_at,
        )
        for ordinal, result in enumerate(results, start=1)
    )
    scorecard = score_evaluation_v2(dataset, results)
    pricing = openai_standard_pricing_catalog(provider=provider, models=(model,))

    evidence = build_standard_report_v2(
        dataset=dataset,
        manifest=manifest,
        results=results,
        scorecard=scorecard,
        attempts=attempts,
        pricing_catalog=pricing,
    )

    assert tuple(item.metric_id for item in evidence.report.gates[0].observations) == tuple(
        metric.value for metric in AdvancedMetricName
    )
    planned_versions = evaluation_scorer_versions(dataset)
    planned_metric_keys = {
        AdvancedMetricName.FAITHFULNESS.value: "faithfulness",
        AdvancedMetricName.CONTEXT_PRECISION.value: "context-precision",
        AdvancedMetricName.ANSWER_COMPLIANCE.value: "answer-compliance",
        AdvancedMetricName.STYLE.value: "style-consistency",
        AdvancedMetricName.REFUSAL_APPROPRIATENESS.value: "refusal-appropriateness",
    }
    assert all(
        item.scorer_version == planned_versions[planned_metric_keys[item.metric_id]]
        for item in evidence.report.gates[0].observations
    )
    assert len(evidence.report.operations_summary.observations) == 17
    assert evidence.report.category_results

    run_root = tmp_path / "runs" / plan.run_id
    run_root.mkdir(parents=True)
    (run_root / "attempt-ledger-v2.jsonl").write_bytes(evidence.attempt_ledger)
    catalog = ArtifactCatalogV2(tmp_path / "published")
    published = publish_standard_report_v2(
        evidence,
        run_root=run_root,
        artifact_catalog=catalog,
        redactor=DEFAULT_REDACTOR,
    )

    assert tuple(item.artifact_id for item in published.manifest.artifacts) == (
        "evaluation-report-json",
        "evaluation-report-html",
        "operations-summary-txt",
        "operations-summary-csv",
    )
    resolved = catalog.resolve(plan.run_id, "evaluation-report-json")
    raw = decode_json_report(resolved.content.decode("utf-8"))
    assert isinstance(raw, dict)
    parsed = parse_report_v2(raw)
    assert parsed.run_id == plan.run_id
    assert parsed.performance_evidence.measured.logical_attempt_count == len(cases)
