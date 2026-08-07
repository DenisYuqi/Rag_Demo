from __future__ import annotations

from pathlib import Path

import pytest

from rag_mvp.domain.ingestion import ChunkLocator
from rag_mvp.domain.qa import (
    AnswerClaim,
    Citation,
    QAErrorCode,
    RefusalReason,
    StreamEventKind,
    ValidatedStreamEvent,
)
from rag_mvp.evaluation.dataset import (
    Answerability,
    CorpusSnapshot,
    DatasetManifest,
    EvaluationCase,
    EvaluationCategory,
    EvaluationDataset,
    EvaluationLanguage,
    ExpectedFact,
    StyleExpectation,
)
from rag_mvp.evaluation.grounding_metrics import MetricName
from rag_mvp.evaluation.runner import EvaluationCaseExecution, PersistedCaseResult
from rag_mvp.evaluation.scoring import (
    MAX_CONCISE_CHARACTERS,
    MAX_REFUSAL_CONCISE_CHARACTERS,
    SCORING_PIPELINE_VERSION,
    EvaluationScorer,
    EvaluationScoringError,
    score_evaluation,
)


def _dataset(*cases: EvaluationCase) -> EvaluationDataset:
    manifest = DatasetManifest.model_construct(dataset_id="mvp-v1", version="1.0.0")
    return EvaluationDataset.model_construct(
        root=Path("evaluations/datasets/mvp-v1"),
        manifest=manifest,
        cases=tuple(cases),
        corpus=CorpusSnapshot.model_construct(),
        category_counts={},
        metric_eligibility_counts={},
    )


def _answer_case(
    case_id: str = "case-answer",
    *,
    evidence_id: str = "chunk-authoritative",
    language: EvaluationLanguage = EvaluationLanguage.ENGLISH,
    styles: tuple[StyleExpectation, ...] = (
        StyleExpectation.ANSWER_IN_REQUEST_LANGUAGE,
        StyleExpectation.CITATIONS_REQUIRED,
        StyleExpectation.CONCISE,
    ),
) -> EvaluationCase:
    category = (
        EvaluationCategory.ANSWERABLE_CHINESE
        if language is EvaluationLanguage.CHINESE
        else EvaluationCategory.PII
        if StyleExpectation.PII_REDACTED in styles
        else EvaluationCategory.ANSWERABLE_ENGLISH
    )
    return EvaluationCase(
        case_id=case_id,
        question="What does the policy say?",
        language=language,
        answerability=Answerability.ANSWERABLE,
        category=category,
        expected_facts=(
            ExpectedFact(
                fact_id=f"fact-{case_id}",
                text="The policy states the expected fact.",
                evidence_ids=(evidence_id,),
            ),
        ),
        authoritative_evidence_ids=(evidence_id,),
        style_expectations=styles,
    )


def _refusal_case(
    case_id: str = "case-refusal",
    *,
    styles: tuple[StyleExpectation, ...] = (
        StyleExpectation.ANSWER_IN_REQUEST_LANGUAGE,
        StyleExpectation.REFUSAL_CONCISE,
    ),
) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        question="What is not in the corpus?",
        language=EvaluationLanguage.ENGLISH,
        answerability=Answerability.UNANSWERABLE,
        category=EvaluationCategory.UNANSWERABLE,
        style_expectations=styles,
    )


def _answer_event(
    *,
    case_id: str,
    content: str = "The policy states the expected fact.",
    citation_ids: tuple[str, ...] = ("chunk-authoritative",),
    response_language: str = "en",
    source_title: str = "Policy",
) -> ValidatedStreamEvent:
    return ValidatedStreamEvent(
        request_id=f"request-{case_id}",
        session_id=f"session-{case_id}",
        sequence=0,
        kind=StreamEventKind.ANSWER,
        response_language=response_language,
        content=content,
        claims=(AnswerClaim(text=content, citation_chunk_ids=citation_ids),),
        citations=tuple(
            Citation(
                source_title=source_title,
                document_version=1,
                chunk_id=chunk_id,
                locator=ChunkLocator(pages=(1,)),
            )
            for chunk_id in dict.fromkeys(citation_ids)
        ),
        terminal=True,
    )


def _refusal_event(
    *,
    case_id: str,
    content: str = "The available evidence does not support an answer.",
) -> ValidatedStreamEvent:
    return ValidatedStreamEvent(
        request_id=f"request-{case_id}",
        session_id=f"session-{case_id}",
        sequence=0,
        kind=StreamEventKind.REFUSAL,
        response_language="en",
        content=content,
        reason=RefusalReason.INSUFFICIENT_EVIDENCE,
        terminal=True,
    )


def _success(
    case_id: str,
    event: ValidatedStreamEvent,
    *,
    retrieved: tuple[str, ...] = (),
    context: tuple[str, ...] = (),
    run_id: str = "run-1",
) -> PersistedCaseResult:
    execution = EvaluationCaseExecution(
        case_id=case_id,
        owner_id=f"owner-{case_id}",
        session_id=event.session_id,
        request_id=event.request_id,
        event=event,
        retrieved_chunk_ids=retrieved,
        context_chunk_ids=context,
        latency_ms=12.0,
    )
    return PersistedCaseResult(
        run_id=run_id,
        case_id=case_id,
        succeeded=True,
        execution=execution,
    )


def test_scoring_produces_five_case_metrics_aggregates_and_passing_gate() -> None:
    answer = _answer_case()
    refusal = _refusal_case()
    results = (
        _success(
            answer.case_id,
            _answer_event(case_id=answer.case_id),
            retrieved=("chunk-authoritative", "chunk-noise"),
            context=("chunk-authoritative",),
        ),
        _success(refusal.case_id, _refusal_event(case_id=refusal.case_id)),
    )

    scorecard = score_evaluation(_dataset(answer, refusal), results)

    assert scorecard.scoring_version == SCORING_PIPELINE_VERSION
    assert scorecard.case_ids == (answer.case_id, refusal.case_id)
    assert len(scorecard.per_case) == 10
    assert tuple(result.metric for result in scorecard.metrics_for_case(answer.case_id)) == tuple(
        MetricName
    )
    answer_metrics = scorecard.per_case_by_id[answer.case_id]
    assert all(result.score == 1 for result in answer_metrics.values())
    refusal_metrics = scorecard.per_case_by_id[refusal.case_id]
    assert all(
        not refusal_metrics[metric].eligible
        for metric in (
            MetricName.FAITHFULNESS,
            MetricName.CONTEXT_PRECISION,
            MetricName.ANSWER_COMPLETENESS,
        )
    )
    assert refusal_metrics[MetricName.STYLE_CONSISTENCY].score == 1
    assert refusal_metrics[MetricName.REFUSAL_APPROPRIATENESS].score == 1
    assert scorecard.aggregates_by_metric[MetricName.FAITHFULNESS].eligible_cases == 1
    assert scorecard.aggregates_by_metric[MetricName.STYLE_CONSISTENCY].eligible_cases == 2
    assert scorecard.quality_gate.valid
    assert scorecard.quality_gate.passed


def test_faithfulness_requires_runner_context_proof_even_when_citation_exists() -> None:
    case = _answer_case()
    result = _success(
        case.case_id,
        _answer_event(case_id=case.case_id),
        retrieved=("chunk-authoritative",),
        context=(),
    )

    metrics = score_evaluation(_dataset(case), (result,)).per_case_by_id[case.case_id]

    assert metrics[MetricName.FAITHFULNESS].score == 0
    assert metrics[MetricName.FAITHFULNESS].evidence[0].rationale == (
        "claim_citation_lacks_runner_context_proof"
    )
    assert metrics[MetricName.CONTEXT_PRECISION].score == 1
    assert metrics[MetricName.ANSWER_COMPLETENESS].score == 0


def test_completeness_uses_expected_fact_evidence_mapping_not_any_grounded_claim() -> None:
    case = _answer_case()
    event = _answer_event(
        case_id=case.case_id,
        content="A different grounded fact.",
        citation_ids=("chunk-other",),
    )
    result = _success(
        case.case_id,
        event,
        retrieved=("chunk-authoritative", "chunk-other"),
        context=("chunk-other",),
    )

    metrics = EvaluationScorer().score(_dataset(case), (result,)).per_case_by_id[case.case_id]

    assert metrics[MetricName.FAITHFULNESS].score == 1
    assert metrics[MetricName.ANSWER_COMPLETENESS].score == 0
    assert metrics[MetricName.ANSWER_COMPLETENESS].evidence[0].rationale == (
        "expected_fact_not_covered"
    )


def test_all_versioned_style_checks_use_deterministic_boundaries() -> None:
    cases = (
        _answer_case(
            "case-language",
            styles=(StyleExpectation.ANSWER_IN_REQUEST_LANGUAGE,),
        ),
        _refusal_case(
            "case-citations",
            styles=(StyleExpectation.CITATIONS_REQUIRED,),
        ),
        _answer_case("case-concise-boundary", styles=(StyleExpectation.CONCISE,)),
        _answer_case("case-concise-over", styles=(StyleExpectation.CONCISE,)),
        _refusal_case(
            "case-refusal-boundary",
            styles=(StyleExpectation.REFUSAL_CONCISE,),
        ),
        _refusal_case(
            "case-refusal-over",
            styles=(StyleExpectation.REFUSAL_CONCISE,),
        ),
        _answer_case("case-pii-safe", styles=(StyleExpectation.PII_REDACTED,)),
        _answer_case("case-pii-raw", styles=(StyleExpectation.PII_REDACTED,)),
        _answer_case("case-pii-title", styles=(StyleExpectation.PII_REDACTED,)),
    )
    results = (
        _success(
            "case-language",
            _answer_event(
                case_id="case-language",
                content="\u8fd9\u662f\u4e2d\u6587\u56de\u7b54\u3002",
                response_language="en",
            ),
            retrieved=("chunk-authoritative",),
            context=("chunk-authoritative",),
        ),
        _success("case-citations", _refusal_event(case_id="case-citations")),
        _success(
            "case-concise-boundary",
            _answer_event(
                case_id="case-concise-boundary",
                content="x" * MAX_CONCISE_CHARACTERS,
            ),
            retrieved=("chunk-authoritative",),
            context=("chunk-authoritative",),
        ),
        _success(
            "case-concise-over",
            _answer_event(
                case_id="case-concise-over",
                content="x" * (MAX_CONCISE_CHARACTERS + 1),
            ),
            retrieved=("chunk-authoritative",),
            context=("chunk-authoritative",),
        ),
        _success(
            "case-refusal-boundary",
            _refusal_event(
                case_id="case-refusal-boundary",
                content="x" * MAX_REFUSAL_CONCISE_CHARACTERS,
            ),
        ),
        _success(
            "case-refusal-over",
            _refusal_event(
                case_id="case-refusal-over",
                content="x" * (MAX_REFUSAL_CONCISE_CHARACTERS + 1),
            ),
        ),
        _success(
            "case-pii-safe",
            _answer_event(
                case_id="case-pii-safe",
                content="Contact [REDACTED:email].",
            ),
            retrieved=("chunk-authoritative",),
            context=("chunk-authoritative",),
        ),
        _success(
            "case-pii-raw",
            _answer_event(
                case_id="case-pii-raw",
                content="Contact alex@example.com.",
            ),
            retrieved=("chunk-authoritative",),
            context=("chunk-authoritative",),
        ),
        _success(
            "case-pii-title",
            _answer_event(
                case_id="case-pii-title",
                content="The contact is redacted.",
                source_title="alex@example.com",
            ),
            retrieved=("chunk-authoritative",),
            context=("chunk-authoritative",),
        ),
    )

    scorecard = score_evaluation(_dataset(*cases), results)
    style_scores = {
        case_id: metrics[MetricName.STYLE_CONSISTENCY]
        for case_id, metrics in scorecard.per_case_by_id.items()
    }

    assert style_scores["case-language"].score == 0
    assert style_scores["case-citations"].score == 0
    assert style_scores["case-concise-boundary"].score == 1
    assert style_scores["case-concise-over"].score == 0
    assert style_scores["case-refusal-boundary"].score == 1
    assert style_scores["case-refusal-over"].score == 0
    assert style_scores["case-pii-safe"].score == 1
    assert style_scores["case-pii-raw"].score == 0
    assert style_scores["case-pii-title"].score == 0
    assert style_scores["case-pii-raw"].evidence[0].rationale == ("raw_supported_pii_detected")


def test_failed_and_error_terminal_cases_keep_all_metrics_ineligible() -> None:
    failed_case = _answer_case("case-failed")
    error_case = _answer_case("case-error")
    failed = PersistedCaseResult(
        run_id="run-1",
        case_id=failed_case.case_id,
        succeeded=False,
        safe_error_code="case_execution_failed",
    )
    error_event = ValidatedStreamEvent(
        request_id="request-case-error",
        session_id="session-case-error",
        sequence=0,
        kind=StreamEventKind.ERROR,
        response_language="en",
        content="A dependency is unavailable.",
        error_code=QAErrorCode.DEPENDENCY_FAILURE,
        retryable=True,
        terminal=True,
    )
    error_execution = EvaluationCaseExecution(
        case_id=error_case.case_id,
        owner_id="owner-case-error",
        session_id=error_event.session_id,
        request_id=error_event.request_id,
        event=error_event,
        latency_ms=10,
    )
    error = PersistedCaseResult(
        run_id="run-1",
        case_id=error_case.case_id,
        succeeded=False,
        execution=error_execution,
        safe_error_code="qa_terminal_error",
    )

    scorecard = score_evaluation(_dataset(failed_case, error_case), (failed, error))

    assert scorecard.failed_case_ids == (failed_case.case_id, error_case.case_id)
    assert len(scorecard.per_case) == 10
    assert all(not result.eligible and result.score is None for result in scorecard.per_case)
    assert all(result.rationale == "case_execution_error" for result in scorecard.per_case)
    assert all(aggregate.eligible_cases == 0 for aggregate in scorecard.aggregates)
    assert not scorecard.quality_gate.valid
    assert not scorecard.quality_gate.passed


def test_one_failed_case_invalidates_otherwise_passing_quality_gate() -> None:
    answer_case = _answer_case("case-answer")
    failed_case = _refusal_case("case-failed")
    successful = _success(
        answer_case.case_id,
        _answer_event(case_id=answer_case.case_id),
        retrieved=("chunk-authoritative",),
        context=("chunk-authoritative",),
    )
    failed = PersistedCaseResult(
        run_id="run-1",
        case_id=failed_case.case_id,
        succeeded=False,
        safe_error_code="case_execution_failed",
    )

    scorecard = score_evaluation(_dataset(answer_case, failed_case), (successful, failed))

    assert scorecard.failed_case_ids == (failed_case.case_id,)
    assert all(decision.valid and decision.passed for decision in scorecard.quality_gate.decisions)
    assert not scorecard.quality_gate.case_executions_complete
    assert not scorecard.quality_gate.valid
    assert not scorecard.quality_gate.passed


def test_scoring_is_reproducible_for_identical_immutable_inputs() -> None:
    case = _answer_case()
    persisted = _success(
        case.case_id,
        _answer_event(case_id=case.case_id),
        retrieved=("chunk-authoritative",),
        context=("chunk-authoritative",),
    )
    scorer = EvaluationScorer()

    first = scorer.score(_dataset(case), (persisted,))
    second = scorer.score(_dataset(case), (persisted,))

    assert first == second


def test_scoring_rejects_missing_results_and_unproven_context_identity() -> None:
    case = _answer_case()
    with pytest.raises(EvaluationScoringError, match="persisted_case_results_invalid"):
        score_evaluation(_dataset(case), ())

    invalid = _success(
        case.case_id,
        _answer_event(case_id=case.case_id),
        retrieved=("chunk-other",),
    )
    execution = invalid.execution
    assert execution is not None
    invalid_execution = EvaluationCaseExecution.model_construct(
        **{
            name: getattr(execution, name)
            for name in type(execution).model_fields
            if name != "context_chunk_ids"
        },
        context_chunk_ids=("chunk-authoritative",),
    )
    invalid = invalid.model_copy(update={"execution": invalid_execution})
    with pytest.raises(EvaluationScoringError, match="case_context_not_in_retrieval"):
        score_evaluation(_dataset(case), (invalid,))


def test_scoring_rejects_mixed_run_or_case_result_sets() -> None:
    first_case = _answer_case("case-1")
    second_case = _answer_case("case-2")
    first = _success(
        first_case.case_id,
        _answer_event(case_id=first_case.case_id),
        retrieved=("chunk-authoritative",),
        context=("chunk-authoritative",),
        run_id="run-1",
    )
    second = _success(
        second_case.case_id,
        _answer_event(case_id=second_case.case_id),
        retrieved=("chunk-authoritative",),
        context=("chunk-authoritative",),
        run_id="run-2",
    )

    with pytest.raises(EvaluationScoringError, match="persisted_run_identity_mismatch"):
        score_evaluation(_dataset(first_case, second_case), (first, second))

    with pytest.raises(EvaluationScoringError, match="persisted_case_set_mismatch"):
        score_evaluation(_dataset(first_case, second_case), (first,))
