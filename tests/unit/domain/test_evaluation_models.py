from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from rag_mvp.domain.evaluation import (
    EvaluationRun,
    EvaluationRunStatus,
    IssueClassification,
    IssueEvidence,
    ModelAttempt,
    ModelAttemptStatus,
    ModelPricing,
    ModelRole,
    TokenUsage,
)


def test_missing_usage_and_pricing_remain_unknown() -> None:
    usage = TokenUsage()
    pricing = ModelPricing(
        pricing_version="2026-08",
        provider="provider-a",
        model="model-a",
        currency="USD",
    )

    assert usage.known_total is None
    assert pricing.input_per_million is None
    assert pricing.output_per_million is None
    assert TokenUsage.model_validate_json(usage.model_dump_json()).known_total is None


def test_reported_token_total_cannot_hide_known_tokens() -> None:
    with pytest.raises(ValidationError):
        TokenUsage(input_tokens=8, output_tokens=3, total_tokens_reported=10)


def test_model_attempt_requires_currency_for_known_cost() -> None:
    values = {
        "attempt_id": "attempt-1",
        "operation_id": "operation-1",
        "request_id": "request-1",
        "role": ModelRole.GENERATION,
        "provider": "primary",
        "model": "chat-v1",
        "status": ModelAttemptStatus.SUCCEEDED,
        "latency_ms": 125.0,
        "estimated_cost": Decimal("0.002"),
    }
    with pytest.raises(ValidationError):
        ModelAttempt.model_validate(values)

    attempt = ModelAttempt.model_validate({**values, "currency": "USD"})
    assert ModelAttempt.model_validate_json(attempt.model_dump_json()) == attempt


def test_evaluation_progress_and_issue_evidence_are_validated() -> None:
    with pytest.raises(ValidationError):
        EvaluationRun(
            run_id="run-1",
            status=EvaluationRunStatus.RUNNING,
            dataset_id="mvp",
            dataset_version="1.0.0",
            dataset_hash="hash-123",
            corpus_version="corpus-v1",
            configuration_id="config-v1",
            code_revision="code-v1",
            scorer_versions={"faithfulness": "v1"},
            cache_policy="acceptance-bypass",
            total_cases=2,
            completed_cases=2,
            failed_cases=1,
        )

    issue = IssueEvidence(
        issue_id="issue-1",
        classification=IssueClassification.GENUINE,
        affected_case_ids=("case-1",),
        symptom="A relevant chunk was ranked below the answer context.",
        metric_references=("metric-1",),
        log_references=("log-1",),
        trace_references=("trace-1",),
        root_cause="The lexical weight was too low.",
        exact_fix="Increase the versioned lexical RRF weight.",
        fix_rationale="The affected cases contain exact policy identifiers.",
        primary_metric="context-precision",
        baseline_value=0.5,
        post_fix_value=0.6,
        relative_improvement_percent=20.0,
    )
    assert IssueEvidence.model_validate_json(issue.model_dump_json()) == issue
