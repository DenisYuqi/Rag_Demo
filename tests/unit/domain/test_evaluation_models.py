from __future__ import annotations

from copy import deepcopy
from decimal import Decimal

import pytest
from pydantic import ValidationError

from rag_mvp.domain.evaluation import (
    AcceptanceContract,
    AcceptanceMetricRequirement,
    ArtifactDescriptor,
    EvaluationRun,
    EvaluationRunStatus,
    EvidenceComparisonOperator,
    IssueClassification,
    IssueEvidence,
    MetricObservation,
    MetricObservationStatus,
    ModelAttempt,
    ModelAttemptStatus,
    ModelPricing,
    ModelRole,
    OperationsSummary,
    TokenUsage,
    UnavailableValue,
    adapt_v1_metric_aggregates,
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
        run_references=("baseline-run", "post-fix-run"),
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


def test_issue_evidence_rejects_missing_observability_references() -> None:
    with pytest.raises(ValidationError):
        IssueEvidence(
            issue_id="issue-1",
            classification=IssueClassification.CONTROLLED,
            affected_case_ids=("case-1",),
            symptom="Controlled failure",
            metric_references=("metric-1",),
            log_references=(),
            run_references=("baseline-run", "post-fix-run"),
            trace_references=("trace-1",),
            root_cause="A test-only configuration defect.",
            exact_fix="Restore the accepted configuration.",
            fix_rationale="The paired run isolates one variable.",
            primary_metric="faithfulness",
            baseline_value=0.7,
            post_fix_value=0.9,
            relative_improvement_percent=28.57,
        )


def test_schema_v2_metric_observation_never_turns_missing_evidence_into_zero() -> None:
    missing = UnavailableValue(reason="no-eligible-cases")
    observation = MetricObservation(
        metric_id="cache-hit-rate",
        unit="ratio",
        value=missing,
        numerator=missing,
        denominator=missing,
        eligible=False,
        scorer_version="cache-rate-v2",
        status=MetricObservationStatus.UNAVAILABLE,
    )

    assert observation.model_dump(mode="json")["value"] == {
        "status": "unavailable",
        "reason": "no-eligible-cases",
    }
    with pytest.raises(ValidationError, match="denominator must be non-zero"):
        MetricObservation(
            metric_id="answer-compliance",
            unit="ratio",
            value=0.0,
            numerator=0.0,
            denominator=0,
            eligible=True,
            scorer_version="answer-compliance-v2",
            status=MetricObservationStatus.OBSERVED,
        )
    with pytest.raises(ValidationError, match="ratio metric value disagrees"):
        MetricObservation(
            metric_id="answer-compliance",
            unit="ratio",
            value=1.0,
            numerator=11.0,
            denominator=10,
            eligible=True,
            scorer_version="answer-compliance-v2",
            status=MetricObservationStatus.OBSERVED,
        )


def test_schema_v2_contract_operations_and_artifact_models_are_immutable() -> None:
    observation = MetricObservation(
        metric_id="answer-compliance",
        unit="ratio",
        value=0.9,
        numerator=9.0,
        denominator=10,
        eligible=True,
        threshold=0.9,
        operator=EvidenceComparisonOperator.GREATER_THAN_OR_EQUAL,
        scorer_version="answer-compliance-v2",
        status=MetricObservationStatus.PASSED,
        evidence_references=("case-1",),
    )
    contract = AcceptanceContract(
        contract_id="original-pdf-advanced",
        version="2.0.0",
        gate_profile_version="rag-advanced-quality-thresholds-v2",
        dataset_schema_version="rag-evaluation-dataset-v2",
        performance_schema_version="rag-performance-evidence-v2",
        cost_schema_version="rag-cost-evidence-v2",
        metric_requirements=(
            AcceptanceMetricRequirement(
                metric_id="answer-compliance",
                threshold=0.9,
                operator=EvidenceComparisonOperator.GREATER_THAN_OR_EQUAL,
            ),
        ),
    )
    operations = OperationsSummary(
        run_id="run-v2",
        configuration_id="config-v2",
        observations=(observation,),
        source_artifact_ids=("report-json",),
    )
    artifact = ArtifactDescriptor(
        schema_version="2.0.0",
        artifact_id="report-json",
        format="json",
        media_type="application/json",
        relative_path="artifacts/report.json",
        sha256_digest=f"sha256:{'a' * 64}",
        byte_size=128,
    )

    assert contract.metric_requirements[0].minimum_denominator == 1
    assert operations.observations == (observation,)
    assert ArtifactDescriptor.model_validate_json(artifact.model_dump_json()) == artifact
    with pytest.raises(ValidationError, match="safe relative POSIX path"):
        ArtifactDescriptor.model_validate(
            {**artifact.model_dump(), "relative_path": "../report.json"}
        )


def test_v1_adapter_is_read_only_and_marks_v2_only_fields_unavailable() -> None:
    legacy = {
        "faithfulness": {
            "value": 0.95,
            "eligible_cases": 8,
            "operator": ">",
            "threshold": 0.85,
            "passed": True,
            "rationale": "threshold_passed",
        }
    }
    original = deepcopy(legacy)

    observations = adapt_v1_metric_aggregates(
        legacy,
        scorer_versions={"faithfulness": "faithfulness-v1"},
    )

    assert legacy == original
    assert len(observations) == 1
    assert observations[0].value == 0.95
    assert observations[0].denominator == 8
    assert isinstance(observations[0].numerator, UnavailableValue)
    assert observations[0].status is MetricObservationStatus.UNAVAILABLE
