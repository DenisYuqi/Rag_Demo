from __future__ import annotations

import hashlib
from decimal import Decimal
from pathlib import Path

import pytest
from test_experiment import _plan
from test_scoring_v2 import _result

from rag_mvp.config.settings import Settings
from rag_mvp.domain import ArtifactDescriptor
from rag_mvp.domain.evaluation import (
    ModelAttempt,
    ModelAttemptStatus,
    ModelRole,
    TokenUsage,
)
from rag_mvp.domain.qa import QAErrorCode, RefusalReason, StreamEventKind
from rag_mvp.domain.retrieval import CacheOutcome, CachePolicy, RetrievalMode
from rag_mvp.evaluation.comparison import (
    COMPARISON_SELECTION_ELIGIBILITY_GATE_ID,
    ComparisonCandidateReference,
    build_comparison_selection_eligibility_gate,
    canonical_candidate_evidence,
    load_verified_candidate_report,
    project_evaluation_identity,
)
from rag_mvp.evaluation.comparison_evidence import (
    ComparisonEvidenceBuildError,
    _cache_outcome,
    build_persisted_candidate_evidence,
    validate_candidate_plan_binding,
)
from rag_mvp.evaluation.comparison_schedule import (
    build_comparison_schedule,
    cache_eligible_case_ids,
    materialize_variant_cases,
)
from rag_mvp.evaluation.dataset import EvaluationCaseV2
from rag_mvp.evaluation.experiment import ExperimentAxis, ExperimentOrderPolicy
from rag_mvp.evaluation.plan import EvaluationDatasetRegistry, build_evaluation_plan
from rag_mvp.evaluation.quality_gate import ADVANCED_QUALITY_GATE_ID
from rag_mvp.evaluation.report_builder import case_ids_content_hash

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DATASETS_ROOT = _REPOSITORY_ROOT / "evaluations" / "datasets"


def _runtime_result(case: EvaluationCaseV2, run_id: str, index: int):
    result = _result(case)
    execution = result.execution
    assert execution is not None
    diagnostics = execution.event.diagnostics.model_copy(
        update={
            "cache_status": {"retrieval": "bypass"},
            "metadata": {"index_revision": "evaluation-revision-1"},
        }
    )
    event = execution.event.model_copy(update={"diagnostics": diagnostics})
    retrieval_digest = f"sha256:{hashlib.sha256(case.case_id.encode()).hexdigest()}"
    return result.model_copy(
        update={
            "run_id": run_id,
            "execution": execution.model_copy(
                update={
                    "event": event,
                    "retrieval_evidence_digest": retrieval_digest,
                }
            ),
            "logical_latency_ms": float(index + 1),
        }
    )


def _run_fixture():
    dataset = EvaluationDatasetRegistry(_DATASETS_ROOT).resolve(
        "original-pdf-acceptance",
        "2.0.0",
    )
    source_plan = build_evaluation_plan(
        dataset,
        Settings(_env_file=None),
        "candidate-run-1",
    )
    pricing_version = _plan().pricing.pricing_version
    evaluation_plan = source_plan.model_copy(
        update={
            "identity": source_plan.identity.model_copy(
                update={
                    "provider_identities": {
                        **source_plan.identity.provider_identities,
                        "embedding": "provider-a",
                        "generation": "provider-a",
                    },
                    "model_identities": {
                        **source_plan.identity.model_identities,
                        "embedding": "embedding-v1",
                        "generation": "generation-v1",
                    },
                    "pricing_version": pricing_version,
                }
            )
        }
    )
    cases = tuple(case for case in dataset.cases if isinstance(case, EvaluationCaseV2))
    results = tuple(
        _runtime_result(case, evaluation_plan.run_id, index) for index, case in enumerate(cases)
    )
    return dataset, evaluation_plan, results


def _experiment_reference(
    dataset,
    evaluation_plan,
    *,
    repeats: int = 1,
    order_policy: ExperimentOrderPolicy = ExperimentOrderPolicy.DECLARED,
    cache_behavior: str | None = None,
):
    source = _plan()
    case_ids = tuple(case.case_id for case in dataset.cases)
    case_set_hash = case_ids_content_hash(case_ids)
    reference_variant_index = 1 if cache_behavior == "warm" else 0
    reference_variant_id = source.variants[reference_variant_index].variant_id
    projection = project_evaluation_identity(
        reference_variant_id,
        evaluation_plan.identity,
        corpus_id=dataset.corpus.manifest.snapshot_id,
        case_set_hash=case_set_hash,
        cache_behavior=cache_behavior,
    )
    if cache_behavior is None:
        axis = source.axis
        variants = (
            source.variants[0].model_copy(
                update={
                    "axis_value": evaluation_plan.identity.model_identities["generation"],
                    "configuration_id": evaluation_plan.identity.configuration_id,
                }
            ),
            source.variants[1].model_copy(update={"axis_value": "generation-v2"}),
        )
    else:
        axis = ExperimentAxis.CACHE_BEHAVIOR
        variants = (
            source.variants[0].model_copy(
                update={
                    "axis_value": "cold",
                    "configuration_id": (
                        evaluation_plan.identity.configuration_id
                        if cache_behavior == "cold"
                        else "semantic-cold-configuration"
                    ),
                }
            ),
            source.variants[1].model_copy(
                update={
                    "axis_value": "warm",
                    "configuration_id": (
                        evaluation_plan.identity.configuration_id
                        if cache_behavior == "warm"
                        else "semantic-warm-configuration"
                    ),
                }
            ),
        )
    experiment_plan = type(source).create(
        **{
            **source.model_dump(exclude={"content_hash"}),
            "axis": axis,
            "fixed_identities": source.fixed_identities.model_copy(
                update={
                    "dataset_id": dataset.manifest.dataset_id,
                    "dataset_version": dataset.manifest.version,
                    "dataset_hash": dataset.manifest.content_hash,
                    "corpus_id": dataset.corpus.manifest.snapshot_id,
                    "corpus_version": dataset.corpus.manifest.version,
                    "corpus_hash": dataset.corpus.manifest.content_hash,
                    "case_set_hash": case_set_hash,
                    "case_count": len(dataset.cases),
                    "controlled": tuple(
                        item for item in projection.identities if item.name != axis.identity_name
                    ),
                }
            ),
            "variants": variants,
            "cache_policy": evaluation_plan.identity.cache_policy,
            "repeat_order_policy": source.repeat_order_policy.model_copy(
                update={
                    "repeats_per_case": repeats,
                    "order_policy": order_policy,
                }
            ),
        }
    )
    reference_variant = experiment_plan.variants[reference_variant_index]
    reference = ComparisonCandidateReference(
        variant_id=reference_variant.variant_id,
        axis_value=reference_variant.axis_value,
        configuration_id=evaluation_plan.identity.configuration_id,
        evaluation_run_id=evaluation_plan.run_id,
    )
    return experiment_plan, reference


def _empty_request_ledger(results):
    return {result.execution.request_id: () for result in results if result.execution is not None}


def _provider_ledger(results, run_id):
    ledger = _empty_request_ledger(results)
    answer_index = 0
    for result in results:
        execution = result.execution
        if execution is None or execution.event.kind is not StreamEventKind.ANSWER:
            continue
        answer_index += 1
        ledger[execution.request_id] = (
            ModelAttempt(
                attempt_id=f"embedding-attempt-{answer_index}",
                operation_id="qa-retrieval",
                request_id=execution.request_id,
                run_id=run_id,
                role=ModelRole.EMBEDDING,
                provider="provider-a",
                model="embedding-v1",
                status=ModelAttemptStatus.SUCCEEDED,
                latency_ms=1.0,
                usage=TokenUsage(input_tokens=3),
            ),
            ModelAttempt(
                attempt_id=f"generation-attempt-{answer_index}",
                operation_id="qa-generation",
                request_id=execution.request_id,
                run_id=run_id,
                role=ModelRole.GENERATION,
                provider="provider-a",
                model="generation-v1",
                status=ModelAttemptStatus.SUCCEEDED,
                latency_ms=2.0,
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            ),
        )
    return ledger


def test_candidate_evidence_is_bound_to_real_non_sla_attempt_and_provider_ledgers() -> None:
    dataset, evaluation_plan, results = _run_fixture()
    first_execution = results[0].execution
    assert first_execution is not None
    experiment_plan, reference = _experiment_reference(dataset, evaluation_plan)
    provider_attempt = ModelAttempt(
        attempt_id="provider-attempt-1",
        operation_id="qa-generation",
        request_id=first_execution.request_id,
        run_id=evaluation_plan.run_id,
        role=ModelRole.GENERATION,
        provider="provider-a",
        model="generation-v1",
        status=ModelAttemptStatus.SUCCEEDED,
        attempt_number=1,
        latency_ms=12.5,
        usage=TokenUsage(input_tokens=100, output_tokens=20),
    )
    embedding_attempt = ModelAttempt(
        attempt_id="provider-attempt-embedding-1",
        operation_id="qa-retrieval",
        request_id=first_execution.request_id,
        run_id=evaluation_plan.run_id,
        role=ModelRole.EMBEDDING,
        provider="provider-a",
        model="embedding-v1",
        status=ModelAttemptStatus.SUCCEEDED,
        attempt_number=1,
        latency_ms=4.5,
        usage=TokenUsage(input_tokens=12),
    )

    evidence = build_persisted_candidate_evidence(
        comparison_id="comparison-1",
        experiment_plan=experiment_plan,
        reference=reference,
        dataset=dataset,
        evaluation_plan=evaluation_plan,
        results=results,
        provider_attempts_by_request={
            **_provider_ledger(results, evaluation_plan.run_id),
            first_execution.request_id: (embedding_attempt, provider_attempt),
        },
    )
    content = canonical_candidate_evidence(evidence)
    descriptor = ArtifactDescriptor(
        schema_version=evidence.schema_version,
        artifact_id=f"comparison-candidate-{reference.variant_id}",
        format="json",
        media_type="application/json",
        relative_path=f"candidates/{reference.variant_id}.json",
        sha256_digest=f"sha256:{hashlib.sha256(content).hexdigest()}",
        byte_size=len(content),
    )

    verified = load_verified_candidate_report(
        reference,
        descriptor,
        content,
        comparison_id="comparison-1",
        plan=experiment_plan,
    )

    assert verified.evidence.attempt_scope == "comparison-logical-case-attempts"
    assert len(verified.evidence.logical_attempts) == 24
    answer_count = sum(
        result.execution is not None and result.execution.event.kind is StreamEventKind.ANSWER
        for result in results
    )
    assert verified.evidence.provider_call_count == answer_count * 2
    embedded = verified.evidence.provider_attempts[1]
    assert embedded.attempt_reference == provider_attempt.attempt_id
    assert embedded.evidence.operation_id == provider_attempt.operation_id
    assert embedded.evidence.role is provider_attempt.role
    assert embedded.evidence.usage == provider_attempt.usage
    assert embedded.total_cost is not None
    assert verified.evidence.logical_attempts[0].output_tokens == 20
    assert any(
        item.terminal_kind == "refusal" and not item.provider_attempt_references
        for item in verified.evidence.logical_attempts
    )
    metric_ids = {item.metric_id for item in verified.evidence.metrics}
    assert "comparison-logical-all-p90-ms" in metric_ids
    assert "all-attempt-p90-ms" not in metric_ids
    gates = {item.gate_id: item for item in verified.evidence.gates}
    assert ADVANCED_QUALITY_GATE_ID in gates
    assert gates[COMPARISON_SELECTION_ELIGIBILITY_GATE_ID].passed is True

    forged_gate = build_comparison_selection_eligibility_gate(
        verified.evidence.logical_attempts,
        verified.evidence.provider_attempts,
        expected_logical_attempt_count=len(verified.evidence.logical_attempts) + 1,
    )
    forged_evidence = verified.evidence.model_copy(
        update={
            "gates": tuple(
                forged_gate if item.gate_id == COMPARISON_SELECTION_ELIGIBILITY_GATE_ID else item
                for item in verified.evidence.gates
            )
        }
    )
    forged_content = canonical_candidate_evidence(forged_evidence)
    forged_descriptor = descriptor.model_copy(
        update={
            "sha256_digest": f"sha256:{hashlib.sha256(forged_content).hexdigest()}",
            "byte_size": len(forged_content),
        }
    )
    with pytest.raises(
        ValueError,
        match="comparison_selection_eligibility_gate_mismatch",
    ):
        load_verified_candidate_report(
            reference,
            forged_descriptor,
            forged_content,
            comparison_id="comparison-1",
            plan=experiment_plan,
        )


def test_timed_out_usage_preserves_priced_retrieval_lower_bound_in_candidate_evidence() -> None:
    dataset, evaluation_plan, original_results = _run_fixture()
    first = original_results[0]
    execution = first.execution
    assert execution is not None
    error_event = type(execution.event).model_validate(
        {
            **execution.event.model_dump(mode="python"),
            "kind": StreamEventKind.ERROR,
            "content": "The evaluation request timed out.",
            "claims": (),
            "citations": (),
            "reason": None,
            "error_code": QAErrorCode.DEADLINE_EXPIRED,
            "retryable": True,
            "terminal": True,
        }
    )
    failed = first.model_copy(
        update={
            "succeeded": False,
            "safe_error_code": "qa_terminal_error",
            "execution": execution.model_copy(update={"event": error_event}),
        }
    )
    results = (failed, *original_results[1:])
    experiment_plan, reference = _experiment_reference(dataset, evaluation_plan)
    ledger = _provider_ledger(results, evaluation_plan.run_id)
    retrieval = ModelAttempt(
        attempt_id="real-shape-retrieval-embedding",
        operation_id="qa-retrieval",
        request_id=execution.request_id,
        run_id=evaluation_plan.run_id,
        role=ModelRole.EMBEDDING,
        provider="provider-a",
        model="embedding-v1",
        status=ModelAttemptStatus.SUCCEEDED,
        latency_ms=1.0,
        usage=TokenUsage(input_tokens=13),
    )
    timeout = ModelAttempt(
        attempt_id="real-shape-fact-timeout",
        operation_id="fact-evidence-assessment",
        request_id=execution.request_id,
        run_id=evaluation_plan.run_id,
        role=ModelRole.EMBEDDING,
        provider="provider-a",
        model="embedding-v1",
        status=ModelAttemptStatus.TIMED_OUT,
        latency_ms=30_000.0,
        safe_error_category="timeout",
        usage=TokenUsage(),
    )
    ledger[execution.request_id] = (retrieval, timeout)

    evidence = build_persisted_candidate_evidence(
        comparison_id="comparison-real-timeout-shape",
        experiment_plan=experiment_plan,
        reference=reference,
        dataset=dataset,
        evaluation_plan=evaluation_plan,
        results=results,
        provider_attempts_by_request=ledger,
    )

    first_logical = evidence.logical_attempts[0]
    retrieval_cost = next(
        item.known_partial_cost
        for item in evidence.provider_attempts
        if item.attempt_reference == retrieval.attempt_id
    )
    assert retrieval_cost == Decimal("0.00000026")
    assert first_logical.known_partial_cost == retrieval_cost
    assert first_logical.estimated_cost is None
    assert first_logical.cost_complete is False
    assert first_logical.cost_unknown_reasons == ("input-usage-unknown",)
    assert evidence.known_partial_cost == sum(
        (item.known_partial_cost for item in evidence.provider_attempts),
        start=Decimal(0),
    )
    assert evidence.total_cost is None
    assert evidence.cost_complete is False
    assert evidence.cost_unknown_reasons == ("input-usage-unknown",)
    selection_gate = next(
        item for item in evidence.gates if item.gate_id == COMPARISON_SELECTION_ELIGIBILITY_GATE_ID
    )
    assert selection_gate.passed is True
    cost_observation = next(
        item
        for item in selection_gate.observations
        if item.metric_id == "comparison-provider-cost-evidence-completeness"
    )
    assert cost_observation.status.value == "unavailable"


def test_candidate_evidence_rejects_claimed_scorer_versions_that_differ_from_v2() -> None:
    dataset, evaluation_plan, results = _run_fixture()
    scorer_versions = dict(evaluation_plan.identity.scorer_versions)
    scorer_versions["scoring-pipeline"] = "deterministic-evaluation-scoring-v1"
    evaluation_plan = evaluation_plan.model_copy(
        update={
            "identity": evaluation_plan.identity.model_copy(
                update={"scorer_versions": scorer_versions}
            )
        }
    )
    experiment_plan, reference = _experiment_reference(dataset, evaluation_plan)

    with pytest.raises(
        ComparisonEvidenceBuildError,
        match="comparison_scorer_version_mismatch",
    ):
        build_persisted_candidate_evidence(
            comparison_id="comparison-scorer-tamper",
            experiment_plan=experiment_plan,
            reference=reference,
            dataset=dataset,
            evaluation_plan=evaluation_plan,
            results=results,
            provider_attempts_by_request=_provider_ledger(
                results,
                evaluation_plan.run_id,
            ),
        )


@pytest.mark.parametrize("mismatch", ["run", "execution"])
def test_candidate_evidence_rejects_cross_run_or_execution_case_results(
    mismatch: str,
) -> None:
    dataset, evaluation_plan, original_results = _run_fixture()
    results = list(original_results)
    if mismatch == "run":
        results[0] = results[0].model_copy(update={"run_id": "foreign-run"})
        expected = "comparison_candidate_case_set_mismatch"
    else:
        assert results[0].execution is not None
        results[0] = results[0].model_copy(
            update={
                "execution": results[0].execution.model_copy(update={"case_id": "foreign-case"})
            }
        )
        expected = "comparison_candidate_execution_identity_mismatch"
    experiment_plan, reference = _experiment_reference(dataset, evaluation_plan)

    with pytest.raises(ComparisonEvidenceBuildError, match=expected):
        build_persisted_candidate_evidence(
            comparison_id="comparison-1",
            experiment_plan=experiment_plan,
            reference=reference,
            dataset=dataset,
            evaluation_plan=evaluation_plan,
            results=results,
            provider_attempts_by_request=_empty_request_ledger(results),
        )


def test_candidate_evidence_rejects_provider_attempt_from_another_run() -> None:
    dataset, evaluation_plan, results = _run_fixture()
    execution = results[0].execution
    assert execution is not None
    experiment_plan, reference = _experiment_reference(dataset, evaluation_plan)
    foreign = ModelAttempt(
        attempt_id="foreign-provider-attempt",
        operation_id="qa-generation",
        request_id=execution.request_id,
        run_id="foreign-run",
        role=ModelRole.GENERATION,
        provider="provider-a",
        model="generation-v1",
        status=ModelAttemptStatus.SUCCEEDED,
        latency_ms=1.0,
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )

    with pytest.raises(
        ComparisonEvidenceBuildError,
        match="comparison_provider_request_mismatch",
    ):
        build_persisted_candidate_evidence(
            comparison_id="comparison-1",
            experiment_plan=experiment_plan,
            reference=reference,
            dataset=dataset,
            evaluation_plan=evaluation_plan,
            results=results,
            provider_attempts_by_request={
                **_empty_request_ledger(results),
                execution.request_id: (foreign,),
            },
        )


@pytest.mark.parametrize("missing_run_id", [False, True])
def test_candidate_evidence_requires_every_request_and_exact_attempt_run_id(
    missing_run_id: bool,
) -> None:
    dataset, evaluation_plan, results = _run_fixture()
    experiment_plan, reference = _experiment_reference(dataset, evaluation_plan)
    ledger = _provider_ledger(results, evaluation_plan.run_id)
    request_id = next(iter(ledger))
    if missing_run_id:
        attempts = ledger[request_id]
        if not attempts:
            answer_result = next(
                item
                for item in results
                if item.execution is not None
                and item.execution.event.kind is StreamEventKind.ANSWER
            )
            request_id = answer_result.execution.request_id
            attempts = ledger[request_id]
        ledger[request_id] = tuple(item.model_copy(update={"run_id": None}) for item in attempts)
        expected = "comparison_provider_request_mismatch"
    else:
        del ledger[request_id]
        expected = "comparison_provider_request_missing"

    with pytest.raises(ComparisonEvidenceBuildError, match=expected):
        build_persisted_candidate_evidence(
            comparison_id="comparison-1",
            experiment_plan=experiment_plan,
            reference=reference,
            dataset=dataset,
            evaluation_plan=evaluation_plan,
            results=results,
            provider_attempts_by_request=ledger,
        )


def test_candidate_evidence_requires_successful_generation_for_every_answer() -> None:
    dataset, evaluation_plan, results = _run_fixture()
    experiment_plan, reference = _experiment_reference(dataset, evaluation_plan)

    with pytest.raises(
        ComparisonEvidenceBuildError,
        match="comparison_answer_generation_evidence_missing",
    ):
        build_persisted_candidate_evidence(
            comparison_id="comparison-1",
            experiment_plan=experiment_plan,
            reference=reference,
            dataset=dataset,
            evaluation_plan=evaluation_plan,
            results=results,
            provider_attempts_by_request=_empty_request_ledger(results),
        )


def test_fact_assessment_embedding_cannot_replace_retrieval_embedding_evidence() -> None:
    dataset, evaluation_plan, results = _run_fixture()
    experiment_plan, reference = _experiment_reference(dataset, evaluation_plan)
    ledger = _provider_ledger(results, evaluation_plan.run_id)
    answer = next(
        item
        for item in results
        if item.execution is not None and item.execution.event.kind is StreamEventKind.ANSWER
    )
    request_id = answer.execution.request_id
    remaining = tuple(item for item in ledger[request_id] if item.role is not ModelRole.EMBEDDING)
    ledger[request_id] = (
        ModelAttempt(
            attempt_id="fact-assessment-embedding",
            operation_id="fact-evidence-assessment",
            request_id=request_id,
            run_id=evaluation_plan.run_id,
            role=ModelRole.EMBEDDING,
            provider="provider-a",
            model="embedding-v1",
            status=ModelAttemptStatus.SUCCEEDED,
            latency_ms=1.0,
            usage=TokenUsage(input_tokens=4),
        ),
        *remaining,
    )

    with pytest.raises(
        ComparisonEvidenceBuildError,
        match="comparison_answer_embedding_evidence_missing",
    ):
        build_persisted_candidate_evidence(
            comparison_id="comparison-1",
            experiment_plan=experiment_plan,
            reference=reference,
            dataset=dataset,
            evaluation_plan=evaluation_plan,
            results=results,
            provider_attempts_by_request=ledger,
        )


def test_hybrid_rerank_answer_requires_bound_reranking_attempt() -> None:
    dataset, evaluation_plan, results = _run_fixture()
    identity = evaluation_plan.identity
    evaluation_plan = evaluation_plan.model_copy(
        update={
            "identity": identity.model_copy(
                update={
                    "provider_identities": {
                        **identity.provider_identities,
                        "reranking": "provider-a",
                    },
                    "model_identities": {
                        **identity.model_identities,
                        "reranking": "reranking-v1",
                    },
                    "retrieval_configuration": {
                        **identity.retrieval_configuration,
                        "mode": "hybrid-rerank",
                        "reranking_enabled": True,
                    },
                }
            ),
            "cases": tuple(
                item.model_copy(update={"retrieval_mode": RetrievalMode.HYBRID_RERANK})
                for item in evaluation_plan.cases
            ),
        }
    )
    experiment_plan, reference = _experiment_reference(dataset, evaluation_plan)

    with pytest.raises(
        ComparisonEvidenceBuildError,
        match="comparison_answer_reranking_evidence_missing",
    ):
        build_persisted_candidate_evidence(
            comparison_id="comparison-rerank",
            experiment_plan=experiment_plan,
            reference=reference,
            dataset=dataset,
            evaluation_plan=evaluation_plan,
            results=results,
            provider_attempts_by_request=_provider_ledger(results, evaluation_plan.run_id),
        )


def test_explicit_warm_cache_hit_may_omit_embedding_and_reranking_calls() -> None:
    dataset, evaluation_plan, original_results = _run_fixture()
    evaluation_plan = evaluation_plan.model_copy(
        update={
            "identity": evaluation_plan.identity.model_copy(
                update={"cache_policy": CachePolicy.USE}
            )
        }
    )
    eligible_case_ids = set(cache_eligible_case_ids(dataset))
    eligible_request_ids: set[str] = set()
    results = []
    for result in original_results:
        execution = result.execution
        if execution is None:
            results.append(result)
            continue
        event = execution.event
        if result.case_id in eligible_case_ids:
            eligible_request_ids.add(execution.request_id)
            diagnostics = event.diagnostics.model_copy(
                update={"cache_status": {"retrieval": "hit"}}
            )
            event = event.model_copy(update={"diagnostics": diagnostics})
        else:
            diagnostics = event.diagnostics.model_copy(
                update={"cache_status": {"retrieval": "not-applicable"}}
            )
            event = event.model_copy(update={"diagnostics": diagnostics})
        results.append(
            result.model_copy(
                update={
                    "execution": execution.model_copy(
                        update={"cache_policy": CachePolicy.USE, "event": event}
                    )
                }
            )
        )
    experiment_plan, reference = _experiment_reference(
        dataset,
        evaluation_plan,
        cache_behavior="warm",
    )
    ledger = _provider_ledger(results, evaluation_plan.run_id)
    for request_id, attempts in ledger.items():
        if request_id in eligible_request_ids:
            ledger[request_id] = tuple(
                item
                for item in attempts
                if item.role not in {ModelRole.EMBEDDING, ModelRole.RERANKING}
            )

    evidence = build_persisted_candidate_evidence(
        comparison_id="comparison-warm",
        experiment_plan=experiment_plan,
        reference=reference,
        dataset=dataset,
        evaluation_plan=evaluation_plan,
        results=results,
        provider_attempts_by_request=ledger,
    )

    eligible_logical_attempt_ids = {
        item.attempt_id for item in evidence.logical_attempts if item.case_id in eligible_case_ids
    }
    assert all(
        item.evidence.role is ModelRole.GENERATION
        for item in evidence.provider_attempts
        if item.logical_attempt_id in eligible_logical_attempt_ids
    )

    with pytest.raises(
        ComparisonEvidenceBuildError,
        match="comparison_cache_hit_provider_call_present",
    ):
        build_persisted_candidate_evidence(
            comparison_id="comparison-warm-tampered",
            experiment_plan=experiment_plan,
            reference=reference,
            dataset=dataset,
            evaluation_plan=evaluation_plan,
            results=results,
            provider_attempts_by_request=_provider_ledger(
                results,
                evaluation_plan.run_id,
            ),
        )


def test_pre_retrieval_refusal_uses_policy_specific_cache_outcome() -> None:
    dataset, base_plan, original_results = _run_fixture()
    target_index = next(
        index
        for index, result in enumerate(original_results)
        if result.execution is not None and result.execution.event.kind is StreamEventKind.REFUSAL
    )

    for policy, expected in (
        (CachePolicy.BYPASS, CacheOutcome.BYPASS),
        (CachePolicy.USE, CacheOutcome.NOT_APPLICABLE),
    ):
        plan = base_plan.model_copy(
            update={"identity": base_plan.identity.model_copy(update={"cache_policy": policy})}
        )
        results = []
        for index, result in enumerate(original_results):
            execution = result.execution
            assert execution is not None
            pre_retrieval = index == target_index or execution.event.reason in {
                RefusalReason.PROMPT_INJECTION,
                RefusalReason.SAFETY,
            }
            diagnostics = execution.event.diagnostics.model_copy(
                update={
                    "cache_status": (
                        {}
                        if pre_retrieval
                        else {
                            "retrieval": (
                                CacheOutcome.BYPASS.value
                                if policy is CachePolicy.BYPASS
                                else CacheOutcome.MISS.value
                            )
                        }
                    ),
                    "metadata": (
                        {} if pre_retrieval else {"index_revision": "evaluation-revision-1"}
                    ),
                }
            )
            event = execution.event.model_copy(
                update={
                    "reason": (
                        RefusalReason.PROMPT_INJECTION
                        if index == target_index
                        else execution.event.reason
                    ),
                    "diagnostics": diagnostics,
                }
            )
            results.append(
                result.model_copy(
                    update={
                        "execution": execution.model_copy(
                            update={
                                "cache_policy": policy,
                                "event": event,
                                "retrieved_chunk_ids": (
                                    () if pre_retrieval else execution.retrieved_chunk_ids
                                ),
                                "context_chunk_ids": (
                                    () if pre_retrieval else execution.context_chunk_ids
                                ),
                                "retrieval_evidence_digest": (
                                    None if pre_retrieval else execution.retrieval_evidence_digest
                                ),
                            }
                        )
                    }
                )
            )
        experiment_plan, reference = _experiment_reference(
            dataset,
            plan,
            cache_behavior="cold" if policy is CachePolicy.USE else None,
        )
        evidence = build_persisted_candidate_evidence(
            comparison_id=f"comparison-{policy.value}",
            experiment_plan=experiment_plan,
            reference=reference,
            dataset=dataset,
            evaluation_plan=plan,
            results=results,
            provider_attempts_by_request=_provider_ledger(results, plan.run_id),
        )

        assert evidence.logical_attempts[target_index].cache_outcome is expected
        if policy is CachePolicy.USE:
            tampered = list(results)
            target = tampered[target_index]
            assert target.execution is not None
            diagnostics = target.execution.event.diagnostics.model_copy(
                update={"cache_status": {"retrieval": CacheOutcome.MISS.value}}
            )
            tampered[target_index] = target.model_copy(
                update={
                    "execution": target.execution.model_copy(
                        update={
                            "event": target.execution.event.model_copy(
                                update={"diagnostics": diagnostics}
                            )
                        }
                    )
                }
            )
            with pytest.raises(
                ComparisonEvidenceBuildError,
                match="comparison_cache_outcome_policy_mismatch",
            ):
                build_persisted_candidate_evidence(
                    comparison_id="comparison-use-tampered",
                    experiment_plan=experiment_plan,
                    reference=reference,
                    dataset=dataset,
                    evaluation_plan=plan,
                    results=tampered,
                    provider_attempts_by_request=_provider_ledger(
                        tampered,
                        plan.run_id,
                    ),
                )


def test_bypass_normalizes_cache_disabled_not_applicable_diagnostics() -> None:
    dataset, evaluation_plan, original_results = _run_fixture()
    assert evaluation_plan.identity.cache_policy is CachePolicy.BYPASS
    results = tuple(
        result.model_copy(
            update={
                "execution": result.execution.model_copy(
                    update={
                        "event": result.execution.event.model_copy(
                            update={
                                "diagnostics": result.execution.event.diagnostics.model_copy(
                                    update={
                                        "cache_status": {
                                            "retrieval": CacheOutcome.NOT_APPLICABLE.value
                                        }
                                    }
                                )
                            }
                        )
                    }
                )
            }
        )
        for result in original_results
        if result.execution is not None
    )
    experiment_plan, reference = _experiment_reference(
        dataset,
        evaluation_plan,
    )

    evidence = build_persisted_candidate_evidence(
        comparison_id="comparison-cache-disabled-bypass",
        experiment_plan=experiment_plan,
        reference=reference,
        dataset=dataset,
        evaluation_plan=evaluation_plan,
        results=results,
        provider_attempts_by_request=_provider_ledger(results, evaluation_plan.run_id),
    )

    assert all(
        attempt.cache_outcome is CacheOutcome.BYPASS for attempt in evidence.logical_attempts
    )


def test_failed_terminal_without_cache_diagnostics_uses_policy_outcome() -> None:
    _, _, original_results = _run_fixture()
    result = original_results[0]
    execution = result.execution
    assert execution is not None
    event = execution.event.model_copy(
        update={
            "kind": StreamEventKind.ERROR,
            "terminal": True,
            "error_code": QAErrorCode.DEADLINE_EXPIRED,
            "reason": None,
            "claims": (),
            "citations": (),
        }
    )

    for policy, expected in (
        (CachePolicy.BYPASS, CacheOutcome.BYPASS),
        (CachePolicy.USE, CacheOutcome.ERROR),
    ):
        failed = result.model_copy(
            update={
                "succeeded": False,
                "safe_error_code": "qa_terminal_error",
                "execution": execution.model_copy(
                    update={
                        "cache_policy": policy,
                        "event": event.model_copy(
                            update={
                                "diagnostics": event.diagnostics.model_copy(
                                    update={"cache_status": {}}
                                )
                            }
                        ),
                    }
                ),
            }
        )
        assert _cache_outcome(failed) is expected

    succeeded = result.model_copy(
        update={
            "execution": execution.model_copy(
                update={
                    "cache_policy": CachePolicy.BYPASS,
                    "event": event.model_copy(
                        update={
                            "kind": StreamEventKind.ANSWER,
                            "error_code": None,
                            "diagnostics": event.diagnostics.model_copy(
                                update={"cache_status": {}}
                            ),
                        }
                    ),
                }
            )
        }
    )
    with pytest.raises(
        ComparisonEvidenceBuildError,
        match="comparison_cache_outcome_unavailable",
    ):
        _cache_outcome(succeeded)


def test_post_retrieval_refusal_requires_cache_equivalence_digest() -> None:
    dataset, base_plan, original_results = _run_fixture()
    plan = base_plan.model_copy(
        update={"identity": base_plan.identity.model_copy(update={"cache_policy": CachePolicy.USE})}
    )
    target_index = next(
        index
        for index, result in enumerate(original_results)
        if result.execution is not None
        and result.execution.event.kind is StreamEventKind.REFUSAL
        and result.execution.event.reason
        not in {RefusalReason.PROMPT_INJECTION, RefusalReason.SAFETY}
    )
    results = []
    for index, result in enumerate(original_results):
        execution = result.execution
        assert execution is not None
        pre_retrieval = execution.event.reason in {
            RefusalReason.PROMPT_INJECTION,
            RefusalReason.SAFETY,
        }
        diagnostics = execution.event.diagnostics.model_copy(
            update={
                "cache_status": ({} if pre_retrieval else {"retrieval": CacheOutcome.MISS.value}),
                "metadata": ({} if pre_retrieval else {"index_revision": "evaluation-revision-1"}),
            }
        )
        results.append(
            result.model_copy(
                update={
                    "execution": execution.model_copy(
                        update={
                            "cache_policy": CachePolicy.USE,
                            "event": execution.event.model_copy(
                                update={"diagnostics": diagnostics}
                            ),
                            "retrieval_evidence_digest": (
                                None
                                if pre_retrieval or index == target_index
                                else execution.retrieval_evidence_digest
                            ),
                        }
                    )
                }
            )
        )
    experiment, reference = _experiment_reference(
        dataset,
        plan,
        cache_behavior="cold",
    )

    with pytest.raises(
        ComparisonEvidenceBuildError,
        match="comparison_retrieval_equivalence_unavailable",
    ):
        build_persisted_candidate_evidence(
            comparison_id="comparison-refusal-missing-digest",
            experiment_plan=experiment,
            reference=reference,
            dataset=dataset,
            evaluation_plan=plan,
            results=results,
            provider_attempts_by_request=_provider_ledger(results, plan.run_id),
        )


@pytest.mark.parametrize("mismatch", ["axis", "case-payload"])
def test_candidate_evidence_rejects_plan_identity_or_case_payload_drift(
    mismatch: str,
) -> None:
    dataset, evaluation_plan, results = _run_fixture()
    experiment_plan, reference = _experiment_reference(dataset, evaluation_plan)
    if mismatch == "axis":
        evaluation_plan = evaluation_plan.model_copy(
            update={
                "identity": evaluation_plan.identity.model_copy(
                    update={
                        "model_identities": {
                            **evaluation_plan.identity.model_identities,
                            "generation": "foreign-generation-model",
                        }
                    }
                )
            }
        )
        expected = "comparison_candidate_axis_identity_mismatch"
    else:
        cases = list(evaluation_plan.cases)
        cases[0] = cases[0].model_copy(update={"question": "Foreign question"})
        evaluation_plan = evaluation_plan.model_copy(update={"cases": tuple(cases)})
        expected = "comparison_candidate_case_binding_mismatch"

    with pytest.raises(ComparisonEvidenceBuildError, match=expected):
        build_persisted_candidate_evidence(
            comparison_id="comparison-1",
            experiment_plan=experiment_plan,
            reference=reference,
            dataset=dataset,
            evaluation_plan=evaluation_plan,
            results=results,
            provider_attempts_by_request=_provider_ledger(
                results,
                evaluation_plan.run_id,
            ),
        )


def test_repeated_candidate_binds_reranker_evidence_to_source_case_and_logical_attempt() -> None:
    dataset, base_plan, _ = _run_fixture()
    source_experiment, _ = _experiment_reference(
        dataset,
        base_plan,
        repeats=2,
        order_policy=ExperimentOrderPolicy.SEEDED_INTERLEAVED,
    )
    schedule = build_comparison_schedule(
        source_experiment,
        dataset,
    )
    variant = source_experiment.variants[0]
    materialized = materialize_variant_cases(
        schedule,
        source_experiment,
        dataset,
        variant.variant_id,
        base_plan.cases,
    )
    evaluation_plan = base_plan.model_copy(
        update={"run_id": "candidate-repeat-run", "cases": materialized}
    )
    dataset_cases = {
        case.case_id: case for case in dataset.cases if isinstance(case, EvaluationCaseV2)
    }
    results = []
    for index, case_input in enumerate(materialized):
        source_id = case_input.source_case_id
        assert source_id is not None
        source_result = _runtime_result(
            dataset_cases[source_id],
            evaluation_plan.run_id,
            index,
        )
        execution = source_result.execution
        assert execution is not None
        request_id = f"repeat-request-{index + 1}"
        session_id = f"repeat-session-{index + 1}"
        event = execution.event.model_copy(
            update={"request_id": request_id, "session_id": session_id}
        )
        updates: dict[str, object] = {
            "case_id": case_input.case_id,
            "request_id": request_id,
            "session_id": session_id,
            "event": event,
        }
        if index == 0:
            ranked = execution.retrieved_chunk_ids
            updates.update(
                {
                    "pre_rerank_chunk_ids": ranked,
                    "post_rerank_chunk_ids": ranked,
                }
            )
        results.append(
            source_result.model_copy(
                update={
                    "run_id": evaluation_plan.run_id,
                    "case_id": case_input.case_id,
                    "execution": execution.model_copy(update=updates),
                    "logical_latency_ms": float(index + 1),
                }
            )
        )
    experiment_plan, _ = _experiment_reference(
        dataset,
        evaluation_plan,
        repeats=2,
        order_policy=ExperimentOrderPolicy.SEEDED_INTERLEAVED,
    )
    variant = experiment_plan.variants[0]
    reference = ComparisonCandidateReference(
        variant_id=variant.variant_id,
        axis_value=variant.axis_value,
        configuration_id=evaluation_plan.identity.configuration_id,
        evaluation_run_id=evaluation_plan.run_id,
    )

    evidence = build_persisted_candidate_evidence(
        comparison_id="comparison-repeat",
        experiment_plan=experiment_plan,
        reference=reference,
        dataset=dataset,
        evaluation_plan=evaluation_plan,
        results=results,
        provider_attempts_by_request=_provider_ledger(results, evaluation_plan.run_id),
    )

    assert len(evidence.logical_attempts) == 48
    assert len({item.case_id for item in evidence.logical_attempts}) == 24
    assert evidence.reranker_evidence
    reranker = evidence.reranker_evidence[0]
    logical = next(
        item for item in evidence.logical_attempts if item.attempt_id == reranker.logical_attempt_id
    )
    assert reranker.case_id == logical.case_id
    assert reranker.case_id in dataset_cases


def test_seeded_candidate_plan_binding_accepts_declared_order_and_rejects_tamper() -> None:
    dataset, base_plan, _ = _run_fixture()
    experiment, reference = _experiment_reference(
        dataset,
        base_plan,
        repeats=2,
        order_policy=ExperimentOrderPolicy.SEEDED_INTERLEAVED,
    )
    schedule = build_comparison_schedule(experiment, dataset)
    cases = materialize_variant_cases(
        schedule,
        experiment,
        dataset,
        reference.variant_id,
        base_plan.cases,
    )
    plan = base_plan.model_copy(update={"cases": cases})

    validate_candidate_plan_binding(experiment, reference, dataset, plan)

    tampered_cases = list(cases)
    tampered_cases[0], tampered_cases[1] = tampered_cases[1], tampered_cases[0]
    tampered = plan.model_copy(update={"cases": tuple(tampered_cases)})
    with pytest.raises(
        ComparisonEvidenceBuildError,
        match="comparison_candidate_schedule_order_mismatch",
    ):
        validate_candidate_plan_binding(experiment, reference, dataset, tampered)


def test_candidate_evidence_supports_cross_validated_declared_dataset_subset() -> None:
    dataset, base_plan, original_results = _run_fixture()
    selected_case_ids = tuple(item.case_id for item in dataset.cases[:3])
    experiment, reference = _experiment_reference(dataset, base_plan)
    experiment = type(experiment).create(
        **{
            **experiment.model_dump(exclude={"content_hash"}),
            "fixed_identities": experiment.fixed_identities.model_copy(
                update={
                    "case_count": len(selected_case_ids),
                    "case_set_hash": case_ids_content_hash(selected_case_ids),
                }
            ),
        }
    )
    schedule = build_comparison_schedule(
        experiment,
        dataset,
        selected_case_ids=selected_case_ids,
    )
    cases = materialize_variant_cases(
        schedule,
        experiment,
        dataset,
        reference.variant_id,
        base_plan.cases,
    )
    plan = base_plan.model_copy(update={"cases": cases})
    by_case = {item.case_id: item for item in original_results}
    results = tuple(by_case[item.source_case_id or item.case_id] for item in cases)

    evidence = build_persisted_candidate_evidence(
        comparison_id="comparison-subset",
        experiment_plan=experiment,
        reference=reference,
        dataset=dataset,
        evaluation_plan=plan,
        results=results,
        provider_attempts_by_request=_provider_ledger(results, plan.run_id),
    )

    assert evidence.case_ids == selected_case_ids
    assert evidence.identity_projection.dataset_hash == dataset.manifest.content_hash
    assert evidence.identity_projection.case_set_hash == case_ids_content_hash(selected_case_ids)
