"""Conservative whole-suite provider-call and monetary preflight."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from rag_mvp.domain.evaluation import ModelRole
from rag_mvp.domain.qa import ConversationRole, ConversationTurn
from rag_mvp.domain.retrieval import CachePolicy
from rag_mvp.evaluation.comparison import ComparisonCandidateReference
from rag_mvp.evaluation.comparison_evidence import (
    ComparisonEvidenceBuildError,
    validate_candidate_plan_binding,
)
from rag_mvp.evaluation.comparison_schedule import (
    ComparisonScheduleError,
    cache_eligible_case_ids,
)
from rag_mvp.evaluation.dataset import EvaluationDataset
from rag_mvp.evaluation.environment import EvaluationIndexReuseKey
from rag_mvp.evaluation.experiment import (
    ExperimentAxis,
    ExperimentPlan,
    ExperimentPricingRate,
    PricingRole,
)
from rag_mvp.evaluation.plan import evaluation_scorer_versions
from rag_mvp.evaluation.report_builder import case_ids_content_hash
from rag_mvp.evaluation.runner import EvaluationRunPlan
from rag_mvp.evaluation.work_budget import (
    ProviderWorkBudget,
    ProviderWorkBudgetError,
    ProviderWorkBudgetSnapshot,
    ProviderWorkEstimate,
)
from rag_mvp.qa.prompt import GENERATOR_PROMPT_VERSION
from rag_mvp.qa.query_rewrite import QueryRewriteError, QueryRewriter
from rag_mvp.retrieval.rerank import RerankStage, RerankTruncationPolicy

_ONE_MILLION = Decimal(1_000_000)
_GENERATION_OUTPUT_TOKENS = 512
# Cache entries are inserted during cold retrieval, before the rest of that case,
# then revisited only after the entire cold phase.  The provider deadline covers
# QA work, but not runner scoring, persistence, or event-loop scheduling.  Reserve
# ten seconds per eligible cold/warm pair plus a fixed phase-transition minute.
CACHE_EXPERIMENT_PER_CASE_MARGIN_SECONDS = 10.0
CACHE_EXPERIMENT_PHASE_MARGIN_SECONDS = 60.0
MAXIMUM_CACHE_TTL_SECONDS = 86_400.0
REGISTERED_GENERATION_PROMPT_VERSION = "grounded-claims-json-v4"
REGISTERED_FAITHFULNESS_SCORER_VERSION = "faithfulness-approved-proposition-support-v4"
REGISTERED_TEXT_SUPPORT_MATCHER_VERSION = "expected-fact-approved-proposition-v5"
REGISTERED_TEXT_SUPPORT_NORMALIZATION_VERSION = "nfkc-casefold-proposition-format-v2"
REGISTERED_SCORING_PIPELINE_VERSION = "deterministic-evaluation-scoring-v3"
_PROMPT_BYTE_OVERHEAD_BY_VERSION = {
    ("generation", REGISTERED_GENERATION_PROMPT_VERSION): 8_192,
    ("reranking", RerankStage.PROMPT_VERSION): 8_192,
}
REGISTERED_COMPARISON_DATASET_ID = "original-pdf-acceptance"
REGISTERED_COMPARISON_DATASET_VERSION = "2.0.0"
REGISTERED_COMPARISON_DATASET_HASH = (
    "sha256:ad1220c67b0542fef8e02f0a7a537533228b56ec8a82d40a69188e1f9895672d"
)
REGISTERED_COMPARISON_CORPUS_ID = "acceptance-bilingual-corpus"
REGISTERED_COMPARISON_CORPUS_VERSION = "2.0.0"
REGISTERED_COMPARISON_CORPUS_HASH = (
    "sha256:019c0b91a453318fc980df3088b1d0fb3ad8802d375a2537d6470c56eb11839e"
)


class ComparisonPreflightError(RuntimeError):
    """Stable fail-closed comparison preflight error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ComparisonWorkPreflight:
    estimates: tuple[ProviderWorkEstimate, ...]
    budget: ProviderWorkBudget
    snapshot: ProviderWorkBudgetSnapshot
    logical_attempt_count: int
    index_build_count: int
    cache_eligible_case_count: int = 0
    minimum_cache_ttl_seconds: float | None = None


def minimum_cache_experiment_ttl_seconds(
    eligible_case_count: int,
    qa_deadline_seconds: float,
) -> float:
    """Return the bounded cold-to-warm survival window committed by the plan."""

    if type(eligible_case_count) is not int or eligible_case_count < 1:
        raise ComparisonPreflightError("comparison_cache_case_count_invalid")
    if (
        isinstance(qa_deadline_seconds, bool)
        or not isinstance(qa_deadline_seconds, (int, float))
        or not math.isfinite(float(qa_deadline_seconds))
        or qa_deadline_seconds <= 0
    ):
        raise ComparisonPreflightError("comparison_cache_expiry_identity_invalid")
    provider_window = eligible_case_count * float(qa_deadline_seconds)
    orchestration_margin = max(
        CACHE_EXPERIMENT_PHASE_MARGIN_SECONDS,
        eligible_case_count * CACHE_EXPERIMENT_PER_CASE_MARGIN_SECONDS,
    )
    required = provider_window + orchestration_margin
    if required > MAXIMUM_CACHE_TTL_SECONDS:
        raise ComparisonPreflightError("comparison_cache_ttl_window_unrepresentable")
    return required


def preflight_comparison_work(
    comparison_id: str,
    experiment_plan: ExperimentPlan,
    dataset: EvaluationDataset,
    candidate_plans: Mapping[str, EvaluationRunPlan],
) -> ComparisonWorkPreflight:
    """Reserve the full worst case before index installation or candidate calls."""

    validate_registered_comparison_dataset(dataset)
    experiment_plan.verify_hash()
    expected_variants = tuple(item.variant_id for item in experiment_plan.variants)
    if set(candidate_plans) != set(expected_variants):
        raise ComparisonPreflightError("comparison_candidate_plan_set_mismatch")
    if len({item.run_id for item in candidate_plans.values()}) != len(candidate_plans):
        raise ComparisonPreflightError("comparison_candidate_run_duplicate")
    expected_scorers = evaluation_scorer_versions(dataset)
    expected_registered_scorers = {
        "faithfulness": REGISTERED_FAITHFULNESS_SCORER_VERSION,
        "faithfulness-text-matcher": REGISTERED_TEXT_SUPPORT_MATCHER_VERSION,
        "faithfulness-text-normalization": REGISTERED_TEXT_SUPPORT_NORMALIZATION_VERSION,
        "scoring-pipeline": REGISTERED_SCORING_PIPELINE_VERSION,
    }
    if any(
        dict(plan.identity.scorer_versions) != expected_scorers
        or any(
            plan.identity.scorer_versions.get(name) != version
            for name, version in expected_registered_scorers.items()
        )
        for plan in candidate_plans.values()
    ):
        raise ComparisonPreflightError("comparison_scorer_version_mismatch")
    cache_case_count, minimum_cache_ttl = _validate_cache_experiment(
        experiment_plan,
        dataset,
        candidate_plans,
    )
    estimates: list[ProviderWorkEstimate] = []
    reuse_installations: dict[str, EvaluationRunPlan] = {}
    for variant_id in expected_variants:
        evaluation_plan = candidate_plans[variant_id]
        variant = next(item for item in experiment_plan.variants if item.variant_id == variant_id)
        try:
            validate_candidate_plan_binding(
                experiment_plan,
                ComparisonCandidateReference(
                    variant_id=variant.variant_id,
                    axis_value=variant.axis_value,
                    configuration_id=variant.configuration_id,
                    evaluation_run_id=evaluation_plan.run_id,
                ),
                dataset,
                evaluation_plan,
            )
        except (ComparisonEvidenceBuildError, TypeError, ValueError) as error:
            code = getattr(error, "code", "comparison_candidate_plan_invalid")
            raise ComparisonPreflightError(str(code)) from None
        key = EvaluationIndexReuseKey.from_plan(evaluation_plan, dataset)
        reuse_installations.setdefault(key.digest, evaluation_plan)
        estimates.extend(
            _case_estimate(
                comparison_id,
                variant_id,
                evaluation_plan,
                case_index,
                experiment_plan,
                dataset,
            )
            for case_index in range(len(evaluation_plan.cases))
        )
    for index, evaluation_plan in enumerate(reuse_installations.values()):
        estimates.append(
            _index_estimate(
                comparison_id,
                index,
                evaluation_plan,
                experiment_plan,
                dataset,
            )
        )
    budget = ProviderWorkBudget(
        experiment_plan.maximum_provider_calls,
        experiment_plan.maximum_cost,
        experiment_plan.pricing.currency,
    )
    try:
        budget.reserve_many(tuple(estimates))
    except ProviderWorkBudgetError as error:
        raise ComparisonPreflightError(error.code) from None
    return ComparisonWorkPreflight(
        estimates=tuple(estimates),
        budget=budget,
        snapshot=budget.snapshot(),
        logical_attempt_count=sum(len(item.cases) for item in candidate_plans.values()),
        index_build_count=len(reuse_installations),
        cache_eligible_case_count=cache_case_count,
        minimum_cache_ttl_seconds=minimum_cache_ttl,
    )


def validate_registered_comparison_dataset(dataset: EvaluationDataset) -> None:
    """Bind paid registered comparisons to the independently pinned acceptance corpus."""

    manifest = dataset.manifest
    corpus = dataset.corpus.manifest
    if (
        manifest.dataset_id != REGISTERED_COMPARISON_DATASET_ID
        or manifest.version != REGISTERED_COMPARISON_DATASET_VERSION
        or manifest.content_hash != REGISTERED_COMPARISON_DATASET_HASH
        or corpus.snapshot_id != REGISTERED_COMPARISON_CORPUS_ID
        or corpus.version != REGISTERED_COMPARISON_CORPUS_VERSION
        or corpus.content_hash != REGISTERED_COMPARISON_CORPUS_HASH
    ):
        raise ComparisonPreflightError("comparison_registered_dataset_identity_mismatch")


def _validate_cache_experiment(
    experiment_plan: ExperimentPlan,
    dataset: EvaluationDataset,
    candidate_plans: Mapping[str, EvaluationRunPlan],
) -> tuple[int, float | None]:
    if experiment_plan.axis is not ExperimentAxis.CACHE_BEHAVIOR:
        return 0, None
    if experiment_plan.cache_policy is not CachePolicy.USE or tuple(
        item.axis_value for item in experiment_plan.variants
    ) != ("cold", "warm"):
        raise ComparisonPreflightError("comparison_cache_plan_invalid")
    if experiment_plan.repeat_order_policy.repeats_per_case != 1:
        raise ComparisonPreflightError("comparison_cache_repeats_invalid")
    ordered_plans = tuple(candidate_plans[item.variant_id] for item in experiment_plan.variants)
    first_sequence = tuple(item.source_case_id or item.case_id for item in ordered_plans[0].cases)
    if not first_sequence or any(
        tuple(item.source_case_id or item.case_id for item in plan.cases) != first_sequence
        for plan in ordered_plans[1:]
    ):
        raise ComparisonPreflightError("comparison_cache_case_order_mismatch")
    selected_set = set(first_sequence)
    canonical_selected = tuple(
        item.case_id for item in dataset.cases if item.case_id in selected_set
    )
    try:
        eligible = cache_eligible_case_ids(dataset)
    except ComparisonScheduleError as error:
        raise ComparisonPreflightError(error.code) from None
    if (
        len(first_sequence) != len(selected_set)
        or canonical_selected != eligible
        or experiment_plan.fixed_identities.case_count != len(eligible)
        or experiment_plan.fixed_identities.case_set_hash != case_ids_content_hash(eligible)
    ):
        raise ComparisonPreflightError("comparison_cache_case_set_ineligible")
    _require_unique_cache_queries(dataset, eligible)
    runtime_ids = {plan.identity.runtime_configuration_id for plan in ordered_plans}
    if None in runtime_ids or len(runtime_ids) != 1:
        raise ComparisonPreflightError("comparison_cache_runtime_identity_mismatch")
    retrieval_identities = tuple(plan.identity.retrieval_configuration for plan in ordered_plans)
    if any(value != retrieval_identities[0] for value in retrieval_identities[1:]):
        raise ComparisonPreflightError("comparison_cache_configuration_mismatch")
    retrieval = retrieval_identities[0]
    if retrieval.get("retrieval_cache_enabled") is not True:
        raise ComparisonPreflightError("comparison_cache_disabled")
    capacity = retrieval.get("retrieval_cache_max_entries")
    ttl = retrieval.get("retrieval_cache_ttl_seconds")
    deadline = ordered_plans[0].identity.generation_settings.get("qa_deadline_seconds")
    if type(capacity) is not int or capacity < len(eligible):
        raise ComparisonPreflightError("comparison_cache_capacity_insufficient")
    if (
        isinstance(ttl, bool)
        or not isinstance(ttl, (int, float))
        or isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or float(ttl) <= 0
        or float(deadline) <= 0
    ):
        raise ComparisonPreflightError("comparison_cache_expiry_identity_invalid")
    minimum_ttl = minimum_cache_experiment_ttl_seconds(len(eligible), float(deadline))
    if float(ttl) < minimum_ttl:
        raise ComparisonPreflightError("comparison_cache_ttl_insufficient")
    return len(eligible), minimum_ttl


def _require_unique_cache_queries(
    dataset: EvaluationDataset,
    eligible_case_ids: tuple[str, ...],
) -> None:
    eligible = set(eligible_case_ids)
    digests: set[str] = set()
    try:
        for case in dataset.cases:
            if case.case_id not in eligible:
                continue
            session_id = "comparison-preflight"
            history = tuple(
                ConversationTurn(
                    turn_id=f"preflight-turn-{ordinal + 1}",
                    session_id=session_id,
                    ordinal=ordinal,
                    role=ConversationRole(turn.role),
                    content=turn.content,
                )
                for ordinal, turn in enumerate(case.history)
            )
            current = ConversationTurn(
                turn_id=f"preflight-turn-{len(history) + 1}",
                session_id=session_id,
                ordinal=len(history),
                role=ConversationRole.USER,
                content=case.question,
            )
            prepared = QueryRewriter().prepare(
                (*history, current),
                requested_language=(
                    case.language.value if case.language.value in {"zh", "en"} else None
                ),
            )
            digest = hashlib.sha256(prepared.query.encode("utf-8")).hexdigest()
            if digest in digests:
                raise ComparisonPreflightError("comparison_cache_query_duplicate")
            digests.add(digest)
    except QueryRewriteError as error:
        raise ComparisonPreflightError(f"comparison_cache_{error.code}") from None


def _case_estimate(
    comparison_id: str,
    variant_id: str,
    evaluation_plan: EvaluationRunPlan,
    case_index: int,
    experiment_plan: ExperimentPlan,
    dataset: EvaluationDataset,
) -> ProviderWorkEstimate:
    case = evaluation_plan.cases[case_index]
    retry_count = _retry_count(evaluation_plan)
    rerank_limit = _retrieval_identity_int(
        evaluation_plan,
        "rerank_candidate_limit",
        minimum=1,
    )
    candidate_token_bound = sum(
        sorted(
            (_utf8_token_upper_bound(chunk.text) for chunk in dataset.production_chunks),
            reverse=True,
        )[:rerank_limit]
    )
    query_tokens = _utf8_token_upper_bound(case.question) + sum(
        _utf8_token_upper_bound(turn.content) for turn in case.history
    )
    context_limit = _retrieval_identity_int(
        evaluation_plan,
        "context_chunk_limit",
        minimum=1,
    )
    context_tokens = sum(
        sorted(
            (_utf8_token_upper_bound(chunk.text) for chunk in dataset.production_chunks),
            reverse=True,
        )[:context_limit]
    )
    operations: list[tuple[ModelRole, int, int]] = [
        (ModelRole.EMBEDDING, query_tokens, 0),
        (ModelRole.EMBEDDING, query_tokens + candidate_token_bound, 0),
        (
            ModelRole.GENERATION,
            _prompt_byte_overhead(evaluation_plan, "generation") + query_tokens + context_tokens,
            _GENERATION_OUTPUT_TOKENS,
        ),
    ]
    if evaluation_plan.identity.retrieval_configuration.get("mode") == "hybrid-rerank":
        submitted = rerank_limit
        operations.append(
            (
                ModelRole.RERANKING,
                _rerank_input_byte_bound(evaluation_plan, dataset, submitted),
                max(64, submitted * 24),
            )
        )
    cost = sum(
        (
            _operation_cost(
                experiment_plan,
                evaluation_plan,
                role,
                input_tokens,
                output_tokens,
            )
            * retry_count
            for role, input_tokens, output_tokens in operations
        ),
        start=Decimal(0),
    )
    return ProviderWorkEstimate(
        work_id=_safe_work_id(
            comparison_id,
            variant_id,
            evaluation_plan.run_id,
            f"case-{case_index + 1}",
        ),
        provider_calls=len(operations) * retry_count,
        conservative_cost=cost,
        currency=experiment_plan.pricing.currency,
    )


def _index_estimate(
    comparison_id: str,
    index: int,
    evaluation_plan: EvaluationRunPlan,
    experiment_plan: ExperimentPlan,
    dataset: EvaluationDataset,
) -> ProviderWorkEstimate:
    retry_count = _retry_count(evaluation_plan)
    cost = sum(
        (
            _operation_cost(
                experiment_plan,
                evaluation_plan,
                ModelRole.EMBEDDING,
                _utf8_token_upper_bound(chunk.text),
                0,
            )
            * retry_count
            for chunk in dataset.production_chunks
        ),
        start=Decimal(0),
    )
    return ProviderWorkEstimate(
        work_id=_safe_work_id(comparison_id, f"index-{index + 1}"),
        provider_calls=len(dataset.production_chunks) * retry_count,
        conservative_cost=cost,
        currency=experiment_plan.pricing.currency,
    )


def _operation_cost(
    experiment_plan: ExperimentPlan,
    evaluation_plan: EvaluationRunPlan,
    role: ModelRole,
    input_tokens: int,
    output_tokens: int,
) -> Decimal:
    provider = evaluation_plan.identity.provider_identities.get(role.value)
    model = evaluation_plan.identity.model_identities.get(role.value)
    if not provider or not model or model == "disabled":
        raise ComparisonPreflightError("comparison_provider_identity_missing")
    rate = _exact_rate(experiment_plan, role, provider, model)
    if rate.input_per_million is None:
        raise ComparisonPreflightError("comparison_input_pricing_missing")
    cost = Decimal(input_tokens) * rate.input_per_million / _ONE_MILLION
    if role is not ModelRole.EMBEDDING:
        if rate.output_per_million is None:
            raise ComparisonPreflightError("comparison_output_pricing_missing")
        cost += Decimal(output_tokens) * rate.output_per_million / _ONE_MILLION
    return cost


def _exact_rate(
    plan: ExperimentPlan,
    role: ModelRole,
    provider: str,
    model: str,
) -> ExperimentPricingRate:
    pricing_role = PricingRole(role.value)
    match = tuple(
        item
        for item in plan.pricing.rate_card
        if item.role is pricing_role and item.provider == provider and item.model == model
    )
    if len(match) != 1:
        raise ComparisonPreflightError("comparison_exact_pricing_missing")
    return match[0]


def _retry_count(plan: EvaluationRunPlan) -> int:
    value = plan.identity.generation_settings.get("provider_retry_limit")
    if type(value) is not int or value < 0:
        raise ComparisonPreflightError("comparison_retry_identity_invalid")
    return value + 1


def _retrieval_identity_int(
    plan: EvaluationRunPlan,
    name: str,
    *,
    minimum: int,
) -> int:
    value = plan.identity.retrieval_configuration.get(name)
    if type(value) is not int or value < minimum:
        raise ComparisonPreflightError("comparison_retrieval_identity_invalid")
    return value


def _prompt_byte_overhead(plan: EvaluationRunPlan, role: str) -> int:
    version = plan.identity.prompt_versions.get(role)
    if role == "generation" and GENERATOR_PROMPT_VERSION != REGISTERED_GENERATION_PROMPT_VERSION:
        raise ComparisonPreflightError("comparison_prompt_overhead_unknown")
    value = _PROMPT_BYTE_OVERHEAD_BY_VERSION.get((role, version or ""))
    if value is None:
        raise ComparisonPreflightError("comparison_prompt_overhead_unknown")
    return value


def _rerank_input_byte_bound(
    plan: EvaluationRunPlan,
    dataset: EvaluationDataset,
    submitted: int,
) -> int:
    truncation = RerankTruncationPolicy()
    candidate_bound = sum(
        sorted(
            (
                _utf8_token_upper_bound(truncation.truncate_candidate(chunk.text))
                for chunk in dataset.production_chunks
            ),
            reverse=True,
        )[:submitted]
    )
    query_bound = (
        min(
            truncation.maximum_query_characters,
            truncation.maximum_query_tokens,
        )
        * 4
    )
    return _prompt_byte_overhead(plan, "reranking") + query_bound + candidate_bound


def _utf8_token_upper_bound(value: str) -> int:
    return max(1, len(value.encode("utf-8")))


def _safe_work_id(*parts: str) -> str:
    raw = ".".join(parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    prefix = "comparison-work-" + "-".join(parts[:2])
    normalized = "".join(
        character if character.isalnum() or character in "_.-" else "-" for character in prefix
    )
    return f"{normalized[:220]}.{digest}"


__all__ = [
    "CACHE_EXPERIMENT_PER_CASE_MARGIN_SECONDS",
    "CACHE_EXPERIMENT_PHASE_MARGIN_SECONDS",
    "MAXIMUM_CACHE_TTL_SECONDS",
    "REGISTERED_COMPARISON_CORPUS_HASH",
    "REGISTERED_COMPARISON_CORPUS_ID",
    "REGISTERED_COMPARISON_CORPUS_VERSION",
    "REGISTERED_COMPARISON_DATASET_HASH",
    "REGISTERED_COMPARISON_DATASET_ID",
    "REGISTERED_COMPARISON_DATASET_VERSION",
    "REGISTERED_FAITHFULNESS_SCORER_VERSION",
    "REGISTERED_GENERATION_PROMPT_VERSION",
    "REGISTERED_SCORING_PIPELINE_VERSION",
    "REGISTERED_TEXT_SUPPORT_MATCHER_VERSION",
    "REGISTERED_TEXT_SUPPORT_NORMALIZATION_VERSION",
    "ComparisonPreflightError",
    "ComparisonWorkPreflight",
    "minimum_cache_experiment_ttl_seconds",
    "preflight_comparison_work",
    "validate_registered_comparison_dataset",
]
