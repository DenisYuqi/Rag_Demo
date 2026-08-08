"""Narrow, fail-closed Ragas integration for schema-v2 evaluation."""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Protocol

from rag_mvp.config.settings import Settings
from rag_mvp.domain.qa import StreamEventKind
from rag_mvp.evaluation.dataset import Answerability, EvaluationCaseV2, EvaluationDataset
from rag_mvp.evaluation.grounding_metrics import (
    EvidenceVerdict,
    MetricEvidence,
    MetricName,
    MetricResult,
)
from rag_mvp.evaluation.runner import PersistedCaseResult
from rag_mvp.evaluation.scoring_v2 import AdvancedEvaluationScorecard, score_evaluation_v2
from rag_mvp.providers.openai_client import OpenAIClientConfig, create_async_openai_client
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, Redactor

RAGAS_BACKEND_VERSION = "ragas-hybrid-v1"
RAGAS_SCORING_PIPELINE_VERSION = "ragas-hybrid-evaluation-scoring-v1"
RAGAS_ADAPTER_VERSION = "ragas-collections-adapter-v1"


def _ragas_package_version() -> str:
    try:
        return version("ragas")
    except PackageNotFoundError:
        return "unavailable"


RAGAS_PACKAGE_VERSION = _ragas_package_version()
RAGAS_FAITHFULNESS_SCORER_VERSION = (
    f"ragas-faithfulness-{RAGAS_PACKAGE_VERSION}-{RAGAS_ADAPTER_VERSION}"
)
RAGAS_CONTEXT_PRECISION_SCORER_VERSION = (
    f"ragas-context-precision-{RAGAS_PACKAGE_VERSION}-{RAGAS_ADAPTER_VERSION}"
)


class EvaluationRagasError(RuntimeError):
    """Stable content-free failure raised by the semantic evaluator."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class RagasCaseSample:
    case_id: str
    user_input: str
    response: str
    reference: str
    retrieved_contexts: tuple[str, ...]
    context_chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (self.case_id, self.user_input, self.response, self.reference)
        ):
            raise ValueError("ragas_sample_text_invalid")
        if not self.retrieved_contexts or any(
            not isinstance(value, str) or not value.strip() for value in self.retrieved_contexts
        ):
            raise ValueError("ragas_sample_context_invalid")
        if len(self.context_chunk_ids) != len(set(self.context_chunk_ids)):
            raise ValueError("ragas_sample_context_ids_duplicate")


class RagasMetricGateway(Protocol):
    async def score_faithfulness(self, sample: RagasCaseSample) -> float: ...

    async def score_context_precision(self, sample: RagasCaseSample) -> float: ...

    async def close(self) -> None: ...


@dataclass(slots=True)
class NativeRagasMetricGateway:
    """Own Ragas and OpenAI objects behind one replaceable adapter boundary."""

    client: object
    faithfulness_metric: object
    context_precision_metric: object

    @classmethod
    def from_settings(cls, settings: Settings) -> NativeRagasMetricGateway:
        if settings.evaluation_scorer_backend != "ragas":
            raise EvaluationRagasError("evaluation_ragas_backend_not_selected")
        if settings.provider_backend != "openai" or settings.openai_api_key is None:
            raise EvaluationRagasError("evaluation_ragas_judge_unavailable")
        os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
        try:
            from ragas.llms import llm_factory
            from ragas.metrics.collections import ContextPrecision, Faithfulness
        except Exception:
            raise EvaluationRagasError("evaluation_ragas_dependency_unavailable") from None
        proxy_url = (
            settings.openai_proxy_url.get_secret_value()
            if settings.openai_proxy_url is not None
            else None
        )
        try:
            client = create_async_openai_client(
                OpenAIClientConfig(
                    base_url=settings.openai_base_url,
                    api_key=settings.openai_api_key.get_secret_value(),
                    secret_reference=":".join(("env", "RAG_MVP_OPENAI_API_KEY")),
                    timeout_seconds=settings.evaluation_ragas_timeout_seconds,
                    proxy_url=proxy_url,
                )
            )
            llm = llm_factory(
                settings.effective_evaluation_judge_model,
                provider="openai",
                client=client,
                temperature=0.0,
            )
            _apply_openai_parameter_compatibility(llm, settings)
            return cls(
                client=client,
                faithfulness_metric=Faithfulness(llm=llm),
                context_precision_metric=ContextPrecision(llm=llm),
            )
        except Exception:
            raise EvaluationRagasError("evaluation_ragas_judge_unavailable") from None

    async def score_faithfulness(self, sample: RagasCaseSample) -> float:
        return await _call_native_metric(
            self.faithfulness_metric,
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=list(sample.retrieved_contexts),
        )

    async def score_context_precision(self, sample: RagasCaseSample) -> float:
        return await _call_native_metric(
            self.context_precision_metric,
            user_input=sample.user_input,
            reference=sample.reference,
            retrieved_contexts=list(sample.retrieved_contexts),
        )

    async def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            result = close()
            if isinstance(result, Awaitable):
                await result


async def _call_native_metric(metric: object, **kwargs: object) -> float:
    call = getattr(metric, "ascore", None)
    if not callable(call):
        raise EvaluationRagasError("evaluation_ragas_metric_unavailable")
    try:
        result = await call(**kwargs)
        return _unit_score(getattr(result, "value", None))
    except EvaluationRagasError:
        raise
    except Exception:
        raise EvaluationRagasError("evaluation_ragas_metric_failed") from None


def _apply_openai_parameter_compatibility(llm: object, settings: Settings) -> None:
    """Honor the application's OpenAI token parameter policy inside Ragas.

    Ragas 0.4's model-name heuristic does not recognize dotted GPT generations
    such as ``gpt-5.4`` and otherwise sends the rejected ``max_tokens`` field.
    """

    model_args = getattr(llm, "model_args", None)
    if not isinstance(model_args, dict):
        raise EvaluationRagasError("evaluation_ragas_judge_unavailable")
    if settings.openai_max_tokens_parameter == "max_completion_tokens":
        maximum = model_args.pop("max_tokens", None)
        if maximum is not None:
            model_args["max_completion_tokens"] = maximum
    model = settings.effective_evaluation_judge_model.lower()
    if model.startswith("gpt-5") or (
        len(model) >= 2 and model[0] == "o" and model[1].isdigit()
    ):
        model_args["temperature"] = 1.0
        model_args.pop("top_p", None)


def build_ragas_samples(
    dataset: EvaluationDataset,
    results: Sequence[PersistedCaseResult],
) -> tuple[RagasCaseSample, ...]:
    cases = tuple(dataset.cases)
    if not cases or any(not isinstance(case, EvaluationCaseV2) for case in cases):
        raise EvaluationRagasError("evaluation_ragas_dataset_v2_required")
    v2_cases = tuple(case for case in cases if isinstance(case, EvaluationCaseV2))
    result_by_id = {result.case_id: result for result in results}
    if len(result_by_id) != len(results) or set(result_by_id) != {
        case.case_id for case in v2_cases
    }:
        raise EvaluationRagasError("evaluation_ragas_case_set_mismatch")
    child_by_id = dataset.corpus.chunks_by_id
    parent_by_id = {parent.parent_chunk_id: parent for parent in dataset.corpus.parents}
    samples: list[RagasCaseSample] = []
    for case in v2_cases:
        persisted = result_by_id[case.case_id]
        execution = persisted.execution
        if (
            not persisted.succeeded
            or execution is None
            or case.answerability is not Answerability.ANSWERABLE
            or execution.event.kind is not StreamEventKind.ANSWER
        ):
            continue
        response = "\n".join(claim.text for claim in execution.event.claims).strip()
        reference = "\n".join(fact.text for fact in case.expected_facts).strip()
        if not response or not reference or not execution.context_chunk_ids:
            raise EvaluationRagasError("evaluation_ragas_required_evidence_missing")
        contexts: list[str] = []
        context_keys: set[str] = set()
        for chunk_id in execution.context_chunk_ids:
            child = child_by_id.get(chunk_id)
            if child is None:
                raise EvaluationRagasError("evaluation_ragas_context_missing")
            parent_id = child.parent_chunk_id or child.chunk_id
            if parent_id in context_keys:
                continue
            parent = parent_by_id.get(parent_id)
            contexts.append(child.text if parent is None else parent.text)
            context_keys.add(parent_id)
        samples.append(
            RagasCaseSample(
                case_id=case.case_id,
                user_input=case.question,
                response=response,
                reference=reference,
                retrieved_contexts=tuple(contexts),
                context_chunk_ids=execution.context_chunk_ids,
            )
        )
    return tuple(samples)


async def score_evaluation_v2_with_ragas(
    dataset: EvaluationDataset,
    results: tuple[PersistedCaseResult, ...],
    *,
    gateway: RagasMetricGateway,
    timeout_seconds: float,
    retry_limit: int,
    maximum_concurrency: int,
    redactor: Redactor = DEFAULT_REDACTOR,
) -> AdvancedEvaluationScorecard:
    samples = build_ragas_samples(dataset, results)
    semaphore = asyncio.Semaphore(maximum_concurrency)

    async def bounded(call: Callable[[], Awaitable[float]]) -> float:
        async with semaphore:
            return await _bounded_metric_call(
                call,
                timeout_seconds=timeout_seconds,
                retry_limit=retry_limit,
            )

    async def score_sample(sample: RagasCaseSample) -> tuple[str, float, float]:
        faithfulness, context_precision = await asyncio.gather(
            bounded(lambda: gateway.score_faithfulness(sample)),
            bounded(lambda: gateway.score_context_precision(sample)),
        )
        return sample.case_id, faithfulness, context_precision

    scored = await asyncio.gather(*(score_sample(sample) for sample in samples))
    score_by_case = {
        case_id: (faithfulness, context_precision)
        for case_id, faithfulness, context_precision in scored
    }
    sample_by_case = {sample.case_id: sample for sample in samples}
    semantic_results: list[MetricResult] = []
    for case in dataset.cases:
        scores = score_by_case.get(case.case_id)
        sample = sample_by_case.get(case.case_id)
        if scores is None or sample is None:
            semantic_results.extend(
                (
                    _ineligible_result(
                        case.case_id,
                        MetricName.FAITHFULNESS,
                        RAGAS_FAITHFULNESS_SCORER_VERSION,
                    ),
                    _ineligible_result(
                        case.case_id,
                        MetricName.CONTEXT_PRECISION,
                        RAGAS_CONTEXT_PRECISION_SCORER_VERSION,
                    ),
                )
            )
            continue
        semantic_results.extend(
            (
                _semantic_result(
                    case_id=case.case_id,
                    metric=MetricName.FAITHFULNESS,
                    scorer_version=RAGAS_FAITHFULNESS_SCORER_VERSION,
                    score=scores[0],
                    evidence_references=sample.context_chunk_ids,
                ),
                _semantic_result(
                    case_id=case.case_id,
                    metric=MetricName.CONTEXT_PRECISION,
                    scorer_version=RAGAS_CONTEXT_PRECISION_SCORER_VERSION,
                    score=scores[1],
                    evidence_references=sample.context_chunk_ids,
                ),
            )
        )
    return score_evaluation_v2(
        dataset,
        results,
        redactor=redactor,
        semantic_results=tuple(semantic_results),
        scoring_version=RAGAS_SCORING_PIPELINE_VERSION,
    )


async def _bounded_metric_call(
    call: Callable[[], Awaitable[float]],
    *,
    timeout_seconds: float,
    retry_limit: int,
) -> float:
    for attempt in range(retry_limit + 1):
        try:
            async with asyncio.timeout(timeout_seconds):
                return _unit_score(await call())
        except Exception:
            if attempt >= retry_limit:
                raise EvaluationRagasError("evaluation_ragas_judge_failed") from None
    raise EvaluationRagasError("evaluation_ragas_judge_failed")


def _semantic_result(
    *,
    case_id: str,
    metric: MetricName,
    scorer_version: str,
    score: float,
    evidence_references: tuple[str, ...],
) -> MetricResult:
    value = _unit_score(score)
    positive = value >= 0.5
    verdict = (
        EvidenceVerdict.SUPPORTED if metric is MetricName.FAITHFULNESS else EvidenceVerdict.RELEVANT
    )
    if not positive:
        verdict = (
            EvidenceVerdict.UNSUPPORTED
            if metric is MetricName.FAITHFULNESS
            else EvidenceVerdict.IRRELEVANT
        )
    return MetricResult(
        case_id=case_id,
        metric=metric,
        scorer_version=scorer_version,
        eligible=True,
        score=value,
        numerator=value,
        denominator=1,
        rationale="ragas_native_score",
        evidence=(
            MetricEvidence(
                reference_id=f"{case_id}-{metric.value}-ragas",
                verdict=verdict,
                rationale="ragas_native_metric_result",
                evidence_references=evidence_references,
            ),
        ),
    )


def _ineligible_result(case_id: str, metric: MetricName, scorer_version: str) -> MetricResult:
    return MetricResult(
        case_id=case_id,
        metric=metric,
        scorer_version=scorer_version,
        eligible=False,
        score=None,
        numerator=None,
        denominator=None,
        rationale="ragas_answer_metric_ineligible",
    )


def _unit_score(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise EvaluationRagasError("evaluation_ragas_score_invalid")
    return float(value)


def ragas_judge_identity(settings: Settings) -> str:
    """Return a stable non-secret identity for scorer-version comparisons."""

    payload = "|".join(
        (
            settings.openai_base_url.rstrip("/"),
            settings.effective_evaluation_judge_model,
            RAGAS_PACKAGE_VERSION,
            RAGAS_ADAPTER_VERSION,
        )
    )
    return f"ragas-judge-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


__all__ = [
    "RAGAS_BACKEND_VERSION",
    "RAGAS_CONTEXT_PRECISION_SCORER_VERSION",
    "RAGAS_FAITHFULNESS_SCORER_VERSION",
    "RAGAS_PACKAGE_VERSION",
    "RAGAS_SCORING_PIPELINE_VERSION",
    "EvaluationRagasError",
    "NativeRagasMetricGateway",
    "RagasCaseSample",
    "RagasMetricGateway",
    "build_ragas_samples",
    "ragas_judge_identity",
    "score_evaluation_v2_with_ragas",
]
