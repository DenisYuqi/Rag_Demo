"""Deadline-bound evidence-first QA orchestration."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from rag_mvp.config.settings import Settings
from rag_mvp.domain.evaluation import (
    ModelAttempt as CostModelAttempt,
)
from rag_mvp.domain.evaluation import (
    ModelAttemptStatus as EvidenceAttemptStatus,
)
from rag_mvp.domain.evaluation import (
    ModelRole as EvidenceModelRole,
)
from rag_mvp.domain.evaluation import (
    ProviderAttemptEvidence,
)
from rag_mvp.domain.evaluation import (
    TokenUsage as EvidenceTokenUsage,
)
from rag_mvp.domain.qa import (
    Citation,
    ConversationRole,
    QAAnswer,
    QAError,
    QAErrorCode,
    QARefusal,
    QAResponse,
    RefusalReason,
    SafeQADiagnostics,
    SessionStatus,
)
from rag_mvp.domain.retrieval import (
    CachePolicy,
    RankingEvidence,
    RetrievalMode,
    RetrievalResult,
)
from rag_mvp.observability.costs import PricingCatalog
from rag_mvp.performance.worker_pools import RagWorkerPools
from rag_mvp.providers.errors import ProviderError, ProviderOperationError
from rag_mvp.providers.models import (
    AttemptStatus,
    Deadline,
    FinishReason,
    GenerationRequest,
    GenerationResult,
    ModelAttempt,
    ProviderCallContext,
    ProviderErrorCategory,
    RoutedResult,
)
from rag_mvp.providers.protocols import EmbeddingProvider, RerankingProvider
from rag_mvp.providers.resilience import capture_provider_attempts, current_provider_attempts
from rag_mvp.providers.routing import ModelProviderRouter
from rag_mvp.qa.citations import StructuredAnswerError, StructuredAnswerParser
from rag_mvp.qa.context import ContextBuilder, ContextSelectionError
from rag_mvp.qa.deadlines import DeadlineRunner, QAStageBudgets
from rag_mvp.qa.evidence_assessor import EvidenceAssessmentError, FactAssessmentResult
from rag_mvp.qa.grounding import (
    GroundingValidationError,
    GroundingValidator,
    ValidatedGroundedAnswer,
)
from rag_mvp.qa.prompt import GeneratorPromptBuilder, PromptBuildError
from rag_mvp.qa.query_rewrite import QueryRewriteError, QueryRewriter, select_response_language
from rag_mvp.qa.refusal import (
    EvidenceDecision,
    EvidenceDecisionCode,
    EvidenceDecisionKind,
    FactEvidence,
    RefusalPolicy,
    RefusalPolicyError,
)
from rag_mvp.qa.sessions import ConversationService
from rag_mvp.retrieval.binding import BoundRetrievalSnapshotFactory
from rag_mvp.retrieval.request import (
    RetrievalRequestContext,
    RetrievalRequestError,
    canonicalize_query,
)
from rag_mvp.retrieval.service import RetrievalService, RetrievalUnavailableError
from rag_mvp.safety.injection import InjectionPolicy
from rag_mvp.storage.repositories import RepositoryError

_OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$")
_ORCHESTRATION_PROOF = object()

_REFUSAL_MESSAGES = {
    "en": {
        RefusalReason.INSUFFICIENT_EVIDENCE: (
            "The available evidence does not support an answer to this request."
        ),
        RefusalReason.CONFLICTING_EVIDENCE: (
            "The available evidence conflicts, so no unsupported conclusion was selected."
        ),
        RefusalReason.UNSAFE_REQUEST: "This request cannot be completed safely.",
    },
    "zh-CN": {
        RefusalReason.INSUFFICIENT_EVIDENCE: "现有证据不足以支持对此请求的回答.",
        RefusalReason.CONFLICTING_EVIDENCE: "现有证据存在冲突, 因此未选择缺乏支持的结论.",
        RefusalReason.UNSAFE_REQUEST: "无法安全地完成此请求.",
    },
}
_ERROR_MESSAGES = {
    "en": {
        QAErrorCode.INDEX_NOT_READY: "The knowledge index is not ready.",
        QAErrorCode.RETRIEVAL_UNAVAILABLE: "Evidence retrieval is currently unavailable.",
        QAErrorCode.DEPENDENCY_FAILURE: "A required dependency could not complete the request.",
        QAErrorCode.DEADLINE_EXPIRED: "The request reached its deadline.",
        QAErrorCode.INTERNAL: "The request could not be completed safely.",
    },
    "zh-CN": {
        QAErrorCode.INDEX_NOT_READY: "知识索引尚未就绪.",
        QAErrorCode.RETRIEVAL_UNAVAILABLE: "当前无法检索证据.",
        QAErrorCode.DEPENDENCY_FAILURE: "必要依赖未能完成请求.",
        QAErrorCode.DEADLINE_EXPIRED: "请求已超过截止时间.",
        QAErrorCode.INTERNAL: "无法安全地完成请求.",
    },
}
_PARTIAL_MESSAGES = {
    "en": "Some requested information is not supported by the available evidence.",
    "zh-CN": "现有证据仅支持请求中的部分信息.",
}


class QARequestError(ValueError):
    """A stable precondition failure for the future HTTP adapter."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class QARetrievalGateway(Protocol):
    async def retrieve(
        self,
        *,
        request_id: str,
        query: str,
        mode: RetrievalMode,
        cache_policy: CachePolicy,
        deadline: Deadline,
    ) -> RetrievalResult: ...


class RoutedGenerationGateway(Protocol):
    async def generate(
        self,
        request: GenerationRequest,
        context: ProviderCallContext,
    ) -> RoutedResult[GenerationResult]: ...


class FactEvidenceAssessor(Protocol):
    async def assess(
        self,
        query: str,
        candidates: Sequence[RankingEvidence],
        *,
        request_id: str,
        revision_id: str,
        deadline: Deadline,
    ) -> tuple[FactEvidence, ...]: ...


@runtime_checkable
class _DiagnosticFactEvidenceAssessor(Protocol):
    async def assess_with_diagnostics(
        self,
        query: str,
        candidates: Sequence[RankingEvidence],
        *,
        request_id: str,
        revision_id: str,
        deadline: Deadline,
    ) -> FactAssessmentResult: ...


class _TokenUsageLike(Protocol):
    @property
    def input_tokens(self) -> int | None: ...

    @property
    def output_tokens(self) -> int | None: ...


class QAStageObserver(Protocol):
    def stage(self, stage: str) -> AbstractAsyncContextManager[None]: ...


@dataclass(frozen=True, slots=True)
class OrchestratedResponse:
    """Internal complete outcome plus the proof required by the release gate."""

    response: QAResponse = field(repr=False)
    grounded_answer: ValidatedGroundedAnswer | None = field(default=None, repr=False)
    application_suffix: str | None = field(default=None, repr=False)
    retrieved_chunk_ids: tuple[str, ...] = field(default=(), repr=False)
    context_chunk_ids: tuple[str, ...] = field(default=(), repr=False)
    _proof: object | None = field(default=None, repr=False, compare=False)

    @classmethod
    def _create(
        cls,
        response: QAResponse,
        *,
        grounded_answer: ValidatedGroundedAnswer | None = None,
        application_suffix: str | None = None,
        retrieved_chunk_ids: tuple[str, ...] = (),
        context_chunk_ids: tuple[str, ...] = (),
    ) -> OrchestratedResponse:
        return cls(
            response=response,
            grounded_answer=grounded_answer,
            application_suffix=application_suffix,
            retrieved_chunk_ids=retrieved_chunk_ids,
            context_chunk_ids=context_chunk_ids,
            _proof=_ORCHESTRATION_PROOF,
        )

    @property
    def trusted_pipeline_result(self) -> bool:
        return self._proof is _ORCHESTRATION_PROOF


class SnapshotRetrievalGateway:
    """Production retrieval composition with one bound snapshot per QA request."""

    def __init__(
        self,
        snapshots: BoundRetrievalSnapshotFactory,
        embedding: EmbeddingProvider | ModelProviderRouter,
        *,
        reranker: RerankingProvider | ModelProviderRouter | None = None,
        settings: Settings | None = None,
        worker_pools: RagWorkerPools | None = None,
    ) -> None:
        if worker_pools is not None and not isinstance(worker_pools, RagWorkerPools):
            raise TypeError("worker_pools must be RagWorkerPools")
        self._snapshots = snapshots
        self._embedding = embedding
        self._reranker = reranker
        self._settings = settings
        self._worker_pools = worker_pools

    async def retrieve(
        self,
        *,
        request_id: str,
        query: str,
        mode: RetrievalMode,
        cache_policy: CachePolicy,
        deadline: Deadline,
    ) -> RetrievalResult:
        async with self._snapshots.bind_async(
            self._worker_pools.chroma if self._worker_pools is not None else None
        ) as snapshot:
            request = RetrievalRequestContext.from_snapshot(
                request_id=request_id,
                query=query,
                mode=mode,
                snapshot=snapshot,
                cache_policy=cache_policy,
            )
            service = RetrievalService.from_snapshot(
                snapshot,
                self._embedding,
                ProviderCallContext(request_id, "qa-retrieval", deadline),
                reranker=self._reranker,
                settings=self._settings,
                worker_pools=self._worker_pools,
            )
            try:
                return await service.retrieve(request)
            finally:
                service.close()


class QAOrchestrator:
    """Compose validated QA stages under one absolute request deadline."""

    def __init__(
        self,
        *,
        conversations: ConversationService,
        retrieval: QARetrievalGateway,
        generation: RoutedGenerationGateway,
        fact_assessor: FactEvidenceAssessor,
        query_rewriter: QueryRewriter | None = None,
        context_builder: ContextBuilder | None = None,
        prompt_builder: GeneratorPromptBuilder | None = None,
        answer_parser: StructuredAnswerParser | None = None,
        grounding_validator: GroundingValidator | None = None,
        refusal_policy: RefusalPolicy | None = None,
        injection_policy: InjectionPolicy | None = None,
        budgets: QAStageBudgets | None = None,
        deadline_runner: DeadlineRunner | None = None,
        maximum_provider_attempts: int = 2,
        stage_observer: QAStageObserver | None = None,
        pricing_catalog: PricingCatalog | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(maximum_provider_attempts) is not int or maximum_provider_attempts < 1:
            raise ValueError("maximum_provider_attempts must be a positive integer")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._conversations = conversations
        self._retrieval = retrieval
        self._generation = generation
        self._fact_assessor = fact_assessor
        self._query_rewriter = query_rewriter or QueryRewriter()
        self._context_builder = context_builder or ContextBuilder()
        self._prompt_builder = prompt_builder or GeneratorPromptBuilder()
        self._answer_parser = answer_parser or StructuredAnswerParser()
        self._grounding_validator = grounding_validator or GroundingValidator()
        self._refusal_policy = refusal_policy or RefusalPolicy()
        self._injection_policy = injection_policy or InjectionPolicy()
        self._budgets = budgets or QAStageBudgets()
        self._deadline_runner = deadline_runner or DeadlineRunner()
        self._maximum_provider_attempts = maximum_provider_attempts
        self._stage_observer = stage_observer
        self._pricing_catalog = pricing_catalog
        self._clock = clock

    def set_stage_observer(self, observer: QAStageObserver | None) -> None:
        """Attach the process telemetry observer before request traffic starts."""

        self._stage_observer = observer

    async def answer(
        self,
        *,
        request_id: str,
        session_id: str,
        owner_id: str,
        question: str,
        mode: RetrievalMode | str,
        requested_language: str | None = None,
        cache_policy: CachePolicy | str = CachePolicy.USE,
    ) -> QAResponse:
        return (
            await self.run(
                request_id=request_id,
                session_id=session_id,
                owner_id=owner_id,
                question=question,
                mode=mode,
                requested_language=requested_language,
                cache_policy=cache_policy,
            )
        ).response

    async def run(
        self,
        *,
        request_id: str,
        session_id: str,
        owner_id: str,
        question: str,
        mode: RetrievalMode | str,
        requested_language: str | None = None,
        cache_policy: CachePolicy | str = CachePolicy.USE,
    ) -> OrchestratedResponse:
        with capture_provider_attempts():
            return await self._run_captured(
                request_id=request_id,
                session_id=session_id,
                owner_id=owner_id,
                question=question,
                mode=mode,
                requested_language=requested_language,
                cache_policy=cache_policy,
            )

    async def _run_captured(
        self,
        *,
        request_id: str,
        session_id: str,
        owner_id: str,
        question: str,
        mode: RetrievalMode | str,
        requested_language: str | None = None,
        cache_policy: CachePolicy | str = CachePolicy.USE,
    ) -> OrchestratedResponse:
        started = self._clock()
        root_deadline = Deadline(started + self._budgets.total_seconds, self._clock)
        timings: dict[str, float] = {}
        retrieval_result: RetrievalResult | None = None
        generation_result: RoutedResult[GenerationResult] | None = None
        fact_assessment: FactAssessmentResult | None = None
        decision: EvidenceDecision | None = None
        safety_reasons: tuple[str, ...] = ()
        failed_provider_attempts: tuple[ProviderAttemptEvidence, ...] = ()

        normalized_question, language, resolved_mode, resolved_cache = self._request_values(
            request_id,
            session_id,
            owner_id,
            question,
            mode,
            requested_language,
            cache_policy,
        )
        try:
            validation_deadline = self._child_deadline(
                root_deadline,
                self._budgets.validation_seconds,
            )
            validation_started = self._clock()
            try:
                async with self._observe_stage("validation"):
                    try:
                        session = self._conversations.get_session(session_id, owner_id)
                    except RepositoryError:
                        raise QARequestError("session_unavailable") from None
                    if session.status is not SessionStatus.ACTIVE:
                        raise QARequestError("session_unavailable")
                    injection = self._injection_policy.assess_user_input(normalized_question)
                    if injection.requires_refusal:
                        self._ensure_deadline(validation_deadline)
                        timings["validation"] = self._elapsed_ms(validation_started)
                        return OrchestratedResponse._create(
                            self._refusal(
                                request_id,
                                session_id,
                                language,
                                RefusalReason.UNSAFE_REQUEST,
                                (),
                                timings,
                                started,
                                metadata={
                                    "input_policy": injection.reason_code or "unsafe_request"
                                },
                            )
                        )
                    try:
                        current_turn = self._conversations.append_turn(
                            session_id,
                            owner_id,
                            ConversationRole.USER,
                            normalized_question,
                        )
                        history = tuple(
                            turn
                            for turn in self._conversations.list_turns(session_id, owner_id)
                            if turn.ordinal <= current_turn.ordinal
                        )
                    except RepositoryError:
                        raise QARequestError("session_unavailable") from None
                    rewrite = self._query_rewriter.prepare(
                        history,
                        requested_language=language,
                    )
                    self._ensure_deadline(validation_deadline)
            finally:
                timings["validation"] = self._elapsed_ms(validation_started)

            retrieval_budget = self._budgets.retrieval_seconds
            if resolved_mode is RetrievalMode.HYBRID_RERANK:
                retrieval_budget += self._budgets.rerank_seconds
            try:
                retrieval_result = await self._run_stage(
                    "retrieval",
                    retrieval_budget,
                    root_deadline,
                    timings,
                    lambda deadline: self._retrieval.retrieve(
                        request_id=request_id,
                        query=rewrite.query,
                        mode=resolved_mode,
                        cache_policy=resolved_cache,
                        deadline=deadline,
                    ),
                )
            except RetrievalRequestError as error:
                code = (
                    QAErrorCode.INDEX_NOT_READY
                    if error.code == "index_not_ready"
                    else QAErrorCode.RETRIEVAL_UNAVAILABLE
                )
                raise _PipelineFailure(
                    code,
                    retryable=code is QAErrorCode.INDEX_NOT_READY,
                ) from None
            except RetrievalUnavailableError as error:
                failed_provider_attempts += tuple(
                    _provider_attempt_evidence(attempt) for attempt in error.provider_attempts
                ) + tuple(
                    _unknown_provider_attempt_evidence(
                        role=EvidenceModelRole.EMBEDDING,
                        operation_id="qa-retrieval",
                        attempt_number=index + 1,
                    )
                    for index in range(error.unrecorded_provider_attempt_count)
                )
                raise _PipelineFailure(QAErrorCode.RETRIEVAL_UNAVAILABLE) from None
            except _DeadlineExpired:
                raise
            except Exception:
                raise _PipelineFailure(QAErrorCode.RETRIEVAL_UNAVAILABLE) from None
            self._validate_retrieval(retrieval_result, request_id, resolved_mode)

            if not retrieval_result.evidence:
                return OrchestratedResponse._create(
                    self._refusal(
                        request_id,
                        session_id,
                        language,
                        RefusalReason.INSUFFICIENT_EVIDENCE,
                        (),
                        timings,
                        started,
                        retrieval=retrieval_result,
                        metadata={
                            "decision_code": EvidenceDecisionCode.INSUFFICIENT_EVIDENCE.value
                        },
                    )
                )

            detected_retrieved_instructions: list[str] = []
            for evidence in retrieval_result.evidence:
                assessment = self._injection_policy.assess_retrieved_content(evidence.text)
                if assessment.matched_rules and assessment.reason_code is not None:
                    detected_retrieved_instructions.append(assessment.reason_code)
            safety_reasons = tuple(dict.fromkeys(detected_retrieved_instructions))
            try:
                fact_assessment = await self._run_stage(
                    "evidence_assessment",
                    self._budgets.evidence_assessment_seconds,
                    root_deadline,
                    timings,
                    lambda deadline: self._assess_facts(
                        rewrite.query,
                        retrieval_result.evidence,
                        request_id=request_id,
                        revision_id=retrieval_result.diagnostics.index_revision,
                        deadline=deadline,
                    ),
                )
                facts = fact_assessment.facts
            except EvidenceAssessmentError as error:
                failed_provider_attempts += tuple(
                    _provider_attempt_evidence(attempt) for attempt in error.provider_attempts
                ) + tuple(
                    _unknown_provider_attempt_evidence(
                        role=EvidenceModelRole.EMBEDDING,
                        operation_id="fact-evidence-assessment",
                        attempt_number=index + 1,
                    )
                    for index in range(error.unrecorded_provider_attempt_count)
                )
                raise _PipelineFailure(QAErrorCode.DEPENDENCY_FAILURE) from None
            except _DeadlineExpired:
                raise
            except Exception:
                raise _PipelineFailure(QAErrorCode.DEPENDENCY_FAILURE) from None
            try:
                decision = self._refusal_policy.decide(
                    facts,
                    candidates=retrieval_result.evidence,
                    revision_id=retrieval_result.diagnostics.index_revision,
                )
            except RefusalPolicyError:
                raise _PipelineFailure(QAErrorCode.INTERNAL) from None

            if decision.requires_refusal:
                reason = decision.reason or RefusalReason.INSUFFICIENT_EVIDENCE
                citations = self._resolve_citations(
                    decision.citation_chunk_ids,
                    retrieval_result.evidence,
                )
                return OrchestratedResponse._create(
                    self._refusal(
                        request_id,
                        session_id,
                        language,
                        reason,
                        citations,
                        timings,
                        started,
                        retrieval=retrieval_result,
                        fact_assessment=fact_assessment,
                        decision=decision,
                        extra_degradation=safety_reasons,
                    ),
                    retrieved_chunk_ids=tuple(
                        evidence.chunk_id for evidence in retrieval_result.evidence
                    ),
                    context_chunk_ids=tuple(
                        evidence.chunk_id for evidence in retrieval_result.evidence
                    ),
                )

            approved_evidence = self._approved_evidence(
                retrieval_result.evidence,
                decision.citation_chunk_ids,
            )
            try:
                context = self._context_builder.build(approved_evidence)
                generation_request = self._prompt_builder.build(
                    question=normalized_question,
                    retrieval_query=rewrite.query,
                    response_language=language,
                    context=context,
                )
            except (ContextSelectionError, PromptBuildError, TypeError, ValueError):
                raise _PipelineFailure(QAErrorCode.INTERNAL) from None

            try:
                generation_result = await self._run_stage(
                    "generation",
                    self._budgets.generation_seconds,
                    root_deadline,
                    timings,
                    lambda deadline: self._generation.generate(
                        generation_request,
                        ProviderCallContext(request_id, "qa-generation", deadline),
                    ),
                    reserve_seconds=self._budgets.finalization_seconds,
                    require_full_budget=True,
                )
            except ProviderOperationError as error:
                failed_provider_attempts += tuple(
                    _provider_attempt_evidence(attempt) for attempt in error.attempts
                )
                raise _PipelineFailure(
                    QAErrorCode.DEPENDENCY_FAILURE,
                    retryable=error.retryable,
                ) from None
            except ProviderError as error:
                failed_provider_attempts += (
                    _unknown_provider_attempt_evidence(
                        role=EvidenceModelRole.GENERATION,
                        operation_id="qa-generation",
                    ),
                )
                raise _PipelineFailure(
                    QAErrorCode.DEPENDENCY_FAILURE,
                    retryable=error.retryable,
                ) from None
            except _DeadlineExpired:
                raise
            except Exception:
                raise _PipelineFailure(QAErrorCode.DEPENDENCY_FAILURE) from None
            generated = self._validated_generation(generation_result)

            finalization_deadline = self._child_deadline(
                root_deadline,
                self._budgets.finalization_seconds,
            )
            finalization_started = self._clock()
            try:
                async with self._observe_stage("finalization"):
                    parsed = self._answer_parser.parse(
                        generated.content,
                        context=context,
                        expected_revision_id=retrieval_result.diagnostics.index_revision,
                    )
                    grounded = self._grounding_validator.validate(
                        parsed,
                        request_id=request_id,
                        revision_id=retrieval_result.diagnostics.index_revision,
                        candidates=retrieval_result.evidence,
                    )
                    answer_text = grounded.answer
                    application_suffix = None
                    if decision.kind is EvidenceDecisionKind.PARTIAL:
                        application_suffix = _PARTIAL_MESSAGES[language]
                        answer_text = f"{answer_text.rstrip()}\n\n{application_suffix}"
                    self._ensure_deadline(finalization_deadline)
            except (StructuredAnswerError, GroundingValidationError) as error:
                raise _PipelineFailure(
                    QAErrorCode.DEPENDENCY_FAILURE,
                    detail_code=error.code,
                ) from None
            finally:
                timings["finalization"] = self._elapsed_ms(finalization_started)
            self._ensure_deadline(root_deadline)
            return OrchestratedResponse._create(
                QAAnswer(
                    request_id=request_id,
                    session_id=session_id,
                    response_language=language,
                    answer=answer_text,
                    claims=grounded.claims,
                    citations=grounded.citations,
                    diagnostics=self._diagnostics(
                        timings,
                        started,
                        request_id=request_id,
                        retrieval=retrieval_result,
                        fact_assessment=fact_assessment,
                        generation=generation_result,
                        decision=decision,
                        extra_degradation=safety_reasons,
                    ),
                ),
                grounded_answer=grounded,
                application_suffix=application_suffix,
                retrieved_chunk_ids=tuple(
                    evidence.chunk_id for evidence in retrieval_result.evidence
                ),
                context_chunk_ids=tuple(chunk.chunk_id for chunk in context.chunks),
            )
        except asyncio.CancelledError:
            raise
        except QARequestError:
            raise
        except _DeadlineExpired:
            return OrchestratedResponse._create(
                self._error(
                    request_id,
                    session_id,
                    language,
                    QAErrorCode.DEADLINE_EXPIRED,
                    True,
                    timings,
                    started,
                    retrieval_result,
                    generation_result,
                    decision,
                    safety_reasons,
                    fact_assessment=fact_assessment,
                    failed_provider_attempts=failed_provider_attempts,
                )
            )
        except _PipelineFailure as error:
            return OrchestratedResponse._create(
                self._error(
                    request_id,
                    session_id,
                    language,
                    error.code,
                    error.retryable,
                    timings,
                    started,
                    retrieval_result,
                    generation_result,
                    decision,
                    safety_reasons,
                    failure_detail=error.detail_code,
                    fact_assessment=fact_assessment,
                    failed_provider_attempts=failed_provider_attempts,
                )
            )
        except (QueryRewriteError, TypeError, ValueError):
            return OrchestratedResponse._create(
                self._error(
                    request_id,
                    session_id,
                    language,
                    QAErrorCode.INTERNAL,
                    False,
                    timings,
                    started,
                    retrieval_result,
                    generation_result,
                    decision,
                    safety_reasons,
                    fact_assessment=fact_assessment,
                    failed_provider_attempts=failed_provider_attempts,
                )
            )
        except Exception:
            return OrchestratedResponse._create(
                self._error(
                    request_id,
                    session_id,
                    language,
                    QAErrorCode.INTERNAL,
                    False,
                    timings,
                    started,
                    retrieval_result,
                    generation_result,
                    decision,
                    safety_reasons,
                    fact_assessment=fact_assessment,
                    failed_provider_attempts=failed_provider_attempts,
                )
            )

    def _request_values(
        self,
        request_id: str,
        session_id: str,
        owner_id: str,
        question: str,
        mode: RetrievalMode | str,
        requested_language: str | None,
        cache_policy: CachePolicy | str,
    ) -> tuple[str, str, RetrievalMode, CachePolicy]:
        if any(
            not isinstance(value, str) or _OPAQUE_ID.fullmatch(value) is None
            for value in (request_id, session_id, owner_id)
        ):
            raise QARequestError("request_identity_invalid")
        try:
            normalized_question = canonicalize_query(question)
            language = select_response_language(
                normalized_question,
                requested_language=requested_language,
            )
            resolved_mode = RetrievalMode(mode)
            resolved_cache = CachePolicy(cache_policy)
        except (RetrievalRequestError, QueryRewriteError, TypeError, ValueError):
            raise QARequestError("request_invalid") from None
        return normalized_question, language, resolved_mode, resolved_cache

    async def _assess_facts(
        self,
        query: str,
        candidates: Sequence[RankingEvidence],
        *,
        request_id: str,
        revision_id: str,
        deadline: Deadline,
    ) -> FactAssessmentResult:
        if isinstance(self._fact_assessor, _DiagnosticFactEvidenceAssessor):
            result = await self._fact_assessor.assess_with_diagnostics(
                query,
                candidates,
                request_id=request_id,
                revision_id=revision_id,
                deadline=deadline,
            )
            if not isinstance(result, FactAssessmentResult):
                raise TypeError("fact assessment result is invalid")
            return result
        facts = await self._fact_assessor.assess(
            query,
            candidates,
            request_id=request_id,
            revision_id=revision_id,
            deadline=deadline,
        )
        return FactAssessmentResult(tuple(facts))

    async def _run_stage[T](
        self,
        name: str,
        budget_seconds: float,
        root_deadline: Deadline,
        timings: dict[str, float],
        operation: Callable[[Deadline], Awaitable[T]],
        *,
        reserve_seconds: float = 0,
        require_full_budget: bool = False,
    ) -> T:
        deadline = self._child_deadline(
            root_deadline,
            budget_seconds,
            reserve_seconds,
            require_full_budget=require_full_budget,
        )
        started = self._clock()
        try:
            async with self._observe_stage(name):
                return await self._deadline_runner.run(
                    lambda: operation(deadline),
                    deadline=deadline,
                )
        except TimeoutError:
            raise _DeadlineExpired from None
        finally:
            timings[name] = self._elapsed_ms(started)

    @asynccontextmanager
    async def _observe_stage(self, name: str) -> AsyncIterator[None]:
        if self._stage_observer is None:
            yield
            return
        async with self._stage_observer.stage(name):
            yield

    def _child_deadline(
        self,
        root: Deadline,
        budget_seconds: float,
        reserve_seconds: float = 0,
        *,
        require_full_budget: bool = False,
    ) -> Deadline:
        current = self._clock()
        if require_full_budget and root.expires_at - current < budget_seconds + reserve_seconds:
            raise _DeadlineExpired
        expires_at = min(current + budget_seconds, root.expires_at - reserve_seconds)
        if expires_at <= current:
            raise _DeadlineExpired
        return Deadline(expires_at, self._clock)

    @staticmethod
    def _ensure_deadline(deadline: Deadline) -> None:
        if deadline.expired:
            raise _DeadlineExpired

    def _elapsed_ms(self, started: float) -> float:
        return max(0.0, (self._clock() - started) * 1000)

    def _validate_retrieval(
        self,
        result: RetrievalResult,
        request_id: str,
        requested_mode: RetrievalMode,
    ) -> None:
        if not isinstance(result, RetrievalResult):
            raise _PipelineFailure(QAErrorCode.RETRIEVAL_UNAVAILABLE)
        diagnostics = result.diagnostics
        allowed_modes = {
            RetrievalMode.DENSE: {RetrievalMode.DENSE},
            RetrievalMode.HYBRID: {RetrievalMode.HYBRID},
            RetrievalMode.HYBRID_RERANK: {
                RetrievalMode.HYBRID,
                RetrievalMode.HYBRID_RERANK,
            },
        }[requested_mode]
        ranks = tuple(evidence.final_rank for evidence in result.evidence)
        if (
            diagnostics.request_id != request_id
            or diagnostics.requested_mode is not requested_mode
            or diagnostics.effective_mode not in allowed_modes
            or ranks != tuple(range(1, len(result.evidence) + 1))
            or any(
                evidence.revision_id != diagnostics.index_revision for evidence in result.evidence
            )
            or any(
                count > self._maximum_provider_attempts
                for count in diagnostics.provider_attempt_counts.values()
            )
        ):
            raise _PipelineFailure(QAErrorCode.RETRIEVAL_UNAVAILABLE)

    def _validated_generation(
        self,
        routed: RoutedResult[GenerationResult],
    ) -> GenerationResult:
        if (
            not isinstance(routed, RoutedResult)
            or not isinstance(routed.value, GenerationResult)
            or len(routed.attempts) > self._maximum_provider_attempts
            or routed.value.finish_reason is not FinishReason.STOP
        ):
            raise _PipelineFailure(QAErrorCode.DEPENDENCY_FAILURE)
        return routed.value

    @staticmethod
    def _approved_evidence(
        evidence: Sequence[RankingEvidence],
        approved_ids: Sequence[str],
    ) -> tuple[RankingEvidence, ...]:
        approved = set(approved_ids)
        selected = tuple(item for item in evidence if item.chunk_id in approved)
        if not approved or len(selected) != len(approved):
            raise _PipelineFailure(QAErrorCode.INTERNAL)
        return tuple(
            item.model_copy(update={"final_rank": rank})
            for rank, item in enumerate(selected, start=1)
        )

    @staticmethod
    def _resolve_citations(
        chunk_ids: Sequence[str],
        evidence: Sequence[RankingEvidence],
    ) -> tuple[Citation, ...]:
        registry = {item.chunk_id: item for item in evidence}
        try:
            return tuple(
                Citation(
                    source_title=registry[chunk_id].display_title,
                    document_version=registry[chunk_id].document_version,
                    chunk_id=chunk_id,
                    locator=registry[chunk_id].locator,
                )
                for chunk_id in chunk_ids
            )
        except (KeyError, TypeError, ValueError):
            raise _PipelineFailure(QAErrorCode.INTERNAL) from None

    def _refusal(
        self,
        request_id: str,
        session_id: str,
        language: str,
        reason: RefusalReason,
        citations: Sequence[Citation],
        timings: dict[str, float],
        started: float,
        *,
        retrieval: RetrievalResult | None = None,
        fact_assessment: FactAssessmentResult | None = None,
        decision: EvidenceDecision | None = None,
        extra_degradation: Sequence[str] = (),
        metadata: dict[str, str | int | float | bool | None] | None = None,
    ) -> QARefusal:
        return QARefusal(
            request_id=request_id,
            session_id=session_id,
            response_language=language,
            message=_REFUSAL_MESSAGES[language][reason],
            reason=reason,
            citations=tuple(citations),
            diagnostics=self._diagnostics(
                timings,
                started,
                request_id=request_id,
                retrieval=retrieval,
                fact_assessment=fact_assessment,
                decision=decision,
                extra_degradation=extra_degradation,
                extra_metadata=metadata,
            ),
        )

    def _error(
        self,
        request_id: str,
        session_id: str,
        language: str,
        code: QAErrorCode,
        retryable: bool,
        timings: dict[str, float],
        started: float,
        retrieval: RetrievalResult | None,
        generation: RoutedResult[GenerationResult] | None,
        decision: EvidenceDecision | None,
        extra_degradation: Sequence[str],
        *,
        failure_detail: str | None = None,
        fact_assessment: FactAssessmentResult | None = None,
        failed_provider_attempts: Sequence[ProviderAttemptEvidence] = (),
    ) -> QAError:
        return QAError(
            request_id=request_id,
            session_id=session_id,
            response_language=language,
            message=_ERROR_MESSAGES[language].get(
                code,
                _ERROR_MESSAGES[language][QAErrorCode.INTERNAL],
            ),
            code=code,
            retryable=retryable,
            diagnostics=self._diagnostics(
                timings,
                started,
                request_id=request_id,
                retrieval=retrieval,
                fact_assessment=fact_assessment,
                generation=generation,
                decision=decision,
                extra_degradation=extra_degradation,
                extra_metadata=(
                    {"failure_detail_code": failure_detail} if failure_detail is not None else None
                ),
                failed_provider_attempts=failed_provider_attempts,
            ),
        )

    def _diagnostics(
        self,
        timings: dict[str, float],
        started: float,
        *,
        request_id: str,
        retrieval: RetrievalResult | None = None,
        fact_assessment: FactAssessmentResult | None = None,
        generation: RoutedResult[GenerationResult] | None = None,
        decision: EvidenceDecision | None = None,
        extra_degradation: Sequence[str] = (),
        extra_metadata: dict[str, str | int | float | bool | None] | None = None,
        failed_provider_attempts: Sequence[ProviderAttemptEvidence] = (),
    ) -> SafeQADiagnostics:
        stage_timings = {**timings, "total": self._elapsed_ms(started)}
        cache_status: dict[str, str] = {}
        model_identities: dict[str, str] = {}
        token_counts: dict[str, int] = {}
        degradation: list[str] = []
        provider_attempts: list[ProviderAttemptEvidence] = [
            _provider_attempt_evidence(attempt) for attempt in current_provider_attempts()
        ]
        metadata: dict[str, str | int | float | bool | None] = dict(extra_metadata or {})
        provider_attempt_count = 0
        provider_failed_attempt_count = 0
        provider_unknown_usage_attempt_count = 0
        if retrieval is not None:
            diagnostics = retrieval.diagnostics
            provider_attempt_count += sum(diagnostics.provider_attempt_counts.values())
            provider_failed_attempt_count += sum(
                diagnostics.provider_failed_attempt_counts.values()
            )
            provider_unknown_usage_attempt_count += sum(
                diagnostics.provider_unknown_usage_attempt_counts.values()
            )
            provider_attempts.extend(diagnostics.provider_attempts)
            for diagnostic_role, declared_count in diagnostics.provider_attempt_counts.items():
                evidence_role = (
                    EvidenceModelRole.RERANKING
                    if diagnostic_role == "reranker"
                    else EvidenceModelRole(diagnostic_role)
                )
                recorded = tuple(
                    attempt
                    for attempt in provider_attempts
                    if attempt.operation_id == "qa-retrieval" and attempt.role is evidence_role
                )
                missing_count = max(0, declared_count - len(recorded))
                recorded_failures = sum(
                    attempt.status is not EvidenceAttemptStatus.SUCCEEDED for attempt in recorded
                )
                missing_failures = max(
                    0,
                    diagnostics.provider_failed_attempt_counts.get(diagnostic_role, 0)
                    - recorded_failures,
                )
                provider_attempts.extend(
                    _unknown_provider_attempt_evidence(
                        role=evidence_role,
                        operation_id="qa-retrieval",
                        attempt_number=len(recorded) + index + 1,
                        status=(
                            EvidenceAttemptStatus.FAILED
                            if index < missing_failures
                            else EvidenceAttemptStatus.SUCCEEDED
                        ),
                    )
                    for index in range(missing_count)
                )
            for stage, duration_ms in diagnostics.stage_timings_ms.items():
                if stage != "total":
                    stage_timings.setdefault(stage, duration_ms)
            cache_status = {
                name: outcome.value for name, outcome in diagnostics.cache_status.items()
            }
            model_identities.update(diagnostics.provider_identities)
            for role, usage in diagnostics.provider_usage.items():
                _merge_token_counts(token_counts, role, usage)
            degradation.extend(diagnostics.degradation_reasons)
            metadata.update(
                {
                    "index_revision": diagnostics.index_revision,
                    "requested_mode": diagnostics.requested_mode.value,
                    "effective_mode": diagnostics.effective_mode.value,
                    "dense_candidate_count": diagnostics.candidate_counts.get("dense", 0),
                    "lexical_candidate_count": diagnostics.candidate_counts.get("bm25", 0),
                    "fused_candidate_count": diagnostics.candidate_counts.get("fused", 0),
                    "reranked_candidate_count": diagnostics.candidate_counts.get("reranked", 0),
                    "candidate_count": diagnostics.candidate_counts.get(
                        "final", len(retrieval.evidence)
                    ),
                }
            )
        if fact_assessment is not None:
            provider_attempt_count += fact_assessment.provider_attempt_count
            provider_failed_attempt_count += fact_assessment.provider_failed_attempt_count
            provider_unknown_usage_attempt_count += (
                fact_assessment.provider_unknown_usage_attempt_count
            )
            for attempt in fact_assessment.provider_attempts:
                _merge_token_counts(token_counts, attempt.role.value, attempt.usage)
                provider_attempts.append(_provider_attempt_evidence(attempt))
            if fact_assessment.direct_provider_usage is not None:
                _merge_token_counts(
                    token_counts,
                    "embedding",
                    fact_assessment.direct_provider_usage,
                )
                assert fact_assessment.direct_provider_identity is not None
                provider_attempts.append(
                    _direct_provider_attempt_evidence(
                        operation_id="fact-evidence-assessment",
                        role=EvidenceModelRole.EMBEDDING,
                        provider=fact_assessment.direct_provider_identity.provider,
                        model=fact_assessment.direct_provider_identity.model,
                        usage=fact_assessment.direct_provider_usage,
                    )
                )
        if generation is not None and isinstance(generation.value, GenerationResult):
            identity = generation.value.identity
            model_identities["generation"] = (
                f"{identity.provider}/{identity.model}/{identity.adapter_version}"
            )
            generation_attempt_count = len(generation.attempts) or 1
            metadata["generation_attempts"] = generation_attempt_count
            metadata["generation_fallback"] = generation.used_fallback
            provider_attempt_count += generation_attempt_count
            provider_failed_attempt_count += sum(
                attempt.status is not AttemptStatus.SUCCEEDED for attempt in generation.attempts
            )
            if generation.attempts:
                for attempt in generation.attempts:
                    _merge_token_counts(token_counts, "generation", attempt.usage)
                    provider_attempts.append(_provider_attempt_evidence(attempt))
                provider_unknown_usage_attempt_count += sum(
                    _attempt_usage_unknown(attempt) for attempt in generation.attempts
                )
            else:
                _merge_token_counts(token_counts, "generation", generation.value.usage)
                provider_unknown_usage_attempt_count += int(
                    _usage_unknown("generation", generation.value.usage)
                )
                provider_attempts.append(
                    _direct_provider_attempt_evidence(
                        operation_id="qa-generation",
                        role=EvidenceModelRole.GENERATION,
                        provider=identity.provider,
                        model=identity.model,
                        usage=generation.value.usage,
                    )
                )
        provider_attempt_count += len(failed_provider_attempts)
        provider_failed_attempt_count += sum(
            attempt.status is not EvidenceAttemptStatus.SUCCEEDED
            for attempt in failed_provider_attempts
        )
        provider_unknown_usage_attempt_count += sum(
            _attempt_usage_unknown(attempt) for attempt in failed_provider_attempts
        )
        for failed_attempt in failed_provider_attempts:
            _merge_token_counts(
                token_counts,
                failed_attempt.role.value,
                failed_attempt.usage,
            )
        provider_attempts.extend(failed_provider_attempts)
        complete_provider_attempts = _deduplicated_provider_attempts(provider_attempts)
        if complete_provider_attempts:
            provider_attempt_count = len(complete_provider_attempts)
            provider_failed_attempt_count = sum(
                attempt.status is not EvidenceAttemptStatus.SUCCEEDED
                for attempt in complete_provider_attempts
            )
            provider_unknown_usage_attempt_count = sum(
                _attempt_usage_unknown(attempt) for attempt in complete_provider_attempts
            )
            token_counts = {}
            for complete_attempt in complete_provider_attempts:
                _merge_token_counts(
                    token_counts,
                    complete_attempt.role.value,
                    complete_attempt.usage,
                )
        metadata["provider_attempt_count"] = provider_attempt_count
        metadata["provider_failed_attempt_count"] = provider_failed_attempt_count
        metadata["provider_unknown_usage_attempt_count"] = provider_unknown_usage_attempt_count
        metadata.update(
            self._provider_cost_metadata(
                request_id=request_id,
                attempts=complete_provider_attempts,
            )
        )
        if decision is not None:
            metadata["decision_code"] = decision.code.value
            metadata["refusal_policy_version"] = decision.policy_version
            metadata["context_count"] = len(decision.citation_chunk_ids)
        degradation.extend(extra_degradation)
        return SafeQADiagnostics(
            stage_timings_ms=stage_timings,
            cache_status=cache_status,
            model_identities=model_identities,
            token_counts=token_counts,
            provider_attempts=complete_provider_attempts,
            degradation_reasons=tuple(dict.fromkeys(degradation)),
            metadata=metadata,
        )

    def _provider_cost_metadata(
        self,
        *,
        request_id: str,
        attempts: Sequence[ProviderAttemptEvidence],
    ) -> dict[str, str | int | float | bool | None]:
        if self._pricing_catalog is None:
            return {
                "pricing_version": "unconfigured",
                "currency": None,
                "estimated_cost": None,
                "cost_complete": False,
                "cost_unknown_reasons": "pricing-not-configured",
            }
        cost_attempts = tuple(
            CostModelAttempt(
                attempt_id=f"qa-cost-attempt-{index + 1}",
                operation_id=attempt.operation_id,
                request_id=request_id,
                role=attempt.role,
                provider=attempt.provider,
                model=attempt.model,
                status=attempt.status,
                attempt_number=attempt.attempt_number,
                fallback=attempt.fallback,
                latency_ms=attempt.latency_ms or 0,
                usage=attempt.usage,
                safe_error_category=attempt.safe_error_category,
            )
            for index, attempt in enumerate(attempts)
        )
        aggregate = self._pricing_catalog.aggregate_request(
            cost_attempts,
            request_id=request_id,
        )
        return {
            "pricing_version": aggregate.pricing_version,
            "currency": aggregate.currency,
            "known_cost": (
                format(aggregate.known_cost, "f") if aggregate.known_cost is not None else None
            ),
            "estimated_cost": (
                format(aggregate.estimated_cost, "f")
                if aggregate.estimated_cost is not None
                else None
            ),
            "cost_complete": aggregate.complete,
            "cost_unknown_reasons": ",".join(reason.value for reason in aggregate.unknown_reasons),
        }


class _DeadlineExpired(TimeoutError):
    pass


class _PipelineFailure(RuntimeError):
    def __init__(
        self,
        code: QAErrorCode,
        *,
        retryable: bool = False,
        detail_code: str | None = None,
    ) -> None:
        self.code = code
        self.retryable = retryable
        self.detail_code = detail_code
        super().__init__(code.value)


def _merge_token_counts(
    target: dict[str, int],
    role: str,
    usage: _TokenUsageLike,
) -> None:
    normalized_role = "reranking" if role == "reranker" else role
    if usage.input_tokens is not None:
        target[f"{normalized_role}-input"] = (
            target.get(f"{normalized_role}-input", 0) + usage.input_tokens
        )
    if usage.output_tokens is not None:
        target[f"{normalized_role}-output"] = (
            target.get(f"{normalized_role}-output", 0) + usage.output_tokens
        )


def _attempt_usage_unknown(
    attempt: ModelAttempt | ProviderAttemptEvidence,
) -> bool:
    return _usage_unknown(attempt.role.value, attempt.usage)


def _usage_unknown(role: str, usage: _TokenUsageLike) -> bool:
    if usage.input_tokens is None:
        return True
    return role != "embedding" and usage.output_tokens is None


def _provider_attempt_evidence(attempt: ModelAttempt) -> ProviderAttemptEvidence:
    return ProviderAttemptEvidence(
        operation_id=attempt.operation_id,
        attempt_number=attempt.attempt_number,
        route_id=attempt.route_id,
        role=EvidenceModelRole(attempt.role.value),
        provider=attempt.provider,
        model=attempt.model,
        status=_evidence_attempt_status(attempt),
        fallback=attempt.is_fallback,
        latency_ms=attempt.latency_ms,
        safe_error_category=(
            attempt.error_category.value if attempt.error_category is not None else None
        ),
        usage=EvidenceTokenUsage(
            input_tokens=attempt.usage.input_tokens,
            output_tokens=attempt.usage.output_tokens,
        ),
    )


def _evidence_attempt_status(attempt: ModelAttempt) -> EvidenceAttemptStatus:
    if attempt.status is AttemptStatus.FAILED and attempt.error_category in {
        ProviderErrorCategory.TIMEOUT,
        ProviderErrorCategory.DEADLINE_EXCEEDED,
    }:
        return EvidenceAttemptStatus.TIMED_OUT
    return EvidenceAttemptStatus(attempt.status.value)


def _direct_provider_attempt_evidence(
    *,
    operation_id: str,
    role: EvidenceModelRole,
    provider: str,
    model: str,
    usage: _TokenUsageLike,
) -> ProviderAttemptEvidence:
    return ProviderAttemptEvidence(
        operation_id=operation_id,
        role=role,
        provider=provider,
        model=model,
        status=EvidenceAttemptStatus.SUCCEEDED,
        usage=EvidenceTokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        ),
    )


def _unknown_provider_attempt_evidence(
    *,
    role: EvidenceModelRole,
    operation_id: str,
    attempt_number: int = 1,
    status: EvidenceAttemptStatus = EvidenceAttemptStatus.FAILED,
) -> ProviderAttemptEvidence:
    return ProviderAttemptEvidence(
        operation_id=operation_id,
        attempt_number=attempt_number,
        role=role,
        provider="unknown",
        model="unknown",
        status=status,
        usage=EvidenceTokenUsage(),
    )


def _deduplicated_provider_attempts(
    attempts: Sequence[ProviderAttemptEvidence],
) -> tuple[ProviderAttemptEvidence, ...]:
    unique: list[ProviderAttemptEvidence] = []
    seen: set[tuple[object, ...]] = set()
    for attempt in attempts:
        key = (
            attempt.operation_id,
            attempt.attempt_number,
            attempt.route_id,
            attempt.role.value,
            attempt.provider,
            attempt.model,
            attempt.status.value,
            attempt.fallback,
            attempt.latency_ms,
            attempt.safe_error_category,
            attempt.usage.input_tokens,
            attempt.usage.output_tokens,
        )
        if key not in seen:
            seen.add(key)
            unique.append(attempt)
    return tuple(unique)
