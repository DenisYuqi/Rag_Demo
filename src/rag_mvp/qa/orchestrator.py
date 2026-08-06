"""Deadline-bound evidence-first QA orchestration."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Protocol

from rag_mvp.config.settings import Settings
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
from rag_mvp.performance.worker_pools import RagWorkerPools
from rag_mvp.providers.errors import ProviderError, ProviderOperationError
from rag_mvp.providers.models import (
    Deadline,
    FinishReason,
    GenerationRequest,
    GenerationResult,
    ProviderCallContext,
    RoutedResult,
)
from rag_mvp.providers.protocols import EmbeddingProvider, RerankingProvider
from rag_mvp.providers.routing import ModelProviderRouter
from rag_mvp.qa.citations import StructuredAnswerError, StructuredAnswerParser
from rag_mvp.qa.context import ContextBuilder, ContextSelectionError
from rag_mvp.qa.deadlines import DeadlineRunner, QAStageBudgets
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
        started = self._clock()
        root_deadline = Deadline(started + self._budgets.total_seconds, self._clock)
        timings: dict[str, float] = {}
        retrieval_result: RetrievalResult | None = None
        generation_result: RoutedResult[GenerationResult] | None = None
        decision: EvidenceDecision | None = None
        safety_reasons: tuple[str, ...] = ()

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
            except RetrievalUnavailableError:
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
                facts = await self._run_stage(
                    "evidence_assessment",
                    self._budgets.evidence_assessment_seconds,
                    root_deadline,
                    timings,
                    lambda deadline: self._fact_assessor.assess(
                        rewrite.query,
                        retrieval_result.evidence,
                        request_id=request_id,
                        revision_id=retrieval_result.diagnostics.index_revision,
                        deadline=deadline,
                    ),
                )
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
                raise _PipelineFailure(
                    QAErrorCode.DEPENDENCY_FAILURE,
                    retryable=error.retryable,
                ) from None
            except ProviderError as error:
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
                        retrieval=retrieval_result,
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
                retrieval=retrieval,
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
                retrieval=retrieval,
                generation=generation,
                decision=decision,
                extra_degradation=extra_degradation,
                extra_metadata=(
                    {"failure_detail_code": failure_detail}
                    if failure_detail is not None
                    else None
                ),
            ),
        )

    def _diagnostics(
        self,
        timings: dict[str, float],
        started: float,
        *,
        retrieval: RetrievalResult | None = None,
        generation: RoutedResult[GenerationResult] | None = None,
        decision: EvidenceDecision | None = None,
        extra_degradation: Sequence[str] = (),
        extra_metadata: dict[str, str | int | float | bool | None] | None = None,
    ) -> SafeQADiagnostics:
        stage_timings = {**timings, "total": self._elapsed_ms(started)}
        cache_status: dict[str, str] = {}
        model_identities: dict[str, str] = {}
        token_counts: dict[str, int] = {}
        degradation: list[str] = []
        metadata: dict[str, str | int | float | bool | None] = dict(extra_metadata or {})
        if retrieval is not None:
            diagnostics = retrieval.diagnostics
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
        if generation is not None and isinstance(generation.value, GenerationResult):
            identity = generation.value.identity
            model_identities["generation"] = (
                f"{identity.provider}/{identity.model}/{identity.adapter_version}"
            )
            metadata["generation_attempts"] = len(generation.attempts)
            metadata["generation_fallback"] = generation.used_fallback
            if generation.attempts:
                for attempt in generation.attempts:
                    _merge_token_counts(token_counts, "generation", attempt.usage)
            else:
                _merge_token_counts(token_counts, "generation", generation.value.usage)
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
            degradation_reasons=tuple(dict.fromkeys(degradation)),
            metadata=metadata,
        )


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
