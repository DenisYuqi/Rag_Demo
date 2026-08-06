"""Executable OpenAI-compatible ingestion and QA service composition."""

from __future__ import annotations

import asyncio
import hashlib
from contextlib import suppress
from dataclasses import dataclass

from openai import AsyncOpenAI

from rag_mvp.api.qa import QARuntimeServices
from rag_mvp.config.settings import Settings
from rag_mvp.ingestion.chunking import ChunkingConfig
from rag_mvp.ingestion.extractors import OcrAdapter
from rag_mvp.ingestion.service import IngestionService
from rag_mvp.observability.diagnostics import SafeRequestDiagnosticStore
from rag_mvp.performance.deadlines import QALatencyBudgets
from rag_mvp.performance.worker_pools import RagWorkerPools
from rag_mvp.providers.models import (
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingSpaceIdentity,
    ModelIdentity,
    NormalizationPolicy,
    ProviderCallContext,
)
from rag_mvp.providers.openai_adapters import (
    OpenAIChatGenerationProvider,
    OpenAIEmbeddingProvider,
    OpenAIListwiseRerankingProvider,
)
from rag_mvp.providers.openai_client import OpenAIClientConfig, create_async_openai_client
from rag_mvp.providers.persistence import PersistentAttemptRecorder
from rag_mvp.providers.resilience import RetryPolicy
from rag_mvp.providers.routing import (
    EmbeddingRoute,
    GenerationRoute,
    ModelProviderRouter,
    ProviderRoute,
    RerankingRoute,
)
from rag_mvp.qa.context import ContextBuilder
from rag_mvp.qa.deadlines import QAStageBudgets
from rag_mvp.qa.evidence_assessor import SemanticFactEvidenceAssessor
from rag_mvp.qa.orchestrator import QAOrchestrator, SnapshotRetrievalGateway
from rag_mvp.qa.sessions import ConversationService
from rag_mvp.qa.streaming import CompleteResponseEmitter
from rag_mvp.retrieval.binding import BoundRetrievalSnapshotFactory
from rag_mvp.retrieval.identity import provider_embedding_identity
from rag_mvp.retrieval.request import RetrievalRequestError
from rag_mvp.safety.injection import InjectionPolicy
from rag_mvp.safety.redactor import Redactor
from rag_mvp.storage.database import Database
from rag_mvp.storage.layout import DataLayout
from rag_mvp.storage.repositories import RuntimeRepositories

_ADAPTER_VERSION = "openai-compatible-v1"
_CLEANUP_TASKS: set[asyncio.Task[None]] = set()


@dataclass(frozen=True, slots=True)
class ExecutableComposition:
    ingestion: IngestionService
    qa: QARuntimeServices
    diagnostics: SafeRequestDiagnosticStore


@dataclass(frozen=True, slots=True)
class _RoutedEmbeddingProvider:
    router: ModelProviderRouter
    identity: EmbeddingSpaceIdentity

    async def embed(
        self,
        request: EmbeddingRequest,
        context: ProviderCallContext,
    ) -> EmbeddingResult:
        routed = await self.router.embed(request, context, required_space=self.identity)
        return routed.value


@dataclass(frozen=True, slots=True)
class _DisabledOcrAdapter:
    version: str = "ocr-disabled-v1"

    def recognize(self, png_bytes: bytes, *, languages: str) -> str:
        del png_bytes, languages
        raise RuntimeError("ocr_disabled")


def compose_openai_services(settings: Settings, redactor: Redactor) -> ExecutableComposition:
    """Build the shared persistent object graph for a configured executable."""

    if settings.provider_backend != "openai" or settings.provider_readiness_errors():
        raise ValueError("openai_provider_configuration_incomplete")
    if not redactor.fully_configured:
        raise ValueError("safety_configuration_incomplete")

    layout = DataLayout.from_root(settings.data_root)
    layout.initialize()
    database = Database(layout.metadata_db)
    database.initialize()
    runtime_repositories = RuntimeRepositories.from_database(database)
    recorder = PersistentAttemptRecorder(runtime_repositories.provider_usage)
    diagnostics = SafeRequestDiagnosticStore(database, redactor=redactor)

    proxy_url = (
        settings.openai_proxy_url.get_secret_value()
        if settings.openai_proxy_url is not None
        else None
    )
    api_key = settings.openai_api_key
    if api_key is None:
        raise ValueError("openai_provider_configuration_incomplete")
    client = create_async_openai_client(
        OpenAIClientConfig(
            base_url=settings.openai_base_url,
            api_key=api_key.get_secret_value(),
            secret_reference=":".join(("env", "RAG_MVP_OPENAI_API_KEY")),
            timeout_seconds=settings.provider_timeout_seconds,
            proxy_url=proxy_url,
        )
    )
    worker_pools = RagWorkerPools()
    try:
        return _compose_with_client(
            settings,
            redactor,
            layout,
            runtime_repositories,
            recorder,
            diagnostics,
            client,
            worker_pools,
        )
    except Exception:
        _close_failed_client(client)
        _close_failed_worker_pools(worker_pools)
        raise


def _compose_with_client(
    settings: Settings,
    redactor: Redactor,
    layout: DataLayout,
    runtime_repositories: RuntimeRepositories,
    recorder: PersistentAttemptRecorder,
    diagnostics: SafeRequestDiagnosticStore,
    client: AsyncOpenAI,
    worker_pools: RagWorkerPools,
) -> ExecutableComposition:
    provider_alias = _provider_alias(settings.openai_base_url)
    embedding_identity = EmbeddingSpaceIdentity(
        provider=provider_alias,
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
        normalization=NormalizationPolicy.NONE,
        adapter_version=_ADAPTER_VERSION,
    )
    embedding = OpenAIEmbeddingProvider(
        client,
        embedding_identity,
        send_dimensions=settings.openai_send_dimensions,
    )
    generation = OpenAIChatGenerationProvider(
        client,
        ModelIdentity(provider_alias, settings.generation_model, _ADAPTER_VERSION),
        max_tokens_parameter=settings.openai_max_tokens_parameter,
    )
    retry_policy = RetryPolicy(
        attempt_timeout_seconds=settings.provider_timeout_seconds,
        max_retries=settings.provider_retry_limit,
    )
    embedding_route: EmbeddingRoute = ProviderRoute(
        "openai-embedding",
        embedding,
        retry_policy,
    )
    generation_route: GenerationRoute = ProviderRoute(
        "openai-generation",
        generation,
        retry_policy,
    )
    reranking_routes: tuple[RerankingRoute, ...] = ()
    if settings.reranking_model is not None:
        reranking_generation = OpenAIChatGenerationProvider(
            client,
            ModelIdentity(provider_alias, settings.reranking_model, _ADAPTER_VERSION),
            max_tokens_parameter=settings.openai_max_tokens_parameter,
        )
        reranker = OpenAIListwiseRerankingProvider(
            reranking_generation,
            max_candidates=settings.rerank_candidate_limit,
        )
        reranking_routes = (ProviderRoute("openai-reranking", reranker, retry_policy),)
    router = ModelProviderRouter(
        embedding_routes=(embedding_route,),
        generation_routes=(generation_route,),
        reranking_routes=reranking_routes,
        recorder=recorder,
    )
    ocr: OcrAdapter | None = None if settings.ocr_enabled else _DisabledOcrAdapter()
    ingestion = IngestionService.create(
        settings.data_root,
        _RoutedEmbeddingProvider(router, embedding_identity),
        ocr=ocr,
        chunking_config=ChunkingConfig(
            target_tokens=settings.chunk_target_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        ),
        upload_max_bytes=settings.upload_max_bytes,
        ocr_languages=settings.ocr_languages,
        worker_pools=worker_pools,
    )
    try:
        return _compose_qa(
            settings,
            redactor,
            layout,
            runtime_repositories,
            client,
            router,
            embedding_identity,
            reranking_routes,
            ingestion,
            diagnostics,
            worker_pools,
        )
    except Exception:
        ingestion.close()
        raise


def _compose_qa(
    settings: Settings,
    redactor: Redactor,
    layout: DataLayout,
    runtime_repositories: RuntimeRepositories,
    client: AsyncOpenAI,
    router: ModelProviderRouter,
    embedding_identity: EmbeddingSpaceIdentity,
    reranking_routes: tuple[RerankingRoute, ...],
    ingestion: IngestionService,
    diagnostics: SafeRequestDiagnosticStore,
    worker_pools: RagWorkerPools,
) -> ExecutableComposition:
    conversations = ConversationService(runtime_repositories.sessions)
    snapshots = BoundRetrievalSnapshotFactory(layout, ingestion.repositories.index_revisions)
    injection_policy = InjectionPolicy()
    orchestrator = QAOrchestrator(
        conversations=conversations,
        retrieval=SnapshotRetrievalGateway(
            snapshots,
            router,
            reranker=router if reranking_routes else None,
            settings=settings,
            worker_pools=worker_pools,
        ),
        generation=router,
        fact_assessor=SemanticFactEvidenceAssessor(
            router,
            required_space=embedding_identity,
        ),
        context_builder=ContextBuilder(maximum_chunks=settings.context_chunk_limit),
        injection_policy=injection_policy,
        budgets=_qa_budgets(settings),
        maximum_provider_attempts=settings.provider_retry_limit + 1,
    )

    def readiness_probe() -> tuple[bool, str | None]:
        if not router.qa_ready:
            return False, "provider_runtime_unavailable"
        try:
            with snapshots.bind() as snapshot:
                active_identity = provider_embedding_identity(snapshot.revision.embedding_space)
                if active_identity != embedding_identity:
                    return False, "index_embedding_incompatible"
        except RetrievalRequestError as error:
            if error.code == "index_not_ready":
                return False, "index_not_ready"
            return False, "index_unavailable"
        except Exception:
            return False, "index_unavailable"
        return True, None

    async def close_runtime_resources() -> None:
        try:
            await client.close()
        finally:
            await worker_pools.aclose()

    qa = QARuntimeServices(
        conversations=conversations,
        orchestrator=orchestrator,
        emitter=CompleteResponseEmitter(
            conversations,
            injection_policy=injection_policy,
            redactor=redactor,
        ),
        readiness_probe=readiness_probe,
        close_callback=close_runtime_resources,
    )
    return ExecutableComposition(ingestion=ingestion, qa=qa, diagnostics=diagnostics)


def _close_failed_client(client: AsyncOpenAI) -> None:
    async def close_safely() -> None:
        with suppress(Exception):
            await client.close()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(close_safely())
    else:
        task = loop.create_task(close_safely(), name="close-failed-openai-client")
        _CLEANUP_TASKS.add(task)
        task.add_done_callback(_CLEANUP_TASKS.discard)


def _close_failed_worker_pools(worker_pools: RagWorkerPools) -> None:
    async def close_safely() -> None:
        with suppress(Exception):
            await worker_pools.aclose()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(close_safely())
    else:
        task = loop.create_task(close_safely(), name="close-failed-worker-pools")
        _CLEANUP_TASKS.add(task)
        task.add_done_callback(_CLEANUP_TASKS.discard)


def _qa_budgets(settings: Settings) -> QAStageBudgets:
    resolved = QALatencyBudgets.from_settings(settings)
    return QAStageBudgets(
        total_seconds=resolved.total_seconds,
        validation_seconds=resolved.validation_seconds,
        retrieval_seconds=(
            settings.qa_retrieval_budget_seconds
            if "qa_retrieval_budget_seconds" in settings.model_fields_set
            else 0.8 * (settings.qa_deadline_seconds / 9.5)
        ),
        rerank_seconds=resolved.rerank_seconds,
        evidence_assessment_seconds=resolved.evidence_assessment_seconds,
        generation_seconds=resolved.generation_seconds,
        finalization_seconds=resolved.finalization_seconds,
    )


def _provider_alias(base_url: str) -> str:
    endpoint_digest = hashlib.sha256(base_url.rstrip("/").encode("utf-8")).hexdigest()[:16]
    return f"openai-compatible-{endpoint_digest}"
