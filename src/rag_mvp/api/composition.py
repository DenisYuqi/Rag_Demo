"""Executable OpenAI-compatible ingestion and QA service composition."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Literal

from openai import AsyncOpenAI

from rag_mvp.api.qa import QARuntimeServices
from rag_mvp.config.settings import Settings
from rag_mvp.evaluation.application import (
    EvaluationApplicationService,
    VerifiedEvaluationArtifactStore,
)
from rag_mvp.evaluation.artifacts_v2 import ArtifactCatalogV2
from rag_mvp.evaluation.comparison_artifacts import ComparisonArtifactCatalog
from rag_mvp.evaluation.comparison_plans import RegisteredComparisonPlanRegistry
from rag_mvp.evaluation.comparison_production import (
    ProductionComparisonJobExecutor,
    RegisteredComparisonLaunchCatalog,
)
from rag_mvp.evaluation.plan import EvaluationDatasetRegistry
from rag_mvp.evaluation.pricing import (
    OPENAI_STANDARD_PRICING_VERSION,
    openai_standard_pricing_catalog,
)
from rag_mvp.evaluation.production import (
    ProductionEvaluationJobExecutor,
    VerifiedLegacyReportStore,
)
from rag_mvp.evaluation.release_evidence import VerifiedReleaseEvidenceStore
from rag_mvp.ingestion.chunking import ChunkingConfig
from rag_mvp.ingestion.extractors import OcrAdapter
from rag_mvp.ingestion.service import IngestionService
from rag_mvp.observability.diagnostics import SafeRequestDiagnosticStore
from rag_mvp.performance.deadlines import QALatencyBudgets
from rag_mvp.performance.worker_pools import RagWorkerPools
from rag_mvp.providers.bge_adapters import (
    LocalBgeEmbeddingProvider,
    LocalBgeRerankingProvider,
)
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
from rag_mvp.qa.refusal import RefusalPolicy
from rag_mvp.qa.sessions import ConversationService
from rag_mvp.qa.streaming import CompleteResponseEmitter
from rag_mvp.retrieval.binding import BoundRetrievalSnapshotFactory
from rag_mvp.retrieval.cache import RetrievalResultCache
from rag_mvp.retrieval.identity import provider_embedding_identity
from rag_mvp.retrieval.request import RetrievalRequestError
from rag_mvp.safety.injection import InjectionPolicy
from rag_mvp.safety.redactor import Redactor
from rag_mvp.storage.database import Database
from rag_mvp.storage.layout import DataLayout
from rag_mvp.storage.repositories import RuntimeRepositories

_ADAPTER_VERSION = "openai-compatible-v1"
_BGE_ADAPTER_VERSION = "flag-embedding-v1"
_CLEANUP_TASKS: set[asyncio.Task[None]] = set()


@dataclass(frozen=True, slots=True)
class ExecutableComposition:
    ingestion: IngestionService
    qa: QARuntimeServices
    diagnostics: SafeRequestDiagnosticStore
    retrieval_cache: RetrievalResultCache | None = None
    evaluation: EvaluationApplicationService | None = None


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


def _compose_retrieval_cache(settings: Settings) -> RetrievalResultCache | None:
    if not settings.retrieval_cache_enabled:
        return None
    return RetrievalResultCache(
        configuration_id=settings.configuration_identity,
        maximum_entries=settings.retrieval_cache_max_entries,
        ttl_seconds=settings.retrieval_cache_ttl_seconds,
    )


def compose_openai_services(
    settings: Settings,
    redactor: Redactor,
    *,
    include_evaluation: bool = True,
) -> ExecutableComposition:
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

    client = _create_openai_client(settings)
    worker_pools = RagWorkerPools()
    try:
        composition = _compose_with_client(
            settings,
            redactor,
            layout,
            runtime_repositories,
            recorder,
            diagnostics,
            client,
            worker_pools,
        )
        if not include_evaluation:
            return composition
        run_root = layout.ensure_within_root(layout.directory("evaluations") / "runs")
        published_root = layout.ensure_within_root(layout.directory("evaluations") / "published")
        comparison_published_root = layout.ensure_within_root(
            layout.directory("evaluations") / "suites" / "published"
        )
        artifact_catalog = ArtifactCatalogV2(published_root)
        comparison_artifact_catalog = ComparisonArtifactCatalog(
            comparison_published_root,
            redactor,
        )
        evaluation_settings = settings.model_copy(
            update={
                "pricing_version": OPENAI_STANDARD_PRICING_VERSION,
                "workbench_enabled": False,
                "retrieval_cache_enabled": False,
            }
        )
        evaluation_executor = ProductionEvaluationJobExecutor(
            settings=evaluation_settings,
            repository=runtime_repositories.evaluation_runs,
            report_repository=runtime_repositories.report_manifests,
            run_artifacts_root=run_root,
            redactor=redactor,
        )
        dataset_registry = EvaluationDatasetRegistry(settings.evaluation_dataset_root)
        comparison_registry = RegisteredComparisonPlanRegistry()
        comparison_catalog = RegisteredComparisonLaunchCatalog(
            registry=comparison_registry,
            datasets=dataset_registry,
            settings=evaluation_settings,
            evaluation_repository=runtime_repositories.evaluation_runs,
            comparison_repository=runtime_repositories.comparisons,
            run_artifacts_root=run_root,
        )
        comparison_executor = ProductionComparisonJobExecutor(
            settings=evaluation_settings,
            evaluation_repository=runtime_repositories.evaluation_runs,
            comparison_repository=runtime_repositories.comparisons,
            run_artifacts_root=run_root,
            artifact_catalog=comparison_artifact_catalog,
            redactor=redactor,
        )
        evaluation = EvaluationApplicationService(
            registry=dataset_registry,
            settings=evaluation_settings,
            repository=runtime_repositories.evaluation_runs,
            run_artifacts_root=run_root,
            executor=evaluation_executor,
            maximum_active_jobs=settings.evaluation_max_active_jobs,
            shutdown_grace_seconds=settings.evaluation_shutdown_grace_seconds,
            artifact_store=VerifiedEvaluationArtifactStore(artifact_catalog),
            legacy_report_store=VerifiedLegacyReportStore(
                runtime_repositories.report_manifests,
                run_root,
            ),
            release_store=VerifiedReleaseEvidenceStore(
                settings.evaluation_release_root,
                redactor,
            ),
            plan_settings_factory=evaluation_executor.isolated_settings,
            comparison_catalog=comparison_catalog,
            comparison_repository=runtime_repositories.comparisons,
            comparison_executor=comparison_executor,
            comparison_artifact_store=comparison_artifact_catalog,
        )
        return replace(composition, evaluation=evaluation)
    except Exception:
        _close_failed_client(client)
        _close_failed_worker_pools(worker_pools)
        raise


def compose_bge_services(settings: Settings, redactor: Redactor) -> ExecutableComposition:
    """Build the isolated local-retrieval profile with shared API generation."""

    if settings.provider_backend != "openai" or settings.provider_readiness_errors():
        raise ValueError("openai_provider_configuration_incomplete")
    if not settings.bge_profile_enabled:
        raise ValueError("bge_profile_disabled")
    if not redactor.fully_configured:
        raise ValueError("safety_configuration_incomplete")

    local_settings = settings.bge_profile_settings()
    layout = DataLayout.from_root(local_settings.data_root)
    layout.initialize()
    database = Database(layout.metadata_db)
    database.initialize()
    runtime_repositories = RuntimeRepositories.from_database(database)
    recorder = PersistentAttemptRecorder(runtime_repositories.provider_usage)
    diagnostics = SafeRequestDiagnosticStore(database, redactor=redactor)
    client = _create_openai_client(local_settings)
    worker_pools = RagWorkerPools()
    try:
        return _compose_with_client(
            local_settings,
            redactor,
            layout,
            runtime_repositories,
            recorder,
            diagnostics,
            client,
            worker_pools,
            retrieval_backend="bge",
        )
    except Exception:
        _close_failed_client(client)
        _close_failed_worker_pools(worker_pools)
        raise


def _create_openai_client(settings: Settings) -> AsyncOpenAI:
    proxy_url = (
        settings.openai_proxy_url.get_secret_value()
        if settings.openai_proxy_url is not None
        else None
    )
    api_key = settings.openai_api_key
    if api_key is None:
        raise ValueError("openai_provider_configuration_incomplete")
    return create_async_openai_client(
        OpenAIClientConfig(
            base_url=settings.openai_base_url,
            api_key=api_key.get_secret_value(),
            secret_reference=":".join(("env", "RAG_MVP_OPENAI_API_KEY")),
            timeout_seconds=settings.provider_timeout_seconds,
            proxy_url=proxy_url,
        )
    )


def _compose_with_client(
    settings: Settings,
    redactor: Redactor,
    layout: DataLayout,
    runtime_repositories: RuntimeRepositories,
    recorder: PersistentAttemptRecorder,
    diagnostics: SafeRequestDiagnosticStore,
    client: AsyncOpenAI,
    worker_pools: RagWorkerPools,
    *,
    retrieval_backend: Literal["openai", "bge"] = "openai",
) -> ExecutableComposition:
    provider_alias = _provider_alias(settings.openai_base_url)
    generation = OpenAIChatGenerationProvider(
        client,
        ModelIdentity(provider_alias, settings.generation_model, _ADAPTER_VERSION),
        max_tokens_parameter=settings.openai_max_tokens_parameter,
    )
    retry_policy = RetryPolicy(
        attempt_timeout_seconds=settings.provider_timeout_seconds,
        max_retries=settings.provider_retry_limit,
    )
    generation_route: GenerationRoute = ProviderRoute(
        "openai-generation",
        generation,
        retry_policy,
    )
    provider_cleanup: Callable[[], None] | None = None
    if retrieval_backend == "openai":
        embedding_identity = EmbeddingSpaceIdentity(
            provider=provider_alias,
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            normalization=NormalizationPolicy.NONE,
            adapter_version=_ADAPTER_VERSION,
        )
        openai_embedding = OpenAIEmbeddingProvider(
            client,
            embedding_identity,
            send_dimensions=settings.openai_send_dimensions,
        )
        embedding_route: EmbeddingRoute = ProviderRoute(
            "openai-embedding",
            openai_embedding,
            retry_policy,
        )
        reranking_routes: tuple[RerankingRoute, ...] = ()
        if settings.reranking_model is not None:
            reranking_generation = OpenAIChatGenerationProvider(
                client,
                ModelIdentity(provider_alias, settings.reranking_model, _ADAPTER_VERSION),
                max_tokens_parameter=settings.openai_max_tokens_parameter,
            )
            openai_reranker = OpenAIListwiseRerankingProvider(
                reranking_generation,
                max_candidates=settings.rerank_candidate_limit,
            )
            reranking_routes = (
                ProviderRoute("openai-reranking", openai_reranker, retry_policy),
            )
    else:
        embedding_identity = EmbeddingSpaceIdentity(
            provider="bge-local",
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            normalization=NormalizationPolicy.L2,
            adapter_version=_BGE_ADAPTER_VERSION,
        )
        bge_embedding = LocalBgeEmbeddingProvider(
            embedding_identity,
            device=settings.bge_device,
            use_fp16=settings.bge_use_fp16,
            batch_size=settings.bge_embedding_batch_size,
            max_length=settings.bge_embedding_max_length,
            cache_dir=settings.bge_model_cache_dir,
        )
        reranking_model = settings.reranking_model
        if reranking_model is None:
            raise ValueError("bge_reranking_model_missing")
        bge_reranker = LocalBgeRerankingProvider(
            ModelIdentity("bge-local", reranking_model, _BGE_ADAPTER_VERSION),
            device=settings.bge_device,
            use_fp16=settings.bge_use_fp16,
            batch_size=settings.bge_reranking_batch_size,
            max_length=settings.bge_reranking_max_length,
            max_candidates=settings.rerank_candidate_limit,
            cache_dir=settings.bge_model_cache_dir,
        )
        embedding_route = ProviderRoute("bge-embedding", bge_embedding, retry_policy)
        reranking_routes = (ProviderRoute("bge-reranking", bge_reranker, retry_policy),)

        def close_bge_models() -> None:
            bge_reranker.close()
            bge_embedding.close()

        provider_cleanup = close_bge_models
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
            provider_cleanup,
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
    provider_cleanup: Callable[[], None] | None = None,
) -> ExecutableComposition:
    conversations = ConversationService(runtime_repositories.sessions)
    snapshots = BoundRetrievalSnapshotFactory(layout, ingestion.repositories.index_revisions)
    retrieval_cache = _compose_retrieval_cache(settings)
    injection_policy = InjectionPolicy()
    orchestrator = QAOrchestrator(
        conversations=conversations,
        retrieval=SnapshotRetrievalGateway(
            snapshots,
            router,
            reranker=router if reranking_routes else None,
            settings=settings,
            worker_pools=worker_pools,
            cache=retrieval_cache,
        ),
        generation=router,
        fact_assessor=SemanticFactEvidenceAssessor(
            router,
            required_space=embedding_identity,
        ),
        context_builder=ContextBuilder(maximum_chunks=settings.context_chunk_limit),
        injection_policy=injection_policy,
        refusal_policy=RefusalPolicy(
            minimum_support_score=settings.qa_minimum_support_score,
        ),
        budgets=_qa_budgets(settings),
        maximum_provider_attempts=settings.provider_retry_limit + 1,
        pricing_catalog=(
            openai_standard_pricing_catalog(
                provider=embedding_identity.provider,
                models=tuple(
                    model
                    for model in (
                        settings.embedding_model,
                        settings.generation_model,
                        settings.reranking_model,
                    )
                    if model is not None
                ),
            )
            if settings.pricing_version == OPENAI_STANDARD_PRICING_VERSION
            else None
        ),
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
            try:
                await worker_pools.aclose()
            finally:
                if provider_cleanup is not None:
                    await asyncio.to_thread(provider_cleanup)

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
    return ExecutableComposition(
        ingestion=ingestion,
        qa=qa,
        diagnostics=diagnostics,
        retrieval_cache=retrieval_cache,
    )


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
    defaults = QAStageBudgets()
    return QAStageBudgets(
        total_seconds=resolved.total_seconds,
        validation_seconds=resolved.validation_seconds,
        retrieval_seconds=(
            settings.qa_retrieval_budget_seconds
            if "qa_retrieval_budget_seconds" in settings.model_fields_set
            else defaults.retrieval_seconds * (settings.qa_deadline_seconds / 9.5)
        ),
        rerank_seconds=resolved.rerank_seconds,
        evidence_assessment_seconds=resolved.evidence_assessment_seconds,
        generation_seconds=resolved.generation_seconds,
        finalization_seconds=resolved.finalization_seconds,
    )


def _provider_alias(base_url: str) -> str:
    endpoint_digest = hashlib.sha256(base_url.rstrip("/").encode("utf-8")).hexdigest()[:16]
    return f"openai-compatible-{endpoint_digest}"
