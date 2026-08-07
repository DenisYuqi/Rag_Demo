from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest
from pydantic import SecretStr
from test_scoring_v2 import _result

from rag_mvp.api.qa import (
    QAEventEmitter,
    QAOrchestratorGateway,
    QARuntimeServices,
)
from rag_mvp.config.settings import Settings
from rag_mvp.domain import UnavailableValue
from rag_mvp.domain.evaluation import (
    ModelAttempt,
    ModelAttemptStatus,
    ModelRole,
)
from rag_mvp.domain.evaluation import (
    TokenUsage as DomainTokenUsage,
)
from rag_mvp.domain.ingestion import (
    EmbeddingSpaceIdentity,
    IndexRevision,
    IndexRevisionStatus,
)
from rag_mvp.domain.qa import QAErrorCode, StreamEventKind
from rag_mvp.domain.retrieval import CachePolicy
from rag_mvp.evaluation.comparison_application import (
    ComparisonRunEntry,
    ComparisonSummary,
)
from rag_mvp.evaluation.comparison_artifacts import ComparisonArtifactCatalog
from rag_mvp.evaluation.comparison_plans import (
    REGISTERED_GENERATION_PLAN_ID,
    MaterializedComparisonCandidate,
    RegisteredComparisonPlanRegistry,
)
from rag_mvp.evaluation.comparison_production import (
    ComparisonCorpusInstaller,
    ComparisonProductionError,
    ComparisonRuntimeComposition,
    ProductionComparisonExecutionContext,
    ProductionComparisonJobExecutor,
    RegisteredComparisonLaunchCatalog,
)
from rag_mvp.evaluation.corpus import InstalledEvaluationCorpus
from rag_mvp.evaluation.dataset import EvaluationCaseV2, EvaluationDataset
from rag_mvp.evaluation.environment import EvaluationIndexReuseKey
from rag_mvp.evaluation.experiment import ExperimentAxis
from rag_mvp.evaluation.plan import EvaluationDatasetRegistry
from rag_mvp.evaluation.runner import (
    EvaluationCaseExecution,
    EvaluationCaseInput,
)
from rag_mvp.evaluation.work_budget import ProviderWorkBudget
from rag_mvp.ingestion.service import IngestionService
from rag_mvp.providers.errors import ProviderError
from rag_mvp.providers.models import (
    Deadline,
    EmbeddingResult,
    ModelIdentity,
    NormalizationPolicy,
    ProviderCallContext,
    ProviderErrorCategory,
    ProviderRole,
    RouteMetadata,
    TokenUsage,
)
from rag_mvp.providers.models import (
    EmbeddingSpaceIdentity as ProviderEmbeddingSpaceIdentity,
)
from rag_mvp.providers.persistence import PersistentAttemptRecorder
from rag_mvp.providers.resilience import RetryPolicy, execute_with_resilience
from rag_mvp.qa.sessions import ConversationService
from rag_mvp.safety.redactor import Redactor
from rag_mvp.storage.database import Database
from rag_mvp.storage.layout import DataLayout
from rag_mvp.storage.repositories import KnowledgeRepositories, RuntimeRepositories

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DATASETS_ROOT = _REPOSITORY_ROOT / "evaluations" / "datasets"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$")


@dataclass(slots=True)
class _Emitter:
    ready: bool = True

    def emit(self, outcome: object, *, owner_id: str) -> object:
        del outcome, owner_id
        return object()


def _qa_services() -> QARuntimeServices:
    return QARuntimeServices(
        conversations=cast(ConversationService, object()),
        orchestrator=cast(QAOrchestratorGateway, object()),
        emitter=cast(QAEventEmitter, _Emitter()),
    )


@dataclass(slots=True)
class _Ingestion:
    data_root: Path
    repositories: KnowledgeRepositories
    closed: bool = False

    def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class _Composition:
    ingestion: _Ingestion
    qa: QARuntimeServices
    retrieval_cache: object | None


@dataclass(slots=True)
class _CompositionFactory:
    retrieval_cache: object | None = field(default_factory=object)
    values: list[_Composition] = field(default_factory=list)

    def __call__(
        self,
        settings: Settings,
        redactor: Redactor,
    ) -> ComparisonRuntimeComposition:
        del redactor
        layout = DataLayout.from_root(settings.data_root)
        layout.initialize()
        database = Database(layout.metadata_db)
        database.initialize()
        composition = _Composition(
            ingestion=_Ingestion(
                data_root=settings.data_root.resolve(),
                repositories=KnowledgeRepositories.from_database(database),
            ),
            qa=_qa_services(),
            retrieval_cache=self.retrieval_cache,
        )
        self.values.append(composition)
        return cast(ComparisonRuntimeComposition, composition)


@dataclass(slots=True)
class _WrongWorkspaceCompositionFactory:
    delegate: _CompositionFactory = field(default_factory=_CompositionFactory)

    @property
    def values(self) -> list[_Composition]:
        return self.delegate.values

    def __call__(
        self,
        settings: Settings,
        redactor: Redactor,
    ) -> ComparisonRuntimeComposition:
        composition = cast(_Composition, self.delegate(settings, redactor))
        composition.ingestion.data_root = settings.data_root.parent.resolve()
        return cast(ComparisonRuntimeComposition, composition)


@dataclass(slots=True)
class _InstallerFactory:
    revision: IndexRevision
    calls: int = 0

    def __call__(self, ingestion: IngestionService) -> ComparisonCorpusInstaller:
        parent = self

        @dataclass(slots=True)
        class Installer:
            async def install(
                self,
                dataset: EvaluationDataset,
            ) -> InstalledEvaluationCorpus:
                parent.calls += 1
                if ingestion.repositories.index_revisions.get(parent.revision.revision_id) is None:
                    ingestion.repositories.index_revisions.create(parent.revision)
                    _set_active_revision(ingestion, parent.revision)
                return InstalledEvaluationCorpus(
                    revision=parent.revision,
                    dataset_id=dataset.manifest.dataset_id,
                    dataset_version=dataset.manifest.version,
                    dataset_hash=dataset.manifest.content_hash,
                    corpus_version=dataset.corpus.manifest.version,
                    corpus_hash=dataset.corpus.manifest.content_hash,
                )

        return Installer()


@dataclass(slots=True)
class _RetryingInstallerFactory:
    revision: IndexRevision
    installed_revision: IndexRevision | None = None
    persist_attempts: bool = True
    calls: int = 0

    def __call__(self, ingestion: IngestionService) -> ComparisonCorpusInstaller:
        parent = self
        repository = RuntimeRepositories.from_database(
            ingestion.repositories.index_revisions.database
        ).provider_usage
        recorder = (
            PersistentAttemptRecorder(
                repository,
                attempt_id_factory=iter(("setup-attempt-1", "setup-attempt-2")).__next__,
            )
            if parent.persist_attempts
            else None
        )

        @dataclass(slots=True)
        class Installer:
            async def install(
                self,
                dataset: EvaluationDataset,
            ) -> InstalledEvaluationCorpus:
                parent.calls += 1
                attempts = 0
                embedding = parent.revision.embedding_space
                identity = ModelIdentity(
                    embedding.provider_alias,
                    embedding.model,
                    embedding.adapter_version,
                )

                async def operation() -> EmbeddingResult:
                    nonlocal attempts
                    attempts += 1
                    if attempts == 1:
                        raise ProviderError(ProviderErrorCategory.NETWORK)
                    return EmbeddingResult(
                        vectors=(tuple(0.0 for _ in range(embedding.dimension)),),
                        identity=ProviderEmbeddingSpaceIdentity(
                            provider=embedding.provider_alias,
                            model=embedding.model,
                            dimension=embedding.dimension,
                            normalization=NormalizationPolicy(embedding.normalization),
                            adapter_version=embedding.adapter_version,
                        ),
                        usage=TokenUsage(input_tokens=11),
                    )

                async def no_sleep(delay: float) -> None:
                    del delay

                digest = dataset.corpus.manifest.content_hash.removeprefix("sha256:")
                await execute_with_resilience(
                    operation,
                    context=ProviderCallContext(
                        f"eval_corpus_{digest}",
                        parent.revision.revision_id,
                        Deadline.after(5),
                    ),
                    route=RouteMetadata(
                        "setup-embedding",
                        ProviderRole.EMBEDDING,
                        identity,
                    ),
                    policy=RetryPolicy(
                        attempt_timeout_seconds=1,
                        max_retries=1,
                        initial_backoff_seconds=0,
                    ),
                    is_fallback=False,
                    recorder=recorder,
                    sleep=no_sleep,
                )
                installed_revision = parent.installed_revision or parent.revision
                ingestion.repositories.index_revisions.create(installed_revision)
                _set_active_revision(ingestion, installed_revision)
                return InstalledEvaluationCorpus(
                    revision=installed_revision,
                    dataset_id=dataset.manifest.dataset_id,
                    dataset_version=dataset.manifest.version,
                    dataset_hash=dataset.manifest.content_hash,
                    corpus_version=dataset.corpus.manifest.version,
                    corpus_hash=dataset.corpus.manifest.content_hash,
                )

        return Installer()


@dataclass(slots=True)
class _SuccessfulInstallerFactory:
    revision: IndexRevision
    calls: int = 0

    def __call__(self, ingestion: IngestionService) -> ComparisonCorpusInstaller:
        parent = self
        repository = RuntimeRepositories.from_database(
            ingestion.repositories.index_revisions.database
        ).provider_usage
        recorder = PersistentAttemptRecorder(
            repository,
            attempt_id_factory=lambda: "setup-success-attempt",
        )

        @dataclass(slots=True)
        class Installer:
            async def install(
                self,
                dataset: EvaluationDataset,
            ) -> InstalledEvaluationCorpus:
                parent.calls += 1
                embedding = parent.revision.embedding_space
                identity = ModelIdentity(
                    embedding.provider_alias,
                    embedding.model,
                    embedding.adapter_version,
                )

                async def operation() -> EmbeddingResult:
                    return EmbeddingResult(
                        vectors=(tuple(0.0 for _ in range(embedding.dimension)),),
                        identity=ProviderEmbeddingSpaceIdentity(
                            provider=embedding.provider_alias,
                            model=embedding.model,
                            dimension=embedding.dimension,
                            normalization=NormalizationPolicy(embedding.normalization),
                            adapter_version=embedding.adapter_version,
                        ),
                        usage=TokenUsage(input_tokens=752),
                    )

                digest = dataset.corpus.manifest.content_hash.removeprefix("sha256:")
                await execute_with_resilience(
                    operation,
                    context=ProviderCallContext(
                        f"eval_corpus_{digest}",
                        parent.revision.revision_id,
                        Deadline.after(5),
                    ),
                    route=RouteMetadata("setup-embedding", ProviderRole.EMBEDDING, identity),
                    policy=RetryPolicy(attempt_timeout_seconds=1, max_retries=0),
                    is_fallback=False,
                    recorder=recorder,
                )
                ingestion.repositories.index_revisions.create(parent.revision)
                _set_active_revision(ingestion, parent.revision)
                return InstalledEvaluationCorpus(
                    revision=parent.revision,
                    dataset_id=dataset.manifest.dataset_id,
                    dataset_version=dataset.manifest.version,
                    dataset_hash=dataset.manifest.content_hash,
                    corpus_version=dataset.corpus.manifest.version,
                    corpus_hash=dataset.corpus.manifest.content_hash,
                )

        return Installer()


@dataclass(slots=True)
class _PersistingCaseExecutor:
    candidate: MaterializedComparisonCandidate
    dataset: EvaluationDataset
    revision_id: str
    executed: list[tuple[str, str]]
    timeout_variant_id: str
    fail_source_case_id: str = "accept-zh-004"
    adjusted_embedding_usage: bool = False
    adjusted_generation_usage: bool = False

    async def execute(
        self,
        case: EvaluationCaseInput,
        *,
        owner_id: str,
        cache_policy: CachePolicy,
    ) -> EvaluationCaseExecution:
        source_id = case.source_case_id or case.case_id
        source = next(item for item in self.dataset.cases if item.case_id == source_id)
        template = _result(cast(EvaluationCaseV2, source))
        template_execution = template.execution
        assert template_execution is not None
        run_id = self.candidate.evaluation_plan.run_id
        digest = hashlib.sha256(f"{run_id}\0{case.case_id}".encode()).hexdigest()
        request_id = f"request-{digest[:32]}"
        session_id = f"session-{digest[:32]}"
        diagnostics = template_execution.event.diagnostics.model_copy(
            update={
                "cache_status": {"retrieval": "bypass"},
                "metadata": {"index_revision": self.revision_id},
            }
        )
        should_timeout = (
            self.candidate.variant_id == self.timeout_variant_id
            and source_id == self.fail_source_case_id
            and case.repeat_index == 1
        )
        event_values = template_execution.event.model_dump(mode="python")
        event_values.update(
            {
                "request_id": request_id,
                "session_id": session_id,
                "diagnostics": diagnostics,
            }
        )
        if should_timeout:
            event_values.update(
                {
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
        event = type(template_execution.event).model_validate(event_values)
        repository = RuntimeRepositories.from_database(
            Database(self.candidate.settings.data_root / "metadata.sqlite3")
        ).provider_usage
        embedding_provider = self.candidate.evaluation_plan.identity.provider_identities[
            "embedding"
        ]
        embedding_model = self.candidate.evaluation_plan.identity.model_identities["embedding"]
        generation_model = self.candidate.evaluation_plan.identity.model_identities["generation"]
        embedding_input_tokens = 13
        if not self.adjusted_embedding_usage:
            embedding_input_tokens = 117 if generation_model == "gpt-5.4" else 19
            self.adjusted_embedding_usage = True
        repository.record(
            ModelAttempt(
                attempt_id=f"attempt-retrieval-{digest}",
                operation_id="qa-retrieval",
                request_id=request_id,
                run_id=run_id,
                role=ModelRole.EMBEDDING,
                provider=embedding_provider,
                model=embedding_model,
                status=ModelAttemptStatus.SUCCEEDED,
                latency_ms=1.0,
                usage=DomainTokenUsage(input_tokens=embedding_input_tokens),
            )
        )
        if should_timeout:
            repository.record(
                ModelAttempt(
                    attempt_id=f"attempt-fact-timeout-{digest}",
                    operation_id="fact-evidence-assessment",
                    request_id=request_id,
                    run_id=run_id,
                    role=ModelRole.EMBEDDING,
                    provider=embedding_provider,
                    model=embedding_model,
                    status=ModelAttemptStatus.TIMED_OUT,
                    latency_ms=30_000.0,
                    safe_error_category="timeout",
                    usage=DomainTokenUsage(),
                )
            )
        elif event.kind is StreamEventKind.ANSWER:
            generation_input_tokens = 10
            if not self.adjusted_generation_usage:
                generation_input_tokens = 6087 if generation_model == "gpt-5.4" else 5837
                self.adjusted_generation_usage = True
            repository.record(
                ModelAttempt(
                    attempt_id=f"attempt-generation-{digest}",
                    operation_id="qa-generation",
                    request_id=request_id,
                    run_id=run_id,
                    role=ModelRole.GENERATION,
                    provider=self.candidate.evaluation_plan.identity.provider_identities[
                        "generation"
                    ],
                    model=generation_model,
                    status=ModelAttemptStatus.SUCCEEDED,
                    latency_ms=2.0,
                    usage=DomainTokenUsage(
                        input_tokens=generation_input_tokens,
                        output_tokens=5,
                    ),
                )
            )
        self.executed.append((self.candidate.variant_id, case.case_id))
        retrieval_digest = f"sha256:{hashlib.sha256(case.case_id.encode()).hexdigest()}"
        return EvaluationCaseExecution(
            case_id=case.case_id,
            owner_id=owner_id,
            session_id=session_id,
            request_id=request_id,
            event=event,
            cache_policy=cache_policy,
            retrieved_chunk_ids=template_execution.retrieved_chunk_ids,
            context_chunk_ids=template_execution.context_chunk_ids,
            retrieval_evidence_digest=retrieval_digest,
            latency_ms=3.0,
        )


@dataclass(slots=True)
class _CancelledCaseExecutor:
    async def execute(
        self,
        case: EvaluationCaseInput,
        *,
        owner_id: str,
        cache_policy: CachePolicy,
    ) -> EvaluationCaseExecution:
        del case, owner_id, cache_policy
        raise asyncio.CancelledError


def _set_active_revision(
    ingestion: IngestionService,
    revision: IndexRevision,
) -> None:
    database = ingestion.repositories.index_revisions.database
    with database.connection() as connection:
        connection.execute(
            """
            INSERT INTO active_index_manifest(singleton_id, revision_id, updated_at)
            VALUES (1, ?, ?)
            """,
            (revision.revision_id, datetime.now(UTC).isoformat()),
        )


def _matching_revision(key: EvaluationIndexReuseKey) -> IndexRevision:
    embedding = {item.name: item.value for item in key.embedding_identity}
    index = {item.name: item.value for item in key.index_identity}
    chunking = {item.name: item.value for item in key.chunking_identity}
    dimension = embedding["dimension"]
    lexical_k1 = index["lexical_k1"]
    lexical_b = index["lexical_b"]
    assert isinstance(dimension, int) and not isinstance(dimension, bool)
    assert isinstance(lexical_k1, (int, float)) and not isinstance(lexical_k1, bool)
    assert isinstance(lexical_b, (int, float)) and not isinstance(lexical_b, bool)
    return IndexRevision(
        revision_id=f"rev_eval_{key.corpus_hash.removeprefix('sha256:')}",
        status=IndexRevisionStatus.ACTIVE,
        active_sources={item.source_id: item.version for item in key.active_sources},
        chunk_set_digest=key.chunk_set_digest,
        embedding_space=EmbeddingSpaceIdentity(
            provider_alias=str(embedding["provider"]),
            model=str(embedding["model"]),
            dimension=dimension,
            normalization=str(embedding["normalization"]),
            adapter_version=str(embedding["adapter_version"]),
        ),
        extraction_version=str(chunking["extraction_version"]),
        chunking_version=str(chunking["chunking_version"]),
        tokenizer_version=str(index["lexical_tokenizer_identity"]),
        dense_index_path="indexes/dense",
        lexical_index_path="indexes/lexical.json",
        chunk_count=key.chunk_count,
        dense_schema_version=str(index["dense_schema_version"]),
        dense_metric=str(index["dense_metric"]),
        lexical_schema_version=str(index["lexical_schema_version"]),
        lexical_algorithm_version=str(index["lexical_algorithm_version"]),
        lexical_k1=float(lexical_k1),
        lexical_b=float(lexical_b),
        record_digest_algorithm=key.record_digest_algorithm,
        published_at=datetime(2026, 8, 7, tzinfo=UTC),
    )


def _settings(data_root: Path) -> Settings:
    return Settings(
        _env_file=None,
        provider_backend="openai",
        openai_api_key=SecretStr("unit-test-key"),
        environment="test",
        data_root=data_root,
        evaluation_dataset_root=_DATASETS_ROOT,
        workbench_enabled=False,
    )


def _catalog_stack(
    tmp_path: Path,
) -> tuple[Settings, RuntimeRepositories, RegisteredComparisonLaunchCatalog]:
    online = tmp_path / "online"
    layout = DataLayout.from_root(online)
    layout.initialize()
    database = Database(layout.metadata_db)
    database.initialize()
    repositories = RuntimeRepositories.from_database(database)
    settings = _settings(online)
    catalog = RegisteredComparisonLaunchCatalog(
        registry=RegisteredComparisonPlanRegistry(),
        datasets=EvaluationDatasetRegistry(_DATASETS_ROOT),
        settings=settings,
        evaluation_repository=repositories.evaluation_runs,
        comparison_repository=repositories.comparisons,
        run_artifacts_root=tmp_path / "runs",
    )
    return settings, repositories, catalog


def test_catalog_prepares_path_isolated_normal_runs_without_database_inserts(
    tmp_path: Path,
) -> None:
    settings, repositories, catalog = _catalog_stack(tmp_path)

    launch = catalog.prepare("comparison-generation-1", REGISTERED_GENERATION_PLAN_ID)

    context = cast(ProductionComparisonExecutionContext, launch.execution_context)
    assert context.comparison_id == launch.suite.comparison_id
    assert (
        context.workspace
        == (
            settings.data_root / "evaluations" / "suites" / launch.suite.comparison_id / "runtime"
        ).resolve()
    )
    assert not context.workspace.exists()
    assert repositories.evaluation_runs.list() == []
    assert repositories.comparisons.get(launch.suite.comparison_id) is None
    assert len(launch.evaluation_runs) == len(context.materialized.candidates)
    assert context.materialized.preflight.snapshot.reservation_count == (
        context.materialized.preflight.logical_attempt_count
        + context.materialized.preflight.index_build_count
    )
    for candidate in context.materialized.candidates:
        run_id = candidate.evaluation_plan.run_id
        assert _SAFE_ID.fullmatch(run_id)
        assert candidate.settings.data_root.resolve() == context.workspace
        assert candidate.evaluation_plan.identity.runtime_configuration_id == (
            candidate.settings.runtime_configuration_identity
        )
        assert (tmp_path / "runs" / run_id / "manifest.json").is_file()


@pytest.mark.asyncio
async def test_missing_index_reservation_fails_before_install_and_closes_runtime(
    tmp_path: Path,
) -> None:
    settings, repositories, catalog = _catalog_stack(tmp_path)
    launch = catalog.prepare("comparison-generation-2", REGISTERED_GENERATION_PLAN_ID)
    repositories.comparisons.create(launch.suite, launch.evaluation_runs)
    context = cast(ProductionComparisonExecutionContext, launch.execution_context)
    key = EvaluationIndexReuseKey.from_plan(
        context.materialized.candidates[0].evaluation_plan,
        context.dataset,
    )
    installer = _InstallerFactory(_matching_revision(key))
    composition = _CompositionFactory()
    empty_budget = ProviderWorkBudget(
        context.materialized.plan.maximum_provider_calls,
        context.materialized.plan.maximum_cost,
        context.materialized.plan.pricing.currency,
    )
    materialized = replace(
        context.materialized,
        preflight=replace(context.materialized.preflight, budget=empty_budget),
    )
    invalid_launch = replace(
        launch,
        execution_context=replace(context, materialized=materialized),
    )
    executor = ProductionComparisonJobExecutor(
        settings=settings,
        evaluation_repository=repositories.evaluation_runs,
        comparison_repository=repositories.comparisons,
        run_artifacts_root=tmp_path / "runs",
        artifact_catalog=cast(ComparisonArtifactCatalog, object()),
        redactor=Redactor(),
        composition_factory=composition,
        corpus_installer_factory=installer,
    )

    with pytest.raises(
        ComparisonProductionError,
        match="provider_work_reservation_missing",
    ):
        await executor.execute(invalid_launch)

    assert installer.calls == 0
    assert composition.values and all(item.ingestion.closed for item in composition.values)
    assert all(item.status.value == "failed" for item in repositories.evaluation_runs.list())
    reconciled = repositories.comparisons.get(launch.suite.comparison_id)
    assert reconciled is not None and reconciled.status.value == "failed"
    assert reconciled.safe_error_code == "result-provider_work_reservation_missing"
    assert all(item.latest.status.value == "failed" for item in reconciled.candidates)
    assert all(
        item.latest.safe_error_code == "provider_work_reservation_missing"
        for item in reconciled.candidates
    )
    setup = repositories.comparisons.get_shared_setup(launch.suite.comparison_id)
    assert setup is not None
    assert setup.status.value == "failed"
    assert setup.safe_error_code == "provider_work_reservation_missing"
    assert setup.provider_call_count == 0
    assert setup.attempts == ()


@pytest.mark.asyncio
async def test_pre_provider_isolation_failure_persists_zero_attempt_setup(
    tmp_path: Path,
) -> None:
    settings, repositories, catalog = _catalog_stack(tmp_path)
    launch = catalog.prepare("comparison-isolation", REGISTERED_GENERATION_PLAN_ID)
    repositories.comparisons.create(launch.suite, launch.evaluation_runs)
    context = cast(ProductionComparisonExecutionContext, launch.execution_context)
    key = EvaluationIndexReuseKey.from_plan(
        context.materialized.candidates[0].evaluation_plan,
        context.dataset,
    )
    installer = _InstallerFactory(_matching_revision(key))
    composition = _WrongWorkspaceCompositionFactory()
    executor = ProductionComparisonJobExecutor(
        settings=settings,
        evaluation_repository=repositories.evaluation_runs,
        comparison_repository=repositories.comparisons,
        run_artifacts_root=tmp_path / "runs",
        artifact_catalog=cast(ComparisonArtifactCatalog, object()),
        redactor=Redactor(),
        composition_factory=composition,
        corpus_installer_factory=installer,
    )

    with pytest.raises(
        ComparisonProductionError,
        match="comparison-index-isolation-failed",
    ):
        await executor.execute(launch)

    assert installer.calls == 0
    setup = repositories.comparisons.get_shared_setup(launch.suite.comparison_id)
    assert setup is not None
    assert setup.status.value == "failed"
    assert setup.safe_error_code == "comparison-index-isolation-failed"
    assert setup.provider_call_count == 0
    assert setup.attempts == ()
    reconciled = repositories.comparisons.get(launch.suite.comparison_id)
    assert reconciled is not None and reconciled.status.value == "failed"
    assert all(item.latest.status.value == "failed" for item in reconciled.candidates)
    assert all(item.status.value == "failed" for item in repositories.evaluation_runs.list())
    assert composition.values and all(item.ingestion.closed for item in composition.values)
    reopened = RuntimeRepositories.from_database(
        Database(DataLayout.from_root(settings.data_root).metadata_db)
    )
    assert reopened.comparisons.get_shared_setup(launch.suite.comparison_id) == setup
    assert reopened.comparisons.get(launch.suite.comparison_id) == reconciled


@pytest.mark.asyncio
async def test_post_embedding_revision_failure_preserves_failed_setup_ledger(
    tmp_path: Path,
) -> None:
    settings, repositories, catalog = _catalog_stack(tmp_path)
    launch = catalog.prepare("comparison-revision", REGISTERED_GENERATION_PLAN_ID)
    repositories.comparisons.create(launch.suite, launch.evaluation_runs)
    context = cast(ProductionComparisonExecutionContext, launch.execution_context)
    key = EvaluationIndexReuseKey.from_plan(
        context.materialized.candidates[0].evaluation_plan,
        context.dataset,
    )
    canonical_revision = _matching_revision(key)
    foreign_revision = canonical_revision.model_copy(update={"revision_id": "revision-foreign"})
    installer = _RetryingInstallerFactory(
        canonical_revision,
        installed_revision=foreign_revision,
    )
    composition = _CompositionFactory()
    executor = ProductionComparisonJobExecutor(
        settings=settings,
        evaluation_repository=repositories.evaluation_runs,
        comparison_repository=repositories.comparisons,
        run_artifacts_root=tmp_path / "runs",
        artifact_catalog=cast(ComparisonArtifactCatalog, object()),
        redactor=Redactor(),
        composition_factory=composition,
        corpus_installer_factory=installer,
    )

    with pytest.raises(
        ComparisonProductionError,
        match="comparison-shared-setup-revision-mismatch",
    ):
        await executor.execute(launch)

    assert installer.calls == 1
    setup = repositories.comparisons.get_shared_setup(launch.suite.comparison_id)
    assert setup is not None
    assert setup.status.value == "failed"
    assert setup.safe_error_code == "comparison-shared-setup-revision-mismatch"
    assert setup.provider_call_count == 2
    assert all(item.source_run_id is None for item in setup.attempts)
    assert [item.evidence.status.value for item in setup.attempts] == [
        "failed",
        "succeeded",
    ]
    reconciled = repositories.comparisons.get(launch.suite.comparison_id)
    assert reconciled is not None and reconciled.status.value == "failed"
    assert all(item.latest.status.value == "failed" for item in reconciled.candidates)
    assert all(item.status.value == "failed" for item in repositories.evaluation_runs.list())
    assert composition.values and all(item.ingestion.closed for item in composition.values)


@pytest.mark.asyncio
async def test_setup_ledger_mismatch_persists_unavailable_calls_and_cost(
    tmp_path: Path,
) -> None:
    settings, repositories, catalog = _catalog_stack(tmp_path)
    launch = catalog.prepare("comparison-ledger-mismatch", REGISTERED_GENERATION_PLAN_ID)
    repositories.comparisons.create(launch.suite, launch.evaluation_runs)
    context = cast(ProductionComparisonExecutionContext, launch.execution_context)
    key = EvaluationIndexReuseKey.from_plan(
        context.materialized.candidates[0].evaluation_plan,
        context.dataset,
    )
    installer = _RetryingInstallerFactory(
        _matching_revision(key),
        persist_attempts=False,
    )
    composition = _CompositionFactory()
    executor = ProductionComparisonJobExecutor(
        settings=settings,
        evaluation_repository=repositories.evaluation_runs,
        comparison_repository=repositories.comparisons,
        run_artifacts_root=tmp_path / "runs",
        artifact_catalog=cast(ComparisonArtifactCatalog, object()),
        redactor=Redactor(),
        composition_factory=composition,
        corpus_installer_factory=installer,
    )

    with pytest.raises(
        ComparisonProductionError,
        match="comparison-shared-setup-ledger-mismatch",
    ):
        await executor.execute(launch)

    assert installer.calls == 1
    setup = repositories.comparisons.get_shared_setup(launch.suite.comparison_id)
    assert setup is not None
    assert setup.status.value == "failed"
    assert setup.provider_calls_complete is False
    assert isinstance(setup.provider_call_count, UnavailableValue)
    assert setup.known_partial_cost == 0
    assert setup.total_cost is None
    assert setup.cost_complete is False
    assert setup.unknown_reasons == ("setup-ledger-integrity-unavailable",)
    assert repositories.comparisons.get_result(launch.suite.comparison_id) is None
    reopened = RuntimeRepositories.from_database(
        Database(DataLayout.from_root(settings.data_root).metadata_db)
    )
    assert reopened.comparisons.get_shared_setup(launch.suite.comparison_id) == setup
    reconciled = reopened.comparisons.get(launch.suite.comparison_id)
    assert reconciled is not None and reconciled.status.value == "failed"
    assert all(item.latest.status.value == "failed" for item in reconciled.candidates)


@pytest.mark.asyncio
async def test_routed_setup_retry_is_persisted_unbound_and_cancellation_reconciles(
    tmp_path: Path,
) -> None:
    settings, repositories, catalog = _catalog_stack(tmp_path)
    launch = catalog.prepare("comparison-cancelled", REGISTERED_GENERATION_PLAN_ID)
    repositories.comparisons.create(launch.suite, launch.evaluation_runs)
    context = cast(ProductionComparisonExecutionContext, launch.execution_context)
    key = EvaluationIndexReuseKey.from_plan(
        context.materialized.candidates[0].evaluation_plan,
        context.dataset,
    )
    installer = _RetryingInstallerFactory(_matching_revision(key))
    composition = _CompositionFactory()
    executor = ProductionComparisonJobExecutor(
        settings=settings,
        evaluation_repository=repositories.evaluation_runs,
        comparison_repository=repositories.comparisons,
        run_artifacts_root=tmp_path / "runs",
        artifact_catalog=cast(ComparisonArtifactCatalog, object()),
        redactor=Redactor(),
        composition_factory=composition,
        corpus_installer_factory=installer,
        case_executor_factory=lambda candidate, services, redactor: _CancelledCaseExecutor(),
    )

    with pytest.raises(asyncio.CancelledError):
        await executor.execute(launch)

    assert installer.calls == 1
    setup = repositories.comparisons.get_shared_setup(launch.suite.comparison_id)
    assert setup is not None
    assert setup.status.value == "completed"
    assert setup.provider_call_count == 2
    assert setup.known_partial_cost > 0
    assert setup.total_cost is None
    assert setup.cost_complete is False
    assert setup.unknown_reasons == ("input-usage-unknown",)
    assert all(item.source_run_id is None for item in setup.attempts)
    assert [item.evidence.status.value for item in setup.attempts] == [
        "failed",
        "succeeded",
    ]
    reconciled = repositories.comparisons.get(launch.suite.comparison_id)
    assert reconciled is not None and reconciled.status.value == "failed"
    assert reconciled.safe_error_code == "result-comparison-interrupted"
    assert all(item.latest.status.value == "interrupted" for item in reconciled.candidates)
    assert all(
        item.status.value == "failed" and item.safe_error_code == "comparison-interrupted"
        for item in repositories.evaluation_runs.list()
    )
    assert composition.values and all(item.ingestion.closed for item in composition.values)


@pytest.mark.asyncio
async def test_unknown_usage_case_continues_later_schedule_and_selects_eligible_candidate(
    tmp_path: Path,
) -> None:
    settings, repositories, catalog = _catalog_stack(tmp_path)
    comparison_id = "comparison-unknown-usage-continuation"
    launch = catalog.prepare(comparison_id, REGISTERED_GENERATION_PLAN_ID)
    repositories.comparisons.create(launch.suite, launch.evaluation_runs)
    context = cast(ProductionComparisonExecutionContext, launch.execution_context)
    key = EvaluationIndexReuseKey.from_plan(
        context.materialized.candidates[0].evaluation_plan,
        context.dataset,
    )
    revision = _matching_revision(key)
    installer = _SuccessfulInstallerFactory(revision)
    composition = _CompositionFactory()
    executed: list[tuple[str, str]] = []
    timeout_variant_id = next(
        candidate.variant_id
        for candidate in context.materialized.candidates
        if candidate.evaluation_plan.identity.model_identities["generation"] == "gpt-5.4"
    )
    artifacts = ComparisonArtifactCatalog(tmp_path / "published")

    def case_executor_factory(
        candidate: MaterializedComparisonCandidate,
        services: QARuntimeServices,
        redactor: Redactor,
    ) -> _PersistingCaseExecutor:
        del services, redactor
        return _PersistingCaseExecutor(
            candidate=candidate,
            dataset=context.dataset,
            revision_id=revision.revision_id,
            executed=executed,
            timeout_variant_id=timeout_variant_id,
        )

    executor = ProductionComparisonJobExecutor(
        settings=settings,
        evaluation_repository=repositories.evaluation_runs,
        comparison_repository=repositories.comparisons,
        run_artifacts_root=tmp_path / "runs",
        artifact_catalog=artifacts,
        redactor=Redactor(),
        composition_factory=composition,
        corpus_installer_factory=installer,
        case_executor_factory=case_executor_factory,
    )

    await executor.execute(launch)

    suite = repositories.comparisons.get(comparison_id)
    result = repositories.comparisons.get_result(comparison_id)
    setup = repositories.comparisons.get_shared_setup(comparison_id)
    assert suite is not None and suite.status.value == "completed"
    assert all(item.latest.status.value == "completed" for item in suite.candidates)
    assert result is not None and setup is not None
    assert setup.known_partial_cost == Decimal("0.00001504")
    assert setup.cost_complete is True
    timed_out = next(
        item for item in result.candidates if item.reference.variant_id == timeout_variant_id
    )
    assert timed_out.failed_case_count >= 1
    assert timed_out.total_cost is None
    assert timed_out.cost_complete is False
    assert timed_out.cost_unknown_reasons == ("input-usage-unknown",)
    assert timed_out.known_partial_cost == Decimal("0.01870706")
    exact_candidate = next(
        item for item in result.candidates if item.reference.variant_id != timeout_variant_id
    )
    assert exact_candidate.known_partial_cost == Decimal("0.00277540")
    assert exact_candidate.total_cost == Decimal("0.00277540")
    assert exact_candidate.cost_complete is True
    assert result.known_partial_cost == setup.known_partial_cost + sum(
        (item.known_partial_cost for item in result.candidates),
        start=Decimal(0),
    )
    assert result.known_partial_cost == Decimal("0.02149750")
    assert result.total_cost is None
    assert result.cost_complete is False
    assert "comparison-cost-lower-bound-only" in result.recommendation.rationale_codes
    selected_variant_id = result.recommendation.selected_variant_id
    assert selected_variant_id is not None
    selection = repositories.comparisons.get_selection(ExperimentAxis.GENERATION_MODEL)
    assert selection is not None
    assert selection.selected_variant_id == selected_variant_id
    target_case = next(
        case.case_id
        for case in context.materialized.candidates[0].evaluation_plan.cases
        if case.source_case_id == "accept-zh-004" and case.repeat_index == 1
    )
    target_index = executed.index((timeout_variant_id, target_case))
    assert target_index < len(executed) - 1
    assert any(variant_id == timeout_variant_id for variant_id, _ in executed[target_index + 1 :])
    reopened_database = Database(DataLayout.from_root(settings.data_root).metadata_db)
    reopened = RuntimeRepositories.from_database(reopened_database)
    reopened_suite = reopened.comparisons.get(comparison_id)
    reopened_result = reopened.comparisons.get_result(comparison_id)
    reopened_setup = reopened.comparisons.get_shared_setup(comparison_id)
    assert reopened_suite == suite
    assert reopened_result == result
    assert reopened_setup == setup
    reopened_entry = ComparisonRunEntry.from_suite(reopened_suite, reopened_setup)
    assert reopened_entry.provider_calls == result.provider_call_count
    assert reopened_entry.known_partial_cost == Decimal("0.02149750")
    assert reopened_entry.incurred_cost is None
    assert reopened_entry.cost_complete is False
    assert reopened_entry.cost_unknown_reasons == ("input-usage-unknown",)
    summary = ComparisonSummary.from_evidence(
        reopened_suite,
        reopened_result,
        reopened_setup,
    )
    assert summary.known_partial_cost == Decimal("0.02149750")
    assert summary.total_cost == UnavailableValue(reason="comparison-cost-incomplete")
    assert summary.cost_complete is False
    assert summary.cost_unknown_reasons == ("input-usage-unknown",)
    assert summary.gate_status == "passed"
    assert summary.recommendation.selected_candidate_id == selected_variant_id

    reopened_artifacts = ComparisonArtifactCatalog(tmp_path / "published")
    manifest = reopened_artifacts.manifest(comparison_id)
    assert manifest is not None
    for artifact_id in (
        "comparison-report-json",
        "comparison-report-html",
        "comparison-report-txt",
        "comparison-report-csv",
    ):
        report = reopened_artifacts.resolve(comparison_id, artifact_id)
        assert report is not None
        assert b"0.02149750" in report.content
        assert b"0.01870706" in report.content
        assert b"0.00277540" in report.content
        assert b"input-usage-unknown" in report.content
    assert composition.values and all(item.ingestion.closed for item in composition.values)


def test_suite_workspace_rejects_existing_symlink_ancestor(tmp_path: Path) -> None:
    settings, _, catalog = _catalog_stack(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    evaluations = settings.data_root / "evaluations"
    try:
        evaluations.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(ComparisonProductionError, match="comparison-workspace-unsafe"):
        catalog.prepare("comparison-symlink", REGISTERED_GENERATION_PLAN_ID)
