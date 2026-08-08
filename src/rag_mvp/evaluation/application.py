"""Shared evaluation catalog, queue, supervisor, and privacy-safe read service."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, Protocol
from uuid import uuid4

from pydantic import AwareDatetime, Field

from rag_mvp.config.settings import Settings
from rag_mvp.domain._base import DomainModel, Identifier, utc_now
from rag_mvp.domain.evaluation import EvaluationRun, EvaluationRunStatus
from rag_mvp.domain.qa import StreamEventKind
from rag_mvp.evaluation.artifacts_v2 import (
    ArtifactCatalogV2,
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    artifact_download_filename_v2,
    artifact_media_type_v2,
)
from rag_mvp.evaluation.comparison import (
    ComparisonArtifactManifest,
    ComparisonCandidateStatus,
    ComparisonResult,
    ComparisonStatus,
)
from rag_mvp.evaluation.comparison_application import (
    ComparisonApplicationError,
    ComparisonArtifactManifestView,
    ComparisonArtifactStore,
    ComparisonCapacityError,
    ComparisonConflictError,
    ComparisonJobExecutor,
    ComparisonLaunchCatalog,
    ComparisonNotFoundError,
    ComparisonPlanCatalogEntry,
    ComparisonRunEntry,
    ComparisonRunStore,
    ComparisonSummary,
    ComparisonUnavailableError,
    ComparisonValidationError,
    PreparedComparisonLaunch,
    ResolvedComparisonDownload,
)
from rag_mvp.evaluation.dataset import EvaluationDataset
from rag_mvp.evaluation.json_report import canonical_json_value, decode_json_report
from rag_mvp.evaluation.plan import (
    EvaluationDatasetRegistry,
    EvaluationPlanError,
    build_evaluation_plan,
)
from rag_mvp.evaluation.report_dispatch import validate_versioned_report
from rag_mvp.evaluation.runner import (
    EvaluationRunner,
    EvaluationRunPlan,
    PersistedCaseResult,
)

STANDARD_EVALUATION_PLAN_ID: Literal["standard-evaluation-v1"] = "standard-evaluation-v1"
STANDARD_EVALUATION_PLAN_VERSION: Literal["1.0.0"] = "1.0.0"
_SAFE_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_SAFE_COMPARISON_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")
_TERMINAL_STATUSES = frozenset(
    {
        EvaluationRunStatus.COMPLETED,
        EvaluationRunStatus.FAILED,
        EvaluationRunStatus.INVALID,
    }
)


class EvaluationApplicationError(RuntimeError):
    """A stable, content-free application-service failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class EvaluationConflictError(EvaluationApplicationError):
    """The same immutable launch is already active."""


class EvaluationCapacityError(EvaluationApplicationError):
    """The separate paid-evaluation supervisor is at capacity."""


class EvaluationRunStore(Protocol):
    def create(self, run: EvaluationRun) -> None: ...

    def get(self, run_id: str) -> EvaluationRun | None: ...

    def update(self, run: EvaluationRun) -> None: ...

    def list(self) -> list[EvaluationRun]: ...


class EvaluationJobExecutor(Protocol):
    async def execute(self, plan: EvaluationRunPlan, dataset: EvaluationDataset) -> None: ...


class EvaluationDatasetCatalogEntry(DomainModel):
    dataset_id: Identifier
    dataset_version: Identifier
    schema_version: Identifier
    content_hash: Identifier
    corpus_version: Identifier
    corpus_hash: Identifier
    case_count: int = Field(gt=0)
    languages: tuple[Identifier, ...]


class EvaluationPlanCatalogEntry(DomainModel):
    plan_id: Literal["standard-evaluation-v1"] = STANDARD_EVALUATION_PLAN_ID
    plan_version: Literal["1.0.0"] = STANDARD_EVALUATION_PLAN_VERSION
    kind: Literal["standard-evaluation"] = "standard-evaluation"
    dataset_id: Identifier
    dataset_version: Identifier
    planned_case_count: int = Field(gt=0)
    candidate_count: Literal[1] = 1
    maximum_logical_calls: int = Field(gt=0)
    maximum_provider_calls: int = Field(gt=0)
    cache_policy: Literal["bypass"] = "bypass"
    cost_estimate_status: Literal["unavailable"] = "unavailable"
    cost_estimate: None = None
    cost_cap: None = None
    maximum_active_jobs: int = Field(gt=0)


class EvaluationRunSummary(DomainModel):
    run_id: Identifier
    status: EvaluationRunStatus
    dataset_id: Identifier
    dataset_version: Identifier
    dataset_hash: Identifier
    corpus_version: Identifier
    corpus_hash: Identifier | None
    plan_id: Literal["standard-evaluation-v1"] = STANDARD_EVALUATION_PLAN_ID
    plan_version: Literal["1.0.0"] = STANDARD_EVALUATION_PLAN_VERSION
    configuration_id: Identifier
    code_revision: Identifier
    cache_policy: Identifier
    total_cases: int = Field(ge=0)
    completed_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    remaining_cases: int = Field(ge=0)
    safe_error_code: str | None = None
    evidence_status: Literal["available", "incomplete", "unavailable"]
    gate_status: Literal["passed", "failed", "unavailable"] = "unavailable"
    completed_at: AwareDatetime | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @classmethod
    def from_run(
        cls,
        run: EvaluationRun,
        *,
        corpus_hash: str | None,
        evidence_status: Literal["available", "incomplete", "unavailable"],
        gate_status: Literal["passed", "failed", "unavailable"],
    ) -> EvaluationRunSummary:
        finished = run.completed_cases + run.failed_cases
        return cls(
            run_id=run.run_id,
            status=run.status,
            dataset_id=run.dataset_id,
            dataset_version=run.dataset_version,
            dataset_hash=run.dataset_hash,
            corpus_version=run.corpus_version,
            corpus_hash=corpus_hash,
            configuration_id=run.configuration_id,
            code_revision=run.code_revision,
            cache_policy=run.cache_policy,
            total_cases=run.total_cases,
            completed_cases=run.completed_cases,
            failed_cases=run.failed_cases,
            remaining_cases=max(0, run.total_cases - finished),
            safe_error_code=run.safe_error_code,
            evidence_status=evidence_status,
            gate_status=gate_status,
            completed_at=run.updated_at if run.status in _TERMINAL_STATUSES else None,
            created_at=run.created_at,
            updated_at=run.updated_at,
        )


class FailedMetricContribution(DomainModel):
    metric_id: Identifier
    status: Literal["passed", "failed", "unavailable"]
    value: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    numerator: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    denominator: int | None = Field(default=None, gt=0)


class FailedCaseDiagnostic(DomainModel):
    case_id: Identifier
    safe_error_code: Identifier
    request_id: Identifier | None = None
    trace_id: Identifier | None = None
    outcome: Identifier | None = None
    refusal_reason: Identifier | None = None
    citation_chunk_ids: tuple[Identifier, ...] = ()
    tags: Annotated[tuple[Identifier, ...], Field(max_length=32)] = ()
    metric_contributions: Annotated[
        tuple[FailedMetricContribution, ...],
        Field(max_length=32),
    ] = ()


class EvaluationArtifactDescriptor(DomainModel):
    artifact_id: Identifier
    schema_version: Identifier
    format: Identifier
    media_type: Identifier
    sha256_digest: Identifier
    byte_size: int = Field(ge=0)
    created_at: AwareDatetime


class EvaluationArtifactManifest(DomainModel):
    run_id: Identifier
    configuration_id: Identifier
    manifest_content_hash: Identifier
    artifacts: tuple[EvaluationArtifactDescriptor, ...]


class ReleaseMetricEvidence(DomainModel):
    """One read-only metric projected from a sealed pre-v2 release."""

    metric_id: Identifier
    value: float | None = Field(default=None, allow_inf_nan=False)
    threshold: float | None = Field(default=None, allow_inf_nan=False)
    operator: str | None = None
    denominator: int | None = Field(default=None, ge=0)
    passed: bool | None = None
    scorer_version: str | None = None


class ReleasePerformanceEvidence(DomainModel):
    """Accepted load/security evidence retained by a sealed legacy release."""

    attempts: int = Field(gt=0)
    successes: int = Field(ge=0)
    errors: int = Field(ge=0)
    configured_concurrency: int = Field(gt=0)
    observed_peak_concurrency: int = Field(gt=0)
    p50_ms: float = Field(ge=0, allow_inf_nan=False)
    p90_ms: float = Field(ge=0, allow_inf_nan=False)
    p95_ms: float = Field(ge=0, allow_inf_nan=False)
    p99_ms: float = Field(ge=0, allow_inf_nan=False)
    provider_attempt_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_cost: Decimal = Field(ge=0)
    cost_per_1000_attempts: Decimal = Field(ge=0)
    cost_per_1000_successes: Decimal = Field(ge=0)
    currency: Identifier
    refusals: int = Field(ge=0)
    answered_requests: int = Field(ge=0)
    security_passed: bool


class ReleaseEvidenceSnapshot(DomainModel):
    """Validated, path-free evidence used to render one sealed release."""

    release_id: Identifier
    source_schema_version: Identifier
    run: EvaluationRun
    corpus_hash: Identifier
    gate_passed: bool
    quality_metrics: tuple[ReleaseMetricEvidence, ...]
    performance: ReleasePerformanceEvidence
    artifact_manifest: EvaluationArtifactManifest


class ReleaseEvidenceStore(Protocol):
    def list(self) -> tuple[ReleaseEvidenceSnapshot, ...]: ...

    def get(self, run_id: str) -> ReleaseEvidenceSnapshot | None: ...

    def resolve(
        self,
        run_id: str,
        artifact_id: str,
    ) -> ResolvedEvaluationArtifact | None: ...


@dataclass(frozen=True, slots=True)
class ResolvedEvaluationArtifact:
    artifact_id: str
    content: bytes
    media_type: str
    filename: str

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("evaluation_artifact_empty")
        if not self.filename or any(character in self.filename for character in '\\/\r\n"'):
            raise ValueError("evaluation_artifact_filename_invalid")


class EvaluationArtifactStore(Protocol):
    def manifest(self, run_id: str) -> EvaluationArtifactManifest | None: ...

    def resolve(self, run_id: str, artifact_id: str) -> ResolvedEvaluationArtifact | None: ...


class EvaluationReportStore(Protocol):
    def resolve(
        self,
        run_id: str,
        report_format: Literal["json", "html"],
    ) -> ResolvedEvaluationArtifact | None: ...


@dataclass(frozen=True, slots=True)
class VerifiedEvaluationArtifactStore:
    """Adapt the hash-verifying v2 catalog to path-free application DTOs."""

    catalog: ArtifactCatalogV2

    def manifest(self, run_id: str) -> EvaluationArtifactManifest | None:
        try:
            manifest = self.catalog.get(run_id)
        except ArtifactNotFoundError:
            return None
        except ArtifactIntegrityError:
            raise EvaluationApplicationError("evaluation_artifact_integrity_failed") from None
        if manifest is None:
            return None
        return EvaluationArtifactManifest(
            run_id=manifest.run_id,
            configuration_id=manifest.configuration_id,
            manifest_content_hash=manifest.manifest_content_hash,
            artifacts=tuple(
                EvaluationArtifactDescriptor(
                    artifact_id=item.artifact_id,
                    schema_version=item.schema_version,
                    format=item.format,
                    media_type=item.media_type,
                    sha256_digest=item.sha256_digest,
                    byte_size=item.byte_size,
                    created_at=item.created_at,
                )
                for item in manifest.artifacts
            ),
        )

    def resolve(self, run_id: str, artifact_id: str) -> ResolvedEvaluationArtifact | None:
        try:
            resolved = self.catalog.resolve(run_id, artifact_id)
        except ArtifactNotFoundError:
            return None
        except ArtifactIntegrityError:
            raise EvaluationApplicationError("evaluation_artifact_integrity_failed") from None
        try:
            filename = artifact_download_filename_v2(resolved.descriptor.artifact_id)
        except ArtifactNotFoundError:
            raise EvaluationApplicationError("evaluation_artifact_integrity_failed") from None
        return ResolvedEvaluationArtifact(
            artifact_id=resolved.descriptor.artifact_id,
            content=resolved.content,
            media_type=resolved.descriptor.media_type,
            filename=filename,
        )


def _default_run_id() -> str:
    return f"eval_{uuid4().hex}"


def _default_comparison_id() -> str:
    return f"comparison_{uuid4().hex}"


@dataclass(slots=True)
class EvaluationApplicationService:
    """One application boundary shared by API, workbench, CLI, and acceptance flows."""

    registry: EvaluationDatasetRegistry
    settings: Settings
    repository: EvaluationRunStore
    run_artifacts_root: Path
    executor: EvaluationJobExecutor
    maximum_active_jobs: int = 1
    shutdown_grace_seconds: float = 2.0
    artifact_store: EvaluationArtifactStore | None = None
    legacy_report_store: EvaluationReportStore | None = None
    release_store: ReleaseEvidenceStore | None = None
    clock: Callable[[], datetime] = field(default=utc_now, repr=False)
    run_id_factory: Callable[[], str] = field(default=_default_run_id, repr=False)
    plan_settings_factory: Callable[[str], Settings] | None = field(
        default=None,
        repr=False,
    )
    comparison_catalog: ComparisonLaunchCatalog | None = None
    comparison_repository: ComparisonRunStore | None = None
    comparison_executor: ComparisonJobExecutor | None = None
    comparison_artifact_store: ComparisonArtifactStore | None = None
    comparison_id_factory: Callable[[], str] = field(
        default=_default_comparison_id,
        repr=False,
    )
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict, init=False, repr=False)
    _launches: dict[tuple[str, str, str, str], str] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _comparison_launches: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _accepting: bool = field(default=False, init=False, repr=False)
    _reconciled: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.run_artifacts_root = Path(self.run_artifacts_root).resolve()
        if self.maximum_active_jobs < 1:
            raise ValueError("evaluation_active_limit_invalid")
        if not 0 <= self.shutdown_grace_seconds <= 10:
            raise ValueError("evaluation_shutdown_grace_invalid")
        configured = (
            self.comparison_catalog,
            self.comparison_repository,
            self.comparison_executor,
        )
        if any(item is not None for item in configured) and any(
            item is None for item in configured
        ):
            raise ValueError("comparison_service_configuration_incomplete")

    async def startup(self) -> None:
        """Reconcile work whose provider outcome became unknowable after a restart."""

        async with self._lock:
            if not self._reconciled:
                for run in self.repository.list():
                    if run.status in {
                        EvaluationRunStatus.QUEUED,
                        EvaluationRunStatus.RUNNING,
                    }:
                        self.repository.update(
                            _updated_run(
                                run,
                                clock=self.clock,
                                status=EvaluationRunStatus.FAILED,
                                safe_error_code="evaluation_interrupted",
                            )
                        )
                if self.comparison_repository is not None:
                    for suite in self.comparison_repository.list():
                        if suite.status in {
                            ComparisonStatus.QUEUED,
                            ComparisonStatus.RUNNING,
                        }:
                            for history in suite.candidates:
                                if history.latest.status in {
                                    ComparisonCandidateStatus.COMPLETED,
                                    ComparisonCandidateStatus.FAILED,
                                    ComparisonCandidateStatus.INTERRUPTED,
                                }:
                                    continue
                                latest = history.latest
                                suite = suite.transition_candidate(
                                    history.reference.variant_id,
                                    status=ComparisonCandidateStatus.INTERRUPTED,
                                    completed_cases=latest.completed_cases,
                                    failed_cases=latest.failed_cases,
                                    provider_calls=latest.provider_calls,
                                    incurred_cost=latest.incurred_cost,
                                    known_partial_cost=latest.known_partial_cost,
                                    cost_complete=latest.cost_complete,
                                    cost_unknown_reasons=latest.cost_unknown_reasons,
                                    currency=latest.currency,
                                    safe_error_code="comparison-interrupted",
                                    recorded_at=max(
                                        self.clock(),
                                        suite.updated_at + timedelta(microseconds=1),
                                    ),
                                )
                                self.comparison_repository.append(suite)
                            suite = suite.fail(
                                "comparison-interrupted",
                                recorded_at=max(
                                    self.clock(),
                                    suite.updated_at + timedelta(microseconds=1),
                                ),
                            )
                            self.comparison_repository.append(suite)
                self._reconciled = True
            self._accepting = True

    async def close(self) -> None:
        """Stop new launches and terminally reconcile every owned background job."""

        async with self._lock:
            self._accepting = False
            tasks = tuple(self._tasks.values())
        if not tasks:
            return
        try:
            _, pending = await asyncio.wait(tasks, timeout=self.shutdown_grace_seconds)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def datasets(self) -> tuple[EvaluationDatasetCatalogEntry, ...]:
        values: list[EvaluationDatasetCatalogEntry] = []
        try:
            datasets = self.registry.list()
        except EvaluationPlanError as error:
            raise EvaluationApplicationError(error.code) from None
        released_identities: set[tuple[str, str, str, str, str]] | None = None
        if self.release_store is not None:
            try:
                released_identities = {
                    (
                        snapshot.run.dataset_id,
                        snapshot.run.dataset_version,
                        snapshot.run.dataset_hash,
                        snapshot.run.corpus_version,
                        snapshot.corpus_hash,
                    )
                    for snapshot in self.release_store.list()
                }
            except Exception:
                raise EvaluationApplicationError("evaluation_release_catalog_unavailable") from None
        for dataset in datasets:
            manifest = dataset.manifest
            identity = (
                manifest.dataset_id,
                manifest.version,
                manifest.content_hash,
                dataset.corpus.manifest.version,
                dataset.corpus.manifest.content_hash,
            )
            if released_identities is not None and identity not in released_identities:
                continue
            schema_version = str(getattr(manifest, "schema_version", "unknown"))
            values.append(
                EvaluationDatasetCatalogEntry(
                    dataset_id=manifest.dataset_id,
                    dataset_version=manifest.version,
                    schema_version=schema_version,
                    content_hash=manifest.content_hash,
                    corpus_version=dataset.corpus.manifest.version,
                    corpus_hash=dataset.corpus.manifest.content_hash,
                    case_count=len(dataset.cases),
                    languages=tuple(sorted(language.value for language in dataset.language_counts)),
                )
            )
        return tuple(values)

    def plans(self) -> tuple[EvaluationPlanCatalogEntry, ...]:
        maximum_operations_per_case = 5 + self.settings.context_chunk_limit
        maximum_attempts = self.settings.provider_retry_limit + 1
        return tuple(
            EvaluationPlanCatalogEntry(
                dataset_id=item.dataset_id,
                dataset_version=item.dataset_version,
                planned_case_count=item.case_count,
                maximum_logical_calls=item.case_count,
                maximum_provider_calls=(
                    item.case_count * maximum_operations_per_case * maximum_attempts
                ),
                maximum_active_jobs=self.maximum_active_jobs,
            )
            for item in self.datasets()
        )

    def comparison_plans(self) -> tuple[ComparisonPlanCatalogEntry, ...]:
        if self.comparison_catalog is None:
            return ()
        try:
            return tuple(self.comparison_catalog.list())
        except Exception as error:
            code = _safe_code(
                getattr(error, "code", None),
                "comparison_catalog_unavailable",
            )
            raise ComparisonUnavailableError(code) from None

    async def start_comparison(self, experiment_plan_id: str) -> ComparisonRunEntry:
        if (
            not isinstance(experiment_plan_id, str)
            or _SAFE_COMPARISON_ID.fullmatch(experiment_plan_id) is None
        ):
            raise ComparisonValidationError("comparison_plan_invalid")
        catalog = self.comparison_catalog
        repository = self.comparison_repository
        executor = self.comparison_executor
        if catalog is None or repository is None or executor is None:
            raise ComparisonUnavailableError("comparison_unavailable")
        async with self._lock:
            if not self._accepting:
                raise ComparisonUnavailableError("comparison_unavailable")
            if experiment_plan_id in self._comparison_launches:
                raise ComparisonConflictError("comparison_duplicate")
            if len(self._tasks) >= self.maximum_active_jobs:
                raise ComparisonCapacityError("comparison_capacity")
            comparison_id = self.comparison_id_factory()
            if (
                not isinstance(comparison_id, str)
                or _SAFE_COMPARISON_ID.fullmatch(comparison_id) is None
            ):
                raise ComparisonUnavailableError("comparison_id_invalid")
            try:
                launch = catalog.prepare(comparison_id, experiment_plan_id)
            except ComparisonApplicationError:
                raise
            except Exception as error:
                raise _comparison_start_error(error, phase="catalog") from None
            if (
                not isinstance(launch, PreparedComparisonLaunch)
                or launch.suite.comparison_id != comparison_id
                or launch.suite.plan.plan_id != experiment_plan_id
            ):
                raise ComparisonValidationError("comparison_plan_invalid")
            try:
                repository.create(launch.suite, launch.evaluation_runs)
            except ComparisonApplicationError:
                raise
            except Exception as error:
                raise _comparison_start_error(error, phase="repository") from None
            task_key = f"comparison:{comparison_id}"
            task = asyncio.create_task(
                self._execute_comparison(launch, experiment_plan_id, task_key),
                name=f"comparison-{comparison_id}",
            )
            self._tasks[task_key] = task
            self._comparison_launches[experiment_plan_id] = comparison_id
            return ComparisonRunEntry.from_suite(
                launch.suite,
                repository.get_shared_setup(comparison_id),
            )

    def get_comparison(self, comparison_id: str) -> ComparisonRunEntry | None:
        repository = self.comparison_repository
        if repository is None:
            return None
        suite = repository.get(comparison_id)
        return (
            None
            if suite is None
            else ComparisonRunEntry.from_suite(
                suite,
                repository.get_shared_setup(comparison_id),
            )
        )

    def list_comparisons(self) -> tuple[ComparisonRunEntry, ...]:
        repository = self.comparison_repository
        if repository is None:
            return ()
        return tuple(
            ComparisonRunEntry.from_suite(
                item,
                repository.get_shared_setup(item.comparison_id),
            )
            for item in repository.list()
        )

    def comparison_summary(self, comparison_id: str) -> ComparisonSummary | None:
        repository = self.comparison_repository
        if repository is None:
            return None
        suite = repository.get(comparison_id)
        if suite is None:
            return None
        result = repository.get_result(comparison_id)
        shared_setup = repository.get_shared_setup(comparison_id)
        return ComparisonSummary.from_evidence(suite, result, shared_setup)

    def comparison_manifest(
        self,
        comparison_id: str,
    ) -> ComparisonArtifactManifestView | None:
        repository = self.comparison_repository
        store = self.comparison_artifact_store
        if repository is None or store is None:
            return None
        suite = repository.get(comparison_id)
        if suite is None or suite.status is not ComparisonStatus.COMPLETED:
            return None
        result = repository.get_result(comparison_id)
        if result is None:
            return None
        ComparisonSummary.from_evidence(
            suite,
            result,
            repository.get_shared_setup(comparison_id),
        )
        manifest = store.manifest(comparison_id)
        if manifest is None:
            return None
        if (
            manifest.comparison_id != suite.comparison_id
            or manifest.plan_id != suite.plan.plan_id
            or manifest.plan_content_hash != suite.plan_content_hash
            or manifest.candidate_variant_ids
            != tuple(item.reference.variant_id for item in suite.candidates)
        ):
            raise ComparisonApplicationError("comparison_artifact_identity_failed")
        _validate_comparison_result_artifact(store, manifest, result)
        return ComparisonArtifactManifestView.from_manifest(manifest)

    def comparison_artifact(
        self,
        comparison_id: str,
        artifact_id: str,
    ) -> ResolvedComparisonDownload | None:
        store = self.comparison_artifact_store
        manifest = self.comparison_manifest(comparison_id)
        if store is None or manifest is None:
            return None
        descriptor = next(
            (item for item in manifest.artifacts if item.artifact_id == artifact_id),
            None,
        )
        if descriptor is None:
            return None
        resolved = store.resolve(comparison_id, artifact_id)
        if resolved is None or (
            resolved.descriptor.artifact_id != descriptor.artifact_id
            or resolved.descriptor.schema_version != descriptor.schema_version
            or resolved.descriptor.format != descriptor.format
            or resolved.descriptor.media_type != descriptor.media_type
            or resolved.descriptor.sha256_digest != descriptor.sha256_digest
            or resolved.descriptor.byte_size != descriptor.byte_size
        ):
            raise ComparisonApplicationError("comparison_artifact_integrity_failed")
        return ResolvedComparisonDownload.from_resolved(resolved)

    async def start(
        self,
        dataset_id: str,
        dataset_version: str | None = None,
        *,
        plan_id: str = STANDARD_EVALUATION_PLAN_ID,
    ) -> EvaluationRun:
        if plan_id != STANDARD_EVALUATION_PLAN_ID:
            raise EvaluationApplicationError("evaluation_plan_not_found")
        try:
            dataset = self.registry.resolve(dataset_id, dataset_version)
        except EvaluationPlanError as error:
            raise EvaluationApplicationError(error.code) from None
        identity = (
            dataset.manifest.dataset_id,
            dataset.manifest.version,
            plan_id,
            self.settings.evaluation_configuration_identity,
        )
        async with self._lock:
            if not self._accepting:
                raise EvaluationApplicationError("evaluation_unavailable")
            if identity in self._launches:
                raise EvaluationConflictError("evaluation_duplicate")
            if len(self._tasks) >= self.maximum_active_jobs:
                raise EvaluationCapacityError("evaluation_capacity")
            run_id = self.run_id_factory()
            try:
                plan_settings = (
                    self.settings
                    if self.plan_settings_factory is None
                    else self.plan_settings_factory(run_id)
                )
                plan = build_evaluation_plan(dataset, plan_settings, run_id)
                runner = EvaluationRunner(
                    self.repository,
                    self.run_artifacts_root,
                    None,
                    clock=self.clock,
                )
                queued = runner.queue(plan)
            except EvaluationPlanError as error:
                raise EvaluationApplicationError(error.code) from None
            except Exception as error:
                code = getattr(error, "code", "evaluation_queue_failed")
                raise EvaluationApplicationError(
                    _safe_code(code, "evaluation_queue_failed")
                ) from None
            task = asyncio.create_task(
                self._execute(plan, dataset, identity),
                name=f"evaluation-{run_id}",
            )
            self._tasks[run_id] = task
            self._launches[identity] = run_id
            return queued

    def get(self, run_id: str) -> EvaluationRun | None:
        persisted = self.repository.get(run_id)
        if persisted is not None or self.release_store is None:
            return persisted
        release = self.release_store.get(run_id)
        return None if release is None else release.run

    def get_run(self, run_id: str) -> EvaluationRun | None:
        return self.get(run_id)

    def list(self) -> tuple[EvaluationRun, ...]:
        persisted = tuple(reversed(self.repository.list()))
        if self.release_store is None:
            return persisted
        persisted_ids = {item.run_id for item in persisted}
        releases = tuple(
            item.run for item in self.release_store.list() if item.run.run_id not in persisted_ids
        )
        return (*releases, *persisted)

    def list_runs(self) -> tuple[EvaluationRun, ...]:
        return self.list()

    def summary(self, run_id: str) -> EvaluationRunSummary | None:
        persisted = self.repository.get(run_id)
        release = None if self.release_store is None else self.release_store.get(run_id)
        run = persisted if persisted is not None else None if release is None else release.run
        if run is None:
            return None
        if persisted is None and release is not None:
            return EvaluationRunSummary.from_run(
                run,
                corpus_hash=release.corpus_hash,
                evidence_status="available",
                gate_status="passed" if release.gate_passed else "failed",
            )
        runner = EvaluationRunner(self.repository, self.run_artifacts_root, None)
        try:
            corpus_hash = runner.load_manifest(run_id).identity.corpus_hash
        except (OSError, ValueError):
            corpus_hash = None
        evidence_status: Literal["available", "incomplete", "unavailable"] = (
            "incomplete"
            if (
                run.completed_cases + run.failed_cases
                and run.status is not EvaluationRunStatus.COMPLETED
            )
            else "unavailable"
        )
        gate_status: Literal["passed", "failed", "unavailable"] = "unavailable"
        if run.status is EvaluationRunStatus.COMPLETED:
            try:
                report = self.report(run_id, "json")
                if report is not None:
                    validated = _validated_report_document(run, report)
                    if "accepted" in validated:
                        accepted = validated.get("accepted")
                    else:
                        gate = validated.get("gate")
                        accepted = gate.get("final_passed") if isinstance(gate, dict) else None
                    if isinstance(accepted, bool):
                        evidence_status = "available"
                        gate_status = "passed" if accepted else "failed"
            except (RuntimeError, UnicodeError, ValueError):
                evidence_status = "unavailable"
                gate_status = "unavailable"
        return EvaluationRunSummary.from_run(
            run,
            corpus_hash=corpus_hash,
            evidence_status=evidence_status,
            gate_status=gate_status,
        )

    def failed_cases(self, run_id: str) -> tuple[FailedCaseDiagnostic, ...]:
        run = self.repository.get(run_id)
        if run is None:
            return ()
        runner = EvaluationRunner(self.repository, self.run_artifacts_root, None)
        try:
            results = runner.load_case_results(run_id)
        except (OSError, ValueError):
            return ()
        tags_by_case: dict[str, tuple[str, ...]] = {}
        try:
            dataset = self.registry.resolve(run.dataset_id, run.dataset_version)
            for case in dataset.cases:
                challenges = tuple(item.value for item in getattr(case, "challenge_tags", ()))
                tags_by_case[case.case_id] = (case.category.value, *challenges)
        except EvaluationPlanError:
            pass
        return tuple(
            diagnostic
            for result in results
            if (
                diagnostic := _failed_case_diagnostic(
                    result,
                    tags=tags_by_case.get(result.case_id, ()),
                )
            )
            is not None
        )

    def artifact_manifest(self, run_id: str) -> EvaluationArtifactManifest | None:
        run = self.get(run_id)
        if run is None:
            return None
        release = None if self.release_store is None else self.release_store.get(run_id)
        if self.repository.get(run_id) is None and release is not None:
            return release.artifact_manifest
        if self.artifact_store is None:
            return None
        manifest = self.artifact_store.manifest(run_id)
        if manifest is not None and (
            manifest.run_id != run.run_id or manifest.configuration_id != run.configuration_id
        ):
            raise EvaluationApplicationError("evaluation_artifact_identity_failed")
        return manifest

    def artifact(self, run_id: str, artifact_id: str) -> ResolvedEvaluationArtifact | None:
        manifest = self.artifact_manifest(run_id)
        if manifest is None:
            return None
        descriptor = next(
            (item for item in manifest.artifacts if item.artifact_id == artifact_id),
            None,
        )
        if descriptor is None:
            return None
        is_release = self.repository.get(run_id) is None
        if is_release:
            resolved = (
                None
                if self.release_store is None
                else self.release_store.resolve(run_id, artifact_id)
            )
        else:
            resolved = (
                None
                if self.artifact_store is None
                else self.artifact_store.resolve(run_id, artifact_id)
            )
        if resolved is None:
            raise EvaluationApplicationError("evaluation_artifact_integrity_failed")
        try:
            expected_filename = artifact_download_filename_v2(artifact_id)
            expected_media_type = artifact_media_type_v2(artifact_id)
        except ArtifactNotFoundError:
            raise EvaluationApplicationError("evaluation_artifact_integrity_failed") from None
        digest = f"sha256:{sha256(resolved.content).hexdigest()}"
        if (
            resolved.artifact_id != artifact_id
            or resolved.filename != expected_filename
            or resolved.media_type != expected_media_type
            or descriptor.media_type != expected_media_type
            or descriptor.byte_size != len(resolved.content)
            or descriptor.sha256_digest != digest
        ):
            raise EvaluationApplicationError("evaluation_artifact_integrity_failed")
        return resolved

    def release_evidence(self, run_id: str) -> ReleaseEvidenceSnapshot | None:
        """Return validated path-free legacy evidence for UI compatibility rendering."""

        if self.release_store is None:
            return None
        return self.release_store.get(run_id)

    def report(
        self,
        run_id: str,
        report_format: Literal["json", "html"],
    ) -> ResolvedEvaluationArtifact | None:
        run = self.get(run_id)
        if run is None or run.status is not EvaluationRunStatus.COMPLETED:
            return None
        manifest = self.artifact_manifest(run_id)
        if manifest is not None:
            report_artifact_id = f"evaluation-report-{report_format}"
            match = next(
                (
                    descriptor
                    for descriptor in manifest.artifacts
                    if descriptor.artifact_id == report_artifact_id
                ),
                None,
            )
            if match is None or match.format != report_format:
                raise EvaluationApplicationError("evaluation_artifact_integrity_failed")
            return self.artifact(run_id, match.artifact_id)
        if self.legacy_report_store is None:
            return None
        json_report = self.legacy_report_store.resolve(run_id, "json")
        if json_report is None:
            return None
        _validated_report_document(run, json_report)
        if report_format == "json":
            return json_report
        html_report = self.legacy_report_store.resolve(run_id, "html")
        if html_report is not None and (
            html_report.artifact_id != "evaluation-report-html"
            or html_report.media_type != "text/html"
            or html_report.filename != "evaluation-report.html"
        ):
            raise EvaluationApplicationError("evaluation_report_integrity_failed")
        return html_report

    def get_report(
        self,
        run_id: str,
        format: Literal["json", "html"],
    ) -> None:
        # The legacy Gradio adapter accepts local paths. Returning one would bypass
        # manifest verification; the typed artifact/API callback replaces it in 2.12.
        del run_id, format
        return None

    def compare_runs(
        self,
        baseline_run_id: str,
        candidate_run_id: str,
    ) -> Mapping[str, object]:
        baseline = self.get(baseline_run_id)
        candidate = self.get(candidate_run_id)
        if baseline is None or candidate is None:
            raise EvaluationApplicationError("evaluation_not_found")
        if (
            baseline.dataset_id,
            baseline.dataset_version,
            baseline.dataset_hash,
            baseline.corpus_version,
        ) != (
            candidate.dataset_id,
            candidate.dataset_version,
            candidate.dataset_hash,
            candidate.corpus_version,
        ):
            raise EvaluationApplicationError("evaluation_runs_incompatible")
        return {
            "baseline_run_id": baseline.run_id,
            "candidate_run_id": candidate.run_id,
            "baseline_status": baseline.status.value,
            "candidate_status": candidate.status.value,
        }

    async def wait(self, run_id: str) -> EvaluationRun | None:
        """CLI/test coordination; API and workbench polling remain read-only."""

        async with self._lock:
            task = self._tasks.get(run_id)
        if task is not None:
            await asyncio.shield(task)
        return self.get(run_id)

    async def wait_comparison(self, comparison_id: str) -> ComparisonRunEntry | None:
        """CLI/test coordination for one already admitted comparison."""

        async with self._lock:
            task = self._tasks.get(f"comparison:{comparison_id}")
        if task is not None:
            await asyncio.shield(task)
        return self.get_comparison(comparison_id)

    async def _execute_comparison(
        self,
        launch: PreparedComparisonLaunch,
        plan_id: str,
        task_key: str,
    ) -> None:
        repository = self.comparison_repository
        executor = self.comparison_executor
        comparison_id = launch.suite.comparison_id
        try:
            if repository is None or executor is None:
                raise ComparisonUnavailableError("comparison_unavailable")
            await executor.execute(launch)
            suite = repository.get(comparison_id)
            if suite is None or suite.status not in {
                ComparisonStatus.COMPLETED,
                ComparisonStatus.FAILED,
                ComparisonStatus.INVALID,
            }:
                self._fail_comparison(comparison_id, "comparison-execution-incomplete")
        except asyncio.CancelledError:
            self._fail_comparison(comparison_id, "comparison-interrupted")
            raise
        except Exception as error:
            self._fail_comparison(
                comparison_id,
                _safe_code(
                    getattr(error, "code", None),
                    "comparison-execution-failed",
                ),
            )
        finally:
            async with self._lock:
                self._tasks.pop(task_key, None)
                if self._comparison_launches.get(plan_id) == comparison_id:
                    self._comparison_launches.pop(plan_id, None)

    def _fail_comparison(self, comparison_id: str, code: str) -> None:
        repository = self.comparison_repository
        if repository is None:
            return
        suite = repository.get(comparison_id)
        if suite is None or suite.status in {
            ComparisonStatus.FAILED,
            ComparisonStatus.INVALID,
        }:
            return
        if suite.status is ComparisonStatus.COMPLETED and not code.startswith(
            ("aggregation-", "artifact-", "integrity-", "publication-", "result-")
        ):
            code = "result-finalization-failed"
        repository.append(
            suite.fail(
                code,
                recorded_at=max(
                    self.clock(),
                    suite.updated_at + timedelta(microseconds=1),
                ),
            )
        )

    async def _execute(
        self,
        plan: EvaluationRunPlan,
        dataset: EvaluationDataset,
        identity: tuple[str, str, str, str],
    ) -> None:
        try:
            await self.executor.execute(plan, dataset)
            run = self.repository.get(plan.run_id)
            if run is None or run.status not in _TERMINAL_STATUSES:
                self._fail_run(plan.run_id, "evaluation_execution_incomplete")
        except asyncio.CancelledError:
            self._fail_run(plan.run_id, "evaluation_interrupted")
            raise
        except Exception as error:
            code = _safe_code(getattr(error, "code", None), "evaluation_execution_failed")
            self._fail_run(plan.run_id, code)
        finally:
            async with self._lock:
                self._tasks.pop(plan.run_id, None)
                if self._launches.get(identity) == plan.run_id:
                    self._launches.pop(identity, None)

    def _fail_run(self, run_id: str, code: str) -> None:
        run = self.repository.get(run_id)
        if run is None or run.status in {
            EvaluationRunStatus.FAILED,
            EvaluationRunStatus.INVALID,
        }:
            return
        self.repository.update(
            _updated_run(
                run,
                clock=self.clock,
                status=EvaluationRunStatus.FAILED,
                safe_error_code=code,
            )
        )


def _updated_run(
    run: EvaluationRun,
    *,
    clock: Callable[[], datetime],
    status: EvaluationRunStatus,
    safe_error_code: str,
) -> EvaluationRun:
    return EvaluationRun.model_validate(
        {
            **run.model_dump(),
            "status": status,
            "safe_error_code": safe_error_code,
            "updated_at": clock(),
        }
    )


def _safe_code(value: object, fallback: str) -> str:
    return value if isinstance(value, str) and _SAFE_ERROR_CODE.fullmatch(value) else fallback


def _comparison_start_error(
    error: Exception,
    *,
    phase: Literal["catalog", "repository"],
) -> ComparisonApplicationError:
    code = _safe_code(
        getattr(error, "code", None),
        "comparison_start_invalid" if phase == "catalog" else "comparison_persistence_failed",
    )
    if type(error).__name__ == "RepositoryConflict":
        return ComparisonConflictError("comparison_duplicate")
    if code in {"comparison_plan_not_found", "comparison_not_found"}:
        return ComparisonNotFoundError("comparison_plan_not_found")
    if code == "comparison_capacity":
        return ComparisonCapacityError(code)
    if (
        code.endswith(("_missing", "_unknown", "_disabled", "_blocked"))
        or "_prerequisite_" in code
        or code == "comparison_registered_dataset_identity_mismatch"
    ):
        return ComparisonConflictError(code)
    if (
        code.endswith("_unavailable")
        or code in {"comparison_persistence_failed", "comparison_start_unavailable"}
        or (phase == "catalog" and code == "comparison_start_invalid")
    ):
        return ComparisonUnavailableError(code)
    return ComparisonValidationError(code)


def _validate_comparison_result_artifact(
    store: ComparisonArtifactStore,
    manifest: ComparisonArtifactManifest,
    result: ComparisonResult,
) -> None:
    descriptor = next(
        (item for item in manifest.artifacts if item.artifact_id == "comparison-report-json"),
        None,
    )
    if descriptor is None:
        raise ComparisonApplicationError("comparison_artifact_integrity_failed")
    try:
        resolved = store.resolve(manifest.comparison_id, descriptor.artifact_id)
        if resolved is None or resolved.descriptor != descriptor:
            raise ValueError("comparison_artifact_descriptor_mismatch")
        document = decode_json_report(resolved.content.decode("utf-8"))
        persisted = ComparisonResult.model_validate(document)
        canonical = (canonical_json_value(persisted.model_dump(mode="json")) + "\n").encode("utf-8")
        if persisted != result or resolved.content != canonical:
            raise ValueError("comparison_result_artifact_mismatch")
    except (UnicodeError, ValueError):
        raise ComparisonApplicationError("comparison_artifact_integrity_failed") from None


def _validated_report_document(
    run: EvaluationRun,
    artifact: ResolvedEvaluationArtifact,
) -> Mapping[str, object]:
    if (
        artifact.artifact_id != "evaluation-report-json"
        or artifact.media_type != "application/json"
        or artifact.filename != "evaluation-report.json"
    ):
        raise EvaluationApplicationError("evaluation_report_integrity_failed")
    try:
        document = decode_json_report(artifact.content.decode("utf-8"))
        if not isinstance(document, dict):
            raise ValueError("evaluation_report_invalid")
        validated = validate_versioned_report(document).document
    except (UnicodeError, ValueError):
        raise EvaluationApplicationError("evaluation_report_integrity_failed") from None
    report_configuration = validated.get("configuration_id")
    if report_configuration is None:
        provenance = validated.get("provenance")
        report_configuration = (
            provenance.get("configuration_id") if isinstance(provenance, dict) else None
        )
    if validated.get("run_id") != run.run_id or report_configuration != run.configuration_id:
        raise EvaluationApplicationError("evaluation_report_identity_failed")
    return validated


def _failed_case_diagnostic(
    result: PersistedCaseResult,
    *,
    tags: tuple[str, ...],
) -> FailedCaseDiagnostic | None:
    execution = result.execution
    if result.succeeded and (
        execution is None or execution.event.kind is not StreamEventKind.ERROR
    ):
        return None
    event = None if execution is None else execution.event
    safe_error_code = result.safe_error_code or "evaluation_case_failed"
    return FailedCaseDiagnostic(
        case_id=result.case_id,
        safe_error_code=_safe_code(safe_error_code, "evaluation_case_failed"),
        request_id=None if execution is None else execution.request_id,
        trace_id=None if event is None else event.diagnostics.trace_id,
        outcome=None if event is None else event.kind.value,
        refusal_reason=(None if event is None or event.reason is None else event.reason.value),
        citation_chunk_ids=(
            () if event is None else tuple(citation.chunk_id for citation in event.citations)
        ),
        tags=tags,
    )


@dataclass(frozen=True, slots=True)
class CallableEvaluationJobExecutor:
    """Small adapter for production composition and focused tests."""

    callback: Callable[[EvaluationRunPlan, EvaluationDataset], Awaitable[None]]

    async def execute(self, plan: EvaluationRunPlan, dataset: EvaluationDataset) -> None:
        await self.callback(plan, dataset)


__all__ = [
    "STANDARD_EVALUATION_PLAN_ID",
    "CallableEvaluationJobExecutor",
    "ComparisonApplicationError",
    "ComparisonCapacityError",
    "ComparisonConflictError",
    "ComparisonNotFoundError",
    "ComparisonUnavailableError",
    "ComparisonValidationError",
    "EvaluationApplicationError",
    "EvaluationApplicationService",
    "EvaluationArtifactDescriptor",
    "EvaluationArtifactManifest",
    "EvaluationArtifactStore",
    "EvaluationCapacityError",
    "EvaluationConflictError",
    "EvaluationDatasetCatalogEntry",
    "EvaluationJobExecutor",
    "EvaluationPlanCatalogEntry",
    "EvaluationReportStore",
    "EvaluationRunSummary",
    "FailedCaseDiagnostic",
    "FailedMetricContribution",
    "ResolvedEvaluationArtifact",
    "VerifiedEvaluationArtifactStore",
]
