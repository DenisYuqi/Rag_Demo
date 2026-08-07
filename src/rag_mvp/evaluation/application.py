"""Shared evaluation catalog, queue, supervisor, and privacy-safe read service."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
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
from rag_mvp.evaluation.dataset import EvaluationDataset
from rag_mvp.evaluation.json_report import decode_json_report
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
    clock: Callable[[], datetime] = field(default=utc_now, repr=False)
    run_id_factory: Callable[[], str] = field(default=_default_run_id, repr=False)
    plan_settings_factory: Callable[[str], Settings] | None = field(
        default=None,
        repr=False,
    )
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict, init=False, repr=False)
    _launches: dict[tuple[str, str, str, str], str] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _accepting: bool = field(default=False, init=False, repr=False)
    _reconciled: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.run_artifacts_root = Path(self.run_artifacts_root).resolve()
        if self.maximum_active_jobs < 1:
            raise ValueError("evaluation_active_limit_invalid")
        if not 0 <= self.shutdown_grace_seconds <= 10:
            raise ValueError("evaluation_shutdown_grace_invalid")

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
        for dataset in datasets:
            manifest = dataset.manifest
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
            self.settings.configuration_identity,
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
        return self.repository.get(run_id)

    def get_run(self, run_id: str) -> EvaluationRun | None:
        return self.get(run_id)

    def list(self) -> tuple[EvaluationRun, ...]:
        return tuple(reversed(self.repository.list()))

    def list_runs(self) -> tuple[EvaluationRun, ...]:
        return self.list()

    def summary(self, run_id: str) -> EvaluationRunSummary | None:
        run = self.get(run_id)
        if run is None:
            return None
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
        run = self.get(run_id)
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
        if run is None or self.artifact_store is None:
            return None
        manifest = self.artifact_store.manifest(run_id)
        if manifest is not None and (
            manifest.run_id != run.run_id or manifest.configuration_id != run.configuration_id
        ):
            raise EvaluationApplicationError("evaluation_artifact_identity_failed")
        return manifest

    def artifact(self, run_id: str, artifact_id: str) -> ResolvedEvaluationArtifact | None:
        if self.artifact_store is None:
            return None
        manifest = self.artifact_manifest(run_id)
        if manifest is None:
            return None
        descriptor = next(
            (item for item in manifest.artifacts if item.artifact_id == artifact_id),
            None,
        )
        if descriptor is None:
            return None
        resolved = self.artifact_store.resolve(run_id, artifact_id)
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
