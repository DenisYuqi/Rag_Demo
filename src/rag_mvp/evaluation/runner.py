"""Reproducible evaluation execution through the production QA boundary."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol, cast
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from rag_mvp.api.qa import QARuntimeServices, stream_qa_events
from rag_mvp.domain._base import DomainModel, Identifier, SafeScalar, utc_now
from rag_mvp.domain.evaluation import EvaluationRun, EvaluationRunStatus
from rag_mvp.domain.qa import (
    ConversationRole,
    StreamEventKind,
    ValidatedStreamEvent,
)
from rag_mvp.domain.retrieval import CachePolicy, RetrievalMode
from rag_mvp.qa.orchestrator import OrchestratedResponse
from rag_mvp.qa.query_rewrite import select_response_language
from rag_mvp.safety.redactor import Redactor

RUN_MANIFEST_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
RUNNER_VERSION: Literal["production-qa-runner-v1"] = "production-qa-runner-v1"
_SAFE_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class EvaluationRunnerError(RuntimeError):
    """A privacy-safe evaluation orchestration failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ImmutableRunError(EvaluationRunnerError):
    """Raised when an immutable artifact would be replaced."""


class EvaluationRunRepository(Protocol):
    def create(self, run: EvaluationRun) -> None: ...

    def get(self, run_id: str) -> EvaluationRun | None: ...

    def update(self, run: EvaluationRun) -> None: ...


class EvaluationConversationTurn(DomainModel):
    role: ConversationRole
    content: str = Field(min_length=1)


class EvaluationCaseInput(DomainModel):
    """The QA-sized portion of one already validated dataset case."""

    case_id: Identifier
    question: str = Field(min_length=1)
    language: Literal["zh", "en", "mixed"]
    history: tuple[EvaluationConversationTurn, ...] = ()
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID

    @field_validator("case_id")
    @classmethod
    def safe_case_id(cls, value: str) -> str:
        if _SAFE_ARTIFACT_ID.fullmatch(value) is None:
            raise ValueError("case_id is unsafe for immutable artifacts")
        return value


class EvaluationEnvironment(DomainModel):
    python_version: Identifier
    platform: Identifier
    deployment: Identifier


class EvaluationRunIdentity(DomainModel):
    """All pinned inputs needed to reproduce an acceptance run."""

    dataset_id: Identifier
    dataset_version: Identifier
    dataset_hash: Identifier
    corpus_version: Identifier
    corpus_hash: Identifier
    configuration_id: Identifier
    code_revision: Identifier
    prompt_versions: dict[str, str]
    provider_identities: dict[str, str]
    model_identities: dict[str, str]
    generation_settings: dict[str, SafeScalar]
    embedding_identity: dict[str, SafeScalar]
    chunking_identity: dict[str, SafeScalar]
    retrieval_configuration: dict[str, SafeScalar]
    scorer_versions: dict[str, str]
    pricing_version: Identifier
    random_seeds: dict[str, int]
    environment: EvaluationEnvironment
    cache_policy: Literal["bypass"] = "bypass"

    @model_validator(mode="after")
    def require_reproducible_identities(self) -> EvaluationRunIdentity:
        required_mappings: tuple[tuple[str, Mapping[str, object]], ...] = (
            ("prompt_versions", self.prompt_versions),
            ("provider_identities", self.provider_identities),
            ("model_identities", self.model_identities),
            ("generation_settings", self.generation_settings),
            ("embedding_identity", self.embedding_identity),
            ("chunking_identity", self.chunking_identity),
            ("retrieval_configuration", self.retrieval_configuration),
            ("scorer_versions", self.scorer_versions),
            ("random_seeds", self.random_seeds),
        )
        missing = [name for name, value in required_mappings if not value]
        if missing:
            raise ValueError("reproducibility identity is incomplete")
        return self


class EvaluationRunPlan(DomainModel):
    run_id: Identifier
    identity: EvaluationRunIdentity
    cases: tuple[EvaluationCaseInput, ...]

    @field_validator("run_id")
    @classmethod
    def safe_run_id(cls, value: str) -> str:
        if _SAFE_ARTIFACT_ID.fullmatch(value) is None:
            raise ValueError("run_id is unsafe for immutable artifacts")
        return value

    @model_validator(mode="after")
    def require_unique_cases(self) -> EvaluationRunPlan:
        case_ids = tuple(case.case_id for case in self.cases)
        if not case_ids:
            raise ValueError("an evaluation run requires at least one case")
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("evaluation case IDs must be unique")
        return self


class EvaluationRunManifest(DomainModel):
    schema_version: Literal["1.0.0"] = RUN_MANIFEST_SCHEMA_VERSION
    runner_version: Literal["production-qa-runner-v1"] = RUNNER_VERSION
    run_id: Identifier
    created_at: datetime
    case_ids: tuple[Identifier, ...]
    identity: EvaluationRunIdentity
    manifest_hash: Identifier

    @classmethod
    def create(
        cls,
        plan: EvaluationRunPlan,
        *,
        created_at: datetime,
    ) -> EvaluationRunManifest:
        case_ids = tuple(case.case_id for case in plan.cases)
        draft = cls.model_construct(
            schema_version=RUN_MANIFEST_SCHEMA_VERSION,
            runner_version=RUNNER_VERSION,
            run_id=plan.run_id,
            created_at=created_at,
            case_ids=case_ids,
            identity=plan.identity,
            manifest_hash="pending",
        )
        unhashed = draft.model_dump(mode="json", exclude={"manifest_hash"})
        digest = hashlib.sha256(_canonical_json(unhashed).encode("utf-8")).hexdigest()
        return cls(
            run_id=plan.run_id,
            created_at=created_at,
            case_ids=case_ids,
            identity=plan.identity,
            manifest_hash=digest,
        )

    @model_validator(mode="after")
    def verify_hash(self) -> EvaluationRunManifest:
        unhashed = self.model_dump(mode="json", exclude={"manifest_hash"})
        digest = hashlib.sha256(_canonical_json(unhashed).encode("utf-8")).hexdigest()
        if digest != self.manifest_hash:
            raise ValueError("run manifest hash mismatch")
        return self


class EvaluationCaseExecution(DomainModel):
    case_id: Identifier
    owner_id: Identifier
    session_id: Identifier
    request_id: Identifier
    event: ValidatedStreamEvent
    cache_policy: Literal["bypass"] = "bypass"
    retrieved_chunk_ids: tuple[Identifier, ...] = ()
    context_chunk_ids: tuple[Identifier, ...] = ()
    latency_ms: float = Field(ge=0, allow_inf_nan=False)


class PersistedCaseResult(DomainModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    run_id: Identifier
    case_id: Identifier
    succeeded: bool
    execution: EvaluationCaseExecution | None = None
    safe_error_code: str | None = None
    completed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_outcome(self) -> PersistedCaseResult:
        if self.succeeded and (self.execution is None or self.safe_error_code is not None):
            raise ValueError("a successful case requires execution evidence only")
        if not self.succeeded and self.safe_error_code is None:
            raise ValueError("a failed case requires a safe_error_code")
        return self


class EvaluationCaseExecutor(Protocol):
    async def execute(
        self,
        case: EvaluationCaseInput,
        *,
        owner_id: str,
        cache_policy: CachePolicy,
    ) -> EvaluationCaseExecution: ...


@dataclass(slots=True)
class _CapturingOrchestrator:
    delegate: object
    outcome: OrchestratedResponse | None = None

    async def run(self, **kwargs: object) -> OrchestratedResponse:
        run = getattr(self.delegate, "run", None)
        if not callable(run):
            raise EvaluationRunnerError("qa_orchestrator_unavailable")
        outcome = await run(**kwargs)
        if not isinstance(outcome, OrchestratedResponse):
            # Runtime duck types are useful for tests, while production remains strict.
            self.outcome = cast(OrchestratedResponse, outcome)
            return cast(OrchestratedResponse, outcome)
        self.outcome = outcome
        return outcome


@dataclass(frozen=True, slots=True)
class ProductionQAExecutor:
    """Invoke the same admission, deadline, safety, and serialization path as HTTP QA."""

    services: QARuntimeServices
    redactor: Redactor

    async def execute(
        self,
        case: EvaluationCaseInput,
        *,
        owner_id: str,
        cache_policy: CachePolicy,
    ) -> EvaluationCaseExecution:
        if cache_policy is not CachePolicy.BYPASS:
            raise EvaluationRunnerError("evaluation_cache_policy_invalid")
        session = self.services.conversations.create_session(owner_id)
        for turn in case.history:
            self.services.conversations.append_turn(
                session.session_id,
                owner_id,
                turn.role,
                turn.content,
            )
        request_id = f"eval_request_{uuid4().hex}"
        requested_language = case.language if case.language in {"zh", "en"} else None
        response_language = select_response_language(
            case.question,
            requested_language=requested_language,
        )
        capture = _CapturingOrchestrator(self.services.orchestrator)
        services = replace(self.services, orchestrator=capture)
        started = perf_counter()
        payloads = [
            payload
            async for payload in stream_qa_events(
                services,
                request_id=request_id,
                session_id=session.session_id,
                owner_id=owner_id,
                question=case.question,
                mode=case.retrieval_mode,
                requested_language=requested_language,
                response_language=response_language,
                redactor=self.redactor,
                cache_policy=CachePolicy.BYPASS,
            )
        ]
        latency_ms = max(0.0, (perf_counter() - started) * 1_000)
        if len(payloads) != 1:
            raise EvaluationRunnerError("validated_stream_contract_invalid")
        try:
            event = ValidatedStreamEvent.model_validate_json(payloads[0])
        except (TypeError, ValueError):
            raise EvaluationRunnerError("validated_stream_contract_invalid") from None
        if event.request_id != request_id or event.session_id != session.session_id:
            raise EvaluationRunnerError("validated_stream_identity_invalid")
        outcome = capture.outcome
        retrieved = tuple(getattr(outcome, "retrieved_chunk_ids", ()))
        context = tuple(getattr(outcome, "context_chunk_ids", ()))
        return EvaluationCaseExecution(
            case_id=case.case_id,
            owner_id=owner_id,
            session_id=session.session_id,
            request_id=request_id,
            event=event,
            retrieved_chunk_ids=retrieved,
            context_chunk_ids=context,
            latency_ms=latency_ms,
        )


@dataclass(slots=True)
class EvaluationRunner:
    repository: EvaluationRunRepository
    artifacts_root: Path
    executor: EvaluationCaseExecutor
    clock: Callable[[], datetime] = field(default=utc_now, repr=False)

    def queue(self, plan: EvaluationRunPlan) -> EvaluationRun:
        """Persist a queued run and its write-once manifest before any provider call."""

        created_at = self.clock()
        manifest = EvaluationRunManifest.create(plan, created_at=created_at)
        run_directory = self._create_run_directory(plan.run_id)
        try:
            self._write_exclusive(run_directory / "manifest.json", manifest)
            (run_directory / "cases").mkdir(mode=0o700)
            run = EvaluationRun(
                run_id=plan.run_id,
                dataset_id=plan.identity.dataset_id,
                dataset_version=plan.identity.dataset_version,
                dataset_hash=plan.identity.dataset_hash,
                corpus_version=plan.identity.corpus_version,
                configuration_id=plan.identity.configuration_id,
                code_revision=plan.identity.code_revision,
                scorer_versions=plan.identity.scorer_versions,
                cache_policy=plan.identity.cache_policy,
                total_cases=len(plan.cases),
                created_at=created_at,
                updated_at=created_at,
            )
            self.repository.create(run)
        except Exception:
            # Artifacts are intentionally retained: silently recycling a run ID would
            # destroy the audit signal that queueing partially completed.
            raise
        return run

    async def execute(self, plan: EvaluationRunPlan) -> EvaluationRun:
        """Execute every case and persist progress after each terminal result."""

        current = self.repository.get(plan.run_id)
        if current is None:
            raise EvaluationRunnerError("evaluation_run_not_queued")
        if current.status is not EvaluationRunStatus.QUEUED:
            raise EvaluationRunnerError("evaluation_run_not_queued")
        current = self._updated(current, status=EvaluationRunStatus.RUNNING)
        self.repository.update(current)
        for case in plan.cases:
            owner_digest = hashlib.sha256(f"{plan.run_id}\0{case.case_id}".encode()).hexdigest()
            try:
                execution = await self.executor.execute(
                    case,
                    owner_id=f"eval_owner_{owner_digest}",
                    cache_policy=CachePolicy.BYPASS,
                )
                if execution.case_id != case.case_id:
                    raise EvaluationRunnerError("evaluation_case_identity_invalid")
                succeeded = execution.event.kind is not StreamEventKind.ERROR
                result = PersistedCaseResult(
                    run_id=plan.run_id,
                    case_id=case.case_id,
                    succeeded=succeeded,
                    execution=execution,
                    safe_error_code=None if succeeded else "qa_terminal_error",
                )
            except Exception:
                succeeded = False
                result = PersistedCaseResult(
                    run_id=plan.run_id,
                    case_id=case.case_id,
                    succeeded=False,
                    safe_error_code="case_execution_failed",
                )
            self._write_exclusive(self._case_path(plan.run_id, case.case_id), result)
            current = self._updated(
                current,
                completed_cases=current.completed_cases + int(succeeded),
                failed_cases=current.failed_cases + int(not succeeded),
            )
            self.repository.update(current)
        current = self._updated(current, status=EvaluationRunStatus.COMPLETED)
        self.repository.update(current)
        return current

    def load_manifest(self, run_id: str) -> EvaluationRunManifest:
        return EvaluationRunManifest.model_validate_json(
            self._manifest_path(run_id).read_text(encoding="utf-8")
        )

    def load_case_results(self, run_id: str) -> tuple[PersistedCaseResult, ...]:
        directory = self._run_path(run_id) / "cases"
        if not directory.is_dir():
            return ()
        return tuple(
            PersistedCaseResult.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*.json"))
        )

    def _updated(self, run: EvaluationRun, **changes: object) -> EvaluationRun:
        return run.model_copy(update={**changes, "updated_at": self.clock()})

    def _run_path(self, run_id: str) -> Path:
        if _SAFE_ARTIFACT_ID.fullmatch(run_id) is None:
            raise EvaluationRunnerError("evaluation_run_id_invalid")
        return self.artifacts_root.resolve() / run_id

    def _create_run_directory(self, run_id: str) -> Path:
        run_directory = self._run_path(run_id)
        self.artifacts_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            run_directory.mkdir(mode=0o700)
        except FileExistsError:
            raise ImmutableRunError("evaluation_run_already_exists") from None
        return run_directory

    def _manifest_path(self, run_id: str) -> Path:
        return self._run_path(run_id) / "manifest.json"

    def _case_path(self, run_id: str, case_id: str) -> Path:
        if _SAFE_ARTIFACT_ID.fullmatch(case_id) is None:
            raise EvaluationRunnerError("evaluation_case_id_invalid")
        return self._run_path(run_id) / "cases" / f"{case_id}.json"

    @staticmethod
    def _write_exclusive(path: Path, model: DomainModel) -> None:
        try:
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(model.model_dump_json(indent=2))
                stream.write("\n")
        except FileExistsError:
            raise ImmutableRunError("evaluation_artifact_already_exists") from None


__all__ = [
    "RUNNER_VERSION",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "EvaluationCaseExecution",
    "EvaluationCaseInput",
    "EvaluationConversationTurn",
    "EvaluationEnvironment",
    "EvaluationRunIdentity",
    "EvaluationRunManifest",
    "EvaluationRunPlan",
    "EvaluationRunner",
    "EvaluationRunnerError",
    "ImmutableRunError",
    "PersistedCaseResult",
    "ProductionQAExecutor",
]
