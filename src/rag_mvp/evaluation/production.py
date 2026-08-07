"""Production evaluator using a fresh, isolated RAG data root for every run."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Protocol, cast

from rag_mvp.api.qa import QARuntimeServices
from rag_mvp.config.settings import Settings
from rag_mvp.domain.evaluation import ReportManifest
from rag_mvp.evaluation.application import ResolvedEvaluationArtifact
from rag_mvp.evaluation.corpus import EvaluationCorpusInstaller
from rag_mvp.evaluation.dataset import EvaluationDataset
from rag_mvp.evaluation.html_report import (
    MAX_HTML_REPORT_BYTES,
    verify_html_parity,
    write_html_report,
)
from rag_mvp.evaluation.json_report import (
    MAX_REPORT_BYTES,
    canonical_report_document,
    decode_json_report,
    report_content_hash,
    validate_report,
    write_json_report,
)
from rag_mvp.evaluation.pricing import openai_standard_pricing_catalog
from rag_mvp.evaluation.report_builder import build_evaluation_report
from rag_mvp.evaluation.runner import (
    EvaluationRunner,
    EvaluationRunPlan,
    ProductionQAExecutor,
)
from rag_mvp.evaluation.scoring import score_evaluation
from rag_mvp.ingestion.service import IngestionService
from rag_mvp.performance.admission import QAAdmissionController
from rag_mvp.performance.deadlines import QALatencyBudgets
from rag_mvp.safety.redactor import Redactor
from rag_mvp.storage.repositories import (
    EvaluationRunRepository,
    ReportManifestRepository,
    RuntimeRepositories,
)

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$")


class EvaluationProductionError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class IsolatedComposition(Protocol):
    @property
    def ingestion(self) -> IngestionService: ...

    @property
    def qa(self) -> QARuntimeServices: ...


type IsolatedCompositionFactory = Callable[[Settings, Redactor], IsolatedComposition]


def _default_composition_factory(settings: Settings, redactor: Redactor) -> IsolatedComposition:
    from rag_mvp.api.composition import compose_openai_services

    return compose_openai_services(settings, redactor, include_evaluation=False)


@dataclass(slots=True)
class ProductionEvaluationJobExecutor:
    """Execute one plan without ever binding evaluation data to the online index."""

    settings: Settings
    repository: EvaluationRunRepository
    report_repository: ReportManifestRepository
    run_artifacts_root: Path
    redactor: Redactor
    issues: Sequence[dict[str, object]] = ()
    composition_factory: IsolatedCompositionFactory = field(
        default=_default_composition_factory,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.run_artifacts_root = self.run_artifacts_root.resolve()

    def isolated_settings(self, run_id: str) -> Settings:
        """Return a per-run root that is provably disjoint from the online root."""

        if not run_id or any(
            character not in "-_" and not character.isalnum() for character in run_id
        ):
            raise EvaluationProductionError("evaluation_run_id_invalid")
        online_root = self.settings.data_root.expanduser().resolve()
        workspace_parent = (online_root / "evaluations" / "workspaces").resolve()
        workspace = (workspace_parent / run_id).resolve()
        if (
            not workspace_parent.is_relative_to(online_root)
            or workspace == online_root
            or not workspace.is_relative_to(workspace_parent)
            or workspace.is_relative_to((online_root / "indexes").resolve())
        ):
            raise EvaluationProductionError("evaluation_workspace_unsafe")
        return self.settings.model_copy(
            update={
                "data_root": workspace,
                "workbench_enabled": False,
                "retrieval_cache_enabled": False,
            }
        )

    async def execute(self, plan: EvaluationRunPlan, dataset: EvaluationDataset) -> None:
        isolated = self.isolated_settings(plan.run_id)
        if plan.identity.configuration_id != isolated.configuration_identity:
            raise EvaluationProductionError("evaluation_configuration_identity_mismatch")
        workspace = isolated.data_root.resolve()
        if workspace.exists() or workspace.is_symlink():
            raise EvaluationProductionError("evaluation_workspace_exists")
        workspace.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        composition = self.composition_factory(isolated, self.redactor)
        ingestion = composition.ingestion
        if _resolved_ingestion_root(ingestion) != workspace:
            await _close_composition(composition)
            raise EvaluationProductionError("evaluation_index_isolation_failed")
        admission = QAAdmissionController(isolated.qa_max_active, isolated.qa_max_queue)
        services = replace(
            composition.qa,
            admission=admission,
            latency_budgets=QALatencyBudgets.from_settings(isolated),
        )
        try:
            await EvaluationCorpusInstaller(ingestion).install(dataset)
            runner = EvaluationRunner(
                self.repository,
                self.run_artifacts_root,
                ProductionQAExecutor(services, self.redactor),
            )
            completed = await runner.execute(plan)
            results = runner.load_case_results(plan.run_id)
            scorecard = score_evaluation(dataset, results, redactor=self.redactor)
            database = ingestion.repositories.index_revisions.database
            repositories = RuntimeRepositories.from_database(database)
            attempts = tuple(
                attempt
                for result in results
                if result.execution is not None
                for attempt in repositories.provider_usage.list_for_request(
                    result.execution.request_id
                )
            )
            provider = plan.identity.provider_identities["generation"]
            models = tuple(
                model
                for model in (
                    isolated.generation_model,
                    isolated.embedding_model,
                    isolated.reranking_model,
                )
                if model is not None
            )
            pricing = openai_standard_pricing_catalog(provider=provider, models=models)
            report = build_evaluation_report(
                dataset=dataset,
                manifest=runner.load_manifest(plan.run_id),
                results=results,
                scorecard=scorecard,
                attempts=attempts,
                pricing_catalog=pricing,
                issues=self.issues,
                redactor=self.redactor,
            )
            gate = cast(dict[str, object], report["gate"])
            final_passed = gate.get("final_passed") is True
            run_root = self.run_artifacts_root / plan.run_id
            json_path = write_json_report(
                report,
                run_root / "evaluation-report.json",
                redactor=self.redactor,
            )
            html_path = write_html_report(
                report,
                run_root / "evaluation-report.html",
                redactor=self.redactor,
            )
            self.report_repository.save(
                ReportManifest(
                    run_id=plan.run_id,
                    schema_version=str(report["schema_version"]),
                    json_report_path=str(json_path),
                    html_report_path=str(html_path),
                    content_hash=report_content_hash(report),
                    metadata={
                        "profile": "standard-evaluation",
                        "quality_passed": scorecard.quality_gate.passed,
                        "final_passed": final_passed,
                    },
                )
            )
            _write_json_exclusive(
                run_root / "summary.json",
                {
                    "run_id": plan.run_id,
                    "status": completed.status.value,
                    "dataset_hash": dataset.manifest.content_hash,
                    "corpus_hash": dataset.corpus.manifest.content_hash,
                    "configuration_id": plan.identity.configuration_id,
                    "quality_passed": scorecard.quality_gate.passed,
                    "final_passed": final_passed,
                    "report_artifact_id": "evaluation-report-json",
                },
            )
        except EvaluationProductionError:
            raise
        except Exception as error:
            code = getattr(error, "code", None)
            if isinstance(code, str) and code.startswith("evaluation_"):
                raise EvaluationProductionError(code) from error
            raise EvaluationProductionError("evaluation_execution_failed") from error
        finally:
            await admission.close()
            await _close_composition(composition)


@dataclass(frozen=True, slots=True)
class VerifiedLegacyReportStore:
    """Resolve v1 compatibility reports within a fixed root and verify parity."""

    repository: ReportManifestRepository
    trusted_root: Path

    def resolve(
        self,
        run_id: str,
        report_format: Literal["json", "html"],
    ) -> ResolvedEvaluationArtifact | None:
        manifest = self.repository.get(run_id)
        if manifest is None:
            return None
        root = self.trusted_root.resolve()
        json_path = _verified_legacy_path(
            root,
            run_id,
            manifest.json_report_path,
            "evaluation-report.json",
        )
        html_path = _verified_legacy_path(
            root,
            run_id,
            manifest.html_report_path,
            "evaluation-report.html",
        )
        report, json_content = _load_verified_json_report(json_path)
        if report_content_hash(report) != manifest.content_hash:
            raise EvaluationProductionError("evaluation_report_integrity_failed")
        if report_format == "json":
            content = json_content
            media_type = "application/json"
            filename = "evaluation-report.json"
        else:
            html, content = _load_verified_html_report(html_path)
            try:
                verify_html_parity(report, html)
            except (TypeError, ValueError):
                raise EvaluationProductionError("evaluation_report_integrity_failed") from None
            media_type = "text/html"
            filename = "evaluation-report.html"
        return ResolvedEvaluationArtifact(
            artifact_id=f"evaluation-report-{report_format}",
            content=content,
            media_type=media_type,
            filename=filename,
        )


async def _close_composition(composition: IsolatedComposition) -> None:
    ingestion = getattr(composition, "ingestion", None)
    if ingestion is not None:
        close_ingestion = getattr(ingestion, "close", None)
        if callable(close_ingestion):
            close_ingestion()
    await composition.qa.close()


def _resolved_ingestion_root(ingestion: IngestionService) -> Path:
    return Path(ingestion.data_root).resolve()


def _verified_legacy_path(
    root: Path,
    run_id: str,
    stored_path: str,
    filename: str,
) -> Path:
    if _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise EvaluationProductionError("evaluation_report_integrity_failed")
    path = Path(stored_path)
    run_root = root / run_id
    expected = run_root / filename
    if (
        not path.is_absolute()
        or run_root.is_symlink()
        or expected.is_symlink()
        or path.is_symlink()
    ):
        raise EvaluationProductionError("evaluation_report_integrity_failed")
    try:
        resolved = path.resolve(strict=True)
        expected_resolved = expected.resolve(strict=True)
    except OSError:
        raise EvaluationProductionError("evaluation_report_unavailable") from None
    if resolved != expected_resolved or not resolved.is_relative_to(root) or not resolved.is_file():
        raise EvaluationProductionError("evaluation_report_integrity_failed")
    return resolved


def _load_verified_json_report(path: Path) -> tuple[dict[str, object], bytes]:
    try:
        content = path.read_bytes()
    except OSError:
        raise EvaluationProductionError("evaluation_report_unavailable") from None
    if not 0 < len(content) <= MAX_REPORT_BYTES:
        raise EvaluationProductionError("evaluation_report_integrity_failed")
    try:
        decoded = decode_json_report(content.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("evaluation_report_invalid")
        report = validate_report(cast(dict[str, object], decoded))
    except (TypeError, UnicodeError, ValueError):
        raise EvaluationProductionError("evaluation_report_integrity_failed") from None
    if canonical_report_document(report) != content:
        raise EvaluationProductionError("evaluation_report_integrity_failed")
    return cast(dict[str, object], report), content


def _load_verified_html_report(path: Path) -> tuple[str, bytes]:
    try:
        content = path.read_bytes()
    except OSError:
        raise EvaluationProductionError("evaluation_report_unavailable") from None
    if not 0 < len(content) <= MAX_HTML_REPORT_BYTES:
        raise EvaluationProductionError("evaluation_report_integrity_failed")
    try:
        return content.decode("utf-8"), content
    except UnicodeError:
        raise EvaluationProductionError("evaluation_report_integrity_failed") from None


def _write_json_exclusive(path: Path, value: object) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError:
        raise EvaluationProductionError("evaluation_artifact_already_exists") from None


__all__ = [
    "EvaluationProductionError",
    "ProductionEvaluationJobExecutor",
    "VerifiedLegacyReportStore",
]
