"""Run one immutable real-provider evaluation and write its evidence bundle."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from rag_mvp.config.settings import Settings
from rag_mvp.evaluation.application import (
    EvaluationApplicationService,
    EvaluationRunSummary,
)
from rag_mvp.evaluation.dataset import load_dataset
from rag_mvp.evaluation.plan import EvaluationDatasetRegistry
from rag_mvp.evaluation.pricing import OPENAI_STANDARD_PRICING_VERSION
from rag_mvp.evaluation.production import (
    ProductionEvaluationJobExecutor,
    VerifiedLegacyReportStore,
)
from rag_mvp.safety.redactor import DEFAULT_REDACTOR
from rag_mvp.storage.database import Database
from rag_mvp.storage.layout import DataLayout
from rag_mvp.storage.repositories import RuntimeRepositories

type EvaluationProfile = Literal["accepted", "controlled-retrieval", "controlled-refusal"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("evaluations/results"))
    parser.add_argument("--run-id")
    parser.add_argument(
        "--profile",
        choices=("accepted", "controlled-retrieval", "controlled-refusal"),
        default="accepted",
    )
    parser.add_argument("--issues", type=Path)
    parser.add_argument("--require-final-pass", action="store_true")
    return parser


def _settings(data_root: Path, profile: EvaluationProfile) -> Settings:
    base = Settings()
    updates: dict[str, object] = {
        "data_root": data_root,
        "workbench_enabled": False,
        "pricing_version": OPENAI_STANDARD_PRICING_VERSION,
    }
    if profile == "controlled-retrieval":
        updates.update(
            {
                "default_retrieval_mode": "dense",
                "dense_candidate_limit": 1,
                "lexical_candidate_limit": 1,
                "context_chunk_limit": 1,
                "rerank_candidate_limit": 1,
            }
        )
    elif profile == "controlled-refusal":
        updates["qa_minimum_support_score"] = 1.0
    return base.model_copy(update=updates)


def _new_run_id(profile: EvaluationProfile) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{profile.replace('-', '_')}_{timestamp}_{uuid4().hex[:8]}"


async def run_real_evaluation(
    *,
    dataset_path: Path,
    data_root: Path,
    output_root: Path,
    run_id: str,
    profile: EvaluationProfile,
    issues_path: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    """Execute and persist one run; dynamic text never enters the summary."""

    dataset = load_dataset(dataset_path)
    settings = _settings(data_root, profile)
    if settings.provider_backend != "openai" or settings.provider_readiness_errors():
        raise RuntimeError("evaluation_provider_unavailable")
    layout = DataLayout.from_root(settings.data_root)
    layout.initialize()
    database = Database(layout.metadata_db)
    database.initialize()
    repositories = RuntimeRepositories.from_database(database)
    run_root = _resolved_path(output_root)
    issues = _load_issues(issues_path)
    executor = ProductionEvaluationJobExecutor(
        settings=settings,
        repository=repositories.evaluation_runs,
        report_repository=repositories.report_manifests,
        run_artifacts_root=run_root,
        redactor=DEFAULT_REDACTOR,
        issues=issues,
    )
    service = EvaluationApplicationService(
        registry=EvaluationDatasetRegistry(dataset.root.parent),
        settings=settings,
        repository=repositories.evaluation_runs,
        run_artifacts_root=run_root,
        executor=executor,
        maximum_active_jobs=1,
        shutdown_grace_seconds=settings.evaluation_shutdown_grace_seconds,
        legacy_report_store=VerifiedLegacyReportStore(
            repositories.report_manifests,
            run_root,
        ),
        run_id_factory=lambda: run_id,
        plan_settings_factory=executor.isolated_settings,
    )
    try:
        await service.startup()
        queued = await service.start(
            dataset.manifest.dataset_id,
            dataset.manifest.version,
        )
        completed = await service.wait(queued.run_id)
        if completed is None or completed.status.value != "completed":
            code = None if completed is None else completed.safe_error_code
            raise RuntimeError(code or "evaluation_execution_failed")
        summary_model = cast(EvaluationRunSummary, service.summary(run_id))
        report_path = run_root / run_id / "evaluation-report.json"
        report = service.report(run_id, "json")
        if report is None or not report_path.is_file():
            raise RuntimeError("evaluation_report_unavailable")
        summary: dict[str, object] = {
            "run_id": run_id,
            "status": completed.status.value,
            "dataset_hash": completed.dataset_hash,
            "corpus_hash": summary_model.corpus_hash,
            "configuration_id": completed.configuration_id,
            "quality_passed": summary_model.gate_status == "passed",
            "final_passed": summary_model.gate_status == "passed",
            "report": str(report_path),
        }
        return report_path, summary
    finally:
        await service.close()


def _load_issues(path: Path | None) -> tuple[dict[str, object], ...]:
    if path is None:
        return ()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("evaluation_issues_invalid")
    return tuple(cast(dict[str, object], item) for item in value)


def _resolved_path(path: Path) -> Path:
    return path.resolve()


def _write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    args = _parser().parse_args()
    run_id = args.run_id or _new_run_id(args.profile)
    report_path, summary = asyncio.run(
        run_real_evaluation(
            dataset_path=args.dataset,
            data_root=args.data_root,
            output_root=args.output_root,
            run_id=run_id,
            profile=args.profile,
            issues_path=args.issues,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if args.require_final_pass and summary["final_passed"] is not True:
        raise SystemExit(1)
    if not report_path.is_file():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
