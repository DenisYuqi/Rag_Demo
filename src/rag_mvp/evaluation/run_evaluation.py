"""Run one immutable real-provider evaluation and write its evidence bundle."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from rag_mvp.api.app import RuntimeState, create_app
from rag_mvp.api.composition import compose_openai_services
from rag_mvp.config.settings import Settings
from rag_mvp.domain.evaluation import ReportManifest
from rag_mvp.evaluation.corpus import EvaluationCorpusInstaller
from rag_mvp.evaluation.dataset import load_dataset
from rag_mvp.evaluation.html_report import write_html_report
from rag_mvp.evaluation.json_report import (
    report_content_hash,
    write_json_report,
)
from rag_mvp.evaluation.plan import build_evaluation_plan
from rag_mvp.evaluation.pricing import (
    OPENAI_STANDARD_PRICING_VERSION,
    openai_standard_pricing_catalog,
)
from rag_mvp.evaluation.report_builder import build_evaluation_report
from rag_mvp.evaluation.runner import EvaluationRunner, ProductionQAExecutor
from rag_mvp.evaluation.scoring import score_evaluation
from rag_mvp.safety.redactor import DEFAULT_REDACTOR
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

    settings = _settings(data_root, profile)
    if settings.provider_backend != "openai" or settings.provider_readiness_errors():
        raise RuntimeError("evaluation_provider_unavailable")
    dataset = load_dataset(dataset_path)
    run_directory = (output_root / run_id).resolve()
    run_directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    composition = compose_openai_services(settings, DEFAULT_REDACTOR)
    runtime: RuntimeState | None = None
    try:
        await EvaluationCorpusInstaller(composition.ingestion).install(dataset)
        app = create_app(
            settings,
            ingestion_service=composition.ingestion,
            qa_services=composition.qa,
            diagnostics_service=composition.diagnostics,
            redactor=DEFAULT_REDACTOR,
        )
        runtime = cast(RuntimeState, app.state.runtime)
        services = runtime.qa_services
        if services is None:
            raise RuntimeError("evaluation_qa_unavailable")
        database = composition.ingestion.repositories.index_revisions.database
        repositories = RuntimeRepositories.from_database(database)
        plan = build_evaluation_plan(dataset, settings, run_id)
        runner = EvaluationRunner(
            repositories.evaluation_runs,
            run_directory / "runs",
            ProductionQAExecutor(services, DEFAULT_REDACTOR),
        )
        runner.queue(plan)
        completed = await runner.execute(plan)
        results = runner.load_case_results(run_id)
        scorecard = score_evaluation(dataset, results, redactor=DEFAULT_REDACTOR)
        attempts = tuple(
            attempt
            for result in results
            if result.execution is not None
            for attempt in repositories.provider_usage.list_for_request(result.execution.request_id)
        )
        provider = plan.identity.provider_identities["generation"]
        models = tuple(
            model
            for model in (
                settings.generation_model,
                settings.embedding_model,
                settings.reranking_model,
            )
            if model is not None
        )
        pricing = openai_standard_pricing_catalog(provider=provider, models=models)
        issues = _load_issues(issues_path)
        manifest = runner.load_manifest(run_id)
        report = build_evaluation_report(
            dataset=dataset,
            manifest=manifest,
            results=results,
            scorecard=scorecard,
            attempts=attempts,
            pricing_catalog=pricing,
            issues=issues,
            redactor=DEFAULT_REDACTOR,
        )
        json_path = write_json_report(report, run_directory / "report.json")
        html_path = write_html_report(report, run_directory / "report.html")
        _write_json_exclusive(run_directory / "configuration.json", settings.safe_dump())
        _write_json_exclusive(
            run_directory / "diagnostics.json",
            {
                "run_id": run_id,
                "requests": [
                    {
                        "case_id": result.case_id,
                        "request_id": result.execution.request_id,
                        "trace_id": result.execution.event.diagnostics.trace_id,
                        "stage_timings_ms": (result.execution.event.diagnostics.stage_timings_ms),
                        "cache_status": result.execution.event.diagnostics.cache_status,
                        "model_identities": (result.execution.event.diagnostics.model_identities),
                        "token_counts": result.execution.event.diagnostics.token_counts,
                        "safe_error_code": result.safe_error_code,
                    }
                    for result in results
                    if result.execution is not None
                ],
            },
        )
        _write_json_exclusive(
            run_directory / "provider-attempts.json",
            {"run_id": run_id, "attempts": [item.model_dump(mode="json") for item in attempts]},
        )
        repositories.report_manifests.save(
            ReportManifest(
                run_id=run_id,
                schema_version=str(report["schema_version"]),
                json_report_path=str(json_path),
                html_report_path=str(html_path),
                content_hash=report_content_hash(report),
                metadata={
                    "profile": profile,
                    "quality_passed": scorecard.quality_gate.passed,
                    "final_passed": cast(dict[str, object], report["gate"])["final_passed"],
                },
            )
        )
        summary = {
            "run_id": run_id,
            "status": completed.status.value,
            "dataset_hash": dataset.manifest.content_hash,
            "corpus_hash": dataset.corpus.manifest.content_hash,
            "configuration_id": settings.configuration_identity,
            "quality_passed": scorecard.quality_gate.passed,
            "final_passed": cast(dict[str, object], report["gate"])["final_passed"],
            "report": str(json_path),
        }
        _write_json_exclusive(run_directory / "summary.json", summary)
        return json_path, summary
    finally:
        if runtime is not None and runtime.owns_qa_admission and runtime.qa_services is not None:
            admission = runtime.qa_services.admission
            if admission is not None:
                await admission.close()
        composition.ingestion.close()
        await composition.qa.close()


def _load_issues(path: Path | None) -> tuple[dict[str, object], ...]:
    if path is None:
        return ()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("evaluation_issues_invalid")
    return tuple(cast(dict[str, object], item) for item in value)


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
