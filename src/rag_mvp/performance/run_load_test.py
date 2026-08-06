"""CLI for an HTTP load run and immutable performance evidence publication."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Never, cast

from pydantic import ValidationError

from rag_mvp.performance.evidence_bundle import (
    PerformanceCostEvidence,
    PerformanceEvidenceIdentity,
    PerformanceEvidenceReferences,
    build_performance_evidence_bundle,
    write_performance_evidence_bundle,
)
from rag_mvp.performance.load_report import LoadReport
from rag_mvp.performance.load_test import (
    HttpLoadTestConfig,
    HttpLoadTestHarness,
    LoadScenario,
)
from rag_mvp.performance.pricing import (
    PerformancePricingEvidence,
    calculate_performance_cost,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the uncached single-instance HTTP QA acceptance workload."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--scenario-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--configuration-id", required=True)
    parser.add_argument("--service-version", required=True)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="ROLE=IDENTITY",
        help="Exact model identity; repeat for every configured role.",
    )
    parser.add_argument("--instance-identity")
    parser.add_argument(
        "--expected-workload-digest",
        required=True,
        help="Pinned sha256 digest of the exact scenario file contents after validation.",
    )
    parser.add_argument(
        "--confirm-acceptance-run",
        action="store_true",
        help="Explicitly confirm quota and acceptance-run spend before any HTTP traffic.",
    )
    parser.add_argument("--metric-reference", action="append", default=[])
    parser.add_argument("--log-reference", action="append", default=[])
    parser.add_argument("--trace-reference", action="append", default=[])
    parser.add_argument(
        "--pricing-evidence",
        type=Path,
        required=True,
        help="Pinned role rate card validated before traffic and used for exact cost.",
    )
    parser.add_argument("--warmup-attempts", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--target-successes", type=int, default=500)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--retry-limit", type=int, default=1)
    parser.add_argument("--request-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--instance-count", type=int, default=1)
    return parser


async def _run(args: argparse.Namespace) -> bool:
    if getattr(args, "confirm_acceptance_run", False) is not True:
        raise ValueError("acceptance run requires explicit quota and spend confirmation")
    scenarios = _load_scenarios(args.scenario_file)
    models = _parse_models(cast(Sequence[str], args.model))
    pricing = _load_pricing(args.pricing_evidence)
    config = HttpLoadTestConfig(
        run_id=args.run_id,
        expected_configuration_id=args.configuration_id,
        base_url=args.base_url,
        scenarios=scenarios,
        warmup_attempts=args.warmup_attempts,
        concurrency=args.concurrency,
        target_successes=args.target_successes,
        max_attempts=args.max_attempts,
        retry_limit=args.retry_limit,
        request_timeout_seconds=args.request_timeout_seconds,
        instance_count=args.instance_count,
    )
    if args.expected_workload_digest != config.workload_digest:
        raise ValueError("scenario workload digest does not match the pinned digest")
    report = await HttpLoadTestHarness(config).run()
    cost = _resolve_cost(report, pricing=pricing)
    observed_instance_identities = {
        attempt.instance_identity
        for attempt in (*report.warmup.attempts, *report.attempts)
        if attempt.succeeded and attempt.instance_identity is not None
    }
    observed_instance_identity = (
        next(iter(observed_instance_identities)) if len(observed_instance_identities) == 1 else None
    )
    if args.instance_identity is not None and args.instance_identity != observed_instance_identity:
        raise ValueError("declared instance identity disagrees with HTTP evidence")
    bundle = build_performance_evidence_bundle(
        report,
        identity=PerformanceEvidenceIdentity(
            code_revision=args.code_revision,
            configuration_id=args.configuration_id,
            service_version=args.service_version,
            model_identities=models,
            instance_identity=observed_instance_identity,
            pricing_evidence_digest=cost.pricing_evidence_digest,
        ),
        references=PerformanceEvidenceReferences(
            metrics=tuple(args.metric_reference),
            logs=tuple(args.log_reference),
            representative_traces=tuple(args.trace_reference),
        ),
        cost=cost,
    )
    write_performance_evidence_bundle(bundle, args.output)
    decision = cast(dict[str, object], bundle["decision"])
    summary = {
        "run_id": report.run_id,
        "attempts": report.attempt_count,
        "successes": report.success_count,
        "errors": report.error_count,
        "valid": decision["valid"],
        "passed": decision["passed"],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, sort_keys=True))
    return decision["passed"] is True


def _load_scenarios(path: Path) -> tuple[LoadScenario, ...]:
    raw = _load_json(path)
    if isinstance(raw, dict):
        raw = raw.get("scenarios")
    if not isinstance(raw, list) or not raw:
        raise ValueError("scenario file must contain a non-empty JSON array")
    try:
        return tuple(LoadScenario.model_validate(value) for value in raw)
    except ValidationError:
        raise ValueError("scenario file contains an invalid scenario") from None


def _load_pricing(pricing_path: Path | None) -> PerformancePricingEvidence:
    if pricing_path is None:
        raise ValueError("pricing evidence is required before acceptance traffic")
    raw = _load_json(pricing_path)
    try:
        return PerformancePricingEvidence.model_validate(raw)
    except ValidationError:
        raise ValueError("pricing evidence does not match the safe contract") from None


def _resolve_cost(
    report: LoadReport,
    *,
    pricing: PerformancePricingEvidence,
) -> PerformanceCostEvidence:
    return calculate_performance_cost(report, pricing)


def _parse_models(values: Sequence[str]) -> dict[str, str]:
    models: dict[str, str] = {}
    for value in values:
        role, separator, identity = value.partition("=")
        if not separator or not role.strip() or not identity.strip() or role in models:
            raise ValueError("each --model must be one unique ROLE=IDENTITY pair")
        models[role.strip()] = identity.strip()
    if not models:
        raise ValueError("at least one --model identity is required")
    return models


def _load_json(path: Path) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("input file is not valid UTF-8 JSON") from error


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("input JSON contains a duplicate key")
        result[key] = value
    return result


def _reject_constant(value: str) -> Never:
    del value
    raise ValueError("input JSON contains a non-finite number")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        passed = asyncio.run(_run(args))
    except (ValueError, OSError):
        print(json.dumps({"error": "load-test-failed"}, sort_keys=True))
        return 2
    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover - exercised as a module CLI
    raise SystemExit(main())
