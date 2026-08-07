from __future__ import annotations

import csv
import hashlib
import os
from io import StringIO
from pathlib import Path

import pytest
from test_comparison import (
    _compatibility,
    _plan,
    _setup_attempt,
    _shared_setup,
    _terminal_suite,
    _verified_reports,
)

from rag_mvp.domain import ModelAttemptStatus
from rag_mvp.evaluation.comparison import (
    ComparisonArtifactManifest,
    ComparisonCandidateEvidence,
    ComparisonCandidateStatus,
    ComparisonResult,
    ComparisonSharedSetupStatus,
    ComparisonSuite,
    VerifiedCandidateReport,
    aggregate_comparison_result,
    build_comparison_candidate_evidence,
    canonical_candidate_evidence,
    canonical_comparison_manifest,
    seal_comparison_candidate_evidence,
)
from rag_mvp.evaluation.comparison_artifacts import (
    COMPARISON_MANIFEST_FILENAME,
    ComparisonArtifactCatalog,
    ComparisonArtifactError,
)
from rag_mvp.safety.scan_artifacts import scan_artifacts

_REPOSITORY_ROOT = Path(__file__).parents[3]
_PRIVACY_FIXTURES = _REPOSITORY_ROOT / "evaluations" / "privacy" / "supported-fixtures-v1.json"


def _evidence_bundle() -> tuple[
    ComparisonSuite,
    ComparisonResult,
    dict[str, VerifiedCandidateReport],
]:
    plan = _plan()
    evidence = _verified_reports(plan, quality_values=(0.9, 1.0))
    suite = _terminal_suite(plan, evidence)
    reports = {
        item.variant_id: seal_comparison_candidate_evidence(
            suite.candidates[index].reference,
            evidence[index],
        )
        for index, item in enumerate(plan.variants)
    }
    result = aggregate_comparison_result(
        suite,
        _compatibility(plan),
        reports,
        shared_setup=_shared_setup(plan),
    )
    return suite, result, reports


def _evidence_bundle_with_latency(
    latency_ms: float,
) -> tuple[
    ComparisonSuite,
    ComparisonResult,
    dict[str, VerifiedCandidateReport],
]:
    plan = _plan()
    originals = _verified_reports(plan, quality_values=(0.9, 1.0))
    references = _terminal_suite(plan, originals).candidates
    evidence = tuple(
        build_comparison_candidate_evidence(
            comparison_id=original.comparison_id,
            plan=plan,
            reference=references[index].reference,
            identity_projection=original.identity_projection,
            expected_case_ids=original.case_ids,
            logical_attempts=(
                type(original.logical_attempts[0]).model_validate(
                    {
                        **original.logical_attempts[0].model_dump(mode="python"),
                        "latency_ms": latency_ms,
                    }
                ),
            ),
            provider_attempts=original.provider_attempts,
            quality_metrics=tuple(
                item for item in original.metrics if not item.metric_id.startswith("comparison-")
            ),
            gates=original.gates,
            category_results=original.category_results,
            reranker_evidence=original.reranker_evidence,
            generated_at=original.generated_at,
        )
        for index, original in enumerate(originals)
    )
    suite = _terminal_suite(plan, evidence)
    reports = {
        item.variant_id: seal_comparison_candidate_evidence(
            suite.candidates[index].reference,
            evidence[index],
        )
        for index, item in enumerate(plan.variants)
    }
    result = aggregate_comparison_result(
        suite,
        _compatibility(plan),
        reports,
        shared_setup=_shared_setup(plan),
    )
    return suite, result, reports


def _rewrite_artifact_with_valid_manifest(
    root: Path,
    suite: ComparisonSuite,
    artifact_id: str,
    content: bytes,
) -> None:
    catalog = ComparisonArtifactCatalog(root)
    manifest = catalog.manifest(suite.comparison_id)
    assert manifest is not None
    descriptors = tuple(
        item.model_copy(
            update={
                "byte_size": len(content),
                "sha256_digest": f"sha256:{hashlib.sha256(content).hexdigest()}",
            }
        )
        if item.artifact_id == artifact_id
        else item
        for item in manifest.artifacts
    )
    replacement = ComparisonArtifactManifest.create(
        comparison_id=suite.comparison_id,
        plan=suite.plan,
        artifacts=descriptors,
        created_at=manifest.created_at,
    )
    descriptor = next(item for item in descriptors if item.artifact_id == artifact_id)
    bundle = root / suite.comparison_id
    (bundle / Path(descriptor.relative_path)).write_bytes(content)
    (bundle / COMPARISON_MANIFEST_FILENAME).write_bytes(canonical_comparison_manifest(replacement))


def test_publish_verify_resolve_and_reject_tamper(tmp_path: Path) -> None:
    suite, result, reports = _evidence_bundle()
    catalog = ComparisonArtifactCatalog(tmp_path / "published")

    published = catalog.publish(suite, result, reports)

    manifest = catalog.manifest("comparison-1")
    assert manifest == published.manifest
    assert {item.artifact_id for item in manifest.artifacts} == {
        "comparison-plan-json",
        "comparison-report-json",
        "comparison-report-html",
        "comparison-report-txt",
        "comparison-report-csv",
        "comparison-candidate-variant-0",
        "comparison-candidate-variant-1",
    }
    for artifact_id in (
        "comparison-report-json",
        "comparison-report-html",
        "comparison-report-txt",
        "comparison-report-csv",
    ):
        resolved = catalog.resolve("comparison-1", artifact_id)
        assert resolved is not None and resolved.content
    assert catalog.resolve("comparison-1", "unknown") is None

    report_path = tmp_path / "published" / "comparison-1" / "comparison-report.json"
    report_path.write_bytes(report_path.read_bytes() + b" ")
    with pytest.raises(ComparisonArtifactError, match="integrity-failed"):
        catalog.manifest("comparison-1")


def test_real_shaped_numeric_evidence_publishes_with_projection_parity(
    tmp_path: Path,
) -> None:
    latency_ms = 17.07899999746587
    suite, result, reports = _evidence_bundle_with_latency(latency_ms)
    catalog = ComparisonArtifactCatalog(tmp_path / "published-real-floats")

    published = catalog.publish(suite, result, reports)

    assert catalog.manifest(suite.comparison_id) == published.manifest
    encoded_latency = str(latency_ms).encode("ascii")
    for artifact_id in (
        "comparison-report-json",
        "comparison-report-html",
        "comparison-report-txt",
        "comparison-report-csv",
    ):
        resolved = catalog.resolve(suite.comparison_id, artifact_id)
        assert resolved is not None
        assert encoded_latency in resolved.content
    csv_report = catalog.resolve(suite.comparison_id, "comparison-report-csv")
    assert csv_report is not None
    rows = tuple(csv.reader(StringIO(csv_report.content.decode("utf-8"), newline="")))
    assert rows
    assert all(len(row) == len(rows[0]) for row in rows)
    privacy_report = scan_artifacts(
        (tmp_path / "published-real-floats" / suite.comparison_id,),
        fixture_path=_PRIVACY_FIXTURES,
    )
    assert privacy_report["passed"] is True
    assert privacy_report["counts"] == {
        "targets": 1,
        "scanned": len(published.manifest.artifacts) + 1,
        "matched": 0,
        "excluded": 0,
        "fixture_matches": 0,
        "detector_matches": 0,
        "errors": 0,
    }


def test_bundle_is_immutable_and_symlink_artifacts_are_rejected(tmp_path: Path) -> None:
    suite, result, reports = _evidence_bundle()
    catalog = ComparisonArtifactCatalog(tmp_path / "published")
    catalog.publish(suite, result, reports)

    with pytest.raises(ComparisonArtifactError, match="bundle-exists"):
        catalog.publish(suite, result, reports)

    report_path = tmp_path / "published" / "comparison-1" / "comparison-report.txt"
    target = tmp_path / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    report_path.unlink()
    try:
        os.symlink(target, report_path)
    except (NotImplementedError, OSError):
        pytest.skip("symbolic links are unavailable")
    with pytest.raises(ComparisonArtifactError, match=r"path-unsafe|entry-set-invalid"):
        catalog.resolve("comparison-1", "comparison-report-txt")


def test_shared_setup_is_present_with_exact_parity_in_every_report_projection(
    tmp_path: Path,
) -> None:
    plan = _plan()
    setup_attempt = _setup_attempt(
        plan,
        attempt_number=1,
        status=ModelAttemptStatus.SUCCEEDED,
    )
    setup = _shared_setup(
        plan,
        status=ComparisonSharedSetupStatus.COMPLETED,
        attempts=(setup_attempt,),
    )
    evidence = _verified_reports(plan, quality_values=(0.9, 1.0))
    suite = _terminal_suite(plan, evidence)
    reports = {
        item.variant_id: seal_comparison_candidate_evidence(
            suite.candidates[index].reference,
            evidence[index],
        )
        for index, item in enumerate(plan.variants)
    }
    result = aggregate_comparison_result(
        suite,
        _compatibility(plan),
        reports,
        shared_setup=setup,
    )
    catalog = ComparisonArtifactCatalog(tmp_path / "published")
    catalog.publish(suite, result, reports)

    for artifact_id in (
        "comparison-report-json",
        "comparison-report-html",
        "comparison-report-txt",
        "comparison-report-csv",
    ):
        resolved = catalog.resolve(suite.comparison_id, artifact_id)
        assert resolved is not None
        text = resolved.content.decode("utf-8")
        assert setup.setup_id in text
        assert str(setup.provider_call_count) in text
        assert str(setup.known_partial_cost) in text


@pytest.mark.parametrize(
    "artifact_id",
    (
        "comparison-report-json",
        "comparison-report-html",
        "comparison-report-txt",
        "comparison-report-csv",
    ),
)
def test_each_projection_rejects_cross_format_consistent_hash_tampering(
    tmp_path: Path,
    artifact_id: str,
) -> None:
    suite, result, reports = _evidence_bundle()
    root = tmp_path / artifact_id
    catalog = ComparisonArtifactCatalog(root)
    catalog.publish(suite, result, reports)
    resolved = catalog.resolve(suite.comparison_id, artifact_id)
    assert resolved is not None
    tampered = resolved.content.replace(
        result.shared_setup.setup_id.encode("utf-8"),
        b"comparison-setup-" + (b"f" * 64),
        1,
    )
    assert tampered != resolved.content
    _rewrite_artifact_with_valid_manifest(root, suite, artifact_id, tampered)

    with pytest.raises(ComparisonArtifactError):
        catalog.manifest(suite.comparison_id)


def test_candidate_semantic_binding_and_partial_failure_artifact_set(tmp_path: Path) -> None:
    suite, result, reports = _evidence_bundle()
    root = tmp_path / "foreign"
    catalog = ComparisonArtifactCatalog(root)
    catalog.publish(suite, result, reports)
    report = reports["variant-0"]
    foreign = ComparisonCandidateEvidence.model_validate(
        report.evidence.model_copy(update={"comparison_id": "comparison-foreign"})
    )
    _rewrite_artifact_with_valid_manifest(
        root,
        suite,
        report.descriptor.artifact_id,
        canonical_candidate_evidence(foreign),
    )
    with pytest.raises(ComparisonArtifactError, match="candidate-invalid"):
        catalog.manifest(suite.comparison_id)

    plan = _plan()
    complete = _verified_reports(plan)
    partial_evidence = (complete[0], None)
    partial_suite = _terminal_suite(plan, partial_evidence)
    partial_reports = {
        "variant-0": seal_comparison_candidate_evidence(
            partial_suite.candidates[0].reference,
            complete[0],
        )
    }
    partial_result = aggregate_comparison_result(
        partial_suite,
        _compatibility(plan),
        {"variant-0": partial_reports["variant-0"], "variant-1": None},
        shared_setup=_shared_setup(plan),
    )
    partial_catalog = ComparisonArtifactCatalog(tmp_path / "partial")
    published = partial_catalog.publish(partial_suite, partial_result, partial_reports)
    artifact_ids = {item.artifact_id for item in published.manifest.artifacts}
    assert "comparison-candidate-variant-0" in artifact_ids
    assert "comparison-candidate-variant-1" not in artifact_ids
    assert partial_result.candidates[1].status is ComparisonCandidateStatus.FAILED
    assert (
        partial_catalog.resolve(
            partial_suite.comparison_id,
            "comparison-report-json",
        )
        is not None
    )


@pytest.mark.parametrize(
    "unsafe_text",
    (
        "sk-secret-value-12345678",
        "C:\\private\\comparison.json",
        "user@example.test",
        "+86 138 0013 8000",
        "4111111111111111",
        "-----BEGIN PRIVATE KEY-----\nprivate-material\n-----END PRIVATE KEY-----",
    ),
)
def test_privacy_checks_run_after_valid_digest_rewrite(
    tmp_path: Path,
    unsafe_text: str,
) -> None:
    suite, result, reports = _evidence_bundle()
    root = tmp_path / hashlib.sha256(unsafe_text.encode("utf-8")).hexdigest()
    catalog = ComparisonArtifactCatalog(root)
    catalog.publish(suite, result, reports)
    artifact_id = "comparison-report-txt"
    resolved = catalog.resolve(suite.comparison_id, artifact_id)
    assert resolved is not None
    _rewrite_artifact_with_valid_manifest(
        root,
        suite,
        artifact_id,
        resolved.content + unsafe_text.encode("utf-8"),
    )

    with pytest.raises(ComparisonArtifactError, match="privacy-failed"):
        catalog.manifest(suite.comparison_id)


def test_extra_entries_are_rejected_even_when_declared_artifacts_are_unchanged(
    tmp_path: Path,
) -> None:
    suite, result, reports = _evidence_bundle()
    catalog = ComparisonArtifactCatalog(tmp_path / "published")
    catalog.publish(suite, result, reports)
    bundle = tmp_path / "published" / suite.comparison_id
    (bundle / "undeclared.txt").write_text("undeclared", encoding="utf-8")

    with pytest.raises(ComparisonArtifactError, match="entry-set-invalid"):
        catalog.manifest(suite.comparison_id)
