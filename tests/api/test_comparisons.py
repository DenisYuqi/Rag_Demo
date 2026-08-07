from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from typing import cast

import pytest
from fastapi.testclient import TestClient

from rag_mvp.api.app import create_app
from rag_mvp.api.evaluation_diagnostics import EvaluationOperations
from rag_mvp.config.settings import Settings
from rag_mvp.domain.evaluation import UnavailableValue
from rag_mvp.evaluation.application import (
    ComparisonCapacityError,
    ComparisonConflictError,
    ComparisonNotFoundError,
    ComparisonUnavailableError,
    ComparisonValidationError,
)
from rag_mvp.evaluation.comparison import ComparisonStatus
from rag_mvp.evaluation.comparison_application import (
    ComparisonArtifactDescriptor,
    ComparisonArtifactManifestView,
    ComparisonPlanCatalogEntry,
    ComparisonPlanVariantEntry,
    ComparisonRecommendationSummary,
    ComparisonRunEntry,
    ComparisonSharedSetupSummary,
    ComparisonSummary,
    ResolvedComparisonDownload,
)


@dataclass
class FakeComparisonService:
    plan: ComparisonPlanCatalogEntry
    run: ComparisonRunEntry | None
    summary: ComparisonSummary | None
    manifest: ComparisonArtifactManifestView | None = None
    artifacts: dict[str, ResolvedComparisonDownload] = field(default_factory=dict)
    start_error: RuntimeError | None = None
    starts: list[str] = field(default_factory=list)

    def comparison_plans(self) -> tuple[ComparisonPlanCatalogEntry, ...]:
        return (self.plan,)

    async def start_comparison(self, experiment_plan_id: str) -> ComparisonRunEntry:
        if self.start_error is not None:
            raise self.start_error
        self.starts.append(experiment_plan_id)
        assert self.run is not None
        return self.run

    def get_comparison(self, comparison_id: str) -> ComparisonRunEntry | None:
        if self.run is None or self.run.comparison_id != comparison_id:
            return None
        return self.run

    def list_comparisons(self) -> tuple[ComparisonRunEntry, ...]:
        return () if self.run is None else (self.run,)

    def comparison_summary(self, comparison_id: str) -> ComparisonSummary | None:
        return self.summary if self.get_comparison(comparison_id) is not None else None

    def comparison_manifest(
        self,
        comparison_id: str,
    ) -> ComparisonArtifactManifestView | None:
        return self.manifest if self.get_comparison(comparison_id) is not None else None

    def comparison_artifact(
        self,
        comparison_id: str,
        artifact_id: str,
    ) -> ResolvedComparisonDownload | None:
        if self.get_comparison(comparison_id) is None:
            return None
        return self.artifacts.get(artifact_id)


def _service() -> FakeComparisonService:
    now = datetime.now(UTC)
    plan_id = "generation-model-plan"
    comparison_id = "comparison-1"
    plan_hash = "sha256:" + "1" * 64
    plan = ComparisonPlanCatalogEntry(
        experiment_plan_id=plan_id,
        plan_content_hash=plan_hash,
        display_name="Generation model comparison",
        axis="generation-model",
        dataset_id="acceptance-v2",
        dataset_version="2.0.0",
        dataset_hash="sha256:" + "2" * 64,
        corpus_id="corpus-v2",
        corpus_version="2.0.0",
        corpus_hash="sha256:" + "3" * 64,
        case_set_hash="sha256:" + "4" * 64,
        planned_case_count=1,
        variants=(
            ComparisonPlanVariantEntry(
                variant_id="variant-0",
                display_name="Baseline",
                axis_value="model-a",
                configuration_id="semantic-config-0",
            ),
            ComparisonPlanVariantEntry(
                variant_id="variant-1",
                display_name="Candidate",
                axis_value="model-b",
                configuration_id="semantic-config-1",
            ),
        ),
        baseline_variant_id="variant-0",
        repeats_per_case=1,
        maximum_logical_calls=2,
        maximum_provider_calls=20,
        cache_policy="bypass",
        cost_estimate_status="available",
        cost_estimate=Decimal("0.10"),
        cost_cap=Decimal("1.00"),
        currency="USD",
        launchable=True,
    )
    run = ComparisonRunEntry(
        comparison_id=comparison_id,
        experiment_plan_id=plan_id,
        plan_content_hash=plan_hash,
        status=ComparisonStatus.QUEUED,
        total_candidates=2,
        completed_candidates=0,
        failed_candidates=0,
        active_candidates=0,
        remaining_candidates=2,
        completed_cases=0,
        failed_cases=0,
        provider_calls=0,
        created_at=now,
        updated_at=now,
    )
    return FakeComparisonService(
        plan=plan,
        run=run,
        summary=ComparisonSummary(
            comparison_id=comparison_id,
            experiment_plan_id=plan_id,
            status=ComparisonStatus.QUEUED,
            evidence_status="unavailable",
            gate_status="unavailable",
            compatibility_state="unavailable",
            candidates=(),
            recommendation=ComparisonRecommendationSummary(
                state="unavailable",
                rationale_codes=("recommendation-not-recorded",),
            ),
            shared_setup=ComparisonSharedSetupSummary.from_evidence(None),
            provider_call_count=UnavailableValue(reason="comparison-cost-evidence-not-recorded"),
            known_partial_cost=UnavailableValue(reason="comparison-cost-evidence-not-recorded"),
            total_cost=UnavailableValue(reason="comparison-cost-evidence-not-recorded"),
            cost_complete=False,
            cost_unknown_reasons=("comparison-cost-evidence-not-recorded",),
            currency=UnavailableValue(reason="comparison-cost-evidence-not-recorded"),
        ),
    )


def _client(tmp_path: object, service: FakeComparisonService | None) -> TestClient:
    return TestClient(
        create_app(
            Settings(
                _env_file=None,
                data_root=tmp_path,
                workbench_enabled=False,
            ),
            evaluation_service=(None if service is None else cast(EvaluationOperations, service)),
        )
    )


def test_comparison_catalog_start_list_get_and_summary_contract(tmp_path: object) -> None:
    service = _service()
    plan_id = service.plan.experiment_plan_id
    assert service.run is not None
    comparison_id = service.run.comparison_id

    with _client(tmp_path, service) as client:
        plans = client.get("/api/v1/comparison-plans")
        started = client.post(
            "/api/v1/comparisons",
            json={"experiment_plan_id": plan_id},
        )
        listed = client.get("/api/v1/comparisons")
        fetched = client.get(f"/api/v1/comparisons/{comparison_id}")
        summary = client.get(f"/api/v1/comparisons/{comparison_id}/summary")
        invalid = client.post(
            "/api/v1/comparisons",
            json={"experiment_plan_id": plan_id, "unexpected": True},
        )

    for response in (plans, started, listed, fetched, summary):
        assert response.status_code in {200, 202}
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "path" not in response.text.casefold()
        assert str(tmp_path) not in response.text
    assert started.status_code == 202
    assert started.headers["location"] == f"/api/v1/comparisons/{comparison_id}"
    assert started.json()["comparison_id"] == comparison_id
    assert listed.json()["comparisons"][0]["experiment_plan_id"] == plan_id
    assert summary.json()["evidence_status"] == "unavailable"
    assert summary.json()["shared_setup"] == {
        "status": "unavailable",
        "safe_error_code": None,
        "provider_call_count": {
            "status": "unavailable",
            "reason": "setup-evidence-not-recorded",
        },
        "known_partial_cost": {
            "status": "unavailable",
            "reason": "setup-evidence-not-recorded",
        },
        "total_cost": {
            "status": "unavailable",
            "reason": "setup-evidence-not-recorded",
        },
        "currency": {
            "status": "unavailable",
            "reason": "setup-evidence-not-recorded",
        },
        "provider_calls_complete": False,
        "cost_complete": False,
        "unknown_reasons": ["setup-evidence-not-recorded"],
    }
    assert "setup_id" not in summary.text
    assert "request_id" not in summary.text
    assert "attempt_reference" not in summary.text
    assert "index_revision_id" not in summary.text
    assert service.starts == [plan_id]
    assert invalid.status_code == 422
    assert invalid.json() == {"error": {"code": "request_invalid"}}
    assert invalid.headers["cache-control"] == "no-store"


def test_failed_setup_unknown_aggregate_is_explicit_at_http_boundary(
    tmp_path: object,
) -> None:
    service = _service()
    assert service.run is not None and service.summary is not None
    unavailable = UnavailableValue(reason="setup-ledger-integrity-unavailable")
    service.summary = service.summary.model_copy(
        update={
            "status": ComparisonStatus.FAILED,
            "evidence_status": "incomplete",
            "shared_setup": ComparisonSharedSetupSummary(
                status="failed",
                safe_error_code="comparison-shared-setup-ledger-mismatch",
                provider_call_count=unavailable,
                known_partial_cost=Decimal(0),
                total_cost=unavailable,
                currency="USD",
                provider_calls_complete=False,
                cost_complete=False,
                unknown_reasons=("setup-ledger-integrity-unavailable",),
            ),
            "provider_call_count": unavailable,
            "total_cost": unavailable,
            "currency": "USD",
        }
    )

    with _client(tmp_path, service) as client:
        response = client.get(f"/api/v1/comparisons/{service.run.comparison_id}/summary")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    setup = response.json()["shared_setup"]
    assert setup["provider_calls_complete"] is False
    assert setup["provider_call_count"]["reason"] == ("setup-ledger-integrity-unavailable")
    assert setup["known_partial_cost"] == "0"
    assert setup["total_cost"]["status"] == "unavailable"
    assert response.json()["provider_call_count"]["status"] == "unavailable"
    assert "setup_id" not in response.text
    assert "request_id" not in response.text
    assert "attempt_reference" not in response.text


def test_incomplete_candidate_cost_exposes_lower_bound_without_blocking_gate_over_http(
    tmp_path: object,
) -> None:
    service = _service()
    assert service.run is not None and service.summary is not None
    service.summary = service.summary.model_copy(
        update={
            "status": ComparisonStatus.COMPLETED,
            "evidence_status": "available",
            "gate_status": "passed",
            "known_partial_cost": Decimal("0.02149750"),
            "total_cost": UnavailableValue(reason="comparison-cost-incomplete"),
            "cost_complete": False,
            "cost_unknown_reasons": ("input-usage-unknown",),
            "currency": "USD",
        }
    )

    with _client(tmp_path, service) as client:
        response = client.get(f"/api/v1/comparisons/{service.run.comparison_id}/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["known_partial_cost"] == "0.02149750"
    assert payload["total_cost"]["status"] == "unavailable"
    assert payload["cost_complete"] is False
    assert payload["cost_unknown_reasons"] == ["input-usage-unknown"]
    assert payload["gate_status"] == "passed"


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    (
        (ComparisonNotFoundError("comparison_plan_not_found"), 404, "comparison_plan_not_found"),
        (ComparisonConflictError("comparison_duplicate"), 409, "comparison_duplicate"),
        (
            ComparisonConflictError("comparison_exact_pricing_missing"),
            409,
            "comparison_prerequisite_unmet",
        ),
        (
            ComparisonValidationError("provider_call_cap_exceeded"),
            422,
            "comparison_preflight_invalid",
        ),
        (ComparisonCapacityError("comparison_capacity"), 429, "comparison_capacity"),
        (
            ComparisonUnavailableError("comparison_runtime_unavailable"),
            503,
            "comparison_unavailable",
        ),
    ),
)
def test_comparison_start_maps_stable_errors_and_headers(
    tmp_path: object,
    error: RuntimeError,
    expected_status: int,
    expected_code: str,
) -> None:
    service = _service()
    service.start_error = error

    with _client(tmp_path, service) as client:
        response = client.post(
            "/api/v1/comparisons",
            json={"experiment_plan_id": service.plan.experiment_plan_id},
        )

    assert response.status_code == expected_status
    assert response.json() == {"error": {"code": expected_code}}
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_comparison_routes_are_stably_unavailable_or_missing(tmp_path: object) -> None:
    with _client(tmp_path, None) as client:
        unavailable = client.get("/api/v1/comparison-plans")

    service = _service()
    service.run = None
    with _client(tmp_path, service) as client:
        missing = client.get("/api/v1/comparisons/comparison-missing")

    assert unavailable.status_code == 503
    assert unavailable.json() == {"error": {"code": "comparison_unavailable"}}
    assert missing.status_code == 404
    assert missing.json() == {"error": {"code": "comparison_not_found"}}


def test_comparison_artifact_manifest_and_exact_download_are_path_free(
    tmp_path: object,
) -> None:
    service = _service()
    assert service.run is not None
    comparison_id = service.run.comparison_id
    content = b'{"schema_version":"comparison-result-v1"}\n'
    digest = f"sha256:{sha256(content).hexdigest()}"
    now = datetime.now(UTC)
    descriptor = ComparisonArtifactDescriptor(
        artifact_id="comparison-report-json",
        schema_version="comparison-result-v1",
        format="json",
        media_type="application/json",
        sha256_digest=digest,
        byte_size=len(content),
        created_at=now,
    )
    service.manifest = ComparisonArtifactManifestView(
        comparison_id=comparison_id,
        experiment_plan_id=service.plan.experiment_plan_id,
        plan_content_hash=service.plan.plan_content_hash,
        manifest_content_hash="sha256:" + "a" * 64,
        artifacts=(descriptor,),
        created_at=now,
    )
    service.artifacts[descriptor.artifact_id] = ResolvedComparisonDownload(
        artifact_id=descriptor.artifact_id,
        content=content,
        media_type=descriptor.media_type,
        filename="comparison-report.json",
    )

    with _client(tmp_path, service) as client:
        manifest = client.get(f"/api/v1/comparisons/{comparison_id}/artifacts")
        downloaded = client.get(
            f"/api/v1/comparisons/{comparison_id}/artifacts/{descriptor.artifact_id}"
        )
        missing = client.get(f"/api/v1/comparisons/{comparison_id}/artifacts/unknown-artifact")

    assert manifest.status_code == 200
    assert "relative_path" not in manifest.text
    assert str(tmp_path) not in manifest.text
    assert downloaded.status_code == 200
    assert downloaded.content == content
    assert downloaded.headers["content-type"] == "application/json"
    assert downloaded.headers["content-disposition"] == (
        'attachment; filename="comparison-report.json"'
    )
    assert downloaded.headers["cache-control"] == "no-store"
    assert downloaded.headers["x-content-type-options"] == "nosniff"
    assert missing.status_code == 404
    assert missing.json() == {"error": {"code": "comparison_artifact_not_found"}}

    service.artifacts[descriptor.artifact_id] = ResolvedComparisonDownload(
        artifact_id=descriptor.artifact_id,
        content=content,
        media_type="text/html",
        filename="comparison-report.json",
    )
    with _client(tmp_path, service) as client:
        mismatched = client.get(
            f"/api/v1/comparisons/{comparison_id}/artifacts/{descriptor.artifact_id}"
        )
    assert mismatched.status_code == 503
    assert mismatched.json() == {"error": {"code": "comparison_artifact_unavailable"}}
