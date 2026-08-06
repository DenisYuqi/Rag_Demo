from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_mvp.performance import run_load_test
from rag_mvp.performance.evidence_bundle import PerformanceEvidenceIdentity
from rag_mvp.performance.load_test import HttpLoadTestConfig, LoadScenario


def _scenario_payload() -> list[dict[str, str]]:
    return [
        {
            "scenario_id": "policy-1",
            "owner_id": "load-owner",
            "question": "What is the policy?",
            "mode": "hybrid",
        }
    ]


def _workload_digest() -> str:
    return HttpLoadTestConfig(
        run_id="acceptance-run",
        expected_configuration_id="config-v1",
        base_url="http://testserver",
        scenarios=(LoadScenario.model_validate(_scenario_payload()[0]),),
        warmup_attempts=1,
        concurrency=5,
        target_successes=5,
    ).workload_digest


def _write_inputs(tmp_path: Path, *, valid_pricing: bool = True) -> tuple[Path, Path]:
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text(json.dumps(_scenario_payload()), encoding="utf-8")
    pricing = tmp_path / "pricing.json"
    pricing.write_text(
        json.dumps(
            {
                "pricing_version": "pricing-v1",
                "currency": "USD",
                "rates": (
                    [
                        {
                            "role": "generation",
                            "provider": "test",
                            "model": "model-v1",
                            "input_per_million": "1",
                            "output_per_million": "2",
                        }
                    ]
                    if valid_pricing
                    else []
                ),
                "source_references": ["https://example.test/pricing-v1"],
            }
        ),
        encoding="utf-8",
    )
    return scenarios, pricing


def _args(
    tmp_path: Path,
    scenario_file: Path,
    pricing_file: Path,
    *,
    confirmed: bool = True,
    expected_workload_digest: str | None = None,
) -> Namespace:
    return Namespace(
        base_url="http://testserver",
        scenario_file=scenario_file,
        output=tmp_path / "performance.json",
        run_id="acceptance-run",
        code_revision="revision-v1",
        configuration_id="config-v1",
        service_version="service-v1",
        model=["generation=test/model-v1"],
        instance_identity=None,
        expected_workload_digest=expected_workload_digest or _workload_digest(),
        confirm_acceptance_run=confirmed,
        metric_reference=["metric-ref-1"],
        log_reference=["log-ref-1"],
        trace_reference=[],
        pricing_evidence=pricing_file,
        warmup_attempts=1,
        concurrency=5,
        target_successes=5,
        max_attempts=5,
        retry_limit=0,
        request_timeout_seconds=1.0,
        instance_count=1,
    )


@pytest.mark.asyncio
async def test_run_requires_explicit_acceptance_confirmation_before_io(tmp_path: Path) -> None:
    args = _args(
        tmp_path,
        tmp_path / "missing-scenarios.json",
        tmp_path / "missing-pricing.json",
        confirmed=False,
    )

    with pytest.raises(ValueError, match="explicit quota and spend confirmation"):
        await run_load_test._run(args)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("pricing", "digest"))
async def test_run_rejects_bad_preflight_before_http_traffic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    scenario_file, pricing_file = _write_inputs(
        tmp_path,
        valid_pricing=failure != "pricing",
    )
    args = _args(
        tmp_path,
        scenario_file,
        pricing_file,
        expected_workload_digest=("sha256:" + "0" * 64 if failure == "digest" else None),
    )
    harness_created = False

    class ForbiddenHarness:
        def __init__(self, config: object) -> None:
            del config
            nonlocal harness_created
            harness_created = True

    monkeypatch.setattr(run_load_test, "HttpLoadTestHarness", ForbiddenHarness)

    with pytest.raises(ValueError):
        await run_load_test._run(args)

    assert harness_created is False


@pytest.mark.asyncio
async def test_run_binds_http_instance_and_cost_pricing_digest_into_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario_file, pricing_file = _write_inputs(tmp_path)
    args = _args(tmp_path, scenario_file, pricing_file)
    observed_attempt = SimpleNamespace(
        succeeded=True,
        instance_identity="instance-http-1",
    )
    report = SimpleNamespace(
        run_id="acceptance-run",
        attempt_count=1,
        success_count=1,
        error_count=0,
        warmup=SimpleNamespace(attempts=(observed_attempt,)),
        attempts=(observed_attempt,),
    )
    pricing_digest = "sha256:" + "a" * 64
    cost = SimpleNamespace(pricing_evidence_digest=pricing_digest)
    captured: dict[str, object] = {}

    class FakeHarness:
        def __init__(self, config: HttpLoadTestConfig) -> None:
            captured["config"] = config

        async def run(self) -> object:
            return report

    def fake_bundle(
        received_report: object,
        *,
        identity: object,
        references: object,
        cost: object,
    ) -> dict[str, object]:
        del references
        captured.update(report=received_report, identity=identity, cost=cost)
        return {"decision": {"valid": True, "passed": True}}

    monkeypatch.setattr(run_load_test, "HttpLoadTestHarness", FakeHarness)
    monkeypatch.setattr(run_load_test, "calculate_performance_cost", lambda *_: cost)
    monkeypatch.setattr(run_load_test, "build_performance_evidence_bundle", fake_bundle)
    monkeypatch.setattr(run_load_test, "write_performance_evidence_bundle", lambda *_: None)

    assert await run_load_test._run(args) is True

    identity = captured["identity"]
    assert isinstance(identity, PerformanceEvidenceIdentity)
    assert identity.instance_identity == "instance-http-1"
    assert identity.pricing_evidence_digest == pricing_digest
    assert captured["cost"] is cost
