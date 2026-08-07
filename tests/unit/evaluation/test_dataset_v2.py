from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from rag_mvp.evaluation import run_evaluation
from rag_mvp.evaluation.dataset import (
    ACCEPTANCE_V2_MINIMUM_CASES,
    ACCEPTANCE_V2_MINIMUM_CHALLENGE_COUNTS,
    ACCEPTANCE_V2_MINIMUM_LANGUAGE_COUNTS,
    ACCEPTANCE_V2_MINIMUM_MULTI_TURN_CASES,
    ACCEPTANCE_V2_REQUIRED_METRICS,
    ChallengeTag,
    DatasetManifestV2,
    DatasetValidationError,
    EvaluationCaseV2,
    EvaluationLanguage,
    EvaluationMetricV2,
    calculate_dataset_content_hash,
    load_dataset,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPOSITORY_ROOT / "evaluations" / "datasets" / "acceptance-v2"
MVP_V1_ROOT = REPOSITORY_ROOT / "evaluations" / "datasets" / "mvp-v1"


def _copy_dataset(tmp_path: Path) -> Path:
    copied = tmp_path / "acceptance-v2"
    shutil.copytree(DATASET_ROOT, copied)
    return copied


def _load_case_payloads(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in (root / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _write_cases_and_refresh_hash(root: Path, payloads: list[dict[str, Any]]) -> None:
    (root / "cases.jsonl").write_text(
        "\n".join(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            for payload in payloads
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = root / "manifest.json"
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = DatasetManifestV2.model_validate(raw_manifest)
    cases = tuple(EvaluationCaseV2.model_validate(item) for item in payloads)
    raw_manifest["content_hash"] = calculate_dataset_content_hash(manifest, cases)
    manifest_path.write_text(
        json.dumps(raw_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_loads_discriminating_acceptance_v2_dataset() -> None:
    dataset = load_dataset(DATASET_ROOT)

    assert isinstance(dataset.manifest, DatasetManifestV2)
    assert dataset.manifest.contract_version == "2.0.0"
    assert len(dataset.cases) == ACCEPTANCE_V2_MINIMUM_CASES
    assert set(dataset.metric_eligibility_counts) == ACCEPTANCE_V2_REQUIRED_METRICS
    assert dataset.metric_eligibility_counts[EvaluationMetricV2.ANSWER_COMPLIANCE] > 0
    assert dataset.corpus.source_manifest is not None
    assert (
        dataset.corpus.source_manifest.content_hash == dataset.corpus.manifest.source_manifest_hash
    )
    assert len(dataset.corpus.documents) == 7

    for language, minimum in ACCEPTANCE_V2_MINIMUM_LANGUAGE_COUNTS.items():
        assert dataset.language_counts[language] >= minimum
    assert sum(bool(case.history) for case in dataset.cases) >= (
        ACCEPTANCE_V2_MINIMUM_MULTI_TURN_CASES
    )
    assert {case.language for case in dataset.cases if case.history} >= {
        EvaluationLanguage.CHINESE,
        EvaluationLanguage.ENGLISH,
    }
    for tag, minimum in ACCEPTANCE_V2_MINIMUM_CHALLENGE_COUNTS.items():
        assert dataset.challenge_counts[tag] >= minimum
    assert all(
        isinstance(case, EvaluationCaseV2) and case.compliance_obligations for case in dataset.cases
    )
    cases = {case.case_id: case for case in dataset.cases}
    zh_code = cases["accept-zh-001"]
    zh_travel = cases["accept-zh-003"]
    en_endpoint = cases["accept-en-004"]
    assert isinstance(zh_code, EvaluationCaseV2)
    assert isinstance(zh_travel, EvaluationCaseV2)
    assert isinstance(en_endpoint, EvaluationCaseV2)
    assert "权威的 RAG 升级代码是 `OPS-RAG-7421`。" in (
        zh_code.expected_facts[0].approved_propositions
    )
    assert "当前有效政策规定\N{FULLWIDTH COMMA}境内航班经济舱票价报销上限为人民币 1,800 元。" in (
        zh_travel.expected_facts[0].approved_propositions
    )
    assert "The production query endpoint is `POST /v2/knowledge/query`." in (
        en_endpoint.expected_facts[0].approved_propositions
    )


def test_acceptance_v2_rejects_challenge_coverage_below_declared_minimum(
    tmp_path: Path,
) -> None:
    copied = _copy_dataset(tmp_path)
    payloads = _load_case_payloads(copied)
    for payload in payloads:
        if ChallengeTag.SCANNED_DOCUMENT.value in payload["challenge_tags"]:
            payload["challenge_tags"].remove(ChallengeTag.SCANNED_DOCUMENT.value)
            break
    _write_cases_and_refresh_hash(copied, payloads)

    with pytest.raises(DatasetValidationError, match="challenge coverage"):
        load_dataset(copied)


@pytest.mark.parametrize("defect", ["required-guidance-absent", "answer-guidance-present"])
def test_acceptance_v2_rejects_inconsistent_refusal_guidance_obligations(
    tmp_path: Path,
    defect: str,
) -> None:
    copied = _copy_dataset(tmp_path)
    payloads = _load_case_payloads(copied)
    if defect == "required-guidance-absent":
        payload = next(item for item in payloads if item["case_id"] == "accept-en-009")
        obligation = next(
            item for item in payload["compliance_obligations"] if item["kind"] == "refusal-guidance"
        )
        obligation["expected_values"] = ["absent"]
    else:
        payload = next(item for item in payloads if item["case_id"] == "accept-en-001")
        payload["response_instructions"].append(
            {
                "instruction_id": "unexpected-guidance",
                "text": "Include refusal guidance.",
            }
        )
        payload["compliance_obligations"].append(
            {
                "obligation_id": "unexpected-guidance-v2",
                "version": "2.0.0",
                "instruction_id": "unexpected-guidance",
                "kind": "refusal-guidance",
                "description": "Unexpected answer guidance requirement.",
                "expected_values": ["present"],
            }
        )
    (copied / "cases.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in payloads) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="invalid schema in evaluation cases"):
        load_dataset(copied)


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "unrelated",
        "trivial",
        "unbound-alternative",
        "missing-propositions",
        "salad-proposition",
        "reversed-proposition",
        "reversed-activation-proposition",
    ],
)
def test_acceptance_v2_rejects_untrusted_expected_fact_support_anchors(
    tmp_path: Path,
    defect: str,
) -> None:
    copied = _copy_dataset(tmp_path)
    payloads = _load_case_payloads(copied)
    fact = payloads[0]["expected_facts"][0]
    if defect == "missing":
        fact.pop("support_anchor_groups")
    elif defect == "unrelated":
        fact["support_anchor_groups"][0]["alternatives"] = ["moon cheese"]
    elif defect == "trivial":
        fact["support_anchor_groups"][0]["alternatives"].append("the")
    elif defect == "unbound-alternative":
        fact["support_anchor_groups"][0]["alternatives"].append("moon cheese")
    elif defect == "missing-propositions":
        fact.pop("approved_propositions")
    elif defect == "salad-proposition":
        fact["approved_propositions"].append("moon cheese OPS-RAG-7421")
    elif defect == "reversed-proposition":
        owner_fact = payloads[-1]["expected_facts"][0]
        owner_fact["approved_propositions"].append("Escalation owns the RAG Operations Desk.")
    else:
        activation_fact = payloads[13]["expected_facts"][0]
        activation_fact["approved_propositions"].append(
            "A validated index activates one atomic revision switch."
        )
    (copied / "cases.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in payloads) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="invalid schema in evaluation cases"):
        load_dataset(copied)


def test_acceptance_v2_rejects_source_manifest_or_source_tampering(tmp_path: Path) -> None:
    copied = _copy_dataset(tmp_path)
    source_manifest = copied / "corpus" / "source-manifest.json"
    source_manifest.write_text(
        source_manifest.read_text(encoding="utf-8").replace(
            "technical-api-spec-en.md",
            "different.md",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError, match="source manifest"):
        load_dataset(copied)

    copied = _copy_dataset(tmp_path / "source")
    source = copied / "corpus" / "sources" / "technical-api-spec-en.md"
    source.write_text(source.read_text(encoding="utf-8") + "\nTampered.\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="source artifact hash mismatch"):
        load_dataset(copied)


def test_mvp_v1_dataset_assets_remain_byte_identical() -> None:
    assert _sha256(MVP_V1_ROOT / "manifest.json") == (
        "2A002C3CDC1F1E2BD2CA5FAB1ACA051E39458130B83B4A1AC03C1CF9C66AA880"
    )
    assert _sha256(MVP_V1_ROOT / "cases.jsonl") == (
        "86494092D0FCF577E21DFEA971CA573E34E387D99BCD00B07B8958C17D4662F0"
    )
    assert _sha256(MVP_V1_ROOT / "corpus" / "manifest.json") == (
        "D94428223384329908724DD7EE062FFC88C84F6E9CBF144AEB317B34A64807C7"
    )
    assert load_dataset(MVP_V1_ROOT).manifest.version == "1.0.0"


async def test_invalid_dataset_fails_before_settings_or_provider_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def unexpected_settings(*args: object, **kwargs: object) -> None:
        raise AssertionError("provider settings must not be inspected before dataset validation")

    monkeypatch.setattr(run_evaluation, "_settings", unexpected_settings)
    with pytest.raises(DatasetValidationError, match="dataset root"):
        await run_evaluation.run_real_evaluation(
            dataset_path=tmp_path / "missing-dataset",
            data_root=tmp_path / "data",
            output_root=tmp_path / "results",
            run_id="fail-fast-dataset",
            profile="accepted",
        )


async def test_invalid_v2_support_contract_fails_before_provider_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    copied = _copy_dataset(tmp_path)
    payloads = _load_case_payloads(copied)
    payloads[0]["expected_facts"][0].pop("support_anchor_groups")
    (copied / "cases.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in payloads) + "\n",
        encoding="utf-8",
    )

    def unexpected_settings(*args: object, **kwargs: object) -> None:
        raise AssertionError("provider settings must not be inspected before dataset validation")

    monkeypatch.setattr(run_evaluation, "_settings", unexpected_settings)
    with pytest.raises(DatasetValidationError, match="invalid schema in evaluation cases"):
        await run_evaluation.run_real_evaluation(
            dataset_path=copied,
            data_root=tmp_path / "data",
            output_root=tmp_path / "results",
            run_id="fail-fast-support-contract",
            profile="accepted",
        )
