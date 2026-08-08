from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from rag_mvp.config.settings import Settings
from rag_mvp.domain.qa import ConversationRole
from rag_mvp.domain.retrieval import RetrievalMode
from rag_mvp.evaluation.answer_metrics import (
    ANSWER_COMPLETENESS_SCORER_VERSION,
    ANSWER_COMPLIANCE_SCORER_VERSION,
    GUIDED_REFUSAL_APPROPRIATENESS_SCORER_VERSION,
    REFUSAL_APPROPRIATENESS_SCORER_VERSION,
    STYLE_CONSISTENCY_SCORER_VERSION,
)
from rag_mvp.evaluation.dataset import (
    CorpusSnapshot,
    DatasetManifest,
    EvaluationDataset,
)
from rag_mvp.evaluation.grounding_metrics import (
    ADJUDICATED_FAITHFULNESS_SCORER_VERSION,
    CONTEXT_PRECISION_SCORER_VERSION,
    FAITHFULNESS_SCORER_VERSION,
    TEXT_SUPPORT_MATCHER_VERSION,
    TEXT_SUPPORT_NORMALIZATION_VERSION,
)
from rag_mvp.evaluation.plan import (
    OPENAI_COMPATIBLE_ADAPTER_VERSION,
    PLAN_IDENTITY_VERSION,
    EvaluationDatasetRegistry,
    EvaluationPlanError,
    build_evaluation_plan,
    source_code_revision,
)
from rag_mvp.evaluation.quality_gate import (
    ADVANCED_QUALITY_GATE_VERSION,
    QUALITY_GATE_VERSION,
)
from rag_mvp.evaluation.ragas_backend import (
    RAGAS_BACKEND_VERSION,
    RAGAS_CONTEXT_PRECISION_SCORER_VERSION,
    RAGAS_FAITHFULNESS_SCORER_VERSION,
    RAGAS_SCORING_PIPELINE_VERSION,
)
from rag_mvp.evaluation.runner import EvaluationEnvironment
from rag_mvp.evaluation.scoring import (
    ADVANCED_SCORING_PIPELINE_VERSION,
    SCORING_PIPELINE_VERSION,
)
from rag_mvp.qa.evidence_assessor import FACT_EVIDENCE_ASSESSOR_VERSION
from rag_mvp.qa.prompt import GENERATOR_OUTPUT_SCHEMA_VERSION, GENERATOR_PROMPT_VERSION
from rag_mvp.qa.query_rewrite import QUERY_REWRITE_VERSION
from rag_mvp.retrieval.rerank import RerankStage

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DATASETS_ROOT = _REPOSITORY_ROOT / "evaluations" / "datasets"
_CODE_REVISION = "sha256:" + ("1" * 64)


def _fake_dataset(path: Path, *, dataset_id: str, version: str) -> EvaluationDataset:
    return EvaluationDataset.model_construct(
        root=path.resolve(),
        manifest=DatasetManifest.model_construct(dataset_id=dataset_id, version=version),
        cases=(),
        corpus=CorpusSnapshot.model_construct(),
        category_counts={},
        metric_eligibility_counts={},
    )


def test_registry_resolves_the_real_validated_dataset_by_id_and_version() -> None:
    registry = EvaluationDatasetRegistry(_DATASETS_ROOT)

    implicit = registry.resolve("mvp-bilingual-rag")
    explicit = registry.resolve("mvp-bilingual-rag", "1.0.0")

    assert implicit.manifest.dataset_id == "mvp-bilingual-rag"
    assert implicit.manifest.version == "1.0.0"
    assert len(implicit.cases) == 8
    assert explicit == implicit


def test_registry_requires_explicit_version_when_multiple_versions_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {"one": "1.0.0", "two": "2.0.0"}
    for name in versions:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "manifest.json").write_text("{}", encoding="utf-8")

    def fake_load(path: str | Path, *, acceptance_mode: bool) -> EvaluationDataset:
        assert acceptance_mode
        resolved = Path(path)
        return _fake_dataset(
            resolved,
            dataset_id="dataset-a",
            version=versions[resolved.name],
        )

    monkeypatch.setattr("rag_mvp.evaluation.plan.load_dataset", fake_load)
    registry = EvaluationDatasetRegistry(tmp_path)

    with pytest.raises(EvaluationPlanError, match="evaluation_dataset_ambiguous"):
        registry.resolve("dataset-a")
    assert registry.resolve("dataset-a", "1.0.0").manifest.version == "1.0.0"
    assert registry.resolve("dataset-a", "2.0.0").manifest.version == "2.0.0"

    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    (duplicate / "manifest.json").write_text("{}", encoding="utf-8")
    versions["duplicate"] = "1.0.0"
    with pytest.raises(EvaluationPlanError, match="evaluation_dataset_ambiguous"):
        registry.resolve("dataset-a", "1.0.0")


def test_registry_fails_closed_for_missing_unsafe_or_malformed_entries(tmp_path: Path) -> None:
    registry = EvaluationDatasetRegistry(_DATASETS_ROOT)
    with pytest.raises(EvaluationPlanError, match="evaluation_dataset_not_found"):
        registry.resolve("dataset-does-not-exist")
    with pytest.raises(EvaluationPlanError, match="dataset_id_invalid"):
        registry.resolve("../mvp-v1")
    with pytest.raises(EvaluationPlanError, match="dataset_version_invalid"):
        registry.resolve("mvp-bilingual-rag", "latest")
    with pytest.raises(EvaluationPlanError, match="evaluation_dataset_registry_missing"):
        EvaluationDatasetRegistry(tmp_path / "missing").resolve("dataset-a")

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    with pytest.raises(EvaluationPlanError, match="evaluation_dataset_registry_entry_invalid"):
        EvaluationDatasetRegistry(tmp_path).resolve("dataset-a")


def test_source_code_revision_is_path_ordered_stable_and_ignores_bytecode(
    tmp_path: Path,
) -> None:
    package = tmp_path / "rag_mvp"
    nested = package / "nested"
    nested.mkdir(parents=True)
    (nested / "b.py").write_text("VALUE = 2\n", encoding="utf-8")
    (package / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "schema.json").write_text('{"version":1}\n', encoding="utf-8")

    first = source_code_revision(package)
    second = source_code_revision(package)
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "a.cpython-312.pyc").write_bytes(b"generated")
    with_bytecode = source_code_revision(package)

    assert re.fullmatch(r"sha256:[0-9a-f]{64}", first)
    assert second == first
    assert with_bytecode == first

    (nested / "b.py").write_text("VALUE = 3\n", encoding="utf-8")
    assert source_code_revision(package) != first


def test_source_code_revision_rejects_missing_or_empty_package(tmp_path: Path) -> None:
    with pytest.raises(EvaluationPlanError, match="source_package_missing"):
        source_code_revision(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(EvaluationPlanError, match="source_package_empty"):
        source_code_revision(empty)


def test_build_plan_pins_complete_safe_identity_and_maps_case_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = EvaluationDatasetRegistry(_DATASETS_ROOT).resolve(
        "mvp-bilingual-rag",
        "1.0.0",
    )
    settings = Settings(
        _env_file=None,
        environment="test",
        provider_backend="openai",
        openai_api_key="acceptance-secret",
        openai_base_url="https://models.example.test/v1",
        embedding_model="embedding-model-v2",
        embedding_dimension=1024,
        generation_model="generation-model-v3",
        reranking_model="reranking-model-v1",
        default_retrieval_mode="hybrid-rerank",
        dense_candidate_limit=24,
        lexical_candidate_limit=22,
        rerank_candidate_limit=9,
        context_chunk_limit=5,
        rrf_k=61,
        dense_weight=1.2,
        lexical_weight=1.4,
        retrieval_cache_enabled=True,
        chunk_target_tokens=500,
        chunk_overlap_tokens=80,
        qa_minimum_support_score=0.72,
        pricing_version="pricing-2026-08",
    )
    monkeypatch.setattr("rag_mvp.evaluation.plan.source_code_revision", lambda: _CODE_REVISION)

    plan = build_evaluation_plan(dataset, settings, "run_20260807_001")

    identity = plan.identity
    assert identity.dataset_id == dataset.manifest.dataset_id
    assert identity.dataset_version == dataset.manifest.version
    assert identity.dataset_hash == dataset.manifest.content_hash
    assert identity.corpus_version == dataset.corpus.manifest.version
    assert identity.corpus_hash == dataset.corpus.manifest.content_hash
    assert identity.configuration_id == settings.evaluation_configuration_identity
    assert identity.runtime_configuration_id == settings.runtime_configuration_identity
    assert identity.code_revision == _CODE_REVISION
    assert identity.cache_policy == "bypass"
    assert identity.prompt_versions == {
        "generation": GENERATOR_PROMPT_VERSION,
        "generation-output-schema": GENERATOR_OUTPUT_SCHEMA_VERSION,
        "query-rewrite": QUERY_REWRITE_VERSION,
        "reranking": RerankStage.PROMPT_VERSION,
        "reranking-parser": RerankStage.PARSER_VERSION,
    }
    endpoint_digest = hashlib.sha256(settings.openai_base_url.encode()).hexdigest()[:16]
    provider_alias = f"openai-compatible-{endpoint_digest}"
    assert identity.provider_identities == {
        "backend": "openai",
        "embedding": provider_alias,
        "generation": provider_alias,
        "reranking": provider_alias,
        "adapter": OPENAI_COMPATIBLE_ADAPTER_VERSION,
    }
    assert identity.model_identities == {
        "embedding": "embedding-model-v2",
        "generation": "generation-model-v3",
        "reranking": "reranking-model-v1",
    }
    assert identity.generation_settings == {
        "temperature": 0.0,
        "maximum_question_characters": 4096,
        "maximum_output_tokens": 512,
        "response_format": "json_object",
        "max_tokens_parameter": "max_completion_tokens",
        "provider_timeout_seconds": 8.0,
        "provider_retry_limit": 1,
        "qa_deadline_seconds": 9.5,
        "qa_generation_budget_seconds": 6.0,
        "qa_finalization_budget_seconds": 0.6,
    }
    assert identity.embedding_identity == {
        "provider": provider_alias,
        "model": "embedding-model-v2",
        "dimension": 1024,
        "normalization": "none",
        "adapter_version": OPENAI_COMPATIBLE_ADAPTER_VERSION,
        "send_dimensions": True,
    }
    assert identity.chunking_identity == {
        "extraction_version": "extraction-v1",
        "normalization_version": "unicode-nfc-lines-v1",
        "chunking_version": "structure-page-token-v1",
        "tokenizer_version": "unicode-word-cjk-v1",
            "target_tokens": 500,
            "overlap_tokens": 80,
            "parent_target_tokens": None,
            "ocr_enabled": True,
        "ocr_languages": "chi_sim+eng",
    }
    assert identity.retrieval_configuration["identity_version"] == PLAN_IDENTITY_VERSION
    assert identity.retrieval_configuration["mode"] == "hybrid-rerank"
    assert identity.retrieval_configuration["dense_candidate_limit"] == 24
    assert identity.retrieval_configuration["lexical_candidate_limit"] == 22
    assert identity.retrieval_configuration["rerank_candidate_limit"] == 9
    assert identity.retrieval_configuration["context_chunk_limit"] == 5
    assert identity.retrieval_configuration["rrf_k"] == 61
    assert identity.retrieval_configuration["dense_weight"] == 1.2
    assert identity.retrieval_configuration["lexical_weight"] == 1.4
    assert identity.retrieval_configuration["retrieval_cache_enabled"] is True
    assert identity.retrieval_configuration["retrieval_cache_ttl_seconds"] == 300.0
    assert identity.retrieval_configuration["retrieval_cache_max_entries"] == 256
    assert identity.retrieval_configuration["qa_retrieval_budget_seconds"] == 4.0
    assert identity.retrieval_configuration["qa_embedding_budget_seconds"] == 0.8
    assert identity.retrieval_configuration["qa_dense_retrieval_budget_seconds"] == 0.8
    assert identity.retrieval_configuration["qa_bm25_budget_seconds"] == 0.8
    assert identity.retrieval_configuration["qa_fusion_budget_seconds"] == 0.2
    assert identity.retrieval_configuration["qa_evidence_assessment_budget_seconds"] == 4.0
    assert identity.retrieval_configuration["minimum_support_score"] == 0.72
    assert identity.scorer_versions == {
        "faithfulness": FAITHFULNESS_SCORER_VERSION,
        "context-precision": CONTEXT_PRECISION_SCORER_VERSION,
        "answer-completeness": ANSWER_COMPLETENESS_SCORER_VERSION,
        "style-consistency": STYLE_CONSISTENCY_SCORER_VERSION,
        "refusal-appropriateness": REFUSAL_APPROPRIATENESS_SCORER_VERSION,
        "answer-compliance": ANSWER_COMPLIANCE_SCORER_VERSION,
        "scoring-pipeline": SCORING_PIPELINE_VERSION,
        "quality-gate": QUALITY_GATE_VERSION,
        "advanced-quality-gate": ADVANCED_QUALITY_GATE_VERSION,
        "evaluation-backend": "legacy-v1",
    }
    assert identity.pricing_version == "pricing-2026-08"
    assert identity.random_seeds == {"case-order": 0, "scoring": 0}
    assert isinstance(identity.environment, EvaluationEnvironment)
    assert identity.environment.deployment == "test"

    assert len(plan.cases) == len(dataset.cases)
    assert all(case.retrieval_mode is RetrievalMode.HYBRID_RERANK for case in plan.cases)
    multi_turn = next(case for case in plan.cases if case.case_id == "case-multi-turn-001")
    assert tuple(turn.role for turn in multi_turn.history) == (
        ConversationRole.USER,
        ConversationRole.ASSISTANT,
    )
    serialized = plan.model_dump_json()
    assert "acceptance-secret" not in serialized
    assert settings.openai_base_url not in serialized


def test_build_v2_plan_pins_the_exact_advanced_scorer_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = EvaluationDatasetRegistry(_DATASETS_ROOT).resolve(
        "original-pdf-acceptance",
        "2.0.0",
    )
    settings = Settings(
        _env_file=None,
        environment="test",
        provider_backend="openai",
        openai_api_key="unit-test-key",
    )
    monkeypatch.setattr("rag_mvp.evaluation.plan.source_code_revision", lambda: _CODE_REVISION)

    plan = build_evaluation_plan(dataset, settings, "advanced_v2_run")

    assert plan.identity.retrieval_configuration["minimum_support_score"] == 0.45
    assert plan.identity.scorer_versions == {
        "faithfulness": ADJUDICATED_FAITHFULNESS_SCORER_VERSION,
        "faithfulness-text-matcher": TEXT_SUPPORT_MATCHER_VERSION,
        "faithfulness-text-normalization": TEXT_SUPPORT_NORMALIZATION_VERSION,
        "context-precision": CONTEXT_PRECISION_SCORER_VERSION,
        "answer-completeness": ANSWER_COMPLETENESS_SCORER_VERSION,
        "style-consistency": STYLE_CONSISTENCY_SCORER_VERSION,
        "refusal-appropriateness": GUIDED_REFUSAL_APPROPRIATENESS_SCORER_VERSION,
        "answer-compliance": ANSWER_COMPLIANCE_SCORER_VERSION,
        "scoring-pipeline": ADVANCED_SCORING_PIPELINE_VERSION,
        "quality-gate": QUALITY_GATE_VERSION,
        "advanced-quality-gate": ADVANCED_QUALITY_GATE_VERSION,
        "evaluation-backend": "legacy-v1",
    }


def test_build_v2_ragas_plan_changes_only_evaluation_scorer_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = EvaluationDatasetRegistry(_DATASETS_ROOT).resolve(
        "original-pdf-acceptance",
        "2.0.0",
    )
    settings = Settings(
        _env_file=None,
        environment="test",
        provider_backend="openai",
        openai_api_key="unit-test-key",
        evaluation_scorer_backend="ragas",
        evaluation_ragas_judge_model="judge-model-v1",
    )
    monkeypatch.setattr("rag_mvp.evaluation.plan.source_code_revision", lambda: _CODE_REVISION)

    plan = build_evaluation_plan(dataset, settings, "advanced_ragas_v2_run")

    identity = plan.identity
    assert (
        identity.retrieval_configuration["fact_evidence_assessor_version"]
        == FACT_EVIDENCE_ASSESSOR_VERSION
    )
    assert identity.model_identities["evaluation-judge"] == "judge-model-v1"
    assert identity.scorer_versions["evaluation-backend"] == RAGAS_BACKEND_VERSION
    assert identity.scorer_versions["faithfulness"] == RAGAS_FAITHFULNESS_SCORER_VERSION
    assert (
        identity.scorer_versions["context-precision"]
        == RAGAS_CONTEXT_PRECISION_SCORER_VERSION
    )
    assert identity.scorer_versions["scoring-pipeline"] == RAGAS_SCORING_PIPELINE_VERSION
    assert "faithfulness-text-matcher" not in identity.scorer_versions
    assert "faithfulness-text-normalization" not in identity.scorer_versions


def test_build_ragas_plan_rejects_legacy_dataset_before_execution() -> None:
    dataset = EvaluationDatasetRegistry(_DATASETS_ROOT).resolve("mvp-bilingual-rag", "1.0.0")
    settings = Settings(
        _env_file=None,
        environment="test",
        provider_backend="openai",
        openai_api_key="unit-test-key",
        evaluation_scorer_backend="ragas",
        chunk_target_tokens=500,
        chunk_overlap_tokens=80,
    )

    with pytest.raises(EvaluationPlanError, match="evaluation_ragas_dataset_v2_required"):
        build_evaluation_plan(dataset, settings, "legacy_ragas_run")


def test_build_plan_fails_when_dataset_derivation_differs_from_runtime_settings() -> None:
    dataset = EvaluationDatasetRegistry(_DATASETS_ROOT).resolve("mvp-bilingual-rag")
    settings = Settings(
        _env_file=None,
        environment="test",
        chunk_target_tokens=501,
        chunk_overlap_tokens=80,
    )

    with pytest.raises(EvaluationPlanError, match="dataset_settings_derivation_mismatch"):
        build_evaluation_plan(dataset, settings, "run_001")


def test_build_plan_rejects_unsafe_run_id() -> None:
    dataset = EvaluationDatasetRegistry(_DATASETS_ROOT).resolve("mvp-bilingual-rag")
    settings = Settings(_env_file=None, environment="test")

    with pytest.raises(EvaluationPlanError, match="evaluation_run_id_invalid"):
        build_evaluation_plan(dataset, settings, "../run")
