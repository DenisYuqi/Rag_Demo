from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag_mvp.config.settings import Settings
from rag_mvp.domain.ingestion import (
    EmbeddingSpaceIdentity,
    IndexRevision,
    IndexRevisionStatus,
)
from rag_mvp.evaluation.corpus import EvaluationCorpusInstaller
from rag_mvp.evaluation.environment import (
    EvaluationEnvironmentError,
    EvaluationIndexReuseKey,
    EvaluationSuiteWorkspaceManager,
)
from rag_mvp.evaluation.plan import EvaluationDatasetRegistry, build_evaluation_plan
from rag_mvp.ingestion.service import IngestionService
from rag_mvp.providers.fakes import DeterministicEmbeddingProvider
from rag_mvp.providers.models import (
    EmbeddingSpaceIdentity as ProviderEmbeddingSpaceIdentity,
)
from rag_mvp.providers.models import NormalizationPolicy
from rag_mvp.retrieval.tokenizer import BILINGUAL_TOKENIZER_IDENTITY

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DATASETS_ROOT = _REPOSITORY_ROOT / "evaluations" / "datasets"


def _dataset_and_plan(tmp_path: Path) -> tuple[object, object]:
    dataset = EvaluationDatasetRegistry(_DATASETS_ROOT).resolve(
        "original-pdf-acceptance",
        "2.0.0",
    )
    settings = Settings(
        data_root=tmp_path / "online",
        chunk_target_tokens=dataset.corpus.manifest.derivation.target_tokens,
        chunk_overlap_tokens=dataset.corpus.manifest.derivation.overlap_tokens,
        _env_file=None,
    )
    return dataset, build_evaluation_plan(dataset, settings, "candidate-a")


def _matching_revision(key: EvaluationIndexReuseKey) -> IndexRevision:
    embedding = {item.name: item.value for item in key.embedding_identity}
    index = {item.name: item.value for item in key.index_identity}
    chunking = {item.name: item.value for item in key.chunking_identity}
    return IndexRevision(
        revision_id="revision-evaluation",
        status=IndexRevisionStatus.ACTIVE,
        active_sources={item.source_id: item.version for item in key.active_sources},
        chunk_set_digest=key.chunk_set_digest,
        embedding_space=EmbeddingSpaceIdentity(
            provider_alias=str(embedding["provider"]),
            model=str(embedding["model"]),
            dimension=int(embedding["dimension"]),
            normalization=str(embedding["normalization"]),
            adapter_version=str(embedding["adapter_version"]),
        ),
        extraction_version=str(chunking["extraction_version"]),
        chunking_version=str(chunking["chunking_version"]),
        tokenizer_version=str(index["lexical_tokenizer_identity"]),
        dense_index_path="indexes/dense",
        lexical_index_path="indexes/lexical.json",
        chunk_count=key.chunk_count,
        dense_schema_version=str(index["dense_schema_version"]),
        dense_metric=str(index["dense_metric"]),
        lexical_schema_version=str(index["lexical_schema_version"]),
        lexical_algorithm_version=str(index["lexical_algorithm_version"]),
        lexical_k1=float(index["lexical_k1"]),
        lexical_b=float(index["lexical_b"]),
        record_digest_algorithm=key.record_digest_algorithm,
        published_at=datetime(2026, 8, 7, tzinfo=UTC),
    )


def test_index_reuse_key_ignores_generation_axis_but_includes_embedding_identity(
    tmp_path: Path,
) -> None:
    dataset, baseline = _dataset_and_plan(tmp_path)
    baseline_key = EvaluationIndexReuseKey.from_plan(baseline, dataset)
    generation_variant = baseline.model_copy(
        update={
            "identity": baseline.identity.model_copy(
                update={
                    "model_identities": {
                        **baseline.identity.model_identities,
                        "generation": "generation-alternative",
                    }
                }
            )
        }
    )
    embedding_variant = baseline.model_copy(
        update={
            "identity": baseline.identity.model_copy(
                update={
                    "embedding_identity": {
                        **baseline.identity.embedding_identity,
                        "model": "embedding-alternative",
                    }
                }
            )
        }
    )

    assert EvaluationIndexReuseKey.from_plan(generation_variant, dataset) == baseline_key
    assert EvaluationIndexReuseKey.from_plan(embedding_variant, dataset) != baseline_key


def test_reuse_requires_every_active_revision_identity() -> None:
    dataset = EvaluationDatasetRegistry(_DATASETS_ROOT).resolve(
        "original-pdf-acceptance",
        "2.0.0",
    )
    settings = Settings(
        chunk_target_tokens=dataset.corpus.manifest.derivation.target_tokens,
        chunk_overlap_tokens=dataset.corpus.manifest.derivation.overlap_tokens,
        _env_file=None,
    )
    plan = build_evaluation_plan(dataset, settings, "candidate-a")
    key = EvaluationIndexReuseKey.from_plan(plan, dataset)

    key.verify_revision(_matching_revision(key))
    wrong = _matching_revision(key).model_copy(update={"chunk_set_digest": "wrong-digest"})
    with pytest.raises(
        EvaluationEnvironmentError,
        match="evaluation_index_reuse_identity_mismatch",
    ):
        key.verify_revision(wrong)
    wrong_lexical_tokenizer = _matching_revision(key).model_copy(
        update={"tokenizer_version": "foreign-lexical-tokenizer-v1"}
    )
    with pytest.raises(
        EvaluationEnvironmentError,
        match="evaluation_index_reuse_identity_mismatch",
    ):
        key.verify_revision(wrong_lexical_tokenizer)


@pytest.mark.asyncio
async def test_reuse_accepts_real_installer_lexical_tokenizer_identity(
    tmp_path: Path,
) -> None:
    dataset, plan = _dataset_and_plan(tmp_path)
    key = EvaluationIndexReuseKey.from_plan(plan, dataset)
    embedding = {item.name: item.value for item in key.embedding_identity}
    chunking = {item.name: item.value for item in key.chunking_identity}
    index = {item.name: item.value for item in key.index_identity}
    provider_identity = ProviderEmbeddingSpaceIdentity(
        provider=str(embedding["provider"]),
        model=str(embedding["model"]),
        dimension=int(embedding["dimension"]),
        normalization=NormalizationPolicy(str(embedding["normalization"])),
        adapter_version=str(embedding["adapter_version"]),
    )
    service = IngestionService.create(
        tmp_path / "installed",
        DeterministicEmbeddingProvider(provider_identity),
    )
    try:
        installed = await EvaluationCorpusInstaller(service).install(dataset)

        assert chunking["tokenizer_version"] == "unicode-word-cjk-v1"
        assert index["lexical_tokenizer_identity"] == BILINGUAL_TOKENIZER_IDENTITY
        assert installed.revision.tokenizer_version == BILINGUAL_TOKENIZER_IDENTITY
        key.verify_revision(installed.revision)
    finally:
        service.close()


def test_suite_workspace_is_shared_only_for_equal_reuse_keys(tmp_path: Path) -> None:
    dataset, plan = _dataset_and_plan(tmp_path)
    baseline = EvaluationIndexReuseKey.from_plan(plan, dataset)
    different = baseline.model_copy(
        update={
            "embedding_identity": tuple(
                item.model_copy(update={"value": "embedding-alternative"})
                if item.name == "model"
                else item
                for item in baseline.embedding_identity
            )
        }
    )
    manager = EvaluationSuiteWorkspaceManager(tmp_path / "online")

    first = manager.workspace_for("suite-001", baseline)
    repeated = manager.workspace_for("suite-001", baseline)
    isolated = manager.workspace_for("suite-001", different)

    assert first == repeated
    assert first != isolated
    assert first.is_relative_to((tmp_path / "online").resolve())
    assert not first.is_relative_to((tmp_path / "online" / "indexes").resolve())


def test_suite_workspace_rejects_ancestor_symlink_escape(tmp_path: Path) -> None:
    dataset, plan = _dataset_and_plan(tmp_path)
    key = EvaluationIndexReuseKey.from_plan(plan, dataset)
    online = tmp_path / "online"
    outside = tmp_path / "outside"
    online.mkdir(parents=True)
    outside.mkdir()
    try:
        (online / "evaluations").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(EvaluationEnvironmentError, match="evaluation_suite_workspace_unsafe"):
        EvaluationSuiteWorkspaceManager(online).workspace_for("suite-001", key)
