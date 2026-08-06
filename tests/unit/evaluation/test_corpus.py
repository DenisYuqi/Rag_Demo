from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from rag_mvp.domain.ingestion import ExtractionMethod, IndexRevisionStatus, IngestionJobStatus
from rag_mvp.evaluation.corpus import (
    EvaluationCorpusInstaller,
    EvaluationCorpusInstallError,
)
from rag_mvp.evaluation.dataset import (
    CorpusSnapshotFormat,
    load_dataset,
    materialize_production_documents,
)
from rag_mvp.ingestion.service import IngestionService
from rag_mvp.providers.fakes import DeterministicEmbeddingProvider
from rag_mvp.retrieval.bm25 import PersistentBm25Index
from rag_mvp.storage.artifacts import ArtifactStore
from rag_mvp.storage.layout import DataLayout


def _dataset_root() -> Path:
    return Path(__file__).resolve().parents[3] / "evaluations" / "datasets" / "mvp-v1"


@pytest.mark.asyncio
async def test_installer_publishes_reproducible_production_snapshot(tmp_path: Path) -> None:
    dataset = load_dataset(_dataset_root())
    provider = DeterministicEmbeddingProvider()
    service = IngestionService.create(tmp_path / "data", provider)
    try:
        installed = await EvaluationCorpusInstaller(service).install(dataset)

        assert installed.revision.status is IndexRevisionStatus.ACTIVE
        assert installed.revision.chunk_count == len(dataset.production_chunks)
        assert installed.corpus_hash == dataset.corpus.manifest.content_hash
        active_revision, documents = service.list_active_documents()
        assert active_revision == installed.revision.revision_id
        assert {document.source_id for document in documents} == set(
            dataset.corpus.manifest.active_sources
        )

        layout = DataLayout.from_root(service.data_root)
        artifacts = ArtifactStore(layout)
        ocr_metadata, source_bytes, expected_extracted = next(
            item
            for item in materialize_production_documents(dataset)
            if item[0].snapshot_format is CorpusSnapshotFormat.FROZEN_OCR_PAGE
        )
        ocr_version = service.repositories.documents.get_version(
            ocr_metadata.source_id,
            ocr_metadata.document_version,
        )
        assert ocr_version is not None
        assert artifacts.load_source(ocr_version) == source_bytes
        assert artifacts.load_canonical(ocr_version) == expected_extracted
        assert all(
            block.extraction_method is ExtractionMethod.OCR
            for block in artifacts.load_canonical(ocr_version).blocks
        )
        with fitz.open(stream=artifacts.load_source(ocr_version), filetype="pdf") as pdf:
            assert pdf.is_repaired is False
            assert all(not page.get_text("text").strip() for page in pdf)

        reindex = service.submit_reindex()
        completed = await service.run(reindex.job_id)
        assert completed.status is IngestionJobStatus.SUCCEEDED
        assert completed.chunk_count == len(dataset.production_chunks)
        reindexed = service.repositories.index_revisions.get_active()
        assert reindexed is not None
        lexical = PersistentBm25Index.load(
            layout.lexical_index_path(reindexed.revision_id),
            expected_revision_id=reindexed.revision_id,
        )
        assert lexical.chunk_ids == frozenset(chunk.chunk_id for chunk in dataset.production_chunks)
    finally:
        service.close()


@pytest.mark.asyncio
async def test_installer_rejects_a_nonempty_data_root(tmp_path: Path) -> None:
    dataset = load_dataset(_dataset_root())
    service = IngestionService.create(tmp_path / "data", DeterministicEmbeddingProvider())
    try:
        installer = EvaluationCorpusInstaller(service)
        await installer.install(dataset)
        with pytest.raises(EvaluationCorpusInstallError, match="evaluation_data_root_not_empty"):
            await installer.install(dataset)
    finally:
        service.close()


@pytest.mark.asyncio
async def test_installer_rejects_orphan_corpus_artifacts(tmp_path: Path) -> None:
    dataset = load_dataset(_dataset_root())
    service = IngestionService.create(tmp_path / "data", DeterministicEmbeddingProvider())
    try:
        orphan = service.data_root / "sources" / "orphan.bin"
        orphan.write_bytes(b"orphan")

        with pytest.raises(EvaluationCorpusInstallError, match="evaluation_data_root_not_empty"):
            await EvaluationCorpusInstaller(service).install(dataset)
        assert service.repositories.documents.list(include_deleted=True) == []
        assert service.repositories.index_revisions.list() == []
    finally:
        service.close()


@pytest.mark.asyncio
async def test_failed_publish_leaves_no_corpus_residue_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = load_dataset(_dataset_root())
    data_root = tmp_path / "data"
    service = IngestionService.create(data_root, DeterministicEmbeddingProvider())

    def fail_publish(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("controlled publish failure")

    monkeypatch.setattr(service._publisher, "publish", fail_publish)
    try:
        with pytest.raises(
            EvaluationCorpusInstallError,
            match="evaluation_corpus_install_failed",
        ):
            await EvaluationCorpusInstaller(service).install(dataset)
        _assert_no_corpus_state(service)
    finally:
        service.close()

    replacement = IngestionService.create(data_root, DeterministicEmbeddingProvider())
    try:
        installed = await EvaluationCorpusInstaller(replacement).install(dataset)
        assert installed.revision.status is IndexRevisionStatus.ACTIVE
    finally:
        replacement.close()


def _assert_no_corpus_state(service: IngestionService) -> None:
    assert service.repositories.documents.list(include_deleted=True) == []
    assert service.repositories.ingestion_jobs.list() == []
    assert service.repositories.index_revisions.list() == []
    assert service.repositories.index_revisions.get_active_revision_id() is None
    layout = DataLayout.from_root(service.data_root)
    for path in (
        layout.directory("sources"),
        layout.directory("canonical"),
        layout.directory("jobs"),
        layout.index_revisions,
    ):
        assert list(path.iterdir()) == []
