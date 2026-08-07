from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from rag_mvp.domain.ingestion import (
    IndexRevisionStatus,
    IngestionJob,
    IngestionJobStatus,
    IngestionStage,
)
from rag_mvp.ingestion.chunking import ChunkingConfig
from rag_mvp.ingestion.indexing import RevisionPublisher
from rag_mvp.ingestion.service import IngestionRecoveryError, IngestionService
from rag_mvp.providers.fakes import DeterministicEmbeddingProvider
from rag_mvp.providers.models import EmbeddingSpaceIdentity, NormalizationPolicy
from rag_mvp.retrieval.bm25 import PersistentBm25Index
from rag_mvp.retrieval.dense import PersistentChromaIndex
from rag_mvp.storage.layout import DataLayout


class NeverOcr:
    version = "never-ocr-v1"

    def recognize(self, png_bytes: bytes, *, languages: str) -> str:
        del png_bytes, languages
        raise AssertionError("text ingestion must not invoke OCR")


class InjectedFailure(RuntimeError):
    pass


def _service(
    root: Path,
    provider: DeterministicEmbeddingProvider | None = None,
    *,
    chunking: ChunkingConfig | None = None,
) -> tuple[IngestionService, DeterministicEmbeddingProvider]:
    resolved = provider or DeterministicEmbeddingProvider()
    return (
        IngestionService.create(
            root,
            resolved,
            ocr=NeverOcr(),
            chunking_config=chunking or ChunkingConfig(target_tokens=8, overlap_tokens=2),
        ),
        resolved,
    )


def _submit(service: IngestionService, content: str, *, source_key: str = "policy") -> IngestionJob:
    return service.submit_upload(
        "policy.txt",
        content.encode(),
        source_key=source_key,
        declared_media_type="text/plain",
        display_title="Policy",
    )


def _active(service: IngestionService):
    active = service.repositories.index_revisions.get_active()
    assert active is not None
    return active


async def _lexical_terms(root: Path, revision_id: str, query: str) -> tuple[str, ...]:
    layout = DataLayout.from_root(root)
    index = PersistentBm25Index.load(
        layout.lexical_index_path(revision_id),
        expected_revision_id=revision_id,
    )
    return tuple(item.text for item in await index.search(query, 20))


@pytest.mark.integration
async def test_first_upload_persists_stages_status_and_restart_visibility(tmp_path: Path) -> None:
    root = tmp_path / "data"
    service, provider = _service(root)
    submitted = _submit(service, "Leave policy ALPHA-101 grants twelve days.")

    completed = await service.run(submitted.job_id)
    active = _active(service)
    document = service.repositories.documents.get_by_source_key("policy")

    assert completed.status is IngestionJobStatus.SUCCEEDED
    assert completed.stage is IngestionStage.COMPLETE
    assert completed.source_id is not None
    assert completed.document_version == 1
    assert completed.chunk_count > 0
    assert completed.ocr_page_count == 0
    assert completed.active_index_revision == active.revision_id
    assert set(completed.stage_timings_ms) == {
        "validating",
        "extracting",
        "normalizing",
        "chunking",
        "embedding",
        "indexing",
        "publishing",
    }
    assert document is not None and document.active_version == 1
    assert provider.call_count > 0
    assert await _lexical_terms(root, active.revision_id, "ALPHA-101")

    service.close()
    reopened, _ = _service(root)
    report = await reopened.recover_startup()

    assert report.active_revision_id == active.revision_id
    assert report.replayed_job_count == 0
    assert reopened.get_job(submitted.job_id) == completed
    assert reopened.repositories.documents.get(document.source_id) == document
    assert await _lexical_terms(root, active.revision_id, "ALPHA-101")
    reopened.close()


@pytest.mark.integration
async def test_exact_duplicate_creates_no_embedding_version_or_revision(tmp_path: Path) -> None:
    root = tmp_path / "data"
    service, provider = _service(root)
    content = "Stable policy DUPLICATE-202 remains unchanged."
    first = await service.run(_submit(service, content).job_id)
    active_before = _active(service)
    calls_before = provider.call_count
    revisions_before = service.repositories.index_revisions.list()

    duplicate = await service.run(_submit(service, content).job_id)

    assert duplicate.status is IngestionJobStatus.SUCCEEDED
    assert duplicate.warnings == ("duplicate",)
    assert duplicate.document_version == 1
    assert duplicate.active_index_revision == active_before.revision_id
    assert provider.call_count == calls_before
    assert service.repositories.index_revisions.list() == revisions_before
    assert first.source_id is not None
    assert len(service.repositories.documents.list_versions(first.source_id)) == 1
    service.close()


@pytest.mark.integration
async def test_changed_upload_creates_v2_and_publishes_complete_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "data"
    service, _ = _service(root)
    first = await service.run(_submit(service, "Original code OLD-303 is active.").job_id)
    second = await service.run(_submit(service, "Replacement code NEW-404 is active.").job_id)
    active = _active(service)

    assert second.source_id == first.source_id
    assert second.document_version == 2
    assert active.active_sources == {first.source_id: 2}
    assert second.active_index_revision == active.revision_id
    assert len(service.repositories.documents.list_versions(first.source_id or "")) == 2
    assert await _lexical_terms(root, active.revision_id, "NEW-404")
    assert await _lexical_terms(root, active.revision_id, "OLD-303") == ()
    service.close()


@pytest.mark.integration
async def test_failed_update_leaves_v1_and_old_revision_queryable(tmp_path: Path) -> None:
    root = tmp_path / "data"
    service, _ = _service(root)
    first = await service.run(_submit(service, "Baseline SAFE-505 remains available.").job_id)
    old_revision = _active(service)

    def fail_inside_transaction(phase: str) -> None:
        if phase == "inside_transaction":
            raise InjectedFailure("private document content must not be persisted")

    service._publisher = RevisionPublisher(
        DataLayout.from_root(root),
        service.repositories.index_revisions,
        failure_hook=fail_inside_transaction,
    )
    failed = await service.run(_submit(service, "Candidate BROKEN-606 must not publish.").job_id)

    document = service.repositories.documents.get(first.source_id or "")
    assert failed.status is IngestionJobStatus.FAILED
    assert failed.safe_error_code == "ingestion_internal_error"
    assert failed.failed_stage is IngestionStage.PUBLISHING
    assert failed.document_version == 2
    assert document is not None and document.active_version == 1
    assert _active(service).revision_id == old_revision.revision_id
    assert await _lexical_terms(root, old_revision.revision_id, "SAFE-505")
    assert await _lexical_terms(root, old_revision.revision_id, "BROKEN-606") == ()
    failed_revision = service.repositories.index_revisions.list()[-1]
    assert failed_revision.status is IndexRevisionStatus.FAILED
    service.close()


@pytest.mark.integration
async def test_concurrent_same_source_updates_publish_in_submission_order(tmp_path: Path) -> None:
    root = tmp_path / "data"
    service, _ = _service(root)
    first = await service.run(_submit(service, "Version one ORDER-1.").job_id)
    second = _submit(service, "Version two ORDER-2.")
    third = _submit(service, "Version three ORDER-3.")

    third_result, second_result = await asyncio.gather(
        service.run(third.job_id),
        service.run(second.job_id),
    )

    assert second_result.document_version == 2
    assert third_result.document_version == 3
    assert first.source_id is not None
    assert [
        version.version for version in service.repositories.documents.list_versions(first.source_id)
    ] == [1, 2, 3]
    assert _active(service).active_sources == {first.source_id: 3}
    revision_ids = [
        revision.revision_id for revision in service.repositories.index_revisions.list()
    ]
    assert revision_ids == [
        first.active_index_revision,
        second_result.active_index_revision,
        third_result.active_index_revision,
    ]
    service.close()


@pytest.mark.integration
async def test_reindex_uses_retained_artifacts_and_same_space_cache(tmp_path: Path) -> None:
    root = tmp_path / "data"
    service, provider = _service(root)
    uploaded = await service.run(_submit(service, "Retained REINDEX-707 source artifact.").job_id)
    calls_before = provider.call_count
    versions_before = service.repositories.documents.list_versions(uploaded.source_id or "")
    revision_before = _active(service)

    reindexed = await service.run(service.submit_reindex().job_id)

    assert reindexed.status is IngestionJobStatus.SUCCEEDED
    assert reindexed.active_index_revision != revision_before.revision_id
    assert provider.call_count == calls_before
    assert service.repositories.documents.list_versions(uploaded.source_id or "") == versions_before
    assert await _lexical_terms(root, reindexed.active_index_revision or "", "REINDEX-707")
    active = _active(service)
    parent_count, parent_digest, _ = service.repositories.parent_chunks.inventory(
        active.revision_id
    )
    assert parent_count == active.parent_chunk_count > 0
    assert parent_digest == active.parent_chunk_set_digest
    service.close()


@pytest.mark.integration
async def test_reindex_embedding_identity_change_reuses_version_but_misses_cache(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    original, _ = _service(root)
    uploaded = await original.run(
        _submit(original, "Embedding identity MODEL-CHANGE-717 source.").job_id
    )
    original.close()

    changed_provider = DeterministicEmbeddingProvider(
        EmbeddingSpaceIdentity(
            provider="deterministic-fake",
            model="hash-vector-v2",
            dimension=16,
            normalization=NormalizationPolicy.L2,
            adapter_version="fake-v1",
        )
    )
    changed, _ = _service(root, changed_provider)
    reindexed = await changed.run(changed.submit_reindex().job_id)

    assert reindexed.status is IngestionJobStatus.SUCCEEDED
    assert changed_provider.call_count > 0
    assert uploaded.source_id is not None
    assert len(changed.repositories.documents.list_versions(uploaded.source_id)) == 1
    assert _active(changed).embedding_space.model == "hash-vector-v2"
    changed.close()


@pytest.mark.integration
async def test_reindex_derivation_change_registers_new_inactive_version_then_publishes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    original, _ = _service(
        root,
        chunking=ChunkingConfig(target_tokens=8, overlap_tokens=2),
    )
    uploaded = await original.run(
        _submit(original, "Derivation config CHUNK-CHANGE-727 source text.").job_id
    )
    original.close()

    changed, _ = _service(
        root,
        chunking=ChunkingConfig(target_tokens=6, overlap_tokens=1),
    )
    reindexed = await changed.run(changed.submit_reindex().job_id)

    assert uploaded.source_id is not None
    assert reindexed.status is IngestionJobStatus.SUCCEEDED
    assert reindexed.source_id == uploaded.source_id
    assert reindexed.document_version == 2
    assert _active(changed).active_sources == {uploaded.source_id: 2}
    changed_active = _active(changed)
    assert changed.repositories.parent_chunks.inventory(changed_active.revision_id)[:2] == (
        changed_active.parent_chunk_count,
        changed_active.parent_chunk_set_digest,
    )
    assert [
        version.version
        for version in changed.repositories.documents.list_versions(uploaded.source_id)
    ] == [1, 2]
    changed.close()


@pytest.mark.integration
async def test_successful_failed_and_last_source_delete(tmp_path: Path) -> None:
    root = tmp_path / "data"
    service, _ = _service(root)
    first = await service.run(_submit(service, "First DELETE-A source.", source_key="first").job_id)
    second = await service.run(
        _submit(service, "Second DELETE-B source.", source_key="second").job_id
    )
    assert first.source_id is not None and second.source_id is not None
    revisions_with_first = tuple(
        revision
        for revision in service.repositories.index_revisions.list()
        if first.source_id in revision.active_sources
    )

    deleted = await service.run(service.submit_delete(first.source_id).job_id)
    assert deleted.status is IngestionJobStatus.SUCCEEDED
    assert _active(service).active_sources == {second.source_id: 1}
    assert await _lexical_terms(root, deleted.active_index_revision or "", "DELETE-A") == ()
    assert service.repositories.documents.get(first.source_id) is None
    assert service.repositories.documents.list_versions(first.source_id) == []
    layout = DataLayout.from_root(root)
    retired_with_first = [
        revision
        for revision in service.repositories.index_revisions.list()
        if first.source_id in revision.active_sources
    ]
    assert retired_with_first == []
    assert all(
        not layout.index_revision_path(revision.revision_id).exists()
        for revision in revisions_with_first
    )
    assert not layout.directory("sources").joinpath(first.source_id).exists()
    assert not layout.directory("canonical").joinpath(first.source_id).exists()

    old_revision = _active(service)

    def fail_pretransaction(phase: str) -> None:
        if phase == "pretransaction":
            raise InjectedFailure("do not expose this")

    service._publisher = RevisionPublisher(
        DataLayout.from_root(root),
        service.repositories.index_revisions,
        failure_hook=fail_pretransaction,
    )
    failed = await service.run(service.submit_delete(second.source_id).job_id)
    assert failed.status is IngestionJobStatus.FAILED
    assert _active(service).revision_id == old_revision.revision_id
    retained = service.repositories.documents.get(second.source_id)
    assert retained is not None and retained.active_version == 1 and retained.deleted_at is None

    service._publisher = RevisionPublisher(
        DataLayout.from_root(root),
        service.repositories.index_revisions,
    )
    last = await service.run(service.submit_delete(second.source_id).job_id)
    empty = _active(service)
    assert last.status is IngestionJobStatus.SUCCEEDED
    assert empty.active_sources == {}
    assert empty.chunk_count == 0
    with PersistentChromaIndex.open_existing(
        layout.dense_index_path(empty.revision_id),
        revision_id=empty.revision_id,
        identity=empty.embedding_space,
    ) as dense:
        assert dense.chunk_ids == frozenset()
    lexical = PersistentBm25Index.load(
        layout.lexical_index_path(empty.revision_id),
        expected_revision_id=empty.revision_id,
    )
    assert lexical.chunk_ids == frozenset()
    service.close()


@pytest.mark.integration
async def test_startup_physically_purges_legacy_soft_deleted_source(tmp_path: Path) -> None:
    root = tmp_path / "data"
    service, _ = _service(root)
    uploaded = await service.run(_submit(service, "Legacy soft delete PURGE-909 source.").job_id)
    assert uploaded.source_id is not None
    source_id = uploaded.source_id
    legacy_revisions = tuple(service.repositories.index_revisions.list())

    async def interrupt_purge(*_args: object) -> IngestionJob | None:
        raise InjectedFailure("simulate legacy soft-delete behavior")

    service._purge_deleted_source = interrupt_purge  # type: ignore[method-assign]
    failed = await service.run(service.submit_delete(source_id).job_id)
    soft_deleted = service.repositories.documents.get(source_id)
    assert failed.status is IngestionJobStatus.FAILED
    assert soft_deleted is not None and soft_deleted.deleted_at is not None
    service.close()

    restarted, _ = _service(root)
    await restarted.recover_startup()

    layout = DataLayout.from_root(root)
    assert restarted.repositories.documents.get(source_id) is None
    assert restarted.repositories.documents.list_versions(source_id) == []
    assert all(
        not layout.index_revision_path(revision.revision_id).exists()
        for revision in legacy_revisions
    )
    assert not layout.directory("sources").joinpath(source_id).exists()
    assert not layout.directory("canonical").joinpath(source_id).exists()
    restarted.close()


@pytest.mark.integration
async def test_fresh_service_sees_same_active_documents_and_indexes(tmp_path: Path) -> None:
    root = tmp_path / "data"
    first_service, _ = _service(root)
    completed = await first_service.run(
        _submit(first_service, "Restart PERSIST-808 remains indexed.").job_id
    )
    active = _active(first_service)
    source_id = completed.source_id
    first_service.close()

    restarted, _ = _service(root)
    assert _active(restarted) == active
    document = restarted.repositories.documents.get(source_id or "")
    assert document is not None and document.active_version == 1
    assert await _lexical_terms(root, active.revision_id, "PERSIST-808")
    restarted.close()


@pytest.mark.integration
async def test_interrupted_registered_version_is_replayed_without_v3(tmp_path: Path) -> None:
    root = tmp_path / "data"
    service, _ = _service(root)
    first = await service.run(_submit(service, "Recovery version one RECOVER-1.").job_id)
    interrupted = _submit(service, "Recovery version two RECOVER-2.")
    command = service._artifacts.load_command(interrupted.job_id)
    job = service.repositories.ingestion_jobs.get(interrupted.job_id)
    assert job is not None
    prepared, _, duplicate = await service._prepare_upload(job, command)
    assert duplicate is False and prepared.document_version == 2
    assert prepared.status is IngestionJobStatus.PROCESSING
    service.close()

    restarted, _ = _service(root)
    report = await restarted.recover_startup()
    recovered = restarted.get_job(interrupted.job_id)

    assert report.requeued_job_count == 1
    assert report.replayed_job_count == 1
    assert recovered is not None and recovered.status is IngestionJobStatus.SUCCEEDED
    assert recovered.document_version == 2
    assert first.source_id is not None
    assert [
        version.version
        for version in restarted.repositories.documents.list_versions(first.source_id)
    ] == [1, 2]
    assert _active(restarted).active_sources == {first.source_id: 2}
    restarted.close()


@pytest.mark.integration
async def test_recovery_abandons_staged_revision_instead_of_promoting_it(tmp_path: Path) -> None:
    root = tmp_path / "data"
    service, _ = _service(root)
    completed = await service.run(_submit(service, "Active STABLE-909 revision.").job_id)
    active = _active(service)
    staged = active.model_copy(
        update={
            "revision_id": "rev_abandoned",
            "status": IndexRevisionStatus.STAGED,
            "published_at": None,
            "dense_index_path": active.dense_index_path.replace(
                active.revision_id, "rev_abandoned"
            ),
            "lexical_index_path": active.lexical_index_path.replace(
                active.revision_id, "rev_abandoned"
            ),
            "ingestion_job_id": None,
        }
    )
    service.repositories.index_revisions.create(staged)
    service.close()

    restarted, _ = _service(root)
    report = await restarted.recover_startup()
    abandoned = restarted.repositories.index_revisions.get("rev_abandoned")

    assert report.abandoned_revision_count == 1
    assert abandoned is not None and abandoned.status is IndexRevisionStatus.FAILED
    assert restarted.repositories.parent_chunks.list_for_revision("rev_abandoned") == ()
    assert _active(restarted).revision_id == active.revision_id
    assert restarted.get_job(completed.job_id) is not None
    restarted.close()


@pytest.mark.integration
async def test_corrupt_active_revision_fails_closed_before_recovery_writes(tmp_path: Path) -> None:
    root = tmp_path / "data"
    service, _ = _service(root)
    completed = await service.run(_submit(service, "Corrupt check CLOSED-010.").job_id)
    active = _active(service)
    queued = _submit(service, "Pending work MUST-NOT-RUN.")
    layout = DataLayout.from_root(root)
    layout.lexical_index_path(active.revision_id).write_text("{corrupt", encoding="utf-8")
    service.close()

    restarted, provider = _service(root)
    with pytest.raises(IngestionRecoveryError, match="active_revision_invalid"):
        await restarted.recover_startup()

    assert provider.call_count == 0
    assert restarted.get_job(queued.job_id) == queued
    assert restarted.get_job(completed.job_id) == completed
    assert _active(restarted).revision_id == active.revision_id
    restarted.close()


@pytest.mark.integration
async def test_terminal_details_and_safe_failure_survive_repository_restart(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    service, _ = _service(root)
    succeeded = await service.run(_submit(service, "Terminal detail FINAL-111.").job_id)
    missing = _submit(service, "Command will be missing.")
    corrupt = _submit(service, "Command will be corrupt.", source_key="corrupt-command")
    service._artifacts.delete_command(missing.job_id)
    layout = DataLayout.from_root(root)
    layout.job_command_path(corrupt.job_id).write_text("{invalid", encoding="utf-8")
    service.close()

    restarted, _ = _service(root)
    report = await restarted.recover_startup()
    failed = restarted.get_job(missing.job_id)
    corrupt_failed = restarted.get_job(corrupt.job_id)

    assert report.failed_job_count == 2
    assert restarted.get_job(succeeded.job_id) == succeeded
    assert failed is not None
    assert failed.status is IngestionJobStatus.FAILED
    assert failed.safe_error_code == "command_missing"
    assert failed.failed_stage is IngestionStage.QUEUED
    assert corrupt_failed is not None
    assert corrupt_failed.safe_error_code == "command_corrupt"
    restarted.close()


@pytest.mark.integration
def test_rejected_upload_creates_no_job_or_command_artifact(tmp_path: Path) -> None:
    root = tmp_path / "data"
    service, _ = _service(root)

    with pytest.raises(ValueError, match="empty_document"):
        _submit(service, "")

    assert service.repositories.ingestion_jobs.list_nonterminal() == []
    layout = DataLayout.from_root(root)
    assert list(layout.directory("jobs").iterdir()) == []
    with sqlite3.connect(layout.metadata_db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM ingestion_jobs").fetchone() == (0,)
    service.close()


@pytest.mark.integration
def test_repository_submission_failure_cleans_durable_command(tmp_path: Path) -> None:
    root = tmp_path / "data"
    service, _ = _service(root)

    def fail_create(job: IngestionJob) -> None:
        del job
        raise RuntimeError("database details must not persist in the command store")

    service.repositories.ingestion_jobs.create = fail_create  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="database details"):
        _submit(service, "Valid content that cannot commit.")

    layout = DataLayout.from_root(root)
    assert list(layout.directory("jobs").iterdir()) == []
    service.close()
