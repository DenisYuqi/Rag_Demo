from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag_mvp.domain.ingestion import (
    Document,
    DocumentKind,
    DocumentVersion,
    EmbeddingSpaceIdentity,
    ExtractionMethod,
    IndexRevision,
    IndexRevisionStatus,
    IngestionJob,
    IngestionJobStatus,
    IngestionStage,
)
from rag_mvp.storage.database import Database
from rag_mvp.storage.repositories import KnowledgeRepositories, RepositoryConflict


def _document() -> Document:
    return Document(
        source_id="source-1",
        source_key="employee-handbook",
        display_title="Employee Handbook",
        media_type="application/pdf",
        kind=DocumentKind.PDF,
    )


def _version(version: int = 1, *, digest: str = "content-123") -> DocumentVersion:
    return DocumentVersion(
        source_id="source-1",
        version=version,
        content_digest=digest,
        derivation_config_digest="derive-123",
        original_filename="handbook.pdf",
        media_type="application/pdf",
        size_bytes=100,
        source_artifact_path=f"sources/source-1/{version}/original.pdf",
        canonical_artifact_path=f"canonical/source-1/{version}/content.txt",
        extraction_method=ExtractionMethod.MIXED,
    )


def _revision(revision_id: str, version: int) -> IndexRevision:
    return IndexRevision(
        revision_id=revision_id,
        active_sources={"source-1": version},
        chunk_set_digest=f"chunks-{revision_id}",
        embedding_space=EmbeddingSpaceIdentity(
            provider_alias="primary",
            model="embed-v1",
            dimension=3,
            normalization="l2",
            adapter_version="v1",
        ),
        extraction_version="v1",
        chunking_version="v1",
        tokenizer_version="v1",
        dense_index_path=f"indexes/{revision_id}/chroma",
        lexical_index_path=f"indexes/{revision_id}/bm25",
    )


@pytest.fixture
def database(tmp_path: Path) -> Database:
    value = Database(tmp_path / "metadata.sqlite3")
    value.initialize()
    return value


def test_document_versions_crud_and_soft_delete(database: Database) -> None:
    repositories = KnowledgeRepositories.from_database(database)
    repositories.documents.create(_document())
    version = _version()
    repositories.documents.add_version(version)

    updated = repositories.documents.set_active_version("source-1", 1)

    assert updated.active_version == 1
    assert repositories.documents.get_by_source_key("employee-handbook") == updated
    assert repositories.documents.get_version("source-1", 1) == version
    assert repositories.documents.list_versions("source-1") == [version]
    assert repositories.documents.list() == [updated]

    deleted = repositories.documents.mark_deleted("source-1")
    assert deleted.deleted_at is not None
    assert deleted.active_version is None
    assert repositories.documents.list() == []
    assert repositories.documents.list(include_deleted=True) == [deleted]


def test_duplicate_document_or_content_derivation_is_rejected(database: Database) -> None:
    repositories = KnowledgeRepositories.from_database(database)
    repositories.documents.create(_document())
    repositories.documents.add_version(_version())

    with pytest.raises(RepositoryConflict):
        repositories.documents.create(
            Document(
                source_id="source-2",
                source_key="employee-handbook",
                display_title="Duplicate key",
                media_type="text/plain",
                kind=DocumentKind.TEXT,
            )
        )

    with pytest.raises(RepositoryConflict):
        repositories.documents.add_version(_version(version=2))


def test_multi_repository_transaction_rolls_back(database: Database) -> None:
    repositories = KnowledgeRepositories.from_database(database)

    with (
        pytest.raises(RuntimeError, match="abort publication"),
        database.transaction() as connection,
    ):
        repositories.documents.create(_document(), connection=connection)
        repositories.documents.add_version(_version(), connection=connection)
        repositories.index_revisions.create(_revision("revision-1", 1), connection=connection)
        raise RuntimeError("abort publication")

    assert repositories.documents.get("source-1") is None
    assert repositories.index_revisions.get("revision-1") is None


def test_index_publication_atomically_supersedes_prior_revision(database: Database) -> None:
    repositories = KnowledgeRepositories.from_database(database)
    repositories.documents.create(_document())
    repositories.documents.add_version(_version())
    repositories.documents.set_active_version("source-1", 1)
    repositories.index_revisions.create(_revision("revision-1", 1))
    repositories.index_revisions.create(_revision("revision-2", 1))

    first = repositories.index_revisions.publish(
        "revision-1", published_at=datetime(2026, 8, 4, 1, tzinfo=UTC)
    )
    second = repositories.index_revisions.publish(
        "revision-2", published_at=datetime(2026, 8, 4, 2, tzinfo=UTC)
    )

    assert first.status is IndexRevisionStatus.ACTIVE
    assert second.status is IndexRevisionStatus.ACTIVE
    assert repositories.index_revisions.get_active() == second
    previous = repositories.index_revisions.get("revision-1")
    assert previous is not None
    assert previous.status is IndexRevisionStatus.SUPERSEDED


def test_terminal_ingestion_job_survives_repository_reopen(database: Database) -> None:
    repositories = KnowledgeRepositories.from_database(database)
    queued = IngestionJob(job_id="job-1", source_key="employee-handbook")
    repositories.ingestion_jobs.create(queued)
    processing = IngestionJob.model_validate(
        {
            **queued.model_dump(),
            "status": IngestionJobStatus.PROCESSING,
            "stage": IngestionStage.EXTRACTING,
        }
    )
    repositories.ingestion_jobs.update(processing)
    failed = IngestionJob.model_validate(
        {
            **processing.model_dump(),
            "status": IngestionJobStatus.FAILED,
            "stage": IngestionStage.FAILED,
            "safe_error_code": "pdf-corrupt",
        }
    )
    repositories.ingestion_jobs.update(failed)

    reopened = KnowledgeRepositories.from_database(Database(database.path))

    assert reopened.ingestion_jobs.get("job-1") == failed
    with pytest.raises(RepositoryConflict):
        reopened.ingestion_jobs.update(processing)
