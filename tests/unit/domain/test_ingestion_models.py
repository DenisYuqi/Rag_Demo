from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rag_mvp.domain.ingestion import (
    Chunk,
    ChunkLocator,
    Document,
    DocumentKind,
    EmbeddingSpaceIdentity,
    IndexRevision,
    IndexRevisionStatus,
    IngestionJob,
    IngestionJobStatus,
    IngestionStage,
    ParentChunk,
)


def test_document_and_chunk_round_trip_json() -> None:
    document = Document(
        source_id="source-1",
        source_key="handbook",
        display_title="员工 Handbook",
        media_type="application/pdf",
        kind=DocumentKind.PDF,
        active_version=1,
    )
    chunk = Chunk(
        chunk_id="chunk-1",
        parent_chunk_id="parent-1",
        source_id=document.source_id,
        document_version=1,
        ordinal=0,
        text="休假 policy",
        content_digest="digest-123",
        locator=ChunkLocator(pages=(1, 2), section_path=("Benefits",)),
        token_count=3,
    )

    assert Document.model_validate_json(document.model_dump_json()) == document
    assert Chunk.model_validate_json(chunk.model_dump_json()) == chunk
    assert chunk.locator.pages == (1, 2)

    parent = ParentChunk(
        parent_chunk_id=chunk.parent_chunk_id,
        source_id=document.source_id,
        document_version=1,
        ordinal=0,
        text="完整的休假 policy 上下文",
        content_digest="parent-digest-123",
        locator=ChunkLocator(pages=(1, 2), section_path=("Benefits",)),
        token_count=8,
    )
    assert ParentChunk.model_validate_json(parent.model_dump_json()) == parent


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"pages": (0,)},
        {"char_start": 10},
        {"char_start": 10, "char_end": 10},
        {"pages": (2, 2)},
    ],
)
def test_chunk_locator_rejects_unstable_or_empty_locations(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ChunkLocator.model_validate(values)


def test_active_index_revision_requires_publication_timestamp() -> None:
    values = {
        "revision_id": "revision-1",
        "status": IndexRevisionStatus.ACTIVE,
        "active_sources": {"source-1": 1},
        "chunk_set_digest": "chunks-123",
        "embedding_space": EmbeddingSpaceIdentity(
            provider_alias="primary",
            model="embedding-v1",
            dimension=3,
            normalization="none",
            adapter_version="v1",
        ),
        "extraction_version": "v1",
        "chunking_version": "v1",
        "tokenizer_version": "v1",
        "dense_index_path": "indexes/revision-1/chroma",
        "lexical_index_path": "indexes/revision-1/bm25",
    }

    with pytest.raises(ValidationError):
        IndexRevision.model_validate(values)

    revision = IndexRevision.model_validate(
        {**values, "published_at": datetime(2026, 8, 4, tzinfo=UTC)}
    )
    assert revision.status is IndexRevisionStatus.ACTIVE


def test_old_index_revision_payload_receives_additive_index_defaults() -> None:
    revision = IndexRevision(
        revision_id="revision-old",
        active_sources={},
        chunk_set_digest="empty-digest",
        embedding_space=EmbeddingSpaceIdentity(
            provider_alias="primary",
            model="embedding-v1",
            dimension=3,
            normalization="none",
            adapter_version="v1",
        ),
        extraction_version="v1",
        chunking_version="v1",
        tokenizer_version="jieba-cjk-ngram-v1",
        dense_index_path="indexes/revisions/revision-old/chroma",
        lexical_index_path="indexes/revisions/revision-old/bm25.json",
    )

    assert revision.chunk_count == 0
    assert revision.dense_metric == "cosine"
    assert revision.lexical_algorithm_version == "bm25-okapi-v1"


def test_ingestion_terminal_states_are_validated() -> None:
    with pytest.raises(ValidationError):
        IngestionJob(
            job_id="job-1",
            source_key="source",
            status=IngestionJobStatus.FAILED,
            stage=IngestionStage.FAILED,
        )

    with pytest.raises(ValidationError):
        IngestionJob(
            job_id="job-1",
            source_key="source",
            status=IngestionJobStatus.SUCCEEDED,
            stage=IngestionStage.INDEXING,
        )

    job = IngestionJob(
        job_id="job-1",
        source_key="source",
        status=IngestionJobStatus.FAILED,
        stage=IngestionStage.FAILED,
        safe_error_code="pdf-corrupt",
    )
    assert IngestionJob.model_validate_json(job.model_dump_json()) == job
