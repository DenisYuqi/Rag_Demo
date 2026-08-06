"""Shared retrieval test builders with a globally unique module name."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from rag_mvp.domain.ingestion import (
    Chunk,
    ChunkLocator,
    Document,
    DocumentKind,
    DocumentVersion,
    ExtractionMethod,
)
from rag_mvp.domain.retrieval import RetrievalCandidate
from rag_mvp.ingestion.embedding import EmbeddingStage
from rag_mvp.ingestion.indexing import RevisionStager
from rag_mvp.providers.fakes import DeterministicEmbeddingProvider
from rag_mvp.providers.models import Deadline, ProviderCallContext
from rag_mvp.retrieval.binding import BoundRetrievalSnapshot, BoundRetrievalSnapshotFactory
from rag_mvp.storage.database import Database
from rag_mvp.storage.embedding_cache import EmbeddingCache
from rag_mvp.storage.layout import DataLayout
from rag_mvp.storage.repositories import KnowledgeRepositories


def candidate(
    chunk_id: str,
    *,
    dense_rank: int | None = None,
    bm25_rank: int | None = None,
    page: int = 1,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        source_id="source-1",
        display_title="Policy",
        document_version=1,
        locator=ChunkLocator(pages=(page,)),
        text=f"Evidence for {chunk_id}",
        dense_rank=dense_rank,
        dense_score=1.0 / dense_rank if dense_rank else None,
        bm25_rank=bm25_rank,
        bm25_score=1.0 / bm25_rank if bm25_rank else None,
    )


def indexed_chunk(
    chunk_id: str,
    text: str,
    *,
    source_id: str = "source-text",
    document_version: int = 1,
    ordinal: int = 0,
    locator: ChunkLocator | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_id=source_id,
        document_version=document_version,
        ordinal=ordinal,
        text=text,
        content_digest=hashlib.sha256(text.encode()).hexdigest(),
        locator=locator or ChunkLocator(char_start=ordinal * 100, char_end=ordinal * 100 + 50),
    )


async def build_bound_snapshot(
    tmp_path: Path,
    *,
    chunks: tuple[Chunk, ...] | None = None,
    titles: dict[str, str] | None = None,
    source_kinds: dict[str, DocumentKind] | None = None,
    revision_id: str = "revision-retrieval",
) -> tuple[BoundRetrievalSnapshot, dict[str, DocumentKind]]:
    resolved_chunks = chunks or (
        indexed_chunk("chunk-leave", "annual leave policy alpha", ordinal=0),
        indexed_chunk("chunk-security", "network security policy beta", ordinal=1),
    )
    resolved_titles = titles or {"source-text": "Employee Handbook"}
    resolved_kinds = source_kinds or {"source-text": DocumentKind.TEXT}
    active_sources = {chunk.source_id: chunk.document_version for chunk in resolved_chunks}
    assert set(active_sources) == set(resolved_titles) == set(resolved_kinds)

    layout = DataLayout.from_root(tmp_path / "data")
    layout.initialize()
    database = Database(layout.metadata_db)
    database.initialize()
    repositories = KnowledgeRepositories.from_database(database)
    for source_id, version in active_sources.items():
        kind = resolved_kinds[source_id]
        extension = {
            DocumentKind.PDF: "pdf",
            DocumentKind.MARKDOWN: "md",
            DocumentKind.TEXT: "txt",
        }[kind]
        media_type = {
            DocumentKind.PDF: "application/pdf",
            DocumentKind.MARKDOWN: "text/markdown",
            DocumentKind.TEXT: "text/plain",
        }[kind]
        repositories.documents.create(
            Document(
                source_id=source_id,
                source_key=f"key-{source_id}",
                display_title=resolved_titles[source_id],
                media_type=media_type,
                kind=kind,
            )
        )
        repositories.documents.add_version(
            DocumentVersion(
                source_id=source_id,
                version=version,
                content_digest=hashlib.sha256(f"document:{source_id}".encode()).hexdigest(),
                derivation_config_digest="derivation-v1",
                original_filename=f"{source_id}.{extension}",
                media_type=media_type,
                size_bytes=100,
                source_artifact_path=f"sources/{source_id}/{version}/source.{extension}",
                canonical_artifact_path=f"canonical/{source_id}/{version}/document.json",
                extraction_method=(
                    ExtractionMethod.NATIVE if kind is DocumentKind.PDF else ExtractionMethod.TEXT
                ),
            )
        )
        repositories.documents.set_active_version(source_id, version)
    context = ProviderCallContext(
        request_id="request-index",
        operation_id="index",
        deadline=Deadline.after(30),
    )
    with EmbeddingCache(layout.directory("caches") / "embeddings.sqlite3") as cache:
        staged = await RevisionStager(
            layout,
            EmbeddingStage(DeterministicEmbeddingProvider(), cache),
        ).stage(
            revision_id,
            resolved_chunks,
            resolved_titles,
            active_sources,
            context,
        )
    repositories.index_revisions.create(staged)
    repositories.index_revisions.publish(
        revision_id,
        published_at=datetime.now(UTC),
        expected_active_revision_id=None,
    )
    snapshot = BoundRetrievalSnapshotFactory(
        layout,
        repositories.index_revisions,
    ).bind()
    return snapshot, resolved_kinds
