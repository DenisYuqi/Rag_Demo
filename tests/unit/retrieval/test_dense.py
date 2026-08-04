from __future__ import annotations

from pathlib import Path

import pytest

from rag_mvp.domain.ingestion import Chunk, ChunkLocator, EmbeddingSpaceIdentity
from rag_mvp.retrieval.dense import DenseIndexError, PersistentChromaIndex


def _identity(model: str = "embedding-model") -> EmbeddingSpaceIdentity:
    return EmbeddingSpaceIdentity(
        provider_alias="test",
        model=model,
        dimension=3,
        normalization="none",
        adapter_version="v1",
    )


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_id="source-1",
        document_version=1,
        ordinal=0 if chunk_id == "chunk-a" else 1,
        text=text,
        content_digest="digest-" + chunk_id,
        locator=ChunkLocator(pages=(1,)),
    )


async def test_chroma_search_is_persistent_and_deterministic(tmp_path: Path) -> None:
    index = PersistentChromaIndex(
        tmp_path / "chroma",
        collection_name="revision_1",
        identity=_identity(),
    )
    index.add(
        (_chunk("chunk-a", "annual leave"), _chunk("chunk-b", "network security")),
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        {"source-1": "Handbook"},
    )

    reopened = PersistentChromaIndex(
        tmp_path / "chroma",
        collection_name="revision_1",
        identity=_identity(),
    )
    results = await reopened.search([1.0, 0.0, 0.0], query_identity=_identity(), limit=2)

    assert reopened.chunk_ids == {"chunk-a", "chunk-b"}
    assert [result.chunk_id for result in results] == ["chunk-a", "chunk-b"]
    assert [result.dense_rank for result in results] == [1, 2]


async def test_incompatible_query_identity_fails_before_query(tmp_path: Path) -> None:
    index = PersistentChromaIndex(
        tmp_path / "chroma",
        collection_name="revision_1",
        identity=_identity(),
    )

    with pytest.raises(DenseIndexError, match="embedding_identity_mismatch"):
        await index.search([1.0, 0.0, 0.0], query_identity=_identity("other-model"), limit=1)
