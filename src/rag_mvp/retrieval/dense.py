"""Revision-specific Chroma dense-vector storage with identity validation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb

from rag_mvp.domain.ingestion import Chunk, ChunkLocator, EmbeddingSpaceIdentity
from rag_mvp.domain.retrieval import RetrievalCandidate


class DenseIndexError(ValueError):
    pass


def _identity_metadata(identity: EmbeddingSpaceIdentity) -> dict[str, str | int]:
    return {
        "provider_alias": identity.provider_alias,
        "model": identity.model,
        "dimension": identity.dimension,
        "normalization": identity.normalization,
        "adapter_version": identity.adapter_version,
    }


class PersistentChromaIndex:
    def __init__(
        self,
        path: Path,
        *,
        collection_name: str,
        identity: EmbeddingSpaceIdentity,
    ) -> None:
        self.path = path
        self.identity = identity
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        existing_names = {collection.name for collection in self._client.list_collections()}
        if collection_name in existing_names:
            collection = self._client.get_collection(collection_name)
            metadata = collection.metadata or {}
            if any(
                metadata.get(key) != value
                for key, value in _identity_metadata(identity).items()
            ):
                raise DenseIndexError("embedding_identity_mismatch")
            self._collection = collection
        else:
            self._collection = self._client.create_collection(
                collection_name,
                metadata={**_identity_metadata(identity), "hnsw:space": "cosine"},
            )

    @property
    def chunk_ids(self) -> frozenset[str]:
        result = self._collection.get(include=[])
        return frozenset(str(item) for item in result["ids"])

    def add(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        titles: dict[str, str],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise DenseIndexError("embedding_count_mismatch")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise DenseIndexError("duplicate_chunk_id")
        vectors: list[list[float]] = []
        metadatas: list[dict[str, str | int]] = []
        for chunk, vector in zip(chunks, embeddings, strict=True):
            normalized = [float(value) for value in vector]
            if len(normalized) != self.identity.dimension or not all(
                math.isfinite(value) for value in normalized
            ):
                raise DenseIndexError("incompatible_embedding")
            vectors.append(normalized)
            metadatas.append(
                {
                    "source_id": chunk.source_id,
                    "display_title": titles[chunk.source_id],
                    "document_version": chunk.document_version,
                    "locator": chunk.locator.model_dump_json(),
                }
            )
        if chunks:
            collection: Any = self._collection
            collection.add(
                ids=[chunk.chunk_id for chunk in chunks],
                embeddings=vectors,
                documents=[chunk.text for chunk in chunks],
                metadatas=metadatas,
            )

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        query_identity: EmbeddingSpaceIdentity,
        limit: int,
    ) -> tuple[RetrievalCandidate, ...]:
        if query_identity != self.identity:
            raise DenseIndexError("embedding_identity_mismatch")
        vector = [float(value) for value in query_vector]
        if len(vector) != self.identity.dimension or not all(
            math.isfinite(value) for value in vector
        ):
            raise DenseIndexError("incompatible_embedding")
        if limit < 1:
            raise ValueError("limit must be positive")
        if not self.chunk_ids:
            return ()
        collection: Any = self._collection
        result: dict[str, Any] = collection.query(
            query_embeddings=[vector],
            n_results=min(limit, len(self.chunk_ids)),
            include=["documents", "metadatas", "distances"],
        )
        rows = list(
            zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        )
        rows.sort(key=lambda row: (float(row[3]), str(row[0])))
        return tuple(
            RetrievalCandidate(
                chunk_id=str(chunk_id),
                source_id=str(metadata["source_id"]),
                display_title=str(metadata["display_title"]),
                document_version=int(metadata["document_version"]),
                locator=ChunkLocator.model_validate_json(str(metadata["locator"])),
                text=str(document),
                dense_rank=rank,
                dense_score=1.0 - float(distance),
            )
            for rank, (chunk_id, document, metadata, distance) in enumerate(rows, start=1)
        )
