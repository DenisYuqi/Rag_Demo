"""Immutable revision-specific Chroma storage with full inventory validation."""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Self, cast

import chromadb
from chromadb.api.collection_configuration import CreateCollectionConfiguration
from pydantic import ValidationError

from rag_mvp.domain.ingestion import Chunk, ChunkLocator, EmbeddingSpaceIdentity
from rag_mvp.domain.retrieval import RetrievalCandidate
from rag_mvp.providers.models import AttemptStatus, ModelAttempt
from rag_mvp.retrieval.snapshot import (
    RECORD_DIGEST_ALGORITHM,
    canonical_locator_json,
    chunk_record_digest,
    chunk_set_digest,
)

# Chroma shares one process-level System for every persistent path, but its
# create/increment and decrement/remove lifecycle transitions are not one
# atomic operation. Serializing client construction and close prevents a cold
# concurrent bind from observing a stopped or replaced shared System. Queries
# deliberately remain outside this lock.
_CHROMA_CLIENT_LIFECYCLE_LOCK: Final = threading.RLock()


def _create_persistent_client(path: Path) -> Any:
    with _CHROMA_CLIENT_LIFECYCLE_LOCK:
        return chromadb.PersistentClient(path=str(path))


def _close_persistent_client(client: Any) -> None:
    with _CHROMA_CLIENT_LIFECYCLE_LOCK:
        client.close()


class DenseIndexError(ValueError):
    """A safe dense index validation or lifecycle error."""

    def __init__(
        self,
        code: str,
        *,
        provider_attempts: tuple[ModelAttempt, ...] = (),
        unrecorded_provider_attempt_count: int = 0,
    ) -> None:
        self.code = code
        self.provider_attempts = tuple(provider_attempts)
        self.unrecorded_provider_attempt_count = unrecorded_provider_attempt_count
        super().__init__(code)

    @property
    def provider_attempt_count(self) -> int:
        return len(self.provider_attempts) + self.unrecorded_provider_attempt_count

    @property
    def provider_failed_attempt_count(self) -> int:
        return (
            sum(attempt.status is not AttemptStatus.SUCCEEDED for attempt in self.provider_attempts)
            + self.unrecorded_provider_attempt_count
        )

    @property
    def provider_unknown_usage_attempt_count(self) -> int:
        return (
            sum(attempt.usage.input_tokens is None for attempt in self.provider_attempts)
            + self.unrecorded_provider_attempt_count
        )


class PersistentChromaIndex:
    """One Chroma collection whose sealed contents can never be mutated."""

    COLLECTION_NAME = "rag_mvp_revision"
    SCHEMA_VERSION = "chroma-revision-v2"
    METRIC = "cosine"
    _PERSISTENCE_MARKER = "chroma.sqlite3"

    def __init__(
        self,
        path: Path,
        *,
        collection_name: str,
        identity: EmbeddingSpaceIdentity,
    ) -> None:
        """Compatibility create-or-open API; strict callers use the classmethods."""

        path = Path(path)
        if (path / self._PERSISTENCE_MARKER).is_file():
            try:
                self._initialize_open(
                    path,
                    collection_name=collection_name,
                    revision_id=collection_name,
                    identity=identity,
                    require_sealed=False,
                )
            except DenseIndexError as error:
                if error.code != "dense_collection_missing":
                    raise
                self._initialize_create(
                    path,
                    collection_name=collection_name,
                    revision_id=collection_name,
                    identity=identity,
                    require_new_path=False,
                )
        else:
            self._initialize_create(
                path,
                collection_name=collection_name,
                revision_id=collection_name,
                identity=identity,
                require_new_path=False,
            )

    @classmethod
    def create_new(
        cls,
        path: Path,
        *,
        revision_id: str,
        identity: EmbeddingSpaceIdentity,
    ) -> PersistentChromaIndex:
        index = cls.__new__(cls)
        index._initialize_create(
            Path(path),
            collection_name=cls.COLLECTION_NAME,
            revision_id=revision_id,
            identity=identity,
            require_new_path=True,
        )
        return index

    @classmethod
    def open_existing(
        cls,
        path: Path,
        *,
        revision_id: str,
        identity: EmbeddingSpaceIdentity,
    ) -> PersistentChromaIndex:
        path = Path(path)
        if not path.is_dir() or not (path / cls._PERSISTENCE_MARKER).is_file():
            raise DenseIndexError("dense_index_missing")
        index = cls.__new__(cls)
        index._initialize_open(
            path,
            collection_name=cls.COLLECTION_NAME,
            revision_id=revision_id,
            identity=identity,
            require_sealed=True,
        )
        return index

    def _initialize_create(
        self,
        path: Path,
        *,
        collection_name: str,
        revision_id: str,
        identity: EmbeddingSpaceIdentity,
        require_new_path: bool,
    ) -> None:
        if not revision_id:
            raise DenseIndexError("revision_id_invalid")
        if require_new_path and path.exists():
            raise DenseIndexError("dense_index_path_exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir(parents=False, exist_ok=not require_new_path)
        self.path = path
        self.identity = identity
        self.revision_id = revision_id
        self.collection_name = collection_name
        self._client: Any | None = None
        self._collection: Any | None = None
        self._sealed = False
        try:
            self._client = _create_persistent_client(path)
            existing_names = {collection.name for collection in self._client.list_collections()}
            if collection_name in existing_names:
                raise DenseIndexError("dense_collection_exists")
            configuration = cast(
                CreateCollectionConfiguration,
                {"hnsw": {"space": self.METRIC}},
            )
            self._collection = self._client.create_collection(
                collection_name,
                configuration=configuration,
                metadata=self._collection_metadata(
                    sealed=False,
                    chunk_count=0,
                    inventory_digest=chunk_set_digest({}),
                ),
                embedding_function=None,
            )
            self._validate_configuration()
        except BaseException:
            self.close()
            raise

    def _initialize_open(
        self,
        path: Path,
        *,
        collection_name: str,
        revision_id: str,
        identity: EmbeddingSpaceIdentity,
        require_sealed: bool,
    ) -> None:
        self.path = path
        self.identity = identity
        self.revision_id = revision_id
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._sealed = False
        try:
            self._client = _create_persistent_client(path)
            existing_names = {collection.name for collection in self._client.list_collections()}
            if collection_name not in existing_names:
                raise DenseIndexError("dense_collection_missing")
            self._collection = self._client.get_collection(
                collection_name,
                embedding_function=None,
            )
            metadata = self._validated_collection_metadata(require_sealed=require_sealed)
            self._sealed = bool(metadata["sealed"])
            self._validate_configuration()
            inventory = self._read_inventory()
            if metadata["chunk_count"] != len(inventory):
                raise DenseIndexError("dense_inventory_count_mismatch")
            if metadata["chunk_set_digest"] != chunk_set_digest(inventory):
                raise DenseIndexError("dense_chunk_set_digest_mismatch")
        except BaseException:
            self.close()
            raise

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    @property
    def chunk_ids(self) -> frozenset[str]:
        return frozenset(self._read_inventory())

    @property
    def record_digests(self) -> dict[str, str]:
        return self._read_inventory()

    @property
    def inventory_digest(self) -> str:
        return chunk_set_digest(self._read_inventory())

    @property
    def configuration(self) -> dict[str, object]:
        collection = self._open_collection()
        return cast(dict[str, object], collection.configuration)

    def add(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        titles: Mapping[str, str],
    ) -> None:
        if self._sealed:
            raise DenseIndexError("dense_index_sealed")
        if len(chunks) != len(embeddings):
            raise DenseIndexError("embedding_count_mismatch")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise DenseIndexError("duplicate_chunk_id")

        vectors: list[list[float]] = []
        metadatas: list[dict[str, str | int]] = []
        requested_digests: dict[str, str] = {}
        try:
            for chunk, vector in zip(chunks, embeddings, strict=True):
                normalized = [float(value) for value in vector]
                if len(normalized) != self.identity.dimension or not all(
                    math.isfinite(value) for value in normalized
                ):
                    raise DenseIndexError("incompatible_embedding")
                title = titles[chunk.source_id]
                digest = chunk_record_digest(chunk, title)
                vectors.append(normalized)
                requested_digests[chunk.chunk_id] = digest
                metadatas.append(
                    {
                        "source_id": chunk.source_id,
                        "parent_chunk_id": chunk.parent_chunk_id,
                        "display_title": title,
                        "document_version": chunk.document_version,
                        "ordinal": chunk.ordinal,
                        "content_digest": chunk.content_digest,
                        "locator": canonical_locator_json(chunk),
                        "record_digest": digest,
                    }
                )
        except KeyError:
            raise DenseIndexError("missing_display_title") from None
        except (TypeError, ValueError, OverflowError) as error:
            if isinstance(error, DenseIndexError):
                raise
            raise DenseIndexError("incompatible_embedding") from None

        before = self._read_inventory()
        collection = self._open_collection()
        client = self._open_client()
        max_batch_size = int(client.get_max_batch_size())
        if max_batch_size < 1:
            raise DenseIndexError("invalid_dense_batch_size")
        ordered_chunks = tuple(chunks)
        try:
            for start in range(0, len(ordered_chunks), max_batch_size):
                end = start + max_batch_size
                batch = ordered_chunks[start:end]
                collection.add(
                    ids=[chunk.chunk_id for chunk in batch],
                    embeddings=vectors[start:end],
                    documents=[chunk.text for chunk in batch],
                    metadatas=metadatas[start:end],
                )
        except Exception:
            raise DenseIndexError("dense_write_failed") from None

        after = self._read_inventory()
        expected = {**before, **requested_digests}
        if len(after) != len(before) + len(requested_digests) or after != expected:
            raise DenseIndexError("dense_post_write_inventory_mismatch")
        self._write_inventory_metadata(after, sealed=False)

    def seal(self) -> str:
        if self._sealed:
            raise DenseIndexError("dense_index_sealed")
        inventory = self._read_inventory()
        self._write_inventory_metadata(inventory, sealed=True)
        self._sealed = True
        return chunk_set_digest(inventory)

    def query(
        self,
        query_vector: Sequence[float],
        *,
        query_identity: EmbeddingSpaceIdentity,
        limit: int,
    ) -> tuple[RetrievalCandidate, ...]:
        if query_identity != self.identity:
            raise DenseIndexError("embedding_identity_mismatch")
        if isinstance(query_vector, (str, bytes, bytearray)) or not isinstance(
            query_vector,
            Sequence,
        ):
            raise DenseIndexError("incompatible_embedding") from None
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in query_vector):
            raise DenseIndexError("incompatible_embedding")
        try:
            vector = [float(value) for value in query_vector]
        except (TypeError, ValueError, OverflowError):
            raise DenseIndexError("incompatible_embedding") from None
        if len(vector) != self.identity.dimension or not all(
            math.isfinite(value) for value in vector
        ):
            raise DenseIndexError("incompatible_embedding")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be positive")
        inventory = self._read_inventory()
        if not inventory:
            return ()
        collection = self._open_collection()
        try:
            result: dict[str, Any] = collection.query(
                query_embeddings=[vector],
                n_results=len(inventory),
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            raise DenseIndexError("dense_query_failed") from None
        try:
            rows = list(
                zip(
                    result["ids"][0],
                    result["documents"][0],
                    result["metadatas"][0],
                    result["distances"][0],
                    strict=True,
                )
            )
            if len(rows) != len(inventory) or {str(row[0]) for row in rows} != set(inventory):
                raise DenseIndexError("dense_query_inventory_mismatch")
            if any(not math.isfinite(float(row[3])) for row in rows):
                raise DenseIndexError("dense_query_record_invalid")
            rows.sort(key=lambda row: (float(row[3]), str(row[0])))
        except DenseIndexError:
            raise
        except (KeyError, TypeError, ValueError, OverflowError):
            raise DenseIndexError("dense_query_record_invalid") from None
        try:
            candidates: list[RetrievalCandidate] = []
            for rank, (chunk_id, document, metadata, distance) in enumerate(
                rows[:limit],
                start=1,
            ):
                if not isinstance(chunk_id, str) or not isinstance(document, str):
                    raise DenseIndexError("dense_query_record_invalid")
                if not isinstance(metadata, dict):
                    raise DenseIndexError("dense_query_record_invalid")
                chunk = Chunk(
                    chunk_id=str(chunk_id),
                    parent_chunk_id=_required_string(metadata, "parent_chunk_id"),
                    source_id=_required_string(metadata, "source_id"),
                    document_version=_required_int(metadata, "document_version", minimum=1),
                    ordinal=_required_int(metadata, "ordinal", minimum=0),
                    text=document,
                    content_digest=_required_string(metadata, "content_digest"),
                    locator=ChunkLocator.model_validate_json(_required_string(metadata, "locator")),
                )
                title = _required_string(metadata, "display_title")
                record_digest = _required_string(metadata, "record_digest")
                if (
                    chunk_record_digest(chunk, title) != record_digest
                    or inventory.get(chunk_id) != record_digest
                ):
                    raise DenseIndexError("dense_query_record_digest_mismatch")
                candidates.append(
                    RetrievalCandidate(
                        chunk_id=chunk.chunk_id,
                        parent_chunk_id=chunk.parent_chunk_id,
                        source_id=chunk.source_id,
                        display_title=title,
                        document_version=chunk.document_version,
                        locator=chunk.locator,
                        text=chunk.text,
                        revision_id=self.revision_id,
                        ordinal=chunk.ordinal,
                        content_digest=chunk.content_digest,
                        record_digest=record_digest,
                        dense_rank=rank,
                        dense_score=1.0 - float(distance),
                    )
                )
            return tuple(candidates)
        except DenseIndexError:
            raise
        except (KeyError, TypeError, ValueError, ValidationError):
            raise DenseIndexError("dense_query_record_invalid") from None

    def close(self) -> None:
        client = getattr(self, "_client", None)
        self._collection = None
        self._client = None
        if client is not None:
            _close_persistent_client(client)

    def __enter__(self) -> Self:
        self._open_client()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def _collection_metadata(
        self,
        *,
        sealed: bool,
        chunk_count: int,
        inventory_digest: str,
    ) -> dict[str, str | int | bool]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "revision_id": self.revision_id,
            "metric": self.METRIC,
            "record_digest_algorithm": RECORD_DIGEST_ALGORITHM,
            "embedding_provider_alias": self.identity.provider_alias,
            "embedding_model": self.identity.model,
            "embedding_dimension": self.identity.dimension,
            "embedding_normalization": self.identity.normalization,
            "embedding_adapter_version": self.identity.adapter_version,
            "sealed": sealed,
            "chunk_count": chunk_count,
            "chunk_set_digest": inventory_digest,
        }

    def _validated_collection_metadata(self, *, require_sealed: bool) -> dict[str, Any]:
        metadata = self._open_collection().metadata
        if not isinstance(metadata, dict):
            raise DenseIndexError("dense_metadata_missing")
        expected = self._collection_metadata(
            sealed=bool(metadata.get("sealed")),
            chunk_count=cast(int, metadata.get("chunk_count", -1)),
            inventory_digest=cast(str, metadata.get("chunk_set_digest", "")),
        )
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise DenseIndexError("dense_metadata_mismatch")
        if type(metadata.get("sealed")) is not bool:
            raise DenseIndexError("dense_metadata_mismatch")
        if type(metadata.get("chunk_count")) is not int or metadata["chunk_count"] < 0:
            raise DenseIndexError("dense_metadata_mismatch")
        if require_sealed and metadata["sealed"] is not True:
            raise DenseIndexError("dense_index_not_sealed")
        return metadata

    def _validate_configuration(self) -> None:
        configuration = self._open_collection().configuration
        try:
            metric = configuration["hnsw"]["space"]
            embedding_function = configuration["embedding_function"]
        except (KeyError, TypeError):
            raise DenseIndexError("dense_configuration_invalid") from None
        if metric != self.METRIC or embedding_function is not None:
            raise DenseIndexError("dense_configuration_mismatch")

    def _read_inventory(self) -> dict[str, str]:
        collection = self._open_collection()
        try:
            result: dict[str, Any] = collection.get(include=["documents", "metadatas"])
            ids = result["ids"]
            documents = result["documents"]
            metadatas = result["metadatas"]
            if documents is None or metadatas is None:
                raise DenseIndexError("dense_inventory_invalid")
            if not (len(ids) == len(documents) == len(metadatas) == collection.count()):
                raise DenseIndexError("dense_inventory_count_mismatch")
            if len(ids) != len(set(ids)):
                raise DenseIndexError("duplicate_chunk_id")
            inventory: dict[str, str] = {}
            for chunk_id, document, metadata in zip(ids, documents, metadatas, strict=True):
                if not isinstance(chunk_id, str) or not isinstance(document, str):
                    raise DenseIndexError("dense_record_invalid")
                if not isinstance(metadata, dict):
                    raise DenseIndexError("dense_record_invalid")
                chunk = Chunk(
                    chunk_id=chunk_id,
                    parent_chunk_id=_required_string(metadata, "parent_chunk_id"),
                    source_id=_required_string(metadata, "source_id"),
                    document_version=_required_int(metadata, "document_version", minimum=1),
                    ordinal=_required_int(metadata, "ordinal", minimum=0),
                    text=document,
                    content_digest=_required_string(metadata, "content_digest"),
                    locator=ChunkLocator.model_validate_json(_required_string(metadata, "locator")),
                )
                title = _required_string(metadata, "display_title")
                recomputed = chunk_record_digest(chunk, title)
                if _required_string(metadata, "record_digest") != recomputed:
                    raise DenseIndexError("dense_record_digest_mismatch")
                inventory[chunk_id] = recomputed
            return inventory
        except DenseIndexError:
            raise
        except (KeyError, TypeError, ValueError, ValidationError, json.JSONDecodeError):
            raise DenseIndexError("dense_inventory_invalid") from None

    def _write_inventory_metadata(self, inventory: Mapping[str, str], *, sealed: bool) -> None:
        self._open_collection().modify(
            metadata=self._collection_metadata(
                sealed=sealed,
                chunk_count=len(inventory),
                inventory_digest=chunk_set_digest(inventory),
            )
        )

    def _open_client(self) -> Any:
        if self._client is None:
            raise DenseIndexError("dense_index_closed")
        return self._client

    def _open_collection(self) -> Any:
        if self._collection is None:
            raise DenseIndexError("dense_index_closed")
        return self._collection


def _required_string(metadata: Mapping[str, object], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise DenseIndexError("dense_record_invalid")
    return value


def _required_int(metadata: Mapping[str, object], key: str, *, minimum: int) -> int:
    value = metadata.get(key)
    if type(value) is not int or value < minimum:
        raise DenseIndexError("dense_record_invalid")
    return value
