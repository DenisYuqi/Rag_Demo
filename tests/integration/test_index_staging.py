from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
import pytest

from rag_mvp.domain.ingestion import (
    Chunk,
    ChunkLocator,
    EmbeddingSpaceIdentity,
    IndexRevisionStatus,
    ParentChunk,
)
from rag_mvp.ingestion.chunking import token_spans
from rag_mvp.ingestion.embedding import EmbeddingStage
from rag_mvp.ingestion.indexing import IndexingError, RevisionStager
from rag_mvp.providers.fakes import DeterministicEmbeddingProvider
from rag_mvp.providers.models import Deadline, ProviderCallContext
from rag_mvp.retrieval.bm25 import LexicalIndexError, PersistentBm25Index
from rag_mvp.retrieval.dense import DenseIndexError, PersistentChromaIndex
from rag_mvp.retrieval.snapshot import chunk_record_digest
from rag_mvp.retrieval.tokenizer import BILINGUAL_TOKENIZER_IDENTITY
from rag_mvp.storage.embedding_cache import EmbeddingCache
from rag_mvp.storage.layout import DataLayout


def _context(operation: str) -> ProviderCallContext:
    return ProviderCallContext(
        request_id=f"request-{operation}",
        operation_id=operation,
        deadline=Deadline.after(30),
    )


def _chunk(
    chunk_id: str,
    source_id: str,
    text: str,
    ordinal: int,
    *,
    version: int = 1,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        parent_chunk_id=f"parent-{chunk_id}",
        source_id=source_id,
        document_version=version,
        ordinal=ordinal,
        text=text,
        content_digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        locator=ChunkLocator(
            section_path=("Policies",),
            char_start=ordinal * 100,
            char_end=ordinal * 100 + len(text),
        ),
    )


def _parents(chunks: tuple[Chunk, ...]) -> tuple[ParentChunk, ...]:
    return tuple(
        ParentChunk(
            parent_chunk_id=chunk.parent_chunk_id,
            source_id=chunk.source_id,
            document_version=chunk.document_version,
            ordinal=chunk.ordinal,
            text=chunk.text,
            content_digest=chunk.content_digest,
            locator=chunk.locator,
            token_count=len(token_spans(chunk.text)),
        )
        for chunk in chunks
    )


async def test_staging_rejects_missing_and_orphan_parent_inventory(tmp_path: Path) -> None:
    layout = DataLayout.from_root(tmp_path / "parent-inventory")
    layout.initialize()
    chunk = _chunk("chunk-one", "source-one", "parent integrity", 0)
    parent = _parents((chunk,))[0]
    orphan = parent.model_copy(update={"parent_chunk_id": "parent-orphan", "ordinal": 1})

    with EmbeddingCache(layout.directory("caches") / "embeddings.sqlite3") as cache:
        stager = RevisionStager(
            layout,
            EmbeddingStage(DeterministicEmbeddingProvider(), cache),
        )
        for revision_id, parents in (
            ("revision-missing-parent", ()),
            ("revision-orphan-parent", (parent, orphan)),
        ):
            with pytest.raises(IndexingError, match="parent_child_inventory_mismatch"):
                await stager.stage(
                    revision_id,
                    (chunk,),
                    {"source-one": "One"},
                    {"source-one": 1},
                    _context(revision_id),
                    parents=parents,
                )

            assert not layout.index_revision_path(revision_id).exists()


async def test_bilingual_snapshot_persists_complete_identity_and_metadata(tmp_path: Path) -> None:
    layout = DataLayout.from_root(tmp_path / "data")
    layout.initialize()
    chunks = (
        _chunk("chunk-english", "source-handbook", "Policy code AKP-204 requires MFA", 0),
        _chunk("chunk-chinese", "source-handbook", "员工休假政策规定年假天数", 1),
    )
    titles = {"source-handbook": "员工 Handbook"}

    with EmbeddingCache(layout.directory("caches") / "embeddings.sqlite3") as cache:
        provider = DeterministicEmbeddingProvider()
        embeddings = EmbeddingStage(provider, cache, batch_size=1)
        revision = await RevisionStager(layout, embeddings).stage(
            "revision-bilingual",
            chunks,
            titles,
            {"source-handbook": 1},
            _context("stage-bilingual"),
            parents=_parents(chunks),
        )
        vectors = (await embeddings.embed(chunks, _context("read-vectors"))).vectors

    assert revision.status is IndexRevisionStatus.STAGED
    assert revision.chunk_count == 2
    assert revision.dense_index_path == "indexes/revisions/revision-bilingual/chroma"
    assert revision.lexical_index_path == "indexes/revisions/revision-bilingual/bm25.json"
    assert revision.tokenizer_version == BILINGUAL_TOKENIZER_IDENTITY
    assert revision.dense_schema_version == PersistentChromaIndex.SCHEMA_VERSION
    assert revision.dense_metric == "cosine"
    assert revision.lexical_schema_version == PersistentBm25Index.SNAPSHOT_SCHEMA
    assert revision.lexical_algorithm_version == PersistentBm25Index.ALGORITHM_VERSION

    expected_digests = {
        chunk.chunk_id: chunk_record_digest(chunk, titles[chunk.source_id]) for chunk in chunks
    }
    dense_path = layout.dense_index_path(revision.revision_id)
    with PersistentChromaIndex.open_existing(
        dense_path,
        revision_id=revision.revision_id,
        identity=revision.embedding_space,
    ) as dense:
        dense_results = dense.query(
            vectors[0],
            query_identity=revision.embedding_space,
            limit=2,
        )
        assert dense.is_sealed
        assert dense.collection_name == PersistentChromaIndex.COLLECTION_NAME
        assert dense.chunk_ids == {"chunk-english", "chunk-chinese"}
        dense_chunk_ids = dense.chunk_ids
        assert dense.record_digests == expected_digests
        assert dense.inventory_digest == revision.chunk_set_digest
        hnsw = dense.configuration["hnsw"]
        assert isinstance(hnsw, dict)
        assert hnsw["space"] == "cosine"

    assert dense_results[0].chunk_id == "chunk-english"
    assert dense_results[0].source_id == "source-handbook"
    assert dense_results[0].display_title == "员工 Handbook"
    assert dense_results[0].document_version == 1
    assert dense_results[0].locator.section_path == ("Policies",)

    lexical = PersistentBm25Index.load(
        layout.lexical_index_path(revision.revision_id),
        expected_revision_id=revision.revision_id,
    )
    assert lexical.chunk_ids == dense_chunk_ids
    assert lexical.record_digests == expected_digests
    assert lexical.chunk_set_digest == revision.chunk_set_digest
    assert lexical.tokenizer_identity == BILINGUAL_TOKENIZER_IDENTITY
    assert lexical.algorithm_version == PersistentBm25Index.ALGORITHM_VERSION
    assert lexical.algorithm_config == {"k1": 1.5, "b": 0.75}
    assert {record.display_title for record in lexical.records} == {"员工 Handbook"}
    english = await lexical.search("AKP-204", 5)
    chinese = await lexical.search("休假政策", 5)
    assert english[0].chunk_id == "chunk-english"
    assert chinese[0].chunk_id == "chunk-chinese"


async def test_staged_paths_and_snapshots_are_immutable_and_empty_is_supported(
    tmp_path: Path,
) -> None:
    layout = DataLayout.from_root(tmp_path / "data")
    layout.initialize()
    chunk = _chunk("chunk-one", "source-one", "immutable record", 0)

    with EmbeddingCache(layout.directory("caches") / "embeddings.sqlite3") as cache:
        embeddings = EmbeddingStage(DeterministicEmbeddingProvider(), cache)
        stager = RevisionStager(layout, embeddings)
        revision = await stager.stage(
            "revision-immutable",
            (chunk,),
            {"source-one": "One"},
            {"source-one": 1},
            _context("immutable"),
            parents=_parents((chunk,)),
        )

        with pytest.raises(IndexingError, match="revision_path_exists"):
            await stager.stage(
                revision.revision_id,
                (chunk,),
                {"source-one": "One"},
                {"source-one": 1},
                _context("duplicate-revision"),
                parents=_parents((chunk,)),
            )

        empty = await stager.stage(
            "revision-empty",
            (),
            {},
            {},
            _context("empty"),
            parents=(),
        )

    with (
        PersistentChromaIndex.open_existing(
            layout.dense_index_path(revision.revision_id),
            revision_id=revision.revision_id,
            identity=revision.embedding_space,
        ) as dense,
        pytest.raises(DenseIndexError, match="dense_index_sealed"),
    ):
        dense.add(
            (chunk,),
            ((1.0,) * revision.embedding_space.dimension,),
            {"source-one": "One"},
        )

    lexical_path = layout.lexical_index_path(revision.revision_id)
    lexical = PersistentBm25Index.load(
        lexical_path,
        expected_revision_id=revision.revision_id,
    )
    with pytest.raises(LexicalIndexError, match="snapshot_exists"):
        lexical.save_new(lexical_path)

    assert empty.chunk_count == 0
    with PersistentChromaIndex.open_existing(
        layout.dense_index_path(empty.revision_id),
        revision_id=empty.revision_id,
        identity=empty.embedding_space,
    ) as empty_dense:
        assert empty_dense.chunk_ids == frozenset()
        assert empty_dense.is_sealed
    empty_lexical = PersistentBm25Index.load(
        layout.lexical_index_path(empty.revision_id),
        expected_revision_id=empty.revision_id,
    )
    assert empty_lexical.chunk_ids == frozenset()
    assert await empty_lexical.search("anything", 5) == ()


async def test_missing_and_corrupt_indexes_fail_open_without_creation(tmp_path: Path) -> None:
    layout = DataLayout.from_root(tmp_path / "data")
    layout.initialize()
    provider = DeterministicEmbeddingProvider()

    missing_dense = layout.dense_index_path("revision-missing")
    with pytest.raises(DenseIndexError, match="dense_index_missing"):
        PersistentChromaIndex.open_existing(
            missing_dense,
            revision_id="revision-missing",
            identity=_domain_identity(provider),
        )
    assert not missing_dense.exists()

    missing_lexical = layout.lexical_index_path("revision-missing")
    with pytest.raises(LexicalIndexError, match="snapshot_missing"):
        PersistentBm25Index.load(
            missing_lexical,
            expected_revision_id="revision-missing",
        )
    assert not missing_lexical.exists()

    corrupt_lexical = layout.lexical_index_path("revision-corrupt-json")
    corrupt_lexical.parent.mkdir(parents=True)
    corrupt_lexical.write_text("{not-json", encoding="utf-8")
    with pytest.raises(LexicalIndexError, match="invalid_snapshot"):
        PersistentBm25Index.load(
            corrupt_lexical,
            expected_revision_id="revision-corrupt-json",
        )
    assert corrupt_lexical.read_text(encoding="utf-8") == "{not-json"

    chunk = _chunk("chunk-corrupt", "source-corrupt", "corrupt metadata", 0)
    with EmbeddingCache(layout.directory("caches") / "embeddings.sqlite3") as cache:
        revision = await RevisionStager(layout, EmbeddingStage(provider, cache)).stage(
            "revision-corrupt-metadata",
            (chunk,),
            {"source-corrupt": "Corrupt"},
            {"source-corrupt": 1},
            _context("corrupt"),
            parents=_parents((chunk,)),
        )

    dense_path = layout.dense_index_path(revision.revision_id)
    client = chromadb.PersistentClient(path=str(dense_path))
    collection = client.get_collection(
        PersistentChromaIndex.COLLECTION_NAME,
        embedding_function=None,
    )
    metadata = dict(collection.metadata or {})
    metadata["revision_id"] = "tampered-revision"
    collection.modify(metadata=metadata)
    client.close()

    with pytest.raises(DenseIndexError, match="dense_metadata_mismatch"):
        PersistentChromaIndex.open_existing(
            dense_path,
            revision_id=revision.revision_id,
            identity=revision.embedding_space,
        )
    assert (dense_path / "chroma.sqlite3").is_file()


async def test_dense_parent_mapping_tampering_fails_closed(tmp_path: Path) -> None:
    layout = DataLayout.from_root(tmp_path / "data-parent-tamper")
    layout.initialize()
    provider = DeterministicEmbeddingProvider()
    chunk = _chunk("chunk-parent-tamper", "source-parent", "parent mapping", 0)
    with EmbeddingCache(layout.directory("caches") / "embeddings.sqlite3") as cache:
        revision = await RevisionStager(layout, EmbeddingStage(provider, cache)).stage(
            "revision-parent-tamper",
            (chunk,),
            {"source-parent": "Parent"},
            {"source-parent": 1},
            _context("parent-tamper"),
            parents=_parents((chunk,)),
        )

    dense_path = layout.dense_index_path(revision.revision_id)
    client = chromadb.PersistentClient(path=str(dense_path))
    collection = client.get_collection(
        PersistentChromaIndex.COLLECTION_NAME,
        embedding_function=None,
    )
    record = collection.get(ids=[chunk.chunk_id], include=["metadatas"])
    metadata = dict(record["metadatas"][0])
    metadata["parent_chunk_id"] = "parent-tampered"
    collection.update(ids=[chunk.chunk_id], metadatas=[metadata])
    client.close()

    with pytest.raises(DenseIndexError, match="dense_record_digest_mismatch"):
        PersistentChromaIndex.open_existing(
            dense_path,
            revision_id=revision.revision_id,
            identity=revision.embedding_space,
        )


def _domain_identity(provider: DeterministicEmbeddingProvider) -> EmbeddingSpaceIdentity:
    identity = provider.identity
    return EmbeddingSpaceIdentity(
        provider_alias=identity.provider,
        model=identity.model,
        dimension=identity.dimension,
        normalization=identity.normalization.value,
        adapter_version=identity.adapter_version,
    )
