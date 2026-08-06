"""Repository-backed binding to one immutable committed retrieval revision."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType, TracebackType
from typing import Self

from rag_mvp.domain.ingestion import DocumentKind, IndexRevision, IndexRevisionStatus
from rag_mvp.retrieval.bm25 import LexicalIndexError, PersistentBm25Index
from rag_mvp.retrieval.dense import DenseIndexError, PersistentChromaIndex
from rag_mvp.retrieval.request import RetrievalRequestError
from rag_mvp.retrieval.snapshot import RECORD_DIGEST_ALGORITHM
from rag_mvp.storage.layout import DataLayout, UnsafeDataPathError
from rag_mvp.storage.repositories import DocumentRepository, IndexRevisionRepository

_BOUND_PROOF = object()


@dataclass(frozen=True, slots=True)
class BoundRetrievalSnapshot:
    """Validated handles for one revision captured while it was active."""

    revision: IndexRevision
    dense: PersistentChromaIndex
    bm25: PersistentBm25Index
    source_kinds: Mapping[str, DocumentKind]
    _proof: object | None = field(default=None, repr=False)
    _binding_token: object = field(default_factory=object, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self._proof is not _BOUND_PROOF:
            raise RetrievalRequestError("invalid_snapshot_binding")
        try:
            source_kinds = {
                source_id: DocumentKind(kind) for source_id, kind in self.source_kinds.items()
            }
        except (AttributeError, TypeError, ValueError):
            raise RetrievalRequestError("invalid_snapshot_binding") from None
        if set(source_kinds) != set(self.revision.active_sources):
            raise RetrievalRequestError("invalid_snapshot_binding")
        object.__setattr__(self, "source_kinds", MappingProxyType(source_kinds))

    @classmethod
    def bind_active(
        cls,
        layout: DataLayout,
        revisions: IndexRevisionRepository,
    ) -> BoundRetrievalSnapshot:
        """Capture the complete active manifest, then open only its immutable artifacts."""

        try:
            revision = revisions.get_active()
        except Exception:
            raise RetrievalRequestError("index_manifest_invalid") from None
        if revision is None:
            raise RetrievalRequestError("index_not_ready")
        if revision.status is not IndexRevisionStatus.ACTIVE:
            raise RetrievalRequestError("index_manifest_invalid")
        return cls._open_captured(layout, revision, _source_kinds(revisions, revision))

    @classmethod
    def open_committed(
        cls,
        layout: DataLayout,
        revisions: IndexRevisionRepository,
        revision_id: str,
    ) -> BoundRetrievalSnapshot:
        """Resolve and open one repository-committed active or superseded revision."""

        try:
            layout.index_revision_path(revision_id)
        except (UnsafeDataPathError, TypeError, ValueError):
            raise RetrievalRequestError("invalid_revision_id") from None
        try:
            revision = revisions.get(revision_id)
        except Exception:
            raise RetrievalRequestError("index_manifest_invalid") from None
        if revision is None or revision.status not in {
            IndexRevisionStatus.ACTIVE,
            IndexRevisionStatus.SUPERSEDED,
        }:
            raise RetrievalRequestError("revision_not_committed")
        return cls._open_captured(layout, revision, _source_kinds(revisions, revision))

    @classmethod
    def _open_captured(
        cls,
        layout: DataLayout,
        revision: IndexRevision,
        source_kinds: Mapping[str, DocumentKind],
    ) -> BoundRetrievalSnapshot:
        dense: PersistentChromaIndex | None = None
        try:
            expected_dense_path = layout.dense_index_path(revision.revision_id)
            expected_bm25_path = layout.lexical_index_path(revision.revision_id)
            if (
                layout.resolve_artifact_path(revision.dense_index_path) != expected_dense_path
                or layout.resolve_artifact_path(revision.lexical_index_path) != expected_bm25_path
            ):
                raise RetrievalRequestError("index_artifact_invalid", detail_code="path_mismatch")
            if (
                revision.dense_schema_version != PersistentChromaIndex.SCHEMA_VERSION
                or revision.dense_metric != PersistentChromaIndex.METRIC
                or revision.lexical_schema_version != PersistentBm25Index.SNAPSHOT_SCHEMA
                or revision.lexical_algorithm_version != PersistentBm25Index.ALGORITHM_VERSION
                or revision.record_digest_algorithm != RECORD_DIGEST_ALGORITHM
            ):
                raise RetrievalRequestError(
                    "index_artifact_invalid",
                    detail_code="index_identity_mismatch",
                )

            dense = PersistentChromaIndex.open_existing(
                expected_dense_path,
                revision_id=revision.revision_id,
                identity=revision.embedding_space,
            )
            bm25 = PersistentBm25Index.load(
                expected_bm25_path,
                expected_revision_id=revision.revision_id,
            )
            _validate_manifest(revision, dense, bm25)
            return cls(
                revision=revision,
                dense=dense,
                bm25=bm25,
                source_kinds=source_kinds,
                _proof=_BOUND_PROOF,
            )
        except RetrievalRequestError:
            if dense is not None:
                dense.close()
            raise
        except (DenseIndexError, LexicalIndexError) as error:
            if dense is not None:
                dense.close()
            missing_codes = {"dense_index_missing", "snapshot_missing"}
            code = (
                "index_artifact_missing"
                if error.code in missing_codes
                else "index_artifact_invalid"
            )
            raise RetrievalRequestError(code, detail_code=error.code) from None
        except Exception:
            if dense is not None:
                dense.close()
            raise RetrievalRequestError("index_artifact_invalid") from None

    @property
    def revision_id(self) -> str:
        return self.revision.revision_id

    @property
    def lexical(self) -> PersistentBm25Index:
        return self.bm25

    @property
    def binding_token(self) -> object:
        return self._binding_token

    @property
    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if not self._closed:
            self.dense.close()
            object.__setattr__(self, "_closed", True)

    def __enter__(self) -> Self:
        if self._closed:
            raise RetrievalRequestError("snapshot_closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


@dataclass(frozen=True, slots=True)
class BoundRetrievalSnapshotFactory:
    layout: DataLayout
    revisions: IndexRevisionRepository

    def bind(self) -> BoundRetrievalSnapshot:
        return BoundRetrievalSnapshot.bind_active(self.layout, self.revisions)

    def open_committed(self, revision_id: str) -> BoundRetrievalSnapshot:
        return BoundRetrievalSnapshot.open_committed(
            self.layout,
            self.revisions,
            revision_id,
        )


def _validate_manifest(
    revision: IndexRevision,
    dense: PersistentChromaIndex,
    bm25: PersistentBm25Index,
) -> None:
    if dense.revision_id != revision.revision_id or bm25.revision_id != revision.revision_id:
        raise RetrievalRequestError("index_artifact_invalid", detail_code="revision_id_mismatch")
    if dense.identity != revision.embedding_space:
        raise RetrievalRequestError(
            "index_artifact_invalid",
            detail_code="embedding_identity_mismatch",
        )
    if (
        bm25.tokenizer_identity != revision.tokenizer_version
        or bm25.algorithm_version != revision.lexical_algorithm_version
        or bm25.k1 != revision.lexical_k1
        or bm25.b != revision.lexical_b
    ):
        raise RetrievalRequestError(
            "index_artifact_invalid",
            detail_code="lexical_identity_mismatch",
        )
    if (
        dense.chunk_ids != bm25.chunk_ids
        or dense.record_digests != bm25.record_digests
        or dense.inventory_digest != revision.chunk_set_digest
        or bm25.chunk_set_digest != revision.chunk_set_digest
        or len(dense.chunk_ids) != revision.chunk_count
    ):
        raise RetrievalRequestError("index_artifact_invalid", detail_code="inventory_mismatch")
    record_source_ids: set[str] = set()
    for record in bm25.records:
        source_id = record.chunk.source_id
        record_source_ids.add(source_id)
        if revision.active_sources.get(source_id) != record.chunk.document_version:
            raise RetrievalRequestError(
                "index_artifact_invalid",
                detail_code="active_sources_mismatch",
            )
    if record_source_ids != set(revision.active_sources):
        raise RetrievalRequestError("index_artifact_invalid", detail_code="active_sources_mismatch")


def _source_kinds(
    revisions: IndexRevisionRepository,
    revision: IndexRevision,
) -> Mapping[str, DocumentKind]:
    documents = DocumentRepository(revisions.database)
    result: dict[str, DocumentKind] = {}
    try:
        for source_id in revision.active_sources:
            document = documents.get(source_id)
            if document is None or document.source_id != source_id:
                raise RetrievalRequestError(
                    "index_manifest_invalid",
                    detail_code="source_kind_missing",
                )
            result[source_id] = document.kind
    except RetrievalRequestError:
        raise
    except Exception:
        raise RetrievalRequestError(
            "index_manifest_invalid",
            detail_code="source_kind_invalid",
        ) from None
    return result
