"""Transactional source deduplication and immutable document version registration."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from rag_mvp.domain._base import utc_now
from rag_mvp.domain.ingestion import (
    Document,
    DocumentVersion,
    ExtractionMethod,
)
from rag_mvp.ingestion.extractors import ExtractedDocument
from rag_mvp.ingestion.normalization import canonical_document_digest, normalize_document
from rag_mvp.ingestion.validation import ValidatedUpload
from rag_mvp.storage.artifacts import ArtifactStore, StoredVersionArtifacts
from rag_mvp.storage.database import Database
from rag_mvp.storage.repositories import DocumentRepository, RepositoryNotFound

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | Mapping[str, "JsonValue"] | Sequence["JsonValue"]

_DERIVATION_SECTIONS = frozenset({"extraction", "ocr", "normalization", "chunking", "tokenizer"})


def derivation_config_digest(config: Mapping[str, JsonValue]) -> str:
    """Hash only the five configuration sections that derive canonical chunks."""
    missing = _DERIVATION_SECTIONS - config.keys()
    if missing:
        raise ValueError(
            "derivation config requires extraction, ocr, normalization, chunking, and tokenizer"
        )
    canonical = _canonical_json_mapping(
        {section: config[section] for section in _DERIVATION_SECTIONS}
    )
    serialized = json.dumps(
        canonical,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _canonical_json_mapping(value: Mapping[str, JsonValue]) -> dict[str, object]:
    canonical: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("derivation config keys must be strings")
        canonical[key] = _canonical_json_value(item)
    return canonical


def _canonical_json_value(value: JsonValue) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("derivation config numbers must be finite")
        return value
    if isinstance(value, Mapping):
        return _canonical_json_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_json_value(item) for item in value]
    raise ValueError("derivation config must contain only JSON-shaped values")


class SourceVersionDisposition(StrEnum):
    CREATED = "created"
    DUPLICATE = "duplicate"


@dataclass(frozen=True, slots=True)
class SourceVersionRegistration:
    disposition: SourceVersionDisposition
    document: Document
    version: DocumentVersion


class DeletedSourceError(RuntimeError):
    """A deleted source key must be explicitly restored by a later workflow."""


class SourceVersioningService:
    """Register source versions; deleted sources are rejected, not implicitly restored."""

    def __init__(
        self,
        database: Database,
        documents: DocumentRepository,
        artifacts: ArtifactStore,
        *,
        source_id_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._database = database
        self._documents = documents
        self._artifacts = artifacts
        self._source_id_factory = source_id_factory or (lambda: f"src_{uuid4().hex}")
        self._clock = clock

    def register(
        self,
        *,
        source_key: str,
        upload: ValidatedUpload,
        extracted_document: ExtractedDocument,
        derivation_config: Mapping[str, JsonValue],
        display_title: str | None = None,
    ) -> SourceVersionRegistration:
        if extracted_document.kind is not upload.kind:
            raise ValueError("extracted document kind does not match the validated upload")

        canonical_document = normalize_document(extracted_document)
        content_digest = canonical_document_digest(canonical_document)
        config_digest = derivation_config_digest(derivation_config)
        timestamp = self._clock()
        written: StoredVersionArtifacts | None = None
        try:
            with self._database.transaction(immediate=True) as connection:
                document = self._documents.get_by_source_key(
                    source_key,
                    connection=connection,
                )
                if document is not None and document.deleted_at is not None:
                    raise DeletedSourceError(
                        "deleted source keys cannot be restored during version registration"
                    )
                if document is not None and document.active_version is not None:
                    active = self._documents.get_version(
                        document.source_id,
                        document.active_version,
                        connection=connection,
                    )
                    if active is None:
                        raise RepositoryNotFound(
                            f"active document version {document.source_id!r}/"
                            f"{document.active_version} was not found"
                        )
                    if (
                        active.content_digest == content_digest
                        and active.derivation_config_digest == config_digest
                    ):
                        return SourceVersionRegistration(
                            disposition=SourceVersionDisposition.DUPLICATE,
                            document=document,
                            version=active,
                        )

                if document is None:
                    document = Document(
                        source_id=self._source_id_factory(),
                        source_key=source_key,
                        display_title=display_title or upload.filename,
                        media_type=upload.media_type,
                        kind=upload.kind,
                        created_at=timestamp,
                        updated_at=timestamp,
                    )
                    self._documents.create(document, connection=connection)

                version_number = self._documents.next_version(
                    document.source_id,
                    connection=connection,
                )
                written = self._artifacts.write_version(
                    source_id=document.source_id,
                    version=version_number,
                    original_filename=upload.filename,
                    source_content=upload.content,
                    canonical_document=canonical_document,
                )
                version = DocumentVersion(
                    source_id=document.source_id,
                    version=version_number,
                    content_digest=content_digest,
                    derivation_config_digest=config_digest,
                    original_filename=upload.filename,
                    media_type=upload.media_type,
                    size_bytes=len(upload.content),
                    source_artifact_path=written.source_artifact_path,
                    canonical_artifact_path=written.canonical_artifact_path,
                    extraction_method=_document_extraction_method(canonical_document),
                    created_at=timestamp,
                )
                self._documents.add_version(version, connection=connection)
                return SourceVersionRegistration(
                    disposition=SourceVersionDisposition.CREATED,
                    document=document,
                    version=version,
                )
        except Exception:
            if written is not None:
                self._artifacts.cleanup(written)
            raise


def _document_extraction_method(document: ExtractedDocument) -> ExtractionMethod:
    methods = {block.extraction_method for block in document.blocks}
    if len(methods) == 1:
        return next(iter(methods))
    return ExtractionMethod.MIXED
