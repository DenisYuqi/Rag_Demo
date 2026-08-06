"""Install an immutable evaluation corpus into the production index runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import fitz

from rag_mvp.domain.ingestion import (
    Document,
    DocumentKind,
    DocumentVersion,
    ExtractionMethod,
    IndexRevision,
)
from rag_mvp.evaluation.dataset import (
    CorpusDocument,
    CorpusSnapshotFormat,
    EvaluationDataset,
    materialize_production_documents,
)
from rag_mvp.ingestion.chunking import chunk_document
from rag_mvp.ingestion.extractors import ExtractedDocument
from rag_mvp.ingestion.service import IngestionService
from rag_mvp.ingestion.validation import ValidatedUpload
from rag_mvp.ingestion.versioning import (
    SourceVersionDisposition,
    SourceVersioningService,
)
from rag_mvp.storage.artifacts import ArtifactStore, StoredVersionArtifacts
from rag_mvp.storage.database import Database
from rag_mvp.storage.layout import DataLayout
from rag_mvp.storage.repositories import KnowledgeRepositories


class EvaluationCorpusInstallError(RuntimeError):
    """A stable failure to install a trusted evaluation snapshot."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class InstalledEvaluationCorpus:
    revision: IndexRevision
    dataset_id: str
    dataset_version: str
    dataset_hash: str
    corpus_version: str
    corpus_hash: str


@dataclass(frozen=True, slots=True)
class EvaluationCorpusInstaller:
    """Register source artifacts and publish their verified production chunks."""

    ingestion: IngestionService

    async def install(self, dataset: EvaluationDataset) -> InstalledEvaluationCorpus:
        if not isinstance(dataset, EvaluationDataset):
            raise EvaluationCorpusInstallError("evaluation_dataset_invalid")
        repositories = self.ingestion.repositories
        layout = DataLayout.from_root(self.ingestion.data_root)
        _require_empty_corpus_target(layout, repositories)
        self._validate_derivation(dataset)
        materialized = materialize_production_documents(dataset)
        for document, source_content, extracted in materialized:
            _validate_materialized_source(document, source_content, extracted)

        artifacts = ArtifactStore(layout)
        database = repositories.index_revisions.database
        versioning = SourceVersioningService(
            database,
            repositories.documents,
            artifacts,
        )
        registered_versions: list[DocumentVersion] = []
        created_source_ids: list[str] = []
        digest = dataset.corpus.manifest.content_hash.removeprefix("sha256:")
        revision_id = f"rev_eval_{digest}"
        try:
            for document, source_content, extracted in materialized:
                repositories.documents.create(
                    Document(
                        source_id=document.source_id,
                        source_key=document.source_key,
                        display_title=document.display_title,
                        media_type=document.media_type,
                        kind=document.kind,
                    )
                )
                created_source_ids.append(document.source_id)
                registration = versioning.register(
                    source_key=document.source_key,
                    upload=ValidatedUpload(
                        filename=_artifact_filename(document.source_id, document.kind),
                        media_type=document.media_type,
                        kind=document.kind,
                        content=source_content,
                    ),
                    extracted_document=extracted,
                    derivation_config=self.ingestion.derivation_config,
                    display_title=document.display_title,
                )
                registered_versions.append(registration.version)
                if (
                    registration.disposition is not SourceVersionDisposition.CREATED
                    or registration.document.source_id != document.source_id
                    or registration.version.version != document.document_version
                ):
                    raise EvaluationCorpusInstallError("evaluation_document_identity_mismatch")

            reproduced = tuple(
                chunk
                for version in registered_versions
                for chunk in chunk_document(
                    artifacts.load_canonical(version),
                    source_id=version.source_id,
                    document_version=version.version,
                    config=self.ingestion.chunking_config,
                )
            )
            expected = {chunk.chunk_id: chunk for chunk in dataset.production_chunks}
            actual = {chunk.chunk_id: chunk for chunk in reproduced}
            if actual != expected or len(actual) != len(reproduced):
                raise EvaluationCorpusInstallError("evaluation_chunk_identity_mismatch")

            revision = await self.ingestion.publish_prechunked_snapshot(
                revision_id=revision_id,
                chunks=dataset.production_chunks,
                titles={
                    document.source_id: document.display_title
                    for document in dataset.corpus.documents
                },
                active_sources=dataset.corpus.manifest.active_sources,
                request_id=f"eval_corpus_{digest}",
                cleanup_failed=True,
            )
        except BaseException as error:
            active_revision_id = repositories.index_revisions.get_active_revision_id()
            if active_revision_id != revision_id:
                try:
                    _rollback_documents(
                        database,
                        artifacts,
                        registered_versions,
                        created_source_ids,
                    )
                except Exception as rollback_error:
                    raise EvaluationCorpusInstallError(
                        "evaluation_corpus_rollback_failed"
                    ) from rollback_error
            if isinstance(error, EvaluationCorpusInstallError):
                raise
            if not isinstance(error, Exception):
                raise
            raise EvaluationCorpusInstallError("evaluation_corpus_install_failed") from error

        return InstalledEvaluationCorpus(
            revision=revision,
            dataset_id=dataset.manifest.dataset_id,
            dataset_version=dataset.manifest.version,
            dataset_hash=dataset.manifest.content_hash,
            corpus_version=dataset.corpus.manifest.version,
            corpus_hash=dataset.corpus.manifest.content_hash,
        )

    def _validate_derivation(self, dataset: EvaluationDataset) -> None:
        declared = dataset.corpus.manifest.derivation
        active = self.ingestion.chunking_config
        configuration = self.ingestion.derivation_config
        try:
            raw_extraction = configuration["extraction"]
            raw_normalization = configuration["normalization"]
            raw_chunking = configuration["chunking"]
        except KeyError:
            raise EvaluationCorpusInstallError("evaluation_derivation_mismatch") from None
        if not all(
            isinstance(section, Mapping)
            for section in (raw_extraction, raw_normalization, raw_chunking)
        ):
            raise EvaluationCorpusInstallError("evaluation_derivation_mismatch")
        extraction = cast(Mapping[str, object], raw_extraction)
        normalization = cast(Mapping[str, object], raw_normalization)
        chunking = cast(Mapping[str, object], raw_chunking)
        if (
            declared.target_tokens != active.target_tokens
            or declared.overlap_tokens != active.overlap_tokens
            or declared.chunking_version != active.version
            or declared.tokenizer_version != active.tokenizer_version
            or extraction.get("version") != declared.extraction_version
            or normalization.get("version") != declared.normalization_version
            or chunking.get("version") != declared.chunking_version
            or chunking.get("tokenizer_version") != declared.tokenizer_version
            or chunking.get("target_tokens") != declared.target_tokens
            or chunking.get("overlap_tokens") != declared.overlap_tokens
        ):
            raise EvaluationCorpusInstallError("evaluation_derivation_mismatch")


def _require_empty_corpus_target(
    layout: DataLayout,
    repositories: KnowledgeRepositories,
) -> None:
    metadata_exists = bool(
        repositories.documents.list(include_deleted=True)
        or repositories.ingestion_jobs.list()
        or repositories.index_revisions.list()
        or repositories.index_revisions.get_active_revision_id()
    )
    artifact_roots = (
        layout.directory("sources"),
        layout.directory("canonical"),
        layout.directory("jobs"),
        layout.index_revisions,
    )
    artifacts_exist = layout.active_manifest.exists() or any(
        any(root.iterdir()) for root in artifact_roots
    )
    if metadata_exists or artifacts_exist:
        raise EvaluationCorpusInstallError("evaluation_data_root_not_empty")


def _validate_materialized_source(
    document: CorpusDocument,
    source_content: bytes,
    extracted: ExtractedDocument,
) -> None:
    if document.snapshot_format is not CorpusSnapshotFormat.FROZEN_OCR_PAGE:
        return
    if (
        document.kind is not DocumentKind.PDF
        or document.media_type != "application/pdf"
        or extracted.kind is not DocumentKind.PDF
        or extracted.ocr_page_count < 1
        or any(block.extraction_method is not ExtractionMethod.OCR for block in extracted.blocks)
    ):
        raise EvaluationCorpusInstallError("evaluation_ocr_snapshot_invalid")
    try:
        pdf = fitz.open(stream=source_content, filetype="pdf")
        try:
            if pdf.needs_pass or pdf.is_repaired or pdf.page_count < 1:
                raise EvaluationCorpusInstallError("evaluation_ocr_source_invalid")
            if extracted.ocr_page_count != pdf.page_count:
                raise EvaluationCorpusInstallError("evaluation_ocr_snapshot_invalid")
            pages = {block.page_number for block in extracted.blocks}
            if pages != set(range(1, pdf.page_count + 1)):
                raise EvaluationCorpusInstallError("evaluation_ocr_snapshot_invalid")
            if any(
                pdf.load_page(index).get_text("text").strip() for index in range(pdf.page_count)
            ):
                raise EvaluationCorpusInstallError("evaluation_ocr_source_has_native_text")
        finally:
            pdf.close()
    except EvaluationCorpusInstallError:
        raise
    except Exception:
        raise EvaluationCorpusInstallError("evaluation_ocr_source_invalid") from None


def _rollback_documents(
    database: Database,
    artifacts: ArtifactStore,
    versions: list[DocumentVersion],
    source_ids: list[str],
) -> None:
    with database.transaction() as connection:
        for version in reversed(versions):
            connection.execute(
                "DELETE FROM document_versions WHERE source_id = ? AND version = ?",
                (version.source_id, version.version),
            )
        for source_id in reversed(source_ids):
            connection.execute("DELETE FROM documents WHERE source_id = ?", (source_id,))
    for version in reversed(versions):
        artifacts.cleanup(
            StoredVersionArtifacts(
                source_artifact_path=version.source_artifact_path,
                canonical_artifact_path=version.canonical_artifact_path,
            )
        )


def _artifact_filename(source_id: str, kind: DocumentKind) -> str:
    extension = {
        DocumentKind.PDF: ".pdf",
        DocumentKind.MARKDOWN: ".md",
        DocumentKind.TEXT: ".txt",
    }[kind]
    return f"{source_id}{extension}"


__all__ = [
    "EvaluationCorpusInstallError",
    "EvaluationCorpusInstaller",
    "InstalledEvaluationCorpus",
]
