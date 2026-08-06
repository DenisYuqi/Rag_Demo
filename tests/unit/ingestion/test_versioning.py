from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pytest

from rag_mvp.domain.ingestion import DocumentKind, ExtractionMethod
from rag_mvp.ingestion.extractors import ExtractedBlock, ExtractedDocument
from rag_mvp.ingestion.validation import ValidatedUpload
from rag_mvp.ingestion.versioning import (
    DeletedSourceError,
    SourceVersionDisposition,
    SourceVersioningService,
    derivation_config_digest,
)
from rag_mvp.storage.artifacts import ArtifactAlreadyExistsError, ArtifactStore
from rag_mvp.storage.database import Database
from rag_mvp.storage.layout import DataLayout, UnsafeDataPathError
from rag_mvp.storage.repositories import DocumentRepository

NOW = datetime(2026, 8, 6, 10, tzinfo=UTC)


def _config(*, chunk_tokens: int = 500) -> dict[str, object]:
    return {
        "extraction": {"adapter": "text-v1"},
        "ocr": {"adapter": "tesseract-v1", "languages": ["chi_sim", "eng"]},
        "normalization": {"version": "unicode-nfc-lines-v1"},
        "chunking": {"version": "structure-v1", "target_tokens": chunk_tokens},
        "tokenizer": {"version": "unicode-cjk-v1"},
    }


def _upload(content: bytes, *, filename: str = "handbook.txt") -> ValidatedUpload:
    return ValidatedUpload(
        filename=filename,
        media_type="text/plain",
        kind=DocumentKind.TEXT,
        content=content,
    )


def _extracted(text: str) -> ExtractedDocument:
    return ExtractedDocument(
        kind=DocumentKind.TEXT,
        blocks=(
            ExtractedBlock(
                text=text,
                section_path=("Policy",),
                extraction_method=ExtractionMethod.TEXT,
            ),
        ),
    )


@pytest.fixture
def versioning(
    tmp_path: Path,
) -> tuple[SourceVersioningService, DocumentRepository, DataLayout]:
    layout = DataLayout.from_root(tmp_path / "data")
    layout.initialize()
    database = Database(layout.metadata_db)
    database.initialize()
    documents = DocumentRepository(database)
    service = SourceVersioningService(
        database,
        documents,
        ArtifactStore(layout),
        source_id_factory=lambda: "src_fixed",
        clock=lambda: NOW,
    )
    return service, documents, layout


def test_derivation_digest_is_canonical_and_changes_only_for_derivation_inputs() -> None:
    first = _config()
    reordered = dict(reversed(list(first.items())))

    assert derivation_config_digest(first) == derivation_config_digest(reordered)
    assert derivation_config_digest(first) != derivation_config_digest(_config(chunk_tokens=300))
    assert derivation_config_digest(first) == derivation_config_digest(
        {**first, "provider_api_key": "ignored"}
    )
    with pytest.raises(ValueError, match="requires"):
        derivation_config_digest({key: value for key, value in first.items() if key != "ocr"})
    with pytest.raises(ValueError, match="finite"):
        derivation_config_digest({**first, "chunking": {"overlap": float("nan")}})


def test_new_source_creates_inactive_version_one_and_round_trippable_artifacts(
    versioning: tuple[SourceVersioningService, DocumentRepository, DataLayout],
) -> None:
    service, documents, layout = versioning
    source_bytes = "Cafe\u0301 policy\r\n制度".encode()

    result = service.register(
        source_key="employee-handbook",
        upload=_upload(source_bytes, filename="Employee Handbook.TXT"),
        extracted_document=_extracted("Cafe\u0301 policy\r\n制度"),
        derivation_config=_config(),
        display_title="Employee Handbook",
    )

    assert result.disposition is SourceVersionDisposition.CREATED
    assert result.version.version == 1
    assert result.document.active_version is None
    assert documents.list_active() == []
    assert PurePosixPath(result.version.source_artifact_path).is_absolute() is False
    assert "Employee Handbook" not in result.version.source_artifact_path
    assert (
        layout.resolve_artifact_path(result.version.source_artifact_path).read_bytes()
        == source_bytes
    )

    canonical = json.loads(
        layout.resolve_artifact_path(result.version.canonical_artifact_path).read_text(
            encoding="utf-8"
        )
    )
    assert canonical == {
        "blocks": [
            {
                "extraction_method": "text",
                "page_number": None,
                "section_path": ["Policy"],
                "text": "Café policy\n制度",
            }
        ],
        "kind": "text",
        "normalization_version": "unicode-nfc-lines-v1",
        "ocr_page_count": 0,
    }


def test_duplicate_compares_only_active_version_and_changed_input_stays_inactive(
    versioning: tuple[SourceVersioningService, DocumentRepository, DataLayout],
) -> None:
    service, documents, layout = versioning
    original = b"original policy"
    first = service.register(
        source_key="employee-handbook",
        upload=_upload(original),
        extracted_document=_extracted(original.decode()),
        derivation_config=_config(),
    )
    documents.set_active_version(first.document.source_id, 1)
    first_source_path = layout.resolve_artifact_path(first.version.source_artifact_path)
    first_canonical_path = layout.resolve_artifact_path(first.version.canonical_artifact_path)
    first_source_bytes = first_source_path.read_bytes()
    first_canonical_bytes = first_canonical_path.read_bytes()

    duplicate = service.register(
        source_key="employee-handbook",
        upload=_upload(original, filename="renamed.txt"),
        extracted_document=_extracted(original.decode()),
        derivation_config=_config(),
    )

    assert duplicate.disposition is SourceVersionDisposition.DUPLICATE
    assert duplicate.version.version == 1
    assert len(documents.list_versions(first.document.source_id)) == 1

    changed = service.register(
        source_key="employee-handbook",
        upload=_upload(b"changed policy"),
        extracted_document=_extracted("changed policy"),
        derivation_config=_config(),
    )
    config_changed = service.register(
        source_key="employee-handbook",
        upload=_upload(original),
        extracted_document=_extracted(original.decode()),
        derivation_config=_config(chunk_tokens=300),
    )

    assert changed.version.version == 2
    assert config_changed.version.version == 3
    document = documents.get(first.document.source_id)
    assert document is not None
    assert document.active_version == 1
    assert [item.version for item in documents.list_versions(document.source_id)] == [1, 2, 3]
    assert first_source_path.read_bytes() == first_source_bytes
    assert first_canonical_path.read_bytes() == first_canonical_bytes


def test_reverting_historical_content_creates_a_new_sequential_version(
    versioning: tuple[SourceVersioningService, DocumentRepository, DataLayout],
) -> None:
    service, documents, _ = versioning
    first = service.register(
        source_key="policy",
        upload=_upload(b"version one"),
        extracted_document=_extracted("version one"),
        derivation_config=_config(),
    )
    documents.set_active_version(first.document.source_id, 1)
    second = service.register(
        source_key="policy",
        upload=_upload(b"version two"),
        extracted_document=_extracted("version two"),
        derivation_config=_config(),
    )
    documents.set_active_version(first.document.source_id, second.version.version)

    reverted = service.register(
        source_key="policy",
        upload=_upload(b"version one"),
        extracted_document=_extracted("version one"),
        derivation_config=_config(),
    )

    assert reverted.disposition is SourceVersionDisposition.CREATED
    assert reverted.version.version == 3


def test_deleted_source_is_rejected_without_new_version_or_artifacts(
    versioning: tuple[SourceVersioningService, DocumentRepository, DataLayout],
) -> None:
    service, documents, layout = versioning
    first = service.register(
        source_key="deleted-policy",
        upload=_upload(b"one"),
        extracted_document=_extracted("one"),
        derivation_config=_config(),
    )
    documents.set_active_version(first.document.source_id, 1)
    documents.mark_deleted(first.document.source_id)
    paths_before = sorted(
        path.relative_to(layout.root) for path in layout.root.rglob("*") if path.is_file()
    )

    with pytest.raises(DeletedSourceError, match="cannot be restored"):
        service.register(
            source_key="deleted-policy",
            upload=_upload(b"two"),
            extracted_document=_extracted("two"),
            derivation_config=_config(),
        )

    assert len(documents.list_versions(first.document.source_id)) == 1
    paths_after = sorted(
        path.relative_to(layout.root) for path in layout.root.rglob("*") if path.is_file()
    )
    assert paths_after == paths_before


def test_artifact_paths_ignore_filename_directories_and_artifacts_are_immutable(
    tmp_path: Path,
) -> None:
    layout = DataLayout.from_root(tmp_path / "data")
    layout.initialize()
    store = ArtifactStore(layout)
    relative = layout.source_artifact_relative_path("src_safe", 1, "../../secret.txt")

    assert relative == "sources/src_safe/1/source.txt"
    stored = store.write_version(
        source_id="src_safe",
        version=1,
        original_filename="../../secret.txt",
        source_content=b"original",
        canonical_document=_extracted("original"),
    )
    with pytest.raises(ArtifactAlreadyExistsError):
        store.write_version(
            source_id="src_safe",
            version=1,
            original_filename="other.txt",
            source_content=b"replacement",
            canonical_document=_extracted("replacement"),
        )

    assert layout.resolve_artifact_path(stored.source_artifact_path).read_bytes() == b"original"
    with pytest.raises(UnsafeDataPathError):
        layout.source_artifact_relative_path("../escape", 1, "source.txt")
    with pytest.raises(UnsafeDataPathError):
        layout.resolve_artifact_path("sources/../escape.txt")


def test_artifacts_are_cleaned_up_when_metadata_insert_fails(tmp_path: Path) -> None:
    class FailingDocumentRepository(DocumentRepository):
        def add_version(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("database write failed")

    layout = DataLayout.from_root(tmp_path / "data")
    layout.initialize()
    database = Database(layout.metadata_db)
    database.initialize()
    documents = FailingDocumentRepository(database)
    service = SourceVersioningService(
        database,
        documents,
        ArtifactStore(layout),
        source_id_factory=lambda: "src_failed",
        clock=lambda: NOW,
    )

    with pytest.raises(RuntimeError, match="database write failed"):
        service.register(
            source_key="failed-source",
            upload=_upload(b"content"),
            extracted_document=_extracted("content"),
            derivation_config=_config(),
        )

    assert documents.get_by_source_key("failed-source") is None
    assert not layout.source_artifact_path("src_failed", 1, "source.txt").exists()
    assert not layout.canonical_artifact_path("src_failed", 1).exists()
