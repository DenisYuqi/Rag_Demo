"""Immutable source and canonical-document artifact storage."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import TypeAdapter, ValidationError

from rag_mvp.domain.ingestion import DocumentVersion, IngestionCommand, UploadCommand
from rag_mvp.storage.layout import DataLayout

if TYPE_CHECKING:
    from rag_mvp.ingestion.extractors import ExtractedDocument

_COMMAND_ADAPTER: TypeAdapter[IngestionCommand] = TypeAdapter(IngestionCommand)


class ArtifactStoreError(RuntimeError):
    """Base class for artifact persistence failures."""


class ArtifactAlreadyExistsError(ArtifactStoreError):
    """Raised rather than replacing an immutable version artifact."""


class ArtifactNotFoundError(ArtifactStoreError):
    """Raised when a required durable artifact is absent."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ArtifactCorruptError(ArtifactStoreError):
    """Raised when a durable artifact fails deterministic validation."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class StoredVersionArtifacts:
    source_artifact_path: str
    canonical_artifact_path: str


def canonical_document_json(document: ExtractedDocument) -> bytes:
    """Serialize every ``ExtractedDocument`` field deterministically."""
    payload = {
        "blocks": [
            {
                "extraction_method": block.extraction_method.value,
                "page_number": block.page_number,
                "section_path": list(block.section_path),
                "text": block.text,
            }
            for block in document.blocks
        ],
        "kind": document.kind.value,
        "normalization_version": document.normalization_version,
        "ocr_page_count": document.ocr_page_count,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return serialized.encode("utf-8")


class ArtifactStore:
    """Create immutable artifacts using temporary files on the target filesystem."""

    def __init__(self, layout: DataLayout) -> None:
        self._layout = layout

    def write_version(
        self,
        *,
        source_id: str,
        version: int,
        original_filename: str,
        source_content: bytes,
        canonical_document: ExtractedDocument,
    ) -> StoredVersionArtifacts:
        source_relative = self._layout.source_artifact_relative_path(
            source_id,
            version,
            original_filename,
        )
        canonical_relative = self._layout.canonical_artifact_relative_path(source_id, version)
        source_path = self._layout.resolve_artifact_path(source_relative)
        canonical_path = self._layout.resolve_artifact_path(canonical_relative)
        created: list[Path] = []
        try:
            self._atomic_create(source_path, source_content)
            created.append(source_path)
            self._atomic_create(canonical_path, canonical_document_json(canonical_document))
            created.append(canonical_path)
        except Exception:
            self._cleanup_paths(created)
            raise
        return StoredVersionArtifacts(
            source_artifact_path=source_relative,
            canonical_artifact_path=canonical_relative,
        )

    def write_command(
        self,
        command: IngestionCommand,
        *,
        upload_content: bytes | None = None,
    ) -> None:
        job_path = self._layout.job_path(command.job_id)
        if job_path.exists():
            raise ArtifactAlreadyExistsError("job artifacts already exist")
        if isinstance(command, UploadCommand) != (upload_content is not None):
            raise ArtifactStoreError("command_upload_mismatch")
        jobs_path = self._layout.directory("jobs")
        jobs_path.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{command.job_id}.", dir=jobs_path))
        try:
            if upload_content is not None:
                self._atomic_create(temporary / "upload.bin", upload_content)
            payload = _COMMAND_ADAPTER.dump_json(command)
            self._atomic_create(temporary / "command.json", payload)
            os.replace(temporary, job_path)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def load_command(self, job_id: str) -> IngestionCommand:
        path = self._layout.job_command_path(job_id)
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            raise ArtifactNotFoundError("command_missing") from None
        except OSError:
            raise ArtifactCorruptError("command_unreadable") from None
        try:
            command: IngestionCommand = _COMMAND_ADAPTER.validate_json(payload)
        except ValidationError:
            raise ArtifactCorruptError("command_corrupt") from None
        if command.job_id != job_id:
            raise ArtifactCorruptError("command_identity_mismatch")
        return command

    def load_upload(self, job_id: str, *, expected_size: int, expected_digest: str) -> bytes:
        path = self._layout.job_upload_path(job_id)
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            raise ArtifactNotFoundError("upload_missing") from None
        except OSError:
            raise ArtifactCorruptError("upload_unreadable") from None
        if len(content) != expected_size or hashlib.sha256(content).hexdigest() != expected_digest:
            raise ArtifactCorruptError("upload_corrupt")
        return content

    def load_source(self, version: DocumentVersion) -> bytes:
        expected = self._layout.source_artifact_relative_path(
            version.source_id,
            version.version,
            version.original_filename,
        )
        if version.source_artifact_path != expected:
            raise ArtifactCorruptError("source_artifact_path_mismatch")
        path = self._layout.resolve_artifact_path(expected)
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            raise ArtifactNotFoundError("source_artifact_missing") from None
        except OSError:
            raise ArtifactCorruptError("source_artifact_unreadable") from None
        if len(content) != version.size_bytes:
            raise ArtifactCorruptError("source_artifact_size_mismatch")
        return content

    def load_canonical(self, version: DocumentVersion) -> ExtractedDocument:
        from rag_mvp.domain.ingestion import DocumentKind, ExtractionMethod
        from rag_mvp.ingestion.extractors import ExtractedBlock, ExtractedDocument
        from rag_mvp.ingestion.normalization import canonical_document_digest

        expected = self._layout.canonical_artifact_relative_path(
            version.source_id,
            version.version,
        )
        if version.canonical_artifact_path != expected:
            raise ArtifactCorruptError("canonical_artifact_path_mismatch")
        path = self._layout.resolve_artifact_path(expected)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            expected_keys = {
                "blocks",
                "kind",
                "normalization_version",
                "ocr_page_count",
            }
            if (
                not isinstance(raw, dict)
                or set(raw) != expected_keys
                or not isinstance(raw["blocks"], list)
                or not raw["blocks"]
                or type(raw["ocr_page_count"]) is not int
                or raw["ocr_page_count"] < 0
                or not (
                    raw["normalization_version"] is None
                    or isinstance(raw["normalization_version"], str)
                )
            ):
                raise ValueError
            blocks: list[ExtractedBlock] = []
            for block in raw["blocks"]:
                if (
                    not isinstance(block, dict)
                    or set(block) != {"extraction_method", "page_number", "section_path", "text"}
                    or not isinstance(block["text"], str)
                    or not block["text"]
                    or not isinstance(block["section_path"], list)
                    or any(not isinstance(part, str) for part in block["section_path"])
                    or not (
                        block["page_number"] is None
                        or (type(block["page_number"]) is int and block["page_number"] > 0)
                    )
                ):
                    raise ValueError
                blocks.append(
                    ExtractedBlock(
                        text=block["text"],
                        page_number=block["page_number"],
                        section_path=tuple(block["section_path"]),
                        extraction_method=ExtractionMethod(block["extraction_method"]),
                    )
                )
            document = ExtractedDocument(
                kind=DocumentKind(raw["kind"]),
                blocks=tuple(blocks),
                ocr_page_count=raw["ocr_page_count"],
                normalization_version=raw["normalization_version"],
            )
        except FileNotFoundError:
            raise ArtifactNotFoundError("canonical_artifact_missing") from None
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            raise ArtifactCorruptError("canonical_artifact_corrupt") from None
        if canonical_document_digest(document) != version.content_digest:
            raise ArtifactCorruptError("canonical_artifact_digest_mismatch")
        return document

    def delete_command(self, job_id: str) -> None:
        job_path = self._layout.job_path(job_id)
        if not job_path.exists():
            return
        for path in (self._layout.job_command_path(job_id), self._layout.job_upload_path(job_id)):
            path.unlink(missing_ok=True)
        with suppress(OSError):
            job_path.rmdir()

    def list_command_job_ids(self) -> tuple[str, ...]:
        jobs_path = self._layout.directory("jobs")
        job_ids: list[str] = []
        for path in jobs_path.iterdir():
            if not path.is_dir():
                continue
            try:
                if self._layout.job_path(path.name) == path.resolve():
                    job_ids.append(path.name)
            except ValueError:
                continue
        return tuple(sorted(job_ids))

    def cleanup(self, artifacts: StoredVersionArtifacts) -> None:
        """Remove artifacts created for metadata that failed to commit."""
        self._cleanup_paths(
            [
                self._layout.resolve_artifact_path(artifacts.source_artifact_path),
                self._layout.resolve_artifact_path(artifacts.canonical_artifact_path),
            ]
        )

    @staticmethod
    def _atomic_create(path: Path, content: bytes) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if path.exists():
            raise ArtifactAlreadyExistsError(f"artifact already exists: {path.name}")

        descriptor, raw_temporary = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(raw_temporary)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            if path.exists():
                raise ArtifactAlreadyExistsError(f"artifact already exists: {path.name}")
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _cleanup_paths(self, paths: list[Path]) -> None:
        for path in reversed(paths):
            path.unlink(missing_ok=True)
            parent = path.parent
            while parent.parent != self._layout.root:
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
