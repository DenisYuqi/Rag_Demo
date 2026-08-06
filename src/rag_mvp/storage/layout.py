"""Safe, configurable on-disk artifact layout."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class UnsafeDataPathError(ValueError):
    """Raised when a path escapes the configured data root."""


@dataclass(frozen=True, slots=True)
class DataLayout:
    """All persistent and temporary application paths below one root."""

    root: Path

    DIRECTORY_NAMES = (
        "sources",
        "canonical",
        "indexes",
        "caches",
        "reports",
        "evaluations",
        "jobs",
        "locks",
        "tmp",
    )
    _OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$")
    _ARTIFACT_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")
    _SOURCE_EXTENSIONS = frozenset({".pdf", ".md", ".markdown", ".txt"})

    @classmethod
    def from_root(cls, root: Path | str) -> DataLayout:
        resolved = Path(root).expanduser().resolve()
        if resolved == Path(resolved.anchor):
            raise UnsafeDataPathError("data root cannot be a filesystem root")
        return cls(root=resolved)

    @property
    def metadata_db(self) -> Path:
        return self.root / "metadata.sqlite3"

    @property
    def active_manifest(self) -> Path:
        return self.root / "indexes" / "active.json"

    @property
    def index_revisions(self) -> Path:
        return self.root / "indexes" / "revisions"

    @property
    def writer_lock(self) -> Path:
        return self.root / "locks" / "writer.lock"

    def directory(self, name: str) -> Path:
        if name not in self.DIRECTORY_NAMES:
            raise UnsafeDataPathError("unknown data directory")
        return self.root / name

    def ensure_within_root(self, candidate: Path | str) -> Path:
        resolved = Path(candidate).expanduser().resolve()
        if not resolved.is_relative_to(self.root):
            raise UnsafeDataPathError("path escapes configured data root")
        return resolved

    def resolve_artifact_path(self, relative_path: str) -> Path:
        """Resolve a restricted POSIX artifact path below the configured root."""
        if not relative_path or "\\" in relative_path:
            raise UnsafeDataPathError("artifact path must be a relative POSIX path")
        pure = PurePosixPath(relative_path)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise UnsafeDataPathError("artifact path must be a relative POSIX path")
        if any(self._ARTIFACT_COMPONENT.fullmatch(part) is None for part in pure.parts):
            raise UnsafeDataPathError("artifact path contains an unsafe component")
        return self.ensure_within_root(self.root.joinpath(*pure.parts))

    def source_artifact_relative_path(
        self,
        source_id: str,
        version: int,
        original_filename: str,
    ) -> str:
        source = self._validated_source_id(source_id)
        number = self._validated_version(version)
        extension = PurePosixPath(original_filename.replace("\\", "/")).suffix.casefold()
        if extension not in self._SOURCE_EXTENSIONS:
            raise UnsafeDataPathError("source filename has an unsupported extension")
        return PurePosixPath("sources", source, str(number), f"source{extension}").as_posix()

    def canonical_artifact_relative_path(self, source_id: str, version: int) -> str:
        source = self._validated_source_id(source_id)
        number = self._validated_version(version)
        return PurePosixPath("canonical", source, str(number), "document.json").as_posix()

    def source_artifact_path(
        self,
        source_id: str,
        version: int,
        original_filename: str,
    ) -> Path:
        return self.resolve_artifact_path(
            self.source_artifact_relative_path(source_id, version, original_filename)
        )

    def canonical_artifact_path(self, source_id: str, version: int) -> Path:
        return self.resolve_artifact_path(self.canonical_artifact_relative_path(source_id, version))

    def index_revision_relative_path(self, revision_id: str) -> str:
        revision = self._validated_opaque_id(revision_id, "revision_id")
        return PurePosixPath("indexes", "revisions", revision).as_posix()

    def dense_index_relative_path(self, revision_id: str) -> str:
        return PurePosixPath(self.index_revision_relative_path(revision_id), "chroma").as_posix()

    def lexical_index_relative_path(self, revision_id: str) -> str:
        return PurePosixPath(
            self.index_revision_relative_path(revision_id),
            "bm25.json",
        ).as_posix()

    def index_revision_path(self, revision_id: str) -> Path:
        return self.resolve_artifact_path(self.index_revision_relative_path(revision_id))

    def dense_index_path(self, revision_id: str) -> Path:
        return self.resolve_artifact_path(self.dense_index_relative_path(revision_id))

    def lexical_index_path(self, revision_id: str) -> Path:
        return self.resolve_artifact_path(self.lexical_index_relative_path(revision_id))

    def job_relative_path(self, job_id: str) -> str:
        job = self._validated_opaque_id(job_id, "job_id")
        return PurePosixPath("jobs", job).as_posix()

    def job_command_relative_path(self, job_id: str) -> str:
        return PurePosixPath(self.job_relative_path(job_id), "command.json").as_posix()

    def job_upload_relative_path(self, job_id: str) -> str:
        return PurePosixPath(self.job_relative_path(job_id), "upload.bin").as_posix()

    def job_path(self, job_id: str) -> Path:
        return self.resolve_artifact_path(self.job_relative_path(job_id))

    def job_command_path(self, job_id: str) -> Path:
        return self.resolve_artifact_path(self.job_command_relative_path(job_id))

    def job_upload_path(self, job_id: str) -> Path:
        return self.resolve_artifact_path(self.job_upload_relative_path(job_id))

    @classmethod
    def _validated_source_id(cls, source_id: str) -> str:
        return cls._validated_opaque_id(source_id, "source_id")

    @classmethod
    def _validated_opaque_id(cls, value: str, field_name: str) -> str:
        if not isinstance(value, str) or cls._OPAQUE_ID.fullmatch(value) is None:
            raise UnsafeDataPathError(f"{field_name} must be an opaque identifier")
        return value

    @staticmethod
    def _validated_version(version: int) -> int:
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise UnsafeDataPathError("document version must be a positive integer")
        return version

    def initialize(self) -> None:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        for name in self.DIRECTORY_NAMES:
            (self.root / name).mkdir(mode=0o700, exist_ok=True)
        self.index_revisions.mkdir(mode=0o700, exist_ok=True)
        self.assert_writable()

    def assert_writable(self) -> None:
        """Perform an actual create/fsync/delete probe inside the data volume."""
        if not self.root.is_dir():
            raise OSError("data root is not a directory")
        descriptor, raw_path = tempfile.mkstemp(prefix=".write-probe-", dir=self.root)
        probe = Path(raw_path)
        try:
            os.write(descriptor, b"ready")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
            probe.unlink(missing_ok=True)
