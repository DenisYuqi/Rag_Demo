"""Safe, configurable on-disk artifact layout."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


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
        "locks",
        "tmp",
    )

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

    def initialize(self) -> None:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        for name in self.DIRECTORY_NAMES:
            (self.root / name).mkdir(mode=0o700, exist_ok=True)
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
