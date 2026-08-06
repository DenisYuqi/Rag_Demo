from __future__ import annotations

from pathlib import Path

import pytest

from rag_mvp.storage.layout import DataLayout, UnsafeDataPathError


def test_layout_creates_all_paths_under_root(tmp_path: Path) -> None:
    layout = DataLayout.from_root(tmp_path / "rag-data")
    layout.initialize()

    assert layout.metadata_db.parent == layout.root
    assert layout.active_manifest.parent == layout.root / "indexes"
    assert all(layout.directory(name).is_dir() for name in layout.DIRECTORY_NAMES)


def test_layout_rejects_escape(tmp_path: Path) -> None:
    layout = DataLayout.from_root(tmp_path / "rag-data")

    with pytest.raises(UnsafeDataPathError):
        layout.ensure_within_root(tmp_path / "outside.txt")


def test_unknown_directory_is_rejected(tmp_path: Path) -> None:
    layout = DataLayout.from_root(tmp_path / "rag-data")

    with pytest.raises(UnsafeDataPathError):
        layout.directory("../escape")


def test_revision_paths_are_isolated_and_reject_non_opaque_ids(tmp_path: Path) -> None:
    layout = DataLayout.from_root(tmp_path / "rag-data")

    assert layout.index_revision_path("revision_123") == (
        layout.root / "indexes" / "revisions" / "revision_123"
    )
    assert layout.dense_index_relative_path("revision_123") == (
        "indexes/revisions/revision_123/chroma"
    )
    assert layout.lexical_index_relative_path("revision_123") == (
        "indexes/revisions/revision_123/bm25.json"
    )
    with pytest.raises(UnsafeDataPathError):
        layout.index_revision_path("../escape")


def test_filesystem_root_is_rejected() -> None:
    with pytest.raises(UnsafeDataPathError):
        DataLayout.from_root(Path("/"))
