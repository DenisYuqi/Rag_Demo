from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rag_mvp.storage.database import SCHEMA_VERSION, Database, DatabaseVersionError


def test_initialize_creates_current_schema_and_is_idempotent(tmp_path: Path) -> None:
    database = Database(tmp_path / "metadata.sqlite3")

    assert database.initialize() == SCHEMA_VERSION
    assert database.initialize() == SCHEMA_VERSION
    assert database.schema_version() == SCHEMA_VERSION

    with database.connection() as connection:
        tables = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        foreign_keys = int(connection.execute("PRAGMA foreign_keys").fetchone()[0])
    assert {
        "documents",
        "document_versions",
        "ingestion_jobs",
        "index_revisions",
        "active_index_manifest",
        "sessions",
        "session_turns",
        "request_diagnostics",
        "provider_usage",
        "evaluation_runs",
        "report_manifests",
    }.issubset(tables)
    assert foreign_keys == 1


def test_initialize_upgrades_an_older_schema(tmp_path: Path) -> None:
    database = Database(tmp_path / "metadata.sqlite3")

    assert database.initialize(target_version=1) == 1
    with database.connection() as connection:
        index_before = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'index' AND name = 'idx_documents_active'
            """
        ).fetchone()
    assert index_before is None

    assert database.initialize() == SCHEMA_VERSION
    with database.connection() as connection:
        index_after = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'index' AND name = 'idx_documents_active'
            """
        ).fetchone()
    assert index_after is not None


def test_initialize_refuses_downgrade(tmp_path: Path) -> None:
    database = Database(tmp_path / "metadata.sqlite3")
    database.initialize()

    with pytest.raises(DatabaseVersionError):
        database.initialize(target_version=1)


def test_transaction_rolls_back_on_error(tmp_path: Path) -> None:
    database = Database(tmp_path / "metadata.sqlite3")
    database.initialize()

    with pytest.raises(RuntimeError, match="force rollback"), database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO documents(
                source_id, source_key, active_version, deleted_at, payload_json
            ) VALUES ('source-1', 'key-1', NULL, NULL, '{}')
            """
        )
        raise RuntimeError("force rollback")

    with database.connection() as connection:
        count = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
    assert count == 0


def test_newer_unknown_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "metadata.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, 'now')",
        (SCHEMA_VERSION + 1,),
    )
    connection.commit()
    connection.close()

    with pytest.raises(DatabaseVersionError):
        Database(path).initialize()
