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
        setup_columns = {
            str(row["name"]): int(row["notnull"])
            for row in connection.execute(
                "PRAGMA table_info('comparison_shared_setup_evidence')"
            ).fetchall()
        }
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
        "comparison_plans",
        "comparison_runs",
        "comparison_candidate_bindings",
        "comparison_candidates",
        "comparison_results",
        "comparison_shared_setup_evidence",
        "comparison_selections",
    }.issubset(tables)
    assert foreign_keys == 1
    assert setup_columns["provider_calls_complete"] == 1
    assert setup_columns["provider_call_count"] == 0


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


def test_latest_migration_preserves_versions_and_removes_historical_digest_uniqueness(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "metadata.sqlite3")
    database.initialize(target_version=2)
    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO documents(source_id, source_key, active_version, deleted_at, payload_json)
            VALUES ('source-1', 'policy', NULL, NULL, '{}')
            """
        )
        connection.execute(
            """
            INSERT INTO document_versions(
                source_id, version, content_digest, derivation_config_digest, payload_json
            ) VALUES ('source-1', 1, 'content-digest', 'config-digest', '{"version": 1}')
            """
        )

    database.initialize()
    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO document_versions(
                source_id, version, content_digest, derivation_config_digest, payload_json
            ) VALUES ('source-1', 2, 'content-digest', 'config-digest', '{"version": 2}')
            """
        )
        rows = connection.execute(
            "SELECT version, payload_json FROM document_versions ORDER BY version"
        ).fetchall()
        index_rows = connection.execute("PRAGMA index_list('document_versions')").fetchall()

    assert [(int(row["version"]), str(row["payload_json"])) for row in rows] == [
        (1, '{"version": 1}'),
        (2, '{"version": 2}'),
    ]
    digest_index = next(row for row in index_rows if row["name"] == "idx_document_versions_digest")
    assert int(digest_index["unique"]) == 0


def test_shared_setup_unavailable_migration_preserves_v4_rows(tmp_path: Path) -> None:
    database = Database(tmp_path / "metadata.sqlite3")
    database.initialize(target_version=4)
    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO comparison_plans(
                comparison_id, plan_id, plan_content_hash, axis, created_at, payload_json
            ) VALUES ('comparison-1', 'plan-1', 'sha256:plan', 'generation-model', 'now', '{}')
            """
        )
        connection.execute(
            """
            INSERT INTO comparison_shared_setup_evidence(
                comparison_id, plan_content_hash, status, provider_call_count,
                known_partial_cost, total_cost, currency, recorded_at,
                evidence_content_hash, payload_json
            ) VALUES (
                'comparison-1', 'sha256:plan', 'failed', 0,
                '0', '0', 'USD', 'now', 'sha256:evidence', '{}'
            )
            """
        )

    database.initialize()

    with database.connection() as connection:
        row = connection.execute(
            """
            SELECT provider_calls_complete, provider_call_count, payload_json
            FROM comparison_shared_setup_evidence
            WHERE comparison_id = 'comparison-1'
            """
        ).fetchone()
    assert row is not None
    assert int(row["provider_calls_complete"]) == 1
    assert int(row["provider_call_count"]) == 0
    assert str(row["payload_json"]) == "{}"


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
