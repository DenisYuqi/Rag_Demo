"""SQLite initialization, migrations, and transaction boundaries."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class DatabaseVersionError(RuntimeError):
    """Raised when an on-disk schema cannot be used by this application."""


_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE documents (
        source_id TEXT PRIMARY KEY,
        source_key TEXT NOT NULL UNIQUE,
        active_version INTEGER,
        deleted_at TEXT,
        payload_json TEXT NOT NULL,
        CHECK (active_version IS NULL OR active_version > 0)
    );

    CREATE TABLE document_versions (
        source_id TEXT NOT NULL,
        version INTEGER NOT NULL CHECK (version > 0),
        content_digest TEXT NOT NULL,
        derivation_config_digest TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        PRIMARY KEY (source_id, version),
        UNIQUE (source_id, content_digest, derivation_config_digest),
        FOREIGN KEY (source_id) REFERENCES documents(source_id) ON DELETE RESTRICT
    );

    CREATE TABLE ingestion_jobs (
        job_id TEXT PRIMARY KEY,
        source_key TEXT NOT NULL,
        source_id TEXT,
        status TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        FOREIGN KEY (source_id) REFERENCES documents(source_id) ON DELETE SET NULL
    );

    CREATE TABLE index_revisions (
        revision_id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        payload_json TEXT NOT NULL
    );

    CREATE TABLE active_index_manifest (
        singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
        revision_id TEXT NOT NULL UNIQUE,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (revision_id) REFERENCES index_revisions(revision_id) ON DELETE RESTRICT
    );

    CREATE TABLE sessions (
        session_id TEXT PRIMARY KEY,
        owner_id TEXT NOT NULL,
        status TEXT NOT NULL,
        payload_json TEXT NOT NULL
    );

    CREATE TABLE session_turns (
        turn_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
        payload_json TEXT NOT NULL,
        UNIQUE (session_id, ordinal),
        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    );

    CREATE TABLE request_diagnostics (
        request_id TEXT PRIMARY KEY,
        session_id TEXT,
        created_at TEXT NOT NULL,
        expires_at TEXT,
        payload_json TEXT NOT NULL
    );

    CREATE TABLE provider_usage (
        attempt_id TEXT PRIMARY KEY,
        request_id TEXT,
        run_id TEXT,
        created_at TEXT NOT NULL,
        payload_json TEXT NOT NULL
    );

    CREATE TABLE evaluation_runs (
        run_id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        payload_json TEXT NOT NULL
    );

    CREATE TABLE report_manifests (
        run_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        FOREIGN KEY (run_id) REFERENCES evaluation_runs(run_id) ON DELETE CASCADE
    );
    """,
    """
    CREATE INDEX idx_documents_active ON documents(deleted_at, source_id);
    CREATE INDEX idx_document_versions_digest
        ON document_versions(content_digest, derivation_config_digest);
    CREATE INDEX idx_ingestion_jobs_source_status
        ON ingestion_jobs(source_key, status);
    CREATE INDEX idx_index_revisions_status ON index_revisions(status);
    CREATE INDEX idx_sessions_owner_status ON sessions(owner_id, status);
    CREATE INDEX idx_session_turns_session_ordinal
        ON session_turns(session_id, ordinal);
    CREATE INDEX idx_request_diagnostics_expiry ON request_diagnostics(expires_at);
    CREATE INDEX idx_provider_usage_request ON provider_usage(request_id, created_at);
    CREATE INDEX idx_provider_usage_run ON provider_usage(run_id, created_at);
    CREATE INDEX idx_evaluation_runs_status ON evaluation_runs(status, created_at);
    """,
)

SCHEMA_VERSION = len(_MIGRATIONS)


class Database:
    """A small connection factory with deterministic, versioned migrations."""

    def __init__(self, path: Path, *, busy_timeout_ms: int = 5_000) -> None:
        self.path = path
        self.busy_timeout_ms = busy_timeout_ms

    def connect(self) -> sqlite3.Connection:
        """Open one configured connection; callers own the returned connection."""

        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1_000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a read connection and always close it."""

        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        """Yield an atomic write transaction, rolling back on every exception."""

        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()

    def initialize(self, *, target_version: int | None = None) -> int:
        """Create or upgrade the database to ``target_version`` atomically per migration."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        requested = SCHEMA_VERSION if target_version is None else target_version
        if requested < 0 or requested > SCHEMA_VERSION:
            raise DatabaseVersionError(
                f"target schema version {requested} is outside 0..{SCHEMA_VERSION}"
            )

        with self.connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            current = self._schema_version(connection)
            if current > SCHEMA_VERSION:
                raise DatabaseVersionError(
                    f"database schema {current} is newer than supported {SCHEMA_VERSION}"
                )
            if requested < current:
                raise DatabaseVersionError(
                    f"refusing to downgrade schema from {current} to {requested}"
                )

            for version in range(current + 1, requested + 1):
                migration = _MIGRATIONS[version - 1]
                script = (
                    "BEGIN IMMEDIATE;\n"
                    f"{migration}\n"
                    "INSERT INTO schema_migrations(version, applied_at) "
                    f"VALUES ({version}, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));\n"
                    "COMMIT;"
                )
                try:
                    connection.executescript(script)
                except sqlite3.Error:
                    if connection.in_transaction:
                        connection.rollback()
                    raise
        return requested

    def schema_version(self) -> int:
        """Return the latest applied migration, treating a new file as version zero."""

        if not self.path.exists():
            return 0
        with self.connection() as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
            ).fetchone()
            if exists is None:
                return 0
            return self._schema_version(connection)

    @staticmethod
    def _schema_version(connection: sqlite3.Connection) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
        ).fetchone()
        if row is None:
            return 0
        return int(row["version"])
