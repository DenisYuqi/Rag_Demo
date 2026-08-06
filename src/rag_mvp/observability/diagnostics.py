"""Persisted, redacted, and retention-bounded request diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

from rag_mvp.domain._base import SafeScalar
from rag_mvp.domain.qa import RequestDiagnostic
from rag_mvp.observability.logging import is_safe_identifier
from rag_mvp.safety.redactor import DEFAULT_REDACTOR, RedactionError, Redactor
from rag_mvp.storage.database import Database
from rag_mvp.storage.repositories import RequestDiagnosticRepository

_OUTCOMES: Final[frozenset[str]] = frozenset({"answer", "refusal", "error", "cancelled", "timeout"})
_STAGES: Final[frozenset[str]] = frozenset(
    {
        "queue",
        "validation",
        "safety",
        "query_embedding",
        "embedding",
        "retrieval",
        "dense",
        "bm25",
        "fusion",
        "rerank",
        "reranking",
        "evidence_assessment",
        "generation",
        "grounding",
        "redaction",
        "finalization",
        "serialization",
        "ingestion",
        "evaluation",
        "total",
    }
)
_CACHE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "document-embedding",
        "query-embedding",
        "query_embedding",
        "retrieval",
        "rerank",
        "reranking",
        "answer",
        "final",
    }
)
_CACHE_OUTCOMES: Final[frozenset[str]] = frozenset(
    {"hit", "miss", "bypass", "disabled", "not-applicable", "error"}
)
_MODEL_ROLES: Final[frozenset[str]] = frozenset(
    {
        "embedding",
        "generation",
        "reranking",
        "reranker",
        "evaluation",
        "dense",
        "bm25",
        "rrf",
    }
)
_TOKEN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "input",
        "output",
        "total",
        "embedding-input",
        "generation-input",
        "generation-output",
        "reranking-input",
        "reranking-output",
        "evaluation-input",
        "evaluation-output",
    }
)
_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "candidate_count",
        "context_count",
        "dense_candidate_count",
        "lexical_candidate_count",
        "fused_candidate_count",
        "reranked_candidate_count",
        "citation_count",
        "configuration_id",
        "currency",
        "estimated_cost",
        "index_revision",
        "redaction_count",
        "retrieval_mode",
        "requested_mode",
        "effective_mode",
        "generation_attempts",
        "generation_fallback",
        "decision_code",
        "refusal_policy_version",
        "input_policy",
    }
)


class DiagnosticSafetyError(RuntimeError):
    """Raised when a diagnostic cannot be proven safe for persistence."""


@dataclass(frozen=True, slots=True)
class DiagnosticRetention:
    max_entries: int = 1_000
    ttl: timedelta = timedelta(hours=24)

    def __post_init__(self) -> None:
        if isinstance(self.max_entries, bool) or self.max_entries <= 0:
            raise ValueError("diagnostic max_entries must be positive")
        if self.ttl <= timedelta(0) or self.ttl > timedelta(days=30):
            raise ValueError("diagnostic TTL must be between zero and 30 days")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _safe_id(value: str, field: str) -> str:
    if not is_safe_identifier(value):
        raise DiagnosticSafetyError(f"{field} is not a bounded opaque identifier")
    return value


def _redact_bounded(value: str, redactor: Redactor, field: str) -> str:
    try:
        redacted = redactor.redact(value).redacted_text
    except RedactionError as error:
        raise DiagnosticSafetyError(f"{field} could not be safely redacted") from error
    if len(redacted) > 255:
        raise DiagnosticSafetyError(f"{field} exceeds the safe diagnostic bound")
    return redacted


def _safe_scalar(value: SafeScalar, redactor: Redactor, field: str) -> SafeScalar:
    if isinstance(value, str):
        return _redact_bounded(value, redactor, field)
    if isinstance(value, float) and not math.isfinite(value):
        raise DiagnosticSafetyError(f"{field} must be finite")
    return value


class SafeRequestDiagnosticStore:
    """Store only content-minimized diagnostics and enforce TTL/count retention."""

    def __init__(
        self,
        database: Database,
        *,
        redactor: Redactor | None = DEFAULT_REDACTOR,
        retention: DiagnosticRetention | None = None,
    ) -> None:
        self._database = database
        self._repository = RequestDiagnosticRepository(database)
        self._redactor = redactor
        self._retention = retention or DiagnosticRetention()

    def save(
        self,
        diagnostic: RequestDiagnostic,
        *,
        now: datetime | None = None,
    ) -> RequestDiagnostic:
        """Redact and atomically save one diagnostic while pruning retained rows."""

        timestamp = now or _utc_now()
        if timestamp.tzinfo is None:
            raise ValueError("diagnostic retention time must be timezone-aware")
        timestamp = timestamp.astimezone(UTC)
        maximum_expiry = timestamp + self._retention.ttl
        expires_at = diagnostic.expires_at
        if expires_at is not None:
            expires_at = expires_at.astimezone(UTC)
        if expires_at is None or expires_at > maximum_expiry:
            expires_at = maximum_expiry
        safe = self._sanitize(diagnostic, expires_at=expires_at)

        with self._database.transaction() as connection:
            existing = connection.execute(
                "SELECT 1 FROM request_diagnostics WHERE request_id = ?",
                (safe.request_id,),
            ).fetchone()
            if existing is not None:
                raise DiagnosticSafetyError("request_id has already been persisted")
            self._repository.save(safe, connection=connection)
            connection.execute(
                """
                DELETE FROM request_diagnostics
                WHERE expires_at IS NOT NULL AND expires_at <= ?
                """,
                (timestamp.isoformat(),),
            )
            connection.execute(
                """
                DELETE FROM request_diagnostics
                WHERE request_id IN (
                    SELECT request_id FROM request_diagnostics
                    ORDER BY created_at DESC, request_id DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (self._retention.max_entries,),
            )
        return safe

    def get(
        self,
        request_id: str,
        *,
        now: datetime | None = None,
    ) -> RequestDiagnostic | None:
        """Look up a recent request by an opaque ID and revalidate before release."""

        if not is_safe_identifier(request_id):
            return None
        timestamp = now
        if timestamp is not None:
            if timestamp.tzinfo is None:
                raise ValueError("diagnostic lookup time must be timezone-aware")
            timestamp = timestamp.astimezone(UTC)
        diagnostic = self._repository.get(request_id, now=timestamp)
        if diagnostic is None:
            return None
        return self._sanitize(diagnostic, expires_at=diagnostic.expires_at)

    def purge(self, *, now: datetime | None = None) -> int:
        """Delete expired rows and any rows above the configured count bound."""

        timestamp = now or _utc_now()
        if timestamp.tzinfo is None:
            raise ValueError("diagnostic retention time must be timezone-aware")
        timestamp = timestamp.astimezone(UTC)
        with self._database.transaction() as connection:
            expired = connection.execute(
                """
                DELETE FROM request_diagnostics
                WHERE expires_at IS NOT NULL AND expires_at <= ?
                """,
                (timestamp.isoformat(),),
            ).rowcount
            overflow = connection.execute(
                """
                DELETE FROM request_diagnostics
                WHERE request_id IN (
                    SELECT request_id FROM request_diagnostics
                    ORDER BY created_at DESC, request_id DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (self._retention.max_entries,),
            ).rowcount
        return expired + overflow

    def count(self) -> int:
        """Return the content-free number of retained diagnostic records."""

        with self._database.connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS diagnostic_count FROM request_diagnostics"
            ).fetchone()
        return 0 if row is None else int(row["diagnostic_count"])

    def _sanitize(
        self,
        diagnostic: RequestDiagnostic,
        *,
        expires_at: datetime | None,
    ) -> RequestDiagnostic:
        redactor = self._redactor
        if redactor is None or not redactor.fully_configured:
            raise DiagnosticSafetyError("diagnostic redaction is unavailable")
        request_id = _safe_id(diagnostic.request_id, "request_id")
        session_id = (
            _safe_id(diagnostic.session_id, "session_id")
            if diagnostic.session_id is not None
            else None
        )
        trace_id = (
            _safe_id(diagnostic.trace_id, "trace_id") if diagnostic.trace_id is not None else None
        )
        if diagnostic.outcome not in _OUTCOMES:
            raise DiagnosticSafetyError("diagnostic outcome is not allowlisted")
        error_category = diagnostic.safe_error_category
        if error_category is not None:
            error_category = _safe_id(error_category, "safe_error_category")

        stage_timings = {
            key: value for key, value in diagnostic.stage_timings_ms.items() if key in _STAGES
        }
        cache_status = {
            key: value
            for key, value in diagnostic.cache_status.items()
            if key in _CACHE_NAMES and value in _CACHE_OUTCOMES
        }
        model_identities = {
            key: _redact_bounded(value, redactor, f"model_identities.{key}")
            for key, value in diagnostic.model_identities.items()
            if key in _MODEL_ROLES
        }
        token_counts = {
            key: value for key, value in diagnostic.token_counts.items() if key in _TOKEN_KEYS
        }
        metadata = {
            key: _safe_scalar(value, redactor, f"metadata.{key}")
            for key, value in diagnostic.metadata.items()
            if key in _METADATA_KEYS
        }
        return RequestDiagnostic(
            request_id=request_id,
            session_id=session_id,
            trace_id=trace_id,
            outcome=diagnostic.outcome,
            safe_error_category=error_category,
            stage_timings_ms=stage_timings,
            cache_status=cache_status,
            model_identities=model_identities,
            token_counts=token_counts,
            metadata=metadata,
            created_at=diagnostic.created_at.astimezone(UTC),
            expires_at=expires_at,
        )
