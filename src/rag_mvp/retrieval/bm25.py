"""Persistent, immutable, and self-validating BM25 snapshots."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from rag_mvp.domain.ingestion import Chunk
from rag_mvp.domain.retrieval import RetrievalCandidate
from rag_mvp.performance.worker_pools import BoundedWorkerPool, default_worker_pools
from rag_mvp.retrieval.snapshot import (
    RECORD_DIGEST_ALGORITHM,
    chunk_record_digest,
    chunk_set_digest,
)
from rag_mvp.retrieval.tokenizer import (
    BILINGUAL_TOKENIZER_IDENTITY,
    BilingualTokenizer,
)


class LexicalIndexError(ValueError):
    """A safe lexical snapshot validation or persistence error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def default_bm25_worker_pool() -> BoundedWorkerPool:
    """Return the bounded process-level fallback for direct snapshot searches."""

    return default_worker_pools().bm25


@dataclass(frozen=True, slots=True)
class LexicalRecord:
    chunk: Chunk
    display_title: str
    tokens: tuple[str, ...]
    record_digest: str


class PersistentBm25Index:
    """An immutable revision-specific lexical index backed by one JSON file."""

    SNAPSHOT_SCHEMA = "bm25-snapshot-v4"
    ALGORITHM_VERSION = "bm25-okapi-v1"
    DEFAULT_REVISION_ID = "standalone"
    DEFAULT_K1 = 1.5
    DEFAULT_B = 0.75

    def __init__(
        self,
        records: tuple[LexicalRecord, ...],
        *,
        revision_id: str = DEFAULT_REVISION_ID,
        tokenizer: BilingualTokenizer | None = None,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        algorithm_version: str = ALGORITHM_VERSION,
        worker_pool: BoundedWorkerPool | None = None,
    ) -> None:
        if not isinstance(revision_id, str) or not revision_id:
            raise LexicalIndexError("revision_id_invalid")
        if algorithm_version != self.ALGORITHM_VERSION:
            raise LexicalIndexError("unsupported_algorithm_version")
        normalized_k1 = _finite_number(k1)
        normalized_b = _finite_number(b)
        if normalized_k1 <= 0 or not 0 <= normalized_b <= 1:
            raise LexicalIndexError("invalid_bm25_parameters")
        if len({record.chunk.chunk_id for record in records}) != len(records):
            raise LexicalIndexError("duplicate_chunk_id")

        self.records = records
        self.revision_id = revision_id
        self.tokenizer = tokenizer or BilingualTokenizer()
        self.k1 = normalized_k1
        self.b = normalized_b
        self.algorithm_version = algorithm_version
        self._worker_pool = worker_pool or default_bm25_worker_pool()
        self._record_digests = {record.chunk.chunk_id: record.record_digest for record in records}
        if any(
            chunk_record_digest(record.chunk, record.display_title) != record.record_digest
            for record in records
        ):
            raise LexicalIndexError("record_digest_mismatch")
        self._chunk_set_digest = chunk_set_digest(self._record_digests)
        self._document_frequencies: Counter[str] = Counter()
        for record in records:
            self._document_frequencies.update(set(record.tokens))
        self._average_length = (
            sum(len(record.tokens) for record in records) / len(records) if records else 0.0
        )

    @property
    def chunk_ids(self) -> frozenset[str]:
        return frozenset(self._record_digests)

    @property
    def record_digests(self) -> dict[str, str]:
        return dict(self._record_digests)

    @property
    def chunk_set_digest(self) -> str:
        return self._chunk_set_digest

    @property
    def tokenizer_identity(self) -> str:
        return self.tokenizer.version

    @property
    def algorithm_config(self) -> dict[str, float]:
        return {"k1": self.k1, "b": self.b}

    @classmethod
    def build(
        cls,
        chunks: Sequence[Chunk],
        titles: Mapping[str, str],
        *,
        revision_id: str = DEFAULT_REVISION_ID,
        tokenizer: BilingualTokenizer | None = None,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> PersistentBm25Index:
        resolved = tokenizer or BilingualTokenizer()
        try:
            records = tuple(
                LexicalRecord(
                    chunk=chunk,
                    display_title=titles[chunk.source_id],
                    tokens=resolved.tokenize(chunk.text),
                    record_digest=chunk_record_digest(chunk, titles[chunk.source_id]),
                )
                for chunk in sorted(chunks, key=lambda item: item.chunk_id)
            )
        except KeyError:
            raise LexicalIndexError("missing_display_title") from None
        return cls(
            records,
            revision_id=revision_id,
            tokenizer=resolved,
            k1=k1,
            b=b,
        )

    def _score(self, query_tokens: tuple[str, ...], record: LexicalRecord) -> float:
        if not self.records or not record.tokens:
            return 0.0
        frequencies = Counter(record.tokens)
        score = 0.0
        document_count = len(self.records)
        length_ratio = len(record.tokens) / self._average_length if self._average_length else 0.0
        for token in query_tokens:
            frequency = frequencies[token]
            if not frequency:
                continue
            document_frequency = self._document_frequencies[token]
            inverse_document_frequency = math.log(
                1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
            score += inverse_document_frequency * frequency * (self.k1 + 1) / denominator
        return score

    async def search(self, query: str, limit: int) -> tuple[RetrievalCandidate, ...]:
        return await self._worker_pool.run_cancel_safe(
            self.search_sync,
            query,
            limit,
        )

    def configure_worker_pool(self, worker_pool: BoundedWorkerPool) -> None:
        """Bind this request-scoped snapshot to its application-owned pool."""

        if not isinstance(worker_pool, BoundedWorkerPool):
            raise TypeError("worker_pool must be BoundedWorkerPool")
        self._worker_pool = worker_pool

    def search_sync(self, query: str, limit: int) -> tuple[RetrievalCandidate, ...]:
        """Score one query synchronously; async callers must dispatch through a pool."""

        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be positive")
        query_tokens = self.tokenizer.tokenize(query)
        ranked = [(self._score(query_tokens, record), record) for record in self.records]
        ranked = [entry for entry in ranked if entry[0] > 0]
        ranked.sort(key=lambda entry: (-entry[0], entry[1].chunk.chunk_id))
        return tuple(
            RetrievalCandidate(
                chunk_id=record.chunk.chunk_id,
                parent_chunk_id=record.chunk.parent_chunk_id,
                source_id=record.chunk.source_id,
                display_title=record.display_title,
                document_version=record.chunk.document_version,
                locator=record.chunk.locator,
                text=record.chunk.text,
                revision_id=self.revision_id,
                ordinal=record.chunk.ordinal,
                content_digest=record.chunk.content_digest,
                record_digest=record.record_digest,
                bm25_rank=rank,
                bm25_score=score,
            )
            for rank, (score, record) in enumerate(ranked[:limit], start=1)
        )

    def save_new(self, path: Path) -> None:
        """Create one snapshot without ever replacing an existing path."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": self.SNAPSHOT_SCHEMA,
            "revision_id": self.revision_id,
            "tokenizer_identity": self.tokenizer_identity,
            "algorithm": {
                "version": self.algorithm_version,
                "k1": self.k1,
                "b": self.b,
            },
            "record_digest_algorithm": RECORD_DIGEST_ALGORITHM,
            "record_digests": self._record_digests,
            "chunk_set_digest": self._chunk_set_digest,
            "records": [
                {
                    "chunk": record.chunk.model_dump(mode="json"),
                    "display_title": record.display_title,
                    "tokens": record.tokens,
                }
                for record in self.records
            ],
        }
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise LexicalIndexError("snapshot_exists") from None
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    def save(self, path: Path) -> None:
        """Compatibility alias retaining immutable create-only semantics."""

        self.save_new(path)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_revision_id: str,
    ) -> PersistentBm25Index:
        path = Path(path)
        if not isinstance(expected_revision_id, str) or not expected_revision_id:
            raise LexicalIndexError("expected_revision_id_invalid")
        if not path.is_file():
            raise LexicalIndexError("snapshot_missing")
        try:
            raw_payload: object = json.loads(path.read_text(encoding="utf-8"))
            payload = cast(dict[str, Any], raw_payload)
            if not isinstance(raw_payload, dict):
                raise LexicalIndexError("invalid_snapshot")
            if payload.get("schema") != cls.SNAPSHOT_SCHEMA:
                raise LexicalIndexError("incompatible_snapshot_schema")
            revision_id = payload.get("revision_id")
            if not isinstance(revision_id, str) or not revision_id:
                raise LexicalIndexError("revision_id_invalid")
            if revision_id != expected_revision_id:
                raise LexicalIndexError("revision_id_mismatch")
            if payload.get("tokenizer_identity") != BILINGUAL_TOKENIZER_IDENTITY:
                raise LexicalIndexError("unsupported_tokenizer_identity")
            if payload.get("record_digest_algorithm") != RECORD_DIGEST_ALGORITHM:
                raise LexicalIndexError("unsupported_record_digest_algorithm")

            algorithm = payload.get("algorithm")
            if not isinstance(algorithm, dict):
                raise LexicalIndexError("invalid_algorithm_config")
            if algorithm.get("version") != cls.ALGORITHM_VERSION:
                raise LexicalIndexError("unsupported_algorithm_version")
            k1 = _finite_number(algorithm.get("k1"))
            b = _finite_number(algorithm.get("b"))
            if k1 <= 0 or not 0 <= b <= 1:
                raise LexicalIndexError("invalid_bm25_parameters")

            tokenizer = BilingualTokenizer()
            raw_records = payload.get("records")
            if not isinstance(raw_records, list):
                raise LexicalIndexError("invalid_records")
            records: list[LexicalRecord] = []
            seen_ids: set[str] = set()
            for raw_record in raw_records:
                if not isinstance(raw_record, dict):
                    raise LexicalIndexError("invalid_record")
                chunk = Chunk.model_validate(raw_record.get("chunk"))
                if chunk.chunk_id in seen_ids:
                    raise LexicalIndexError("duplicate_chunk_id")
                seen_ids.add(chunk.chunk_id)
                title = raw_record.get("display_title")
                tokens = raw_record.get("tokens")
                if not isinstance(title, str) or not title:
                    raise LexicalIndexError("invalid_display_title")
                if not isinstance(tokens, list) or any(
                    not isinstance(token, str) for token in tokens
                ):
                    raise LexicalIndexError("invalid_tokens")
                normalized_tokens = tuple(tokens)
                if normalized_tokens != tokenizer.tokenize(chunk.text):
                    raise LexicalIndexError("token_inventory_mismatch")
                records.append(
                    LexicalRecord(
                        chunk=chunk,
                        display_title=title,
                        tokens=normalized_tokens,
                        record_digest=chunk_record_digest(chunk, title),
                    )
                )

            raw_digests = payload.get("record_digests")
            if not isinstance(raw_digests, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in raw_digests.items()
            ):
                raise LexicalIndexError("invalid_record_digests")
            stored_digests = cast(dict[str, str], raw_digests)
            recomputed_digests = {record.chunk.chunk_id: record.record_digest for record in records}
            if stored_digests != recomputed_digests:
                raise LexicalIndexError("record_digest_mismatch")
            recomputed_set_digest = chunk_set_digest(recomputed_digests)
            if payload.get("chunk_set_digest") != recomputed_set_digest:
                raise LexicalIndexError("chunk_set_digest_mismatch")

            return cls(
                tuple(records),
                revision_id=revision_id,
                tokenizer=tokenizer,
                k1=k1,
                b=b,
            )
        except LexicalIndexError:
            raise
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValidationError,
            TypeError,
            ValueError,
        ):
            raise LexicalIndexError("invalid_snapshot") from None


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LexicalIndexError("invalid_algorithm_config")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise LexicalIndexError("invalid_algorithm_config")
    return normalized
