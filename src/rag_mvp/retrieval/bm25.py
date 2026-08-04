"""Persistent deterministic BM25 snapshot and search."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_mvp.domain.ingestion import Chunk
from rag_mvp.domain.retrieval import RetrievalCandidate
from rag_mvp.retrieval.tokenizer import BilingualTokenizer


class LexicalIndexError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LexicalRecord:
    chunk: Chunk
    display_title: str
    tokens: tuple[str, ...]


class PersistentBm25Index:
    """An immutable revision-specific lexical index backed by one JSON snapshot."""

    SNAPSHOT_SCHEMA = "bm25-snapshot-v1"

    def __init__(
        self,
        records: tuple[LexicalRecord, ...],
        *,
        tokenizer: BilingualTokenizer | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer or BilingualTokenizer()
        self.k1 = k1
        self.b = b
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("invalid BM25 parameters")
        self._document_frequencies: Counter[str] = Counter()
        for record in records:
            self._document_frequencies.update(set(record.tokens))
        self._average_length = (
            sum(len(record.tokens) for record in records) / len(records) if records else 0.0
        )

    @classmethod
    def build(
        cls,
        chunks: tuple[Chunk, ...],
        titles: dict[str, str],
        *,
        tokenizer: BilingualTokenizer | None = None,
    ) -> PersistentBm25Index:
        resolved = tokenizer or BilingualTokenizer()
        records = tuple(
            LexicalRecord(
                chunk=chunk,
                display_title=titles[chunk.source_id],
                tokens=resolved.tokenize(chunk.text),
            )
            for chunk in sorted(chunks, key=lambda item: item.chunk_id)
        )
        if len({record.chunk.chunk_id for record in records}) != len(records):
            raise LexicalIndexError("duplicate_chunk_id")
        return cls(records, tokenizer=resolved)

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
        if limit < 1:
            raise ValueError("limit must be positive")
        query_tokens = self.tokenizer.tokenize(query)
        ranked = [
            (self._score(query_tokens, record), record)
            for record in self.records
        ]
        ranked = [entry for entry in ranked if entry[0] > 0]
        ranked.sort(key=lambda entry: (-entry[0], entry[1].chunk.chunk_id))
        return tuple(
            RetrievalCandidate(
                chunk_id=record.chunk.chunk_id,
                source_id=record.chunk.source_id,
                display_title=record.display_title,
                document_version=record.chunk.document_version,
                locator=record.chunk.locator,
                text=record.chunk.text,
                bm25_rank=rank,
                bm25_score=score,
            )
            for rank, (score, record) in enumerate(ranked[:limit], start=1)
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": self.SNAPSHOT_SCHEMA,
            "tokenizer_version": self.tokenizer.version,
            "k1": self.k1,
            "b": self.b,
            "records": [
                {
                    "chunk": record.chunk.model_dump(mode="json"),
                    "display_title": record.display_title,
                    "tokens": record.tokens,
                }
                for record in self.records
            ],
        }
        descriptor, raw_path = tempfile.mkstemp(prefix=".bm25-", suffix=".json", dir=path.parent)
        temporary = Path(raw_path)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path) -> PersistentBm25Index:
        try:
            payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            if payload["schema"] != cls.SNAPSHOT_SCHEMA:
                raise LexicalIndexError("incompatible_snapshot")
            tokenizer = BilingualTokenizer(version=str(payload["tokenizer_version"]))
            records = tuple(
                LexicalRecord(
                    chunk=Chunk.model_validate(item["chunk"]),
                    display_title=str(item["display_title"]),
                    tokens=tuple(str(token) for token in item["tokens"]),
                )
                for item in payload["records"]
            )
            return cls(
                records,
                tokenizer=tokenizer,
                k1=float(payload["k1"]),
                b=float(payload["b"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise LexicalIndexError("invalid_snapshot") from error
