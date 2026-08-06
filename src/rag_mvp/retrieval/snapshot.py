"""Canonical record and complete-snapshot digests shared by persistent indexes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from rag_mvp.domain.ingestion import Chunk

RECORD_DIGEST_ALGORITHM = "chunk-record-sha256-v1"
CHUNK_SET_DIGEST_ALGORITHM = "chunk-set-sha256-v1"


def chunk_record_digest(chunk: Chunk, display_title: str) -> str:
    """Digest every field that must agree between dense and lexical records."""

    payload = {
        "algorithm": RECORD_DIGEST_ALGORITHM,
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "document_version": chunk.document_version,
        "ordinal": chunk.ordinal,
        "text": chunk.text,
        "content_digest": chunk.content_digest,
        "locator": chunk.locator.model_dump(mode="json"),
        "display_title": display_title,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def chunk_set_digest(record_digests: Mapping[str, str]) -> str:
    """Digest a complete inventory independent of insertion order."""

    payload = {
        "algorithm": CHUNK_SET_DIGEST_ALGORITHM,
        "records": [
            {"chunk_id": chunk_id, "record_digest": record_digests[chunk_id]}
            for chunk_id in sorted(record_digests)
        ],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def canonical_locator_json(chunk: Chunk) -> str:
    return _canonical_json(chunk.locator.model_dump(mode="json"))


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
