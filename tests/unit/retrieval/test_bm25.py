from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rag_mvp.domain.ingestion import Chunk, ChunkLocator
from rag_mvp.retrieval.bm25 import LexicalIndexError, PersistentBm25Index
from rag_mvp.retrieval.tokenizer import (
    BILINGUAL_TOKENIZER_IDENTITY,
    JIEBA_DICTIONARY_SHA256,
    JIEBA_PACKAGE_VERSION,
)


def _chunk(chunk_id: str, text: str, ordinal: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_id="source-1",
        document_version=1,
        ordinal=ordinal,
        text=text,
        content_digest=hashlib.sha256(text.encode()).hexdigest(),
        locator=ChunkLocator(char_start=ordinal * 100, char_end=ordinal * 100 + len(text)),
    )


def _save_index(
    path: Path,
    chunks: tuple[Chunk, ...],
    *,
    revision_id: str = "revision-bm25",
) -> None:
    PersistentBm25Index.build(
        chunks,
        {"source-1": "Handbook"},
        revision_id=revision_id,
    ).save_new(path)


async def test_bm25_finds_exact_english_and_unspaced_chinese_after_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bm25.json"
    _save_index(
        path,
        (
            _chunk("english", "Policy code AKP-204 requires MFA", 0),
            _chunk("chinese", "员工休假政策规定年假天数", 1),
        ),
    )

    reopened = PersistentBm25Index.load(path, expected_revision_id="revision-bm25")
    english = await reopened.search("AKP-204", 5)
    chinese = await reopened.search("休假政策", 5)

    assert english[0].chunk_id == "english"
    assert chinese[0].chunk_id == "chinese"
    assert english[0].bm25_score is not None and english[0].bm25_score > 0
    assert english[0].revision_id == "revision-bm25"
    assert english[0].ordinal == 0
    assert english[0].content_digest is not None
    assert english[0].record_digest == reopened.record_digests["english"]


async def test_bm25_empty_query_match_returns_empty() -> None:
    index = PersistentBm25Index.build(
        (_chunk("english", "annual leave", 0),),
        {"source-1": "Handbook"},
        revision_id="revision-empty-query",
    )

    assert await index.search("unrelated-token", 5) == ()


def test_bm25_load_requires_exact_expected_revision(tmp_path: Path) -> None:
    path = tmp_path / "bm25.json"
    _save_index(path, (_chunk("chunk", "policy", 0),))

    with pytest.raises(TypeError):
        PersistentBm25Index.load(path)  # type: ignore[call-arg]
    with pytest.raises(LexicalIndexError, match="expected_revision_id_invalid"):
        PersistentBm25Index.load(path, expected_revision_id="")
    with pytest.raises(LexicalIndexError, match="revision_id_mismatch"):
        PersistentBm25Index.load(path, expected_revision_id="revision-other")


def test_tokenizer_identity_pins_package_dictionary_and_implementation() -> None:
    assert f"jieba-{JIEBA_PACKAGE_VERSION}" in BILINGUAL_TOKENIZER_IDENTITY
    assert f"dict-sha256-{JIEBA_DICTIONARY_SHA256}" in BILINGUAL_TOKENIZER_IDENTITY
    assert BILINGUAL_TOKENIZER_IDENTITY.endswith("hmm-false")


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda payload: payload.__setitem__("tokenizer_identity", "other-tokenizer"),
            "unsupported_tokenizer_identity",
        ),
        (
            lambda payload: payload["algorithm"].__setitem__("version", "other-bm25"),
            "unsupported_algorithm_version",
        ),
        (
            lambda payload: payload["records"][0].__setitem__("tokens", ["tampered"]),
            "token_inventory_mismatch",
        ),
        (
            lambda payload: payload["record_digests"].__setitem__("chunk", "0" * 64),
            "record_digest_mismatch",
        ),
        (
            lambda payload: payload.__setitem__("chunk_set_digest", "0" * 64),
            "chunk_set_digest_mismatch",
        ),
    ],
)
def test_bm25_snapshot_rejects_identity_and_inventory_tampering(
    tmp_path: Path,
    mutate: object,
    code: str,
) -> None:
    path = tmp_path / "bm25.json"
    _save_index(path, (_chunk("chunk", "exact policy", 0),))
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)  # type: ignore[operator]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(LexicalIndexError, match=code):
        PersistentBm25Index.load(path, expected_revision_id="revision-bm25")


async def test_bm25_ties_are_stable_by_chunk_id_across_input_order_and_restart(
    tmp_path: Path,
) -> None:
    expected = ["chunk-a", "chunk-b"]
    for number, chunks in enumerate(
        (
            (
                _chunk("chunk-b", "shared term", 1),
                _chunk("chunk-a", "shared term", 0),
            ),
            (
                _chunk("chunk-a", "shared term", 0),
                _chunk("chunk-b", "shared term", 1),
            ),
        ),
        start=1,
    ):
        path = tmp_path / f"bm25-{number}.json"
        revision_id = f"revision-{number}"
        _save_index(path, chunks, revision_id=revision_id)
        for _ in range(3):
            reopened = PersistentBm25Index.load(path, expected_revision_id=revision_id)
            assert [item.chunk_id for item in await reopened.search("shared", 2)] == expected


async def test_boundary_whitespace_survives_bm25_snapshot_and_candidate_round_trip(
    tmp_path: Path,
) -> None:
    text = "  exact policy text\n"
    chunk = _chunk("chunk-space", text, 0)
    assert Chunk.model_validate_json(chunk.model_dump_json()).text == text
    path = tmp_path / "bm25-space.json"
    _save_index(path, (chunk,), revision_id="revision-space")

    reopened = PersistentBm25Index.load(path, expected_revision_id="revision-space")
    result = (await reopened.search("policy", 1))[0]

    assert reopened.records[0].chunk.text == text
    assert result.text == text
    assert result.model_validate_json(result.model_dump_json()).text == text


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
async def test_bm25_limit_must_be_a_positive_integer(limit: object) -> None:
    index = PersistentBm25Index.build(
        (_chunk("chunk", "policy", 0),),
        {"source-1": "Handbook"},
        revision_id="revision-limit",
    )
    with pytest.raises(ValueError, match="limit must be positive"):
        await index.search("policy", limit)  # type: ignore[arg-type]
