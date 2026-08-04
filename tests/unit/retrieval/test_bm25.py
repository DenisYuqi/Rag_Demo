from __future__ import annotations

from pathlib import Path

from rag_mvp.domain.ingestion import Chunk, ChunkLocator
from rag_mvp.retrieval.bm25 import PersistentBm25Index


def _chunk(chunk_id: str, text: str, ordinal: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_id="source-1",
        document_version=1,
        ordinal=ordinal,
        text=text,
        content_digest="digest-" + chunk_id,
        locator=ChunkLocator(char_start=ordinal * 10, char_end=ordinal * 10 + len(text)),
    )


async def test_bm25_finds_exact_english_and_unspaced_chinese(tmp_path: Path) -> None:
    index = PersistentBm25Index.build(
        (
            _chunk("english", "Policy code AKP-204 requires MFA", 0),
            _chunk("chinese", "员工休假政策规定年假天数", 1),
        ),
        {"source-1": "Handbook"},
    )
    path = tmp_path / "bm25.json"
    index.save(path)

    reopened = PersistentBm25Index.load(path)
    english = await reopened.search("AKP-204", 5)
    chinese = await reopened.search("休假政策", 5)

    assert english[0].chunk_id == "english"
    assert chinese[0].chunk_id == "chinese"
    assert english[0].bm25_score and english[0].bm25_score > 0


async def test_bm25_empty_query_match_returns_empty() -> None:
    index = PersistentBm25Index.build(
        (_chunk("english", "annual leave", 0),),
        {"source-1": "Handbook"},
    )

    assert await index.search("unrelated-token", 5) == ()
