from __future__ import annotations

from rag_mvp.domain.ingestion import DocumentKind
from rag_mvp.ingestion.chunking import ChunkingConfig, chunk_document, token_spans
from rag_mvp.ingestion.extractors import ExtractedBlock, ExtractedDocument


def test_chunking_is_bounded_overlapping_and_deterministic() -> None:
    document = ExtractedDocument(
        kind=DocumentKind.TEXT,
        blocks=(
            ExtractedBlock(
                text="one two three four five six seven",
                section_path=("Policy",),
            ),
        ),
    )
    config = ChunkingConfig(target_tokens=4, overlap_tokens=1)

    first = chunk_document(document, source_id="source-1", document_version=1, config=config)
    second = chunk_document(document, source_id="source-1", document_version=1, config=config)

    assert first == second
    assert [chunk.token_count for chunk in first] == [4, 4]
    assert first[0].text.endswith("four")
    assert first[1].text.startswith("four")
    assert first[0].locator.section_path == ("Policy",)


def test_pdf_chunks_keep_explicit_page_locators() -> None:
    document = ExtractedDocument(
        kind=DocumentKind.PDF,
        blocks=(
            ExtractedBlock(text="第一页 policy terms", page_number=1),
            ExtractedBlock(text="第二页 more terms", page_number=2),
        ),
    )

    chunks = chunk_document(
        document,
        source_id="pdf-source",
        document_version=2,
        config=ChunkingConfig(target_tokens=3, overlap_tokens=0),
    )

    assert {chunk.locator.pages for chunk in chunks} == {(1,), (2,)}
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_chinese_without_spaces_is_tokenized_deterministically() -> None:
    text = "员工休假政策"

    assert len(token_spans(text)) == len(text)
