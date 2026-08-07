from __future__ import annotations

import pytest

from rag_mvp.domain.ingestion import DocumentKind
from rag_mvp.ingestion.chunking import (
    ChunkingConfig,
    chunk_document,
    chunk_document_hierarchy,
    token_spans,
)
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
    assert first[0].text.rstrip().endswith("four")
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


@pytest.mark.parametrize(
    "values",
    [
        {"target_tokens": 4.0},
        {"target_tokens": True},
        {"overlap_tokens": 1.0},
        {"overlap_tokens": False},
        {"parent_target_tokens": 4.0},
        {"parent_target_tokens": True},
    ],
)
def test_chunking_config_requires_integer_token_counts(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(**values)  # type: ignore[arg-type]


def test_chunk_ranges_preserve_all_source_content_including_boundary_whitespace() -> None:
    text = "  one \n two\t three  four \n"
    document = ExtractedDocument(
        kind=DocumentKind.TEXT,
        blocks=(ExtractedBlock(text=text),),
    )

    chunks = chunk_document(
        document,
        source_id="source-whitespace",
        document_version=1,
        config=ChunkingConfig(target_tokens=2, overlap_tokens=0),
    )

    assert "".join(chunk.text for chunk in chunks) == text
    assert chunks[0].text.startswith("  ")
    assert chunks[-1].text.endswith(" \n")
    for chunk in chunks:
        start = chunk.locator.char_start
        end = chunk.locator.char_end
        assert start is not None and end is not None
        assert chunk.text == text[start:end]


def test_overlapping_chunk_ranges_are_bounded_and_cover_every_character() -> None:
    text = " one  two\nthree\tfour "
    document = ExtractedDocument(
        kind=DocumentKind.TEXT,
        blocks=(ExtractedBlock(text=text),),
    )

    chunks = chunk_document(
        document,
        source_id="source-overlap",
        document_version=1,
        config=ChunkingConfig(target_tokens=2, overlap_tokens=1),
    )

    covered = [False] * len(text)
    previous_end = 0
    for index, chunk in enumerate(chunks):
        start = chunk.locator.char_start
        end = chunk.locator.char_end
        assert start is not None and end is not None
        assert chunk.token_count is not None and chunk.token_count <= 2
        assert chunk.text == text[start:end]
        if index:
            assert start < previous_end
        for position in range(start, end):
            covered[position] = True
        previous_end = end

    assert all(covered)


def test_parent_child_derivation_is_bounded_contained_and_deterministic() -> None:
    text = " one two three four five six seven eight nine ten "
    document = ExtractedDocument(
        kind=DocumentKind.TEXT,
        blocks=(ExtractedBlock(text=text, section_path=("Policy",)),),
    )
    config = ChunkingConfig(
        parent_target_tokens=6,
        target_tokens=4,
        overlap_tokens=1,
    )

    first = chunk_document_hierarchy(
        document,
        source_id="parent-child-source",
        document_version=3,
        config=config,
    )
    second = chunk_document_hierarchy(
        document,
        source_id="parent-child-source",
        document_version=3,
        config=config,
    )

    assert first == second
    assert len(first.parents) == 2
    assert "".join(parent.text for parent in first.parents) == text
    assert all(parent.token_count <= 6 for parent in first.parents)
    parent_by_id = {parent.parent_chunk_id: parent for parent in first.parents}
    for child in first.children:
        parent = parent_by_id[child.parent_chunk_id]
        assert child.token_count is not None and child.token_count <= 4
        assert child.text in parent.text
        assert child.locator.section_path == parent.locator.section_path
        assert parent.locator.char_start is not None
        assert parent.locator.char_end is not None
        assert child.locator.char_start is not None
        assert child.locator.char_end is not None
        assert parent.locator.char_start <= child.locator.char_start
        assert child.locator.char_end <= parent.locator.char_end


def test_parent_target_must_not_be_smaller_than_child_target() -> None:
    with pytest.raises(ValueError, match="parent_target_tokens"):
        ChunkingConfig(parent_target_tokens=3, target_tokens=4, overlap_tokens=1)
