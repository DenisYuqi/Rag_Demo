from __future__ import annotations

import json

import pytest

from rag_mvp.domain.ingestion import ChunkLocator
from rag_mvp.domain.retrieval import RankingEvidence
from rag_mvp.qa.citations import StructuredAnswerError, StructuredAnswerParser
from rag_mvp.qa.context import ContextBuilder
from rag_mvp.qa.prompt import GENERATOR_OUTPUT_SCHEMA_VERSION


def _evidence(
    chunk_id: str,
    rank: int,
    *,
    revision_id: str = "revision-current",
    locator: ChunkLocator | None = None,
) -> RankingEvidence:
    return RankingEvidence(
        chunk_id=chunk_id,
        parent_chunk_id=chunk_id,
        source_id=f"source-{rank}",
        display_title=f"Policy {rank}",
        document_version=rank,
        locator=locator or ChunkLocator(pages=(rank,)),
        text=f"Evidence {rank}",
        revision_id=revision_id,
        final_rank=rank,
    )


def _content(
    *,
    answer: str = "Employees receive ten days of annual leave.",
    claims: object | None = None,
    schema_version: str = GENERATOR_OUTPUT_SCHEMA_VERSION,
) -> str:
    return json.dumps(
        {
            "schema_version": schema_version,
            "answer": answer,
            "claims": claims
            if claims is not None
            else [
                {
                    "text": "Employees receive ten days of annual leave.",
                    "citation_chunk_ids": ["chunk-1"],
                }
            ],
        }
    )


def test_parser_resolves_citation_metadata_from_current_context() -> None:
    context = ContextBuilder().build(
        (
            _evidence("chunk-1", 1),
            _evidence("chunk-2", 2, locator=ChunkLocator(section_path=("Leave",))),
        )
    )
    content = _content(
        claims=[
            {"text": "Ten days are available.", "citation_chunk_ids": ["chunk-1"]},
            {
                "text": "The leave section explains eligibility.",
                "citation_chunk_ids": ["chunk-2", "chunk-1"],
            },
        ]
    )

    result = StructuredAnswerParser().parse(
        content,
        context=context,
        expected_revision_id="revision-current",
    )

    assert result.cited_chunk_ids == ("chunk-1", "chunk-2")
    assert result.citations[0].source_title == "Policy 1"
    assert result.citations[0].document_version == 1
    assert result.citations[0].locator.pages == (1,)
    assert result.citations[1].locator.section_path == ("Leave",)


def test_parser_rejects_invented_citation_id() -> None:
    context = ContextBuilder().build((_evidence("chunk-1", 1),))

    with pytest.raises(StructuredAnswerError, match="citation_unknown"):
        StructuredAnswerParser().parse(
            _content(
                claims=[
                    {
                        "text": "Invented claim.",
                        "citation_chunk_ids": ["chunk-invented"],
                    }
                ]
            ),
            context=context,
            expected_revision_id="revision-current",
        )


def test_parser_rejects_citation_from_a_stale_revision() -> None:
    context = ContextBuilder().build((_evidence("chunk-1", 1, revision_id="revision-old"),))

    with pytest.raises(StructuredAnswerError, match="citation_stale"):
        StructuredAnswerParser().parse(
            _content(),
            context=context,
            expected_revision_id="revision-current",
        )


def test_parser_rejects_invalid_registry_locator() -> None:
    invalid = _evidence("chunk-1", 1).model_copy(update={"locator": object()})
    context = ContextBuilder().build((invalid,))

    with pytest.raises(StructuredAnswerError, match="citation_locator_invalid"):
        StructuredAnswerParser().parse(
            _content(),
            context=context,
            expected_revision_id="revision-current",
        )


def test_model_cannot_supply_or_override_citation_locator() -> None:
    context = ContextBuilder().build((_evidence("chunk-1", 1),))

    with pytest.raises(StructuredAnswerError, match="structured_answer_invalid"):
        StructuredAnswerParser().parse(
            _content(
                claims=[
                    {
                        "text": "Claim.",
                        "citation_chunk_ids": ["chunk-1"],
                        "locator": {"pages": [999]},
                    }
                ]
            ),
            context=context,
            expected_revision_id="revision-current",
        )


@pytest.mark.parametrize(
    ("content", "code"),
    [
        ("not-json", "structured_answer_invalid"),
        ("[]", "structured_answer_invalid"),
        (
            _content(schema_version="grounded-answer-old"),
            "output_schema_version_invalid",
        ),
        (
            '{"schema_version":"grounded-answer-v1","answer":"one","answer":"two","claims":[]}',
            "structured_answer_invalid",
        ),
        (
            _content(
                claims=[
                    {
                        "text": "Claim.",
                        "citation_chunk_ids": ["chunk-1", "chunk-1"],
                    }
                ]
            ),
            "citation_duplicate",
        ),
    ],
)
def test_parser_rejects_malformed_or_ambiguous_output(content: str, code: str) -> None:
    with pytest.raises(StructuredAnswerError, match=code):
        StructuredAnswerParser().parse(
            content,
            context=ContextBuilder().build((_evidence("chunk-1", 1),)),
            expected_revision_id="revision-current",
        )
