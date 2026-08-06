"""Strict structured-answer parsing and deterministic citation resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Annotated, Never, cast

from pydantic import Field, ValidationError

from rag_mvp.domain._base import DomainModel, NonEmptyText
from rag_mvp.domain.ingestion import ChunkLocator
from rag_mvp.domain.qa import AnswerClaim, Citation
from rag_mvp.domain.retrieval import RankingEvidence
from rag_mvp.qa.context import (
    CONTEXT_SELECTION_VERSION,
    CONTEXT_TOKENIZER_VERSION,
    ContextSelection,
)
from rag_mvp.qa.prompt import (
    GENERATOR_OUTPUT_SCHEMA_VERSION,
    MAX_CITATIONS_PER_CLAIM,
    MAX_GENERATED_CLAIMS,
)

MAX_STRUCTURED_ANSWER_CHARACTERS = 65_536


class StructuredAnswerError(ValueError):
    """A safe, stable generated-answer validation failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _GeneratedAnswer(DomainModel):
    schema_version: str
    answer: NonEmptyText
    claims: Annotated[
        tuple[AnswerClaim, ...],
        Field(min_length=1, max_length=MAX_GENERATED_CLAIMS),
    ]


@dataclass(frozen=True, slots=True)
class ParsedAnswer:
    answer: str = field(repr=False)
    claims: tuple[AnswerClaim, ...] = field(repr=False)
    citations: tuple[Citation, ...] = field(repr=False)
    schema_version: str = GENERATOR_OUTPUT_SCHEMA_VERSION

    @property
    def cited_chunk_ids(self) -> tuple[str, ...]:
        return tuple(citation.chunk_id for citation in self.citations)


class StructuredAnswerParser:
    """Resolve model-supplied IDs only through the selected revision-bound context."""

    def __init__(
        self,
        *,
        maximum_characters: int = MAX_STRUCTURED_ANSWER_CHARACTERS,
    ) -> None:
        if type(maximum_characters) is not int or maximum_characters < 1:
            raise ValueError("maximum_characters must be a positive integer")
        self.maximum_characters = maximum_characters

    def parse(
        self,
        content: str,
        *,
        context: ContextSelection,
        expected_revision_id: str,
    ) -> ParsedAnswer:
        if not isinstance(content, str) or not content or len(content) > self.maximum_characters:
            raise StructuredAnswerError("structured_answer_invalid")
        if not isinstance(expected_revision_id, str) or not expected_revision_id.strip():
            raise StructuredAnswerError("expected_revision_invalid")
        raw = self._load(content)
        if raw.get("schema_version") != GENERATOR_OUTPUT_SCHEMA_VERSION:
            raise StructuredAnswerError("output_schema_version_invalid")
        try:
            generated = _GeneratedAnswer.model_validate(raw)
        except (TypeError, ValueError, ValidationError):
            raise StructuredAnswerError("structured_answer_invalid") from None

        registry = self._context_registry(context)
        citations: list[Citation] = []
        resolved_ids: set[str] = set()
        for claim in generated.claims:
            if len(claim.citation_chunk_ids) > MAX_CITATIONS_PER_CLAIM:
                raise StructuredAnswerError("citation_limit_exceeded")
            if len(claim.citation_chunk_ids) != len(set(claim.citation_chunk_ids)):
                raise StructuredAnswerError("citation_duplicate")
            for chunk_id in claim.citation_chunk_ids:
                evidence = registry.get(chunk_id)
                if evidence is None:
                    raise StructuredAnswerError("citation_unknown")
                if evidence.revision_id != expected_revision_id:
                    raise StructuredAnswerError("citation_stale")
                if chunk_id in resolved_ids:
                    continue
                citations.append(self._citation(evidence))
                resolved_ids.add(chunk_id)

        return ParsedAnswer(
            answer=generated.answer,
            claims=generated.claims,
            citations=tuple(citations),
        )

    @staticmethod
    def _load(content: str) -> dict[str, object]:
        try:
            raw: object = json.loads(
                content,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, UnicodeError, ValueError, _DuplicateJsonKey):
            raise StructuredAnswerError("structured_answer_invalid") from None
        if not isinstance(raw, dict):
            raise StructuredAnswerError("structured_answer_invalid")
        return cast(dict[str, object], raw)

    @staticmethod
    def _context_registry(context: ContextSelection) -> dict[str, RankingEvidence]:
        if not isinstance(context, ContextSelection):
            raise StructuredAnswerError("citation_context_invalid")
        if (
            context.version != CONTEXT_SELECTION_VERSION
            or context.tokenizer_version != CONTEXT_TOKENIZER_VERSION
        ):
            raise StructuredAnswerError("citation_context_invalid")
        registry: dict[str, RankingEvidence] = {}
        for chunk in context.chunks:
            if chunk.chunk_id in registry:
                raise StructuredAnswerError("citation_context_invalid")
            registry[chunk.chunk_id] = chunk.evidence
        return registry

    @staticmethod
    def _citation(evidence: RankingEvidence) -> Citation:
        try:
            chunk_id = evidence.chunk_id
            source_title = evidence.display_title
            document_version = evidence.document_version
            raw_locator: object = evidence.locator
            if not isinstance(raw_locator, ChunkLocator):
                raise TypeError
            locator = ChunkLocator.model_validate(raw_locator.model_dump())
            return Citation(
                source_title=source_title,
                document_version=document_version,
                chunk_id=chunk_id,
                locator=locator,
            )
        except (AttributeError, TypeError, ValueError, ValidationError):
            raise StructuredAnswerError("citation_locator_invalid") from None


class _DuplicateJsonKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Never:
    del value
    raise ValueError
