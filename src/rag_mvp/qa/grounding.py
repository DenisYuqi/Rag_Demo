"""Deterministic factual-unit coverage and request-scoped grounding validation."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from pydantic import ValidationError

from rag_mvp.domain.qa import AnswerClaim, Citation
from rag_mvp.domain.retrieval import RankingEvidence
from rag_mvp.qa.citations import ParsedAnswer

GROUNDING_VALIDATOR_VERSION = "exact-claim-coverage-request-registry-v1"
_OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$")


class GroundingValidationError(ValueError):
    """A content-free reason that a complete generated answer was withheld."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ValidatedGroundedAnswer:
    request_id: str
    revision_id: str
    answer: str = field(repr=False)
    claims: tuple[AnswerClaim, ...] = field(repr=False)
    citations: tuple[Citation, ...] = field(repr=False)
    validator_version: str = GROUNDING_VALIDATOR_VERSION


class GroundingValidator:
    """Validate deterministic release invariants without semantic re-scoring."""

    def validate(
        self,
        parsed: ParsedAnswer,
        *,
        request_id: str,
        revision_id: str,
        candidates: Sequence[RankingEvidence],
    ) -> ValidatedGroundedAnswer:
        self._validate_request_identity(request_id, revision_id)
        if not isinstance(parsed, ParsedAnswer) or not parsed.claims:
            raise GroundingValidationError("parsed_answer_invalid")
        if _coverage_key(parsed.answer) != _coverage_key(
            "".join(claim.text for claim in parsed.claims)
        ):
            raise GroundingValidationError("factual_unit_coverage_invalid")

        registry = self._candidate_registry(candidates, revision_id)
        referenced_ids = tuple(
            dict.fromkeys(
                chunk_id for claim in parsed.claims for chunk_id in claim.citation_chunk_ids
            )
        )
        if not referenced_ids:
            raise GroundingValidationError("factual_unit_citation_missing")
        for chunk_id in referenced_ids:
            if chunk_id not in registry:
                raise GroundingValidationError("citation_not_in_request")

        if any(not isinstance(citation, Citation) for citation in parsed.citations):
            raise GroundingValidationError("citation_set_invalid")
        citation_ids = tuple(citation.chunk_id for citation in parsed.citations)
        if citation_ids != referenced_ids:
            raise GroundingValidationError("citation_set_invalid")
        for citation in parsed.citations:
            expected = self._citation_from_candidate(registry[citation.chunk_id])
            if citation != expected:
                raise GroundingValidationError("citation_metadata_mismatch")

        return ValidatedGroundedAnswer(
            request_id=request_id,
            revision_id=revision_id,
            answer=parsed.answer,
            claims=parsed.claims,
            citations=parsed.citations,
        )

    @staticmethod
    def _validate_request_identity(request_id: str, revision_id: str) -> None:
        if not isinstance(request_id, str) or _OPAQUE_ID.fullmatch(request_id) is None:
            raise GroundingValidationError("request_id_invalid")
        if not isinstance(revision_id, str) or _OPAQUE_ID.fullmatch(revision_id) is None:
            raise GroundingValidationError("revision_id_invalid")

    @staticmethod
    def _candidate_registry(
        candidates: object,
        revision_id: str,
    ) -> dict[str, RankingEvidence]:
        if isinstance(candidates, (str, bytes, bytearray)) or not isinstance(candidates, Sequence):
            raise GroundingValidationError("candidate_registry_invalid")
        values: list[RankingEvidence] = []
        for candidate in cast(Sequence[object], candidates):
            if not isinstance(candidate, RankingEvidence):
                raise GroundingValidationError("candidate_registry_invalid")
            values.append(candidate)
        ordered = tuple(sorted(values, key=lambda candidate: candidate.final_rank))
        ranks = tuple(candidate.final_rank for candidate in ordered)
        if ranks != tuple(range(1, len(ordered) + 1)):
            raise GroundingValidationError("candidate_registry_invalid")
        registry = {candidate.chunk_id: candidate for candidate in ordered}
        if len(registry) != len(ordered) or any(
            candidate.revision_id != revision_id for candidate in ordered
        ):
            raise GroundingValidationError("candidate_registry_invalid")
        return registry

    @staticmethod
    def _citation_from_candidate(candidate: RankingEvidence) -> Citation:
        try:
            return Citation(
                source_title=candidate.display_title,
                document_version=candidate.document_version,
                chunk_id=candidate.chunk_id,
                locator=candidate.locator,
            )
        except (TypeError, ValueError, ValidationError):
            raise GroundingValidationError("candidate_registry_invalid") from None


def _coverage_key(value: str) -> str:
    try:
        return "".join(unicodedata.normalize("NFC", value).split())
    except (TypeError, ValueError):
        raise GroundingValidationError("factual_unit_coverage_invalid") from None
