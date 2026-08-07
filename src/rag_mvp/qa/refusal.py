"""Calibrated, content-free evidence sufficiency and conflict decisions."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final, cast

from rag_mvp.domain.qa import RefusalReason
from rag_mvp.domain.retrieval import RankingEvidence

LEGACY_REFUSAL_POLICY_VERSION = "normalized-fact-support-v1"
REFUSAL_POLICY_VERSION = "normalized-fact-support-v2"
DEFAULT_MINIMUM_SUPPORT_SCORE = 0.55
PARTIAL_EVIDENCE_MESSAGES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "en": "Some requested information is not supported by the available evidence.",
        "zh": "现有证据仅支持请求中的部分信息.",
        "zh-CN": "现有证据仅支持请求中的部分信息.",
    }
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")


class EvidenceDecisionKind(StrEnum):
    ANSWER = "answer"
    PARTIAL = "partial"
    REFUSAL = "refusal"


class EvidenceDecisionCode(StrEnum):
    ANSWERABLE = "answerable"
    PARTIAL_EVIDENCE = "partial-evidence"
    LOW_CONFIDENCE = RefusalReason.LOW_CONFIDENCE
    INSUFFICIENT_EVIDENCE = RefusalReason.INSUFFICIENT_EVIDENCE
    CONFLICTING_EVIDENCE = RefusalReason.CONFLICTING_EVIDENCE


class RefusalPolicyError(ValueError):
    """A content-free evidence-policy input failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class FactEvidence:
    """Normalized support for one request fact, without storing the fact text."""

    fact_id: str
    support_score: float
    supporting_chunk_ids: tuple[str, ...] = ()
    conflicting_chunk_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "supporting_chunk_ids", tuple(self.supporting_chunk_ids))
        object.__setattr__(self, "conflicting_chunk_ids", tuple(self.conflicting_chunk_ids))
        if not isinstance(self.fact_id, str) or _SAFE_ID.fullmatch(self.fact_id) is None:
            raise ValueError("fact_id_invalid")
        if (
            isinstance(self.support_score, bool)
            or not isinstance(self.support_score, (int, float))
            or not math.isfinite(self.support_score)
            or not 0 <= self.support_score <= 1
        ):
            raise ValueError("support_score_invalid")
        support_ids = self.supporting_chunk_ids
        conflict_ids = self.conflicting_chunk_ids
        if any(
            not isinstance(chunk_id, str) or not chunk_id
            for chunk_id in (*support_ids, *conflict_ids)
        ):
            raise ValueError("fact_evidence_chunk_id_invalid")
        if len(support_ids) != len(set(support_ids)) or len(conflict_ids) != len(set(conflict_ids)):
            raise ValueError("fact_evidence_chunk_id_duplicate")
        if set(support_ids) & set(conflict_ids):
            raise ValueError("fact_evidence_sides_overlap")
        if conflict_ids and not support_ids:
            raise ValueError("conflict_requires_both_sides")
        if not support_ids and self.support_score != 0:
            raise ValueError("score_requires_supporting_evidence")


@dataclass(frozen=True, slots=True)
class EvidenceDecision:
    kind: EvidenceDecisionKind
    code: EvidenceDecisionCode
    supported_fact_ids: tuple[str, ...]
    unsupported_fact_ids: tuple[str, ...]
    conflicting_fact_ids: tuple[str, ...]
    citation_chunk_ids: tuple[str, ...]
    reason: RefusalReason | None
    minimum_support_score: float
    policy_version: str = REFUSAL_POLICY_VERSION

    @property
    def answer_allowed(self) -> bool:
        return self.kind in {EvidenceDecisionKind.ANSWER, EvidenceDecisionKind.PARTIAL}

    @property
    def requires_refusal(self) -> bool:
        return self.kind is EvidenceDecisionKind.REFUSAL


class RefusalPolicy:
    """Apply a versioned support threshold; material unresolved conflict always wins."""

    def __init__(
        self,
        *,
        minimum_support_score: float = DEFAULT_MINIMUM_SUPPORT_SCORE,
    ) -> None:
        if (
            isinstance(minimum_support_score, bool)
            or not isinstance(minimum_support_score, (int, float))
            or not math.isfinite(minimum_support_score)
            or not 0 < minimum_support_score <= 1
        ):
            raise ValueError("minimum_support_score must be finite and in (0, 1]")
        self.minimum_support_score = float(minimum_support_score)

    def decide(
        self,
        facts: Sequence[FactEvidence],
        *,
        candidates: Sequence[RankingEvidence],
        revision_id: str,
    ) -> EvidenceDecision:
        assessments = self._validated_facts(facts)
        registry = self._candidate_registry(candidates, revision_id)
        supported: list[str] = []
        unsupported: list[str] = []
        conflicting: list[str] = []
        supporting_citations: list[str] = []
        conflict_citations: list[str] = []

        for fact in assessments:
            referenced = (*fact.supporting_chunk_ids, *fact.conflicting_chunk_ids)
            if any(chunk_id not in registry for chunk_id in referenced):
                raise RefusalPolicyError("fact_evidence_not_in_request")
            if fact.conflicting_chunk_ids:
                conflicting.append(fact.fact_id)
                conflict_citations.extend(referenced)
            elif fact.supporting_chunk_ids and fact.support_score >= self.minimum_support_score:
                supported.append(fact.fact_id)
                supporting_citations.extend(fact.supporting_chunk_ids)
            else:
                unsupported.append(fact.fact_id)

        if conflicting:
            return self._decision(
                EvidenceDecisionKind.REFUSAL,
                EvidenceDecisionCode.CONFLICTING_EVIDENCE,
                supported,
                unsupported,
                conflicting,
                conflict_citations,
                RefusalReason.CONFLICTING_EVIDENCE,
            )
        if not supported:
            return self._decision(
                EvidenceDecisionKind.REFUSAL,
                EvidenceDecisionCode.LOW_CONFIDENCE,
                supported,
                unsupported,
                conflicting,
                (),
                RefusalReason.LOW_CONFIDENCE,
            )
        if unsupported:
            return self._decision(
                EvidenceDecisionKind.PARTIAL,
                EvidenceDecisionCode.PARTIAL_EVIDENCE,
                supported,
                unsupported,
                conflicting,
                supporting_citations,
                None,
            )
        return self._decision(
            EvidenceDecisionKind.ANSWER,
            EvidenceDecisionCode.ANSWERABLE,
            supported,
            unsupported,
            conflicting,
            supporting_citations,
            None,
        )

    def _decision(
        self,
        kind: EvidenceDecisionKind,
        code: EvidenceDecisionCode,
        supported: Sequence[str],
        unsupported: Sequence[str],
        conflicting: Sequence[str],
        citations: Sequence[str],
        reason: RefusalReason | None,
    ) -> EvidenceDecision:
        return EvidenceDecision(
            kind=kind,
            code=code,
            supported_fact_ids=tuple(supported),
            unsupported_fact_ids=tuple(unsupported),
            conflicting_fact_ids=tuple(conflicting),
            citation_chunk_ids=tuple(dict.fromkeys(citations)),
            reason=reason,
            minimum_support_score=self.minimum_support_score,
        )

    @staticmethod
    def _validated_facts(facts: object) -> tuple[FactEvidence, ...]:
        if isinstance(facts, (str, bytes, bytearray)) or not isinstance(facts, Sequence):
            raise RefusalPolicyError("fact_assessment_invalid")
        values: list[FactEvidence] = []
        for fact in cast(Sequence[object], facts):
            if not isinstance(fact, FactEvidence):
                raise RefusalPolicyError("fact_assessment_invalid")
            values.append(fact)
        if not values or len({fact.fact_id for fact in values}) != len(values):
            raise RefusalPolicyError("fact_assessment_invalid")
        return tuple(values)

    @staticmethod
    def _candidate_registry(
        candidates: object,
        revision_id: str,
    ) -> dict[str, RankingEvidence]:
        if not isinstance(revision_id, str) or _SAFE_ID.fullmatch(revision_id) is None:
            raise RefusalPolicyError("revision_id_invalid")
        if isinstance(candidates, (str, bytes, bytearray)) or not isinstance(candidates, Sequence):
            raise RefusalPolicyError("candidate_registry_invalid")
        values: list[RankingEvidence] = []
        for candidate in cast(Sequence[object], candidates):
            if not isinstance(candidate, RankingEvidence):
                raise RefusalPolicyError("candidate_registry_invalid")
            values.append(candidate)
        ordered = tuple(sorted(values, key=lambda candidate: candidate.final_rank))
        ranks = tuple(candidate.final_rank for candidate in ordered)
        if ranks != tuple(range(1, len(ordered) + 1)):
            raise RefusalPolicyError("candidate_registry_invalid")
        registry = {candidate.chunk_id: candidate for candidate in ordered}
        if len(registry) != len(ordered) or any(
            candidate.revision_id != revision_id for candidate in ordered
        ):
            raise RefusalPolicyError("candidate_registry_invalid")
        return registry
