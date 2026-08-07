"""Fail-closed assembly of final evidence from a bound immutable revision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from rag_mvp.domain.ingestion import DocumentKind
from rag_mvp.domain.retrieval import RankingEvidence, RetrievalCandidate, RetrievalMode
from rag_mvp.retrieval.binding import BoundRetrievalSnapshot
from rag_mvp.retrieval.fusion import RrfConfig, validate_ranked_channel, weighted_rrf
from rag_mvp.retrieval.rerank import (
    RerankIntegrityError,
    RerankStageResult,
    validate_rerank_stage_result,
)
from rag_mvp.retrieval.snapshot import chunk_record_digest


class EvidenceIntegrityError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class EvidenceAssembler:
    """Resolve final candidates only through the snapshot's exact record registry."""

    def __init__(
        self,
        snapshot: BoundRetrievalSnapshot,
        source_kinds: Mapping[str, DocumentKind | str],
        *,
        final_limit: int,
        rrf: RrfConfig | None = None,
    ) -> None:
        if not isinstance(snapshot, BoundRetrievalSnapshot) or snapshot.is_closed:
            raise EvidenceIntegrityError("invalid_snapshot_binding")
        if type(final_limit) is not int or final_limit < 1:
            raise ValueError("final_limit_invalid")
        if not isinstance(source_kinds, Mapping):
            raise EvidenceIntegrityError("source_kind_mapping_invalid")
        normalized: dict[str, DocumentKind] = {}
        try:
            for source_id, raw_kind in source_kinds.items():
                if not isinstance(source_id, str) or not source_id:
                    raise EvidenceIntegrityError("source_kind_mapping_invalid")
                normalized[source_id] = DocumentKind(raw_kind)
        except (TypeError, ValueError):
            raise EvidenceIntegrityError("source_kind_mapping_invalid") from None
        if set(normalized) != set(snapshot.revision.active_sources):
            raise EvidenceIntegrityError("source_kind_mapping_incomplete")
        if normalized != dict(snapshot.source_kinds):
            raise EvidenceIntegrityError("source_kind_mapping_mismatch")
        self._snapshot = snapshot
        self._source_kinds = normalized
        self.final_limit = final_limit
        self._rrf = rrf or RrfConfig()
        self._records = {record.chunk.chunk_id: record for record in snapshot.bm25.records}

    def assemble(
        self,
        candidates: Sequence[RetrievalCandidate],
        *,
        mode: RetrievalMode,
        dense_candidates: Sequence[RetrievalCandidate] = (),
        bm25_candidates: Sequence[RetrievalCandidate] = (),
        fused_candidates: Sequence[RetrievalCandidate] = (),
        rerank_result: RerankStageResult | None = None,
    ) -> tuple[RankingEvidence, ...]:
        if self._snapshot.is_closed:
            raise EvidenceIntegrityError("snapshot_closed")
        try:
            resolved_mode = RetrievalMode(mode)
        except (TypeError, ValueError):
            raise EvidenceIntegrityError("retrieval_mode_invalid") from None
        bounded = _candidate_sequence(candidates)
        if len(bounded) > self.final_limit:
            raise EvidenceIntegrityError("final_limit_exceeded")
        if len({candidate.chunk_id for candidate in bounded}) != len(bounded):
            raise EvidenceIntegrityError("duplicate_final_candidate")

        dense = self._validated_channel(dense_candidates, "dense")
        bm25 = self._validated_channel(bm25_candidates, "bm25")
        dense_registry = {candidate.chunk_id: candidate for candidate in dense}
        bm25_registry = {candidate.chunk_id: candidate for candidate in bm25}
        fused = _candidate_sequence(fused_candidates)
        fused_registry = {candidate.chunk_id: candidate for candidate in fused}
        if len(fused_registry) != len(fused):
            raise EvidenceIntegrityError("duplicate_fused_candidate")

        if resolved_mode is RetrievalMode.DENSE:
            if bm25 or fused or rerank_result is not None:
                raise EvidenceIntegrityError("dense_stage_pollution")
            expected_registry = dense_registry
        else:
            if not fused and bounded:
                raise EvidenceIntegrityError("fused_candidates_missing")
            if rerank_result is not None and resolved_mode is not RetrievalMode.HYBRID_RERANK:
                raise EvidenceIntegrityError("rerank_mode_mismatch")
            if resolved_mode is RetrievalMode.HYBRID_RERANK and bounded and rerank_result is None:
                raise EvidenceIntegrityError("rerank_stage_result_missing")
            expected_fused = weighted_rrf(dense, bm25, config=self._rrf)
            if fused != expected_fused:
                raise EvidenceIntegrityError("fused_ranking_invalid")
            reranked = fused
            if rerank_result is not None:
                try:
                    reranked = validate_rerank_stage_result(fused, rerank_result)
                except RerankIntegrityError:
                    raise EvidenceIntegrityError("rerank_stage_result_invalid") from None
                if not rerank_result.applied or rerank_result.degraded:
                    raise EvidenceIntegrityError("unapplied_rerank_result")
            expected_order = reranked[: len(bounded)]
            if tuple(candidate.chunk_id for candidate in bounded) != tuple(
                candidate.chunk_id for candidate in expected_order
            ):
                raise EvidenceIntegrityError("final_order_invalid")
            expected_registry = fused_registry

        evidence: list[RankingEvidence] = []
        for final_rank, candidate in enumerate(bounded, start=1):
            self._validate_snapshot_record(candidate)
            expected = expected_registry.get(candidate.chunk_id)
            if expected is None:
                raise EvidenceIntegrityError("final_candidate_not_ranked")
            if resolved_mode is RetrievalMode.DENSE:
                if final_rank > len(dense) or dense[final_rank - 1].chunk_id != candidate.chunk_id:
                    raise EvidenceIntegrityError("dense_final_order_invalid")
                if candidate != expected:
                    raise EvidenceIntegrityError("dense_evidence_mutated")
                if (
                    candidate.bm25_rank is not None
                    or candidate.bm25_score is not None
                    or candidate.rrf_score is not None
                    or candidate.reranking_rank is not None
                ):
                    raise EvidenceIntegrityError("dense_score_pollution")
            else:
                self._validate_hybrid_candidate(
                    candidate,
                    expected,
                    dense_registry,
                    bm25_registry,
                    rerank_applied=rerank_result is not None,
                    final_rank=final_rank,
                )
            evidence.append(
                RankingEvidence.model_validate({**candidate.model_dump(), "final_rank": final_rank})
            )
        return tuple(evidence)

    def validate_cached_evidence(
        self,
        evidence: object,
    ) -> tuple[RankingEvidence, ...]:
        """Validate cached final evidence against this snapshot's record registry."""

        if self._snapshot.is_closed:
            raise EvidenceIntegrityError("snapshot_closed")
        if isinstance(evidence, (str, bytes, bytearray)) or not isinstance(evidence, Sequence):
            raise EvidenceIntegrityError("cached_evidence_sequence_invalid")
        raw_values = cast(Sequence[object], evidence)
        if len(raw_values) > self.final_limit:
            raise EvidenceIntegrityError("final_limit_exceeded")
        if any(not isinstance(item, RankingEvidence) for item in raw_values):
            raise EvidenceIntegrityError("cached_evidence_invalid")
        values = tuple(cast(RankingEvidence, item) for item in raw_values)
        if len({item.chunk_id for item in values}) != len(values):
            raise EvidenceIntegrityError("duplicate_final_candidate")
        if tuple(item.final_rank for item in values) != tuple(range(1, len(values) + 1)):
            raise EvidenceIntegrityError("cached_evidence_ranks_invalid")
        for item in values:
            self._validate_snapshot_record(item)
        return values

    def _validated_channel(
        self,
        candidates: Sequence[RetrievalCandidate],
        channel: str,
    ) -> tuple[RetrievalCandidate, ...]:
        try:
            return validate_ranked_channel(
                candidates,
                channel=channel,
                expected_revision_id=self._snapshot.revision_id,
                require_complete_identity=True,
                require_positional_ranks=True,
                require_scores=True,
            )
        except (TypeError, ValueError) as error:
            code = getattr(error, "code", f"{channel}_channel_invalid")
            raise EvidenceIntegrityError(str(code)) from None

    def _validate_snapshot_record(self, candidate: RetrievalCandidate) -> None:
        if candidate.revision_id != self._snapshot.revision_id:
            raise EvidenceIntegrityError("candidate_revision_mismatch")
        if any(
            value is None
            for value in (
                candidate.ordinal,
                candidate.content_digest,
                candidate.record_digest,
            )
        ):
            raise EvidenceIntegrityError("candidate_identity_incomplete")
        active_version = self._snapshot.revision.active_sources.get(candidate.source_id)
        if active_version is None or candidate.document_version != active_version:
            raise EvidenceIntegrityError("candidate_source_version_inactive")
        record = self._records.get(candidate.chunk_id)
        if record is None:
            raise EvidenceIntegrityError("candidate_not_in_snapshot")
        chunk = record.chunk
        if (
            candidate.source_id != chunk.source_id
            or candidate.parent_chunk_id != chunk.parent_chunk_id
            or candidate.document_version != chunk.document_version
            or candidate.ordinal != chunk.ordinal
            or candidate.text != chunk.text
            or candidate.locator != chunk.locator
            or candidate.content_digest != chunk.content_digest
            or candidate.display_title != record.display_title
            or candidate.record_digest != record.record_digest
            or candidate.record_digest != chunk_record_digest(chunk, record.display_title)
        ):
            raise EvidenceIntegrityError("candidate_snapshot_record_mismatch")
        kind = self._source_kinds[candidate.source_id]
        locator = candidate.locator
        if kind is DocumentKind.PDF:
            if not locator.pages:
                raise EvidenceIntegrityError("pdf_pages_missing")
        elif not locator.section_path and (locator.char_start is None or locator.char_end is None):
            raise EvidenceIntegrityError("text_locator_missing")

    @staticmethod
    def _validate_hybrid_candidate(
        candidate: RetrievalCandidate,
        expected_fused: RetrievalCandidate,
        dense_registry: Mapping[str, RetrievalCandidate],
        bm25_registry: Mapping[str, RetrievalCandidate],
        *,
        rerank_applied: bool,
        final_rank: int,
    ) -> None:
        expected_values = expected_fused.model_dump()
        actual_values = candidate.model_dump()
        actual_rerank_rank = actual_values.pop("reranking_rank")
        expected_values.pop("reranking_rank")
        if actual_values != expected_values:
            raise EvidenceIntegrityError("hybrid_evidence_mutated")
        if candidate.rrf_score is None:
            raise EvidenceIntegrityError("rrf_score_missing")
        dense = dense_registry.get(candidate.chunk_id)
        bm25 = bm25_registry.get(candidate.chunk_id)
        if dense is None:
            if candidate.dense_rank is not None or candidate.dense_score is not None:
                raise EvidenceIntegrityError("dense_score_pollution")
        elif candidate.dense_rank != dense.dense_rank or candidate.dense_score != dense.dense_score:
            raise EvidenceIntegrityError("dense_score_mismatch")
        if bm25 is None:
            if candidate.bm25_rank is not None or candidate.bm25_score is not None:
                raise EvidenceIntegrityError("bm25_score_pollution")
        elif candidate.bm25_rank != bm25.bm25_rank or candidate.bm25_score != bm25.bm25_score:
            raise EvidenceIntegrityError("bm25_score_mismatch")
        if rerank_applied:
            if actual_rerank_rank != final_rank:
                raise EvidenceIntegrityError("rerank_rank_inconsistent")
        elif actual_rerank_rank is not None:
            raise EvidenceIntegrityError("rerank_rank_pollution")


def assemble_legacy_evidence(
    candidates: Sequence[RetrievalCandidate],
    *,
    final_limit: int,
) -> tuple[RankingEvidence, ...]:
    """Minimal compatibility path for isolated, provenance-free test doubles."""

    if type(final_limit) is not int or final_limit < 1:
        raise ValueError("final_limit_invalid")
    bounded = _candidate_sequence(candidates)[:final_limit]
    if len({candidate.chunk_id for candidate in bounded}) != len(bounded):
        raise EvidenceIntegrityError("duplicate_final_candidate")
    return tuple(
        RankingEvidence.model_validate({**candidate.model_dump(), "final_rank": rank})
        for rank, candidate in enumerate(bounded, start=1)
    )


def _candidate_sequence(
    candidates: object,
) -> tuple[RetrievalCandidate, ...]:
    if isinstance(candidates, (str, bytes, bytearray)) or not isinstance(candidates, Sequence):
        raise EvidenceIntegrityError("candidate_sequence_invalid")
    raw_candidates = cast(Sequence[object], candidates)
    values: list[RetrievalCandidate] = []
    for candidate in raw_candidates:
        if not isinstance(candidate, RetrievalCandidate):
            raise EvidenceIntegrityError("candidate_invalid")
        try:
            values.append(RetrievalCandidate.model_validate(candidate.model_dump()))
        except (TypeError, ValueError, RerankIntegrityError):
            raise EvidenceIntegrityError("candidate_invalid") from None
    return tuple(values)
