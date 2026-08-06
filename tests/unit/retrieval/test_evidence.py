from __future__ import annotations

from pathlib import Path

import pytest
from retrieval_test_helpers import build_bound_snapshot, candidate, indexed_chunk

from rag_mvp.domain.ingestion import ChunkLocator, DocumentKind
from rag_mvp.domain.retrieval import RetrievalCandidate, RetrievalMode
from rag_mvp.providers.fakes import (
    DeterministicEmbeddingProvider,
    DeterministicRerankingProvider,
)
from rag_mvp.providers.models import Deadline, ProviderCallContext
from rag_mvp.retrieval.evidence import EvidenceAssembler, EvidenceIntegrityError
from rag_mvp.retrieval.fusion import weighted_rrf
from rag_mvp.retrieval.identity import provider_embedding_identity
from rag_mvp.retrieval.query_dense import BoundDenseRetriever
from rag_mvp.retrieval.request import RetrievalRequestContext
from rag_mvp.retrieval.rerank import RerankStage
from rag_mvp.retrieval.service import RetrievalService


class StaticRetriever:
    async def search(self, query: str, limit: int) -> tuple[object, ...]:
        del query, limit
        return (candidate("pdf-evidence", dense_rank=1, page=7),)


def _context() -> ProviderCallContext:
    return ProviderCallContext("request", "retrieval", Deadline.after(10))


async def _ranked_channels(
    snapshot: object,
) -> tuple[tuple[RetrievalCandidate, ...], tuple[RetrievalCandidate, ...]]:
    provider = DeterministicEmbeddingProvider(
        provider_embedding_identity(snapshot.revision.embedding_space)  # type: ignore[union-attr]
    )
    dense = await BoundDenseRetriever(snapshot, provider, _context()).search("policy", 10)  # type: ignore[arg-type]
    bm25 = await snapshot.bm25.search("policy", 10)  # type: ignore[union-attr]
    return dense, bm25


async def test_legacy_evidence_contains_exact_text_locator_and_real_scores() -> None:
    service = RetrievalService(dense=StaticRetriever(), lexical=StaticRetriever())  # type: ignore[arg-type]

    result = await service.retrieve(
        RetrievalRequestContext("req", "policy", RetrievalMode.DENSE, "rev-1")
    )

    evidence = result.evidence[0]
    assert evidence.text == "Evidence for pdf-evidence"
    assert evidence.locator.pages == (7,)
    assert evidence.final_rank == 1
    assert evidence.dense_score == 1.0
    assert evidence.bm25_score is None


async def test_pdf_and_text_evidence_preserve_exact_bound_records_and_stage_scores(
    tmp_path: Path,
) -> None:
    pdf_text = "  PDF policy evidence\n"
    markdown_text = "# Policy\nExact markdown evidence\n"
    chunks = (
        indexed_chunk(
            "chunk-pdf",
            pdf_text,
            source_id="source-pdf",
            ordinal=0,
            locator=ChunkLocator(pages=(3, 4)),
        ),
        indexed_chunk(
            "chunk-markdown",
            markdown_text,
            source_id="source-markdown",
            ordinal=0,
            locator=ChunkLocator(section_path=("Policy",), char_start=0, char_end=33),
        ),
    )
    snapshot, source_kinds = await build_bound_snapshot(
        tmp_path,
        chunks=chunks,
        titles={"source-pdf": "PDF Policy", "source-markdown": "Markdown Policy"},
        source_kinds={
            "source-pdf": DocumentKind.PDF,
            "source-markdown": DocumentKind.MARKDOWN,
        },
    )
    dense, bm25 = await _ranked_channels(snapshot)
    fused = weighted_rrf(dense, bm25)
    assembler = EvidenceAssembler(snapshot, source_kinds, final_limit=2)

    evidence = assembler.assemble(
        fused[:2],
        mode=RetrievalMode.HYBRID,
        dense_candidates=dense,
        bm25_candidates=bm25,
        fused_candidates=fused,
    )

    by_id = {item.chunk_id: item for item in evidence}
    assert by_id["chunk-pdf"].text == pdf_text
    assert by_id["chunk-pdf"].display_title == "PDF Policy"
    assert by_id["chunk-pdf"].locator.pages == (3, 4)
    assert by_id["chunk-markdown"].text == markdown_text
    assert by_id["chunk-markdown"].locator.section_path == ("Policy",)
    assert all(item.rrf_score is not None for item in evidence)
    assert [item.final_rank for item in evidence] == [1, 2]
    snapshot.close()


async def test_dense_evidence_has_only_dense_fields(tmp_path: Path) -> None:
    snapshot, source_kinds = await build_bound_snapshot(tmp_path)
    dense, _ = await _ranked_channels(snapshot)
    assembler = EvidenceAssembler(snapshot, source_kinds, final_limit=2)

    evidence = assembler.assemble(
        dense[:2],
        mode=RetrievalMode.DENSE,
        dense_candidates=dense,
    )

    assert all(item.dense_rank is not None and item.dense_score is not None for item in evidence)
    assert all(
        item.bm25_rank is None
        and item.bm25_score is None
        and item.rrf_score is None
        and item.reranking_rank is None
        for item in evidence
    )
    snapshot.close()


async def test_rerank_rank_survives_only_with_validated_applied_stage_result(
    tmp_path: Path,
) -> None:
    snapshot, source_kinds = await build_bound_snapshot(tmp_path)
    dense, bm25 = await _ranked_channels(snapshot)
    fused = weighted_rrf(dense, bm25)
    rerank = await RerankStage(
        DeterministicRerankingProvider(),
        candidate_limit=2,
    ).run("policy", fused, _context())
    assembler = EvidenceAssembler(snapshot, source_kinds, final_limit=2)

    evidence = assembler.assemble(
        rerank.ordered_candidates[:2],
        mode=RetrievalMode.HYBRID_RERANK,
        dense_candidates=dense,
        bm25_candidates=bm25,
        fused_candidates=fused,
        rerank_result=rerank,
    )

    assert [item.reranking_rank for item in evidence] == [1, 2]
    with pytest.raises(
        EvidenceIntegrityError,
        match=r"final_order_invalid|rerank_rank_pollution",
    ):
        assembler.assemble(
            rerank.ordered_candidates[:2],
            mode=RetrievalMode.HYBRID,
            dense_candidates=dense,
            bm25_candidates=bm25,
            fused_candidates=fused,
        )
    snapshot.close()


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ({"text": "fabricated"}, "candidate_snapshot_record_mismatch"),
        ({"revision_id": "other-revision"}, "candidate_revision_mismatch"),
        ({"document_version": 2}, "candidate_source_version_inactive"),
        ({"record_digest": None}, "candidate_identity_incomplete"),
    ],
)
async def test_evidence_rejects_stale_incomplete_or_mutated_identity(
    tmp_path: Path,
    mutation: dict[str, object],
    code: str,
) -> None:
    snapshot, source_kinds = await build_bound_snapshot(tmp_path)
    dense, _ = await _ranked_channels(snapshot)
    polluted = RetrievalCandidate.model_validate({**dense[0].model_dump(), **mutation})
    assembler = EvidenceAssembler(snapshot, source_kinds, final_limit=2)

    with pytest.raises(EvidenceIntegrityError, match=code):
        assembler.assemble(
            (polluted,),
            mode=RetrievalMode.DENSE,
            dense_candidates=dense,
        )
    snapshot.close()


async def test_evidence_rejects_score_pollution_and_fabricated_rrf(tmp_path: Path) -> None:
    snapshot, source_kinds = await build_bound_snapshot(tmp_path)
    dense, bm25 = await _ranked_channels(snapshot)
    assembler = EvidenceAssembler(snapshot, source_kinds, final_limit=2)
    polluted_dense = RetrievalCandidate.model_validate(
        {**dense[0].model_dump(), "bm25_rank": 1, "bm25_score": 99.0}
    )
    with pytest.raises(EvidenceIntegrityError, match="dense_provenance_invalid"):
        assembler.assemble(
            (polluted_dense,),
            mode=RetrievalMode.DENSE,
            dense_candidates=(polluted_dense,),
        )

    fused = weighted_rrf(dense, bm25)
    fabricated = RetrievalCandidate.model_validate(
        {**fused[0].model_dump(), "rrf_score": fused[0].rrf_score + 1.0}  # type: ignore[operator]
    )
    with pytest.raises(EvidenceIntegrityError, match="fused_ranking_invalid"):
        assembler.assemble(
            (fabricated,),
            mode=RetrievalMode.HYBRID,
            dense_candidates=dense,
            bm25_candidates=bm25,
            fused_candidates=(fabricated, *fused[1:]),
        )
    snapshot.close()


async def test_evidence_rejects_invalid_locator_for_safe_source_kind(tmp_path: Path) -> None:
    pdf_with_section_only = indexed_chunk(
        "chunk-pdf",
        "PDF policy",
        source_id="source-pdf",
        locator=ChunkLocator(section_path=("Policy",)),
    )
    snapshot, source_kinds = await build_bound_snapshot(
        tmp_path,
        chunks=(pdf_with_section_only,),
        titles={"source-pdf": "PDF Policy"},
        source_kinds={"source-pdf": DocumentKind.PDF},
    )
    dense, _ = await _ranked_channels(snapshot)

    with pytest.raises(EvidenceIntegrityError, match="pdf_pages_missing"):
        EvidenceAssembler(snapshot, source_kinds, final_limit=1).assemble(
            dense,
            mode=RetrievalMode.DENSE,
            dense_candidates=dense,
        )
    snapshot.close()


async def test_evidence_rejects_duplicate_order_and_final_limit(tmp_path: Path) -> None:
    snapshot, source_kinds = await build_bound_snapshot(tmp_path)
    dense, _ = await _ranked_channels(snapshot)
    assembler = EvidenceAssembler(snapshot, source_kinds, final_limit=1)

    with pytest.raises(EvidenceIntegrityError, match="final_limit_exceeded"):
        assembler.assemble(
            dense[:2],
            mode=RetrievalMode.DENSE,
            dense_candidates=dense,
        )
    duplicate_assembler = EvidenceAssembler(snapshot, source_kinds, final_limit=2)
    with pytest.raises(EvidenceIntegrityError, match="duplicate_final_candidate"):
        duplicate_assembler.assemble(
            (dense[0], dense[0]),
            mode=RetrievalMode.DENSE,
            dense_candidates=dense,
        )
    snapshot.close()
