from __future__ import annotations

import pytest

from rag_mvp.domain.ingestion import ChunkLocator
from rag_mvp.domain.retrieval import RankingEvidence
from rag_mvp.providers.models import (
    Deadline,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingSpaceIdentity,
    NormalizationPolicy,
    ProviderCallContext,
)
from rag_mvp.qa.evidence_assessor import (
    EvidenceAssessmentError,
    EvidenceLifecycleStatus,
    FactAssessmentConfig,
    SemanticFactEvidenceAssessor,
)

IDENTITY = EmbeddingSpaceIdentity(
    provider="test",
    model="semantic-v1",
    dimension=2,
    normalization=NormalizationPolicy.L2,
    adapter_version="v1",
)


class MappingEmbeddingProvider:
    identity = IDENTITY

    def __init__(self, vectors: dict[str, tuple[float, float]]) -> None:
        self.vectors = vectors
        self.calls = 0
        self.requests: list[EmbeddingRequest] = []
        self.contexts: list[ProviderCallContext] = []

    async def embed(
        self,
        request: EmbeddingRequest,
        context: ProviderCallContext,
    ) -> EmbeddingResult:
        self.calls += 1
        self.requests.append(request)
        self.contexts.append(context)
        return EmbeddingResult(
            vectors=tuple(self.vectors[text] for text in request.texts),
            identity=self.identity,
        )


def _candidate(
    chunk_id: str,
    rank: int,
    text: str,
    *,
    source_id: str | None = None,
    revision_id: str = "revision-current",
) -> RankingEvidence:
    return RankingEvidence(
        chunk_id=chunk_id,
        parent_chunk_id=chunk_id,
        source_id=source_id or f"source-{rank}",
        display_title=f"Policy {rank}",
        document_version=1,
        locator=ChunkLocator(pages=(rank,)),
        text=text,
        revision_id=revision_id,
        final_rank=rank,
    )


def _deadline() -> Deadline:
    return Deadline.after(10)


async def test_cross_language_semantic_support_uses_dedicated_embeddings() -> None:
    chinese_fact = "年假有多少天"
    english_evidence = "Employees receive ten days of annual leave."
    unrelated = "The office network maintenance schedule."
    provider = MappingEmbeddingProvider(
        {
            chinese_fact: (1.0, 0.0),
            english_evidence: (1.0, 0.0),
            unrelated: (0.0, 1.0),
        }
    )
    assessor = SemanticFactEvidenceAssessor(
        provider,
        config=FactAssessmentConfig(candidate_similarity_floor=0.6),
    )

    result = await assessor.assess(
        "年假有多少天\uff1f",
        (
            _candidate("chunk-leave", 1, english_evidence),
            _candidate("chunk-network", 2, unrelated),
        ),
        request_id="request-1",
        revision_id="revision-current",
        deadline=_deadline(),
    )

    assert result[0].support_score == 1
    assert result[0].supporting_chunk_ids == ("chunk-leave",)
    assert result[0].conflicting_chunk_ids == ()
    assert provider.requests[0].texts == (chinese_fact, english_evidence, unrelated)


async def test_successful_fact_assessment_reports_direct_provider_usage_state() -> None:
    fact = "What is the leave allowance"
    evidence = "Employees receive ten days of annual leave."
    assessor = SemanticFactEvidenceAssessor(
        MappingEmbeddingProvider({fact: (1.0, 0.0), evidence: (1.0, 0.0)})
    )

    result = await assessor.assess_with_diagnostics(
        f"{fact}?",
        (_candidate("chunk-leave", 1, evidence),),
        request_id="request-1",
        revision_id="revision-current",
        deadline=_deadline(),
    )

    assert result.facts[0].supporting_chunk_ids == ("chunk-leave",)
    assert result.provider_attempt_count == 1
    assert result.provider_failed_attempt_count == 0
    assert result.provider_unknown_usage_attempt_count == 1
    assert result.direct_provider_identity is not None


async def test_multi_fact_query_can_produce_partial_support() -> None:
    leave_fact = "What is the leave allowance"
    stipend_fact = "what is the remote stipend"
    leave_evidence = "The leave allowance is ten days."
    provider = MappingEmbeddingProvider(
        {
            leave_fact: (1.0, 0.0),
            stipend_fact: (0.0, 1.0),
            leave_evidence: (1.0, 0.0),
        }
    )
    assessor = SemanticFactEvidenceAssessor(
        provider,
        config=FactAssessmentConfig(candidate_similarity_floor=0.6),
    )

    result = await assessor.assess(
        "What is the leave allowance; what is the remote stipend?",
        (_candidate("chunk-leave", 1, leave_evidence),),
        request_id="request-1",
        revision_id="revision-current",
        deadline=_deadline(),
    )

    assert [fact.fact_id for fact in result] == ["fact-1", "fact-2"]
    assert result[0].supporting_chunk_ids == ("chunk-leave",)
    assert result[1].support_score == 0
    assert result[1].supporting_chunk_ids == ()


async def test_raw_retrieval_scores_do_not_affect_support_decision() -> None:
    fact = "What is the leave allowance"
    relevant = _candidate("chunk-relevant", 1, "Relevant evidence")
    irrelevant = _candidate("chunk-irrelevant", 2, "Irrelevant evidence")
    relevant = relevant.model_copy(
        update={"dense_score": -999.0, "bm25_score": -999.0, "rrf_score": 0.0}
    )
    irrelevant = irrelevant.model_copy(
        update={"dense_score": 999.0, "bm25_score": 999.0, "rrf_score": 999.0}
    )
    provider = MappingEmbeddingProvider(
        {
            fact: (1.0, 0.0),
            relevant.text: (1.0, 0.0),
            irrelevant.text: (0.0, 1.0),
        }
    )

    result = await SemanticFactEvidenceAssessor(provider).assess(
        f"{fact}?",
        (relevant, irrelevant),
        request_id="request-1",
        revision_id="revision-current",
        deadline=_deadline(),
    )

    assert result[0].supporting_chunk_ids == ("chunk-relevant",)


async def test_different_single_values_across_sources_are_material_conflict() -> None:
    fact = "How many annual leave days"
    ten_days = "Employees receive 10 days of annual leave."
    fifteen_days = "Employees receive 15 days of annual leave."
    provider = MappingEmbeddingProvider(
        {fact: (1.0, 0.0), ten_days: (1.0, 0.0), fifteen_days: (0.99, 0.01)}
    )

    result = await SemanticFactEvidenceAssessor(provider).assess(
        f"{fact}?",
        (
            _candidate("chunk-ten", 1, ten_days, source_id="source-a"),
            _candidate("chunk-fifteen", 2, fifteen_days, source_id="source-b"),
        ),
        request_id="request-1",
        revision_id="revision-current",
        deadline=_deadline(),
    )

    assert result[0].supporting_chunk_ids == ("chunk-ten",)
    assert result[0].conflicting_chunk_ids == ("chunk-fifteen",)


async def test_same_source_values_are_not_automatically_treated_as_conflict() -> None:
    fact = "How many annual leave days"
    base = "Base allowance is 10 days."
    exception = "Senior employees receive 15 days."
    provider = MappingEmbeddingProvider(
        {fact: (1.0, 0.0), base: (1.0, 0.0), exception: (0.99, 0.01)}
    )

    result = await SemanticFactEvidenceAssessor(provider).assess(
        f"{fact}?",
        (
            _candidate("chunk-base", 1, base, source_id="source-policy"),
            _candidate("chunk-exception", 2, exception, source_id="source-policy"),
        ),
        request_id="request-1",
        revision_id="revision-current",
        deadline=_deadline(),
    )

    assert result[0].conflicting_chunk_ids == ()
    assert result[0].supporting_chunk_ids == ("chunk-base", "chunk-exception")


async def test_explicit_opposite_policy_assertions_are_conflict() -> None:
    fact = "Is remote work allowed"
    allowed = "Remote work is allowed."
    prohibited = "Remote work is prohibited."
    provider = MappingEmbeddingProvider(
        {fact: (1.0, 0.0), allowed: (1.0, 0.0), prohibited: (0.99, 0.01)}
    )

    result = await SemanticFactEvidenceAssessor(provider).assess(
        f"{fact}?",
        (
            _candidate("chunk-allowed", 1, allowed, source_id="source-a"),
            _candidate("chunk-prohibited", 2, prohibited, source_id="source-b"),
        ),
        request_id="request-1",
        revision_id="revision-current",
        deadline=_deadline(),
    )

    assert result[0].conflicting_chunk_ids == ("chunk-prohibited",)


async def test_rerank_bounded_current_evidence_supersedes_withdrawn_draft() -> None:
    fact = "Which travel identifier is authoritative rather than the draft"
    withdrawn = "Withdrawn draft: the cap is 2800 yuan."
    current = (
        "\u672c\u6587\u4ef6\u662f\u5f53\u524d\u6709\u6548\u7684\u5dee\u65c5\u653f\u7b56; "
        "the cap is 1800 yuan."
    )
    current_lifecycle = (
        "\u672c\u6587\u4ef6\u662f\u5f53\u524d\u6709\u6548\u7684\u5dee\u65c5\u653f\u7b56;"
    )
    current_cap = "the cap is 1800 yuan."
    unrelated = "The authoritative RAG escalation code is OPS-RAG-7421."
    provider = MappingEmbeddingProvider(
        {
            fact: (1.0, 0.0),
            withdrawn: (0.8, 0.6),
            current_lifecycle: (0.31, 0.95074),
            current_cap: (0.35, 0.93675),
            unrelated: (0.7, 0.71414),
        }
    )
    assessor = SemanticFactEvidenceAssessor(provider)

    result = await assessor.assess_with_diagnostics(
        f"{fact}?",
        (
            _candidate("chunk-draft", 1, withdrawn, source_id="source-draft"),
            _candidate("chunk-current", 2, current, source_id="source-current"),
            _candidate("chunk-unrelated", 3, unrelated, source_id="source-other"),
        ),
        request_id="request-authority",
        revision_id="revision-current",
        deadline=_deadline(),
    )

    assert result.facts[0].support_score == 0.5
    assert result.facts[0].supporting_chunk_ids == ("chunk-current",)
    assert result.facts[0].conflicting_chunk_ids == ()
    assert result.authority_resolution_count == 1
    assert {
        item.chunk_id: (item.status, item.authority_level) for item in result.authority_metadata
    } == {
        "chunk-draft": (EvidenceLifecycleStatus.WITHDRAWN, 0),
        "chunk-current": (EvidenceLifecycleStatus.CURRENT, 2),
        "chunk-unrelated": (EvidenceLifecycleStatus.UNSPECIFIED, 1),
    }


async def test_unqualified_withdrawn_selection_retains_authority_conflict() -> None:
    fact = "Choose the withdrawn draft amount without qualification"
    withdrawn = "Withdrawn draft: the cap is 2800 yuan."
    current = "This is the current authoritative policy; the cap is 1800 yuan."
    current_lifecycle = "This is the current authoritative policy;"
    current_cap = "the cap is 1800 yuan."
    provider = MappingEmbeddingProvider(
        {
            fact: (1.0, 0.0),
            withdrawn: (0.8, 0.6),
            current_lifecycle: (0.31, 0.95074),
            current_cap: (0.35, 0.93675),
        }
    )

    result = await SemanticFactEvidenceAssessor(provider).assess(
        f"{fact}.",
        (
            _candidate("chunk-draft", 1, withdrawn, source_id="source-draft"),
            _candidate("chunk-current", 2, current, source_id="source-current"),
        ),
        request_id="request-withdrawn",
        revision_id="revision-current",
        deadline=_deadline(),
    )

    assert result[0].supporting_chunk_ids == ("chunk-current",)
    assert result[0].conflicting_chunk_ids == ("chunk-draft",)


async def test_response_language_instruction_is_not_assessed_as_a_fact() -> None:
    fact = (
        "\u77e5\u8bc6\u67e5\u8be2\u63a5\u53e3\u8981\u6c42\u54ea\u4e2a\u79df\u6237\u8bf7\u6c42\u5934"
    )
    evidence = "Every request must carry the tenant header X-Atlas-Tenant."
    provider = MappingEmbeddingProvider({fact: (1.0, 0.0), evidence: (1.0, 0.0)})

    result = await SemanticFactEvidenceAssessor(provider).assess(
        f"{fact}? \u8bf7\u7528\u4e2d\u6587\u56de\u7b54\u3002",
        (_candidate("chunk-header", 1, evidence),),
        request_id="request-language",
        revision_id="revision-current",
        deadline=_deadline(),
    )

    assert len(result) == 1
    assert result[0].supporting_chunk_ids == ("chunk-header",)


async def test_conflict_uses_matching_assertions_not_unrelated_chunk_polarity() -> None:
    fact = "What tenant header is required"
    header = "Every request must carry X-Atlas-Tenant."
    unrelated_prohibition = "The service must not expose local filesystem paths."
    other_policy = "Evaluation jobs must use an isolated data root."
    primary = f"{header} {unrelated_prohibition}"
    provider = MappingEmbeddingProvider(
        {
            fact: (1.0, 0.0),
            header: (1.0, 0.0),
            unrelated_prohibition: (0.0, 1.0),
            other_policy: (0.99, 0.01),
        }
    )

    result = await SemanticFactEvidenceAssessor(provider).assess(
        f"{fact}?",
        (
            _candidate("chunk-header", 1, primary, source_id="source-header"),
            _candidate("chunk-other", 2, other_policy, source_id="source-other"),
        ),
        request_id="request-assertion-conflict",
        revision_id="revision-current",
        deadline=_deadline(),
    )

    assert result[0].supporting_chunk_ids == ("chunk-header", "chunk-other")
    assert result[0].conflicting_chunk_ids == ()


async def test_chinese_semicolon_separates_candidate_assertions() -> None:
    fact = "How does a validated index become the active revision"
    raw_unrelated = "在线索引使用不可变修订\uff1b"
    normalized_unrelated = "在线索引使用不可变修订;"
    supporting = "新索引校验后通过一次原子操作切换活动修订。"
    provider = MappingEmbeddingProvider(
        {
            fact: (1.0, 0.0),
            normalized_unrelated: (0.0, 1.0),
            supporting: (0.8, 0.6),
        }
    )

    result = await SemanticFactEvidenceAssessor(provider).assess(
        f"{fact}?",
        (_candidate("chunk-architecture", 1, f"{raw_unrelated}{supporting}"),),
        request_id="request-cjk-semicolon",
        revision_id="revision-current",
        deadline=_deadline(),
    )

    assert result[0].supporting_chunk_ids == ("chunk-architecture",)
    assert result[0].support_score == 0.8


async def test_weak_assertion_match_remains_unsupported() -> None:
    fact = "What is the private board calendar"
    unrelated = "The production API has a stable specification identifier."
    provider = MappingEmbeddingProvider({fact: (1.0, 0.0), unrelated: (0.46, 0.88792)})

    result = await SemanticFactEvidenceAssessor(provider).assess(
        f"{fact}?",
        (_candidate("chunk-unrelated", 1, unrelated),),
        request_id="request-weak-assertion",
        revision_id="revision-current",
        deadline=_deadline(),
    )

    assert result[0].support_score == 0
    assert result[0].supporting_chunk_ids == ()


async def test_stale_candidate_registry_fails_before_provider_call() -> None:
    provider = MappingEmbeddingProvider({})
    assessor = SemanticFactEvidenceAssessor(provider)

    with pytest.raises(EvidenceAssessmentError, match="candidate_registry_invalid"):
        await assessor.assess(
            "Question?",
            (_candidate("chunk-1", 1, "Evidence", revision_id="revision-old"),),
            request_id="request-1",
            revision_id="revision-current",
            deadline=_deadline(),
        )

    assert provider.calls == 0


async def test_expired_deadline_fails_before_provider_call() -> None:
    provider = MappingEmbeddingProvider({})
    assessor = SemanticFactEvidenceAssessor(provider)
    clock_value = 10.0
    deadline = Deadline(clock_value, lambda: clock_value)

    with pytest.raises(TimeoutError):
        await assessor.assess(
            "Question?",
            (_candidate("chunk-1", 1, "Evidence"),),
            request_id="request-1",
            revision_id="revision-current",
            deadline=deadline,
        )

    assert provider.calls == 0


@pytest.mark.parametrize("floor", [0, -0.1, 1.1, float("nan"), True])
def test_invalid_similarity_calibration_is_rejected(floor: object) -> None:
    with pytest.raises(ValueError, match="candidate_similarity_floor"):
        FactAssessmentConfig(candidate_similarity_floor=floor)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    ["assertion_similarity_floor", "authority_similarity_floor", "authority_score_bonus"],
)
@pytest.mark.parametrize("value", [0, -0.1, 1.1, float("nan"), True])
def test_invalid_authority_calibration_is_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        FactAssessmentConfig(**{field: value})  # type: ignore[arg-type]
