"""Dense, lexical, and reranked retrieval."""

from rag_mvp.domain.retrieval import CachePolicy
from rag_mvp.retrieval.binding import BoundRetrievalSnapshot, BoundRetrievalSnapshotFactory
from rag_mvp.retrieval.bm25 import LexicalIndexError, PersistentBm25Index
from rag_mvp.retrieval.cache import (
    RETRIEVAL_CACHE_IDENTITY_VERSION,
    BoundedTtlCache,
    RetrievalCacheIdentity,
)
from rag_mvp.retrieval.collection import (
    BoundBm25Retriever,
    HybridCandidateCollection,
    HybridCollectionError,
    RevisionBoundRetriever,
    collect_hybrid_candidates,
)
from rag_mvp.retrieval.dense import DenseIndexError, PersistentChromaIndex
from rag_mvp.retrieval.evidence import EvidenceAssembler, EvidenceIntegrityError
from rag_mvp.retrieval.fusion import (
    CandidateIntegrityError,
    RrfConfig,
    merge_ranked_candidates,
    validate_ranked_channel,
    weighted_rrf,
)
from rag_mvp.retrieval.identity import (
    EmbeddingIdentityError,
    domain_embedding_identity,
    provider_embedding_identity,
)
from rag_mvp.retrieval.query_dense import (
    BoundDenseRetriever,
    DenseSearchResult,
    QueryDenseRetriever,
)
from rag_mvp.retrieval.request import (
    QUERY_CANONICALIZATION_VERSION,
    RetrievalRequestContext,
    RetrievalRequestError,
    canonicalize_query,
)
from rag_mvp.retrieval.rerank import (
    RerankIntegrityError,
    RerankStage,
    RerankStageResult,
    RerankTruncationPolicy,
    validate_rerank_stage_result,
)
from rag_mvp.retrieval.service import (
    DEGRADATION_POLICY_VERSION,
    RRF_TIE_POLICY_VERSION,
    RetrievalLimits,
    RetrievalService,
    RetrievalUnavailableError,
)
from rag_mvp.retrieval.snapshot import (
    CHUNK_SET_DIGEST_ALGORITHM,
    RECORD_DIGEST_ALGORITHM,
    chunk_record_digest,
    chunk_set_digest,
)
from rag_mvp.retrieval.tokenizer import BILINGUAL_TOKENIZER_IDENTITY, BilingualTokenizer

__all__ = [
    "BILINGUAL_TOKENIZER_IDENTITY",
    "CHUNK_SET_DIGEST_ALGORITHM",
    "DEGRADATION_POLICY_VERSION",
    "QUERY_CANONICALIZATION_VERSION",
    "RECORD_DIGEST_ALGORITHM",
    "RETRIEVAL_CACHE_IDENTITY_VERSION",
    "RRF_TIE_POLICY_VERSION",
    "BilingualTokenizer",
    "BoundBm25Retriever",
    "BoundDenseRetriever",
    "BoundRetrievalSnapshot",
    "BoundRetrievalSnapshotFactory",
    "BoundedTtlCache",
    "CachePolicy",
    "CandidateIntegrityError",
    "DenseIndexError",
    "DenseSearchResult",
    "EmbeddingIdentityError",
    "EvidenceAssembler",
    "EvidenceIntegrityError",
    "HybridCandidateCollection",
    "HybridCollectionError",
    "LexicalIndexError",
    "PersistentBm25Index",
    "PersistentChromaIndex",
    "QueryDenseRetriever",
    "RerankIntegrityError",
    "RerankStage",
    "RerankStageResult",
    "RerankTruncationPolicy",
    "RetrievalCacheIdentity",
    "RetrievalLimits",
    "RetrievalRequestContext",
    "RetrievalRequestError",
    "RetrievalService",
    "RetrievalUnavailableError",
    "RevisionBoundRetriever",
    "RrfConfig",
    "canonicalize_query",
    "chunk_record_digest",
    "chunk_set_digest",
    "collect_hybrid_candidates",
    "domain_embedding_identity",
    "merge_ranked_candidates",
    "provider_embedding_identity",
    "validate_ranked_channel",
    "validate_rerank_stage_result",
    "weighted_rrf",
]
