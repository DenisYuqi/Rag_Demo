from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

from rag_mvp.api.app import create_app
from rag_mvp.api.qa import NDJSON_MEDIA_TYPE, QARuntimeServices
from rag_mvp.config.settings import Settings
from rag_mvp.domain.ingestion import (
    Chunk,
    ChunkLocator,
    Document,
    DocumentKind,
    DocumentVersion,
    ExtractionMethod,
    ParentChunk,
)
from rag_mvp.domain.qa import ConversationRole, StreamEventKind, ValidatedStreamEvent
from rag_mvp.ingestion.chunking import token_spans
from rag_mvp.ingestion.embedding import EmbeddingStage
from rag_mvp.ingestion.indexing import RevisionPublisher, RevisionStager
from rag_mvp.performance.admission import QAAdmissionController
from rag_mvp.providers.models import (
    Deadline,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingSpaceIdentity,
    FinishReason,
    GenerationFormat,
    GenerationRequest,
    GenerationResult,
    ModelIdentity,
    NormalizationPolicy,
    ProviderCallContext,
    TokenUsage,
)
from rag_mvp.providers.resilience import RetryPolicy
from rag_mvp.providers.routing import ModelProviderRouter, ProviderRoute
from rag_mvp.qa.context import ContextBuilder
from rag_mvp.qa.deadlines import QAStageBudgets
from rag_mvp.qa.evidence_assessor import FactAssessmentConfig, SemanticFactEvidenceAssessor
from rag_mvp.qa.orchestrator import QAOrchestrator, SnapshotRetrievalGateway
from rag_mvp.qa.sessions import ConversationService
from rag_mvp.qa.streaming import CompleteResponseEmitter
from rag_mvp.retrieval.binding import BoundRetrievalSnapshotFactory
from rag_mvp.safety.injection import InjectionPolicy
from rag_mvp.storage.database import Database
from rag_mvp.storage.embedding_cache import EmbeddingCache
from rag_mvp.storage.layout import DataLayout
from rag_mvp.storage.repositories import KnowledgeRepositories, SessionRepository

pytestmark = pytest.mark.integration

_SOURCE_TEXT = "Employees receive ten days of annual leave."
_ANSWER = "Employees receive ten days of annual leave."
_CHUNK_ID = "chunk-concurrency-leave"
_SPACE = EmbeddingSpaceIdentity(
    provider="concurrency-proof",
    model="constant-concept-v1",
    dimension=4,
    normalization=NormalizationPolicy.L2,
    adapter_version="integration-v1",
)
_GENERATION_IDENTITY = ModelIdentity(
    provider="concurrency-proof",
    model="gated-structured-generation-v1",
    adapter_version="integration-v1",
)


class DeterministicEmbeddingProvider:
    """Provide one stable semantic space across indexing and live retrieval."""

    identity = _SPACE

    def __init__(self) -> None:
        self.requests: list[EmbeddingRequest] = []

    async def embed(
        self,
        request: EmbeddingRequest,
        context: ProviderCallContext,
    ) -> EmbeddingResult:
        if context.deadline.expired:
            raise TimeoutError
        self.requests.append(request)
        return EmbeddingResult(
            vectors=tuple((1.0, 0.0, 0.0, 0.0) for _ in request.texts),
            identity=self.identity,
            usage=TokenUsage(
                input_tokens=sum(max(1, len(text) // 4) for text in request.texts),
                output_tokens=0,
            ),
        )


class FiveWayGenerationProvider:
    """Hold the fake provider call until five real orchestrations overlap."""

    identity = _GENERATION_IDENTITY

    def __init__(self, expected: int = 5) -> None:
        self.expected = expected
        self.active = 0
        self.peak = 0
        self.requests: list[GenerationRequest] = []
        self.all_entered = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(
        self,
        request: GenerationRequest,
        context: ProviderCallContext,
    ) -> GenerationResult:
        if context.deadline.expired:
            raise TimeoutError
        assert request.response_format is GenerationFormat.JSON_OBJECT
        payload = json.loads(request.messages[-1].content)
        assert isinstance(payload, dict)
        allowed_ids = payload.get("allowed_chunk_ids")
        assert isinstance(allowed_ids, list) and allowed_ids == [_CHUNK_ID]

        self.requests.append(request)
        self.active += 1
        self.peak = max(self.peak, self.active)
        if self.active == self.expected:
            self.all_entered.set()
        try:
            await self.release.wait()
        finally:
            self.active -= 1

        return GenerationResult(
            content=json.dumps(
                {
                    "schema_version": "grounded-answer-v1",
                    "answer": _ANSWER,
                    "claims": [
                        {
                            "text": _ANSWER,
                            "citation_chunk_ids": [_CHUNK_ID],
                        }
                    ],
                },
                separators=(",", ":"),
            ),
            identity=self.identity,
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(input_tokens=32, output_tokens=12),
        )


@dataclass(frozen=True, slots=True)
class ProductionQAHarness:
    services: QARuntimeServices
    conversations: ConversationService
    orchestrator: QAOrchestrator
    embedding: DeterministicEmbeddingProvider


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


async def _production_qa_harness(
    root: Path,
    generation: FiveWayGenerationProvider,
    admission: QAAdmissionController,
) -> ProductionQAHarness:
    layout = DataLayout.from_root(root / "data")
    layout.initialize()
    database = Database(layout.metadata_db)
    database.initialize()
    repositories = KnowledgeRepositories.from_database(database)

    source_id = "source-concurrency-leave"
    source_key = "concurrency-leave-policy"
    filename = "concurrency-leave-policy.txt"
    repositories.documents.create(
        Document(
            source_id=source_id,
            source_key=source_key,
            display_title="Concurrency Leave Policy",
            media_type="text/plain",
            kind=DocumentKind.TEXT,
            active_version=1,
        )
    )
    repositories.documents.add_version(
        DocumentVersion(
            source_id=source_id,
            version=1,
            content_digest=_digest(_SOURCE_TEXT),
            derivation_config_digest=_digest("concurrency-proof-v1"),
            original_filename=filename,
            media_type="text/plain",
            size_bytes=len(_SOURCE_TEXT.encode("utf-8")),
            source_artifact_path=layout.source_artifact_relative_path(source_id, 1, filename),
            canonical_artifact_path=layout.canonical_artifact_relative_path(source_id, 1),
            extraction_method=ExtractionMethod.TEXT,
        )
    )
    chunk = Chunk(
        chunk_id=_CHUNK_ID,
        parent_chunk_id=f"parent-{_CHUNK_ID}",
        source_id=source_id,
        document_version=1,
        ordinal=0,
        text=_SOURCE_TEXT,
        content_digest=_digest(_SOURCE_TEXT),
        locator=ChunkLocator(section_path=("Annual leave",)),
    )
    parent = ParentChunk(
        parent_chunk_id=chunk.parent_chunk_id,
        source_id=chunk.source_id,
        document_version=chunk.document_version,
        ordinal=chunk.ordinal,
        text=chunk.text,
        content_digest=chunk.content_digest,
        locator=chunk.locator,
        token_count=len(token_spans(chunk.text)),
    )

    embedding = DeterministicEmbeddingProvider()
    revision_id = "revision-concurrency-proof"
    with EmbeddingCache(layout.directory("caches") / "concurrency-embeddings.sqlite3") as cache:
        staged = await RevisionStager(layout, EmbeddingStage(embedding, cache)).stage(
            revision_id,
            (chunk,),
            {source_id: "Concurrency Leave Policy"},
            {source_id: 1},
            ProviderCallContext(
                "request-concurrency-index",
                "concurrency-index",
                Deadline.after(30),
            ),
            parents=(parent,),
        )
    with database.transaction() as connection:
        repositories.index_revisions.create(staged, connection=connection)
        repositories.parent_chunks.insert_many(
            revision_id,
            (parent,),
            connection=connection,
        )
    RevisionPublisher(layout, repositories.index_revisions).publish(
        revision_id,
        expected_active_revision_id=None,
    )
    embedding.requests.clear()

    retry_policy = RetryPolicy(attempt_timeout_seconds=25, max_retries=0)
    provider_router = ModelProviderRouter(
        embedding_routes=(ProviderRoute("concurrency-embedding", embedding, retry_policy),),
        generation_routes=(ProviderRoute("concurrency-generation", generation, retry_policy),),
    )
    conversations = ConversationService(SessionRepository(database))
    injection_policy = InjectionPolicy()
    orchestrator = QAOrchestrator(
        conversations=conversations,
        retrieval=SnapshotRetrievalGateway(
            BoundRetrievalSnapshotFactory(layout, repositories.index_revisions),
            provider_router,
        ),
        generation=provider_router,
        fact_assessor=SemanticFactEvidenceAssessor(
            provider_router,
            required_space=embedding.identity,
            config=FactAssessmentConfig(candidate_similarity_floor=0.6),
        ),
        context_builder=ContextBuilder(parent_resolver=repositories.parent_chunks),
        injection_policy=injection_policy,
        budgets=QAStageBudgets(
            total_seconds=45,
            validation_seconds=5,
            retrieval_seconds=15,
            rerank_seconds=2,
            evidence_assessment_seconds=10,
            generation_seconds=20,
            finalization_seconds=5,
        ),
        maximum_provider_attempts=1,
    )
    services = QARuntimeServices(
        conversations=conversations,
        orchestrator=orchestrator,
        emitter=CompleteResponseEmitter(
            conversations,
            injection_policy=injection_policy,
        ),
        admission=admission,
    )
    return ProductionQAHarness(services, conversations, orchestrator, embedding)


@pytest.mark.asyncio
async def test_five_http_qa_pipelines_overlap_inside_the_generation_provider(
    tmp_path: Path,
) -> None:
    provider_gate = FiveWayGenerationProvider()
    admission = QAAdmissionController(max_active=5, max_queue=0)
    harness = await _production_qa_harness(tmp_path, provider_gate, admission)
    owners = tuple(f"owner_{index}" for index in range(5))
    sessions = tuple(harness.conversations.create_session(owner) for owner in owners)
    assert type(harness.orchestrator) is QAOrchestrator

    app = create_app(
        Settings(
            _env_file=None,
            data_root=tmp_path / "data",
            environment="test",
            workbench_enabled=False,
            qa_total_deadline_seconds=30,
        ),
        qa_services=harness.services,
    )
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        requests = [
            asyncio.create_task(
                client.post(
                    "/api/v1/qa",
                    headers={"X-RAG-Cache-Policy": "bypass"},
                    json={
                        "owner_id": owner,
                        "session_id": session.session_id,
                        "question": "How many days of annual leave do employees receive?",
                        "mode": "hybrid",
                        "requested_language": "en",
                    },
                )
            )
            for owner, session in zip(owners, sessions, strict=True)
        ]
        try:
            await asyncio.wait_for(provider_gate.all_entered.wait(), timeout=15)
            assert provider_gate.peak == 5
            assert len(provider_gate.requests) == 5
            assert admission.active_count == 5
            assert admission.queued_count == 0
        finally:
            provider_gate.release.set()
        responses = await asyncio.wait_for(asyncio.gather(*requests), timeout=15)

    events = [ValidatedStreamEvent.model_validate_json(response.text) for response in responses]
    assert all(response.status_code == 200 for response in responses)
    assert all(
        response.headers["content-type"].startswith(NDJSON_MEDIA_TYPE) for response in responses
    )
    assert [event.session_id for event in events] == [session.session_id for session in sessions]
    assert all(event.kind is StreamEventKind.ANSWER and event.terminal for event in events)
    assert all(event.content == _ANSWER for event in events)
    assert all(
        [citation.chunk_id for citation in event.citations] == [_CHUNK_ID] for event in events
    )
    assert len({event.request_id for event in events}) == 5
    assert len(harness.embedding.requests) == 10
    assert provider_gate.active == 0

    for owner, session in zip(owners, sessions, strict=True):
        turns = harness.conversations.list_turns(session.session_id, owner)
        assert [turn.role for turn in turns] == [
            ConversationRole.USER,
            ConversationRole.ASSISTANT,
        ]
        assert turns[-1].content == _ANSWER

    snapshot = await admission.snapshot()
    assert snapshot.active == 0
    assert snapshot.queued == 0
    assert snapshot.admitted_total == 5
    assert snapshot.rejected_total == 0
