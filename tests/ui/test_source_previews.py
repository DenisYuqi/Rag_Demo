from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from rag_mvp.api.qa import QARuntimeServices
from rag_mvp.config.settings import Settings
from rag_mvp.domain.ingestion import Chunk, ChunkLocator
from rag_mvp.domain.qa import (
    AnswerClaim,
    Citation,
    SafeQADiagnostics,
    StreamEventKind,
    ValidatedStreamEvent,
)
from rag_mvp.domain.retrieval import RetrievalMode
from rag_mvp.ingestion.service import IngestionService
from rag_mvp.performance.worker_pools import BoundedWorkerPool
from rag_mvp.retrieval.binding import BoundRetrievalSnapshotFactory
from rag_mvp.retrieval.bm25 import LexicalRecord
from rag_mvp.ui.services import (
    SharedQAGateway,
    SnapshotSourcePreviewLookup,
    configured_workbench_services,
)

pytestmark = pytest.mark.ui


def _citation(*, locator: ChunkLocator | None = None) -> Citation:
    return Citation(
        source_title="Handbook",
        document_version=2,
        chunk_id="chunk_policy",
        locator=locator or ChunkLocator(pages=(4,)),
    )


def _record() -> LexicalRecord:
    text = "Employees receive twelve days."
    chunk = Chunk(
        chunk_id="chunk_policy",
        parent_chunk_id="parent_policy",
        source_id="source_policy",
        document_version=2,
        ordinal=0,
        text=text,
        content_digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        locator=ChunkLocator(pages=(4,)),
        token_count=5,
    )
    return LexicalRecord(
        chunk=chunk,
        display_title="Handbook",
        tokens=("employees", "receive", "twelve", "days"),
        record_digest="unused-by-preview-lookup",
    )


@dataclass
class _FakeSnapshot:
    bm25: object
    closed: bool = False

    def __enter__(self) -> _FakeSnapshot:
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self.closed = True


@dataclass
class _FakeSnapshotFactory:
    snapshot: _FakeSnapshot
    opened_revision_id: str | None = None

    def open_committed(self, revision_id: str) -> _FakeSnapshot:
        self.opened_revision_id = revision_id
        return self.snapshot


@dataclass
class _InlineWorkerPool:
    calls: int = 0

    async def run_cancel_safe(self, function: object, /, *args: object) -> object:
        self.calls += 1
        assert callable(function)
        return function(*args)


@pytest.mark.asyncio
async def test_snapshot_preview_uses_exact_committed_revision_and_citation_identity() -> None:
    snapshot = _FakeSnapshot(SimpleNamespace(records=(_record(),)))
    factory = _FakeSnapshotFactory(snapshot)
    pool = _InlineWorkerPool()
    lookup = SnapshotSourcePreviewLookup(
        cast(BoundRetrievalSnapshotFactory, factory),
        cast(BoundedWorkerPool, pool),
    )

    values = await lookup.get_previews(
        "request_answer",
        "revision_answer",
        (
            _citation(),
            _citation(locator=ChunkLocator(pages=(99,))),
        ),
    )

    assert values == {"chunk_policy": "Employees receive twelve days."}
    assert factory.opened_revision_id == "revision_answer"
    assert pool.calls == 1
    assert snapshot.closed is True


class _Conversations:
    @staticmethod
    def create_session(owner_id: str) -> object:
        del owner_id
        return SimpleNamespace(session_id="session_answer")


@dataclass
class _PreviewLookup:
    fail: bool = False
    call: tuple[str, str, tuple[Citation, ...]] | None = None

    async def get_previews(
        self,
        request_id: str,
        revision_id: str,
        citations: Sequence[Citation],
    ) -> dict[str, str]:
        self.call = (request_id, revision_id, tuple(citations))
        if self.fail:
            raise RuntimeError("preview lookup failed")
        return {citations[0].chunk_id: "Employees receive twelve days."}


@pytest.mark.asyncio
@pytest.mark.parametrize("lookup_fails", [False, True])
async def test_shared_qa_gateway_passes_request_revision_and_fails_preview_only(
    monkeypatch: pytest.MonkeyPatch,
    lookup_fails: bool,
) -> None:
    citation = _citation()

    async def fake_stream(*args: object, **kwargs: object) -> AsyncIterator[bytes]:
        del args, kwargs
        event = ValidatedStreamEvent(
            request_id=request_id,
            session_id="session_answer",
            sequence=0,
            kind=StreamEventKind.ANSWER,
            response_language="en",
            content="Employees receive twelve days.",
            claims=(
                AnswerClaim(
                    text="Employees receive twelve days.",
                    citation_chunk_ids=(citation.chunk_id,),
                ),
            ),
            citations=(citation,),
            diagnostics=SafeQADiagnostics(metadata={"index_revision": "revision_answer"}),
            terminal=True,
        )
        yield f"{event.model_dump_json()}\n".encode()

    monkeypatch.setattr("rag_mvp.ui.services.stream_qa_events", fake_stream)
    lookup = _PreviewLookup(fail=lookup_fails)
    runtime = cast(QARuntimeServices, SimpleNamespace(conversations=_Conversations()))
    gateway = SharedQAGateway(runtime, preview_lookup=lookup)

    request_id = "request_" + "a" * 32
    monkeypatch.setattr("rag_mvp.ui.services.uuid4", lambda: SimpleNamespace(hex="a" * 32))
    result = await gateway.submit(
        owner_id="owner_answer",
        session_id=None,
        question="How many days?",
        mode=RetrievalMode.HYBRID,
    )

    assert lookup.call == (request_id, "revision_answer", (citation,))
    assert result.previews[0].preview == (
        None if lookup_fails else "Employees receive twelve days."
    )


def test_production_workbench_composition_wires_previews_for_every_profile(
    tmp_path: Path,
) -> None:
    settings = Settings(data_root=tmp_path, _env_file=None)
    qa = cast(QARuntimeServices, SimpleNamespace(conversations=_Conversations()))

    def ingestion(root: Path) -> IngestionService:
        value = SimpleNamespace(
            data_root=root,
            repositories=SimpleNamespace(index_revisions=object()),
        )
        return cast(IngestionService, value)

    default_ingestion = ingestion(tmp_path)
    default_services = configured_workbench_services(
        settings=settings,
        qa=qa,
        ingestion=default_ingestion,
    )
    default_chat = default_services.chat_for("openai-api")

    assert isinstance(default_chat, SharedQAGateway)
    assert isinstance(default_chat.preview_lookup, SnapshotSourcePreviewLookup)

    profile_services = configured_workbench_services(
        settings=settings,
        qa=None,
        ingestion=None,
        profile_services={
            "openai-api": (qa, ingestion(tmp_path)),
            "bge-local": (qa, ingestion(tmp_path)),
        },
    )

    for profile_id in ("openai-api", "bge-local"):
        profile_chat = profile_services.chat_for(profile_id)
        assert isinstance(profile_chat, SharedQAGateway)
        assert isinstance(profile_chat.preview_lookup, SnapshotSourcePreviewLookup)
