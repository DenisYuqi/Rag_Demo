from __future__ import annotations

from pathlib import Path

import pytest

from rag_mvp.config.settings import Settings
from rag_mvp.domain.ingestion import (
    Document,
    DocumentKind,
    IngestionJob,
    IngestionJobStatus,
    IngestionOperation,
    IngestionStage,
)
from rag_mvp.ui.callbacks import WorkbenchCallbacks
from rag_mvp.ui.models import UploadPayload
from rag_mvp.ui.services import WorkbenchServices
from rag_mvp.ui.workbench import create_workbench

pytestmark = pytest.mark.ui


class FakeDocumentGateway:
    def __init__(self, *, with_document: bool = False) -> None:
        self.revision = "revision_initial" if with_document else None
        self.documents: list[Document] = []
        if with_document:
            self.documents.append(self._document("Leave Policy"))
        self.jobs: list[IngestionJob] = []
        self.uploads: list[UploadPayload] = []
        self.reindex_submissions = 0
        self.delete_submissions: list[str] = []

    @staticmethod
    def _document(title: str) -> Document:
        return Document(
            source_id="source_policy",
            source_key="policy-key",
            display_title=title,
            media_type="text/plain",
            kind=DocumentKind.TEXT,
            active_version=2,
        )

    @staticmethod
    def _queued(job_id: str, operation: IngestionOperation) -> IngestionJob:
        return IngestionJob(
            job_id=job_id,
            source_key="policy-key",
            operation=operation,
        )

    def submit_upload(self, payload: UploadPayload) -> IngestionJob:
        self.uploads.append(payload)
        job = self._queued("job_upload", IngestionOperation.UPLOAD)
        self.jobs.append(job)
        return job

    def submit_reindex(self) -> IngestionJob:
        self.reindex_submissions += 1
        job = self._queued("job_reindex", IngestionOperation.REINDEX)
        self.jobs.append(job)
        return job

    def submit_delete(self, source_id: str) -> IngestionJob:
        self.delete_submissions.append(source_id)
        job = self._queued("job_delete", IngestionOperation.DELETE)
        self.jobs.append(job)
        return job

    async def run_job(self, job_id: str) -> IngestionJob:
        submitted = next(job for job in self.jobs if job.job_id == job_id)
        revision = f"revision_{submitted.operation.value}"
        completed = IngestionJob(
            job_id=submitted.job_id,
            source_key=submitted.source_key,
            operation=submitted.operation,
            status=IngestionJobStatus.SUCCEEDED,
            stage=IngestionStage.COMPLETE,
            source_id="source_policy",
            document_version=2,
            ocr_page_count=1,
            chunk_count=7,
            active_index_revision=revision,
            stage_timings_ms={"publishing": 2.5},
        )
        self.jobs = [completed if job.job_id == job_id else job for job in self.jobs]
        self.revision = revision
        if submitted.operation is IngestionOperation.UPLOAD:
            payload = self.uploads[-1]
            self.documents = [self._document(payload.display_title or payload.filename)]
        elif submitted.operation is IngestionOperation.DELETE:
            self.documents = []
        return completed

    def get_job(self, job_id: str) -> IngestionJob | None:
        return next((job for job in self.jobs if job.job_id == job_id), None)

    def list_active_documents(self) -> tuple[str | None, tuple[Document, ...]]:
        return self.revision, tuple(self.documents)

    def list_jobs(self) -> tuple[IngestionJob, ...]:
        return tuple(self.jobs)


def test_existing_documents_are_preloaded_and_refreshed_on_page_load() -> None:
    gateway = FakeDocumentGateway(with_document=True)
    blocks = create_workbench(
        settings=Settings(_env_file=None),
        services=WorkbenchServices(documents=gateway),
    )
    config = blocks.get_config_file()
    components = config["components"]
    components_by_label = {
        component["props"].get("label"): component
        for component in components
        if component.get("props")
    }
    documents = components_by_label["Active documents / 活跃文档"]
    jobs = components_by_label["Ingestion progress / 摄取进度"]
    status = components_by_label["Document status / 文档状态"]

    assert documents["props"]["value"]["data"] == [
        ["source_policy", "Leave Policy", 2, "text", "text/plain"]
    ]
    assert "revision_initial" in status["props"]["value"]
    expected_outputs = {documents["id"], jobs["id"], status["id"]}
    assert any(
        any(event == "load" for _, event in dependency["targets"])
        and dependency["inputs"] == []
        and set(dependency["outputs"]) == expected_outputs
        for dependency in config["dependencies"]
    )


@pytest.mark.asyncio
async def test_upload_shows_published_document_progress_and_safe_metadata(tmp_path: Path) -> None:
    path = tmp_path / "policy.txt"
    path.write_text("Employees receive twelve days.", encoding="utf-8")
    gateway = FakeDocumentGateway()
    callbacks = WorkbenchCallbacks(WorkbenchServices(documents=gateway))

    rendered = await callbacks.upload_document(
        str(path),
        "policy-key",
        "Leave Policy person@example.com",
    )

    assert len(gateway.uploads) == 1
    assert gateway.uploads[0] == UploadPayload(
        filename="policy.txt",
        content=b"Employees receive twelve days.",
        declared_media_type="text/plain",
        source_key="policy-key",
        display_title="Leave Policy person@example.com",
    )
    assert rendered.document_rows == (
        (
            "source_policy",
            "Leave Policy [REDACTED_EMAIL]",
            2,
            "text",
            "text/plain",
        ),
    )
    assert rendered.job_rows == (
        (
            "job_upload",
            "upload",
            "succeeded",
            "complete",
            "source_policy",
            2,
            1,
            7,
            "revision_upload",
        ),
    )
    assert "upload complete" in rendered.status_markdown
    assert "person@example.com" not in repr(rendered)


@pytest.mark.asyncio
async def test_reindex_reports_new_active_revision_and_completed_job() -> None:
    gateway = FakeDocumentGateway(with_document=True)
    callbacks = WorkbenchCallbacks(WorkbenchServices(documents=gateway))

    rendered = await callbacks.reindex_documents()

    assert gateway.reindex_submissions == 1
    assert rendered.document_rows[0][0] == "source_policy"
    assert rendered.document_rows[0][2:] == (2, "text", "text/plain")
    assert rendered.job_rows[-1][1:4] == ("reindex", "succeeded", "complete")
    assert rendered.job_rows[-1][-1] == "revision_reindex"
    assert "reindex complete" in rendered.status_markdown


@pytest.mark.asyncio
async def test_delete_requires_confirmation_and_only_reports_after_publication() -> None:
    gateway = FakeDocumentGateway(with_document=True)
    callbacks = WorkbenchCallbacks(WorkbenchServices(documents=gateway))

    unconfirmed = await callbacks.delete_document("source_policy", False)

    assert gateway.delete_submissions == []
    assert unconfirmed.document_rows == ()
    assert "Confirm deletion first" in unconfirmed.status_markdown

    confirmed = await callbacks.delete_document("source_policy", True)

    assert gateway.delete_submissions == ["source_policy"]
    assert gateway.documents == []
    assert confirmed.document_rows == ()
    assert confirmed.job_rows[-1][1:4] == ("delete", "succeeded", "complete")
    assert confirmed.job_rows[-1][-1] == "revision_delete"
    assert "Deletion published" in confirmed.status_markdown
