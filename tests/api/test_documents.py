from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_mvp.api.app import create_app
from rag_mvp.config.settings import Settings
from rag_mvp.domain.ingestion import IngestionJobStatus
from rag_mvp.ingestion.chunking import ChunkingConfig
from rag_mvp.ingestion.indexing import RevisionPublisher
from rag_mvp.ingestion.service import IngestionService
from rag_mvp.providers.fakes import DeterministicEmbeddingProvider
from rag_mvp.providers.models import EmbeddingRequest, EmbeddingResult, ProviderCallContext
from rag_mvp.storage.layout import DataLayout

pytestmark = pytest.mark.api


class NeverOcr:
    version = "never-ocr-v1"

    def recognize(self, png_bytes: bytes, *, languages: str) -> str:
        del png_bytes, languages
        raise AssertionError("text API tests must not invoke OCR")


class BlockingEmbeddingProvider(DeterministicEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    async def embed(
        self,
        request: EmbeddingRequest,
        context: ProviderCallContext,
    ) -> EmbeddingResult:
        self.started.set()
        await asyncio.to_thread(self.release.wait)
        return await super().embed(request, context)


class InjectedFailure(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ApiHarness:
    client: TestClient
    service: IngestionService
    provider: DeterministicEmbeddingProvider


@pytest.fixture
def api_harness(tmp_path: Path) -> ApiHarness:
    root = tmp_path / "data"
    provider = DeterministicEmbeddingProvider()
    service = _service(root, provider)
    app = create_app(
        _settings(root),
        ingestion_service=service,
        owns_ingestion_service=True,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield ApiHarness(client, service, provider)


def _settings(root: Path) -> Settings:
    return Settings(
        data_root=root,
        upload_max_bytes=64,
        shutdown_grace_seconds=5,
        _env_file=None,
    )


def _service(
    root: Path,
    provider: DeterministicEmbeddingProvider,
) -> IngestionService:
    return IngestionService.create(
        root,
        provider,
        ocr=NeverOcr(),
        chunking_config=ChunkingConfig(target_tokens=8, overlap_tokens=2),
        upload_max_bytes=64,
    )


def _upload(
    client: TestClient,
    content: bytes,
    *,
    filename: str = "policy.txt",
    media_type: str = "text/plain",
    source_key: str = "policy",
    display_title: str = "Policy",
):
    return client.post(
        "/api/v1/documents",
        files={"file": (filename, content, media_type)},
        data={"source_key": source_key, "display_title": display_title},
    )


def _wait_for_terminal(client: TestClient, location: str) -> dict[str, object]:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        response = client.get(location)
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {
            IngestionJobStatus.SUCCEEDED.value,
            IngestionJobStatus.FAILED.value,
        }:
            return payload
        time.sleep(0.02)
    pytest.fail("ingestion job did not reach a terminal state")


def test_upload_list_rebuild_and_delete_contract(api_harness: ApiHarness) -> None:
    client = api_harness.client
    assert client.get("/api/v1/documents").json() == {
        "active_index_revision": None,
        "documents": [],
    }

    accepted = _upload(
        client,
        b"Leave policy ALPHA-101 grants twelve days.",
        display_title="Policy owner@example.com",
    )

    assert accepted.status_code == 202
    assert accepted.json()["status"] == "queued"
    assert accepted.json()["stage"] == "queued"
    assert "source_key" not in accepted.json()
    location = accepted.headers["location"]
    assert location.endswith(accepted.json()["job_id"])
    uploaded = _wait_for_terminal(client, location)
    assert uploaded["status"] == "succeeded"
    assert uploaded["document_version"] == 1
    assert uploaded["chunk_count"] > 0
    assert uploaded["ocr_page_count"] == 0
    assert uploaded["active_index_revision"]
    assert set(uploaded["stage_timings_ms"]) == {
        "validating",
        "extracting",
        "normalizing",
        "chunking",
        "embedding",
        "indexing",
        "publishing",
    }

    listed = client.get("/api/v1/documents")
    assert listed.status_code == 200
    listed_payload = listed.json()
    assert listed_payload["active_index_revision"] == uploaded["active_index_revision"]
    assert len(listed_payload["documents"]) == 1
    document = listed_payload["documents"][0]
    assert document["display_title"] == "Policy [REDACTED_EMAIL]"
    assert document["active_version"] == 1
    assert document["kind"] == "text"
    rendered = listed.text
    assert "owner@example.com" not in rendered
    assert "source_key" not in rendered
    assert "artifact_path" not in rendered

    rebuild = client.post("/api/v1/index/rebuild")
    assert rebuild.status_code == 202
    assert rebuild.headers["location"].endswith(rebuild.json()["job_id"])
    rebuilt = _wait_for_terminal(client, rebuild.headers["location"])
    assert rebuilt["status"] == "succeeded"
    assert rebuilt["active_index_revision"] != uploaded["active_index_revision"]
    after_rebuild = client.get("/api/v1/documents").json()
    assert after_rebuild["active_index_revision"] == rebuilt["active_index_revision"]
    assert after_rebuild["documents"][0]["active_version"] == 1

    deleted = client.delete(f"/api/v1/documents/{document['source_id']}")
    assert deleted.status_code == 202
    deletion = _wait_for_terminal(client, deleted.headers["location"])
    assert deletion["status"] == "succeeded"
    assert deletion["active_index_revision"] != rebuilt["active_index_revision"]
    assert client.get("/api/v1/documents").json() == {
        "active_index_revision": deletion["active_index_revision"],
        "documents": [],
    }
    missing = client.delete(f"/api/v1/documents/{document['source_id']}")
    assert missing.status_code == 404
    assert missing.json() == {"error": {"code": "source_not_active"}}


def test_unpublished_document_is_hidden_until_atomic_publication(tmp_path: Path) -> None:
    root = tmp_path / "data"
    provider = BlockingEmbeddingProvider()
    service = _service(root, provider)
    app = create_app(
        _settings(root),
        ingestion_service=service,
        owns_ingestion_service=True,
    )
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            accepted = _upload(client, b"Pending publication BETA-202 remains hidden.")
            assert accepted.status_code == 202
            assert provider.started.wait(timeout=5)

            during = client.get("/api/v1/documents")
            assert during.status_code == 200
            assert during.json() == {"active_index_revision": None, "documents": []}

            provider.release.set()
            completed = _wait_for_terminal(client, accepted.headers["location"])
            assert completed["status"] == "succeeded"
            after = client.get("/api/v1/documents").json()
            assert len(after["documents"]) == 1
            assert after["active_index_revision"] == completed["active_index_revision"]
    finally:
        provider.release.set()


def test_failed_publication_keeps_prior_revision_and_document_version(
    api_harness: ApiHarness,
) -> None:
    client = api_harness.client
    first = _upload(client, b"Baseline SAFE-303 remains active.")
    completed = _wait_for_terminal(client, first.headers["location"])
    before = client.get("/api/v1/documents").json()

    def fail_inside_transaction(phase: str) -> None:
        if phase == "inside_transaction":
            raise InjectedFailure("private-marker-person@example.com")

    api_harness.service._publisher = RevisionPublisher(
        DataLayout.from_root(api_harness.service.repositories.index_revisions.database.path.parent),
        api_harness.service.repositories.index_revisions,
        failure_hook=fail_inside_transaction,
    )
    changed = _upload(client, b"Candidate BROKEN-404 must not publish.")
    failed = _wait_for_terminal(client, changed.headers["location"])

    assert completed["document_version"] == 1
    assert failed["status"] == "failed"
    assert failed["safe_error_code"] == "ingestion_internal_error"
    assert failed["failed_stage"] == "publishing"
    assert failed["document_version"] == 2
    assert failed["active_index_revision"] == before["active_index_revision"]
    assert "private-marker" not in str(failed)
    after = client.get("/api/v1/documents").json()
    assert after == before
    assert after["documents"][0]["active_version"] == 1


@pytest.mark.parametrize(
    ("filename", "content", "media_type", "expected_status", "expected_code"),
    [
        ("policy.exe", b"valid text", "text/plain", 415, "unsupported_extension"),
        ("policy.txt", b"", "text/plain", 422, "empty_document"),
        ("policy.txt", b"x" * 65, "text/plain", 413, "document_too_large"),
        ("policy.txt", b"valid text", "application/pdf", 415, "media_type_mismatch"),
    ],
)
def test_invalid_uploads_return_safe_codes_without_persistent_side_effects(
    api_harness: ApiHarness,
    filename: str,
    content: bytes,
    media_type: str,
    expected_status: int,
    expected_code: str,
) -> None:
    response = _upload(
        api_harness.client,
        content,
        filename=filename,
        media_type=media_type,
        display_title="private-title-person@example.com",
    )

    assert response.status_code == expected_status
    assert response.json() == {"error": {"code": expected_code}}
    assert "person@example.com" not in response.text
    assert api_harness.service.repositories.ingestion_jobs.list() == []
    assert api_harness.service.repositories.index_revisions.get_active() is None
    layout = DataLayout.from_root(
        api_harness.service.repositories.index_revisions.database.path.parent
    )
    assert list(layout.directory("jobs").iterdir()) == []


def test_missing_resources_and_request_validation_are_content_free(
    api_harness: ApiHarness,
) -> None:
    unknown = api_harness.client.get("/api/v1/ingestion-jobs/job_missing")
    assert unknown.status_code == 404
    assert unknown.json() == {"error": {"code": "ingestion_job_not_found"}}

    missing_file = api_harness.client.post(
        "/api/v1/documents",
        data={"display_title": "person@example.com"},
    )
    assert missing_file.status_code == 422
    assert missing_file.json() == {"error": {"code": "request_invalid"}}
    assert "person@example.com" not in missing_file.text


def test_unexpected_exception_and_unready_service_return_only_fixed_errors(
    api_harness: ApiHarness,
    tmp_path: Path,
) -> None:
    def explode(job_id: str):
        del job_id
        raise RuntimeError("secret person@example.com must not cross the API")

    api_harness.service.get_job = explode  # type: ignore[method-assign]
    failed = api_harness.client.get("/api/v1/ingestion-jobs/job_failure")
    assert failed.status_code == 500
    assert failed.json() == {"error": {"code": "internal_error"}}
    assert "person@example.com" not in failed.text

    unavailable = create_app(_settings(tmp_path / "unavailable"))
    with TestClient(unavailable, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/documents")
    assert response.status_code == 503
    assert response.json() == {"error": {"code": "ingestion_unavailable"}}


def test_ingestion_service_configuration_must_match_app(
    api_harness: ApiHarness,
) -> None:
    settings = _settings(api_harness.service.data_root).model_copy(update={"upload_max_bytes": 65})

    with pytest.raises(ValueError, match="ingestion_service_configuration_mismatch"):
        create_app(settings, ingestion_service=api_harness.service)


def test_openapi_declares_multipart_and_allowlisted_response_models(
    api_harness: ApiHarness,
) -> None:
    schema = api_harness.client.get("/openapi.json").json()
    paths = schema["paths"]
    assert set(paths) >= {
        "/api/v1/documents",
        "/api/v1/documents/{source_id}",
        "/api/v1/index/rebuild",
        "/api/v1/ingestion-jobs/{job_id}",
    }
    assert paths["/api/v1/documents"]["post"]["responses"].keys() >= {"202", "413", "415", "422"}
    assert "multipart/form-data" in paths["/api/v1/documents"]["post"]["requestBody"]["content"]
    job_properties = schema["components"]["schemas"]["IngestionJobResponse"]["properties"]
    document_properties = schema["components"]["schemas"]["ActiveDocumentResponse"]["properties"]
    assert "source_key" not in job_properties
    assert "source_artifact_path" not in document_properties
    assert "canonical_artifact_path" not in document_properties
