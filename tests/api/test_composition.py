from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import SecretStr

from rag_mvp.api.composition import (
    _BgeModelLoader,
    _compose_bge_isolated_services,
    _compose_evaluation_service,
    _compose_retrieval_cache,
    _provider_alias,
    compose_bge_services,
)
from rag_mvp.config.settings import Settings
from rag_mvp.evaluation.comparison_production import ProductionComparisonJobExecutor
from rag_mvp.evaluation.production import ProductionEvaluationJobExecutor
from rag_mvp.safety.redactor import DEFAULT_REDACTOR
from rag_mvp.storage.database import Database
from rag_mvp.storage.layout import DataLayout
from rag_mvp.storage.repositories import RuntimeRepositories


class _BlockingWarmup:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def warmup(self) -> None:
        self.calls += 1
        self.started.set()
        if not self.release.wait(timeout=2):
            raise RuntimeError("test warmup was not released")


def _wait_for_status(loader: _BgeModelLoader, state: str) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if loader.status().state == state:
            return
        time.sleep(0.01)
    raise AssertionError(f"loader did not reach {state}")


def test_bge_model_loader_reports_two_background_stages() -> None:
    embedding = _BlockingWarmup()
    reranker = _BlockingWarmup()
    loader = _BgeModelLoader(
        cast(Any, embedding),
        cast(Any, reranker),
    )

    initial = loader.status()
    assert embedding.started.wait(timeout=1)
    assert initial.state == "loading"
    assert initial.completed_steps == 0
    assert initial.active_step == "embedding-model"

    embedding.release.set()
    assert reranker.started.wait(timeout=1)
    reranking = loader.status()
    assert reranking.state == "loading"
    assert reranking.completed_steps == 1
    assert reranking.active_step == "reranker-model"

    reranker.release.set()
    _wait_for_status(loader, "ready")
    ready = loader.status()
    assert ready.completed_steps == ready.total_steps == 2
    assert embedding.calls == reranker.calls == 1


def test_provider_alias_binds_embedding_identity_to_the_endpoint() -> None:
    official = _provider_alias("https://api.openai.com/v1")

    assert official == _provider_alias("https://api.openai.com/v1/")
    assert official != _provider_alias("https://models.internal.example/v1")
    assert "api.openai.com" not in official


def test_production_cache_composition_is_disabled_by_default_and_shared_when_enabled(
    tmp_path: Path,
) -> None:
    disabled = Settings(data_root=tmp_path, _env_file=None)
    enabled = Settings(
        data_root=tmp_path,
        retrieval_cache_enabled=True,
        retrieval_cache_max_entries=3,
        retrieval_cache_ttl_seconds=12,
        _env_file=None,
    )

    assert _compose_retrieval_cache(disabled) is None
    cache = _compose_retrieval_cache(enabled)
    assert cache is not None
    assert cache.configuration_id == enabled.configuration_identity
    assert cache.metrics.snapshot().hit_rate is None


def test_bge_evaluation_composition_keeps_local_models_and_profile_storage(
    tmp_path: Path,
) -> None:
    settings = Settings(
        data_root=tmp_path / "openai",
        bge_data_root=tmp_path / "bge",
        evaluation_dataset_root=Path("evaluations/datasets"),
        _env_file=None,
    ).bge_profile_settings()
    layout = DataLayout.from_root(settings.data_root)
    layout.initialize()
    database = Database(layout.metadata_db)
    database.initialize()

    service = _compose_evaluation_service(
        settings,
        DEFAULT_REDACTOR,
        layout,
        RuntimeRepositories.from_database(database),
        include_release_evidence=False,
        isolated_composition_factory=_compose_bge_isolated_services,
    )

    assert service.settings.data_root == (tmp_path / "bge").resolve()
    assert service.settings.embedding_model == "BAAI/bge-m3"
    assert service.settings.reranking_model == "BAAI/bge-reranker-v2-m3"
    assert service.release_store is None
    assert isinstance(service.executor, ProductionEvaluationJobExecutor)
    assert service.executor.composition_factory is _compose_bge_isolated_services
    isolated = service.executor.isolated_settings("run-bge")
    assert isolated.data_root.is_relative_to((tmp_path / "bge").resolve())
    assert isolated.embedding_model == "BAAI/bge-m3"
    assert isinstance(service.comparison_executor, ProductionComparisonJobExecutor)
    assert service.comparison_executor.composition_factory is _compose_bge_isolated_services


def test_bge_composition_rejects_a_disabled_profile_before_loading_models(
    tmp_path: Path,
) -> None:
    settings = Settings(
        data_root=tmp_path,
        provider_backend="openai",
        openai_api_key=SecretStr("test-key"),
        bge_profile_enabled=False,
        _env_file=None,
    )

    with pytest.raises(ValueError, match="bge_profile_disabled"):
        compose_bge_services(settings, DEFAULT_REDACTOR)
