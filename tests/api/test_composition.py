from __future__ import annotations

from pathlib import Path

from rag_mvp.api.composition import _compose_retrieval_cache, _provider_alias
from rag_mvp.config.settings import Settings


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
