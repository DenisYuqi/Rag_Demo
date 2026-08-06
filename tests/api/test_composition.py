from __future__ import annotations

from rag_mvp.api.composition import _provider_alias


def test_provider_alias_binds_embedding_identity_to_the_endpoint() -> None:
    official = _provider_alias("https://api.openai.com/v1")

    assert official == _provider_alias("https://api.openai.com/v1/")
    assert official != _provider_alias("https://models.internal.example/v1")
    assert "api.openai.com" not in official
