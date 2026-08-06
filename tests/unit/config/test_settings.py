from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from rag_mvp.config.settings import Settings


def test_defaults_are_safe_and_offline_ready(tmp_path: object) -> None:
    settings = Settings(data_root=tmp_path, _env_file=None)

    assert settings.host == "127.0.0.1"
    assert settings.provider_readiness_errors() == ()
    assert settings.safe_dump()["openai_api_key"] is None
    assert len(settings.configuration_identity) == 16
    assert not settings.allow_single_retriever_degradation
    assert not settings.retrieval_cache_enabled
    assert settings.reranking_model is None


def test_secret_never_appears_in_repr_or_safe_dump(tmp_path: object) -> None:
    raw_secret = "sk-test-do-not-log"
    settings = Settings(
        data_root=tmp_path,
        openai_api_key=SecretStr(raw_secret),
        _env_file=None,
    )

    assert raw_secret not in repr(settings)
    assert raw_secret not in str(settings.safe_dump())
    assert settings.safe_dump()["openai_api_key"] == "[REDACTED_SECRET]"


def test_openai_backend_reports_safe_missing_credentials(tmp_path: object) -> None:
    settings = Settings(data_root=tmp_path, provider_backend="openai", _env_file=None)

    assert settings.provider_readiness_errors() == ("provider_credentials_missing",)


def test_openai_compatible_request_dialect_is_explicitly_configurable(tmp_path: object) -> None:
    settings = Settings(
        data_root=tmp_path,
        openai_send_dimensions=False,
        openai_max_tokens_parameter="max_tokens",
        _env_file=None,
    )

    assert not settings.openai_send_dimensions
    assert settings.openai_max_tokens_parameter == "max_tokens"

    with pytest.raises(ValidationError):
        Settings(
            data_root=tmp_path,
            openai_max_tokens_parameter="unsupported",
            _env_file=None,
        )


@pytest.mark.parametrize("path", ["/", "/api", "/api/v1", "/healthz", "/metrics"])
def test_reserved_workbench_paths_are_rejected(tmp_path: object, path: str) -> None:
    with pytest.raises(ValidationError):
        Settings(data_root=tmp_path, workbench_path=path, _env_file=None)


def test_limit_invariants_are_validated(tmp_path: object) -> None:
    with pytest.raises(ValidationError):
        Settings(
            data_root=tmp_path,
            chunk_target_tokens=100,
            chunk_overlap_tokens=100,
            _env_file=None,
        )


def test_qa_stage_budgets_are_configurable_and_bounded_by_total(tmp_path: object) -> None:
    settings = Settings(
        data_root=tmp_path,
        qa_generation_budget_seconds=4.5,
        qa_finalization_budget_seconds=0.5,
        _env_file=None,
    )

    assert settings.qa_generation_budget_seconds == 4.5
    assert settings.qa_finalization_budget_seconds == 0.5

    with pytest.raises(ValidationError):
        Settings(
            data_root=tmp_path,
            qa_generation_budget_seconds=9.0,
            qa_finalization_budget_seconds=1.0,
            _env_file=None,
        )


def test_hybrid_rerank_default_requires_configured_model(tmp_path: object) -> None:
    with pytest.raises(ValidationError):
        Settings(
            data_root=tmp_path,
            default_retrieval_mode="hybrid-rerank",
            _env_file=None,
        )

    settings = Settings(
        data_root=tmp_path,
        default_retrieval_mode="hybrid-rerank",
        reranking_model="reranker",
        _env_file=None,
    )
    assert settings.reranking_model == "reranker"


def test_proxy_credentials_never_appear_in_diagnostics(tmp_path: object) -> None:
    raw_proxy = "http://proxy-user:proxy-password@127.0.0.1:7890"
    settings = Settings(
        data_root=tmp_path,
        openai_proxy_url=SecretStr(raw_proxy),
        _env_file=None,
    )

    assert raw_proxy not in repr(settings)
    assert raw_proxy not in str(settings.safe_dump())
    assert settings.safe_dump()["openai_proxy_url"] == "[REDACTED_SECRET]"


@pytest.mark.parametrize("proxy_url", ["127.0.0.1:7890", "file:///tmp/proxy"])
def test_unsafe_proxy_urls_are_rejected(tmp_path: object, proxy_url: str) -> None:
    with pytest.raises(ValidationError):
        Settings(data_root=tmp_path, openai_proxy_url=proxy_url, _env_file=None)
