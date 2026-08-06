from __future__ import annotations

from pathlib import Path

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
    assert settings.server_shutdown_grace_seconds == 4
    assert settings.app_shutdown_grace_seconds == 15
    assert settings.total_shutdown_budget_seconds == 19
    assert settings.total_shutdown_budget_seconds < settings.CONTAINER_STOP_GRACE_SECONDS


def test_server_and_application_shutdown_budgets_must_fit_container_grace(
    tmp_path: object,
) -> None:
    with pytest.raises(ValidationError, match="shutdown budgets must fit container grace"):
        Settings(
            data_root=tmp_path,
            server_shutdown_grace_seconds=5,
            shutdown_grace_seconds=15,
            _env_file=None,
        )


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


def test_openai_backend_reads_runtime_credential_file(tmp_path: Path) -> None:
    secret_path = tmp_path / "openai_api_key"
    secret_path.write_text("runtime-secret\n", encoding="utf-8")

    settings = Settings(
        data_root=tmp_path,
        provider_backend="openai",
        openai_api_key_file=secret_path,
        _env_file=None,
    )

    assert settings.provider_readiness_errors() == ()
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "runtime-secret"
    assert settings.safe_dump()["openai_api_key"] == "[REDACTED_SECRET]"


@pytest.mark.parametrize("content", [None, "", "   \n"])
def test_openai_backend_fails_readiness_safely_for_unusable_credential_file(
    tmp_path: Path,
    content: str | None,
) -> None:
    secret_path = tmp_path / "openai_api_key"
    if content is not None:
        secret_path.write_text(content, encoding="utf-8")

    settings = Settings(
        provider_backend="openai",
        openai_api_key_file=secret_path,
        _env_file=None,
    )

    assert settings.openai_api_key is None
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


def test_otlp_telemetry_requires_an_explicit_endpoint_for_readiness(tmp_path: object) -> None:
    settings = Settings(data_root=tmp_path, telemetry_exporter="otlp", _env_file=None)

    assert settings.telemetry_readiness_errors() == ("telemetry_otlp_endpoint_missing",)


def test_otlp_telemetry_endpoint_and_timeout_are_normalized(tmp_path: object) -> None:
    settings = Settings(
        data_root=tmp_path,
        telemetry_exporter="otlp",
        telemetry_otlp_traces_endpoint=" https://collector.example/v1/traces/ ",
        telemetry_export_timeout_seconds=2.5,
        _env_file=None,
    )

    assert settings.telemetry_readiness_errors() == ()
    assert settings.telemetry_otlp_traces_endpoint == "https://collector.example/v1/traces"
    assert settings.telemetry_export_timeout_seconds == 2.5


@pytest.mark.parametrize(
    "endpoint",
    [
        "collector:4318/v1/traces",
        "ftp://collector.example/v1/traces",
        "https://user:secret@collector.example/v1/traces",
        "https://collector.example/v1/traces?api_key=secret",
        "https://collector.example/v1/traces#secret",
    ],
)
def test_unsafe_otlp_telemetry_endpoints_are_rejected(
    tmp_path: object,
    endpoint: str,
) -> None:
    with pytest.raises(ValidationError):
        Settings(
            data_root=tmp_path,
            telemetry_exporter="otlp",
            telemetry_otlp_traces_endpoint=endpoint,
            _env_file=None,
        )
