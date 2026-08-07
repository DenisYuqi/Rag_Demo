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
    assert len(settings.runtime_configuration_identity) == 16
    assert len(settings.evaluation_configuration_identity) == 16
    assert not settings.allow_single_retriever_degradation
    assert not settings.retrieval_cache_enabled
    assert settings.retrieval_cache_max_entries == 256
    assert settings.retrieval_cache_ttl_seconds == 300
    assert settings.evaluation_max_active_jobs == 1
    assert settings.evaluation_shutdown_grace_seconds == 2
    assert settings.reranking_model is None
    assert settings.default_retrieval_profile == "openai-api"
    assert settings.bge_profile_enabled
    assert settings.bge_embedding_model == "BAAI/bge-m3"
    assert settings.bge_embedding_dimension == 1024
    assert settings.bge_reranking_model == "BAAI/bge-reranker-v2-m3"
    assert settings.bge_provider_timeout_seconds == 60
    assert settings.bge_qa_deadline_seconds == 45
    assert settings.bge_qa_retrieval_budget_seconds == 20
    assert settings.bge_rerank_deadline_seconds == 10
    assert settings.bge_qa_evidence_assessment_budget_seconds == 10
    assert settings.parent_chunk_target_tokens == 1536
    assert settings.chunk_target_tokens == 512
    assert settings.chunk_overlap_tokens == 128
    assert settings.qa_minimum_support_score == 0.45
    assert settings.server_shutdown_grace_seconds == 4
    assert settings.app_shutdown_grace_seconds == 15
    assert settings.total_shutdown_budget_seconds == 19
    assert settings.total_shutdown_budget_seconds < settings.CONTAINER_STOP_GRACE_SECONDS


def test_bge_profile_settings_are_isolated_and_truthful(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path / "openai", _env_file=None)

    local = settings.bge_profile_settings()

    assert local.data_root == (tmp_path / "openai" / "profiles" / "bge-local").resolve()
    assert local.data_root != settings.data_root
    assert local.embedding_model == "BAAI/bge-m3"
    assert local.embedding_dimension == 1024
    assert local.reranking_model == "BAAI/bge-reranker-v2-m3"
    assert local.default_retrieval_mode == "hybrid-rerank"
    assert local.provider_timeout_seconds == 60
    assert local.qa_deadline_seconds == 45
    assert local.qa_retrieval_budget_seconds == 20
    assert local.rerank_deadline_seconds == 10
    assert local.qa_evidence_assessment_budget_seconds == 10
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.reranking_model is None
    assert settings.provider_timeout_seconds == 8
    assert settings.qa_deadline_seconds == 9.5
    assert settings.rerank_deadline_seconds == 3
    assert settings.qa_evidence_assessment_budget_seconds == 4


def test_bge_profile_configuration_invariants_are_enforced(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="default BGE retrieval profile"):
        Settings(
            data_root=tmp_path,
            default_retrieval_profile="bge-local",
            bge_profile_enabled=False,
            _env_file=None,
        )

    with pytest.raises(ValidationError, match="1024-dimensional"):
        Settings(data_root=tmp_path, bge_embedding_dimension=768, _env_file=None)

    with pytest.raises(ValidationError, match="different data roots"):
        Settings(data_root=tmp_path, bge_data_root=tmp_path, _env_file=None)

    with pytest.raises(ValidationError, match="each BGE stage budget"):
        Settings(
            data_root=tmp_path,
            bge_qa_deadline_seconds=8,
            bge_qa_evidence_assessment_budget_seconds=8,
            _env_file=None,
        )

    with pytest.raises(ValidationError, match="parent chunk target"):
        Settings(
            data_root=tmp_path,
            parent_chunk_target_tokens=255,
            chunk_target_tokens=256,
            _env_file=None,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("retrieval_cache_max_entries", 0),
        ("retrieval_cache_max_entries", 10_001),
        ("retrieval_cache_ttl_seconds", 0),
        ("retrieval_cache_ttl_seconds", 86_401),
    ],
)
def test_retrieval_cache_bounds_are_validated(
    tmp_path: object,
    field: str,
    value: int,
) -> None:
    with pytest.raises(ValidationError):
        Settings(data_root=tmp_path, _env_file=None, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evaluation_max_active_jobs", 0),
        ("evaluation_max_active_jobs", 5),
        ("evaluation_shutdown_grace_seconds", -1),
        ("evaluation_shutdown_grace_seconds", 11),
    ],
)
def test_evaluation_supervisor_bounds_are_validated(
    tmp_path: object,
    field: str,
    value: int,
) -> None:
    with pytest.raises(ValidationError):
        Settings(data_root=tmp_path, _env_file=None, **{field: value})


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


def test_evaluation_identity_excludes_runtime_paths_ui_and_lifecycle(tmp_path: Path) -> None:
    baseline = Settings(data_root=tmp_path / "online", _env_file=None)
    isolated = baseline.model_copy(
        update={
            "data_root": tmp_path / "evaluations" / "workspaces" / "run-001",
            "evaluation_dataset_root": tmp_path / "datasets-copy",
            "workbench_enabled": False,
            "workbench_path": "/operator",
            "evaluation_max_active_jobs": 4,
            "evaluation_shutdown_grace_seconds": 7.0,
            "telemetry_exporter": "console",
        }
    )

    assert baseline.runtime_configuration_identity != isolated.runtime_configuration_identity
    assert baseline.configuration_identity == baseline.runtime_configuration_identity
    assert isolated.configuration_identity == isolated.runtime_configuration_identity
    assert baseline.evaluation_configuration_identity == isolated.evaluation_configuration_identity


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generation_model", "generation-alternative"),
        ("default_retrieval_mode", "dense"),
        ("dense_candidate_limit", 21),
        ("provider_retry_limit", 2),
        ("retrieval_cache_enabled", True),
        ("chunk_target_tokens", 640),
        ("pricing_version", "pricing-v2"),
        ("qa_max_active", 6),
        ("qa_max_queue", 11),
        ("qa_minimum_support_score", 0.55),
        ("environment", "production"),
    ],
)
def test_evaluation_identity_includes_behavioral_configuration(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    baseline = Settings(data_root=tmp_path, _env_file=None)
    changed = baseline.model_copy(update={field: value})

    assert baseline.evaluation_configuration_identity != changed.evaluation_configuration_identity


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
    defaults = Settings(data_root=tmp_path, _env_file=None)
    assert defaults.qa_validation_budget_seconds == 0.8
    assert defaults.qa_retrieval_budget_seconds == 4.0
    assert defaults.qa_evidence_assessment_budget_seconds == 4.0
    assert defaults.qa_generation_budget_seconds == 6.0
    assert all(
        stage < defaults.qa_deadline_seconds
        for stage in (
            defaults.qa_retrieval_budget_seconds,
            defaults.qa_evidence_assessment_budget_seconds,
            defaults.qa_generation_budget_seconds,
        )
    )

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
