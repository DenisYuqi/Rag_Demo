from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from rag_mvp.config.settings import Settings


def test_defaults_are_safe_and_offline_ready(tmp_path: object) -> None:
    settings = Settings(data_root=tmp_path)

    assert settings.host == "127.0.0.1"
    assert settings.provider_readiness_errors() == ()
    assert settings.safe_dump()["openai_api_key"] is None
    assert len(settings.configuration_identity) == 16


def test_secret_never_appears_in_repr_or_safe_dump(tmp_path: object) -> None:
    raw_secret = "sk-test-do-not-log"
    settings = Settings(data_root=tmp_path, openai_api_key=SecretStr(raw_secret))

    assert raw_secret not in repr(settings)
    assert raw_secret not in str(settings.safe_dump())
    assert settings.safe_dump()["openai_api_key"] == "[REDACTED_SECRET]"


def test_openai_backend_reports_safe_missing_credentials(tmp_path: object) -> None:
    settings = Settings(data_root=tmp_path, provider_backend="openai")

    assert settings.provider_readiness_errors() == ("provider_credentials_missing",)


@pytest.mark.parametrize("path", ["/", "/api", "/api/v1", "/healthz", "/metrics"])
def test_reserved_workbench_paths_are_rejected(tmp_path: object, path: str) -> None:
    with pytest.raises(ValidationError):
        Settings(data_root=tmp_path, workbench_path=path)


def test_limit_invariants_are_validated(tmp_path: object) -> None:
    with pytest.raises(ValidationError):
        Settings(data_root=tmp_path, chunk_target_tokens=100, chunk_overlap_tokens=100)
