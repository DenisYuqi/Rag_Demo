"""Environment-backed application settings with safe diagnostics."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_mvp import __version__


class Settings(BaseSettings):
    """Validated runtime settings.

    The default ``offline`` backend makes development and deterministic tests usable
    without credentials. Selecting ``openai`` keeps the process live but makes provider
    readiness fail safely until all required runtime configuration is supplied.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_MVP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    service_name: str = "rag-mvp"
    service_version: str = __version__
    environment: Literal["development", "test", "production"] = "development"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    data_root: Path = Path("data")
    workbench_enabled: bool = True
    workbench_path: str = "/workbench"

    provider_backend: Literal["offline", "openai"] = "offline"
    openai_api_key: SecretStr | None = Field(default=None, repr=False)
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = Field(default=1536, ge=1, le=65536)
    generation_model: str = "gpt-4.1-mini"
    reranking_model: str | None = None
    provider_timeout_seconds: float = Field(default=8.0, gt=0, le=60)
    provider_retry_limit: int = Field(default=1, ge=0, le=5)

    default_retrieval_mode: Literal["dense", "hybrid", "hybrid-rerank"] = "hybrid"
    dense_candidate_limit: int = Field(default=20, ge=1, le=100)
    lexical_candidate_limit: int = Field(default=20, ge=1, le=100)
    rerank_candidate_limit: int = Field(default=10, ge=1, le=50)
    context_chunk_limit: int = Field(default=5, ge=1, le=20)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    dense_weight: float = Field(default=1.0, gt=0, le=10)
    lexical_weight: float = Field(default=1.0, gt=0, le=10)

    upload_max_bytes: int = Field(default=25 * 1024 * 1024, ge=1)
    chunk_target_tokens: int = Field(default=500, ge=64, le=4096)
    chunk_overlap_tokens: int = Field(default=80, ge=0, le=1024)
    ocr_enabled: bool = True
    ocr_languages: str = "chi_sim+eng"

    qa_max_active: int = Field(default=5, ge=5, le=100)
    qa_max_queue: int = Field(default=10, ge=0, le=1000)
    qa_deadline_seconds: float = Field(default=9.5, gt=0, le=60)
    rerank_deadline_seconds: float = Field(default=1.2, gt=0, le=10)
    shutdown_grace_seconds: float = Field(default=15.0, gt=0, le=120)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    telemetry_exporter: Literal["none", "console", "otlp"] = "none"
    pricing_version: str = "unconfigured"

    @field_validator("workbench_path")
    @classmethod
    def validate_workbench_path(cls, value: str) -> str:
        normalized = "/" + value.strip().strip("/")
        reserved = {"/", "/api", "/api/v1", "/healthz", "/readyz", "/metrics"}
        if normalized in reserved or normalized.startswith("/api/"):
            raise ValueError("workbench path conflicts with a reserved route")
        return normalized

    @field_validator("openai_base_url")
    @classmethod
    def validate_openai_base_url(cls, value: str) -> str:
        normalized = value.rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("provider base URL must use http or https")
        return normalized

    @model_validator(mode="after")
    def validate_limits(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            raise ValueError("chunk overlap must be smaller than chunk target")
        if self.rerank_candidate_limit < self.context_chunk_limit:
            raise ValueError("rerank candidate limit cannot be below context chunk limit")
        if self.rerank_deadline_seconds >= self.qa_deadline_seconds:
            raise ValueError("rerank deadline must be below the total QA deadline")
        return self

    def provider_readiness_errors(self) -> tuple[str, ...]:
        """Return safe configuration categories without exposing secret values."""
        if self.provider_backend == "offline":
            return ()
        errors: list[str] = []
        if self.openai_api_key is None or not self.openai_api_key.get_secret_value():
            errors.append("provider_credentials_missing")
        if not self.embedding_model:
            errors.append("embedding_model_missing")
        if not self.generation_model:
            errors.append("generation_model_missing")
        return tuple(errors)

    def safe_dump(self) -> dict[str, Any]:
        """Return diagnostics-safe settings with credentials replaced, never revealed."""
        values = self.model_dump(mode="json")
        values["openai_api_key"] = (
            "[REDACTED_SECRET]" if self.openai_api_key is not None else None
        )
        return values

    @property
    def configuration_identity(self) -> str:
        """Hash non-secret configuration for logs, caches, and reports."""
        payload = self.safe_dump()
        payload.pop("openai_api_key", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once for process-level application composition."""
    return Settings()
