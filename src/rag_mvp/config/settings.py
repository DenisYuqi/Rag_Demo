"""Environment-backed application settings with safe diagnostics."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Literal
from urllib.parse import urlsplit

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
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    CONTAINER_STOP_GRACE_SECONDS: ClassVar[float] = 20.0
    _EVALUATION_RUNTIME_ONLY_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "data_root",
            "evaluation_dataset_root",
            "evaluation_release_root",
            "evaluation_max_active_jobs",
            "evaluation_shutdown_grace_seconds",
            "host",
            "log_level",
            "openai_api_key_file",
            "port",
            "server_shutdown_grace_seconds",
            "shutdown_grace_seconds",
            "telemetry_export_timeout_seconds",
            "telemetry_exporter",
            "telemetry_otlp_traces_endpoint",
            "upload_max_bytes",
            "workbench_enabled",
            "workbench_path",
        }
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
    openai_api_key_file: Path | None = Field(default=None, repr=False)
    openai_base_url: str = "https://api.openai.com/v1"
    openai_proxy_url: SecretStr | None = Field(default=None, repr=False)
    openai_send_dimensions: bool = True
    openai_max_tokens_parameter: Literal["max_tokens", "max_completion_tokens"] = (
        "max_completion_tokens"
    )
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = Field(default=1536, ge=1, le=65536)
    generation_model: str = "gpt-4.1-mini"
    reranking_model: str | None = None
    provider_timeout_seconds: float = Field(default=8.0, gt=0, le=60)
    provider_retry_limit: int = Field(default=1, ge=0, le=5)

    default_retrieval_profile: Literal["openai-api", "bge-local"] = "openai-api"
    bge_profile_enabled: bool = True
    bge_data_root: Path | None = None
    bge_embedding_model: str = "BAAI/bge-m3"
    bge_embedding_dimension: int = Field(default=1024, ge=1, le=65536)
    bge_reranking_model: str = "BAAI/bge-reranker-v2-m3"
    bge_device: str = "auto"
    bge_use_fp16: bool = False
    bge_embedding_batch_size: int = Field(default=8, ge=1, le=256)
    bge_reranking_batch_size: int = Field(default=8, ge=1, le=256)
    bge_embedding_max_length: int = Field(default=8192, ge=64, le=8192)
    bge_reranking_max_length: int = Field(default=1024, ge=64, le=8192)
    bge_model_cache_dir: Path | None = None
    bge_provider_timeout_seconds: float = Field(default=60.0, gt=0, le=60)
    bge_qa_deadline_seconds: float = Field(default=45.0, gt=0, le=60)
    bge_qa_retrieval_budget_seconds: float = Field(default=20.0, gt=0, le=60)
    bge_rerank_deadline_seconds: float = Field(default=10.0, gt=0, le=60)
    bge_qa_evidence_assessment_budget_seconds: float = Field(default=10.0, gt=0, le=60)

    default_retrieval_mode: Literal["dense", "hybrid", "hybrid-rerank"] = "hybrid"
    dense_candidate_limit: int = Field(default=20, ge=1, le=100)
    lexical_candidate_limit: int = Field(default=20, ge=1, le=100)
    rerank_candidate_limit: int = Field(default=10, ge=1, le=50)
    context_chunk_limit: int = Field(default=5, ge=1, le=20)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    dense_weight: float = Field(default=1.0, gt=0, le=10)
    lexical_weight: float = Field(default=1.0, gt=0, le=10)
    allow_single_retriever_degradation: bool = False
    retrieval_cache_enabled: bool = False
    retrieval_cache_max_entries: int = Field(default=256, ge=1, le=10_000)
    retrieval_cache_ttl_seconds: float = Field(default=300.0, gt=0, le=86_400)

    evaluation_dataset_root: Path = Path("evaluations/datasets")
    evaluation_release_root: Path = Path("evaluations/releases")
    evaluation_max_active_jobs: int = Field(default=1, ge=1, le=4)
    evaluation_shutdown_grace_seconds: float = Field(default=2.0, ge=0, le=10)

    upload_max_bytes: int = Field(default=25 * 1024 * 1024, ge=1)
    parent_chunk_target_tokens: int = Field(default=1536, ge=64, le=8192)
    chunk_target_tokens: int = Field(default=512, ge=64, le=4096)
    chunk_overlap_tokens: int = Field(default=128, ge=0, le=1024)
    ocr_enabled: bool = True
    ocr_languages: str = "chi_sim+eng"

    qa_max_active: int = Field(default=5, ge=5, le=100)
    qa_max_queue: int = Field(default=10, ge=0, le=1000)
    qa_deadline_seconds: float = Field(default=9.5, gt=0, le=60)
    qa_queue_budget_seconds: float = Field(default=0.2, gt=0, le=60)
    qa_validation_budget_seconds: float = Field(default=0.8, gt=0, le=60)
    qa_retrieval_budget_seconds: float = Field(default=4.0, gt=0, le=60)
    qa_embedding_budget_seconds: float = Field(default=0.8, gt=0, le=60)
    qa_dense_retrieval_budget_seconds: float = Field(default=0.8, gt=0, le=60)
    qa_bm25_budget_seconds: float = Field(default=0.8, gt=0, le=60)
    qa_fusion_budget_seconds: float = Field(default=0.2, gt=0, le=60)
    rerank_deadline_seconds: float = Field(default=3.0, gt=0, le=10)
    qa_evidence_assessment_budget_seconds: float = Field(default=4.0, gt=0, le=60)
    qa_minimum_support_score: float = Field(default=0.45, gt=0, le=1)
    qa_generation_budget_seconds: float = Field(default=6.0, gt=0, le=60)
    qa_grounding_budget_seconds: float = Field(default=0.3, gt=0, le=60)
    qa_redaction_budget_seconds: float = Field(default=0.2, gt=0, le=60)
    qa_serialization_budget_seconds: float = Field(default=0.1, gt=0, le=60)
    qa_finalization_budget_seconds: float = Field(default=0.6, gt=0, le=60)
    server_shutdown_grace_seconds: int = Field(default=4, ge=1, le=19)
    shutdown_grace_seconds: float = Field(default=15.0, gt=0, lt=20)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    telemetry_exporter: Literal["none", "console", "otlp"] = "none"
    telemetry_otlp_traces_endpoint: str | None = None
    telemetry_export_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
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

    @field_validator("telemetry_otlp_traces_endpoint", mode="before")
    @classmethod
    def validate_telemetry_otlp_traces_endpoint(cls, value: object) -> object:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        normalized = str(value).strip().rstrip("/")
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("OTLP traces endpoint must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("OTLP traces endpoint must not contain credentials or query data")
        return normalized

    @field_validator("reranking_model", mode="before")
    @classmethod
    def normalize_optional_reranking_model(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "bge_embedding_model",
        "bge_reranking_model",
        "bge_device",
        mode="before",
    )
    @classmethod
    def normalize_required_bge_text(cls, value: object) -> object:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("BGE model and device settings must not be empty")
        return value.strip()

    @field_validator("openai_proxy_url", mode="before")
    @classmethod
    def validate_openai_proxy_url(cls, value: object) -> object:
        if value is None or value == "":
            return None
        raw_value = value.get_secret_value() if isinstance(value, SecretStr) else str(value)
        parsed = urlsplit(raw_value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("provider proxy URL must be an absolute HTTP(S) URL")
        return raw_value.rstrip("/")

    @model_validator(mode="after")
    def validate_limits(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            raise ValueError("chunk overlap must be smaller than chunk target")
        if self.parent_chunk_target_tokens < self.chunk_target_tokens:
            raise ValueError("parent chunk target cannot be below child chunk target")
        if self.rerank_candidate_limit < self.context_chunk_limit:
            raise ValueError("rerank candidate limit cannot be below context chunk limit")
        stage_budgets = {
            "queue": self.qa_queue_budget_seconds,
            "validation": self.qa_validation_budget_seconds,
            "retrieval": self.qa_retrieval_budget_seconds,
            "embedding": self.qa_embedding_budget_seconds,
            "dense retrieval": self.qa_dense_retrieval_budget_seconds,
            "BM25": self.qa_bm25_budget_seconds,
            "fusion": self.qa_fusion_budget_seconds,
            "rerank": self.rerank_deadline_seconds,
            "evidence assessment": self.qa_evidence_assessment_budget_seconds,
            "generation": self.qa_generation_budget_seconds,
            "grounding": self.qa_grounding_budget_seconds,
            "redaction": self.qa_redaction_budget_seconds,
            "serialization": self.qa_serialization_budget_seconds,
            "finalization": self.qa_finalization_budget_seconds,
        }
        if any(value >= self.qa_deadline_seconds for value in stage_budgets.values()):
            raise ValueError("each stage budget must be below the total QA deadline")
        if (
            self.qa_generation_budget_seconds + self.qa_finalization_budget_seconds
            > self.qa_deadline_seconds
        ):
            raise ValueError("generation and finalization budgets exceed the total QA deadline")
        if self.default_retrieval_mode == "hybrid-rerank" and self.reranking_model is None:
            raise ValueError("hybrid-rerank default requires a configured reranking model")
        if self.default_retrieval_profile == "bge-local" and not self.bge_profile_enabled:
            raise ValueError("default BGE retrieval profile must be enabled")
        if self.bge_embedding_model == "BAAI/bge-m3" and self.bge_embedding_dimension != 1024:
            raise ValueError("BAAI/bge-m3 requires a 1024-dimensional embedding space")
        if self.resolved_bge_data_root == self.data_root.resolve(strict=False):
            raise ValueError("BGE and OpenAI profiles must use different data roots")
        bge_stage_budgets = {
            "retrieval": self.bge_qa_retrieval_budget_seconds,
            "rerank": self.bge_rerank_deadline_seconds,
            "evidence assessment": self.bge_qa_evidence_assessment_budget_seconds,
            "generation": self.qa_generation_budget_seconds,
            "finalization": self.qa_finalization_budget_seconds,
        }
        if any(value >= self.bge_qa_deadline_seconds for value in bge_stage_budgets.values()):
            raise ValueError("each BGE stage budget must be below the BGE total QA deadline")
        if (
            self.qa_generation_budget_seconds + self.qa_finalization_budget_seconds
            > self.bge_qa_deadline_seconds
        ):
            raise ValueError("generation and finalization budgets exceed the BGE total QA deadline")
        if self.total_shutdown_budget_seconds >= self.CONTAINER_STOP_GRACE_SECONDS:
            raise ValueError("server and application shutdown budgets must fit container grace")
        if self.openai_api_key is None and self.openai_api_key_file is not None:
            try:
                secret = self.openai_api_key_file.read_text(encoding="utf-8").strip()
            except OSError:
                secret = ""
            if secret:
                self.openai_api_key = SecretStr(secret)
        return self

    @property
    def app_shutdown_grace_seconds(self) -> float:
        """Budget reserved for ASGI lifespan cleanup after server-side draining."""

        return self.shutdown_grace_seconds

    @property
    def resolved_bge_data_root(self) -> Path:
        configured = self.bge_data_root
        if configured is None:
            configured = self.data_root / "profiles" / "bge-local"
        return configured.resolve(strict=False)

    def bge_profile_settings(self) -> Settings:
        """Return a truthful generic settings view for the isolated local profile."""

        return self.model_copy(
            update={
                "data_root": self.resolved_bge_data_root,
                "embedding_model": self.bge_embedding_model,
                "embedding_dimension": self.bge_embedding_dimension,
                "reranking_model": self.bge_reranking_model,
                "default_retrieval_mode": "hybrid-rerank",
                "provider_timeout_seconds": self.bge_provider_timeout_seconds,
                "qa_deadline_seconds": self.bge_qa_deadline_seconds,
                "qa_retrieval_budget_seconds": self.bge_qa_retrieval_budget_seconds,
                "rerank_deadline_seconds": self.bge_rerank_deadline_seconds,
                "qa_evidence_assessment_budget_seconds": (
                    self.bge_qa_evidence_assessment_budget_seconds
                ),
                "pricing_version": "unconfigured",
                "workbench_enabled": False,
            }
        )

    @property
    def total_shutdown_budget_seconds(self) -> float:
        """Maximum configured server drain plus application cleanup duration."""

        return self.server_shutdown_grace_seconds + self.app_shutdown_grace_seconds

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

    def telemetry_readiness_errors(self) -> tuple[str, ...]:
        """Return safe telemetry configuration categories without probing the network."""

        if self.telemetry_exporter == "otlp" and self.telemetry_otlp_traces_endpoint is None:
            return ("telemetry_otlp_endpoint_missing",)
        return ()

    def safe_dump(self) -> dict[str, Any]:
        """Return diagnostics-safe settings with credentials replaced, never revealed."""
        values = self.model_dump(mode="json")
        values["openai_api_key"] = "[REDACTED_SECRET]" if self.openai_api_key is not None else None
        values["openai_proxy_url"] = (
            "[REDACTED_SECRET]" if self.openai_proxy_url is not None else None
        )
        return values

    @property
    def configuration_identity(self) -> str:
        """Backward-compatible alias for the complete runtime configuration hash."""

        return self.runtime_configuration_identity

    @property
    def runtime_configuration_identity(self) -> str:
        """Hash all non-secret runtime configuration for logs and cache isolation."""

        payload = self.safe_dump()
        payload.pop("openai_api_key", None)
        return _configuration_digest(payload)

    @property
    def evaluation_configuration_identity(self) -> str:
        """Hash behaviorally relevant evaluation settings independent of runtime paths.

        Evaluation workspaces are unique per run. Their storage paths, UI settings,
        lifecycle budgets, and telemetry destinations must not make otherwise identical
        experiment candidates incomparable. The complete runtime hash remains available
        separately for operational diagnostics.
        """

        payload = self.safe_dump()
        payload.pop("openai_api_key", None)
        for field_name in self._EVALUATION_RUNTIME_ONLY_FIELDS:
            payload.pop(field_name, None)
        return _configuration_digest(payload)


def _configuration_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once for process-level application composition."""
    return Settings()
