"""Safe construction of the shared asynchronous OpenAI-compatible client."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx
from openai import AsyncOpenAI


@dataclass(frozen=True, slots=True)
class OpenAIClientConfig:
    """Resolved runtime client configuration.

    The secret value is required to construct the SDK client but is deliberately
    excluded from representations.  APIs and diagnostics should retain only
    ``secret_reference``.
    """

    base_url: str
    api_key: str = field(repr=False)
    secret_reference: str
    timeout_seconds: float
    proxy_url: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("provider base URL must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("provider base URL must not contain credentials")
        if not self.api_key:
            raise ValueError("provider API key is required")
        if not self.secret_reference.strip():
            raise ValueError("provider secret reference is required")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("provider timeout must be positive and finite")
        if self.proxy_url is not None:
            parsed_proxy = urlsplit(self.proxy_url)
            if parsed_proxy.scheme not in {"http", "https"} or not parsed_proxy.netloc:
                raise ValueError("provider proxy URL must be an absolute HTTP(S) URL")


def create_async_openai_client(
    config: OpenAIClientConfig,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> AsyncOpenAI:
    """Build an SDK client with SDK retries disabled.

    Retry and deadline behavior belongs to :mod:`rag_mvp.providers.resilience`, where
    every attempt is visible to usage accounting.  Allowing hidden SDK retries would
    undercount attempts and could overrun the request deadline.
    """

    kwargs: dict[str, object] = {
        "api_key": config.api_key,
        "base_url": config.base_url.rstrip("/"),
        "timeout": config.timeout_seconds,
        "max_retries": 0,
    }
    resolved_http_client = http_client
    if resolved_http_client is None and config.proxy_url is not None:
        resolved_http_client = httpx.AsyncClient(
            proxy=config.proxy_url,
            timeout=config.timeout_seconds,
        )
    if resolved_http_client is not None:
        kwargs["http_client"] = resolved_http_client
    return AsyncOpenAI(**kwargs)  # type: ignore[arg-type]
