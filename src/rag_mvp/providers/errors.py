"""Safe provider errors and vendor-exception classification."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Final

from rag_mvp.providers.models import ModelAttempt, ProviderErrorCategory


_SAFE_MESSAGES: Final[Mapping[ProviderErrorCategory, str]] = {
    ProviderErrorCategory.NETWORK: "provider_network_error",
    ProviderErrorCategory.TIMEOUT: "provider_timeout",
    ProviderErrorCategory.RATE_LIMIT: "provider_rate_limited",
    ProviderErrorCategory.SERVER: "provider_server_error",
    ProviderErrorCategory.AUTHENTICATION: "provider_authentication_failed",
    ProviderErrorCategory.INVALID_REQUEST: "provider_invalid_request",
    ProviderErrorCategory.INCOMPATIBLE_RESPONSE: "provider_incompatible_response",
    ProviderErrorCategory.CANCELLED: "provider_call_cancelled",
    ProviderErrorCategory.DEADLINE_EXCEEDED: "provider_deadline_exceeded",
    ProviderErrorCategory.UNAVAILABLE: "provider_unavailable",
}

_TRANSIENT: Final[frozenset[ProviderErrorCategory]] = frozenset(
    {
        ProviderErrorCategory.NETWORK,
        ProviderErrorCategory.TIMEOUT,
        ProviderErrorCategory.RATE_LIMIT,
        ProviderErrorCategory.SERVER,
    }
)


class ProviderError(Exception):
    """An error safe to expose without the original provider payload."""

    def __init__(
        self,
        category: ProviderErrorCategory,
        *,
        retryable: bool | None = None,
        fallback_eligible: bool | None = None,
    ) -> None:
        self.category = category
        self.retryable = category in _TRANSIENT if retryable is None else retryable
        self.fallback_eligible = (
            category
            not in {
                ProviderErrorCategory.INVALID_REQUEST,
                ProviderErrorCategory.CANCELLED,
                ProviderErrorCategory.DEADLINE_EXCEEDED,
            }
            if fallback_eligible is None
            else fallback_eligible
        )
        super().__init__(_SAFE_MESSAGES[category])


class ProviderOperationError(ProviderError):
    """A logical provider operation failed after zero or more safe attempts."""

    def __init__(
        self,
        category: ProviderErrorCategory,
        attempts: tuple[ModelAttempt, ...] = (),
        *,
        retryable: bool | None = None,
        fallback_eligible: bool | None = None,
    ) -> None:
        self.attempts = attempts
        super().__init__(
            category,
            retryable=retryable,
            fallback_eligible=fallback_eligible,
        )


class ProviderConfigurationError(ValueError):
    """Safe startup-time role or route configuration failure."""


def classify_provider_exception(error: Exception) -> ProviderError:
    """Map an SDK/transport exception without copying its potentially unsafe text."""

    if isinstance(error, ProviderError):
        return error
    if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
        return ProviderError(ProviderErrorCategory.TIMEOUT)
    if isinstance(error, (ConnectionError, OSError)):
        return ProviderError(ProviderErrorCategory.NETWORK)

    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        if status_code == 429:
            return ProviderError(ProviderErrorCategory.RATE_LIMIT)
        if status_code in {401, 403}:
            return ProviderError(ProviderErrorCategory.AUTHENTICATION)
        if status_code >= 500:
            return ProviderError(ProviderErrorCategory.SERVER)
        if 400 <= status_code < 500:
            return ProviderError(ProviderErrorCategory.INVALID_REQUEST)

    class_name = type(error).__name__.lower()
    if "authentication" in class_name or "permissiondenied" in class_name:
        return ProviderError(ProviderErrorCategory.AUTHENTICATION)
    if "ratelimit" in class_name:
        return ProviderError(ProviderErrorCategory.RATE_LIMIT)
    if "timeout" in class_name:
        return ProviderError(ProviderErrorCategory.TIMEOUT)
    if "connection" in class_name:
        return ProviderError(ProviderErrorCategory.NETWORK)
    if "badrequest" in class_name or "unprocessable" in class_name:
        return ProviderError(ProviderErrorCategory.INVALID_REQUEST)
    return ProviderError(ProviderErrorCategory.UNAVAILABLE, retryable=False)

