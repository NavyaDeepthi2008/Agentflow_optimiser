"""Errors raised by LLM clients.

Every error has a short machine-readable ``code`` the agent records, and a
human-readable message that never contains the API key.
"""

from __future__ import annotations


class LLMError(Exception):
    code = "llm_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.retryable = retryable
        self.status_code = status_code


class LLMConfigurationError(LLMError):
    """The client cannot be used as configured (e.g. the API key is missing)."""

    code = "llm_not_configured"


class LLMProviderError(LLMError):
    """The provider failed: rate limit, outage, auth problem, timeout, network."""

    code = "provider_error"


class LLMResponseError(LLMError):
    """The provider answered, but the answer is unusable (empty, blocked, malformed)."""

    code = "malformed_response"


def redact(text: str, secret: str | None, max_length: int = 300) -> str:
    """Remove ``secret`` from ``text`` and shorten it, before it is stored or shown."""
    if secret:
        text = text.replace(secret, "[REDACTED]")
    if len(text) > max_length:
        text = text[: max_length - 3] + "..."
    return text
