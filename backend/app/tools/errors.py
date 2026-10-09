"""Errors a tool may raise on purpose.

The gateway turns these into structured results the agent can read. Anything a
tool raises that is *not* one of these is treated as an internal tool bug.
"""

from __future__ import annotations


class ToolError(Exception):
    """Base class for expected tool failures."""

    code = "tool_error"
    retryable = False

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ToolInputError(ToolError):
    """The request was understood but cannot be satisfied (permanent).

    Example: dividing by zero, or asking for a document that does not exist.
    Retrying the same request will fail the same way.
    """

    code = "invalid_request"


class ToolNotFoundError(ToolInputError):
    """The requested item (document, record, symbol) does not exist."""

    code = "not_found"


class ToolTransientError(ToolError):
    """A temporary failure (timeout, rate limit, outage). May succeed later."""

    code = "transient"
    retryable = True
