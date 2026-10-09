"""Data shapes that go into and come out of the ToolGateway."""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class GatewayMode(StrEnum):
    PASSTHROUGH = "passthrough"  # baseline: every permitted request executes
    OPTIMIZED = "optimized"  # eligible read-only results may be reused


class Decision(StrEnum):
    # The tool ran
    EXECUTED_LIVE = "executed_live"  # normal execution (cache miss or not cacheable)
    STALE_REFRESHED = "stale_refreshed"  # a cached result existed but was too old
    FORCED_FRESH = "forced_fresh"  # caller asked for a fresh result
    FAILED = "failed"  # the tool was invoked and reported an error
    # The tool did NOT run - an earlier result was reused (optimized mode only)
    CACHE_HIT = "cache_hit"  # result from an earlier run in the same scope
    REUSED_IN_RUN = "reused_in_run"  # result from earlier in this same run
    SHARED_IN_FLIGHT = "shared_in_flight"  # joined an identical request already running
    # The tool did NOT run - the request was refused
    BLOCKED_UNKNOWN_TOOL = "blocked_unknown_tool"
    BLOCKED_INVALID_ARGS = "blocked_invalid_args"
    BLOCKED_PERMISSION = "blocked_permission"
    BLOCKED_LIMIT = "blocked_limit"
    BLOCKED_RUN_STOPPED = "blocked_run_stopped"
    BLOCKED_SAFETY_ERROR = "blocked_safety_error"  # a safety check crashed -> fail closed


REUSE_DECISIONS = frozenset({Decision.CACHE_HIT, Decision.REUSED_IN_RUN, Decision.SHARED_IN_FLIGHT})


class ResultSource(StrEnum):
    LIVE = "live"  # produced by running the tool for this request
    CACHE = "cache"  # stored result from an earlier run
    IN_RUN = "in_run"  # stored result from earlier in this run
    IN_FLIGHT = "in_flight"  # shared from a concurrent identical execution
    NONE = "none"  # no tool output (blocked before execution)


class ToolCallRequest(BaseModel):
    """One tool request from the agent.

    ``args`` is typed ``Any`` on purpose: whatever the LLM produced is accepted
    here and validated by the gateway, so malformed requests become a clear
    ``BLOCKED_INVALID_ARGS`` result instead of a crash.

    ``force_fresh`` asks for a new result instead of a reused one. The model can
    ask for the same thing with an optional ``force_fresh`` argument.
    """

    model_config = ConfigDict(frozen=True)

    tool_name: str
    args: Any = Field(default_factory=dict)
    call_id: str = Field(default_factory=lambda: uuid4().hex)
    force_fresh: bool = False

    @classmethod
    def from_function_call(
        cls, name: str, args: Any, call_id: str | None = None
    ) -> ToolCallRequest:
        """Build a request from an LLM function call (e.g. Gemini's name + args)."""
        if call_id:
            return cls(tool_name=name, args=args if args is not None else {}, call_id=call_id)
        return cls(tool_name=name, args=args if args is not None else {})


class ToolErrorInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    retryable: bool = False


class ShadowObservation(BaseModel):
    """A shadow rule would have reused a stored result, but the request executed
    live (shadow rules never change behavior). ``matched`` says whether the
    stored result equals the live one: a mismatch means the rule would have
    served out-of-date data."""

    model_config = ConfigDict(frozen=True)

    rule_ids: list[str]
    would_reuse: bool = True
    matched: bool
    age_seconds: float


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_id: str
    tool_name: str
    decision: Decision
    ok: bool
    executed: bool  # True if the tool's code was invoked for this request
    executions: int = 0  # how many times (1 + retries); 0 for reused/blocked
    source: ResultSource
    output: Any = None
    error: ToolErrorInfo | None = None
    started_at: str
    ended_at: str
    duration_ms: float  # whole gateway call
    tool_duration_ms: float | None = None  # time inside the tool (all attempts)
    cache_status: str | None = None  # disabled | not_eligible | miss | hit | stale | ...
    cache_age_seconds: float | None = None  # age of a reused result
    forced_fresh: bool = False
    invalidated_entries: int = 0
    note: str | None = None  # short explanation shown to the model for reused results
    # Evidence for learned rules (Stage 7). None when not applicable.
    fingerprint: str | None = None  # identifies "the same request" (tool+version+args)
    output_digest: str | None = None  # sha256 of the output, for successful results
    data_time: float | None = None  # data-clock seconds when the request started
    learned_rules: list[str] = Field(default_factory=list)  # ACTIVE rules that changed this
    shadow: ShadowObservation | None = None  # what a SHADOW rule would have done

    @property
    def reused(self) -> bool:
        return self.decision in REUSE_DECISIONS

    def for_model(self) -> dict[str, Any]:
        """The part of the result the LLM should see (no timings or internals)."""
        if self.ok:
            body: dict[str, Any] = {"ok": True, "result": self.output}
            if self.note:
                body["note"] = self.note
            return body
        assert self.error is not None
        return {
            "ok": False,
            "error": {"code": self.error.code, "message": self.error.message},
        }
