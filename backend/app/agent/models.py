"""Shapes for agent runs: limits, statuses, and the recorded result of a run."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.gateway.models import ToolResult
from app.llm.base import TokenUsage


@dataclass(frozen=True)
class AgentLimits:
    """Hard limits on the agent loop. They never depend on the gateway mode.

    The tool-request limit lives in ``RunContext.limits`` because the gateway
    enforces it on every request.
    """

    max_steps: int = 12  # one step = one LLM call
    run_timeout_seconds: float = 90.0
    max_task_chars: int = 4000

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.run_timeout_seconds <= 0:
            raise ValueError("run_timeout_seconds must be positive")


class RunStatus(StrEnum):
    SUCCESS = "success"  # the model gave a final answer
    FAILED = "failed"  # the LLM failed or returned an unusable response
    LIMIT_STEPS = "limit_steps"  # max_steps reached without a final answer
    LIMIT_TOOL_CALLS = "limit_tool_calls"  # the gateway's request limit was reached
    LOOP_DETECTED = "loop_detected"  # the same tool request was repeated too often
    LIMIT_TIME = "limit_time"  # the run timeout was reached
    ERROR = "error"  # unexpected internal error in our code


class RunError(BaseModel):
    model_config = ConfigDict(frozen=True)
    code: str
    message: str


class LLMCallRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    step: int
    provider: str
    model_id: str
    model_version: str | None = None
    ok: bool
    error_code: str | None = None
    finish_reason: str | None = None
    latency_ms: float
    usage: TokenUsage | None = None
    tool_calls_requested: list[str] = Field(default_factory=list)


class UsageTotals(BaseModel):
    """Token totals across the run's LLM calls.

    ``complete`` is False when at least one successful call did not report
    input and output tokens - totals are then a lower bound, not the true value.
    When no call reported usage at all, the totals are ``None`` ("unavailable"),
    never 0. Failed calls report no usage.
    """

    model_config = ConfigDict(frozen=True)

    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    cached_input_tokens: int | None = None
    total_tokens: int | None = None
    complete: bool = True
    calls_with_usage: int = 0


class RunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    session_id: str
    mode: str
    task: str
    status: RunStatus
    final_answer: str | None = None
    error: RunError | None = None
    steps: int
    llm_calls: list[LLMCallRecord]
    tool_results: list[ToolResult]
    usage: UsageTotals
    started_at: str
    ended_at: str
    duration_ms: float
    llm_time_ms: float
    tool_time_ms: float

    @property
    def llm_call_count(self) -> int:
        return len(self.llm_calls)

    @property
    def tool_request_count(self) -> int:
        return len(self.tool_results)

    @property
    def tool_execution_count(self) -> int:
        """Actual tool executions, including retries (reused results count 0)."""
        return sum(r.executions for r in self.tool_results)

    def tool_decision_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.tool_results:
            counts[r.decision.value] = counts.get(r.decision.value, 0) + 1
        return counts

    def summary(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "status": self.status.value,
            "final_answer": self.final_answer,
            "error": self.error.model_dump() if self.error else None,
            "steps": self.steps,
            "llm_calls": self.llm_call_count,
            "tool_requests": self.tool_request_count,
            "tool_executions": self.tool_execution_count,
            "reused_results": sum(1 for r in self.tool_results if r.reused),
            "tool_retries": sum(max(0, r.executions - 1) for r in self.tool_results),
            "tool_decisions": self.tool_decision_counts(),
            "tokens": self.usage.model_dump(),
            "models": sorted({c.model_version or c.model_id for c in self.llm_calls}),
            "duration_ms": round(self.duration_ms, 1),
            "llm_time_ms": round(self.llm_time_ms, 1),
            "tool_time_ms": round(self.tool_time_ms, 1),
        }
