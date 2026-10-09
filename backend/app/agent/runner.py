"""The agent loop.

    task -> LLM -> (tool calls -> ToolGateway -> results -> LLM)* -> final answer

The agent holds only an LLM client and a ToolGateway. It never imports tool
code; every tool request it receives from the model goes through
``gateway.call``. The architecture tests enforce this.

Limits (identical in every gateway mode - the code below never looks at the mode):
* ``max_steps``: maximum number of LLM calls in a run.
* ``run_timeout_seconds``: checked before every LLM call, and the LLM call is
  given only the remaining time.
* tool requests: enforced by the gateway via ``RunContext.limits``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.agent.models import (
    AgentLimits,
    LLMCallRecord,
    RunError,
    RunResult,
    RunStatus,
    UsageTotals,
)
from app.agent.prompts import SYSTEM_PROMPT
from app.gateway.context import RunContext
from app.gateway.gateway import ToolGateway
from app.gateway.models import ToolCallRequest, ToolResult
from app.llm.base import (
    AssistantMessage,
    LLMClient,
    LLMRequest,
    Message,
    ToolResultItem,
    ToolResultsMessage,
    UserMessage,
)
from app.llm.errors import LLMError
from app.telemetry.clock import Clock, SystemClock
from app.telemetry.trace import InMemoryTraceRecorder, TraceEvent, TraceRecorder

logger = logging.getLogger(__name__)


class AgentRunner:
    def __init__(
        self,
        llm: LLMClient,
        gateway: ToolGateway,
        *,
        limits: AgentLimits | None = None,
        recorder: TraceRecorder | None = None,
        clock: Clock | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self._llm = llm
        self._gateway = gateway
        self._limits = limits or AgentLimits()
        self._recorder: TraceRecorder = recorder or InMemoryTraceRecorder()
        self._clock = clock or SystemClock()
        self._system_prompt = system_prompt
        self._trace_errors = 0

    @property
    def trace_errors(self) -> int:
        return self._trace_errors

    async def run(self, task: str, ctx: RunContext, *, tools: list[str] | None = None) -> RunResult:
        task = task.strip() if isinstance(task, str) else ""
        if not task:
            raise ValueError("task must be a non-empty string")
        if len(task) > self._limits.max_task_chars:
            raise ValueError(f"task is longer than {self._limits.max_task_chars} characters")
        if not ctx.is_fresh:
            raise ValueError("each run needs a fresh RunContext")

        declarations = self._gateway.tool_declarations(tools)
        started_at = self._clock.utcnow().isoformat()
        t0 = self._clock.monotonic()
        deadline = t0 + self._limits.run_timeout_seconds
        ctx.deadline = deadline  # no tool may start or keep running past it

        self._emit(
            ctx,
            "run.started",
            {
                "task": task,
                "mode": self._gateway.mode.value,
                "provider": self._llm.provider,
                "model_id": self._llm.model_id,
                "tools_offered": [d["name"] for d in declarations],
                "max_steps": self._limits.max_steps,
                "max_tool_requests": ctx.limits.max_tool_requests,
                "run_timeout_seconds": self._limits.run_timeout_seconds,
            },
        )

        messages: list[Message] = [UserMessage(text=task)]
        llm_calls: list[LLMCallRecord] = []
        tool_results: list[ToolResult] = []
        status: RunStatus | None = None
        final_answer: str | None = None
        error: RunError | None = None
        steps = 0

        try:
            while True:
                if steps >= self._limits.max_steps:
                    status = RunStatus.LIMIT_STEPS
                    error = RunError(
                        code="max_steps",
                        message=f"no final answer after {self._limits.max_steps} LLM calls",
                    )
                    break
                remaining = deadline - self._clock.monotonic()
                if remaining <= 0:
                    status, error = RunStatus.LIMIT_TIME, _timeout_error(self._limits)
                    break

                steps += 1
                request = LLMRequest(
                    system_prompt=self._system_prompt,
                    messages=list(messages),
                    tools=declarations,
                )
                t_llm = self._clock.monotonic()
                try:
                    response = await asyncio.wait_for(self._llm.generate(request), remaining)
                except TimeoutError:
                    llm_calls.append(self._record_llm_failure(ctx, steps, t_llm, "run_timeout"))
                    status, error = RunStatus.LIMIT_TIME, _timeout_error(self._limits)
                    break
                except LLMError as exc:
                    llm_calls.append(self._record_llm_failure(ctx, steps, t_llm, exc.code))
                    status = RunStatus.FAILED
                    error = RunError(code=exc.code, message=exc.message)
                    break

                record = LLMCallRecord(
                    step=steps,
                    provider=self._llm.provider,
                    model_id=response.model_id,
                    model_version=response.model_version,
                    ok=True,
                    finish_reason=response.finish_reason,
                    latency_ms=(self._clock.monotonic() - t_llm) * 1000,
                    usage=response.usage,
                    tool_calls_requested=[c.name for c in response.tool_calls],
                )
                llm_calls.append(record)
                self._emit(ctx, "llm_call.completed", record.model_dump(mode="json"))

                if response.tool_calls:
                    messages.append(
                        AssistantMessage(
                            text=response.text,
                            tool_calls=response.tool_calls,
                            provider_payload=response.provider_payload,
                        )
                    )
                    items: list[ToolResultItem] = []
                    for call in response.tool_calls:
                        # The ONLY way a tool runs: through the gateway.
                        result = await self._gateway.call(
                            ToolCallRequest.from_function_call(call.name, call.args, call.id),
                            ctx,
                        )
                        tool_results.append(result)
                        items.append(
                            ToolResultItem(
                                call_id=call.id,
                                name=call.name,
                                response=result.for_model(),
                                provider_call_id=call.provider_call_id,
                            )
                        )
                    messages.append(ToolResultsMessage(results=items))
                    if ctx.stop_reason is not None:
                        status, error = _stop_status(ctx)
                        break
                    continue

                if not response.text:
                    status = RunStatus.FAILED
                    error = RunError(
                        code="empty_response",
                        message="the model returned neither a final answer nor a tool call",
                    )
                    break
                status, final_answer = RunStatus.SUCCESS, response.text
                break
        except Exception:
            logger.exception("unexpected error in agent run %s", ctx.run_id)
            status = RunStatus.ERROR
            error = RunError(code="internal_error", message="the agent run failed unexpectedly")

        assert status is not None
        result = RunResult(
            run_id=ctx.run_id,
            session_id=ctx.session_id,
            mode=self._gateway.mode.value,
            task=task,
            status=status,
            final_answer=final_answer,
            error=error,
            steps=steps,
            llm_calls=llm_calls,
            tool_results=tool_results,
            usage=_usage_totals(llm_calls),
            started_at=started_at,
            ended_at=self._clock.utcnow().isoformat(),
            duration_ms=(self._clock.monotonic() - t0) * 1000,
            llm_time_ms=sum(c.latency_ms for c in llm_calls),
            tool_time_ms=sum(r.duration_ms for r in tool_results),
        )
        self._emit(
            ctx,
            "run.completed",
            {
                **result.summary(),
                "tool_decision_sequence": [r.decision.value for r in tool_results],
            },
        )
        return result

    # ------------------------------------------------------------ helpers
    def _record_llm_failure(
        self, ctx: RunContext, step: int, t_llm: float, code: str
    ) -> LLMCallRecord:
        record = LLMCallRecord(
            step=step,
            provider=self._llm.provider,
            model_id=self._llm.model_id,
            ok=False,
            error_code=code,
            latency_ms=(self._clock.monotonic() - t_llm) * 1000,
        )
        self._emit(ctx, "llm_call.failed", record.model_dump(mode="json"))
        return record

    def _emit(self, ctx: RunContext, event_type: str, data: dict[str, Any]) -> None:
        try:
            self._recorder.record(
                TraceEvent(
                    type=event_type,
                    run_id=ctx.run_id,
                    session_id=ctx.session_id,
                    at=self._clock.utcnow().isoformat(),
                    data=data,
                )
            )
        except Exception:
            self._trace_errors += 1
            logger.exception("failed to record trace event %s", event_type)


def _stop_status(ctx: RunContext) -> tuple[RunStatus, RunError]:
    """Translate the gateway's stop reason into a run status."""
    reason = ctx.stop_reason
    if reason == "max_identical_requests":
        return RunStatus.LOOP_DETECTED, RunError(
            code=reason,
            message="the same tool request was repeated more than "
            f"{ctx.limits.max_identical_requests} times",
        )
    if reason == "run_timeout":
        return RunStatus.LIMIT_TIME, RunError(
            code=reason, message="the run's time limit was reached during tool calls"
        )
    return RunStatus.LIMIT_TOOL_CALLS, RunError(
        code=reason or "max_tool_requests",
        message=f"tool request limit of {ctx.limits.max_tool_requests} reached",
    )


def _timeout_error(limits: AgentLimits) -> RunError:
    return RunError(
        code="run_timeout",
        message=f"the run exceeded {limits.run_timeout_seconds:g} seconds",
    )


def _usage_totals(calls: list[LLMCallRecord]) -> UsageTotals:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "cached_input_tokens": 0,
        "total_tokens": 0,
    }
    complete = True
    with_usage = 0
    for call in calls:
        if not call.ok:
            continue
        usage = call.usage
        if usage is None or not usage.is_complete:
            complete = False
        if usage is None:
            continue
        with_usage += 1
        for key in totals:
            totals[key] += getattr(usage, key) or 0
    if with_usage == 0:
        return UsageTotals(complete=complete and not any(c.ok for c in calls))
    return UsageTotals(**totals, complete=complete, calls_with_usage=with_usage)
