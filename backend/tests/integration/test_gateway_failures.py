"""Failures inside tools, safety checks and telemetry."""

from __future__ import annotations

from app.gateway.context import RunContext
from app.gateway.factory import build_gateway
from app.gateway.models import Decision, ResultSource, ToolCallRequest
from app.safety.permissions import RunPermissions


async def test_permanent_tool_error(gateway, make_ctx, call):
    result = await call(make_ctx(), "doc_read", {"doc_id": "kb-999"})
    assert result.decision is Decision.FAILED
    assert result.executed  # the tool did run
    assert result.source is ResultSource.LIVE
    assert result.error.code == "not_found"
    assert result.error.retryable is False
    assert result.for_model()["ok"] is False


async def test_transient_error_is_retried_for_read_only_tools(gateway, make_ctx, call, world):
    # flaky_tool fails once with a transient error, then succeeds (fast retry policy
    # is not set here, so the default policy's small delay is used).
    result = await call(make_ctx(), "flaky_tool", {"text": "hi"})
    assert result.decision is Decision.EXECUTED_LIVE
    assert result.ok and result.output == {"echo": "hi"}
    assert result.executions == 2
    assert world.execution_counts["flaky_tool"] == 2
    assert gateway.stats["retries"] == 1


async def test_transient_error_after_retries_is_reported(gateway, make_ctx, call, world):
    world.flaky_fail_times = 10  # more failures than the retry budget (2 retries)
    result = await call(make_ctx(), "flaky_tool", {"text": "hi"})
    assert result.decision is Decision.FAILED
    assert result.error.code == "transient"
    assert result.executions == 3
    assert world.execution_counts["flaky_tool"] == 3


async def test_unexpected_tool_crash_is_contained(gateway, make_ctx, call, world, recorder):
    result = await call(make_ctx(), "crashing_tool", {})
    assert result.decision is Decision.FAILED
    assert result.error.code == "tool_internal_error"
    assert "secret-token-123" not in result.error.message
    assert world.execution_counts["crashing_tool"] == 1
    completed = recorder.events_for_call(result.call_id)[-1]
    assert "secret-token-123" not in str(completed.data)


async def test_non_json_output_is_rejected(gateway, make_ctx, call):
    result = await call(make_ctx(), "bad_output_tool", {})
    assert result.decision is Decision.FAILED
    assert result.error.code == "invalid_tool_output"
    assert result.output is None
    assert result.executed


class ExplodingPermissions(RunPermissions):
    def denial_reason(self, spec):  # noqa: ARG002
        raise RuntimeError("bug in safety check")


async def test_safety_check_crash_fails_closed(gateway, world):
    ctx = RunContext(session_id="s", permissions=ExplodingPermissions())
    result = await gateway.call(
        ToolCallRequest(tool_name="calculator", args={"expression": "1+1"}), ctx
    )
    assert result.decision is Decision.BLOCKED_SAFETY_ERROR
    assert not result.executed
    assert world.execution_counts["calculator"] == 0


class BrokenRecorder:
    def record(self, event):  # noqa: ARG002
        raise OSError("disk full")


async def test_trace_failure_does_not_break_the_call(world, make_ctx):
    gateway = build_gateway(world=world, recorder=BrokenRecorder())
    result = await gateway.call(
        ToolCallRequest(tool_name="calculator", args={"expression": "2*2"}), make_ctx()
    )
    assert result.ok and result.output["result"] == 4
    assert gateway.stats["trace_errors"] == 2  # requested + completed events both failed


async def test_every_request_produces_exactly_two_events(gateway, make_ctx, call, recorder):
    ctx = make_ctx(max_tool_requests=4)
    results = [
        await call(ctx, "calculator", {"expression": "1"}),
        await call(ctx, "nope", {}),
        await call(ctx, "calculator", {}),
        await call(ctx, "send_notification", {"recipient": "team:x", "message": "m"}),
        await call(ctx, "calculator", {"expression": "1"}),  # over the limit
    ]
    for r in results:
        types = [e.type for e in recorder.events_for_call(r.call_id)]
        assert types == ["tool_call.requested", "tool_call.completed"]
    decisions = [r.decision for r in results]
    assert decisions == [
        Decision.EXECUTED_LIVE,
        Decision.BLOCKED_UNKNOWN_TOOL,
        Decision.BLOCKED_INVALID_ARGS,
        Decision.BLOCKED_PERMISSION,
        Decision.BLOCKED_LIMIT,
    ]
