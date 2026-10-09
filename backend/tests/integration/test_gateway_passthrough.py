"""End-to-end behaviour of the ToolGateway in pass-through (baseline) mode."""

from __future__ import annotations

import asyncio

import pytest

from app.gateway.factory import build_gateway
from app.gateway.gateway import ToolGateway
from app.gateway.models import Decision, GatewayMode, ResultSource, ToolCallRequest
from app.tools.builtin.prices import TICK_SECONDS

# --------------------------------------------------------------- happy path


async def test_read_only_call_executes_and_is_traced(gateway, make_ctx, call, recorder, world):
    ctx = make_ctx()
    result = await call(ctx, "calculator", {"expression": "29 * 12"})

    assert result.decision is Decision.EXECUTED_LIVE
    assert result.ok and result.executed
    assert result.source is ResultSource.LIVE
    assert result.output == {"result": 348}
    assert result.for_model() == {"ok": True, "result": result.output}
    assert world.execution_counts["calculator"] == 1

    events = recorder.events_for_call(result.call_id)
    assert [e.type for e in events] == ["tool_call.requested", "tool_call.completed"]
    done = events[1].data
    assert done["decision"] == "executed_live"
    assert done["effect"] == "read_only"
    assert done["args"] == {"expression": "29 * 12"}
    assert done["started_at"] and done["ended_at"]
    assert done["duration_ms"] >= 0 and done["tool_duration_ms"] >= 0
    assert all(e.run_id == ctx.run_id and e.session_id == ctx.session_id for e in events)


async def test_timestamps_follow_the_clock(gateway, make_ctx, call, clock):
    clock.advance(42)
    result = await call(make_ctx(), "calculator", {"expression": "1+1"})
    assert result.started_at == clock.utcnow().isoformat()
    assert result.started_at.endswith("+00:00")  # UTC


async def test_real_durations_are_measured(world, recorder, make_ctx):
    from app.telemetry.clock import SystemClock
    from app.tools.mock_world import MockWorld

    gw = build_gateway(world=MockWorld.create(SystemClock()), include_test_tools=True)
    result = await gw.call(
        ToolCallRequest(tool_name="slow_tool", args={"seconds": 0.05}), make_ctx()
    )
    assert result.decision is Decision.EXECUTED_LIVE
    assert result.tool_duration_ms >= 45
    assert result.duration_ms >= result.tool_duration_ms


async def test_defaults_are_filled_in(gateway, make_ctx, call, recorder):
    result = await call(make_ctx(), "kb_search", {"query": "refund"})
    assert result.ok
    completed = recorder.events_for_call(result.call_id)[-1]
    assert completed.data["args"] == {"query": "refund", "limit": 3}


# --------------------------------------------------------------- no caching in baseline


async def test_identical_read_only_calls_all_execute(gateway, make_ctx, call, world):
    ctx = make_ctx()
    for _ in range(3):
        result = await call(ctx, "kb_search", {"query": "support hours"})
        assert result.decision is Decision.EXECUTED_LIVE
    assert world.execution_counts["kb_search"] == 3
    assert gateway.stats["live_executions"]["kb_search"] == 3


async def test_price_reflects_current_time(gateway, make_ctx, call, clock):
    ctx = make_ctx()
    first = await call(ctx, "price_lookup", {"symbol": "KST-X"})
    clock.advance(TICK_SECONDS)
    second = await call(ctx, "price_lookup", {"symbol": "KST-X"})
    assert first.output["price"] != second.output["price"]


async def test_read_after_write_sees_new_value(gateway, make_ctx, call):
    ctx = make_ctx(allow_side_effects=True)
    before = await call(ctx, "record_read", {"record_id": "REC-1001"})
    await call(
        ctx, "record_update", {"record_id": "REC-1001", "field": "owner", "value": "user:ravi"}
    )
    after = await call(ctx, "record_read", {"record_id": "REC-1001"})
    assert before.output["owner"] == "team:orbit"
    assert after.output["owner"] == "user:ravi"


# --------------------------------------------------------------- side effects


async def test_side_effects_always_execute(gateway, make_ctx, call, world):
    ctx = make_ctx(allow_side_effects=True)
    args = {"recipient": "team:billing", "message": "Refund approved"}
    r1 = await call(ctx, "send_notification", args)
    r2 = await call(ctx, "send_notification", args)
    assert r1.decision is r2.decision is Decision.EXECUTED_LIVE
    assert len(world.outbox) == 2
    assert world.execution_counts["send_notification"] == 2


async def test_side_effect_without_permission_is_blocked(gateway, make_ctx, call, world):
    result = await call(
        make_ctx(), "send_notification", {"recipient": "team:billing", "message": "hi"}
    )
    assert result.decision is Decision.BLOCKED_PERMISSION
    assert not result.executed and not result.ok
    assert result.error.code == "permission_denied"
    assert world.outbox == []
    assert world.execution_counts["send_notification"] == 0


async def test_unknown_effect_tool_needs_permission_and_always_executes(
    gateway, make_ctx, call, world
):
    blocked = await call(make_ctx(), "unknown_effect_tool", {"text": "a"})
    assert blocked.decision is Decision.BLOCKED_PERMISSION
    assert world.execution_counts["unknown_effect_tool"] == 0

    ctx = make_ctx(allow_side_effects=True)
    for _ in range(2):
        assert (await call(ctx, "unknown_effect_tool", {"text": "a"})).executed
    assert world.execution_counts["unknown_effect_tool"] == 2


async def test_allowlist_restricts_tools(gateway, make_ctx, call, world):
    ctx = make_ctx(allowed_tools=frozenset({"calculator"}))
    assert (await call(ctx, "calculator", {"expression": "1+1"})).ok
    blocked = await call(ctx, "kb_search", {"query": "orbit"})
    assert blocked.decision is Decision.BLOCKED_PERMISSION
    assert world.execution_counts["kb_search"] == 0


# --------------------------------------------------------------- invalid requests


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("calculator", {}),  # missing field
        ("calculator", {"expression": 5}),  # wrong type
        ("calculator", {"expression": "1+1", "extra": True}),  # unexpected field
        ("calculator", ["1+1"]),  # not an object
        ("calculator", "1+1"),  # not an object
        ("kb_search", {"query": "x", "limit": 99}),  # out of range
        ("doc_read", {"doc_id": "../../etc/passwd"}),  # pattern violation
        ("record_update", {"record_id": "REC-1001", "field": "version", "value": "9"}),
        ("calculator", {"expression": "1" * 9000}),  # too large
    ],
)
async def test_invalid_arguments_are_blocked_before_execution(
    gateway, make_ctx, call, world, tool, args
):
    ctx = make_ctx(allow_side_effects=True)
    result = await call(ctx, tool, args)
    assert result.decision is Decision.BLOCKED_INVALID_ARGS
    assert not result.executed
    assert result.error.code == "invalid_arguments"
    assert sum(world.execution_counts.values()) == 0


async def test_validation_errors_do_not_echo_input_values(gateway, make_ctx, call):
    result = await call(make_ctx(), "kb_search", {"query": "x", "limit": "my-secret-value"})
    assert result.decision is Decision.BLOCKED_INVALID_ARGS
    assert "my-secret-value" not in result.error.message
    assert "limit" in result.error.message


async def test_unknown_tool_is_blocked(gateway, make_ctx, call):
    result = await call(make_ctx(), "delete_everything", {})
    assert result.decision is Decision.BLOCKED_UNKNOWN_TOOL
    assert result.error.code == "unknown_tool"
    assert not result.executed


async def test_from_function_call_handles_missing_args(gateway, make_ctx):
    request = ToolCallRequest.from_function_call("kb_search", None, call_id="fc-1")
    result = await gateway.call(request, make_ctx())
    assert result.call_id == "fc-1"
    assert result.decision is Decision.BLOCKED_INVALID_ARGS  # query is required


# --------------------------------------------------------------- limits


async def test_request_limit_stops_the_run(gateway, make_ctx, call, world):
    ctx = make_ctx(max_tool_requests=3)
    for _ in range(3):
        assert (await call(ctx, "calculator", {"expression": "1+1"})).ok
    over = await call(ctx, "calculator", {"expression": "1+1"})
    assert over.decision is Decision.BLOCKED_LIMIT
    assert ctx.stop_reason == "max_tool_requests"
    after = await call(ctx, "calculator", {"expression": "1+1"})
    assert after.decision is Decision.BLOCKED_RUN_STOPPED
    assert world.execution_counts["calculator"] == 3


async def test_invalid_requests_count_toward_the_limit(gateway, make_ctx, call):
    ctx = make_ctx(max_tool_requests=2)
    await call(ctx, "no_such_tool", {})
    await call(ctx, "calculator", {})
    assert (await call(ctx, "calculator", {"expression": "1"})).decision is Decision.BLOCKED_LIMIT


async def test_limits_are_per_run(gateway, make_ctx, call):
    first = make_ctx(max_tool_requests=1)
    second = make_ctx(max_tool_requests=1)
    assert (await call(first, "calculator", {"expression": "1"})).ok
    assert (await call(second, "calculator", {"expression": "1"})).ok


# --------------------------------------------------------------- modes & concurrency


def test_both_modes_can_be_built(world):
    assert build_gateway(world=world, mode=GatewayMode.OPTIMIZED).mode is GatewayMode.OPTIMIZED
    assert build_gateway(world=world, mode="passthrough").mode is GatewayMode.PASSTHROUGH


def test_unfrozen_registry_rejected():
    from app.tools.registry import ToolRegistry

    with pytest.raises(ValueError):
        ToolGateway(ToolRegistry())


async def test_data_clock_and_timing_clock_are_separate(world, clock, make_ctx):
    """World clock = data time (TTL, prices); gateway clock = real timing."""
    from app.telemetry.clock import SystemClock

    gw = build_gateway(world=world, clock=SystemClock(), mode=GatewayMode.OPTIMIZED)
    ctx = make_ctx(session_id="A")
    await gw.call(ToolCallRequest(tool_name="price_lookup", args={"symbol": "NMB-X"}), ctx)
    clock.advance(61)  # data time moves past the 60 s TTL; real time does not
    later = await gw.call(
        ToolCallRequest(tool_name="price_lookup", args={"symbol": "NMB-X"}),
        make_ctx(session_id="A"),
    )
    assert later.decision.value == "stale_refreshed"
    assert later.duration_ms < 61_000  # durations are real, not simulated


async def test_concurrent_calls_are_all_counted(gateway, make_ctx, world):
    ctx = make_ctx(max_tool_requests=50)
    requests = [
        ToolCallRequest(tool_name="slow_tool", args={"seconds": 0.01 + i / 1000}) for i in range(10)
    ] + [ToolCallRequest(tool_name="calculator", args={"expression": f"{i}+1"}) for i in range(10)]
    results = await asyncio.gather(*(gateway.call(r, ctx) for r in requests))
    assert all(r.ok for r in results)
    assert ctx.tool_requests == 20
    assert world.execution_counts["slow_tool"] == 10
    assert gateway.stats["live_executions"] == {"slow_tool": 10, "calculator": 10}
    assert len({r.call_id for r in results}) == 20


async def test_stats_snapshot_cannot_modify_gateway(gateway, make_ctx, call):
    await call(make_ctx(), "calculator", {"expression": "1"})
    snapshot = gateway.stats
    snapshot["live_executions"]["calculator"] = 999
    assert gateway.stats["live_executions"]["calculator"] == 1
