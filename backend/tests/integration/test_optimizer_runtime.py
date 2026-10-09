"""Retries, timeouts, concurrency, safety precedence and mode parity."""

from __future__ import annotations

import asyncio
import time

import pytest

from app.agent.models import AgentLimits, RunStatus
from app.agent.runner import AgentRunner
from app.gateway.factory import build_gateway
from app.gateway.models import Decision, GatewayMode, ResultSource, ToolCallRequest
from app.llm.scripted import ScriptedLLM
from app.optimizer.retry import RetryPolicy
from app.telemetry.clock import SystemClock
from app.tools.mock_world import MockWorld
from tests.conftest import FAST_RETRIES

S = ScriptedLLM
BOTH_MODES = pytest.mark.parametrize("mode", [GatewayMode.PASSTHROUGH, GatewayMode.OPTIMIZED])


def gw_for(mode, world, recorder=None, **kw):
    kw.setdefault("retry_policy", FAST_RETRIES)
    return build_gateway(world=world, recorder=recorder, include_test_tools=True, mode=mode, **kw)


def req(tool, args=None, **kw):
    return ToolCallRequest(tool_name=tool, args=args or {}, **kw)


# ------------------------------------------------------------------ retries


@BOTH_MODES
async def test_read_only_transient_failure_is_retried(mode, world, make_ctx):
    gw = gw_for(mode, world)
    r = await gw.call(req("flaky_tool", {"text": "a"}), make_ctx())
    assert r.ok and r.executions == 2
    assert gw.stats["retries"] == 1
    assert gw.stats["live_executions"]["flaky_tool"] == world.execution_counts["flaky_tool"] == 2


@BOTH_MODES
async def test_side_effect_without_key_is_never_retried(mode, world, make_ctx, recorder):
    gw = gw_for(mode, world, recorder)
    r = await gw.call(req("flaky_write_tool", {"value": "pay"}), make_ctx(allow_side_effects=True))
    assert r.decision is Decision.FAILED and r.executions == 1
    assert len(world.test_writes) == 1  # it DID write once; a retry would have doubled it
    failed = [
        e for e in recorder.events_for_call(r.call_id) if e.type == "tool_call.attempt_failed"
    ]
    assert failed[0].data["will_retry"] is False
    assert "idempotency key" in failed[0].data["retry_blocked_because"]


@BOTH_MODES
async def test_side_effect_with_idempotency_key_is_retried_safely(mode, world, make_ctx):
    gw = gw_for(mode, world)
    r = await gw.call(
        req("flaky_write_tool", {"value": "pay", "request_key": "order-0001"}),
        make_ctx(allow_side_effects=True),
    )
    assert r.ok and r.executions == 2
    assert r.output["duplicate"] is True  # the retry recognised the earlier write
    assert len(world.test_writes) == 1


async def test_permanent_errors_and_bugs_are_not_retried(gateway, make_ctx, world):
    a = await gateway.call(req("doc_read", {"doc_id": "kb-999"}), make_ctx())
    b = await gateway.call(req("crashing_tool"), make_ctx())
    assert a.executions == 1 and b.executions == 1
    assert gateway.stats["retries"] == 0


async def test_retry_budget(world, make_ctx):
    world.flaky_fail_times = 100
    gw = gw_for("passthrough", world, retry_policy=RetryPolicy(max_retries=1, base_delay_seconds=0))
    r = await gw.call(req("flaky_tool"), make_ctx())
    assert r.decision is Decision.FAILED and r.executions == 2


async def test_no_retry_when_the_run_deadline_is_near(make_ctx):
    world = MockWorld.create(SystemClock())
    world.flaky_fail_times = 100
    gw = gw_for("passthrough", world, retry_policy=RetryPolicy(base_delay_seconds=5.0))
    ctx = make_ctx(deadline=SystemClock().monotonic() + 1.0)
    start = time.monotonic()
    r = await gw.call(req("flaky_tool"), ctx)
    assert r.executions == 1 and time.monotonic() - start < 1.0


# ------------------------------------------------------------------ timeouts


@BOTH_MODES
async def test_sync_tool_timeout(mode, world, make_ctx):
    gw = gw_for(mode, world)
    start = time.monotonic()
    r = await gw.call(req("blocking_tool", {"seconds": 1.0}), make_ctx())
    assert time.monotonic() - start < 0.8  # timed out at ~0.2s, not 1s
    assert r.decision is Decision.FAILED and r.error.code == "timeout"
    assert gw.stats["timeouts"] == 1


async def test_side_effect_timeout_is_reported_as_unknown_and_not_retried(world, make_ctx):
    gw = gw_for("optimized", world)
    r = await gw.call(req("slow_write_tool", {"seconds": 1.0}), make_ctx(allow_side_effects=True))
    assert r.error.code == "timeout_outcome_unknown"
    assert r.executions == 1
    assert len(world.test_writes) == 1  # the write happened even though we timed out


async def test_side_effect_timeout_with_key_retries_safely(world, make_ctx):
    gw = gw_for("optimized", world)
    r = await gw.call(
        req("slow_write_tool", {"seconds": 1.0, "request_key": "write-0001"}),
        make_ctx(allow_side_effects=True),
    )
    assert r.ok and r.executions == 2 and r.output["duplicate"] is True
    assert len(world.test_writes) == 1


async def test_tool_timeout_is_capped_by_run_deadline(make_ctx):
    world = MockWorld.create(SystemClock())
    gw = gw_for("passthrough", world)
    ctx = make_ctx(deadline=SystemClock().monotonic() + 0.15)
    start = time.monotonic()
    r = await gw.call(req("slow_tool", {"seconds": 5}), ctx)  # its own timeout is 10s
    assert time.monotonic() - start < 1.0
    assert r.error.code == "timeout"
    after = await gw.call(req("calculator", {"expression": "1"}), ctx)
    assert after.decision is Decision.BLOCKED_LIMIT and after.error.code == "run_timeout"
    assert ctx.stop_reason == "run_timeout"


# ------------------------------------------------------------------ concurrency


# Sharing only happens when executions really overlap in time, so these tests use
# a tool that takes 50 ms. (A near-instant tool usually finishes before the next
# request is even processed; then later requests get an ordinary cache hit.)
SLOW_READ = {"record_id": "REC-1001", "seconds": 0.05}


async def test_concurrent_identical_requests_share_one_execution(opt_gateway, make_ctx, world):
    ctxs = [make_ctx(session_id="A") for _ in range(5)]  # 5 runs, same session
    results = await asyncio.gather(
        *(opt_gateway.call(req("slow_record_read", SLOW_READ), c) for c in ctxs)
    )
    decisions = sorted(r.decision.value for r in results)
    assert decisions == ["executed_live"] + ["shared_in_flight"] * 4
    assert world.execution_counts["slow_record_read"] == 1
    assert len({str(r.output) for r in results}) == 1
    shared = [r for r in results if r.decision is Decision.SHARED_IN_FLIGHT]
    assert all(r.source is ResultSource.IN_FLIGHT and r.executions == 0 for r in shared)
    # Each caller got its own copy: changing one does not change the others.
    leader = next(r for r in results if r.decision is Decision.EXECUTED_LIVE)
    shared[0].output["owner"] = "tampered"
    assert leader.output["owner"] == "team:orbit"
    assert shared[1].output["owner"] == "team:orbit"


async def test_fast_concurrent_duplicates_still_execute_at_most_once(opt_gateway, make_ctx, world):
    ctxs = [make_ctx(session_id="A") for _ in range(5)]
    results = await asyncio.gather(
        *(opt_gateway.call(req("kb_search", {"query": "support hours"}), c) for c in ctxs)
    )
    assert world.execution_counts["kb_search"] == 1  # shared or cached, never re-run
    assert sum(1 for r in results if r.reused) == 4


async def test_concurrent_identical_requests_in_baseline_all_execute(gateway, make_ctx, world):
    ctxs = [make_ctx(session_id="A") for _ in range(5)]
    await asyncio.gather(*(gateway.call(req("slow_record_read", SLOW_READ), c) for c in ctxs))
    assert world.execution_counts["slow_record_read"] == 5


async def test_failed_leader_is_shared_without_extra_executions(opt_gateway, make_ctx, world):
    ctxs = [make_ctx(session_id="A") for _ in range(4)]
    missing = {"record_id": "REC-9999", "seconds": 0.05}  # fails after 50 ms
    results = await asyncio.gather(
        *(opt_gateway.call(req("slow_record_read", missing), c) for c in ctxs)
    )
    assert all(r.decision is Decision.FAILED for r in results)
    assert world.execution_counts["slow_record_read"] == 1
    assert sum(r.executions for r in results) == 1


async def test_concurrent_side_effects_all_execute(opt_gateway, make_ctx, world):
    ctxs = [make_ctx(session_id="A", allow_side_effects=True) for _ in range(4)]
    args = {"recipient": "team:a", "message": "same"}
    await asyncio.gather(*(opt_gateway.call(req("send_notification", args), c) for c in ctxs))
    assert len(world.outbox) == 4


# ------------------------------------------------------------------ safety comes first


async def test_cached_result_not_served_when_permission_is_missing(ocall, make_ctx, world):
    await ocall(make_ctx(session_id="A"), "kb_search", {"query": "orbit"})
    restricted = make_ctx(session_id="A", allowed_tools=frozenset({"calculator"}))
    r = await ocall(restricted, "kb_search", {"query": "orbit"})
    assert r.decision is Decision.BLOCKED_PERMISSION and r.output is None


async def test_invalid_arguments_never_hit_the_cache(ocall, make_ctx):
    ctx = make_ctx()
    await ocall(ctx, "kb_search", {"query": "orbit"})
    r = await ocall(ctx, "kb_search", {"query": "orbit", "unexpected": 1})
    assert r.decision is Decision.BLOCKED_INVALID_ARGS


@BOTH_MODES
async def test_identical_request_limit_is_the_same_in_both_modes(mode, world, make_ctx):
    gw = gw_for(mode, world)
    ctx = make_ctx()
    results = [await gw.call(req("kb_search", {"query": "loop"}), ctx) for _ in range(5)]
    assert [r.decision for r in results[3:]] == [
        Decision.BLOCKED_LIMIT,
        Decision.BLOCKED_RUN_STOPPED,
    ]
    assert results[3].error.code == "repeated_request"
    assert ctx.stop_reason == "max_identical_requests"


@BOTH_MODES
async def test_request_limit_counts_reused_results(mode, world, make_ctx):
    gw = gw_for(mode, world)
    ctx = make_ctx(max_tool_requests=2)
    await gw.call(req("calculator", {"expression": "1"}), ctx)
    await gw.call(req("calculator", {"expression": "1"}), ctx)
    third = await gw.call(req("calculator", {"expression": "2"}), ctx)
    assert third.decision is Decision.BLOCKED_LIMIT


async def test_cache_hit_blocked_after_deadline(ocall, make_ctx, clock):
    await ocall(make_ctx(), "calculator", {"expression": "3*3"})
    expired = make_ctx(deadline=clock.monotonic() - 1)
    r = await ocall(expired, "calculator", {"expression": "3*3"})
    assert r.decision is Decision.BLOCKED_LIMIT and r.error.code == "run_timeout"


async def test_optimizer_failure_fails_open(opt_gateway, make_ctx, world, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("cache bug")

    monkeypatch.setattr(opt_gateway._cache, "get", broken)
    r = await opt_gateway.call(req("kb_search", {"query": "x"}), make_ctx())
    assert r.decision is Decision.EXECUTED_LIVE and r.ok
    assert r.cache_status == "optimizer_error"
    assert opt_gateway.stats["optimizer_errors"] == 1


async def test_loop_guard_failure_fails_closed(opt_gateway, make_ctx, world, monkeypatch):
    import app.gateway.gateway as gateway_module

    def broken(*a, **k):
        raise RuntimeError("fingerprint bug")

    monkeypatch.setattr(gateway_module, "request_fingerprint", broken)
    r = await opt_gateway.call(req("kb_search", {"query": "x"}), make_ctx())
    assert r.decision is Decision.BLOCKED_SAFETY_ERROR
    assert world.execution_counts["kb_search"] == 0


# ------------------------------------------------------------------ gateway-only accounting


async def test_counts_match_tool_internal_counts_in_optimized_mode(opt_gateway, make_ctx, world):
    ctx = make_ctx(allow_side_effects=True, max_tool_requests=100)
    workload = [
        ("kb_search", {"query": "refund"}),
        ("kb_search", {"query": "Refund"}),  # reused
        ("doc_read", {"doc_id": "kb-004"}),
        ("doc_read", {"doc_id": "kb-004"}),  # reused
        ("record_read", {"record_id": "REC-1001"}),
        ("record_update", {"record_id": "REC-1001", "field": "note", "value": "n"}),
        ("record_read", {"record_id": "REC-1001"}),  # invalidated -> live
        ("send_notification", {"recipient": "team:a", "message": "m"}),
        ("send_notification", {"recipient": "team:a", "message": "m"}),  # never reused
        ("flaky_tool", {"text": "x"}),  # retried once
        ("calculator", {"expression": "nope"}),  # tool error
        ("no_such_tool", {}),
    ]
    results = [await opt_gateway.call(req(t, a), ctx) for t, a in workload]
    assert opt_gateway.stats["live_executions"] == dict(world.execution_counts)
    assert sum(r.executions for r in results) == sum(world.execution_counts.values()) == 10
    assert sum(1 for r in results if r.reused) == 2
    assert opt_gateway.stats["requests_total"] == len(workload)


# ------------------------------------------------------------------ agent in both modes


def redundant_script():
    return S(
        [
            S.tool_calls(("kb_search", {"query": "Orbit price"})),
            S.tool_calls(
                ("kb_search", {"query": "orbit price"}), ("doc_read", {"doc_id": "kb-002"})
            ),
            S.tool_calls(("doc_read", {"doc_id": "kb-002"})),
            S.final("Orbit costs 29 USD per user per month."),
        ]
    )


async def test_same_script_in_both_modes(make_ctx):
    results = {}
    for mode in GatewayMode:
        world = MockWorld.create()
        gw = gw_for(mode, world)
        llm = redundant_script()
        result = await AgentRunner(llm, gw).run("Orbit price?", make_ctx())
        results[mode] = (result, llm, world)
    base, base_llm, base_world = results[GatewayMode.PASSTHROUGH]
    opt, opt_llm, opt_world = results[GatewayMode.OPTIMIZED]

    assert base.status is opt.status is RunStatus.SUCCESS
    assert base.final_answer == opt.final_answer
    assert base_llm.requests[0].tools == opt_llm.requests[0].tools  # identical tool definitions
    assert base.tool_request_count == opt.tool_request_count == 4
    assert base.tool_execution_count == 4
    assert opt.tool_execution_count == 2
    assert opt.summary()["reused_results"] == 2
    assert sum(base_world.execution_counts.values()) == 4
    assert sum(opt_world.execution_counts.values()) == 2


@BOTH_MODES
async def test_agent_loop_detected(mode, make_ctx):
    gw = gw_for(mode, MockWorld.create())
    llm = S([S.tool_calls(("kb_search", {"query": "same"})) for _ in range(6)])
    result = await AgentRunner(llm, gw).run("t", make_ctx())
    assert result.status is RunStatus.LOOP_DETECTED
    assert result.error.code == "max_identical_requests"
    assert result.llm_call_count == 4


async def test_agent_run_timeout_reached_inside_a_tool(make_ctx):
    gw = gw_for("passthrough", MockWorld.create(SystemClock()))
    llm = S([S.tool_calls(("slow_tool", {"seconds": 5})), S.final("never")])
    runner = AgentRunner(llm, gw, clock=SystemClock(), limits=AgentLimits(run_timeout_seconds=0.3))
    start = time.monotonic()
    result = await runner.run("t", make_ctx())
    assert time.monotonic() - start < 2
    assert result.status is RunStatus.LIMIT_TIME
    assert result.tool_results[0].error.code == "timeout"


# ------------------------------------------------------------------ abandoned sync threads
# A synchronous tool that times out keeps running in its thread: Python cannot
# cancel it. These tests show what the gateway does about that.


async def test_late_write_from_a_timed_out_sync_tool_never_leaves_stale_data(world, make_ctx):
    # slow_record_read is async, so it can run while the blocked write still
    # holds the mock database lock (a sync record_read would simply wait).
    gw = gw_for("optimized", world)
    ctx = make_ctx(allow_side_effects=True, max_identical_requests=10)
    rec = {"record_id": "REC-1001", "seconds": 0.01}
    first = await gw.call(req("slow_record_read", rec), ctx)
    assert first.output["status"] == "open"
    assert (await gw.call(req("slow_record_read", rec), ctx)).decision is Decision.REUSED_IN_RUN

    write = await gw.call(
        req("blocking_write_tool", {"record_id": "REC-1001", "status": "closed", "seconds": 0.5}),
        ctx,
    )
    assert write.error.code == "timeout_outcome_unknown"  # outcome honestly unknown
    assert gw.stats["abandoned_threads"] == 1

    # While the write may still land, reads execute live and are NOT stored,
    # so the old "open" can never be served after the write completes.
    during = await gw.call(req("slow_record_read", rec), ctx)
    assert during.decision is Decision.EXECUTED_LIVE and during.output["status"] == "open"
    again = await gw.call(req("slow_record_read", rec), ctx)
    assert again.decision is Decision.EXECUTED_LIVE
    assert gw.stats["not_cached"]["write_outcome_pending"] == 2

    await asyncio.sleep(0.6)  # the abandoned thread finishes the write
    assert world.records["REC-1001"]["status"] == "closed"
    assert gw.stats["late_completions"] == 1
    after = await gw.call(req("slow_record_read", rec), ctx)
    assert after.decision is Decision.EXECUTED_LIVE and after.output["status"] == "closed"
    cached = await gw.call(req("slow_record_read", rec), ctx)
    assert cached.decision is Decision.REUSED_IN_RUN  # normal caching resumes
    assert cached.output["status"] == "closed"


async def test_unrelated_results_are_still_cached_during_a_pending_write(world, make_ctx):
    gw = gw_for("optimized", world)
    ctx = make_ctx(allow_side_effects=True)
    await gw.call(
        req("blocking_write_tool", {"record_id": "REC-1001", "status": "x", "seconds": 0.4}), ctx
    )
    other = {"record_id": "REC-1002", "seconds": 0.01}
    await gw.call(req("slow_record_read", other), ctx)
    hit = await gw.call(req("slow_record_read", other), ctx)
    assert hit.decision is Decision.REUSED_IN_RUN
    await asyncio.sleep(0.5)


async def test_abandoned_read_only_thread_is_counted_and_traced(world, make_ctx, recorder):
    gw = build_gateway(
        world=world,
        recorder=recorder,
        include_test_tools=True,
        retry_policy=FAST_RETRIES,
        mode="optimized",
    )
    r = await gw.call(req("blocking_tool", {"seconds": 0.3}), make_ctx())
    assert r.error.code == "timeout"
    assert gw.stats["abandoned_threads"] >= 1
    await asyncio.sleep(0.5)
    types = [e.type for e in recorder.events]
    assert "tool_call.abandoned" in types and "tool_call.late_completion" in types


async def test_read_that_straddles_the_late_write_is_not_stored(world, make_ctx):
    """A read that started BEFORE the abandoned write landed and finishes AFTER
    it may hold the old value; the late completion makes sure it is not kept."""
    gw = gw_for("optimized", world)
    ctx = make_ctx(allow_side_effects=True, max_identical_requests=10)
    await gw.call(
        req("blocking_write_tool", {"record_id": "REC-1001", "status": "closed", "seconds": 0.4}),
        ctx,
    )
    # Starts at ~0.2 s (reads "open"), the write lands at ~0.4 s, the read
    # replies at ~0.8 s - after the pending window has closed.
    slow = await gw.call(req("slow_record_read", {"record_id": "REC-1001", "seconds": 0.6}), ctx)
    assert slow.output["status"] == "open"
    assert world.records["REC-1001"]["status"] == "closed"
    nxt = await gw.call(req("slow_record_read", {"record_id": "REC-1001", "seconds": 0.6}), ctx)
    assert nxt.decision is Decision.EXECUTED_LIVE and nxt.output["status"] == "closed"
