"""Optimized mode: caching, reuse, TTL, sessions, staleness, invalidation, forced fresh."""

from __future__ import annotations

import pytest

from app.gateway.models import Decision, ResultSource
from app.tools.builtin.prices import TICK_SECONDS, price_at

# ------------------------------------------------------------------ basic reuse


async def test_repeat_in_same_run_is_reused(ocall, make_ctx, world, recorder, opt_gateway):
    ctx = make_ctx()
    first = await ocall(ctx, "kb_search", {"query": "refund policy"})
    second = await ocall(ctx, "kb_search", {"query": "refund policy"})

    assert first.decision is Decision.EXECUTED_LIVE and first.cache_status == "miss"
    assert second.decision is Decision.REUSED_IN_RUN
    assert second.source is ResultSource.IN_RUN
    assert second.executed is False and second.executions == 0
    assert second.output == first.output
    assert world.execution_counts["kb_search"] == 1
    assert "Reused" in second.for_model()["note"]
    assert "note" not in first.for_model()
    done = recorder.events_for_call(second.call_id)[-1].data
    assert done["decision"] == "reused_in_run" and done["executed"] is False
    assert done["cache_status"] == "hit"
    assert opt_gateway.stats["decisions"]["reused_in_run"] == 1


async def test_later_run_same_session_gets_cache_hit(ocall, make_ctx, world):
    await ocall(make_ctx(session_id="A"), "doc_read", {"doc_id": "kb-002"})
    hit = await ocall(make_ctx(session_id="A"), "doc_read", {"doc_id": "kb-002"})
    assert hit.decision is Decision.CACHE_HIT and hit.source is ResultSource.CACHE
    assert world.execution_counts["doc_read"] == 1


async def test_sessions_never_share_session_scoped_results(ocall, make_ctx, world):
    a = await ocall(make_ctx(session_id="A"), "kb_search", {"query": "orbit"})
    b = await ocall(make_ctx(session_id="B"), "kb_search", {"query": "orbit"})
    assert a.decision is b.decision is Decision.EXECUTED_LIVE
    assert world.execution_counts["kb_search"] == 2


async def test_run_scoped_results_are_not_shared_between_runs(ocall, make_ctx, world):
    ctx = make_ctx(session_id="A")
    await ocall(ctx, "record_read", {"record_id": "REC-1001"})
    same_run = await ocall(ctx, "record_read", {"record_id": "REC-1001"})
    next_run = await ocall(make_ctx(session_id="A"), "record_read", {"record_id": "REC-1001"})
    assert same_run.decision is Decision.REUSED_IN_RUN
    assert next_run.decision is Decision.EXECUTED_LIVE
    assert world.execution_counts["record_read"] == 2


async def test_global_scope_only_for_public_static_tool(ocall, make_ctx, world):
    await ocall(make_ctx(session_id="A"), "calculator", {"expression": "29*12"})
    hit = await ocall(make_ctx(session_id="B"), "calculator", {"expression": "29 * 12"})
    # Different spacing is not normalized away inside the expression ("29*12" vs
    # "29 * 12" differ), so this is a NEW request:
    assert hit.decision is Decision.EXECUTED_LIVE
    again = await ocall(make_ctx(session_id="C"), "calculator", {"expression": "  29*12 "})
    assert again.decision is Decision.CACHE_HIT  # surrounding spaces ARE normalized
    assert world.execution_counts["calculator"] == 2


async def test_normalized_equivalents_are_reused(ocall, make_ctx, world):
    ctx = make_ctx()
    await ocall(ctx, "kb_search", {"query": "Refund  Policy"})
    same = await ocall(ctx, "kb_search", {"query": " refund policy", "limit": 3})  # default
    different = await ocall(ctx, "kb_search", {"query": "refund policy", "limit": 4})
    assert same.decision is Decision.REUSED_IN_RUN
    assert different.decision is Decision.EXECUTED_LIVE
    assert world.execution_counts["kb_search"] == 2


async def test_baseline_never_reuses(call, make_ctx, world, gateway):
    ctx = make_ctx()
    for _ in range(3):
        r = await call(ctx, "kb_search", {"query": "refund"})
        assert r.decision is Decision.EXECUTED_LIVE and r.cache_status == "disabled"
    assert world.execution_counts["kb_search"] == 3
    assert gateway.stats["cache_entries"] == 0


# ------------------------------------------------------------------ TTL & staleness


async def test_ttl_expiry_refreshes(ocall, make_ctx, clock, world):
    ctx = make_ctx(session_id="A")
    first = await ocall(ctx, "price_lookup", {"symbol": "NMB-X"})
    clock.advance(59)
    hit = await ocall(make_ctx(session_id="A"), "price_lookup", {"symbol": "NMB-X"})
    assert hit.decision is Decision.CACHE_HIT and hit.cache_age_seconds == pytest.approx(59)
    clock.advance(1)  # now exactly 60s old = stale
    refreshed = await ocall(make_ctx(session_id="A"), "price_lookup", {"symbol": "NMB-X"})
    assert refreshed.decision is Decision.STALE_REFRESHED and refreshed.executions == 1
    assert refreshed.cache_status == "stale"
    assert world.execution_counts["price_lookup"] == 2
    assert first.output["price"] == price_at("NMB-X", 0)


async def test_stale_price_is_never_served(ocall, make_ctx, clock):
    await ocall(make_ctx(session_id="A"), "price_lookup", {"symbol": "ORB-X"})
    clock.advance(TICK_SECONDS)  # the true price has changed
    later = await ocall(make_ctx(session_id="A"), "price_lookup", {"symbol": "ORB-X"})
    assert later.decision is Decision.STALE_REFRESHED
    assert later.output["price"] == price_at("ORB-X", 1)


async def test_static_results_do_not_expire(ocall, make_ctx, clock):
    await ocall(make_ctx(), "calculator", {"expression": "2**10"})
    clock.advance(10**7)
    assert (await ocall(make_ctx(), "calculator", {"expression": "2**10"})).decision is (
        Decision.CACHE_HIT
    )


async def test_failed_refresh_does_not_fall_back_to_stale(ocall, make_ctx, clock, world):
    ctx = make_ctx(session_id="A")  # record_read is shared within one run (Stage 7)
    await ocall(ctx, "record_read", {"record_id": "REC-1003"})
    del world.records["REC-1003"]
    clock.advance(301)
    result = await ocall(ctx, "record_read", {"record_id": "REC-1003"})
    assert result.decision is Decision.FAILED and result.error.code == "not_found"
    assert result.output is None
    assert result.cache_status == "stale"


async def test_errors_are_never_cached(ocall, make_ctx, world):
    ctx = make_ctx()
    await ocall(ctx, "doc_read", {"doc_id": "kb-999"})
    again = await ocall(ctx, "doc_read", {"doc_id": "kb-999"})
    assert again.decision is Decision.FAILED and again.executions == 1
    assert world.execution_counts["doc_read"] == 2


# ------------------------------------------------------------------ invalidation


async def test_write_invalidates_the_affected_read(ocall, make_ctx, world):
    ctx = make_ctx(allow_side_effects=True)
    before = await ocall(ctx, "record_read", {"record_id": "REC-1001"})
    other = await ocall(ctx, "record_read", {"record_id": "REC-1002"})
    update = await ocall(
        ctx, "record_update", {"record_id": "REC-1001", "field": "status", "value": "resolved"}
    )
    assert update.invalidated_entries == 1  # only REC-1001's read
    after = await ocall(ctx, "record_read", {"record_id": "REC-1001"})
    untouched = await ocall(ctx, "record_read", {"record_id": "REC-1002"})
    assert after.decision is Decision.EXECUTED_LIVE
    assert before.output["status"] == "open" and after.output["status"] == "resolved"
    assert untouched.decision is Decision.REUSED_IN_RUN and untouched.output == other.output


async def test_failed_write_still_invalidates(ocall, make_ctx):
    ctx = make_ctx(allow_side_effects=True)
    await ocall(ctx, "record_read", {"record_id": "REC-1001"})
    bad = await ocall(
        ctx, "record_update", {"record_id": "REC-1001", "field": "status", "value": "maybe"}
    )
    assert bad.decision is Decision.FAILED and bad.invalidated_entries == 1


async def test_unknown_tool_clears_everything(ocall, make_ctx, opt_gateway):
    ctx = make_ctx(allow_side_effects=True)
    await ocall(ctx, "kb_search", {"query": "orbit"})
    await ocall(ctx, "calculator", {"expression": "1+1"})
    assert opt_gateway.stats["cache_entries"] == 2
    r = await ocall(ctx, "unknown_effect_tool", {"text": "x"})
    assert r.invalidated_entries == 2
    assert opt_gateway.stats["cache_entries"] == 0


async def test_side_effect_declaring_no_changes_keeps_cache(ocall, make_ctx, opt_gateway):
    ctx = make_ctx(allow_side_effects=True)
    await ocall(ctx, "kb_search", {"query": "orbit"})
    r = await ocall(ctx, "send_notification", {"recipient": "team:a", "message": "hi"})
    assert r.invalidated_entries == 0
    assert opt_gateway.stats["cache_entries"] == 1


async def test_read_overlapping_a_write_is_not_cached(opt_gateway, make_ctx, world):
    import asyncio

    from app.gateway.models import ToolCallRequest

    ctx = make_ctx(allow_side_effects=True)
    read = asyncio.create_task(
        opt_gateway.call(
            ToolCallRequest(
                tool_name="slow_record_read", args={"record_id": "REC-1001", "seconds": 0.05}
            ),
            ctx,
        )
    )
    await asyncio.sleep(0.01)  # the read has taken its snapshot and is waiting
    await opt_gateway.call(
        ToolCallRequest(
            tool_name="record_update",
            args={"record_id": "REC-1001", "field": "owner", "value": "user:x"},
        ),
        ctx,
    )
    old = await read
    assert old.output["owner"] == "team:orbit"  # value from before the write
    assert opt_gateway.stats["not_cached"] == {"write_during_fetch": 1}
    fresh = await opt_gateway.call(
        ToolCallRequest(
            tool_name="slow_record_read", args={"record_id": "REC-1001", "seconds": 0.0}
        ),
        ctx,
    )
    assert fresh.decision is Decision.EXECUTED_LIVE and fresh.output["owner"] == "user:x"


async def test_clear_cache_for_cold_starts(ocall, make_ctx, opt_gateway):
    await ocall(make_ctx(session_id="A"), "kb_search", {"query": "a"})
    await ocall(make_ctx(session_id="B"), "kb_search", {"query": "a"})
    assert opt_gateway.clear_cache(session_id="A") == 1
    assert opt_gateway.clear_cache() == 1
    assert opt_gateway.stats["cache_entries"] == 0


# ------------------------------------------------------------------ forced fresh


async def test_force_fresh_request_flag(ocall, make_ctx, world):
    ctx = make_ctx()
    await ocall(ctx, "price_lookup", {"symbol": "KST-X"})
    forced = await ocall(ctx, "price_lookup", {"symbol": "KST-X"}, force_fresh=True)
    assert forced.decision is Decision.FORCED_FRESH and forced.forced_fresh
    assert forced.executions == 1
    # Honoured once per identical request per run: a second "force" is reused.
    again = await ocall(ctx, "price_lookup", {"symbol": "KST-X"}, force_fresh=True)
    assert again.decision is Decision.REUSED_IN_RUN
    assert world.execution_counts["price_lookup"] == 2


async def test_force_fresh_argument_from_the_model(ocall, make_ctx, world, recorder):
    ctx = make_ctx()
    await ocall(ctx, "doc_read", {"doc_id": "kb-001"})
    forced = await ocall(ctx, "doc_read", {"doc_id": "kb-001", "force_fresh": True})
    assert forced.decision is Decision.FORCED_FRESH
    stored = recorder.events_for_call(forced.call_id)[-1].data["args"]
    assert stored == {"doc_id": "kb-001"}  # reserved argument stripped before validation
    bad = await ocall(ctx, "doc_read", {"doc_id": "kb-001", "force_fresh": "yes"})
    assert bad.decision is Decision.BLOCKED_INVALID_ARGS


async def test_task_level_fresh_tools(ocall, make_ctx, world):
    await ocall(make_ctx(session_id="A"), "price_lookup", {"symbol": "NMB-X"})
    verify = make_ctx(session_id="A", fresh_tools=frozenset({"price_lookup"}))
    r = await ocall(verify, "price_lookup", {"symbol": "NMB-X"})
    assert r.decision is Decision.FORCED_FRESH
    assert world.execution_counts["price_lookup"] == 2


async def test_force_fresh_in_baseline_is_just_live(gateway, make_ctx):
    from app.gateway.models import ToolCallRequest

    r = await gateway.call(
        ToolCallRequest(tool_name="kb_search", args={"query": "x"}, force_fresh=True), make_ctx()
    )
    assert r.decision is Decision.EXECUTED_LIVE


# ------------------------------------------------------------------ never reused


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("send_notification", {"recipient": "team:billing", "message": "same text"}),
        ("record_update", {"record_id": "REC-1002", "field": "note", "value": "same"}),
        ("unknown_effect_tool", {"text": "same"}),
        ("flaky_tool", {"text": "live tool"}),  # read-only but freshness=live
    ],
)
async def test_ineligible_tools_always_execute(ocall, make_ctx, world, tool, args):
    world.flaky_fail_times = 0
    ctx = make_ctx(allow_side_effects=True)
    results = [await ocall(ctx, tool, args) for _ in range(3)]
    assert all(r.decision is Decision.EXECUTED_LIVE for r in results)
    assert all(r.cache_status == "not_eligible" and r.executions == 1 for r in results)
    assert world.execution_counts[tool] == 3


async def test_side_effects_are_not_skipped_in_optimized_mode(ocall, make_ctx, world):
    ctx = make_ctx(allow_side_effects=True)
    for _ in range(3):
        await ocall(ctx, "send_notification", {"recipient": "team:a", "message": "same"})
    assert len(world.outbox) == 3


async def test_oversized_results_are_not_cached(world, recorder, make_ctx):
    from app.gateway.factory import build_gateway
    from app.gateway.gateway import OptimizerConfig
    from app.gateway.models import ToolCallRequest

    gw = build_gateway(
        world=world, mode="optimized", optimizer_config=OptimizerConfig(max_cached_output_bytes=50)
    )
    ctx = make_ctx()
    for _ in range(2):
        r = await gw.call(ToolCallRequest(tool_name="doc_read", args={"doc_id": "kb-001"}), ctx)
        assert r.decision is Decision.EXECUTED_LIVE
    assert gw.stats["not_cached"] == {"too_large": 2}


async def test_reuse_note_can_be_disabled(world, make_ctx):
    from app.gateway.factory import build_gateway
    from app.gateway.gateway import OptimizerConfig
    from app.gateway.models import ToolCallRequest

    gw = build_gateway(
        world=world,
        mode="optimized",
        optimizer_config=OptimizerConfig(annotate_reused_results=False),
    )
    ctx = make_ctx()
    req = ToolCallRequest(tool_name="kb_search", args={"query": "x"})
    await gw.call(req, ctx)
    reused = await gw.call(req.model_copy(update={"call_id": "second"}), ctx)
    assert reused.decision is Decision.REUSED_IN_RUN and "note" not in reused.for_model()
