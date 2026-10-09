"""Learned rules inside the gateway: active rules, shadow rules, and the fixed
safety rules they can never override."""

from __future__ import annotations

from app.gateway.factory import build_gateway
from app.gateway.models import Decision, GatewayMode, ToolCallRequest
from app.optimizer.learned import LearnedPolicy, RuleKind, ToolOverride
from app.tools.spec import CacheScope
from tests.conftest import FAST_RETRIES


def rule(tool, kind, value, rule_id, gw):
    spec = gw.tool_spec(tool)
    if kind is RuleKind.TTL:
        return ToolOverride(rule_id, tool, spec.version, kind, ttl_seconds=value)
    return ToolOverride(rule_id, tool, spec.version, kind, cache_scope=value)


def make(world, recorder, policy=None, mode=GatewayMode.OPTIMIZED):
    holder = {"policy": policy or LearnedPolicy()}
    gw = build_gateway(
        world=world,
        recorder=recorder,
        include_test_tools=True,
        retry_policy=FAST_RETRIES,
        mode=mode,
        policy=lambda: holder["policy"],  # tests change the rules between calls
    )
    return gw, holder


async def call(gw, ctx, tool, args, **kw):
    return await gw.call(ToolCallRequest(tool_name=tool, args=args, **kw), ctx)


REC = {"record_id": "REC-1003"}


# ----------------------------------------------------------------- active rules
async def test_active_ttl_rule_extends_reuse_within_the_ceiling(world, recorder, make_ctx, clock):
    gw, holder = make(world, recorder)
    holder["policy"] = LearnedPolicy(active=(rule("record_read", RuleKind.TTL, 600, "ttl1", gw),))
    ctx = make_ctx()
    first = await call(gw, ctx, "record_read", REC)
    clock.advance(450)  # older than the default 300 s, younger than the rule's 600 s
    second = await call(gw, ctx, "record_read", REC)
    assert first.decision is Decision.EXECUTED_LIVE and first.learned_rules == []
    assert second.decision is Decision.REUSED_IN_RUN
    assert second.learned_rules == ["ttl1"]
    assert world.execution_counts["record_read"] == 1
    clock.advance(200)  # 650 s: past the learned TTL too
    third = await call(gw, ctx, "record_read", REC)
    assert third.decision is Decision.STALE_REFRESHED
    assert gw.stats["learned_rules_applied"] == {"ttl1": 1}


async def test_rule_not_credited_when_default_would_have_reused(world, recorder, make_ctx, clock):
    gw, holder = make(world, recorder)
    holder["policy"] = LearnedPolicy(active=(rule("record_read", RuleKind.TTL, 600, "ttl1", gw),))
    ctx = make_ctx()
    await call(gw, ctx, "record_read", REC)
    clock.advance(10)
    again = await call(gw, ctx, "record_read", REC)
    assert again.decision is Decision.REUSED_IN_RUN and again.learned_rules == []


async def test_active_scope_rule_shares_within_session_only(world, recorder, make_ctx):
    gw, holder = make(world, recorder)
    holder["policy"] = LearnedPolicy(
        active=(rule("record_read", RuleKind.CACHE_SCOPE, CacheScope.SESSION, "sc1", gw),)
    )
    await call(gw, make_ctx(session_id="A"), "record_read", REC)
    hit = await call(gw, make_ctx(session_id="A"), "record_read", REC)
    other = await call(gw, make_ctx(session_id="B"), "record_read", REC)
    assert hit.decision is Decision.CACHE_HIT and hit.learned_rules == ["sc1"]
    assert other.decision is Decision.EXECUTED_LIVE  # never across sessions
    assert world.execution_counts["record_read"] == 2


async def test_disabling_a_rule_takes_effect_on_the_next_request(world, recorder, make_ctx, clock):
    gw, holder = make(world, recorder)
    holder["policy"] = LearnedPolicy(active=(rule("record_read", RuleKind.TTL, 600, "ttl1", gw),))
    ctx = make_ctx()
    await call(gw, ctx, "record_read", REC)
    clock.advance(450)
    holder["policy"] = LearnedPolicy()  # rule disabled
    again = await call(gw, ctx, "record_read", REC)
    # The stored entry was made under the 600 s rule, but freshness is judged
    # with the TTL in force now (300 s).
    assert again.decision is Decision.STALE_REFRESHED


# ----------------------------------------------------- fixed rules still win
async def test_write_still_invalidates_results_kept_by_a_learned_rule(
    world, recorder, make_ctx, clock
):
    gw, holder = make(world, recorder)
    holder["policy"] = LearnedPolicy(
        active=(
            rule("record_read", RuleKind.TTL, 900, "ttl1", gw),
            rule("record_read", RuleKind.CACHE_SCOPE, CacheScope.SESSION, "sc1", gw),
        )
    )
    await call(gw, make_ctx(session_id="A"), "record_read", REC)
    await call(
        gw,
        make_ctx(session_id="A", allow_side_effects=True),
        "record_update",
        {"record_id": "REC-1003", "field": "status", "value": "closed"},
    )
    after = await call(gw, make_ctx(session_id="A"), "record_read", REC)
    assert after.decision is Decision.EXECUTED_LIVE
    assert after.output["status"] == "closed"


async def test_write_through_baseline_sibling_invalidates_optimized_cache(
    world, recorder, make_ctx
):
    """Found while building Stage 7: sibling gateways had separate caches, so a
    write made by a BASELINE run left a stale read in the optimized cache of the
    same session. Siblings now share the cache store."""
    gw, holder = make(world, recorder)
    holder["policy"] = LearnedPolicy(
        active=(rule("record_read", RuleKind.CACHE_SCOPE, CacheScope.SESSION, "sc1", gw),)
    )
    baseline = gw.with_mode(GatewayMode.PASSTHROUGH)
    await call(gw, make_ctx(session_id="A"), "record_read", REC)
    await call(
        baseline,
        make_ctx(session_id="A", allow_side_effects=True),
        "record_update",
        {"record_id": "REC-1003", "field": "status", "value": "closed"},
    )
    after = await call(gw, make_ctx(session_id="A"), "record_read", REC)
    assert after.decision is Decision.EXECUTED_LIVE and after.output["status"] == "closed"


async def test_force_fresh_and_fresh_tools_still_bypass_learned_rules(
    world, recorder, make_ctx, clock
):
    gw, holder = make(world, recorder)
    holder["policy"] = LearnedPolicy(active=(rule("record_read", RuleKind.TTL, 900, "t", gw),))
    ctx = make_ctx()
    await call(gw, ctx, "record_read", REC)
    forced = await call(gw, ctx, "record_read", REC, force_fresh=True)
    assert forced.decision is Decision.FORCED_FRESH
    ctx2 = make_ctx(fresh_tools=frozenset({"record_read"}))
    await call(gw, ctx2, "record_read", REC)
    assert world.execution_counts["record_read"] == 3


async def test_loop_guard_and_permissions_still_apply(world, recorder, make_ctx):
    gw, holder = make(world, recorder)
    holder["policy"] = LearnedPolicy(active=(rule("record_read", RuleKind.TTL, 900, "t", gw),))
    ctx = make_ctx(max_identical_requests=2)
    results = [await call(gw, ctx, "record_read", REC) for _ in range(3)]
    assert results[-1].decision is Decision.BLOCKED_LIMIT
    denied = await call(gw, make_ctx(allowed_tools=frozenset({"kb_search"})), "record_read", REC)
    assert denied.decision is Decision.BLOCKED_PERMISSION


async def test_tampered_rules_for_unsafe_tools_are_ignored(world, recorder, make_ctx):
    """Even if a bad rule reached the gateway (e.g. edited in the database), the
    gateway re-checks it and ignores it: side effects still always execute."""
    gw, holder = make(world, recorder)
    spec = gw.tool_spec("send_notification")
    bad = (
        ToolOverride("evil1", "send_notification", spec.version, RuleKind.TTL, ttl_seconds=60),
        ToolOverride(
            "evil2",
            "send_notification",
            spec.version,
            RuleKind.CACHE_SCOPE,
            cache_scope=CacheScope.SESSION,
        ),
        ToolOverride("evil3", "record_read", "0", RuleKind.TTL, ttl_seconds=600),  # stale version
        ToolOverride(
            "evil4", "record_read", gw.tool_spec("record_read").version, RuleKind.TTL,
            ttl_seconds=10**6,  # above the ceiling
        ),
    )
    holder["policy"] = LearnedPolicy(active=bad, shadow=bad)
    ctx = make_ctx(allow_side_effects=True)
    args = {"recipient": "team:orbit", "message": "hi"}
    a = await call(gw, ctx, "send_notification", args)
    b = await call(gw, ctx, "send_notification", args)
    assert a.decision is b.decision is Decision.EXECUTED_LIVE
    assert len(world.outbox) == 2
    await call(gw, ctx, "record_read", REC)
    assert gw.stats["learned_rules_ignored"].get("evil4", 0) >= 1
    assert gw.stats["learned_rules_ignored"].get("evil3", 0) >= 1
    assert gw.stats["learned_rules_applied"] == {}
    ignored = [e for e in recorder.events if e.type == "learning.rule_ignored"]
    assert {e.data["rule_id"] for e in ignored} == {"evil3", "evil4"}  # unsafe tools: never reach


async def test_broken_policy_provider_fails_open_to_defaults(world, recorder, make_ctx, clock):
    gw, holder = make(world, recorder)

    def boom():
        raise RuntimeError("db down")

    gw._policy = boom  # noqa: SLF001
    ctx = make_ctx()
    await call(gw, ctx, "record_read", REC)
    again = await call(gw, ctx, "record_read", REC)
    assert again.decision is Decision.REUSED_IN_RUN  # default behavior continues
    assert gw.stats["optimizer_errors"] >= 1


async def test_baseline_mode_ignores_learned_rules(world, recorder, make_ctx):
    gw, holder = make(world, recorder, mode=GatewayMode.PASSTHROUGH)
    holder["policy"] = LearnedPolicy(active=(rule("record_read", RuleKind.TTL, 900, "t", gw),))
    ctx = make_ctx()
    await call(gw, ctx, "record_read", REC)
    await call(gw, ctx, "record_read", REC)
    assert world.execution_counts["record_read"] == 2


# ----------------------------------------------------------------- shadow
async def test_shadow_rule_never_changes_execution(world, recorder, make_ctx, clock):
    gw, holder = make(world, recorder)
    holder["policy"] = LearnedPolicy(shadow=(rule("record_read", RuleKind.TTL, 600, "sh1", gw),))
    ctx = make_ctx()
    first = await call(gw, ctx, "record_read", REC)
    clock.advance(450)
    second = await call(gw, ctx, "record_read", REC)
    assert second.decision is Decision.STALE_REFRESHED  # executed, as without the rule
    assert world.execution_counts["record_read"] == 2
    assert first.shadow is None
    assert second.shadow is not None
    assert second.shadow.rule_ids == ["sh1"] and second.shadow.matched is True
    assert second.learned_rules == []
    events = [e for e in recorder.events if e.type == "learning.shadow"]
    assert len(events) == 1 and events[0].data["matched"] is True
    assert gw.stats["shadow"] == {"would_reuse": 1, "matched": 1}


async def test_shadow_detects_stale_data(world, recorder, make_ctx, clock):
    gw, holder = make(world, recorder)
    holder["policy"] = LearnedPolicy(shadow=(rule("record_read", RuleKind.TTL, 600, "sh1", gw),))
    ctx = make_ctx()
    await call(gw, ctx, "record_read", REC)
    world.records["REC-1003"]["status"] = "changed outside the agent"
    clock.advance(450)
    second = await call(gw, ctx, "record_read", REC)
    assert second.output["status"] == "changed outside the agent"  # real answer is live
    assert second.shadow.matched is False
    assert gw.stats["shadow"]["mismatched"] == 1


async def test_shadow_scope_rule_observes_across_runs(world, recorder, make_ctx):
    gw, holder = make(world, recorder)
    holder["policy"] = LearnedPolicy(
        shadow=(rule("record_read", RuleKind.CACHE_SCOPE, CacheScope.SESSION, "sh2", gw),)
    )
    await call(gw, make_ctx(session_id="A"), "record_read", REC)
    nxt = await call(gw, make_ctx(session_id="A"), "record_read", REC)
    assert nxt.decision is Decision.EXECUTED_LIVE
    assert nxt.shadow.rule_ids == ["sh2"] and nxt.shadow.matched
    other = await call(gw, make_ctx(session_id="B"), "record_read", REC)
    assert other.shadow is None  # shadow rules never look across sessions either


async def test_shadow_respects_invalidation(world, recorder, make_ctx, clock):
    gw, holder = make(world, recorder)
    holder["policy"] = LearnedPolicy(shadow=(rule("record_read", RuleKind.TTL, 900, "sh", gw),))
    ctx = make_ctx(allow_side_effects=True)
    await call(gw, ctx, "record_read", REC)
    await call(
        gw, ctx, "record_update", {"record_id": "REC-1003", "field": "note", "value": "x"}
    )
    clock.advance(400)
    after = await call(gw, ctx, "record_read", REC)
    assert after.shadow is None  # the write removed the shadow copy too


async def test_results_carry_evidence_fields(world, recorder, make_ctx, clock):
    gw, _ = make(world, recorder)
    ctx = make_ctx()
    a = await call(gw, ctx, "record_read", REC)
    clock.advance(5)
    b = await call(gw, ctx, "record_read", REC)
    assert a.fingerprint and a.fingerprint == b.fingerprint
    assert a.output_digest and a.output_digest == b.output_digest
    assert b.data_time - a.data_time == 5
    failed = await call(gw, ctx, "record_read", {"record_id": "REC-9999"})
    assert failed.output_digest is None
