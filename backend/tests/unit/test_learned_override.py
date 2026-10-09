"""check_override: the fixed limits no learned rule can cross."""

from __future__ import annotations

import pytest

from app.gateway.factory import build_gateway
from app.optimizer.learned import (
    LearnedPolicy,
    OverrideRejected,
    RuleKind,
    ToolOverride,
    check_override,
    effective,
)
from app.tools.spec import CacheScope

GW = build_gateway(include_test_tools=True)


def spec(name):
    return GW.tool_spec(name)


def ttl(tool, seconds, version=None, rule_id="r1"):
    return ToolOverride(
        rule_id=rule_id,
        tool_name=tool,
        tool_version=version or spec(tool).version,
        kind=RuleKind.TTL,
        ttl_seconds=seconds,
    )


def scope(tool, value=CacheScope.SESSION, version=None, rule_id="r2"):
    return ToolOverride(
        rule_id=rule_id,
        tool_name=tool,
        tool_version=version or spec(tool).version,
        kind=RuleKind.CACHE_SCOPE,
        cache_scope=value,
    )


def test_valid_rules_pass():
    check_override(spec("record_read"), ttl("record_read", 600))
    check_override(spec("record_read"), ttl("record_read", 900))  # exactly the ceiling
    check_override(spec("record_read"), scope("record_read"))


@pytest.mark.parametrize(
    "tool",
    ["record_update", "send_notification", "unknown_effect_tool", "flaky_write_tool"],
)
def test_side_effect_and_unknown_tools_are_never_eligible(tool):
    with pytest.raises(OverrideRejected, match="never reuses|test-only"):
        check_override(spec(tool), ttl(tool, 10, version=spec(tool).version))


def test_live_tools_are_never_eligible():
    with pytest.raises(OverrideRejected):
        check_override(spec("slow_tool"), ttl("slow_tool", 10))


@pytest.mark.parametrize("seconds", [0, -5, 901, 10**9])
def test_ttl_must_stay_within_the_tools_ceiling(seconds):
    with pytest.raises(OverrideRejected, match="ceiling"):
        check_override(spec("record_read"), ttl("record_read", seconds))


def test_static_tools_have_no_ttl_to_learn():
    with pytest.raises(OverrideRejected, match="no TTL"):
        check_override(spec("calculator"), ttl("calculator", 60))


def test_scope_can_only_widen_run_to_session():
    with pytest.raises(OverrideRejected, match="only widen"):
        check_override(spec("record_read"), scope("record_read", CacheScope.GLOBAL))
    with pytest.raises(OverrideRejected, match="only widen"):
        check_override(spec("record_read"), scope("record_read", CacheScope.RUN))
    with pytest.raises(OverrideRejected, match="already shared"):
        check_override(spec("kb_search"), scope("kb_search"))


def test_rule_for_a_changed_tool_version_is_rejected():
    with pytest.raises(OverrideRejected, match="changed"):
        check_override(spec("record_read"), ttl("record_read", 600, version="0"))


def test_missing_tool_and_mixed_up_fields_are_rejected():
    with pytest.raises(OverrideRejected, match="no tool"):
        check_override(None, ttl("record_read", 600))
    bad = ToolOverride(
        rule_id="x",
        tool_name="record_read",
        tool_version=spec("record_read").version,
        kind=RuleKind.TTL,
        ttl_seconds=600,
        cache_scope=CacheScope.SESSION,
    )
    with pytest.raises(OverrideRejected, match="only ttl_seconds"):
        check_override(spec("record_read"), bad)
    with pytest.raises(OverrideRejected, match="different tool"):
        check_override(spec("kb_search"), ttl("record_read", 600))


def test_effective_skips_invalid_rules_and_reports_them():
    rejected = []
    eff = effective(
        spec("record_read"),
        (ttl("record_read", 10_000, rule_id="bad"), ttl("record_read", 600, rule_id="good")),
        on_rejected=lambda o, why: rejected.append(o.rule_id),
    )
    assert eff.ttl_seconds == 600 and eff.ttl_rule == "good"
    assert eff.cache_scope is CacheScope.RUN and eff.scope_rule is None
    assert rejected == ["bad"]


def test_effective_without_rules_is_the_tool_default():
    eff = effective(spec("record_read"), ())
    assert (eff.ttl_seconds, eff.cache_scope) == (300, CacheScope.RUN)
    assert LearnedPolicy().is_empty
