"""Learned rules: narrow, checked adjustments to how long and how widely a
read-only tool's results may be reused.

A learned rule can only change two things, and only for tools whose results
may be reused at all (read-only, not ``live``):

* ``ttl``: a new TTL for a ``freshness=ttl`` tool, never above the developer's
  ceiling ``max_ttl_seconds``;
* ``cache_scope``: widen ``run`` -> ``session``. Never ``global``.

It can never touch side-effecting, unknown or live tools, permissions, limits,
validation, retries or anything else in the gateway pipeline. These checks run
when a rule is proposed, approved, loaded and - again - every time the gateway
uses it (a rule that fails is ignored, falling back to the developer default).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from app.optimizer.eligibility import may_reuse_result
from app.tools.spec import CacheScope, Freshness, ToolSpec


class RuleKind(StrEnum):
    TTL = "ttl"
    CACHE_SCOPE = "cache_scope"


class OverrideRejected(ValueError):
    """The rule would break a fixed rule; it must not be used."""


@dataclass(frozen=True)
class ToolOverride:
    rule_id: str
    tool_name: str
    tool_version: str
    kind: RuleKind
    ttl_seconds: float | None = None
    cache_scope: CacheScope | None = None


def check_override(spec: ToolSpec | None, override: ToolOverride) -> None:
    """Raise OverrideRejected if ``override`` may not be applied to ``spec``."""
    if spec is None:
        raise OverrideRejected(f"no tool named '{override.tool_name}'")
    if spec.name != override.tool_name:
        raise OverrideRejected("rule is for a different tool")
    if spec.version != override.tool_version:
        raise OverrideRejected(
            f"tool '{spec.name}' changed (version {override.tool_version} -> {spec.version}) "
            "since the rule was learned"
        )
    if spec.test_only:
        raise OverrideRejected("test-only tools cannot have learned rules")
    if not may_reuse_result(spec):
        raise OverrideRejected(
            f"tool '{spec.name}' never reuses results (side effects, unknown effect, or live "
            "data); learned rules cannot change that"
        )
    if override.kind is RuleKind.TTL:
        if override.cache_scope is not None or override.ttl_seconds is None:
            raise OverrideRejected("a TTL rule sets only ttl_seconds")
        if spec.freshness is not Freshness.TTL:
            raise OverrideRejected(f"tool '{spec.name}' has no TTL (freshness={spec.freshness})")
        if spec.max_ttl_seconds is None:
            raise OverrideRejected(f"tool '{spec.name}' has no TTL ceiling, so it cannot learn one")
        if not 0 < override.ttl_seconds <= spec.max_ttl_seconds:
            raise OverrideRejected(
                f"TTL must be between 0 and the tool's ceiling of {spec.max_ttl_seconds:g} s"
            )
    elif override.kind is RuleKind.CACHE_SCOPE:
        if override.ttl_seconds is not None or override.cache_scope is None:
            raise OverrideRejected("a scope rule sets only cache_scope")
        if override.cache_scope is not CacheScope.SESSION:
            raise OverrideRejected("learned scope rules may only widen to 'session'")
        if spec.cache_scope is not CacheScope.RUN:
            raise OverrideRejected(
                f"tool '{spec.name}' is already shared within {spec.cache_scope} scope"
            )
    else:  # pragma: no cover - closed enum
        raise OverrideRejected(f"unknown rule kind {override.kind}")


@dataclass(frozen=True)
class Effective:
    ttl_seconds: float | None  # None = never expires (static tools)
    cache_scope: CacheScope
    ttl_rule: str | None = None
    scope_rule: str | None = None


@dataclass(frozen=True)
class LearnedPolicy:
    """Rules that are ACTIVE (change behavior) and rules in SHADOW (only observed)."""

    active: tuple[ToolOverride, ...] = ()
    shadow: tuple[ToolOverride, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.active and not self.shadow

    def active_ids(self) -> list[str]:
        return [o.rule_id for o in self.active]

    def shadow_ids(self) -> list[str]:
        return [o.rule_id for o in self.shadow]


def effective(
    spec: ToolSpec,
    overrides: tuple[ToolOverride, ...],
    on_rejected: Callable[[ToolOverride, str], None] | None = None,
) -> Effective:
    """The TTL and scope to use for ``spec``, after applying valid overrides.
    Invalid overrides are skipped (and reported through ``on_rejected``)."""
    ttl, scope = spec.ttl_seconds, spec.cache_scope
    ttl_rule = scope_rule = None
    for o in overrides:
        if o.tool_name != spec.name:
            continue
        try:
            check_override(spec, o)
        except OverrideRejected as exc:
            if on_rejected is not None:
                on_rejected(o, str(exc))
            continue
        if o.kind is RuleKind.TTL and ttl_rule is None:
            ttl, ttl_rule = o.ttl_seconds, o.rule_id
        elif o.kind is RuleKind.CACHE_SCOPE and scope_rule is None:
            scope, scope_rule = o.cache_scope or scope, o.rule_id
    return Effective(ttl_seconds=ttl, cache_scope=scope, ttl_rule=ttl_rule, scope_rule=scope_rule)


def make_override(
    *, rule_id: str, tool_name: str, tool_version: str, kind: str, setting: dict
) -> ToolOverride:
    """Build an override from stored values. Unknown values become ``None``,
    which ``check_override`` then rejects (so a damaged row is never used)."""
    rule_kind = RuleKind(kind)
    if rule_kind is RuleKind.TTL:
        ttl = setting.get("ttl_seconds")
        return ToolOverride(
            rule_id,
            tool_name,
            tool_version,
            rule_kind,
            ttl_seconds=float(ttl) if isinstance(ttl, int | float) else None,
        )
    try:
        scope: CacheScope | None = CacheScope(setting.get("cache_scope"))
    except ValueError:
        scope = None
    return ToolOverride(rule_id, tool_name, tool_version, rule_kind, cache_scope=scope)


PolicyProvider = Callable[[], LearnedPolicy]
NO_POLICY = LearnedPolicy()
