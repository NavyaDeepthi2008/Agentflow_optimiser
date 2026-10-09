"""The fixed rule for which tool results may ever be reused instead of executed.

This is the single source of truth the Stage 3 cache and the Stage 8 learned
rules must consult. It is deliberately strict:

* Only READ_ONLY tools can ever be reused.
* LIVE tools are never reused, even if read-only.
* SIDE_EFFECT and UNKNOWN tools are never reused, skipped, or deduplicated.

Learned rules may only make reuse *less* likely for a tool that passes this
check; they can never make an ineligible tool eligible.
"""

from __future__ import annotations

from app.tools.spec import Freshness, ToolEffect, ToolSpec


def may_reuse_result(spec: ToolSpec) -> bool:
    return spec.effect is ToolEffect.READ_ONLY and spec.freshness is not Freshness.LIVE


def must_always_execute(spec: ToolSpec) -> bool:
    """True for tools whose requests must always run (never cached, skipped or merged)."""
    return not may_reuse_result(spec)
