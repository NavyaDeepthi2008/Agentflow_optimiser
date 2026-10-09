"""The fixed reuse rule: only read-only, non-live tools may ever be reused."""

from __future__ import annotations

import itertools

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from app.optimizer.eligibility import may_reuse_result, must_always_execute
from app.tools.catalog import build_registry
from app.tools.mock_world import MockWorld
from app.tools.spec import Freshness, ToolEffect, ToolSpec


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _try_spec(effect: ToolEffect, freshness: Freshness) -> ToolSpec | None:
    kwargs = {"ttl_seconds": 30} if freshness is Freshness.TTL else {}
    try:
        return ToolSpec(
            name="combo_tool",
            description="Combination test tool.",
            args_model=Args,
            effect=effect,
            freshness=freshness,
            **kwargs,
        )
    except ValidationError:
        return None  # combination is forbidden outright - even safer


@pytest.mark.parametrize(("effect", "freshness"), list(itertools.product(ToolEffect, Freshness)))
def test_every_combination(effect, freshness):
    spec = _try_spec(effect, freshness)
    expected_reusable = effect is ToolEffect.READ_ONLY and freshness is not Freshness.LIVE
    if spec is None:
        assert not expected_reusable, "a reusable combination should be constructible"
        return
    assert may_reuse_result(spec) is expected_reusable
    assert must_always_execute(spec) is (not expected_reusable)


def test_builtin_side_effect_tools_are_never_reusable(clock):
    registry = build_registry(MockWorld.create(clock), include_test_tools=True)
    for spec in registry.specs():
        if spec.effect is not ToolEffect.READ_ONLY:
            assert not may_reuse_result(spec), spec.name
            assert must_always_execute(spec), spec.name


def test_builtin_classification_table(clock):
    """Pin down the classification of every demo tool so changes are deliberate."""
    registry = build_registry(MockWorld.create(clock))
    table = {s.name: (s.effect, s.freshness) for s in registry.specs()}
    assert table == {
        "calculator": (ToolEffect.READ_ONLY, Freshness.STATIC),
        "kb_search": (ToolEffect.READ_ONLY, Freshness.TTL),
        "doc_read": (ToolEffect.READ_ONLY, Freshness.TTL),
        "price_lookup": (ToolEffect.READ_ONLY, Freshness.TTL),
        "record_read": (ToolEffect.READ_ONLY, Freshness.TTL),
        "record_update": (ToolEffect.SIDE_EFFECT, Freshness.LIVE),
        "send_notification": (ToolEffect.SIDE_EFFECT, Freshness.LIVE),
    }
