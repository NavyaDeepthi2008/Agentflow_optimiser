"""ToolSpec validation: unsafe metadata is rejected when the spec is created."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from app.tools.spec import CacheScope, Freshness, ToolEffect, ToolSpec


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: str


class LooseArgs(BaseModel):  # does not forbid extra fields
    item_id: str


def make(**overrides):
    base = {"name": "sample_tool", "description": "A sample tool for tests.", "args_model": Args}
    base.update(overrides)
    return ToolSpec(**base)


def test_defaults_are_the_safe_choice():
    spec = make()
    assert spec.effect is ToolEffect.UNKNOWN
    assert spec.freshness is Freshness.LIVE
    assert spec.cache_scope is CacheScope.RUN
    assert spec.requires_explicit_permission


def test_valid_read_only_ttl_tool():
    spec = make(
        effect=ToolEffect.READ_ONLY,
        freshness=Freshness.TTL,
        ttl_seconds=60,
        max_ttl_seconds=300,
        resource_tags=("item:{item_id}",),
    )
    assert spec.is_read_only
    assert not spec.requires_explicit_permission
    assert spec.render_tags(spec.resource_tags, {"item_id": "A1"}) == ["item:A1"]


@pytest.mark.parametrize(
    "overrides",
    [
        # side-effecting / unknown tools can never be reusable
        {"effect": ToolEffect.SIDE_EFFECT, "freshness": Freshness.TTL, "ttl_seconds": 60},
        {"effect": ToolEffect.SIDE_EFFECT, "freshness": Freshness.STATIC},
        {"effect": ToolEffect.UNKNOWN, "freshness": Freshness.STATIC},
        {"effect": ToolEffect.UNKNOWN, "freshness": Freshness.TTL, "ttl_seconds": 5},
        {"effect": ToolEffect.SIDE_EFFECT, "cache_scope": CacheScope.SESSION},
        {"effect": ToolEffect.SIDE_EFFECT, "retryable": True},  # no idempotency support
        {"effect": ToolEffect.SIDE_EFFECT, "resource_tags": ("item:{item_id}",)},
        # TTL consistency
        {"effect": ToolEffect.READ_ONLY, "freshness": Freshness.TTL},
        {"effect": ToolEffect.READ_ONLY, "freshness": Freshness.TTL, "ttl_seconds": 0},
        {"effect": ToolEffect.READ_ONLY, "freshness": Freshness.LIVE, "ttl_seconds": 10},
        {
            "effect": ToolEffect.READ_ONLY,
            "freshness": Freshness.TTL,
            "ttl_seconds": 60,
            "max_ttl_seconds": 30,
        },
        {"effect": ToolEffect.READ_ONLY, "freshness": Freshness.LIVE, "max_ttl_seconds": 30},
        # scope / invalidation rules
        {
            "effect": ToolEffect.READ_ONLY,
            "freshness": Freshness.STATIC,
            "cache_scope": CacheScope.GLOBAL,
        },  # global requires public data
        {
            "effect": ToolEffect.READ_ONLY,
            "freshness": Freshness.STATIC,
            "invalidates": ("item:{item_id}",),
        },  # only side effects invalidate
        {
            "effect": ToolEffect.READ_ONLY,
            "freshness": Freshness.STATIC,
            "resource_tags": ("item:{missing}",),
        },  # tag references unknown argument
        # malformed metadata
        {"args_model": LooseArgs},
        {"name": "Bad Name"},
        {"description": "short"},
    ],
)
def test_unsafe_or_inconsistent_specs_are_rejected(overrides):
    with pytest.raises(ValidationError):
        make(**overrides)


def test_side_effect_tool_with_idempotency_may_be_retryable():
    spec = make(effect=ToolEffect.SIDE_EFFECT, retryable=True, idempotency_arg="item_id")
    assert spec.retryable


def test_spec_is_immutable():
    spec = make()
    with pytest.raises(ValidationError):
        spec.effect = ToolEffect.READ_ONLY  # type: ignore[misc]
