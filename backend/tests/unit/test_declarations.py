"""Tool declarations for the LLM are complete and use a simple, portable schema."""

from __future__ import annotations

import json

from app.gateway.factory import build_gateway
from app.tools.catalog import BUILTIN_TOOLS


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def test_declarations_cover_demo_tools_only(world):
    gateway = build_gateway(world=world, include_test_tools=True)
    names = [d["name"] for d in gateway.tool_declarations()]
    assert sorted(names) == sorted(cls.spec.name for cls in BUILTIN_TOOLS)


def test_declaration_schemas_are_simple_and_serializable(world):
    gateway = build_gateway(world=world)
    for decl in gateway.tool_declarations():
        json.dumps(decl)
        params = decl["parameters"]
        assert params["type"] == "object"
        for node in _walk(params):
            assert "$ref" not in node and "$defs" not in node
            assert "anyOf" not in node
            assert "title" not in node
        assert set(params.get("required", [])) <= set(params["properties"])


def test_optional_argument_is_not_required(world):
    gateway = build_gateway(world=world)
    (decl,) = gateway.tool_declarations(["send_notification"])
    assert "idempotency_key" in decl["parameters"]["properties"]
    assert "idempotency_key" not in decl["parameters"]["required"]
    (search,) = gateway.tool_declarations(["kb_search"])
    assert search["parameters"]["properties"]["limit"]["default"] == 3
