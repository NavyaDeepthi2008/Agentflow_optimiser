"""Tool declarations for an LLM's function-calling API.

Produces, for each tool, ``{"name", "description", "parameters"}`` where
``parameters`` is plain JSON Schema with no ``$ref``, no ``title`` keys and no
``anyOf`` for optional fields. This simple shape is accepted by common
function-calling APIs, including Gemini's. It is provider-neutral on purpose;
Stage 2 wires it into the Gemini client and tests it against the real SDK.
"""

from __future__ import annotations

import copy
from typing import Any

from app.optimizer.eligibility import may_reuse_result
from app.tools.spec import RESERVED_ARG, ToolSpec


def _resolve(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            ref_name = node["$ref"].rsplit("/", 1)[-1]
            merged = {
                **copy.deepcopy(defs[ref_name]),
                **{k: v for k, v in node.items() if k != "$ref"},
            }
            return _resolve(merged, defs)
        if "anyOf" in node:
            options = [o for o in node["anyOf"] if o.get("type") != "null"]
            if len(options) == 1:  # "X or null" -> just X (the field stays optional)
                rest = {k: v for k, v in node.items() if k != "anyOf"}
                return _resolve({**options[0], **rest}, defs)
        return {
            key: _resolve(value, defs)
            for key, value in node.items()
            if key not in ("title", "$defs") and not (key == "default" and value is None)
        }
    if isinstance(node, list):
        return [_resolve(item, defs) for item in node]
    return node


FORCE_FRESH_SCHEMA = {
    "type": "boolean",
    "description": "Optional. Set to true only if you need a newer result than one already "
    "obtained for the same request.",
}


def parameters_schema(spec: ToolSpec) -> dict[str, Any]:
    raw = spec.args_model.model_json_schema()
    defs = raw.get("$defs", {})
    schema = _resolve(raw, defs)
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    # Offered for every tool whose results can ever be reused, in BOTH modes, so
    # the model sees identical tool definitions in baseline and optimized runs.
    if may_reuse_result(spec):
        schema["properties"][RESERVED_ARG] = dict(FORCE_FRESH_SCHEMA)
    return schema


def tool_declarations(specs: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"name": s.name, "description": s.description, "parameters": parameters_schema(s)}
        for s in specs
    ]
