"""Request fingerprints and cache keys.

* The **fingerprint** identifies "the same request": tool name + tool version +
  normalized arguments. The loop guard counts identical requests by fingerprint
  in BOTH modes.
* The **cache key** is the fingerprint plus a scope (run, session or global), so
  results can never leak across runs or sessions that should not share them.

Normalization is deliberately minimal. Arguments were already validated by
pydantic (types fixed, defaults filled in). On top of that, only the per-tool
hints declared in the ToolSpec are applied: collapsing whitespace and ignoring
case, for arguments where the tool itself ignores those differences. Numbers
and words are never reinterpreted.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.tools.spec import CacheScope, ToolSpec


def normalized_args(spec: ToolSpec, validated_args: dict[str, Any]) -> dict[str, Any]:
    args = dict(validated_args)
    for name in spec.normalize_whitespace_args:
        if isinstance(args.get(name), str):
            args[name] = " ".join(args[name].split())
    for name in spec.casefold_args:
        if isinstance(args.get(name), str):
            args[name] = args[name].casefold()
    return args


def _digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def request_fingerprint(spec: ToolSpec, norm_args: dict[str, Any]) -> str:
    return _digest({"tool": spec.name, "version": spec.version, "args": norm_args})


def scope_id(
    spec: ToolSpec, *, run_id: str, session_id: str, scope: CacheScope | None = None
) -> str:
    """The cache partition for a result. ``scope`` overrides the tool's own
    ``cache_scope`` (used for learned scope rules, already checked by the caller)."""
    scope = scope or spec.cache_scope
    if scope is CacheScope.RUN:
        return f"run:{run_id}"
    if scope is CacheScope.SESSION:
        return f"session:{session_id}"
    return "global"


def cache_key(fingerprint: str, scope: str) -> str:
    return _digest({"fingerprint": fingerprint, "scope": scope})


def output_digest(output: Any) -> str:
    """A fingerprint of a tool's OUTPUT, so stored traces can show whether a
    repeated request returned the same result (evidence for learned rules)."""
    canonical = json.dumps(output, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
