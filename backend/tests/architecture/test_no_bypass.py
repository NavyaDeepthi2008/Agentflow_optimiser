"""Proof that tool code is reachable only through the ToolGateway.

Two independent kinds of evidence:

1. Static: the import/attribute rules in ``import_rules.py`` hold for every file
   in ``app/`` (and the checker really does catch violations).
2. Runtime: for a mixed workload, the number of executions the gateway counted
   equals the number each tool counted itself, tool by tool. Blocked calls
   reach no tool at all.
"""

from __future__ import annotations

import inspect
import textwrap
from pathlib import Path

import pytest

from app.gateway.gateway import ToolGateway
from app.gateway.models import Decision, ToolCallRequest
from app.tools.builtin.base import MockTool
from app.tools.registry import ToolRegistry
from tests.architecture.import_rules import find_violations

APP_ROOT = Path(__file__).resolve().parents[2] / "app"


# ----------------------------------------------------------------- static rules


def test_app_code_follows_import_rules():
    violations = find_violations(APP_ROOT)
    assert violations == [], "\n".join(str(v) for v in violations)


def _write(root: Path, rel: str, source: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    for parent in path.relative_to(root.parent).parents:
        init = root.parent / parent / "__init__.py"
        if parent != Path(".") and not init.exists():
            init.write_text("")
    path.write_text(textwrap.dedent(source))


@pytest.mark.parametrize(
    ("rel_path", "source", "rule"),
    [
        ("agent/loop.py", "from app.tools.builtin.calculator import Calculator\n", "R1"),
        ("agent/loop.py", "from app.tools.registry import ToolRegistry\n", "R2"),
        ("agent/loop.py", "import app.tools.catalog\n", "R2"),
        ("api/routes.py", "from ..tools.builtin import notify\n", "R1"),
        ("api/routes.py", "from app.tools import builtin\n", "R1"),
        ("agent/loop.py", "def f(gw):\n    return gw._registry\n", "R3"),
        ("agent/loop.py", "def f(r):\n    return r._handler_for('x')\n", "R3"),
        ("agent/loop.py", "def f(r):\n    return getattr(r, '_handler_for')\n", "R3"),
        ("evaluation/run.py", "import importlib\n", "R4"),
        ("learning/x.py", "m = __import__('app.tools.builtin')\n", "R4"),
        ("llm/gemini.py", "from app.gateway.gateway import ToolGateway\n", "R5"),
        ("llm/other.py", "from app.agent.runner import AgentRunner\n", "R5"),
        ("llm/gemini.py", "from app.tools.declarations import tool_declarations\n", "R2"),
        ("agent/runner.py", "from google import genai\n", "R6"),
        ("gateway/x.py", "import google.genai\n", "R6"),
        ("agent/runner.py", "import dotenv\n", "R6"),
        ("agent/runner.py", "from app.llm.gemini import GeminiClient\n", "R7"),
        ("cli.py", "from app.tools.catalog import build_registry\n", "R2"),
        ("optimizer/cache.py", "from app.gateway.gateway import ToolGateway\n", "R8"),
        ("services/runs.py", "from app.tools.catalog import build_registry\n", "R2"),
        ("storage/repository.py", "from app.tools.registry import ToolRegistry\n", "R2"),
        ("agent/runner.py", "import sqlalchemy\n", "R6"),
        ("services/runs.py", "from fastapi import FastAPI\n", "R6"),
        ("api/routes.py", "from alembic import command\n", "R6"),
        ("evaluation/x.py", "from app.tools.catalog import build_registry\n", "R2"),
        ("evaluation/x.py", "from app.tools.registry import ToolRegistry\n", "R2"),
        ("evaluation/x.py", "from app.tools.builtin.prices import price_at\n", "R1"),
        ("api/x.py", "from app.tools.mock_world import MockWorld\n", "R2"),
        ("optimizer/cache.py", "from app.tools.builtin.calculator import Calculator\n", "R1"),
        ("optimizer/cache.py", "def f(r):\n    return r._handler_for('x')\n", "R3"),
        ("learning/x.py", "from app.gateway.gateway import ToolGateway\n", "R9"),
        ("learning/x.py", "from app.gateway.factory import build_gateway\n", "R9"),
        ("learning/x.py", "from app.optimizer.cache_store import CacheStore\n", "R9"),
        ("learning/x.py", "from app.tools.spec import CacheScope\n", "R2"),
    ],
)
def test_checker_catches_violations(tmp_path, rel_path, source, rule):
    app_root = tmp_path / "app"
    _write(app_root, rel_path, source)
    rules = {v.rule for v in find_violations(app_root)}
    assert rule in rules


def test_checker_allows_legitimate_code(tmp_path):
    app_root = tmp_path / "app"
    _write(
        app_root,
        "agent/loop.py",
        "from app.gateway.models import ToolCallRequest\n"
        "async def step(gateway, ctx):\n"
        "    return await gateway.call(ToolCallRequest(tool_name='calculator'), ctx)\n",
    )
    _write(app_root, "tools/catalog.py", "from app.tools.builtin.calculator import Calculator\n")
    _write(app_root, "evaluation/runner.py", "from app.tools.mock_world import MockWorld\n")
    _write(app_root, "learning/v.py", "from app.gateway.models import REUSE_DECISIONS\n")
    assert find_violations(app_root) == []


# ----------------------------------------------------------------- public surface


def test_registry_exposes_no_public_way_to_get_handlers():
    public = {n for n in dir(ToolRegistry) if not n.startswith("_")}
    assert public == {"register", "freeze", "is_frozen", "get_spec", "specs", "names"}


def test_gateway_public_attributes_do_not_leak_tools(gateway):
    for name in dir(gateway):
        if name.startswith("_"):
            continue
        value = getattr(gateway, name)
        if inspect.ismethod(value):
            continue
        assert not isinstance(value, ToolRegistry | MockTool), name
    public_methods = {n for n in dir(ToolGateway) if not n.startswith("_")}
    assert public_methods == {
        "call",
        "clear_cache",
        "mode",
        "stats",
        "tool_declarations",
        "tool_names",
        "tool_spec",
        "with_mode",
    }


# ----------------------------------------------------------------- runtime counting


MIXED_WORKLOAD = [
    ("calculator", {"expression": "1+2"}),
    ("calculator", {"expression": "1+2"}),
    ("kb_search", {"query": "refund policy"}),
    ("doc_read", {"doc_id": "kb-004"}),
    ("doc_read", {"doc_id": "kb-404"}),  # not found -> FAILED, but executed
    ("price_lookup", {"symbol": "NMB-X"}),
    ("record_read", {"record_id": "REC-1002"}),
    ("record_update", {"record_id": "REC-1002", "field": "note", "value": "checked"}),
    ("send_notification", {"recipient": "team:keystone", "message": "note added"}),
    ("send_notification", {"recipient": "team:keystone", "message": "note added"}),
    ("unknown_effect_tool", {"text": "x"}),
    ("flaky_tool", {"text": "x"}),  # transient failure, executed
    ("crashing_tool", {}),  # internal error, executed
    ("bad_output_tool", {}),  # bad output, executed
    ("calculator", {"expression": 7}),  # invalid args -> not executed
    ("no_such_tool", {}),  # unknown -> not executed
]


async def test_gateway_count_equals_tool_internal_count(gateway, make_ctx, world):
    ctx = make_ctx(allow_side_effects=True, max_tool_requests=100)
    for tool, args in MIXED_WORKLOAD:
        await gateway.call(ToolCallRequest(tool_name=tool, args=args), ctx)

    gateway_counts = gateway.stats["live_executions"]
    tool_counts = dict(world.execution_counts)
    assert gateway_counts == tool_counts
    # 16 requests: 2 never reached a tool; flaky_tool ran twice (1 retry) -> 15.
    assert sum(tool_counts.values()) == 15
    assert gateway.stats["requests_total"] == len(MIXED_WORKLOAD)


async def test_blocked_calls_reach_no_tool(gateway, make_ctx, world):
    ctx = make_ctx(max_tool_requests=100)  # no side-effect permission
    blocked = [
        ("send_notification", {"recipient": "team:a", "message": "m"}),
        ("record_update", {"record_id": "REC-1001", "field": "note", "value": "v"}),
        ("unknown_effect_tool", {"text": "x"}),
        ("calculator", {"expression": "1", "x": 1}),
        ("missing", {}),
    ]
    for tool, args in blocked:
        result = await gateway.call(ToolCallRequest(tool_name=tool, args=args), ctx)
        assert result.decision is not Decision.EXECUTED_LIVE
        assert result.executed is False
    assert sum(world.execution_counts.values()) == 0
    assert gateway.stats["live_executions"] == {}
    assert world.outbox == []
