"""Optional checks against the REAL Gemini API.

Skipped unless both are set:
    GEMINI_API_KEY=...      (your key, e.g. in backend/.env)
    RUN_LIVE_TESTS=1

Run with:  .venv/bin/python -m pytest -m live
These calls cost a small amount of money and depend on the network.
"""

from __future__ import annotations

import os

import pytest

from app.agent.models import RunStatus
from app.agent.runner import AgentRunner
from app.config import load_settings
from app.gateway.context import RunContext
from app.gateway.factory import build_gateway
from app.llm.gemini import GeminiClient
from app.tools.mock_world import MockWorld

pytestmark = pytest.mark.live

settings = load_settings()
if os.environ.get("RUN_LIVE_TESTS") != "1" or not settings.has_gemini_key:
    pytest.skip(
        "live tests disabled (set RUN_LIVE_TESTS=1 and GEMINI_API_KEY)", allow_module_level=True
    )


async def test_live_agent_uses_a_tool_through_the_gateway():
    world = MockWorld.create()
    gateway = build_gateway(world=world)
    runner = AgentRunner(GeminiClient.from_settings(settings), gateway)
    result = await runner.run(
        "Use the calculator tool to compute 1234 * 5678 and reply with just the number.",
        RunContext(session_id="live-test"),
        tools=["calculator"],
    )
    assert result.status is RunStatus.SUCCESS, result.summary()
    assert world.execution_counts["calculator"] >= 1
    assert gateway.stats["live_executions"] == dict(world.execution_counts)
    assert "7006652" in (result.final_answer or "").replace(",", "")
    assert result.llm_calls[0].usage is not None


async def test_live_same_task_in_both_modes():
    """One real task in baseline and optimized mode. This checks that the real
    model works through both gateways; it is ONE sample and says nothing
    general about savings (use the evaluation engine with repetitions for that)."""
    results = {}
    for mode in ("passthrough", "optimized"):
        world = MockWorld.create()
        gateway = build_gateway(world=world, mode=mode)
        runner = AgentRunner(GeminiClient.from_settings(settings), gateway)
        results[mode] = await runner.run(
            "What does Orbit cost per user per month? Use the knowledge base tools.",
            RunContext(session_id=f"live-{mode}"),
            tools=["kb_search", "doc_read"],
        )
        assert results[mode].status is RunStatus.SUCCESS, results[mode].summary()
        assert gateway.stats["live_executions"] == dict(world.execution_counts)
    for mode, r in results.items():
        print(mode, r.tool_request_count, r.tool_execution_count, r.final_answer)
