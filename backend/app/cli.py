"""Run one agent task from the command line.

    python -m app.cli "What does Orbit cost per user per month?"
    python -m app.cli --allow-side-effects "Set REC-1001 to resolved and notify team:orbit"

Uses the real Gemini API (needs GEMINI_API_KEY) and the deterministic mock
tools, in baseline (pass-through) mode. Prints a JSON summary; never prints the
API key.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from app.agent.models import AgentLimits
from app.agent.runner import AgentRunner
from app.config import load_settings
from app.gateway.context import RunContext
from app.gateway.factory import build_gateway
from app.llm.errors import LLMConfigurationError
from app.llm.gemini import GeminiClient
from app.safety.permissions import RunLimits, RunPermissions
from app.telemetry.trace import InMemoryTraceRecorder

SIDE_EFFECT_TOOLS = frozenset({"record_update", "send_notification"})


async def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    parser.add_argument("task", help="the task for the agent")
    parser.add_argument(
        "--allow-side-effects",
        action="store_true",
        help="allow record_update and send_notification (mock tools only)",
    )
    parser.add_argument("--trace", action="store_true", help="also print trace events")
    args = parser.parse_args(argv)

    settings = load_settings()
    try:
        llm = GeminiClient.from_settings(settings)
    except LLMConfigurationError as exc:
        print(f"Configuration error: {exc.message}", file=sys.stderr)
        return 2

    recorder = InMemoryTraceRecorder()
    gateway = build_gateway(recorder=recorder)
    runner = AgentRunner(
        llm,
        gateway,
        recorder=recorder,
        limits=AgentLimits(
            max_steps=settings.agent_max_steps,
            run_timeout_seconds=settings.agent_run_timeout_seconds,
        ),
    )
    ctx = RunContext(
        session_id="cli",
        permissions=RunPermissions(
            side_effect_tools=SIDE_EFFECT_TOOLS if args.allow_side_effects else frozenset()
        ),
        limits=RunLimits(max_tool_requests=settings.agent_max_tool_requests),
    )
    result = await runner.run(args.task, ctx)
    print(json.dumps(result.summary(), indent=2))
    if args.trace:
        for event in recorder.events_for_run(ctx.run_id):
            print(json.dumps(event.model_dump(), default=str))
    return 0 if result.status.value == "success" else 1


def main() -> None:
    raise SystemExit(asyncio.run(_main(sys.argv[1:])))


if __name__ == "__main__":
    main()
