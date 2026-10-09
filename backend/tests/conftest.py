from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from app.gateway.context import RunContext
from app.gateway.factory import build_gateway
from app.gateway.gateway import ToolGateway
from app.gateway.models import GatewayMode, ToolCallRequest, ToolResult
from app.optimizer.retry import RetryPolicy
from app.safety.permissions import RunLimits, RunPermissions
from app.telemetry.clock import FakeClock
from app.telemetry.trace import InMemoryTraceRecorder
from app.tools.mock_world import MockWorld

SIDE_EFFECT_TOOLS = frozenset(
    {
        "record_update",
        "send_notification",
        "unknown_effect_tool",
        "flaky_write_tool",
        "slow_write_tool",
        "blocking_write_tool",
    }
)
# Same retry rules as production, but no waiting between attempts (fast tests).
FAST_RETRIES = RetryPolicy(max_retries=2, base_delay_seconds=0.0, jitter_ratio=0.0)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def world(clock: FakeClock) -> MockWorld:
    return MockWorld.create(clock)


@pytest.fixture
def recorder() -> InMemoryTraceRecorder:
    return InMemoryTraceRecorder()


@pytest.fixture
def gateway(world: MockWorld, recorder: InMemoryTraceRecorder) -> ToolGateway:
    """Baseline (pass-through) gateway."""
    return build_gateway(
        world=world, recorder=recorder, include_test_tools=True, retry_policy=FAST_RETRIES
    )


@pytest.fixture
def opt_gateway(world: MockWorld, recorder: InMemoryTraceRecorder) -> ToolGateway:
    """Optimized gateway over the same mock world."""
    return build_gateway(
        world=world,
        recorder=recorder,
        include_test_tools=True,
        retry_policy=FAST_RETRIES,
        mode=GatewayMode.OPTIMIZED,
    )


@pytest.fixture
def make_ctx() -> Callable[..., RunContext]:
    def _make(
        *,
        allow_side_effects: bool = False,
        allowed_tools: frozenset[str] | None = None,
        max_tool_requests: int = 20,
        max_identical_requests: int = 3,
        session_id: str = "session-test",
        fresh_tools: frozenset[str] = frozenset(),
        deadline: float | None = None,
    ) -> RunContext:
        return RunContext(
            session_id=session_id,
            permissions=RunPermissions(
                allowed_tools=allowed_tools,
                side_effect_tools=SIDE_EFFECT_TOOLS if allow_side_effects else frozenset(),
            ),
            limits=RunLimits(
                max_tool_requests=max_tool_requests,
                max_identical_requests=max_identical_requests,
            ),
            fresh_tools=fresh_tools,
            deadline=deadline,
        )

    return _make


@pytest.fixture
def call(gateway: ToolGateway) -> Callable[..., Any]:
    async def _call(ctx: RunContext, tool_name: str, args: Any = None) -> ToolResult:
        return await gateway.call(
            ToolCallRequest(tool_name=tool_name, args={} if args is None else args), ctx
        )

    return _call


@pytest.fixture
def ocall(opt_gateway: ToolGateway) -> Callable[..., Any]:
    async def _call(
        ctx: RunContext, tool_name: str, args: Any = None, *, force_fresh: bool = False
    ) -> ToolResult:
        return await opt_gateway.call(
            ToolCallRequest(
                tool_name=tool_name,
                args={} if args is None else args,
                force_fresh=force_fresh,
            ),
            ctx,
        )

    return _call
