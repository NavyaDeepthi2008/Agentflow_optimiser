"""Convenience constructor: the normal way to get a ready-to-use gateway."""

from __future__ import annotations

from app.gateway.gateway import OptimizerConfig, ToolGateway
from app.gateway.models import GatewayMode
from app.optimizer.learned import PolicyProvider
from app.optimizer.retry import RetryPolicy
from app.telemetry.clock import Clock, SystemClock
from app.telemetry.trace import TraceRecorder
from app.tools.catalog import build_registry
from app.tools.mock_world import MockWorld


def build_gateway(
    *,
    world: MockWorld | None = None,
    clock: Clock | None = None,
    recorder: TraceRecorder | None = None,
    include_test_tools: bool = False,
    mode: GatewayMode | str = GatewayMode.PASSTHROUGH,
    retry_policy: RetryPolicy | None = None,
    optimizer_config: OptimizerConfig | None = None,
    policy: PolicyProvider | None = None,
) -> ToolGateway:
    """Build a gateway over the mock tools.

    * ``world`` holds the mock data. Its clock is the DATA clock: it drives
      prices and cache ages (TTL).
    * ``clock`` is the TIMING clock used for durations and run deadlines. If
      omitted it is the world's clock (or the real clock when no world is given).
    * ``policy`` returns the learned rules (active and shadow) to use; it is
      called once per request, so approving or disabling a rule takes effect
      on the next request.
    """
    timing_clock = clock or (world.clock if world else SystemClock())
    world = world or MockWorld.create(timing_clock)
    registry = build_registry(world, include_test_tools=include_test_tools)
    return ToolGateway(
        registry,
        mode=GatewayMode(mode),
        clock=timing_clock,
        data_clock=world.clock,
        recorder=recorder,
        retry_policy=retry_policy,
        optimizer_config=optimizer_config,
        policy=policy,
    )
