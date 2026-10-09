"""Tools that exist only to test failure handling and safety rules.

They are registered only when ``include_test_tools=True`` and are never offered
to the agent in the demo.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.tools.builtin.base import MockTool
from app.tools.errors import ToolNotFoundError, ToolTransientError
from app.tools.spec import CacheScope, Freshness, ToolEffect, ToolSpec


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(default="ping", max_length=100)


class _SleepArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seconds: float = Field(default=0.01, ge=0, le=30)


class FlakyTool(MockTool):
    """Fails with a transient error for the first ``world.flaky_fail_times`` calls."""

    spec = ToolSpec(
        name="flaky_tool",
        description="Test tool that fails temporarily before succeeding.",
        args_model=_EchoArgs,
        effect=ToolEffect.READ_ONLY,
        freshness=Freshness.LIVE,
        retryable=True,
        test_only=True,
    )

    def run(self, args: _EchoArgs) -> dict[str, Any]:
        if self.world.execution_counts[self.spec.name] <= self.world.flaky_fail_times:
            raise ToolTransientError("temporary failure (simulated)")
        return {"echo": args.text}


class SlowTool(MockTool):
    """An async tool that sleeps for real time. Used for timing and (Stage 3) timeouts."""

    spec = ToolSpec(
        name="slow_tool",
        description="Test tool that waits for the requested number of seconds.",
        args_model=_SleepArgs,
        effect=ToolEffect.READ_ONLY,
        freshness=Freshness.LIVE,
        test_only=True,
    )

    async def run(self, args: _SleepArgs) -> dict[str, Any]:  # type: ignore[override]
        await asyncio.sleep(args.seconds)
        return {"slept_seconds": args.seconds}


class UnknownEffectTool(MockTool):
    """A tool whose effect was never classified (uses the safe defaults)."""

    spec = ToolSpec(
        name="unknown_effect_tool",
        description="Test tool with no effect classification; must be treated as unsafe.",
        args_model=_EchoArgs,
        test_only=True,
    )

    def run(self, args: _EchoArgs) -> dict[str, Any]:
        return {"echo": args.text}


class CrashingTool(MockTool):
    """Raises an unexpected exception, simulating a bug inside a tool."""

    spec = ToolSpec(
        name="crashing_tool",
        description="Test tool that raises an unexpected internal error.",
        args_model=_NoArgs,
        effect=ToolEffect.READ_ONLY,
        freshness=Freshness.LIVE,
        test_only=True,
    )

    def run(self, args: _NoArgs) -> dict[str, Any]:
        raise RuntimeError("internal detail that must not leak: secret-token-123")


class BadOutputTool(MockTool):
    """Returns something that cannot be turned into JSON."""

    spec = ToolSpec(
        name="bad_output_tool",
        description="Test tool that returns a value that is not JSON-serializable.",
        args_model=_NoArgs,
        effect=ToolEffect.READ_ONLY,
        freshness=Freshness.LIVE,
        test_only=True,
    )

    def run(self, args: _NoArgs) -> Any:
        return {"value": object()}


class _WriteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str = Field(default="v", max_length=100)
    request_key: str | None = Field(default=None, min_length=8, max_length=64)
    seconds: float = Field(default=0.0, ge=0, le=30)


class BlockingTool(MockTool):
    """A *synchronous* tool that blocks with time.sleep (tests sync-tool timeouts)."""

    spec = ToolSpec(
        name="blocking_tool",
        description="Test tool that blocks its thread for the requested seconds.",
        args_model=_SleepArgs,
        effect=ToolEffect.READ_ONLY,
        freshness=Freshness.LIVE,
        timeout_seconds=0.2,
        test_only=True,
    )
    run_under_lock = False  # it blocks; must not hold the world lock

    def run(self, args: _SleepArgs) -> dict[str, Any]:
        time.sleep(args.seconds)
        return {"slept_seconds": args.seconds}


def _record_write(world: Any, args: _WriteArgs) -> dict[str, Any] | None:
    """Write once per idempotency key. Returns the earlier write for a repeated key."""
    if args.request_key and args.request_key in world.test_write_keys:
        return {**world.test_writes[world.test_write_keys[args.request_key]], "duplicate": True}
    world.test_writes.append({"value": args.value, "n": len(world.test_writes) + 1})
    if args.request_key:
        world.test_write_keys[args.request_key] = len(world.test_writes) - 1
    return None


class FlakyWriteTool(MockTool):
    """Side effect whose response is 'lost' after the write happened.

    The first ``world.flaky_write_failures`` calls perform the write and THEN raise
    a transient error - the dangerous case for retries. With a ``request_key``
    (idempotency key) a retry is harmless; without one it would write twice.
    """

    spec = ToolSpec(
        name="flaky_write_tool",
        description="Test side-effect tool that writes, then reports a transient failure.",
        args_model=_WriteArgs,
        effect=ToolEffect.SIDE_EFFECT,
        freshness=Freshness.LIVE,
        invalidates=(),
        idempotency_arg="request_key",
        retryable=True,
        test_only=True,
    )

    def run(self, args: _WriteArgs) -> dict[str, Any]:
        earlier = _record_write(self.world, args)
        if earlier is not None:
            return earlier
        if self.world.flaky_write_failures > 0:
            self.world.flaky_write_failures -= 1
            raise ToolTransientError("connection lost after the write (simulated)")
        return {**self.world.test_writes[-1], "duplicate": False}


class SlowWriteTool(MockTool):
    """Async side effect that writes immediately, then takes ``seconds`` to reply."""

    spec = ToolSpec(
        name="slow_write_tool",
        description="Test side-effect tool that writes and then responds slowly.",
        args_model=_WriteArgs,
        effect=ToolEffect.SIDE_EFFECT,
        freshness=Freshness.LIVE,
        invalidates=(),
        idempotency_arg="request_key",
        retryable=True,
        timeout_seconds=0.2,
        test_only=True,
    )

    async def run(self, args: _WriteArgs) -> dict[str, Any]:  # type: ignore[override]
        earlier = _record_write(self.world, args)
        if earlier is not None:
            return earlier
        await asyncio.sleep(args.seconds)
        return {**self.world.test_writes[-1], "duplicate": False}


class _SlowReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_id: str = Field(pattern=r"^REC-\d{4}$")
    seconds: float = Field(default=0.05, ge=0, le=5)


class SlowRecordRead(MockTool):
    """Cacheable async read that reads the record FIRST, then waits.

    Used to prove that a result fetched while a write happened is not cached.
    """

    spec = ToolSpec(
        name="slow_record_read",
        description="Test tool: read a record, then respond slowly.",
        args_model=_SlowReadArgs,
        effect=ToolEffect.READ_ONLY,
        freshness=Freshness.TTL,
        ttl_seconds=300,
        cache_scope=CacheScope.SESSION,
        resource_tags=("record:{record_id}",),
        test_only=True,
    )

    async def run(self, args: _SlowReadArgs) -> dict[str, Any]:  # type: ignore[override]
        record = self.world.records.get(args.record_id)
        snapshot = dict(record) if record is not None else None
        await asyncio.sleep(args.seconds)
        if snapshot is None:
            raise ToolNotFoundError(f"record {args.record_id} does not exist")
        return snapshot


class _BlockingWriteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_id: str = Field(pattern=r"^REC-\d{4}$")
    status: str = Field(min_length=1, max_length=40)
    seconds: float = Field(default=0.5, ge=0, le=5)


class BlockingWriteTool(MockTool):
    """A *synchronous* side effect that blocks, THEN changes a record.

    If it times out, Python cannot stop its thread: the write still happens
    later. Used to prove the gateway keeps affected results out of the cache
    until the abandoned call has really finished.
    """

    spec = ToolSpec(
        name="blocking_write_tool",
        description="Test side-effect tool: waits, then sets a record's status.",
        args_model=_BlockingWriteArgs,
        effect=ToolEffect.SIDE_EFFECT,
        freshness=Freshness.LIVE,
        invalidates=("record:{record_id}",),
        timeout_seconds=0.2,
        test_only=True,
    )
    run_under_lock = False  # it blocks; it locks only the write itself

    def run(self, args: _BlockingWriteArgs) -> dict[str, Any]:
        time.sleep(args.seconds)
        with self.world.lock:
            self.world.records[args.record_id]["status"] = args.status
        return {"record_id": args.record_id, "status": args.status}


TEST_TOOLS: tuple[type[MockTool], ...] = (
    SlowRecordRead,
    FlakyTool,
    SlowTool,
    UnknownEffectTool,
    CrashingTool,
    BadOutputTool,
    BlockingTool,
    FlakyWriteTool,
    SlowWriteTool,
    BlockingWriteTool,
)
