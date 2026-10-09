"""The mock world: all mutable state the mock tools read and change.

One ``MockWorld`` is created per session (and per test), so tests never leak
state into each other. ``execution_counts`` is incremented *inside the tools
themselves*, independently of the gateway; the architecture tests compare the
two counts to prove nothing reaches a tool without passing the gateway.
"""

from __future__ import annotations

import copy
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from app.telemetry.clock import Clock, SystemClock
from app.tools.builtin.corpus import SEED_RECORDS


@dataclass
class MockWorld:
    clock: Clock
    started_at: float
    records: dict[str, dict[str, Any]]
    outbox: list[dict[str, Any]] = field(default_factory=list)
    idempotency_index: dict[str, int] = field(default_factory=dict)
    execution_counts: Counter[str] = field(default_factory=Counter)
    # Sync tools run in worker threads (so they can be timed out); this lock makes
    # each mock tool execution atomic with respect to the shared state above.
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    # Fault injection: tool name -> number of upcoming executions that fail with
    # a transient error (simulated outage). Used by evaluations and tests.
    injected_failures: Counter[str] = field(default_factory=Counter)
    # Test-only knobs and state
    flaky_fail_times: int = 1
    flaky_write_failures: int = 1
    test_writes: list[dict[str, Any]] = field(default_factory=list)
    test_write_keys: dict[str, int] = field(default_factory=dict)

    @classmethod
    def create(cls, clock: Clock | None = None) -> MockWorld:
        clock = clock or SystemClock()
        return cls(
            clock=clock,
            started_at=clock.monotonic(),
            records=copy.deepcopy(SEED_RECORDS),
        )

    def elapsed_seconds(self) -> float:
        return self.clock.monotonic() - self.started_at
