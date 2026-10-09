"""Per-run state the gateway needs (which run, which session, what is allowed)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from uuid import uuid4

from app.safety.permissions import RunLimits, RunPermissions


@dataclass
class RunContext:
    session_id: str
    permissions: RunPermissions = field(default_factory=RunPermissions)
    limits: RunLimits = field(default_factory=RunLimits)
    run_id: str = field(default_factory=lambda: uuid4().hex)
    # Tools whose results must always be fetched fresh in this task (verification).
    fresh_tools: frozenset[str] = frozenset()
    # Monotonic time after which no tool may start; set by the agent from its
    # run timeout. Tool timeouts are shortened so they never run past it.
    deadline: float | None = None

    # Mutated by the gateway only.
    tool_requests: int = 0
    stop_reason: str | None = None
    identical_requests: Counter[str] = field(default_factory=Counter)
    forced_fresh_keys: set[str] = field(default_factory=set)

    @property
    def is_fresh(self) -> bool:
        return self.tool_requests == 0 and self.stop_reason is None
