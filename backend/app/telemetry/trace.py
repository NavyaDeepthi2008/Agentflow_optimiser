"""Trace events: the recorded history of everything the gateway did.

Stage 1 keeps events in memory. Stage 4 adds a SQLite-backed recorder with the
same ``record`` method, so the gateway does not need to change.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class TraceEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    type: str
    run_id: str
    session_id: str
    call_id: str | None = None
    at: str  # UTC timestamp, ISO 8601
    data: dict[str, Any] = Field(default_factory=dict)


class TraceRecorder(Protocol):
    def record(self, event: TraceEvent) -> None: ...


class InMemoryTraceRecorder:
    """Keeps events in a list, in the order they were recorded."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []
        self._lock = threading.Lock()

    def record(self, event: TraceEvent) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> list[TraceEvent]:
        with self._lock:
            return list(self._events)

    def events_for_run(self, run_id: str) -> list[TraceEvent]:
        return [e for e in self.events if e.run_id == run_id]

    def events_for_call(self, call_id: str) -> list[TraceEvent]:
        return [e for e in self.events if e.call_id == call_id]
