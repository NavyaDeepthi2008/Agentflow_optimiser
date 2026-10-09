"""Clocks.

Two kinds of time are used everywhere in AgentFlow:

* ``monotonic()`` - a number of seconds that only ever moves forward. Used for
  measuring durations and (later) cache ages, because the wall clock can jump.
* ``utcnow()`` - a timezone-aware UTC timestamp, used for display and storage.

Tests use :class:`FakeClock`, which only moves when told to, so time-dependent
behaviour (price changes, TTL expiry) is fully deterministic.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def monotonic(self) -> float: ...

    def utcnow(self) -> datetime: ...


class SystemClock:
    """Real time."""

    def monotonic(self) -> float:
        return time.monotonic()

    def utcnow(self) -> datetime:
        return datetime.now(UTC)


class FakeClock:
    """A clock that only moves when :meth:`advance` is called."""

    def __init__(self, start: datetime | None = None) -> None:
        self._elapsed = 0.0
        self._start = start or datetime(2026, 1, 1, tzinfo=UTC)
        if self._start.tzinfo is None:
            raise ValueError("FakeClock start must be timezone-aware")

    def monotonic(self) -> float:
        return self._elapsed

    def utcnow(self) -> datetime:
        return self._start + timedelta(seconds=self._elapsed)

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("time cannot move backwards")
        self._elapsed += seconds


class OffsetClock:
    """Real time plus a manual offset: ``advance`` simulates time passing.

    Used as the DATA clock in evaluations, so a script can say "five minutes
    later" without waiting, while latency is still measured with real time.
    """

    def __init__(self, base: Clock | None = None) -> None:
        self._base = base or SystemClock()
        self._offset = 0.0

    @property
    def offset_seconds(self) -> float:
        return self._offset

    def monotonic(self) -> float:
        return self._base.monotonic() + self._offset

    def utcnow(self) -> datetime:
        return self._base.utcnow() + timedelta(seconds=self._offset)

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("time cannot move backwards")
        self._offset += seconds
