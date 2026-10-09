"""In-memory cache of tool results.

* Values are stored as JSON text, and every read decodes a fresh copy, so a
  caller can never modify a cached result in place.
* Each entry records when it was created (monotonic clock), its TTL, the
  resource tags it depends on, and which run created it.
* ``generation`` increases on every invalidation. A result that was being
  fetched while an invalidation happened is not stored, because it may reflect
  data from before the write.
* Size limits: at most ``max_entries`` entries (oldest evicted first) and
  ``max_entry_bytes`` per value (larger results are not cached).

Freshness (TTL) decisions are made by the gateway with its clock; the store
only keeps entries.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


class _Unset:
    pass


UNSET = _Unset()


@dataclass(frozen=True)
class CacheEntry:
    key: str
    tool_name: str
    scope: str
    value_json: str
    created_at: float  # monotonic seconds
    created_at_iso: str
    ttl_seconds: float | None  # None = never expires (static tools)
    tags: frozenset[str]
    run_id: str
    call_id: str

    def value(self) -> Any:
        return json.loads(self.value_json)

    def age(self, now: float) -> float:
        return max(0.0, now - self.created_at)

    def is_fresh(self, now: float, ttl_seconds: float | None | _Unset = UNSET, /) -> bool:
        """Fresh under this entry's own TTL, or under ``ttl_seconds`` when given
        (the TTL in force now, which a learned rule may have changed)."""
        ttl = self.ttl_seconds if isinstance(ttl_seconds, _Unset) else ttl_seconds
        return ttl is None or self.age(now) < ttl


class CacheStore:
    def __init__(self, *, max_entries: int = 1000, max_entry_bytes: int = 65_536) -> None:
        if max_entries < 1 or max_entry_bytes < 1:
            raise ValueError("cache limits must be positive")
        self._entries: dict[str, CacheEntry] = {}
        self._max_entries = max_entries
        self._max_entry_bytes = max_entry_bytes
        self.generation = 0
        self.evictions = 0
        # Writes whose outcome is still unknown (a timed-out synchronous tool
        # whose thread is still running). While one is pending, no result
        # depending on what it may change can be stored.
        self._pending_all = 0
        self._pending_tags: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> CacheEntry | None:
        return self._entries.get(key)

    def fits(self, value_json: str) -> bool:
        return len(value_json.encode("utf-8")) <= self._max_entry_bytes

    def put(self, entry: CacheEntry) -> bool:
        if not self.fits(entry.value_json):
            return False
        self._entries.pop(entry.key, None)  # re-insert so it counts as newest
        self._entries[entry.key] = entry
        while len(self._entries) > self._max_entries:
            oldest = next(iter(self._entries))
            del self._entries[oldest]
            self.evictions += 1
        return True

    def delete(self, key: str) -> bool:
        return self._entries.pop(key, None) is not None

    def invalidate_tags(self, tags: Iterable[str]) -> int:
        tag_set = set(tags)
        self.generation += 1
        if not tag_set:
            return 0
        doomed = [k for k, e in self._entries.items() if e.tags & tag_set]
        for key in doomed:
            del self._entries[key]
        return len(doomed)

    def invalidate_all(self) -> int:
        self.generation += 1
        count = len(self._entries)
        self._entries.clear()
        return count

    # ------------------------------------------------------------ pending writes
    def begin_pending(self, tags: Iterable[str] | None) -> None:
        """``tags=None`` means the write may change anything."""
        if tags is None:
            self._pending_all += 1
        else:
            self._pending_tags.update(set(tags))

    def end_pending(self, tags: Iterable[str] | None) -> int:
        """The abandoned write finished (or failed): drop what it may have changed."""
        if tags is None:
            self._pending_all = max(0, self._pending_all - 1)
            return self.invalidate_all()
        tag_set = set(tags)
        self._pending_tags.subtract(tag_set)
        self._pending_tags = +self._pending_tags  # drop zero counts
        return self.invalidate_tags(tag_set)

    def blocked_by_pending(self, tags: Iterable[str]) -> bool:
        return self._pending_all > 0 or any(t in self._pending_tags for t in tags)

    @property
    def pending_writes(self) -> int:
        return self._pending_all + sum(self._pending_tags.values())

    def clear_scope(self, scope: str) -> int:
        doomed = [k for k, e in self._entries.items() if e.scope == scope]
        for key in doomed:
            del self._entries[key]
        return len(doomed)
