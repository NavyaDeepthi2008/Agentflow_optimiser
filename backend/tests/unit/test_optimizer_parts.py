"""Unit tests for cache keys, the cache store and the retry policy."""

from __future__ import annotations

import json
import random

import pytest

from app.optimizer.cache_store import CacheEntry, CacheStore
from app.optimizer.keys import cache_key, normalized_args, request_fingerprint, scope_id
from app.optimizer.retry import RetryPolicy, retry_blocker
from app.tools.builtin.calculator import Calculator
from app.tools.builtin.knowledge import DocRead, KbSearch
from app.tools.builtin.notify import SendNotification
from app.tools.builtin.records import RecordUpdate
from app.tools.test_only.tools import FlakyTool, FlakyWriteTool, UnknownEffectTool

# ------------------------------------------------------------------ keys


def test_whitespace_and_case_hints_apply_only_where_declared():
    kb = KbSearch.spec
    assert normalized_args(kb, {"query": "  Refund\t POLICY ", "limit": 3}) == {
        "query": "refund policy",
        "limit": 3,
    }
    doc = DocRead.spec  # no hints: nothing changes
    assert normalized_args(doc, {"doc_id": "kb-001"}) == {"doc_id": "kb-001"}


def test_calculator_normalization_never_changes_meaning():
    calc = Calculator.spec
    assert normalized_args(calc, {"expression": " 1  +  2 "}) == {"expression": "1 + 2"}
    # "1 2" is invalid arithmetic; normalization must not turn it into "12".
    assert normalized_args(calc, {"expression": "1 2"}) == {"expression": "1 2"}
    # Case is NOT folded for the calculator (no hint declared).
    assert calc.casefold_args == ()


def test_fingerprint_is_stable_and_specific():
    kb = KbSearch.spec
    a = request_fingerprint(kb, {"query": "x", "limit": 3})
    assert a == request_fingerprint(kb, {"limit": 3, "query": "x"})  # key order
    assert a != request_fingerprint(kb, {"query": "x", "limit": 4})
    assert a != request_fingerprint(kb, {"query": "y", "limit": 3})
    assert a != request_fingerprint(DocRead.spec, {"query": "x", "limit": 3})
    newer = KbSearch.spec.model_copy(update={"version": "2"})
    assert a != request_fingerprint(newer, {"query": "x", "limit": 3})


def test_scopes_separate_keys():
    kb = KbSearch.spec
    fp = request_fingerprint(kb, {"query": "x", "limit": 3})
    s1 = scope_id(kb, run_id="r1", session_id="A")
    s2 = scope_id(kb, run_id="r2", session_id="A")
    s3 = scope_id(kb, run_id="r1", session_id="B")
    assert s1 == s2 == "session:A"  # session-scoped tool: shared across runs
    assert cache_key(fp, s1) != cache_key(fp, s3)  # never across sessions
    assert scope_id(Calculator.spec, run_id="r", session_id="A") == "global"
    assert scope_id(FlakyTool.spec, run_id="r9", session_id="A") == "run:r9"


# ------------------------------------------------------------------ cache store


def entry(key="k", *, ttl=10.0, created=0.0, tags=(), scope="session:A", value=None):
    return CacheEntry(
        key=key,
        tool_name="t",
        scope=scope,
        value_json=json.dumps(value if value is not None else {"v": 1}),
        created_at=created,
        created_at_iso="2026-01-01T00:00:00+00:00",
        ttl_seconds=ttl,
        tags=frozenset(tags),
        run_id="r1",
        call_id="c1",
    )


def test_ttl_boundary():
    e = entry(ttl=60, created=100)
    assert e.is_fresh(159.999)
    assert not e.is_fresh(160)  # exactly TTL old = stale
    assert entry(ttl=None).is_fresh(10**9)  # static never expires


def test_values_are_copies():
    store = CacheStore()
    store.put(entry(value={"items": [1, 2]}))
    first = store.get("k").value()
    first["items"].append(99)
    assert store.get("k").value() == {"items": [1, 2]}


def test_tag_invalidation_is_targeted_and_bumps_generation():
    store = CacheStore()
    store.put(entry("a", tags={"record:REC-1"}))
    store.put(entry("b", tags={"record:REC-2"}))
    g = store.generation
    assert store.invalidate_tags(["record:REC-1"]) == 1
    assert store.get("a") is None and store.get("b") is not None
    assert store.generation == g + 1
    assert store.invalidate_tags([]) == 0 and store.generation == g + 2


def test_invalidate_all_and_clear_scope():
    store = CacheStore()
    store.put(entry("a", scope="session:A"))
    store.put(entry("b", scope="session:B"))
    assert store.clear_scope("session:A") == 1 and len(store) == 1
    assert store.invalidate_all() == 1 and len(store) == 0


def test_size_limits():
    store = CacheStore(max_entries=2, max_entry_bytes=30)
    assert not store.put(entry("big", value={"x": "y" * 100}))
    for k in ("a", "b", "c"):
        assert store.put(entry(k))
    assert store.get("a") is None and len(store) == 2 and store.evictions == 1


# ------------------------------------------------------------------ retry rules


@pytest.mark.parametrize(
    ("spec", "args", "error_retryable", "allowed"),
    [
        (KbSearch.spec, {"query": "x"}, True, True),  # read-only + transient
        (KbSearch.spec, {"query": "x"}, False, False),  # permanent error
        (DocRead.spec, {}, True, True),
        (FlakyTool.spec, {}, True, True),
        (RecordUpdate.spec, {}, True, False),  # side effect, not retryable, no key
        (SendNotification.spec, {"idempotency_key": None}, True, False),  # no key given
        (SendNotification.spec, {"idempotency_key": "key-12345"}, True, True),  # key given
        (FlakyWriteTool.spec, {"request_key": None}, True, False),
        (FlakyWriteTool.spec, {"request_key": "abcdefgh"}, True, True),
        (UnknownEffectTool.spec, {}, True, False),  # unknown: never
    ],
)
def test_retry_blocker(spec, args, error_retryable, allowed):
    assert (retry_blocker(spec, args, error_retryable) is None) is allowed


def test_backoff_is_exponential_capped_and_deterministic():
    policy = RetryPolicy(base_delay_seconds=0.2, max_delay_seconds=1.0, jitter_ratio=0.0)
    rng = random.Random(0)
    assert [policy.delay(n, rng) for n in (1, 2, 3, 4)] == [0.2, 0.4, 0.8, 1.0]
    jittery = RetryPolicy(base_delay_seconds=1.0, jitter_ratio=0.1)
    a = [jittery.delay(1, random.Random(7)) for _ in range(2)]
    assert a[0] == a[1] and 0.9 <= a[0] <= 1.1


@pytest.mark.parametrize(
    "kwargs", [{"max_retries": -1}, {"base_delay_seconds": -1}, {"jitter_ratio": 2}]
)
def test_invalid_policies(kwargs):
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)
