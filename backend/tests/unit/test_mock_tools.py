"""Direct tests of each mock tool's logic (tests may call tools directly)."""

from __future__ import annotations

import pytest

from app.tools.builtin.calculator import Calculator, CalculatorArgs, evaluate_expression
from app.tools.builtin.knowledge import DocRead, DocReadArgs, KbSearch, KbSearchArgs
from app.tools.builtin.notify import SendNotification, SendNotificationArgs
from app.tools.builtin.prices import TICK_SECONDS, PriceLookup, PriceLookupArgs, price_at
from app.tools.builtin.records import (
    RecordRead,
    RecordReadArgs,
    RecordUpdate,
    RecordUpdateArgs,
)
from app.tools.errors import ToolInputError, ToolNotFoundError

# ----------------------------------------------------------------- calculator


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("10 / 4", 2.5),
        ("10 // 4", 2),
        ("10 % 4", 2),
        ("2 ** 10", 1024),
        ("-3 + +5", 2),
        ("29 * 12", 348),
        ("0.1 + 0.2", 0.3),
    ],
)
def test_calculator_arithmetic(expr, expected):
    assert evaluate_expression(expr) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hacked')",
        "open('/etc/passwd')",
        "x + 1",
        "(1).__class__",
        "[1, 2]",
        "'a' * 3",
        "True + 1",
        "1 if 1 else 2",
        "lambda: 1",
        "2 ** 1000",
        "10 ** 20",
        "1 / 0",
        "5 % 0",
        "(-8) ** 0.5",
        "2 +",
        "",
    ],
)
def test_calculator_rejects_unsafe_or_invalid(expr):
    with pytest.raises(ToolInputError):
        evaluate_expression(expr)


def test_calculator_tool_counts_execution(world):
    Calculator(world)(CalculatorArgs(expression="1+1"))
    assert world.execution_counts["calculator"] == 1


# ----------------------------------------------------------------- knowledge


def test_kb_search_is_deterministic_and_ranked(world):
    tool = KbSearch(world)
    first = tool(KbSearchArgs(query="Orbit price per user"))
    second = tool(KbSearchArgs(query="Orbit price per user"))
    assert first == second
    assert first["results"][0]["doc_id"] == "kb-002"
    scores = [r["score"] for r in first["results"]]
    assert scores == sorted(scores, reverse=True)


def test_kb_search_limit_and_empty(world):
    tool = KbSearch(world)
    assert len(tool(KbSearchArgs(query="Orbit Keystone Nimbus", limit=1))["results"]) == 1
    assert tool(KbSearchArgs(query="zebra quantum"))["results"] == []


def test_doc_read_found_and_missing(world):
    tool = DocRead(world)
    assert tool(DocReadArgs(doc_id="kb-004"))["title"] == "Refund policy"
    with pytest.raises(ToolNotFoundError):
        tool(DocReadArgs(doc_id="kb-999"))


# ----------------------------------------------------------------- prices


def test_price_changes_only_when_tick_changes(world, clock):
    tool = PriceLookup(world)
    p0 = tool(PriceLookupArgs(symbol="NMB-X"))["price"]
    clock.advance(TICK_SECONDS - 1)
    assert tool(PriceLookupArgs(symbol="NMB-X"))["price"] == p0
    clock.advance(1)
    p1 = tool(PriceLookupArgs(symbol="NMB-X"))["price"]
    assert p1 == price_at("NMB-X", 1)
    assert p1 != p0


def test_prices_stay_within_one_percent():
    for tick in range(50):
        assert abs(price_at("ORB-X", tick) - 45.50) <= 45.50 * 0.01 + 1e-9


def test_unknown_symbol(world):
    with pytest.raises(ToolNotFoundError):
        PriceLookup(world)(PriceLookupArgs(symbol="ZZZ-X"))


# ----------------------------------------------------------------- records


def test_record_update_changes_data_and_version(world):
    update = RecordUpdate(world)
    read = RecordRead(world)
    before = read(RecordReadArgs(record_id="REC-1001"))
    after = update(RecordUpdateArgs(record_id="REC-1001", field="status", value="resolved"))
    assert after["status"] == "resolved"
    assert after["version"] == before["version"] + 1
    assert read(RecordReadArgs(record_id="REC-1001"))["status"] == "resolved"


def test_record_read_returns_a_copy(world):
    record = RecordRead(world)(RecordReadArgs(record_id="REC-1002"))
    record["status"] = "tampered"
    assert world.records["REC-1002"]["status"] == "in_progress"


def test_record_update_rejects_bad_status_and_missing_record(world):
    update = RecordUpdate(world)
    with pytest.raises(ToolInputError):
        update(RecordUpdateArgs(record_id="REC-1001", field="status", value="maybe"))
    with pytest.raises(ToolNotFoundError):
        update(RecordUpdateArgs(record_id="REC-9999", field="note", value="x"))


def test_worlds_are_isolated(clock):
    from app.tools.mock_world import MockWorld

    a, b = MockWorld.create(clock), MockWorld.create(clock)
    RecordUpdate(a)(RecordUpdateArgs(record_id="REC-1001", field="note", value="changed"))
    assert b.records["REC-1001"]["note"] == ""


# ----------------------------------------------------------------- notify


def test_notification_writes_to_outbox_each_time(world):
    tool = SendNotification(world)
    args = SendNotificationArgs(recipient="team:billing", message="hello")
    tool(args)
    tool(args)
    assert len(world.outbox) == 2
    assert world.execution_counts["send_notification"] == 2


def test_notification_idempotency_key_prevents_duplicate_send(world):
    tool = SendNotification(world)
    args = SendNotificationArgs(recipient="user:asha", message="hello", idempotency_key="key-12345")
    first = tool(args)
    second = tool(args)
    assert len(world.outbox) == 1
    assert second["duplicate_suppressed"] is True
    assert second["message_id"] == first["message_id"]
    # The tool still ran both times - nothing was skipped by anyone else.
    assert world.execution_counts["send_notification"] == 2
