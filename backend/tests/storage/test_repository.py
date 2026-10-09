"""RunRepository: lifecycle, atomic results, redaction, truncation, cascades."""

from __future__ import annotations

import pytest
from sqlalchemy import delete, func, select

from app.agent.models import RunStatus
from app.agent.runner import AgentRunner
from app.gateway.context import RunContext
from app.gateway.factory import build_gateway
from app.llm.base import TokenUsage
from app.llm.scripted import ScriptedLLM
from app.storage.db import Database
from app.storage.models import LLMCallRow, RunRow, ToolCallRow, TraceEventRow
from app.storage.redaction import REDACTED, Redactor
from app.storage.repository import (
    MAX_STORED_OUTPUT_BYTES,
    RunNotFoundError,
    RunRepository,
    SqlTraceRecorder,
    _bounded_output,
)

SECRET = "configured-secret-value-9876"
GOOGLE_SHAPED = "AIza" + "B" * 35
S = ScriptedLLM


@pytest.fixture
def db(tmp_path):
    database = Database(f"sqlite:///{tmp_path}/r.db")
    database.migrate()
    yield database
    database.dispose()


@pytest.fixture
def repo(db):
    return RunRepository(db, Redactor([SECRET]))


def queue(repo, run_id="a" * 32, task="t", session="s1", created="2026-01-01T00:00:00+00:00"):
    repo.create_queued(
        run_id=run_id,
        session_id=session,
        task=task,
        mode="passthrough",
        provider="scripted",
        model_id="scripted-llm",
        created_at=created,
        request_options={"llm": "scripted"},
        config_snapshot={"max_steps": 12},
    )
    return run_id


async def finished_result(repo, run_id, llm):
    ctx = RunContext(session_id="s1", run_id=run_id)
    runner = AgentRunner(llm, build_gateway(), recorder=SqlTraceRecorder(repo))
    return await runner.run("What does Orbit cost?", ctx)


def test_queued_run_has_no_metrics(repo):
    rid = queue(repo)
    row = repo.get(rid)
    assert row.status == "queued" and row.steps is None and row.final_answer is None


def test_unknown_run(repo):
    with pytest.raises(RunNotFoundError):
        repo.get("f" * 32)
    with pytest.raises(RunNotFoundError):
        repo.trace("f" * 32)


async def test_full_result_is_saved_atomically(repo, db):
    rid = queue(repo)
    repo.mark_running(rid, "2026-01-01T00:00:01+00:00")
    usage = TokenUsage(input_tokens=10, output_tokens=2, total_tokens=12)
    llm = S(
        [
            S.tool_calls(("kb_search", {"query": "orbit"}), usage=usage),
            S.tool_calls(("doc_read", {"doc_id": "kb-404"}), usage=usage),
            S.final("29 USD", usage=usage),
        ]
    )
    result = await finished_result(repo, rid, llm)
    repo.save_result(result)

    row = repo.get(rid)
    assert row.status == "success" and row.final_answer == "29 USD"
    assert (row.steps, row.llm_call_count, row.tool_request_count) == (3, 3, 2)
    assert row.tool_execution_count == 2 and row.reused_result_count == 0
    assert (row.input_tokens, row.output_tokens, row.usage_complete) == (30, 6, True)
    assert row.decision_counts == {"executed_live": 1, "failed": 1}
    assert row.started_at == "2026-01-01T00:00:01+00:00"
    assert [c.step for c in repo.llm_calls(rid)] == [1, 2, 3]
    calls = repo.tool_calls(rid)
    assert [(c.seq, c.tool_name, c.decision) for c in calls] == [
        (1, "kb_search", "executed_live"),
        (2, "doc_read", "failed"),
    ]
    assert calls[1].error_code == "not_found" and calls[1].output is None
    events = repo.trace(rid)
    assert events[0].type == "run.started" and events[-1].type == "run.completed"
    assert [e.id for e in events] == sorted(e.id for e in events)


async def test_missing_usage_is_stored_as_null(repo):
    rid = queue(repo)
    repo.save_result(await finished_result(repo, rid, S([S.final("x")])))
    row = repo.get(rid)
    assert row.input_tokens is None and row.total_tokens is None
    assert row.usage_complete is False


async def test_error_never_overwrites_a_finished_run(repo):
    rid = queue(repo)
    repo.save_result(await finished_result(repo, rid, S([S.final("done")])))
    repo.mark_error(rid, "internal_error", "late failure", "2026-01-01T00:01:00+00:00")
    assert repo.get(rid).status == "success"


def test_interrupted_only_touches_active_runs(repo, db):
    a, b, c = queue(repo, "a" * 32), queue(repo, "b" * 32), queue(repo, "c" * 32)
    repo.mark_running(b, "2026-01-01T00:00:01+00:00")
    repo.mark_error(c, "internal_error", "boom", "2026-01-01T00:00:02+00:00")
    assert repo.mark_interrupted_runs("2026-01-01T00:00:03+00:00") == 2
    assert [repo.get(x).status for x in (a, b, c)] == ["interrupted", "interrupted", "error"]


def test_mark_running_only_from_queued(repo):
    rid = queue(repo)
    repo.mark_error(rid, "x", "y", "2026-01-01T00:00:00+00:00")
    repo.mark_running(rid, "2026-01-01T00:00:01+00:00")
    assert repo.get(rid).status == "error"


async def test_secrets_are_redacted_everywhere(repo, db):
    task = f"use key {SECRET} and {GOOGLE_SHAPED}"
    rid = queue(repo, task=task)
    llm = S(
        [
            S.tool_calls(("kb_search", {"query": f"x {SECRET}"})),
            S.final(f"answer with {SECRET}"),
        ]
    )
    repo.save_result(await finished_result(repo, rid, llm))
    repo.mark_error(rid, "x", f"message {SECRET}", "2026-01-01T00:00:00+00:00")  # no-op
    row = repo.get(rid)
    assert SECRET not in row.task and GOOGLE_SHAPED not in row.task and REDACTED in row.task
    assert SECRET not in row.final_answer
    for event in repo.trace(rid):
        assert SECRET not in str(event.data)
    assert any(REDACTED in str(e.data) for e in repo.trace(rid))


def test_redactor_handles_nesting_and_short_values():
    r = Redactor([SECRET, "short"])  # values under 8 chars are ignored (too risky)
    data = {"a": [f"x{SECRET}y", {"b": SECRET}], SECRET: 1, "n": 5}
    assert r(data) == {"a": [f"x{REDACTED}y", {"b": REDACTED}], REDACTED: 1, "n": 5}
    assert r("a short word") == "a short word"


def test_large_outputs_are_truncated():
    big = {"text": "x" * (MAX_STORED_OUTPUT_BYTES + 10)}
    stored, truncated = _bounded_output(big)
    assert truncated and stored["truncated"] is True
    assert _bounded_output({"a": 1}) == ({"a": 1}, False)


async def test_deleting_a_run_deletes_its_details(repo, db):
    rid = queue(repo)
    llm = S([S.tool_calls(("calculator", {"expression": "1+1"})), S.final("2")])
    repo.save_result(await finished_result(repo, rid, llm))
    with db.session() as s, s.begin():
        s.execute(delete(RunRow).where(RunRow.id == rid))
    with db.session() as s:
        for table in (LLMCallRow, ToolCallRow, TraceEventRow):
            assert s.scalar(select(func.count()).select_from(table)) == 0


def test_list_runs_filters_and_orders(repo):
    queue(repo, "a" * 32, session="s1", created="2026-01-01T00:00:01+00:00")
    queue(repo, "b" * 32, session="s2", created="2026-01-01T00:00:02+00:00")
    queue(repo, "c" * 32, session="s1", created="2026-01-01T00:00:03+00:00")
    assert [r.id[0] for r in repo.list_runs()] == ["c", "b", "a"]
    assert [r.id[0] for r in repo.list_runs(session_id="s1")] == ["c", "a"]
    assert [r.id[0] for r in repo.list_runs(limit=1)] == ["c"]
    assert repo.list_runs(status="success") == []


async def test_trace_paging(repo):
    rid = queue(repo)
    repo.save_result(await finished_result(repo, rid, S([S.final("x")])))
    all_events = repo.trace(rid)
    first = repo.trace(rid, limit=1)
    rest = repo.trace(rid, after_id=first[-1].id)
    assert [e.id for e in first + rest] == [e.id for e in all_events]


async def test_result_for_unknown_run_is_rejected(repo):
    result = await finished_result(repo, "e" * 32, S([S.final("x")]))
    with pytest.raises(Exception):  # noqa: B017 - FK or not-found, either is a refusal
        repo.save_result(result)
    assert result.status is RunStatus.SUCCESS  # the agent itself was fine
