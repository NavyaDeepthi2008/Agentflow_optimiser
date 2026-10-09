"""Run lifecycle, failures, restarts and secrets through the real API."""

from __future__ import annotations

import logging
import time

from app.llm.errors import LLMProviderError
from app.llm.gemini import GeminiClient
from tests.api.conftest import (
    GateLLM,
    calls,
    final,
    scripted_body,
    wait_finished,
    wait_status,
)
from tests.fakes.gemini_server import FAKE_KEY, FakeGeminiServer, call_reply, text_reply

# ------------------------------------------------------------------ lifecycle


def test_run_is_not_finished_until_it_completes(make_client):
    gate = GateLLM()
    with make_client(gemini_factory=lambda: gate) as client:
        rid = client.post("/runs", json={"task": "wait for me"}).json()["run_id"]
        assert gate.started.wait(5)
        wait_status(client, rid, "running")
        run = client.get(f"/runs/{rid}").json()
        assert run["finished"] is False
        assert run["final_answer"] is None and run["metrics"] is None
        assert client.get("/health").json()["active_runs"] == 1
        assert client.get(f"/runs/{rid}/details").json()["llm_calls"] == []

        gate.release.set()
        done = wait_finished(client, rid)
        assert done["status"] == "success" and done["final_answer"] == "released"
        assert done["metrics"]["llm_calls"] == 1
        assert done["started_at"] <= done["ended_at"]


def test_concurrency_limit_keeps_extra_runs_queued(make_client):
    gate = GateLLM()
    with make_client(gemini_factory=lambda: gate, max_concurrent_runs=1) as client:
        first = client.post("/runs", json={"task": "one"}).json()["run_id"]
        second = client.post("/runs", json={"task": "two"}).json()["run_id"]
        wait_status(client, first, "running")
        time.sleep(0.1)
        assert client.get(f"/runs/{second}").json()["status"] == "queued"
        gate.release.set()
        assert wait_finished(client, first)["status"] == "success"
        assert wait_finished(client, second)["status"] == "success"


def test_llm_provider_failure_is_recorded_as_failed(make_client):
    class FailingLLM:
        provider, model_id = "gemini", "gemini-test"

        async def generate(self, request):
            raise LLMProviderError("rate limited (simulated)", code="rate_limited")

    with make_client(gemini_factory=FailingLLM) as client:
        rid = client.post("/runs", json={"task": "t"}).json()["run_id"]
        run = wait_finished(client, rid)
        assert run["status"] == "failed"
        assert run["error"] == {"code": "rate_limited", "message": "rate limited (simulated)"}
        llm_calls = client.get(f"/runs/{rid}/details").json()["llm_calls"]
        assert llm_calls[0]["ok"] is False and llm_calls[0]["error_code"] == "rate_limited"


def test_failure_while_saving_marks_error_not_success(make_client, monkeypatch):
    with make_client() as client:
        repo = client.app.state.repo

        def broken_save(result):
            raise RuntimeError("disk problem")

        monkeypatch.setattr(repo, "save_result", broken_save)
        rid = client.post("/runs", json=scripted_body(final("would be success"))).json()["run_id"]
        run = wait_finished(client, rid)
        assert run["status"] == "error"
        assert run["error"]["code"] == "internal_error"
        assert run["final_answer"] is None


# ------------------------------------------------------------------ restarts


def test_completed_runs_survive_a_restart(make_client):
    body = scripted_body(
        calls(("calculator", {"expression": "29*12"})), final("348"), mode="optimized"
    )
    with make_client() as client:
        rid = client.post("/runs", json=body).json()["run_id"]
        before = wait_finished(client, rid)
        details_before = client.get(f"/runs/{rid}/details").json()
        trace_before = client.get(f"/runs/{rid}/trace").json()

    with make_client() as restarted:  # new app, same database file
        assert restarted.get(f"/runs/{rid}").json() == before
        assert restarted.get(f"/runs/{rid}/details").json() == details_before
        assert restarted.get(f"/runs/{rid}/trace").json() == trace_before


def test_shutdown_during_a_run_marks_it_interrupted(make_client):
    gate = GateLLM()
    with make_client(gemini_factory=lambda: gate) as client:
        rid = client.post("/runs", json={"task": "never finishes"}).json()["run_id"]
        wait_status(client, rid, "running")
    # leaving the block shut the app down while the LLM was still waiting
    with make_client() as restarted:
        run = restarted.get(f"/runs/{rid}").json()
        assert run["status"] == "interrupted" and run["finished"] is True
        assert run["final_answer"] is None
        types = [e["type"] for e in restarted.get(f"/runs/{rid}/trace").json()["events"]]
        assert types[0] == "run.started"  # events written before the stop were kept


def test_runs_left_active_by_a_crash_are_recovered_at_startup(make_client):
    with make_client() as client:
        repo = client.app.state.repo
        repo.create_queued(
            run_id="d" * 32,
            session_id="s",
            task="t",
            mode="passthrough",
            provider="gemini",
            model_id="m",
            created_at="2026-01-01T00:00:00+00:00",
            request_options={},
            config_snapshot={},
        )
        repo.mark_running("d" * 32, "2026-01-01T00:00:01+00:00")
    # simulate a crash: the row stays "running"; the next start must fix it
    with make_client() as restarted:
        run = restarted.get("/runs/" + "d" * 32).json()
        assert run["status"] == "interrupted"
        assert run["error"]["code"] == "interrupted"


# ------------------------------------------------------------------ secrets


def test_api_key_never_reaches_responses_database_or_logs(make_client, db_url, caplog):
    server = FakeGeminiServer(
        [
            call_reply(("kb_search", {"query": "refund"}, "fc-1"), signature=b"sig"),
            text_reply(f"Refunds within 30 days. (debug: {FAKE_KEY})"),  # model echoes a secret
        ]
    )

    def gemini():
        return GeminiClient(api_key=FAKE_KEY, model="gemini-test", http_client=server.http_client())

    caplog.set_level(logging.DEBUG)
    with make_client(gemini_factory=gemini, gemini_api_key=FAKE_KEY) as client:
        assert client.get("/health").json()["gemini_configured"] is True
        rid = client.post("/runs", json={"task": f"refund window? key={FAKE_KEY}"}).json()["run_id"]
        run = wait_finished(client, rid)
        assert run["status"] == "success"
        responses = [
            client.get(f"/runs/{rid}").text,
            client.get(f"/runs/{rid}/details").text,
            client.get(f"/runs/{rid}/trace").text,
            client.get("/runs").text,
            client.get("/health").text,
            client.get("/tools").text,
        ]
        for text in responses:
            assert FAKE_KEY not in text
        assert "[REDACTED]" in run["final_answer"] and "[REDACTED]" in run["task"]

    db_file = db_url.removeprefix("sqlite:///")
    from pathlib import Path

    for path in Path(db_file).parent.glob(Path(db_file).name + "*"):  # db, -wal, -shm
        assert FAKE_KEY.encode() not in path.read_bytes()
    assert FAKE_KEY not in caplog.text
