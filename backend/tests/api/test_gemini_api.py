"""The real Gemini code path through the whole API, offline.

Every test uses the genuine ``GeminiClient`` and ``google-genai`` SDK; only the
HTTP transport is replaced by ``FakeGeminiServer``. No key, no network, no
cost. What these tests can NOT show: how the real model behaves (which tools it
calls, how often it repeats itself, answer quality). That needs a live run.
"""

from __future__ import annotations

import time

from app.llm.gemini import GeminiClient
from tests.api.conftest import wait_finished
from tests.fakes.gemini_server import (
    FAKE_KEY,
    FakeGeminiServer,
    _candidate,
    call_reply,
    error_body,
    text_reply,
)


def factory(server, **kw):
    def make():
        return GeminiClient(
            api_key=FAKE_KEY, model="gemini-test", http_client=server.http_client(), **kw
        )

    return make


def run(client, **body):
    rid = client.post("/runs", json={"task": "What does Orbit cost?", **body}).json()["run_id"]
    return wait_finished(client, rid), rid


def test_invalid_key_fails_the_run_with_a_safe_message(make_client, db_url):
    server = FakeGeminiServer(
        [(400, error_body(400, f"API key not valid: {FAKE_KEY}", "INVALID_ARGUMENT"))]
        + [(401, error_body(401, "unauthenticated", "UNAUTHENTICATED"))]
    )
    with make_client(gemini_factory=factory(server), gemini_api_key=FAKE_KEY) as client:
        assert client.get("/health").json()["gemini_configured"] is True
        first, rid = run(client)
        assert first["status"] == "failed"
        assert first["error"]["code"] == "provider_error"
        second, rid2 = run(client)
        assert second["error"]["code"] == "auth_error"
        for r in (rid, rid2):
            for path in (f"/runs/{r}", f"/runs/{r}/details", f"/runs/{r}/trace"):
                assert FAKE_KEY not in client.get(path).text
    assert FAKE_KEY.encode() not in open(db_url.removeprefix("sqlite:///"), "rb").read()


def test_unknown_model_is_reported_clearly(make_client):
    server = FakeGeminiServer([(404, error_body(404, "model not found", "NOT_FOUND"))])
    with make_client(gemini_factory=factory(server)) as client:
        result, _ = run(client)
        assert result["status"] == "failed"
        assert result["error"]["code"] == "model_not_found"


def test_slow_model_times_out_and_the_run_fails_cleanly(make_client):
    server = FakeGeminiServer([("sleep", 2.0)])
    with make_client(gemini_factory=factory(server, timeout_seconds=0.2)) as client:
        start = time.monotonic()
        result, rid = run(client)
        assert time.monotonic() - start < 2.0
        assert result["status"] == "failed" and result["error"]["code"] == "timeout"
        llm = client.get(f"/runs/{rid}/details").json()["llm_calls"]
        assert llm[0]["ok"] is False and llm[0]["error_code"] == "timeout"


def test_malformed_function_call_fails_the_run(make_client):
    server = FakeGeminiServer([_candidate([], usage=True, finish="MALFORMED_FUNCTION_CALL")])
    with make_client(gemini_factory=factory(server)) as client:
        result, _ = run(client)
        assert result["error"]["code"] == "malformed_function_call"


def test_unknown_tool_and_bad_args_from_the_model_are_blocked_by_the_gateway(make_client):
    server = FakeGeminiServer(
        [
            call_reply(
                ("delete_everything", {}, "c1"),
                ("kb_search", {"query": 5}, "c2"),
                ("send_notification", {"recipient": "team:orbit", "message": "hi"}, "c3"),
            ),
            text_reply("I could not do that."),
        ]
    )
    with make_client(gemini_factory=factory(server)) as client:
        result, rid = run(client)
        assert result["status"] == "success"
        calls = client.get(f"/runs/{rid}/details").json()["tool_calls"]
        assert [c["decision"] for c in calls] == [
            "blocked_unknown_tool",
            "blocked_invalid_args",
            "blocked_permission",  # side effects are never allowed for free-text tasks
        ]
        assert all(c["executions"] == 0 for c in calls)
        # The model saw the refusals as function responses.
        responses = server.bodies[1]["contents"][-1]["parts"]
        assert all(p["functionResponse"]["response"]["ok"] is False for p in responses)


def gemini_script():
    """The same model behavior for both modes: search, repeat the search,
    read, answer. (A real model may or may not repeat itself like this.)"""
    return [
        call_reply(("kb_search", {"query": "orbit price"}, "a")),
        call_reply(("kb_search", {"query": "Orbit  price"}, "b")),
        call_reply(("doc_read", {"doc_id": "kb-002"}, "c")),
        text_reply("Orbit costs 29 USD per user per month."),
    ]


def test_baseline_and_optimized_with_the_gemini_client(make_client):
    server = FakeGeminiServer(gemini_script() + gemini_script())
    with make_client(gemini_factory=factory(server)) as client:
        base, base_id = run(client, mode="passthrough", session_id="g-b")
        opt, opt_id = run(client, mode="optimized", session_id="g-o")
        assert base["status"] == opt["status"] == "success"
        assert base["final_answer"] == opt["final_answer"]
        assert base["provider"] == opt["provider"] == "gemini"
        assert base["metrics"]["tool_requests"] == opt["metrics"]["tool_requests"] == 3
        assert base["metrics"]["tool_executions"] == 3
        assert opt["metrics"]["tool_executions"] == 2
        # Usage reported by the (fake) API is recorded, never invented.
        assert opt["metrics"]["input_tokens"] == 400 and opt["metrics"]["usage_complete"]
        # The reused result told the model it was reused.
        third_request = server.bodies[6]  # optimized run, 3rd LLM call (after the reuse)
        part = third_request["contents"][-1]["parts"][0]["functionResponse"]["response"]
        assert "Reused" in part["note"]


def test_gemini_evaluation_is_labelled_live_and_skips_simulated_time_tasks(make_client):
    replies = []
    for _ in range(2):
        replies += [call_reply(("calculator", {"expression": "29*12"}, "a")), text_reply("348")]
    server = FakeGeminiServer(replies)
    body = {
        "llm": "gemini",
        "task_ids": ["control_annual_price", "freshness_price_changed"],
    }
    with make_client(gemini_factory=factory(server)) as client:
        eid = client.post("/evaluations", json=body).json()["evaluation_id"]
        for _ in range(300):
            e = client.get(f"/evaluations/{eid}").json()
            if e["finished"]:
                break
            time.sleep(0.02)
        assert e["status"] == "completed"
        report = e["report"]
        assert report["provider"] == "gemini" and report["label"].startswith("LIVE MODEL")
        assert [s["task_id"] for s in report["skipped_tasks"]] == ["freshness_price_changed"]
