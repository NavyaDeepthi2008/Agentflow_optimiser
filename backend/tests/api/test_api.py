"""HTTP API: endpoints, validation, modes, sessions, safety rules."""

from __future__ import annotations

from tests.api.conftest import calls, final, scripted_body, wait_finished

# ------------------------------------------------------------------ basics


def test_health(client):
    body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "database": "ok",
        "gemini_configured": False,
        "gemini_model": None,
        "pricing_configured": False,
        "active_runs": 0,
        "active_evaluations": 0,
        "active_validations": 0,
        "active_learned_rules": 0,
    }


def test_tools_lists_demo_tools_with_metadata(client):
    tools = {t["name"]: t for t in client.get("/tools").json()["tools"]}
    assert set(tools) == {
        "calculator",
        "kb_search",
        "doc_read",
        "price_lookup",
        "record_read",
        "record_update",
        "send_notification",
    }
    assert tools["send_notification"]["effect"] == "side_effect"
    assert tools["send_notification"]["requires_permission"] is True
    assert tools["price_lookup"]["ttl_seconds"] == 60
    assert "force_fresh" in tools["kb_search"]["parameters"]["properties"]
    assert "force_fresh" not in tools["record_update"]["parameters"]["properties"]


def test_scripted_run_completes_with_details_and_trace(client):
    created = client.post(
        "/runs",
        json=scripted_body(
            calls(("kb_search", {"query": "orbit price"})),
            calls(("doc_read", {"doc_id": "kb-002"})),
            final("29 USD per user per month"),
        ),
    )
    assert created.status_code == 202
    body = created.json()
    assert body["status"] == "queued" and body["links"]["self"] == f"/runs/{body['run_id']}"

    run = wait_finished(client, body["run_id"])
    assert run["status"] == "success"
    assert run["final_answer"] == "29 USD per user per month"
    assert run["provider"] == "scripted" and run["mode"] == "passthrough"
    m = run["metrics"]
    assert (m["steps"], m["llm_calls"], m["tool_requests"], m["tool_executions"]) == (3, 3, 2, 2)
    assert m["input_tokens"] is None and m["usage_complete"] is False  # scripted: no usage
    assert run["config_snapshot"]["max_steps"] == 12

    details = client.get(f"/runs/{body['run_id']}/details").json()
    assert [c["step"] for c in details["llm_calls"]] == [1, 2, 3]
    assert [c["tool_name"] for c in details["tool_calls"]] == ["kb_search", "doc_read"]
    assert details["tool_calls"][1]["output"]["doc_id"] == "kb-002"

    trace = client.get(f"/runs/{body['run_id']}/trace").json()
    types = [e["type"] for e in trace["events"]]
    assert types[0] == "run.started" and types[-1] == "run.completed"
    assert types.count("tool_call.completed") == 2
    assert trace["next_after_id"] is None


def test_trace_paging(client):
    run_id = client.post("/runs", json=scripted_body(final("hi"))).json()["run_id"]
    wait_finished(client, run_id)
    page = client.get(f"/runs/{run_id}/trace?limit=1").json()
    assert len(page["events"]) == 1 and page["next_after_id"] == page["events"][0]["id"]
    rest = client.get(f"/runs/{run_id}/trace?after_id={page['next_after_id']}").json()
    assert rest["events"][0]["id"] > page["events"][0]["id"]


def test_list_runs_by_session(client):
    a = client.post("/runs", json=scripted_body(final("a"), session_id="alpha")).json()["run_id"]
    b = client.post("/runs", json=scripted_body(final("b"), session_id="beta")).json()["run_id"]
    wait_finished(client, a)
    wait_finished(client, b)
    assert [r["run_id"] for r in client.get("/runs?session_id=alpha").json()] == [a]
    assert len(client.get("/runs").json()) == 2
    assert len(client.get("/runs?limit=1").json()) == 1


# ------------------------------------------------------------------ errors & validation


def test_unknown_run_is_404(client):
    response = client.get("/runs/" + "0" * 32)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "run_not_found"
    assert client.get("/runs/" + "0" * 32 + "/trace").status_code == 404
    assert client.get("/runs/" + "0" * 32 + "/details").status_code == 404


def test_malformed_run_id_is_rejected(client):
    assert client.get("/runs/not-an-id").status_code == 422
    assert client.get("/runs/../../etc/passwd").status_code == 404  # path never matches


def test_validation_errors_do_not_echo_submitted_values(client):
    secret_task = "SECRET-VALUE-" + "x" * 5000  # too long
    response = client.post(
        "/runs", json={"task": secret_task, "llm": "scripted", "scripted_replies": [{"text": "x"}]}
    )
    assert response.status_code == 422
    assert "SECRET-VALUE" not in response.text
    assert response.json()["error"]["details"][0]["loc"] == ["body", "task"]


def test_invalid_requests(client):
    bad_bodies = [
        {"task": "t", "llm": "scripted"},  # no replies
        {"task": "t", "llm": "gemini", "scripted_replies": [{"text": "x"}]},
        {"task": "   ", "llm": "scripted", "scripted_replies": [{"text": "x"}]},
        scripted_body({"text": "x", "tool_calls": [{"name": "a"}]}),  # both kinds
        scripted_body(final("x"), mode="turbo"),
        scripted_body(final("x"), session_id="bad id!"),
        scripted_body(final("x"), unexpected=True),
        scripted_body(final("x"), max_steps=0),
    ]
    for body in bad_bodies:
        assert client.post("/runs", json=body).status_code == 422, body
    assert client.get("/runs").json() == []  # nothing was stored


def test_unknown_tool_in_request_is_rejected_before_storing(client):
    response = client.post("/runs", json=scripted_body(final("x"), tools=["rm_rf"]))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unknown_tool"
    test_only = client.post("/runs", json=scripted_body(final("x"), tools=["crashing_tool"]))
    assert test_only.status_code == 422  # test-only tools are not available via the API
    assert client.get("/runs").json() == []


def test_gemini_without_key_is_503_and_nothing_stored(client):
    response = client.post("/runs", json={"task": "hello"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_not_configured"
    assert client.get("/runs").json() == []


def test_unexpected_error_is_generic_500(db_url, monkeypatch):
    from fastapi.testclient import TestClient

    from app.api.main import create_app
    from app.config import Settings

    app = create_app(Settings(database_url=db_url))
    with TestClient(app, raise_server_exceptions=False) as c:

        def boom(*a, **k):
            raise RuntimeError("internal detail with secret-ish text")

        monkeypatch.setattr(c.app.state.repo, "list_runs", boom)
        response = c.get("/runs")
        assert response.status_code == 500
        assert response.json() == {
            "error": {"code": "internal_error", "message": "an unexpected error occurred"}
        }


# ------------------------------------------------------------------ modes, safety, sessions


REDUNDANT = (
    calls(("kb_search", {"query": "Orbit price"})),
    calls(("kb_search", {"query": "orbit price"})),
    calls(("doc_read", {"doc_id": "kb-002"})),
    calls(("doc_read", {"doc_id": "kb-002"})),
    final("29 USD"),
)


def test_baseline_and_optimized_through_the_api(client):
    ids = {
        mode: client.post(
            "/runs", json=scripted_body(*REDUNDANT, mode=mode, session_id=f"s-{mode}")
        ).json()["run_id"]
        for mode in ("passthrough", "optimized")
    }
    base = wait_finished(client, ids["passthrough"])
    opt = wait_finished(client, ids["optimized"])
    assert base["final_answer"] == opt["final_answer"] == "29 USD"
    assert base["metrics"]["tool_requests"] == opt["metrics"]["tool_requests"] == 4
    assert base["metrics"]["tool_executions"] == 4
    assert opt["metrics"]["tool_executions"] == 2
    assert opt["metrics"]["reused_results"] == 2
    assert opt["metrics"]["decision_counts"] == {"executed_live": 2, "reused_in_run": 2}
    assert base["metrics"]["decision_counts"] == {"executed_live": 4}


def test_side_effects_need_permission(client):
    notify = calls(("send_notification", {"recipient": "team:a", "message": "hi"}))
    denied = client.post("/runs", json=scripted_body(notify, final("ok"))).json()["run_id"]
    allowed = client.post(
        "/runs", json=scripted_body(notify, final("ok"), allow_side_effects=True)
    ).json()["run_id"]
    for rid, decision in ((denied, "blocked_permission"), (allowed, "executed_live")):
        wait_finished(client, rid)
        tool = client.get(f"/runs/{rid}/details").json()["tool_calls"][0]
        assert tool["decision"] == decision


def test_tool_allowlist_from_request(client):
    rid = client.post(
        "/runs",
        json=scripted_body(calls(("kb_search", {"query": "x"})), final("ok"), tools=["calculator"]),
    ).json()["run_id"]
    wait_finished(client, rid)
    assert client.get(f"/runs/{rid}/details").json()["tool_calls"][0]["decision"] == (
        "blocked_permission"
    )


def test_session_cache_is_shared_within_a_session_and_clearable(client):
    body = scripted_body(
        calls(("doc_read", {"doc_id": "kb-004"})),
        final("ok"),
        mode="optimized",
        session_id="cachetest",
    )

    def decision_of_new_run():
        rid = client.post("/runs", json=body).json()["run_id"]
        wait_finished(client, rid)
        return client.get(f"/runs/{rid}/details").json()["tool_calls"][0]["decision"]

    assert decision_of_new_run() == "executed_live"
    assert decision_of_new_run() == "cache_hit"
    other = dict(body, session_id="someone-else")
    rid = client.post("/runs", json=other).json()["run_id"]
    wait_finished(client, rid)
    assert client.get(f"/runs/{rid}/details").json()["tool_calls"][0]["decision"] == (
        "executed_live"
    )
    cleared = client.delete("/sessions/cachetest/cache").json()
    assert cleared == {"session_id": "cachetest", "cleared_entries": 1}
    assert decision_of_new_run() == "executed_live"
    assert client.delete("/sessions/never-used/cache").json()["cleared_entries"] == 0


def test_loop_detection_via_api(client):
    loop = [calls(("kb_search", {"query": "same"})) for _ in range(6)]
    rid = client.post("/runs", json=scripted_body(*loop, mode="optimized")).json()["run_id"]
    run = wait_finished(client, rid)
    assert run["status"] == "loop_detected"
    assert run["error"]["code"] == "max_identical_requests"


def test_request_limits_override(client):
    many = [calls(("calculator", {"expression": f"{i}+1"})) for i in range(5)]
    rid = client.post("/runs", json=scripted_body(*many, final("x"), max_tool_requests=2)).json()[
        "run_id"
    ]
    assert wait_finished(client, rid)["status"] == "limit_tool_calls"
    rid = client.post("/runs", json=scripted_body(*many, final("x"), max_steps=2)).json()["run_id"]
    assert wait_finished(client, rid)["status"] == "limit_steps"
