"""Backend features added for the dashboard: CORS, scenario runs, cost, API contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.llm.gemini import GeminiClient
from tests.api.conftest import wait_finished
from tests.fakes.gemini_server import FAKE_KEY, FakeGeminiServer, call_reply, text_reply

# ------------------------------------------------------------------ CORS


def test_cors_allows_only_configured_origins(client):
    ok = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-credentials" not in ok.headers
    other = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_cors_preflight(client):
    response = client.options(
        "/runs",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert "POST" in response.headers["access-control-allow-methods"]
    blocked = client.options(
        "/runs",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "PUT",
        },
    )
    assert blocked.status_code == 400


@pytest.mark.parametrize("value", ["*", "localhost:5173", "http://a.test,*"])
def test_wildcard_or_malformed_origins_are_rejected(value):
    with pytest.raises(ValidationError):
        Settings(cors_origins=value)


# ------------------------------------------------------------------ scenarios


def scenario(task_id, **extra):
    return {"scenario": {"task_id": task_id}, "llm": "scripted", **extra}


def test_scripted_scenario_uses_the_dataset_task_and_script(client):
    rid = client.post(
        "/runs", json=scenario("redundant_orbit_price", mode="optimized", session_id="sc1")
    ).json()["run_id"]
    run = wait_finished(client, rid)
    assert run["status"] == "success"
    assert run["task"].startswith("How much does Orbit cost")
    assert "29" in run["final_answer"]
    assert run["metrics"]["tool_requests"] == 4 and run["metrics"]["tool_executions"] == 2
    assert run["request_options"]["scenario"]["task_id"] == "redundant_orbit_price"
    assert "expected" not in json.dumps(run)  # nothing from the hidden expectations


def test_scenario_permissions_come_from_the_dataset(client):
    rid = client.post("/runs", json=scenario("side_effect_double_reminder")).json()["run_id"]
    wait_finished(client, rid)
    decisions = [t["decision"] for t in client.get(f"/runs/{rid}/details").json()["tool_calls"]]
    assert decisions == ["executed_live", "executed_live"]
    rid = client.post("/runs", json=scenario("side_effect_not_permitted")).json()["run_id"]
    wait_finished(client, rid)
    tool = client.get(f"/runs/{rid}/details").json()["tool_calls"][0]
    assert tool["decision"] == "blocked_permission"


def test_freshness_scenario_advances_simulated_time(client):
    rid = client.post(
        "/runs", json=scenario("freshness_price_changed", mode="optimized", session_id="sc-fresh")
    ).json()["run_id"]
    run = wait_finished(client, rid)
    assert "45.91" in run["final_answer"]
    tools = client.get(f"/runs/{rid}/details").json()["tool_calls"]
    assert [t["decision"] for t in tools] == ["executed_live", "stale_refreshed"]


def test_fault_injection_scenario(client):
    rid = client.post("/runs", json=scenario("failure_transient_outage")).json()["run_id"]
    run = wait_finished(client, rid)
    assert run["status"] == "success" and run["metrics"]["tool_retries"] == 1


def test_invalid_scenario_requests(client):
    assert client.post("/runs", json=scenario("no_such_task")).json()["error"]["code"] == (
        "unknown_scenario"
    )
    assert client.post("/runs", json=scenario("control_annual_price", task="x")).status_code == 422
    assert (
        client.post("/runs", json=scenario("control_annual_price", tools=["kb_search"])).status_code
        == 422
    )
    assert (
        client.post(
            "/runs", json=scenario("control_annual_price", allow_side_effects=True)
        ).status_code
        == 422
    )
    assert client.get("/runs").json() == []


def test_scripted_only_scenarios_cannot_use_gemini(make_client):
    with make_client(gemini_factory=lambda: None) as client:
        body = {"scenario": {"task_id": "freshness_price_changed"}, "llm": "gemini"}
        response = client.post("/runs", json=body)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "scenario_scripted_only"


# ------------------------------------------------------------------ cost


def write_pricing(tmp_path, model="gemini-test") -> str:
    path = tmp_path / "pricing.json"
    path.write_text(
        json.dumps(
            {
                "source": "test fixture (not real prices)",
                "retrieved_on": "2026-01-01",
                "models": {
                    model: {
                        "input_per_million": 1.0,
                        "output_per_million": 2.0,
                        "thinking_billed_as_output": True,
                    }
                },
            }
        )
    )
    return str(path)


def test_cost_shown_only_with_pricing_and_reported_usage(make_client, tmp_path):
    server = FakeGeminiServer(
        [
            call_reply(("calculator", {"expression": "29*12"}, "c1")),
            text_reply("348"),
        ]
    )

    def gemini():
        return GeminiClient(api_key=FAKE_KEY, model="gemini-test", http_client=server.http_client())

    with make_client(gemini_factory=gemini, pricing_file=write_pricing(tmp_path)) as client:
        assert client.get("/health").json()["pricing_configured"] is True
        rid = client.post("/runs", json={"task": "29*12?"}).json()["run_id"]
        run = wait_finished(client, rid)
        cost = run["estimated_cost"]
        # 2 calls: 200 input * 1 + 40 output * 2 + 10 thinking * 2 (per million)
        assert cost["llm"] == pytest.approx(300 / 1_000_000)
        assert cost["pricing_source"].startswith("test fixture")
        scripted = client.post("/runs", json=scenario("control_annual_price")).json()["run_id"]
        assert wait_finished(client, scripted)["estimated_cost"] is None  # no usage reported


def test_no_cost_without_pricing(client):
    rid = client.post("/runs", json=scenario("control_annual_price")).json()["run_id"]
    assert wait_finished(client, rid)["estimated_cost"] is None
    assert client.get("/health").json()["pricing_configured"] is False


# ------------------------------------------------------------------ contract


def test_committed_openapi_schema_is_current():
    """The frontend's types are generated from backend/openapi.json."""
    from scripts.export_openapi import OUTPUT, current_schema

    committed = json.loads(Path(OUTPUT).read_text())
    assert committed == json.loads(json.dumps(current_schema(), sort_keys=True)), (
        "openapi.json is stale: run scripts/export_openapi.py and npm run gen:api"
    )
