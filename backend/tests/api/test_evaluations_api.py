"""Evaluations through the HTTP API: lifecycle, storage, restarts, validation."""

from __future__ import annotations

import time

from tests.api.conftest import GateLLM


def wait_eval(client, evaluation_id, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        e = client.get(f"/evaluations/{evaluation_id}").json()
        if e["finished"]:
            return e
        time.sleep(0.05)
    raise AssertionError("evaluation did not finish")


SMALL = {"task_ids": ["control_annual_price", "redundant_orbit_price"], "repetitions": 2}


def test_evaluation_runs_and_is_stored(client):
    created = client.post("/evaluations", json=SMALL)
    assert created.status_code == 202
    body = created.json()
    assert body["planned_runs"] == 8 and body["status"] == "queued"

    e = wait_eval(client, body["evaluation_id"])
    assert e["status"] == "completed" and e["completed_runs"] == 8
    report = e["report"]
    assert report["provider"] == "scripted" and report["label"].startswith("MOCK")
    assert report["dataset"]["checksum"] == e["dataset"]["checksum"]
    cold = report["phases"]["cold"]["modes"]
    assert cold["passthrough"]["accuracy"] == cold["optimized"]["accuracy"] == 1.0

    runs = client.get(f"/evaluations/{body['evaluation_id']}/runs").json()
    assert len(runs) == 8 and [r["seq"] for r in runs] == list(range(1, 9))
    assert {r["mode"] for r in runs} == {"passthrough", "optimized"}
    # Every measured run links to a full stored agent run.
    linked = client.get(f"/runs/{runs[0]['run_id']}/details").json()
    assert linked["run"]["status"] == "success" and linked["tool_calls"]
    listing = client.get("/evaluations").json()
    assert listing[0]["evaluation_id"] == body["evaluation_id"]
    assert listing[0]["report"] is None  # list view stays small


def test_evaluation_survives_restart(make_client):
    with make_client() as client:
        eid = client.post("/evaluations", json=SMALL).json()["evaluation_id"]
        before = wait_eval(client, eid)
        runs_before = client.get(f"/evaluations/{eid}/runs").json()
    with make_client() as restarted:
        assert restarted.get(f"/evaluations/{eid}").json() == before
        assert restarted.get(f"/evaluations/{eid}/runs").json() == runs_before


def test_shutdown_mid_evaluation_marks_it_interrupted(make_client):
    gate = GateLLM()
    config = {"llm": "gemini", "task_ids": ["control_annual_price"]}
    with make_client(gemini_factory=lambda: gate) as client:
        eid = client.post("/evaluations", json=config).json()["evaluation_id"]
        assert gate.started.wait(5)
    with make_client() as restarted:
        e = restarted.get(f"/evaluations/{eid}").json()
        assert e["status"] == "interrupted" and e["report"] is None


def test_invalid_evaluation_requests(client):
    assert client.post("/evaluations", json={"repetitions": 0}).status_code == 422
    assert client.post("/evaluations", json={"modes": []}).status_code == 422
    assert client.post("/evaluations", json={"surprise": 1}).status_code == 422
    unknown_task = client.post("/evaluations", json={"task_ids": ["nope"]})
    assert unknown_task.status_code == 422
    assert unknown_task.json()["error"]["code"] == "invalid_evaluation"
    unknown_ds = client.post("/evaluations", json={"dataset": "missing"})
    assert unknown_ds.json()["error"]["code"] == "unknown_dataset"
    too_many = client.post("/evaluations", json={"repetitions": 20, "warm_cache": True})
    assert too_many.status_code == 422
    no_key = client.post("/evaluations", json={"llm": "gemini"})
    assert no_key.status_code == 503
    assert client.get("/evaluations").json() == []  # nothing stored
    assert client.get("/evaluations/" + "0" * 32).status_code == 404


def test_dataset_endpoint_hides_expectations_and_scripts(client):
    body = client.get("/datasets/core/1").json()
    assert len(body["tasks"]) == 18
    for task in body["tasks"]:
        assert set(task) == {
            "id",
            "category",
            "split",
            "task",
            "tools",
            "allow_side_effects",
            "scripted_only",
        }
    text = str(body["tasks"])
    for secret_answer in ("45.91", "MSG-0002", "permission_denied", "1500"):
        assert secret_answer not in text
    assert client.get("/datasets/core/9").status_code == 404
