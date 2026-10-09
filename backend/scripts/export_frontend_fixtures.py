"""Record REAL API responses for the frontend tests.

Starts the API on a temporary database, runs a few scripted (mock) scenarios and
one small evaluation, and saves the exact JSON responses to
frontend/src/test/fixtures/. The frontend tests use these instead of
hand-written data, so they always reflect what the backend really returns.

    .venv/bin/python scripts/export_frontend_fixtures.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.api.main import create_app  # noqa: E402
from app.config import Settings  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "frontend" / "src" / "test" / "fixtures"


def wait(client: TestClient, path: str) -> dict:
    for _ in range(600):
        body = client.get(path).json()
        if body.get("finished"):
            return body
        time.sleep(0.02)
    raise RuntimeError(f"{path} did not finish")


def run_bundle(client: TestClient, body: dict) -> dict:
    run_id = client.post("/runs", json=body).json()["run_id"]
    run = wait(client, f"/runs/{run_id}")
    return {
        "run": run,
        "details": client.get(f"/runs/{run_id}/details").json(),
        "trace": client.get(f"/runs/{run_id}/trace").json(),
    }


def learning_fixtures(client: TestClient) -> dict:
    """Stage 7: a real training evaluation, mining, validation, approval."""
    out: dict = {}
    train = client.post(
        "/evaluations",
        json={"dataset_version": "2", "split": "train", "warm_cache": True, "repetitions": 2},
    ).json()["evaluation_id"]
    wait(client, f"/evaluations/{train}")
    out["mining"] = client.post("/learning/minings", json={"evaluation_ids": [train]}).json()
    out["minings_list"] = client.get("/learning/minings").json()
    out["learning_rules"] = client.get("/learning/rules").json()
    scope = next(r for r in out["learning_rules"] if r["kind"] == "cache_scope")
    rid = scope["rule_id"]
    out["rule_detail_proposed"] = client.get(f"/learning/rules/{rid}").json()
    vid = client.post(f"/learning/rules/{rid}/validations", json={"repetitions": 1}).json()[
        "validation_id"
    ]
    out["validation_passed"] = wait(client, f"/learning/validations/{vid}")
    out["rule_detail_validated"] = client.get(f"/learning/rules/{rid}").json()
    out["rule_approved"] = client.post(
        f"/learning/rules/{rid}/approve",
        json={"approved_by": "Navya", "note": "evidence reviewed", "confirm": True},
    ).json()
    out["rule_detail_active"] = client.get(f"/learning/rules/{rid}").json()
    out["tools_with_rule"] = client.get("/tools").json()
    out["learning_rules_after_approval"] = client.get("/learning/rules").json()
    body = {
        "scenario": {"task_id": "control_record_status", "version": "2"},
        "llm": "scripted",
        "mode": "optimized",
        "session_id": "fx-learned",
    }
    run_bundle(client, body)  # first run stores the result
    out["learned_rule_run"] = run_bundle(client, body)  # second run reuses it via the rule
    out["rule_disabled"] = client.post(
        f"/learning/rules/{rid}/disable", json={"reason": "demo finished"}
    ).json()
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(Settings(database_url=f"sqlite:///{tmp}/fixtures.db"))
        with TestClient(app) as client:
            data = {
                "health": client.get("/health").json(),
                "tools": client.get("/tools").json(),
                "dataset": client.get("/datasets/core/1").json(),
                "optimized_run": run_bundle(
                    client,
                    {
                        "scenario": {"task_id": "redundant_orbit_price"},
                        "llm": "scripted",
                        "mode": "optimized",
                        "session_id": "fx-opt",
                    },
                ),
                "baseline_run": run_bundle(
                    client,
                    {
                        "scenario": {"task_id": "redundant_orbit_price"},
                        "llm": "scripted",
                        "mode": "passthrough",
                        "session_id": "fx-base",
                    },
                ),
                "retry_run": run_bundle(
                    client,
                    {
                        "scenario": {"task_id": "failure_transient_outage"},
                        "llm": "scripted",
                        "mode": "optimized",
                        "session_id": "fx-retry",
                    },
                ),
                "blocked_run": run_bundle(
                    client,
                    {
                        "scenario": {"task_id": "side_effect_not_permitted"},
                        "llm": "scripted",
                        "mode": "optimized",
                        "session_id": "fx-blocked",
                    },
                ),
            }
            eval_id = client.post(
                "/evaluations",
                json={
                    "repetitions": 1,
                    "warm_cache": True,
                    "categories": ["redundant", "freshness", "control"],
                },
            ).json()["evaluation_id"]
            data["evaluation"] = wait(client, f"/evaluations/{eval_id}")
            data["evaluation_list"] = client.get("/evaluations").json()
            data["runs_list"] = client.get("/runs?limit=5").json()
            data.update(learning_fixtures(client))

    for name, value in data.items():
        (OUT / f"{name}.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(data)} fixtures to {OUT}")


if __name__ == "__main__":
    main()
