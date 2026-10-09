"""Learning engine over HTTP: mine -> review evidence -> shadow -> validate ->
approve -> active in real runs -> disable, persisted across restarts."""

from __future__ import annotations

import time

import pytest

from tests.api.conftest import wait_finished

TRAIN = {
    "dataset_version": "2",
    "split": "train",
    "warm_cache": True,
    "repetitions": 2,
    "llm": "scripted",
}


def wait_eval(client, evaluation_id, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        e = client.get(f"/evaluations/{evaluation_id}").json()
        if e["finished"]:
            assert e["status"] == "completed", e
            return e
        time.sleep(0.05)
    raise AssertionError("evaluation did not finish")


def wait_validation(client, validation_id, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = client.get(f"/learning/validations/{validation_id}").json()
        if v["finished"]:
            return v
        time.sleep(0.05)
    raise AssertionError("validation did not finish")


def train_and_mine(client, **mine_body):
    eid = client.post("/evaluations", json=TRAIN).json()["evaluation_id"]
    wait_eval(client, eid)
    resp = client.post("/learning/minings", json={"evaluation_ids": [eid], **mine_body})
    assert resp.status_code == 201, resp.json()
    return eid, resp.json()


def rule_by(client, tool, kind):
    rules = client.get("/learning/rules").json()
    (rule,) = [r for r in rules if r["tool_name"] == tool and r["kind"] == kind]
    return rule


@pytest.fixture
def mined(client):
    eid, mining = train_and_mine(client)
    return client, eid, mining


# ------------------------------------------------------------------ mining
def test_mining_without_training_data_is_refused(client):
    resp = client.post("/learning/minings", json={})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "no_training_data"
    assert client.get("/learning/rules").json() == []


def test_mining_rejects_weakened_thresholds(client):
    resp = client.post("/learning/minings", json={"thresholds": {"min_support": 1}})
    assert resp.status_code == 422
    resp = client.post("/learning/minings", json={"thresholds": {"required_consistency": 0.9}})
    assert resp.status_code == 422


def test_mining_unknown_or_wrong_evaluation_is_refused(client):
    resp = client.post("/learning/minings", json={"evaluation_ids": ["0" * 32]})
    assert resp.json()["error"]["code"] == "invalid_evaluations"


def test_mining_proposes_with_evidence(mined):
    client, eid, mining = mined
    report = mining["report"]
    assert report["runs_used"] > 0 and report["created_rule_ids"]
    cands = {(c["tool_name"], c["kind"]): c for c in report["candidates"]}
    scope = cands[("record_read", "cache_scope")]
    assert scope["proposed"] is True
    assert scope["current_setting"] == {"cache_scope": "run"}
    assert scope["proposed_setting"] == {"cache_scope": "session"}
    ev = scope["evidence"]
    assert ev["repeat_count"] >= 5 and ev["distinct_runs"] >= 3 and ev["consistency"] == 1.0
    assert ev["supporting_run_ids"] and ev["evaluation_ids"] == [eid]
    ttl = cands[("record_read", "ttl")]
    assert ttl["proposed"] is True
    assert ttl["current_setting"] == {"ttl_seconds": 300.0}
    assert ttl["proposed_setting"] == {"ttl_seconds": 600.0}
    # Not proposed, with a reason:
    assert cands[("price_lookup", "ttl")]["proposed"] is False
    assert "changed" in cands[("price_lookup", "ttl")]["reasons"][0]
    assert cands[("kb_search", "ttl")]["proposed"] is False
    # Never considered:
    assert {t["tool_name"] for t in report["not_eligible"]} == {
        "record_update",
        "send_notification",
    }
    # Every supporting run id is a real stored run that was correct.
    for run_id in ev["supporting_run_ids"][:4]:
        run = client.get(f"/runs/{run_id}").json()
        assert run["status"] == "success"


def test_rules_list_and_detail(mined):
    client, _, _ = mined
    rule = rule_by(client, "record_read", "cache_scope")
    assert rule["status"] == "proposed"
    detail = client.get(f"/learning/rules/{rule['rule_id']}").json()
    assert detail["rule"]["evidence"]["reasons"]
    assert [e["to_status"] for e in detail["events"]] == ["proposed"]
    assert detail["validations"] == []
    assert client.get("/learning/rules?status=active").json() == []
    assert client.get("/learning/rules?status=bogus").status_code == 422
    assert client.get("/learning/rules/" + "f" * 32).status_code == 404


def test_mining_again_does_not_duplicate(mined):
    client, eid, _ = mined
    again = client.post("/learning/minings", json={"evaluation_ids": [eid]}).json()
    assert again["report"]["created_rule_ids"] == []
    assert len(again["report"]["duplicates"]) == 2
    assert len(client.get("/learning/rules").json()) == 2


# ------------------------------------------------------------------ lifecycle
def test_full_lifecycle_scope_rule(mined):
    client, _, _ = mined
    rule = rule_by(client, "record_read", "cache_scope")
    rid = rule["rule_id"]

    # Approval before validation is refused.
    early = client.post(
        f"/learning/rules/{rid}/approve", json={"approved_by": "Navya", "confirm": True}
    )
    assert early.status_code == 409 and early.json()["error"]["code"] == "not_validated"

    created = client.post(f"/learning/rules/{rid}/validations", json={"repetitions": 1})
    assert created.status_code == 202
    v = wait_validation(client, created.json()["validation_id"])
    assert v["status"] == "passed", v["result"]["criteria"]
    result = v["result"]
    assert result["arms"]["on"]["tool_executions"] < result["arms"]["off"]["tool_executions"]
    assert result["regressions"] == [] and result["safety_violations"] == []
    assert result["shadow"]["would_reuse"] >= 1 and result["shadow"]["mismatched"] == 0
    assert all(t["off"]["correct"] for t in result["tasks"])
    assert {t["task_id"] for t in result["tasks"]} >= {"control_record_status"}
    # Validation runs are stored as normal runs (traceable), not in history evals.
    run_id = result["tasks"][0]["on"]["run_id"]
    assert client.get(f"/runs/{run_id}").json()["status"] == "success"

    assert client.get(f"/learning/rules/{rid}").json()["rule"]["status"] == "validated"
    # Still not active: needs a human.
    assert client.get("/health").json()["active_learned_rules"] == 0

    no_confirm = client.post(
        f"/learning/rules/{rid}/approve", json={"approved_by": "Navya", "confirm": False}
    )
    assert no_confirm.json()["error"]["code"] == "confirmation_required"
    blank = client.post(
        f"/learning/rules/{rid}/approve", json={"approved_by": "  ", "confirm": True}
    )
    assert blank.status_code == 422

    ok = client.post(
        f"/learning/rules/{rid}/approve",
        json={"approved_by": "Navya", "note": "evidence reviewed", "confirm": True},
    )
    assert ok.status_code == 200 and ok.json()["status"] == "active"
    assert ok.json()["approved_by"] == "Navya"
    twice = client.post(
        f"/learning/rules/{rid}/approve", json={"approved_by": "Navya", "confirm": True}
    )
    assert twice.status_code == 409

    tools = {t["name"]: t for t in client.get("/tools").json()["tools"]}
    assert tools["record_read"]["cache_scope"] == "run"  # developer setting unchanged
    assert tools["record_read"]["effective_cache_scope"] == "session"
    assert tools["record_read"]["learned_rules"][0]["rule_id"] == rid
    assert tools["send_notification"]["learning_eligible"] is False

    # Real runs in one session now share the read; another session does not.
    first = scenario_run(client, "s1")
    second = scenario_run(client, "s1")
    other = scenario_run(client, "s2")
    assert first["tool_execution_count"] == 1 and second["tool_execution_count"] == 0
    assert other["tool_execution_count"] == 1
    details = client.get(f"/runs/{second['run_id']}/details").json()
    assert details["tool_calls"][0]["learned_rules"] == [rid]
    assert details["run"]["config_snapshot"]["learned_rules_active"] == [rid]
    # Baseline is never affected.
    base = scenario_run(client, "s1", mode="passthrough")
    assert base["tool_execution_count"] == 1

    off = client.post(
        f"/learning/rules/{rid}/disable", json={"reason": "demo over", "disabled_by": "Navya"}
    )
    assert off.json()["status"] == "disabled"
    after = scenario_run(client, "s1")
    assert after["tool_execution_count"] == 1  # back to run-only sharing
    events = [e["to_status"] for e in client.get(f"/learning/rules/{rid}").json()["events"]]
    assert events == ["proposed", "validating", "validated", "active", "disabled"]


def scenario_run(client, session, mode="optimized"):
    body = {
        "scenario": {"task_id": "control_record_status", "version": "2"},
        "llm": "scripted",
        "mode": mode,
        "session_id": session,
    }
    run_id = client.post("/runs", json=body).json()["run_id"]
    run = wait_finished(client, run_id)
    assert run["status"] == "success", run
    return {**run, "tool_execution_count": run["metrics"]["tool_executions"]}


def test_shadow_mode_observes_without_changing_runs(mined):
    client, _, _ = mined
    rid = rule_by(client, "record_read", "cache_scope")["rule_id"]
    assert (
        client.post(f"/learning/rules/{rid}/shadow", json={"enabled": True}).json()["status"]
        == "shadow"
    )
    a = scenario_run(client, "sh")
    b = scenario_run(client, "sh")
    assert a["tool_execution_count"] == b["tool_execution_count"] == 1  # unchanged
    detail = client.get(f"/learning/rules/{rid}").json()
    assert detail["shadow_live"] == {"would_reuse": 1, "matched": 1, "mismatched": 0}
    calls = client.get(f"/runs/{b['run_id']}/details").json()["tool_calls"]
    assert calls[0]["shadow"]["rule_ids"] == [rid] and calls[0]["learned_rules"] is None
    stop = client.post(f"/learning/rules/{rid}/shadow", json={"enabled": False})
    assert stop.json()["status"] == "proposed"


def test_ttl_rule_validates_and_reduces_executions(mined):
    client, _, _ = mined
    rid = rule_by(client, "record_read", "ttl")["rule_id"]
    vid = client.post(f"/learning/rules/{rid}/validations", json={}).json()["validation_id"]
    v = wait_validation(client, vid)
    assert v["status"] == "passed", v["result"]["criteria"]
    followup = [t for t in v["result"]["tasks"] if t["task_id"] == "recheck_ticket_followup"]
    assert followup
    assert all(t["on"]["tool_executions"] < t["off"]["tool_executions"] for t in followup)
    # The read-after-write task stays correct with the longer TTL.
    raw = [t for t in v["result"]["tasks"] if t["task_id"] == "freshness_read_after_write"]
    assert raw and all(t["on"]["correct"] for t in raw)


def test_approval_requires_revalidation_when_active_rules_change(mined):
    client, _, _ = mined
    scope = rule_by(client, "record_read", "cache_scope")["rule_id"]
    ttl = rule_by(client, "record_read", "ttl")["rule_id"]
    for rid in (scope, ttl):
        vid = client.post(f"/learning/rules/{rid}/validations", json={"repetitions": 1}).json()[
            "validation_id"
        ]
        assert wait_validation(client, vid)["status"] == "passed"
    approve = {"approved_by": "Navya", "confirm": True}
    assert client.post(f"/learning/rules/{scope}/approve", json=approve).status_code == 200
    # ttl was validated WITHOUT the scope rule active -> must be validated again.
    stale = client.post(f"/learning/rules/{ttl}/approve", json=approve)
    assert stale.status_code == 409 and stale.json()["error"]["code"] == "revalidation_required"
    vid = client.post(f"/learning/rules/{ttl}/validations", json={"repetitions": 1}).json()[
        "validation_id"
    ]
    v = wait_validation(client, vid)
    assert v["base_rule_ids"] == [scope]
    assert v["status"] == "passed"
    assert client.post(f"/learning/rules/{ttl}/approve", json=approve).status_code == 200
    assert client.get("/health").json()["active_learned_rules"] == 2


def test_rules_and_validations_survive_restart(make_client):
    with make_client() as client:
        train_and_mine(client)
        rid = rule_by(client, "record_read", "cache_scope")["rule_id"]
        vid = client.post(f"/learning/rules/{rid}/validations", json={"repetitions": 1}).json()[
            "validation_id"
        ]
        wait_validation(client, vid)
        client.post(
            f"/learning/rules/{rid}/approve", json={"approved_by": "Navya", "confirm": True}
        )
        before = client.get(f"/learning/rules/{rid}").json()
        validation = client.get(f"/learning/validations/{vid}").json()
    with make_client() as restarted:
        assert restarted.get(f"/learning/rules/{rid}").json() == before
        assert restarted.get(f"/learning/validations/{vid}").json() == validation
        assert restarted.get("/health").json()["active_learned_rules"] == 1
        a = scenario_run(restarted, "r1")
        b = scenario_run(restarted, "r1")
        assert (a["tool_execution_count"], b["tool_execution_count"]) == (1, 0)
        assert len(restarted.get("/learning/minings").json()) == 1


def test_interrupted_validation_restores_rule_status(make_client, db_url):
    """A validation still 'running' when the server stopped (simulated by
    writing that state directly) is marked interrupted at the next startup,
    and the rule goes back to the status it had before."""
    from app.storage.db import Database
    from app.storage.learning import V_RUNNING, VALIDATING, LearningRepository
    from app.storage.models import RuleValidationRow

    with make_client() as client:
        train_and_mine(client)
        rid = rule_by(client, "record_read", "cache_scope")["rule_id"]
    db = Database(db_url)
    repo = LearningRepository(db)
    repo.transition(
        rid, to=VALIDATING, allowed_from=("proposed",), at="2026-01-01T00:00:00", actor="t"
    )
    repo.create_validation(
        RuleValidationRow(
            id="a" * 32,
            rule_id=rid,
            status=V_RUNNING,
            created_at="2026-01-01T00:00:00",
            previous_rule_status="proposed",
            config={},
            base_rule_ids=[],
        )
    )
    db.dispose()
    with make_client() as restarted:
        v = restarted.get("/learning/validations/" + "a" * 32).json()
        detail = restarted.get(f"/learning/rules/{rid}").json()
        assert v["status"] == "interrupted" and v["finished"]
        assert detail["rule"]["status"] == "proposed"
        assert detail["events"][-1]["note"] == "validation interrupted by a server restart"


def test_tampered_active_rule_for_unsafe_tool_is_disabled_at_startup(make_client, db_url):
    """Even a rule written straight into the database cannot make a
    side-effecting tool reusable: startup re-checks and disables it."""
    from app.gateway.factory import build_gateway
    from app.storage.db import Database
    from app.storage.learning import LearningRepository
    from app.storage.models import LearnedRuleRow

    with make_client() as client:
        version = {t["name"]: t for t in client.get("/tools").json()["tools"]}
        assert version  # app started once (schema created)
    db = Database(db_url)
    LearningRepository(db).create_rule(
        LearnedRuleRow(
            id="b" * 32,
            status="active",
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:00:00",
            tool_name="send_notification",
            tool_version=build_gateway().tool_spec("send_notification").version,
            kind="cache_scope",
            current_setting={},
            proposed_setting={"cache_scope": "session"},
            evidence={},
            dataset_name="core",
            dataset_version="2",
            approved_at="2026-01-01T00:00:00",
        )
    )
    db.dispose()
    with make_client() as restarted:
        rule = restarted.get("/learning/rules/" + "b" * 32).json()["rule"]
        assert rule["status"] == "disabled"
        assert "no longer allowed" in rule["status_reason"]
        assert "never reuses" in rule["status_reason"]  # the eligibility rule, not a version
        assert restarted.get("/health").json()["active_learned_rules"] == 0


def test_validation_and_actions_reject_bad_input(mined):
    client, _, _ = mined
    rid = rule_by(client, "record_read", "ttl")["rule_id"]
    base = f"/learning/rules/{rid}"
    assert client.post(f"{base}/validations", json={"repetitions": 0}).status_code == 422
    assert client.post(f"{base}/validations", json={"llm": "gemini"}).status_code == 503
    assert client.post(f"{base}/disable", json={"reason": " "}).status_code == 422
    assert client.post(f"{base}/shadow", json={}).status_code == 422
    missing = "/learning/rules/" + "0" * 32 + "/shadow"
    assert client.post(missing, json={"enabled": True}).status_code == 404
    # Turning shadow off is only valid for a rule that is in shadow.
    assert client.post(f"{base}/shadow", json={"enabled": False}).status_code == 409
    # Rule state unchanged by all of the above.
    assert client.get(f"/learning/rules/{rid}").json()["rule"]["status"] == "proposed"


def test_a_dismissed_rule_cannot_be_approved_even_after_passing(mined):
    client, _, _ = mined
    rid = rule_by(client, "record_read", "cache_scope")["rule_id"]
    vid = client.post(f"/learning/rules/{rid}/validations", json={"repetitions": 1}).json()[
        "validation_id"
    ]
    assert wait_validation(client, vid)["status"] == "passed"
    client.post(f"/learning/rules/{rid}/disable", json={"reason": "not now"})
    resp = client.post(
        f"/learning/rules/{rid}/approve", json={"approved_by": "Navya", "confirm": True}
    )
    assert resp.status_code == 409 and resp.json()["error"]["code"] == "not_validated"
    # Re-validating is the only way back.
    vid = client.post(f"/learning/rules/{rid}/validations", json={"repetitions": 1}).json()[
        "validation_id"
    ]
    assert wait_validation(client, vid)["status"] == "passed"
    ok = client.post(
        f"/learning/rules/{rid}/approve", json={"approved_by": "Navya", "confirm": True}
    )
    assert ok.status_code == 200
