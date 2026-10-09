"""Reproducible end-to-end demo with the MOCK (scripted) agent. No API key needed.

    # in-process, on a throw-away database:
    .venv/bin/python scripts/demo.py

    # against the running API server, so every run also appears in the dashboard:
    .venv/bin/python scripts/demo.py --base-url http://127.0.0.1:8000 --keep-rule

It drives the real HTTP API, exactly like the dashboard does:

1. Redundant lookups: the same task in baseline and optimized mode, each in a
   new session (empty cache). Same answer; fewer real tool executions.
2. Cases where reuse would be WRONG, and what the gateway does instead:
   a price that changed (stale result refreshed), a read after a write
   (cached read invalidated), a repeated notification (side effect executed
   every time), and a model stuck repeating itself (stopped by the loop guard).
3. An evaluation of every core task, both modes, cold and warm cache.
4. Learning: training evidence -> mining -> held-out validation -> human
   approval -> the rule's effect in real runs -> disabling it.

Every number printed comes from the API. These are MOCK results: a scripted
agent replays fixed tool calls, so they show what the optimizer does to tool
calls - not what a real Gemini agent would save.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MOCK_NOTE = (
    "MOCK results: scripted agent, deterministic mock tools. They show what the gateway does "
    "with tool calls; they do not measure real Gemini savings, cost or answer quality."
)


class Api:
    def __init__(self, client: Any) -> None:
        self.c = client

    def get(self, path: str) -> Any:
        r = self.c.get(path)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, body: dict) -> Any:
        r = self.c.post(path, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text}")
        return r.json()

    def wait(self, path: str, done=lambda b: b.get("finished"), timeout: float = 300) -> Any:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            body = self.get(path)
            if done(body):
                return body
            time.sleep(0.05)
        raise TimeoutError(path)

    def scenario(self, task_id: str, mode: str, session: str, version: str = "1") -> dict:
        body = {
            "scenario": {"task_id": task_id, "version": version},
            "llm": "scripted",
            "mode": mode,
            "session_id": session,
        }
        run_id = self.post("/runs", body)["run_id"]
        run = self.wait(f"/runs/{run_id}")
        calls = self.get(f"/runs/{run_id}/details")["tool_calls"]
        return {"run": run, "calls": calls}


def line(char: str = "-") -> None:
    print(char * 78)


def decisions(calls: list[dict]) -> str:
    return ", ".join(f"{c['tool_name']}:{c['decision']}" for c in calls)


def compare(api: Api, task_id: str, tag: str, version: str = "1") -> dict:
    base = api.scenario(task_id, "passthrough", f"demo-{tag}-base", version)
    opt = api.scenario(task_id, "optimized", f"demo-{tag}-opt", version)
    b, o = base["run"], opt["run"]
    same = b["final_answer"] == o["final_answer"]
    print(f"\n{task_id}")
    for label, r in (("baseline ", b), ("optimized", o)):
        m = r["metrics"]
        print(
            f"  {label}: status={r['status']:<15} requests={m['tool_requests']} "
            f"executions={m['tool_executions']} reused={m['reused_results']} "
            f"time={m['duration_ms']:.1f} ms"
        )
    if o["final_answer"] is None and b["final_answer"] is None:
        print(f"  no final answer in either mode (both ended with status {o['status']})")
    else:
        print(f"  answers identical: {same}   answer: {o['final_answer']!r}")
    print(f"  optimized decisions: {decisions(opt['calls'])}")
    print(f"  dashboard: /compare?base={b['run_id']}&opt={o['run_id']}")
    return {
        "task_id": task_id,
        "answers_identical": same,
        "baseline": {k: b["metrics"][k] for k in ("tool_requests", "tool_executions")},
        "optimized": {k: o["metrics"][k] for k in ("tool_requests", "tool_executions")},
        "baseline_status": b["status"],
        "optimized_status": o["status"],
        "optimized_decisions": [c["decision"] for c in opt["calls"]],
        "baseline_run_id": b["run_id"],
        "optimized_run_id": o["run_id"],
    }


def section_redundant(api: Api) -> list[dict]:
    line("=")
    print("1. Redundant lookups: same answer, fewer real tool executions")
    line("=")
    return [
        compare(api, t, f"r{i}")
        for i, t in enumerate(
            ["redundant_orbit_price", "redundant_support_hours", "redundant_keystone_total"]
        )
    ]


def section_safety(api: Api) -> list[dict]:
    line("=")
    print("2. When reuse would be wrong - the gateway does NOT reuse")
    line("=")
    notes = {
        "freshness_price_changed": "price changed after 400 s (TTL 60 s): stored price is "
        "stale, so it is refreshed (stale_refreshed) and the answer uses the new price",
        "freshness_read_after_write": "record_update invalidates the stored read, so the second "
        "read executes and sees the new status",
        "side_effect_double_reminder": "a notification is a side effect: both requests execute "
        "(2 messages sent), never merged into one",
        "failure_loop": "the model repeats the same search; the identical-request guard stops "
        "the run in BOTH modes",
    }
    out = []
    for i, (task, why) in enumerate(notes.items()):
        result = compare(api, task, f"s{i}")
        print(f"  why: {why}")
        out.append({**result, "why": why})
    return out


def section_evaluation(api: Api, reps: int) -> dict:
    line("=")
    print(f"3. Evaluation: every core v1 task, both modes, cold + warm cache, {reps} repetitions")
    line("=")
    created = api.post("/evaluations", {"repetitions": reps, "warm_cache": True})
    ev = api.wait(f"/evaluations/{created['evaluation_id']}", timeout=600)
    report = ev["report"]
    print(f"  {report['label']}")
    summary: dict[str, Any] = {"evaluation_id": ev["evaluation_id"], "phases": {}}
    for phase, data in report["phases"].items():
        print(f"  {phase} cache:")
        for mode, m in data["modes"].items():
            print(
                f"    {mode:<11} runs={m['runs']:<3} accuracy={m['accuracy']:.0%}  "
                f"requests={m['tool_requests']:<4} executions={m['tool_executions']:<4} "
                f"latency p50={m['latency_ms']['median']:.1f} ms "
                f"p95={m['latency_ms']['p95']:.1f} ms"
            )
        cmp = data.get("comparison") or {}
        change = cmp.get("tool_executions_change_pct")
        acc = cmp.get("accuracy_change")
        print(
            f"    optimized vs baseline: executions "
            f"{'n/a' if change is None else f'{change:+.1f}%'}, accuracy "
            f"{'n/a' if acc is None else f'{acc * 100:+.1f} points'}"
        )
        summary["phases"][phase] = {
            mode: {k: m[k] for k in ("runs", "accuracy", "tool_requests", "tool_executions")}
            for mode, m in data["modes"].items()
        }
    print(f"  regressions (correct in baseline, wrong in optimized): {len(report['regressions'])}")
    print("  cost: not reported (the mock agent reports no token usage; no pricing file)")
    print(f"  dashboard: /evaluations/{ev['evaluation_id']}")
    summary["regressions"] = len(report["regressions"])
    return summary


def section_learning(api: Api, keep_rule: bool) -> dict:
    line("=")
    print("4. Learning: evidence -> proposal -> held-out validation -> approval -> effect")
    line("=")
    created = api.post(
        "/evaluations",
        {"dataset_version": "2", "split": "train", "warm_cache": True, "repetitions": 2},
    )
    api.wait(f"/evaluations/{created['evaluation_id']}", timeout=600)
    mining = api.post(
        "/learning/minings",
        {"dataset_version": "2", "evaluation_ids": [created["evaluation_id"]]},
    )
    report = mining["report"]
    print(f"  training runs used as evidence: {report['runs_used']} of {report['runs_examined']}")
    for c in report["candidates"]:
        verdict = f"PROPOSE {c['proposed_setting']}" if c["proposed"] else "no proposal"
        print(f"    {c['tool_name']:<13} {c['kind']:<12} {verdict:<38} {c['reasons'][0]}")
    for t in report["not_eligible"]:
        print(f"    {t['tool_name']:<13} never learned: {t['reason']}")

    rules = api.get("/learning/rules?status=proposed")
    rule = next(r for r in rules if r["tool_name"] == "record_read" and r["kind"] == "cache_scope")
    ev = rule["evidence"]
    print(
        f"  rule {rule['rule_id'][:8]}: record_read sharing run -> session; "
        f"{ev['repeat_count']} supporting repeats in {ev['distinct_runs']} sessions, "
        f"consistency {ev['consistency']:.0%}"
    )
    vid = api.post(f"/learning/rules/{rule['rule_id']}/validations", {"repetitions": 2})[
        "validation_id"
    ]
    v = api.wait(f"/learning/validations/{vid}", timeout=600)
    res = v["result"]
    print(f"  held-out validation: {v['status'].upper()}")
    for c in res["criteria"]:
        print(f"    [{'x' if c['passed'] else ' '}] {c['label']} ({c['detail']})")

    before = api.scenario("control_record_status", "optimized", "demo-learn-before", "2")
    before2 = api.scenario("control_record_status", "optimized", "demo-learn-before", "2")
    api.post(
        f"/learning/rules/{rule['rule_id']}/approve",
        {"approved_by": "demo script", "note": "demo approval", "confirm": True},
    )
    after = api.scenario("control_record_status", "optimized", "demo-learn-after", "2")
    after2 = api.scenario("control_record_status", "optimized", "demo-learn-after", "2")
    ex = lambda r: r["run"]["metrics"]["tool_executions"]  # noqa: E731
    print(
        f"  two runs in one session, before approval: executions {ex(before)} + {ex(before2)}; "
        f"after approval: {ex(after)} + {ex(after2)}"
    )
    print(
        f"  second run after approval: {decisions(after2['calls'])} "
        f"(learned rule {after2['calls'][0]['learned_rules']})"
    )
    if not keep_rule:
        api.post(f"/learning/rules/{rule['rule_id']}/disable", {"reason": "demo finished"})
        print("  rule disabled again (use --keep-rule to leave it active for the dashboard)")
    print(f"  dashboard: /learning/rules/{rule['rule_id']}")
    return {
        "rule_id": rule["rule_id"],
        "validation": v["status"],
        "arms": {k: a["tool_executions"] for k, a in res["arms"].items()},
        "executions_before_approval": [ex(before), ex(before2)],
        "executions_after_approval": [ex(after), ex(after2)],
        "kept_active": keep_rule,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-url", help="use a running API server instead of in-process")
    parser.add_argument("--reps", type=int, default=3, help="evaluation repetitions")
    parser.add_argument("--keep-rule", action="store_true", help="leave the learned rule active")
    parser.add_argument("--json", type=Path, help="also write a JSON summary to this file")
    args = parser.parse_args(argv)

    print(MOCK_NOTE)

    def run(api: Api) -> dict:
        health = api.get("/health")
        if health["status"] != "ok":
            raise RuntimeError(f"backend not healthy: {health}")
        if health.get("active_learned_rules"):
            print(
                f"NOTE: {health['active_learned_rules']} learned rule(s) already active; "
                "optimized numbers below include them."
            )
        return {
            "redundant": section_redundant(api),
            "safety": section_safety(api),
            "evaluation": section_evaluation(api, args.reps),
            "learning": section_learning(api, args.keep_rule),
        }

    if args.base_url:
        import httpx

        with httpx.Client(base_url=args.base_url, timeout=30) as client:
            summary = run(Api(client))
    else:
        from fastapi.testclient import TestClient

        from app.api.main import create_app
        from app.config import Settings

        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(database_url=f"sqlite:///{tmp}/demo.db")
            with TestClient(create_app(settings)) as client:
                summary = run(Api(client))

    line("=")
    print(MOCK_NOTE)
    if args.json:
        args.json.write_text(json.dumps({"note": MOCK_NOTE, **summary}, indent=2) + "\n")
        print(f"summary written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
