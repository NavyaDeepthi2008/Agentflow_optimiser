"""The documented demo (scripts/demo.py) runs end to end and shows what it claims."""

from __future__ import annotations

import json

from scripts.demo import main


def test_demo_runs_and_its_claims_hold(tmp_path, capsys):
    out = tmp_path / "demo.json"
    assert main(["--reps", "1", "--json", str(out)]) == 0
    summary = json.loads(out.read_text())
    assert "MOCK" in summary["note"]

    for case in summary["redundant"]:
        assert case["answers_identical"]
        assert case["baseline"]["tool_requests"] == case["optimized"]["tool_requests"]
        assert case["optimized"]["tool_executions"] < case["baseline"]["tool_executions"]

    safety = {c["task_id"]: c for c in summary["safety"]}
    assert all(c["answers_identical"] for c in safety.values())
    assert "stale_refreshed" in safety["freshness_price_changed"]["optimized_decisions"]
    for task in (
        "freshness_price_changed",
        "freshness_read_after_write",
        "side_effect_double_reminder",
    ):
        c = safety[task]
        assert c["optimized"]["tool_executions"] == c["baseline"]["tool_executions"]
    loop = safety["failure_loop"]
    assert loop["baseline_status"] == loop["optimized_status"] == "loop_detected"

    assert summary["evaluation"]["regressions"] == 0
    learning = summary["learning"]
    assert learning["validation"] == "passed"
    assert learning["arms"]["on"] < learning["arms"]["off"]
    assert learning["executions_before_approval"] == [1, 1]
    assert learning["executions_after_approval"] == [1, 0]
    assert "MOCK results" in capsys.readouterr().out
