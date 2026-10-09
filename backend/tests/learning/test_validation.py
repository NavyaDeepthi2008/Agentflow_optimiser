"""Held-out validation: every rejection criterion, on real runs where possible."""

from __future__ import annotations

from types import SimpleNamespace

from app.evaluation.dataset import Dataset, EvalTask, load_dataset
from app.evaluation.runner import EvaluationRunner
from app.gateway.factory import build_gateway
from app.learning.models import ValidationConfig
from app.learning.validation import judge, run_validation
from app.optimizer.learned import RuleKind, ToolOverride
from app.tools.builtin.prices import price_at
from app.tools.spec import CacheScope

GW = build_gateway(include_test_tools=True)
CORE2 = load_dataset("core", "2")


def override(tool, kind, value, rule_id="cand"):
    v = GW.tool_spec(tool).version
    if kind is RuleKind.TTL:
        return ToolOverride(rule_id, tool, v, kind, ttl_seconds=value)
    return ToolOverride(rule_id, tool, v, kind, cache_scope=value)


async def validate(rule, dataset=CORE2, base=(), reps=1):
    return await run_validation(
        EvaluationRunner(),
        dataset,
        rule,
        base,
        ValidationConfig(repetitions=reps),
        validation_id="v" * 32,
        spec_of=GW.tool_spec,
    )


def crit(result, name):
    return next(c for c in result["criteria"] if c["name"] == name)


# ------------------------------------------------------------------ pass
async def test_good_rules_pass_on_held_out_tasks():
    for rule in (
        override("record_read", RuleKind.TTL, 600.0),
        override("record_read", RuleKind.CACHE_SCOPE, CacheScope.SESSION),
    ):
        result = await validate(rule)
        assert result["passed"], result["criteria"]
        assert result["arms"]["on"]["tool_executions"] < result["arms"]["off"]["tool_executions"]
        assert result["arms"]["on"]["rule_applied"] >= 1
        assert result["arms"]["off"]["rule_applied"] == 0  # shadow never acts
        # Only held-out tasks are used.
        heldout = {t.id for t in CORE2.tasks if t.split == "heldout"}
        assert {t["task_id"] for t in result["tasks"]} <= heldout


async def test_arms_differ_only_by_the_rule():
    result = await validate(override("record_read", RuleKind.TTL, 600.0), reps=2)
    for row in result["tasks"]:
        if row["rule_applied"] == 0:
            assert row["on"]["tool_executions"] == row["off"]["tool_executions"]
            assert row["on"]["final_answer"] == row["off"]["final_answer"]


# ------------------------------------------------------------------ reject
async def test_rule_without_measurable_savings_is_rejected():
    # kb_search repeats on held-out tasks are already reused within the default TTL.
    result = await validate(override("kb_search", RuleKind.TTL, 7200.0))
    assert not result["passed"]
    assert result["failed_criteria"] == ["executions_decreased"]


def stale_price_dataset() -> Dataset:
    """Price is read at t=150 s and again at t=350 s: 200 s apart, but across a
    price change at 300 s. A 300 s TTL would answer with the old price."""
    task = {
        "id": "price_across_tick",
        "category": "freshness",
        "split": "heldout",
        "scripted_only": True,
        "task": "What is KST-X now? Check, then check again a few minutes later.",
        "tools": ["price_lookup"],
        "script": [
            {
                "advance_seconds": 150,
                "tool_calls": [{"name": "price_lookup", "args": {"symbol": "KST-X"}}],
            },
            {
                "advance_seconds": 200,
                "tool_calls": [{"name": "price_lookup", "args": {"symbol": "KST-X"}}],
            },
            {
                "answer_template": "KST-X is {p} USD.",
                "values": {"p": {"tool": "price_lookup", "path": "result.price"}},
            },
        ],
        "expected": {"answer": {"numbers": [{"value": price_at("KST-X", 1), "tolerance": 0.001}]}},
    }
    control = next(t for t in CORE2.tasks if t.id == "control_record_status")
    return Dataset(name="test", version="1", tasks=[EvalTask.model_validate(task), control])


async def test_rule_that_breaks_a_correct_answer_is_rejected():
    assert price_at("KST-X", 0) != price_at("KST-X", 1)
    rule = override("price_lookup", RuleKind.TTL, 300.0)  # within the tool's ceiling!
    result = await validate(rule, dataset=stale_price_dataset())
    assert not result["passed"]
    failed = set(result["failed_criteria"])
    assert {"no_correct_to_incorrect", "accuracy_not_lower", "no_stale_data_in_shadow"} <= failed
    assert result["regressions"][0]["task_id"] == "price_across_tick"
    assert result["shadow"]["mismatched"] >= 1
    # It DID save executions - savings never outweigh a wrong answer.
    assert crit(result, "executions_decreased")["passed"]


async def test_scope_rule_on_tool_held_out_tasks_never_repeat_is_rejected():
    ds = Dataset(
        name="t",
        version="1",
        tasks=[
            t for t in CORE2.tasks if t.id in ("failure_missing_document", "control_flight_class")
        ],
    )
    result = await validate(override("record_read", RuleKind.CACHE_SCOPE, CacheScope.SESSION), ds)
    assert not result["passed"] and "executions_decreased" in result["failed_criteria"]
    assert result["arms"]["on"]["rule_applied"] == 0


async def test_side_effect_tasks_execute_identically_in_both_arms():
    result = await validate(override("record_read", RuleKind.TTL, 900.0), reps=1)
    by_task = {(t["task_id"], t["phase"]): t for t in result["tasks"]}
    for key in [("side_effect_close_and_notify", "cold"), ("freshness_read_after_write", "cold")]:
        row = by_task[key]
        assert row["on"]["correct"] and row["off"]["correct"]
    assert result["safety_violations"] == []


# ------------------------------------------------------------------ judge (synthetic)
def fake(arm, task_id, correct, calls, failed=()):
    planned = SimpleNamespace(
        task=SimpleNamespace(id=task_id, category="control"), phase="cold", repetition=1
    )
    metrics = SimpleNamespace(
        status="success",
        tool_executions=sum(c.executions for c in calls),
        tool_requests=len(calls),
        cache_hits=0,
        reused_in_run=0,
        shared_in_flight=0,
    )
    record = SimpleNamespace(
        correct=correct,
        failed_checks=list(failed),
        metrics=metrics,
        run_id=f"{arm}-{task_id}",
        warmup_run_id=None,
        final_answer="x",
    )
    ar = SimpleNamespace(
        arm=arm,
        planned=planned,
        record=record,
        results={"measured": SimpleNamespace(tool_results=calls)},
    )
    return ar


def tc(tool, executions=1, decision="executed_live", learned=(), shadow=None):
    return SimpleNamespace(
        tool_name=tool,
        executions=executions,
        decision=decision,
        learned_rules=list(learned),
        shadow=shadow,
    )


def run_judge(off_calls, on_calls, **kw):
    rule = override("record_read", RuleKind.TTL, 600.0)
    arms = [
        fake("off", "t1", True, off_calls),
        fake("on", "t1", kw.pop("on_correct", True), on_calls, kw.pop("failed", ())),
    ]
    from app.evaluation.runner import EvaluationConfig

    return judge(arms, rule, GW.tool_spec, EvaluationConfig(), [])


def test_judge_rejects_a_change_in_side_effect_executions():
    r = run_judge(
        [tc("record_read"), tc("record_read"), tc("send_notification")],
        [tc("record_read"), tc("record_read", 0, "reused_in_run", ["cand"])],
    )
    assert "no_safety_violation" in r["failed_criteria"]
    assert "side-effecting tools executed 0 times" in crit(r, "no_safety_violation")["detail"]


def test_judge_rejects_reuse_of_an_ineligible_tool():
    r = run_judge(
        [tc("send_notification")],
        [tc("send_notification"), tc("send_notification", 0, "cache_hit")],
    )
    assert "no_safety_violation" in r["failed_criteria"]


def test_judge_rejects_world_check_failure():
    r = run_judge(
        [tc("record_read"), tc("record_read")],
        [tc("record_read"), tc("record_read", 0, "reused_in_run", ["cand"])],
        failed=["world:outbox_count"],
    )
    assert "no_safety_violation" in r["failed_criteria"]


def test_judge_rejects_accuracy_drop_and_flip():
    r = run_judge(
        [tc("record_read"), tc("record_read")],
        [tc("record_read"), tc("record_read", 0, "reused_in_run", ["cand"])],
        on_correct=False,
    )
    assert {"no_correct_to_incorrect", "accuracy_not_lower"} <= set(r["failed_criteria"])


def test_judge_passes_a_clean_saving():
    r = run_judge(
        [tc("record_read"), tc("record_read")],
        [tc("record_read"), tc("record_read", 0, "reused_in_run", ["cand"])],
    )
    assert r["passed"], r["criteria"]
