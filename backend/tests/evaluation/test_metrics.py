"""Metric definitions, cost rules, statistics and report building."""

from __future__ import annotations

import pytest

from app.agent.runner import AgentRunner
from app.evaluation.metrics import (
    ModelPrice,
    Pricing,
    RunMetrics,
    llm_cost,
    median,
    pct_change,
    percentile,
    run_metrics,
    tool_cost,
)
from app.evaluation.report import EvalRunRecord, build_report, compare, summarize
from app.gateway.context import RunContext
from app.gateway.factory import build_gateway
from app.llm.base import TokenUsage
from app.llm.scripted import ScriptedLLM

S = ScriptedLLM

# ------------------------------------------------------------------ statistics


def test_percentile_nearest_rank():
    values = [float(v) for v in range(1, 21)]  # 1..20
    assert percentile(values, 50) == 10
    assert percentile(values, 95) == 19
    assert percentile(values, 100) == 20
    assert percentile([5.0], 95) == 5.0
    # Sizes where nearest-rank differs from rounding or interpolation:
    five = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(five, 50) == 30.0  # rank ceil(2.5)=3, not 2
    assert percentile(five, 90) == 50.0  # rank ceil(4.5)=5, not 4
    assert percentile([1.0, 100.0], 50) == 1.0  # no averaging between values
    assert percentile([], 95) is None
    with pytest.raises(ValueError):
        percentile(values, 0)


def test_median_and_pct_change():
    assert median([3.0, 1.0, 2.0]) == 2.0 and median([]) is None
    assert pct_change(100, 75) == -25.0
    assert pct_change(0, 5) is None and pct_change(None, 5) is None


# ------------------------------------------------------------------ per-run metrics


async def make_result(mode="optimized", usage=None, model_id="m-test"):
    gw = build_gateway(mode=mode)
    llm = S(
        [
            S.tool_calls(("kb_search", {"query": "x"}), usage=usage),
            S.tool_calls(("kb_search", {"query": "x"}), usage=usage),
            S.tool_calls(("calculator", {"expression": "1+1"}), usage=usage),
            S.final("done", usage=usage),
        ],
        model_id=model_id,
    )
    return await AgentRunner(llm, gw).run("t", RunContext(session_id="s"))


async def test_requests_vs_executions_and_overhead():
    result = await make_result("optimized")
    m = run_metrics(result)
    assert (m.tool_requests, m.tool_executions, m.reused_in_run, m.cache_hits) == (3, 2, 1, 0)
    assert m.llm_calls == 4 and m.tool_retries == 0
    expected_overhead = sum(r.duration_ms - (r.tool_duration_ms or 0) for r in result.tool_results)
    assert m.gateway_overhead_ms == pytest.approx(expected_overhead)
    assert m.gateway_overhead_ms >= 0
    assert m.input_tokens is None and m.llm_cost is None  # no usage reported
    base = run_metrics(await make_result("passthrough"))
    assert (base.tool_requests, base.tool_executions) == (3, 3)


PRICING = Pricing(
    source="test fixture (not real prices)",
    retrieved_on="2026-01-01",
    models={
        "m-test": ModelPrice(
            input_per_million=1.0, output_per_million=4.0, thinking_billed_as_output=True
        ),
        "m-nothinking-price": ModelPrice(input_per_million=1.0, output_per_million=4.0),
    },
    tool_per_execution={"kb_search": 0.001, "calculator": 0.0},
)


async def test_cost_only_with_pricing_and_complete_usage():
    u = TokenUsage(input_tokens=1000, output_tokens=100, thinking_tokens=50, total_tokens=1150)
    result = await make_result(usage=u)
    # 4 calls: input 4000 * 1 + output 400 * 4 + thinking 200 * 4 (billed as output)
    assert llm_cost(PRICING, result) == pytest.approx((4000 + 1600 + 800) / 1_000_000)
    assert tool_cost(PRICING, result) == pytest.approx(0.001)  # reused call is not charged
    assert llm_cost(None, result) is None
    other = await make_result(usage=u, model_id="unknown-model")
    assert llm_cost(PRICING, other) is None
    no_thinking_price = await make_result(usage=u, model_id="m-nothinking-price")
    assert llm_cost(PRICING, no_thinking_price) is None  # thinking billing unknown
    partial = await make_result(usage=TokenUsage(input_tokens=10))
    assert llm_cost(PRICING, partial) is None
    unpriced_tool = PRICING.model_copy(update={"tool_per_execution": {"kb_search": 0.001}})
    assert tool_cost(unpriced_tool, result) is None  # calculator ran but has no price


# ------------------------------------------------------------------ summaries & reports


def metrics(**overrides) -> RunMetrics:
    base = {
        "status": "success",
        "tool_requests": 4,
        "tool_executions": 4,
        "cache_hits": 0,
        "reused_in_run": 0,
        "shared_in_flight": 0,
        "tool_retries": 0,
        "llm_calls": 3,
        "input_tokens": None,
        "output_tokens": None,
        "thinking_tokens": None,
        "total_tokens": None,
        "usage_complete": False,
        "latency_ms": 10.0,
        "llm_time_ms": 1.0,
        "tool_time_ms": 1.0,
        "gateway_overhead_ms": 0.5,
        "llm_cost": None,
        "tool_cost": None,
    }
    base.update(overrides)
    return RunMetrics(**base)


def record(task, mode, *, rep=1, correct=True, phase="cold", **m) -> EvalRunRecord:
    return EvalRunRecord(
        seq=0,
        task_id=task,
        category="control",
        split="train",
        mode=mode,
        phase=phase,
        repetition=rep,
        run_id=f"{task}-{mode}-{rep}",
        correct=correct,
        failed_checks=[] if correct else ["number:1"],
        grade_checks=[],
        final_answer="x",
        metrics=metrics(**m),
    )


def test_summarize():
    rows = [
        record("a", "optimized", latency_ms=10.0, tool_executions=2),
        record("b", "optimized", latency_ms=30.0, tool_executions=1, correct=False),
        record("c", "optimized", latency_ms=20.0, tool_executions=3),
    ]
    s = summarize(rows)
    assert s["runs"] == 3 and s["correct"] == 2 and s["accuracy"] == pytest.approx(2 / 3)
    assert s["tool_executions"] == 6 and s["tool_requests"] == 12
    assert s["latency_ms"] == {"median": 20.0, "p95": 30.0, "min": 10.0, "max": 30.0}
    assert s["tokens"]["total"] is None and s["tokens"]["complete"] is False
    assert s["llm_cost"] is None


def test_costs_total_only_when_every_run_has_one():
    rows = [
        record("a", "optimized", llm_cost=0.5, tool_cost=0.1),
        record("b", "optimized", llm_cost=None, tool_cost=0.1),
    ]
    assert summarize(rows)["llm_cost"] is None
    assert summarize(rows)["tool_cost"] == pytest.approx(0.2)


def test_compare():
    base = summarize([record("a", "passthrough", tool_executions=4, latency_ms=20.0)])
    opt = summarize([record("a", "optimized", tool_executions=2, latency_ms=15.0)])
    c = compare(base, opt)
    assert c["tool_executions_change_pct"] == -50.0
    assert c["latency_median_change_pct"] == -25.0
    assert c["accuracy_change"] == 0.0
    assert c["tokens_change_pct"] is None and c["cost_change_pct"] is None


def test_report_lists_every_regression_and_improvement():
    rows = [
        record("t1", "passthrough", rep=1),
        record("t1", "optimized", rep=1, correct=False),
        record("t1", "passthrough", rep=2),
        record("t1", "optimized", rep=2),
        record("t2", "passthrough", rep=1, correct=False),
        record("t2", "optimized", rep=1),
        record("t3", "passthrough", rep=1, phase="warm"),
        record("t3", "optimized", rep=1, phase="warm", correct=False),
    ]
    report = build_report(rows, meta={"label": "x"}, skipped=[])
    assert [(r["task_id"], r["phase"], r["repetition"]) for r in report["regressions"]] == [
        ("t1", "cold", 1),
        ("t3", "warm", 1),
    ]
    assert report["regressions"][0]["optimized_run_id"] == "t1-optimized-1"
    assert [(r["task_id"], r["repetition"]) for r in report["improvements"]] == [("t2", 1)]
    assert set(report["phases"]) == {"cold", "warm"}  # never mixed
    t1 = next(p for p in report["per_task"] if p["task_id"] == "t1")
    assert t1["optimized"]["correct"] == 1 and t1["optimized"]["runs"] == 2
