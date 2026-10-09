"""Build the evaluation report from per-run records.

The report never mixes:
* cold-cache and warm-cache runs (separate ``phases``),
* scripted (mock) and real-model results (one provider per evaluation, and
  the report's ``label`` says which).

Accuracy changes are reported per task and repetition: a **regression** is a
(task, phase, repetition) where the baseline run was correct and the optimized
run was not. Every regression is listed - none are averaged away.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from pydantic import BaseModel

from app.evaluation.metrics import RunMetrics, median, pct_change, percentile

BASELINE, OPTIMIZED = "passthrough", "optimized"


class EvalRunRecord(BaseModel):
    seq: int
    task_id: str
    category: str
    split: str
    mode: str
    phase: str  # cold | warm
    repetition: int
    run_id: str
    correct: bool
    failed_checks: list[str]
    grade_checks: list[dict[str, Any]]
    final_answer: str | None
    metrics: RunMetrics
    # Warm phase only: the warm-up run in the same session and its grade.
    warmup_run_id: str | None = None
    warmup_correct: bool | None = None


def _sum_optional(values: list[int | float | None]) -> tuple[float | None, int]:
    present = [v for v in values if v is not None]
    return (sum(present) if present else None), len(present)


def summarize(records: list[EvalRunRecord]) -> dict[str, Any]:
    n = len(records)
    m = [r.metrics for r in records]
    latencies = [x.latency_ms for x in m]
    overheads = [x.gateway_overhead_ms for x in m]
    tokens_total, with_usage = _sum_optional([x.total_tokens for x in m])
    llm_costs = [x.llm_cost for x in m]
    tool_costs = [x.tool_cost for x in m]
    correct = sum(r.correct for r in records)
    return {
        "runs": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "status_counts": dict(Counter(x.status for x in m)),
        "tool_requests": sum(x.tool_requests for x in m),
        "tool_executions": sum(x.tool_executions for x in m),
        "cache_hits": sum(x.cache_hits for x in m),
        "reused_in_run": sum(x.reused_in_run for x in m),
        "shared_in_flight": sum(x.shared_in_flight for x in m),
        "tool_retries": sum(x.tool_retries for x in m),
        "llm_calls": sum(x.llm_calls for x in m),
        "tokens": {
            "total": tokens_total,
            "input": _sum_optional([x.input_tokens for x in m])[0],
            "output": _sum_optional([x.output_tokens for x in m])[0],
            "thinking": _sum_optional([x.thinking_tokens for x in m])[0],
            "runs_with_usage": with_usage,
            "complete": n > 0 and all(x.usage_complete for x in m),
        },
        "latency_ms": {
            "median": median(latencies),
            "p95": percentile(latencies, 95),
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "gateway_overhead_ms": {"median": median(overheads), "p95": percentile(overheads, 95)},
        # Cost totals only when EVERY run has a cost; otherwise unavailable.
        "llm_cost": sum(llm_costs) if n and all(c is not None for c in llm_costs) else None,
        "tool_cost": sum(tool_costs) if n and all(c is not None for c in tool_costs) else None,
    }


def compare(base: dict[str, Any], opt: dict[str, Any]) -> dict[str, Any]:
    def total_cost(s: dict[str, Any]) -> float | None:
        if s["llm_cost"] is None or s["tool_cost"] is None:
            return None
        return s["llm_cost"] + s["tool_cost"]

    both_complete = base["tokens"]["complete"] and opt["tokens"]["complete"]
    return {
        "accuracy_change": (
            opt["accuracy"] - base["accuracy"]
            if base["accuracy"] is not None and opt["accuracy"] is not None
            else None
        ),
        "tool_requests_change_pct": pct_change(base["tool_requests"], opt["tool_requests"]),
        "tool_executions_change_pct": pct_change(base["tool_executions"], opt["tool_executions"]),
        "llm_calls_change_pct": pct_change(base["llm_calls"], opt["llm_calls"]),
        "tokens_change_pct": (
            pct_change(base["tokens"]["total"], opt["tokens"]["total"]) if both_complete else None
        ),
        "latency_median_change_pct": pct_change(
            base["latency_ms"]["median"], opt["latency_ms"]["median"]
        ),
        "latency_p95_change_pct": pct_change(base["latency_ms"]["p95"], opt["latency_ms"]["p95"]),
        "cost_change_pct": pct_change(total_cost(base), total_cost(opt)),
    }


def build_report(
    records: list[EvalRunRecord],
    *,
    meta: dict[str, Any],
    skipped: list[dict[str, str]],
) -> dict[str, Any]:
    phases: dict[str, Any] = {}
    by_phase: dict[str, list[EvalRunRecord]] = defaultdict(list)
    for r in records:
        by_phase[r.phase].append(r)

    for phase, rows in sorted(by_phase.items()):
        modes = {
            mode: summarize([r for r in rows if r.mode == mode])
            for mode in sorted({r.mode for r in rows})
        }
        entry: dict[str, Any] = {"modes": modes}
        if BASELINE in modes and OPTIMIZED in modes:
            entry["comparison"] = compare(modes[BASELINE], modes[OPTIMIZED])
        categories: dict[str, dict[str, Any]] = defaultdict(dict)
        for category in sorted({r.category for r in rows}):
            for mode in modes:
                sel = [r for r in rows if r.category == category and r.mode == mode]
                categories[category][mode] = {
                    "runs": len(sel),
                    "accuracy": sum(r.correct for r in sel) / len(sel) if sel else None,
                    "tool_executions": sum(r.metrics.tool_executions for r in sel),
                    "tool_requests": sum(r.metrics.tool_requests for r in sel),
                }
        entry["by_category"] = dict(categories)
        phases[phase] = entry

    # Pair runs by (task, phase, repetition) to find accuracy changes.
    pairs: dict[tuple[str, str, int], dict[str, EvalRunRecord]] = defaultdict(dict)
    for r in records:
        pairs[(r.task_id, r.phase, r.repetition)][r.mode] = r
    regressions, improvements = [], []
    for (task_id, phase, rep), modes in sorted(pairs.items()):
        if BASELINE not in modes or OPTIMIZED not in modes:
            continue
        b, o = modes[BASELINE], modes[OPTIMIZED]
        item = {
            "task_id": task_id,
            "phase": phase,
            "repetition": rep,
            "baseline_run_id": b.run_id,
            "optimized_run_id": o.run_id,
        }
        if b.correct and not o.correct:
            regressions.append({**item, "optimized_failed_checks": o.failed_checks})
        elif o.correct and not b.correct:
            improvements.append({**item, "baseline_failed_checks": b.failed_checks})

    per_task = []
    for task_id in sorted({r.task_id for r in records}):
        for phase in sorted({r.phase for r in records if r.task_id == task_id}):
            row: dict[str, Any] = {"task_id": task_id, "phase": phase}
            for mode in (BASELINE, OPTIMIZED):
                sel = [
                    r
                    for r in records
                    if r.task_id == task_id and r.phase == phase and r.mode == mode
                ]
                if sel:
                    row[mode] = {
                        "correct": sum(r.correct for r in sel),
                        "runs": len(sel),
                        "tool_executions_median": median([r.metrics.tool_executions for r in sel]),
                        "tool_requests_median": median([r.metrics.tool_requests for r in sel]),
                        "latency_ms_median": median([r.metrics.latency_ms for r in sel]),
                    }
            row["category"] = next(r.category for r in records if r.task_id == task_id)
            per_task.append(row)

    return {
        **meta,
        "phases": phases,
        "regressions": regressions,
        "improvements": improvements,
        "per_task": per_task,
        "skipped_tasks": skipped,
    }
