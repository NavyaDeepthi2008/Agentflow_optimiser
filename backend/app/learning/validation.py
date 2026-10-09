"""Held-out validation of one proposed rule: rules OFF versus rules ON.

* Tasks: every task in the dataset's **held-out** split (never the training
  tasks the rule was mined from), all of them - not only those that use the
  rule's tool - so a regression anywhere is caught.
* Cold caches: every run gets a new mock world, gateway and session. The warm
  phase is included too (a warm-up run then the measured run in one new
  session), because a scope rule can only matter between runs of a session.
* Two arms, both in OPTIMIZED mode with the currently active rules:
    - **off**: the candidate runs in SHADOW (observed, never used);
    - **on**:  the candidate is ACTIVE.
  For each planned run the two arms run back to back, in a seeded random order.

The rule is REJECTED if any of these fails:

1. no task/phase/repetition that is correct with the rule off becomes
   incorrect with it on;
2. overall accuracy does not decrease;
3. no safety violation: side-effecting/unknown tools execute exactly as often
   in both arms, no such tool's result is ever reused, and no world check
   (outbox, record state) fails with the rule on;
4. measured tool executions strictly decrease;
5. the shadow arm never saw the rule serve data that differed from a live
   result (a stale result that happened not to change the answer).
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from app.agent.models import RunResult
from app.evaluation.dataset import Dataset
from app.evaluation.report import EvalRunRecord
from app.evaluation.runner import (
    EvaluationConfig,
    EvaluationRunner,
    PlannedRun,
    plan_runs,
)
from app.gateway.models import REUSE_DECISIONS, GatewayMode
from app.learning.models import ValidationConfig
from app.optimizer.eligibility import may_reuse_result
from app.optimizer.learned import LearnedPolicy, ToolOverride

SpecLookup = Callable[[str], Any]
ARMS = ("off", "on")


def evaluation_config(dataset: Dataset, config: ValidationConfig) -> EvaluationConfig:
    return EvaluationConfig(
        dataset=dataset.name,
        dataset_version=dataset.version,
        llm=config.llm,
        modes=[GatewayMode.OPTIMIZED],
        repetitions=config.repetitions,
        warm_cache=True,
        warm_gap_seconds=0.0,
        seed=config.seed,
        split="heldout",
    )


def arm_policies(
    candidate: ToolOverride, base_active: tuple[ToolOverride, ...]
) -> dict[str, LearnedPolicy]:
    return {
        "off": LearnedPolicy(active=base_active, shadow=(candidate,)),
        "on": LearnedPolicy(active=(candidate, *base_active)),
    }


class _ArmRun:
    def __init__(self, arm: str, planned: PlannedRun) -> None:
        self.arm = arm
        self.planned = planned
        self.record: EvalRunRecord | None = None
        self.results: dict[str, RunResult] = {}  # kind -> result


async def run_validation(
    runner: EvaluationRunner,
    dataset: Dataset,
    candidate: ToolOverride,
    base_active: tuple[ToolOverride, ...],
    config: ValidationConfig,
    *,
    validation_id: str,
    spec_of: SpecLookup,
    before_run: Callable[[str, str, PlannedRun, Any, Any], None] | None = None,
    after_run: Callable[[str, str, PlannedRun, RunResult], None] | None = None,
) -> dict[str, Any]:
    """Run both arms and return the validation result (``passed`` and why)."""
    eval_config = evaluation_config(dataset, config)
    plan = plan_runs(eval_config, dataset)
    policies = arm_policies(candidate, base_active)
    rng = random.Random(config.seed)
    arm_runs: list[_ArmRun] = []

    for planned in plan.runs:
        order = list(ARMS)
        rng.shuffle(order)
        for arm in order:
            ar = _ArmRun(arm, planned)

            def before(kind, p, ctx, llm, _arm=arm) -> None:
                if before_run is not None:
                    before_run(_arm, kind, p, ctx, llm)

            def after(kind, p, result, record, _ar=ar) -> None:
                _ar.results[kind] = result
                if after_run is not None:
                    after_run(_ar.arm, kind, p, result)

            ar.record = await runner.run_planned(
                eval_config,
                planned,
                evaluation_id=validation_id,
                policy=policies[arm],
                before_run=before,
                after_run=after,
            )
            arm_runs.append(ar)

    return judge(arm_runs, candidate, spec_of, eval_config, plan.skipped)


# ---------------------------------------------------------------------- judging
def _tool_calls(ar: _ArmRun) -> list:
    return [c for result in ar.results.values() for c in result.tool_results]


def judge(
    arm_runs: list[_ArmRun],
    candidate: ToolOverride,
    spec_of: SpecLookup,
    eval_config: EvaluationConfig,
    skipped: list[dict[str, str]],
) -> dict[str, Any]:
    by_key: dict[tuple, dict[str, _ArmRun]] = defaultdict(dict)
    for ar in arm_runs:
        p = ar.planned
        by_key[(p.task.id, p.phase, p.repetition)][ar.arm] = ar

    def unsafe(name: str) -> bool:
        spec = spec_of(name)
        return spec is None or not spec.is_read_only

    def eligible(name: str) -> bool:
        spec = spec_of(name)
        return spec is not None and may_reuse_result(spec)

    arms: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        runs = [ar for ar in arm_runs if ar.arm == arm]
        correct = sum(1 for ar in runs if ar.record and ar.record.correct)
        arms[arm] = {
            "runs": len(runs),
            "correct": correct,
            "accuracy": correct / len(runs) if runs else None,
            "tool_executions": sum(ar.record.metrics.tool_executions for ar in runs if ar.record),
            "tool_executions_incl_warmups": sum(
                c.executions for ar in runs for c in _tool_calls(ar)
            ),
            "tool_requests": sum(ar.record.metrics.tool_requests for ar in runs if ar.record),
            "reused_results": sum(
                ar.record.metrics.cache_hits
                + ar.record.metrics.reused_in_run
                + ar.record.metrics.shared_in_flight
                for ar in runs
                if ar.record
            ),
            "rule_applied": sum(
                1 for ar in runs for c in _tool_calls(ar) if candidate.rule_id in c.learned_rules
            ),
        }

    tasks: list[dict[str, Any]] = []
    flips: list[dict[str, Any]] = []
    violations: list[str] = []
    for (task_id, phase, rep), pair in sorted(by_key.items()):
        off, on = pair.get("off"), pair.get("on")
        if off is None or on is None or off.record is None or on.record is None:
            violations.append(f"{task_id} ({phase}, rep {rep}): an arm did not finish")
            continue
        row = {
            "task_id": task_id,
            "phase": phase,
            "repetition": rep,
            "category": off.planned.task.category,
            "off": _arm_row(off),
            "on": _arm_row(on),
            "rule_applied": sum(1 for c in _tool_calls(on) if candidate.rule_id in c.learned_rules),
        }
        tasks.append(row)
        if off.record.correct and not on.record.correct:
            flips.append(
                {
                    "task_id": task_id,
                    "phase": phase,
                    "repetition": rep,
                    "failed_checks": on.record.failed_checks,
                    "off_run_id": off.record.run_id,
                    "on_run_id": on.record.run_id,
                }
            )
        # Safety: side effects must happen exactly as often, never be reused.
        for kind in ("warmup", "measured"):
            a = off.results[kind].tool_results if kind in off.results else []
            b = on.results[kind].tool_results if kind in on.results else []
            fx_off = sum(c.executions for c in a if unsafe(c.tool_name))
            fx_on = sum(c.executions for c in b if unsafe(c.tool_name))
            if fx_off != fx_on:
                violations.append(
                    f"{task_id} ({phase}, rep {rep}, {kind}): side-effecting tools executed "
                    f"{fx_on} times with the rule on vs {fx_off} with it off"
                )
        for c in _tool_calls(on):
            if c.decision in REUSE_DECISIONS and not eligible(c.tool_name):
                violations.append(f"{task_id}: result of '{c.tool_name}' was reused")
        world_fail = [n for n in on.record.failed_checks if n.startswith("world:")]
        if world_fail:
            violations.append(f"{task_id} ({phase}, rep {rep}): world check failed: {world_fail}")

    shadow = {"would_reuse": 0, "matched": 0, "mismatched": 0, "examples": []}
    for ar in arm_runs:
        if ar.arm != "off":
            continue
        for c in _tool_calls(ar):
            if c.shadow is not None and candidate.rule_id in c.shadow.rule_ids:
                shadow["would_reuse"] += 1
                shadow["matched" if c.shadow.matched else "mismatched"] += 1
                if not c.shadow.matched and len(shadow["examples"]) < 10:
                    shadow["examples"].append(
                        {
                            "task_id": ar.planned.task.id,
                            "run_id": ar.record.run_id if ar.record else None,
                            "tool_name": c.tool_name,
                            "age_seconds": c.shadow.age_seconds,
                        }
                    )

    off_a, on_a = arms["off"]["accuracy"], arms["on"]["accuracy"]
    off_x, on_x = arms["off"]["tool_executions"], arms["on"]["tool_executions"]
    criteria = [
        {
            "name": "no_correct_to_incorrect",
            "label": "No task that was correct with the rule off became incorrect",
            "passed": not flips,
            "detail": f"{len(flips)} correct-to-incorrect changes",
        },
        {
            "name": "accuracy_not_lower",
            "label": "Accuracy did not decrease",
            "passed": off_a is not None and on_a is not None and on_a >= off_a,
            "detail": f"off {_pct(off_a)}, on {_pct(on_a)}",
        },
        {
            "name": "no_safety_violation",
            "label": "No safety violation",
            "passed": not violations,
            "detail": "; ".join(violations[:5]) if violations else "none found",
        },
        {
            "name": "executions_decreased",
            "label": "Tool executions decreased",
            "passed": on_x < off_x,
            "detail": f"off {off_x}, on {on_x} (measured runs)",
        },
        {
            "name": "no_stale_data_in_shadow",
            "label": "In shadow, every result the rule would have reused matched the live result",
            "passed": shadow["mismatched"] == 0,
            "detail": (
                f"{shadow['would_reuse']} would-be reuses, {shadow['mismatched']} differed "
                "from the live result"
            ),
        },
    ]
    passed = all(c["passed"] for c in criteria)
    return {
        "passed": passed,
        "failed_criteria": [c["name"] for c in criteria if not c["passed"]],
        "criteria": criteria,
        "arms": arms,
        "regressions": flips,
        "safety_violations": violations,
        "shadow": shadow,
        "tasks": tasks,
        "skipped": skipped,
        "method": {
            "split": "heldout",
            "cold_cache": "new mock world, gateway and session for every planned run",
            "warm_phase": "included: warm-up run then measured run in one new session",
            "arms": "off = rule in shadow; on = rule active; both optimized, same active rules",
            "order": f"arms run back to back per planned run, shuffled with seed "
            f"{eval_config.seed}",
            "repetitions": eval_config.repetitions,
            "llm": eval_config.llm,
            "dataset": f"{eval_config.dataset} v{eval_config.dataset_version}",
        },
    }


def _arm_row(ar: _ArmRun) -> dict[str, Any]:
    assert ar.record is not None
    return {
        "run_id": ar.record.run_id,
        "warmup_run_id": ar.record.warmup_run_id,
        "correct": ar.record.correct,
        "status": ar.record.metrics.status,
        "tool_executions": ar.record.metrics.tool_executions,
        "tool_requests": ar.record.metrics.tool_requests,
        "final_answer": ar.record.final_answer,
    }


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"
