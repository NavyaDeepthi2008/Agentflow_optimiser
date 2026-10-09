"""Run an evaluation from the command line and store it in the database.

    python -m app.evaluation.cli                      # scripted, 1 repetition
    python -m app.evaluation.cli --reps 3 --warm      # + warm-cache phase
    python -m app.evaluation.cli --llm gemini --reps 3   # real model (costs money)

Prints a summary table. The full report is stored and can be read later via
GET /evaluations/{id}, or printed with --json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from app.config import load_settings
from app.evaluation.dataset import load_dataset
from app.evaluation.metrics import load_pricing
from app.evaluation.runner import EvaluationConfig, EvaluationRunner, plan_runs
from app.llm.errors import LLMConfigurationError
from app.llm.gemini import GeminiClient
from app.storage.db import Database
from app.storage.evaluations import EvaluationRepository
from app.storage.redaction import Redactor
from app.storage.repository import RunRepository, SqlTraceRecorder
from app.telemetry.clock import SystemClock


def _fmt(value, digits=1) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def print_summary(report: dict) -> None:
    print(f"\n{report['label']}")
    d = report["dataset"]
    print(
        f"dataset {d['name']} v{d['version']} ({d['checksum'][:12]}), "
        f"model {report['model_id']}, evaluation {report['evaluation_id']}"
    )
    for phase, entry in report["phases"].items():
        print(f"\n[{phase} cache]")
        print(
            f"{'mode':<12}{'runs':>5}{'acc':>7}{'req':>6}{'exec':>6}{'hits':>6}{'reuse':>6}"
            f"{'llm':>5}{'tokens':>8}{'lat p50 ms':>11}{'p95 ms':>9}{'ovh p50 ms':>11}"
        )
        for mode, s in entry["modes"].items():
            print(
                f"{mode:<12}{s['runs']:>5}{_fmt(s['accuracy'] * 100, 0) + '%':>7}"
                f"{s['tool_requests']:>6}{s['tool_executions']:>6}{s['cache_hits']:>6}"
                f"{s['reused_in_run']:>6}{s['llm_calls']:>5}{_fmt(s['tokens']['total']):>8}"
                f"{_fmt(s['latency_ms']['median'], 2):>11}{_fmt(s['latency_ms']['p95'], 2):>9}"
                f"{_fmt(s['gateway_overhead_ms']['median'], 3):>11}"
            )
        if "comparison" in entry:
            c = entry["comparison"]
            print(
                f"tool executions change: {_fmt(c['tool_executions_change_pct'])}%   "
                f"latency p50 change: {_fmt(c['latency_median_change_pct'])}%   "
                f"accuracy change: {_fmt(c['accuracy_change'], 3)}"
            )
    print(f"\nregressions (baseline correct, optimized wrong): {len(report['regressions'])}")
    for r in report["regressions"]:
        print(
            f"  - {r['task_id']} [{r['phase']} rep {r['repetition']}]: "
            f"{', '.join(r['optimized_failed_checks'])}"
        )
    if report["skipped_tasks"]:
        print(f"skipped: {len(report['skipped_tasks'])} task/phase combinations (see report)")


async def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.evaluation.cli")
    parser.add_argument("--llm", choices=["scripted", "gemini"], default="scripted")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--warm", action="store_true", help="also run a warm-cache phase")
    parser.add_argument("--warm-gap", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["train", "heldout"])
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args(argv)

    settings = load_settings()
    config = EvaluationConfig(
        llm=args.llm,
        repetitions=args.reps,
        warm_cache=args.warm,
        warm_gap_seconds=args.warm_gap,
        seed=args.seed,
        split=args.split,
    )
    dataset = load_dataset(config.dataset, config.dataset_version)
    planned = len(plan_runs(config, dataset).runs)
    gemini = None
    if args.llm == "gemini":
        try:
            gemini = GeminiClient.from_settings(settings)
        except LLMConfigurationError as exc:
            print(f"Configuration error: {exc.message}", file=sys.stderr)
            return 2
        print(f"About to make {planned} real Gemini runs (this costs money).", file=sys.stderr)

    db = Database(settings.database_url)
    db.migrate()
    redactor = Redactor(settings.secret_values())
    runs, evals = RunRepository(db, redactor), EvaluationRepository(db, redactor)
    clock = SystemClock()
    runner = EvaluationRunner(
        gemini_factory=(lambda: gemini) if gemini else None,
        recorder=SqlTraceRecorder(runs),
        pricing=load_pricing(settings.pricing_file),
        clock=clock,
    )
    from uuid import uuid4

    evaluation_id = uuid4().hex
    evals.create(
        evaluation_id=evaluation_id,
        created_at=clock.utcnow().isoformat(),
        provider=config.llm,
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_checksum=dataset.checksum,
        config=config.model_dump(mode="json"),
        planned_runs=planned,
    )
    evals.mark_running(evaluation_id, clock.utcnow().isoformat())

    def before(kind, planned_run, ctx, llm):
        runs.create_queued(
            run_id=ctx.run_id,
            session_id=ctx.session_id,
            task=planned_run.task.task,
            mode=planned_run.mode.value,
            provider=llm.provider,
            model_id=llm.model_id,
            created_at=clock.utcnow().isoformat(),
            request_options={
                "evaluation_id": evaluation_id,
                "kind": kind,
                "task_id": planned_run.task.id,
            },
            config_snapshot={},
        )
        runs.mark_running(ctx.run_id, clock.utcnow().isoformat())

    def after(kind, planned_run, result, record):
        runs.save_result(result)
        if record is not None:
            evals.add_run(evaluation_id, record)
            print(
                f"  {record.seq:>3}/{planned} {record.task_id:<30} {record.mode:<12} "
                f"{record.phase:<5} {'ok ' if record.correct else 'BAD'}",
                file=sys.stderr,
            )

    try:
        _records, report = await runner.run(
            config, dataset, evaluation_id=evaluation_id, before_run=before, after_run=after
        )
    except BaseException:
        evals.fail(
            evaluation_id,
            "internal_error",
            "the evaluation stopped early",
            clock.utcnow().isoformat(),
        )
        raise
    evals.complete(evaluation_id, report, clock.utcnow().isoformat())
    db.dispose()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_summary(report)
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_main(sys.argv[1:])))


if __name__ == "__main__":
    main()
