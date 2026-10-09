"""Evaluation service: runs evaluations in the background, one at a time.

Only one evaluation runs at a time so evaluations never compete with each
other for CPU, which would distort latency. (Individual ``/runs`` requests can
still run alongside; avoid that during a benchmark you plan to quote.)

Every agent run made by an evaluation is also stored as a normal run (with LLM
calls, tool calls and trace), so any number in a report can be traced back to
the exact run that produced it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from uuid import uuid4

from app.agent.models import RunResult
from app.config import Settings
from app.evaluation.dataset import load_dataset
from app.evaluation.metrics import Pricing
from app.evaluation.report import EvalRunRecord
from app.evaluation.runner import EvaluationConfig, EvaluationRunner, PlannedRun, plan_runs
from app.gateway.context import RunContext
from app.llm.base import LLMClient
from app.optimizer.learned import LearnedPolicy
from app.services.runs import RequestRejected
from app.storage.evaluations import EvaluationRepository
from app.storage.repository import RunRepository
from app.telemetry.clock import Clock, SystemClock
from app.telemetry.trace import TraceRecorder

logger = logging.getLogger(__name__)


class EvaluationService:
    def __init__(
        self,
        *,
        settings: Settings,
        evaluations: EvaluationRepository,
        runs: RunRepository,
        recorder: TraceRecorder,
        gemini_factory: Callable[[], LLMClient],
        pricing: Pricing | None = None,
        clock: Clock | None = None,
        active_rules: Callable[[], LearnedPolicy] | None = None,
    ) -> None:
        self._active_rules = active_rules or LearnedPolicy
        self._settings = settings
        self._evals = evaluations
        self._runs = runs
        self._recorder = recorder
        self._gemini_factory = gemini_factory
        self._pricing = pricing
        self._clock = clock or SystemClock()
        self._one_at_a_time = asyncio.Semaphore(1)
        self._tasks: dict[str, asyncio.Task[None]] = {}

    @property
    def active_ids(self) -> list[str]:
        return list(self._tasks)

    def submit(self, config: EvaluationConfig) -> str:
        try:
            dataset = load_dataset(config.dataset, config.dataset_version)
            plan = plan_runs(config, dataset)
        except FileNotFoundError as exc:
            raise RequestRejected("unknown_dataset", str(exc)) from None
        except ValueError as exc:
            raise RequestRejected("invalid_evaluation", str(exc)) from None
        gemini: LLMClient | None = None
        if config.llm == "gemini":
            gemini = self._gemini_factory()  # raises LLMConfigurationError without a key

        evaluation_id = uuid4().hex
        self._evals.create(
            evaluation_id=evaluation_id,
            created_at=self._clock.utcnow().isoformat(),
            provider=config.llm,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            dataset_checksum=dataset.checksum,
            config=config.model_dump(mode="json"),
            planned_runs=len(plan.runs),
        )
        # The rules active when the evaluation starts stay fixed for all its runs.
        snapshot = self._active_rules()
        runner = EvaluationRunner(
            gemini_factory=(lambda: gemini) if gemini is not None else None,
            recorder=self._recorder,
            pricing=self._pricing,
            clock=self._clock,
            policy=lambda: snapshot,
        )
        task = asyncio.create_task(
            self._execute(evaluation_id, runner, config, dataset, snapshot)
        )
        self._tasks[evaluation_id] = task
        task.add_done_callback(lambda _t, eid=evaluation_id: self._tasks.pop(eid, None))
        return evaluation_id

    async def _execute(self, evaluation_id, runner, config, dataset, snapshot) -> None:
        def before_run(kind: str, planned: PlannedRun, ctx: RunContext, llm: LLMClient) -> None:
            self._runs.create_queued(
                run_id=ctx.run_id,
                session_id=ctx.session_id,
                task=planned.task.task,
                mode=planned.mode.value,
                provider=llm.provider,
                model_id=llm.model_id,
                created_at=self._clock.utcnow().isoformat(),
                request_options={
                    "evaluation_id": evaluation_id,
                    "task_id": planned.task.id,
                    "phase": planned.phase,
                    "repetition": planned.repetition,
                    "kind": kind,
                },
                config_snapshot={
                    "max_steps": config.max_steps,
                    "max_tool_requests": config.max_tool_requests,
                    "learned_rules_active": (
                        [o.rule_id for o in snapshot.active]
                        if planned.mode.value == "optimized"
                        else []
                    ),
                },
            )
            self._runs.mark_running(ctx.run_id, self._clock.utcnow().isoformat())

        def after_run(
            kind: str, planned: PlannedRun, result: RunResult, record: EvalRunRecord | None
        ) -> None:
            self._runs.save_result(result)
            if record is not None:  # warm-up runs are stored as runs but not graded
                self._evals.add_run(evaluation_id, record)

        try:
            async with self._one_at_a_time:
                self._evals.mark_running(evaluation_id, self._clock.utcnow().isoformat())
                _records, report = await runner.run(
                    config,
                    dataset,
                    evaluation_id=evaluation_id,
                    before_run=before_run,
                    after_run=after_run,
                )
                self._evals.complete(evaluation_id, report, self._clock.utcnow().isoformat())
        except asyncio.CancelledError:
            self._safe_fail(
                evaluation_id,
                "interrupted",
                "the evaluation was stopped before it finished",
                interrupted=True,
            )
            raise
        except Exception:
            logger.exception("evaluation %s failed", evaluation_id)
            self._safe_fail(evaluation_id, "internal_error", "the evaluation failed unexpectedly")

    def _safe_fail(
        self, evaluation_id: str, code: str, message: str, *, interrupted: bool = False
    ) -> None:
        try:
            self._evals.fail(
                evaluation_id,
                code,
                message,
                self._clock.utcnow().isoformat(),
                interrupted=interrupted,
            )
        except Exception:
            logger.exception("could not record the end of evaluation %s", evaluation_id)

    async def shutdown(self, timeout: float = 5.0) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)
