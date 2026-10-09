"""Run the same tasks in baseline and optimized modes and compare them fairly.

Fairness rules (enforced here, checked by tests):
* **Same everything except the mode.** Same dataset, LLM client, system
  prompt, tool declarations, permissions, agent limits, gateway limits and
  retry policy for both modes.
* **Cold cache per run.** Every measured run gets a NEW mock world, a NEW
  gateway and a NEW session id, so nothing can leak between runs.
* **Interleaved, seeded order.** For each repetition the task order is shuffled
  and, for each task, the two modes run back to back in a shuffled order (with
  a fixed seed, so it is reproducible). Neither mode always goes first.
* **Warm cache (optional) is reported separately.** A warm run is the same task
  run a second time in the SAME session, after a warm-up run (and an optional
  simulated time gap). Tasks with side effects or fault injection are not run
  warm, because the warm-up would already have changed the world.
* **Sequential.** Runs never overlap, so they don't slow each other down.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agent.models import AgentLimits, RunResult
from app.agent.runner import AgentRunner
from app.evaluation.dataset import Dataset, EvalTask
from app.evaluation.grading import grade
from app.evaluation.metrics import Pricing, run_metrics
from app.evaluation.report import EvalRunRecord, build_report
from app.evaluation.scripted_agent import scripted_agent_for
from app.gateway.context import RunContext
from app.gateway.factory import build_gateway
from app.gateway.gateway import ToolGateway
from app.gateway.models import GatewayMode
from app.llm.base import LLMClient
from app.optimizer.learned import LearnedPolicy, PolicyProvider
from app.optimizer.retry import RetryPolicy
from app.safety.permissions import RunLimits, RunPermissions
from app.telemetry.clock import Clock, OffsetClock, SystemClock
from app.telemetry.trace import InMemoryTraceRecorder, TraceRecorder
from app.tools.mock_world import MockWorld

MAX_PLANNED_RUNS = 400


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(default="core", pattern=r"^[a-z][a-z0-9_]{0,40}$")
    dataset_version: str = Field(default="1", pattern=r"^[0-9a-z]{1,10}$")
    llm: Literal["scripted", "gemini"] = "scripted"
    modes: list[GatewayMode] = Field(
        default_factory=lambda: [GatewayMode.PASSTHROUGH, GatewayMode.OPTIMIZED], min_length=1
    )
    repetitions: int = Field(default=1, ge=1, le=20)
    warm_cache: bool = False
    warm_gap_seconds: float = Field(default=0.0, ge=0, le=3600)
    seed: int = Field(default=0, ge=0)
    task_ids: list[str] | None = None
    split: Literal["train", "heldout"] | None = None
    categories: list[str] | None = None
    max_steps: int = Field(default=12, ge=1, le=50)
    max_tool_requests: int = Field(default=20, ge=1, le=200)
    run_timeout_seconds: float = Field(default=90.0, gt=0, le=600)

    @model_validator(mode="after")
    def _unique_modes(self) -> EvaluationConfig:
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("modes must not repeat")
        return self


class PlannedRun(BaseModel):
    seq: int
    task: EvalTask
    mode: GatewayMode
    phase: Literal["cold", "warm"]
    repetition: int


class EvaluationPlan(BaseModel):
    runs: list[PlannedRun]
    skipped: list[dict[str, str]]


def plan_runs(config: EvaluationConfig, dataset: Dataset) -> EvaluationPlan:
    tasks = dataset.select(
        task_ids=config.task_ids, split=config.split, categories=config.categories
    )
    skipped: list[dict[str, str]] = []
    if config.llm == "gemini":
        for t in tasks:
            if t.scripted_only:
                skipped.append(
                    {
                        "task_id": t.id,
                        "phase": "all",
                        "reason": "needs simulated time; scripted agent only",
                    }
                )
        tasks = [t for t in tasks if not t.scripted_only]
    if not tasks:
        raise ValueError("no tasks selected")

    phases: list[Literal["cold", "warm"]] = ["cold"] + (["warm"] if config.warm_cache else [])
    rng = random.Random(config.seed)
    runs: list[PlannedRun] = []
    for phase in phases:
        phase_tasks = tasks
        if phase == "warm":
            phase_tasks = [t for t in tasks if _warm_eligible(t)]
            skipped += [
                {
                    "task_id": t.id,
                    "phase": "warm",
                    "reason": "its warm-up would change the world or the simulated time",
                }
                for t in tasks
                if not _warm_eligible(t)
            ]
        for rep in range(1, config.repetitions + 1):
            order = list(phase_tasks)
            rng.shuffle(order)
            for task in order:
                modes = list(config.modes)
                rng.shuffle(modes)
                for mode in modes:
                    runs.append(
                        PlannedRun(
                            seq=len(runs) + 1, task=task, mode=mode, phase=phase, repetition=rep
                        )
                    )
    if len(runs) > MAX_PLANNED_RUNS:
        raise ValueError(f"{len(runs)} runs planned; the maximum is {MAX_PLANNED_RUNS}")
    return EvaluationPlan(runs=runs, skipped=skipped)


def _warm_eligible(task: EvalTask) -> bool:
    """A warm run repeats the task in the same session, so the warm-up must not
    have changed anything the expectation depends on: the world (side effects,
    injected faults) or the simulated time (scripted time jumps)."""
    return (
        not task.allow_side_effects
        and not task.expected.world
        and not task.setup.inject_failures
        and not any(step.advance_seconds for step in task.script)
    )


# Hooks let the caller persist runs. ``kind`` is "warmup" or "measured".
BeforeRun = Callable[[str, PlannedRun, RunContext, LLMClient], None]
AfterRun = Callable[[str, PlannedRun, RunResult, EvalRunRecord | None], None]


class EvaluationRunner:
    def __init__(
        self,
        *,
        gemini_factory: Callable[[], LLMClient] | None = None,
        recorder: TraceRecorder | None = None,
        pricing: Pricing | None = None,
        clock: Clock | None = None,
        retry_policy: RetryPolicy | None = None,
        policy: PolicyProvider | None = None,
    ) -> None:
        """``policy`` supplies the learned rules for OPTIMIZED runs (the same
        rules for every run of the evaluation; baseline runs never use them)."""
        self._policy = policy
        self._gemini_factory = gemini_factory
        self._gemini: LLMClient | None = None
        self._recorder = recorder or InMemoryTraceRecorder()
        self._pricing = pricing
        self._clock = clock or SystemClock()
        self._retry = retry_policy

    def _llm(self, config: EvaluationConfig, task: EvalTask, data_clock: OffsetClock) -> LLMClient:
        if config.llm == "scripted":
            return scripted_agent_for(task, data_clock)
        if self._gemini is None:
            if self._gemini_factory is None:
                raise ValueError("no Gemini client configured")
            self._gemini = self._gemini_factory()
        return self._gemini

    async def run(
        self,
        config: EvaluationConfig,
        dataset: Dataset,
        *,
        evaluation_id: str | None = None,
        before_run: BeforeRun | None = None,
        after_run: AfterRun | None = None,
    ) -> tuple[list[EvalRunRecord], dict]:
        evaluation_id = evaluation_id or uuid4().hex
        plan = plan_runs(config, dataset)
        records: list[EvalRunRecord] = []
        providers: set[str] = set()
        model_ids: set[str] = set()
        for planned in plan.runs:
            record, provider, model_id = await self._run_one(
                config, planned, evaluation_id, before_run, after_run
            )
            records.append(record)
            providers.add(provider)
            model_ids.add(model_id)
        report = build_report(
            records,
            meta=self._meta(config, dataset, evaluation_id, providers, model_ids),
            skipped=plan.skipped,
        )
        return records, report

    async def _run_one(
        self,
        config,
        planned: PlannedRun,
        evaluation_id,
        before_run,
        after_run,
        policy: LearnedPolicy | None = None,
    ):
        task = planned.task
        data_clock = OffsetClock(self._clock)
        world = MockWorld.create(data_clock)
        provider = (lambda: policy) if policy is not None else self._policy
        gateway = build_gateway(
            world=world,
            clock=self._clock,
            recorder=self._recorder,
            mode=planned.mode,
            retry_policy=self._retry,
            policy=provider,
        )
        session_id = f"eval-{evaluation_id[:8]}-{planned.seq}"
        limits = AgentLimits(
            max_steps=config.max_steps, run_timeout_seconds=config.run_timeout_seconds
        )

        warmup_run_id: str | None = None
        warmup_correct: bool | None = None
        if planned.phase == "warm":
            # Warm-up: same task, same session and gateway. Recorded, not graded.
            warm_llm = self._llm(config, task, data_clock)
            warm_ctx = self._context(config, task, gateway, session_id)
            if before_run is not None:
                before_run("warmup", planned, warm_ctx, warm_llm)
            warm_result = await AgentRunner(
                warm_llm, gateway, limits=limits, recorder=self._recorder, clock=self._clock
            ).run(task.task, warm_ctx, tools=task.tools)
            if after_run is not None:
                after_run("warmup", planned, warm_result, None)
            # Graded too (warm-eligible tasks don't change the world), so the
            # learning engine can tell whether the warm-up itself was correct.
            warmup_run_id = warm_ctx.run_id
            warmup_correct = grade(
                task,
                status=warm_result.status.value,
                final_answer=warm_result.final_answer,
                outbox_count=len(world.outbox),
                records=world.records,
            ).correct
            data_clock.advance(config.warm_gap_seconds)

        world.injected_failures.update(task.setup.inject_failures)
        llm = self._llm(config, task, data_clock)
        ctx = self._context(config, task, gateway, session_id)
        if before_run is not None:
            before_run("measured", planned, ctx, llm)
        runner = AgentRunner(
            llm, gateway, limits=limits, recorder=self._recorder, clock=self._clock
        )
        # The agent receives ONLY the task text; expectations stay here.
        result = await runner.run(task.task, ctx, tools=task.tools)
        verdict = grade(
            task,
            status=result.status.value,
            final_answer=result.final_answer,
            outbox_count=len(world.outbox),
            records=world.records,
        )
        record = EvalRunRecord(
            seq=planned.seq,
            task_id=task.id,
            category=task.category,
            split=task.split,
            mode=planned.mode.value,
            phase=planned.phase,
            repetition=planned.repetition,
            run_id=ctx.run_id,
            correct=verdict.correct,
            failed_checks=verdict.failed_checks,
            grade_checks=verdict.checks,
            final_answer=result.final_answer,
            metrics=run_metrics(result, self._pricing),
            warmup_run_id=warmup_run_id,
            warmup_correct=warmup_correct,
        )
        if after_run is not None:
            after_run("measured", planned, result, record)
        return record, llm.provider, llm.model_id

    async def run_planned(
        self,
        config: EvaluationConfig,
        planned: PlannedRun,
        *,
        evaluation_id: str,
        policy: LearnedPolicy,
        before_run: BeforeRun | None = None,
        after_run: AfterRun | None = None,
    ) -> EvalRunRecord:
        """Run ONE planned run (with its warm-up, if warm) using ``policy``.
        Used by rule validation, which runs every planned run in two arms."""
        record, _provider, _model = await self._run_one(
            config, planned, evaluation_id, before_run, after_run, policy=policy
        )
        return record

    @staticmethod
    def _context(
        config: EvaluationConfig, task: EvalTask, gateway: ToolGateway, session_id: str
    ) -> RunContext:
        side_effect_tools = frozenset(
            name
            for name in task.tools
            if (spec := gateway.tool_spec(name)) and spec.requires_explicit_permission
        )
        return RunContext(
            session_id=session_id,
            permissions=RunPermissions(
                allowed_tools=frozenset(task.tools),
                side_effect_tools=side_effect_tools if task.allow_side_effects else frozenset(),
            ),
            limits=RunLimits(max_tool_requests=config.max_tool_requests),
            fresh_tools=frozenset(task.fresh_tools),
        )

    def _learned_rules(self) -> list[dict]:
        if self._policy is None:
            return []
        return [
            {
                "rule_id": o.rule_id,
                "tool_name": o.tool_name,
                "kind": o.kind.value,
                "ttl_seconds": o.ttl_seconds,
                "cache_scope": o.cache_scope.value if o.cache_scope else None,
            }
            for o in self._policy().active
        ]

    def _meta(self, config, dataset, evaluation_id, providers, model_ids) -> dict:
        provider = providers.pop() if len(providers) == 1 else "mixed"
        model = ", ".join(sorted(model_ids))
        if provider == "scripted":
            label = (
                "MOCK - scripted reference agent with deterministic mock tools. Shows what "
                "the optimizer does to tool calls; says nothing about real model quality. "
                "Savings reflect the redundancy written into this dataset's scripts."
            )
        else:
            label = (
                f"LIVE MODEL - {provider} ({model}) with deterministic mock tools. "
                "Model output varies between runs; compare across repetitions."
            )
        return {
            "evaluation_id": evaluation_id,
            "label": label,
            "provider": provider,
            "model_id": model,
            "dataset": {
                "name": dataset.name,
                "version": dataset.version,
                "checksum": dataset.checksum,
            },
            "config": config.model_dump(mode="json"),
            # Learned rules active in OPTIMIZED runs (baseline never uses them).
            "learned_rules": self._learned_rules(),
            "pricing": (
                {
                    "source": self._pricing.source,
                    "retrieved_on": self._pricing.retrieved_on,
                    "currency": self._pricing.currency,
                }
                if self._pricing
                else None
            ),
            "fairness": {
                "cold_cache": "new session, mock world and gateway for every measured run",
                "order": f"interleaved; shuffled with seed {config.seed}",
                "same_for_all_modes": [
                    "dataset",
                    "llm client",
                    "system prompt",
                    "tool declarations",
                    "permissions",
                    "agent limits",
                    "gateway limits",
                    "retry policy",
                ],
                "execution": "sequential (no overlapping runs)",
            },
            "notes": [
                "p95 uses the nearest-rank method; with few runs it equals the slowest run.",
                "Token and cost figures are null when the provider did not report usage or "
                "no pricing is configured.",
            ],
        }
