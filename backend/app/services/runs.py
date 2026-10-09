"""Run service: starts agent runs in the background and records them.

* ``submit`` validates the request, creates a ``queued`` row, and starts an
  asyncio task. It returns immediately; the caller polls the run's status.
* At most ``max_concurrent_runs`` runs execute at once; others wait as queued.
* A run reaches a final status only after ``RunRepository.save_result`` has
  committed its full result. Failures in our own code mark it ``error``;
  cancellation at shutdown marks it ``interrupted``.
* Sessions: each session gets its own mock world (data) and gateways (one per
  mode, so each has its own cache). Sessions beyond ``max_sessions`` evict the
  least recently used idle session.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import uuid4

from app.agent.models import AgentLimits
from app.agent.runner import AgentRunner
from app.config import Settings
from app.evaluation.dataset import EvalTask, load_dataset
from app.evaluation.scripted_agent import scripted_agent_for
from app.gateway.context import RunContext
from app.gateway.factory import build_gateway
from app.gateway.gateway import ToolGateway
from app.gateway.models import GatewayMode
from app.llm.base import LLMClient, LLMResponse
from app.llm.scripted import ScriptedLLM
from app.optimizer.learned import PolicyProvider
from app.safety.permissions import RunLimits, RunPermissions
from app.services.schemas import RunRequest
from app.storage.repository import RunRepository
from app.telemetry.clock import Clock, OffsetClock, SystemClock
from app.telemetry.trace import TraceRecorder
from app.tools.mock_world import MockWorld

logger = logging.getLogger(__name__)

LLMFactory = Callable[[RunRequest], LLMClient]


class RequestRejected(ValueError):
    """The request is well-formed but cannot be accepted (e.g. unknown tool)."""

    def __init__(self, code: str, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass
class _Session:
    gateways: dict[GatewayMode, ToolGateway]
    world: MockWorld
    data_clock: OffsetClock
    active_runs: int = 0


@dataclass
class SessionManager:
    recorder: TraceRecorder
    clock: Clock
    max_sessions: int
    policy: PolicyProvider | None = None  # learned rules (optimized gateways only use them)
    _sessions: OrderedDict[str, _Session] = field(default_factory=OrderedDict)

    def get(self, session_id: str) -> _Session:
        session = self._sessions.get(session_id)
        if session is not None:
            self._sessions.move_to_end(session_id)
            return session
        if len(self._sessions) >= self.max_sessions:
            idle = next((k for k, s in self._sessions.items() if s.active_runs == 0), None)
            if idle is None:
                raise RequestRejected(
                    "too_many_sessions", "all sessions are busy; try again later", 503
                )
            del self._sessions[idle]
        # Both modes share one mock world (the session's data) but have separate
        # gateways, so the baseline gateway never sees the optimized cache.
        # The session's data clock can be moved forward by scripted scenarios
        # ("a few minutes later"); latency is still measured in real time.
        data_clock = OffsetClock(self.clock)
        world = MockWorld.create(data_clock)
        base = build_gateway(
            world=world, clock=self.clock, recorder=self.recorder, policy=self.policy
        )
        session = _Session(
            gateways={GatewayMode.PASSTHROUGH: base}, world=world, data_clock=data_clock
        )
        session.gateways[GatewayMode.OPTIMIZED] = base.with_mode(GatewayMode.OPTIMIZED)
        self._sessions[session_id] = session
        return session

    def peek(self, session_id: str) -> _Session | None:
        return self._sessions.get(session_id)


def scripted_llm(request: RunRequest) -> ScriptedLLM:
    steps: list[LLMResponse] = []
    for reply in request.scripted_replies or []:
        if reply.text:
            steps.append(ScriptedLLM.final(reply.text))
        else:
            steps.append(ScriptedLLM.tool_calls(*[(c.name, c.args) for c in reply.tool_calls]))
    return ScriptedLLM(steps)


class RunService:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: RunRepository,
        recorder: TraceRecorder,
        gemini_factory: Callable[[], LLMClient],
        clock: Clock | None = None,
        policy: PolicyProvider | None = None,
    ) -> None:
        self._settings = settings
        self._policy = policy
        self._repo = repository
        self._recorder = recorder
        self._gemini_factory = gemini_factory
        self._gemini: LLMClient | None = None
        self._clock = clock or SystemClock()
        self._sessions = SessionManager(recorder, self._clock, settings.max_sessions, policy)
        self._slots = asyncio.Semaphore(settings.max_concurrent_runs)
        self._tasks: dict[str, asyncio.Task[None]] = {}

    # ------------------------------------------------------------ public
    @property
    def active_run_ids(self) -> list[str]:
        return list(self._tasks)

    def available_tools(self) -> list[dict]:
        gateway = self._sessions.get("default").gateways[GatewayMode.PASSTHROUGH]
        out = []
        for decl in gateway.tool_declarations():
            spec = gateway.tool_spec(decl["name"])
            assert spec is not None
            out.append(
                {
                    **decl,
                    "effect": spec.effect.value,
                    "freshness": spec.freshness.value,
                    "ttl_seconds": spec.ttl_seconds,
                    "cache_scope": spec.cache_scope.value,
                    "requires_permission": spec.requires_explicit_permission,
                    "timeout_seconds": spec.timeout_seconds,
                }
            )
        return out

    def submit(self, request: RunRequest) -> str:
        session = self._sessions.get(request.session_id)
        gateway = session.gateways[request.mode]
        scenario = self._resolve_scenario(request)
        if scenario is not None:
            request = request.model_copy(
                update={
                    "task": scenario.task,
                    "tools": scenario.tools,
                    "allow_side_effects": scenario.allow_side_effects,
                    "fresh_tools": scenario.fresh_tools,
                }
            )
        self._check_tools(gateway, request)
        if scenario is not None and request.llm == "scripted":
            llm: LLMClient = scripted_agent_for(scenario, session.data_clock)
        else:
            llm = self._llm_for(request)  # may raise before anything is stored
        if scenario is not None:
            session.world.injected_failures.update(scenario.setup.inject_failures)
        assert request.task is not None

        run_id = uuid4().hex
        side_effect_tools = frozenset(
            name
            for name in gateway.tool_names()
            if (spec := gateway.tool_spec(name)) and spec.requires_explicit_permission
        )
        ctx = RunContext(
            session_id=request.session_id,
            run_id=run_id,
            permissions=RunPermissions(
                allowed_tools=frozenset(request.tools) if request.tools is not None else None,
                side_effect_tools=side_effect_tools if request.allow_side_effects else frozenset(),
            ),
            limits=RunLimits(
                max_tool_requests=request.max_tool_requests
                or self._settings.agent_max_tool_requests
            ),
            fresh_tools=frozenset(request.fresh_tools),
        )
        limits = AgentLimits(
            max_steps=request.max_steps or self._settings.agent_max_steps,
            run_timeout_seconds=self._settings.agent_run_timeout_seconds,
        )
        self._repo.create_queued(
            run_id=run_id,
            session_id=request.session_id,
            task=request.task,
            mode=request.mode.value,
            provider=llm.provider,
            model_id=llm.model_id,
            created_at=self._clock.utcnow().isoformat(),
            request_options=request.model_dump(mode="json", exclude={"task"}),
            config_snapshot=self._snapshot(llm, ctx, limits),
        )
        runner = AgentRunner(
            llm, gateway, limits=limits, recorder=self._recorder, clock=self._clock
        )
        session.active_runs += 1
        task = asyncio.create_task(self._execute(run_id, runner, request, ctx, session))
        self._tasks[run_id] = task
        task.add_done_callback(lambda _t, rid=run_id: self._tasks.pop(rid, None))
        return run_id

    @staticmethod
    def _resolve_scenario(request: RunRequest) -> EvalTask | None:
        if request.scenario is None:
            return None
        ref = request.scenario
        try:
            dataset = load_dataset(ref.dataset, ref.version)
            (task,) = dataset.select(task_ids=[ref.task_id])
        except (FileNotFoundError, ValueError):
            raise RequestRejected("unknown_scenario", "no such scenario") from None
        if task.scripted_only and request.llm != "scripted":
            raise RequestRejected(
                "scenario_scripted_only",
                "this scenario needs simulated time and can only run with the scripted LLM",
            )
        return task

    def clear_session_cache(self, session_id: str) -> int:
        session = self._sessions.peek(session_id)
        if session is None:
            return 0
        return session.gateways[GatewayMode.OPTIMIZED].clear_cache()

    async def wait_for(self, run_id: str, timeout: float = 30.0) -> None:
        task = self._tasks.get(run_id)
        if task is not None:
            await asyncio.wait_for(asyncio.shield(task), timeout)

    async def shutdown(self, timeout: float = 5.0) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)

    # ------------------------------------------------------------ internals
    async def _execute(
        self,
        run_id: str,
        runner: AgentRunner,
        request: RunRequest,
        ctx: RunContext,
        session: _Session,
    ) -> None:
        try:
            async with self._slots:
                self._repo.mark_running(run_id, self._clock.utcnow().isoformat())
                result = await runner.run(request.task, ctx, tools=request.tools)
                self._repo.save_result(result)
        except asyncio.CancelledError:
            self._safe_mark(run_id, interrupted=True)
            raise
        except Exception:
            logger.exception("run %s failed in the service layer", run_id)
            self._safe_mark(run_id, interrupted=False)
        finally:
            session.active_runs -= 1

    def _safe_mark(self, run_id: str, *, interrupted: bool) -> None:
        now = self._clock.utcnow().isoformat()
        try:
            if interrupted:
                self._repo.mark_interrupted(run_id, "the run was stopped before it finished", now)
            else:
                self._repo.mark_error(run_id, "internal_error", "the run failed unexpectedly", now)
        except Exception:
            logger.exception("could not record the end of run %s", run_id)

    def _check_tools(self, gateway: ToolGateway, request: RunRequest) -> None:
        known = {d["name"] for d in gateway.tool_declarations()}
        for name in [*(request.tools or []), *request.fresh_tools]:
            if name not in known:
                raise RequestRejected("unknown_tool", f"unknown tool: {name}")

    def _llm_for(self, request: RunRequest) -> LLMClient:
        if request.llm == "scripted":
            return scripted_llm(request)
        if self._gemini is None:
            self._gemini = self._gemini_factory()  # raises LLMConfigurationError if no key
        return self._gemini

    def _snapshot(self, llm: LLMClient, ctx: RunContext, limits: AgentLimits) -> dict:
        s = self._settings
        policy = self._policy() if self._policy else None
        return {
            "provider": llm.provider,
            "model_id": llm.model_id,
            "gemini_temperature": s.gemini_temperature if llm.provider == "gemini" else None,
            "gemini_thinking_level": s.gemini_thinking_level if llm.provider == "gemini" else None,
            "max_steps": limits.max_steps,
            "run_timeout_seconds": limits.run_timeout_seconds,
            "max_tool_requests": ctx.limits.max_tool_requests,
            "max_identical_requests": ctx.limits.max_identical_requests,
            # Learned rules in force when the run was submitted (used only by
            # optimized runs; shadow rules never change behavior).
            "learned_rules_active": list(policy.active_ids()) if policy else [],
            "learned_rules_shadow": list(policy.shadow_ids()) if policy else [],
        }
