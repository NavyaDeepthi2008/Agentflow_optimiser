"""ToolGateway: the single wrapper every tool request must pass through.

Pipeline for each request. Steps 0-6 and 8-9 are IDENTICAL in both modes; only
step 7 differs.

    0. run already stopped?            -> BLOCKED_RUN_STOPPED
    1. request budget                  -> BLOCKED_LIMIT (run stops)
    2. tool exists?                    -> BLOCKED_UNKNOWN_TOOL
    3. arguments valid?                -> BLOCKED_INVALID_ARGS
    4. fixed safety rules              -> BLOCKED_PERMISSION / BLOCKED_SAFETY_ERROR
    5. identical-request limit         -> BLOCKED_LIMIT (run stops)
    6. run deadline                    -> BLOCKED_LIMIT (run stops)
    7. reuse decision (OPTIMIZED mode, eligible read-only tools only)
         fresh cached result           -> CACHE_HIT / REUSED_IN_RUN
         identical request running     -> SHARED_IN_FLIGHT
         stale / missing / forced      -> continue to 8
    8. execute with timeout + retry policy
                                       -> EXECUTED_LIVE / STALE_REFRESHED /
                                          FORCED_FRESH / FAILED
    9. store result (optimized + eligible) / invalidate after writes / record

Because reuse happens only AFTER validation and safety checks, a cached result
can never be returned for a request that would have been refused.

Learned rules (Stage 7) can only change step 7 and the TTL/scope used in step
9, and only for tools that pass ``may_reuse_result`` - each rule is re-checked
against the tool's fixed limits on every request (``check_override``) and
ignored if it fails. ACTIVE rules change behavior. SHADOW rules never do: they
keep a separate shadow cache and only record whether they WOULD have reused a
result, and whether that result matched what the live execution returned.

Design rules:
* Safety checks fail CLOSED: if a check itself crashes, the call is blocked.
* Optimization fails OPEN: if cache logic crashes, the request simply executes
  live (and ``stats.optimizer_errors`` counts it).
* Telemetry fails OPEN: a broken trace recorder never breaks a tool call.
* The gateway never raises for a bad request or a failing tool.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from app.gateway.context import RunContext
from app.gateway.models import (
    Decision,
    GatewayMode,
    ResultSource,
    ShadowObservation,
    ToolCallRequest,
    ToolErrorInfo,
    ToolResult,
)
from app.optimizer.cache_store import CacheEntry, CacheStore
from app.optimizer.eligibility import may_reuse_result
from app.optimizer.keys import (
    cache_key,
    normalized_args,
    output_digest,
    request_fingerprint,
    scope_id,
)
from app.optimizer.learned import (
    NO_POLICY,
    Effective,
    LearnedPolicy,
    PolicyProvider,
    ToolOverride,
    effective,
)
from app.optimizer.retry import RetryPolicy, retry_blocker
from app.telemetry.clock import Clock, SystemClock
from app.telemetry.trace import InMemoryTraceRecorder, TraceEvent, TraceRecorder
from app.tools.declarations import tool_declarations
from app.tools.errors import ToolError
from app.tools.registry import ToolRegistry
from app.tools.spec import RESERVED_ARG, ToolSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizerConfig:
    annotate_reused_results: bool = True  # tell the model a result was reused
    max_cache_entries: int = 1000
    max_cached_output_bytes: int = 65_536


@dataclass
class GatewayStats:
    requests_total: int = 0
    live_executions: Counter[str] = field(default_factory=Counter)  # every attempt
    decisions: Counter[str] = field(default_factory=Counter)
    retries: int = 0
    timeouts: int = 0
    invalidated_entries: int = 0
    not_cached: Counter[str] = field(default_factory=Counter)
    optimizer_errors: int = 0
    trace_errors: int = 0
    abandoned_threads: int = 0  # timed-out sync tools whose thread kept running
    late_completions: int = 0  # ...and later finished
    learned_rules_applied: Counter[str] = field(default_factory=Counter)
    learned_rules_ignored: Counter[str] = field(default_factory=Counter)
    shadow: Counter[str] = field(default_factory=Counter)  # would_reuse / matched / mismatched

    def snapshot(self, cache_entries: int) -> dict[str, Any]:
        return {
            "requests_total": self.requests_total,
            "live_executions": dict(self.live_executions),
            "decisions": dict(self.decisions),
            "retries": self.retries,
            "timeouts": self.timeouts,
            "invalidated_entries": self.invalidated_entries,
            "not_cached": dict(self.not_cached),
            "optimizer_errors": self.optimizer_errors,
            "trace_errors": self.trace_errors,
            "abandoned_threads": self.abandoned_threads,
            "late_completions": self.late_completions,
            "learned_rules_applied": dict(self.learned_rules_applied),
            "learned_rules_ignored": dict(self.learned_rules_ignored),
            "shadow": dict(self.shadow),
            "cache_entries": cache_entries,
        }


@dataclass
class _Outcome:
    """What happened when the tool was actually run (shared with in-flight waiters)."""

    output: Any = None
    error: ToolErrorInfo | None = None
    executions: int = 0
    tool_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class _CallState:
    request: ToolCallRequest
    ctx: RunContext
    started_at: str
    t0: float
    spec: ToolSpec | None = None
    validated_args: dict[str, Any] | None = None
    args_model: BaseModel | None = None
    force_fresh_arg: bool = False
    fingerprint: str | None = None
    cache_status: str | None = None
    forced_fresh: bool = False
    data_t0: float | None = None
    eff: Effective | None = None  # TTL/scope after ACTIVE learned rules
    shadow_eff: Effective | None = None  # ...if SHADOW rules were active (None: no change)
    shadow_key: str | None = None
    shadow_hit: tuple[CacheEntry, float] | None = None  # (entry, age)
    shadow_generation: int = 0
    shadow: ShadowObservation | None = None
    applied_rules: list[str] = field(default_factory=list)


class ToolGateway:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        mode: GatewayMode = GatewayMode.PASSTHROUGH,
        clock: Clock | None = None,
        recorder: TraceRecorder | None = None,
        retry_policy: RetryPolicy | None = None,
        optimizer_config: OptimizerConfig | None = None,
        rng_seed: int = 0,
        data_clock: Clock | None = None,
        policy: PolicyProvider | None = None,
    ) -> None:
        if not registry.is_frozen:
            raise ValueError("the registry must be frozen before it is given to the gateway")
        self._registry = registry
        self._mode = GatewayMode(mode)
        self._clock = clock or SystemClock()
        # The data clock decides how OLD a cached result is (TTL). It is normally
        # the same as the timing clock; evaluations use a clock that can be moved
        # forward to simulate time passing, while latency stays real.
        self._data_clock = data_clock or self._clock
        self._recorder: TraceRecorder = recorder or InMemoryTraceRecorder()
        self._retry = retry_policy or RetryPolicy()
        self._config = optimizer_config or OptimizerConfig()
        self._rng = random.Random(rng_seed)
        self._cache = CacheStore(
            max_entries=self._config.max_cache_entries,
            max_entry_bytes=self._config.max_cached_output_bytes,
        )
        # Results seen by SHADOW rules only. Never read to answer a request.
        self._shadow_cache = CacheStore(
            max_entries=self._config.max_cache_entries,
            max_entry_bytes=self._config.max_cached_output_bytes,
        )
        self._inflight: dict[str, asyncio.Future[_Outcome]] = {}
        self._stats = GatewayStats()
        # Called once per request; returns the learned rules to use (if any).
        self._policy: PolicyProvider = policy or (lambda: NO_POLICY)

    # ------------------------------------------------------------ public API
    @property
    def mode(self) -> GatewayMode:
        return self._mode

    @property
    def stats(self) -> dict[str, Any]:
        """A copy of the counters (callers cannot change the real ones)."""
        return self._stats.snapshot(cache_entries=len(self._cache))

    def tool_names(self) -> list[str]:
        return self._registry.names()

    def tool_spec(self, name: str) -> ToolSpec | None:
        return self._registry.get_spec(name)

    def tool_declarations(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """Declarations for the LLM (identical in both modes). Never test-only tools."""
        specs = [s for s in self._registry.specs() if not s.test_only]
        if names is not None:
            wanted = set(names)
            specs = [s for s in specs if s.name in wanted]
        return tool_declarations(specs)

    def with_mode(self, mode: GatewayMode | str) -> ToolGateway:
        """A sibling gateway over the SAME tools (and therefore the same data),
        with the same settings and learned-rule source, but its own mode and
        counters.

        Siblings SHARE the cache store. Only an optimized gateway ever reads or
        stores results, but a write made through ANY sibling must invalidate the
        stored results it affects; otherwise a baseline run that updates a
        record in a session would leave a stale read in the optimized cache.
        """
        sibling = ToolGateway(
            self._registry,
            mode=GatewayMode(mode),
            clock=self._clock,
            recorder=self._recorder,
            retry_policy=self._retry,
            optimizer_config=self._config,
            data_clock=self._data_clock,
            policy=self._policy,
        )
        sibling._cache = self._cache
        sibling._shadow_cache = self._shadow_cache
        sibling._inflight = self._inflight
        return sibling

    def clear_cache(self, session_id: str | None = None) -> int:
        """Empty the whole cache, or only one session's entries (for a cold start)."""
        if session_id is None:
            self._shadow_cache.invalidate_all()
            return self._cache.invalidate_all()
        self._shadow_cache.clear_scope(f"session:{session_id}")
        return self._cache.clear_scope(f"session:{session_id}")

    async def call(self, request: ToolCallRequest, ctx: RunContext) -> ToolResult:
        state = _CallState(
            request=request,
            ctx=ctx,
            started_at=self._clock.utcnow().isoformat(),
            t0=self._clock.monotonic(),
            cache_status="disabled" if self._mode is GatewayMode.PASSTHROUGH else None,
            data_t0=self._data_clock.monotonic(),
        )
        self._stats.requests_total += 1
        self._emit(
            "tool_call.requested",
            state,
            {"tool_name": request.tool_name, "mode": self._mode.value},
        )

        # 0. Run already stopped by an earlier limit.
        if ctx.stop_reason is not None:
            return self._blocked(
                state,
                Decision.BLOCKED_RUN_STOPPED,
                "run_stopped",
                f"the run was stopped earlier ({ctx.stop_reason}); no more tool calls are allowed",
            )

        # 1. Request budget (counts every request, valid or not).
        ctx.tool_requests += 1
        if ctx.tool_requests > ctx.limits.max_tool_requests:
            ctx.stop_reason = "max_tool_requests"
            return self._blocked(
                state,
                Decision.BLOCKED_LIMIT,
                "limit_reached",
                f"tool request limit of {ctx.limits.max_tool_requests} reached for this run",
            )

        # 2. Does the tool exist?
        spec = self._registry.get_spec(request.tool_name)
        if spec is None:
            return self._blocked(
                state,
                Decision.BLOCKED_UNKNOWN_TOOL,
                "unknown_tool",
                f"no tool named '{request.tool_name}' is available",
            )
        state.spec = spec

        # 3. Validate arguments.
        problem = self._validate_args(state)
        if problem is not None:
            return self._blocked(state, Decision.BLOCKED_INVALID_ARGS, "invalid_arguments", problem)

        # 4. Fixed safety rules (fail closed).
        try:
            reason = ctx.permissions.denial_reason(spec)
        except Exception:
            logger.exception("safety check failed for %s; blocking (fail closed)", spec.name)
            return self._safety_error(state)
        if reason is not None:
            return self._blocked(state, Decision.BLOCKED_PERMISSION, "permission_denied", reason)

        # 5. Identical-request limit (a safety limit: fail closed, both modes).
        try:
            assert state.validated_args is not None
            state.fingerprint = request_fingerprint(
                spec, normalized_args(spec, state.validated_args)
            )
        except Exception:
            logger.exception("could not fingerprint request for %s; blocking", spec.name)
            return self._safety_error(state)
        ctx.identical_requests[state.fingerprint] += 1
        if ctx.identical_requests[state.fingerprint] > ctx.limits.max_identical_requests:
            ctx.stop_reason = "max_identical_requests"
            return self._blocked(
                state,
                Decision.BLOCKED_LIMIT,
                "repeated_request",
                f"the same request was made more than {ctx.limits.max_identical_requests} "
                "times in this run; the run was stopped to prevent a loop",
            )

        # 6. Run deadline.
        if ctx.deadline is not None and self._clock.monotonic() >= ctx.deadline:
            ctx.stop_reason = "run_timeout"
            return self._blocked(
                state,
                Decision.BLOCKED_LIMIT,
                "run_timeout",
                "the run's time limit was reached before this tool could start",
            )

        # 7. Reuse decision (optimized mode only) - fails open.
        key: str | None = None
        if self._mode is GatewayMode.OPTIMIZED:
            if not may_reuse_result(spec):
                state.cache_status = "not_eligible"
            else:
                try:
                    self._apply_policy(state)
                    assert state.eff is not None
                    key = cache_key(
                        state.fingerprint,
                        scope_id(
                            spec,
                            run_id=ctx.run_id,
                            session_id=ctx.session_id,
                            scope=state.eff.cache_scope,
                        ),
                    )
                    reused = await self._try_reuse(state, key)
                    if reused is not None:
                        return reused
                    self._shadow_lookup(state)
                except Exception:
                    logger.exception("optimizer failed for %s; executing live", spec.name)
                    self._stats.optimizer_errors += 1
                    state.cache_status = "optimizer_error"
                    key = None

        # 8-9. Execute (as the single in-flight leader when cacheable).
        return await self._execute_and_store(state, key)

    # ------------------------------------------------------------ learned rules
    def _apply_policy(self, state: _CallState) -> None:
        """Work out the TTL and scope for this request (fixed spec + valid ACTIVE
        rules), and what they would be if the SHADOW rules were active too."""
        spec = state.spec
        assert spec is not None
        try:
            policy: LearnedPolicy = self._policy()
        except Exception:
            # Fail open = no learned rules = the developer's own defaults.
            logger.exception("could not load learned rules; using tool defaults")
            self._stats.optimizer_errors += 1
            policy = NO_POLICY

        def ignored(override: ToolOverride, why: str) -> None:
            self._stats.learned_rules_ignored[override.rule_id] += 1
            self._emit(
                "learning.rule_ignored",
                state,
                {"rule_id": override.rule_id, "tool_name": spec.name, "reason": why},
            )

        state.eff = effective(spec, policy.active, on_rejected=ignored)
        if policy.shadow:
            # Shadow rules first, so they win over an active rule of the same kind.
            shadow_eff = effective(spec, policy.shadow + policy.active, on_rejected=ignored)
            if (shadow_eff.ttl_seconds, shadow_eff.cache_scope) != (
                state.eff.ttl_seconds,
                state.eff.cache_scope,
            ):
                state.shadow_eff = shadow_eff

    @staticmethod
    def _shadow_rule_ids(state: _CallState) -> list[str]:
        eff, base = state.shadow_eff, state.eff
        assert eff is not None and base is not None
        ids = []
        if eff.ttl_rule and eff.ttl_rule != base.ttl_rule:
            ids.append(eff.ttl_rule)
        if eff.scope_rule and eff.scope_rule != base.scope_rule:
            ids.append(eff.scope_rule)
        return ids

    def _shadow_lookup(self, state: _CallState) -> None:
        """The request will execute. Would the shadow rules have reused a result?"""
        if state.shadow_eff is None or state.forced_fresh:
            return
        try:
            ctx, spec = state.ctx, state.spec
            assert spec is not None and state.fingerprint is not None
            state.shadow_key = cache_key(
                state.fingerprint,
                scope_id(
                    spec,
                    run_id=ctx.run_id,
                    session_id=ctx.session_id,
                    scope=state.shadow_eff.cache_scope,
                ),
            )
            state.shadow_generation = self._shadow_cache.generation
            entry = self._shadow_cache.get(state.shadow_key)
            now = self._data_clock.monotonic()
            if entry is not None and _fresh(entry, now, state.shadow_eff.ttl_seconds):
                state.shadow_hit = (entry, entry.age(now))
        except Exception:
            logger.exception("shadow lookup failed (ignored)")
            self._stats.optimizer_errors += 1
            state.shadow_key = None

    def _shadow_record(self, state: _CallState, outcome: _Outcome) -> None:
        """After a successful live execution: compare with what the shadow rule
        would have returned, then remember the live result for the shadow rule."""
        if state.shadow_eff is None or state.shadow_key is None or not outcome.ok:
            return
        try:
            spec = state.spec
            assert spec is not None and state.validated_args is not None
            value_json = json.dumps(outcome.output)
            if state.shadow_hit is not None:
                entry, age = state.shadow_hit
                matched = output_digest(entry.value()) == output_digest(outcome.output)
                rule_ids = self._shadow_rule_ids(state)
                state.shadow = ShadowObservation(
                    rule_ids=rule_ids, matched=matched, age_seconds=age
                )
                self._stats.shadow["would_reuse"] += 1
                self._stats.shadow["matched" if matched else "mismatched"] += 1
                self._emit(
                    "learning.shadow",
                    state,
                    {
                        "tool_name": spec.name,
                        "rule_ids": rule_ids,
                        "matched": matched,
                        "age_seconds": age,
                    },
                )
            if self._shadow_cache.generation != state.shadow_generation:
                return  # data changed during the fetch
            if self._shadow_cache.blocked_by_pending(
                spec.render_tags(spec.resource_tags, state.validated_args)
            ):
                return
            if not self._shadow_cache.fits(value_json):
                return
            self._shadow_cache.put(
                CacheEntry(
                    key=state.shadow_key,
                    tool_name=spec.name,
                    scope=scope_id(
                        spec,
                        run_id=state.ctx.run_id,
                        session_id=state.ctx.session_id,
                        scope=state.shadow_eff.cache_scope,
                    ),
                    value_json=value_json,
                    created_at=self._data_clock.monotonic(),
                    created_at_iso=self._data_clock.utcnow().isoformat(),
                    ttl_seconds=state.shadow_eff.ttl_seconds,
                    tags=frozenset(spec.render_tags(spec.resource_tags, state.validated_args)),
                    run_id=state.ctx.run_id,
                    call_id=state.request.call_id,
                )
            )
        except Exception:
            logger.exception("shadow bookkeeping failed (ignored)")
            self._stats.optimizer_errors += 1

    # ------------------------------------------------------------ reuse
    async def _try_reuse(self, state: _CallState, key: str) -> ToolResult | None:
        """Return a reused result, or None if the tool must execute."""
        spec, ctx, request = state.spec, state.ctx, state.request
        assert spec is not None
        wants_fresh = request.force_fresh or state.force_fresh_arg or spec.name in ctx.fresh_tools
        if wants_fresh and key not in ctx.forced_fresh_keys:
            # Honoured once per identical request per run, so "verify" can't loop.
            ctx.forced_fresh_keys.add(key)
            state.forced_fresh = True
            state.cache_status = "bypassed_forced"
            return None

        now = self._data_clock.monotonic()
        entry = self._cache.get(key)
        assert state.eff is not None
        if entry is not None:
            # Freshness uses the TTL in force NOW, so disabling a learned rule
            # takes effect immediately for results stored under it.
            if _fresh(entry, now, state.eff.ttl_seconds):
                return self._reused_from_cache(state, entry, now)
            self._cache.delete(key)
            state.cache_status = "stale"
        else:
            state.cache_status = "miss"

        leader = self._inflight.get(key)
        if leader is not None:
            outcome = await asyncio.shield(leader)
            return self._reused_from_flight(state, outcome)
        return None

    def _reused_from_cache(self, state: _CallState, entry: CacheEntry, now: float) -> ToolResult:
        same_run = entry.run_id == state.ctx.run_id
        age = entry.age(now)
        state.cache_status = "hit"
        spec, eff = state.spec, state.eff
        assert spec is not None and eff is not None
        # Record a learned rule only when it made the difference.
        if eff.ttl_rule and spec.ttl_seconds is not None and age >= spec.ttl_seconds:
            state.applied_rules.append(eff.ttl_rule)
        if eff.scope_rule and not same_run:
            state.applied_rules.append(eff.scope_rule)
        for rule_id in state.applied_rules:
            self._stats.learned_rules_applied[rule_id] += 1
        note = None
        if self._config.annotate_reused_results:
            note = f"Reused the result of an identical earlier request ({age:.0f}s old)." + (
                " Repeating this request returns the same result." if same_run else ""
            )
        return self._finish(
            state,
            Decision.REUSED_IN_RUN if same_run else Decision.CACHE_HIT,
            ok=True,
            source=ResultSource.IN_RUN if same_run else ResultSource.CACHE,
            output=entry.value(),
            cache_age_seconds=age,
            note=note,
        )

    def _reused_from_flight(self, state: _CallState, outcome: _Outcome) -> ToolResult:
        state.cache_status = "in_flight"
        if not outcome.ok:
            return self._finish(
                state,
                Decision.FAILED,
                ok=False,
                source=ResultSource.IN_FLIGHT,
                error=outcome.error,
            )
        note = (
            "Reused the result of an identical request that was running at the same time."
            if self._config.annotate_reused_results
            else None
        )
        return self._finish(
            state,
            Decision.SHARED_IN_FLIGHT,
            ok=True,
            source=ResultSource.IN_FLIGHT,
            output=json.loads(json.dumps(outcome.output)),  # independent copy
            cache_age_seconds=0.0,
            note=note,
        )

    # ------------------------------------------------------------ execution
    async def _execute_and_store(self, state: _CallState, key: str | None) -> ToolResult:
        spec = state.spec
        assert spec is not None
        future: asyncio.Future[_Outcome] | None = None
        if key is not None and not state.forced_fresh:
            future = asyncio.get_running_loop().create_future()
            self._inflight[key] = future
        generation = self._cache.generation
        outcome = _Outcome(
            error=ToolErrorInfo(code="cancelled", message="the tool call was cancelled")
        )
        try:
            outcome = await self._run_with_retries(state)
        finally:
            if future is not None:
                if self._inflight.get(key) is future:  # type: ignore[arg-type]
                    del self._inflight[key]  # type: ignore[arg-type]
                if not future.done():
                    future.set_result(outcome)

        invalidated = self._invalidate_after(state, outcome)
        if key is not None and outcome.ok:
            self._store(state, key, outcome, generation)
            self._shadow_record(state, outcome)

        if not outcome.ok:
            decision = Decision.FAILED
        elif state.forced_fresh:
            decision = Decision.FORCED_FRESH
        elif state.cache_status == "stale":
            decision = Decision.STALE_REFRESHED
        else:
            decision = Decision.EXECUTED_LIVE
        return self._finish(
            state,
            decision,
            ok=outcome.ok,
            source=ResultSource.LIVE,
            output=outcome.output if outcome.ok else None,
            error=outcome.error,
            executions=outcome.executions,
            tool_ms=outcome.tool_ms,
            invalidated=invalidated,
        )

    async def _run_with_retries(self, state: _CallState) -> _Outcome:
        spec, ctx = state.spec, state.ctx
        assert spec is not None and state.args_model is not None
        assert state.validated_args is not None
        handler = self._registry._handler_for(spec.name)
        outcome = _Outcome()
        while True:
            timeout = spec.timeout_seconds
            if ctx.deadline is not None:
                timeout = min(timeout, ctx.deadline - self._clock.monotonic())
            if timeout <= 0:
                if outcome.executions == 0:
                    outcome.error = ToolErrorInfo(
                        code="run_timeout", message="no time left in this run for the tool"
                    )
                return outcome
            outcome.executions += 1
            self._stats.live_executions[spec.name] += 1
            t_attempt = self._clock.monotonic()
            output, error = await self._attempt(handler, state, timeout)
            outcome.tool_ms += (self._clock.monotonic() - t_attempt) * 1000
            outcome.output, outcome.error = output, error
            if error is None:
                return outcome

            blocker = retry_blocker(spec, state.validated_args, error.retryable)
            if blocker is None and outcome.executions > self._retry.max_retries:
                blocker = "retry budget used up"
            delay = self._retry.delay(outcome.executions, self._rng)
            if (
                blocker is None
                and ctx.deadline is not None
                and self._clock.monotonic() + delay >= ctx.deadline
            ):
                blocker = "not enough run time left"
            self._emit(
                "tool_call.attempt_failed",
                state,
                {
                    "tool_name": spec.name,
                    "attempt": outcome.executions,
                    "error_code": error.code,
                    "will_retry": blocker is None,
                    "retry_blocked_because": blocker,
                    "retry_delay_seconds": delay if blocker is None else None,
                },
            )
            if blocker is not None:
                return outcome
            self._stats.retries += 1
            await asyncio.sleep(delay)

    async def _attempt(
        self, handler: Any, state: _CallState, timeout: float
    ) -> tuple[Any, ToolErrorInfo | None]:
        spec, model = state.spec, state.args_model
        assert spec is not None and model is not None
        t_start = self._clock.monotonic()
        try:
            if _is_async(handler):
                # Async tools are really cancelled on timeout.
                result = await asyncio.wait_for(handler(model), timeout)
            else:
                # Sync tools run in a worker thread so a hanging tool can be timed
                # out without freezing every other request. Python cannot stop
                # that thread: see _abandon().
                worker = asyncio.get_running_loop().run_in_executor(None, handler, model)
                try:
                    result = await asyncio.wait_for(asyncio.shield(worker), timeout)
                except TimeoutError:
                    self._abandon(state, worker)
                    raise
                if inspect.isawaitable(result):
                    remaining = max(0.001, timeout - (self._clock.monotonic() - t_start))
                    result = await asyncio.wait_for(result, remaining)
        except TimeoutError:
            self._stats.timeouts += 1
            if spec.is_read_only:
                return None, ToolErrorInfo(
                    code="timeout",
                    message=f"tool '{spec.name}' did not respond within {timeout:.2g} seconds",
                    retryable=True,
                )
            return None, ToolErrorInfo(
                code="timeout_outcome_unknown",
                message=f"tool '{spec.name}' did not respond within {timeout:.2g} seconds; "
                "the action may or may not have happened",
                retryable=True,
            )
        except ToolError as exc:
            return None, ToolErrorInfo(code=exc.code, message=exc.message, retryable=exc.retryable)
        except Exception:
            # Unexpected bug inside a tool: details go to the developer log only.
            logger.exception("tool %s raised an unexpected exception", spec.name)
            return None, ToolErrorInfo(
                code="tool_internal_error", message=f"tool '{spec.name}' failed unexpectedly"
            )
        try:
            json.dumps(result)
        except (TypeError, ValueError):
            logger.error("tool %s returned output that is not JSON-serializable", spec.name)
            return None, ToolErrorInfo(
                code="invalid_tool_output",
                message=f"tool '{spec.name}' returned an unreadable result",
            )
        return result, None

    def _abandon(self, state: _CallState, worker: asyncio.Future[Any]) -> None:
        """A synchronous tool timed out but its thread keeps running, so its
        outcome is unknown and may arrive LATER. For a tool that may change data
        (side effect or unknown effect), every result it could affect is kept
        out of the cache until the thread really finishes, and dropped again
        when it does. Read-only tools change nothing, so only the counter moves.
        (The thread itself cannot be cancelled; it still uses a worker thread.)"""
        spec = state.spec
        assert spec is not None and state.validated_args is not None
        self._stats.abandoned_threads += 1
        tags: frozenset[str] | None = None
        if not spec.is_read_only:
            if not spec.invalidates_everything:
                tags = frozenset(spec.render_tags(spec.invalidates or (), state.validated_args))
            for store in (self._cache, self._shadow_cache):
                store.begin_pending(tags)
        self._emit(
            "tool_call.abandoned",
            state,
            {
                "tool_name": spec.name,
                "effect": spec.effect.value,
                "pending_invalidation": None if spec.is_read_only else sorted(tags or ["*"]),
            },
        )

        def finished(fut: asyncio.Future[Any]) -> None:
            self._stats.late_completions += 1
            removed = 0
            if not spec.is_read_only:
                for store in (self._cache, self._shadow_cache):
                    removed += store.end_pending(tags)
                self._inflight.clear()
            failed = fut.cancelled() or fut.exception() is not None
            self._emit(
                "tool_call.late_completion",
                state,
                {"tool_name": spec.name, "failed": failed, "invalidated_entries": removed},
            )

        worker.add_done_callback(finished)

    # ------------------------------------------------------------ cache upkeep
    def _store(self, state: _CallState, key: str, outcome: _Outcome, generation: int) -> None:
        spec = state.spec
        assert spec is not None and state.validated_args is not None
        try:
            if self._cache.generation != generation:
                # Data changed while this result was being fetched: don't keep it.
                self._stats.not_cached["write_during_fetch"] += 1
                return
            tags = frozenset(spec.render_tags(spec.resource_tags, state.validated_args))
            if self._cache.blocked_by_pending(tags):
                # A timed-out write may still change this data.
                self._stats.not_cached["write_outcome_pending"] += 1
                return
            value_json = json.dumps(outcome.output)
            if not self._cache.fits(value_json):
                self._stats.not_cached["too_large"] += 1
                return
            self._cache.put(
                CacheEntry(
                    key=key,
                    tool_name=spec.name,
                    scope=scope_id(
                        spec,
                        run_id=state.ctx.run_id,
                        session_id=state.ctx.session_id,
                        scope=state.eff.cache_scope if state.eff else None,
                    ),
                    value_json=value_json,
                    created_at=self._data_clock.monotonic(),
                    created_at_iso=self._data_clock.utcnow().isoformat(),
                    ttl_seconds=state.eff.ttl_seconds if state.eff else spec.ttl_seconds,
                    tags=frozenset(spec.render_tags(spec.resource_tags, state.validated_args)),
                    run_id=state.ctx.run_id,
                    call_id=state.request.call_id,
                )
            )
        except Exception:
            logger.exception("could not store result of %s (ignored)", spec.name)
            self._stats.optimizer_errors += 1

    def _invalidate_after(self, state: _CallState, outcome: _Outcome) -> int:
        """After a side-effecting/unknown tool ran (even if it failed), drop stale data."""
        spec = state.spec
        assert spec is not None and state.validated_args is not None
        if spec.is_read_only or outcome.executions == 0:
            return 0
        try:
            if spec.invalidates_everything:
                removed = self._cache.invalidate_all()
                self._shadow_cache.invalidate_all()
            else:
                tags = spec.render_tags(spec.invalidates or (), state.validated_args)
                removed = self._cache.invalidate_tags(tags)
                self._shadow_cache.invalidate_tags(tags)
        except Exception:
            # Safety first: if targeted invalidation fails, drop everything.
            logger.exception("targeted invalidation failed for %s; clearing cache", spec.name)
            removed = self._cache.invalidate_all()
            self._shadow_cache.invalidate_all()
        # Requests started after this write must not join reads begun before it.
        self._inflight.clear()
        self._stats.invalidated_entries += removed
        return removed

    # ------------------------------------------------------------ validation
    def _validate_args(self, state: _CallState) -> str | None:
        spec = state.spec
        assert spec is not None
        args = state.request.args
        if not isinstance(args, dict):
            return f"arguments must be an object, got {type(args).__name__}"
        if RESERVED_ARG in args:
            flag = args[RESERVED_ARG]
            if not isinstance(flag, bool):
                return f"{RESERVED_ARG}: must be true or false"
            state.force_fresh_arg = flag
            args = {k: v for k, v in args.items() if k != RESERVED_ARG}
        try:
            encoded = json.dumps(args)
        except (TypeError, ValueError):
            return "arguments must be plain JSON values"
        if len(encoded.encode()) > state.ctx.limits.max_args_bytes:
            return f"arguments are larger than {state.ctx.limits.max_args_bytes} bytes"
        try:
            model = spec.args_model.model_validate(args)
        except ValidationError as exc:
            # Report field names and messages only - never echo the input values back.
            parts = [
                f"{'.'.join(str(p) for p in err['loc']) or '(arguments)'}: {err['msg']}"
                for err in exc.errors()
            ]
            return "; ".join(parts)
        state.validated_args = model.model_dump(mode="json")
        state.args_model = model
        return None

    # ------------------------------------------------------------ results
    def _safety_error(self, state: _CallState) -> ToolResult:
        return self._blocked(
            state,
            Decision.BLOCKED_SAFETY_ERROR,
            "safety_check_error",
            "a safety check could not be completed, so the call was blocked",
        )

    def _blocked(
        self, state: _CallState, decision: Decision, code: str, message: str
    ) -> ToolResult:
        return self._finish(
            state,
            decision,
            ok=False,
            source=ResultSource.NONE,
            error=ToolErrorInfo(code=code, message=message),
        )

    def _finish(
        self,
        state: _CallState,
        decision: Decision,
        *,
        ok: bool,
        source: ResultSource,
        output: Any = None,
        error: ToolErrorInfo | None = None,
        executions: int = 0,
        tool_ms: float | None = None,
        cache_age_seconds: float | None = None,
        invalidated: int = 0,
        note: str | None = None,
    ) -> ToolResult:
        result = ToolResult(
            call_id=state.request.call_id,
            tool_name=state.request.tool_name,
            decision=decision,
            ok=ok,
            executed=executions > 0,
            executions=executions,
            source=source,
            output=output,
            error=error,
            started_at=state.started_at,
            ended_at=self._clock.utcnow().isoformat(),
            duration_ms=(self._clock.monotonic() - state.t0) * 1000,
            tool_duration_ms=tool_ms,
            cache_status=state.cache_status,
            cache_age_seconds=cache_age_seconds,
            forced_fresh=state.forced_fresh,
            invalidated_entries=invalidated,
            note=note,
            fingerprint=state.fingerprint,
            output_digest=_digest_or_none(output) if ok else None,
            data_time=state.data_t0,
            learned_rules=list(state.applied_rules),
            shadow=state.shadow,
        )
        self._stats.decisions[decision.value] += 1
        spec = state.spec
        self._emit(
            "tool_call.completed",
            state,
            {
                "tool_name": result.tool_name,
                "mode": self._mode.value,
                "decision": decision.value,
                "ok": ok,
                "executed": result.executed,
                "executions": executions,
                "source": source.value,
                "cache_status": state.cache_status,
                "cache_age_seconds": cache_age_seconds,
                "forced_fresh": state.forced_fresh,
                "invalidated_entries": invalidated,
                "effect": spec.effect.value if spec else None,
                "freshness": spec.freshness.value if spec else None,
                "tool_version": spec.version if spec else None,
                "fingerprint": state.fingerprint,
                "output_digest": result.output_digest,
                "data_time": state.data_t0,
                "learned_rules": list(state.applied_rules),
                "shadow": state.shadow.model_dump() if state.shadow else None,
                "args": state.validated_args,  # only validated args are stored
                "error_code": error.code if error else None,
                "error_message": error.message if error else None,
                "started_at": result.started_at,
                "ended_at": result.ended_at,
                "duration_ms": result.duration_ms,
                "tool_duration_ms": tool_ms,
                "run_tool_requests": state.ctx.tool_requests,
            },
        )
        return result

    def _emit(self, event_type: str, state: _CallState, data: dict[str, Any]) -> None:
        try:
            self._recorder.record(
                TraceEvent(
                    type=event_type,
                    run_id=state.ctx.run_id,
                    session_id=state.ctx.session_id,
                    call_id=state.request.call_id,
                    at=self._clock.utcnow().isoformat(),
                    data=data,
                )
            )
        except Exception:
            self._stats.trace_errors += 1
            logger.exception("failed to record trace event %s", event_type)


def _fresh(entry: CacheEntry, now: float, ttl_seconds: float | None) -> bool:
    return entry.is_fresh(now, ttl_seconds)


def _digest_or_none(output: Any) -> str | None:
    try:
        return output_digest(output)
    except (TypeError, ValueError):
        return None


def _is_async(handler: Any) -> bool:
    return (
        inspect.iscoroutinefunction(handler)
        or inspect.iscoroutinefunction(getattr(handler, "run", None))
        or inspect.iscoroutinefunction(getattr(handler, "__call__", None))  # noqa: B004
    )
