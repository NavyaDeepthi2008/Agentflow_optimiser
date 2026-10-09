"""Learning service: mining, shadow, validation, human approval, disabling.

Lifecycle of a rule (every change is stored in ``rule_events``):

    proposed --(watch)--> shadow --(stop)--> proposed
    proposed / shadow / rejected / validated / disabled --(validate)--> validating
    validating --> validated (all criteria passed) | rejected (any failed)
    validated --(human approval)--> active --(disable)--> disabled
    (any not-yet-active status) --(dismiss)--> disabled

Safety, in layers:
* the miner only considers tools that may reuse results at all;
* ``check_override`` (the tool's fixed limits) runs when a rule is created,
  validated, approved and loaded at startup - and the gateway runs it again on
  every request;
* only a validation that PASSED against the CURRENT set of active rules can be
  approved, only by a person, and only with explicit confirmation.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from app.agent.models import RunResult
from app.config import Settings
from app.evaluation.dataset import load_dataset
from app.evaluation.metrics import Pricing
from app.evaluation.runner import EvaluationRunner, PlannedRun, plan_runs
from app.gateway.context import RunContext
from app.gateway.factory import build_gateway
from app.learning.miner import mine
from app.learning.models import Thresholds, ValidationConfig
from app.learning.validation import evaluation_config, run_validation
from app.llm.base import LLMClient
from app.optimizer.learned import (
    LearnedPolicy,
    OverrideRejected,
    ToolOverride,
    check_override,
    effective,
    make_override,
)
from app.services.runs import RequestRejected
from app.storage.learning import (
    ACTIVE,
    DISABLED,
    PROPOSED,
    REJECTED,
    SHADOW,
    SUPERSEDED,
    V_ERROR,
    V_FAILED,
    V_INTERRUPTED,
    V_PASSED,
    V_RUNNING,
    VALIDATED,
    VALIDATING,
    InvalidTransition,
    LearningRepository,
    RuleNotFoundError,
)
from app.storage.models import LearnedRuleRow, RuleValidationRow
from app.storage.redaction import Redactor
from app.storage.repository import RunRepository
from app.telemetry.clock import Clock, SystemClock
from app.telemetry.trace import TraceRecorder

logger = logging.getLogger(__name__)

OPEN_STATUSES = (PROPOSED, SHADOW, VALIDATING, VALIDATED, ACTIVE)
CAN_VALIDATE_FROM = (PROPOSED, SHADOW, REJECTED, VALIDATED, DISABLED)
CAN_SHADOW_FROM = (PROPOSED, VALIDATED, REJECTED, DISABLED)
CAN_DISABLE_FROM = (PROPOSED, SHADOW, VALIDATED, REJECTED, ACTIVE)
SUPERSEDE_FROM = (PROPOSED, SHADOW, VALIDATED, REJECTED)


def to_override(row: LearnedRuleRow) -> ToolOverride:
    return make_override(
        rule_id=row.id,
        tool_name=row.tool_name,
        tool_version=row.tool_version,
        kind=row.kind,
        setting=row.proposed_setting or {},
    )


class LearnedRules:
    """The rules the gateways use right now (in memory, loaded from the database).
    Gateways call ``policy()`` once per request."""

    def __init__(self) -> None:
        self._policy = LearnedPolicy()
        self._rows: dict[str, LearnedRuleRow] = {}

    def policy(self) -> LearnedPolicy:
        return self._policy

    def active_only(self) -> LearnedPolicy:
        return LearnedPolicy(active=self._policy.active)

    def active_rows(self) -> list[LearnedRuleRow]:
        return [self._rows[o.rule_id] for o in self._policy.active if o.rule_id in self._rows]

    def set(self, active: list[LearnedRuleRow], shadow: list[LearnedRuleRow]) -> None:
        self._rows = {r.id: r for r in [*active, *shadow]}
        self._policy = LearnedPolicy(
            active=tuple(to_override(r) for r in active),
            shadow=tuple(to_override(r) for r in shadow),
        )


class LearningService:
    def __init__(
        self,
        *,
        settings: Settings,
        repo: LearningRepository,
        runs: RunRepository,
        recorder: TraceRecorder,
        gemini_factory: Callable[[], LLMClient],
        rules: LearnedRules,
        pricing: Pricing | None = None,
        clock: Clock | None = None,
        redactor: Redactor | None = None,
    ) -> None:
        self._settings = settings
        self._repo = repo
        self._runs = runs
        self._recorder = recorder
        self._gemini_factory = gemini_factory
        self._pricing = pricing
        self._clock = clock or SystemClock()
        self._redact = redactor or Redactor()
        self.rules = rules
        self._specs = build_gateway()  # spec lookup only; never executes a tool
        self._one_at_a_time = asyncio.Semaphore(1)
        self._tasks: dict[str, asyncio.Task[None]] = {}

    # ------------------------------------------------------------ helpers
    def _now(self) -> str:
        return self._clock.utcnow().isoformat()

    def spec(self, name: str):
        return self._specs.tool_spec(name)

    def tool_names(self) -> list[str]:
        return self._specs.tool_names()

    @property
    def active_validation_ids(self) -> list[str]:
        return list(self._tasks)

    def current_settings(self) -> dict[str, dict[str, Any]]:
        out = {}
        for name in self.tool_names():
            spec = self.spec(name)
            eff = effective(spec, self.rules.policy().active)
            out[name] = {
                "ttl_seconds": eff.ttl_seconds,
                "cache_scope": eff.cache_scope.value,
                "ttl_rule": eff.ttl_rule,
                "scope_rule": eff.scope_rule,
            }
        return out

    def _get(self, rule_id: str) -> LearnedRuleRow:
        try:
            return self._repo.get_rule(rule_id)
        except RuleNotFoundError:
            raise RequestRejected("rule_not_found", "no such rule", 404) from None

    def _transition(self, rule_id: str, **kw) -> LearnedRuleRow:
        try:
            return self._repo.transition(rule_id, at=self._now(), **kw)
        except RuleNotFoundError:
            raise RequestRejected("rule_not_found", "no such rule", 404) from None
        except InvalidTransition as exc:
            raise RequestRejected("invalid_rule_status", str(exc), 409) from None

    def _check(self, row: LearnedRuleRow) -> None:
        try:
            check_override(self.spec(row.tool_name), to_override(row))
        except OverrideRejected as exc:
            raise RequestRejected("rule_not_allowed", str(exc), 409) from None

    # ------------------------------------------------------------ startup
    def start(self) -> None:
        recovered = self._repo.recover_interrupted(self._now())
        if recovered:
            logger.warning("%d rule validation(s) interrupted by a restart", len(recovered))
        self.reload()

    def reload(self) -> None:
        """Load active and shadow rules. A rule that no longer passes the fixed
        checks (e.g. its tool changed) is disabled, never silently used."""
        keep: dict[str, list[LearnedRuleRow]] = {ACTIVE: [], SHADOW: []}
        for row in self._repo.list_rules(statuses=(ACTIVE, SHADOW), limit=1000):
            try:
                check_override(self.spec(row.tool_name), to_override(row))
            except OverrideRejected as exc:
                logger.warning("disabling learned rule %s: %s", row.id, exc)
                try:
                    self._repo.transition(
                        row.id,
                        to=DISABLED,
                        allowed_from=(row.status,),
                        at=self._now(),
                        actor="system",
                        note=f"no longer allowed: {exc}",
                        values={"disabled_at": self._now()},
                    )
                except InvalidTransition:
                    pass
                continue
            keep[row.status].append(row)
        # One active rule per tool and kind (the most recently approved wins).
        seen: set[tuple[str, str]] = set()
        active = []
        for row in sorted(keep[ACTIVE], key=lambda r: r.approved_at or "", reverse=True):
            if (row.tool_name, row.kind) not in seen:
                seen.add((row.tool_name, row.kind))
                active.append(row)
        self.rules.set(active, keep[SHADOW])

    # ------------------------------------------------------------ mining
    def mine(
        self,
        *,
        dataset: str,
        dataset_version: str,
        evaluation_ids: list[str] | None,
        thresholds: Thresholds,
    ) -> str:
        evaluations = self._repo.completed_evaluations(
            dataset_name=dataset, dataset_version=dataset_version, evaluation_ids=evaluation_ids
        )
        if evaluation_ids is not None and len(evaluations) != len(set(evaluation_ids)):
            raise RequestRejected(
                "invalid_evaluations",
                f"every evaluation must exist, be completed and use {dataset} v{dataset_version}",
            )
        if not evaluations:
            raise RequestRejected(
                "no_training_data",
                f"no completed evaluation of {dataset} v{dataset_version} to learn from; run one "
                "on the train split (with the warm phase) first",
            )
        ids = [e.id for e in evaluations]
        observations = self._repo.training_observations(ids)
        report = mine(
            observations,
            tool_names=self.tool_names(),
            spec_of=self.spec,
            current={k: v for k, v in self.current_settings().items()},
            thresholds=thresholds,
        )
        mining_id = uuid4().hex
        now = self._now()
        created: list[str] = []
        duplicates: list[dict[str, str]] = []
        report_json = report.model_dump(mode="json")
        self._repo.create_mining(
            mining_id=mining_id,
            created_at=now,
            evaluation_ids=ids,
            thresholds=thresholds.model_dump(mode="json"),
            report=report_json,
        )
        existing = self._repo.list_rules(statuses=OPEN_STATUSES, limit=1000)
        for cand in report.candidates:
            if not cand.proposed or cand.proposed_setting is None:
                continue
            dup = next(
                (
                    r
                    for r in existing
                    if r.tool_name == cand.tool_name
                    and r.kind == cand.kind
                    and r.tool_version == cand.tool_version
                    and r.proposed_setting == cand.proposed_setting
                ),
                None,
            )
            if dup is not None:
                duplicates.append(
                    {"tool_name": cand.tool_name, "kind": cand.kind, "rule_id": dup.id}
                )
                continue
            row = LearnedRuleRow(
                id=uuid4().hex,
                status=PROPOSED,
                created_at=now,
                updated_at=now,
                tool_name=cand.tool_name,
                tool_version=cand.tool_version,
                kind=cand.kind,
                current_setting=cand.current_setting,
                proposed_setting=cand.proposed_setting,
                evidence={
                    **(cand.evidence.model_dump(mode="json") if cand.evidence else {}),
                    "reasons": cand.reasons,
                    "thresholds": thresholds.model_dump(mode="json"),
                },
                mining_id=mining_id,
                dataset_name=dataset,
                dataset_version=dataset_version,
            )
            self._check(row)  # never store a proposal that breaks a fixed rule
            # Older open proposals for the same tool and kind are replaced.
            for old in existing:
                if (
                    old.tool_name == row.tool_name
                    and old.kind == row.kind
                    and old.status in SUPERSEDE_FROM
                ):
                    try:
                        self._repo.transition(
                            old.id,
                            to=SUPERSEDED,
                            allowed_from=SUPERSEDE_FROM,
                            at=now,
                            actor="system",
                            note=f"replaced by a newer proposal ({row.id[:8]})",
                        )
                    except InvalidTransition:
                        pass
            self._repo.create_rule(row, note=f"proposed by mining {mining_id[:8]}")
            created.append(row.id)
        report_json["created_rule_ids"] = created
        report_json["duplicates"] = duplicates
        self._repo.set_mining_report(mining_id, report_json)
        self.reload()  # a superseded shadow rule stops being observed
        return mining_id

    # ------------------------------------------------------------ shadow
    def set_shadow(self, rule_id: str, enabled: bool) -> LearnedRuleRow:
        row = self._get(rule_id)
        if enabled:
            self._check(row)
            row = self._transition(
                rule_id,
                to=SHADOW,
                allowed_from=CAN_SHADOW_FROM,
                actor="dashboard user",
                note="observing in shadow mode (no effect on execution)",
            )
        else:
            row = self._transition(
                rule_id,
                to=PROPOSED,
                allowed_from=(SHADOW,),
                actor="dashboard user",
                note="shadow observation stopped",
            )
        self.reload()
        return row

    # ------------------------------------------------------------ validation
    def submit_validation(self, rule_id: str, config: ValidationConfig) -> str:
        row = self._get(rule_id)
        self._check(row)
        try:
            dataset = load_dataset(row.dataset_name, row.dataset_version)
            plan_runs(evaluation_config(dataset, config), dataset)
        except (FileNotFoundError, ValueError) as exc:
            raise RequestRejected("invalid_validation", str(exc)) from None
        gemini: LLMClient | None = None
        if config.llm == "gemini":
            gemini = self._gemini_factory()  # raises LLMConfigurationError without a key

        base = tuple(
            o
            for o in self.rules.policy().active
            if not (o.tool_name == row.tool_name and o.kind.value == row.kind)
        )
        previous = row.status
        self._transition(
            rule_id,
            to=VALIDATING,
            allowed_from=CAN_VALIDATE_FROM,
            actor="dashboard user",
            note="held-out validation started",
        )
        validation_id = uuid4().hex
        self._repo.create_validation(
            RuleValidationRow(
                id=validation_id,
                rule_id=rule_id,
                status=V_RUNNING,
                created_at=self._now(),
                previous_rule_status=previous,
                config={
                    **config.model_dump(mode="json"),
                    "dataset": row.dataset_name,
                    "dataset_version": row.dataset_version,
                },
                base_rule_ids=sorted(o.rule_id for o in base),
            )
        )
        self._repo.set_fields(rule_id, {"latest_validation_id": validation_id})
        if previous == SHADOW:
            self.reload()  # no longer observed in live runs while validating
        runner = EvaluationRunner(
            gemini_factory=(lambda: gemini) if gemini is not None else None,
            recorder=self._recorder,
            pricing=self._pricing,
            clock=self._clock,
        )
        task = asyncio.create_task(
            self._validate(validation_id, row, base, config, runner, dataset, previous)
        )
        self._tasks[validation_id] = task
        task.add_done_callback(lambda _t, vid=validation_id: self._tasks.pop(vid, None))
        return validation_id

    async def _validate(self, validation_id, row, base, config, runner, dataset, previous):
        candidate = to_override(row)

        def before_run(arm: str, kind: str, planned: PlannedRun, ctx: RunContext, llm) -> None:
            self._runs.create_queued(
                run_id=ctx.run_id,
                session_id=ctx.session_id,
                task=planned.task.task,
                mode=planned.mode.value,
                provider=llm.provider,
                model_id=llm.model_id,
                created_at=self._now(),
                request_options={
                    "validation_id": validation_id,
                    "rule_id": row.id,
                    "arm": arm,
                    "task_id": planned.task.id,
                    "phase": planned.phase,
                    "repetition": planned.repetition,
                    "kind": kind,
                },
                config_snapshot={
                    "learned_rules_active": [o.rule_id for o in base]
                    + ([row.id] if arm == "on" else []),
                    "learned_rules_shadow": [row.id] if arm == "off" else [],
                },
            )
            self._runs.mark_running(ctx.run_id, self._now())

        def after_run(arm: str, kind: str, planned: PlannedRun, result: RunResult) -> None:
            self._runs.save_result(result)

        try:
            async with self._one_at_a_time:
                result = await run_validation(
                    runner,
                    dataset,
                    candidate,
                    base,
                    config,
                    validation_id=validation_id,
                    spec_of=self.spec,
                    before_run=before_run,
                    after_run=after_run,
                )
            passed = bool(result["passed"])
            self._repo.finish_validation(
                validation_id,
                status=V_PASSED if passed else V_FAILED,
                ended_at=self._now(),
                result=result,
            )
            failed = [c["label"] for c in result["criteria"] if not c["passed"]]
            self._repo.transition(
                row.id,
                to=VALIDATED if passed else REJECTED,
                allowed_from=(VALIDATING,),
                at=self._now(),
                actor="system",
                note=(
                    "passed every held-out check; waiting for human approval"
                    if passed
                    else "rejected: " + "; ".join(failed)
                ),
            )
        except asyncio.CancelledError:
            self._end_badly(validation_id, row.id, previous, V_INTERRUPTED, "interrupted")
            raise
        except Exception:
            logger.exception("validation %s failed", validation_id)
            self._end_badly(validation_id, row.id, previous, V_ERROR, "failed unexpectedly")

    def _end_badly(self, validation_id, rule_id, previous, status, what) -> None:
        try:
            self._repo.finish_validation(
                validation_id,
                status=status,
                ended_at=self._now(),
                error_message=f"the validation {what}",
            )
            self._repo.transition(
                rule_id,
                to=previous,
                allowed_from=(VALIDATING,),
                at=self._now(),
                actor="system",
                note=f"validation {what}; status restored",
            )
            self.reload()
        except Exception:
            logger.exception("could not record the end of validation %s", validation_id)

    # ------------------------------------------------------------ approval
    def approve(self, rule_id: str, *, approved_by: str, note: str | None, confirm: bool):
        if not confirm:
            raise RequestRejected(
                "confirmation_required",
                "approval must be confirmed explicitly (confirm: true) after reviewing "
                "the evidence",
            )
        row = self._get(rule_id)
        if row.status != VALIDATED:
            raise RequestRejected(
                "not_validated",
                f"only a validated rule can be approved (this one is {row.status})",
                409,
            )
        self._check(row)
        if not row.latest_validation_id:
            raise RequestRejected("not_validated", "the rule has no validation", 409)
        validation = self._repo.get_validation(row.latest_validation_id)
        if validation.status != V_PASSED:
            raise RequestRejected("not_validated", "the latest validation did not pass", 409)
        current_base = sorted(
            o.rule_id
            for o in self.rules.policy().active
            if not (o.tool_name == row.tool_name and o.kind.value == row.kind)
        )
        if current_base != sorted(validation.base_rule_ids):
            raise RequestRejected(
                "revalidation_required",
                "the active rules changed since this rule was validated; validate it again",
                409,
            )
        now = self._now()
        approver = self._redact(approved_by.strip())
        for old in self.rules.active_rows():
            if old.tool_name == row.tool_name and old.kind == row.kind and old.id != row.id:
                try:
                    self._repo.transition(
                        old.id,
                        to=DISABLED,
                        allowed_from=(ACTIVE,),
                        at=now,
                        actor=approver,
                        note=f"replaced by rule {row.id[:8]}",
                        values={"disabled_at": now},
                    )
                except InvalidTransition:
                    pass
        row = self._transition(
            rule_id,
            to=ACTIVE,
            allowed_from=(VALIDATED,),
            actor=approver,
            note=self._redact(note) if note else "approved after reviewing evidence and validation",
            values={
                "approved_at": now,
                "approved_by": approver,
                "approval_note": self._redact(note) if note else None,
                "disabled_at": None,
            },
        )
        self.reload()
        return row

    def disable(self, rule_id: str, *, reason: str, disabled_by: str) -> LearnedRuleRow:
        now = self._now()
        row = self._transition(
            rule_id,
            to=DISABLED,
            allowed_from=CAN_DISABLE_FROM,
            actor=self._redact(disabled_by.strip()),
            note=self._redact(reason.strip()),
            values={"disabled_at": now},
        )
        self.reload()
        return row

    # ------------------------------------------------------------ reads
    def get_mining(self, mining_id: str):
        return self._repo.get_mining(mining_id)

    def list_minings(self, limit: int = 10):
        return self._repo.list_minings(limit=limit)

    def list_rules(self, *, statuses=None, limit: int = 100):
        return self._repo.list_rules(statuses=statuses, limit=limit)

    def get_validation(self, validation_id: str):
        return self._repo.get_validation(validation_id)

    def detail(self, rule_id: str) -> dict[str, Any]:
        row = self._get(rule_id)
        return {
            "rule": row,
            "events": self._repo.events(rule_id),
            "validations": self._repo.validations_for(rule_id),
            "shadow_live": self._repo.shadow_observations().get(rule_id),
        }

    async def wait_for(self, validation_id: str, timeout: float = 60.0) -> None:
        task = self._tasks.get(validation_id)
        if task is not None:
            await asyncio.wait_for(asyncio.shield(task), timeout)

    async def shutdown(self, timeout: float = 5.0) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)
