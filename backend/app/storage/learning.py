"""Reading and writing learned rules, minings, validations and the audit log.

Every status change goes through ``transition``: one transaction that updates
the rule ONLY if it is still in an allowed status (so two clicks on "approve"
cannot both succeed) and appends a row to ``rule_events``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from sqlalchemy import select, update

from app.learning.models import CallObservation, RunObservation
from app.storage.db import Database
from app.storage.models import (
    EvaluationRow,
    EvaluationRunRow,
    LearnedRuleRow,
    RuleEventRow,
    RuleMiningRow,
    RuleValidationRow,
    RunRow,
    ToolCallRow,
    TraceEventRow,
)

# Rule statuses
PROPOSED = "proposed"
SHADOW = "shadow"
VALIDATING = "validating"
VALIDATED = "validated"
REJECTED = "rejected"
ACTIVE = "active"
DISABLED = "disabled"
SUPERSEDED = "superseded"
RULE_STATUSES = (PROPOSED, SHADOW, VALIDATING, VALIDATED, REJECTED, ACTIVE, DISABLED, SUPERSEDED)

# Validation statuses
V_RUNNING, V_PASSED, V_FAILED, V_ERROR, V_INTERRUPTED = (
    "running",
    "passed",
    "failed",
    "error",
    "interrupted",
)


class RuleNotFoundError(LookupError):
    pass


class InvalidTransition(RuntimeError):
    def __init__(self, current: str | None, wanted: str) -> None:
        super().__init__(f"cannot change a {current} rule to {wanted}")
        self.current = current
        self.wanted = wanted


class LearningRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    # ------------------------------------------------------------ minings
    def create_mining(
        self,
        *,
        mining_id: str,
        created_at: str,
        evaluation_ids: list[str],
        thresholds: dict[str, Any],
        report: dict[str, Any],
    ) -> None:
        with self._db.session() as s, s.begin():
            s.add(
                RuleMiningRow(
                    id=mining_id,
                    created_at=created_at,
                    evaluation_ids=evaluation_ids,
                    thresholds=thresholds,
                    report=report,
                )
            )

    def set_mining_report(self, mining_id: str, report: dict[str, Any]) -> None:
        with self._db.session() as s, s.begin():
            s.execute(
                update(RuleMiningRow).where(RuleMiningRow.id == mining_id).values(report=report)
            )

    def get_mining(self, mining_id: str) -> RuleMiningRow:
        with self._db.session() as s:
            row = s.get(RuleMiningRow, mining_id)
        if row is None:
            raise RuleNotFoundError(mining_id)
        return row

    def list_minings(self, limit: int = 20) -> list[RuleMiningRow]:
        with self._db.session() as s:
            q = select(RuleMiningRow).order_by(RuleMiningRow.created_at.desc()).limit(limit)
            return list(s.scalars(q))

    # ------------------------------------------------------------ rules
    def create_rule(self, row: LearnedRuleRow, *, actor: str = "system", note: str | None = None):
        with self._db.session() as s, s.begin():
            s.add(row)
            s.flush()
            s.add(
                RuleEventRow(
                    rule_id=row.id,
                    at=row.created_at,
                    from_status=None,
                    to_status=row.status,
                    actor=actor,
                    note=note,
                )
            )

    def get_rule(self, rule_id: str) -> LearnedRuleRow:
        with self._db.session() as s:
            row = s.get(LearnedRuleRow, rule_id)
        if row is None:
            raise RuleNotFoundError(rule_id)
        return row

    def list_rules(
        self, *, statuses: Iterable[str] | None = None, limit: int = 200
    ) -> list[LearnedRuleRow]:
        with self._db.session() as s:
            q = select(LearnedRuleRow).order_by(LearnedRuleRow.created_at.desc()).limit(limit)
            if statuses is not None:
                q = q.where(LearnedRuleRow.status.in_(list(statuses)))
            return list(s.scalars(q))

    def transition(
        self,
        rule_id: str,
        *,
        to: str,
        allowed_from: Iterable[str],
        at: str,
        actor: str,
        note: str | None = None,
        values: dict[str, Any] | None = None,
    ) -> LearnedRuleRow:
        allowed = list(allowed_from)
        with self._db.session() as s, s.begin():
            current = s.get(LearnedRuleRow, rule_id)
            if current is None:
                raise RuleNotFoundError(rule_id)
            before = current.status
            changed = s.execute(
                update(LearnedRuleRow)
                .where(LearnedRuleRow.id == rule_id, LearnedRuleRow.status.in_(allowed))
                .values(status=to, updated_at=at, status_reason=note, **(values or {}))
            ).rowcount
            if changed != 1:
                raise InvalidTransition(before, to)
            s.add(
                RuleEventRow(
                    rule_id=rule_id,
                    at=at,
                    from_status=before,
                    to_status=to,
                    actor=actor,
                    note=note,
                )
            )
        return self.get_rule(rule_id)

    def set_fields(self, rule_id: str, values: dict[str, Any]) -> None:
        with self._db.session() as s, s.begin():
            s.execute(update(LearnedRuleRow).where(LearnedRuleRow.id == rule_id).values(**values))

    def events(self, rule_id: str) -> list[RuleEventRow]:
        with self._db.session() as s:
            q = (
                select(RuleEventRow)
                .where(RuleEventRow.rule_id == rule_id)
                .order_by(RuleEventRow.id)
            )
            return list(s.scalars(q))

    # ------------------------------------------------------------ validations
    def create_validation(self, row: RuleValidationRow) -> None:
        with self._db.session() as s, s.begin():
            s.add(row)

    def finish_validation(
        self,
        validation_id: str,
        *,
        status: str,
        ended_at: str,
        result: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._db.session() as s, s.begin():
            s.execute(
                update(RuleValidationRow)
                .where(RuleValidationRow.id == validation_id, RuleValidationRow.status == V_RUNNING)
                .values(
                    status=status, ended_at=ended_at, result=result, error_message=error_message
                )
            )

    def get_validation(self, validation_id: str) -> RuleValidationRow:
        with self._db.session() as s:
            row = s.get(RuleValidationRow, validation_id)
        if row is None:
            raise RuleNotFoundError(validation_id)
        return row

    def validations_for(self, rule_id: str) -> list[RuleValidationRow]:
        with self._db.session() as s:
            q = (
                select(RuleValidationRow)
                .where(RuleValidationRow.rule_id == rule_id)
                .order_by(RuleValidationRow.created_at.desc())
            )
            return list(s.scalars(q))

    def recover_interrupted(self, at: str) -> list[str]:
        """After a restart: validations that were running are marked
        ``interrupted`` and their rules go back to the status they had."""
        recovered: list[str] = []
        with self._db.session() as s, s.begin():
            running = list(
                s.scalars(select(RuleValidationRow).where(RuleValidationRow.status == V_RUNNING))
            )
            for v in running:
                v.status = V_INTERRUPTED
                v.ended_at = at
                v.error_message = "the server stopped before the validation finished"
                rule = s.get(LearnedRuleRow, v.rule_id)
                if rule is not None and rule.status == VALIDATING:
                    s.add(
                        RuleEventRow(
                            rule_id=rule.id,
                            at=at,
                            from_status=VALIDATING,
                            to_status=v.previous_rule_status,
                            actor="system",
                            note="validation interrupted by a server restart",
                        )
                    )
                    rule.status = v.previous_rule_status
                    rule.updated_at = at
                    rule.status_reason = "validation interrupted by a server restart"
                    recovered.append(rule.id)
            # A rule left "validating" without a running validation row.
            for rule in s.scalars(
                select(LearnedRuleRow).where(LearnedRuleRow.status == VALIDATING)
            ):
                if rule.id in recovered:
                    continue
                s.add(
                    RuleEventRow(
                        rule_id=rule.id,
                        at=at,
                        from_status=VALIDATING,
                        to_status=PROPOSED,
                        actor="system",
                        note="validation state lost; back to proposed",
                    )
                )
                rule.status = PROPOSED
                rule.updated_at = at
                recovered.append(rule.id)
        return recovered

    # ------------------------------------------------------------ evidence
    def completed_evaluations(
        self, *, dataset_name: str, dataset_version: str, evaluation_ids: list[str] | None
    ) -> list[EvaluationRow]:
        with self._db.session() as s:
            q = select(EvaluationRow).where(
                EvaluationRow.status == "completed",
                EvaluationRow.dataset_name == dataset_name,
                EvaluationRow.dataset_version == dataset_version,
            )
            if evaluation_ids is not None:
                q = q.where(EvaluationRow.id.in_(evaluation_ids))
            return list(s.scalars(q.order_by(EvaluationRow.created_at)))

    def training_observations(self, evaluation_ids: list[str]) -> list[RunObservation]:
        """Every measured TRAIN-split run of these evaluations, with its tool
        calls (and its warm-up run's calls in the warm phase)."""
        if not evaluation_ids:
            return []
        with self._db.session() as s:
            rows = list(
                s.scalars(
                    select(EvaluationRunRow)
                    .where(
                        EvaluationRunRow.evaluation_id.in_(evaluation_ids),
                        EvaluationRunRow.split == "train",
                        EvaluationRunRow.run_id.is_not(None),
                    )
                    .order_by(EvaluationRunRow.evaluation_id, EvaluationRunRow.seq)
                )
            )
            run_ids = {r.run_id for r in rows} | {r.warmup_run_id for r in rows if r.warmup_run_id}
            runs = {r.id: r for r in s.scalars(select(RunRow).where(RunRow.id.in_(run_ids)))}
            calls: dict[str, list[CallObservation]] = defaultdict(list)
            for c in s.scalars(
                select(ToolCallRow)
                .where(ToolCallRow.run_id.in_(run_ids))
                .order_by(ToolCallRow.run_id, ToolCallRow.seq)
            ):
                calls[c.run_id].append(
                    CallObservation(
                        seq=c.seq,
                        tool_name=c.tool_name,
                        decision=c.decision,
                        ok=c.ok,
                        executed=c.executed,
                        fingerprint=c.fingerprint,
                        output_digest=c.output_digest,
                        data_time=c.data_time,
                    )
                )
        out: list[RunObservation] = []
        for r in rows:
            run = runs.get(r.run_id or "")
            if run is None:
                continue
            warm = runs.get(r.warmup_run_id or "")
            out.append(
                RunObservation(
                    run_id=run.id,
                    session_id=run.session_id,
                    evaluation_id=r.evaluation_id,
                    task_id=r.task_id,
                    split=r.split,
                    mode=r.mode,
                    phase=r.phase,
                    status=r.status,
                    correct=r.correct,
                    calls=calls.get(run.id, []),
                    warmup=(
                        RunObservation(
                            run_id=warm.id,
                            session_id=warm.session_id,
                            evaluation_id=r.evaluation_id,
                            task_id=r.task_id,
                            split=r.split,
                            mode=r.mode,
                            phase="warmup",
                            status=warm.status,
                            correct=bool(r.warmup_correct),
                            calls=calls.get(warm.id, []),
                        )
                        if warm is not None
                        else None
                    ),
                    warmup_correct=r.warmup_correct,
                )
            )
        return out

    def shadow_observations(self, limit: int = 20_000) -> dict[str, dict[str, int]]:
        """Shadow results seen in ordinary runs (not validation runs), per rule id."""
        counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"would_reuse": 0, "matched": 0, "mismatched": 0}
        )
        with self._db.session() as s:
            q = (
                select(TraceEventRow.data, RunRow.request_options)
                .join(RunRow, RunRow.id == TraceEventRow.run_id)
                .where(TraceEventRow.type == "learning.shadow")
                .order_by(TraceEventRow.id.desc())
                .limit(limit)
            )
            for data, options in s.execute(q):
                if (options or {}).get("validation_id"):
                    continue
                for rule_id in data.get("rule_ids") or []:
                    c = counts[rule_id]
                    c["would_reuse"] += 1
                    c["matched" if data.get("matched") else "mismatched"] += 1
        return dict(counts)
