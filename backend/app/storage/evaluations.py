"""Reading and writing evaluations.

An evaluation is ``completed`` only when its report has been written, in the
same transaction as the status. Each finished run is stored as it happens, so
progress is visible and partial results survive a crash (the evaluation itself
is then marked ``interrupted`` at the next startup).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select, update

from app.evaluation.report import EvalRunRecord
from app.storage.db import Database
from app.storage.models import EvaluationRow, EvaluationRunRow
from app.storage.redaction import Redactor

QUEUED, RUNNING, COMPLETED, FAILED, INTERRUPTED = (
    "queued",
    "running",
    "completed",
    "failed",
    "interrupted",
)
ACTIVE = (QUEUED, RUNNING)


class EvaluationNotFoundError(LookupError):
    pass


class EvaluationRepository:
    def __init__(self, db: Database, redactor: Redactor | None = None) -> None:
        self._db = db
        self._redact = redactor or Redactor()

    def create(
        self,
        *,
        evaluation_id: str,
        created_at: str,
        provider: str,
        dataset_name: str,
        dataset_version: str,
        dataset_checksum: str,
        config: dict[str, Any],
        planned_runs: int,
    ) -> None:
        with self._db.session() as s, s.begin():
            s.add(
                EvaluationRow(
                    id=evaluation_id,
                    status=QUEUED,
                    created_at=created_at,
                    provider=provider,
                    dataset_name=dataset_name,
                    dataset_version=dataset_version,
                    dataset_checksum=dataset_checksum,
                    config=config,
                    planned_runs=planned_runs,
                    completed_runs=0,
                )
            )

    def mark_running(self, evaluation_id: str, started_at: str) -> None:
        self._set(evaluation_id, {"status": RUNNING, "started_at": started_at}, (QUEUED,))

    def add_run(self, evaluation_id: str, record: EvalRunRecord) -> None:
        r = self._redact
        with self._db.session() as s, s.begin():
            s.add(
                EvaluationRunRow(
                    evaluation_id=evaluation_id,
                    seq=record.seq,
                    task_id=record.task_id,
                    category=record.category,
                    split=record.split,
                    mode=record.mode,
                    phase=record.phase,
                    repetition=record.repetition,
                    run_id=record.run_id,
                    status=record.metrics.status,
                    correct=record.correct,
                    failed_checks=record.failed_checks,
                    grade_checks=r(record.grade_checks),
                    final_answer=r(record.final_answer) if record.final_answer else None,
                    tool_requests=record.metrics.tool_requests,
                    tool_executions=record.metrics.tool_executions,
                    latency_ms=record.metrics.latency_ms,
                    metrics=record.metrics.model_dump(mode="json"),
                    warmup_run_id=record.warmup_run_id,
                    warmup_correct=record.warmup_correct,
                )
            )
            s.execute(
                update(EvaluationRow)
                .where(EvaluationRow.id == evaluation_id)
                .values(completed_runs=EvaluationRow.completed_runs + 1)
            )

    def complete(self, evaluation_id: str, report: dict[str, Any], ended_at: str) -> None:
        self._set(
            evaluation_id,
            {"status": COMPLETED, "report": self._redact(report), "ended_at": ended_at},
            ACTIVE,
        )

    def fail(
        self,
        evaluation_id: str,
        code: str,
        message: str,
        ended_at: str,
        *,
        interrupted: bool = False,
    ) -> None:
        self._set(
            evaluation_id,
            {
                "status": INTERRUPTED if interrupted else FAILED,
                "error_code": code,
                "error_message": self._redact(message),
                "ended_at": ended_at,
            },
            ACTIVE,
        )

    def mark_interrupted_all(self, ended_at: str) -> int:
        with self._db.session() as s, s.begin():
            result = s.execute(
                update(EvaluationRow)
                .where(EvaluationRow.status.in_(ACTIVE))
                .values(
                    status=INTERRUPTED,
                    error_code="interrupted",
                    error_message="the server stopped before this evaluation finished",
                    ended_at=ended_at,
                )
            )
            return result.rowcount or 0

    def _set(self, evaluation_id: str, values: dict[str, Any], only_from: tuple[str, ...]) -> None:
        with self._db.session() as s, s.begin():
            s.execute(
                update(EvaluationRow)
                .where(EvaluationRow.id == evaluation_id, EvaluationRow.status.in_(only_from))
                .values(**values)
            )

    def get(self, evaluation_id: str) -> EvaluationRow:
        with self._db.session() as s:
            row = s.get(EvaluationRow, evaluation_id)
            if row is None:
                raise EvaluationNotFoundError(evaluation_id)
            return row

    def list(self, limit: int = 50) -> list[EvaluationRow]:
        with self._db.session() as s:
            return list(
                s.scalars(
                    select(EvaluationRow)
                    .order_by(EvaluationRow.created_at.desc(), EvaluationRow.id.desc())
                    .limit(limit)
                )
            )

    def runs(self, evaluation_id: str) -> list[EvaluationRunRow]:
        self.get(evaluation_id)
        with self._db.session() as s:
            return list(
                s.scalars(
                    select(EvaluationRunRow)
                    .where(EvaluationRunRow.evaluation_id == evaluation_id)
                    .order_by(EvaluationRunRow.seq)
                )
            )
