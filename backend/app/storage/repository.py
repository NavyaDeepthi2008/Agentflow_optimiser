"""Reading and writing runs.

Lifecycle of a run row:

    queued  -> running -> <final status from RunStatus>
        \\          \\
         \\-> error  \\-> error          (our code failed)
          \\-> interrupted               (server stopped before the run finished)

A run gets its final status ONLY in ``save_result``, which writes the summary,
every LLM call and every tool call in one transaction. Readers can never see a
"success" run without its details.

Every text value is passed through the Redactor before it is written.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, select, update

from app.agent.models import RunResult
from app.storage.db import Database
from app.storage.models import LLMCallRow, RunRow, ToolCallRow, TraceEventRow
from app.storage.redaction import Redactor
from app.telemetry.trace import TraceEvent

QUEUED = "queued"
RUNNING = "running"
INTERRUPTED = "interrupted"
INTERNAL_ERROR = "error"
ACTIVE_STATUSES = (QUEUED, RUNNING)

MAX_STORED_OUTPUT_BYTES = 16_384


class RunNotFoundError(LookupError):
    pass


class RunRepository:
    def __init__(self, db: Database, redactor: Redactor | None = None) -> None:
        self._db = db
        self._redact = redactor or Redactor()

    # ------------------------------------------------------------ writes
    def create_queued(
        self,
        *,
        run_id: str,
        session_id: str,
        task: str,
        mode: str,
        provider: str,
        model_id: str,
        created_at: str,
        request_options: dict[str, Any],
        config_snapshot: dict[str, Any],
    ) -> None:
        with self._db.session() as s, s.begin():
            s.add(
                RunRow(
                    id=run_id,
                    session_id=session_id,
                    task=self._redact(task),
                    mode=mode,
                    provider=provider,
                    model_id=model_id,
                    status=QUEUED,
                    created_at=created_at,
                    request_options=self._redact(request_options),
                    config_snapshot=self._redact(config_snapshot),
                )
            )

    def mark_running(self, run_id: str, started_at: str) -> None:
        self._set(run_id, {"status": RUNNING, "started_at": started_at}, only_from=(QUEUED,))

    def mark_error(self, run_id: str, code: str, message: str, ended_at: str) -> None:
        """Record that our own code failed. Never overwrites a finished run."""
        self._set(
            run_id,
            {
                "status": INTERNAL_ERROR,
                "error_code": code,
                "error_message": self._redact(message),
                "ended_at": ended_at,
            },
            only_from=ACTIVE_STATUSES,
        )

    def mark_interrupted(self, run_id: str, message: str, ended_at: str) -> None:
        """The run was stopped (e.g. server shutdown) before it could finish."""
        self._set(
            run_id,
            {
                "status": INTERRUPTED,
                "error_code": "interrupted",
                "error_message": self._redact(message),
                "ended_at": ended_at,
            },
            only_from=ACTIVE_STATUSES,
        )

    def mark_interrupted_runs(self, ended_at: str) -> int:
        """At startup: runs left queued/running by a previous process can't finish."""
        with self._db.session() as s, s.begin():
            result = s.execute(
                update(RunRow)
                .where(RunRow.status.in_(ACTIVE_STATUSES))
                .values(
                    status=INTERRUPTED,
                    error_code="interrupted",
                    error_message="the server stopped before this run finished",
                    ended_at=ended_at,
                )
            )
            return result.rowcount or 0

    def save_result(self, result: RunResult) -> None:
        """Write the final status, metrics, LLM calls and tool calls atomically."""
        r = self._redact
        usage = result.usage
        with self._db.session() as s, s.begin():
            row = s.get(RunRow, result.run_id)
            if row is None:
                raise RunNotFoundError(result.run_id)
            row.status = result.status.value
            row.final_answer = r(result.final_answer) if result.final_answer else None
            row.error_code = result.error.code if result.error else None
            row.error_message = r(result.error.message) if result.error else None
            row.started_at = row.started_at or result.started_at
            row.ended_at = result.ended_at
            row.steps = result.steps
            row.llm_call_count = result.llm_call_count
            row.tool_request_count = result.tool_request_count
            row.tool_execution_count = result.tool_execution_count
            row.reused_result_count = sum(1 for t in result.tool_results if t.reused)
            row.tool_retry_count = sum(max(0, t.executions - 1) for t in result.tool_results)
            row.input_tokens = usage.input_tokens
            row.output_tokens = usage.output_tokens
            row.thinking_tokens = usage.thinking_tokens
            row.cached_input_tokens = usage.cached_input_tokens
            row.total_tokens = usage.total_tokens
            row.usage_complete = usage.complete
            row.duration_ms = result.duration_ms
            row.llm_time_ms = result.llm_time_ms
            row.tool_time_ms = result.tool_time_ms
            row.decision_counts = result.tool_decision_counts()
            row.model_versions = sorted({c.model_version or c.model_id for c in result.llm_calls})

            for call in result.llm_calls:
                u = call.usage
                s.add(
                    LLMCallRow(
                        run_id=result.run_id,
                        step=call.step,
                        provider=call.provider,
                        model_id=call.model_id,
                        model_version=call.model_version,
                        ok=call.ok,
                        error_code=call.error_code,
                        finish_reason=call.finish_reason,
                        latency_ms=call.latency_ms,
                        input_tokens=u.input_tokens if u else None,
                        output_tokens=u.output_tokens if u else None,
                        thinking_tokens=u.thinking_tokens if u else None,
                        cached_input_tokens=u.cached_input_tokens if u else None,
                        total_tokens=u.total_tokens if u else None,
                        tool_calls_requested=call.tool_calls_requested,
                    )
                )
            for seq, tool in enumerate(result.tool_results, start=1):
                output, truncated = _bounded_output(r(tool.output))
                s.add(
                    ToolCallRow(
                        run_id=result.run_id,
                        seq=seq,
                        call_id=tool.call_id,
                        tool_name=tool.tool_name,
                        decision=tool.decision.value,
                        ok=tool.ok,
                        executed=tool.executed,
                        executions=tool.executions,
                        source=tool.source.value,
                        cache_status=tool.cache_status,
                        cache_age_seconds=tool.cache_age_seconds,
                        forced_fresh=tool.forced_fresh,
                        invalidated_entries=tool.invalidated_entries,
                        error_code=tool.error.code if tool.error else None,
                        error_message=r(tool.error.message) if tool.error else None,
                        output=output,
                        output_truncated=truncated,
                        duration_ms=tool.duration_ms,
                        tool_duration_ms=tool.tool_duration_ms,
                        started_at=tool.started_at,
                        ended_at=tool.ended_at,
                        fingerprint=tool.fingerprint,
                        output_digest=tool.output_digest,
                        data_time=tool.data_time,
                        learned_rules=tool.learned_rules or None,
                        shadow=tool.shadow.model_dump() if tool.shadow else None,
                    )
                )

    def add_event(self, event: TraceEvent) -> None:
        with self._db.session() as s, s.begin():
            s.add(
                TraceEventRow(
                    run_id=event.run_id,
                    session_id=event.session_id,
                    call_id=event.call_id,
                    type=event.type,
                    at=event.at,
                    data=self._redact(event.data),
                )
            )

    def _set(self, run_id: str, values: dict[str, Any], *, only_from: Sequence[str]) -> bool:
        with self._db.session() as s, s.begin():
            result = s.execute(
                update(RunRow)
                .where(RunRow.id == run_id, RunRow.status.in_(only_from))
                .values(**values)
            )
            return bool(result.rowcount)

    # ------------------------------------------------------------ reads
    def get(self, run_id: str) -> RunRow:
        with self._db.session() as s:
            row = s.get(RunRow, run_id)
            if row is None:
                raise RunNotFoundError(run_id)
            return row

    def list_runs(
        self, *, session_id: str | None = None, status: str | None = None, limit: int = 50
    ) -> list[RunRow]:
        query = select(RunRow).order_by(RunRow.created_at.desc(), RunRow.id.desc()).limit(limit)
        if session_id is not None:
            query = query.where(RunRow.session_id == session_id)
        if status is not None:
            query = query.where(RunRow.status == status)
        with self._db.session() as s:
            return list(s.scalars(query))

    def llm_calls(self, run_id: str) -> list[LLMCallRow]:
        self.get(run_id)
        with self._db.session() as s:
            return list(
                s.scalars(
                    select(LLMCallRow)
                    .where(LLMCallRow.run_id == run_id)
                    .order_by(LLMCallRow.step, LLMCallRow.id)
                )
            )

    def tool_calls(self, run_id: str) -> list[ToolCallRow]:
        self.get(run_id)
        with self._db.session() as s:
            return list(
                s.scalars(
                    select(ToolCallRow)
                    .where(ToolCallRow.run_id == run_id)
                    .order_by(ToolCallRow.seq)
                )
            )

    def trace(self, run_id: str, *, after_id: int = 0, limit: int = 1000) -> list[TraceEventRow]:
        self.get(run_id)
        with self._db.session() as s:
            return list(
                s.scalars(
                    select(TraceEventRow)
                    .where(TraceEventRow.run_id == run_id, TraceEventRow.id > after_id)
                    .order_by(TraceEventRow.id)
                    .limit(limit)
                )
            )

    def count_runs(self) -> int:
        with self._db.session() as s:
            return int(s.scalar(select(func.count()).select_from(RunRow)) or 0)


def _bounded_output(output: Any) -> tuple[Any, bool]:
    """Keep stored tool outputs small; very large ones are replaced by a marker."""
    if output is None:
        return None, False
    encoded = json.dumps(output)
    if len(encoded.encode("utf-8")) <= MAX_STORED_OUTPUT_BYTES:
        return output, False
    return {"truncated": True, "original_bytes": len(encoded.encode("utf-8"))}, True


class SqlTraceRecorder:
    """Writes each trace event to the database the moment it happens.

    So if the process dies mid-run, everything recorded until then is kept.
    """

    def __init__(self, repository: RunRepository) -> None:
        self._repo = repository

    def record(self, event: TraceEvent) -> None:
        self._repo.add_event(event)
