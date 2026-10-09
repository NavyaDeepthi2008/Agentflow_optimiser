"""Database tables.

Timestamps are stored as ISO 8601 UTC strings (SQLite has no timezone-aware
datetime type, and strings keep the exact value the app produced).

``runs`` holds one row per agent run, with every metric the evaluation engine
will need, so results can be compared later without re-running anything.
Detail tables (``llm_calls``, ``tool_calls``, ``trace_events``) are deleted
together with their run.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RunRow(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    task: Mapped[str] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String(20))
    provider: Mapped[str] = mapped_column(String(20))  # gemini | scripted
    model_id: Mapped[str] = mapped_column(String(100))

    # queued -> running -> one final status (see RunStatus) or "interrupted"
    status: Mapped[str] = mapped_column(String(30), index=True)
    final_answer: Mapped[str | None] = mapped_column(Text)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[str] = mapped_column(String(40), index=True)
    started_at: Mapped[str | None] = mapped_column(String(40))
    ended_at: Mapped[str | None] = mapped_column(String(40))

    # Evaluation-ready metrics (NULL until the run finishes; token fields stay
    # NULL when the provider reported no usage - never stored as 0).
    steps: Mapped[int | None] = mapped_column(Integer)
    llm_call_count: Mapped[int | None] = mapped_column(Integer)
    tool_request_count: Mapped[int | None] = mapped_column(Integer)
    tool_execution_count: Mapped[int | None] = mapped_column(Integer)
    reused_result_count: Mapped[int | None] = mapped_column(Integer)
    tool_retry_count: Mapped[int | None] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    thinking_tokens: Mapped[int | None] = mapped_column(Integer)
    cached_input_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    usage_complete: Mapped[bool | None] = mapped_column(Boolean)
    duration_ms: Mapped[float | None] = mapped_column(Float)
    llm_time_ms: Mapped[float | None] = mapped_column(Float)
    tool_time_ms: Mapped[float | None] = mapped_column(Float)
    decision_counts: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    model_versions: Mapped[list[str] | None] = mapped_column(JSON)

    # What the run was allowed to do and how it was configured (no secrets).
    request_options: Mapped[dict[str, Any]] = mapped_column(JSON)
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON)


class LLMCallRow(Base):
    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    step: Mapped[int] = mapped_column(Integer)
    provider: Mapped[str] = mapped_column(String(20))
    model_id: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str | None] = mapped_column(String(100))
    ok: Mapped[bool] = mapped_column(Boolean)
    error_code: Mapped[str | None] = mapped_column(String(64))
    finish_reason: Mapped[str | None] = mapped_column(String(64))
    latency_ms: Mapped[float] = mapped_column(Float)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    thinking_tokens: Mapped[int | None] = mapped_column(Integer)
    cached_input_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    tool_calls_requested: Mapped[list[str]] = mapped_column(JSON)


class ToolCallRow(Base):
    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    seq: Mapped[int] = mapped_column(Integer)  # order within the run
    call_id: Mapped[str] = mapped_column(String(64))
    tool_name: Mapped[str] = mapped_column(String(100))
    decision: Mapped[str] = mapped_column(String(40), index=True)
    ok: Mapped[bool] = mapped_column(Boolean)
    executed: Mapped[bool] = mapped_column(Boolean)
    executions: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(20))
    cache_status: Mapped[str | None] = mapped_column(String(30))
    cache_age_seconds: Mapped[float | None] = mapped_column(Float)
    forced_fresh: Mapped[bool] = mapped_column(Boolean)
    invalidated_entries: Mapped[int] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    output: Mapped[Any | None] = mapped_column(JSON)
    output_truncated: Mapped[bool] = mapped_column(Boolean)
    duration_ms: Mapped[float] = mapped_column(Float)
    tool_duration_ms: Mapped[float | None] = mapped_column(Float)
    started_at: Mapped[str] = mapped_column(String(40))
    ended_at: Mapped[str] = mapped_column(String(40))
    # Evidence for learned rules (Stage 7). NULL in rows recorded before that.
    fingerprint: Mapped[str | None] = mapped_column(String(64))
    output_digest: Mapped[str | None] = mapped_column(String(64))
    data_time: Mapped[float | None] = mapped_column(Float)
    learned_rules: Mapped[list[str] | None] = mapped_column(JSON)
    shadow: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class TraceEventRow(Base):
    __tablename__ = "trace_events"
    __table_args__ = (Index("ix_trace_events_run_id_id", "run_id", "id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    session_id: Mapped[str] = mapped_column(String(64))
    call_id: Mapped[str | None] = mapped_column(String(64))
    type: Mapped[str] = mapped_column(String(64))
    at: Mapped[str] = mapped_column(String(40))
    data: Mapped[dict[str, Any]] = mapped_column(JSON)


class EvaluationRow(Base):
    __tablename__ = "evaluations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # queued -> running -> completed | failed | interrupted
    status: Mapped[str] = mapped_column(String(20), index=True)
    created_at: Mapped[str] = mapped_column(String(40), index=True)
    started_at: Mapped[str | None] = mapped_column(String(40))
    ended_at: Mapped[str | None] = mapped_column(String(40))
    provider: Mapped[str] = mapped_column(String(20))  # scripted | gemini
    dataset_name: Mapped[str] = mapped_column(String(64))
    dataset_version: Mapped[str] = mapped_column(String(16))
    dataset_checksum: Mapped[str] = mapped_column(String(64))
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    planned_runs: Mapped[int] = mapped_column(Integer)
    completed_runs: Mapped[int] = mapped_column(Integer, default=0)
    report: Mapped[dict[str, Any] | None] = mapped_column(JSON)  # set only when completed
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)


class EvaluationRunRow(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evaluation_id: Mapped[str] = mapped_column(
        ForeignKey("evaluations.id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer)
    task_id: Mapped[str] = mapped_column(String(64), index=True)
    category: Mapped[str] = mapped_column(String(20))
    split: Mapped[str] = mapped_column(String(10))
    mode: Mapped[str] = mapped_column(String(20))
    phase: Mapped[str] = mapped_column(String(10))
    repetition: Mapped[int] = mapped_column(Integer)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id", ondelete="SET NULL"))
    status: Mapped[str] = mapped_column(String(30))
    correct: Mapped[bool] = mapped_column(Boolean)
    failed_checks: Mapped[list[str]] = mapped_column(JSON)
    grade_checks: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    final_answer: Mapped[str | None] = mapped_column(Text)
    tool_requests: Mapped[int] = mapped_column(Integer)
    tool_executions: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[float] = mapped_column(Float)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON)
    # Warm phase only (Stage 7): the warm-up run in the same session, and
    # whether it was graded correct. NULL for cold runs and older rows.
    warmup_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL")
    )
    warmup_correct: Mapped[bool | None] = mapped_column(Boolean)


# ---------------------------------------------------------------- learning
class LearnedRuleRow(Base):
    """One learned rule and its current lifecycle status.

    proposed -> (shadow) -> validating -> validated | rejected
    validated -> active (human approval only) -> disabled
    Every change is also written to ``rule_events``.
    """

    __tablename__ = "learned_rules"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(20), index=True)
    status_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(40), index=True)
    updated_at: Mapped[str] = mapped_column(String(40))
    tool_name: Mapped[str] = mapped_column(String(100), index=True)
    tool_version: Mapped[str] = mapped_column(String(40))
    kind: Mapped[str] = mapped_column(String(20))  # ttl | cache_scope
    current_setting: Mapped[dict[str, Any]] = mapped_column(JSON)
    proposed_setting: Mapped[dict[str, Any]] = mapped_column(JSON)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON)
    mining_id: Mapped[str | None] = mapped_column(
        ForeignKey("rule_minings.id", ondelete="SET NULL"), index=True
    )
    dataset_name: Mapped[str] = mapped_column(String(64))
    dataset_version: Mapped[str] = mapped_column(String(16))
    latest_validation_id: Mapped[str | None] = mapped_column(String(64))
    approved_at: Mapped[str | None] = mapped_column(String(40))
    approved_by: Mapped[str | None] = mapped_column(String(100))
    approval_note: Mapped[str | None] = mapped_column(Text)
    disabled_at: Mapped[str | None] = mapped_column(String(40))


class RuleMiningRow(Base):
    """One mining pass: what was examined and every candidate considered."""

    __tablename__ = "rule_minings"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[str] = mapped_column(String(40), index=True)
    evaluation_ids: Mapped[list[str]] = mapped_column(JSON)
    thresholds: Mapped[dict[str, Any]] = mapped_column(JSON)
    report: Mapped[dict[str, Any]] = mapped_column(JSON)


class RuleValidationRow(Base):
    __tablename__ = "rule_validations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    rule_id: Mapped[str] = mapped_column(
        ForeignKey("learned_rules.id", ondelete="CASCADE"), index=True
    )
    # running -> passed | failed | error | interrupted
    status: Mapped[str] = mapped_column(String(20), index=True)
    created_at: Mapped[str] = mapped_column(String(40))
    ended_at: Mapped[str | None] = mapped_column(String(40))
    previous_rule_status: Mapped[str] = mapped_column(String(20))
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    # Active rules (ids) both arms ran with, besides the rule under test.
    base_rule_ids: Mapped[list[str]] = mapped_column(JSON)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error_message: Mapped[str | None] = mapped_column(Text)


class RuleEventRow(Base):
    """Audit log of every lifecycle change (who, when, from, to, why)."""

    __tablename__ = "rule_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rule_id: Mapped[str] = mapped_column(
        ForeignKey("learned_rules.id", ondelete="CASCADE"), index=True
    )
    at: Mapped[str] = mapped_column(String(40))
    from_status: Mapped[str | None] = mapped_column(String(20))
    to_status: Mapped[str] = mapped_column(String(20))
    actor: Mapped[str] = mapped_column(String(100))  # "system" or the approver's name
    note: Mapped[str | None] = mapped_column(Text)
