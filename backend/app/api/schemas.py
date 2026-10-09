"""Response shapes for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from app.evaluation.metrics import Pricing, estimate_llm_cost
from app.storage.models import (
    EvaluationRow,
    LLMCallRow,
    RunRow,
    ToolCallRow,
    TraceEventRow,
)
from app.storage.repository import ACTIVE_STATUSES


class RunCreated(BaseModel):
    run_id: str
    status: str
    links: dict[str, str]


class RunMetrics(BaseModel):
    steps: int | None
    llm_calls: int | None
    tool_requests: int | None
    tool_executions: int | None
    reused_results: int | None
    tool_retries: int | None
    input_tokens: int | None
    output_tokens: int | None
    thinking_tokens: int | None
    cached_input_tokens: int | None
    total_tokens: int | None
    usage_complete: bool | None
    duration_ms: float | None
    llm_time_ms: float | None
    tool_time_ms: float | None
    decision_counts: dict[str, int] | None
    model_versions: list[str] | None


class CostEstimate(BaseModel):
    llm: float
    currency: str
    pricing_source: str
    pricing_retrieved_on: str
    note: str = "LLM tokens only; tool costs are reported in evaluations."


def _cost(row: RunRow, pricing: Pricing | None) -> CostEstimate | None:
    if pricing is None or row.steps is None:
        return None
    value = estimate_llm_cost(
        pricing,
        row.model_id,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        thinking_tokens=row.thinking_tokens,
        complete=row.usage_complete,
    )
    if value is None:
        return None
    return CostEstimate(
        llm=value,
        currency=pricing.currency,
        pricing_source=pricing.source,
        pricing_retrieved_on=pricing.retrieved_on,
    )


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: str
    session_id: str
    task: str
    mode: str
    provider: str
    model_id: str
    status: str
    finished: bool  # False while queued or running
    final_answer: str | None
    error: dict[str, str] | None
    created_at: str
    started_at: str | None
    ended_at: str | None
    metrics: RunMetrics | None  # None until the run has finished
    # Only when a pricing file is configured AND the provider reported complete
    # usage; otherwise null (never 0).
    estimated_cost: CostEstimate | None
    request_options: dict[str, Any]
    config_snapshot: dict[str, Any]

    @classmethod
    def from_row(cls, row: RunRow, pricing: Pricing | None = None) -> RunOut:
        finished = row.status not in ACTIVE_STATUSES
        has_metrics = row.steps is not None
        return cls(
            run_id=row.id,
            session_id=row.session_id,
            task=row.task,
            mode=row.mode,
            provider=row.provider,
            model_id=row.model_id,
            status=row.status,
            finished=finished,
            final_answer=row.final_answer,
            error=(
                {"code": row.error_code, "message": row.error_message or ""}
                if row.error_code
                else None
            ),
            created_at=row.created_at,
            started_at=row.started_at,
            ended_at=row.ended_at,
            metrics=RunMetrics(
                steps=row.steps,
                llm_calls=row.llm_call_count,
                tool_requests=row.tool_request_count,
                tool_executions=row.tool_execution_count,
                reused_results=row.reused_result_count,
                tool_retries=row.tool_retry_count,
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                thinking_tokens=row.thinking_tokens,
                cached_input_tokens=row.cached_input_tokens,
                total_tokens=row.total_tokens,
                usage_complete=row.usage_complete,
                duration_ms=row.duration_ms,
                llm_time_ms=row.llm_time_ms,
                tool_time_ms=row.tool_time_ms,
                decision_counts=row.decision_counts,
                model_versions=row.model_versions,
            )
            if has_metrics
            else None,
            estimated_cost=_cost(row, pricing),
            request_options=row.request_options,
            config_snapshot=row.config_snapshot,
        )


class LLMCallOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    step: int
    provider: str
    model_id: str
    model_version: str | None
    ok: bool
    error_code: str | None
    finish_reason: str | None
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    thinking_tokens: int | None
    cached_input_tokens: int | None
    total_tokens: int | None
    tool_calls_requested: list[str]

    @classmethod
    def from_row(cls, row: LLMCallRow) -> LLMCallOut:
        return cls.model_validate(row)


class ToolCallOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    seq: int
    call_id: str
    tool_name: str
    decision: str
    ok: bool
    executed: bool
    executions: int
    source: str
    cache_status: str | None
    cache_age_seconds: float | None
    forced_fresh: bool
    invalidated_entries: int
    error_code: str | None
    error_message: str | None
    output: Any
    output_truncated: bool
    duration_ms: float
    tool_duration_ms: float | None
    started_at: str
    ended_at: str
    # Stage 7: active learned rules that changed this decision; what a shadow
    # rule would have done. Empty/null for older rows.
    learned_rules: list[str] | None = None
    shadow: dict[str, Any] | None = None

    @classmethod
    def from_row(cls, row: ToolCallRow) -> ToolCallOut:
        return cls.model_validate(row)


class RunDetailsOut(BaseModel):
    run: RunOut
    llm_calls: list[LLMCallOut]
    tool_calls: list[ToolCallOut]


class TraceEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    type: str
    at: str
    call_id: str | None
    data: dict[str, Any]

    @classmethod
    def from_row(cls, row: TraceEventRow) -> TraceEventOut:
        return cls.model_validate(row)


class TraceOut(BaseModel):
    run_id: str
    events: list[TraceEventOut]
    next_after_id: int | None  # pass as ?after_id= to get the next page


class CacheCleared(BaseModel):
    session_id: str
    cleared_entries: int


class Health(BaseModel):
    status: str
    database: str
    gemini_configured: bool
    gemini_model: str | None
    pricing_configured: bool
    active_runs: int
    active_evaluations: int
    active_validations: int = 0
    active_learned_rules: int = 0


class EvaluationCreated(BaseModel):
    evaluation_id: str
    status: str
    planned_runs: int
    links: dict[str, str]


class EvaluationOut(BaseModel):
    evaluation_id: str
    status: str
    finished: bool
    provider: str
    dataset: dict[str, str]
    config: dict[str, Any]
    planned_runs: int
    completed_runs: int
    created_at: str
    started_at: str | None
    ended_at: str | None
    error: dict[str, str] | None
    report: dict[str, Any] | None  # only once completed

    @classmethod
    def from_row(cls, row: EvaluationRow, *, include_report: bool = True) -> EvaluationOut:
        return cls(
            evaluation_id=row.id,
            status=row.status,
            finished=row.status not in ("queued", "running"),
            provider=row.provider,
            dataset={
                "name": row.dataset_name,
                "version": row.dataset_version,
                "checksum": row.dataset_checksum,
            },
            config=row.config,
            planned_runs=row.planned_runs,
            completed_runs=row.completed_runs,
            created_at=row.created_at,
            started_at=row.started_at,
            ended_at=row.ended_at,
            error=(
                {"code": row.error_code, "message": row.error_message or ""}
                if row.error_code
                else None
            ),
            report=row.report if include_report else None,
        )


class EvaluationRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    seq: int
    task_id: str
    category: str
    split: str
    mode: str
    phase: str
    repetition: int
    run_id: str | None
    status: str
    correct: bool
    failed_checks: list[str]
    grade_checks: list[dict[str, Any]]
    final_answer: str | None
    tool_requests: int
    tool_executions: int
    latency_ms: float
    metrics: dict[str, Any]
    warmup_run_id: str | None = None
    warmup_correct: bool | None = None


class ActiveRuleRef(BaseModel):
    rule_id: str
    kind: str
    setting: dict[str, Any]


class ToolOut(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    effect: str
    freshness: str
    ttl_seconds: float | None  # the tool's own (developer) setting
    cache_scope: str
    requires_permission: bool
    timeout_seconds: float
    max_ttl_seconds: float | None = None
    # After ACTIVE learned rules (equal to the tool's own setting when none apply).
    effective_ttl_seconds: float | None = None
    effective_cache_scope: str | None = None
    learned_rules: list[ActiveRuleRef] = []
    learning_eligible: bool = False


class ToolsOut(BaseModel):
    tools: list[ToolOut]


class DatasetTaskOut(BaseModel):
    id: str
    category: str
    split: str
    task: str
    tools: list[str]
    allow_side_effects: bool
    scripted_only: bool


class DatasetOut(BaseModel):
    name: str
    version: str
    checksum: str
    description: str
    tasks: list[DatasetTaskOut]


# ---------------------------------------------------------------- learning
class RuleOut(BaseModel):
    rule_id: str
    status: str
    status_reason: str | None
    created_at: str
    updated_at: str
    tool_name: str
    tool_version: str
    kind: str
    current_setting: dict[str, Any]
    proposed_setting: dict[str, Any]
    evidence: dict[str, Any]
    mining_id: str | None
    dataset: dict[str, str]
    latest_validation_id: str | None
    approved_at: str | None
    approved_by: str | None
    approval_note: str | None
    disabled_at: str | None

    @classmethod
    def from_row(cls, row: Any) -> RuleOut:
        return cls(
            rule_id=row.id,
            status=row.status,
            status_reason=row.status_reason,
            created_at=row.created_at,
            updated_at=row.updated_at,
            tool_name=row.tool_name,
            tool_version=row.tool_version,
            kind=row.kind,
            current_setting=row.current_setting,
            proposed_setting=row.proposed_setting,
            evidence=row.evidence,
            mining_id=row.mining_id,
            dataset={"name": row.dataset_name, "version": row.dataset_version},
            latest_validation_id=row.latest_validation_id,
            approved_at=row.approved_at,
            approved_by=row.approved_by,
            approval_note=row.approval_note,
            disabled_at=row.disabled_at,
        )


class RuleEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    at: str
    from_status: str | None
    to_status: str
    actor: str
    note: str | None


class ValidationOut(BaseModel):
    validation_id: str
    rule_id: str
    status: str
    finished: bool
    created_at: str
    ended_at: str | None
    config: dict[str, Any]
    base_rule_ids: list[str]
    result: dict[str, Any] | None
    error_message: str | None

    @classmethod
    def from_row(cls, row: Any, *, include_result: bool = True) -> ValidationOut:
        return cls(
            validation_id=row.id,
            rule_id=row.rule_id,
            status=row.status,
            finished=row.status != "running",
            created_at=row.created_at,
            ended_at=row.ended_at,
            config=row.config,
            base_rule_ids=row.base_rule_ids,
            result=(
                row.result
                if include_result
                else (
                    {
                        "passed": row.result.get("passed"),
                        "failed_criteria": row.result.get("failed_criteria"),
                        "arms": row.result.get("arms"),
                    }
                    if row.result
                    else None
                )
            ),
            error_message=row.error_message,
        )


class ShadowCounts(BaseModel):
    would_reuse: int
    matched: int
    mismatched: int


class RuleDetailOut(BaseModel):
    rule: RuleOut
    events: list[RuleEventOut]
    validations: list[ValidationOut]  # newest first, without per-task tables
    shadow_live: ShadowCounts | None  # from ordinary optimized runs while in shadow


class MiningOut(BaseModel):
    mining_id: str
    created_at: str
    evaluation_ids: list[str]
    thresholds: dict[str, Any]
    report: dict[str, Any]

    @classmethod
    def from_row(cls, row: Any) -> MiningOut:
        return cls(
            mining_id=row.id,
            created_at=row.created_at,
            evaluation_ids=row.evaluation_ids,
            thresholds=row.thresholds,
            report=row.report,
        )


class ValidationCreated(BaseModel):
    validation_id: str
    rule_id: str
    status: str
    links: dict[str, str]
