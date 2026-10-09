"""Data shapes for the learning engine (mining, evidence, validation)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Decisions where the tool really ran for this request (so the output is a
# genuine observation of the data at that moment).
LIVE_DECISIONS = frozenset({"executed_live", "stale_refreshed", "forced_fresh"})

# Hard floors: an API caller may ask for STRONGER evidence, never weaker.
MIN_SUPPORT_FLOOR = 3
MIN_DISTINCT_RUNS_FLOOR = 2


class Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_support: int = Field(
        default=5,
        ge=MIN_SUPPORT_FLOOR,
        le=1000,
        description="Minimum repeated requests that support the proposal.",
    )
    min_distinct_runs: int = Field(
        default=3,
        ge=MIN_DISTINCT_RUNS_FLOOR,
        le=1000,
        description="The support must come from at least this many different runs (or sessions).",
    )
    # Consistency is fixed at 100%: a single changed result rules a proposal out.
    required_consistency: Literal[1.0] = 1.0


class CallObservation(BaseModel):
    seq: int
    tool_name: str
    decision: str
    ok: bool
    executed: bool
    fingerprint: str | None
    output_digest: str | None
    data_time: float | None


class RunObservation(BaseModel):
    """One stored run, with what the miner needs to know about it."""

    run_id: str
    session_id: str
    evaluation_id: str
    task_id: str
    split: str
    mode: str
    phase: str  # cold | warm
    status: str
    correct: bool
    calls: list[CallObservation]
    warmup: RunObservation | None = None  # warm phase: the graded warm-up run
    warmup_correct: bool | None = None


class Pair(BaseModel):
    """Two live executions of the identical request."""

    tool_name: str
    run_id: str  # run of the second execution
    first_run_id: str
    session_id: str
    task_id: str
    gap_seconds: float
    same_output: bool


class Evidence(BaseModel):
    supporting_run_ids: list[str]
    supporting_task_ids: list[str]
    evaluation_ids: list[str]
    repeat_count: int  # repeated requests that support the proposal
    distinct_runs: int
    consistency: float | None  # share of supporting-range repeats with identical output
    observed_repeats: int  # all repeats of this tool examined
    changed_repeats: int
    max_unchanged_gap_seconds: float | None = None
    min_changed_gap_seconds: float | None = None
    capped_by_ceiling: bool = False
    sample: list[Pair] = Field(default_factory=list)


class Candidate(BaseModel):
    tool_name: str
    tool_version: str
    kind: Literal["ttl", "cache_scope"]
    current_setting: dict[str, Any]
    proposed_setting: dict[str, Any] | None
    proposed: bool
    reasons: list[str]  # why it was (not) proposed, in plain words
    evidence: Evidence | None = None


class MiningReport(BaseModel):
    thresholds: Thresholds
    runs_examined: int
    runs_used: int
    excluded: dict[str, int]  # reason -> number of runs
    candidates: list[Candidate]
    not_eligible: list[dict[str, str]]  # tools learning may never touch, and why


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: Literal["scripted", "gemini"] = "scripted"
    repetitions: int = Field(default=2, ge=1, le=5)
    seed: int = Field(default=0, ge=0)
