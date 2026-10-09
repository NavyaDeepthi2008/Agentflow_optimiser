// Types for the AgentFlow API.
//
// Everything except the evaluation report comes straight from the backend's
// OpenAPI schema (src/api/schema.gen.ts, regenerated with `npm run gen:api`).
// The report is a free-form JSON object on the backend, so its shape is
// described by hand below; every metric that can be unknown is `| null`.
import type { components } from "./schema.gen";

type Schemas = components["schemas"];

export type Health = Schemas["Health"];
export type RunOut = Schemas["RunOut"];
export type RunMetrics = Schemas["RunMetrics"];
export type RunDetails = Schemas["RunDetailsOut"];
export type ToolCall = Schemas["ToolCallOut"];
export type LLMCall = Schemas["LLMCallOut"];
export type TraceEvent = Schemas["TraceEventOut"];
export type TracePage = Schemas["TraceOut"];
export type Tool = Schemas["ToolOut"];
export type Dataset = Schemas["DatasetOut"];
export type DatasetTask = Schemas["DatasetTaskOut"];
export type RunCreated = Schemas["RunCreated"];
export type RunRequest = Schemas["RunRequest"];
export type EvaluationOut = Schemas["EvaluationOut"];
export type EvaluationRun = Schemas["EvaluationRunOut"];
export type EvaluationCreated = Schemas["EvaluationCreated"];
export type EvaluationConfig = Schemas["EvaluationConfig"];
export type CacheCleared = Schemas["CacheCleared"];
export type RuleOut = Schemas["RuleOut"];
export type RuleDetail = Schemas["RuleDetailOut"];
export type RuleEvent = Schemas["RuleEventOut"];
export type ValidationOut = Schemas["ValidationOut"];
export type MiningOut = Schemas["MiningOut"];
export type ValidationCreated = Schemas["ValidationCreated"];
export type ValidationConfig = Schemas["ValidationConfig"];
export type MineRequest = Schemas["MineRequest"];

export type Mode = "passthrough" | "optimized";

export interface ModeSummary {
  runs: number;
  correct: number;
  accuracy: number | null;
  status_counts: Record<string, number>;
  tool_requests: number;
  tool_executions: number;
  cache_hits: number;
  reused_in_run: number;
  shared_in_flight: number;
  tool_retries: number;
  llm_calls: number;
  tokens: {
    total: number | null;
    input: number | null;
    output: number | null;
    thinking: number | null;
    runs_with_usage: number;
    complete: boolean;
  };
  latency_ms: { median: number | null; p95: number | null; min: number | null; max: number | null };
  gateway_overhead_ms: { median: number | null; p95: number | null };
  llm_cost: number | null;
  tool_cost: number | null;
}

export interface Comparison {
  accuracy_change: number | null;
  tool_requests_change_pct: number | null;
  tool_executions_change_pct: number | null;
  llm_calls_change_pct: number | null;
  tokens_change_pct: number | null;
  latency_median_change_pct: number | null;
  latency_p95_change_pct: number | null;
  cost_change_pct: number | null;
}

export interface CategorySummary {
  runs: number;
  accuracy: number | null;
  tool_executions: number;
  tool_requests: number;
}

export interface PhaseReport {
  modes: Partial<Record<Mode, ModeSummary>>;
  comparison?: Comparison;
  by_category: Record<string, Partial<Record<Mode, CategorySummary>>>;
}

export interface AccuracyChange {
  task_id: string;
  phase: string;
  repetition: number;
  baseline_run_id: string;
  optimized_run_id: string;
  optimized_failed_checks?: string[];
  baseline_failed_checks?: string[];
}

export interface PerTaskModeRow {
  correct: number;
  runs: number;
  tool_executions_median: number | null;
  tool_requests_median: number | null;
  latency_ms_median: number | null;
}

export interface PerTaskRow {
  task_id: string;
  phase: string;
  category: string;
  passthrough?: PerTaskModeRow;
  optimized?: PerTaskModeRow;
}

export interface EvaluationReport {
  evaluation_id: string;
  label: string;
  provider: string;
  model_id: string;
  dataset: { name: string; version: string; checksum: string };
  config: Record<string, unknown>;
  pricing: { source: string; retrieved_on: string; currency: string } | null;
  fairness: Record<string, string | string[]>;
  notes: string[];
  phases: Record<string, PhaseReport>;
  regressions: AccuracyChange[];
  improvements: AccuracyChange[];
  per_task: PerTaskRow[];
  skipped_tasks: { task_id: string; phase: string; reason: string }[];
}

// ---------------------------------------------------------------- learning
// Evidence, mining reports and validation results are free-form JSON on the
// backend (app/learning), so their shapes are written out here by hand.

export interface EvidencePair {
  tool_name: string;
  run_id: string;
  first_run_id: string;
  session_id: string;
  task_id: string;
  gap_seconds: number;
  same_output: boolean;
}

export interface Evidence {
  supporting_run_ids: string[];
  supporting_task_ids: string[];
  evaluation_ids: string[];
  repeat_count: number;
  distinct_runs: number;
  consistency: number | null;
  observed_repeats: number;
  changed_repeats: number;
  max_unchanged_gap_seconds: number | null;
  min_changed_gap_seconds: number | null;
  capped_by_ceiling: boolean;
  sample: EvidencePair[];
  reasons?: string[];
  thresholds?: { min_support: number; min_distinct_runs: number; required_consistency: number };
}

export interface Candidate {
  tool_name: string;
  tool_version: string;
  kind: "ttl" | "cache_scope";
  current_setting: Setting;
  proposed_setting: Setting | null;
  proposed: boolean;
  reasons: string[];
  evidence: Evidence | null;
}

export interface Setting {
  ttl_seconds?: number | null;
  cache_scope?: string | null;
}

export interface MiningReport {
  thresholds: { min_support: number; min_distinct_runs: number; required_consistency: number };
  runs_examined: number;
  runs_used: number;
  excluded: Record<string, number>;
  candidates: Candidate[];
  not_eligible: { tool_name: string; reason: string }[];
  created_rule_ids?: string[];
  duplicates?: { tool_name: string; kind: string; rule_id: string }[];
}

export interface Criterion {
  name: string;
  label: string;
  passed: boolean;
  detail: string;
}

export interface ArmSummary {
  runs: number;
  correct: number;
  accuracy: number | null;
  tool_executions: number;
  tool_executions_incl_warmups: number;
  tool_requests: number;
  reused_results: number;
  rule_applied: number;
}

export interface ArmRow {
  run_id: string;
  warmup_run_id: string | null;
  correct: boolean;
  status: string;
  tool_executions: number;
  tool_requests: number;
  final_answer: string | null;
}

export interface ValidationTaskRow {
  task_id: string;
  phase: string;
  repetition: number;
  category: string;
  off: ArmRow;
  on: ArmRow;
  rule_applied: number;
}

export interface ValidationResult {
  passed: boolean;
  failed_criteria: string[];
  criteria?: Criterion[];
  arms: { off: ArmSummary; on: ArmSummary };
  regressions?: { task_id: string; phase: string; repetition: number; failed_checks: string[]; off_run_id: string; on_run_id: string }[];
  safety_violations?: string[];
  shadow?: { would_reuse: number; matched: number; mismatched: number };
  tasks?: ValidationTaskRow[];
  skipped?: { task_id: string; phase: string; reason: string }[];
  method?: Record<string, string | number>;
}
