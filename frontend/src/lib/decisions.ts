// What happened to a tool request, in plain words, derived only from the
// gateway's recorded decision, executions and error code.
import type { ToolCall } from "../api/types";
import { MISSING } from "./format";

export type Route = "live" | "reused" | "failed" | "blocked";

export interface RequestOutcome {
  route: Route;
  label: string;
  detail: string | null;
  retries: number;
  timedOut: boolean;
}

const BLOCK_REASONS: Record<string, string> = {
  blocked_permission: "Blocked: the task is not allowed to use this tool",
  blocked_invalid_args: "Blocked: invalid arguments",
  blocked_unknown_tool: "Blocked: no such tool",
  blocked_limit: "Blocked: a run limit was reached",
  blocked_run_stopped: "Blocked: the run had already stopped",
  blocked_safety_error: "Blocked: a safety check failed",
};

function age(seconds: number | null | undefined): string {
  if (seconds == null) return "";
  if (seconds < 1) return " (under 1 s old)";
  return ` (${Math.round(seconds)} s old)`;
}

export function describeRequest(call: Pick<
  ToolCall,
  "decision" | "executions" | "error_code" | "error_message" | "cache_age_seconds"
>): RequestOutcome {
  const retries = Math.max(0, call.executions - 1);
  const timedOut = (call.error_code ?? "").startsWith("timeout");
  const base = { retries, timedOut };
  switch (call.decision) {
    case "executed_live":
      return { ...base, route: "live", label: "Executed live", detail: null };
    case "stale_refreshed":
      return { ...base, route: "live", label: "Executed live",
               detail: "A stored result existed but was too old, so it was refreshed" };
    case "forced_fresh":
      return { ...base, route: "live", label: "Executed live",
               detail: "A fresh result was requested" };
    case "cache_hit":
      return { ...base, route: "reused", label: "Cache hit",
               detail: `Result stored by an earlier run${age(call.cache_age_seconds)}` };
    case "reused_in_run":
      return { ...base, route: "reused", label: "Reused",
               detail: `Same request earlier in this run${age(call.cache_age_seconds)}` };
    case "shared_in_flight":
      return { ...base, route: "reused", label: "Shared",
               detail: "Joined an identical request that was already running" };
    case "failed": {
      const label = timedOut ? "Timed out" : "Failed";
      return { ...base, route: "failed", label,
               detail: call.error_message ?? call.error_code ?? MISSING };
    }
    default:
      if (call.decision.startsWith("blocked_")) {
        return { ...base, route: "blocked",
                 label: BLOCK_REASONS[call.decision] ?? "Blocked",
                 detail: call.error_message ?? null };
      }
      return { ...base, route: "failed", label: call.decision, detail: call.error_message ?? null };
  }
}

export interface RequestTally {
  requested: number;
  executions: number;
  live: number;
  reused: number;
  failed: number;
  blocked: number;
  retries: number;
}

export function tallyRequests(calls: ToolCall[]): RequestTally {
  const tally: RequestTally = {
    requested: calls.length, executions: 0, live: 0, reused: 0, failed: 0, blocked: 0, retries: 0,
  };
  for (const call of calls) {
    const outcome = describeRequest(call);
    tally[outcome.route] += 1;
    tally.executions += call.executions;
    tally.retries += outcome.retries;
  }
  return tally;
}
