import { describe, expect, it } from "vitest";
import retryRun from "../test/fixtures/retry_run.json";
import optimizedRun from "../test/fixtures/optimized_run.json";
import blockedRun from "../test/fixtures/blocked_run.json";
import type { ToolCall } from "../api/types";
import { describeRequest, tallyRequests } from "./decisions";
import { MISSING, fmtChange, fmtCost, fmtInt, fmtMs, fmtPct } from "./format";

describe("formatting never turns missing values into zero", () => {
  it("shows a dash for null and undefined", () => {
    for (const fn of [fmtInt, fmtMs, fmtPct, fmtChange, fmtCost]) {
      expect(fn(null)).toBe(MISSING);
      expect(fn(undefined)).toBe(MISSING);
    }
  });

  it("keeps a real zero as zero", () => {
    expect(fmtInt(0)).toBe("0");
    expect(fmtPct(0)).toBe("0%");
    expect(fmtMs(0)).toBe("0.00 ms");
  });

  it("formats numbers readably", () => {
    expect(fmtInt(1234)).toBe("1,234");
    expect(fmtMs(9.534)).toBe("9.53 ms");
    expect(fmtMs(250)).toBe("250 ms");
    expect(fmtMs(1500)).toBe("1.50 s");
    expect(fmtPct(0.925, 1)).toBe("92.5%");
    expect(fmtChange(-27.5)).toBe("−27.5%");
    expect(fmtChange(1.7)).toBe("+1.7%");
    expect(fmtChange(0.01)).toBe("±0.0%");
    expect(fmtCost(0.0003)).toBe("0.000300 USD");
  });
});

describe("request outcomes", () => {
  const base = { executions: 1, error_code: null, error_message: null, cache_age_seconds: null };

  it.each([
    ["executed_live", "live", "Executed live"],
    ["stale_refreshed", "live", "Executed live"],
    ["forced_fresh", "live", "Executed live"],
    ["cache_hit", "reused", "Cache hit"],
    ["reused_in_run", "reused", "Reused"],
    ["shared_in_flight", "reused", "Shared"],
    ["failed", "failed", "Failed"],
    ["blocked_permission", "blocked", "Blocked: the task is not allowed to use this tool"],
    ["blocked_limit", "blocked", "Blocked: a run limit was reached"],
  ])("%s is shown as %s", (decision, route, label) => {
    const outcome = describeRequest({ ...base, decision, executions: route === "live" || route === "failed" ? 1 : 0 });
    expect(outcome.route).toBe(route);
    expect(outcome.label).toBe(label);
  });

  it("detects timeouts and retries", () => {
    const t = describeRequest({ ...base, decision: "failed", executions: 3, error_code: "timeout" });
    expect(t.label).toBe("Timed out");
    expect(t.retries).toBe(2);
    expect(t.timedOut).toBe(true);
  });

  it("tallies real recorded runs", () => {
    const opt = tallyRequests(optimizedRun.details.tool_calls as ToolCall[]);
    expect(opt).toMatchObject({ requested: 4, executions: 2, live: 2, reused: 2, blocked: 0 });
    const retry = tallyRequests(retryRun.details.tool_calls as ToolCall[]);
    expect(retry).toMatchObject({ requested: 2, executions: 3, retries: 1 });
    const blocked = tallyRequests(blockedRun.details.tool_calls as ToolCall[]);
    expect(blocked).toMatchObject({ requested: 1, executions: 0, blocked: 1 });
  });
});
