import type { RunOut, ToolCall } from "../api/types";
import { fmtCost, fmtInt, fmtMs } from "../lib/format";
import { RequestBar } from "./RequestBar";
import { AgentTag, type Fact, Facts, StatusBadge, isActive, modeName } from "./ui";

/** The core facts of one run. Metrics only appear once the run has finished;
 *  values the backend did not report are shown as "—", never as zero. */
export function RunSummary({
  run,
  toolCalls,
  pricingConfigured,
}: {
  run: RunOut;
  toolCalls: ToolCall[] | undefined;
  pricingConfigured: boolean;
}) {
  const m = run.metrics;
  const facts: Fact[] = m
    ? [
        { label: "LLM calls", value: fmtInt(m.llm_calls) },
        tokensFact(m.input_tokens, m.output_tokens, m.thinking_tokens, m.usage_complete),
        { label: "Run time", value: fmtMs(m.duration_ms) },
        { label: "Time in LLM", value: fmtMs(m.llm_time_ms) },
        { label: "Time in tools", value: fmtMs(m.tool_time_ms), note: "incl. gateway" },
        { label: "Reused results", value: fmtInt(m.reused_results) },
      ]
    : [];
  if (m && pricingConfigured) {
    facts.push(
      run.estimated_cost
        ? { label: "Estimated cost", value: fmtCost(run.estimated_cost.llm, run.estimated_cost.currency),
            note: "LLM tokens only" }
        : { label: "Estimated cost", value: "—", note: "no complete usage reported" },
    );
  }

  return (
    <div className="run-summary" style={{ display: "grid", gap: 18 }}>
      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <StatusBadge status={run.status} />
        <span className="tag neutral">{modeName(run.mode)}</span>
        <AgentTag provider={run.provider} model={run.provider === "scripted" ? null : run.model_id} />
      </div>

      {run.status === "interrupted" && (
        <div className="notice warn" role="status">
          <strong>This run was interrupted</strong>
          <span>The server stopped before the run finished, so it has no final answer or metrics.</span>
        </div>
      )}
      {run.error && run.status !== "interrupted" && (
        <div className="notice error">
          <strong>{run.error.code}</strong>
          <span>{run.error.message}</span>
        </div>
      )}

      {isActive(run.status) ? (
        <p className="hint">Results appear when the run finishes.</p>
      ) : (
        <p className={run.final_answer ? "answer" : "answer none"}>
          {run.final_answer ?? "No final answer"}
        </p>
      )}

      {toolCalls && !isActive(run.status) && <RequestBar calls={toolCalls} />}
      {facts.length > 0 && <Facts items={facts} />}
    </div>
  );
}

function tokensFact(
  input: number | null,
  output: number | null,
  thinking: number | null,
  complete: boolean | null,
): Fact {
  if (input === null && output === null) {
    return { label: "Tokens", value: "—", note: "not reported by this agent" };
  }
  const parts = [`${fmtInt(input)} in`, `${fmtInt(output)} out`];
  if (thinking !== null) parts.push(`${fmtInt(thinking)} thinking`);
  const total = input !== null && output !== null ? input + output + (thinking ?? 0) : null;
  return {
    label: "Tokens",
    value: fmtInt(total),
    note: parts.join(", ") + (complete ? "" : " (partial: some calls did not report usage)"),
  };
}
