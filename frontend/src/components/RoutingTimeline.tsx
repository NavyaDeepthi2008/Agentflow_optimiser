import type { TraceEvent } from "../api/types";
import { describeRequest } from "../lib/decisions";
import { fmtInt, fmtMs } from "../lib/format";

interface Attempt {
  attempt: number;
  error_code: string;
  will_retry: boolean;
  retry_blocked_because: string | null;
}

interface ToolStep {
  callId: string;
  tool: string;
  args: Record<string, unknown> | null;
  decision: string;
  executions: number;
  errorCode: string | null;
  errorMessage: string | null;
  cacheAge: number | null;
  durationMs: number | null;
  invalidated: number;
  attempts: Attempt[];
  learnedRules: string[];
  shadow: { matched: boolean } | null;
}

interface LlmStep {
  step: number;
  ok: boolean;
  errorCode: string | null;
  latencyMs: number | null;
  inputTokens: number | null;
  outputTokens: number | null;
  tools: ToolStep[];
}

export interface Timeline {
  steps: LlmStep[];
  finalAnswer: string | null;
  finished: boolean;
}

type Data = Record<string, unknown>;
const num = (v: unknown): number | null => (typeof v === "number" ? v : null);
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);

/** Turn the stored trace events into LLM steps with their tool requests. */
export function buildTimeline(events: TraceEvent[]): Timeline {
  const ordered = [...events].sort((a, b) => a.id - b.id);
  const attempts = new Map<string, Attempt[]>();
  for (const e of ordered) {
    if (e.type === "tool_call.attempt_failed" && e.call_id) {
      const d = e.data as Data;
      const list = attempts.get(e.call_id) ?? [];
      list.push({
        attempt: num(d.attempt) ?? list.length + 1,
        error_code: str(d.error_code) ?? "error",
        will_retry: d.will_retry === true,
        retry_blocked_because: str(d.retry_blocked_because),
      });
      attempts.set(e.call_id, list);
    }
  }

  const steps: LlmStep[] = [];
  let finalAnswer: string | null = null;
  let finished = false;
  for (const e of ordered) {
    const d = e.data as Data;
    if (e.type === "llm_call.completed" || e.type === "llm_call.failed") {
      const usage = (d.usage ?? null) as Data | null;
      steps.push({
        step: num(d.step) ?? steps.length + 1,
        ok: e.type === "llm_call.completed",
        errorCode: str(d.error_code),
        latencyMs: num(d.latency_ms),
        inputTokens: usage ? num(usage.input_tokens) : null,
        outputTokens: usage ? num(usage.output_tokens) : null,
        tools: [],
      });
    } else if (e.type === "tool_call.completed") {
      const current = steps[steps.length - 1];
      const tool: ToolStep = {
        callId: e.call_id ?? String(e.id),
        tool: str(d.tool_name) ?? "unknown",
        args: (d.args ?? null) as Record<string, unknown> | null,
        decision: str(d.decision) ?? "unknown",
        executions: num(d.executions) ?? 0,
        errorCode: str(d.error_code),
        errorMessage: str(d.error_message),
        cacheAge: num(d.cache_age_seconds),
        durationMs: num(d.duration_ms),
        invalidated: num(d.invalidated_entries) ?? 0,
        attempts: attempts.get(e.call_id ?? "") ?? [],
        learnedRules: Array.isArray(d.learned_rules) ? (d.learned_rules as string[]) : [],
        shadow:
          d.shadow && typeof d.shadow === "object"
            ? { matched: Boolean((d.shadow as { matched?: unknown }).matched) }
            : null,
      };
      if (current) current.tools.push(tool);
    } else if (e.type === "run.completed") {
      finished = true;
      finalAnswer = str(d.final_answer);
    }
  }
  return { steps, finalAnswer, finished };
}

function argsSummary(args: Record<string, unknown> | null): string {
  if (!args) return "";
  return Object.entries(args)
    .filter(([, v]) => v !== null && v !== undefined)
    .map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`)
    .join(", ");
}

function ToolRoute({ t }: { t: ToolStep }) {
  const outcome = describeRequest({
    decision: t.decision,
    executions: t.executions,
    error_code: t.errorCode,
    error_message: t.errorMessage,
    cache_age_seconds: t.cacheAge,
  });
  return (
    <div className={`route ${outcome.route}`} data-testid="tool-route" data-route={outcome.route}>
      <div className="route-name">
        {t.tool}
        <small>{argsSummary(t.args)}</small>
      </div>
      <div>
        <div className="track" aria-hidden="true">
          <span className="seg" />
          <span className="node agent" />
          <span className="node gate" />
          <span className="node tool" />
          {outcome.route === "reused" && <span className="loop" />}
          {outcome.route === "blocked" && <span className="stop" />}
        </div>
        <div className="outcome">
          <b>{outcome.label}</b>
          {outcome.detail && <span className="extra">{outcome.detail}</span>}
          {outcome.retries > 0 && (
            <span className="flag">
              Retried {outcome.retries} {outcome.retries === 1 ? "time" : "times"}
            </span>
          )}
          {outcome.timedOut && outcome.route !== "failed" && <span className="flag">Timed out</span>}
          {t.invalidated > 0 && (
            <span className="extra">
              Cleared {t.invalidated} stored {t.invalidated === 1 ? "result" : "results"}
            </span>
          )}
          {t.learnedRules.length > 0 && (
            <span className="flag learned" title={`Learned rule ${t.learnedRules.join(", ")}`}>
              Reused because of an approved learned rule
            </span>
          )}
          {t.shadow && (
            <span className="extra">
              Shadow rule would have reused a {t.shadow.matched ? "matching" : "DIFFERENT"} result
            </span>
          )}
          {t.durationMs !== null && <span className="extra">{fmtMs(t.durationMs)}</span>}
        </div>
        {t.attempts.length > 0 && (
          <ul className="attempts">
            {t.attempts.map((a) => (
              <li key={a.attempt}>
                Attempt {a.attempt} failed ({a.error_code})
                {a.will_retry ? ", retrying" : a.retry_blocked_because ? `, not retried: ${a.retry_blocked_because}` : ""}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

export function RoutingTimeline({ timeline }: { timeline: Timeline }) {
  if (timeline.steps.length === 0) {
    return <p className="hint">No steps recorded yet.</p>;
  }
  const last = timeline.steps[timeline.steps.length - 1];
  return (
    <ol className="timeline" aria-label="Run timeline">
      {timeline.steps.map((s) => (
        <li className="step" key={s.step}>
          <div className="step-meta">
            <strong>Step {s.step}</strong>
            <span>LLM {fmtMs(s.latencyMs)}</span>
            <span>
              {s.inputTokens === null && s.outputTokens === null
                ? "tokens not reported"
                : `${fmtInt(s.inputTokens)} in, ${fmtInt(s.outputTokens)} out`}
            </span>
          </div>
          <div className="step-body">
            {!s.ok && <p className="notice error">LLM call failed ({s.errorCode ?? "unknown error"})</p>}
            {s.tools.map((t) => (
              <ToolRoute key={t.callId} t={t} />
            ))}
            {s.ok && s.tools.length === 0 && s === last && timeline.finished && (
              <div className="step-answer">
                {timeline.finalAnswer ?? <span className="hint">No final answer</span>}
              </div>
            )}
          </div>
        </li>
      ))}
    </ol>
  );
}
