import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { ApiError } from "../api/client";

// ---------------------------------------------------------------- status

const STATUS: Record<string, { text: string; tone: "ok" | "warn" | "bad" | "active" | "" }> = {
  queued: { text: "Queued", tone: "active" },
  running: { text: "Running", tone: "active" },
  success: { text: "Finished", tone: "ok" },
  failed: { text: "Failed", tone: "bad" },
  error: { text: "Internal error", tone: "bad" },
  interrupted: { text: "Interrupted", tone: "warn" },
  limit_steps: { text: "Stopped: step limit", tone: "warn" },
  limit_tool_calls: { text: "Stopped: tool request limit", tone: "warn" },
  limit_time: { text: "Stopped: time limit", tone: "warn" },
  loop_detected: { text: "Stopped: repeated request", tone: "warn" },
  completed: { text: "Completed", tone: "ok" },
  // learned rules and their validations
  proposed: { text: "Proposed", tone: "" },
  shadow: { text: "Shadow (observing)", tone: "active" },
  validating: { text: "Validating", tone: "active" },
  validated: { text: "Validated: awaiting approval", tone: "warn" },
  rejected: { text: "Rejected", tone: "bad" },
  active: { text: "Active", tone: "ok" },
  disabled: { text: "Disabled", tone: "" },
  superseded: { text: "Superseded", tone: "" },
  passed: { text: "Passed", tone: "ok" },
};

export function StatusBadge({ status }: { status: string }) {
  const s = STATUS[status] ?? { text: status, tone: "" };
  return (
    <span className={`status ${s.tone}`} data-status={status}>
      {s.text}
    </span>
  );
}

export function isActive(status: string): boolean {
  return status === "queued" || status === "running" || status === "validating";
}

// ---------------------------------------------------------------- provider

export function AgentTag({ provider, model }: { provider: string; model?: string | null }) {
  if (provider === "scripted") {
    return (
      <span className="tag mock" title="Scripted mock agent: deterministic, not a real model">
        MOCK
      </span>
    );
  }
  return (
    <span className="tag live-model" title={model ? `Real model: ${model}` : "Real model"}>
      LIVE{model ? ` ${model}` : ""}
    </span>
  );
}

export function modeName(mode: string): string {
  return mode === "optimized" ? "Optimized" : mode === "passthrough" ? "Baseline" : mode;
}

// ---------------------------------------------------------------- states

export function Loading({ what }: { what: string }) {
  return (
    <p className="loading" role="status">
      Loading {what}
    </p>
  );
}

export function ErrorNotice({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  return (
    <div className="notice error" role="alert">
      <strong>{error.isNetwork ? "Can't reach the backend" : "Request failed"}</strong>
      <span>{error.message}</span>
      {error.details.length > 0 && (
        <ul className="attempts">
          {error.details.map((d, i) => (
            <li key={i}>
              {d.loc.filter((p) => p !== "body").join(".") || "request"}: {d.msg}
            </li>
          ))}
        </ul>
      )}
      {onRetry && (
        <div>
          <button className="quiet" type="button" onClick={onRetry}>
            Try again
          </button>
        </div>
      )}
    </div>
  );
}

export function Empty({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty">
      <strong>{title}</strong>
      {children}
      {action}
    </div>
  );
}

export function NotFound() {
  return (
    <Empty title="Page not found" action={<Link className="button quiet" to="/">Go to the run launcher</Link>}>
      <p>The address doesn't match any page in the dashboard.</p>
    </Empty>
  );
}

// ---------------------------------------------------------------- facts

export interface Fact {
  label: string;
  value: string;
  note?: string;
}

export function Facts({ items }: { items: Fact[] }) {
  return (
    <dl className="facts">
      {items.map((f) => (
        <div key={f.label}>
          <dt>{f.label}</dt>
          <dd>
            {f.value}
            {f.note && <small>{f.note}</small>}
          </dd>
        </div>
      ))}
    </dl>
  );
}

// ---------------------------------------------------------------- mock vs live

export const MOCK_DISCLAIMER =
  "These numbers show what the gateway does with tool calls. They do not show what a real Gemini agent would save, cost or answer.";

/** Says plainly whether results come from the scripted mock agent or a real model. */
export function ResultsLabel({ provider, model }: { provider: string; model?: string | null }) {
  const mock = provider === "scripted";
  return (
    <div className={`label-banner ${mock ? "mock" : "live-model"}`} role="note" data-testid="results-label">
      <strong>{mock ? "MOCK results" : `LIVE model results${model ? ` (${model})` : ""}`}</strong>
      <span>
        {mock
          ? `Scripted mock agent replaying fixed tool calls against mock tools. ${MOCK_DISCLAIMER}`
          : "A real model against mock tools. Model output varies from run to run, so one comparison is a single sample."}
      </span>
    </div>
  );
}
