import type { ToolCall } from "../api/types";
import { tallyRequests } from "../lib/decisions";

const SEGMENTS = [
  { key: "live", label: "executed live" },
  { key: "reused", label: "reused stored result" },
  { key: "failed", label: "tool reported an error" },
  { key: "blocked", label: "blocked before the tool" },
] as const;

/** The headline of a run: how many tool requests there were, and how many
 *  actually ran tool code. Built only from recorded gateway decisions. */
export function RequestBar({ calls }: { calls: ToolCall[] }) {
  const t = tallyRequests(calls);
  if (t.requested === 0) {
    return <p className="hint">The agent made no tool requests in this run.</p>;
  }
  return (
    <div className="ledger">
      <div className="ledger-headline">
        <span className="figure" aria-label="tool requests">
          {t.requested}
        </span>
        <span className="of">{t.requested === 1 ? "tool request" : "tool requests"}, tool code ran</span>
        <span className="figure" aria-label="tool executions">
          {t.executions}
        </span>
        <span className="of">
          {t.executions === 1 ? "time" : "times"}
          {t.retries > 0 && ` (including ${t.retries} ${t.retries === 1 ? "retry" : "retries"})`}
        </span>
      </div>
      <div className="bar" role="img" aria-label={describe(t)}>
        {SEGMENTS.map(({ key }) =>
          t[key] > 0 ? (
            <span key={key} className={key} style={{ width: `${(t[key] / t.requested) * 100}%` }} />
          ) : null,
        )}
      </div>
      <ul className="legend">
        {SEGMENTS.filter(({ key }) => t[key] > 0).map(({ key, label }) => (
          <li key={key}>
            <i className={key} />
            {t[key]} {label}
          </li>
        ))}
      </ul>
    </div>
  );
}

function describe(t: ReturnType<typeof tallyRequests>): string {
  return SEGMENTS.filter(({ key }) => t[key] > 0)
    .map(({ key, label }) => `${t[key]} ${label}`)
    .join(", ");
}
