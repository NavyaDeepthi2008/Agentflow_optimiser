import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { RunOut } from "../api/types";
import { AgentTag, Empty, ErrorNotice, Loading, StatusBadge, isActive, modeName } from "../components/ui";
import { fmtInt, fmtMs, fmtTime, truncate } from "../lib/format";
import { useResource } from "../lib/useResource";

const STATUSES = [
  "", "running", "queued", "success", "failed", "loop_detected", "limit_steps",
  "limit_tool_calls", "limit_time", "interrupted", "error",
];
const LIMIT = 100;

export function HistoryPage() {
  const [status, setStatus] = useState("");
  const [session, setSession] = useState("");
  const [sessionFilter, setSessionFilter] = useState("");
  const runs = useResource(
    () => api.listRuns({ status: status || undefined, session_id: sessionFilter || undefined, limit: LIMIT }),
    [status, sessionFilter],
    { pollMs: 3000, pollWhile: (list: RunOut[]) => list.some((r) => isActive(r.status)) },
  );

  return (
    <>
      <header className="page-head">
        <h1>Run history</h1>
        <p>Every run stored by the backend, newest first, including the runs made by evaluations.</p>
      </header>

      <form
        className="form-grid"
        style={{ marginBottom: 20 }}
        onSubmit={(e) => {
          e.preventDefault();
          setSessionFilter(session.trim());
        }}
      >
        <div className="field">
          <label htmlFor="status-filter">Status</label>
          <select id="status-filter" value={status} onChange={(e) => setStatus(e.target.value)}>
            {STATUSES.map((s) => (
              <option key={s} value={s}>{s === "" ? "Any status" : s}</option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="session-filter">Session</label>
          <input
            id="session-filter"
            type="text"
            value={session}
            pattern="[A-Za-z0-9_\-]{1,64}"
            placeholder="Any session"
            onChange={(e) => setSession(e.target.value)}
          />
        </div>
        <div>
          <button type="submit" className="quiet">Apply filters</button>
        </div>
      </form>

      {runs.error ? (
        <ErrorNotice error={runs.error} onRetry={runs.reload} />
      ) : !runs.data ? (
        <Loading what="runs" />
      ) : runs.data.length === 0 ? (
        <Empty
          title={status || sessionFilter ? "No runs match these filters" : "No runs yet"}
          action={<Link className="button" to="/">Run a task</Link>}
        >
          <p>Runs appear here as soon as they are started.</p>
        </Empty>
      ) : (
        <>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Started</th>
                  <th>Task</th>
                  <th>Mode</th>
                  <th>Agent</th>
                  <th>Status</th>
                  <th className="num">Requests</th>
                  <th className="num">Executions</th>
                  <th className="num">Run time</th>
                </tr>
              </thead>
              <tbody>
                {runs.data.map((r) => (
                  <tr key={r.run_id}>
                    <td>{fmtTime(r.created_at)}</td>
                    <td><Link to={`/runs/${r.run_id}`}>{truncate(r.task, 70)}</Link></td>
                    <td>{modeName(r.mode)}</td>
                    <td><AgentTag provider={r.provider} /></td>
                    <td><StatusBadge status={r.status} /></td>
                    <td className="num">{fmtInt(r.metrics?.tool_requests)}</td>
                    <td className="num">{fmtInt(r.metrics?.tool_executions)}</td>
                    <td className="num">{fmtMs(r.metrics?.duration_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {runs.data.length === LIMIT && <p className="hint" style={{ marginTop: 10 }}>Showing the latest {LIMIT} runs.</p>}
        </>
      )}
    </>
  );
}
