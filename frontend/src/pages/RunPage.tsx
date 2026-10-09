import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useBackend } from "../App";
import { ApiError, api } from "../api/client";
import type { RunOut, RunRequest } from "../api/types";
import { RoutingTimeline, buildTimeline } from "../components/RoutingTimeline";
import { RunSummary } from "../components/RunSummary";
import { ErrorNotice, Facts, Loading, isActive } from "../components/ui";
import { fmtTime, shortId } from "../lib/format";
import { useRun } from "../lib/useRun";

export function RunPage() {
  const { runId = "" } = useParams();
  return <RunView key={runId} runId={runId} />;
}

function RunView({ runId }: { runId: string }) {
  const { run, details, trace } = useRun(runId);
  const { health } = useBackend();

  if (run.error) {
    return (
      <>
        <header className="page-head">
          <h1>Run {shortId(runId)}</h1>
        </header>
        {run.error.status === 404 ? (
          <div className="notice warn" role="alert">
            <strong>Run not found</strong>
            <span>There is no run with this id. It may belong to a different database.</span>
            <Link to="/runs">Open the run history</Link>
          </div>
        ) : (
          <ErrorNotice error={run.error} onRetry={run.reload} />
        )}
      </>
    );
  }
  if (!run.data) return <Loading what="run" />;
  const r = run.data;

  return (
    <>
      <header className="page-head">
        <h1>Run {shortId(r.run_id)}</h1>
        <p>{r.task}</p>
      </header>

      <section aria-label="Result">
        <RunSummary run={r} toolCalls={details.data?.tool_calls} pricingConfigured={!!health?.pricing_configured} />
        {details.error && <ErrorNotice error={details.error} onRetry={details.reload} />}
      </section>

      <section aria-labelledby="timeline-title">
        <div className="section-head">
          <h2 id="timeline-title">What happened to each request</h2>
          <span className="hint">
            {isActive(r.status) ? "Updating while the run is active" : "Recorded by the gateway"}
          </span>
        </div>
        <div className="panel">
          {trace.error ? (
            <ErrorNotice error={trace.error} onRetry={trace.reload} />
          ) : trace.data ? (
            <RoutingTimeline timeline={buildTimeline(trace.data)} />
          ) : (
            <Loading what="timeline" />
          )}
        </div>
      </section>

      <section aria-labelledby="context-title">
        <div className="section-head">
          <h2 id="context-title">Session and settings</h2>
        </div>
        <SessionActions run={r} />
        <div style={{ marginTop: 18 }}>
          <Facts
            items={[
              { label: "Session", value: r.session_id },
              { label: "Created", value: fmtTime(r.created_at) },
              { label: "Ended", value: fmtTime(r.ended_at) },
              { label: "Step limit", value: String(r.config_snapshot.max_steps ?? "—") },
              { label: "Tool request limit", value: String(r.config_snapshot.max_tool_requests ?? "—") },
              { label: "Model", value: r.model_id },
            ]}
          />
        </div>
      </section>
    </>
  );
}

function SessionActions({ run }: { run: RunOut }) {
  const navigate = useNavigate();
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string>();
  const [error, setError] = useState<ApiError>();
  const scenario = run.request_options.scenario as { task_id?: string } | undefined;
  const llm = run.provider === "scripted" ? "scripted" : "gemini";

  async function runAgain() {
    setBusy(true);
    setError(undefined);
    const body: RunRequest = scenario?.task_id
      ? { scenario: { task_id: scenario.task_id }, llm }
      : { task: run.task, llm: "gemini" };
    try {
      const created = await api.createRun({ ...body, mode: run.mode as RunRequest["mode"], session_id: run.session_id });
      navigate(`/runs/${created.run_id}`);
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(-1, "client_error", String(err)));
      setBusy(false);
    }
  }

  async function clearCache() {
    setBusy(true);
    setError(undefined);
    try {
      const result = await api.clearCache(run.session_id);
      setMessage(
        result.cleared_entries === 0
          ? "The session's cache was already empty."
          : `Cleared ${result.cleared_entries} stored ${result.cleared_entries === 1 ? "result" : "results"}.`,
      );
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(-1, "client_error", String(err)));
    } finally {
      setBusy(false);
    }
  }

  const canRerun = !isActive(run.status) && (scenario?.task_id || run.provider !== "scripted");
  return (
    <div style={{ display: "grid", gap: 10 }}>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        {canRerun && (
          <button type="button" onClick={runAgain} disabled={busy}>
            Run again in this session
          </button>
        )}
        {run.mode === "optimized" && (
          <button type="button" className="quiet" onClick={clearCache} disabled={busy}>
            Clear this session's cache
          </button>
        )}
      </div>
      <p className="hint">
        {run.mode === "optimized"
          ? "Running again in the same session can reuse results this run stored (a warm cache)."
          : "Baseline runs never reuse results, even in the same session."}
      </p>
      {message && <p role="status">{message}</p>}
      {error && <ErrorNotice error={error} />}
    </div>
  );
}
