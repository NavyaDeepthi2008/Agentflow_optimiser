import { type FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useBackend } from "../App";
import { ApiError, api } from "../api/client";
import type { EvaluationConfig, EvaluationOut } from "../api/types";
import { AgentTag, Empty, ErrorNotice, Loading, StatusBadge, isActive } from "../components/ui";
import { fmtTime, shortId } from "../lib/format";
import { useResource } from "../lib/useResource";

export function EvaluationsPage() {
  const { health } = useBackend();
  const navigate = useNavigate();
  const list = useResource(() => api.listEvaluations(), [], {
    pollMs: 2000,
    pollWhile: (items: EvaluationOut[]) => items.some((e) => isActive(e.status)),
  });
  const [llm, setLlm] = useState<"scripted" | "gemini">("scripted");
  const [repetitions, setRepetitions] = useState(3);
  const [warm, setWarm] = useState(true);
  const [split, setSplit] = useState<"" | "train" | "heldout">("");
  const [seed, setSeed] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError>();

  async function start(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(undefined);
    const config: EvaluationConfig = {
      llm,
      repetitions,
      warm_cache: warm,
      seed,
      ...(split ? { split } : {}),
    };
    try {
      const created = await api.createEvaluation(config);
      navigate(`/evaluations/${created.evaluation_id}`);
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(-1, "client_error", String(err)));
      setBusy(false);
    }
  }

  return (
    <>
      <header className="page-head">
        <h1>Evaluations</h1>
        <p>
          Runs every task of the core dataset in baseline and optimized mode with the same agent,
          prompts, tools and limits, grades the answers against hidden expectations, and compares the
          results. Each measured run starts with an empty cache.
        </p>
      </header>

      <section aria-labelledby="new-eval">
        <div className="section-head"><h2 id="new-eval">Start an evaluation</h2></div>
        <form className="panel" onSubmit={start} style={{ display: "grid", gap: 16 }}>
          <div className="form-grid">
            <div className="field">
              <label htmlFor="eval-llm">Agent</label>
              <select id="eval-llm" value={llm} onChange={(e) => setLlm(e.target.value as "scripted" | "gemini")}>
                <option value="scripted">Mock agent (scripted)</option>
                <option value="gemini" disabled={!health?.gemini_configured}>
                  Gemini{health?.gemini_configured ? "" : " (not configured)"}
                </option>
              </select>
            </div>
            <div className="field">
              <label htmlFor="eval-reps">Repetitions</label>
              <input id="eval-reps" type="number" min={1} max={20} value={repetitions}
                onChange={(e) => setRepetitions(Number(e.target.value))} />
            </div>
            <div className="field">
              <label htmlFor="eval-split">Tasks</label>
              <select id="eval-split" value={split} onChange={(e) => setSplit(e.target.value as typeof split)}>
                <option value="">All tasks</option>
                <option value="train">Train split</option>
                <option value="heldout">Held-out split</option>
              </select>
            </div>
            <div className="field">
              <label htmlFor="eval-seed">Order seed</label>
              <input id="eval-seed" type="number" min={0} value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
            </div>
          </div>
          <label className="check">
            <input type="checkbox" checked={warm} onChange={(e) => setWarm(e.target.checked)} />
            Also measure warm-cache runs (reported separately)
          </label>
          {llm === "gemini" && (
            <p className="notice warn">Gemini evaluations make real API calls for every run and cost money.</p>
          )}
          {error && <ErrorNotice error={error} />}
          <div>
            <button type="submit" disabled={busy}>{busy ? "Starting…" : "Start evaluation"}</button>
          </div>
        </form>
      </section>

      <section aria-labelledby="past-evals">
        <div className="section-head"><h2 id="past-evals">Past evaluations</h2></div>
        {list.error ? (
          <ErrorNotice error={list.error} onRetry={list.reload} />
        ) : !list.data ? (
          <Loading what="evaluations" />
        ) : list.data.length === 0 ? (
          <Empty title="No evaluations yet">
            <p>Start one above. A mock evaluation with 3 repetitions takes a few seconds.</p>
          </Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Started</th><th>Evaluation</th><th>Agent</th><th>Status</th><th className="num">Runs</th></tr>
              </thead>
              <tbody>
                {list.data.map((e) => (
                  <tr key={e.evaluation_id}>
                    <td>{fmtTime(e.created_at)}</td>
                    <td><Link to={`/evaluations/${e.evaluation_id}`}>Evaluation {shortId(e.evaluation_id)}</Link></td>
                    <td><AgentTag provider={e.provider} /></td>
                    <td><StatusBadge status={e.status} /></td>
                    <td className="num">{e.completed_runs} of {e.planned_runs}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}
