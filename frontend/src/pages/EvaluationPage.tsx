import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import type { EvaluationReport, Mode, ModeSummary, PhaseReport } from "../api/types";
import { ErrorNotice, Loading, MOCK_DISCLAIMER, StatusBadge, isActive, modeName } from "../components/ui";
import { fmtChange, fmtCost, fmtInt, fmtMs, fmtPct, shortId } from "../lib/format";
import { useResource } from "../lib/useResource";
import { CATEGORY_NAMES } from "./LaunchPage";

const PHASE_TITLES: Record<string, string> = {
  cold: "Cold cache: every run starts empty",
  warm: "Warm cache: the same task repeated in the same session",
};
const MODES: Mode[] = ["passthrough", "optimized"];

export function EvaluationPage() {
  const { evaluationId = "" } = useParams();
  return <EvaluationView key={evaluationId} evaluationId={evaluationId} />;
}

function EvaluationView({ evaluationId }: { evaluationId: string }) {
  const evaluation = useResource(() => api.evaluation(evaluationId), [evaluationId], {
    pollMs: 1000,
    pollWhile: (e) => isActive(e.status),
  });

  if (evaluation.error) {
    return (
      <>
        <header className="page-head"><h1>Evaluation {shortId(evaluationId)}</h1></header>
        <ErrorNotice error={evaluation.error} onRetry={evaluation.reload} />
      </>
    );
  }
  if (!evaluation.data) return <Loading what="evaluation" />;
  const e = evaluation.data;
  const pct = e.planned_runs > 0 ? (e.completed_runs / e.planned_runs) * 100 : 0;

  return (
    <>
      <header className="page-head">
        <h1>Evaluation {shortId(e.evaluation_id)}</h1>
        <p>
          Dataset {e.dataset.name} v{e.dataset.version}, {String(e.config.repetitions)} repetition
          {e.config.repetitions === 1 ? "" : "s"}, order seed {String(e.config.seed)}.
        </p>
        <div style={{ display: "flex", gap: 14, alignItems: "center", flexWrap: "wrap" }}>
          <StatusBadge status={e.status} />
          <span className="hint">{e.completed_runs} of {e.planned_runs} runs</span>
        </div>
      </header>

      {isActive(e.status) && (
        <div style={{ display: "grid", gap: 8, marginBottom: 28 }}>
          <div className="progress" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
            <span style={{ width: `${pct}%` }} />
          </div>
          <p className="hint">The report appears when every run has finished.</p>
        </div>
      )}
      {e.status === "interrupted" && (
        <div className="notice warn" role="status">
          <strong>This evaluation was interrupted</strong>
          <span>
            The server stopped after {e.completed_runs} of {e.planned_runs} runs. Partial runs are kept
            in the run history, but no report was produced.
          </span>
        </div>
      )}
      {e.status === "failed" && e.error && (
        <div className="notice error"><strong>{e.error.code}</strong><span>{e.error.message}</span></div>
      )}
      {e.report && <Report report={e.report} />}
    </>
  );
}

function Report({ report }: { report: EvaluationReport }) {
  const isMock = report.provider === "scripted";
  const phases = Object.entries(report.phases).sort(([a], [b]) => (a === "cold" ? -1 : b === "cold" ? 1 : 0));
  return (
    <>
      <div className={`label-banner ${isMock ? "mock" : "live-model"}`} role="note">
        <strong>{isMock ? "MOCK results" : `LIVE results from ${report.model_id}`}</strong>
        <span>{report.label}</span>
        {isMock && <span>{MOCK_DISCLAIMER}</span>}
      </div>

      {phases.map(([phase, data]) => (
        <PhaseSection key={phase} phase={phase} data={data} showCost={report.pricing !== null} currency={report.pricing?.currency} />
      ))}

      <section aria-labelledby="regressions">
        <div className="section-head">
          <h2 id="regressions">Accuracy changes</h2>
          <span className="hint">Compared task by task and repetition by repetition</span>
        </div>
        {report.regressions.length === 0 ? (
          <div className="notice info">
            <strong>No regressions</strong>
            <span>Every task the baseline answered correctly was also answered correctly by the optimized run.</span>
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <caption className="hint" style={{ textAlign: "left", padding: 10 }}>
                Regressions: the baseline was correct and the optimized run was not
              </caption>
              <thead><tr><th>Task</th><th>Phase</th><th className="num">Repetition</th><th>Failed checks</th><th>Runs</th></tr></thead>
              <tbody>
                {report.regressions.map((r) => (
                  <tr className="bad" key={`${r.task_id}-${r.phase}-${r.repetition}`}>
                    <td>{r.task_id}</td>
                    <td>{r.phase}</td>
                    <td className="num">{r.repetition}</td>
                    <td>{(r.optimized_failed_checks ?? []).join(", ")}</td>
                    <td>
                      <Link to={`/compare?base=${r.baseline_run_id}&opt=${r.optimized_run_id}`}>Compare runs</Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {report.improvements.length > 0 && (
          <p className="hint" style={{ marginTop: 10 }}>
            {report.improvements.length} task run{report.improvements.length === 1 ? " was" : "s were"} correct
            only in optimized mode: {report.improvements.map((r) => r.task_id).join(", ")}.
          </p>
        )}
      </section>

      <section aria-labelledby="per-task">
        <div className="section-head"><h2 id="per-task">Per task</h2></div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Task</th><th>Phase</th>
                <th className="num">Baseline correct</th><th className="num">Optimized correct</th>
                <th className="num">Executions (median)</th>
              </tr>
            </thead>
            <tbody>
              {report.per_task.map((t) => {
                const b = t.passthrough;
                const o = t.optimized;
                const worse = b && o && o.correct < b.correct;
                return (
                  <tr key={`${t.task_id}-${t.phase}`} className={worse ? "bad" : undefined}>
                    <td>{t.task_id}<br /><span className="hint">{CATEGORY_NAMES[t.category]?.title ?? t.category}</span></td>
                    <td>{t.phase}</td>
                    <td className="num">{b ? `${b.correct} of ${b.runs}` : "—"}</td>
                    <td className="num">{o ? `${o.correct} of ${o.runs}` : "—"}</td>
                    <td className="num">{fmtInt(b?.tool_executions_median)} → {fmtInt(o?.tool_executions_median)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {report.skipped_tasks.length > 0 && (
        <section aria-labelledby="skipped">
          <div className="section-head"><h2 id="skipped">Not measured</h2></div>
          <ul>
            {report.skipped_tasks.map((s) => (
              <li key={`${s.task_id}-${s.phase}`}>{s.task_id} ({s.phase} phase): {s.reason}</li>
            ))}
          </ul>
        </section>
      )}

      <section aria-labelledby="method">
        <div className="section-head"><h2 id="method">How this was measured</h2></div>
        <ul>
          {Object.entries(report.fairness).map(([k, v]) => (
            <li key={k}><strong>{k.replace(/_/g, " ")}:</strong> {Array.isArray(v) ? v.join(", ") : v}</li>
          ))}
          {report.notes.map((n) => <li key={n}>{n}</li>)}
          <li>Dataset checksum {report.dataset.checksum.slice(0, 12)}</li>
          {report.pricing && <li>Prices from {report.pricing.source}, retrieved {report.pricing.retrieved_on}</li>}
        </ul>
      </section>
    </>
  );
}

function PhaseSection({ phase, data, showCost, currency }: { phase: string; data: PhaseReport; showCost: boolean; currency?: string }) {
  const base = data.modes.passthrough;
  const opt = data.modes.optimized;
  const c = data.comparison;
  return (
    <section aria-labelledby={`phase-${phase}`}>
      <div className="section-head">
        <h2 id={`phase-${phase}`}>{PHASE_TITLES[phase] ?? phase}</h2>
      </div>
      {base && opt && c && (
        <div className="headline-figures">
          <div>
            <span>Tool executions</span>
            <strong>{fmtInt(base.tool_executions)} → {fmtInt(opt.tool_executions)}</strong>
            <span>{fmtChange(c.tool_executions_change_pct)} for the same {fmtInt(base.tool_requests)} requests</span>
          </div>
          <div>
            <span>Accuracy</span>
            <strong>{fmtPct(base.accuracy)} → {fmtPct(opt.accuracy)}</strong>
            <span>baseline → optimized</span>
          </div>
          <div>
            <span>Run time p50</span>
            <strong>{fmtMs(base.latency_ms.median)} → {fmtMs(opt.latency_ms.median)}</strong>
            <span>{fmtChange(c.latency_median_change_pct)}; p95 {fmtChange(c.latency_p95_change_pct)}</span>
          </div>
        </div>
      )}
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Mode</th><th className="num">Runs</th><th className="num">Accuracy</th>
              <th className="num">Requests</th><th className="num">Executions</th>
              <th className="num">Cache hits</th><th className="num">Reused in run</th>
              <th className="num">Retries</th><th className="num">LLM calls</th><th className="num">Tokens</th>
              <th className="num">Run time p50</th><th className="num">p95</th><th className="num">Gateway p50</th>
              {showCost && <th className="num">Cost</th>}
            </tr>
          </thead>
          <tbody>
            {MODES.map((mode) => {
              const s = data.modes[mode];
              return s ? <ModeRow key={mode} mode={mode} s={s} showCost={showCost} currency={currency} /> : null;
            })}
          </tbody>
        </table>
      </div>
      <CategoryTable data={data} />
    </section>
  );
}

function tokens(s: ModeSummary): string {
  if (s.tokens.runs_with_usage === 0) return "not reported";
  return `${fmtInt(s.tokens.total)}${s.tokens.complete ? "" : " (partial)"}`;
}

function ModeRow({ mode, s, showCost, currency }: { mode: Mode; s: ModeSummary; showCost: boolean; currency?: string }) {
  const cost = s.llm_cost !== null && s.tool_cost !== null ? s.llm_cost + s.tool_cost : null;
  return (
    <tr>
      <td>{modeName(mode)}</td>
      <td className="num">{s.runs}</td>
      <td className="num">{fmtPct(s.accuracy)} <span className="hint">({s.correct})</span></td>
      <td className="num">{fmtInt(s.tool_requests)}</td>
      <td className="num">{fmtInt(s.tool_executions)}</td>
      <td className="num">{fmtInt(s.cache_hits)}</td>
      <td className="num">{fmtInt(s.reused_in_run)}</td>
      <td className="num">{fmtInt(s.tool_retries)}</td>
      <td className="num">{fmtInt(s.llm_calls)}</td>
      <td className="num">{tokens(s)}</td>
      <td className="num">{fmtMs(s.latency_ms.median)}</td>
      <td className="num">{fmtMs(s.latency_ms.p95)}</td>
      <td className="num">{fmtMs(s.gateway_overhead_ms.median)}</td>
      {showCost && <td className="num">{cost === null ? "unavailable" : fmtCost(cost, currency)}</td>}
    </tr>
  );
}

function CategoryTable({ data }: { data: PhaseReport }) {
  const rows = Object.entries(data.by_category);
  if (rows.length === 0) return null;
  return (
    <details style={{ marginTop: 14 }}>
      <summary>By category</summary>
      <div className="table-wrap" style={{ marginTop: 10 }}>
        <table>
          <thead>
            <tr><th>Category</th><th className="num">Baseline accuracy</th><th className="num">Optimized accuracy</th><th className="num">Executions</th></tr>
          </thead>
          <tbody>
            {rows.map(([category, modes]) => (
              <tr key={category}>
                <td>{CATEGORY_NAMES[category]?.title ?? category}</td>
                <td className="num">{fmtPct(modes.passthrough?.accuracy)}</td>
                <td className="num">{fmtPct(modes.optimized?.accuracy)}</td>
                <td className="num">{fmtInt(modes.passthrough?.tool_executions)} → {fmtInt(modes.optimized?.tool_executions)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}
