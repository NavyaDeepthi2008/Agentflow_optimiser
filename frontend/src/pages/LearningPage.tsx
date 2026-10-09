import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ApiError, api } from "../api/client";
import type { Candidate, EvaluationOut, MiningOut, MiningReport, RuleOut, Setting } from "../api/types";
import { Empty, ErrorNotice, Loading, StatusBadge, isActive } from "../components/ui";
import { fmtInt, fmtPct, fmtTime, shortId } from "../lib/format";
import { describeSetting, kindName } from "../lib/learning";
import { useResource } from "../lib/useResource";

const DATASET = { name: "core", version: "2" };
export const TRAINING_EVALUATION = {
  dataset_version: DATASET.version,
  split: "train" as const,
  warm_cache: true,
  repetitions: 2,
  llm: "scripted" as const,
};

type Mining = Omit<MiningOut, "report"> & { report: MiningReport };

export function LearningPage() {
  const navigate = useNavigate();
  const rules = useResource(() => api.rules(), [], {
    pollMs: 1500,
    pollWhile: (items: RuleOut[]) => items.some((r) => isActive(r.status)),
  });
  const evaluations = useResource(() => api.listEvaluations(), [], {
    pollMs: 2000,
    pollWhile: (items: EvaluationOut[]) => items.some((e) => isActive(e.status)),
  });
  const minings = useResource(() => api.minings(), []);
  const [mining, setMining] = useState<Mining>();
  const [busy, setBusy] = useState<"" | "train" | "mine">("");
  const [error, setError] = useState<ApiError>();

  const training = (evaluations.data ?? []).filter(
    (e) => e.dataset.name === DATASET.name && e.dataset.version === DATASET.version,
  );
  const completed = training.filter((e) => e.status === "completed");
  const running = training.some((e) => isActive(e.status));
  const latest: Mining | undefined = mining ?? minings.data?.[0];

  async function startTraining() {
    setBusy("train");
    setError(undefined);
    try {
      const created = await api.createEvaluation(TRAINING_EVALUATION);
      navigate(`/evaluations/${created.evaluation_id}`);
    } catch (err) {
      setError(asApiError(err));
      setBusy("");
    }
  }

  async function mine() {
    setBusy("mine");
    setError(undefined);
    try {
      setMining(await api.mine({ dataset: DATASET.name, dataset_version: DATASET.version }));
      rules.reload();
    } catch (err) {
      setError(asApiError(err));
    } finally {
      setBusy("");
    }
  }

  return (
    <>
      <header className="page-head">
        <h1>Learning</h1>
        <p>
          Proposes conservative changes to how long, and how widely, read-only tool results may be
          reused, using only correctly graded training runs as evidence. Every proposal is checked on
          held-out tasks with the rule off and on, and becomes active only after a person approves it.
          Tools that change data, have no known effect, or return live data are never touched.
        </p>
      </header>

      <ol className="steps" aria-label="How a rule is learned">
        <li><strong>Collect evidence</strong><span>Training evaluation with a warm phase</span></li>
        <li><strong>Mine</strong><span>Repeated identical requests that returned identical results</span></li>
        <li><strong>Validate</strong><span>Held-out tasks, rule off vs on, empty caches</span></li>
        <li><strong>Approve</strong><span>A person reviews the evidence and switches it on</span></li>
      </ol>

      {error && <ErrorNotice error={error} />}

      <section aria-labelledby="evidence-head">
        <div className="section-head">
          <h2 id="evidence-head">1–2. Evidence and mining</h2>
        </div>
        <div className="panel learn-actions">
          <div>
            <p>
              Training evaluations of {DATASET.name} v{DATASET.version}:{" "}
              {evaluations.error ? "—" : evaluations.data ? `${completed.length} completed` : "loading"}
              {running && " (one is running)"}
            </p>
            <small className="hint">
              Mock agent, train split, warm phase on, {TRAINING_EVALUATION.repetitions} repetitions. The
              warm phase repeats each task in the same session, which is the only evidence for sharing
              results between runs.
            </small>
          </div>
          <div className="button-row">
            <button type="button" className="quiet" onClick={startTraining} disabled={busy !== ""}>
              {busy === "train" ? "Starting…" : "Run a training evaluation"}
            </button>
            <button
              type="button"
              onClick={mine}
              disabled={busy !== "" || completed.length === 0}
              title={completed.length === 0 ? "Run a training evaluation first" : undefined}
            >
              {busy === "mine" ? "Mining…" : "Mine proposals"}
            </button>
          </div>
        </div>
        {evaluations.error && <ErrorNotice error={evaluations.error} onRetry={evaluations.reload} />}
        {latest ? <MiningSummary mining={latest} /> : minings.loading ? <Loading what="mining results" /> : null}
      </section>

      <section aria-labelledby="rules-head">
        <div className="section-head">
          <h2 id="rules-head">3–4. Rules</h2>
        </div>
        {rules.error ? (
          <ErrorNotice error={rules.error} onRetry={rules.reload} />
        ) : !rules.data ? (
          <Loading what="rules" />
        ) : rules.data.length === 0 ? (
          <Empty title="No learned rules yet">
            <p>Run a training evaluation, then mine proposals. Nothing changes until you approve a rule.</p>
          </Empty>
        ) : (
          <RulesTable rules={rules.data} />
        )}
      </section>
    </>
  );
}

function RulesTable({ rules }: { rules: RuleOut[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Tool</th>
            <th>Rule</th>
            <th>Now</th>
            <th>Proposed</th>
            <th className="num">Supporting repeats</th>
            <th className="num">Consistency</th>
            <th>Status</th>
            <th>Created</th>
          </tr>
        </thead>
        <tbody>
          {rules.map((r) => {
            const ev = r.evidence as { repeat_count?: number; consistency?: number | null };
            return (
              <tr key={r.rule_id}>
                <td><Link to={`/learning/rules/${r.rule_id}`}><strong>{r.tool_name}</strong></Link></td>
                <td>{kindName(r.kind)}</td>
                <td>{describeSetting(r.kind, r.current_setting as Setting)}</td>
                <td>{describeSetting(r.kind, r.proposed_setting as Setting)}</td>
                <td className="num">{fmtInt(ev.repeat_count)}</td>
                <td className="num">{fmtPct(ev.consistency)}</td>
                <td><StatusBadge status={r.status} /></td>
                <td>{fmtTime(r.created_at)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function MiningSummary({ mining }: { mining: Mining }) {
  const r = mining.report;
  const excluded = Object.entries(r.excluded);
  return (
    <div className="panel mining" aria-label="Latest mining">
      <p>
        <strong>Latest mining</strong> ({fmtTime(mining.created_at)}, {mining.evaluation_ids.length} evaluation
        {mining.evaluation_ids.length === 1 ? "" : "s"}): used {fmtInt(r.runs_used)} of {fmtInt(r.runs_examined)} training
        runs. Thresholds: at least {r.thresholds.min_support} supporting repeats from {r.thresholds.min_distinct_runs}{" "}
        different runs, and 100% identical results.
      </p>
      {excluded.length > 0 && (
        <p className="hint">
          Not used as evidence: {excluded.map(([why, n]) => `${n} (${why})`).join(", ")}.
        </p>
      )}
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Tool</th><th>Rule</th><th>Now</th><th>Proposal</th><th>Why</th></tr>
          </thead>
          <tbody>
            {r.candidates.map((c) => (
              <CandidateRow key={`${c.tool_name}-${c.kind}`} c={c} />
            ))}
            {r.not_eligible.map((t) => (
              <tr key={t.tool_name} className="muted-row">
                <td>{t.tool_name}</td>
                <td>—</td>
                <td>—</td>
                <td><span className="tag effect">never learned</span></td>
                <td>{t.reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {(r.created_rule_ids?.length ?? 0) > 0 && (
        <p className="hint">New proposals: {r.created_rule_ids?.map(shortId).join(", ")}.</p>
      )}
      {(r.duplicates?.length ?? 0) > 0 && (
        <p className="hint">Already proposed before: {r.duplicates?.map((d) => `${d.tool_name} (${kindName(d.kind)})`).join(", ")}.</p>
      )}
    </div>
  );
}

function CandidateRow({ c }: { c: Candidate }) {
  return (
    <tr>
      <td>{c.tool_name}</td>
      <td>{kindName(c.kind)}</td>
      <td>{describeSetting(c.kind, c.current_setting)}</td>
      <td>{c.proposed ? <strong>{describeSetting(c.kind, c.proposed_setting)}</strong> : <span className="tag neutral">not proposed</span>}</td>
      <td>{c.reasons.join("; ")}</td>
    </tr>
  );
}

function asApiError(err: unknown): ApiError {
  return err instanceof ApiError ? err : new ApiError(-1, "client_error", String(err));
}
