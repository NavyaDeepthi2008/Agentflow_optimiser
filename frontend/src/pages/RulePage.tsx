import { type FormEvent, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError, api } from "../api/client";
import type { Evidence, RuleDetail, Setting, ValidationResult } from "../api/types";
import { ErrorNotice, type Fact, Facts, Loading, NotFound, ResultsLabel, StatusBadge } from "../components/ui";
import { MISSING, fmtInt, fmtPct, fmtTime, shortId } from "../lib/format";
import { CAN, describeSetting, fmtSeconds, kindName } from "../lib/learning";
import { useResource } from "../lib/useResource";

export function RulePage() {
  const { ruleId = "" } = useParams();
  const detail = useResource(() => api.rule(ruleId), [ruleId], {
    pollMs: 1200,
    pollWhile: (d: RuleDetail) => d.rule.status === "validating",
  });
  const latestId = detail.data?.rule.latest_validation_id ?? null;
  const status = detail.data?.rule.status;
  const validation = useResource(
    () => (latestId ? api.validation(latestId) : Promise.resolve(null)),
    [latestId, status],
  );

  if (detail.error?.status === 404) return <NotFound />;
  if (detail.error && !detail.data) return <ErrorNotice error={detail.error} onRetry={detail.reload} />;
  if (!detail.data) return <Loading what="rule" />;

  const { rule, events, shadow_live } = detail.data;
  const ev = rule.evidence as unknown as Evidence;
  const refresh = () => {
    detail.reload();
    validation.reload();
  };

  return (
    <>
      <header className="page-head">
        <p className="crumb"><Link to="/learning">Learning</Link> / rule {shortId(rule.rule_id)}</p>
        <h1>
          {rule.tool_name}: {kindName(rule.kind).toLowerCase()}
        </h1>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <StatusBadge status={rule.status} />
          <span className="tag neutral">tool version {rule.tool_version}</span>
          <span className="tag neutral">
            learned from {rule.dataset.name} v{rule.dataset.version}
          </span>
        </div>
      </header>

      <div className="change panel" aria-label="Proposed change">
        <div>
          <small>Now</small>
          <strong>{describeSetting(rule.kind, rule.current_setting as Setting)}</strong>
        </div>
        <span aria-hidden="true">→</span>
        <div>
          <small>Proposed</small>
          <strong>{describeSetting(rule.kind, rule.proposed_setting as Setting)}</strong>
        </div>
        {rule.status_reason && <p className="hint">{rule.status_reason}</p>}
      </div>

      <EvidenceSection ev={ev} />

      <section aria-labelledby="act-head">
        <div className="section-head"><h2 id="act-head">Actions</h2></div>
        <Actions rule={detail.data.rule} onChange={refresh} />
        {(rule.status === "shadow" || shadow_live) && (
          <p className="hint" style={{ marginTop: 12 }}>
            Shadow observations in ordinary optimized runs:{" "}
            {shadow_live
              ? `${fmtInt(shadow_live.would_reuse)} times the rule would have reused a result; ${fmtInt(
                  shadow_live.matched,
                )} matched the live result, ${fmtInt(shadow_live.mismatched)} differed.`
              : "none yet. Run tasks that use this tool in optimized mode."}
          </p>
        )}
      </section>

      <section aria-labelledby="val-head">
        <div className="section-head"><h2 id="val-head">Held-out validation</h2></div>
        {!latestId ? (
          <p className="hint">Not validated yet.</p>
        ) : validation.error ? (
          <ErrorNotice error={validation.error} onRetry={validation.reload} />
        ) : !validation.data ? (
          <Loading what="validation" />
        ) : validation.data.status === "running" ? (
          <p className="loading" role="status">Validating: running held-out tasks with the rule off and on</p>
        ) : (
          <ValidationView
            llm={String(validation.data.config.llm ?? "scripted")}
            ruleStatus={rule.status}
            status={validation.data.status}
            result={validation.data.result}
            error={validation.data.error_message}
            baseRules={validation.data.base_rule_ids}
          />
        )}
      </section>

      <section aria-labelledby="hist-head">
        <div className="section-head"><h2 id="hist-head">History</h2></div>
        <ol className="history">
          {events.map((e, i) => (
            <li key={i}>
              <span>{fmtTime(e.at)}</span>
              <StatusBadge status={e.to_status} />
              <span>by {e.actor}</span>
              {e.note && <span className="hint">{e.note}</span>}
            </li>
          ))}
        </ol>
      </section>
    </>
  );
}

function EvidenceSection({ ev }: { ev: Evidence }) {
  const facts: Fact[] = [
    { label: "Supporting repeats", value: fmtInt(ev.repeat_count), note: ev.thresholds ? `at least ${ev.thresholds.min_support} needed` : undefined },
    { label: "Different runs or sessions", value: fmtInt(ev.distinct_runs), note: ev.thresholds ? `at least ${ev.thresholds.min_distinct_runs} needed` : undefined },
    { label: "Consistency", value: fmtPct(ev.consistency), note: "identical results; 100% required" },
    { label: "All repeats examined", value: fmtInt(ev.observed_repeats), note: `${fmtInt(ev.changed_repeats)} with a different result` },
    { label: "Longest unchanged gap", value: ev.max_unchanged_gap_seconds != null ? fmtSeconds(ev.max_unchanged_gap_seconds) : MISSING },
    { label: "Shortest gap with a change", value: ev.min_changed_gap_seconds != null ? fmtSeconds(ev.min_changed_gap_seconds) : "none seen" },
  ];
  return (
    <section aria-labelledby="ev-head">
      <div className="section-head"><h2 id="ev-head">Evidence</h2></div>
      {ev.reasons && ev.reasons.length > 0 && (
        <ul className="reasons">{ev.reasons.map((r) => <li key={r}>{r}</li>)}</ul>
      )}
      <Facts items={facts} />
      {ev.capped_by_ceiling && <p className="hint">Capped at the limit the tool's developer set.</p>}
      <p className="hint" style={{ marginTop: 12 }}>
        From evaluation{ev.evaluation_ids.length === 1 ? "" : "s"}{" "}
        {ev.evaluation_ids.map((id, i) => (
          <span key={id}>{i > 0 && ", "}<Link to={`/evaluations/${id}`}>{shortId(id)}</Link></span>
        ))}
        ; tasks {ev.supporting_task_ids.join(", ") || MISSING}.
      </p>
      {ev.sample.length > 0 && (
        <div className="table-wrap" style={{ marginTop: 12 }}>
          <table aria-label="Supporting repeats">
            <thead>
              <tr><th>Task</th><th>First request</th><th>Repeat</th><th className="num">Gap</th><th>Result</th></tr>
            </thead>
            <tbody>
              {ev.sample.map((p, i) => (
                <tr key={i}>
                  <td>{p.task_id}</td>
                  <td><Link to={`/runs/${p.first_run_id}`}>{shortId(p.first_run_id)}</Link></td>
                  <td><Link to={`/runs/${p.run_id}`}>{shortId(p.run_id)}</Link>{p.first_run_id !== p.run_id && " (later run, same session)"}</td>
                  <td className="num">{fmtSeconds(p.gap_seconds)}</td>
                  <td>{p.same_output ? "identical" : <span className="tag effect">different</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="hint" style={{ marginTop: 8 }}>
        {fmtInt(ev.supporting_run_ids.length)} supporting runs in total; the table shows up to 20 repeats.
      </p>
    </section>
  );
}

function Actions({ rule, onChange }: { rule: RuleDetail["rule"]; onChange: () => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError>();
  const [reps, setReps] = useState(2);
  const [approver, setApprover] = useState("");
  const [note, setNote] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [reason, setReason] = useState("");
  const s = rule.status;

  async function act(fn: () => Promise<unknown>) {
    setBusy(true);
    setError(undefined);
    try {
      await fn();
      onChange();
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(-1, "client_error", String(err)));
    } finally {
      setBusy(false);
    }
  }

  const approve = (e: FormEvent) => {
    e.preventDefault();
    void act(() =>
      api.approveRule(rule.rule_id, { approved_by: approver.trim(), note: note.trim() || undefined, confirm: confirmed }),
    );
  };
  const disable = (e: FormEvent) => {
    e.preventDefault();
    void act(() => api.disableRule(rule.rule_id, { reason: reason.trim() }));
  };

  return (
    <div className="actions-grid">
      {error && <ErrorNotice error={error} />}
      <div className="panel">
        <h3>Shadow mode</h3>
        <p className="hint">Watch the rule in ordinary optimized runs without letting it change anything.</p>
        {CAN.shadowOff(s) ? (
          <button type="button" className="quiet" disabled={busy} onClick={() => act(() => api.setShadow(rule.rule_id, false))}>
            Stop observing
          </button>
        ) : (
          <button type="button" className="quiet" disabled={busy || !CAN.shadowOn(s)} onClick={() => act(() => api.setShadow(rule.rule_id, true))}>
            Observe in shadow
          </button>
        )}
      </div>

      <div className="panel">
        <h3>Validate on held-out tasks</h3>
        <p className="hint">Every held-out task, empty caches, rule off (shadow) vs on.</p>
        <div className="field">
          <label htmlFor="val-reps">Repetitions</label>
          <input id="val-reps" type="number" min={1} max={5} value={reps} onChange={(e) => setReps(Number(e.target.value))} />
        </div>
        <button
          type="button"
          disabled={busy || !CAN.validate(s)}
          onClick={() => act(() => api.validateRule(rule.rule_id, { repetitions: reps, llm: "scripted" }))}
        >
          {s === "validating" ? "Validating…" : "Validate"}
        </button>
      </div>

      {CAN.approve(s) && (
        <form className="panel approve" onSubmit={approve} aria-label="Approve rule">
          <h3>Approve</h3>
          <p className="hint">The rule passed validation. It starts affecting optimized runs only after you approve it.</p>
          <div className="field">
            <label htmlFor="approver">Your name</label>
            <input type="text" id="approver" value={approver} maxLength={100} onChange={(e) => setApprover(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="approve-note">Note (optional)</label>
            <input type="text" id="approve-note" value={note} maxLength={500} onChange={(e) => setNote(e.target.value)} />
          </div>
          <label className="check">
            <input type="checkbox" checked={confirmed} onChange={(e) => setConfirmed(e.target.checked)} />
            I reviewed the evidence and the validation results
          </label>
          <button type="submit" disabled={busy || !confirmed || !approver.trim()}>Approve and activate</button>
        </form>
      )}

      {CAN.disable(s) && (
        <form className="panel" onSubmit={disable} aria-label="Disable rule">
          <h3>{s === "active" ? "Disable" : "Dismiss"}</h3>
          <p className="hint">
            {s === "active"
              ? "Optimized runs go back to the tool's own setting on their next request."
              : "The proposal will not be used."}
          </p>
          <div className="field">
            <label htmlFor="disable-reason">Reason</label>
            <input type="text" id="disable-reason" value={reason} maxLength={500} onChange={(e) => setReason(e.target.value)} />
          </div>
          <button type="submit" className="danger" disabled={busy || !reason.trim()}>
            {s === "active" ? "Disable rule" : "Dismiss proposal"}
          </button>
        </form>
      )}
    </div>
  );
}

export function ValidationView({
  llm,
  ruleStatus,
  status,
  result,
  error,
  baseRules,
}: {
  llm: string;
  ruleStatus: string;
  status: string;
  result: ValidationResult | null;
  error: string | null;
  baseRules: string[];
}) {
  if (!result) {
    return (
      <div className="notice warn" role="status">
        <strong>Validation {status}</strong>
        <span>{error ?? "No result was recorded."}</span>
      </div>
    );
  }
  const { off, on } = result.arms;
  return (
    <div style={{ display: "grid", gap: 16 }}>
      <ResultsLabel provider={llm === "scripted" ? "scripted" : "gemini"} />
      <div className={`notice ${result.passed ? "info" : "error"}`} role="status">
        <strong>{result.passed ? "Passed every check" : "Rejected"}</strong>
        <span>
          {result.passed
            ? ruleStatus === "validated"
              ? "The rule may be approved."
              : ruleStatus === "active"
                ? "The rule was approved after this validation."
                : "This was the latest validation of the rule."
            : `Failed: ${(result.criteria ?? []).filter((c) => !c.passed).map((c) => c.label).join("; ")}`}
        </span>
      </div>
      <ul className="criteria">
        {(result.criteria ?? []).map((c) => (
          <li key={c.name} className={c.passed ? "pass" : "fail"}>
            <span aria-label={c.passed ? "passed" : "failed"}>{c.passed ? "✓" : "✗"}</span>
            <strong>{c.label}</strong>
            <small>{c.detail}</small>
          </li>
        ))}
      </ul>
      <div className="table-wrap">
        <table aria-label="Rule off versus on">
          <thead>
            <tr><th></th><th className="num">Runs</th><th className="num">Accuracy</th><th className="num">Tool executions</th><th className="num">incl. warm-ups</th><th className="num">Rule used</th></tr>
          </thead>
          <tbody>
            {([["Rule off (shadow)", off], ["Rule on", on]] as const).map(([label, a]) => (
              <tr key={label}>
                <th scope="row">{label}</th>
                <td className="num">{fmtInt(a.runs)}</td>
                <td className="num">{fmtPct(a.accuracy)}</td>
                <td className="num">{fmtInt(a.tool_executions)}</td>
                <td className="num">{fmtInt(a.tool_executions_incl_warmups)}</td>
                <td className="num">{fmtInt(a.rule_applied)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {result.shadow && (
        <p className="hint">
          In shadow (rule off) the rule would have reused {fmtInt(result.shadow.would_reuse)}{" "}
          {result.shadow.would_reuse === 1 ? "result" : "results"}; {fmtInt(result.shadow.mismatched)} differed
          from the live result.
        </p>
      )}
      {baseRules.length > 0 && (
        <p className="hint">Both arms also ran with the active rules {baseRules.map(shortId).join(", ")}.</p>
      )}
      {result.tasks && result.tasks.length > 0 && (
        <details>
          <summary>Per task ({result.tasks.length})</summary>
          <div className="table-wrap">
            <table aria-label="Per task">
              <thead>
                <tr><th>Task</th><th>Phase</th><th>Off</th><th>On</th><th className="num">Executions off → on</th></tr>
              </thead>
              <tbody>
                {result.tasks.map((t) => (
                  <tr key={`${t.task_id}-${t.phase}-${t.repetition}`}>
                    <td>{t.task_id}</td>
                    <td>{t.phase}</td>
                    <td><Link to={`/runs/${t.off.run_id}`}>{t.off.correct ? "correct" : "incorrect"}</Link></td>
                    <td><Link to={`/runs/${t.on.run_id}`}>{t.on.correct ? "correct" : "incorrect"}</Link></td>
                    <td className="num">{t.off.tool_executions} → {t.on.tool_executions}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </div>
  );
}
