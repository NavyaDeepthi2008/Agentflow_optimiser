import { Link, useSearchParams } from "react-router-dom";
import { useBackend } from "../App";
import type { RunOut, ToolCall } from "../api/types";
import { RunSummary } from "../components/RunSummary";
import { Empty, ErrorNotice, Loading, ResultsLabel, isActive, modeName } from "../components/ui";
import { tallyRequests } from "../lib/decisions";
import { fmtChange, fmtMs, shortId, truncate } from "../lib/format";
import { useRun } from "../lib/useRun";

export function ComparePage() {
  const [params] = useSearchParams();
  const base = params.get("base");
  const opt = params.get("opt");
  if (!base || !opt) {
    return (
      <Empty title="Nothing to compare" action={<Link className="button" to="/">Run both modes</Link>}>
        <p>Start a task in "Compare both" mode to see baseline and optimized runs side by side.</p>
      </Empty>
    );
  }
  return <Compare key={`${base}-${opt}`} baseId={base} optId={opt} />;
}

function Compare({ baseId, optId }: { baseId: string; optId: string }) {
  const base = useRun(baseId);
  const opt = useRun(optId);
  const both =
    base.run.data && opt.run.data && base.details.data && opt.details.data &&
    !isActive(base.run.data.status) && !isActive(opt.run.data.status)
      ? {
          base: { run: base.run.data, calls: base.details.data.tool_calls },
          opt: { run: opt.run.data, calls: opt.details.data.tool_calls },
        }
      : null;

  return (
    <>
      <header className="page-head">
        <h1>Baseline vs optimized</h1>
        <p>{base.run.data?.task ?? opt.run.data?.task ?? "Loading the task"}</p>
      </header>
      {(base.run.data ?? opt.run.data) && (
        <ResultsLabel
          provider={(base.run.data ?? opt.run.data)!.provider}
          model={(base.run.data ?? opt.run.data)!.model_id}
        />
      )}
      {both ? <DeltaStrip base={both.base} opt={both.opt} /> : (
        <p className="hint" style={{ marginBottom: 24 }}>The comparison appears when both runs have finished.</p>
      )}
      <div className="compare">
        <Column title="Baseline" bundle={base} />
        <Column title="Optimized" bundle={opt} />
      </div>
    </>
  );
}

function Column({ title, bundle }: { title: string; bundle: ReturnType<typeof useRun> }) {
  const { health } = useBackend();
  const { run, details } = bundle;
  return (
    <div className="panel">
      <div className="section-head" style={{ marginBottom: 0 }}>
        <h2>{title}</h2>
        {run.data && <Link to={`/runs/${run.data.run_id}`}>Open run {shortId(run.data.run_id)}</Link>}
      </div>
      {run.error ? (
        <ErrorNotice error={run.error} onRetry={run.reload} />
      ) : run.data ? (
        <>
          {run.data.mode !== (title === "Baseline" ? "passthrough" : "optimized") && (
            <p className="notice warn">This run is in {modeName(run.data.mode)} mode.</p>
          )}
          <RunSummary run={run.data} toolCalls={details.data?.tool_calls} pricingConfigured={!!health?.pricing_configured} />
        </>
      ) : (
        <Loading what="run" />
      )}
    </div>
  );
}

function DeltaStrip({ base, opt }: { base: { run: RunOut; calls: ToolCall[] }; opt: { run: RunOut; calls: ToolCall[] } }) {
  const b = tallyRequests(base.calls);
  const o = tallyRequests(opt.calls);
  const change = b.executions > 0 ? ((o.executions - b.executions) / b.executions) * 100 : null;
  const timeChange =
    base.run.metrics?.duration_ms && opt.run.metrics?.duration_ms
      ? ((opt.run.metrics.duration_ms - base.run.metrics.duration_ms) / base.run.metrics.duration_ms) * 100
      : null;
  const sameAnswer = base.run.final_answer !== null && base.run.final_answer === opt.run.final_answer;
  return (
    <div className="delta-strip" aria-label="Comparison">
      <div>
        <span>Tool executions</span>
        <strong>{b.executions} → {o.executions}</strong>
      </div>
      <div>
        <span>Change</span>
        <strong>{fmtChange(change)}</strong>
      </div>
      <div>
        <span>Run time</span>
        <strong>{fmtMs(base.run.metrics?.duration_ms)} → {fmtMs(opt.run.metrics?.duration_ms)}</strong>
      </div>
      <div>
        <span>Run time change</span>
        <strong>{fmtChange(timeChange)}</strong>
      </div>
      <div>
        <span>Final answers</span>
        <strong>{sameAnswer ? "Identical" : "Different"}</strong>
        {!sameAnswer && (
          <span>{truncate(base.run.final_answer ?? "none", 40)} vs {truncate(opt.run.final_answer ?? "none", 40)}</span>
        )}
      </div>
    </div>
  );
}
