import { type FormEvent, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useBackend } from "../App";
import { ApiError, api } from "../api/client";
import type { DatasetTask, RunRequest } from "../api/types";
import { Empty, ErrorNotice, Loading } from "../components/ui";
import { useResource } from "../lib/useResource";

export const CATEGORY_NAMES: Record<string, { title: string; about: string }> = {
  redundant: { title: "Repeated lookups", about: "The agent asks for the same data more than once." },
  freshness: { title: "Freshness", about: "Data changes over time or after a write." },
  side_effect: { title: "Side effects", about: "Actions that must never be skipped or merged." },
  failure: { title: "Failures", about: "Outages, missing data and loops." },
  control: { title: "Controls", about: "Nothing to optimize: results should not change." },
};

type Mode = "passthrough" | "optimized" | "both";
type Llm = "scripted" | "gemini";

export function newSessionId(): string {
  return `s${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
}

export function LaunchPage() {
  const { health } = useBackend();
  const dataset = useResource(() => api.dataset(), []);
  const navigate = useNavigate();
  const geminiReady = health?.gemini_configured === true;

  const [source, setSource] = useState<"scenario" | "custom">("scenario");
  const [scenarioId, setScenarioId] = useState<string>("redundant_orbit_price");
  const [customTask, setCustomTask] = useState("");
  const [llm, setLlm] = useState<Llm>("scripted");
  const [mode, setMode] = useState<Mode>("both");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<ApiError>();

  const tasks = useMemo(() => dataset.data?.tasks ?? [], [dataset.data]);
  const scenario = tasks.find((t) => t.id === scenarioId);
  const grouped = useMemo(() => {
    const groups = new Map<string, DatasetTask[]>();
    for (const t of tasks) groups.set(t.category, [...(groups.get(t.category) ?? []), t]);
    return groups;
  }, [tasks]);

  const effectiveLlm: Llm = source === "custom" ? "gemini" : llm;
  const blocker =
    effectiveLlm === "gemini" && !geminiReady
      ? "Gemini isn't configured on the backend (set GEMINI_API_KEY). Use the mock agent with a scenario."
      : source === "custom" && !customTask.trim()
        ? "Write a task for the agent."
        : source === "scenario" && !scenario
          ? "Choose a scenario."
          : source === "scenario" && scenario?.scripted_only && effectiveLlm === "gemini"
            ? "This scenario simulates time passing, so it only runs with the mock agent."
            : null;

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (blocker) return;
    setSubmitting(true);
    setError(undefined);
    const base: RunRequest =
      source === "scenario"
        ? { scenario: { task_id: scenarioId }, llm: effectiveLlm }
        : { task: customTask.trim(), llm: "gemini" };
    try {
      if (mode === "both") {
        // Separate new sessions, so each mode starts with an empty cache.
        const session = newSessionId();
        const baseline = await api.createRun({ ...base, mode: "passthrough", session_id: `${session}-b` });
        const optimized = await api.createRun({ ...base, mode: "optimized", session_id: `${session}-o` });
        navigate(`/compare?base=${baseline.run_id}&opt=${optimized.run_id}`);
      } else {
        const run = await api.createRun({ ...base, mode, session_id: newSessionId() });
        navigate(`/runs/${run.run_id}`);
      }
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(-1, "client_error", String(err)));
      setSubmitting(false);
    }
  }

  return (
    <>
      <header className="page-head">
        <h1>Run a task</h1>
        <p>
          Every tool request the agent makes passes through the gateway. Run a task in baseline mode
          (every request executes) or optimized mode (safe, identical read-only requests reuse a stored
          result), or both side by side.
        </p>
      </header>

      <form className="launcher" onSubmit={submit}>
        <div className="scenarios">
          <div className="segmented" role="radiogroup" aria-label="Task source">
            <label>
              <input type="radio" name="source" checked={source === "scenario"} onChange={() => setSource("scenario")} />
              Scenario
            </label>
            <label>
              <input type="radio" name="source" checked={source === "custom"} onChange={() => setSource("custom")} />
              Your own task
            </label>
          </div>

          {source === "custom" ? (
            <div className="field">
              <label htmlFor="custom-task">Task for the agent</label>
              <textarea
                id="custom-task"
                value={customTask}
                maxLength={4000}
                onChange={(e) => setCustomTask(e.target.value)}
                placeholder="For example: What does Orbit cost per user per month, and what is the refund window?"
              />
              <small className="hint">
                Your own tasks run with the real Gemini model against the mock Nimbus Labs tools. Side-effecting
                tools stay blocked.
              </small>
            </div>
          ) : dataset.error ? (
            <ErrorNotice error={dataset.error} onRetry={dataset.reload} />
          ) : dataset.loading && !dataset.data ? (
            <Loading what="scenarios" />
          ) : tasks.length === 0 ? (
            <Empty title="No scenarios available" />
          ) : (
            [...grouped.entries()].map(([category, list]) => (
              <fieldset className="scenario-group" key={category} style={{ border: 0, padding: 0, margin: 0 }}>
                <legend style={{ padding: 0, marginBottom: 8 }}>
                  <h3>{CATEGORY_NAMES[category]?.title ?? category}</h3>
                  <span className="hint">{CATEGORY_NAMES[category]?.about}</span>
                </legend>
                <div className="scenario-list">
                  {list.map((t) => (
                    <label className="scenario" key={t.id}>
                      <input
                        type="radio"
                        name="scenario"
                        value={t.id}
                        checked={scenarioId === t.id}
                        onChange={() => setScenarioId(t.id)}
                      />
                      <span>
                        <p>{t.task}</p>
                        <span className="meta">
                          <span>{t.tools.join(", ")}</span>
                          {t.allow_side_effects && <span className="tag effect">changes data</span>}
                          {t.scripted_only && <span className="tag neutral">mock agent only</span>}
                        </span>
                      </span>
                    </label>
                  ))}
                </div>
              </fieldset>
            ))
          )}
        </div>

        <div className="launch-panel panel">
          <div className="field">
            <span id="agent-label" style={{ fontWeight: 560 }}>Agent</span>
            <div className="segmented" role="radiogroup" aria-labelledby="agent-label">
              <label>
                <input
                  type="radio"
                  name="llm"
                  checked={effectiveLlm === "scripted"}
                  disabled={source === "custom"}
                  onChange={() => setLlm("scripted")}
                />
                Mock agent
              </label>
              <label>
                <input
                  type="radio"
                  name="llm"
                  checked={effectiveLlm === "gemini"}
                  disabled={!geminiReady}
                  onChange={() => setLlm("gemini")}
                />
                Gemini
              </label>
            </div>
            <small>
              {effectiveLlm === "scripted"
                ? "Replays the scenario's fixed script. Results are labelled MOCK."
                : geminiReady
                  ? `Real model (${health?.gemini_model}). Results are labelled LIVE and may vary.`
                  : "Gemini isn't configured on the backend."}
            </small>
          </div>

          <div className="field">
            <span id="mode-label" style={{ fontWeight: 560 }}>Gateway mode</span>
            <div className="segmented" role="radiogroup" aria-labelledby="mode-label">
              {(
                [
                  ["both", "Compare both"],
                  ["passthrough", "Baseline"],
                  ["optimized", "Optimized"],
                ] as const
              ).map(([value, text]) => (
                <label key={value}>
                  <input type="radio" name="mode" checked={mode === value} onChange={() => setMode(value)} />
                  {text}
                </label>
              ))}
            </div>
            <small>Each run starts in a new session, so its cache starts empty.</small>
          </div>

          {error && <ErrorNotice error={error} />}
          {blocker && <p className="hint" role="note">{blocker}</p>}

          <div>
            <button type="submit" disabled={submitting || blocker !== null}>
              {submitting ? "Starting…" : mode === "both" ? "Run both modes" : "Start run"}
            </button>
          </div>
        </div>
      </form>
    </>
  );
}
