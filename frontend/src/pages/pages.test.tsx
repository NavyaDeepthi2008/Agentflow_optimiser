import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import baselineRun from "../test/fixtures/baseline_run.json";
import blockedRun from "../test/fixtures/blocked_run.json";
import dataset from "../test/fixtures/dataset.json";
import evaluation from "../test/fixtures/evaluation.json";
import health from "../test/fixtures/health.json";
import optimizedRun from "../test/fixtures/optimized_run.json";
import retryRun from "../test/fixtures/retry_run.json";
import tools from "../test/fixtures/tools.json";
import { mockApi, renderApp, reply, seq } from "../test/mockApi";

type Bundle = typeof optimizedRun;
const clone = <T,>(v: T): T => JSON.parse(JSON.stringify(v)) as T;

function runRoutes(bundle: Bundle, runOverride?: unknown) {
  const id = bundle.run.run_id;
  return {
    [`GET /runs/${id}`]: runOverride ?? bundle.run,
    [`GET /runs/${id}/details`]: bundle.details,
    [`GET /runs/${id}/trace`]: bundle.trace,
  };
}

function activeVersion(bundle: Bundle) {
  const run = clone(bundle.run) as Record<string, unknown>;
  Object.assign(run, { status: "running", finished: false, final_answer: null, metrics: null, ended_at: null });
  return run;
}

// ---------------------------------------------------------------- run page

describe("run page", () => {
  it("shows a finished optimized run from real recorded data", async () => {
    mockApi(runRoutes(optimizedRun));
    renderApp(`/runs/${optimizedRun.run.run_id}`);

    expect(await screen.findByText("Finished")).toBeInTheDocument();
    // The answer appears in the summary and at the end of the timeline.
    expect(screen.getAllByText("Orbit costs 29 USD per user per month.")).toHaveLength(2);
    expect(screen.getByLabelText("tool requests")).toHaveTextContent("4");
    expect(screen.getByLabelText("tool executions")).toHaveTextContent("2");
    expect(screen.getByText("MOCK")).toBeInTheDocument();

    const routes = await screen.findAllByTestId("tool-route");
    expect(routes.map((r) => r.dataset.route)).toEqual(["live", "reused", "live", "reused"]);
    expect(within(routes[1]!).getByText("Reused")).toBeInTheDocument();

    // The mock agent reports no tokens: shown as missing, never as 0.
    const tokens = screen.getByText("Tokens").closest("div")!;
    expect(within(tokens).getByText("—")).toBeInTheDocument();
    expect(within(tokens).getByText("not reported by this agent")).toBeInTheDocument();
    // Pricing is not configured, so no cost appears at all.
    expect(screen.queryByText("Estimated cost")).not.toBeInTheDocument();
  });

  it("shows retries from the recorded attempt events", async () => {
    mockApi(runRoutes(retryRun as unknown as Bundle));
    renderApp(`/runs/${retryRun.run.run_id}`);
    expect(await screen.findByText("Retried 1 time")).toBeInTheDocument();
    expect(screen.getByText(/Attempt 1 failed \(transient\), retrying/)).toBeInTheDocument();
    expect(screen.getByText(/including 1 retry/)).toBeInTheDocument();
  });

  it("shows blocked requests as blocked", async () => {
    mockApi(runRoutes(blockedRun as unknown as Bundle));
    renderApp(`/runs/${blockedRun.run.run_id}`);
    const route = await screen.findByTestId("tool-route");
    expect(route.dataset.route).toBe("blocked");
    expect(within(route).getByText(/Blocked: the task is not allowed/)).toBeInTheDocument();
  });

  it("keeps polling while running and never shows metrics early", async () => {
    const id = optimizedRun.run.run_id;
    mockApi({
      ...runRoutes(optimizedRun),
      [`GET /runs/${id}`]: seq(activeVersion(optimizedRun), optimizedRun.run),
      [`GET /runs/${id}/trace`]: seq({ run_id: id, events: [], next_after_id: null }, optimizedRun.trace),
    });
    renderApp(`/runs/${id}`);
    expect(await screen.findByText("Running")).toBeInTheDocument();
    expect(screen.getByText("Results appear when the run finishes.")).toBeInTheDocument();
    expect(screen.queryByText("LLM calls")).not.toBeInTheDocument();
    expect(await screen.findByText("Finished", {}, { timeout: 4000 })).toBeInTheDocument();
    expect(screen.getByText("LLM calls")).toBeInTheDocument();
  });

  it("explains an interrupted run without inventing numbers", async () => {
    const run = activeVersion(optimizedRun);
    Object.assign(run, { status: "interrupted", finished: true,
      error: { code: "interrupted", message: "the server stopped before this run finished" } });
    mockApi({ ...runRoutes(optimizedRun, run),
      [`GET /runs/${optimizedRun.run.run_id}/details`]: { run, llm_calls: [], tool_calls: [] } });
    renderApp(`/runs/${optimizedRun.run.run_id}`);
    expect(await screen.findByText("This run was interrupted")).toBeInTheDocument();
    expect(screen.getByText("No final answer")).toBeInTheDocument();
    expect(screen.queryByText("LLM calls")).not.toBeInTheDocument();
  });

  it("shows a clear not-found state", async () => {
    mockApi({ "GET /runs/0123": reply(404, { error: { code: "run_not_found", message: "no run with this id" } }) });
    renderApp("/runs/0123");
    expect(await screen.findByText("Run not found")).toBeInTheDocument();
  });

  it("shows cost only when pricing is configured", async () => {
    const priced = { ...health, pricing_configured: true };
    const withCost = clone(optimizedRun.run) as Record<string, unknown>;
    withCost.estimated_cost = { llm: 0.0003, currency: "USD", pricing_source: "test", pricing_retrieved_on: "2026-01-01", note: "" };
    mockApi({ "GET /health": priced, ...runRoutes(optimizedRun, withCost) });
    renderApp(`/runs/${optimizedRun.run.run_id}`);
    expect(await screen.findByText("Estimated cost")).toBeInTheDocument();
    expect(screen.getByText("0.000300 USD")).toBeInTheDocument();
  });

  it("says why cost is unavailable when pricing exists but usage was not reported", async () => {
    mockApi({ "GET /health": { ...health, pricing_configured: true }, ...runRoutes(optimizedRun) });
    renderApp(`/runs/${optimizedRun.run.run_id}`);
    expect(await screen.findByText("no complete usage reported")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------- launcher & compare

describe("run launcher", () => {
  it("disables Gemini when it is not configured", async () => {
    mockApi({ "GET /datasets/core/1": dataset });
    renderApp("/");
    expect(await screen.findByText(/How much does Orbit cost/)).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Gemini" })).toBeDisabled();
    await userEvent.click(screen.getByRole("radio", { name: "Your own task" }));
    expect(screen.getByRole("note")).toHaveTextContent("Gemini isn't configured");
    expect(screen.getByRole("button", { name: "Run both modes" })).toBeDisabled();
  });

  it("starts both modes in separate new sessions and opens the comparison", async () => {
    const created = [baselineRun.run.run_id, optimizedRun.run.run_id];
    const { calls } = mockApi({
      "GET /datasets/core/1": dataset,
      "POST /runs": seq(
        { run_id: created[0], status: "queued", links: {} },
        { run_id: created[1], status: "queued", links: {} },
      ),
      ...runRoutes(baselineRun as unknown as Bundle),
      ...runRoutes(optimizedRun),
    });
    renderApp("/");
    await screen.findByText(/How much does Orbit cost/);
    await userEvent.click(screen.getByRole("button", { name: "Run both modes" }));

    expect(await screen.findByRole("heading", { name: "Baseline vs optimized" })).toBeInTheDocument();
    const posts = calls.filter((c) => c.method === "POST");
    expect(posts.map((p) => (p.body as { mode: string }).mode)).toEqual(["passthrough", "optimized"]);
    for (const p of posts) {
      expect(p.body).toMatchObject({ scenario: { task_id: "redundant_orbit_price" }, llm: "scripted" });
    }
    const sessions = posts.map((p) => (p.body as { session_id: string }).session_id);
    expect(new Set(sessions).size).toBe(2);

    const strip = await screen.findByLabelText("Comparison");
    expect(within(strip).getByText("4 → 2")).toBeInTheDocument();
    expect(within(strip).getByText("Identical")).toBeInTheDocument();
  });

  it("shows the server's error when a run cannot start", async () => {
    mockApi({
      "GET /datasets/core/1": dataset,
      "POST /runs": reply(503, { error: { code: "too_many_sessions", message: "all sessions are busy; try again later" } }),
    });
    renderApp("/");
    await screen.findByText(/How much does Orbit cost/);
    await userEvent.click(screen.getByRole("radio", { name: "Optimized" }));
    await userEvent.click(screen.getByRole("button", { name: "Start run" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("all sessions are busy");
  });
});

// ---------------------------------------------------------------- history

describe("run history", () => {
  it("invites the first run when empty", async () => {
    mockApi({ "GET /runs": [] });
    renderApp("/runs");
    expect(await screen.findByText("No runs yet")).toBeInTheDocument();
    const main = screen.getByRole("main");
    expect(within(main).getByRole("link", { name: "Run a task" })).toHaveAttribute("href", "/");
  });

  it("lists runs and shows missing metrics as dashes", async () => {
    mockApi({ "GET /runs": [activeVersion(optimizedRun), baselineRun.run] });
    renderApp("/runs");
    const rows = await screen.findAllByRole("row");
    const active = rows[1]!;
    expect(within(active).getByText("Running")).toBeInTheDocument();
    expect(within(active).getAllByText("—")).toHaveLength(3);
    expect(within(rows[2]!).getByText("Finished")).toBeInTheDocument();
    expect(within(rows[2]!).getAllByText("4")).toHaveLength(2); // baseline: 4 requests, 4 executions
  });
});

// ---------------------------------------------------------------- evaluations

describe("evaluation page", () => {
  const id = evaluation.evaluation_id;

  it("labels mock results and shows the comparison from the real report", async () => {
    mockApi({ [`GET /evaluations/${id}`]: evaluation });
    renderApp(`/evaluations/${id}`);
    expect(await screen.findByText("MOCK results")).toBeInTheDocument();
    expect(screen.getByText("No regressions")).toBeInTheDocument();
    const cold = evaluation.report.phases.cold.modes;
    expect(screen.getAllByText(`${cold.passthrough.tool_executions} → ${cold.optimized.tool_executions}`).length).toBeGreaterThan(0);
    expect(screen.getAllByText("not reported").length).toBeGreaterThan(0); // tokens
    expect(screen.queryByText("Cost")).not.toBeInTheDocument(); // no pricing configured
    expect(screen.getByRole("heading", { name: /Warm cache/ })).toBeInTheDocument();
  });

  it("lists regressions with links to both runs", async () => {
    const withRegression = clone(evaluation);
    withRegression.report.regressions = [{
      task_id: "freshness_price_changed", phase: "cold", repetition: 1,
      baseline_run_id: "b".repeat(32), optimized_run_id: "c".repeat(32),
      optimized_failed_checks: ["number:45.91"],
    }] as never;
    mockApi({ [`GET /evaluations/${id}`]: withRegression });
    renderApp(`/evaluations/${id}`);
    const link = await screen.findByRole("link", { name: "Compare runs" });
    expect(link).toHaveAttribute("href", `/compare?base=${"b".repeat(32)}&opt=${"c".repeat(32)}`);
    expect(screen.getByText("number:45.91")).toBeInTheDocument();
    expect(screen.queryByText("No regressions")).not.toBeInTheDocument();
  });

  it("labels real-model results as LIVE", async () => {
    const live = clone(evaluation);
    live.report.provider = "gemini";
    live.report.model_id = "gemini-x";
    mockApi({ [`GET /evaluations/${id}`]: live });
    renderApp(`/evaluations/${id}`);
    expect(await screen.findByText("LIVE results from gemini-x")).toBeInTheDocument();
  });

  it("shows progress while running and an explanation when interrupted", async () => {
    const running = { ...clone(evaluation), status: "running", finished: false, report: null, completed_runs: 3 };
    mockApi({ [`GET /evaluations/${id}`]: running });
    renderApp(`/evaluations/${id}`);
    expect(await screen.findByRole("progressbar")).toBeInTheDocument();
    expect(screen.getByText("The report appears when every run has finished.")).toBeInTheDocument();
  });

  it("explains an interrupted evaluation", async () => {
    const interrupted = { ...clone(evaluation), status: "interrupted", finished: true, report: null, completed_runs: 3 };
    mockApi({ [`GET /evaluations/${id}`]: interrupted });
    renderApp(`/evaluations/${id}`);
    expect(await screen.findByText("This evaluation was interrupted")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------- shell & tools

describe("app shell", () => {
  it("tells the user when the backend is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    renderApp("/");
    expect(await screen.findByText("Backend unreachable")).toBeInTheDocument();
    expect(await screen.findByText("Can't reach the backend")).toBeInTheDocument();
  });

  it("lists tools with their safety rules", async () => {
    mockApi({ "GET /tools": tools });
    renderApp("/tools");
    const row = (await screen.findByText("send_notification")).closest("tr")!;
    expect(within(row).getByText("Changes data")).toBeInTheDocument();
    expect(within(row).getByText("Never reused: always executes")).toBeInTheDocument();
    const calc = screen.getByText("calculator").closest("tr")!;
    expect(within(calc).getByText(/Reusable indefinitely/)).toBeInTheDocument();
  });

  it("shows a not-found page for unknown addresses", async () => {
    mockApi({});
    renderApp("/nope");
    expect(await screen.findByText("Page not found")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("Backend connected")).toBeInTheDocument());
  });
});
