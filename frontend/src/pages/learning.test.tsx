import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import evaluationList from "../test/fixtures/evaluation_list.json";
import rulesAfterApproval from "../test/fixtures/learning_rules_after_approval.json";
import rules from "../test/fixtures/learning_rules.json";
import mining from "../test/fixtures/mining.json";
import minings from "../test/fixtures/minings_list.json";
import ruleApproved from "../test/fixtures/rule_approved.json";
import ruleActive from "../test/fixtures/rule_detail_active.json";
import ruleProposed from "../test/fixtures/rule_detail_proposed.json";
import ruleValidated from "../test/fixtures/rule_detail_validated.json";
import ruleDisabled from "../test/fixtures/rule_disabled.json";
import toolsWithRule from "../test/fixtures/tools_with_rule.json";
import validation from "../test/fixtures/validation_passed.json";
import { mockApi, renderApp, reply, seq } from "../test/mockApi";

const clone = <T,>(v: T): T => JSON.parse(JSON.stringify(v)) as T;
const rid = ruleValidated.rule.rule_id;
const vid = validation.validation_id;

// A training evaluation of core v2, as the learning page lists it. The recorded
// evaluation list is core v1, so only its dataset field is changed here.
const trainingEvals = [
  { ...clone(evaluationList[0]), dataset: { name: "core", version: "2", checksum: "x" } },
];

describe("learning page", () => {
  it("guides the first steps when nothing has been learned", async () => {
    mockApi({ "GET /learning/rules": [], "GET /learning/minings": [], "GET /evaluations": [] });
    renderApp("/learning");
    expect(await screen.findByText("No learned rules yet")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Mine proposals" })).toBeDisabled();
    expect(screen.getByText(/0 completed/)).toBeInTheDocument();
  });

  it("starts a training evaluation with the warm phase on the train split", async () => {
    const { calls } = mockApi({
      "GET /learning/rules": [],
      "GET /learning/minings": [],
      "GET /evaluations": [],
      "POST /evaluations": reply(202, { evaluation_id: "e".repeat(32), status: "queued", planned_runs: 1, links: {} }),
      [`GET /evaluations/${"e".repeat(32)}`]: evaluationList[0],
      [`GET /evaluations/${"e".repeat(32)}/runs`]: [],
    });
    renderApp("/learning");
    await userEvent.click(await screen.findByRole("button", { name: "Run a training evaluation" }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST")).toBe(true));
    const post = calls.find((c) => c.method === "POST")!;
    expect(post.body).toMatchObject({ dataset_version: "2", split: "train", warm_cache: true, llm: "scripted" });
  });

  it("mines and shows every candidate with its reason, from the real report", async () => {
    const { calls } = mockApi({
      "GET /learning/rules": seq([], rules),
      "GET /learning/minings": [],
      "GET /evaluations": trainingEvals,
      "POST /learning/minings": reply(201, mining),
    });
    renderApp("/learning");
    const mine = await screen.findByRole("button", { name: "Mine proposals" });
    await waitFor(() => expect(mine).toBeEnabled());
    await userEvent.click(mine);

    const summary = await screen.findByLabelText("Latest mining");
    expect(within(summary).getByText(/at least 5 supporting repeats/)).toBeInTheDocument();
    // Proposed and not proposed, each with a reason in plain words.
    expect(within(summary).getAllByText("share within one session").length).toBeGreaterThan(0);
    expect(within(summary).getAllByText("not proposed").length).toBeGreaterThan(0);
    expect(within(summary).getByText(/changed at every gap/)).toBeInTheDocument();
    // Side-effecting tools are listed as never learned.
    const notify = within(summary).getByText("send_notification").closest("tr")!;
    expect(within(notify).getByText("never learned")).toBeInTheDocument();
    expect(calls.find((c) => c.method === "POST")!.body).toEqual({ dataset: "core", dataset_version: "2" });
    // The rules table now lists the new proposals.
    expect(await screen.findAllByText("Proposed")).not.toHaveLength(0);
  });

  it("shows the latest stored mining after a reload", async () => {
    mockApi({ "GET /learning/rules": rules, "GET /learning/minings": minings, "GET /evaluations": [] });
    renderApp("/learning");
    expect(await screen.findByLabelText("Latest mining")).toBeInTheDocument();
    const table = screen.getByRole("heading", { name: "3–4. Rules" }).closest("section")!;
    expect(within(table).getAllByRole("row")).toHaveLength(rules.length + 1);
  });

  it("shows a server error when mining is refused", async () => {
    mockApi({
      "GET /learning/rules": [],
      "GET /learning/minings": [],
      "GET /evaluations": trainingEvals,
      "POST /learning/minings": reply(422, { error: { code: "no_training_data", message: "no completed evaluation" } }),
    });
    renderApp("/learning");
    const mine = await screen.findByRole("button", { name: "Mine proposals" });
    await waitFor(() => expect(mine).toBeEnabled());
    await userEvent.click(mine);
    expect(await screen.findByText("no completed evaluation")).toBeInTheDocument();
  });
});

describe("rule page", () => {
  it("shows the evidence: repeats, runs, consistency and links to supporting runs", async () => {
    mockApi({ [`GET /learning/rules/${rid}`]: ruleProposed });
    renderApp(`/learning/rules/${rid}`);
    expect(await screen.findByText("Proposed", { selector: ".page-head .status" })).toBeInTheDocument();
    const ev = ruleProposed.rule.evidence;
    const repeats = screen.getByText("Supporting repeats").closest("div")!;
    expect(within(repeats).getByText(String(ev.repeat_count))).toBeInTheDocument();
    const consistency = screen.getByText("Consistency").closest("div")!;
    expect(within(consistency).getByText("100%")).toBeInTheDocument();
    expect(screen.getByText("share within one run")).toBeInTheDocument();
    expect(screen.getByText("share within one session")).toBeInTheDocument();
    const sample = screen.getByRole("table", { name: "Supporting repeats" });
    const firstRun = ev.sample[0]!.first_run_id.slice(0, 8);
    expect(within(sample).getAllByRole("link", { name: firstRun })[0]).toHaveAttribute(
      "href",
      `/runs/${ev.sample[0]!.first_run_id}`,
    );
    // Not validated -> no approval form.
    expect(screen.queryByRole("form", { name: "Approve rule" })).not.toBeInTheDocument();
    expect(screen.getByText("Not validated yet.")).toBeInTheDocument();
  });

  it("starts a validation and polls until it finishes", async () => {
    const validating = clone(ruleProposed);
    validating.rule.status = "validating";
    validating.rule.latest_validation_id = vid as never;
    const { calls } = mockApi({
      [`GET /learning/rules/${rid}`]: seq(ruleProposed, validating, validating, ruleValidated),
      [`POST /learning/rules/${rid}/validations`]: reply(202, { validation_id: vid, rule_id: rid, status: "running", links: {} }),
      [`GET /learning/validations/${vid}`]: seq({ ...validation, status: "running", finished: false, result: null }, validation),
    });
    renderApp(`/learning/rules/${rid}`);
    await userEvent.click(await screen.findByRole("button", { name: "Validate" }));
    expect(calls.find((c) => c.method === "POST")!.body).toEqual({ repetitions: 2, llm: "scripted" });
    expect(await screen.findByText("Passed every check", {}, { timeout: 5000 })).toBeInTheDocument();
    expect(await screen.findByText("Validated: awaiting approval", { selector: ".page-head .status" })).toBeInTheDocument();
  });

  it("shows every validation criterion and the off-versus-on comparison", async () => {
    mockApi({ [`GET /learning/rules/${rid}`]: ruleValidated, [`GET /learning/validations/${vid}`]: validation });
    renderApp(`/learning/rules/${rid}`);
    expect(await screen.findByText("Passed every check")).toBeInTheDocument();
    for (const c of validation.result.criteria) {
      expect(screen.getByText(c.label)).toBeInTheDocument();
    }
    const table = screen.getByRole("table", { name: "Rule off versus on" });
    const off = within(table).getByRole("row", { name: /Rule off/ });
    const on = within(table).getByRole("row", { name: /Rule on/ });
    expect(off).toHaveTextContent(String(validation.result.arms.off.tool_executions));
    expect(on).toHaveTextContent(String(validation.result.arms.on.tool_executions));
  });

  it("shows a rejected validation with the failed criteria", async () => {
    // Derived from the real recorded result: one criterion is flipped here to
    // exercise the rejected view (the backend tests produce real rejections).
    const rejected = clone(validation);
    rejected.status = "failed";
    rejected.result.passed = false;
    rejected.result.criteria[3]!.passed = false;
    (rejected.result as { failed_criteria: string[] }).failed_criteria = [rejected.result.criteria[3]!.name];
    const rule = clone(ruleValidated);
    rule.rule.status = "rejected";
    mockApi({ [`GET /learning/rules/${rid}`]: rule, [`GET /learning/validations/${vid}`]: rejected });
    renderApp(`/learning/rules/${rid}`);
    expect(await screen.findByText("Rejected", { selector: "strong" })).toBeInTheDocument();
    expect(screen.getByText(new RegExp(`Failed: ${rejected.result.criteria[3]!.label}`))).toBeInTheDocument();
    expect(screen.queryByRole("form", { name: "Approve rule" })).not.toBeInTheDocument();
  });

  it("requires a name and an explicit confirmation before approving", async () => {
    const { calls } = mockApi({
      [`GET /learning/rules/${rid}`]: seq(ruleValidated, ruleActive),
      [`GET /learning/validations/${vid}`]: validation,
      [`POST /learning/rules/${rid}/approve`]: ruleApproved,
    });
    renderApp(`/learning/rules/${rid}`);
    const form = await screen.findByRole("form", { name: "Approve rule" });
    const submit = within(form).getByRole("button", { name: "Approve and activate" });
    expect(submit).toBeDisabled();
    await userEvent.type(within(form).getByLabelText("Your name"), "Navya");
    expect(submit).toBeDisabled(); // still needs the confirmation
    await userEvent.click(within(form).getByLabelText(/I reviewed the evidence/));
    await userEvent.click(submit);
    await waitFor(() => expect(calls.some((c) => c.path.endsWith("/approve"))).toBe(true));
    expect(calls.find((c) => c.path.endsWith("/approve"))!.body).toEqual({ approved_by: "Navya", confirm: true });
    expect(await screen.findByText("Active", { selector: ".page-head .status" })).toBeInTheDocument();
  });

  it("shows the approval refusal from the server", async () => {
    mockApi({
      [`GET /learning/rules/${rid}`]: ruleValidated,
      [`GET /learning/validations/${vid}`]: validation,
      [`POST /learning/rules/${rid}/approve`]: reply(409, {
        error: { code: "revalidation_required", message: "the active rules changed since this rule was validated" },
      }),
    });
    renderApp(`/learning/rules/${rid}`);
    const form = await screen.findByRole("form", { name: "Approve rule" });
    await userEvent.type(within(form).getByLabelText("Your name"), "Navya");
    await userEvent.click(within(form).getByLabelText(/I reviewed the evidence/));
    await userEvent.click(within(form).getByRole("button", { name: "Approve and activate" }));
    expect(await screen.findByText(/active rules changed/)).toBeInTheDocument();
  });

  it("disables an active rule with a reason", async () => {
    const disabledDetail = clone(ruleActive);
    disabledDetail.rule = { ...disabledDetail.rule, ...clone(ruleDisabled) } as never;
    const { calls } = mockApi({
      [`GET /learning/rules/${rid}`]: seq(ruleActive, disabledDetail),
      [`GET /learning/validations/${vid}`]: validation,
      [`POST /learning/rules/${rid}/disable`]: ruleDisabled,
    });
    renderApp(`/learning/rules/${rid}`);
    expect(await screen.findByText("Active", { selector: ".page-head .status" })).toBeInTheDocument();
    expect(screen.getAllByText(/by Navya/).length).toBeGreaterThan(0);
    const form = screen.getByRole("form", { name: "Disable rule" });
    const submit = within(form).getByRole("button", { name: "Disable rule" });
    expect(submit).toBeDisabled();
    await userEvent.type(within(form).getByLabelText("Reason"), "demo finished");
    await userEvent.click(submit);
    await waitFor(() => expect(calls.some((c) => c.path.endsWith("/disable"))).toBe(true));
    expect(calls.find((c) => c.path.endsWith("/disable"))!.body).toEqual({ reason: "demo finished" });
    expect(await screen.findByText("Disabled", { selector: ".page-head .status" })).toBeInTheDocument();
  });

  it("toggles shadow mode", async () => {
    const shadow = clone(ruleProposed);
    shadow.rule.status = "shadow";
    const { calls } = mockApi({
      [`GET /learning/rules/${rid}`]: seq(ruleProposed, shadow),
      [`POST /learning/rules/${rid}/shadow`]: shadow.rule,
    });
    renderApp(`/learning/rules/${rid}`);
    await userEvent.click(await screen.findByRole("button", { name: "Observe in shadow" }));
    expect(calls.find((c) => c.method === "POST")!.body).toEqual({ enabled: true });
    expect(await screen.findByRole("button", { name: "Stop observing" })).toBeInTheDocument();
    expect(screen.getByText(/none yet/)).toBeInTheDocument();
  });

  it("shows a not-found page for an unknown rule", async () => {
    mockApi({ [`GET /learning/rules/${"0".repeat(32)}`]: reply(404, { error: { code: "rule_not_found", message: "no such rule" } }) });
    renderApp(`/learning/rules/${"0".repeat(32)}`);
    expect(await screen.findByText("Page not found")).toBeInTheDocument();
  });
});

describe("tools page with a learned rule", () => {
  it("shows the active rule next to the tool's own setting", async () => {
    mockApi({ "GET /tools": toolsWithRule });
    renderApp("/tools");
    const row = (await screen.findByText("record_read")).closest("tr")!;
    const link = within(row).getByRole("link", { name: /Shared within session/ });
    expect(link).toHaveAttribute("href", `/learning/rules/${rid}`);
    expect(within(row).getByText("run")).toBeInTheDocument(); // the tool's own scope
    const notify = screen.getByText("send_notification").closest("tr")!;
    expect(within(notify).getByText("Never learned")).toBeInTheDocument();
  });

  it("lists rule statuses after approval", async () => {
    mockApi({ "GET /learning/rules": rulesAfterApproval, "GET /learning/minings": [], "GET /evaluations": [] });
    renderApp("/learning");
    expect(await screen.findByText("Active")).toBeInTheDocument();
  });
});

describe("run timeline with a learned rule", () => {
  it("marks a reuse that happened because of an approved rule", async () => {
    const { default: bundle } = await import("../test/fixtures/learned_rule_run.json");
    const id = bundle.run.run_id;
    mockApi({
      [`GET /runs/${id}`]: bundle.run,
      [`GET /runs/${id}/details`]: bundle.details,
      [`GET /runs/${id}/trace`]: bundle.trace,
    });
    renderApp(`/runs/${id}`);
    const routes = await screen.findAllByTestId("tool-route");
    expect(routes.map((r) => r.dataset.route)).toEqual(["reused"]);
    expect(within(routes[0]!).getByText("Reused because of an approved learned rule")).toBeInTheDocument();
  });
});

describe("mock versus live labels", () => {
  it("the comparison says the numbers are mock and do not show real Gemini savings", async () => {
    const { default: base } = await import("../test/fixtures/baseline_run.json");
    const { default: opt } = await import("../test/fixtures/optimized_run.json");
    const routes: Record<string, unknown> = {};
    for (const b of [base, opt]) {
      routes[`GET /runs/${b.run.run_id}`] = b.run;
      routes[`GET /runs/${b.run.run_id}/details`] = b.details;
      routes[`GET /runs/${b.run.run_id}/trace`] = b.trace;
    }
    mockApi(routes);
    renderApp(`/compare?base=${base.run.run_id}&opt=${opt.run.run_id}`);
    const label = await screen.findByTestId("results-label");
    expect(label).toHaveTextContent("MOCK results");
    expect(label).toHaveTextContent("do not show what a real Gemini agent would save");
  });

  it("a validation result carries the mock label", async () => {
    mockApi({ [`GET /learning/rules/${rid}`]: ruleValidated, [`GET /learning/validations/${vid}`]: validation });
    renderApp(`/learning/rules/${rid}`);
    expect(await screen.findByTestId("results-label")).toHaveTextContent("MOCK results");
  });
});
