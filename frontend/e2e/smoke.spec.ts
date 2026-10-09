// Browser smoke tests: the full demo workflow in a real browser against the
// real backend (mock agent only). One worker, in order: later steps use what
// earlier ones created. Screenshots go to e2e/.tmp/ for a human to look at.
import { type ConsoleMessage, type Page, expect, test } from "@playwright/test";

test.describe.configure({ mode: "serial" });

const shots = (page: Page, name: string) =>
  page.screenshot({ path: `e2e/.tmp/${name}.png`, fullPage: true });

let consoleErrors: string[] = [];
test.beforeEach(({ page }) => {
  consoleErrors = [];
  page.on("console", (msg: ConsoleMessage) => {
    // Expected 4xx answers (e.g. the not-found test) are logged by the browser
    // as "Failed to load resource"; server errors are caught below instead.
    if (msg.type() === "error" && !msg.text().startsWith("Failed to load resource")) {
      consoleErrors.push(msg.text());
    }
  });
  page.on("response", (res) => {
    if (res.status() >= 500) consoleErrors.push(`${res.status()} ${res.url()}`);
  });
  page.on("pageerror", (err) => consoleErrors.push(String(err)));
});
test.afterEach(() => {
  expect(consoleErrors, "browser console errors").toEqual([]);
});

test("app shell connects to the backend and every page opens", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Backend connected")).toBeVisible();
  await expect(page.getByText("Gemini not configured")).toBeVisible();
  for (const [link, heading] of [
    ["Run history", "Run history"],
    ["Evaluations", "Evaluations"],
    ["Learning", "Learning"],
    ["Tools", "Tools"],
    ["Run a task", "Run a task"],
  ]) {
    await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: link, exact: true }).click();
    await expect(page.getByRole("heading", { level: 1, name: heading })).toBeVisible();
  }
  await page.goto("/no/such/page");
  await expect(page.getByText("Page not found")).toBeVisible();
});

test("compare both modes: same answer, fewer executions, mock label", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("radio", { name: /How much does Orbit cost per user per month/ })).toBeChecked();
  // Gemini is not configured, so it cannot be chosen.
  await expect(page.getByRole("radio", { name: "Gemini" })).toBeDisabled();
  await page.getByRole("button", { name: "Run both modes" }).click();
  await expect(page).toHaveURL(/\/compare\?base=/);
  const strip = page.getByLabel("Comparison");
  await expect(strip).toContainText("4 → 2");
  await expect(strip).toContainText("Identical");
  await expect(page.getByTestId("results-label")).toContainText("MOCK results");
  await shots(page, "compare");

  await page.getByRole("link", { name: /^Open run/ }).nth(1).click();
  await expect(page.getByRole("heading", { name: "What happened to each request" })).toBeVisible();
  const routes = page.getByTestId("tool-route");
  await expect(routes).toHaveCount(4);
  await expect(routes.nth(1)).toHaveAttribute("data-route", "reused");
  await expect(page.getByLabel("tool executions")).toContainText("2");
  await shots(page, "run-optimized");
});

test("a stale price is refreshed, not reused", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("radio", { name: /current price of ORB-X/ }).check();
  await page.getByRole("button", { name: "Run both modes" }).click();
  await expect(page.getByLabel("Comparison")).toContainText("2 → 2");
  await expect(page.getByLabel("Comparison")).toContainText("Identical");
  await page.getByRole("link", { name: /^Open run/ }).nth(1).click();
  await expect(page.getByTestId("tool-route").nth(1)).toHaveAttribute("data-route", "live");
  await expect(page.getByText(/too old|stale/i).first()).toBeVisible();
  await shots(page, "run-stale-refreshed");
});

test("history lists the runs", async ({ page }) => {
  await page.goto("/runs");
  await expect(page.getByRole("heading", { name: "Run history" })).toBeVisible();
  await expect(page.locator("tbody tr")).toHaveCount(4);
  await shots(page, "history");
});

test("an evaluation runs and reports mock results honestly", async ({ page }) => {
  await page.goto("/evaluations");
  await page.getByLabel("Repetitions").fill("1");
  await page.getByRole("button", { name: "Start evaluation" }).click();
  await expect(page).toHaveURL(/\/evaluations\/[0-9a-f]{32}/);
  await expect(page.getByText("MOCK results")).toBeVisible({ timeout: 90_000 });
  await expect(page.getByText(/do not show what a real Gemini agent would save/)).toBeVisible();
  await expect(page.getByText("No regressions")).toBeVisible();
  await shots(page, "evaluation");
});

test("learning lifecycle: evidence, mining, validation, approval, effect, disable", async ({ page }) => {
  test.setTimeout(240_000);
  await page.goto("/learning");
  await expect(page.getByText("No learned rules yet")).toBeVisible();
  await page.getByRole("button", { name: "Run a training evaluation" }).click();
  await expect(page).toHaveURL(/\/evaluations\/[0-9a-f]{32}/);
  await expect(page.getByText("MOCK results")).toBeVisible({ timeout: 120_000 });

  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Learning" }).click();
  await page.getByRole("button", { name: "Mine proposals" }).click();
  const mining = page.getByLabel("Latest mining");
  await expect(mining).toContainText("share within one session");
  await expect(mining).toContainText("never learned");
  await shots(page, "learning-mined");

  const rulesSection = page.locator("section", { has: page.getByRole("heading", { name: "3–4. Rules" }) });
  await rulesSection.getByRole("row", { name: /Sharing scope/ }).getByRole("link").click();
  await expect(page.getByRole("heading", { name: "Evidence" })).toBeVisible();
  await page.getByLabel("Repetitions").fill("1");
  await page.getByRole("button", { name: "Validate" }).click();
  await expect(page.getByText("Passed every check")).toBeVisible({ timeout: 120_000 });
  await expect(page.getByTestId("results-label")).toContainText("MOCK results");
  await shots(page, "rule-validated");

  const approve = page.getByRole("form", { name: "Approve rule" });
  await expect(approve.getByRole("button", { name: "Approve and activate" })).toBeDisabled();
  await approve.getByLabel("Your name").fill("Demo presenter");
  await approve.getByLabel(/I reviewed the evidence/).check();
  await approve.getByRole("button", { name: "Approve and activate" }).click();
  await expect(page.locator(".page-head .status")).toHaveText("Active");
  await shots(page, "rule-active");

  await page.goto("/tools");
  await expect(page.getByRole("row", { name: /record_read/ }).getByRole("link", { name: /Shared within session/ })).toBeVisible();

  // Effect in real runs: the second run in the same session reuses the read.
  await page.goto("/");
  await page.getByRole("radio", { name: /current status of ticket REC-1003/ }).check();
  await page.getByRole("radiogroup", { name: "Gateway mode" }).getByText("Optimized", { exact: true }).click();
  await page.getByRole("button", { name: "Start run" }).click();
  await expect(page.getByLabel("tool executions")).toContainText("1");
  await page.getByRole("button", { name: "Run again in this session" }).click();
  await expect(page.getByText("Reused because of an approved learned rule")).toBeVisible();
  await expect(page.getByLabel("tool executions")).toContainText("0");
  await shots(page, "run-learned-rule");

  await page.goto("/learning");
  await rulesSection.getByRole("row", { name: /Sharing scope/ }).getByRole("link").click();
  const disable = page.getByRole("form", { name: "Disable rule" });
  await disable.getByLabel("Reason").fill("demo finished");
  await disable.getByRole("button", { name: "Disable rule" }).click();
  await expect(page.locator(".page-head .status")).toHaveText("Disabled");
});

test("an unknown run shows a not-found state", async ({ page }) => {
  await page.goto(`/runs/${"0".repeat(32)}`);
  await expect(page.getByText(/not found|no run/i).first()).toBeVisible();
});

test("pages fit a phone-width screen without sideways scrolling", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  for (const path of ["/", "/runs", "/evaluations", "/learning", "/tools"]) {
    await page.goto(path);
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow, `horizontal overflow on ${path}`).toBeLessThanOrEqual(1);
  }
  await page.goto("/learning");
  await shots(page, "phone-learning");
});
