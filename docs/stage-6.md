# Stage 6: Frontend dashboard

React 19 + Vite 8 + TypeScript 5.9, in `frontend/`. Every number on screen
comes from the backend API; nothing is computed from guesses or hardcoded.

## Pages

| Page | What it shows |
|---|---|
| **Run a task** (`/`) | Pick a scenario from the core dataset (task text only; expected answers stay on the server) or write your own task (Gemini only). Choose the agent (mock or Gemini, which is disabled with a reason when not configured) and the mode: Baseline, Optimized, or Compare both. |
| **Run** (`/runs/:id`) | Status, final answer, the request bar ("N tool requests, tool code ran M times"), and the run's metrics. Also the routing timeline: every LLM step and every tool request, drawn as a line that runs through to the tool (executed live), loops back to a stored result (cache hit / reused / shared), or stops at a barrier (blocked), with retries, timeouts and invalidations marked. It also offers "Run again in this session" (to show a warm cache) and "Clear this session's cache". |
| **Compare** (`/compare?base=&opt=`) | Baseline and optimized side by side, with a strip showing tool executions, run time and whether the final answers are identical. Both runs use separate new sessions, so each starts with an empty cache. |
| **Run history** (`/runs`) | All stored runs, filterable by status and session. |
| **Evaluations** (`/evaluations`, `/evaluations/:id`) | Start an evaluation (agent, repetitions, warm phase, split, seed); progress while it runs. The report shows a MOCK or LIVE banner and, per phase: tool executions, accuracy, run time p50 and p95, a per-mode table, a per-category breakdown, every regression with a link comparing both runs, a per-task table, skipped tasks and the measurement method. |
| **Tools** (`/tools`) | Each tool's effect, reuse rule, sharing scope, permission rule and timeout. |

The run metrics on the run page are LLM calls, tokens, run time, time in the
LLM, time in tools, reused results and, only when pricing is configured,
estimated cost.

States handled everywhere:
- **Loading** and **error**: errors show the server's code and message, with
  "Try again". An unreachable backend is called out in the navigation and on
  each page.
- **Empty**: with an action to take.
- **Not found**.
- **Interrupted**: runs and evaluations.
- **Running**: polls about every 0.8 s and stops once finished; no metrics are
  shown until the run has finished.

## Never zero for missing data

- **Unknown values show "—":** `fmtInt`, `fmtMs` and the other formatters
  return "—" for `null` or `undefined`.
- **Tokens the agent did not report** show "not reported" (the mock agent never
  reports tokens).
- **Cost** appears only when the backend has a pricing file
  (`pricing_configured`). If prices exist but usage was not reported, it says
  so instead of showing 0.

## Backend additions for this stage

- **CORS**: `CORS_ORIGINS` (default `http://localhost:5173` and
  `http://127.0.0.1:5173`). `*` and malformed origins are rejected at startup.
  Credentials are not allowed, and only GET, POST and DELETE with
  `Content-Type` are permitted. In development the dashboard does not need
  CORS at all: Vite proxies `/api/*` to `http://127.0.0.1:8000`.
- **Scenario runs**: `POST /runs` with
  `{"scenario": {"task_id": ...}, "llm": "scripted"}` runs a dataset task with
  its server-side reference script. The task text, tools and permissions come
  from the dataset; the expectations are never sent anywhere. Scripted-only
  scenarios refuse Gemini. Each session now has its own data clock, so
  scripted "a few minutes later" steps work in the dashboard too.
- **Health** now includes `gemini_model` (only when configured) and
  `pricing_configured`.
- **Runs** include `estimated_cost` (LLM tokens only), present only with
  pricing data and complete usage.
- **Typed responses**: `/tools` and `/datasets` now have response models.
- **API contract**: `backend/scripts/export_openapi.py` writes
  `backend/openapi.json`, from which `npm run gen:api` generates
  `src/api/schema.gen.ts`. A backend test fails if `openapi.json` is out of
  date.
- **Recorded test data**: `backend/scripts/export_frontend_fixtures.py`
  records real API responses (scenario runs, a retried run, a blocked run, an
  evaluation) into `frontend/src/test/fixtures/`. The frontend tests use these
  instead of hand-written data.

## Running it

```bash
# terminal 1 - backend (local only, no authentication)
cd backend && .venv/bin/uvicorn app.api.main:create_app --factory --host 127.0.0.1 --port 8000

# terminal 2 - dashboard
cd frontend && npm install && npm run dev      # http://127.0.0.1:5173
```

Checks: `npm run typecheck`, `npm run lint`, `npm test`, `npm run build`.

## Known limitations (Stage 6)

- **Not reviewed in a real browser.** Downloading a headless browser was
  blocked by this environment's network policy (`cdn.playwright.dev`).
  - **What I tested instead:** the UI with jsdom component tests, and by
    serving the production build through its proxy against the real backend.
  - **What is unchecked:** the visual design and layout have not been reviewed
    in a browser or with screenshots. Please look at it before the demo.
- **No end-to-end browser tests.** There are no browser-driven tests
  (Playwright or Cypress) yet; the page tests use recorded API responses and a
  fake `fetch`.
- **Polling, not live updates.** Live updates poll the API; there is no
  streaming.
- **No authentication or deployment.** There is no login, and no production
  deployment configuration (the backend is local-only).
- **Accessibility checked by hand only.** Accessibility basics (labels, focus
  styles, reduced motion, status roles) were added but not audited with
  automated tooling or a screen reader.
- **The report is typed by hand.** The evaluation report is free-form JSON on
  the backend, so its frontend type is written by hand and not covered by the
  OpenAPI contract test.
- **Node version requirement.** Node 22.12+ is required (Vite 8 and Vitest 5).
  jsdom is pinned to 29 because jsdom 30 needs Node 22.22.2+.
