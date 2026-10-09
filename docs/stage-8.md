# Stage 8: Integration, hardening and demo readiness

There is no new architecture and no new features in this stage. It covers
fixes, tests, a reproducible demo, browser checks and the handoff
documentation.

## Issues found and fixed

| # | Issue | Fix | Test |
|---|---|---|---|
| 1 | **A timed-out synchronous write could leave stale data in the cache.** Python cannot stop a thread. If a side-effecting sync tool timed out, the gateway invalidated affected results at once, but the write could still land *later*. A read cached in between would then be served stale. | The gateway now tracks such an *abandoned* call. Until its thread really finishes, no result that it may change is stored (`not_cached["write_outcome_pending"]`). When it finishes, the affected results are invalidated again, and so is any read that was in progress at that time. Read-only abandoned calls are only counted. New stats: `abandoned_threads`, `late_completions`. New trace events: `tool_call.abandoned`, `tool_call.late_completion`. | 4 tests in `tests/integration/test_optimizer_runtime.py`, including a read that starts before the late write lands and finishes after it. |
| 2 | **One blocking mock tool could freeze the whole server.** Mock sync tools ran their whole body under the world lock. Async tools count their calls under the same lock *on the event loop*, so a sync tool that blocks for a long time would stall every run. | Mock tools that can block (`run_under_lock = False`) lock only their own write. The built-in tools are instant and still run fully under the lock. | Covered by the tests in #1, which failed before this fix. |
| 3 | **On phone-width screens, wide tables stretched the page sideways,** including the navigation (58 px overflow on Evaluations, more on Tools). | Grid columns get `min-width: 0`, so tables scroll inside their own frame. The mining table keeps a readable minimum width and scrolls. | New browser test: no horizontal overflow at 390 px on five pages. |
| 4 | **The brand text ran together** ("AgentFlowTool-call optimizer"), because the generic nav-link style overrode it. | More specific `.nav a.brand` rule. | Seen in screenshots. |
| 5 | **Approval and disable text fields had no styling** (no `type`, so the input styles didn't apply): they rendered as thin lines. | `type="text"`. | Seen in screenshots. |
| 6 | **An active rule's validation still said "The rule may be approved."** | The message now depends on the rule's status. | Seen in screenshots. |
| 7 | **Small wording issues:** "1 tool requests", "would have reused 1 results". | Singular forms. | — |
| 8 | **The comparison and validation views didn't say that mock numbers are not real Gemini savings.** | A shared **MOCK / LIVE** banner on Compare, Evaluation and rule validation. The mock text says the numbers "do not show what a real Gemini agent would save, cost or answer". | 2 UI tests plus browser tests. |

## Gemini readiness: what is verified and what is not

The real Gemini path is `GeminiClient` on the official `google-genai` SDK. It:
- turns off automatic function calling (the SDK never runs tools; all tool
  calls go through the gateway);
- turns off SDK retries;
- sends thought signatures back;
- removes the key from error messages.

**Verified offline:** the genuine SDK runs against a fake HTTP transport, so
no key, no network and no cost are involved. This covers:
- the missing key (503, nothing stored) and a blank key;
- invalid-key responses (400 and 401), with the key absent from API
  responses, traces and the database file;
- an unknown model (404 → `model_not_found`) and the other HTTP error codes;
- the LLM timeout (the run fails cleanly with `timeout` in under the fake
  delay);
- malformed function calls;
- unknown tools, bad arguments and side effects requested by the model, which
  the gateway blocks; the model receives the refusals;
- parallel calls, missing ids or arguments, thinking text, usage parsing
  (including missing usage shown as "not reported", never 0), and cost only
  with pricing plus complete usage;
- baseline vs optimized through the Gemini client: same answer, 3 → 2
  executions, the reuse note sent to the model;
- a Gemini evaluation labelled LIVE that skips tasks needing simulated time.

New this stage: `tests/api/test_gemini_api.py` (7 tests, through the whole
API).

**Not verified (no key was available or used):**
- real model behaviour: which tools it calls, whether it repeats itself,
  answer quality;
- real latency, token counts and cost;
- the live API's exact handling of `thinking_level` and JSON-schema
  parameters;
- whether real runs show any savings at all.

Two optional live tests exist (`RUN_LIVE_TESTS=1 pytest -m live`): one tool
call, and the same task in both modes. They were skipped here.

The default model `gemini-3.5-flash` was listed as a stable model on Google's
model page on 9 Oct 2026 (https://ai.google.dev/gemini-api/docs/models). That
is not a test with a key.

## Correctness and safety audit (what was re-checked)

- **Caching:** TTL expiry, invalidation after writes (now also across sibling
  gateways and after late writes), session and run scope, forced-fresh
  requests, single-flight, the generation check, and retries only for safe
  cases (side effects only with an idempotency key).
- **Loop guard and limits:** both modes.
- **Learned rules:** re-checked on every request. A tampered database row for
  a side-effecting tool is disabled at startup and ignored by the gateway.
  Shadow mismatch detection works. Held-out validation, approval and disabling
  work, and restarts are recovered. Notes and approver names are redacted
  before they are stored.
- **Secrets:** no key-shaped string is tracked in git, there is no `.env` or
  `*.db` in the repository, and the server logs from the smoke runs contain no
  key.
- **Break-on-purpose checks** for the new mitigation: 3 deliberate breaks. All
  were caught after one extra test was added; the first version of the tests
  missed "late completion doesn't invalidate".

**Remaining limits of the timeout mitigation (documented, not solved):**
- An abandoned thread keeps running and keeps occupying a worker thread until
  it finishes. Many hung sync tools could exhaust the default thread pool,
  after which later sync tool calls also wait and time out.
- The outcome is still unknown to the agent (`timeout_outcome_unknown`).
- A read-only tool that times out may be retried while its first thread is
  still running.
- Async tools are really cancelled. Whether a cancelled real-world write
  happened is still unknown, and is reported the same way.

## Demo evidence (actually run, mock agent)

`backend/scripts/demo.py` drives the real API, either in-process or against a
running server with `--base-url`, and `tests/api/test_demo_script.py` checks
its claims on every test run.

**Redundant lookups:**

| Task | Answers | Requests | Executions (baseline → optimized) |
|---|---|---|---|
| redundant_orbit_price | identical | 4 / 4 | 4 → 2 |
| redundant_support_hours | identical | 3 / 3 | 3 → 1 |
| redundant_keystone_total | identical | 4 / 4 | 4 → 2 |

**Cases where reuse would be wrong:**

| Task | What happened | Executions |
|---|---|---|
| freshness_price_changed | stale price refreshed, not reused | 2 → 2 |
| freshness_read_after_write | write invalidated the read | 3 → 3 |
| side_effect_double_reminder | both notifications sent | 2 → 2 |
| failure_loop | stopped by the loop guard in both modes | (looping model, no answer) |

**Evaluation** (core v1, 3 repetitions, in-process run):

| Phase | Accuracy (both modes) | Executions | Run time p50 (baseline / optimized) |
|---|---|---|---|
| cold | 100% | 120 → 87 | 8.5 / 9.0 ms |
| warm | 100% | 72 → 9 | 9.2 / 8.1 ms |

There were 0 regressions. Cost was not reported (no usage, no pricing).

**Learning:**
- 2 rules proposed: `record_read` TTL 300 → 600 s, and scope run → session.
- The scope rule passed held-out validation: 38 → 36 executions, 100%
  accuracy in both arms.
- Two runs in one session: 1 + 1 executions before approval, 1 + 0 after.

All of these are MOCK numbers. Run time in milliseconds against mock tools
says nothing about real latency.

## Browser smoke tests (Playwright, real browser)

`npm run e2e` starts the real backend (fresh database, no key) and the
production build, then runs 8 tests in Chromium. A test fails on any console
error or server 5xx.

1. The app shell connects; every page opens; unknown addresses show not-found.
2. Compare both modes: 4 → 2, identical answers, MOCK banner; the run timeline
   shows the reused routes.
3. A stale price is refreshed and not reused.
4. History lists the runs.
5. An evaluation completes with the MOCK disclaimer and no regressions.
6. The full learning lifecycle in the UI: training evaluation, mining,
   evidence, validation, approval (blocked until a name and the confirmation
   are given), Tools page, a run reused via the rule, disable.
7. Unknown run → not found.
8. No horizontal overflow at phone width.

Results: 8 passed, in each of two runs. Screenshots were reviewed by me; the
issues in #3–#7 came from that review.

**Still needs a human in a browser:**
- overall visual polish and colour contrast;
- keyboard-only navigation and a screen reader;
- browsers other than Chromium;
- the dev server (`npm run dev`) itself. The tests use the production build,
  which shares the same code and proxy settings.

## Final check results

**Backend:**
- `ruff check app tests migrations scripts`: clean.
- pytest: 519 passed, 1 skipped in each of three runs. The skipped item is the
  live Gemini module (2 tests, no key).
  - unit 166
  - integration 146
  - architecture 45
  - storage 19
  - api 68
  - evaluation 37
  - learning 38

**Frontend:**
- `tsc` and `eslint`: clean.
- vitest: 60 passed, in each of three runs.
- `vite build`: OK.
- Playwright: 8 passed, in each of two runs.

**Clean-copy install** (only git-tracked files, new venv, `pip install -e
".[dev]"`, `npm ci`):
- backend: 518 passed, 2 skipped (the extra skip is the git-history secrets
  test, which has no `.git` there);
- frontend: 60 tests passed and the build succeeded.

**Demo:** `scripts/demo.py` passed in-process and against a running server.

**Hygiene:** `.env`, databases, virtual environments, `node_modules`, `dist`
and test artifacts are all git-ignored and none are tracked.
`backend/.env.example` and `frontend/.env.example` contain placeholders only.

## Known limitations (whole project)

- **No live Gemini measurement.** All savings shown are MOCK and come from the
  redundancy written into our datasets. The real Gemini path is tested only
  against a fake HTTP server.
- **Timed-out sync tools can't be stopped** (see above). Async cancellation
  doesn't reveal whether a write happened.
- **Caches live in memory per process.** They are lost on restart (rules and
  history persist). This is single-process only, with no multi-worker
  deployment.
- **The API is local-only, with no authentication.** Approver names are free
  text.
- **Learning is narrow:** only TTL and run → session scope; one active rule
  per tool and kind; rules don't expire and aren't re-validated over time.
  Validation with a real model may reject good rules, because of run-to-run
  variation.
- **The browser checks are smoke tests.** They don't cover visual regression,
  accessibility tooling or other browsers.
- **The dashboard polls;** there are no live updates.
