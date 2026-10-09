# Stage 7: Evidence-based learning engine

The learning engine proposes conservative changes to how long, and how widely,
the results of read-only tools may be reused. It learns only from correctly
graded training runs, checks every proposal on held-out tasks with the rule off
and on, and activates a rule only after a person approves it.

## What a rule can change, and what it never can

A learned rule changes exactly one setting of one tool:

| Kind | Changes | Allowed range |
|---|---|---|
| `ttl` | How long a stored result may be reused | Above the current TTL, at most the tool's developer-set ceiling (`max_ttl_seconds`). Only for `freshness=ttl` tools. |
| `cache_scope` | Who may reuse it | `run` → `session` only. Never `global`, and never across sessions. |

The following are fixed and can never be changed by a rule: side-effecting
tools, tools with unknown effect, live-data tools, test-only tools,
permissions, validation, the loop guard, limits, retries, invalidation after
writes, and forced-fresh requests.

`check_override` (`app/optimizer/learned.py`) enforces these limits. It runs
at five points:

1. when a proposal is created;
2. before validation;
3. at approval;
4. when rules are loaded at startup (a rule that fails is disabled with the
   reason);
5. by the gateway on every request (a rule that fails is ignored and logged as
   `learning.rule_ignored`).

## Pipeline

1. **Collect evidence.** Run a training evaluation on `core` v2, train split,
   with the warm phase on. The Learning page has a button for this.
2. **Mine** (`app/learning/miner.py`). Only runs that pass all of these are
   used: training split, status `success`, graded correct, and no failed tool
   call. Only live executions are observations; reused results say nothing
   about whether data changed. A *repeat* is two live executions of the same
   request fingerprint.
   - **TTL evidence.** Repeats within one run, with the gap measured on the
     data clock. A pair is skipped if a side-effecting or unknown tool executed
     between the two executions.
   - **Scope evidence.** The graded warm-up run against the graded measured
     run of the same session.
   - **The TTL proposal** is the longest gap at which the output was still
     identical, rounded down. Every repeat up to that gap must have been
     identical, and the proposal must be shorter than the first gap at which
     the output changed. It is capped at the tool's ceiling, and must be
     longer than the current TTL.
   - **The scope proposal** requires every cross-run repeat within the TTL to
     be identical.
   - **Thresholds.** By default, at least 5 supporting repeats from at least 3
     different runs or sessions, with 100% consistency. Callers may raise the
     thresholds but not lower them below 3 repeats and 2 runs. Consistency is
     fixed at 100%.
   - **Reporting.** Every candidate that is not proposed is listed with its
     reason, and tools that may never be learned are listed with theirs.
3. **Shadow** (optional). A shadow rule rides along in ordinary optimized runs
   and never changes behavior:
   - it keeps its own shadow cache;
   - when the real request executes, it records whether it *would* have reused
     a result, and whether that result matched the live one (a mismatch means
     stale data);
   - writes invalidate the shadow cache too.
4. **Validate** (`app/learning/validation.py`).
   - **Tasks:** all held-out tasks, cold phase plus warm phase, with a new
     world, gateway and session for every planned run.
   - **Arms:** two arms in optimized mode with the same active rules. Off =
     candidate in shadow; on = candidate active. They run back to back per
     planned run, in seeded random order.
   - **Rejection:** the rule is rejected if any of the following holds:
     1. any task/phase/repetition that was correct with the rule off is
        incorrect with it on;
     2. accuracy drops;
     3. there is a safety violation: side-effecting tools ran a different
        number of times, an ineligible tool's result was reused, or a world
        check failed;
     4. tool executions did not strictly decrease;
     5. *(stricter than requested)* the shadow arm saw the rule would have
        served data that differed from the live result.
   - **Traceability:** every validation run is stored as a normal run, so it
     can be traced.
5. **Approve.** Only a `validated` rule can be approved. Approval requires:
   - the approver's name and `confirm: true`;
   - that the set of active rules is the same as when the rule was validated
     (otherwise: "validate again").

   Approving a rule disables any older active rule for the same tool and kind.
6. **Disable.** Any rule that is not yet terminal can be disabled or dismissed,
   with a reason. Disabling takes effect on the next request, because freshness
   is judged with the TTL in force *now*, not the one an entry was stored with.

## Lifecycle and persistence

Statuses: `proposed`, `shadow`, `validating`, `validated`, `rejected`,
`active`, `disabled`, `superseded`. Migration `0003` adds:

- `learned_rules`: setting, evidence, approval fields;
- `rule_minings`: full report, including candidates that were not proposed;
- `rule_validations`: config, the active rules it ran with, and the full
  result;
- `rule_events`: an audit log of every status change (who, when, from, to,
  why);
- evidence columns on `tool_calls`: `fingerprint`, `output_digest`,
  `data_time`, `learned_rules`, `shadow`;
- `warmup_run_id` and `warmup_correct` on `evaluation_runs`.

Status changes are conditional updates, so a double click cannot approve a
rule twice. At startup:

- validations interrupted by a restart are marked `interrupted`, and their
  rule returns to its previous status;
- active and shadow rules are reloaded and re-checked.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/learning/minings` | Mine from completed evaluations of a dataset version (default `core` v2) |
| GET | `/learning/minings`, `/learning/minings/{id}` | Stored mining reports |
| GET | `/learning/rules?status=` | List rules |
| GET | `/learning/rules/{id}` | Rule, evidence, audit log, validations, live shadow counts |
| POST | `/learning/rules/{id}/shadow` | `{"enabled": true or false}` |
| POST | `/learning/rules/{id}/validations` | Start held-out validation (`repetitions`, `seed`, `llm`) |
| GET | `/learning/validations/{id}` | Validation result |
| POST | `/learning/rules/{id}/approve` | `{"approved_by", "note"?, "confirm": true}` |
| POST | `/learning/rules/{id}/disable` | `{"reason", "disabled_by"?}` |

Other API changes:

- `/tools` now also returns the effective TTL and scope, the active rules and
  `learning_eligible`.
- `/health` reports `active_validations` and `active_learned_rules`.
- Run `config_snapshot` lists the active and shadow rule ids.
- Evaluation reports list the learned rules that were active.

## Dashboard

**Learning** (`/learning`):
- the four steps;
- "Run a training evaluation";
- "Mine proposals";
- the latest mining report (every candidate and its reason, plus tools that
  are never learned);
- a rules table.

**Rule page** (`/learning/rules/:id`):
- the current and proposed setting;
- the evidence: repeats, runs, consistency, gaps, and the supporting repeats
  with links to their runs;
- shadow toggle and live shadow counts;
- validation: every criterion, off vs on, and a per-task table with run links;
- an approval form (name plus a confirmation checkbox, shown only when
  validated);
- disable/dismiss with a reason;
- the history log.

**Other pages:**
- The Tools page shows active learned rules next to each tool's own setting.
- The run timeline marks reuses caused by a learned rule and shadow
  observations.

## Changes to earlier behavior (deliberate)

- **`record_read` is now shared only within one run by default** (was:
  session). This gives learning a real, evidence-backed decision to make. As a
  result, without an approved rule, a later run in the same session re-reads
  the record. Two earlier tests in `tests/integration/test_optimizer_cache.py`
  were adjusted for this, and a run-scope test was added.
- **Other updates to earlier tests:**
  - `test_health` now expects the two new health fields.
  - `test_a_broken_optimizer_is_caught_as_a_regression` patches
    `CacheEntry.is_fresh` with a signature that accepts the new optional TTL
    argument. Stage 5/6 evaluation numbers for `record_read` tasks in the warm
  phase change accordingly.
- **`core` v2 dataset.** v1 is unchanged. v2 adds 5 tasks: re-checking ticket
  status or owner after 8–10 minutes (train), two single ticket lookups
  (train), and one re-check after 7.5 minutes (held-out).
- **The evaluation runner grades warm-up runs** (stored in `warmup_correct`).
  This does not change reports.

## Bug found and fixed while building this stage

Sibling gateways (`with_mode`) had separate caches. If a **baseline** run in a
session updated a record, the **optimized** gateway of the same session kept
its stale read. Siblings now share one cache store. Only the optimized gateway
reads or stores results, but every sibling's writes invalidate it. This is
covered by `test_write_through_baseline_sibling_invalidates_optimized_cache`.

## Results (actually run)

**Backend:**
- ruff clean.
- pytest: 507 passed, 1 skipped (the live Gemini test), three runs.
  - unit 166
  - integration 142
  - architecture 45
  - storage 19
  - api 60
  - evaluation 37
  - learning 38
- Mutation checks: 19 deliberate breaks of the learning code; all are caught.
  One break survived the first pass ("approve from a non-validated status"),
  because another guard also blocked it; a test now pins that check
  specifically.

**Frontend:**
- tsc and eslint clean.
- vitest: 58 passed, three runs.
- Build OK.

**End-to-end against the real backend**, through the built dashboard's proxy,
with the mock agent:
- Training evaluation: mining used 68 of 76 training runs and proposed two
  rules:
  - `record_read` TTL 300 s → 600 s;
  - `record_read` scope run → session.
- Validating the scope rule (2 repetitions) passed every criterion: tool
  executions went from 38 (off) to 36 (on), with accuracy 100% in both arms.
- After approval, two runs in one session executed `record_read` once, then
  zero times.
- After a server restart, the rule was still active.

These numbers come from the scripted mock agent on a dataset whose redundancy
was written by us. They show that the mechanism works, not that a real agent
saves this much.

## Known limitations (Stage 7)

- **Mock data only.** No validation has been run with Gemini (no key).
  - With a real model, run-to-run variation can flip an answer for reasons
    unrelated to the rule. The strict criteria would then reject a good rule.
  - Use more repetitions in that case. Nothing statistical is done yet.
- **Little evidence available.** The only evidence available is what the
  `core` v2 tasks produce. Real data may change in patterns these tasks don't
  show, for example changes made by other people between two reads. The
  developer-set ceiling (`max_ttl_seconds`) remains the outer bound.
- **Rules don't expire or re-check themselves.** An approved rule stays active
  until disabled. A tool version change disables it at the next startup (and
  the gateway ignores it immediately).
- **Approval has no authentication.** The approver's name is free text; there
  is no login.
- **Shadow counts are approximate.**
  - Live shadow counts scan the most recent 20,000 shadow trace events.
  - Requests that join an identical request already running are not credited
    to a scope rule.
- **Only two kinds of rule.** One active rule per tool and kind; only TTL and
  run → session scope.
- **The UI has not been viewed in a real browser.** It has been checked by
  component tests and through the production build's proxy, but not visually
  (as in Stage 6).
