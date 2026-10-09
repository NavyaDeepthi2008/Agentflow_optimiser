# Stage 5: Evaluation engine

## What exists

- **Dataset** (`app/evaluation/datasets/core_v1.json`, loaded by `dataset.py`):
  18 tasks over the fictional mock tools.
  - **Categories:** `redundant` (4), `freshness` (4), `side_effect` (3),
    `failure` (4) and `control` (3).
  - **Splits:** `train` and `heldout`, so the later learning stage can be
    tested on tasks it never learned from.
  - **Checksum:** the file's SHA-256 hash is stored with every evaluation.

  Each task has:
  - `task`: the only text the agent sees.
  - `expected`: hidden checks on the run status, the answer (required phrases,
    forbidden phrases such as a stale price, numbers with a tolerance), and the
    mock world's final state (outbox size, record fields).
  - `script`: a reference behavior for the scripted LLM. Its answers are built
    from the tool results the agent really received, so a wrong or stale result
    from the optimizer produces a wrong answer.
  - `setup`: fault injection, for example one simulated outage.
  - `scripted_only`: tasks that need simulated time to pass mid-run. They are
    skipped (and listed as skipped) for real-model evaluations.
- **Grading** (`grading.py`): deterministic, never an LLM. A run is correct
  only if every check passes, and every check's result is stored.
- **Runner** (`runner.py`): runs the same tasks in each mode.
  - Every measured run gets a new session, mock world and gateway, so it starts
    with a cold cache.
  - Modes are interleaved per task, in an order shuffled with a fixed seed.
  - Runs execute one at a time, with the same LLM client, prompt, tool
    definitions, permissions, limits and retry policy in both modes.
  - Repetitions are supported (1–20).
  - **Warm phase** (optional): each task runs once as a warm-up in the same
    session, simulated time moves forward by `warm_gap_seconds`, then the task
    runs again and is measured. Warm results are reported separately from cold
    ones. Tasks whose warm-up changes the world or the simulated time are
    excluded from the warm phase (listed with the reason).
- **Data clock vs timing clock**: the gateway now takes a separate data clock
  (`OffsetClock`) for TTL and mock data. Scripts can simulate "five minutes
  later" while latency stays real.
- **Metrics** (`metrics.py`), per run:
  - tool requests vs actual tool executions
  - cache hits, in-run reuse and in-flight sharing
  - retries and LLM calls
  - tokens (null when not reported)
  - latency
  - gateway overhead: time in the gateway outside tool code
  - LLM and tool cost

  Cost is computed only when a dated pricing file is configured
  (`PRICING_FILE`, see `backend/pricing.example.json`) and usage is complete;
  otherwise it is null. No prices are shipped.
- **Report** (`report.py`), per phase and mode:
  - accuracy and status counts
  - totals
  - latency median, p95 (nearest rank), min and max
  - gateway overhead median and p95
  - tokens and cost (only when available for every run)
  - a baseline vs optimized comparison, plus a per-category and per-task breakdown

  Every regression (baseline correct, optimized wrong, for the same task,
  phase and repetition) is listed individually, with the failed checks and both
  run ids. The `label` says MOCK (scripted) or LIVE MODEL (Gemini).
- **Storage**: migration `0002` adds `evaluations` and `evaluation_runs`. Each
  agent run made by an evaluation, including warm-up runs, is also stored as a
  normal run with full details and trace.
- **API**:
  - `POST /evaluations` (202), `GET /evaluations`, `GET /evaluations/{id}`
    (with the report once completed), `GET /evaluations/{id}/runs`
  - `GET /datasets/{name}/{version}` (task list only, with no expectations or
    scripts)
  - One evaluation runs at a time. Unfinished evaluations are marked
    `interrupted` at shutdown or at the next startup.
- **CLI**: `python -m app.evaluation.cli [--reps N] [--warm] [--llm gemini]`.

## Example scripted result

This is a mock run: the scripted reference agent with mock tools. The numbers
show only what the optimizer does to the redundancy written into this
dataset's scripts. They say nothing about real model behavior.

`python -m app.evaluation.cli --reps 3 --warm` produced, on the development
container:

| Phase | Mode | Runs | Accuracy | Tool requests | Tool executions | Latency p50 |
|---|---|---|---|---|---|---|
| cold | baseline | 54 | 100% | 123 | 120 | 9.53 ms |
| cold | optimized | 54 | 100% | 123 | 87 | 9.70 ms |
| warm | baseline | 30 | 100% | 75 | 72 | 10.76 ms |
| warm | optimized | 30 | 100% | 75 | 6 | 8.80 ms |

There were zero regressions. Cold-cache latency was not better (+1.7%): mock
tools take microseconds, so skipping them saves almost nothing, while each
trace event is written to SQLite in both modes. Timings will differ on other
machines.

## Known limitations (Stage 5)

- **Scripted results are mechanics, not intelligence.** The scripted agent
  repeats requests because its script says so. Real-model savings depend on how
  often a real model repeats itself, which has not been measured.
- **No real-model evaluation yet.** The Gemini evaluation path is tested
  only against a fake server. 4 of the 18 tasks are scripted-only, so a Gemini
  evaluation covers 14.
- **Small dataset.** 18 tasks, written by us. It may favor the optimizer;
  control and failure tasks are included to show where it changes nothing.
  The grading rules are simple: phrases and numbers.
- **Latency includes database writes.** Each trace event is written to SQLite,
  identically in both modes. Gateway overhead therefore includes that
  recording. Latency was measured on one machine; there is no
  significance test.
- **p95 needs many runs.** It uses the nearest-rank method, so with few runs
  it is close to the maximum.
- **Simulated time is approximate.** The data clock is real time plus an
  offset. A real-model run lasting minutes could cross a price boundary;
  expected prices assume runs start well inside the first five-minute price
  window.
- **Warm-phase coverage.** The warm phase covers 10 of the 18 tasks; the others
  are excluded because their warm-up would change the world or the simulated
  time.
- **Runs can overlap with evaluations.** Individual `/runs` requests can still
  run in parallel with an evaluation and affect its latency.
- **No cost figures yet.** No pricing data is shipped, so cost is
  "unavailable" until a pricing file is added.
- **Narrow import exception.** Evaluation code is allowed to import
  `app.tools.mock_world`, to read final state and inject faults. This is a
  documented exception to the import rules; it still cannot import tool
  implementations or the registry.
