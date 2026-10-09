# Stage 3: Runtime optimization engine

## What exists

All optimization lives inside the existing `ToolGateway` (`app/gateway/gateway.py`).
The decision logic is in `app/optimizer/`:

- `keys.py`: request fingerprints and cache keys
- `cache_store.py`: the cache, with TTL, tags and a generation counter
- `retry.py`: the retry policy
- `eligibility.py`: the fixed reuse rule

The agent and the LLM clients are unchanged, apart from new run statuses.

### Pipeline

Every step except step 7 is identical in both modes:

| # | Step | Both modes? |
|---|---|---|
| 0 | run already stopped | yes |
| 1 | tool-request limit | yes |
| 2 | tool exists | yes |
| 3 | argument validation (strips the reserved `force_fresh` argument) | yes |
| 4 | permission rules, fail closed | yes |
| 5 | identical-request limit (loop guard), fail closed | yes |
| 6 | run deadline | yes |
| 7 | **reuse decision**, fail open | **optimized only** |
| 8 | execute with timeout and retry policy | yes |
| 9 | store result / invalidate after writes / record | store: optimized only |

Reuse happens after validation and safety checks, so a cached result is never
returned for a request that would be refused.

### What can be reused

Only tools that are read-only and not `live`, as decided by
`may_reuse_result`. Side-effecting tools, unknown tools and `live` tools always
execute. They are never cached, skipped, merged or shared, in either mode.

| Decision | Tool ran? | Meaning |
|---|---|---|
| `executed_live` | yes | cache miss, not eligible, or baseline mode |
| `stale_refreshed` | yes | a cached result existed but was past its TTL |
| `forced_fresh` | yes | the caller asked for a fresh result |
| `failed` | yes, unless shared | the tool reported an error (after any allowed retries) |
| `reused_in_run` | no | identical request earlier in this run |
| `cache_hit` | no | identical request in an earlier run of the same scope |
| `shared_in_flight` | no | joined an identical request that was running at the same time |

### Rules

- **Cache keys** combine the tool name, the tool version, the validated
  arguments (defaults filled in) and the scope:
  - `run`: only within one run
  - `session`: across runs of one session
  - `global`: across sessions, only for tools marked public (currently the
    calculator)

  The only extra normalization is per-tool hints for whitespace and case,
  applied where the tool itself ignores those differences.
- **TTL**: a result is fresh while its age is below `ttl_seconds`. Static
  tools never expire. A stale result is deleted and the tool runs again. If
  that refresh fails, the stale value is **not** returned.
- **Invalidation**: after a side-effecting tool runs, even if it failed, the
  cached reads it declares (`invalidates`, e.g. `record:{record_id}`) are
  removed.
  - A side-effecting tool that declares nothing (`invalidates=None`) clears the
    whole cache. So does any unknown tool.
  - `invalidates=()` explicitly means "changes no cached data". This is used by
    `send_notification`.
  - A result fetched while a write happened is not stored (generation check).
- **Forced fresh**: requested by `ToolCallRequest.force_fresh`, by an optional
  `force_fresh` argument from the model, or by a task-level `fresh_tools`
  setting. It is honoured once per identical request per run, so "verify"
  cannot loop.
- **Errors** and results larger than `max_cached_output_bytes` (64 KB) are
  never cached.
- **Timeouts**: every execution has the tool's `timeout_seconds`, shortened to
  the run's remaining time. Sync tools run in a worker thread so they can be
  timed out. A timed-out side effect reports `timeout_outcome_unknown`.
- **Retries** use the same policy in both modes (default 2 retries, with
  exponential backoff and jitter). A retry happens only when all of these hold:
  - the error is transient
  - the tool is marked retryable
  - the tool is read-only, **or** it is side-effecting, supports idempotency
    keys, **and** this request carries a key

  Unknown tools are never retried, and a retry is skipped if the run deadline
  is too close.
- **Loop guard**: more than `max_identical_requests` (default 3) identical
  requests in one run stops the run with status `loop_detected`, in both
  modes.
- **Recording**:
  - Each result has `executions` (attempts), `cache_status`,
    `cache_age_seconds`, `forced_fresh` and `invalidated_entries`.
  - Trace events include `tool_call.attempt_failed`, which says whether a retry
    happened and why not.
  - Gateway stats count requests, executions per tool, decisions, retries,
    timeouts, invalidations, reasons results were not cached, optimizer errors,
    and cache size.
  - `RunResult.summary()` reports tool requests, tool executions, reused
    results and retries separately.
- **Identical tool definitions in both modes**: the optional `force_fresh`
  argument is declared for reusable tools in baseline mode too.
- **Reuse note**: reused results carry a short note for the model. It can be
  turned off with `OptimizerConfig(annotate_reused_results=False)`, since the
  note itself may change model behavior.

## Known limitations (Stage 3)

- **In-flight sharing needs real overlap.** It only helps when identical
  executions truly overlap in time. Near-instant tools usually finish before
  the next request is processed, and then later requests get an ordinary cache
  hit instead. The total number of executions is the same either way.
- **Timed-out sync tools keep running.** Python cannot stop a thread, so a
  timed-out sync tool keeps running in the background until it finishes. For
  side effects this is reported as `timeout_outcome_unknown`. Async tools are
  cancelled.
- **The cache is in memory and per gateway instance.** It is lost on restart
  and not shared between processes. Persistence and multi-process use are not
  implemented.
- **Mock tools run one at a time.** Each mock tool execution holds a lock on
  the mock world, so mock tools never run in parallel with each other. Real
  network tools would not need this.
- **Normalization hints are trusted.** They are written by developers. A wrong
  hint could make two different requests share a result. The hints exist only
  on `kb_search` (whitespace and case) and `calculator` (surrounding
  whitespace).
- **Invalidation is limited to declared tags.** Targeted invalidation only
  covers the tags a tool declares. Data changed outside the gateway (by other
  systems) is only noticed when the TTL expires.
- **No real model tested yet.** The effect of the reuse note on a real model's
  behavior has not been measured. That is evaluation work in a later stage.
- **No LLM retries.** LLM calls are still not retried; this stage only added
  tool retries.
- **No cost estimates yet.** That needs pricing data.
