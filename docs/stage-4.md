# Stage 4: Persistence and API

## What exists

- **Database** (`app/storage/`): SQLAlchemy models plus one Alembic migration
  (`migrations/versions/0001_initial_schema.py`). The schema is created only
  by migrations, and a test checks the migrations match the models exactly.
  SQLite runs with foreign keys on, WAL mode, and a 5 s busy timeout.
- **Tables**:

  | Table | Contents |
  |---|---|
  | `runs` | task, mode, provider/model, status, final answer, error, timestamps, evaluation metrics, request options, config snapshot |
  | `llm_calls` | one row per LLM call: model version, finish reason, latency, token usage, requested tools |
  | `tool_calls` | one row per tool request: decision, executions, cache status/age, forced fresh, invalidations, error, output (capped at 16 KB), timings |
  | `trace_events` | every trace event, written the moment it happens |

  Run metrics stored on `runs`:
  - steps, LLM calls, tool requests, tool executions, reused results, retries
  - tokens: input, output, thinking, cached
  - durations
  - decision counts and model versions

  Token fields stay `NULL` when the provider reported no usage, never 0.
- **Redaction**: every value is passed through `Redactor` before it is stored,
  so the database (and therefore the API) never contains the configured API
  key, or anything shaped like a Google API key.
- **Run service** (`app/services/runs.py`): validates a request, stores a
  `queued` row, and runs the agent in a background asyncio task. At most
  `MAX_CONCURRENT_RUNS` run at once. Each session has its own mock data and
  its own gateways (baseline and optimized), so caches never cross sessions.
- **API** (`app/api/main.py`):

  | Method | Path | Purpose |
  |---|---|---|
  | GET | `/health` | database status, whether Gemini is configured (never the key), active runs |
  | GET | `/tools` | demo tools with schema and safety metadata (no test-only tools) |
  | POST | `/runs` | start a run → `202` with `run_id` (mode, session, `gemini` or `scripted`, tool allowlist, side-effect permission, fresh tools, limits) |
  | GET | `/runs` | list runs (filter by session or status) |
  | GET | `/runs/{id}` | status, final answer, metrics, config |
  | GET | `/runs/{id}/details` | LLM calls and tool calls |
  | GET | `/runs/{id}/trace` | trace events, paged with `after_id` |
  | DELETE | `/sessions/{id}/cache` | clear that session's optimizer cache |

## Run lifecycle

```
queued -> running -> success | failed | limit_steps | limit_tool_calls |
                     limit_time | loop_detected | error
  (any active run) -> interrupted   (server stopped before it finished)
```

- **Final status is atomic.** It is written only by `save_result`, together
  with all LLM calls and tool calls, in one transaction. A run is never shown
  as finished without its details, and `finished` is `false` while queued or
  running.
- **Failures in our code are labelled honestly.** If our own code fails (for
  example, saving the result), the run becomes `error`, never `success`.
- **Shutdown and crashes.** On shutdown, running tasks are cancelled and marked
  `interrupted`. At startup, any run left `queued` or `running` by a crashed
  process is also marked `interrupted`. Trace events written before the stop
  are kept.

## Response and log safety

- **Validation errors** return the field location and message only, never the
  submitted value.
- **Unexpected errors** return a generic 500; details go to the server log.
- **Request bodies are never logged.** Uvicorn's access log shows the method
  and path only.
- **Gemini without a key** is refused with `503 llm_not_configured`, and no
  run is stored.

## Known limitations (Stage 4)

- **No authentication.** Bind the server to `127.0.0.1` only. Anyone who can
  reach the port can start runs and read every run.
- **Blocking database writes.** Writes happen synchronously inside the async
  event loop. Each trace event is one small transaction (fast with SQLite and
  WAL), but under heavy load this would slow request handling. Not
  load-tested.
- **Single process only.** Background runs live inside one server process.
  Running several Uvicorn workers would give each its own sessions, caches and
  run queue. SQLite also limits concurrent writers.
- **Session state is in memory.** Session data (mock world and caches) is lost
  on restart; only run records persist. At most `MAX_SESSIONS` sessions are
  kept, and the least recently used idle session is dropped first.
- **No retention policy yet.** The database grows without limit. Deleting a
  run removes its details (cascade), but there is no API or job for it.
- **Redaction scope.** Redaction only knows the configured key and
  Google-shaped keys. Other personal data (emails, phone numbers) is stored as
  given; masking is planned (P1). Demo data is synthetic.
- **No live updates.** No streaming or WebSocket; clients poll `GET /runs/{id}`.
- **No CORS yet.** CORS settings are not configured; the frontend stage will
  need them.
- **PostgreSQL untested.** Only SQLAlchemy/Alembic features are used, but
  PostgreSQL has not been tested.
- **Still not run live.** As before, nothing has been run against the real
  Gemini API.
