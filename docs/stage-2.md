# Stage 2: Gemini integration and the agent loop

## What exists

- **LLM interface** (`app/llm/base.py`): provider-neutral messages, tool calls,
  token usage, request and response. The agent depends only on this.
- **GeminiClient** (`app/llm/gemini.py`), on the official `google-genai` SDK:
  - Tool declarations from the gateway become `FunctionDeclaration`s
    (`parameters_json_schema`).
  - The SDK's automatic function calling is disabled. The SDK never runs tools;
    function calls come back to the agent.
  - The SDK's automatic HTTP retries are disabled (`attempts=1`), so every API
    call is visible and counted.
  - The model's previous turn is sent back unchanged, preserving Gemini thought
    signatures across tool-calling turns.
  - Errors are mapped to: `auth_error`, `rate_limited`, `provider_unavailable`,
    `model_not_found`, `provider_error`, `timeout`, `blocked_response`,
    `empty_response`, `malformed_function_call`. The API key is removed from any
    error text.
  - Temperature is left at the provider default unless configured. Google
    recommends the default for Gemini 3 models and warns that lower values may
    cause looping.
- **ScriptedLLM** (`app/llm/scripted.py`): deterministic replies for offline
  tests. It never reports token usage unless a test scripts it.
- **AgentRunner** (`app/agent/runner.py`): task, then LLM, then tool calls
  through `ToolGateway.call` only, then results back to the LLM, until a final
  answer or a limit.
- **Limits**, the same in every gateway mode:
  - `max_steps`: LLM calls per run.
  - `run_timeout_seconds`: checked before each LLM call, and the LLM call only
    gets the remaining time.
  - `max_tool_requests`: enforced by the gateway.
- **Recording**: trace events `run.started`, `llm_call.completed` /
  `llm_call.failed`, the gateway's `tool_call.*` events, and `run.completed`
  (including the final answer). `RunResult` has per-call model id, reported
  model version, finish reason, latency and token usage. Token totals are
  `None` (unavailable) when the provider reported none, and `complete=false`
  when only some calls reported usage.
- **Settings** (`app/config.py`): environment variables, optionally from
  `backend/.env`. The key is held as a `SecretStr`.
- **CLI** (`python -m app.cli "task"`): one real Gemini run against the mock
  tools, printing a JSON summary.

## Run statuses

| Status | Meaning |
|---|---|
| `success` | the model gave a final answer |
| `failed` | the LLM failed or returned an unusable response (error code recorded) |
| `limit_steps` | `max_steps` LLM calls without a final answer |
| `limit_tool_calls` | the gateway's tool-request limit was reached |
| `limit_time` | the run timeout was reached |
| `error` | unexpected bug in our code (details logged, not shown to users) |

## Architecture rules added

Checked statically on every test run, in `tests/architecture/import_rules.py`:

- R5: LLM clients cannot import the gateway or the agent.
- R6: only `app/llm/gemini.py` may import the Google SDK, and only
  `app/config.py` may import `dotenv`.
- R7: the agent cannot import a specific provider client.
- The CLI is treated like other consumers and may not import tool code.

## Known limitations (Stage 2)

- **Not yet run against the real Gemini API.** No key was available while
  building. All Gemini tests use the real SDK against a fake HTTP server, which
  checks request and response formats offline. Two things only a live run can
  confirm:
  - The SDK sends `parameters_json_schema` and `thinking_level` in snake_case
    in the JSON body. This is the SDK's own behavior, but it is unverified
    against the live API.
  - The default model id `gemini-3.5-flash`, which was taken from Google's
    model list.

  `tests/live/test_gemini_live.py` performs that check when enabled.
- No LLM retries. A rate limit or outage ends the run with `failed`. Whether
  and how to retry is an explicit later decision.
- A malformed function call or an empty reply ends the run. The model is not
  given a second chance.
- Tool calls from one model turn run one after another, not in parallel.
- Tool calls still have no timeout (Stage 3). The run timeout is checked
  between steps, so a hanging tool could hold a run past its timeout.
- Failed LLM calls record no token usage, even though some providers may bill
  partial work.
- No cost estimates yet. That needs a pricing table and is planned with the
  evaluation engine.
- Traces are kept in memory only (SQLite is Stage 4). Task text and tool
  arguments are stored unmasked.
- "Limits are identical in every mode" is guaranteed by the code never reading
  the mode. It can only be tested with pass-through mode until optimized mode
  exists in Stage 3.
