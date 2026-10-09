# Stage 1: Tools and ToolGateway (pass-through mode)

## What exists

- **ToolGateway** (`app/gateway/gateway.py`): the single wrapper for every tool
  request. Pass-through (baseline) mode only: every permitted, valid request is
  executed; nothing is cached, reused or skipped.
- **Tool registry and metadata** (`app/tools/spec.py`, `app/tools/registry.py`):
  name, version, argument schema, effect (read-only / side-effect / unknown),
  freshness (static / ttl / live), TTL and TTL ceiling, scope, resource tags,
  invalidation tags, timeout, retry and idempotency flags. Unsafe combinations
  are rejected when a spec is created.
- **Mock tools** (`app/tools/builtin/`): `calculator`, `kb_search`, `doc_read`,
  `price_lookup`, `record_read`, `record_update`, `send_notification`, plus
  test-only tools for failure cases (`app/tools/test_only/`).
- **Fixed reuse rule** (`app/optimizer/eligibility.py`): only read-only, non-live
  tools may ever be reused. Not used for caching yet; Stage 3 must consult it.
- **Safety** (`app/safety/permissions.py`): per-task tool allowlist, explicit
  permission for side-effecting and unknown tools, per-run request limit,
  argument size limit.
- **Trace events** (`app/telemetry/trace.py`): two events per request
  (`tool_call.requested`, `tool_call.completed`) with decision, timestamps,
  durations, validated arguments and error codes.
- **LLM-ready declarations** (`app/tools/declarations.py`): plain JSON Schema per
  tool for a function-calling API. Not yet tested against the Gemini SDK.

## Gateway decisions

| Decision | Tool code ran? | When |
|---|---|---|
| `executed_live` | yes | valid, permitted, tool succeeded |
| `failed` | yes | tool raised an error, crashed, or returned non-JSON output |
| `blocked_unknown_tool` | no | no tool with that name |
| `blocked_invalid_args` | no | arguments missing, wrong type, extra, or too large |
| `blocked_permission` | no | tool not allowed, or side-effecting/unknown without permission |
| `blocked_limit` | no | the run's request limit was exceeded (run stops) |
| `blocked_run_stopped` | no | any request after the run stopped |
| `blocked_safety_error` | no | a safety check itself crashed (fail closed) |

## How bypassing the gateway is prevented

1. Tool handlers are stored privately in the registry; there is no public way to
   get them. Only the gateway calls the private accessor.
2. Static architecture rules (`tests/architecture/import_rules.py`), checked on
   every test run: tool implementations may only be imported inside
   `app/tools/`; future consumer packages (`agent`, `llm`, `api`, `evaluation`,
   `learning`) may not import `app.tools` at all; private accessors may only be
   used in the gateway; no dynamic imports.
3. Runtime check: for a mixed workload, the gateway's per-tool execution counts
   must equal each tool's own internal counts, and blocked calls reach no tool.

This is a structural guarantee for our own code, enforced by tests. It is not a
security sandbox against deliberately malicious code in the same process.

## Known limitations (Stage 1)

- No caching, reuse or deduplication yet (by design: baseline first).
- No tool timeouts or retries yet. A tool that hangs would block its call.
  Planned for Stage 3. Transient errors are reported to the agent, not retried.
- The per-run request limit is the only loop protection. Identical-call limits,
  step limits and run timeouts arrive with the agent loop (Stage 2) and the
  optimizer (Stage 3).
- No limit on tool output size yet; mock outputs are small. A consistent
  truncation rule is needed before real tools are added.
- Traces are kept in memory only (SQLite arrives in Stage 4). Validated
  arguments are stored in full; masking of personal data is planned (P1).
- Tool metadata lives in code, not yet in `config/tools.yaml`.
- `send_notification` is marked non-retryable until Stage 3 defines when a retry
  is safe (only with an idempotency key).
- Tool declarations are not yet verified against the real Gemini SDK.
- Concurrency was tested with asyncio inside one process only; there is no
  multi-process or database-level concurrency yet.
