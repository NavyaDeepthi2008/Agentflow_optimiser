# Demo script (about 6 minutes)

Everything below uses the **mock agent**: no key, no cost, and the same
result every time. Say so at the start. The numbers show what the gateway does
with tool calls, not what a real Gemini agent would save.

## Before the presentation

```bash
# terminal 1
cd backend && rm -f agentflow.db      # optional: start with an empty history
.venv/bin/uvicorn app.api.main:create_app --factory --host 127.0.0.1 --port 8000
# terminal 2
cd frontend && npm run dev            # open http://127.0.0.1:5173
```

To speed up step 5, start a training evaluation about a minute before you
present: open the **Learning** page and click **Run a training evaluation**.
It takes a few seconds. As a backup, have this ready in a third terminal:

```bash
cd backend && .venv/bin/python scripts/demo.py --base-url http://127.0.0.1:8000 --keep-rule
```

## 1. The problem (30 s)

Agents repeat themselves: the same search twice, the same document twice,
re-checking something that has not changed. Every repeat is a real tool call.
But some repeats must happen: a price can change, a record changes after a
write, and a notification is an action, not a lookup.

## 2. Same answer, fewer executions (1 min)

1. **Run a task.** The scenario "How much does Orbit cost…" is preselected.
   Mode: **Compare both**. Agent: **Mock agent**. Click **Run both modes**.
2. Point at:
   - the **MOCK results** banner;
   - **Tool executions 4 → 2**;
   - **Final answers: Identical**.

   Both runs made 4 tool requests; in optimized mode, 2 were answered from an
   earlier identical request.
3. Click **Open run** on the Optimized column. Walk through the routing
   timeline: two lines run all the way to the tool (**Executed live**), and two
   loop back at the gateway (**Reused**: same request earlier in this run).

## 3. When reuse would be wrong (1.5 min)

Back to **Run a task** with **Compare both**, once for each scenario:

- **"What is the current price of ORB-X? … a few minutes later"**: executions
  2 → 2. Open the optimized run: the second request says "A stored result
  existed but was too old, so it was refreshed". The answer uses the new
  price.
- **"Mark ticket REC-1001 as resolved, then confirm its current status…"**
  (Freshness group): the write clears the stored read, so the second read runs
  live and sees "resolved".
- **"Send the message 'Standup at 10' to team:orbit twice…"** (Side effects
  group): 2 → 2 and two messages are sent. Actions are never merged or
  skipped.

Optionally open **Tools** to show which tools can reuse results. Tools that
change data, have an unknown effect or return live data never do.

## 4. Measured, not anecdotal (1 min)

**Evaluations** → Repetitions 3, warm cache on → **Start evaluation** (about 5
seconds). Show:

- **MOCK** banner;
- cold cache executions (for example 120 → 87 with 3 repetitions) at **100% →
  100%** accuracy;
- warm cache executions;
- **No regressions**: every task the baseline got right, the optimized run
  also got right;
- requests and executions in separate columns; run time p50 and p95; tokens
  "not reported" (the mock agent has none), and no cost row (no pricing
  configured).

Quote the numbers on screen; they depend on the dataset's scripted
redundancy.

## 5. Learning a rule, safely (1.5 min)

1. **Learning** → **Mine proposals**. The table shows every candidate and why:
   - two proposals for `record_read`: keep for 10 min instead of 5, and share
     within a session instead of one run;
   - "not proposed" with reasons (e.g. price output changed every time);
   - "never learned" for the tools that change data.
2. Open the **Sharing scope** rule. Show the evidence: supporting repeats,
   100% consistency, and links to the exact training runs.
3. Click **Validate** (1 repetition). This runs held-out tasks with empty
   caches, with the rule off (in shadow) and on. All five checks pass, and
   executions drop (for example 19 → 18).
4. **Approve**: type a name and tick "I reviewed the evidence…". The button
   stays disabled until both are done. The rule becomes **Active**.
5. **Run a task** → "What is the current status of ticket REC-1003?" →
   **Optimized** → **Start run**: 1 execution. Click **Run again in this
   session**: 0 executions, marked "Reused because of an approved learned
   rule".
6. Back on the rule: **Disable** with a reason. It takes effect on the next
   request.

## 6. Close (30 s)

- **What is proven here:** with the mock agent, the gateway reuses only what
  is safe, keeps answers identical, and refuses unsafe reuse. Learned rules
  need evidence, held-out validation and a human.
- **What is not proven yet:** real Gemini savings. That needs a key and the
  same evaluation run with `llm=gemini` and several repetitions. Model output
  varies, so it must be measured, not assumed.

## If something goes wrong

- **"Backend unreachable"** in the sidebar: terminal 1 is not running, or not
  on port 8000.
- **Mine proposals is disabled:** no completed training evaluation of core v2
  yet. Click **Run a training evaluation** first.
- **An old rule is still active from a previous demo:** disable it on its rule
  page, or start with `rm -f backend/agentflow.db`.
