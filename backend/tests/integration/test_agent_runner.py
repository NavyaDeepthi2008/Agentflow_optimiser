"""The agent loop with ScriptedLLM (and, at the end, the real SDK via a fake server)."""

from __future__ import annotations

import asyncio
import json

import pytest

from app.agent.models import AgentLimits, RunStatus
from app.agent.runner import AgentRunner
from app.gateway.models import Decision
from app.llm.base import LLMRequest, LLMResponse, TokenUsage, ToolResultsMessage
from app.llm.errors import LLMProviderError, LLMResponseError
from app.llm.gemini import GeminiClient
from app.llm.scripted import ScriptedLLM
from app.telemetry.clock import SystemClock
from tests.fakes.gemini_server import FAKE_KEY, FakeGeminiServer, call_reply, text_reply

S = ScriptedLLM


@pytest.fixture
def runner_for(gateway, recorder, clock):
    def _make(llm, **limits):
        return AgentRunner(
            llm, gateway, recorder=recorder, clock=clock, limits=AgentLimits(**limits)
        )

    return _make


def last_tool_results(request: LLMRequest) -> ToolResultsMessage:
    message = request.messages[-1]
    assert isinstance(message, ToolResultsMessage)
    return message


# ------------------------------------------------------------------ success paths


async def test_direct_answer_without_tools(runner_for, make_ctx):
    result = await runner_for(S([S.final("Hello!")])).run("Say hello", make_ctx())
    assert result.status is RunStatus.SUCCESS
    assert result.final_answer == "Hello!"
    assert (result.steps, result.llm_call_count, result.tool_request_count) == (1, 1, 0)


async def test_multi_step_run_records_everything(runner_for, make_ctx, recorder, world):
    llm = S(
        [
            S.tool_calls(("kb_search", {"query": "Orbit price"})),
            S.tool_calls(("doc_read", {"doc_id": "kb-002"})),
            S.final("Orbit costs 29 USD per user per month."),
        ]
    )
    ctx = make_ctx()
    result = await runner_for(llm).run("How much does Orbit cost per month?", ctx)

    assert result.status is RunStatus.SUCCESS
    assert result.final_answer == "Orbit costs 29 USD per user per month."
    assert result.llm_call_count == 3
    assert [r.tool_name for r in result.tool_results] == ["kb_search", "doc_read"]
    assert result.tool_execution_count == 2
    assert world.execution_counts == {"kb_search": 1, "doc_read": 1}
    assert all(c.model_id == "scripted-llm" for c in result.llm_calls)

    # The model saw the real tool output on the next turn.
    second_request = llm.requests[1]
    seen = last_tool_results(second_request).results[0]
    assert seen.name == "kb_search"
    assert seen.response["ok"] is True
    assert seen.response["result"]["results"][0]["doc_id"] == "kb-002"

    events = recorder.events_for_run(ctx.run_id)
    types_ = [e.type for e in events]
    assert types_[0] == "run.started" and types_[-1] == "run.completed"
    assert types_.count("llm_call.completed") == 3
    assert types_.count("tool_call.completed") == 2
    done = events[-1].data
    assert done["final_answer"] == result.final_answer
    assert done["status"] == "success"
    assert done["tool_decision_sequence"] == ["executed_live", "executed_live"]
    started = events[0].data
    assert started["task"] == "How much does Orbit cost per month?"
    assert started["mode"] == "passthrough"
    assert "calculator" in started["tools_offered"]


async def test_parallel_tool_calls_in_one_turn(runner_for, make_ctx):
    a = S.call("calculator", {"expression": "29*12"})
    b = S.call("price_lookup", {"symbol": "NMB-X"})
    llm = S([S.tool_calls(a, b), S.final("done")])
    result = await runner_for(llm).run("Two things", make_ctx())
    assert result.status is RunStatus.SUCCESS
    assert [r.call_id for r in result.tool_results] == [a.id, b.id]
    items = last_tool_results(llm.requests[1]).results
    assert [(i.call_id, i.provider_call_id) for i in items] == [
        (a.id, a.provider_call_id),
        (b.id, b.provider_call_id),
    ]
    assert items[0].response["result"]["result"] == 348


async def test_tool_subset_is_offered(runner_for, make_ctx):
    llm = S([S.final("ok")])
    await runner_for(llm).run("t", make_ctx(), tools=["calculator", "kb_search"])
    assert sorted(t["name"] for t in llm.requests[0].tools) == ["calculator", "kb_search"]


async def test_test_only_tools_are_never_offered(runner_for, make_ctx):
    llm = S([S.final("ok")])
    await runner_for(llm).run("t", make_ctx())
    offered = {t["name"] for t in llm.requests[0].tools}
    assert "crashing_tool" not in offered and "flaky_tool" not in offered


# ------------------------------------------------------------------ gateway only


async def test_every_tool_execution_went_through_the_gateway(runner_for, make_ctx, gateway, world):
    llm = S(
        [
            S.tool_calls(("calculator", {"expression": "1+1"}), ("kb_search", {"query": "x"})),
            S.tool_calls(("send_notification", {"recipient": "team:a", "message": "m"})),
            S.tool_calls(("record_read", {"record_id": "REC-1001"})),
            S.final("done"),
        ]
    )
    result = await runner_for(llm).run("t", make_ctx())  # no side-effect permission
    assert result.status is RunStatus.SUCCESS
    assert gateway.stats["live_executions"] == dict(world.execution_counts)
    assert world.outbox == []  # the model asked, the gateway refused
    blocked = result.tool_results[2]
    assert blocked.decision is Decision.BLOCKED_PERMISSION and not blocked.executed
    seen = last_tool_results(llm.requests[2]).results[0].response
    assert seen["ok"] is False and seen["error"]["code"] == "permission_denied"


async def test_side_effects_run_when_permitted(runner_for, make_ctx, world):
    llm = S(
        [
            S.tool_calls(
                ("record_update", {"record_id": "REC-1001", "field": "status", "value": "resolved"})
            ),
            S.tool_calls(
                ("send_notification", {"recipient": "team:orbit", "message": "REC-1001 resolved"})
            ),
            S.final("Done."),
        ]
    )
    result = await runner_for(llm).run("Resolve and notify", make_ctx(allow_side_effects=True))
    assert result.status is RunStatus.SUCCESS
    assert world.records["REC-1001"]["status"] == "resolved"
    assert len(world.outbox) == 1


# ------------------------------------------------------------------ malformed model output


async def test_malformed_tool_requests_are_handled_by_the_gateway(runner_for, make_ctx, world):
    llm = S(
        [
            S.tool_calls(("delete_database", {})),  # unknown tool
            S.tool_calls(("calculator", ["not", "an", "object"])),  # args not an object
            S.tool_calls(("calculator", {"expr": "1+1"})),  # wrong argument name
            S.tool_calls(("doc_read", {"doc_id": "kb-999"})),  # valid but not found
            S.final("I could not complete that."),
        ]
    )
    result = await runner_for(llm).run("t", make_ctx())
    assert result.status is RunStatus.SUCCESS
    assert [r.decision for r in result.tool_results] == [
        Decision.BLOCKED_UNKNOWN_TOOL,
        Decision.BLOCKED_INVALID_ARGS,
        Decision.BLOCKED_INVALID_ARGS,
        Decision.FAILED,
    ]
    assert sum(world.execution_counts.values()) == 1  # only doc_read actually ran
    for request, code in zip(
        llm.requests[1:5],
        ["unknown_tool", "invalid_arguments", "invalid_arguments", "not_found"],
        strict=True,
    ):
        assert last_tool_results(request).results[0].response["error"]["code"] == code


async def test_empty_model_response_fails_the_run(runner_for, make_ctx):
    empty = LLMResponse(model_id="scripted-llm", text=None)
    result = await runner_for(S([empty])).run("t", make_ctx())
    assert result.status is RunStatus.FAILED
    assert result.error.code == "empty_response"


@pytest.mark.parametrize(
    "exc",
    [
        LLMResponseError("bad call", code="malformed_function_call"),
        LLMResponseError("blocked", code="blocked_response"),
    ],
)
async def test_unusable_llm_response_fails_the_run(runner_for, make_ctx, recorder, exc):
    ctx = make_ctx()
    result = await runner_for(S([exc])).run("t", ctx)
    assert result.status is RunStatus.FAILED
    assert result.error.code == exc.code
    assert result.llm_calls[0].ok is False and result.llm_calls[0].error_code == exc.code
    assert "llm_call.failed" in [e.type for e in recorder.events_for_run(ctx.run_id)]


# ------------------------------------------------------------------ provider errors


@pytest.mark.parametrize("code", ["rate_limited", "auth_error", "provider_unavailable", "timeout"])
async def test_provider_errors_fail_the_run_cleanly(runner_for, make_ctx, code):
    llm = S(
        [
            S.tool_calls(("calculator", {"expression": "1+1"})),
            LLMProviderError(f"simulated {code}", code=code),
        ]
    )
    result = await runner_for(llm).run("t", make_ctx())
    assert result.status is RunStatus.FAILED
    assert result.error.code == code
    assert result.llm_call_count == 2
    assert [c.ok for c in result.llm_calls] == [True, False]
    assert result.tool_execution_count == 1  # work done before the failure is recorded


async def test_script_exhausted_is_a_failure(runner_for, make_ctx):
    llm = S([S.tool_calls(("calculator", {"expression": "1"}))])
    result = await runner_for(llm).run("t", make_ctx())
    assert result.status is RunStatus.FAILED
    assert result.error.code == "script_exhausted"


async def test_unexpected_bug_becomes_error_status(runner_for, make_ctx, recorder):
    ctx = make_ctx()
    result = await runner_for(S([RuntimeError("secret internal detail")])).run("t", ctx)
    assert result.status is RunStatus.ERROR
    assert result.error.code == "internal_error"
    assert "secret internal detail" not in result.error.message
    assert recorder.events_for_run(ctx.run_id)[-1].type == "run.completed"


# ------------------------------------------------------------------ limits


async def test_step_limit(runner_for, make_ctx):
    llm = S([S.tool_calls(("calculator", {"expression": f"{i}+1"})) for i in range(10)])
    result = await runner_for(llm, max_steps=3).run("loop forever", make_ctx())
    assert result.status is RunStatus.LIMIT_STEPS
    assert result.error.code == "max_steps"
    assert result.llm_call_count == 3
    assert result.tool_request_count == 3


async def test_tool_request_limit(runner_for, make_ctx, world):
    llm = S(
        [
            S.tool_calls(*[("calculator", {"expression": f"{t}*10+{i}"}) for i in range(3)])
            for t in range(5)
        ]
    )
    result = await runner_for(llm).run("t", make_ctx(max_tool_requests=4))
    assert result.status is RunStatus.LIMIT_TOOL_CALLS
    assert world.execution_counts["calculator"] == 4
    assert result.tool_request_count == 6  # 2nd turn: 1 ok, 1 over the limit, 1 run-stopped
    assert result.tool_results[4].decision is Decision.BLOCKED_LIMIT
    assert result.tool_results[5].decision is Decision.BLOCKED_RUN_STOPPED
    assert result.llm_call_count == 2


class SlowLLM:
    provider = "slow"
    model_id = "slow-model"

    async def generate(self, request):
        await asyncio.sleep(5)


async def test_run_timeout(gateway, recorder, make_ctx):
    runner = AgentRunner(
        SlowLLM(),
        gateway,
        recorder=recorder,
        clock=SystemClock(),
        limits=AgentLimits(run_timeout_seconds=0.1),
    )
    result = await runner.run("t", make_ctx())
    assert result.status is RunStatus.LIMIT_TIME
    assert result.error.code == "run_timeout"
    assert result.llm_calls[0].error_code == "run_timeout"
    assert result.duration_ms < 2000


async def test_runs_need_fresh_context_and_valid_task(runner_for, make_ctx):
    ctx = make_ctx()
    await runner_for(S([S.tool_calls(("calculator", {"expression": "1"})), S.final("x")])).run(
        "t", ctx
    )
    with pytest.raises(ValueError):
        await runner_for(S([S.final("x")])).run("again", ctx)
    for bad in ["", "   ", "x" * 5000]:
        with pytest.raises(ValueError):
            await runner_for(S([S.final("x")])).run(bad, make_ctx())


# ------------------------------------------------------------------ usage accounting


async def test_usage_totals(runner_for, make_ctx):
    u = TokenUsage(input_tokens=100, output_tokens=10, thinking_tokens=4, total_tokens=114)
    llm = S([S.tool_calls(("calculator", {"expression": "1"}), usage=u), S.final("x", usage=u)])
    result = await runner_for(llm).run("t", make_ctx())
    assert result.usage.input_tokens == 200 and result.usage.output_tokens == 20
    assert result.usage.thinking_tokens == 8 and result.usage.total_tokens == 228
    assert result.usage.complete and result.usage.calls_with_usage == 2


async def test_partial_usage_is_flagged(runner_for, make_ctx):
    u = TokenUsage(input_tokens=100, output_tokens=10)
    llm = S([S.tool_calls(("calculator", {"expression": "1"}), usage=u), S.final("x")])
    result = await runner_for(llm).run("t", make_ctx())
    assert result.usage.input_tokens == 100
    assert result.usage.complete is False


async def test_no_usage_is_unavailable_not_zero(runner_for, make_ctx):
    result = await runner_for(S([S.final("x")])).run("t", make_ctx())
    assert result.usage.input_tokens is None and result.usage.total_tokens is None
    assert result.usage.complete is False
    assert result.summary()["tokens"]["input_tokens"] is None


# ------------------------------------------------------------------ real SDK end to end


async def test_full_run_with_gemini_sdk_against_fake_server(gateway, recorder, make_ctx, world):
    server = FakeGeminiServer(
        [
            call_reply(("kb_search", {"query": "refund policy"}, "fc-1"), signature=b"sig-1"),
            call_reply(("doc_read", {"doc_id": "kb-004"}, "fc-2"), signature=b"sig-2"),
            text_reply("Refunds are available within 30 days of purchase."),
        ]
    )
    llm = GeminiClient(
        api_key=FAKE_KEY, model="gemini-test-model", http_client=server.http_client()
    )
    runner = AgentRunner(llm, gateway, recorder=recorder)
    ctx = make_ctx()
    result = await runner.run("What is the refund window?", ctx)

    assert result.status is RunStatus.SUCCESS
    assert result.final_answer == "Refunds are available within 30 days of purchase."
    assert world.execution_counts == {"kb_search": 1, "doc_read": 1}
    assert result.usage.input_tokens == 300 and result.usage.complete
    assert {c.model_version for c in result.llm_calls} == {"gemini-fake-001"}
    # Third request carries both earlier model turns (with signatures) and both results.
    contents = server.bodies[2]["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user", "model", "user"]
    assert contents[2]["parts"][0]["functionResponse"]["id"] == "fc-1"
    assert "thoughtSignature" in contents[3]["parts"][0]
    # The key appears nowhere in what we recorded.
    everything = json.dumps([e.model_dump() for e in recorder.events]) + result.model_dump_json()
    assert FAKE_KEY not in everything
