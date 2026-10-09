from __future__ import annotations

import pytest

from app.llm.base import LLMRequest, UserMessage
from app.llm.errors import LLMProviderError, LLMResponseError
from app.llm.scripted import ScriptedLLM


def _request(text: str = "hi") -> LLMRequest:
    return LLMRequest(system_prompt="sys", messages=[UserMessage(text=text)])


async def test_replies_in_order_and_records_requests():
    llm = ScriptedLLM(
        [ScriptedLLM.tool_calls(("calculator", {"expression": "1+1"})), ScriptedLLM.final("done")]
    )
    first = await llm.generate(_request("a"))
    second = await llm.generate(_request("b"))
    assert first.tool_calls[0].name == "calculator"
    assert first.tool_calls[0].provider_call_id is not None
    assert second.text == "done"
    assert [r.messages[0].text for r in llm.requests] == ["a", "b"]
    assert llm.remaining_steps == 0


async def test_exceptions_are_raised():
    llm = ScriptedLLM([LLMProviderError("slow down", code="rate_limited", retryable=True)])
    with pytest.raises(LLMProviderError) as info:
        await llm.generate(_request())
    assert info.value.code == "rate_limited"


async def test_callable_step_sees_the_request():
    llm = ScriptedLLM([lambda req: ScriptedLLM.final(f"echo:{req.messages[0].text}")])
    assert (await llm.generate(_request("ping"))).text == "echo:ping"


async def test_exhausted_script_raises():
    llm = ScriptedLLM([])
    with pytest.raises(LLMResponseError) as info:
        await llm.generate(_request())
    assert info.value.code == "script_exhausted"


async def test_no_usage_unless_scripted():
    assert (await ScriptedLLM([ScriptedLLM.final("x")]).generate(_request())).usage is None


async def test_model_id_is_reported():
    llm = ScriptedLLM([ScriptedLLM.final("x")], model_id="scripted-test")
    assert (await llm.generate(_request())).model_id == "scripted-test"
