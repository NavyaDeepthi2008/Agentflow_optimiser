"""GeminiClient against the real google-genai SDK, using a fake HTTP server.

No network, no API key, no cost. The SDK builds the real REST request and parses
the real REST response format.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest
from pydantic import SecretStr

from app.config import load_settings
from app.gateway.factory import build_gateway
from app.llm.base import (
    AssistantMessage,
    LLMRequest,
    LLMToolCall,
    ToolResultItem,
    ToolResultsMessage,
    UserMessage,
)
from app.llm.errors import LLMConfigurationError, LLMProviderError, LLMResponseError
from app.llm.gemini import GeminiClient
from tests.fakes.gemini_server import (
    FAKE_KEY,
    FakeGeminiServer,
    _candidate,
    call_reply,
    error_body,
    text_reply,
)


def make_client(server: FakeGeminiServer, **kwargs) -> GeminiClient:
    params = {"api_key": FAKE_KEY, "model": "gemini-test-model"}
    params.update(kwargs)
    return GeminiClient(**params, http_client=server.http_client())


def simple_request(tools=None) -> LLMRequest:
    return LLMRequest(
        system_prompt="You are a test.",
        messages=[UserMessage(text="What is 2+2?")],
        tools=tools or [],
    )


# --------------------------------------------------------------------- keys


@pytest.mark.parametrize("key", [None, "", "   ", SecretStr(""), SecretStr("  ")])
def test_missing_key_is_a_configuration_error(key):
    with pytest.raises(LLMConfigurationError) as info:
        GeminiClient(api_key=key, model="m")
    assert "GEMINI_API_KEY" in info.value.message


def test_from_settings_without_key_fails_clearly():
    with pytest.raises(LLMConfigurationError):
        GeminiClient.from_settings(load_settings(environ={}, env_file=None))


def test_key_not_in_repr_and_sent_only_as_header():
    server = FakeGeminiServer([text_reply("4")])
    client = make_client(server)
    assert FAKE_KEY not in repr(client) and FAKE_KEY not in str(client)


async def test_key_goes_in_header_not_url():
    server = FakeGeminiServer([text_reply("4")])
    await make_client(server).generate(simple_request())
    assert FAKE_KEY not in server.urls[0]
    assert server.headers[0].get("x-goog-api-key") == FAKE_KEY
    assert FAKE_KEY not in json.dumps(server.bodies[0])


# --------------------------------------------------------------------- request


async def test_request_contains_system_prompt_and_function_declarations(world):
    declarations = build_gateway(world=world).tool_declarations()
    server = FakeGeminiServer([text_reply("ok")])
    await make_client(server).generate(simple_request(tools=declarations))
    body = server.bodies[0]
    assert body["systemInstruction"]["parts"][0]["text"] == "You are a test."
    sent = body["tools"][0]["functionDeclarations"]
    assert sorted(d["name"] for d in sent) == sorted(d["name"] for d in declarations)
    calc = next(d for d in sent if d["name"] == "calculator")
    schema = calc.get("parametersJsonSchema") or calc.get("parameters_json_schema")
    assert schema["required"] == ["expression"]
    assert "/models/gemini-test-model:generateContent" in server.urls[0]


async def test_no_tools_means_no_tools_field():
    server = FakeGeminiServer([text_reply("ok")])
    await make_client(server).generate(simple_request())
    assert "tools" not in server.bodies[0]


async def test_temperature_is_provider_default_unless_configured():
    server = FakeGeminiServer([text_reply("a"), text_reply("b")])
    await make_client(server).generate(simple_request())
    assert "temperature" not in server.bodies[0].get("generationConfig", {})
    await make_client(server, temperature=0.3, thinking_level="low").generate(simple_request())
    config = server.bodies[1]["generationConfig"]
    assert config["temperature"] == 0.3
    thinking = config["thinkingConfig"]
    # The SDK serializes this field in snake_case (as it does parameters_json_schema).
    assert (thinking.get("thinkingLevel") or thinking.get("thinking_level")) == "LOW"


async def test_history_round_trip_preserves_signature_and_call_ids():
    signature = b"opaque-thought-signature"
    server = FakeGeminiServer(
        [
            call_reply(("calculator", {"expression": "2+2"}, "fc-1"), signature=signature),
            text_reply("4"),
        ]
    )
    client = make_client(server)
    first = await client.generate(simple_request())
    call = first.tool_calls[0]
    assert call.provider_call_id == "fc-1"

    follow_up = LLMRequest(
        system_prompt="You are a test.",
        messages=[
            UserMessage(text="What is 2+2?"),
            AssistantMessage(tool_calls=first.tool_calls, provider_payload=first.provider_payload),
            ToolResultsMessage(
                results=[
                    ToolResultItem(
                        call_id=call.id,
                        name="calculator",
                        provider_call_id="fc-1",
                        response={"ok": True, "result": {"result": 4}},
                    )
                ]
            ),
        ],
    )
    await client.generate(follow_up)
    model_turn, tool_turn = server.bodies[1]["contents"][1:]
    assert model_turn["role"] == "model"
    assert base64.b64decode(model_turn["parts"][0]["thoughtSignature"]) == signature
    response_part = tool_turn["parts"][0]["functionResponse"]
    assert response_part["id"] == "fc-1"
    assert response_part["name"] == "calculator"
    assert response_part["response"] == {"ok": True, "result": {"result": 4}}


async def test_history_without_payload_is_rebuilt():
    server = FakeGeminiServer([text_reply("ok")])
    request = LLMRequest(
        system_prompt="s",
        messages=[
            UserMessage(text="q"),
            AssistantMessage(
                text="checking",
                tool_calls=[
                    LLMToolCall(
                        id="x", name="kb_search", args={"query": "a"}, provider_call_id="p1"
                    )
                ],
            ),
        ],
    )
    await make_client(server).generate(request)
    parts = server.bodies[0]["contents"][1]["parts"]
    assert parts[0]["text"] == "checking"
    assert parts[1]["functionCall"] == {"id": "p1", "name": "kb_search", "args": {"query": "a"}}


# --------------------------------------------------------------------- responses


async def test_text_response_with_usage():
    server = FakeGeminiServer([text_reply("Four.", thought="internal reasoning")])
    response = await make_client(server).generate(simple_request())
    assert response.text == "Four."  # thinking text is excluded
    assert response.tool_calls == []
    assert response.model_id == "gemini-test-model"
    assert response.model_version == "gemini-fake-001"
    assert response.finish_reason == "STOP"
    usage = response.usage
    assert (usage.input_tokens, usage.output_tokens, usage.thinking_tokens) == (100, 20, 5)
    assert usage.total_tokens == 125


async def test_missing_usage_is_none_not_zero():
    server = FakeGeminiServer([text_reply("x", usage=False)])
    assert (await make_client(server).generate(simple_request())).usage is None


async def test_parallel_function_calls_and_missing_args_or_ids():
    server = FakeGeminiServer(
        [
            call_reply(
                ("kb_search", {"query": "refund"}, "c1"),
                ("calculator", None, None),
                text="Let me look.",
            )
        ]
    )
    response = await make_client(server).generate(simple_request())
    assert [c.name for c in response.tool_calls] == ["kb_search", "calculator"]
    assert response.tool_calls[1].args == {}
    assert response.tool_calls[1].provider_call_id is None
    assert len({c.id for c in response.tool_calls}) == 2
    assert response.text == "Let me look."


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"candidates": [], "modelVersion": "x"}, "empty_response"),
        ({"promptFeedback": {"blockReason": "SAFETY"}}, "blocked_response"),
        (_candidate([], usage=False, finish="SAFETY"), "blocked_response"),
        (_candidate([], usage=False, finish="MALFORMED_FUNCTION_CALL"), "malformed_function_call"),
        (_candidate([{"functionCall": {"name": ""}}], usage=False), "malformed_function_call"),
        (_candidate([], usage=False, finish="MAX_TOKENS"), "empty_response"),
        (_candidate([{"text": "   "}], usage=False), "empty_response"),
        (_candidate([{"text": "only thinking", "thought": True}], usage=False), "empty_response"),
    ],
)
async def test_unusable_responses(body, code):
    server = FakeGeminiServer([body])
    with pytest.raises(LLMResponseError) as info:
        await make_client(server).generate(simple_request())
    assert info.value.code == code


# --------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (400, "provider_error", False),
        (401, "auth_error", False),
        (403, "auth_error", False),
        (404, "model_not_found", False),
        (429, "rate_limited", True),
        (500, "provider_unavailable", True),
        (503, "provider_unavailable", True),
    ],
)
async def test_http_errors_are_mapped_and_not_retried(status, code, retryable):
    server = FakeGeminiServer([(status, error_body(status, "problem"))])
    with pytest.raises(LLMProviderError) as info:
        await make_client(server).generate(simple_request())
    assert info.value.code == code
    assert info.value.retryable is retryable
    assert info.value.status_code == status
    assert server.calls == 1  # the SDK did not silently retry


async def test_error_message_never_contains_the_key():
    leaky = f"API key {FAKE_KEY} not valid. Please pass a valid API key."
    server = FakeGeminiServer([(400, error_body(400, leaky, "INVALID_ARGUMENT"))])
    with pytest.raises(LLMProviderError) as info:
        await make_client(server).generate(simple_request())
    assert FAKE_KEY not in info.value.message
    assert FAKE_KEY not in str(info.value)
    assert "[REDACTED]" in info.value.message


async def test_network_failure():
    server = FakeGeminiServer([httpx.ConnectError("connection refused")])
    with pytest.raises(LLMProviderError) as info:
        await make_client(server).generate(simple_request())
    assert info.value.code == "provider_unavailable"
    assert info.value.retryable


async def test_timeout():
    server = FakeGeminiServer([("sleep", 1.0)])
    with pytest.raises(LLMProviderError) as info:
        await make_client(server, timeout_seconds=0.05).generate(simple_request())
    assert info.value.code == "timeout"
