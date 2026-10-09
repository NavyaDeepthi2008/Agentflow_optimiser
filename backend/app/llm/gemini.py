"""Google Gemini client, built on the official Google Gen AI SDK (``google-genai``).

What this module does:
* Converts gateway tool declarations into Gemini ``FunctionDeclaration``s
  (plain JSON Schema via ``parameters_json_schema``).
* Turns *off* the SDK's automatic function calling, so the SDK never runs a
  Python function itself. Function calls come back to our agent, which sends
  them through the ToolGateway.
* Turns *off* the SDK's automatic HTTP retries (``attempts=1``), so every API
  call is visible and counted. Retry policy is a later, explicit decision.
* Sends the model's previous turn back unchanged (``provider_payload``), which
  preserves Gemini's thought signatures in multi-turn function calling.
* Maps SDK and network failures to our :mod:`app.llm.errors` classes, with any
  occurrence of the API key removed from messages.

The API key is passed in by the caller (from settings). It is never logged,
printed, or included in errors, and ``repr(client)`` does not show it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import uuid4

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import SecretStr

from app.config import Settings
from app.llm.base import (
    AssistantMessage,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
    TokenUsage,
    ToolResultsMessage,
    UserMessage,
)
from app.llm.errors import (
    LLMConfigurationError,
    LLMError,
    LLMProviderError,
    LLMResponseError,
    redact,
)

logger = logging.getLogger(__name__)

_BLOCKED_FINISH_REASONS = {
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "LANGUAGE",
}
_MALFORMED_FINISH_REASONS = {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"}

MISSING_KEY_MESSAGE = (
    "GEMINI_API_KEY is not set. Add it to backend/.env (see .env.example) or export it "
    "in your shell. ScriptedLLM can be used without a key."
)


class GeminiClient:
    provider = "gemini"

    def __init__(
        self,
        *,
        api_key: SecretStr | str | None,
        model: str,
        temperature: float | None = None,
        thinking_level: str | None = None,
        max_output_tokens: int | None = 4096,
        timeout_seconds: float = 45.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        if not key or not key.strip():
            raise LLMConfigurationError(MISSING_KEY_MESSAGE)
        if not model.strip():
            raise LLMConfigurationError("a Gemini model id is required (GEMINI_MODEL)")
        self.__key = key.strip()  # name-mangled; used only for redaction
        self._model = model
        self._temperature = temperature
        self._thinking_level = thinking_level
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout_seconds
        http_options = types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
            httpx_async_client=http_client,  # tests inject a fake transport here
        )
        self._sdk = genai.Client(api_key=self.__key, http_options=http_options)

    @classmethod
    def from_settings(
        cls, settings: Settings, *, http_client: httpx.AsyncClient | None = None
    ) -> GeminiClient:
        return cls(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            temperature=settings.gemini_temperature,
            thinking_level=settings.gemini_thinking_level,
            max_output_tokens=settings.gemini_max_output_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
            http_client=http_client,
        )

    def __repr__(self) -> str:
        return f"GeminiClient(model={self._model!r})"

    __str__ = __repr__

    @property
    def model_id(self) -> str:
        return self._model

    # --------------------------------------------------------------- request
    def build_contents(self, request: LLMRequest) -> list[types.Content]:
        contents: list[types.Content] = []
        for message in request.messages:
            if isinstance(message, UserMessage):
                contents.append(types.Content(role="user", parts=[types.Part(text=message.text)]))
            elif isinstance(message, AssistantMessage):
                if isinstance(message.provider_payload, types.Content):
                    # Send Gemini's own turn back unchanged (keeps thought signatures).
                    contents.append(message.provider_payload)
                    continue
                parts: list[types.Part] = []
                if message.text:
                    parts.append(types.Part(text=message.text))
                for call in message.tool_calls:
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                id=call.provider_call_id,
                                name=call.name,
                                args=call.args if isinstance(call.args, dict) else {},
                            )
                        )
                    )
                contents.append(types.Content(role="model", parts=parts))
            elif isinstance(message, ToolResultsMessage):
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    id=item.provider_call_id,
                                    name=item.name,
                                    response=item.response,
                                )
                            )
                            for item in message.results
                        ],
                    )
                )
            else:  # pragma: no cover - the Message union is closed
                raise TypeError(f"unsupported message type {type(message).__name__}")
        return contents

    @staticmethod
    def build_function_declarations(
        declarations: list[dict[str, Any]],
    ) -> list[types.FunctionDeclaration]:
        return [
            types.FunctionDeclaration(
                name=d["name"],
                description=d["description"],
                parameters_json_schema=d["parameters"],
            )
            for d in declarations
        ]

    def build_config(self, request: LLMRequest) -> types.GenerateContentConfig:
        config: dict[str, Any] = {
            "system_instruction": request.system_prompt,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
            "max_output_tokens": self._max_output_tokens,
        }
        if request.tools:
            config["tools"] = [
                types.Tool(function_declarations=self.build_function_declarations(request.tools))
            ]
        if self._temperature is not None:
            config["temperature"] = self._temperature
        if self._thinking_level is not None:
            config["thinking_config"] = types.ThinkingConfig(
                thinking_level=self._thinking_level.upper()
            )
        return types.GenerateContentConfig(**config)

    # --------------------------------------------------------------- call
    async def generate(self, request: LLMRequest) -> LLMResponse:
        contents = self.build_contents(request)
        config = self.build_config(request)
        try:
            response = await asyncio.wait_for(
                self._sdk.aio.models.generate_content(
                    model=self._model, contents=contents, config=config
                ),
                timeout=self._timeout,
            )
        except TimeoutError:
            raise LLMProviderError(
                f"Gemini did not respond within {self._timeout:g} seconds",
                code="timeout",
                retryable=True,
            ) from None
        except genai_errors.APIError as exc:
            raise self._map_api_error(exc) from None
        except LLMError:
            raise
        except (httpx.HTTPError, OSError) as exc:
            raise LLMProviderError(
                f"could not reach Gemini ({type(exc).__name__})",
                code="provider_unavailable",
                retryable=True,
            ) from None
        return self.parse_response(response)

    def _map_api_error(self, exc: genai_errors.APIError) -> LLMProviderError:
        status = exc.code if isinstance(exc.code, int) else None
        detail = redact(f"{exc.status or ''} {exc.message or ''}".strip(), self.__key)
        if status in (401, 403):
            code, retryable = "auth_error", False
        elif status == 429:
            code, retryable = "rate_limited", True
        elif status is not None and status >= 500:
            code, retryable = "provider_unavailable", True
        elif status == 404:
            code, retryable = "model_not_found", False
        else:
            code, retryable = "provider_error", False
        return LLMProviderError(
            f"Gemini API error {status}: {detail}" if detail else f"Gemini API error {status}",
            code=code,
            retryable=retryable,
            status_code=status,
        )

    # --------------------------------------------------------------- response
    def parse_response(self, response: types.GenerateContentResponse) -> LLMResponse:
        usage = self._parse_usage(response.usage_metadata)
        candidates = response.candidates or []
        if not candidates:
            feedback = response.prompt_feedback
            reason = getattr(feedback, "block_reason", None) if feedback else None
            if reason:
                raise LLMResponseError(
                    f"Gemini blocked the prompt ({_enum_name(reason)})", code="blocked_response"
                )
            raise LLMResponseError("Gemini returned no candidates", code="empty_response")

        candidate = candidates[0]
        finish = _enum_name(candidate.finish_reason)
        if finish in _MALFORMED_FINISH_REASONS:
            raise LLMResponseError(
                f"Gemini produced an unusable function call ({finish})",
                code="malformed_function_call",
            )
        parts = (candidate.content.parts if candidate.content else None) or []
        if not parts and finish in _BLOCKED_FINISH_REASONS:
            raise LLMResponseError(
                f"Gemini stopped the response ({finish})", code="blocked_response"
            )

        texts: list[str] = []
        calls: list[LLMToolCall] = []
        for part in parts:
            if part.function_call is not None:
                name = part.function_call.name
                if not isinstance(name, str) or not name.strip():
                    raise LLMResponseError(
                        "Gemini returned a function call without a name",
                        code="malformed_function_call",
                    )
                args = part.function_call.args
                calls.append(
                    LLMToolCall(
                        id=uuid4().hex,
                        name=name,
                        args={} if args is None else args,
                        provider_call_id=part.function_call.id,
                    )
                )
            elif part.text and not part.thought:  # thinking text is not the answer
                texts.append(part.text)

        text = "".join(texts).strip() or None
        if text is None and not calls:
            raise LLMResponseError(
                f"Gemini returned neither text nor a function call (finish reason: {finish})",
                code="empty_response",
            )
        return LLMResponse(
            text=text,
            tool_calls=calls,
            usage=usage,
            model_id=self._model,
            model_version=response.model_version,
            finish_reason=finish,
            provider_payload=candidate.content,
        )

    @staticmethod
    def _parse_usage(meta: types.GenerateContentResponseUsageMetadata | None) -> TokenUsage | None:
        if meta is None:
            return None
        return TokenUsage(
            input_tokens=meta.prompt_token_count,
            output_tokens=meta.candidates_token_count,
            thinking_tokens=meta.thoughts_token_count,
            cached_input_tokens=meta.cached_content_token_count,
            total_tokens=meta.total_token_count,
        )


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "name", None) or str(value)
