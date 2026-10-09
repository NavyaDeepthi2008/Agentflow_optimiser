"""Provider-neutral LLM types.

The agent talks to any LLM through :class:`LLMClient` using these shapes, so the
agent does not depend on Gemini. Each provider client converts to and from its
own SDK format.

``provider_payload`` lets a client keep the provider's original message object
(for Gemini: the exact model ``Content``, including *thought signatures*) so it
can be sent back unchanged on the next turn, as Google requires for multi-turn
function calling.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class LLMToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str  # our id, used as the gateway call_id
    name: str
    args: Any = Field(default_factory=dict)  # unvalidated; the gateway validates
    provider_call_id: str | None = None  # the provider's own id, echoed back


class TokenUsage(BaseModel):
    """Token counts as reported by the provider. ``None`` means "not reported"."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    cached_input_tokens: int | None = None
    total_tokens: int | None = None

    @property
    def is_complete(self) -> bool:
        return self.input_tokens is not None and self.output_tokens is not None


class UserMessage(BaseModel):
    model_config = ConfigDict(frozen=True)
    role: Literal["user"] = "user"
    text: str


class AssistantMessage(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    role: Literal["assistant"] = "assistant"
    text: str | None = None
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    provider_payload: Any = Field(default=None, exclude=True, repr=False)


class ToolResultItem(BaseModel):
    model_config = ConfigDict(frozen=True)
    call_id: str
    name: str
    response: dict[str, Any]
    provider_call_id: str | None = None


class ToolResultsMessage(BaseModel):
    model_config = ConfigDict(frozen=True)
    role: Literal["tool"] = "tool"
    results: list[ToolResultItem]


Message = UserMessage | AssistantMessage | ToolResultsMessage


class LLMRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    system_prompt: str
    messages: list[Message]
    tools: list[dict[str, Any]] = Field(default_factory=list)  # gateway declarations


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    text: str | None = None
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    usage: TokenUsage | None = None
    model_id: str  # the model we asked for
    model_version: str | None = None  # the exact version the provider reports
    finish_reason: str | None = None
    provider_payload: Any = Field(default=None, exclude=True, repr=False)


class LLMClient(Protocol):
    provider: str

    @property
    def model_id(self) -> str: ...

    async def generate(self, request: LLMRequest) -> LLMResponse: ...
