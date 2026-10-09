"""ScriptedLLM: a fake LLM that replies from a fixed script.

Used for deterministic, offline, free tests and demos. Each call to
``generate`` consumes the next step of the script:

* an ``LLMResponse`` is returned as-is (with ``model_id`` filled in),
* an ``Exception`` is raised (to simulate provider errors),
* a callable receives the request and returns a response (to react to tool
  results, e.g. "answer with whatever the calculator returned").

It records every request it receives so tests can check what the agent sent.
It never reports token usage unless the script says so, so fake numbers can
never be mistaken for real ones.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any

from app.llm.base import LLMRequest, LLMResponse, LLMToolCall, TokenUsage
from app.llm.errors import LLMResponseError

ScriptStep = LLMResponse | Exception | Callable[[LLMRequest], LLMResponse]

SCRIPTED_MODEL_ID = "scripted-llm"


class ScriptedLLM:
    provider = "scripted"

    def __init__(self, steps: Sequence[ScriptStep], model_id: str = SCRIPTED_MODEL_ID) -> None:
        self._steps = list(steps)
        self._model_id = model_id
        self.requests: list[LLMRequest] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def remaining_steps(self) -> int:
        return len(self._steps)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self._steps:
            raise LLMResponseError("the script has no more replies", code="script_exhausted")
        step = self._steps.pop(0)
        if isinstance(step, Exception):
            raise step
        response = step(request) if callable(step) else step
        return response.model_copy(update={"model_id": self._model_id})

    # ----------------------------------------------------------- helpers
    _ids = itertools.count(1)

    @classmethod
    def call(cls, name: str, args: Any = None) -> LLMToolCall:
        n = next(cls._ids)
        return LLMToolCall(
            id=f"scripted-call-{n}",
            name=name,
            args={} if args is None else args,
            provider_call_id=f"scripted-fc-{n}",
        )

    @classmethod
    def tool_calls(
        cls,
        *calls: tuple[str, Any] | LLMToolCall,
        text: str | None = None,
        usage: TokenUsage | None = None,
    ) -> LLMResponse:
        built = [c if isinstance(c, LLMToolCall) else cls.call(c[0], c[1]) for c in calls]
        return LLMResponse(
            text=text,
            tool_calls=built,
            usage=usage,
            model_id=SCRIPTED_MODEL_ID,
            finish_reason="STOP",
        )

    @staticmethod
    def final(text: str, usage: TokenUsage | None = None) -> LLMResponse:
        return LLMResponse(text=text, usage=usage, model_id=SCRIPTED_MODEL_ID, finish_reason="STOP")
