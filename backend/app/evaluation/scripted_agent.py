"""Turn a task's reference script into a ScriptedLLM.

* ``tool_calls`` steps become tool-call replies.
* ``advance_seconds`` moves the DATA clock forward before the step's reply,
  simulating time passing (prices change, cache entries age). Latency is still
  measured with real time.
* ``answer_template`` steps read values from the most recent result of a tool
  that the agent actually received in this run. If that result is missing,
  the placeholder becomes ``<missing>`` and the answer will usually be graded
  wrong - which is the point: the answer reflects what the tools returned.
"""

from __future__ import annotations

import re
from typing import Any

from app.evaluation.dataset import EvalTask, ScriptStep, ValueSource
from app.llm.base import LLMRequest, LLMResponse, ToolResultsMessage
from app.llm.scripted import ScriptedLLM
from app.telemetry.clock import OffsetClock

MISSING = "<missing>"


def _latest_response(request: LLMRequest, tool: str) -> dict[str, Any] | None:
    for message in reversed(request.messages):
        if isinstance(message, ToolResultsMessage):
            for item in reversed(message.results):
                if item.name == tool:
                    return item.response
    return None


def _follow(data: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(data, dict) and part in data:
            data = data[part]
        elif isinstance(data, list) and part.isdigit() and int(part) < len(data):
            data = data[int(part)]
        else:
            return None
    return data


def extract(request: LLMRequest, source: ValueSource) -> str:
    response = _latest_response(request, source.tool)
    value = _follow(response, source.path) if response is not None else None
    if value is None:
        return MISSING
    text = str(value)
    if source.regex:
        match = re.search(source.regex, text)
        return match.group(1) if match else MISSING
    return text


def _step(step: ScriptStep, data_clock: OffsetClock | None):
    def reply(request: LLMRequest) -> LLMResponse:
        if step.advance_seconds and data_clock is not None:
            data_clock.advance(step.advance_seconds)
        if step.tool_calls:
            return ScriptedLLM.tool_calls(
                *[(c["name"], c.get("args", {})) for c in step.tool_calls]
            )
        if step.answer is not None:
            return ScriptedLLM.final(step.answer)
        assert step.answer_template is not None
        values = {k: extract(request, v) for k, v in step.values.items()}
        return ScriptedLLM.final(step.answer_template.format(**values))

    return reply


def scripted_agent_for(task: EvalTask, data_clock: OffsetClock | None) -> ScriptedLLM:
    return ScriptedLLM(
        [_step(step, data_clock) for step in task.script], model_id="scripted-reference"
    )
