"""Input shapes for starting a run (used by the API and the run service)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.gateway.models import GatewayMode
from app.learning.models import Thresholds

SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class ScriptedToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    args: dict[str, Any] = Field(default_factory=dict)


class ScriptedReply(BaseModel):
    """One canned LLM reply: either a final text, or one or more tool calls."""

    model_config = ConfigDict(extra="forbid")
    text: str | None = Field(default=None, max_length=4000)
    tool_calls: list[ScriptedToolCall] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def _one_kind(self) -> ScriptedReply:
        if bool(self.text) == bool(self.tool_calls):
            raise ValueError("a scripted reply needs either text or tool_calls (not both)")
        return self


class ScenarioRef(BaseModel):
    """Run a task from an evaluation dataset. The task text, tools and permissions
    come from the dataset; with llm='scripted' its server-side reference script is
    used. Expected answers are never sent to the agent or the client."""

    model_config = ConfigDict(extra="forbid")
    dataset: str = Field(default="core", pattern=r"^[a-z][a-z0-9_]{0,40}$")
    version: str = Field(default="1", pattern=r"^[0-9a-z]{1,10}$")
    task_id: str = Field(pattern=r"^[a-z0-9_]{3,64}$")


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str | None = Field(default=None, min_length=1, max_length=4000)
    scenario: ScenarioRef | None = None
    mode: GatewayMode = GatewayMode.PASSTHROUGH
    session_id: str = Field(default="default", pattern=SESSION_ID_PATTERN)
    llm: Literal["gemini", "scripted"] = "gemini"
    scripted_replies: list[ScriptedReply] | None = Field(default=None, max_length=30)
    # If set, only these tools are offered to the model AND allowed by the gateway.
    tools: list[str] | None = Field(default=None, max_length=50)
    allow_side_effects: bool = False
    fresh_tools: list[str] = Field(default_factory=list, max_length=50)
    max_steps: int | None = Field(default=None, ge=1, le=50)
    max_tool_requests: int | None = Field(default=None, ge=1, le=200)

    @model_validator(mode="after")
    def _check(self) -> RunRequest:
        if self.scenario is not None:
            if self.task is not None or self.scripted_replies:
                raise ValueError("a scenario run takes its task and script from the dataset")
            defined_by_scenario = {"tools", "allow_side_effects", "fresh_tools"}
            if defined_by_scenario & self.model_fields_set:
                raise ValueError("tools and permissions come from the scenario")
            return self
        if self.task is None or not self.task.strip():
            raise ValueError("task must not be blank")
        if self.llm == "scripted" and not self.scripted_replies:
            raise ValueError("llm='scripted' requires scripted_replies (or use a scenario)")
        if self.llm == "gemini" and self.scripted_replies:
            raise ValueError("scripted_replies are only allowed with llm='scripted'")
        return self


# ---------------------------------------------------------------- learning
class MineRequest(BaseModel):
    """Mine proposals from completed evaluations of one dataset version."""

    model_config = ConfigDict(extra="forbid")
    dataset: str = Field(default="core", pattern=r"^[a-z][a-z0-9_]{0,40}$")
    dataset_version: str = Field(default="2", pattern=r"^[0-9a-z]{1,10}$")
    # None = every completed evaluation of that dataset version.
    evaluation_ids: list[Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]] | None = Field(
        default=None, min_length=1, max_length=50
    )
    thresholds: Thresholds = Field(default_factory=Thresholds)


class ShadowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool


class ApproveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approved_by: str = Field(min_length=1, max_length=100)
    note: str | None = Field(default=None, max_length=500)
    confirm: bool = Field(
        description="Must be true: the approver has reviewed the evidence and validation."
    )

    @model_validator(mode="after")
    def _not_blank(self) -> ApproveRequest:
        if not self.approved_by.strip():
            raise ValueError("approved_by must not be blank")
        return self


class DisableRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(min_length=1, max_length=500)
    disabled_by: str = Field(default="dashboard user", min_length=1, max_length=100)

    @model_validator(mode="after")
    def _not_blank(self) -> DisableRequest:
        if not self.reason.strip() or not self.disabled_by.strip():
            raise ValueError("reason and disabled_by must not be blank")
        return self
