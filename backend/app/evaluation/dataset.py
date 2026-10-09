"""Evaluation dataset: tasks, hidden expectations and scripted reference behavior.

Each task has:
* ``task``: the ONLY text the agent ever sees.
* ``expected``: how the result is graded (answer checks, expected run status,
  and checks on the mock world's final state). Never sent to the agent.
* ``script``: a reference behavior used when evaluating with the scripted LLM.
  Its final answer is built from the tool results the agent actually received,
  so if the optimizer returned a wrong or stale result, the answer is wrong too.
* ``scripted_only``: the task depends on simulated time passing mid-run, which
  only the scripted agent can do; it is skipped for real-model evaluations.

The dataset file is hashed (SHA-256), and the hash is stored with every
evaluation, so a report always says exactly which data produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
Category = Literal["redundant", "freshness", "side_effect", "failure", "control"]


class ValueSource(BaseModel):
    """Where a scripted answer takes a value from: the latest result of ``tool``."""

    model_config = ConfigDict(extra="forbid")
    tool: str
    path: str = Field(description="dotted path in the tool's response, e.g. result.price")
    regex: str | None = None  # optional: first capture group of this regex


class ScriptStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    advance_seconds: float = Field(default=0, ge=0, le=86_400)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    answer: str | None = None
    answer_template: str | None = None
    values: dict[str, ValueSource] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_kind(self) -> ScriptStep:
        kinds = sum(bool(x) for x in (self.tool_calls, self.answer, self.answer_template))
        if kinds != 1:
            raise ValueError("a step needs exactly one of tool_calls, answer, answer_template")
        return self


class NumberCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: float
    tolerance: float = Field(default=0.0, ge=0)


class AnswerCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contains_all: list[str] = Field(default_factory=list)
    contains_none: list[str] = Field(default_factory=list)
    numbers: list[NumberCheck] = Field(default_factory=list)


class WorldCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outbox_count: int | None = None
    record: str | None = None
    field: str | None = None
    equals: str | None = None

    @model_validator(mode="after")
    def _shape(self) -> WorldCheck:
        is_outbox = self.outbox_count is not None
        is_record = self.record is not None and self.field is not None and self.equals is not None
        if is_outbox == is_record:
            raise ValueError("a world check is either outbox_count or record+field+equals")
        return self


class Expectation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str = "success"
    answer: AnswerCheck | None = None
    world: list[WorldCheck] = Field(default_factory=list)


class TaskSetup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inject_failures: dict[str, int] = Field(default_factory=dict)


class EvalTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]{3,64}$")
    category: Category
    split: Literal["train", "heldout"]
    task: str = Field(min_length=1, max_length=4000)
    tools: list[str] = Field(min_length=1)
    allow_side_effects: bool = False
    fresh_tools: list[str] = Field(default_factory=list)
    scripted_only: bool = False
    setup: TaskSetup = Field(default_factory=TaskSetup)
    script: list[ScriptStep] = Field(min_length=1)
    expected: Expectation

    @model_validator(mode="after")
    def _check(self) -> EvalTask:
        last = self.script[-1]
        if last.tool_calls:
            raise ValueError(f"{self.id}: the script must end with an answer")
        if any(step.advance_seconds for step in self.script) and not self.scripted_only:
            raise ValueError(f"{self.id}: tasks that advance time must be scripted_only")
        if self.expected.status == "success" and self.expected.answer is None:
            raise ValueError(f"{self.id}: successful tasks need an answer check")
        return self


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    version: str
    description: str = ""
    tasks: list[EvalTask]
    checksum: str = ""  # filled in by load_dataset

    @model_validator(mode="after")
    def _unique(self) -> Dataset:
        ids = [t.id for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task ids must be unique")
        return self

    def select(
        self,
        *,
        task_ids: list[str] | None = None,
        split: str | None = None,
        categories: list[str] | None = None,
    ) -> list[EvalTask]:
        tasks = self.tasks
        if task_ids is not None:
            unknown = set(task_ids) - {t.id for t in tasks}
            if unknown:
                raise ValueError(f"unknown task ids: {sorted(unknown)}")
            tasks = [t for t in tasks if t.id in set(task_ids)]
        if split is not None:
            tasks = [t for t in tasks if t.split == split]
        if categories is not None:
            tasks = [t for t in tasks if t.category in set(categories)]
        return tasks


def load_dataset(name: str = "core", version: str = "1") -> Dataset:
    if not name.isidentifier() or not version.isalnum():
        raise ValueError("invalid dataset name or version")
    path = DATASETS_DIR / f"{name}_v{version}.json"
    if not path.is_file():
        raise FileNotFoundError(f"dataset {name} v{version} not found")
    raw = path.read_bytes()
    dataset = Dataset.model_validate(json.loads(raw))
    return dataset.model_copy(update={"checksum": hashlib.sha256(raw).hexdigest()})
