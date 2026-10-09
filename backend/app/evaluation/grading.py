"""Grade one run against a task's hidden expectations.

A run is correct only if EVERY check passes:
1. the run status equals the expected status (usually "success");
2. answer checks (case-insensitive):
   * ``contains_all``: each phrase appears in the final answer;
   * ``contains_none``: none of these phrases appears (e.g. a stale value);
   * ``numbers``: each expected number appears, within its tolerance;
3. world checks: the mock world's final state (outbox size, record fields).
   This catches side effects that were skipped or duplicated even when the
   answer *sounds* right.

Grading is deterministic and never uses an LLM.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from app.evaluation.dataset import EvalTask

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


class Grade(BaseModel):
    correct: bool
    checks: list[dict[str, Any]] = Field(default_factory=list)  # each: name, passed, detail

    @property
    def failed_checks(self) -> list[str]:
        return [c["name"] for c in self.checks if not c["passed"]]


def normalize(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 1,200 -> 1200
    return " ".join(text.split())


def numbers_in(text: str) -> list[float]:
    return [float(n) for n in _NUMBER.findall(normalize(text))]


def grade(
    task: EvalTask,
    *,
    status: str,
    final_answer: str | None,
    outbox_count: int,
    records: dict[str, dict[str, Any]],
) -> Grade:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    expected = task.expected
    check("status", status == expected.status, f"expected {expected.status}, got {status}")

    if expected.answer is not None:
        answer = normalize(final_answer or "")
        for phrase in expected.answer.contains_all:
            check(f"contains:{phrase}", normalize(phrase) in answer, "required phrase")
        for phrase in expected.answer.contains_none:
            check(f"excludes:{phrase}", normalize(phrase) not in answer, "forbidden phrase")
        found = numbers_in(final_answer or "")
        for number in expected.answer.numbers:
            ok = any(abs(n - number.value) <= number.tolerance for n in found)
            check(f"number:{number.value:g}", ok, f"numbers in answer: {found[:10]}")

    for world in expected.world:
        if world.outbox_count is not None:
            check(
                "world:outbox_count",
                outbox_count == world.outbox_count,
                f"expected {world.outbox_count}, got {outbox_count}",
            )
        else:
            actual = (records.get(world.record or "") or {}).get(world.field or "")
            check(
                f"world:{world.record}.{world.field}",
                actual == world.equals,
                f"expected {world.equals!r}, got {actual!r}",
            )

    return Grade(correct=all(c["passed"] for c in checks), checks=checks)
