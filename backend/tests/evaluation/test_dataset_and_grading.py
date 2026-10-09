"""Dataset integrity and deterministic grading."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.evaluation.dataset import Dataset, EvalTask, load_dataset
from app.evaluation.grading import grade, normalize, numbers_in

# ------------------------------------------------------------------ dataset


def test_core_dataset_loads_with_all_required_categories():
    ds = load_dataset("core", "1")
    assert len(ds.tasks) == 18
    assert {t.category for t in ds.tasks} == {
        "redundant",
        "freshness",
        "side_effect",
        "failure",
        "control",
    }
    assert {t.split for t in ds.tasks} == {"train", "heldout"}
    assert len(ds.checksum) == 64
    assert load_dataset().checksum == ds.checksum  # stable


def test_dataset_names_are_validated():
    with pytest.raises(ValueError):
        load_dataset("../etc", "1")
    with pytest.raises(FileNotFoundError):
        load_dataset("nope", "1")


def test_selection_filters():
    ds = load_dataset()
    assert {t.split for t in ds.select(split="train")} == {"train"}
    assert [t.id for t in ds.select(task_ids=["control_annual_price"])] == ["control_annual_price"]
    assert {t.category for t in ds.select(categories=["freshness"])} == {"freshness"}
    with pytest.raises(ValueError):
        ds.select(task_ids=["does_not_exist"])


def _task(**overrides):
    base = {
        "id": "sample_task",
        "category": "control",
        "split": "train",
        "task": "Compute 1+1.",
        "tools": ["calculator"],
        "script": [{"answer": "2"}],
        "expected": {"answer": {"numbers": [{"value": 2}]}},
    }
    base.update(overrides)
    return EvalTask.model_validate(base)


@pytest.mark.parametrize(
    "overrides",
    [
        {"script": [{"tool_calls": [{"name": "calculator", "args": {}}]}]},  # no final answer
        {"script": [{"advance_seconds": 5, "answer": "2"}]},  # time travel needs scripted_only
        {"script": [{"answer": "2", "tool_calls": [{"name": "x"}]}]},  # two kinds in one step
        {"expected": {"status": "success"}},  # success without answer check
        {
            "expected": {
                "answer": {},
                "world": [  # a check cannot be both kinds at once
                    {"outbox_count": 1, "record": "R", "field": "status", "equals": "x"}
                ],
            }
        },
        {"expected": {"answer": {}, "world": [{"record": "R"}]}},  # incomplete record check
        {"category": "other"},
        {"id": "Bad Id"},
    ],
)
def test_invalid_tasks_are_rejected(overrides):
    with pytest.raises(ValidationError):
        _task(**overrides)


def test_duplicate_task_ids_rejected():
    t = _task().model_dump()
    with pytest.raises(ValidationError):
        Dataset.model_validate({"name": "x", "version": "1", "tasks": [t, t]})


# ------------------------------------------------------------------ grading


def test_normalization_and_number_extraction():
    assert normalize("  Hello\n  WORLD ") == "hello world"
    assert numbers_in("costs 1,200 USD and 3.5 more") == [1200.0, 3.5]


def run_grade(task, answer, *, status="success", outbox=0, records=None):
    return grade(
        task, status=status, final_answer=answer, outbox_count=outbox, records=records or {}
    )


def test_answer_checks():
    task = _task(
        expected={
            "answer": {
                "contains_all": ["Monday", "09:00"],
                "contains_none": ["Sunday"],
                "numbers": [{"value": 45.91, "tolerance": 0.001}],
            }
        }
    )
    assert run_grade(task, "monday to friday from 09:00; price 45.91").correct
    g = run_grade(task, "Sunday 09:00, price 45.18")
    assert not g.correct
    assert set(g.failed_checks) == {"contains:Monday", "excludes:Sunday", "number:45.91"}
    assert not run_grade(task, None).correct


def test_status_must_match():
    task = _task(expected={"status": "loop_detected"})
    assert run_grade(task, None, status="loop_detected").correct
    g = run_grade(task, "an answer", status="success")
    assert not g.correct and g.failed_checks == ["status"]


def test_world_checks_catch_wrong_side_effects_even_with_a_good_answer():
    task = _task(
        expected={
            "answer": {"contains_all": ["sent"]},
            "world": [
                {"outbox_count": 2},
                {"record": "REC-1", "field": "status", "equals": "closed"},
            ],
        }
    )
    good = run_grade(task, "sent", outbox=2, records={"REC-1": {"status": "closed"}})
    assert good.correct
    bad = run_grade(task, "sent", outbox=1, records={"REC-1": {"status": "open"}})
    assert not bad.correct
    assert set(bad.failed_checks) == {"world:outbox_count", "world:REC-1.status"}
    assert len(bad.checks) == 4  # every check is reported, not just the first failure
