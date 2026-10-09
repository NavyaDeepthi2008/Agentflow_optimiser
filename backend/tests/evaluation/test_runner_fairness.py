"""Evaluation runner: fairness, cold-cache isolation, repeats, warm phase, regressions."""

from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from app.evaluation.dataset import load_dataset
from app.evaluation.metrics import ModelPrice, Pricing
from app.evaluation.runner import EvaluationConfig, EvaluationRunner, plan_runs
from app.gateway.models import GatewayMode
from app.llm.base import UserMessage
from app.llm.gemini import GeminiClient
from app.optimizer.retry import RetryPolicy
from tests.fakes.gemini_server import FAKE_KEY, FakeGeminiServer, call_reply, text_reply

FAST = RetryPolicy(base_delay_seconds=0.0, jitter_ratio=0.0)
DATASET = load_dataset()


def runner(**kw):
    kw.setdefault("retry_policy", FAST)
    return EvaluationRunner(**kw)


# ------------------------------------------------------------------ planning


def test_plan_is_reproducible_and_interleaved():
    config = EvaluationConfig(repetitions=2, seed=7)
    a = plan_runs(config, DATASET).runs
    b = plan_runs(config, DATASET).runs
    assert [(r.task.id, r.mode, r.repetition) for r in a] == [
        (r.task.id, r.mode, r.repetition) for r in b
    ]
    other = plan_runs(EvaluationConfig(repetitions=2, seed=8), DATASET).runs
    assert [r.task.id for r in a] != [r.task.id for r in other]
    assert len(a) == 18 * 2 * 2
    # Both modes of the same task+repetition run back to back.
    for first, second in zip(a[::2], a[1::2], strict=True):
        assert (first.task.id, first.repetition) == (second.task.id, second.repetition)
        assert {first.mode, second.mode} == set(GatewayMode)
    # Neither mode always goes first.
    assert len({pair.mode for pair in a[::2]}) == 2


def test_plan_limits_and_filters():
    with pytest.raises(ValueError):
        plan_runs(EvaluationConfig(task_ids=["missing"]), DATASET)
    with pytest.raises(ValueError):
        plan_runs(EvaluationConfig(repetitions=20, warm_cache=True), DATASET)  # > 400 runs
    one = plan_runs(
        EvaluationConfig(task_ids=["control_annual_price"], modes=[GatewayMode.OPTIMIZED]), DATASET
    )
    assert [(r.task.id, r.mode) for r in one.runs] == [("control_annual_price", "optimized")]
    with pytest.raises(ValueError):
        EvaluationConfig(modes=[GatewayMode.OPTIMIZED, GatewayMode.OPTIMIZED])


def test_gemini_plans_skip_scripted_only_tasks():
    plan = plan_runs(EvaluationConfig(llm="gemini"), DATASET)
    skipped = {s["task_id"] for s in plan.skipped}
    assert skipped == {t.id for t in DATASET.tasks if t.scripted_only}
    assert not any(r.task.scripted_only for r in plan.runs)


def test_warm_phase_excludes_tasks_whose_warm_up_changes_state():
    plan = plan_runs(EvaluationConfig(warm_cache=True), DATASET)
    warm_ids = {r.task.id for r in plan.runs if r.phase == "warm"}
    for task in DATASET.tasks:
        changes_state = (
            task.allow_side_effects
            or task.expected.world
            or task.setup.inject_failures
            or any(s.advance_seconds for s in task.script)
        )
        assert (task.id in warm_ids) is (not changes_state), task.id


# ------------------------------------------------------------------ full scripted runs


async def test_reference_behavior_is_correct_in_both_modes():
    """The core regression test: the optimizer must not break any task."""
    records, report = await runner().run(EvaluationConfig(), DATASET)
    wrong = [(r.task_id, r.mode, r.failed_checks) for r in records if not r.correct]
    assert wrong == []
    assert report["regressions"] == [] and report["improvements"] == []
    cold = report["phases"]["cold"]["modes"]
    assert cold["passthrough"]["accuracy"] == cold["optimized"]["accuracy"] == 1.0
    assert cold["passthrough"]["tool_requests"] == cold["optimized"]["tool_requests"]
    assert cold["optimized"]["tool_executions"] < cold["passthrough"]["tool_executions"]
    assert report["provider"] == "scripted" and report["label"].startswith("MOCK")
    assert report["dataset"]["checksum"] == DATASET.checksum


async def test_every_measured_run_is_isolated_and_cold():
    seen = []

    def before(kind, planned, ctx, llm):
        seen.append((kind, ctx.session_id, ctx.run_id))

    records, _ = await runner().run(EvaluationConfig(repetitions=2), DATASET, before_run=before)
    sessions = [s for _, s, _ in seen]
    assert len(set(sessions)) == len(sessions) == len(records)  # new session per run
    assert all(r.metrics.cache_hits == 0 for r in records)  # nothing carried over
    # Repeating a task does not get faster/cheaper through leftover cache.
    by_task = defaultdict(set)
    for r in records:
        by_task[(r.task_id, r.mode)].add(r.metrics.tool_executions)
    assert all(len(v) == 1 for v in by_task.values())


async def test_both_modes_get_identical_inputs():
    captured = defaultdict(dict)

    def before(kind, planned, ctx, llm):
        captured[(planned.task.id, planned.repetition)][planned.mode] = (ctx, llm)

    await runner().run(EvaluationConfig(), DATASET, before_run=before)
    for (task_id, _rep), modes in captured.items():
        (ctx_a, llm_a), (ctx_b, llm_b) = (
            modes[GatewayMode.PASSTHROUGH],
            modes[GatewayMode.OPTIMIZED],
        )
        assert ctx_a.permissions == ctx_b.permissions
        assert ctx_a.limits == ctx_b.limits and ctx_a.fresh_tools == ctx_b.fresh_tools
        first_a, first_b = llm_a.requests[0], llm_b.requests[0]
        assert first_a.system_prompt == first_b.system_prompt
        assert first_a.tools == first_b.tools
        # The agent sees ONLY the task text - no expectations.
        task = next(t for t in DATASET.tasks if t.id == task_id)
        assert first_a.messages == [UserMessage(text=task.task)]


async def test_repeated_runs_and_warm_phase_are_reported_separately():
    config = EvaluationConfig(
        repetitions=3, warm_cache=True, warm_gap_seconds=10, categories=["redundant", "control"]
    )
    records, report = await runner().run(config, DATASET)
    counts = Counter((r.task_id, r.mode, r.phase) for r in records)
    assert set(counts.values()) == {3}
    assert set(report["phases"]) == {"cold", "warm"}
    warm = report["phases"]["warm"]["modes"]
    assert warm["optimized"]["cache_hits"] > 0
    assert warm["passthrough"]["cache_hits"] == 0
    assert warm["optimized"]["accuracy"] == 1.0
    assert all(p["optimized"]["runs"] == 3 for p in report["per_task"])


async def test_fault_injection_applies_to_each_run():
    records, _ = await runner().run(
        EvaluationConfig(task_ids=["failure_transient_outage"], repetitions=2), DATASET
    )
    assert all(r.correct and r.metrics.tool_retries == 1 for r in records)


async def test_a_broken_optimizer_is_caught_as_a_regression(monkeypatch):
    from app.optimizer import cache_store

    monkeypatch.setattr(cache_store.CacheEntry, "is_fresh", lambda self, *_a: True)  # no TTL
    _, report = await runner().run(EvaluationConfig(categories=["freshness"]), DATASET)
    regressed = {r["task_id"] for r in report["regressions"]}
    assert "freshness_price_changed" in regressed
    item = next(r for r in report["regressions"] if r["task_id"] == "freshness_price_changed")
    assert "number:45.91" in item["optimized_failed_checks"]
    assert report["phases"]["cold"]["modes"]["passthrough"]["accuracy"] == 1.0


async def test_breaking_only_the_eligibility_rule_is_still_safe(monkeypatch):
    """Defense in depth: even if side effects were wrongly marked reusable, a write
    invalidates the cache, and a result produced during an invalidation is never
    stored - so the side effect still runs every time."""
    import app.gateway.gateway as gw

    monkeypatch.setattr(gw, "may_reuse_result", lambda spec: True)
    _, report = await runner().run(EvaluationConfig(categories=["side_effect"]), DATASET)
    assert report["regressions"] == []


async def test_skipped_side_effects_are_caught_by_the_evaluation(monkeypatch):
    import app.gateway.gateway as gw

    monkeypatch.setattr(gw, "may_reuse_result", lambda spec: True)  # cache everything
    monkeypatch.setattr(gw.ToolGateway, "_invalidate_after", lambda self, state, outcome: 0)
    _, report = await runner().run(EvaluationConfig(categories=["side_effect"]), DATASET)
    regression = next(
        r for r in report["regressions"] if r["task_id"] == "side_effect_double_reminder"
    )
    assert "world:outbox_count" in regression["optimized_failed_checks"]


# ------------------------------------------------------------------ real-model path (faked)


async def test_gemini_evaluation_is_labelled_live_and_reports_tokens_and_cost():
    server = FakeGeminiServer(
        [
            call_reply(("calculator", {"expression": "29*12"}, "c1")),
            text_reply("29 x 12 = 348"),
            call_reply(("calculator", {"expression": "29*12"}, "c2")),
            text_reply("It is 348."),
        ]
    )
    gemini = GeminiClient(api_key=FAKE_KEY, model="gemini-test", http_client=server.http_client())
    pricing = Pricing(
        source="test fixture",
        retrieved_on="2026-01-01",
        models={
            "gemini-test": ModelPrice(
                input_per_million=1, output_per_million=2, thinking_billed_as_output=True
            )
        },
        tool_per_execution={"calculator": 0.0},
    )
    records, report = await runner(gemini_factory=lambda: gemini, pricing=pricing).run(
        EvaluationConfig(llm="gemini", task_ids=["control_annual_price"]), DATASET
    )
    assert report["provider"] == "gemini" and report["label"].startswith("LIVE MODEL")
    assert all(r.correct for r in records)
    m = records[0].metrics
    assert m.input_tokens == 200 and m.usage_complete
    assert m.llm_cost == pytest.approx((200 * 1 + 40 * 2 + 10 * 2) / 1_000_000)
    s = report["phases"]["cold"]["modes"]["optimized"]
    assert s["tokens"]["complete"] and s["llm_cost"] is not None
    assert FAKE_KEY not in str(report)
