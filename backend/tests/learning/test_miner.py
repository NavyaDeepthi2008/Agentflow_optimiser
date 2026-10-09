"""Evidence miner: what counts as evidence, and when NOT to propose."""

from __future__ import annotations

import itertools

import pytest

from app.gateway.factory import build_gateway
from app.learning.miner import mine
from app.learning.models import CallObservation, RunObservation, Thresholds

GW = build_gateway(include_test_tools=True)
NAMES = GW.tool_names()
_ids = itertools.count()


def call(tool, t, digest="d1", fp=None, decision="executed_live", ok=True, executed=True):
    return CallObservation(
        seq=next(_ids),
        tool_name=tool,
        decision=decision,
        ok=ok,
        executed=executed,
        fingerprint=fp or f"fp-{tool}",
        output_digest=digest if ok else None,
        data_time=t,
    )


def run(*calls, split="train", status="success", correct=True, phase="cold", **kw):
    n = next(_ids)
    return RunObservation(
        run_id=f"run{n}",
        session_id=kw.pop("session_id", f"sess{n}"),
        evaluation_id="ev1",
        task_id=kw.pop("task_id", "t1"),
        split=split,
        mode="passthrough",
        phase=phase,
        status=status,
        correct=correct,
        calls=list(calls),
        **kw,
    )


def recheck(gap, same=True, **kw):
    return run(call("record_read", 0), call("record_read", gap, "d1" if same else "d2"), **kw)


def candidates(runs, **kw):
    report = mine(runs, tool_names=NAMES, spec_of=GW.tool_spec, current={}, **kw)
    return report, {(c.tool_name, c.kind): c for c in report.candidates}


# ------------------------------------------------------------------ TTL
def test_consistent_repeats_propose_the_longest_observed_gap():
    runs = [recheck(600.4) for _ in range(3)] + [recheck(480) for _ in range(3)]
    report, c = candidates(runs)
    ttl = c[("record_read", "ttl")]
    assert ttl.proposed and ttl.proposed_setting == {"ttl_seconds": 600.0}  # rounded down
    assert ttl.current_setting == {"ttl_seconds": 300}
    ev = ttl.evidence
    assert ev.repeat_count == 6 and ev.distinct_runs == 6 and ev.consistency == 1.0
    assert len(ev.supporting_run_ids) == 6 and ev.evaluation_ids == ["ev1"]
    assert report.runs_used == 6


@pytest.mark.parametrize("n", [1, 4])
def test_weak_evidence_is_not_proposed(n):
    _, c = candidates([recheck(600) for _ in range(n)])
    ttl = c[("record_read", "ttl")]
    assert not ttl.proposed and ttl.proposed_setting is None
    assert any("weak evidence" in r for r in ttl.reasons)


def test_support_from_too_few_runs_is_weak():
    many_in_one = run(*[call("record_read", 400 * i) for i in range(7)])
    _, c = candidates([many_in_one])
    ttl = c[("record_read", "ttl")]
    assert not ttl.proposed
    assert any("1 runs" in r for r in ttl.reasons)


def test_stronger_thresholds_can_be_requested():
    runs = [recheck(600) for _ in range(6)]
    _, c = candidates(runs, thresholds=Thresholds(min_support=10))
    assert not c[("record_read", "ttl")].proposed


def test_any_change_within_the_current_ttl_blocks_a_proposal():
    runs = [recheck(600) for _ in range(6)] + [recheck(100, same=False)]
    _, c = candidates(runs)
    ttl = c[("record_read", "ttl")]
    assert not ttl.proposed and "even within the current TTL" in ttl.reasons[0]


def test_proposal_stays_below_the_first_observed_change():
    runs = [recheck(450) for _ in range(5)] + [recheck(700) for _ in range(5)]
    runs += [recheck(600, same=False)]
    _, c = candidates(runs)
    ttl = c[("record_read", "ttl")]
    assert ttl.proposed and ttl.proposed_setting == {"ttl_seconds": 450.0}
    assert ttl.evidence.min_changed_gap_seconds == 600
    assert ttl.evidence.repeat_count == 5  # only the 450 s repeats support it


def test_proposal_is_capped_at_the_tools_ceiling():
    _, c = candidates([recheck(5000) for _ in range(6)])
    ttl = c[("record_read", "ttl")]
    assert ttl.proposed_setting == {"ttl_seconds": 900.0}  # record_read max_ttl
    assert ttl.evidence.capped_by_ceiling


def test_repeats_within_current_ttl_propose_nothing():
    _, c = candidates([recheck(30) for _ in range(10)])
    ttl = c[("record_read", "ttl")]
    assert not ttl.proposed and "would save nothing" in ttl.reasons[0]


def test_outputs_that_always_differ_propose_nothing():
    runs = [run(call("price_lookup", 0, "a"), call("price_lookup", 120, "b")) for _ in range(6)]
    _, c = candidates(runs)
    assert not c[("price_lookup", "ttl")].proposed


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"split": "heldout"}, "not a training task"),
        ({"status": "limit_steps"}, "run did not succeed"),
        ({"correct": False}, "answer graded incorrect"),
    ],
)
def test_only_successful_correct_training_runs_count(kwargs, reason):
    report, c = candidates([recheck(600, **kwargs) for _ in range(6)])
    assert report.runs_used == 0 and report.excluded == {reason: 6}
    assert not c[("record_read", "ttl")].proposed


def test_runs_with_a_failed_tool_call_are_excluded():
    runs = [
        run(
            call("record_read", 0),
            call("doc_read", 1, decision="failed", ok=False),
            call("record_read", 600),
        )
        for _ in range(6)
    ]
    report, _ = candidates(runs)
    assert report.excluded == {"a tool call failed": 6}


def test_runs_recorded_before_stage_7_are_excluded():
    old = [
        run(call("record_read", 0, fp=None).model_copy(update={"fingerprint": None}))
        for _ in range(3)
    ]
    report, _ = candidates(old)
    assert report.excluded == {"recorded before Stage 7 (no evidence fields)": 3}


def test_reused_results_are_not_observations():
    runs = [
        run(call("record_read", 0), call("record_read", 600, decision="reused_in_run"))
        for _ in range(6)
    ]
    _, c = candidates(runs)
    assert c[("record_read", "ttl")].reasons == ["no repeated requests in the usable training runs"]


def test_a_write_between_repeats_breaks_the_pair():
    runs = [
        run(
            call("record_read", 0),
            call("record_update", 1, fp="w"),
            call("record_read", 600, "d2"),
        )
        for _ in range(6)
    ]
    _, c = candidates(runs)
    # The changed output after a write is NOT evidence of fast-changing data,
    # and the pair is not evidence of stable data either.
    assert c[("record_read", "ttl")].reasons == ["no repeated requests in the usable training runs"]


def test_different_requests_are_never_paired():
    runs = [
        run(call("record_read", 0, fp="a"), call("record_read", 600, "d2", fp="b"))
        for _ in range(6)
    ]
    _, c = candidates(runs)
    assert not c[("record_read", "ttl")].proposed


def test_current_setting_includes_active_rules():
    runs = [recheck(600) for _ in range(6)]
    report = mine(
        runs,
        tool_names=NAMES,
        spec_of=GW.tool_spec,
        current={"record_read": {"ttl_seconds": 600.0, "cache_scope": "session"}},
    )
    c = {(x.tool_name, x.kind): x for x in report.candidates}
    assert not c[("record_read", "ttl")].proposed  # already at 600
    assert ("record_read", "cache_scope") not in c  # already session-wide


# ------------------------------------------------------------------ safety
def test_side_effect_unknown_and_live_tools_are_never_candidates():
    runs = [
        run(
            call("send_notification", 0, fp="n"),
            call("send_notification", 600, fp="n"),
            call("unknown_effect_tool", 0, fp="u"),
            call("unknown_effect_tool", 600, fp="u"),
            call("slow_tool", 0, fp="s"),
            call("slow_tool", 600, fp="s"),
        )
        for _ in range(10)
    ]
    report, c = candidates(runs)
    tools = {name for name, _ in c}
    assert tools.isdisjoint({"send_notification", "record_update", "unknown_effect_tool"})
    assert "slow_tool" not in tools
    assert {t["tool_name"] for t in report.not_eligible} == {"record_update", "send_notification"}
    assert all(not cand.proposed for cand in report.candidates)


def test_static_and_global_tools_get_no_candidates():
    _, c = candidates([run(call("calculator", 0), call("calculator", 9999)) for _ in range(6)])
    assert not any(name == "calculator" for name, _ in c)


# ------------------------------------------------------------------ scope
def warm_pair(same=True, warm_ok=True, gap=1.0, session_id=None, **kw):
    session_id = session_id or f"ws{next(_ids)}"
    warm = run(call("record_read", 0), phase="warmup", session_id=session_id)
    measured = run(
        call("record_read", gap, "d1" if same else "d2"),
        session_id=session_id,
        phase="warm",
        warmup=warm,
        warmup_correct=warm_ok,
        **kw,
    )
    return measured


def test_scope_widening_needs_consistent_cross_run_repeats():
    _, c = candidates([warm_pair() for _ in range(5)])
    scope = c[("record_read", "cache_scope")]
    assert scope.proposed and scope.proposed_setting == {"cache_scope": "session"}
    assert scope.current_setting == {"cache_scope": "run"}
    assert scope.evidence.repeat_count == 5 and scope.evidence.distinct_runs == 5
    assert len(scope.evidence.supporting_run_ids) == 10  # warm-up + measured runs


def test_scope_not_proposed_when_output_differs_between_runs():
    _, c = candidates([warm_pair() for _ in range(5)] + [warm_pair(same=False)])
    scope = c[("record_read", "cache_scope")]
    assert not scope.proposed and "differed" in scope.reasons[0]


def test_scope_ignores_ungraded_or_incorrect_warmups():
    _, c = candidates([warm_pair(warm_ok=False) for _ in range(5)])
    assert not c[("record_read", "cache_scope")].proposed


def test_scope_ignores_repeats_older_than_the_ttl():
    _, c = candidates([warm_pair(gap=400) for _ in range(5)])
    assert not c[("record_read", "cache_scope")].proposed


def test_scope_needs_distinct_sessions():
    _, c = candidates([warm_pair(session_id="same") for _ in range(6)])
    scope = c[("record_read", "cache_scope")]
    assert not scope.proposed and any("1 sessions" in r for r in scope.reasons)


def test_scope_ignores_a_warmup_from_another_session():
    pairs = []
    for _ in range(5):
        p = warm_pair()
        p.warmup.session_id = "elsewhere"
        pairs.append(p)
    _, c = candidates(pairs)
    assert not c[("record_read", "cache_scope")].proposed
