"""Evidence miner: propose conservative TTL and cache-scope rules from stored,
correctly graded training runs.

What counts as evidence
-----------------------
Only runs that are ALL of: from the training split, finished with status
``success``, graded correct, and without any failed tool call. Within those,
only LIVE executions are observations (a reused result says nothing about
whether the data changed). Two live executions of the identical request
(same fingerprint: tool + version + normalized arguments) form a *repeat*:

* **TTL repeats**: inside one run. The gap is measured on the data clock.
* **Scope repeats**: the graded warm-up run and the graded measured run of the
  warm phase, which share one session. These show whether a result could be
  shared between runs of a session.

A repeat with an executed side-effecting/unknown tool between the two
executions is skipped: invalidation handles writes, so it says nothing about
how fast the data changes on its own.

What gets proposed
------------------
* **TTL** (freshness=ttl tools): the longest gap at which the output was still
  identical, rounded down, only if every repeat up to that gap was identical
  (100% consistency), it is strictly shorter than the shortest gap at which
  the output changed, it is longer than the current TTL (otherwise nothing is
  gained) and it is capped at the tool's ``max_ttl_seconds``.
* **Scope** (tools shared only within a run): widen to ``session`` if every
  cross-run repeat within the current TTL returned identical output.

Both need at least ``min_support`` supporting repeats from at least
``min_distinct_runs`` different runs (or sessions). Tools that may never reuse
results (side effects, unknown effect, live data, test-only) are never
considered. Everything not proposed is reported with the reason.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

from app.learning.models import (
    LIVE_DECISIONS,
    CallObservation,
    Candidate,
    Evidence,
    MiningReport,
    Pair,
    RunObservation,
    Thresholds,
)
from app.optimizer.eligibility import may_reuse_result

SpecLookup = Callable[[str], Any]  # tool name -> ToolSpec | None
SAMPLE_SIZE = 20


def _is_live(call: CallObservation) -> bool:
    return (
        call.ok
        and call.executed
        and call.decision in LIVE_DECISIONS
        and call.fingerprint is not None
        and call.output_digest is not None
        and call.data_time is not None
    )


def _is_write(call: CallObservation, spec_of: SpecLookup) -> bool:
    if not call.executed:
        return False
    spec = spec_of(call.tool_name)
    return spec is None or not spec.is_read_only  # unknown tools count as writes


def exclusion_reason(run: RunObservation) -> str | None:
    if run.split != "train":
        return "not a training task"
    if run.status != "success":
        return "run did not succeed"
    if not run.correct:
        return "answer graded incorrect"
    if any(c.decision == "failed" for c in run.calls):
        return "a tool call failed"
    if any(c.executed and c.fingerprint is None for c in run.calls):
        return "recorded before Stage 7 (no evidence fields)"
    return None


def ttl_pairs(run: RunObservation, spec_of: SpecLookup) -> list[Pair]:
    pairs: list[Pair] = []
    writes = 0
    last: dict[str, tuple[CallObservation, int]] = {}
    for call in sorted(run.calls, key=lambda c: c.seq):
        if _is_write(call, spec_of):
            writes += 1
            continue
        if not _is_live(call):
            continue
        assert call.fingerprint is not None and call.data_time is not None
        prev = last.get(call.fingerprint)
        if prev is not None and prev[1] == writes:
            first = prev[0]
            assert first.data_time is not None
            pairs.append(
                Pair(
                    tool_name=call.tool_name,
                    run_id=run.run_id,
                    first_run_id=run.run_id,
                    session_id=run.session_id,
                    task_id=run.task_id,
                    gap_seconds=call.data_time - first.data_time,
                    same_output=call.output_digest == first.output_digest,
                )
            )
        last[call.fingerprint] = (call, writes)
    return pairs


def scope_pairs(run: RunObservation, spec_of: SpecLookup) -> list[Pair]:
    """First live execution in the measured warm run vs. the warm-up's last."""
    warm = run.warmup
    if run.phase != "warm" or warm is None or run.warmup_correct is not True:
        return []
    if warm.session_id != run.session_id:
        return []  # only runs of ONE session can show what session sharing would do
    if warm.status != "success" or any(c.decision == "failed" for c in warm.calls):
        return []
    last_in_warmup: dict[str, CallObservation] = {}
    for call in sorted(warm.calls, key=lambda c: c.seq):
        if _is_write(call, spec_of):
            last_in_warmup.clear()  # nothing before a write may be compared
        elif _is_live(call):
            assert call.fingerprint is not None
            last_in_warmup[call.fingerprint] = call
    pairs: list[Pair] = []
    seen: set[str] = set()
    for call in sorted(run.calls, key=lambda c: c.seq):
        if _is_write(call, spec_of):
            break
        if not _is_live(call) or call.fingerprint in seen:
            continue
        assert call.fingerprint is not None and call.data_time is not None
        seen.add(call.fingerprint)
        first = last_in_warmup.get(call.fingerprint)
        if first is None:
            continue
        assert first.data_time is not None
        pairs.append(
            Pair(
                tool_name=call.tool_name,
                run_id=run.run_id,
                first_run_id=warm.run_id,
                session_id=run.session_id,
                task_id=run.task_id,
                gap_seconds=call.data_time - first.data_time,
                same_output=call.output_digest == first.output_digest,
            )
        )
    return pairs


def _evidence(
    supporting: list[Pair],
    all_pairs: list[Pair],
    runs_by_id: dict[str, RunObservation],
    *,
    distinct_key: Callable[[Pair], str],
    in_range: list[Pair],
    capped: bool = False,
) -> Evidence:
    changed = [p for p in all_pairs if not p.same_output]
    unchanged = [p for p in all_pairs if p.same_output]
    run_ids = sorted({r for p in supporting for r in (p.first_run_id, p.run_id)})
    return Evidence(
        supporting_run_ids=run_ids,
        supporting_task_ids=sorted({p.task_id for p in supporting}),
        evaluation_ids=sorted(
            {runs_by_id[p.run_id].evaluation_id for p in supporting if p.run_id in runs_by_id}
        ),
        repeat_count=len(supporting),
        distinct_runs=len({distinct_key(p) for p in supporting}),
        consistency=(sum(p.same_output for p in in_range) / len(in_range) if in_range else None),
        observed_repeats=len(all_pairs),
        changed_repeats=len(changed),
        max_unchanged_gap_seconds=max((p.gap_seconds for p in unchanged), default=None),
        min_changed_gap_seconds=min((p.gap_seconds for p in changed), default=None),
        capped_by_ceiling=capped,
        sample=sorted(supporting or all_pairs, key=lambda p: -p.gap_seconds)[:SAMPLE_SIZE],
    )


def _weak(evidence: Evidence, t: Thresholds, unit: str) -> list[str]:
    reasons = []
    if evidence.repeat_count < t.min_support:
        reasons.append(
            f"weak evidence: {evidence.repeat_count} supporting repeats "
            f"(at least {t.min_support} needed)"
        )
    if evidence.distinct_runs < t.min_distinct_runs:
        reasons.append(
            f"weak evidence: support comes from {evidence.distinct_runs} {unit} "
            f"(at least {t.min_distinct_runs} needed)"
        )
    return reasons


def _ttl_candidate(spec, pairs, current, runs_by_id, t: Thresholds) -> Candidate:
    current_ttl = current["ttl_seconds"]
    base = {
        "tool_name": spec.name,
        "tool_version": spec.version,
        "kind": "ttl",
        "current_setting": {"ttl_seconds": current_ttl},
    }
    if not pairs:
        return Candidate(
            **base,
            proposed_setting=None,
            proposed=False,
            reasons=["no repeated requests in the usable training runs"],
        )
    changed = [p for p in pairs if not p.same_output]
    min_changed = min((p.gap_seconds for p in changed), default=math.inf)
    early_changes = [p for p in changed if p.gap_seconds <= current_ttl]
    if early_changes:
        ev = _evidence([], pairs, runs_by_id, distinct_key=lambda p: p.run_id, in_range=pairs)
        return Candidate(
            **base,
            proposed_setting=None,
            proposed=False,
            reasons=[
                f"the output changed between identical requests even within the current TTL "
                f"({len(early_changes)} of {len(pairs)} repeats); keeping the current TTL"
            ],
            evidence=ev,
        )
    longer = [
        p.gap_seconds
        for p in pairs
        if p.same_output and p.gap_seconds > current_ttl and p.gap_seconds < min_changed
    ]
    if not longer:
        ev = _evidence([], pairs, runs_by_id, distinct_key=lambda p: p.run_id, in_range=pairs)
        reason = (
            "every repeat already fell within the current TTL; a longer TTL would save nothing"
            if not changed
            else "the output changed at every gap longer than the current TTL"
        )
        return Candidate(
            **base, proposed_setting=None, proposed=False, reasons=[reason], evidence=ev
        )
    longest = max(longer)
    # Rounded down, then capped: the proposal never exceeds what was observed.
    proposed_ttl = float(math.floor(longest))
    capped = proposed_ttl > spec.max_ttl_seconds
    proposed_ttl = min(proposed_ttl, float(spec.max_ttl_seconds))
    in_range = [p for p in pairs if p.gap_seconds <= longest]
    supporting = [p for p in in_range if p.gap_seconds > current_ttl]
    ev = _evidence(
        supporting,
        pairs,
        runs_by_id,
        distinct_key=lambda p: p.run_id,
        in_range=in_range,
        capped=capped,
    )
    reasons: list[str] = []
    if proposed_ttl <= current_ttl:
        reasons.append("no longer TTL fits under the tool's ceiling")
    if ev.consistency != t.required_consistency:  # cannot happen by construction; checked anyway
        reasons.append("not every repeat in range returned identical output")
    reasons += _weak(ev, t, "runs")
    if reasons:
        return Candidate(
            **base, proposed_setting=None, proposed=False, reasons=reasons, evidence=ev
        )
    why = [
        f"{ev.repeat_count} repeats {current_ttl:g}-{proposed_ttl:g} s apart all returned "
        f"identical output, in {ev.distinct_runs} different runs"
    ]
    if ev.min_changed_gap_seconds is not None:
        why.append(
            f"the output changed after {ev.min_changed_gap_seconds:.0f} s, so the proposal "
            "stays below that"
        )
    if capped:
        why.append(f"capped at the tool's ceiling of {spec.max_ttl_seconds:g} s")
    return Candidate(
        **base,
        proposed_setting={"ttl_seconds": proposed_ttl},
        proposed=True,
        reasons=why,
        evidence=ev,
    )


def _scope_candidate(spec, pairs, current, runs_by_id, t: Thresholds) -> Candidate:
    base = {
        "tool_name": spec.name,
        "tool_version": spec.version,
        "kind": "cache_scope",
        "current_setting": {"cache_scope": current["cache_scope"]},
    }
    ttl = current["ttl_seconds"]
    relevant = [p for p in pairs if ttl is None or p.gap_seconds < ttl]
    if not relevant:
        return Candidate(
            **base,
            proposed_setting=None,
            proposed=False,
            reasons=[
                "no identical requests in two graded runs of the same session within the TTL "
                "(run a training evaluation with the warm phase turned on)"
            ],
        )
    ev = _evidence(
        relevant,
        relevant,
        runs_by_id,
        distinct_key=lambda p: p.session_id,
        in_range=relevant,
    )
    reasons: list[str] = []
    if ev.changed_repeats:
        reasons.append(
            f"the output differed between runs of the same session in "
            f"{ev.changed_repeats} of {ev.observed_repeats} repeats"
        )
    reasons += _weak(ev, t, "sessions")
    if reasons:
        return Candidate(
            **base, proposed_setting=None, proposed=False, reasons=reasons, evidence=ev
        )
    return Candidate(
        **base,
        proposed_setting={"cache_scope": "session"},
        proposed=True,
        reasons=[
            f"{ev.repeat_count} identical requests in a later run of the same session all "
            f"returned identical output, in {ev.distinct_runs} different sessions",
            "results would still never be shared between sessions",
        ],
        evidence=ev,
    )


def not_eligible_reason(spec) -> str | None:
    if spec.test_only:
        return "test-only tool"
    if not may_reuse_result(spec):
        if not spec.is_read_only:
            return f"{spec.effect.value} tool: always executes (fixed safety rule)"
        return "live data: always executes (fixed safety rule)"
    return None


def mine(
    runs: list[RunObservation],
    *,
    tool_names: list[str],
    spec_of: SpecLookup,
    current: dict[str, dict[str, Any]],
    thresholds: Thresholds | None = None,
) -> MiningReport:
    """``current[tool]`` is the setting in force now: ``ttl_seconds`` and
    ``cache_scope`` after any active learned rule."""
    t = thresholds or Thresholds()
    excluded: Counter[str] = Counter()
    used: list[RunObservation] = []
    for run in runs:
        reason = exclusion_reason(run)
        if reason:
            excluded[reason] += 1
        else:
            used.append(run)
    runs_by_id = {r.run_id: r for r in used}

    ttl_by_tool: dict[str, list[Pair]] = defaultdict(list)
    scope_by_tool: dict[str, list[Pair]] = defaultdict(list)
    for run in used:
        for p in ttl_pairs(run, spec_of):
            ttl_by_tool[p.tool_name].append(p)
        for p in scope_pairs(run, spec_of):
            scope_by_tool[p.tool_name].append(p)

    candidates: list[Candidate] = []
    not_eligible: list[dict[str, str]] = []
    for name in sorted(tool_names):
        spec = spec_of(name)
        if spec is None:
            continue
        reason = not_eligible_reason(spec)
        if reason:
            if not spec.test_only:
                not_eligible.append({"tool_name": name, "reason": reason})
            continue
        now = current.get(name) or {
            "ttl_seconds": spec.ttl_seconds,
            "cache_scope": spec.cache_scope.value,
        }
        if spec.freshness.value == "ttl" and spec.max_ttl_seconds is not None:
            candidates.append(_ttl_candidate(spec, ttl_by_tool[name], now, runs_by_id, t))
        if now["cache_scope"] == "run":
            candidates.append(_scope_candidate(spec, scope_by_tool[name], now, runs_by_id, t))

    return MiningReport(
        thresholds=t,
        runs_examined=len(runs),
        runs_used=len(used),
        excluded=dict(excluded),
        candidates=candidates,
        not_eligible=not_eligible,
    )
