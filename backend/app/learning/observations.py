"""Turn finished runs (in memory) into miner input. The learning service
builds the same shapes from stored rows."""

from __future__ import annotations

from app.agent.models import RunResult
from app.evaluation.report import EvalRunRecord
from app.learning.models import CallObservation, RunObservation


def calls_of(result: RunResult) -> list[CallObservation]:
    return [
        CallObservation(
            seq=i,
            tool_name=t.tool_name,
            decision=t.decision.value,
            ok=t.ok,
            executed=t.executed,
            fingerprint=t.fingerprint,
            output_digest=t.output_digest,
            data_time=t.data_time,
        )
        for i, t in enumerate(result.tool_results)
    ]


def observation(
    record: EvalRunRecord,
    result: RunResult,
    *,
    evaluation_id: str,
    warmup: RunResult | None = None,
) -> RunObservation:
    warm = None
    if warmup is not None:
        warm = RunObservation(
            run_id=warmup.run_id,
            session_id=warmup.session_id,
            evaluation_id=evaluation_id,
            task_id=record.task_id,
            split=record.split,
            mode=record.mode,
            phase="warmup",
            status=warmup.status.value,
            correct=bool(record.warmup_correct),
            calls=calls_of(warmup),
        )
    return RunObservation(
        run_id=record.run_id,
        session_id=result.session_id,
        evaluation_id=evaluation_id,
        task_id=record.task_id,
        split=record.split,
        mode=record.mode,
        phase=record.phase,
        status=record.metrics.status,
        correct=record.correct,
        calls=calls_of(result),
        warmup=warm,
        warmup_correct=record.warmup_correct,
    )
