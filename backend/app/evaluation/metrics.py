"""Per-run metrics, cost estimation and summary statistics.

Definitions (used everywhere in reports):

* tool_requests      - tool calls the agent asked for (including blocked ones)
* tool_executions    - times tool code actually ran (retries included; reused = 0)
* cache_hits         - requests answered from an earlier run's stored result
* reused_in_run      - requests answered from earlier in the same run
* shared_in_flight   - requests that joined an identical request already running
* gateway_overhead   - time spent inside the gateway OUTSIDE tool code
                       (validation, safety checks, cache lookups, recording),
                       summed over the run's tool requests
* latency            - wall-clock duration of the whole run

Cost is only computed when real pricing data is configured AND the provider
reported complete token usage. Otherwise it is ``None`` ("unavailable") - never
guessed and never 0.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.agent.models import RunResult
from app.gateway.models import Decision

# ------------------------------------------------------------------ pricing


class ModelPrice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_per_million: float | None = Field(default=None, ge=0)
    output_per_million: float | None = Field(default=None, ge=0)
    # How thinking tokens are billed. If neither is set, cost is unavailable for
    # any run that used thinking tokens.
    thinking_per_million: float | None = Field(default=None, ge=0)
    thinking_billed_as_output: bool | None = None


class Pricing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(min_length=1)
    retrieved_on: str = Field(min_length=1)
    currency: str = "USD"
    models: dict[str, ModelPrice] = Field(default_factory=dict)
    tool_per_execution: dict[str, float] = Field(default_factory=dict)


def load_pricing(path: str | Path | None) -> Pricing | None:
    if not path:
        return None
    return Pricing.model_validate(json.loads(Path(path).read_text()))


def llm_cost(pricing: Pricing | None, result: RunResult) -> float | None:
    if pricing is None or not result.llm_calls:
        return None
    model_ids = {c.model_id for c in result.llm_calls}
    if len(model_ids) != 1:
        return None
    usage = result.usage
    return estimate_llm_cost(
        pricing,
        model_ids.pop(),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        thinking_tokens=usage.thinking_tokens,
        complete=usage.complete,
    )


def estimate_llm_cost(
    pricing: Pricing | None,
    model_id: str,
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    thinking_tokens: int | None,
    complete: bool | None,
) -> float | None:
    """Cost from token counts, or None when anything needed is unknown."""
    if pricing is None or not complete or input_tokens is None or output_tokens is None:
        return None
    price = pricing.models.get(model_id)
    if price is None or price.input_per_million is None or price.output_per_million is None:
        return None
    thinking = thinking_tokens or 0
    if thinking:
        if price.thinking_per_million is not None:
            thinking_cost = thinking * price.thinking_per_million
        elif price.thinking_billed_as_output:
            thinking_cost = thinking * price.output_per_million
        else:
            return None
    else:
        thinking_cost = 0.0
    total = (
        input_tokens * price.input_per_million
        + output_tokens * price.output_per_million
        + thinking_cost
    )
    return total / 1_000_000


def tool_cost(pricing: Pricing | None, result: RunResult) -> float | None:
    if pricing is None:
        return None
    total = 0.0
    for r in result.tool_results:
        if r.executions == 0:
            continue
        price = pricing.tool_per_execution.get(r.tool_name)
        if price is None:
            return None  # an executed tool has no known price
        total += price * r.executions
    return total


# ------------------------------------------------------------------ per run


class RunMetrics(BaseModel):
    status: str
    tool_requests: int
    tool_executions: int
    cache_hits: int
    reused_in_run: int
    shared_in_flight: int
    tool_retries: int
    llm_calls: int
    input_tokens: int | None
    output_tokens: int | None
    thinking_tokens: int | None
    total_tokens: int | None
    usage_complete: bool
    latency_ms: float
    llm_time_ms: float
    tool_time_ms: float
    gateway_overhead_ms: float
    llm_cost: float | None
    tool_cost: float | None


def run_metrics(result: RunResult, pricing: Pricing | None = None) -> RunMetrics:
    decisions = [r.decision for r in result.tool_results]
    overhead = sum(
        max(0.0, r.duration_ms - (r.tool_duration_ms or 0.0)) for r in result.tool_results
    )
    return RunMetrics(
        status=result.status.value,
        tool_requests=result.tool_request_count,
        tool_executions=result.tool_execution_count,
        cache_hits=decisions.count(Decision.CACHE_HIT),
        reused_in_run=decisions.count(Decision.REUSED_IN_RUN),
        shared_in_flight=decisions.count(Decision.SHARED_IN_FLIGHT),
        tool_retries=sum(max(0, r.executions - 1) for r in result.tool_results),
        llm_calls=result.llm_call_count,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        thinking_tokens=result.usage.thinking_tokens,
        total_tokens=result.usage.total_tokens,
        usage_complete=result.usage.complete,
        latency_ms=result.duration_ms,
        llm_time_ms=result.llm_time_ms,
        tool_time_ms=result.tool_time_ms,
        gateway_overhead_ms=overhead,
        llm_cost=llm_cost(pricing, result),
        tool_cost=tool_cost(pricing, result),
    )


# ------------------------------------------------------------------ statistics


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (no interpolation). None for an empty list."""
    if not values:
        return None
    if not 0 < p <= 100:
        raise ValueError("p must be in (0, 100]")
    ordered = sorted(values)
    rank = math.ceil(p / 100 * len(ordered))
    return ordered[rank - 1]


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def pct_change(before: float | None, after: float | None) -> float | None:
    """Percentage change from ``before`` to ``after``; None if undefined."""
    if before is None or after is None or before == 0:
        return None
    return (after - before) / before * 100
