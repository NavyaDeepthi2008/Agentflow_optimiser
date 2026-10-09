"""Retry policy for failed tool executions.

The same policy is used in baseline and optimized modes, so retries never make
one mode look better than the other.

A failed execution is retried only if ALL of these hold:
1. the error is transient (timeout, temporary outage) - never for bad input
   or internal bugs,
2. the tool is marked ``retryable``,
3. the tool is read-only, OR it is side-effecting AND supports idempotency keys
   AND this request actually carries a key (so a repeat cannot act twice).
   Unknown tools are never retried.
4. the retry budget and the run's remaining time allow it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from app.tools.spec import ToolEffect, ToolSpec


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2
    base_delay_seconds: float = 0.2
    max_delay_seconds: float = 2.0
    jitter_ratio: float = 0.1  # +/-10% randomness so retries don't line up

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("delays cannot be negative")
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be between 0 and 1")

    def delay(self, retry_number: int, rng: random.Random) -> float:
        """Wait before retry number ``retry_number`` (1 = first retry)."""
        base = min(self.max_delay_seconds, self.base_delay_seconds * 2 ** (retry_number - 1))
        return max(0.0, base * (1 + self.jitter_ratio * rng.uniform(-1, 1)))


NO_RETRIES = RetryPolicy(max_retries=0)


def retry_blocker(
    spec: ToolSpec, validated_args: dict[str, Any], error_retryable: bool
) -> str | None:
    """Return why a failed execution may NOT be retried, or None if it may."""
    if not error_retryable:
        return "error is not transient"
    if not spec.retryable:
        return "tool is not marked retryable"
    if spec.effect is ToolEffect.READ_ONLY:
        return None
    if spec.effect is ToolEffect.UNKNOWN:
        return "tool effect is unknown"
    if spec.idempotency_arg is None:
        return "side-effecting tool without idempotency support"
    if not validated_args.get(spec.idempotency_arg):
        return "side-effecting request without an idempotency key"
    return None
