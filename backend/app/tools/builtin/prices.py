"""Mock price tool whose value changes over (simulated) time.

Prices move every ``TICK_SECONDS`` using a fixed formula, so with a FakeClock the
exact price at any moment is known. This is what lets tests and the evaluation
prove that a stale (too old) price gives a wrong answer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.tools.builtin.base import MockTool
from app.tools.builtin.corpus import BASE_PRICES
from app.tools.errors import ToolNotFoundError
from app.tools.spec import CacheScope, Freshness, ToolEffect, ToolSpec

TICK_SECONDS = 300


def price_at(symbol: str, tick: int) -> float:
    """Deterministic price for ``symbol`` at time-step ``tick`` (within +/-1% of base)."""
    base = BASE_PRICES[symbol]
    seed = sum(ord(c) for c in symbol)
    offset = ((tick * 37 + seed) % 21) - 10  # -10 .. +10
    return round(base * (1 + offset / 1000), 2)


class PriceLookupArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str = Field(
        pattern=r"^[A-Z]{3}-X$",
        description="Fictional instrument symbol, e.g. 'NMB-X'. Available: NMB-X, ORB-X, KST-X.",
    )


class PriceLookup(MockTool):
    spec = ToolSpec(
        name="price_lookup",
        description=(
            "Get the current price of a fictional instrument. Prices change every few minutes."
        ),
        args_model=PriceLookupArgs,
        effect=ToolEffect.READ_ONLY,
        freshness=Freshness.TTL,
        ttl_seconds=60,
        max_ttl_seconds=TICK_SECONDS,
        cache_scope=CacheScope.SESSION,
        resource_tags=("price:{symbol}",),
        retryable=True,
        timeout_seconds=5.0,
    )

    def run(self, args: PriceLookupArgs) -> dict[str, Any]:
        if args.symbol not in BASE_PRICES:
            raise ToolNotFoundError(f"unknown symbol {args.symbol}")
        tick = int(self.world.elapsed_seconds() // TICK_SECONDS)
        return {
            "symbol": args.symbol,
            "price": price_at(args.symbol, tick),
            "currency": "USD",
            "as_of": self.world.clock.utcnow().isoformat(),
        }
