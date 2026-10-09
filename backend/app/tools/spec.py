"""Tool metadata (``ToolSpec``).

Every tool declares *what kind* of tool it is. The gateway and the future
optimizer make all their safety decisions from this metadata, so the rules are
checked here, when a spec is created, rather than trusted later.

Safe defaults: a spec that does not say otherwise is ``effect=UNKNOWN`` and
``freshness=LIVE``. Unknown tools are handled exactly like side-effecting ones:
they are never reused or skipped, and they need explicit task permission.
"""

from __future__ import annotations

import string
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Argument name the gateway reserves for "fetch a fresh result" requests.
RESERVED_ARG = "force_fresh"


class ToolEffect(StrEnum):
    READ_ONLY = "read_only"  # only reads; running it twice changes nothing
    SIDE_EFFECT = "side_effect"  # changes the world (writes, sends, pays)
    UNKNOWN = "unknown"  # not classified -> treated as SIDE_EFFECT


class Freshness(StrEnum):
    STATIC = "static"  # result never changes for the same arguments
    TTL = "ttl"  # result may be reused for ``ttl_seconds``
    LIVE = "live"  # result must never be reused


class CacheScope(StrEnum):
    RUN = "run"  # reuse only inside one agent run
    SESSION = "session"  # reuse across runs of the same session
    GLOBAL = "global"  # reuse across sessions - public, non-personal data only


class ToolSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,47}$")
    version: str = Field(default="1", min_length=1, max_length=20)
    description: str = Field(min_length=10, max_length=500)
    args_model: type[BaseModel]

    effect: ToolEffect = ToolEffect.UNKNOWN
    freshness: Freshness = Freshness.LIVE
    ttl_seconds: float | None = None
    max_ttl_seconds: float | None = None  # ceiling that learning may never exceed
    cache_scope: CacheScope = CacheScope.RUN
    public_data: bool = False

    resource_tags: tuple[str, ...] = ()  # e.g. "record:{record_id}" (what a read depends on)
    # What a side-effecting tool makes stale, e.g. ("record:{record_id}",).
    # None = not declared -> the safe assumption: it may have changed anything, so
    # every cached result is invalidated. () = explicitly changes no cached data.
    invalidates: tuple[str, ...] | None = None

    # Cache-key normalization hints. Only use them where they cannot change the
    # tool's meaning (e.g. the search tool ignores case and extra spaces).
    normalize_whitespace_args: tuple[str, ...] = ()  # strip + collapse runs of spaces
    casefold_args: tuple[str, ...] = ()  # compare case-insensitively

    timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    retryable: bool = False
    # Name of the argument carrying an idempotency key, if the tool supports one.
    idempotency_arg: str | None = None
    test_only: bool = False

    # ----------------------------------------------------------------- checks
    @model_validator(mode="after")
    def _check_consistency(self) -> ToolSpec:
        extra = self.args_model.model_config.get("extra")
        if extra != "forbid":
            raise ValueError(
                f"{self.name}: args_model must set extra='forbid' so unexpected "
                "arguments are rejected"
            )

        if self.freshness is Freshness.TTL:
            if self.ttl_seconds is None or self.ttl_seconds <= 0:
                raise ValueError(f"{self.name}: freshness=ttl requires ttl_seconds > 0")
        elif self.ttl_seconds is not None:
            raise ValueError(f"{self.name}: ttl_seconds is only valid with freshness=ttl")

        if self.max_ttl_seconds is not None:
            if self.freshness is Freshness.LIVE:
                raise ValueError(f"{self.name}: a live tool cannot have max_ttl_seconds")
            if self.ttl_seconds is not None and self.max_ttl_seconds < self.ttl_seconds:
                raise ValueError(f"{self.name}: max_ttl_seconds must be >= ttl_seconds")

        if not self.is_read_only:
            # Side-effecting and unknown tools: the fixed safety rule.
            if self.freshness is not Freshness.LIVE:
                raise ValueError(
                    f"{self.name}: side-effecting/unknown tools must use freshness=live "
                    "(their results may never be reused)"
                )
            if self.cache_scope is not CacheScope.RUN:
                raise ValueError(f"{self.name}: side-effecting/unknown tools cannot widen scope")
            if self.retryable and not self.supports_idempotency_key:
                raise ValueError(
                    f"{self.name}: a side-effecting tool may only be retryable if it "
                    "supports idempotency keys"
                )
            if self.resource_tags:
                raise ValueError(f"{self.name}: resource_tags are for read-only tools")

        if self.invalidates and self.effect is not ToolEffect.SIDE_EFFECT:
            raise ValueError(f"{self.name}: only side-effecting tools can invalidate")

        if self.cache_scope is CacheScope.GLOBAL and not self.public_data:
            raise ValueError(f"{self.name}: global scope requires public_data=True")

        fields = set(self.args_model.model_fields)
        if RESERVED_ARG in fields:
            raise ValueError(f"{self.name}: argument name {RESERVED_ARG!r} is reserved")
        for arg in (*self.normalize_whitespace_args, *self.casefold_args):
            if arg not in fields:
                raise ValueError(f"{self.name}: normalization hint for unknown argument {arg!r}")
        if self.idempotency_arg is not None and self.idempotency_arg not in fields:
            raise ValueError(f"{self.name}: idempotency_arg {self.idempotency_arg!r} is unknown")
        for template in (*self.resource_tags, *(self.invalidates or ())):
            for _, placeholder, _, _ in string.Formatter().parse(template):
                if placeholder is not None and placeholder not in fields:
                    raise ValueError(
                        f"{self.name}: tag {template!r} uses unknown argument {placeholder!r}"
                    )
        return self

    # ----------------------------------------------------------------- helpers
    @property
    def is_read_only(self) -> bool:
        return self.effect is ToolEffect.READ_ONLY

    @property
    def supports_idempotency_key(self) -> bool:
        return self.idempotency_arg is not None

    @property
    def invalidates_everything(self) -> bool:
        """True if running this tool must clear every cached result.

        Unknown tools always do; side-effecting tools do unless they declared
        exactly what they change.
        """
        if self.effect is ToolEffect.UNKNOWN:
            return True
        return self.effect is ToolEffect.SIDE_EFFECT and self.invalidates is None

    @property
    def requires_explicit_permission(self) -> bool:
        """Side-effecting AND unknown tools need the task to allow them explicitly."""
        return not self.is_read_only

    @staticmethod
    def render_tags(templates: tuple[str, ...], args: dict[str, Any]) -> list[str]:
        return [t.format(**args) for t in templates]
