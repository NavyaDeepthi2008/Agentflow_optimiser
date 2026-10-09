"""Fixed safety rules that decide whether a tool may run in a given task.

These rules are written by developers, never learned, and always applied before
any optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.tools.spec import ToolSpec


@dataclass(frozen=True)
class RunPermissions:
    """What one task is allowed to do.

    * ``allowed_tools``: if set, only these tools may be used in the task.
      ``None`` means every registered tool.
    * ``side_effect_tools``: side-effecting AND unknown tools must be listed here
      explicitly, otherwise they are blocked. Read-only tools need no listing.
    """

    allowed_tools: frozenset[str] | None = None
    side_effect_tools: frozenset[str] = frozenset()

    def denial_reason(self, spec: ToolSpec) -> str | None:
        """Return why ``spec`` may not run, or ``None`` if it is permitted."""
        if self.allowed_tools is not None and spec.name not in self.allowed_tools:
            return f"tool '{spec.name}' is not allowed for this task"
        if spec.requires_explicit_permission and spec.name not in self.side_effect_tools:
            return (
                f"tool '{spec.name}' can change data or is unclassified, and this task has "
                "not been given permission to use it"
            )
        return None


@dataclass(frozen=True)
class RunLimits:
    """Hard limits applied identically in baseline and optimized modes."""

    max_tool_requests: int = 20
    max_identical_requests: int = 3  # same tool + same normalized args, per run
    max_args_bytes: int = 8_192

    def __post_init__(self) -> None:
        if self.max_tool_requests < 1:
            raise ValueError("max_tool_requests must be at least 1")
        if self.max_identical_requests < 1:
            raise ValueError("max_identical_requests must be at least 1")
        if self.max_args_bytes < 64:
            raise ValueError("max_args_bytes must be at least 64")
