"""The tool registry: which tools exist, their metadata, and their code.

Only the gateway should run tool code. The registry therefore exposes tool
*metadata* publicly but keeps the callable handlers behind the private
``_handler_for`` method. The architecture tests fail the build if any module
other than the gateway calls it.

(Python cannot make this impossible for deliberately malicious code in the same
process. It is a structural rule for our own code, enforced by tests.)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.tools.spec import ToolSpec

Handler = Callable[[BaseModel], Any]


@dataclass(frozen=True)
class _Entry:
    spec: ToolSpec
    handler: Handler


class RegistryFrozenError(RuntimeError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._frozen = False

    def register(self, spec: ToolSpec, handler: Handler) -> None:
        if self._frozen:
            raise RegistryFrozenError("registry is frozen; tools cannot be added")
        if spec.name in self._entries:
            raise ValueError(f"tool {spec.name!r} is already registered")
        if not callable(handler):
            raise TypeError(f"handler for {spec.name!r} is not callable")
        self._entries[spec.name] = _Entry(spec=spec, handler=handler)

    def freeze(self) -> None:
        self._frozen = True

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def get_spec(self, name: str) -> ToolSpec | None:
        entry = self._entries.get(name)
        return entry.spec if entry else None

    def specs(self) -> list[ToolSpec]:
        return [self._entries[n].spec for n in sorted(self._entries)]

    def names(self) -> list[str]:
        return sorted(self._entries)

    # Private: only app.gateway.gateway may call this (enforced by tests).
    def _handler_for(self, name: str) -> Handler:
        return self._entries[name].handler
