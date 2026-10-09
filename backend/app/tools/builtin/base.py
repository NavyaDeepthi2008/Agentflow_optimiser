"""Base class for mock tools.

Each mock tool counts its own executions in ``world.execution_counts`` the moment
it is invoked. This count is independent of the gateway's own counters.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel

from app.tools.errors import ToolTransientError
from app.tools.mock_world import MockWorld
from app.tools.spec import ToolSpec


class MockTool:
    spec: ClassVar[ToolSpec]
    # Built-in sync tools are instant, so their whole body runs under the world
    # lock (atomic). A tool that may block for a long time must not hold it:
    # async tools count their calls under the same lock ON THE EVENT LOOP, so a
    # long-held lock would freeze every run. Such tools lock only their writes.
    run_under_lock: ClassVar[bool] = True

    def __init__(self, world: MockWorld) -> None:
        self.world = world

    def __call__(self, args: BaseModel) -> Any:
        # For sync tools the whole run happens under the lock. For async tools
        # only the counting does; their body runs when the gateway awaits it.
        with self.world.lock:
            self.world.execution_counts[self.spec.name] += 1
            if self.world.injected_failures[self.spec.name] > 0:
                self.world.injected_failures[self.spec.name] -= 1
                raise ToolTransientError("simulated temporary outage")
            if self.run_under_lock:
                return self.run(args)
        return self.run(args)

    def run(self, args: Any) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError
