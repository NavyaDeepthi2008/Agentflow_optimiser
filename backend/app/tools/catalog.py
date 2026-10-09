"""Builds the registry of available tools.

This is the only place outside the tool modules themselves that imports tool
implementations. Everything else talks to tools through the ToolGateway.
"""

from __future__ import annotations

from app.tools.builtin.base import MockTool
from app.tools.builtin.calculator import Calculator
from app.tools.builtin.knowledge import DocRead, KbSearch
from app.tools.builtin.notify import SendNotification
from app.tools.builtin.prices import PriceLookup
from app.tools.builtin.records import RecordRead, RecordUpdate
from app.tools.mock_world import MockWorld
from app.tools.registry import ToolRegistry

BUILTIN_TOOLS: tuple[type[MockTool], ...] = (
    Calculator,
    KbSearch,
    DocRead,
    PriceLookup,
    RecordRead,
    RecordUpdate,
    SendNotification,
)


def build_registry(world: MockWorld, *, include_test_tools: bool = False) -> ToolRegistry:
    registry = ToolRegistry()
    tool_classes: list[type[MockTool]] = list(BUILTIN_TOOLS)
    if include_test_tools:
        from app.tools.test_only.tools import TEST_TOOLS

        tool_classes.extend(TEST_TOOLS)
    for cls in tool_classes:
        tool = cls(world)
        registry.register(tool.spec, tool)
    registry.freeze()
    return registry
