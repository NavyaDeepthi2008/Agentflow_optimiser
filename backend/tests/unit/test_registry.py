from __future__ import annotations

import pytest

from app.tools.builtin.calculator import Calculator
from app.tools.catalog import BUILTIN_TOOLS, build_registry
from app.tools.mock_world import MockWorld
from app.tools.registry import RegistryFrozenError, ToolRegistry


def test_duplicate_names_rejected(clock):
    registry = ToolRegistry()
    tool = Calculator(MockWorld.create(clock))
    registry.register(tool.spec, tool)
    with pytest.raises(ValueError):
        registry.register(tool.spec, tool)


def test_non_callable_handler_rejected():
    with pytest.raises(TypeError):
        ToolRegistry().register(Calculator.spec, "not a function")  # type: ignore[arg-type]


def test_frozen_registry_cannot_change(clock):
    registry = build_registry(MockWorld.create(clock))
    assert registry.is_frozen
    with pytest.raises(RegistryFrozenError):
        registry.register(Calculator.spec, Calculator(MockWorld.create(clock)))


def test_demo_registry_excludes_test_tools(clock):
    registry = build_registry(MockWorld.create(clock))
    assert registry.names() == sorted(cls.spec.name for cls in BUILTIN_TOOLS)
    assert not any(s.test_only for s in registry.specs())


def test_unknown_spec_is_none(clock):
    assert build_registry(MockWorld.create(clock)).get_spec("does_not_exist") is None
