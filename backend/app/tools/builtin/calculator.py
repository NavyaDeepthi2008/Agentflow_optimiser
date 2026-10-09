"""Calculator tool: safe arithmetic without ``eval``.

The expression is parsed into a syntax tree and only numbers, + - * / // % **
and parentheses are allowed. Names, function calls, attributes and anything else
are rejected, so an LLM (or a prompt injection) cannot run code through it.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.tools.builtin.base import MockTool
from app.tools.errors import ToolInputError
from app.tools.spec import CacheScope, Freshness, ToolEffect, ToolSpec

MAX_ABS_VALUE = 1e15
MAX_EXPONENT = 64

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class CalculatorArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: str = Field(
        min_length=1,
        max_length=200,
        description="Arithmetic expression using numbers, + - * / // % ** and parentheses.",
    )


def _check(value: float | int) -> float | int:
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolInputError("result is not a finite number")
    if abs(value) > MAX_ABS_VALUE:
        raise ToolInputError(f"result exceeds the allowed magnitude ({MAX_ABS_VALUE:g})")
    return value


def _evaluate(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ToolInputError("only numbers are allowed")
        return _check(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _check(_UNARY_OPS[type(node.op)](_evaluate(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolInputError(f"exponent larger than {MAX_EXPONENT} is not allowed")
        if isinstance(node.op, ast.Div | ast.FloorDiv | ast.Mod) and right == 0:
            raise ToolInputError("division by zero")
        try:
            result = _BIN_OPS[type(node.op)](left, right)
        except (OverflowError, ZeroDivisionError) as exc:
            raise ToolInputError(f"cannot compute: {exc}") from None
        if isinstance(result, complex):
            raise ToolInputError("result is not a real number")
        return _check(result)
    raise ToolInputError("unsupported expression: only arithmetic is allowed")


def evaluate_expression(expression: str) -> float | int:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ToolInputError("expression is not valid arithmetic") from None
    result = _evaluate(tree)
    if isinstance(result, float) and result.is_integer():
        return int(result)
    if isinstance(result, float):
        return round(result, 10)
    return result


class Calculator(MockTool):
    spec = ToolSpec(
        name="calculator",
        description="Evaluate an arithmetic expression and return the numeric result.",
        args_model=CalculatorArgs,
        effect=ToolEffect.READ_ONLY,
        freshness=Freshness.STATIC,
        cache_scope=CacheScope.GLOBAL,
        public_data=True,
        normalize_whitespace_args=("expression",),
        retryable=True,
        timeout_seconds=2.0,
    )

    def run(self, args: CalculatorArgs) -> dict[str, Any]:
        return {"result": evaluate_expression(args.expression)}
