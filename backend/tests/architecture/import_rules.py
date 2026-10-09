"""Static checks that keep tool code reachable only through the ToolGateway.

The checker reads every Python file under ``app/`` as a syntax tree (it does not
run the code) and reports rule violations.

Rules
-----
R1  Tool implementations (``app.tools.builtin`` / ``app.tools.test_only``) may be
    imported only by modules inside ``app/tools/``.
R2  Consumer packages (agent, llm, api, evaluation, learning) may not import
    ``app.tools`` at all - they must use the gateway's public methods.
R3  The private handler accessor ``_handler_for`` and the gateway's private
    ``_registry`` may be used only inside ``app/gateway/gateway.py``. This also
    catches string tricks such as ``getattr(x, "_handler_for")``.
R4  No dynamic imports (``importlib`` / ``__import__``) anywhere in ``app/``,
    because they would let code slip past R1 and R2.
R5  LLM clients (``app.llm``) may not import the gateway or the agent: an LLM
    client can describe tool calls but can never execute them.
R6  Third-party isolation: the Google Gen AI SDK (``google``) may be imported
    only by ``app.llm.gemini``; ``dotenv`` only by ``app.config``; SQLAlchemy
    only by ``app.storage`` (and the API's health check); Alembic only by
    ``app.storage.db``; FastAPI/Starlette only by ``app.api``.
R7  The agent (``app.agent``) may not import a specific provider client; it works
    with any ``LLMClient``.
R8  The optimizer (``app.optimizer``) may not import the gateway, agent or LLM
    code: it only decides and stores; it can never execute a tool itself.
R9  The learning engine (``app.learning``) may use only the gateway's data
    shapes (``app.gateway.models``) and must not touch the cache store
    (``app.optimizer.cache_store``): it proposes and validates rules, and its
    only way to affect reuse is a checked ``ToolOverride`` that a person
    approves.

Narrow, documented exception to R2:
* ``app.evaluation`` may import ``app.tools.mock_world`` (and nothing else from
  ``app.tools``). The grader must read the mock world's final state (outbox,
  records) independently of the agent, and the runner injects faults into it.
  ``MockWorld`` is data only; it has no way to execute a tool.
* ``app.services.runs`` may import ``app.tools.mock_world`` too: each session
  owns its mock data, and demo scenarios inject faults into it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

TOOL_IMPL_PREFIXES = ("app.tools.builtin", "app.tools.test_only")
CONSUMER_PACKAGES = (
    "app.agent",
    "app.llm",
    "app.api",
    "app.evaluation",
    "app.learning",
    "app.cli",
    "app.services",
    "app.storage",
)
PRIVATE_NAMES = ("_handler_for", "_registry")
GATEWAY_MODULE = "app.gateway.gateway"
REGISTRY_MODULE = "app.tools.registry"
# Third-party package -> the only modules (or packages) allowed to import it.
ISOLATED_THIRD_PARTY = {
    "google": ("app.llm.gemini",),
    "dotenv": ("app.config",),
    "sqlalchemy": ("app.storage", "app.api.main"),
    "alembic": ("app.storage.db",),
    "fastapi": ("app.api",),
    "starlette": ("app.api",),
}
PROVIDER_CLIENT_MODULES = ("app.llm.gemini",)
# (importing package, exact module it may import despite R2)
R2_EXCEPTIONS = {
    ("app.evaluation", "app.tools.mock_world"),
    ("app.services.runs", "app.tools.mock_world"),
}


def _r2_allowed(module: str, imported: str) -> bool:
    for package, allowed in R2_EXCEPTIONS:
        if _in(module, (package,)) and (imported == allowed or imported.startswith(allowed + ".")):
            return True
    return False


@dataclass(frozen=True)
class Violation:
    rule: str
    module: str
    line: int
    detail: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.module}:{self.line} {self.detail}"


def module_name(app_root: Path, path: Path) -> str:
    rel = path.relative_to(app_root.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative(module: str, is_package: bool, level: int, target: str | None) -> str:
    base = module.split(".")
    if not is_package:
        base = base[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join([*base, target] if target else base)


def _imported_modules(tree: ast.AST, module: str, is_package: bool):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = _resolve_relative(module, is_package, node.level, node.module)
            else:
                base = node.module or ""
            yield node.lineno, base
            for alias in node.names:  # "from app.tools import builtin"
                yield node.lineno, f"{base}.{alias.name}"


def _in(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def find_violations(app_root: Path) -> list[Violation]:
    violations: list[Violation] = []
    for path in sorted(app_root.rglob("*.py")):
        module = module_name(app_root, path)
        is_package = path.name == "__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        for line, imported in _imported_modules(tree, module, is_package):
            if _in(imported, TOOL_IMPL_PREFIXES) and not _in(module, ("app.tools",)):
                violations.append(Violation("R1", module, line, f"imports {imported}"))
            if (
                _in(module, CONSUMER_PACKAGES)
                and _in(imported, ("app.tools",))
                and not _r2_allowed(module, imported)
            ):
                violations.append(Violation("R2", module, line, f"imports {imported}"))
            if imported.split(".")[0] == "importlib":
                violations.append(Violation("R4", module, line, "uses importlib"))
            if _in(module, ("app.llm",)) and _in(imported, ("app.gateway", "app.agent")):
                violations.append(Violation("R5", module, line, f"imports {imported}"))
            top = imported.split(".")[0]
            if top in ISOLATED_THIRD_PARTY and not _in(module, ISOLATED_THIRD_PARTY[top]):
                violations.append(Violation("R6", module, line, f"imports {imported}"))
            if _in(module, ("app.agent",)) and _in(imported, PROVIDER_CLIENT_MODULES):
                violations.append(Violation("R7", module, line, f"imports {imported}"))
            if _in(module, ("app.optimizer",)) and _in(
                imported, ("app.gateway", "app.agent", "app.llm")
            ):
                violations.append(Violation("R8", module, line, f"imports {imported}"))
            if _in(module, ("app.learning",)) and (
                (_in(imported, ("app.gateway",)) and not _in(imported, ("app.gateway.models",)))
                or _in(imported, ("app.optimizer.cache_store",))
            ):
                violations.append(Violation("R9", module, line, f"imports {imported}"))

        allowed_private = module in (GATEWAY_MODULE, REGISTRY_MODULE)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in PRIVATE_NAMES:
                if module != GATEWAY_MODULE:
                    violations.append(
                        Violation("R3", module, node.lineno, f"accesses .{node.attr}")
                    )
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in PRIVATE_NAMES
                and not allowed_private
            ):
                violations.append(
                    Violation("R3", module, node.lineno, f"string reference {node.value!r}")
                )
            elif isinstance(node, ast.Name) and node.id == "__import__":
                violations.append(Violation("R4", module, node.lineno, "uses __import__"))
    return violations
