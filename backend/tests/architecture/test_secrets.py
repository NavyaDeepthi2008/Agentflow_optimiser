"""Guards against committing or hardcoding API keys."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
# Google API keys start with "AIza" followed by 35 URL-safe characters.
GOOGLE_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{35}")
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache"}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".example",
    ".cfg",
    ".ini",
    ".env",
    ".js",
    ".ts",
    ".tsx",
    ".html",
}


def _project_files():
    for path in REPO_ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts):
            continue
        if path.is_file() and (path.suffix in TEXT_SUFFIXES or path.name.startswith(".env")):
            yield path


def test_no_google_api_key_in_project_files():
    offenders = [
        str(p.relative_to(REPO_ROOT))
        for p in _project_files()
        if p.name != ".env" and GOOGLE_KEY_PATTERN.search(p.read_text(errors="ignore"))
    ]
    assert offenders == []


def test_no_hardcoded_key_assignment_in_app_code():
    app = REPO_ROOT / "backend" / "app"
    pattern = re.compile(r"api_key\s*=\s*['\"][^'\"]+['\"]")
    offenders = [str(p) for p in app.rglob("*.py") if pattern.search(p.read_text())]
    assert offenders == []


def test_env_example_has_no_real_value():
    example = REPO_ROOT / "backend" / ".env.example"
    assert example.is_file()
    for line in example.read_text().splitlines():
        if line.startswith("GEMINI_API_KEY="):
            assert line.strip() == "GEMINI_API_KEY="


def test_dotenv_is_git_ignored():
    gitignore = (REPO_ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in gitignore


def test_dotenv_is_not_tracked_by_git():
    try:
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is not available")
    assert not [f for f in tracked if Path(f).name == ".env"]
