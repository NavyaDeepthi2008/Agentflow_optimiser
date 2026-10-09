from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import Settings
from app.llm.base import LLMRequest, LLMResponse


class GateLLM:
    """An LLM that waits until the test releases it (to observe 'running')."""

    provider = "gate"
    model_id = "gate-model"

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        self.started.set()
        while not self.release.is_set():
            await asyncio.sleep(0.01)
        return LLMResponse(text="released", model_id=self.model_id, finish_reason="STOP")


def scripted_body(*replies: dict[str, Any], **extra: Any) -> dict[str, Any]:
    body = {"task": "What does Orbit cost?", "llm": "scripted", "scripted_replies": list(replies)}
    body.update(extra)
    return body


def calls(*pairs: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    return {"tool_calls": [{"name": n, "args": a} for n, a in pairs]}


def final(text: str) -> dict[str, Any]:
    return {"text": text}


def wait_finished(client: TestClient, run_id: str, timeout: float = 10.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = client.get(f"/runs/{run_id}").json()
        if run["finished"]:
            return run
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish in {timeout}s")


def wait_status(client: TestClient, run_id: str, status: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.get(f"/runs/{run_id}").json()["status"] == status:
            return
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} never reached {status}")


@pytest.fixture
def db_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path}/api.db"


@pytest.fixture
def make_client(db_url) -> Callable[..., TestClient]:
    def _make(gemini_factory=None, **settings: Any) -> TestClient:
        settings.setdefault("database_url", db_url)
        return TestClient(create_app(Settings(**settings), gemini_factory=gemini_factory))

    return _make


@pytest.fixture
def client(make_client):
    with make_client() as c:
        yield c
