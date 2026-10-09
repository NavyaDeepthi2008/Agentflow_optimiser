"""An in-process fake of the Gemini REST API.

It plugs into the real ``google-genai`` SDK through an ``httpx`` mock transport,
so tests exercise the SDK's genuine request building and response parsing,
offline, with no API key and no cost.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import httpx

FAKE_KEY = "test-key-not-a-real-key-000"


def text_reply(text: str, *, usage: bool = True, thought: str | None = None) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    if thought:
        parts.append({"text": thought, "thought": True})
    parts.append({"text": text})
    return _candidate(parts, usage=usage)


def call_reply(
    *calls: tuple[str, dict[str, Any] | None, str | None],
    signature: bytes | None = None,
    usage: bool = True,
    text: str | None = None,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    if text:
        parts.append({"text": text})
    for i, (name, args, call_id) in enumerate(calls):
        fc: dict[str, Any] = {"name": name}
        if args is not None:
            fc["args"] = args
        if call_id is not None:
            fc["id"] = call_id
        part: dict[str, Any] = {"functionCall": fc}
        if signature and i == 0:
            part["thoughtSignature"] = base64.b64encode(signature).decode()
        parts.append(part)
    return _candidate(parts, usage=usage)


def _candidate(parts: list[dict[str, Any]], *, usage: bool, finish: str = "STOP") -> dict:
    body: dict[str, Any] = {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}],
        "modelVersion": "gemini-fake-001",
    }
    if usage:
        body["usageMetadata"] = {
            "promptTokenCount": 100,
            "candidatesTokenCount": 20,
            "thoughtsTokenCount": 5,
            "totalTokenCount": 125,
        }
    return body


def error_body(status: int, message: str, status_text: str = "ERROR") -> dict[str, Any]:
    return {"error": {"code": status, "message": message, "status": status_text}}


class FakeGeminiServer:
    """Replies are consumed in order. Each reply may be:

    * a dict -> HTTP 200 with that JSON body
    * ``(status, dict)`` -> that status and body
    * an ``Exception`` -> raised inside the transport (network failure)
    * ``("sleep", seconds)`` -> waits, then returns a text reply
    """

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.bodies: list[dict[str, Any]] = []
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []

    @property
    def calls(self) -> int:
        return len(self.bodies)

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        self.headers.append(dict(request.headers))
        self.bodies.append(json.loads(request.content))
        if not self.replies:
            return httpx.Response(500, json=error_body(500, "fake server has no reply left"))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, tuple) and reply[0] == "sleep":
            await asyncio.sleep(reply[1])
            return httpx.Response(200, json=text_reply("late"))
        if isinstance(reply, tuple):
            return httpx.Response(reply[0], json=reply[1])
        return httpx.Response(200, json=reply)

    def http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
