"""Remove secrets from anything before it is stored or returned by the API.

Two layers:
1. Exact known secrets (the configured API key) are replaced wherever they
   appear, even inside longer text.
2. Anything shaped like a Google API key is replaced, in case a key that is
   not ours appears (for example, pasted into a task).

Applied recursively to strings inside dicts and lists.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

REDACTED = "[REDACTED]"
_KEY_PATTERNS = (re.compile(r"AIza[0-9A-Za-z_\-]{35}"),)


class Redactor:
    def __init__(self, secrets: Iterable[str] = ()) -> None:
        # Longest first, so a secret that contains another is fully removed.
        self._secrets = sorted({s for s in secrets if s and len(s) >= 8}, key=len, reverse=True)

    def text(self, value: str) -> str:
        for secret in self._secrets:
            value = value.replace(secret, REDACTED)
        for pattern in _KEY_PATTERNS:
            value = pattern.sub(REDACTED, value)
        return value

    def __call__(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {self(k) if isinstance(k, str) else k: self(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [self(v) for v in value]
        return value
