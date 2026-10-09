"""Write the API's OpenAPI schema to backend/openapi.json.

The frontend generates its TypeScript API types from this file
(``npm run gen:api``), and a backend test fails if the file is out of date.

    .venv/bin/python scripts/export_openapi.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.main import create_app  # noqa: E402
from app.config import Settings  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "openapi.json"


def current_schema() -> dict:
    return create_app(Settings()).openapi()


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(current_schema(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUTPUT}")
