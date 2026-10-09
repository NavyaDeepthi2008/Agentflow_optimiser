"""Settings loaded from environment variables (and optionally a local ``.env`` file).

Secrets:
* ``GEMINI_API_KEY`` is held as a ``SecretStr``: printing or logging the settings
  shows ``**********`` instead of the key.
* Real environment variables win over values in ``.env``.
* ``.env`` is listed in ``.gitignore``; only ``.env.example`` (no real key) is
  committed.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

THINKING_LEVELS = ("minimal", "low", "medium", "high")


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    gemini_api_key: SecretStr | None = None
    gemini_model: str = Field(default="gemini-3.5-flash", min_length=1)
    # None = use the provider's default. Google recommends the default (1.0) for
    # Gemini 3 models and warns lower values may cause looping.
    gemini_temperature: float | None = Field(default=None, ge=0, le=2)
    gemini_thinking_level: str | None = None
    gemini_max_output_tokens: int = Field(default=4096, ge=64, le=65536)
    llm_timeout_seconds: float = Field(default=45.0, gt=0, le=600)

    agent_max_steps: int = Field(default=12, ge=1, le=100)
    agent_max_tool_requests: int = Field(default=20, ge=1, le=500)
    agent_run_timeout_seconds: float = Field(default=90.0, gt=0, le=3600)

    # Persistence and API
    database_url: str = Field(default="sqlite:///./agentflow.db", min_length=1)
    max_concurrent_runs: int = Field(default=2, ge=1, le=32)
    max_sessions: int = Field(default=100, ge=1, le=10_000)
    # Optional JSON file with real, dated prices (see pricing.example.json).
    # Without it, cost estimates are reported as unavailable.
    pricing_file: str | None = None
    # Browser origins allowed to call the API (local dashboard during development).
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @field_validator("cors_origins")
    @classmethod
    def _no_wildcard_origin(cls, value: str) -> str:
        origins = [o.strip() for o in value.split(",") if o.strip()]
        for origin in origins:
            if origin == "*" or not origin.startswith(("http://", "https://")):
                raise ValueError(
                    "CORS origins must be explicit http(s) origins; '*' is not allowed"
                )
        return ",".join(origins)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o for o in self.cors_origins.split(",") if o]

    def secret_values(self) -> list[str]:
        """Every secret value, so storage and logs can scrub them."""
        return [self.gemini_api_key.get_secret_value()] if self.gemini_api_key else []

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def _blank_key_is_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("gemini_thinking_level", mode="before")
    @classmethod
    def _check_thinking_level(cls, value: object) -> object:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        if not isinstance(value, str) or value.strip().lower() not in THINKING_LEVELS:
            raise ValueError(f"must be one of {', '.join(THINKING_LEVELS)}")
        return value.strip().lower()

    @property
    def has_gemini_key(self) -> bool:
        return self.gemini_api_key is not None


_ENV_FIELDS = {
    "GEMINI_API_KEY": "gemini_api_key",
    "GEMINI_MODEL": "gemini_model",
    "GEMINI_TEMPERATURE": "gemini_temperature",
    "GEMINI_THINKING_LEVEL": "gemini_thinking_level",
    "GEMINI_MAX_OUTPUT_TOKENS": "gemini_max_output_tokens",
    "LLM_TIMEOUT_SECONDS": "llm_timeout_seconds",
    "AGENT_MAX_STEPS": "agent_max_steps",
    "AGENT_MAX_TOOL_REQUESTS": "agent_max_tool_requests",
    "AGENT_RUN_TIMEOUT_SECONDS": "agent_run_timeout_seconds",
    "DATABASE_URL": "database_url",
    "MAX_CONCURRENT_RUNS": "max_concurrent_runs",
    "MAX_SESSIONS": "max_sessions",
    "PRICING_FILE": "pricing_file",
    "CORS_ORIGINS": "cors_origins",
}


def load_settings(
    environ: Mapping[str, str] | None = None,
    env_file: str | Path | None = ".env",
) -> Settings:
    """Build settings from ``environ`` (default: the process environment).

    Values from ``env_file`` are used only for variables not already set in the
    environment. Pass ``env_file=None`` to ignore any ``.env`` file (tests do).
    """
    values: dict[str, str] = {}
    if env_file is not None and Path(env_file).is_file():
        values.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
    values.update(os.environ if environ is None else environ)
    kwargs = {
        field: values[env]
        for env, field in _ENV_FIELDS.items()
        if env in values and values[env].strip() != ""
    }
    if "GEMINI_API_KEY" in values:  # keep an explicitly blank key as "missing"
        kwargs["gemini_api_key"] = values["GEMINI_API_KEY"]
    return Settings(**kwargs)
