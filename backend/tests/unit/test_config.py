"""Settings: secure key handling and environment loading."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings, load_settings

KEY = "unit-test-secret-value-123"


def test_defaults_without_key():
    settings = load_settings(environ={}, env_file=None)
    assert settings.gemini_api_key is None
    assert not settings.has_gemini_key
    assert settings.gemini_temperature is None  # provider default
    assert settings.agent_max_steps == 12


def test_key_is_read_and_hidden():
    settings = load_settings(environ={"GEMINI_API_KEY": f"  {KEY}  "}, env_file=None)
    assert settings.gemini_api_key.get_secret_value() == KEY  # whitespace stripped
    for text in (repr(settings), str(settings), settings.model_dump_json()):
        assert KEY not in text


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_blank_key_counts_as_missing(blank):
    assert load_settings(environ={"GEMINI_API_KEY": blank}, env_file=None).gemini_api_key is None


def test_env_file_is_used_but_real_environment_wins(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"GEMINI_API_KEY={KEY}\nGEMINI_MODEL=from-file\nAGENT_MAX_STEPS=5\n")
    settings = load_settings(environ={"GEMINI_MODEL": "from-env"}, env_file=env_file)
    assert settings.gemini_api_key.get_secret_value() == KEY
    assert settings.gemini_model == "from-env"
    assert settings.agent_max_steps == 5


def test_missing_env_file_is_fine(tmp_path):
    assert load_settings(environ={}, env_file=tmp_path / "nope.env").gemini_api_key is None


def test_numeric_and_choice_values_are_validated():
    settings = load_settings(
        environ={"GEMINI_TEMPERATURE": "0.5", "GEMINI_THINKING_LEVEL": "LOW"}, env_file=None
    )
    assert settings.gemini_temperature == 0.5
    assert settings.gemini_thinking_level == "low"
    with pytest.raises(ValidationError):
        load_settings(environ={"GEMINI_THINKING_LEVEL": "extreme"}, env_file=None)
    with pytest.raises(ValidationError):
        load_settings(environ={"AGENT_MAX_STEPS": "0"}, env_file=None)
    with pytest.raises(ValidationError):
        Settings(gemini_temperature=5)
