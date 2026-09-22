"""Shared test setup.

The tests must be hermetic. `Settings` reads `<repo>/.env`, so on a developer
machine that has real credentials in it the suite would otherwise resolve live
providers, change its own expectations, and in the worst case make paid calls.

Process environment variables take priority over `env_file` in
pydantic-settings, so setting each credential to an empty string neutralises
whatever the `.env` says. Individual tests then opt back in to the specific
provider they are exercising with `monkeypatch.setenv`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from orca.config import get_settings  # noqa: E402

#: every credential and provider selector that could leak in from a real .env
NEUTRALISED = (
    "ORCA_OPENROUTER_API_KEY",
    "ORCA_GEMINI_API_KEY",
    "ORCA_OPENAI_API_KEY",
    "ORCA_SARVAM_API_KEY",
    "ORCA_JEV_API_KEY",
    "ORCA_IMD_API_KEY",
    "ORCA_N8N_BASE",
    "ORCA_N8N_WEBHOOK_TOKEN",
    # unprefixed aliases the effective_* properties also consult
    "OPENROUTER_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "SARVAM_API_KEY",
    "JEV_API_KEY",
)


@pytest.fixture(autouse=True)
def hermetic_settings(monkeypatch: pytest.MonkeyPatch):
    """Blank every credential before each test, and reset the settings cache."""
    for name in NEUTRALISED:
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("ORCA_LLM_PROVIDER", "auto")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
