"""Tests for ORCA LLM Providers: OpenRouter, Gemini, and Ollama."""

from __future__ import annotations

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from orca.config import Settings, get_settings
from orca.llm import LlmClient, LlmReply, LANGUAGE_NAMES, get_system_prompt


def test_language_system_prompts():
    """Verify system prompts for coastal Indian languages and English."""
    en_prompt = get_system_prompt("en")
    assert "ORCA" in en_prompt
    assert "Reply in English" in en_prompt

    kn_prompt = get_system_prompt("kn")
    assert "Kannada" in kn_prompt
    assert "ಕನ್ನಡ" in kn_prompt

    ta_prompt = get_system_prompt("ta")
    assert "Tamil" in ta_prompt
    assert "தமிழ்" in ta_prompt

    hi_prompt = get_system_prompt("hi")
    assert "Hindi" in hi_prompt


@pytest.mark.asyncio
async def test_provider_resolution_openrouter(monkeypatch):
    """Verify auto-resolution chooses OpenRouter when OPENROUTER_API_KEY is set."""
    monkeypatch.setenv("ORCA_LLM_PROVIDER", "auto")
    monkeypatch.setenv("ORCA_OPENROUTER_API_KEY", "sk-or-v1-mock-key")
    monkeypatch.delenv("ORCA_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ORCA_OPENAI_API_KEY", raising=False)

    get_settings.cache_clear()
    client = LlmClient()
    provider = await client.provider()
    assert provider == "openrouter"


@pytest.mark.asyncio
async def test_provider_resolution_gemini(monkeypatch):
    """Verify auto-resolution chooses Gemini when GEMINI_API_KEY is set."""
    monkeypatch.setenv("ORCA_LLM_PROVIDER", "auto")
    monkeypatch.delenv("ORCA_OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("ORCA_GEMINI_API_KEY", "AIzaSyMockGeminiKey")
    monkeypatch.delenv("ORCA_OPENAI_API_KEY", raising=False)

    get_settings.cache_clear()
    client = LlmClient()
    provider = await client.provider()
    assert provider == "gemini"


@pytest.mark.asyncio
async def test_provider_resolution_ggl_alias(monkeypatch):
    """Verify provider 'ggl' resolves to 'gemini'."""
    monkeypatch.setenv("ORCA_LLM_PROVIDER", "ggl")

    get_settings.cache_clear()
    client = LlmClient()
    provider = await client.provider()
    assert provider == "gemini"


@pytest.mark.asyncio
async def test_openrouter_complete_call():
    """Verify OpenRouter HTTP request structure."""
    client = LlmClient()
    client.settings.openrouter_api_key = "sk-or-test"
    client.settings.openrouter_model = "anthropic/claude-3.5-sonnet"

    mock_resp_data = {
        "id": "gen-123",
        "model": "anthropic/claude-3.5-sonnet",
        "choices": [
            {"message": {"content": "Sea conditions off Rameswaram are calm."}}
        ]
    }

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.json = MagicMock(return_value=mock_resp_data)

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)) as mock_post:
        reply = await client._openrouter("Hello", "System", 0.2, 500)
        assert reply is not None
        assert reply.provider == "openrouter"
        assert reply.text == "Sea conditions off Rameswaram are calm."
        assert reply.model == "anthropic/claude-3.5-sonnet"

        # Assert headers sent
        call_kwargs = mock_post.call_args.kwargs
        assert "Authorization" in call_kwargs["headers"]
        assert call_kwargs["headers"]["Authorization"] == "Bearer sk-or-test"
        assert "HTTP-Referer" in call_kwargs["headers"]


@pytest.mark.asyncio
async def test_gemini_complete_call():
    """Verify Gemini HTTP request structure."""
    client = LlmClient()
    client.settings.gemini_api_key = "AIzaSyMockKey"
    client.settings.gemini_model = "gemini-2.5-flash"

    mock_resp_data = {
        "candidates": [
            {"content": {"parts": [{"text": "Wave heights stay below 1.2 m."}]}}
        ]
    }

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.json = MagicMock(return_value=mock_resp_data)

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)) as mock_post:
        reply = await client._gemini("Hello", "System", 0.2, 500)
        assert reply is not None
        assert reply.provider == "gemini"
        assert reply.text == "Wave heights stay below 1.2 m."
        assert reply.model == "gemini-2.5-flash"
