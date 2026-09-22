"""Tests for ORCA LLM Providers: OpenRouter, Gemini, and Ollama."""

from __future__ import annotations

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from orca.config import Settings, get_settings
from orca.llm import (
    LlmClient,
    LlmReply,
    LlmRole,
    LANGUAGE_NAMES,
    extract_json_object,
    get_system_prompt,
)


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
    monkeypatch.setenv("ORCA_GEMINI_API_KEY", "")
    monkeypatch.setenv("ORCA_OPENAI_API_KEY", "")

    get_settings.cache_clear()
    client = LlmClient()
    provider = await client.provider()
    assert provider == "openrouter"


@pytest.mark.asyncio
async def test_provider_resolution_gemini(monkeypatch):
    """Verify auto-resolution chooses Gemini when GEMINI_API_KEY is set."""
    monkeypatch.setenv("ORCA_LLM_PROVIDER", "auto")
    # blank, not delete: deleting would let a real .env value win again
    monkeypatch.setenv("ORCA_OPENROUTER_API_KEY", "")
    monkeypatch.setenv("ORCA_GEMINI_API_KEY", "AIzaSyMockGeminiKey")
    monkeypatch.setenv("ORCA_OPENAI_API_KEY", "")

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


# --------------------------------------------------------------------------- #
# role based routing: GLM 5.2 for agentic work, Gemini for synthesis
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_role_routing_agentic_and_synthesis(monkeypatch):
    """Agentic work goes to GLM on OpenRouter, synthesis to a Google model."""
    monkeypatch.setenv("ORCA_LLM_PROVIDER", "auto")
    monkeypatch.setenv("ORCA_OPENROUTER_API_KEY", "sk-or-v1-mock-key")
    monkeypatch.setenv("ORCA_GEMINI_API_KEY", "AIzaSyMockGeminiKey")
    monkeypatch.delenv("ORCA_OPENAI_API_KEY", raising=False)

    get_settings.cache_clear()
    client = LlmClient()

    agentic = await client.resolve_role(LlmRole.AGENTIC)
    synthesis = await client.resolve_role(LlmRole.SYNTHESIS)

    assert agentic == ("openrouter", "z-ai/glm-5.2")
    # one key covers both roles; synthesis is a Google model served by OpenRouter
    assert synthesis[0] == "openrouter"
    assert synthesis[1].startswith("google/gemini")

    routing = await client.routing()
    assert routing["agentic"]["model"] == "z-ai/glm-5.2"
    assert routing["synthesis"]["provider"] == "openrouter"
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_direct_gemini_provider_still_supported(monkeypatch):
    """Setting provider=gemini uses Google's own API instead of OpenRouter."""
    monkeypatch.setenv("ORCA_LLM_PROVIDER", "auto")
    monkeypatch.setenv("ORCA_OPENROUTER_API_KEY", "")
    monkeypatch.setenv("ORCA_GEMINI_API_KEY", "AIzaSyMockGeminiKey")
    monkeypatch.setenv("ORCA_LLM_SYNTHESIS_PROVIDER", "gemini")
    monkeypatch.setenv("ORCA_LLM_SYNTHESIS_MODEL", "gemini-3.8-flash")

    get_settings.cache_clear()
    client = LlmClient()
    provider, model = await client.resolve_role(LlmRole.SYNTHESIS)
    assert provider == "gemini"
    assert model == "gemini-3.8-flash"
    get_settings.cache_clear()


def test_reasoning_param_differs_by_model_family():
    """GLM wants reasoning off; Gemini on OpenRouter refuses to have it off."""
    from orca.llm import _reasoning_param

    assert _reasoning_param("z-ai/glm-5.2") == {"enabled": False}
    assert _reasoning_param("google/gemini-3.8-flash") == {"effort": "low"}


@pytest.mark.asyncio
async def test_role_falls_back_when_its_provider_has_no_key(monkeypatch):
    """With only a Gemini key, the agentic role must not silently no-op."""
    monkeypatch.setenv("ORCA_LLM_PROVIDER", "auto")
    monkeypatch.setenv("ORCA_OPENROUTER_API_KEY", "")
    monkeypatch.setenv("ORCA_GEMINI_API_KEY", "AIzaSyMockGeminiKey")
    monkeypatch.setenv("ORCA_OPENAI_API_KEY", "")

    get_settings.cache_clear()
    client = LlmClient()

    provider, model = await client.resolve_role(LlmRole.AGENTIC)
    assert provider == "gemini"
    assert model.startswith("gemini-")  # not the GLM slug, which gemini cannot serve
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_provider_none_disables_every_role(monkeypatch):
    monkeypatch.setenv("ORCA_LLM_PROVIDER", "none")
    get_settings.cache_clear()
    client = LlmClient()

    assert await client.resolve_role(LlmRole.AGENTIC) == ("none", "")
    assert await client.resolve_role(LlmRole.SYNTHESIS) == ("none", "")
    assert await client.complete("anything") is None
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_complete_tags_the_reply_with_its_role(monkeypatch):
    monkeypatch.setenv("ORCA_LLM_PROVIDER", "auto")
    monkeypatch.setenv("ORCA_OPENROUTER_API_KEY", "sk-or-v1-mock-key")
    monkeypatch.setenv("ORCA_GEMINI_API_KEY", "")
    monkeypatch.setenv("ORCA_OPENAI_API_KEY", "")
    get_settings.cache_clear()

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.json = MagicMock(
        return_value={
            "model": "z-ai/glm-5.2",
            "choices": [{"message": {"content": '{"intent": "safety_go_nogo"}'}}],
        }
    )

    client = LlmClient()
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)) as mock_post:
        reply = await client.complete("classify this", role=LlmRole.AGENTIC)
        assert reply is not None
        assert reply.role == "agentic"
        assert mock_post.call_args.kwargs["json"]["model"] == "z-ai/glm-5.2"

        parsed = await client.complete_json("classify this", role=LlmRole.AGENTIC)
        assert parsed == {"intent": "safety_go_nogo"}
    get_settings.cache_clear()


def test_extract_json_object_handles_fences_and_prose():
    assert extract_json_object('{"a": 1}') == {"a": 1}
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('Here you go: {"a": 1} hope that helps') == {"a": 1}
    assert extract_json_object("no json here") is None
    assert extract_json_object("") is None
    assert extract_json_object("[1, 2, 3]") is None  # arrays are not objects
