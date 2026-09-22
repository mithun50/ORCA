"""End-to-end integration tests for ORCA Voice and Chat API endpoints."""

from __future__ import annotations

import base64
import io
import wave
import pytest
import httpx

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from orca.main import app
from orca.services import get_services


def create_dummy_wav_bytes(duration_s: float = 0.2, sample_rate: int = 16000) -> bytes:
    """Helper to generate small dummy WAV bytes."""
    num_samples = int(sample_rate * duration_s)
    out_io = io.BytesIO()
    with wave.open(out_io, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * num_samples)
    return out_io.getvalue()


@pytest.mark.asyncio
async def test_health_endpoint():
    """Verify /health reports LLM, Jev, and Sarvam voice status."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "llm_provider" in data
        assert "jev_configured" in data
        assert "sarvam_voice_configured" in data


@pytest.mark.asyncio
async def test_api_voice_tts_endpoint():
    """Verify /api/voice/tts returns valid base64 audio."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/voice/tts", json={
            "text": "Weather is clear off Chennai.",
            "language": "en",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "audio_base64" in data
        assert len(data["audio_base64"]) > 50
        assert data["format"] == "wav"


@pytest.mark.asyncio
async def test_api_voice_stt_endpoint():
    """Verify /api/voice/stt transcribes uploaded audio."""
    wav_bytes = create_dummy_wav_bytes()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("deck.wav", wav_bytes, "audio/wav")}
        resp = await client.post("/api/voice/stt", files=files, data={"language": "kn"})
        assert resp.status_code == 200
        data = resp.json()
        assert "transcript" in data
        assert data["detected_locale"] == "kn"
        assert len(data["transcript"]) > 0


@pytest.mark.asyncio
async def test_chat_multilingual_kannada():
    """Verify /chat with language='kn' returns Kannada response with audio."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/chat", json={
            "message": "Is it safe to venture out off Rameswaram tomorrow morning?",
            "language": "kn",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        assert data["language"] == "kn"
        assert data["audio_base64"] is not None
        assert "risk" in data
        assert data["risk"]["band"] in ["safe", "caution", "unsafe", "unknown"]
        # Verify Jev decision was recorded
        assert data["risk"].get("jev_decision") is not None


@pytest.mark.asyncio
async def test_chat_voice_e2e():
    """Verify /chat/voice full audio pipeline."""
    wav_bytes = create_dummy_wav_bytes()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("deck.wav", wav_bytes, "audio/wav")}
        resp = await client.post("/chat/voice", files=files, data={"language": "ta"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert data["language"] == "ta"
        assert data["audio_base64"] is not None
        assert "evidence" in data
        assert "trace" in data


@pytest.mark.asyncio
async def test_chat_voice_returns_what_it_heard():
    """A voice turn must echo the transcript back.

    The user cannot sanity-check an answer whose question they never see, and a
    misheard place name is the likeliest failure on a noisy deck. Without this
    the UI had nothing to show for what was said.
    """
    wav_bytes = create_dummy_wav_bytes()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("deck.wav", wav_bytes, "audio/wav")}
        resp = await client.post("/chat/voice", files=files, data={"language": "en"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["transcript"], "the transcript must be returned to the caller"
        assert isinstance(data["transcript"], str)
        assert 0.0 < data["transcript_confidence"] <= 1.0


@pytest.mark.asyncio
async def test_chat_voice_rejects_unintelligible_audio_helpfully():
    """Silence must produce advice, not a bare error code."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # empty body: no file and no base64
        resp = await client.post("/chat/voice", data={"language": "en"})
        assert resp.status_code == 422
        assert "audio" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_text_chat_has_no_transcript():
    """A typed question has nothing to transcribe, so the field stays empty."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/chat", json={
            "message": "Is it safe to venture out off Rameswaram tomorrow morning?",
        })
        assert resp.status_code == 200
        assert resp.json()["transcript"] == ""
