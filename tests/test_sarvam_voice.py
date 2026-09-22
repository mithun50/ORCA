"""Tests for Sarvam AI Voice Engine: STT, TTS, and Acoustic Noise Filtering."""

from __future__ import annotations

import base64
import io
import wave
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from orca.voice import SarvamVoiceClient, AcousticFilter, LOCALE_MAP


def create_dummy_wav_bytes(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Helper to generate dummy 16-bit PCM WAV bytes."""
    num_samples = int(sample_rate * duration_s)
    out_io = io.BytesIO()
    with wave.open(out_io, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * num_samples)
    return out_io.getvalue()


def test_acoustic_filter_preserves_wav():
    """Verify acoustic filter processes 16-bit PCM WAV without corruption."""
    wav_bytes = create_dummy_wav_bytes()
    filtered = AcousticFilter.filter_wav_bytes(wav_bytes, cutoff_hz=120.0)
    assert len(filtered) > 44
    assert filtered.startswith(b"RIFF")


def test_locale_mapping():
    """Verify standard coastal Indian locales map to Sarvam BCP-47 tags."""
    assert LOCALE_MAP["kn"] == "kn-IN"
    assert LOCALE_MAP["ta"] == "ta-IN"
    assert LOCALE_MAP["te"] == "te-IN"
    assert LOCALE_MAP["ml"] == "ml-IN"
    assert LOCALE_MAP["hi"] == "hi-IN"
    assert LOCALE_MAP["en"] == "en-IN"


@pytest.mark.asyncio
async def test_stt_offline_fallback():
    """Verify STT returns a sensible fallback when SARVAM_API_KEY is not set."""
    client = SarvamVoiceClient()
    client.settings.sarvam_api_key = ""

    wav_bytes = create_dummy_wav_bytes()
    res = await client.speech_to_text(wav_bytes, language_code="kn")

    assert res.provider == "mock-saaras-offline"
    assert res.detected_locale == "kn"
    assert "ಸುರಕ್ಷಿತವೇ" in res.transcript or len(res.transcript) > 5


@pytest.mark.asyncio
async def test_tts_offline_fallback():
    """Verify TTS offline fallback generates valid base64-encoded WAV audio."""
    client = SarvamVoiceClient()
    client.settings.sarvam_api_key = ""

    res = await client.text_to_speech("Conditions look safe today.", language_code="en")

    assert res.provider == "mock-bulbul-offline"
    assert res.audio_format == "wav"
    assert len(res.audio_base64) > 100

    # Verify base64 decodes into valid RIFF WAV header
    decoded = base64.b64decode(res.audio_base64)
    assert decoded.startswith(b"RIFF")


@pytest.mark.asyncio
async def test_sarvam_stt_cloud_call():
    """Verify Sarvam STT cloud API invocation."""
    client = SarvamVoiceClient()
    client.settings.sarvam_api_key = "sarvam-sub-key"

    mock_resp_data = {
        "transcript": "ಸಮುದ್ರ ಪರಿಸ್ಥಿತಿ ಚೆನ್ನಾಗಿದೆ",
        "language_code": "kn-IN",
    }

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.json = MagicMock(return_value=mock_resp_data)

    wav_bytes = create_dummy_wav_bytes()
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)) as mock_post:
        res = await client.speech_to_text(wav_bytes, language_code="kn")

        assert res.provider == "sarvam-saaras"
        assert res.transcript == "ಸಮುದ್ರ ಪರಿಸ್ಥಿತಿ ಚೆನ್ನಾಗಿದೆ"
        assert res.detected_locale == "kn"

        call_kwargs = mock_post.call_args.kwargs
        assert "api-subscription-key" in call_kwargs["headers"]
        assert call_kwargs["headers"]["api-subscription-key"] == "sarvam-sub-key"


@pytest.mark.asyncio
async def test_sarvam_tts_cloud_call():
    """Verify Sarvam TTS cloud API invocation."""
    client = SarvamVoiceClient()
    client.settings.sarvam_api_key = "sarvam-sub-key"

    fake_b64 = base64.b64encode(create_dummy_wav_bytes()).decode("ascii")
    mock_resp_data = {
        "audios": [fake_b64]
    }

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.json = MagicMock(return_value=mock_resp_data)

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)) as mock_post:
        res = await client.text_to_speech("Caution off Rameswaram", language_code="ta", speaker="pavithra")

        assert res.provider == "sarvam-bulbul"
        assert res.audio_base64 == fake_b64
        assert res.target_language == "ta-IN"
        assert res.speaker == "pavithra"

        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["json"]["target_language_code"] == "ta-IN"
        assert call_kwargs["json"]["speaker"] == "pavithra"
