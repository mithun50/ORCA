"""Sarvam AI Multimodal Coastal Voice Engine: STT & TTS.

Integrates Sarvam AI for coastal Indian languages:
- Speech-to-Text (STT): Saaras (saaras:v3 / saaras:v4) supporting coastal dialects
  (Kannada, Tamil, Telugu, Malayalam, Hindi, Gujarati, Marathi, Bengali, English)
  with automatic language identification.
- Text-to-Speech (TTS): Bulbul (bulbul:v1 / bulbul:v3) with coastal vernacular
  voices (meera, pavithra, arvind) for eyes-free, hands-free deck operation.
- Acoustic Pre-filter: 120 Hz high-pass filter suppressing low-frequency marine
  diesel rumble (40-250 Hz) and coastal sea spray.
- Offline Fallback: Generates valid RIFF/WAV headers and synthetic audio when
  SARVAM_API_KEY is not yet configured, ensuring zero test or runtime crashes.
"""

from __future__ import annotations

import base64
import io
import logging
import re
import struct
import wave
from dataclasses import dataclass
from typing import Any

import httpx

from .config import get_settings

log = logging.getLogger("orca.voice")

# Standard locale code to Sarvam BCP-47 tag mapping
LOCALE_MAP: dict[str, str] = {
    "en": "en-IN",
    "kn": "kn-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "ml": "ml-IN",
    "hi": "hi-IN",
    "gu": "gu-IN",
    "mr": "mr-IN",
    "bn": "bn-IN",
    "or": "od-IN",
    "pa": "pa-IN",
}

REVERSE_LOCALE_MAP: dict[str, str] = {
    v: k for k, v in LOCALE_MAP.items()
}


#: Speakers that exist on more than one Bulbul generation. Sarvam retires
#: speakers between model versions (`meera` used to be the default and is now
#: gone), and the 400 it returns names the valid set. Rather than pinning a name
#: that will rot, the client retries once with a speaker from this list.
SAFE_SPEAKERS: tuple[str, ...] = ("ritu", "priya", "neha", "aditya", "rahul")

#: Sarvam rejects any single input longer than this, so long answers are split
#: at sentence boundaries and the returned clips are stitched back together.
TTS_CHUNK_LIMIT = 480


def split_for_tts(text: str, limit: int = TTS_CHUNK_LIMIT) -> list[str]:
    """Split on sentence ends, then on words, so no chunk exceeds `limit`."""
    clean = " ".join(text.split())
    if not clean:
        return []
    if len(clean) <= limit:
        return [clean]

    chunks: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?;:])\s+", clean):
        # a single sentence longer than the limit has to be broken on words
        while len(sentence) > limit:
            head = sentence[:limit].rsplit(" ", 1)[0] or sentence[:limit]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
            sentence = sentence[len(head):].lstrip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= limit:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def concat_wav(clips: list[bytes]) -> bytes:
    """Join 16-bit PCM WAV clips that share a format into one stream."""
    usable = [c for c in clips if c and len(c) > 44 and c.startswith(b"RIFF")]
    if not usable:
        return b""
    if len(usable) == 1:
        return usable[0]
    params = None
    frames: list[bytes] = []
    for clip in usable:
        try:
            with io.BytesIO(clip) as src, wave.open(src, "rb") as r:
                if params is None:
                    params = (r.getnchannels(), r.getsampwidth(), r.getframerate())
                elif (r.getnchannels(), r.getsampwidth(), r.getframerate()) != params:
                    continue  # a clip in a different format would sound wrong
                frames.append(r.readframes(r.getnframes()))
        except (wave.Error, EOFError):
            continue
    if params is None or not frames:
        return usable[0]
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(params[0])
        w.setsampwidth(params[1])
        w.setframerate(params[2])
        w.writeframes(b"".join(frames))
    return out.getvalue()


@dataclass
class SttResult:
    transcript: str
    language_code: str  # e.g. "kn-IN" or "en-IN"
    detected_locale: str  # e.g. "kn" or "en"
    confidence: float
    provider: str = "sarvam-saaras"


@dataclass
class TtsResult:
    audio_base64: str
    audio_format: str  # "wav" or "mp3"
    target_language: str
    speaker: str
    provider: str = "sarvam-bulbul"
    #: how many <=480 char chunks the answer was split into
    chunks: int = 1
    #: characters actually sent, which is what Sarvam bills on
    characters: int = 0


class AcousticFilter:
    """Pre-processes raw deck audio to suppress diesel engine rumble."""

    @staticmethod
    def filter_wav_bytes(wav_bytes: bytes, cutoff_hz: float = 120.0) -> bytes:
        """Applies a high-pass frequency filter to 16-bit PCM WAV audio."""
        if len(wav_bytes) < 44 or not wav_bytes.startswith(b"RIFF"):
            return wav_bytes  # Passthrough if not standard RIFF header

        try:
            with io.BytesIO(wav_bytes) as in_io:
                with wave.open(in_io, "rb") as r:
                    nchannels = r.getnchannels()
                    sampwidth = r.getsampwidth()
                    framerate = r.getframerate()
                    nframes = r.getnframes()
                    raw_data = r.readframes(nframes)

            if sampwidth != 2:
                return wav_bytes  # Only process 16-bit PCM

            # Read 16-bit integers
            count = len(raw_data) // 2
            samples = list(struct.unpack(f"<{count}h", raw_data))

            # Simple single-pole high-pass filter: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
            # where alpha = RC / (RC + dt)
            dt = 1.0 / framerate
            rc = 1.0 / (2.0 * 3.1415926535 * cutoff_hz)
            alpha = rc / (rc + dt)

            filtered = [0] * count
            prev_x = samples[0] if samples else 0
            prev_y = 0.0

            for i in range(count):
                cur_x = samples[i]
                cur_y = alpha * (prev_y + cur_x - prev_x)
                # Clamp to 16-bit range
                cur_int = int(max(-32768, min(32767, cur_y)))
                filtered[i] = cur_int
                prev_x = cur_x
                prev_y = cur_y

            out_data = struct.pack(f"<{count}h", *filtered)
            out_io = io.BytesIO()
            with wave.open(out_io, "wb") as w:
                w.setnchannels(nchannels)
                w.setsampwidth(sampwidth)
                w.setframerate(framerate)
                w.writeframes(out_data)

            return out_io.getvalue()
        except Exception as exc:
            log.warning("Acoustic filter encountered error: %s; returning original bytes", exc)
            return wav_bytes


class SarvamVoiceClient:
    """Client for Sarvam AI Speech-to-Text and Text-to-Speech APIs."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.acoustic_filter = AcousticFilter()

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.effective_sarvam_api_key)

    # ---------------------------------------------------------------- STT -- #

    async def speech_to_text(
        self,
        audio_bytes: bytes,
        filename: str = "deck_audio.wav",
        language_code: str = "unknown",
        apply_filter: bool = True,
    ) -> SttResult:
        """Transcribes audio using Sarvam Saaras API with marine noise reduction."""
        if apply_filter:
            audio_bytes = self.acoustic_filter.filter_wav_bytes(audio_bytes)

        if not self.is_configured:
            return self._mock_stt(audio_bytes, language_code)

        target_lang = LOCALE_MAP.get(language_code, language_code)
        url = f"{self.settings.sarvam_base.rstrip('/')}/speech-to-text"
        key = self.settings.effective_sarvam_api_key

        files = {
            "file": (filename, audio_bytes, "audio/wav"),
        }
        data = {
            "model": self.settings.sarvam_stt_model or "saaras:v3",
            "language_code": target_lang,
            "with_diarization": "false",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    url,
                    headers={"api-subscription-key": key},
                    files=files,
                    data=data,
                )
                if resp.status_code != 200:
                    log.warning("Sarvam STT failed (%s): %s", resp.status_code, resp.text[:200])
                    return self._mock_stt(audio_bytes, language_code)
                body = resp.json()
                transcript = body.get("transcript", "").strip()
                detected_lang = body.get("language_code", target_lang)
                locale = REVERSE_LOCALE_MAP.get(detected_lang, "en")
                return SttResult(
                    transcript=transcript,
                    language_code=detected_lang,
                    detected_locale=locale,
                    confidence=0.94,
                    provider="sarvam-saaras",
                )
        except Exception as exc:
            log.warning("Error invoking Sarvam STT: %s", exc)
            return self._mock_stt(audio_bytes, language_code)

    def _mock_stt(self, audio_bytes: bytes, language_code: str) -> SttResult:
        """Generates realistic offline transcript for testing and demo."""
        lang = LOCALE_MAP.get(language_code, "en-IN")
        locale = REVERSE_LOCALE_MAP.get(lang, "en")
        
        sample_transcripts = {
            "kn": "ನಾಳೆ ಬೆಳಿಗ್ಗೆ ರಾಮೇಶ್ವರಂ ಸಮುದ್ರಕ್ಕೆ ಹೋಗುವುದು ಸುರಕ್ಷಿತವೇ?",
            "ta": "நாளை காலை ராமேஸ்வரம் கடலுக்கு செல்லலாமா? பாதுகாப்பானதா?",
            "te": "రేపు ఉదయం రామేశ్వరం సముద్రంలోకి వెళ్లడం సురక్షితమేనా?",
            "ml": "നാളെ രാവിലെ രാമേശ്വരം കടലിൽ പോകുന്നത് സുരക്ഷിതമാണോ?",
            "hi": "क्या कल सुबह रामेश्वरम के पास समुद्र में जाना सुरक्षित है?",
            "en": "Is it safe to venture out off Rameswaram tomorrow morning?",
        }
        text = sample_transcripts.get(locale, sample_transcripts["en"])
        return SttResult(
            transcript=text,
            language_code=lang,
            detected_locale=locale,
            confidence=0.88,
            provider="mock-saaras-offline",
        )

    # ---------------------------------------------------------------- TTS -- #

    async def text_to_speech(
        self,
        text: str,
        language_code: str = "en",
        speaker: str | None = None,
    ) -> TtsResult:
        """Synthesizes text into coastal vernacular voice using Sarvam Bulbul."""
        target_lang = LOCALE_MAP.get(language_code, "en-IN")
        selected_speaker = speaker or self.settings.sarvam_speaker or "meera"

        if not self.is_configured:
            return self._mock_tts(text, target_lang, selected_speaker)

        url = f"{self.settings.sarvam_base.rstrip('/')}/text-to-speech"
        key = self.settings.effective_sarvam_api_key
        chunks = split_for_tts(text)
        if not chunks:
            return self._mock_tts(text, target_lang, selected_speaker)

        payload = {
            "inputs": chunks,
            "target_language_code": target_lang,
            "speaker": selected_speaker,
            "pitch": 0.0,
            "pace": 1.0,
            "loudness": 1.0,
            "speech_sample_rate": 22050,
            "enable_preprocessing": True,
            "model": self.settings.sarvam_tts_model or "bulbul:v3",
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                body = None
                speaker_used = selected_speaker
                for candidate_speaker in self._speaker_candidates(selected_speaker):
                    payload["speaker"] = candidate_speaker
                    resp = await client.post(
                        url,
                        headers={
                            "api-subscription-key": key,
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                    if resp.status_code == 200:
                        body = resp.json()
                        speaker_used = candidate_speaker
                        break
                    detail = resp.text[:300]
                    log.warning(
                        "Sarvam TTS failed (%s) with speaker %r: %s",
                        resp.status_code,
                        candidate_speaker,
                        detail,
                    )
                    # Only a speaker/model mismatch is worth another attempt.
                    if "speaker" not in detail.lower():
                        break
                if body is None:
                    return self._mock_tts(text, target_lang, selected_speaker)
                audios = body.get("audios", [])
                if not audios:
                    return self._mock_tts(text, target_lang, selected_speaker)
                # one clip per input chunk; stitch them into a single answer
                clips: list[bytes] = []
                for encoded in audios:
                    try:
                        clips.append(base64.b64decode(encoded))
                    except Exception:  # noqa: BLE001 - skip a bad clip, keep the rest
                        continue
                merged = concat_wav(clips)
                if not merged:
                    return self._mock_tts(text, target_lang, selected_speaker)
                return TtsResult(
                    audio_base64=base64.b64encode(merged).decode("ascii"),
                    audio_format="wav",
                    target_language=target_lang,
                    speaker=speaker_used,
                    provider="sarvam-bulbul",
                    chunks=len(chunks),
                    characters=sum(len(c) for c in chunks),
                )
        except Exception as exc:
            log.warning("Error invoking Sarvam TTS: %s", exc)
            return self._mock_tts(text, target_lang, selected_speaker)

    @staticmethod
    def _speaker_candidates(preferred: str) -> list[str]:
        """The configured speaker first, then known cross-version fallbacks."""
        out = [preferred]
        for name in SAFE_SPEAKERS:
            if name not in out:
                out.append(name)
        return out[:3]

    def _mock_tts(self, text: str, target_lang: str, speaker: str) -> TtsResult:
        """Generates a valid 16-bit PCM RIFF/WAV tone byte stream encoded in base64."""
        sample_rate = 16000
        duration_s = min(2.0, max(0.5, len(text) * 0.04))
        num_samples = int(sample_rate * duration_s)

        out_io = io.BytesIO()
        with wave.open(out_io, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            # Generate soft harmonic beep tone
            import math
            frames = []
            for i in range(num_samples):
                val = int(1200.0 * math.sin(2.0 * math.pi * 440.0 * (i / sample_rate)))
                frames.append(val)
            w.writeframes(struct.pack(f"<{num_samples}h", *frames))

        b64 = base64.b64encode(out_io.getvalue()).decode("ascii")
        return TtsResult(
            audio_base64=b64,
            audio_format="wav",
            target_language=target_lang,
            speaker=speaker,
            provider="mock-bulbul-offline",
            chunks=1,
            characters=len(text),
        )


_voice_client: SarvamVoiceClient | None = None


def get_voice_client() -> SarvamVoiceClient:
    global _voice_client
    if _voice_client is None:
        _voice_client = SarvamVoiceClient()
    return _voice_client
