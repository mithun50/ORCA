"""LLM access layer.

Deliberately optional. ORCA answers every supported query with deterministic
templates built from the retrieved evidence, and uses an LLM only to make the
wording natural and to arbitrate intent when the lexical router is unsure.

That ordering matters for a decision-support system: the numbers, thresholds and
safety verdicts are computed in code and are reproducible. The LLM never invents
a value, and if it is unavailable the answer is still correct, just blunter.

Providers, tried in this order when `ORCA_LLM_PROVIDER=auto`:
  1. Ollama on `ORCA_OLLAMA_BASE` (local, no key, good for offline demos)
  2. OpenAI-compatible endpoint if `ORCA_OPENAI_API_KEY` is set
  3. Gemini if `ORCA_GEMINI_API_KEY` is set
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from .config import get_settings

log = logging.getLogger("orca.llm")

SYSTEM_PROMPT = """You are ORCA, a marine information assistant for Indian coastal
users: fishermen, coastal authorities and maritime operators.

Hard rules:
- Use ONLY the evidence supplied in the prompt. Never invent a number, a place, a
  dataset name or an advisory.
- If the evidence is missing or stale, say so plainly.
- Keep the safety verdict exactly as given. Do not soften or escalate it.
- Name the source agency when you state a number (ISRO/MOSDAC, INCOIS, IMD, or
  "fallback model" for the non-official tier).
- Reply in English. Short paragraphs or a few bullets. No preamble, no headings.
- Speak plainly, as to a boat owner, not an oceanographer.
- Never use em dashes or en dashes. Use a plain hyphen.
"""


@dataclass
class LlmReply:
    text: str
    provider: str
    model: str


class LlmClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._provider: str | None = None
        self._checked = False

    # ------------------------------------------------------------ discovery #

    async def provider(self) -> str:
        if self._checked and self._provider is not None:
            return self._provider
        self._checked = True
        configured = (self.settings.llm_provider or "auto").lower()
        if configured == "none":
            self._provider = "none"
            return self._provider
        if configured != "auto":
            self._provider = configured
            return self._provider

        if await self._ollama_alive():
            self._provider = "ollama"
        elif self.settings.openai_api_key:
            self._provider = "openai"
        elif self.settings.gemini_api_key:
            self._provider = "gemini"
        else:
            self._provider = "none"
        log.info("LLM provider resolved to %s", self._provider)
        return self._provider

    async def _ollama_alive(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.settings.ollama_base}/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False

    @property
    def available_hint(self) -> str:
        return self._provider or "unchecked"

    # ----------------------------------------------------------- completion #

    async def complete(
        self,
        prompt: str,
        *,
        system: str = SYSTEM_PROMPT,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> LlmReply | None:
        provider = await self.provider()
        try:
            if provider == "ollama":
                return await self._ollama(prompt, system, temperature, max_tokens)
            if provider == "openai":
                return await self._openai(prompt, system, temperature, max_tokens)
            if provider == "gemini":
                return await self._gemini(prompt, system, temperature, max_tokens)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            log.warning("LLM call failed (%s): %s", provider, exc)
            return None
        return None

    async def complete_json(
        self, prompt: str, *, system: str = "", max_tokens: int = 400
    ) -> dict[str, Any] | None:
        reply = await self.complete(
            prompt,
            system=system or "Reply with a single JSON object and nothing else.",
            temperature=0.0,
            max_tokens=max_tokens,
        )
        if not reply:
            return None
        text = reply.text.strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except ValueError:
            return None

    # ------------------------------------------------------------ providers #

    async def _ollama(
        self, prompt: str, system: str, temperature: float, max_tokens: int
    ) -> LlmReply | None:
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_s) as client:
            resp = await client.post(
                f"{self.settings.ollama_base}/api/chat",
                json={
                    "model": self.settings.llm_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            if resp.status_code != 200:
                return None
            body = resp.json()
            text = (body.get("message") or {}).get("content", "").strip()
            return LlmReply(text, "ollama", self.settings.llm_model) if text else None

    async def _openai(
        self, prompt: str, system: str, temperature: float, max_tokens: int
    ) -> LlmReply | None:
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_s) as client:
            resp = await client.post(
                f"{self.settings.openai_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                json={
                    "model": self.settings.llm_model or "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            if resp.status_code != 200:
                log.warning("openai returned %s", resp.status_code)
                return None
            body = resp.json()
            text = body["choices"][0]["message"]["content"].strip()
            return LlmReply(text, "openai", body.get("model", "")) if text else None

    async def _gemini(
        self, prompt: str, system: str, temperature: float, max_tokens: int
    ) -> LlmReply | None:
        model = self.settings.llm_model or "gemini-2.0-flash"
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_s) as client:
            resp = await client.post(
                url,
                params={"key": self.settings.gemini_api_key},
                json={
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_tokens,
                    },
                },
            )
            if resp.status_code != 200:
                log.warning("gemini returned %s", resp.status_code)
                return None
            body = resp.json()
            parts = body["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts).strip()
            return LlmReply(text, "gemini", model) if text else None


_client: LlmClient | None = None


def get_llm() -> LlmClient:
    global _client
    if _client is None:
        _client = LlmClient()
    return _client
