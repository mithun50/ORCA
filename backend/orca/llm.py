"""LLM access layer.

Deliberately optional. ORCA answers every supported query with deterministic
templates built from the retrieved evidence, and uses an LLM only to make the
wording natural and to arbitrate intent when the lexical router is unsure.

That ordering matters for a decision-support system: the numbers, thresholds and
safety verdicts are computed in code and are reproducible. The LLM never invents
a value, and if it is unavailable the answer is still correct, just blunter.

Two roles, two models, because the jobs are not alike:

* `LlmRole.AGENTIC` - intent arbitration and structured judgment. Wants a
  reasoning model. Defaults to GLM 5.2 on OpenRouter (`z-ai/glm-5.2`).
* `LlmRole.SYNTHESIS` - wording and Indian regional language translation. Wants a
  fast, fluent, multilingual model. Defaults to Gemini (`gemini-3.5-flash`).

A role whose provider has no key silently falls back to the generic provider
resolved from whatever credentials do exist, and then to templates. When
`ORCA_LLM_PROVIDER=auto` that generic order is:
  1. OpenRouter if `ORCA_OPENROUTER_API_KEY` is set
  2. Gemini if `ORCA_GEMINI_API_KEY` is set
  3. OpenAI-compatible endpoint if `ORCA_OPENAI_API_KEY` is set
  4. Ollama on `ORCA_OLLAMA_BASE` (local, no key, good for offline demos)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from .config import get_settings

log = logging.getLogger("orca.llm")


def _reasoning_param(model: str) -> dict[str, Any]:
    """How to keep a reasoning model from spending its whole budget thinking.

    Rewording a finished draft needs no deliberation, and letting a reasoning
    model deliberate anyway is not merely wasteful: it hits the token cap, the
    answer comes back truncated or empty, and the citation guard then rejects it.

    The two families need opposite settings, verified against OpenRouter on
    2026-09-22:

    * `z-ai/glm-5.2` honours `{"enabled": false}`. Given `{"effort": "low"}` it
      still burns the full budget and returns empty content.
    * `google/gemini-*` **rejects** `{"enabled": false}` outright with
      "Reasoning is mandatory for this endpoint and cannot be disabled", and
      needs `{"effort": "low"}`. Left unset it spends ~900 tokens thinking and
      truncates, at roughly 14x the cost of the same call with low effort.
    """
    if model.startswith("google/"):
        return {"effort": "low"}
    return {"enabled": False}


class LlmRole(str, Enum):
    """What the model is being asked to do, which decides which model runs."""

    #: intent arbitration, structured decisions, anything judgment shaped
    AGENTIC = "agentic"
    #: rewording and translating a draft that is already factually complete
    SYNTHESIS = "synthesis"

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

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "kn": "Kannada (ಕನ್ನಡ)",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
    "ml": "Malayalam (മലയാളം)",
    "hi": "Hindi (हिन्दी)",
    "mr": "Marathi (मराठी)",
    "gu": "Gujarati (ગુજરાતી)",
    "bn": "Bengali (বাংলা)",
}


def get_system_prompt(language: str = "en") -> str:
    lang_name = LANGUAGE_NAMES.get(language, "English")
    if language == "en":
        return SYSTEM_PROMPT
    return f"""You are ORCA, a marine information assistant for Indian coastal
users: fishermen, coastal authorities and maritime operators.

Hard rules:
- Use ONLY the evidence supplied in the prompt. Never invent a number, a place, a
  dataset name or an advisory.
- If the evidence is missing or stale, say so plainly.
- Keep the safety verdict exactly as given. Do not soften or escalate it.
- Name the source agency when you state a number (ISRO/MOSDAC, INCOIS, IMD, or
  "fallback model" for the non-official tier).
- Reply in {lang_name} naturally and respectfully so a local coastal fisherman or boat owner easily understands. Short paragraphs or a few bullets. No preamble, no headings.
- Keep maritime technical terms clear and transliterated or translated accurately.
- Speak plainly, as to a boat owner, not an oceanographer.
- Never use em dashes or en dashes. Use a plain hyphen.
"""


@dataclass
class LlmReply:
    text: str
    provider: str
    model: str
    role: str = ""
    #: token counts and cost, when the provider reports them. OpenRouter returns
    #: a real dollar figure; Gemini reports tokens only, so cost stays None and
    #: the UI says "not reported" rather than inventing an estimate.
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


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
        if configured in ("ggl", "gemini"):
            self._provider = "gemini"
            return self._provider
        if configured != "auto":
            self._provider = configured
            return self._provider

        # Auto-discovery priority: OpenRouter -> Gemini -> OpenAI -> Ollama
        if self.settings.effective_openrouter_api_key:
            self._provider = "openrouter"
        elif self.settings.effective_gemini_api_key:
            self._provider = "gemini"
        elif self.settings.openai_api_key:
            self._provider = "openai"
        elif await self._ollama_alive():
            self._provider = "ollama"
        else:
            self._provider = "none"
        log.info("LLM provider resolved to %s", self._provider)
        return self._provider

    def _has_credentials(self, provider: str) -> bool:
        """Can this provider actually be called right now?"""
        if provider == "openrouter":
            return bool(self.settings.effective_openrouter_api_key)
        if provider == "gemini":
            return bool(self.settings.effective_gemini_api_key)
        if provider == "openai":
            return bool(self.settings.openai_api_key)
        if provider == "ollama":
            return True  # probed lazily; a dead daemon just returns None
        return False

    def _default_model(self, provider: str) -> str:
        return {
            "openrouter": self.settings.openrouter_model,
            "gemini": self.settings.gemini_model,
            "openai": self.settings.llm_model or "gpt-4o-mini",
            "ollama": self.settings.llm_model,
        }.get(provider, "")

    async def resolve_role(self, role: LlmRole) -> tuple[str, str]:
        """Pick (provider, model) for a role, falling back when unconfigured."""
        if (self.settings.llm_provider or "auto").lower() == "none":
            return "none", ""
        if role is LlmRole.AGENTIC:
            wanted = (self.settings.llm_agentic_provider or "").lower()
            model = self.settings.llm_agentic_model
        else:
            wanted = (self.settings.llm_synthesis_provider or "").lower()
            model = self.settings.llm_synthesis_model
        if wanted in ("ggl",):
            wanted = "gemini"
        if wanted and wanted != "none" and self._has_credentials(wanted):
            return wanted, model or self._default_model(wanted)
        generic = await self.provider()
        if generic == "none":
            return "none", ""
        # keep the role's model only if it belongs to the provider we fell back to
        fallback_model = model if wanted == generic else self._default_model(generic)
        log.info(
            "LLM role %s wanted %s but fell back to %s",
            role.value,
            wanted or "unset",
            generic,
        )
        return generic, fallback_model or self._default_model(generic)

    async def routing(self) -> dict[str, dict[str, str]]:
        """What each role will actually call. Surfaced on /health."""
        agentic = await self.resolve_role(LlmRole.AGENTIC)
        synthesis = await self.resolve_role(LlmRole.SYNTHESIS)
        return {
            "agentic": {"provider": agentic[0], "model": agentic[1]},
            "synthesis": {"provider": synthesis[0], "model": synthesis[1]},
        }

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
        role: LlmRole = LlmRole.SYNTHESIS,
    ) -> LlmReply | None:
        provider, model = await self.resolve_role(role)
        if provider == "none":
            return None
        # Synthesis is a wording job over a finished draft. If the role resolved
        # to a reasoning model, tell it not to think, or it returns empty content.
        suppress = role is LlmRole.SYNTHESIS
        try:
            reply: LlmReply | None = None
            if provider == "openrouter":
                reply = await self._openrouter(
                    prompt,
                    system,
                    temperature,
                    max_tokens,
                    model,
                    suppress_reasoning=suppress,
                )
            elif provider == "gemini":
                reply = await self._gemini(
                    prompt, system, temperature, max_tokens, model
                )
            elif provider == "openai":
                reply = await self._openai(
                    prompt, system, temperature, max_tokens, model
                )
            elif provider == "ollama":
                reply = await self._ollama(
                    prompt, system, temperature, max_tokens, model
                )
            if reply is not None:
                reply.role = role.value
            return reply
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            log.warning("LLM call failed (%s/%s): %s", provider, model, exc)
            return None

    async def complete_json(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 400,
        role: LlmRole = LlmRole.AGENTIC,
    ) -> dict[str, Any] | None:
        reply = await self.complete(
            prompt,
            system=system or "Reply with a single JSON object and nothing else.",
            temperature=0.0,
            max_tokens=max_tokens,
            role=role,
        )
        if not reply:
            return None
        return extract_json_object(reply.text)

    # ------------------------------------------------------------ providers #

    async def _openrouter(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
        model: str = "",
        *,
        suppress_reasoning: bool = False,
    ) -> LlmReply | None:
        key = self.settings.effective_openrouter_api_key
        if not key:
            return None
        model = model or self.settings.openrouter_model or "z-ai/glm-5.2"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            # ask OpenRouter to report what this call actually cost
            "usage": {"include": True},
        }
        if suppress_reasoning:
            payload["reasoning"] = _reasoning_param(model)
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_s) as client:
            resp = await client.post(
                f"{self.settings.openrouter_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "HTTP-Referer": "https://github.com/mithun50/ORCA",
                    "X-Title": "ORCA Marine Intelligence",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            # Some endpoints refuse to have reasoning switched off. Rather than
            # maintaining a list of which, take the API at its word and retry.
            if resp.status_code == 400 and "reasoning is mandatory" in resp.text.lower():
                log.info(
                    "%s mandates reasoning; retrying at low effort", model
                )
                payload["reasoning"] = {"effort": "low"}
                resp = await client.post(
                    f"{self.settings.openrouter_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "HTTP-Referer": "https://github.com/mithun50/ORCA",
                        "X-Title": "ORCA Marine Intelligence",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            if resp.status_code != 200:
                log.warning("openrouter returned %s: %s", resp.status_code, resp.text[:200])
                return None
            body = resp.json()
            choices = body.get("choices", [])
            if not choices:
                return None
            # A reasoning model can return `content: null` when it spent its
            # budget on reasoning tokens, so the key exists but holds None.
            message = choices[0].get("message") or {}
            text = (message.get("content") or "").strip()
            if not text:
                reason = choices[0].get("finish_reason") or "no content"
                log.warning(
                    "openrouter %s returned no usable content (%s)", model, reason
                )
                return None
            usage = body.get("usage") or {}
            cost = usage.get("cost")
            return LlmReply(
                text,
                "openrouter",
                body.get("model", model),
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
            )

    async def _ollama(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
        model: str = "",
    ) -> LlmReply | None:
        model = model or self.settings.llm_model
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_s) as client:
            resp = await client.post(
                f"{self.settings.ollama_base}/api/chat",
                json={
                    "model": model,
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
            return LlmReply(text, "ollama", model) if text else None

    async def _openai(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
        model: str = "",
    ) -> LlmReply | None:
        model = model or self.settings.llm_model or "gpt-4o-mini"
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_s) as client:
            resp = await client.post(
                f"{self.settings.openai_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                json={
                    "model": model,
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
            message = (body.get("choices") or [{}])[0].get("message") or {}
            text = (message.get("content") or "").strip()
            return LlmReply(text, "openai", body.get("model", model)) if text else None

    async def _gemini(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
        model: str = "",
    ) -> LlmReply | None:
        key = self.settings.effective_gemini_api_key
        if not key:
            return None
        model = model or self.settings.gemini_model or "gemini-3.5-flash"
        if not model.startswith("gemini-"):
            model = self.settings.gemini_model or "gemini-3.5-flash"
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_s) as client:
            resp = await client.post(
                url,
                params={"key": key},
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
                log.warning("gemini returned %s: %s", resp.status_code, resp.text[:200])
                return None
            body = resp.json()
            candidates = body.get("candidates", [])
            if not candidates:
                return None
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
            if not text:
                return None
            meta = body.get("usageMetadata") or {}
            return LlmReply(
                text,
                "gemini",
                model,
                input_tokens=int(meta.get("promptTokenCount") or 0),
                output_tokens=int(meta.get("candidatesTokenCount") or 0),
                cost_usd=None,  # Gemini does not price the call in its response
            )


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a model reply, fences and prose included."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned[3:]
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


_client: LlmClient | None = None


def get_llm() -> LlmClient:
    global _client
    if _client is None:
        _client = LlmClient()
    return _client
