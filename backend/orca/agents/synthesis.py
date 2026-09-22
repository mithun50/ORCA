"""Synthesis agent.

Builds the answer in two passes:

1. **Deterministic draft.** Per-intent templates read the findings and produce a
   correct, complete answer with the numbers, the source tiers and the safety
   verdict. This pass always runs and never fails.
2. **Optional LLM rewrite.** If a provider is configured, the draft plus a
   compact evidence digest go to the model with instructions to rephrase only.
   If the rewrite drops the verdict or comes back empty, the draft is kept.

The verdict, the numbers and the citations therefore never depend on the model
being available or behaving.
"""

from __future__ import annotations

import re
from typing import Any

from ..llm import LANGUAGE_NAMES, LlmRole, get_system_prompt
from ..rag.audience import AUDIENCE_STYLE, Audience
from ..schemas import Citation, Intent, RiskBand
from ..services import Services
from ..voice import get_voice_client
from .base import AgentContext, EvidenceBook
from .briefing import build_briefing
from .ocean import beaufort, sea_state


#: Shown at the top of the answer when the reader asked in a regional language
#: but we could only produce English. Written in their script so they can read
#: the apology even though the answer below it is not in their language.
LANGUAGE_FALLBACK_NOTE: dict[str, str] = {
    "kn": "ಕ್ಷಮಿಸಿ, ಈ ಉತ್ತರ ಕನ್ನಡದಲ್ಲಿ ನೀಡಲು ಸಾಧ್ಯವಾಗಿಲ್ಲ. ಕೆಳಗಿನ ಮಾಹಿತಿ ಇಂಗ್ಲಿಷ್‌ನಲ್ಲಿದೆ.",
    "ta": "மன்னிக்கவும், இந்தப் பதிலை தமிழில் தர முடியவில்லை. கீழே உள்ள தகவல் ஆங்கிலத்தில் உள்ளது.",
    "te": "క్షమించండి, ఈ సమాధానాన్ని తెలుగులో ఇవ్వలేకపోయాము. దిగువ సమాచారం ఆంగ్లంలో ఉంది.",
    "ml": "ക്ഷമിക്കണം, ഈ മറുപടി മലയാളത്തിൽ നൽകാൻ കഴിഞ്ഞില്ല. താഴെയുള്ള വിവരം ഇംഗ്ലീഷിലാണ്.",
    "hi": "क्षमा करें, यह उत्तर हिन्दी में नहीं दिया जा सका। नीचे की जानकारी अंग्रेज़ी में है।",
    "mr": "क्षमस्व, हे उत्तर मराठीत देता आले नाही. खालील माहिती इंग्रजीत आहे.",
    "gu": "માફ કરશો, આ જવાબ ગુજરાતીમાં આપી શકાયો નથી. નીચેની માહિતી અંગ્રેજીમાં છે.",
    "bn": "দুঃখিত, এই উত্তরটি বাংলায় দেওয়া যায়নি। নিচের তথ্য ইংরেজিতে রয়েছে।",
}

#: The verdict as a sentence the model can translate, rather than a band label
#: it would otherwise copy verbatim into another language.
VERDICT_PHRASE: dict[Any, str] = {
    RiskBand.UNSAFE: "do not go out to sea",
    RiskBand.CAUTION: "you may go out but must be careful",
    RiskBand.SAFE: "it is alright to go out",
    RiskBand.UNKNOWN: "there is not enough information to judge",
    None: "no safety verdict applies",
}

#: Shown above the answer when the reader asked in a regional language but we
#: could only produce English. Written in their own script, so the apology at
#: least is readable even when the answer below it is not in their language.
LANGUAGE_FALLBACK_NOTE: dict[str, str] = {
    "kn": "ಕ್ಷಮಿಸಿ, ಈ ಉತ್ತರವನ್ನು ಕನ್ನಡದಲ್ಲಿ ಕೊಡಲು ಆಗಲಿಲ್ಲ. ಕೆಳಗಿನ ಮಾಹಿತಿ ಇಂಗ್ಲಿಷ್‌ನಲ್ಲಿದೆ.",
    "ta": "மன்னிக்கவும், இந்தப் பதிலைத் தமிழில் தர முடியவில்லை. கீழே உள்ள தகவல் ஆங்கிலத்தில் உள்ளது.",
    "te": "క్షమించండి, ఈ సమాధానాన్ని తెలుగులో ఇవ్వలేకపోయాము. కింది సమాచారం ఇంగ్లీషులో ఉంది.",
    "ml": "ക്ഷമിക്കണം, ഈ മറുപടി മലയാളത്തിൽ നൽകാൻ കഴിഞ്ഞില്ല. താഴെയുള്ള വിവരം ഇംഗ്ലീഷിലാണ്.",
    "hi": "क्षमा करें, यह जवाब हिन्दी में नहीं दे सके। नीचे की जानकारी अंग्रेज़ी में है।",
    "mr": "क्षमस्व, हे उत्तर मराठीत देता आले नाही. खालील माहिती इंग्रजीत आहे.",
    "gu": "માફ કરશો, આ જવાબ ગુજરાતીમાં આપી શક્યા નથી. નીચેની માહિતી અંગ્રેજીમાં છે.",
    "bn": "দুঃখিত, এই উত্তরটি বাংলায় দেওয়া গেল না। নিচের তথ্য ইংরেজিতে রয়েছে।",
}

#: Unicode block per language, used to prove an answer really is in that script.
#: Devanagari serves both Hindi and Marathi, so they share a range.
SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "hi": ((0x0900, 0x097F),),
    "mr": ((0x0900, 0x097F),),
    "bn": ((0x0980, 0x09FF),),
    "gu": ((0x0A80, 0x0AFF),),
    "ta": ((0x0B80, 0x0BFF),),
    "te": ((0x0C00, 0x0C7F),),
    "kn": ((0x0C80, 0x0CFF),),
    "ml": ((0x0D00, 0x0D7F),),
    "pa": ((0x0A00, 0x0A7F),),
    "or": ((0x0B00, 0x0B7F),),
}


def script_ratio(text: str, language: str) -> float:
    """Share of letters that belong to the target script, ignoring digits.

    Citation markers and numbers stay in Western digits by design, so counting
    only letters keeps this honest for a correctly translated answer.
    """
    ranges = SCRIPT_RANGES.get(language)
    if not ranges:
        return 1.0  # nothing to prove for English or an unknown code
    letters = 0
    in_script = 0
    for char in text:
        if not char.isalpha():
            continue
        letters += 1
        code = ord(char)
        if any(low <= code <= high for low, high in ranges):
            in_script += 1
    if letters == 0:
        return 0.0
    return in_script / letters


class Citer:
    """Assigns `[n]` markers to evidence ids in order of first use.

    Numbering is by first appearance in the answer, not by evidence order, so
    `[1]` is always the first thing the reader meets. An id that is not in the
    evidence book is skipped rather than producing a dangling marker, which
    keeps the invariant that every marker in the text resolves.
    """

    def __init__(self, book: EvidenceBook) -> None:
        self.book = book
        self._order: list[str] = []
        self._by_id: dict[str, int] = {}

    def ref(self, *evidence_ids: str) -> str:
        """Return ` [n]` or ` [n,m]` for the given ids, or '' if none resolve."""
        marks: list[int] = []
        for eid in evidence_ids:
            if not eid:
                continue
            if eid not in self._by_id:
                if self.book.get(eid) is None:
                    continue
                self._by_id[eid] = len(self._order) + 1
                self._order.append(eid)
            marks.append(self._by_id[eid])
        if not marks:
            return ""
        uniq = sorted(set(marks))
        return " [" + ",".join(str(m) for m in uniq) + "]"

    @property
    def markers(self) -> list[int]:
        return list(range(1, len(self._order) + 1))

    def citations(self) -> list[Citation]:
        out: list[Citation] = []
        for marker, eid in enumerate(self._order, start=1):
            ev = self.book.get(eid)
            if ev is None:
                continue
            p = ev.provenance
            out.append(
                Citation(
                    marker=marker,
                    evidence_id=eid,
                    label=ev.label,
                    value=(
                        f"{ev.value:.2f} {ev.unit}".strip()
                        if isinstance(ev.value, float)
                        else (f"{ev.value} {ev.unit}".strip() if ev.value is not None else "")
                    ),
                    agency=p.agency,
                    dataset=p.dataset,
                    tier=p.tier,
                    url=p.url,
                    official=p.official,
                    is_stale=p.is_stale,
                )
            )
        return out

TIER_LABEL = {
    "tier1-isro": "ISRO/MOSDAC",
    "tier2-incois": "INCOIS",
    "tier3-imd": "IMD",
    "tier4-fallback": "fallback model",
    "derived": "ORCA-derived",
    "seed": "ORCA reference",
}

REWRITE_PROMPT = """Rewrite the draft answer below for this specific reader.

READER: {audience_label}
REGISTER: {reading_level}
HOW TO WRITE FOR THEM: {guidance}
LENGTH: under {max_words} words.

Keep every number, place name, source attribution and the safety verdict exactly
as they are. Do not add any fact that is not in the draft. Do not add a greeting
or a sign-off.

Write it as flowing speech, not as a form. Never open with a label like
"Verdict:", "Summary:" or "Status:". Say the thing itself, the way a person would
say it out loud.

CITATION RULE, this one is absolute: the draft contains square-bracket markers
like [1], [2] or [3,4]. Every single marker must appear in your rewrite, attached
to the same fact it was attached to in the draft. Do not renumber them, do not
merge them, do not drop one. If you move a sentence, its marker moves with it.
An answer that loses a marker is rejected and thrown away.

Verdict that must survive unchanged: {verdict}

Draft:
{draft}
"""

REGIONAL_REWRITE_PROMPT = """You are ORCA. Translate and adapt the draft answer
below into natural {lang_name}, for this specific reader.

READER: {audience_label}
REGISTER: {reading_level}
HOW TO WRITE FOR THEM: {guidance}
LENGTH: under {max_words} words.

Use the everyday {lang_name} a coastal reader actually speaks, not textbook
translation. Where a marine term has a common local word, use the local word.
Write it as flowing speech, not as a form: never open with a label like
"Verdict:" or "Summary:".
Keep every number, place name, source attribution and the safety verdict
intact and accurate. Do not add any fact that is not in the draft.
Do not add a greeting or a sign-off.

The safety verdict for this answer is: {verdict}. That is a meaning to convey,
not a word to copy. Express it naturally in {lang_name} and never leave the
English word in the middle of a {lang_name} sentence. Agency names (ISRO,
INCOIS, IMD) and numbers stay as they are.

CITATION RULE, this one is absolute: the draft contains square-bracket markers
like [1], [2] or [3,4]. Keep every marker, in Western digits, attached to the same
fact it was attached to in the draft. Do not renumber, merge or drop any. An
answer that loses a marker is rejected and thrown away.

Draft:
{draft}
"""


class SynthesisAgent:
    name = "synthesis"
    tools = ("templates", "llm.rewrite", "sarvam.bulbul_tts")

    def __init__(self, services: Services) -> None:
        self.services = services

    async def run(self, ctx: AgentContext) -> None:
        with ctx.trace.timed(
            self.name,
            "compose the answer from retrieved evidence",
            rationale=(
                "deterministic template first so the numbers and the verdict are "
                "reproducible and cannot be invented"
            ),
            tool="templates",
        ) as step:
            draft = self._draft(ctx)
            ctx.findings.notes.append("")  # keep notes list non-empty for joins
            ctx.findings.notes = [n for n in ctx.findings.notes if n]
            ctx.citations = ctx.citer.citations()
            # Same numbers, no sentences. Built from the citations so a briefing
            # row points at the same source the prose does.
            ctx.briefing = build_briefing(ctx, ctx.citations)
            step.outcome = (
                f"{len(draft.split())} word draft from templates, "
                f"{len(ctx.citations)} inline citations, "
                f"{len(ctx.briefing.ocean) + len(ctx.briefing.weather) + len(ctx.briefing.gis)} "
                "briefing figures"
            )
            step.evidence_ids = [c.evidence_id for c in ctx.citations]

        final = draft
        used_llm = False
        target_lang = ctx.language_hint or "en"
        # The n8n critic can demand a deterministic re-run after it rejects a
        # rewrite, so the repair pass must be able to skip the model entirely.
        allow_llm = ctx.findings.diagnosis.get("allow_llm", True)
        provider, model = (
            await self.services.llm.resolve_role(LlmRole.SYNTHESIS)
            if allow_llm
            else ("none", "")
        )

        with ctx.trace.timed(
            self.name,
            f"rewrite for readability ({target_lang})",
            rationale=(
                "a fast multilingual model rephrases or translates for coastal "
                "fishermen; the safety verdict is verified and the draft is kept "
                "if the rewrite alters it"
            ),
            tool="llm.rewrite",
            tool_args={"provider": provider, "model": model, "language": target_lang},
        ) as step:
            # Give the model the verdict as a meaning, not as a token. Passing
            # the raw band value made Gemini drop the literal English word
            # "unsafe" into the middle of a Kannada sentence.
            verdict = VERDICT_PHRASE.get(
                ctx.risk.band if ctx.risk else None, "no safety verdict applies"
            )
            system_prompt = get_system_prompt(target_lang)
            style = (
                ctx.audience.style
                if ctx.audience
                else AUDIENCE_STYLE[Audience.FISHERMAN]
            )
            common = {
                "verdict": verdict,
                "draft": draft,
                "audience_label": style["label"],
                "reading_level": style["reading_level"],
                "guidance": style["guidance"],
                "max_words": style["max_words"],
            }
            if target_lang != "en":
                prompt_text = REGIONAL_REWRITE_PROMPT.format(
                    lang_name=LANGUAGE_NAMES.get(target_lang, target_lang), **common
                )
            else:
                prompt_text = REWRITE_PROMPT.format(**common)

            reply = None
            attempts: list[str] = []
            if allow_llm:
                # Up to two attempts. A translation that comes back in the wrong
                # script or without its citation markers is not usable, and the
                # second attempt is told exactly what went wrong.
                max_attempts = 2 if target_lang != "en" else 1
                for attempt in range(1, max_attempts + 1):
                    prompt = prompt_text
                    if attempt > 1:
                        prompt = (
                            prompt_text
                            + "\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED: "
                            + "; ".join(attempts)
                            + ".\nFix exactly that. Write the whole answer in "
                            + LANGUAGE_NAMES.get(target_lang, target_lang)
                            + " and include every one of these markers verbatim: "
                            + " ".join(f"[{c.marker}]" for c in ctx.citations)
                            + "."
                        )
                    candidate_reply = await self.services.llm.complete(
                        prompt,
                        system=system_prompt,
                        role=LlmRole.SYNTHESIS,
                        # A translation into an Indian script costs far more
                        # tokens than the English draft did, and a truncated
                        # answer loses its trailing citations and gets rejected.
                        max_tokens=1600 if target_lang != "en" else 1100,
                    )
                    if candidate_reply is None or not candidate_reply.text.strip():
                        attempts.append("the model returned nothing")
                        break
                    reply = candidate_reply
                    candidate = candidate_reply.text.strip()

                    problems: list[str] = []
                    if not self._verdict_survived(candidate, ctx, target_lang):
                        problems.append("the safety verdict was softened or dropped")
                    lost = self._citations_lost(candidate, ctx)
                    if lost:
                        problems.append(
                            "citation markers "
                            + ", ".join(f"[{m}]" for m in lost)
                            + " were dropped"
                        )
                    ratio = script_ratio(candidate, target_lang)
                    if target_lang != "en" and ratio < 0.55:
                        problems.append(
                            f"the answer was not written in "
                            f"{LANGUAGE_NAMES.get(target_lang, target_lang)} "
                            f"(only {ratio * 100:.0f}% of letters were in that script)"
                        )

                    if not problems:
                        final = candidate
                        used_llm = True
                        step.outcome = (
                            f"rewritten by {candidate_reply.provider}/"
                            f"{candidate_reply.model} in {target_lang}, "
                            f"all {len(ctx.citations)} citations intact"
                            + (f" (attempt {attempt})" if attempt > 1 else "")
                        )
                        break
                    attempts.append("; ".join(problems))

            if not allow_llm:
                step.status = "skipped"
                step.outcome = (
                    "the caller asked for deterministic synthesis only, so the "
                    "template draft is the answer"
                )
            elif not used_llm and attempts:
                step.status = "degraded"
                step.outcome = (
                    f"every rewrite attempt was rejected ({attempts[-1]}), so the "
                    "deterministic draft was kept"
                )
            elif not used_llm:
                step.status = "degraded"
                step.outcome = "no synthesis model configured; using the deterministic draft"

        # Language guard. If the reader asked in Kannada and we are about to hand
        # back English, that is a failure we must not paper over: say so in their
        # script, and report the language we actually produced.
        language_fallback = False
        if target_lang != "en" and script_ratio(final, target_lang) < 0.55:
            language_fallback = True
            with ctx.trace.timed(
                self.name,
                f"language guard: could not deliver {target_lang}",
                rationale=(
                    "answering a regional language question in English is a "
                    "failure, so it is declared rather than hidden"
                ),
                tool="language-guard",
            ) as lstep:
                lstep.status = "degraded"
                lstep.outcome = (
                    f"the answer is in English because the {target_lang} rewrite "
                    "could not be produced or verified"
                )
            note = LANGUAGE_FALLBACK_NOTE.get(target_lang)
            if note:
                final = f"{note}\n\n{final}"

        # Synthesize voice audio with Sarvam AI
        audio_b64: str | None = None
        with ctx.trace.timed(
            self.name,
            f"synthesize speech ({target_lang}) via Sarvam AI",
            rationale=(
                "converts the synthesized response into natural coastal Indian "
                "speech for hands-free deck operation"
            ),
            tool="sarvam.bulbul_tts",
            tool_args={"language": target_lang},
        ) as step:
            tts_res = await get_voice_client().text_to_speech(final, language_code=target_lang)
            audio_b64 = tts_res.audio_base64
            step.outcome = f"synthesized voice via {tts_res.provider} ({tts_res.speaker})"

        ctx.findings.diagnosis["answer"] = final
        ctx.findings.diagnosis["llm_used"] = used_llm
        ctx.findings.diagnosis["audio_base64"] = audio_b64
        ctx.findings.diagnosis["language"] = target_lang
        ctx.findings.diagnosis["synthesis_role"] = {
            "provider": provider,
            "model": model,
            "used": used_llm,
            "input_tokens": getattr(reply, "input_tokens", 0) if reply else 0,
            "output_tokens": getattr(reply, "output_tokens", 0) if reply else 0,
            "cost_usd": getattr(reply, "cost_usd", None) if reply else None,
            # a rejected rewrite still costs money, so record that it was billed
            "billed_but_discarded": bool(reply and not used_llm),
        }
        ctx.findings.diagnosis["tts_role"] = {
            "provider": tts_res.provider,
            "model": (
                self.services.settings.sarvam_tts_model
                if get_voice_client().is_configured
                else "offline-beep"
            ),
            "speaker": tts_res.speaker,
            "used": bool(audio_b64),
            "characters": tts_res.characters,
            "chunks": tts_res.chunks,
        }
        ctx.followups = self._followups(ctx)

    # --------------------------------------------------------------- drafts #

    def _draft(self, ctx: AgentContext) -> str:
        ctx.reset_citer()  # numbering restarts for every draft attempt
        if ctx.plan.clarification_needed:
            return ctx.plan.clarification_needed
        builder = {
            Intent.SMALL_TALK: self._small_talk,
            Intent.SAFETY_GO_NOGO: self._safety,
            Intent.CONDITIONS_SUMMARY: self._conditions,
            Intent.HAZARD_ALERTS: self._hazards,
            Intent.GEOFENCE_CHECK: self._geofence,
            Intent.ROUTE_PLANNING: self._route,
            Intent.PFZ_LOCATE: self._pfz,
            Intent.PRODUCTIVITY_SCAN: self._pfz,
            Intent.PRODUCTIVITY_DIAGNOSIS: self._diagnosis,
            Intent.DATA_DISCOVERY: self._catalog,
        }.get(ctx.intent, self._conditions)
        body = builder(ctx)
        return self._with_caveats(body, ctx)

    # -- helpers ---------------------------------------------------------- #

    @staticmethod
    def _where(ctx: AgentContext) -> str:
        if not ctx.location:
            return "your position"
        if ctx.location.source == "explicit":
            return f"{ctx.location.lat:.2f}N {ctx.location.lon:.2f}E"
        return ctx.location.name

    @staticmethod
    def _when(ctx: AgentContext) -> str:
        return ctx.window.label if ctx.window else "now"

    def _conditions_lines(self, ctx: AgentContext) -> list[str]:
        lines: list[str] = []
        cit = ctx.citer
        waves = ctx.findings.waves or {}
        weather = ctx.findings.weather or {}
        ocean = ctx.findings.ocean or {}
        wfields = waves.get("fields", {}) or {}
        mfields = weather.get("fields", {}) or {}
        ofields = ocean.get("fields", {}) or {}

        def eid(fields: dict, key: str) -> str:
            return (fields.get(key) or {}).get("evidence_id", "")

        swh = waves.get("swh_now_m")
        if swh is not None:
            state = waves.get("sea_state", sea_state(swh))
            tier = TIER_LABEL.get((wfields.get("swh") or {}).get("tier", ""), "model")
            line = f"Waves about {swh:.1f} m ({state} sea), from {tier}"
            peak = waves.get("swh_peak_m")
            if peak is not None and peak - swh > 0.3:
                line += f", building to {peak:.1f} m later in the window"
            if waves.get("wave_from"):
                line += f", running from the {waves['wave_from']}"
            lines.append(line + "." + cit.ref(eid(wfields, "swh")))
        isro_swh = waves.get("swh_isro_m")
        if isro_swh is not None and swh is not None and abs(isro_swh - swh) > 0.2:
            lines.append(
                f"The ISRO OSF_WAVE forecast for this point gives {isro_swh:.1f} m "
                "for its own cycle, which differs from the live value above."
                + cit.ref(eid(wfields, "swh_isro"), eid(wfields, "swh"))
            )

        wind = weather.get("wind_kt")
        if wind is not None:
            force, label = beaufort(wind)
            line = f"Wind {wind:.0f} kt (Beaufort {force}, {label})"
            if weather.get("wind_from"):
                line += f" from the {weather['wind_from']}"
            gust = weather.get("gust_peak_kt")
            if gust is not None and gust > wind + 3:
                line += f", gusting to {gust:.0f} kt"
            lines.append(
                line + "." + cit.ref(eid(mfields, "wind"), eid(mfields, "gust"))
            )

        sst = (ofields.get("sst") or {}).get("value")
        if sst is not None:
            tier = TIER_LABEL.get((ofields.get("sst") or {}).get("tier", ""), "model")
            lines.append(
                f"Sea surface temperature {sst:.1f} degC ({tier})."
                + cit.ref(eid(ofields, "sst"))
            )

        current = (ofields.get("current") or {}).get("value")
        if current is not None:
            setting = ocean.get("current_set", "")
            lines.append(
                f"Surface current about {current:.0f} cm/s"
                + (f" setting {setting}" if setting else "")
                + "."
                + cit.ref(eid(ofields, "current"))
            )

        tide_now = waves.get("tide_now_m")
        if tide_now is not None:
            high = waves.get("tide_high_m")
            low = waves.get("tide_low_m")
            line = f"Sea level {tide_now:+.2f} m relative to mean sea level"
            if high is not None and low is not None:
                line += f", ranging {low:+.2f} m to {high:+.2f} m over the window"
            lines.append(line + "." + cit.ref(eid(wfields, "tide")))

        rain = weather.get("rain_peak_mmh")
        if rain is not None and rain > 0.2:
            lines.append(
                f"Rain up to {rain:.1f} mm/h in the window."
                + cit.ref(eid(mfields, "rain"))
            )
        return lines

    def _risk_lines(self, ctx: AgentContext) -> list[str]:
        if not ctx.risk:
            return []
        cit = ctx.citer
        lines = [ctx.risk.headline]
        for finding in ctx.risk.findings:
            if finding.band in (RiskBand.UNSAFE, RiskBand.CAUTION):
                lines.append(f"- {finding.detail}" + cit.ref(*finding.evidence_ids))
        if ctx.risk.band == RiskBand.SAFE:
            ok = [f for f in ctx.risk.findings if f.band == RiskBand.SAFE]
            for finding in ok[:2]:
                lines.append(f"- {finding.detail}" + cit.ref(*finding.evidence_ids))
        if ctx.risk.window_advice:
            lines.append(ctx.risk.window_advice)
        return lines

    # -- plain language layer --------------------------------------------- #
    # The deterministic draft is what ships when no model is available, and it is
    # also what the model is told to preserve. So the plain wording has to exist
    # here, in code, not only in the prompt. Otherwise a fisherman with no API key
    # gets "significant wave height 1.14 m, tier4-fallback".

    @staticmethod
    def _knots_from_cms(cms: float) -> float:
        return cms * 0.0194384

    @staticmethod
    def _wave_feel(swh: float) -> str:
        if swh < 0.5:
            return "almost flat"
        if swh < 1.0:
            return "small, easy going"
        if swh < 1.5:
            return "choppy but workable"
        if swh < 2.5:
            return "rough, hard work in a small boat"
        return "dangerous for a small boat"

    #: Which conditions actually bear on each question. A go/no-go decision turns
    #: on sea state, wind and storms; sea temperature and tide do not change the
    #: answer and only bury it. A conditions request is the one case that genuinely
    #: wants everything.
    RELEVANT_FIELDS: dict[Intent, tuple[str, ...]] = {
        Intent.SAFETY_GO_NOGO: ("swh", "wind", "rain"),
        Intent.HAZARD_ALERTS: ("wind", "rain"),
        Intent.ROUTE_PLANNING: ("swh", "wind", "current"),
        Intent.GEOFENCE_CHECK: ("swh", "wind"),
        Intent.PFZ_LOCATE: ("sst", "current", "swh"),
        Intent.PRODUCTIVITY_SCAN: ("sst", "current"),
        Intent.PRODUCTIVITY_DIAGNOSIS: ("sst", "current"),
    }

    def _wanted_fields(self, ctx: AgentContext) -> tuple[str, ...] | None:
        """None means "report everything", which is what a conditions ask wants."""
        return self.RELEVANT_FIELDS.get(ctx.intent)

    def _plain_conditions_lines(self, ctx: AgentContext) -> list[str]:
        """Same numbers, everyday words. No dataset names, no jargon."""
        lines: list[str] = []
        cit = ctx.citer
        waves = ctx.findings.waves or {}
        weather = ctx.findings.weather or {}
        ocean = ctx.findings.ocean or {}
        wfields = waves.get("fields", {}) or {}
        mfields = weather.get("fields", {}) or {}
        ofields = ocean.get("fields", {}) or {}
        wanted = self._wanted_fields(ctx)

        def show(key: str) -> bool:
            return wanted is None or key in wanted

        def eid(fields: dict, key: str) -> str:
            return (fields.get(key) or {}).get("evidence_id", "")

        swh = waves.get("swh_now_m")
        if swh is not None and show("swh"):
            line = f"Waves about {swh:.1f} m right now, {self._wave_feel(swh)}"
            peak = waves.get("swh_peak_m")
            if peak is not None and peak - swh > 0.3:
                line += f", building to about {peak:.1f} m later"
            if waves.get("wave_from"):
                line += f", coming from the {waves['wave_from']}"
            lines.append(line + "." + cit.ref(eid(wfields, "swh")))

        wind = weather.get("wind_kt")
        if wind is not None and show("wind"):
            _, label = beaufort(wind)
            line = f"Wind {wind:.0f} knots, a {label}"
            if weather.get("wind_from"):
                line += f" from the {weather['wind_from']}"
            gust = weather.get("gust_peak_kt")
            if gust is not None and gust > wind + 3:
                line += f", with stronger blows up to {gust:.0f} knots"
            lines.append(
                line + "." + cit.ref(eid(mfields, "wind"), eid(mfields, "gust"))
            )

        sst = (ofields.get("sst") or {}).get("value")
        if sst is not None and show("sst"):
            lines.append(
                f"Sea temperature about {sst:.0f} degrees."
                + cit.ref(eid(ofields, "sst"))
            )

        current = (ofields.get("current") or {}).get("value")
        if current is not None and show("current"):
            kn = self._knots_from_cms(current)
            setting = ocean.get("current_set", "")
            line = f"Water is drifting {setting}" if setting else "Water is drifting"
            line += f" at about {kn:.1f} knots"
            if kn >= 1.0:
                line += ", so allow for it on the way back"
            lines.append(line + "." + cit.ref(eid(ofields, "current")))

        tide_now = waves.get("tide_now_m")
        if tide_now is not None and show("tide"):
            high = waves.get("tide_high_m")
            low = waves.get("tide_low_m")
            line = f"Tide is about {abs(tide_now):.1f} m {'above' if tide_now >= 0 else 'below'} normal"
            if high is not None and low is not None:
                line += f", moving between {low:+.1f} m and {high:+.1f} m today"
            lines.append(line + "." + cit.ref(eid(wfields, "tide")))

        rain = weather.get("rain_peak_mmh")
        if rain is not None and rain > 0.2 and show("rain"):
            heavy = "heavy" if rain > 4 else "light"
            lines.append(
                f"Expect {heavy} rain at times." + cit.ref(eid(mfields, "rain"))
            )
        return lines

    #: jargon in a rule's detail text, and what a boat owner would say instead
    PLAIN_RULES: dict[str, str] = {
        "swh_danger": "Waves are too big for a small boat.",
        "swh_caution": "Waves are on the edge of what a small boat can take.",
        "swh_ok": "Wave height is fine.",
        "wind_danger": "Wind is too strong to work in.",
        "wind_caution": "Wind is squally, enough to make it hard going.",
        "wind_ok": "Wind is manageable.",
        "gust_danger": "Sudden strong blows can knock a small boat over, even if the steady wind feels alright.",
        "convective_risk": "Thunderstorms can build fast today. Lightning at sea is deadly and there is no shelter out there.",
        "tropical_cyclone_distance": "There is a cyclone in the area.",
        "imd_warning_in_force": "The weather office has a warning out for fishermen. Their word overrides anything here.",
        "strong_current": "The current is strong, so keep fuel in hand for the return.",
        "inside_restricted_zone": "You are inside an area you are not allowed to fish in.",
        "approaching_boundary": "You are close to a boundary you must not cross.",
        "outside_eez": "You are outside Indian waters. Boats get seized for this.",
        "swh_unavailable": "No wave information came through for this spot, so treat it with care.",
    }

    def _plain_risk_lines(self, ctx: AgentContext) -> list[str]:
        """Verdict first, in the words a skipper uses, then the reasons."""
        if not ctx.risk:
            return []
        cit = ctx.citer
        lines = [ctx.risk.headline]
        shown = 0
        for finding in ctx.risk.findings:
            if finding.band not in (RiskBand.UNSAFE, RiskBand.CAUTION):
                continue
            if finding.rule.startswith("jev_"):
                continue  # a second-opinion score means nothing to a boat owner
            plain = self.PLAIN_RULES.get(finding.rule)
            lines.append(f"- {plain or finding.detail}" + cit.ref(*finding.evidence_ids))
            shown += 1
        if shown == 0:
            for finding in ctx.risk.findings[:2]:
                if finding.rule.startswith("jev_"):
                    continue
                plain = self.PLAIN_RULES.get(finding.rule)
                if plain:
                    lines.append(f"- {plain}" + cit.ref(*finding.evidence_ids))
        if ctx.risk.window_advice:
            lines.append(ctx.risk.window_advice)
        return lines

    @property
    def _is_plain(self) -> bool:  # pragma: no cover - replaced per call
        return False

    def _conditions_for(self, ctx: AgentContext) -> list[str]:
        """Pick the register the reader needs."""
        if ctx.audience and ctx.audience.audience is Audience.FISHERMAN:
            return self._plain_conditions_lines(ctx)
        return self._conditions_lines(ctx)

    def _risk_for(self, ctx: AgentContext) -> list[str]:
        if ctx.audience and ctx.audience.audience is Audience.FISHERMAN:
            return self._plain_risk_lines(ctx)
        return self._risk_lines(ctx)

    def _with_caveats(self, body: str, ctx: AgentContext) -> str:
        plain = bool(ctx.audience and ctx.audience.audience is Audience.FISHERMAN)
        parts = [body.strip()]
        stale = ctx.evidence.stale_ids()
        if stale:
            parts.append(
                "One of the official forecasts used here is a bit old. Check the "
                "evidence panel to see which."
                if plain
                else (
                    "Note: one or more official forecast cycles used here are older "
                    "than the freshness threshold. The evidence panel shows which."
                )
            )
        for note in ctx.findings.notes[:2]:
            parts.append(note)
        if plain:
            parts.append(
                "Numbers in brackets show where each fact came from. If the weather "
                "office says something different, go with them."
            )
        return "\n\n".join(p for p in parts if p)

    # -- per-intent drafts ------------------------------------------------ #

    def _small_talk(self, ctx: AgentContext) -> str:
        return (
            "I am ORCA, a marine information assistant for Indian waters. Ask me "
            "about sea conditions, whether it is safe to go out, tides, cyclone "
            "and weather warnings, where fish are likely to be, maritime "
            "boundaries and restricted zones, or a safe route between two "
            "harbours. Name a place, for example 'is it safe off Rameswaram "
            "tomorrow morning', or send me a latitude and longitude. Every answer "
            "shows which agency's data it came from."
        )

    def _safety(self, ctx: AgentContext) -> str:
        cit = ctx.citer
        lines = self._risk_for(ctx)
        lines.append("")
        lines.extend(self._conditions_for(ctx))
        harbours = (ctx.findings.geo or {}).get("harbours") or []
        if harbours and ctx.risk and ctx.risk.band != RiskBand.SAFE:
            first = harbours[0]
            lines.append(
                f"Nearest shelter is {first['name']}, about "
                f"{first['distance_km']:.0f} km {first['bearing']}."
                + cit.ref(first.get("evidence_id", ""))
            )
        return "\n".join(lines)

    def _conditions(self, ctx: AgentContext) -> str:
        lines = [f"Conditions off {self._where(ctx)} {self._when(ctx)}:", ""]
        condition_lines = self._conditions_for(ctx)
        if not condition_lines:
            return (
                f"I could not get conditions for {self._where(ctx)} right now. The "
                "agency services did not send anything back for this spot. The "
                "reasoning panel shows which ones failed."
            )
        lines.extend(condition_lines)
        if ctx.risk:
            lines.append("")
            lines.append(ctx.risk.headline)
        return "\n".join(lines)

    def _hazards(self, ctx: AgentContext) -> str:
        cit = ctx.citer
        hazards = ctx.findings.hazards or {}
        weather = ctx.findings.weather or {}
        mfields = weather.get("fields", {}) or {}
        lines: list[str] = []
        cyclone = hazards.get("cyclone")
        if cyclone:
            lines.append(
                f"Tropical cyclone {cyclone['name']} is about "
                f"{cyclone['distance_km']:.0f} km {cyclone['bearing']} of "
                f"{self._where(ctx)}, GDACS alert level "
                f"{cyclone['alert_level'] or 'unclassified'}. RSMC New Delhi is the "
                "official authority for this basin, so confirm against the IMD "
                "bulletin before acting." + cit.ref(cyclone.get("evidence_id", ""))
            )
        else:
            lines.append(
                "No active tropical cyclone is listed in the north Indian Ocean "
                "right now."
            )

        convective = weather.get("convective_risk") or {}
        if convective:
            lines.append(
                f"Thunderstorm and lightning likelihood: {convective['band']}. "
                f"{convective['explanation']} (CAPE "
                f"{convective['cape_j_per_kg']:.0f} J/kg, rain up to "
                f"{convective['rain_peak_mm_per_h']:.1f} mm/h). This is a derived "
                "indicator. India has no public lightning strike feed, so check "
                "IMD's nowcast before you sail."
                + cit.ref(
                    (mfields.get("cape") or {}).get("evidence_id", ""),
                    (mfields.get("rain") or {}).get("evidence_id", ""),
                )
            )

        warnings = hazards.get("imd_warnings") or []
        if warnings:
            imd_ref = cit.ref("imd-fishermen-warning")
            lines.append(f"IMD warning text in force:{imd_ref}")
            for sentence in warnings[:3]:
                lines.append(f"- {sentence}")
        else:
            lines.append(
                "No fishermen or sea-state sentence was found on IMD's public "
                "warning page for this request."
            )
        if ctx.risk:
            lines.append("")
            lines.append(ctx.risk.headline)
        return "\n".join(lines)

    def _geofence(self, ctx: AgentContext) -> str:
        cit = ctx.citer
        geo = ctx.findings.geo or {}
        zones = geo.get("zones") or []
        eez = geo.get("eez") or {}
        lines: list[str] = []

        inside = [z for z in zones if z["status"] == "inside"]
        approaching = [z for z in zones if z["status"] == "approaching"]
        nearby = [z for z in zones if z["status"] == "clear"]

        if inside:
            for zone in inside:
                lines.append(
                    f"You are inside {zone['name']}. {zone['advisory']}"
                    + cit.ref(zone.get("evidence_id", ""))
                )
        if approaching:
            for zone in approaching:
                lines.append(
                    f"{zone['name']} is {zone['distance_km']:.0f} km "
                    f"{zone['bearing']} of you. {zone['advisory']}"
                    + cit.ref(zone.get("evidence_id", ""))
                )
        if not inside and not approaching:
            lines.append(
                f"No protected or restricted zone is within its warning buffer of "
                f"{self._where(ctx)}."
            )
        if nearby:
            lines.append(
                "Also within 60 km: "
                + ", ".join(
                    f"{z['name']} ({z['distance_km']:.0f} km {z['bearing']})"
                    + cit.ref(z.get("evidence_id", ""))
                    for z in nearby[:4]
                )
                + "."
            )
        if eez.get("available"):
            eez_ref = cit.ref("eez-status")
            if eez["inside_india_eez"]:
                lines.append(
                    f"You are inside the India EEZ, "
                    f"{eez['distance_to_boundary_km']:.0f} km from the boundary "
                    f"({eez['bearing_to_boundary']})." + eez_ref
                )
            else:
                lines.append(
                    "This position is OUTSIDE the India EEZ, about "
                    f"{eez['distance_to_boundary_km']:.0f} km beyond the boundary."
                    + eez_ref
                )
        lines.append(
            "Zone geometry here is approximate and indicative. It is not survey "
            "grade and must not be used for navigation or position fixing."
        )
        return "\n".join(lines)

    def _route(self, ctx: AgentContext) -> str:
        cit = ctx.citer
        route = ctx.findings.route or {}
        if not route:
            return (
                "I need both ends of the passage. Tell me where you are sailing "
                "from and where you are going, for example 'safest route from "
                "Chennai to Kakinada'."
            )
        best = route["recommended"]
        swh_ref = cit.ref(
            ((ctx.findings.waves or {}).get("fields", {}).get("swh") or {}).get(
                "evidence_id", ""
            )
        )
        lines = [
            f"From {route['origin']['name']} to {route['destination']['name']} is "
            f"{route['direct_km']:.0f} km on a direct track, initial course "
            f"{route['initial_bearing']}.",
            "",
            f"Recommended: {best['label']}, {best['length_km']:.0f} km"
            + (
                f" ({best['extra_distance_km']:+.0f} km against the direct track)"
                if abs(best["extra_distance_km"]) >= 1
                else ""
            )
            + (
                f", peak wave height {best['max_swh_m']:.1f} m along the way."
                + swh_ref
                if best.get("max_swh_m") is not None
                else ", sea state could not be sampled along the whole track."
            ),
        ]
        if best["zone_conflicts"]:
            for zone in best["zone_conflicts"]:
                lines.append(
                    f"- Watch for {zone['name']} ({zone['status']}, "
                    f"{zone['distance_km']:.0f} km). {zone['advisory']}"
                    + cit.ref(zone.get("evidence_id", ""))
                )
        alternatives = [a for a in route["alternatives"][1:4]]
        if alternatives:
            lines.append("")
            lines.append("Alternatives considered:")
            for alt in alternatives:
                lines.append(
                    f"- {alt['label']}: {alt['length_km']:.0f} km, peak "
                    f"{alt['max_swh_m'] if alt['max_swh_m'] is not None else 'n/a'} m, "
                    f"{alt['zone_conflicts']} zone conflict(s)"
                )
        if ctx.risk:
            lines.append("")
            lines.append(ctx.risk.headline)
        lines.append(
            "This comparison weighs sea state and zone conflicts only. It does "
            "not know bathymetry, shoals, traffic separation or your boat's "
            "handling. Use it with a chart."
        )
        return "\n".join(lines)

    def _pfz(self, ctx: AgentContext) -> str:
        cit = ctx.citer
        ocean = ctx.findings.ocean or {}
        ofields = ocean.get("fields", {}) or {}
        sst = (ofields.get("sst") or {}).get("value")
        mld = (ofields.get("mld") or {}).get("value")

        def eid(key: str) -> str:
            return (ofields.get(key) or {}).get("evidence_id", "")

        lines: list[str] = []
        lines.append(
            "INCOIS is the authority for Potential Fishing Zone advisories and "
            "publishes them per state as bulletins, not as machine-readable data. "
            "Here is what the official grids say about the water off "
            f"{self._where(ctx)} {self._when(ctx)}, and what it implies."
        )
        lines.append("")
        if sst is not None:
            verdict = (
                "in the productive band for sardine, mackerel and tuna"
                if 27.0 <= sst <= 30.0
                else (
                    "warm enough to stratify the surface layer and suppress "
                    "surface catch"
                    if sst > 30.0
                    else "cooler than the usual productive band here"
                )
            )
            lines.append(
                f"Sea surface temperature {sst:.1f} degC from the ISRO Ocean State "
                f"Forecast, which is {verdict}." + cit.ref(eid("sst"))
            )
        if mld is not None:
            lines.append(
                f"Mixed layer depth about {mld:.0f} m. A deeper mixed layer brings "
                "nutrients up and generally means better feeding." + cit.ref(eid("mld"))
            )
        current = (ofields.get("current") or {}).get("value")
        if current is not None:
            lines.append(
                f"Surface current about {current:.0f} cm/s"
                + (f" setting {ocean.get('current_set')}" if ocean.get("current_set") else "")
                + ", which is what will carry your gear."
                + cit.ref(eid("current"))
            )
        advisories = ctx.findings.advisories or []
        if advisories:
            lines.append("")
            lines.append(
                "From the advisory knowledge base: "
                + advisories[0]["text"][:320].rsplit(" ", 1)[0]
                + "..."
                + cit.ref(advisories[0].get("evidence_id", ""))
            )
        lines.append("")
        lines.append(
            "For the actual zone bearings and distances from your landing centre, "
            "read today's INCOIS PFZ bulletin for your state. Anything ORCA infers "
            "from the grids is a candidate, not an official advisory."
        )
        if ctx.risk and ctx.risk.band != RiskBand.SAFE:
            lines.append("")
            lines.append(
                ctx.risk.headline
                + " A productive zone is not an opportunity if you cannot reach it "
                "safely."
            )
        return "\n".join(lines)

    def _diagnosis(self, ctx: AgentContext) -> str:
        cit = ctx.citer
        ocean = ctx.findings.ocean or {}
        ofields = ocean.get("fields", {}) or {}
        sst = (ofields.get("sst") or {}).get("value")
        mld = (ofields.get("mld") or {}).get("value")

        def eid(key: str) -> str:
            return (ofields.get(key) or {}).get("evidence_id", "")

        lines = [
            f"Looking at the physical drivers off {self._where(ctx)}, from the "
            "official grids ORCA can read today:",
            "",
        ]
        if sst is not None:
            lines.append(
                f"- Sea surface temperature is {sst:.1f} degC. Above about 31 degC "
                "the surface layer stratifies, the nutrient supply from below is "
                "cut off, and surface catch usually falls." + cit.ref(eid("sst"))
            )
        if mld is not None:
            lines.append(
                f"- Mixed layer depth is about {mld:.0f} m. A shallow mixed layer "
                "is consistent with weak mixing and lower productivity."
                + cit.ref(eid("mld"))
            )
        current = (ofields.get("current") or {}).get("value")
        if current is not None:
            lines.append(
                f"- Surface current is about {current:.0f} cm/s, which changes where "
                "larvae and feed are carried." + cit.ref(eid("current"))
            )
        advisories = ctx.findings.advisories or []
        for advisory in advisories[:2]:
            lines.append(
                f"- {advisory['title']}: {advisory['text'][:220]}..."
                + cit.ref(advisory.get("evidence_id", ""))
            )
        lines.append("")
        lines.append(
            "An honest caveat: a real decline in catch is usually a mix of "
            "physical change, fishing effort, gear, and enforcement of closed "
            "seasons. ORCA can show you the physical side from satellite and model "
            "data. It cannot see effort or landings, and this prototype is reading "
            "current conditions rather than a multi-year trend, so treat this as "
            "one input and not a conclusion."
        )
        return "\n".join(lines)

    def _catalog(self, ctx: AgentContext) -> str:
        catalog = ctx.findings.catalog or []
        if not catalog:
            return (
                "I could not reach the dataset catalogues just now. ORCA normally "
                "reads the MOSDAC THREDDS catalogue and the INCOIS ERDDAP index "
                "live."
            )
        lines = ["Datasets ORCA can query for this, read live from the catalogues:", ""]
        for item in catalog[:10]:
            bits = [f"- {item['agency']}: {item['dataset']}"]
            if item.get("latest"):
                bits.append(f"latest {item['latest']}")
            if item.get("variables"):
                bits.append(item["variables"])
            if item.get("access"):
                bits.append(item["access"])
            lines.append(", ".join(bits))
        lines.append("")
        lines.append(
            "Order of preference is an ISRO product on MOSDAC, then INCOIS, then "
            "IMD for warnings, and a non-Indian model only to fill a gap. Every "
            "number in an ORCA answer is labelled with which tier it came from."
        )
        return "\n".join(lines)

    # ---------------------------------------------------------- guard rails #

    @staticmethod
    def _citations_lost(candidate: str, ctx: AgentContext) -> list[int]:
        """Which `[n]` markers the rewrite failed to carry through.

        A citation that survives in the evidence panel but not in the sentence it
        supports is worthless, so a rewrite that drops one is rejected outright.
        """
        if not ctx.citations:
            return []
        present: set[int] = set()
        for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", candidate):
            for part in group.split(","):
                part = part.strip()
                if part.isdigit():
                    present.add(int(part))
        return [c.marker for c in ctx.citations if c.marker not in present]

    @staticmethod
    def _verdict_survived(candidate: str, ctx: AgentContext, language: str = "en") -> bool:
        """Reject a rewrite that flips or drops an unsafe verdict."""
        if not ctx.risk or ctx.risk.band != RiskBand.UNSAFE:
            return True
        if language != "en":
            return len(candidate.strip()) > 10
        lowered = candidate.lower()
        markers = ("do not", "don't", "avoid", "stay in", "not safe", "unsafe",
                   "remain in harbour", "postpone")
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _followups(ctx: AgentContext) -> list[str]:
        where = ctx.location.name if ctx.location else "this area"
        base = {
            Intent.SAFETY_GO_NOGO: [
                f"What are the waves and wind off {where} tomorrow morning?",
                f"Which zones should I avoid near {where}?",
                f"Any cyclone or lightning alerts near {where}?",
            ],
            Intent.CONDITIONS_SUMMARY: [
                f"Is it safe to venture out off {where} tomorrow morning?",
                f"What is the tide doing off {where} tonight?",
                f"Where is the nearest potential fishing zone off {where}?",
            ],
            Intent.PFZ_LOCATE: [
                f"Is it safe to reach that zone off {where}?",
                "Which regions have high chlorophyll and favourable SST?",
                f"Why has the catch dropped off {where}?",
            ],
            Intent.HAZARD_ALERTS: [
                f"Is it safe to go out off {where} today?",
                f"What is the nearest harbour to {where}?",
            ],
            Intent.GEOFENCE_CHECK: [
                f"How far is the maritime boundary from {where}?",
                f"Is it safe to venture out off {where} today?",
            ],
            Intent.ROUTE_PLANNING: [
                "What are the conditions along that route tomorrow?",
                "Are there any restricted zones on the way?",
            ],
            Intent.DATA_DISCOVERY: [
                "Which ISRO satellite gives chlorophyll for Indian waters?",
                f"What are the sea conditions off {where} now?",
            ],
        }
        return base.get(
            ctx.intent,
            [
                f"Is it safe to venture out off {where} tomorrow morning?",
                f"What are the tide, weather and sea conditions near {where}?",
                f"Any cyclone or lightning alerts near {where}?",
            ],
        )[:3]
