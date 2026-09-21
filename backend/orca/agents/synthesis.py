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

from typing import Any

from ..schemas import Intent, RiskBand
from ..services import Services
from .base import AgentContext
from .ocean import beaufort, sea_state

TIER_LABEL = {
    "tier1-isro": "ISRO/MOSDAC",
    "tier2-incois": "INCOIS",
    "tier3-imd": "IMD",
    "tier4-fallback": "fallback model",
    "derived": "ORCA-derived",
    "seed": "ORCA reference",
}

REWRITE_PROMPT = """Rewrite the draft answer below so it reads naturally for an
Indian fisherman or boat owner. Keep every number, place name, source attribution
and the safety verdict exactly as they are. Do not add any fact that is not in the
draft. Do not add a greeting or a sign-off. Keep it under 180 words.

Verdict that must survive unchanged: {verdict}

Draft:
{draft}
"""


class SynthesisAgent:
    name = "synthesis"
    tools = ("templates", "llm.rewrite")

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
            step.outcome = f"{len(draft.split())} word draft from templates"

        final = draft
        used_llm = False
        with ctx.trace.timed(
            self.name,
            "rewrite for readability",
            rationale=(
                "the model may only rephrase; the verdict is checked afterwards "
                "and the draft is kept if the rewrite drops it"
            ),
            tool="llm.rewrite",
        ) as step:
            verdict = ctx.risk.band.value if ctx.risk else "none"
            reply = await self.services.llm.complete(
                REWRITE_PROMPT.format(verdict=verdict, draft=draft)
            )
            if reply and reply.text.strip():
                candidate = reply.text.strip()
                if self._verdict_survived(candidate, ctx):
                    final = candidate
                    used_llm = True
                    step.outcome = f"rewritten by {reply.provider}/{reply.model}"
                else:
                    step.status = "degraded"
                    step.outcome = (
                        "rewrite dropped or altered the safety verdict, so the "
                        "deterministic draft was kept"
                    )
            else:
                step.status = "degraded"
                step.outcome = "no LLM configured; using the deterministic draft"

        ctx.findings.diagnosis["answer"] = final
        ctx.findings.diagnosis["llm_used"] = used_llm
        ctx.followups = self._followups(ctx)

    # --------------------------------------------------------------- drafts #

    def _draft(self, ctx: AgentContext) -> str:
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
        waves = ctx.findings.waves or {}
        weather = ctx.findings.weather or {}
        ocean = ctx.findings.ocean or {}

        swh = waves.get("swh_now_m")
        if swh is not None:
            state = waves.get("sea_state", sea_state(swh))
            tier = TIER_LABEL.get(
                (waves.get("fields", {}).get("swh") or {}).get("tier", ""), "model"
            )
            line = f"Waves about {swh:.1f} m ({state} sea), from {tier}"
            peak = waves.get("swh_peak_m")
            if peak is not None and peak - swh > 0.3:
                line += f", building to {peak:.1f} m later in the window"
            if waves.get("wave_from"):
                line += f", running from the {waves['wave_from']}"
            lines.append(line + ".")
        isro_swh = waves.get("swh_isro_m")
        if isro_swh is not None and swh is not None and abs(isro_swh - swh) > 0.2:
            lines.append(
                f"The ISRO OSF_WAVE forecast for this point gives {isro_swh:.1f} m "
                "for its own cycle, which differs from the live value above."
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
            lines.append(line + ".")

        sst = (ocean.get("fields", {}).get("sst") or {}).get("value")
        if sst is not None:
            tier = TIER_LABEL.get(
                (ocean.get("fields", {}).get("sst") or {}).get("tier", ""), "model"
            )
            lines.append(f"Sea surface temperature {sst:.1f} degC ({tier}).")

        current = (ocean.get("fields", {}).get("current") or {}).get("value")
        if current is not None:
            setting = ocean.get("current_set", "")
            lines.append(
                f"Surface current about {current:.0f} cm/s"
                + (f" setting {setting}" if setting else "")
                + "."
            )

        tide_now = waves.get("tide_now_m")
        if tide_now is not None:
            high = waves.get("tide_high_m")
            low = waves.get("tide_low_m")
            line = f"Sea level {tide_now:+.2f} m relative to mean sea level"
            if high is not None and low is not None:
                line += f", ranging {low:+.2f} m to {high:+.2f} m over the window"
            lines.append(line + ".")

        rain = weather.get("rain_peak_mmh")
        if rain is not None and rain > 0.2:
            lines.append(f"Rain up to {rain:.1f} mm/h in the window.")
        return lines

    def _risk_lines(self, ctx: AgentContext) -> list[str]:
        if not ctx.risk:
            return []
        lines = [ctx.risk.headline]
        for finding in ctx.risk.findings:
            if finding.band in (RiskBand.UNSAFE, RiskBand.CAUTION):
                lines.append(f"- {finding.detail}")
        if ctx.risk.band == RiskBand.SAFE:
            ok = [f for f in ctx.risk.findings if f.band == RiskBand.SAFE]
            for finding in ok[:2]:
                lines.append(f"- {finding.detail}")
        if ctx.risk.window_advice:
            lines.append(ctx.risk.window_advice)
        return lines

    def _with_caveats(self, body: str, ctx: AgentContext) -> str:
        parts = [body.strip()]
        stale = ctx.evidence.stale_ids()
        if stale:
            parts.append(
                "Note: one or more official forecast cycles used here are older "
                "than the freshness threshold. The evidence panel shows which."
            )
        for note in ctx.findings.notes[:2]:
            parts.append(note)
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
        lines = self._risk_lines(ctx)
        lines.append("")
        lines.extend(self._conditions_lines(ctx))
        harbours = (ctx.findings.geo or {}).get("harbours") or []
        if harbours and ctx.risk and ctx.risk.band != RiskBand.SAFE:
            first = harbours[0]
            lines.append(
                f"Nearest shelter is {first['name']}, about "
                f"{first['distance_km']:.0f} km {first['bearing']}."
            )
        return "\n".join(lines)

    def _conditions(self, ctx: AgentContext) -> str:
        lines = [f"Conditions off {self._where(ctx)} {self._when(ctx)}:", ""]
        condition_lines = self._conditions_lines(ctx)
        if not condition_lines:
            return (
                f"I could not retrieve conditions for {self._where(ctx)} right now. "
                "The upstream agency services did not return data for this point. "
                "The reasoning trace shows which calls failed."
            )
        lines.extend(condition_lines)
        if ctx.risk:
            lines.append("")
            lines.append(ctx.risk.headline)
        return "\n".join(lines)

    def _hazards(self, ctx: AgentContext) -> str:
        hazards = ctx.findings.hazards or {}
        weather = ctx.findings.weather or {}
        lines: list[str] = []
        cyclone = hazards.get("cyclone")
        if cyclone:
            lines.append(
                f"Tropical cyclone {cyclone['name']} is about "
                f"{cyclone['distance_km']:.0f} km {cyclone['bearing']} of "
                f"{self._where(ctx)}, GDACS alert level "
                f"{cyclone['alert_level'] or 'unclassified'}. RSMC New Delhi is the "
                "official authority for this basin, so confirm against the IMD "
                "bulletin before acting."
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
            )

        warnings = hazards.get("imd_warnings") or []
        if warnings:
            lines.append("IMD warning text in force:")
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
        geo = ctx.findings.geo or {}
        zones = geo.get("zones") or []
        eez = geo.get("eez") or {}
        lines: list[str] = []

        inside = [z for z in zones if z["status"] == "inside"]
        approaching = [z for z in zones if z["status"] == "approaching"]
        nearby = [z for z in zones if z["status"] == "clear"]

        if inside:
            for zone in inside:
                lines.append(f"You are inside {zone['name']}. {zone['advisory']}")
        if approaching:
            for zone in approaching:
                lines.append(
                    f"{zone['name']} is {zone['distance_km']:.0f} km "
                    f"{zone['bearing']} of you. {zone['advisory']}"
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
                    for z in nearby[:4]
                )
                + "."
            )
        if eez.get("available"):
            if eez["inside_india_eez"]:
                lines.append(
                    f"You are inside the India EEZ, "
                    f"{eez['distance_to_boundary_km']:.0f} km from the boundary "
                    f"({eez['bearing_to_boundary']})."
                )
            else:
                lines.append(
                    "This position is OUTSIDE the India EEZ, about "
                    f"{eez['distance_to_boundary_km']:.0f} km beyond the boundary."
                )
        lines.append(
            "Zone geometry here is approximate and indicative. It is not survey "
            "grade and must not be used for navigation or position fixing."
        )
        return "\n".join(lines)

    def _route(self, ctx: AgentContext) -> str:
        route = ctx.findings.route or {}
        if not route:
            return (
                "I need both ends of the passage. Tell me where you are sailing "
                "from and where you are going, for example 'safest route from "
                "Chennai to Kakinada'."
            )
        best = route["recommended"]
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
                if best.get("max_swh_m") is not None
                else ", sea state could not be sampled along the whole track."
            ),
        ]
        if best["zone_conflicts"]:
            for zone in best["zone_conflicts"]:
                lines.append(
                    f"- Watch for {zone['name']} ({zone['status']}, "
                    f"{zone['distance_km']:.0f} km). {zone['advisory']}"
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
        ocean = ctx.findings.ocean or {}
        sst = (ocean.get("fields", {}).get("sst") or {}).get("value")
        mld = (ocean.get("fields", {}).get("mld") or {}).get("value")
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
                f"Forecast, which is {verdict}."
            )
        if mld is not None:
            lines.append(
                f"Mixed layer depth about {mld:.0f} m. A deeper mixed layer brings "
                "nutrients up and generally means better feeding."
            )
        current = (ocean.get("fields", {}).get("current") or {}).get("value")
        if current is not None:
            lines.append(
                f"Surface current about {current:.0f} cm/s"
                + (f" setting {ocean.get('current_set')}" if ocean.get("current_set") else "")
                + ", which is what will carry your gear."
            )
        advisories = ctx.findings.advisories or []
        if advisories:
            lines.append("")
            lines.append(
                "From the advisory knowledge base: "
                + advisories[0]["text"][:320].rsplit(" ", 1)[0]
                + "..."
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
        ocean = ctx.findings.ocean or {}
        sst = (ocean.get("fields", {}).get("sst") or {}).get("value")
        mld = (ocean.get("fields", {}).get("mld") or {}).get("value")
        lines = [
            f"Looking at the physical drivers off {self._where(ctx)}, from the "
            "official grids ORCA can read today:",
            "",
        ]
        if sst is not None:
            lines.append(
                f"- Sea surface temperature is {sst:.1f} degC. Above about 31 degC "
                "the surface layer stratifies, the nutrient supply from below is "
                "cut off, and surface catch usually falls."
            )
        if mld is not None:
            lines.append(
                f"- Mixed layer depth is about {mld:.0f} m. A shallow mixed layer "
                "is consistent with weak mixing and lower productivity."
            )
        current = (ocean.get("fields", {}).get("current") or {}).get("value")
        if current is not None:
            lines.append(
                f"- Surface current is about {current:.0f} cm/s, which changes where "
                "larvae and feed are carried."
            )
        advisories = ctx.findings.advisories or []
        for advisory in advisories[:2]:
            lines.append(f"- {advisory['title']}: {advisory['text'][:220]}...")
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
    def _verdict_survived(candidate: str, ctx: AgentContext) -> bool:
        """Reject a rewrite that flips or drops an unsafe verdict."""
        if not ctx.risk or ctx.risk.band != RiskBand.UNSAFE:
            return True
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
