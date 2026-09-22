"""Hazard retrieval and risk assessment.

The verdict is computed by explicit rules, not by the language model. Each rule
that fires is recorded with the evidence that triggered it, which is what makes
the recommendation auditable: a user can see that "do not venture" came from
`swh_danger` firing on a 2.9 m wave height from the ISRO forecast, and not from
a model's intuition.

Rule severity is combined by taking the worst band, never by averaging. A single
gale warning is not cancelled out by calm seas.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..geo.geometry import bearing_deg, compass
from ..jev import JevSafetyDecision, get_jev_engine
from ..schemas import Evidence, MapMarker, RiskAssessment, RiskBand, RiskFinding
from ..services import Services
from .base import AgentContext

BAND_ORDER = {
    RiskBand.UNKNOWN: 0,
    RiskBand.SAFE: 1,
    RiskBand.CAUTION: 2,
    RiskBand.UNSAFE: 3,
}


class RiskAssessmentAgent:
    name = "risk-assessment"
    tools = (
        "gdacs.nearest_cyclone",
        "imd.fishermen_warnings",
        "threshold-rules",
        "jev.system_one",
    )

    def __init__(self, services: Services) -> None:
        self.services = services

    async def run(self, ctx: AgentContext) -> None:
        await self._retrieve_hazards(ctx)
        await self._assess(ctx)

    # ------------------------------------------------------------- retrieval #

    async def _retrieve_hazards(self, ctx: AgentContext) -> None:
        with ctx.trace.timed(
            self.name,
            "check active cyclones and official warnings",
            rationale=(
                "GDACS supplies cyclone geometry so ORCA can compute a real "
                "distance; IMD supplies the operative warning wording and "
                "overrides GDACS on alert status for the north Indian Ocean"
            ),
            tool="gdacs.nearest_cyclone",
            tool_args={"lat": round(ctx.lat, 3), "lon": round(ctx.lon, 3)},
        ) as step:
            nearest, warnings = await asyncio.gather(
                self.services.gdacs.nearest_cyclone(ctx.lat, ctx.lon),
                self.services.imd.fishermen_warnings(),
            )
            hazards: dict[str, Any] = {"cyclone": None, "imd_warnings": warnings}
            if nearest:
                event, distance_km = nearest
                evidence = Evidence(
                    id=f"tc-{event.get('event_id', 'unknown')}",
                    label=f"tropical cyclone {event.get('name', 'unnamed')}",
                    value=round(distance_km, 0),
                    unit="km away",
                    at_lat=event.get("lat"),
                    at_lon=event.get("lon"),
                    text=(
                        f"GDACS alert level {event.get('alert_level') or 'unknown'}; "
                        f"{event.get('severity', '')}"
                    ),
                    provenance=self.services.gdacs.provenance(),
                )
                ctx.evidence.add(evidence)
                hazards["cyclone"] = {
                    "name": event.get("name"),
                    "alert_level": event.get("alert_level"),
                    "distance_km": round(distance_km, 0),
                    "bearing": compass(
                        bearing_deg(ctx.lat, ctx.lon, event["lat"], event["lon"])
                    ),
                    "lat": event["lat"],
                    "lon": event["lon"],
                    "url": event.get("url", ""),
                    "evidence_id": evidence.id,
                }
                step.evidence_ids = [evidence.id]
                step.outcome = (
                    f"{event.get('name')} at {distance_km:.0f} km, GDACS level "
                    f"{event.get('alert_level')}"
                )
            else:
                step.outcome = "no active tropical cyclone in the north Indian Ocean"

            if warnings:
                evidence = Evidence(
                    id="imd-fishermen-warning",
                    label="IMD warning text mentioning fishermen or sea state",
                    text=" ".join(warnings)[:900],
                    provenance=self.services.imd.provenance(
                        "sub-division wise weather warning",
                        self.services.settings.imd_subdivision_warning_url,
                    ),
                )
                ctx.evidence.add(evidence)
                step.evidence_ids.append(evidence.id)
                step.outcome += f"; {len(warnings)} IMD warning sentence(s) matched"
            ctx.findings.hazards = hazards

    # ------------------------------------------------------------ assessment #

    async def _assess(self, ctx: AgentContext) -> None:
        settings = self.services.settings
        with ctx.trace.timed(
            self.name,
            "apply safety thresholds",
            rationale=(
                "explicit rules over retrieved values; the worst band wins so a "
                "single severe factor is never averaged away"
            ),
            tool="threshold-rules",
            tool_args={
                "swh_caution_m": settings.swh_caution_m,
                "swh_danger_m": settings.swh_danger_m,
                "wind_caution_kt": settings.wind_caution_kt,
                "wind_danger_kt": settings.wind_danger_kt,
                "gust_danger_kt": settings.gust_danger_kt,
            },
        ) as step:
            findings: list[RiskFinding] = []
            waves = ctx.findings.waves or {}
            weather = ctx.findings.weather or {}
            ocean = ctx.findings.ocean or {}
            hazards = ctx.findings.hazards or {}
            geo = ctx.findings.geo or {}

            swh_peak = waves.get("swh_peak_m")
            swh_id = (waves.get("fields", {}).get("swh") or {}).get("evidence_id", "")
            if swh_peak is None:
                findings.append(
                    RiskFinding(
                        rule="swh_unavailable",
                        band=RiskBand.UNKNOWN,
                        detail="No wave height was available for this position.",
                    )
                )
            elif swh_peak >= settings.swh_danger_m:
                findings.append(
                    RiskFinding(
                        rule="swh_danger",
                        band=RiskBand.UNSAFE,
                        detail=(
                            f"Peak wave height {swh_peak:.1f} m is at or above the "
                            f"{settings.swh_danger_m} m danger threshold for small "
                            "craft."
                        ),
                        evidence_ids=[swh_id] if swh_id else [],
                    )
                )
            elif swh_peak >= settings.swh_caution_m:
                findings.append(
                    RiskFinding(
                        rule="swh_caution",
                        band=RiskBand.CAUTION,
                        detail=(
                            f"Peak wave height {swh_peak:.1f} m is above the "
                            f"{settings.swh_caution_m} m caution threshold. Small "
                            "and non-mechanised craft should stay in."
                        ),
                        evidence_ids=[swh_id] if swh_id else [],
                    )
                )
            else:
                findings.append(
                    RiskFinding(
                        rule="swh_ok",
                        band=RiskBand.SAFE,
                        detail=(
                            f"Peak wave height {swh_peak:.1f} m stays below the "
                            f"{settings.swh_caution_m} m caution threshold."
                        ),
                        evidence_ids=[swh_id] if swh_id else [],
                    )
                )

            wind_peak = weather.get("wind_peak_kt")
            wind_id = (weather.get("fields", {}).get("wind") or {}).get("evidence_id", "")
            if wind_peak is not None:
                if wind_peak >= settings.wind_danger_kt:
                    findings.append(
                        RiskFinding(
                            rule="wind_danger",
                            band=RiskBand.UNSAFE,
                            detail=(
                                f"Wind peaks at {wind_peak:.0f} kt, at or above the "
                                f"{settings.wind_danger_kt} kt near-gale threshold."
                            ),
                            evidence_ids=[wind_id] if wind_id else [],
                        )
                    )
                elif wind_peak >= settings.wind_caution_kt:
                    findings.append(
                        RiskFinding(
                            rule="wind_caution",
                            band=RiskBand.CAUTION,
                            detail=(
                                f"Wind peaks at {wind_peak:.0f} kt, in the squally "
                                "band where IMD normally advises caution."
                            ),
                            evidence_ids=[wind_id] if wind_id else [],
                        )
                    )
                else:
                    findings.append(
                        RiskFinding(
                            rule="wind_ok",
                            band=RiskBand.SAFE,
                            detail=f"Wind stays at or below {wind_peak:.0f} kt.",
                            evidence_ids=[wind_id] if wind_id else [],
                        )
                    )

            gust_peak = weather.get("gust_peak_kt")
            if gust_peak is not None and gust_peak >= settings.gust_danger_kt:
                gust_id = (weather.get("fields", {}).get("gust") or {}).get(
                    "evidence_id", ""
                )
                findings.append(
                    RiskFinding(
                        rule="gust_danger",
                        band=RiskBand.UNSAFE,
                        detail=(
                            f"Gusts reach {gust_peak:.0f} kt. Gusts of this strength "
                            "knock a small boat down even when the mean wind looks "
                            "manageable."
                        ),
                        evidence_ids=[gust_id] if gust_id else [],
                    )
                )

            convective = weather.get("convective_risk") or {}
            if convective.get("band") in ("high", "moderate"):
                cape_id = (weather.get("fields", {}).get("cape") or {}).get(
                    "evidence_id", ""
                )
                rain_id = (weather.get("fields", {}).get("rain") or {}).get(
                    "evidence_id", ""
                )
                findings.append(
                    RiskFinding(
                        rule="convective_risk",
                        band=(
                            RiskBand.UNSAFE
                            if convective["band"] == "high"
                            else RiskBand.CAUTION
                        ),
                        detail=(
                            f"Thunderstorm risk {convective['band']}: "
                            f"{convective['explanation']} (CAPE "
                            f"{convective['cape_j_per_kg']:.0f} J/kg). This is a "
                            "derived indicator, not an observed lightning strike."
                        ),
                        evidence_ids=[i for i in (cape_id, rain_id) if i],
                    )
                )

            cyclone = hazards.get("cyclone")
            if cyclone:
                distance = cyclone["distance_km"]
                if distance <= 300:
                    band = RiskBand.UNSAFE
                elif distance <= 800:
                    band = RiskBand.CAUTION
                else:
                    band = RiskBand.SAFE
                findings.append(
                    RiskFinding(
                        rule="tropical_cyclone_distance",
                        band=band,
                        detail=(
                            f"Tropical cyclone {cyclone['name']} is about "
                            f"{distance:.0f} km {cyclone['bearing']} of this "
                            "position. Confirm against the RSMC New Delhi bulletin, "
                            "which is the official authority."
                        ),
                        evidence_ids=[cyclone["evidence_id"]],
                    )
                )

            if hazards.get("imd_warnings"):
                findings.append(
                    RiskFinding(
                        rule="imd_warning_in_force",
                        band=RiskBand.CAUTION,
                        detail=(
                            "IMD's warning page currently carries sea-state or "
                            "fishermen advisory text. Its wording overrides any "
                            "model number quoted here: "
                            + hazards["imd_warnings"][0][:220]
                        ),
                        evidence_ids=["imd-fishermen-warning"],
                    )
                )

            current = (ocean.get("fields", {}).get("current") or {}).get("value")
            if current is not None and current >= settings.current_caution_cms:
                current_id = (ocean.get("fields", {}).get("current") or {}).get(
                    "evidence_id", ""
                )
                findings.append(
                    RiskFinding(
                        rule="strong_current",
                        band=RiskBand.CAUTION,
                        detail=(
                            f"Surface current about {current:.0f} cm/s. Allow for "
                            "set and drift, and for extra fuel on the return leg."
                        ),
                        evidence_ids=[current_id] if current_id else [],
                    )
                )

            for zone in geo.get("zones", []):
                if zone["status"] == "inside":
                    findings.append(
                        RiskFinding(
                            rule="inside_restricted_zone",
                            band=RiskBand.UNSAFE if zone["danger"] else RiskBand.CAUTION,
                            detail=(
                                f"This position is inside {zone['name']}. "
                                f"{zone['advisory']}"
                            ),
                            evidence_ids=[zone["evidence_id"]],
                        )
                    )
                elif zone["status"] == "approaching" and zone["danger"]:
                    findings.append(
                        RiskFinding(
                            rule="approaching_boundary",
                            band=RiskBand.CAUTION,
                            detail=(
                                f"{zone['name']} is only {zone['distance_km']} km "
                                f"{zone['bearing']} of this position. "
                                f"{zone['advisory']}"
                            ),
                            evidence_ids=[zone["evidence_id"]],
                        )
                    )

            eez = geo.get("eez") or {}
            if eez.get("available") and not eez.get("inside_india_eez"):
                findings.append(
                    RiskFinding(
                        rule="outside_eez",
                        band=RiskBand.UNSAFE,
                        detail=(
                            "This position is outside the India EEZ, about "
                            f"{eez['distance_to_boundary_km']} km from the boundary. "
                            "Fishing here risks arrest and confiscation."
                        ),
                        evidence_ids=["eez-status"],
                    )
                )

            # TypeSafe AI Jev System-One Decision Integration
            swh_val = waves.get("swh_peak_m") or waves.get("swh_now_m")
            wind_val = weather.get("wind_peak_kt") or weather.get("wind_now_kt")
            gust_val = weather.get("gust_peak_kt")
            cyclone_present = bool(hazards.get("cyclone"))
            cyclone_distance = (hazards.get("cyclone") or {}).get("distance_km")
            imd_active = bool(hazards.get("imd_warnings"))

            min_zone_dist = None
            nearest_zone_name = None
            for z in geo.get("zones", []):
                d = z.get("distance_km")
                if d is not None and (min_zone_dist is None or d < min_zone_dist):
                    min_zone_dist = d
                    nearest_zone_name = z.get("name")

            jev_dec = await get_jev_engine().evaluate_safety(
                location_name=(ctx.location.name if ctx.location else "this position"),
                swh_m=swh_val,
                wind_kt=wind_val,
                gust_kt=gust_val,
                cyclone_alert=cyclone_present,
                cyclone_dist_km=cyclone_distance,
                nearest_boundary_dist_km=min_zone_dist,
                restricted_zone_name=nearest_zone_name,
                imd_warning_active=imd_active,
            )

            jev_band = {
                "SAFE": RiskBand.SAFE,
                "CAUTION": RiskBand.CAUTION,
                "UNSAFE": RiskBand.UNSAFE,
            }.get(jev_dec.safety_verdict, RiskBand.CAUTION)

            findings.append(
                RiskFinding(
                    rule=f"jev_{jev_dec.action.lower()}",
                    band=jev_band,
                    detail=(
                        f"Jev System-One judgment: {jev_dec.safety_verdict} "
                        f"({jev_dec.action}). Severity {jev_dec.risk_score:.0f}/100, "
                        f"boundary breach {jev_dec.breach_probability * 100:.0f}%, "
                        f"capsizing {jev_dec.capsizing_probability * 100:.0f}%, "
                        f"confidence {jev_dec.confidence:.2f}. This is a structured "
                        f"second opinion from {jev_dec.engine}; it can only make the "
                        "verdict more conservative, never less."
                    ),
                    evidence_ids=[],
                )
            )
            hazards["jev_decision"] = {
                "verdict": jev_dec.safety_verdict,
                "action": jev_dec.action,
                "risk_score": jev_dec.risk_score,
                "breach_probability": jev_dec.breach_probability,
                "capsizing_probability": jev_dec.capsizing_probability,
                "engine": jev_dec.engine,
                "model": jev_dec.model,
                "confidence": jev_dec.confidence,
                "verdict_probabilities": jev_dec.verdict_probabilities,
                "rationale": jev_dec.rationale,
                "input_tokens": jev_dec.input_tokens,
                "output_tokens": jev_dec.output_tokens,
                "cost_usd": jev_dec.cost_usd,
                "elapsed_ms": jev_dec.elapsed_ms,
            }

            band = RiskBand.UNKNOWN
            for finding in findings:
                if BAND_ORDER[finding.band] > BAND_ORDER[band]:
                    band = finding.band
            score = self._score(findings)

            ctx.risk = RiskAssessment(
                band=band,
                score=score,
                headline=self._headline(band, ctx),
                findings=findings,
                window_advice=self._window_advice(ctx),
                jev_decision=hazards["jev_decision"],
            )
            step.outcome = (
                f"verdict {band.value} (score {score:.0f}/100) from "
                f"{len(findings)} rules; worst: "
                + next(
                    (f.rule for f in findings if f.band == band),
                    "none",
                )
            )
            step.evidence_ids = [
                eid for f in findings for eid in f.evidence_ids if eid
            ]

            marker_kind = {
                RiskBand.SAFE: "good",
                RiskBand.CAUTION: "caution",
                RiskBand.UNSAFE: "danger",
                RiskBand.UNKNOWN: "info",
            }[band]
            ctx.findings.hazards["marker"] = MapMarker(
                lat=ctx.lat,
                lon=ctx.lon,
                label=(ctx.location.name if ctx.location else "your position"),
                kind=marker_kind,
                detail=ctx.risk.headline,
            ).model_dump()

    @staticmethod
    def _score(findings: list[RiskFinding]) -> float:
        weights = {
            RiskBand.SAFE: 5.0,
            RiskBand.CAUTION: 35.0,
            RiskBand.UNSAFE: 85.0,
            RiskBand.UNKNOWN: 20.0,
        }
        if not findings:
            return 0.0
        worst = max(weights[f.band] for f in findings)
        extra = sum(
            8.0 for f in findings if f.band in (RiskBand.CAUTION, RiskBand.UNSAFE)
        )
        return min(100.0, worst + min(extra, 15.0))

    @staticmethod
    def _headline(band: RiskBand, ctx: AgentContext) -> str:
        where = ctx.location.name if ctx.location else "this position"
        when = ctx.window.label if ctx.window else "now"
        return {
            RiskBand.SAFE: f"Yes, you can go out off {where} {when}.",
            RiskBand.CAUTION: f"You can go out off {where} {when}, but be careful.",
            RiskBand.UNSAFE: f"Do not go out off {where} {when}.",
            RiskBand.UNKNOWN: (
                f"I cannot say whether it is safe off {where} {when}. Not enough "
                "information came through."
            ),
        }[band]

    def _window_advice(self, ctx: AgentContext) -> str:
        """Point at the calmest part of the window, if the series allows it."""
        waves = ctx.findings.waves or {}
        swh_field = (waves.get("fields", {}) or {}).get("swh") or {}
        peak_time = swh_field.get("peak_time") or waves.get("swh_peak_time")
        peak = waves.get("swh_peak_m")
        now = waves.get("swh_now_m")
        if peak is None or now is None:
            return ""
        if peak - now > 0.4 and peak_time:
            return (
                f"The sea gets rougher as the day goes on, worst around {peak_time} "
                f"at about {peak:.1f} m. Go early if you go at all."
            )
        if now - peak > 0.4:
            return "The sea is settling down through the day."
        return ""
