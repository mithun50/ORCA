"""Geospatial reasoning and route planning agents.

The geospatial agent answers the questions a chart would: am I inside anything I
should not be, how far is the nearest boundary, where is shelter. The route
planner then samples conditions along candidate tracks and scores them, so
"safest route" is a comparison of alternatives rather than a single suggestion.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..geo.geometry import (
    bearing_deg,
    compass,
    great_circle_waypoints,
    haversine_km,
    offset_track,
    track_length_km,
)
from ..schemas import Evidence, MapLayer, MapMarker, Tier
from ..services import Services
from .base import AgentContext


class GeospatialReasoningAgent:
    name = "geospatial-reasoning"
    tools = ("geo_rag.zones_near", "geo_rag.eez_status", "geo_rag.nearest_harbours")

    def __init__(self, services: Services) -> None:
        self.services = services

    async def run(self, ctx: AgentContext) -> None:
        geo = self.services.geo_rag
        out: dict[str, Any] = {}

        with ctx.trace.timed(
            self.name,
            "check protected, restricted and boundary zones",
            rationale=(
                "geofencing is retrieval by distance: every zone within the "
                "search radius is returned with its advisory text, so the answer "
                "can cite the rule rather than paraphrase it"
            ),
            tool="geo_rag.zones_near",
            tool_args={"lat": round(ctx.lat, 3), "lon": round(ctx.lon, 3),
                       "radius_km": 60},
        ) as step:
            zones = geo.zones_near(ctx.lat, ctx.lon, radius_km=60.0)
            hits = []
            for zone in zones:
                evidence = Evidence(
                    id=f"zone-{zone.zone_id}",
                    label=f"{zone.name} - {zone.status}",
                    value=zone.distance_km,
                    unit="km",
                    at_lat=round(ctx.lat, 4),
                    at_lon=round(ctx.lon, 4),
                    text=zone.advisory,
                    provenance=geo.zone_provenance(zone),
                )
                ctx.evidence.add(evidence)
                hits.append(
                    {
                        "id": zone.zone_id,
                        "name": zone.name,
                        "kind": zone.kind,
                        "status": zone.status,
                        "distance_km": zone.distance_km,
                        "bearing": zone.bearing_to_zone,
                        "advisory": zone.advisory,
                        "authority": zone.authority,
                        "danger": zone.danger,
                        "evidence_id": evidence.id,
                    }
                )
            out["zones"] = hits
            inside = [z for z in hits if z["status"] == "inside"]
            approaching = [z for z in hits if z["status"] == "approaching"]
            step.evidence_ids = [z["evidence_id"] for z in hits]
            step.outcome = (
                f"{len(inside)} zone(s) entered, {len(approaching)} being "
                f"approached, {len(hits)} within 60 km"
            )
            if inside or approaching:
                step.status = "degraded" if inside else "ok"

        with ctx.trace.timed(
            self.name,
            "check position against the India EEZ",
            rationale=(
                "crossing the EEZ or an international maritime boundary is the "
                "single highest-consequence geofence for Indian fishermen"
            ),
            tool="geo_rag.eez_status",
        ) as step:
            eez = await geo.eez_status(ctx.lat, ctx.lon)
            out["eez"] = eez
            if not eez.get("available"):
                step.status = "degraded"
                step.outcome = str(eez.get("note", "EEZ layer unavailable"))
            else:
                evidence = Evidence(
                    id="eez-status",
                    label=(
                        "inside the India EEZ"
                        if eez["inside_india_eez"]
                        else "outside the India EEZ"
                    ),
                    value=eez["distance_to_boundary_km"],
                    unit="km to the boundary",
                    at_lat=round(ctx.lat, 4),
                    at_lon=round(ctx.lon, 4),
                    provenance=geo.eez_provenance(),
                )
                ctx.evidence.add(evidence)
                step.evidence_ids = [evidence.id]
                step.outcome = (
                    f"{'inside' if eez['inside_india_eez'] else 'OUTSIDE'} the "
                    f"India EEZ, {eez['distance_to_boundary_km']} km from the "
                    f"boundary ({eez['bearing_to_boundary']})"
                )

        with ctx.trace.timed(
            self.name,
            "find nearest harbours for shelter",
            rationale="a safety answer is more useful when it names where to run to",
            tool="geo_rag.nearest_harbours",
        ) as step:
            harbours = geo.nearest_harbours(ctx.lat, ctx.lon, limit=3)
            out["harbours"] = [
                {
                    "name": h.name,
                    "distance_km": h.distance_km,
                    "bearing": h.bearing,
                    "state": h.state,
                    "kind": h.kind,
                    "lat": h.lat,
                    "lon": h.lon,
                }
                for h in harbours
            ]
            if harbours:
                evidence = Evidence(
                    id="nearest-harbour",
                    label=f"nearest harbour: {harbours[0].name}",
                    value=harbours[0].distance_km,
                    unit="km",
                    text=f"bearing {harbours[0].bearing}",
                    provenance=geo.gazetteer_provenance(),
                )
                ctx.evidence.add(evidence)
                step.evidence_ids = [evidence.id]
                step.outcome = (
                    f"{harbours[0].name} at {harbours[0].distance_km} km "
                    f"({harbours[0].bearing})"
                )
            else:
                step.status = "degraded"
                step.outcome = "no harbour found in the gazetteer"

        ctx.findings.geo = out


class RoutePlannerAgent:
    """Scores the direct track against bowed alternatives."""

    name = "route-planner"
    tools = ("timeseries_rag.wave_state", "geo_rag.zones_along_track")

    #: candidate lateral offsets at the midpoint, in km
    OFFSETS_KM = (0.0, 25.0, -25.0, 50.0, -50.0)
    SAMPLES = 5

    def __init__(self, services: Services) -> None:
        self.services = services

    async def run(self, ctx: AgentContext) -> None:
        if ctx.destination is None:
            ctx.trace.add(
                self.name,
                "skip route planning",
                rationale="no destination could be resolved from the request",
                status="skipped",
            )
            return

        origin = (ctx.lat, ctx.lon)
        target = (ctx.destination.lat, ctx.destination.lon)
        direct_km = haversine_km(*origin, *target)

        with ctx.trace.timed(
            self.name,
            "build candidate tracks",
            rationale=(
                "the direct great-circle track plus tracks bowed to either side, "
                "so the recommendation is the best of several options rather than "
                "the only one considered"
            ),
            tool="geo.offset_track",
            tool_args={
                "from": ctx.location.name if ctx.location else "origin",
                "to": ctx.destination.name,
                "direct_km": round(direct_km, 1),
                "candidates": len(self.OFFSETS_KM),
            },
        ) as step:
            candidates: list[dict[str, Any]] = []
            for offset in self.OFFSETS_KM:
                if abs(offset) < 1.0:
                    track = great_circle_waypoints(*origin, *target, segments=self.SAMPLES)
                    label = "direct track"
                else:
                    track = offset_track(
                        *origin, *target, offset_km=offset, segments=self.SAMPLES
                    )
                    side = "seaward" if offset > 0 else "landward"
                    label = f"{abs(offset):.0f} km {side} of the direct track"
                candidates.append(
                    {
                        "offset_km": offset,
                        "label": label,
                        "track": track,
                        "length_km": round(track_length_km(track), 1),
                    }
                )
            step.outcome = f"{len(candidates)} candidate tracks built"

        with ctx.trace.timed(
            self.name,
            "sample sea state along each track",
            rationale=(
                "wave height is sampled at waypoints on every candidate; the "
                "worst point on a track decides its score, because that is what "
                "a small boat actually has to survive"
            ),
            tool="timeseries_rag.wave_state",
            tool_args={"waypoints_per_track": self.SAMPLES + 1},
        ) as step:
            rag = self.services.timeseries_rag
            for candidate in candidates:
                sampled = await asyncio.gather(
                    *(
                        rag.wave_state(la, lo, ctx.start, ctx.end)
                        for la, lo in candidate["track"]
                    )
                )
                heights: list[float] = []
                for waves in sampled:
                    swh = waves.get("swh") or waves.get("swh_isro")
                    if swh is None:
                        continue
                    value = swh.max()
                    if value is not None:
                        heights.append(value)
                candidate["max_swh_m"] = round(max(heights), 2) if heights else None
                candidate["mean_swh_m"] = (
                    round(sum(heights) / len(heights), 2) if heights else None
                )
                candidate["samples"] = len(heights)
            sampled_ok = sum(1 for c in candidates if c["max_swh_m"] is not None)
            step.outcome = f"sea state sampled on {sampled_ok}/{len(candidates)} tracks"
            if sampled_ok == 0:
                step.status = "degraded"

        with ctx.trace.timed(
            self.name,
            "check each track for zone conflicts",
            rationale="a shorter track that crosses the IMBL is not a safer track",
            tool="geo_rag.zones_along_track",
        ) as step:
            conflicts_found = 0
            for candidate in candidates:
                zones = self.services.geo_rag.zones_along_track(
                    candidate["track"], corridor_km=12.0
                )
                candidate["zone_conflicts"] = [
                    {
                        "name": z.name,
                        "kind": z.kind,
                        "status": z.status,
                        "distance_km": z.distance_km,
                        "danger": z.danger,
                        "advisory": z.advisory,
                    }
                    for z in zones
                ]
                conflicts_found += len(candidate["zone_conflicts"])
            step.outcome = f"{conflicts_found} track/zone interactions found"

        scored = sorted(candidates, key=self._score)
        best = scored[0]
        settings = self.services.settings

        evidence = Evidence(
            id="route-best",
            label=f"recommended track: {best['label']}",
            value=best.get("max_swh_m"),
            unit="m peak SWH",
            text=(
                f"{best['length_km']} km, "
                f"{len(best['zone_conflicts'])} zone conflict(s)"
            ),
            provenance=self.services.mosdac.provenance(
                "OSF_WAVE/SAC_OSF_WAVE_10KM.nc",
                title="route scored on wave height sampled along each track",
                access_method="derived",
                caveat=(
                    "ORCA-derived route comparison. It weighs sea state and zone "
                    "conflicts only. It does not know about bathymetry, shoals, "
                    "traffic separation schemes, your vessel's draught or its "
                    "seakeeping. Use it as advice, alongside a chart."
                ),
            ),
        )
        evidence.provenance.tier = Tier.DERIVED
        evidence.provenance.official = False
        ctx.evidence.add(evidence)

        ctx.findings.route = {
            "origin": {
                "name": ctx.location.name if ctx.location else "origin",
                "lat": ctx.lat,
                "lon": ctx.lon,
            },
            "destination": {
                "name": ctx.destination.name,
                "lat": ctx.destination.lat,
                "lon": ctx.destination.lon,
            },
            "direct_km": round(direct_km, 1),
            "initial_bearing": compass(bearing_deg(*origin, *target)),
            "recommended": {
                "label": best["label"],
                "length_km": best["length_km"],
                "max_swh_m": best.get("max_swh_m"),
                "mean_swh_m": best.get("mean_swh_m"),
                "zone_conflicts": best["zone_conflicts"],
                "extra_distance_km": round(best["length_km"] - direct_km, 1),
            },
            "alternatives": [
                {
                    "label": c["label"],
                    "length_km": c["length_km"],
                    "max_swh_m": c.get("max_swh_m"),
                    "zone_conflicts": len(c["zone_conflicts"]),
                    "score": round(self._score(c), 2),
                }
                for c in scored
            ],
            "swh_caution_m": settings.swh_caution_m,
            "evidence_id": evidence.id,
        }

        for index, candidate in enumerate(scored[:3]):
            ctx.add_layer(
                MapLayer(
                    id=f"route-{index}",
                    title=(
                        f"{'Recommended' if index == 0 else 'Alternative'}: "
                        f"{candidate['label']}"
                    ),
                    kind="line",
                    geojson={
                        "type": "Feature",
                        "properties": {
                            "label": candidate["label"],
                            "max_swh_m": candidate.get("max_swh_m"),
                            "recommended": index == 0,
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [lo, la] for la, lo in candidate["track"]
                            ],
                        },
                    },
                    attribution="ORCA-derived track, scored on ISRO/fallback wave data",
                    visible_by_default=index == 0,
                    legend=(
                        f"peak SWH {candidate.get('max_swh_m')} m over "
                        f"{candidate['length_km']} km"
                    ),
                )
            )

    def _score(self, candidate: dict[str, Any]) -> float:
        """Lower is better: sea state dominates, then detour cost, then zones."""
        settings = self.services.settings
        swh = candidate.get("max_swh_m")
        if swh is None:
            swh_penalty = 5.0  # unknown conditions are not treated as safe
        else:
            swh_penalty = swh * 2.0
            if swh > settings.swh_danger_m:
                swh_penalty += 8.0
            elif swh > settings.swh_caution_m:
                swh_penalty += 3.0
        detour = abs(candidate["offset_km"]) / 50.0
        zone_penalty = 0.0
        for zone in candidate.get("zone_conflicts", []):
            if zone["danger"]:
                zone_penalty += 20.0 if zone["status"] == "inside" else 8.0
            else:
                zone_penalty += 6.0 if zone["status"] == "inside" else 1.5
        return swh_penalty + detour + zone_penalty
