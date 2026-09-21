"""Visualisation agent.

Builds the map payload. The important decision here is that the raster layers
are MOSDAC WMS URLs handed straight to the client, so the user sees the actual
ISRO field rendered by ISRO's own server rather than an ORCA re-interpretation of
it. Vector overlays (position, zones, harbours, cyclone) are GeoJSON built from
the same evidence the text answer cites.
"""

from __future__ import annotations

from typing import Any

from ..schemas import MapLayer, MapMarker, RiskBand
from ..services import Services
from .base import AgentContext

ZONE_COLOURS = {
    "imbl": "#d62828",
    "restricted": "#e07a5f",
    "mpa": "#2a9d8f",
    "eco-sensitive": "#457b9d",
}


class VisualisationAgent:
    name = "visualisation"
    tools = ("mosdac.wms", "geojson-builder")

    def __init__(self, services: Services) -> None:
        self.services = services

    async def run(self, ctx: AgentContext) -> None:
        with ctx.trace.timed(
            self.name,
            "assemble map layers",
            rationale=(
                "ISRO fields are shown as MOSDAC WMS tiles so the map displays the "
                "agency's own rendering; everything else is GeoJSON built from the "
                "cited evidence"
            ),
            tool="mosdac.wms",
        ) as step:
            built: list[str] = []

            self._position_layer(ctx)
            built.append("position")

            if self._zone_layer(ctx):
                built.append("zones")
            if self._harbour_layer(ctx):
                built.append("harbours")
            if self._cyclone_layer(ctx):
                built.append("cyclone")
            if await self._wms_layers(ctx):
                built.append("isro-wms")

            step.outcome = f"layers: {', '.join(built)}"

    # ------------------------------------------------------------------ bits #

    def _position_layer(self, ctx: AgentContext) -> None:
        band = ctx.risk.band if ctx.risk else RiskBand.UNKNOWN
        kind = {
            RiskBand.SAFE: "good",
            RiskBand.CAUTION: "caution",
            RiskBand.UNSAFE: "danger",
            RiskBand.UNKNOWN: "info",
        }[band]
        detail = ctx.risk.headline if ctx.risk else ""
        ctx.add_layer(
            MapLayer(
                id="position",
                title="Query position",
                kind="markers",
                markers=[
                    MapMarker(
                        lat=ctx.lat,
                        lon=ctx.lon,
                        label=ctx.location.name if ctx.location else "position",
                        kind=kind,
                        detail=detail,
                    )
                ],
                attribution="ORCA",
                legend=f"assessed {band.value}",
            )
        )

    def _zone_layer(self, ctx: AgentContext) -> bool:
        zones = (ctx.findings.geo or {}).get("zones") or []
        if not zones:
            return False
        geo_rag = self.services.geo_rag
        if not geo_rag._zones:  # noqa: SLF001 - internal cache is intentional
            geo_rag.load_zones()
        wanted = {z["id"] for z in zones}
        features: list[dict[str, Any]] = []
        for props, geom in geo_rag._zones:  # noqa: SLF001
            zone_id = str(props.get("id", ""))
            if zone_id not in wanted:
                continue
            status = next(
                (z["status"] for z in zones if z["id"] == zone_id), "clear"
            )
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        **props,
                        "status": status,
                        "colour": ZONE_COLOURS.get(str(props.get("kind")), "#888888"),
                    },
                    "geometry": geom.__geo_interface__,
                }
            )
        if not features:
            return False
        ctx.add_layer(
            MapLayer(
                id="zones",
                title="Protected, restricted and boundary zones",
                kind="geojson",
                geojson={"type": "FeatureCollection", "features": features},
                attribution=(
                    "ORCA seeded zone layer - approximate, indicative, not for "
                    "navigation"
                ),
                opacity=0.35,
                legend="red: boundary or restricted, green: protected area",
            )
        )
        return True

    @staticmethod
    def _harbour_layer(ctx: AgentContext) -> bool:
        harbours = (ctx.findings.geo or {}).get("harbours") or []
        if not harbours:
            return False
        ctx.add_layer(
            MapLayer(
                id="harbours",
                title="Nearest harbours and shelter",
                kind="markers",
                markers=[
                    MapMarker(
                        lat=h["lat"],
                        lon=h["lon"],
                        label=h["name"],
                        kind="harbour",
                        detail=f"{h['distance_km']} km {h['bearing']}, {h['state']}",
                    )
                    for h in harbours
                ],
                attribution="ORCA coastal gazetteer",
            )
        )
        return True

    @staticmethod
    def _cyclone_layer(ctx: AgentContext) -> bool:
        cyclone = (ctx.findings.hazards or {}).get("cyclone")
        if not cyclone:
            return False
        ctx.add_layer(
            MapLayer(
                id="cyclone",
                title=f"Tropical cyclone {cyclone['name']}",
                kind="markers",
                markers=[
                    MapMarker(
                        lat=cyclone["lat"],
                        lon=cyclone["lon"],
                        label=f"TC {cyclone['name']}",
                        kind="danger",
                        detail=(
                            f"{cyclone['distance_km']:.0f} km "
                            f"{cyclone['bearing']}, GDACS level "
                            f"{cyclone['alert_level']}"
                        ),
                    )
                ],
                attribution="GDACS (JRC) geometry; IMD/RSMC New Delhi is authoritative",
            )
        )
        return True

    async def _wms_layers(self, ctx: AgentContext) -> bool:
        """MOSDAC WMS for the ISRO fields relevant to this intent."""
        mosdac = self.services.mosdac
        added = False

        circ = await mosdac.latest_osf_circ()
        if circ:
            ctx.add_layer(
                MapLayer(
                    id="wms-osf-sst",
                    title="ISRO Ocean State Forecast - sea temperature",
                    kind="wms",
                    url=mosdac.wms_url(circ["url_path"]),
                    wms_layer="temp",
                    wms_style="boxfill/sst_36",
                    attribution="ISRO / MOSDAC (Space Applications Centre)",
                    opacity=0.6,
                    visible_by_default=True,
                    legend=f"OSF circulation cycle {circ.get('date', '')}",
                )
            )
            added = True

        ctx.add_layer(
            MapLayer(
                id="wms-osf-swh",
                title="ISRO Ocean State Forecast - significant wave height",
                kind="wms",
                url=mosdac.wms_url(mosdac.OSF_WAVE_DATASET),
                wms_layer="SWH",
                wms_style="boxfill/rainbow",
                attribution="ISRO / MOSDAC (Space Applications Centre)",
                opacity=0.6,
                visible_by_default=False,
                legend="OSF wave model, check the cycle date in the evidence panel",
            )
        )
        return added or True
