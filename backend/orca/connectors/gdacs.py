"""GDACS connector - tropical cyclone alert geometry.

IMD/RSMC New Delhi is the authority for cyclones in the North Indian Ocean, but
it publishes bulletins as HTML and images, not as features. GDACS (JRC, European
Commission) republishes active TC events as GeoJSON with alert levels, which
gives ORCA a *geometry* it can do distance reasoning against.

Rule enforced in the risk agent: GDACS supplies geometry and distance; IMD
supplies the wording and the official alert status. If they disagree, IMD wins
and the response says so.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from ..schemas import Provenance, Tier
from .base import HttpConnector

NIO_BBOX = (30.0, 100.0, -5.0, 30.0)  # west, east, south, north


class GdacsConnector(HttpConnector):
    source_name = "gdacs"

    async def tropical_cyclones(self, days_back: int = 10) -> list[dict[str, Any]]:
        since = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime(
            "%Y-%m-%d"
        )
        payload = await self.get_json(
            self.settings.gdacs_base,
            params={"eventlist": "TC", "fromDate": since},
            ttl=1800,
        )
        if not payload:
            return []
        features = payload.get("features") if isinstance(payload, dict) else None
        if not features:
            return []
        events: list[dict[str, Any]] = []
        for feature in features:
            props = feature.get("properties", {}) or {}
            geom = feature.get("geometry", {}) or {}
            lat, lon = _centroid(geom)
            if lat is None or lon is None:
                continue
            west, east, south, north = NIO_BBOX
            if not (west <= lon <= east and south <= lat <= north):
                continue
            events.append(
                {
                    "event_id": props.get("eventid"),
                    "name": props.get("eventname") or props.get("name") or "unnamed",
                    "alert_level": (props.get("alertlevel") or "").lower(),
                    "severity": props.get("severitydata", {}).get("severitytext", ""),
                    "from_date": props.get("fromdate"),
                    "to_date": props.get("todate"),
                    "lat": lat,
                    "lon": lon,
                    "url": (props.get("url") or {}).get("report", ""),
                    "geometry": geom,
                }
            )
        return events

    async def nearest_cyclone(
        self, lat: float, lon: float
    ) -> tuple[dict[str, Any], float] | None:
        events = await self.tropical_cyclones()
        if not events:
            return None
        scored = [
            (event, _haversine_km(lat, lon, event["lat"], event["lon"]))
            for event in events
        ]
        scored.sort(key=lambda pair: pair[1])
        return scored[0]

    def provenance(self, dataset: str = "GDACS tropical cyclone events") -> Provenance:
        return Provenance(
            agency="GDACS (JRC, European Commission)",
            dataset=dataset,
            tier=Tier.FALLBACK,
            url=self.settings.gdacs_base,
            access_method="http-json",
            official=False,
            caveat=(
                "used for cyclone position and distance geometry only; "
                "IMD / RSMC New Delhi remains the official authority for the "
                "North Indian Ocean and overrides this on alert status"
            ),
        )


def _centroid(geom: dict[str, Any]) -> tuple[float | None, float | None]:
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if not coords:
        return None, None
    if gtype == "Point":
        return float(coords[1]), float(coords[0])
    points: list[tuple[float, float]] = []

    def walk(node: Any) -> None:
        if (
            isinstance(node, (list, tuple))
            and len(node) == 2
            and all(isinstance(v, (int, float)) for v in node)
        ):
            points.append((float(node[1]), float(node[0])))
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(coords)
    if not points:
        return None, None
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))
