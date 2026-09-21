"""Geospatial RAG.

Retrieval here is spatial rather than lexical: given a position (or a track), it
returns the GIS features that bear on it, ranked by distance, each with the
advisory text attached. That is what turns "which zones should I avoid?" into a
citable answer instead of a guess.

Layers:
  * `data/layers/marine_zones.geojson` - MPAs, ecologically sensitive areas,
    restricted areas, indicative IMBL segments (seeded, approximate).
  * India EEZ polygon from the Marine Regions WFS, disk cached.
  * the coastal gazetteer, used for nearest-harbour and shelter answers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from ..config import get_settings
from ..connectors.marine_regions import MarineRegionsConnector
from ..geo import gazetteer
from ..geo.geometry import (
    as_linestring,
    bearing_deg,
    compass,
    geodesic_distance_km,
    haversine_km,
    load_features,
    nearest_point_on,
)
from ..schemas import Provenance, Tier

log = logging.getLogger("orca.rag.geo")


@dataclass
class ZoneHit:
    zone_id: str
    name: str
    kind: str
    authority: str
    advisory: str
    distance_km: float
    inside: bool
    approaching: bool
    warn_buffer_km: float
    bearing_to_zone: str
    approximate: bool
    danger: bool = False

    @property
    def status(self) -> str:
        if self.inside:
            return "inside"
        if self.approaching:
            return "approaching"
        return "clear"


@dataclass
class HarbourHit:
    name: str
    lat: float
    lon: float
    state: str
    district: str
    kind: str
    distance_km: float
    bearing: str


class GeoRag:
    def __init__(self, marine_regions: MarineRegionsConnector | None = None) -> None:
        self.settings = get_settings()
        self.marine_regions = marine_regions or MarineRegionsConnector()
        self._zones: list[tuple[dict[str, Any], BaseGeometry]] = []
        self._eez: BaseGeometry | None = None
        self._eez_loaded = False

    # -------------------------------------------------------------- loading #

    def load_zones(self) -> int:
        path = Path(self.settings.layers_dir) / "marine_zones.geojson"
        if not path.exists():
            log.warning("zone layer %s missing; geofencing disabled", path)
            return 0
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("zone layer unreadable: %s", exc)
            return 0
        self._zones = load_features(payload)
        return len(self._zones)

    async def ensure_eez(self) -> BaseGeometry | None:
        if self._eez_loaded:
            return self._eez
        self._eez_loaded = True
        payload = await self.marine_regions.india_eez()
        if not payload:
            return None
        features = load_features(payload)
        if not features:
            return None
        geom = features[0][1]
        for _, extra in features[1:]:
            geom = geom.union(extra)
        self._eez = geom
        return self._eez

    # ------------------------------------------------------------ retrieval #

    def zones_near(
        self, lat: float, lon: float, radius_km: float = 60.0
    ) -> list[ZoneHit]:
        if not self._zones:
            self.load_zones()
        point = Point(lon, lat)
        hits: list[ZoneHit] = []
        for props, geom in self._zones:
            inside = bool(geom.geom_type in ("Polygon", "MultiPolygon") and geom.contains(point))
            distance = 0.0 if inside else geodesic_distance_km(lat, lon, geom)
            buffer_km = float(props.get("warn_buffer_km", 5) or 5)
            if not inside and distance > max(radius_km, buffer_km):
                continue
            near_lat, near_lon = nearest_point_on(lat, lon, geom)
            hits.append(
                ZoneHit(
                    zone_id=str(props.get("id", props.get("name", "zone"))),
                    name=str(props.get("name", "unnamed zone")),
                    kind=str(props.get("kind", "zone")),
                    authority=str(props.get("authority", "")),
                    advisory=str(props.get("advisory", "")),
                    distance_km=round(distance, 1),
                    inside=inside,
                    approaching=(not inside and distance <= buffer_km),
                    warn_buffer_km=buffer_km,
                    bearing_to_zone=compass(bearing_deg(lat, lon, near_lat, near_lon)),
                    approximate=bool(props.get("approximate", True)),
                    danger=bool(props.get("danger", False)),
                )
            )
        hits.sort(key=lambda h: (not h.inside, h.distance_km))
        return hits

    def zones_along_track(
        self, track: list[tuple[float, float]], corridor_km: float = 10.0
    ) -> list[ZoneHit]:
        """Zones a planned track enters or passes close to."""
        if not self._zones:
            self.load_zones()
        if len(track) < 2:
            return []
        line = as_linestring(track)
        hits: list[ZoneHit] = []
        for props, geom in self._zones:
            if geom.intersects(line):
                distance = 0.0
                inside = True
            else:
                mid = track[len(track) // 2]
                distance = min(
                    geodesic_distance_km(la, lo, geom) for la, lo in track
                )
                inside = False
                if distance > corridor_km:
                    continue
                del mid
            buffer_km = float(props.get("warn_buffer_km", 5) or 5)
            first_lat, first_lon = track[0]
            near_lat, near_lon = nearest_point_on(first_lat, first_lon, geom)
            hits.append(
                ZoneHit(
                    zone_id=str(props.get("id", "zone")),
                    name=str(props.get("name", "unnamed zone")),
                    kind=str(props.get("kind", "zone")),
                    authority=str(props.get("authority", "")),
                    advisory=str(props.get("advisory", "")),
                    distance_km=round(distance, 1),
                    inside=inside,
                    approaching=distance <= buffer_km,
                    warn_buffer_km=buffer_km,
                    bearing_to_zone=compass(
                        bearing_deg(first_lat, first_lon, near_lat, near_lon)
                    ),
                    approximate=bool(props.get("approximate", True)),
                    danger=bool(props.get("danger", False)),
                )
            )
        hits.sort(key=lambda h: (not h.inside, h.distance_km))
        return hits

    async def eez_status(self, lat: float, lon: float) -> dict[str, Any]:
        geom = await self.ensure_eez()
        if geom is None:
            return {
                "available": False,
                "note": "India EEZ polygon unavailable (upstream WFS unreachable)",
            }
        point = Point(lon, lat)
        inside = bool(geom.contains(point))
        distance = geodesic_distance_km(lat, lon, geom.boundary)
        near_lat, near_lon = nearest_point_on(lat, lon, geom.boundary)
        return {
            "available": True,
            "inside_india_eez": inside,
            "distance_to_boundary_km": round(distance, 1),
            "bearing_to_boundary": compass(bearing_deg(lat, lon, near_lat, near_lon)),
            "boundary_point": {"lat": round(near_lat, 4), "lon": round(near_lon, 4)},
        }

    @staticmethod
    def nearest_harbours(
        lat: float, lon: float, limit: int = 3, harbours_only: bool = True
    ) -> list[HarbourHit]:
        candidates = [
            p
            for p in gazetteer.PLACES
            if (p.kind in ("harbour", "landing-centre") if harbours_only else True)
        ]
        scored = [
            (haversine_km(lat, lon, p.lat, p.lon), p) for p in candidates
        ]
        scored.sort(key=lambda pair: pair[0])
        return [
            HarbourHit(
                name=p.name,
                lat=p.lat,
                lon=p.lon,
                state=p.state,
                district=p.district,
                kind=p.kind,
                distance_km=round(distance, 1),
                bearing=compass(bearing_deg(lat, lon, p.lat, p.lon)),
            )
            for distance, p in scored[:limit]
        ]

    # ----------------------------------------------------------- provenance #

    @staticmethod
    def zone_provenance(hit: ZoneHit) -> Provenance:
        return Provenance(
            agency=hit.authority or "ORCA seeded zone layer",
            dataset=f"{hit.name} ({hit.kind})",
            tier=Tier.SEED,
            url="file://data/layers/marine_zones.geojson",
            access_method="geospatial-rag",
            official=False,
            caveat=(
                "APPROXIMATE simplified boundary seeded for the prototype. "
                "Indicative only - not survey grade, not for navigation. "
                "Replace with gazetted MoEFCC / NHO / INCOIS layers in production."
            ),
        )

    def eez_provenance(self) -> Provenance:
        return self.marine_regions.provenance()

    @staticmethod
    def gazetteer_provenance() -> Provenance:
        return Provenance(
            agency="ORCA coastal gazetteer",
            dataset="Indian fishing harbours and landing centres",
            tier=Tier.SEED,
            url="file://backend/orca/geo/gazetteer.py",
            access_method="gazetteer",
            official=False,
            caveat="curated coordinate list, positions nudged offshore of each harbour",
        )

    def stats(self) -> dict[str, Any]:
        if not self._zones:
            self.load_zones()
        kinds: dict[str, int] = {}
        for props, _ in self._zones:
            kind = str(props.get("kind", "zone"))
            kinds[kind] = kinds.get(kind, 0) + 1
        return {
            "zone_features": len(self._zones),
            "zones_by_kind": kinds,
            "eez_loaded": self._eez is not None,
            "gazetteer_places": len(gazetteer.PLACES),
        }
