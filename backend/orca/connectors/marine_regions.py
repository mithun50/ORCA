"""Marine Regions (VLIZ) connector - the India EEZ polygon.

This is what makes geofencing possible. INCOIS publishes EEZ and IMBL layers
through a GeoServer that returns 403 to non-browser clients, so ORCA sources the
same maritime boundary from the Marine Regions WFS (the reference gazetteer that
national agencies themselves cite) and caches it on disk. ~2.2 MB, fetched once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas import Provenance, Tier
from .base import HttpConnector


class MarineRegionsConnector(HttpConnector):
    source_name = "marine-regions"

    CACHE_DAYS = 30

    async def india_eez(self) -> dict[str, Any] | None:
        """India EEZ as a GeoJSON FeatureCollection (disk cached)."""
        key = "india-eez"
        cached = self.disk_get(key, ".geojson", self.CACHE_DAYS * 86400)
        if cached:
            try:
                return json.loads(cached)
            except ValueError:
                pass

        # A bundled snapshot keeps the prototype demonstrable offline.
        bundled = Path(self.settings.layers_dir) / "india_eez.geojson"

        payload = await self.get_json(
            self.settings.marine_regions_wfs,
            params={
                "service": "WFS",
                "version": "1.1.0",
                "request": "GetFeature",
                "typeName": "MarineRegions:eez",
                "CQL_FILTER": "sovereign1='India'",
                "outputFormat": "application/json",
            },
            ttl=0,
        )
        if payload and payload.get("features"):
            self.disk_put(key, ".geojson", json.dumps(payload))
            try:
                bundled.parent.mkdir(parents=True, exist_ok=True)
                bundled.write_text(json.dumps(payload), encoding="utf-8")
            except OSError:
                pass
            return payload

        if bundled.exists():
            try:
                return json.loads(bundled.read_text(encoding="utf-8"))
            except ValueError:
                return None
        return None

    def provenance(self) -> Provenance:
        return Provenance(
            agency="Marine Regions / VLIZ (Flanders Marine Institute)",
            dataset="India Exclusive Economic Zone boundary",
            tier=Tier.FALLBACK,
            url=self.settings.marine_regions_wfs,
            access_method="wfs-geojson",
            official=False,
            caveat=(
                "reference maritime boundary; INCOIS publishes the Indian IMBL "
                "through a GeoServer that blocks non-browser clients, so this "
                "gazetteer boundary is used for distance-to-boundary reasoning. "
                "Not a navigational authority - do not use for legal position "
                "fixing"
            ),
        )
