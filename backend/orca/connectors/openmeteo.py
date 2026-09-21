"""Open-Meteo connector - the Tier 4 gap filler.

Used only where an Indian official machine endpoint does not exist or its cycle
has gone stale, and every value it produces is tagged
`tier4-fallback / official=False` so the UI can show the user that this number
is not from ISRO, INCOIS or IMD.

It covers three real gaps:

* **tide** - `sea_level_height_msl` is the only free tide/sea-level series we
  found without a key. INCOIS publishes tide tables as PDFs only.
* **live waves** - when the MOSDAC OSF_WAVE cycle is older than
  `ORCA_STALE_HOURS`, the wave numbers here are the current ones.
* **convective proxy** - CAPE + precipitation, used to flag thunderstorm risk
  because IMD's Damini lightning network has no public API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..schemas import Provenance, Tier
from .base import HttpConnector

MARINE_HOURLY = (
    "wave_height",
    "wave_direction",
    "wave_period",
    "wind_wave_height",
    "swell_wave_height",
    "swell_wave_period",
    "sea_surface_temperature",
    "sea_level_height_msl",
    "ocean_current_velocity",
    "ocean_current_direction",
)

WEATHER_HOURLY = (
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "precipitation",
    "cape",
    "visibility",
    "cloud_cover",
    "temperature_2m",
    "relative_humidity_2m",
)


class OpenMeteoConnector(HttpConnector):
    source_name = "open-meteo"

    async def marine(
        self, lat: float, lon: float, forecast_days: int = 3
    ) -> dict[str, Any] | None:
        return await self.get_json(
            self.settings.openmeteo_marine_base,
            params={
                "latitude": f"{lat:.4f}",
                "longitude": f"{lon:.4f}",
                "hourly": ",".join(MARINE_HOURLY),
                "forecast_days": forecast_days,
                "timezone": "UTC",
            },
            ttl=1800,
        )

    async def weather(
        self, lat: float, lon: float, forecast_days: int = 3
    ) -> dict[str, Any] | None:
        return await self.get_json(
            self.settings.openmeteo_forecast_base,
            params={
                "latitude": f"{lat:.4f}",
                "longitude": f"{lon:.4f}",
                "hourly": ",".join(WEATHER_HOURLY),
                "forecast_days": forecast_days,
                "timezone": "UTC",
            },
            ttl=1800,
        )

    # ------------------------------------------------------------- reshaping #

    @staticmethod
    def to_series(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Flatten Open-Meteo's column-oriented hourly block into rows."""
        if not payload or "hourly" not in payload:
            return []
        hourly = payload["hourly"]
        times = hourly.get("time", [])
        units = payload.get("hourly_units", {})
        rows: list[dict[str, Any]] = []
        for i, stamp in enumerate(times):
            row: dict[str, Any] = {"time": _iso_z(stamp), "_units": units}
            for key, values in hourly.items():
                if key == "time":
                    continue
                row[key] = values[i] if i < len(values) else None
            rows.append(row)
        return rows

    def provenance(self, dataset: str, url: str = "") -> Provenance:
        return Provenance(
            agency="Open-Meteo (ECMWF/GFS/MFWAM derived)",
            dataset=dataset,
            tier=Tier.FALLBACK,
            url=url or self.settings.openmeteo_marine_base,
            access_method="http-json",
            official=False,
            caveat=(
                "non-Indian fallback source, used because no free official "
                "machine endpoint exists for this variable; treat ISRO/INCOIS/IMD "
                "products as authoritative where they disagree"
            ),
        )


def _iso_z(stamp: str) -> str:
    try:
        dt = datetime.fromisoformat(stamp)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return stamp
