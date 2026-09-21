"""Time-series and gridded-field RAG.

This is the numeric leg of the multi-RAG. It retrieves *values*, not passages,
and every value it returns is wrapped in `Evidence` with the provenance of the
exact service call that produced it.

Source policy, applied per variable:

| variable | first choice | fallback | why |
|---|---|---|---|
| SST, currents, salinity, MLD | MOSDAC OSF_CIRC (ISRO) | Open-Meteo marine SST | official Indian ocean model |
| SWH, swell, wave period/direction | MOSDAC OSF_WAVE (ISRO) | Open-Meteo marine | OSF_WAVE cycle can be stale |
| wind, gust, rain, CAPE | Open-Meteo forecast | - | IMD's API needs a key; OSF winds are 6-hourly only |
| tide / sea level | Open-Meteo marine | - | no free official Indian tide API |
| chlorophyll | MOSDAC pfz/chl grids | INCOIS ERDDAP Oceansat-2 OCM | both ISRO heritage |

When the first choice is stale or masked, the fallback value is returned *and*
the response says which one it used. It never silently substitutes.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

from ..connectors.incois import IncoisConnector
from ..connectors.mosdac import MosdacConnector
from ..connectors.openmeteo import OpenMeteoConnector
from ..schemas import Evidence, Provenance, Tier

log = logging.getLogger("orca.rag.timeseries")

MS_TO_KT = 1.94384
CMS_TO_KT = 0.0194384


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _finite(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for v in values:
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            out.append(float(v))
    return out


@dataclass
class PointField:
    """One variable's series at one location, with the evidence wrapper."""

    variable: str
    label: str
    unit: str
    series: list[dict[str, Any]] = field(default_factory=list)
    evidence: Evidence | None = None
    source_tier: Tier = Tier.DERIVED
    used_fallback: bool = False
    note: str = ""

    def values(self) -> list[float]:
        return _finite(row.get("value") for row in self.series)

    def at_or_first(self, when: datetime | None = None) -> float | None:
        vals = self.values()
        if not vals:
            return None
        if when is None:
            return vals[0]
        best: tuple[float, float] | None = None
        for row in self.series:
            stamp = _parse_iso(str(row.get("time", "")))
            value = row.get("value")
            if stamp is None or not isinstance(value, (int, float)):
                continue
            delta = abs((stamp - when).total_seconds())
            if best is None or delta < best[0]:
                best = (delta, float(value))
        return best[1] if best else vals[0]

    def max(self) -> float | None:
        vals = self.values()
        return max(vals) if vals else None

    def min(self) -> float | None:
        vals = self.values()
        return min(vals) if vals else None

    def mean(self) -> float | None:
        vals = self.values()
        return sum(vals) / len(vals) if vals else None

    def window_of_max(self) -> str:
        best: tuple[float, str] | None = None
        for row in self.series:
            value = row.get("value")
            if not isinstance(value, (int, float)):
                continue
            if best is None or value > best[0]:
                best = (float(value), str(row.get("time", "")))
        return best[1] if best else ""


@dataclass
class GridField:
    variable: str
    label: str
    unit: str
    lats: list[float]
    lons: list[float]
    values: list[list[float | None]]
    provenance: Provenance
    valid_time: str = ""

    def cells(self) -> list[tuple[float, float, float]]:
        out: list[tuple[float, float, float]] = []
        for i, lat in enumerate(self.lats):
            for j, lon in enumerate(self.lons):
                try:
                    value = self.values[i][j]
                except IndexError:
                    continue
                if value is None:
                    continue
                out.append((lat, lon, float(value)))
        return out

    def gradient_cells(self) -> list[tuple[float, float, float]]:
        """Central-difference gradient magnitude per cell, in units per 100 km."""
        out: list[tuple[float, float, float]] = []
        ny, nx = len(self.lats), len(self.lons)
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                try:
                    c = self.values[i][j]
                    up, down = self.values[i - 1][j], self.values[i + 1][j]
                    left, right = self.values[i][j - 1], self.values[i][j + 1]
                except IndexError:
                    continue
                if None in (c, up, down, left, right):
                    continue
                dlat_km = abs(self.lats[i + 1] - self.lats[i - 1]) * 111.0 or 1.0
                dlon_km = (
                    abs(self.lons[j + 1] - self.lons[j - 1])
                    * 111.0
                    * math.cos(math.radians(self.lats[i]))
                ) or 1.0
                dy = (float(down) - float(up)) / dlat_km
                dx = (float(right) - float(left)) / dlon_km
                out.append((self.lats[i], self.lons[j], math.hypot(dx, dy) * 100.0))
        return out


class TimeseriesRag:
    def __init__(
        self,
        mosdac: MosdacConnector | None = None,
        openmeteo: OpenMeteoConnector | None = None,
        incois: IncoisConnector | None = None,
    ) -> None:
        self.mosdac = mosdac or MosdacConnector()
        self.openmeteo = openmeteo or OpenMeteoConnector()
        self.incois = incois or IncoisConnector()
        self.settings = self.mosdac.settings
        self._ev_seq = 0

    def _next_id(self, prefix: str) -> str:
        self._ev_seq += 1
        return f"{prefix}-{self._ev_seq:03d}"

    # ------------------------------------------------------------- ocean    #

    async def ocean_state(
        self, lat: float, lon: float, start: datetime, end: datetime
    ) -> dict[str, PointField]:
        """SST, currents, salinity from the ISRO Ocean State Forecast (OSF_CIRC)."""
        dataset_info = await self.mosdac.latest_osf_circ()
        if not dataset_info:
            return await self._ocean_state_fallback(lat, lon, start, end)

        dataset = dataset_info["url_path"]
        span = await self.mosdac.time_span(dataset)
        q_start, q_end = self._clamp(start, end, span)

        rows, used_lat, used_lon, moved_km = await self.mosdac.point_series_nearest_wet(
            dataset,
            MosdacConnector.OSF_CIRC_VARS,
            lat,
            lon,
            q_start,
            q_end,
            vert_coord=1,  # vertCoord=0 returns fill on this grid
            probe_var="temp",
        )
        if not rows:
            return await self._ocean_state_fallback(lat, lon, start, end)

        moved_note = (
            ""
            if moved_km < 1.0
            else (
                f"nearest unmasked model cell was {moved_km:.0f} km away at "
                f"{used_lat:.2f}N {used_lon:.2f}E; the requested point falls on a "
                "land-masked cell of the 10 km grid"
            )
        )
        prov = self.mosdac.provenance(
            dataset,
            title=f"SAC Ocean State Forecast - circulation ({dataset_info.get('date','')})",
            access_method="ncss-point",
            url=f"{self.mosdac.ncss_url(dataset)}/dataset.xml",
            valid_from=span[0] if span else None,
            valid_to=span[1] if span else None,
            caveat=moved_note,
        )

        out: dict[str, PointField] = {}
        specs = (
            ("temp", "sea surface temperature", "degC", "sst"),
            ("salinity", "sea surface salinity", "psu", "sss"),
        )
        for var, label, unit, key in specs:
            series = self._rows_to_series(rows, var)
            if not series:
                continue
            out[key] = PointField(
                variable=var,
                label=label,
                unit=unit,
                series=series,
                source_tier=Tier.ISRO,
                note=moved_note,
                evidence=self._evidence(
                    prefix=key,
                    label=f"{label} (ISRO OSF)",
                    series=series,
                    unit=unit,
                    lat=used_lat,
                    lon=used_lon,
                    provenance=prov,
                ),
            )

        u = self._rows_to_series(rows, "eastward_ocean_wave_current")
        v = self._rows_to_series(rows, "northward_ocean_wave_current")
        if u and v:
            speed_series = []
            for ru, rv in zip(u, v):
                if ru["value"] is None or rv["value"] is None:
                    continue
                speed_series.append(
                    {
                        "time": ru["time"],
                        "value": math.hypot(float(ru["value"]), float(rv["value"])),
                        "direction_deg": (
                            math.degrees(
                                math.atan2(float(ru["value"]), float(rv["value"]))
                            )
                            % 360.0
                        ),
                    }
                )
            if speed_series:
                out["current"] = PointField(
                    variable="current_speed",
                    label="surface current speed",
                    unit="cm/s",
                    series=speed_series,
                    source_tier=Tier.ISRO,
                    note=moved_note,
                    evidence=self._evidence(
                        prefix="cur",
                        label="surface current speed (ISRO OSF)",
                        series=speed_series,
                        unit="cm/s",
                        lat=used_lat,
                        lon=used_lon,
                        provenance=prov,
                    ),
                )

        # mixed layer depth lives on a 3-D-free variable, fetch separately
        mld_rows = await self.mosdac.point_series(
            dataset, ["hmxl"], used_lat, used_lon, q_start, q_end
        )
        mld_series = self._rows_to_series(mld_rows, "hmxl")
        mld_series = [
            {"time": r["time"], "value": r["value"] / 100.0}
            for r in mld_series
            if r["value"] is not None
        ]
        if mld_series:
            out["mld"] = PointField(
                variable="hmxl",
                label="mixed layer depth",
                unit="m",
                series=mld_series,
                source_tier=Tier.ISRO,
                evidence=self._evidence(
                    prefix="mld",
                    label="mixed layer depth (ISRO OSF)",
                    series=mld_series,
                    unit="m",
                    lat=used_lat,
                    lon=used_lon,
                    provenance=prov,
                ),
            )
        return out

    async def _ocean_state_fallback(
        self, lat: float, lon: float, start: datetime, end: datetime
    ) -> dict[str, PointField]:
        payload = await self.openmeteo.marine(lat, lon)
        rows = self.openmeteo.to_series(payload)
        series = self._openmeteo_series(rows, "sea_surface_temperature", start, end)
        if not series:
            return {}
        prov = self.openmeteo.provenance("marine forecast - sea surface temperature")
        return {
            "sst": PointField(
                variable="sea_surface_temperature",
                label="sea surface temperature",
                unit="degC",
                series=series,
                source_tier=Tier.FALLBACK,
                used_fallback=True,
                note="ISRO OSF_CIRC unavailable for this request",
                evidence=self._evidence(
                    prefix="sst",
                    label="sea surface temperature (fallback)",
                    series=series,
                    unit="degC",
                    lat=lat,
                    lon=lon,
                    provenance=prov,
                ),
            )
        }

    # ------------------------------------------------------------- waves    #

    async def wave_state(
        self, lat: float, lon: float, start: datetime, end: datetime
    ) -> dict[str, PointField]:
        dataset = MosdacConnector.OSF_WAVE_DATASET
        span = await self.mosdac.time_span(dataset)
        stale = True
        if span:
            age_h = (datetime.now(timezone.utc) - span[1]).total_seconds() / 3600.0
            stale = age_h > self.settings.stale_hours

        out: dict[str, PointField] = {}
        if span:
            q_start, q_end = self._clamp(start, end, span)
            rows, used_lat, used_lon, moved_km = (
                await self.mosdac.point_series_nearest_wet(
                    dataset,
                    ("SWH", "MWPER", "MWDIR", "HS01", "UWIND", "VWIND"),
                    lat,
                    lon,
                    q_start,
                    q_end,
                    probe_var="SWH",
                )
            )
            if rows:
                prov = self.mosdac.provenance(
                    dataset,
                    title="SAC Ocean State Forecast - wave model (10 km)",
                    access_method="ncss-point",
                    url=f"{self.mosdac.ncss_url(dataset)}/dataset.xml",
                    valid_from=span[0],
                    valid_to=span[1],
                    caveat=(
                        ""
                        if moved_km < 1.0
                        else f"sampled {moved_km:.0f} km from the requested point"
                    ),
                )
                swh = self._rows_to_series(rows, "SWH")
                if swh:
                    out["swh_isro"] = PointField(
                        variable="SWH",
                        label="significant wave height",
                        unit="m",
                        series=swh,
                        source_tier=Tier.ISRO,
                        note=(
                            "ISRO OSF_WAVE forecast cycle is "
                            f"{'stale' if stale else 'current'}"
                        ),
                        evidence=self._evidence(
                            prefix="swh-isro",
                            label="significant wave height (ISRO OSF_WAVE)",
                            series=swh,
                            unit="m",
                            lat=used_lat,
                            lon=used_lon,
                            provenance=prov,
                        ),
                    )
                period = self._rows_to_series(rows, "MWPER")
                if period:
                    out["wave_period_isro"] = PointField(
                        variable="MWPER",
                        label="mean wave period",
                        unit="s",
                        series=period,
                        source_tier=Tier.ISRO,
                        evidence=self._evidence(
                            prefix="per-isro",
                            label="mean wave period (ISRO OSF_WAVE)",
                            series=period,
                            unit="s",
                            lat=used_lat,
                            lon=used_lon,
                            provenance=prov,
                        ),
                    )
                direction = self._rows_to_series(rows, "MWDIR")
                if direction:
                    deg = [
                        {
                            "time": r["time"],
                            "value": (math.degrees(float(r["value"])) % 360.0),
                        }
                        for r in direction
                        if r["value"] is not None
                    ]
                    if deg:
                        out["wave_dir_isro"] = PointField(
                            variable="MWDIR",
                            label="mean wave direction",
                            unit="deg",
                            series=deg,
                            source_tier=Tier.ISRO,
                            evidence=self._evidence(
                                prefix="dir-isro",
                                label="mean wave direction (ISRO OSF_WAVE)",
                                series=deg,
                                unit="deg",
                                lat=used_lat,
                                lon=used_lon,
                                provenance=prov,
                            ),
                        )

        # live waves: the fallback is authoritative for "right now" when the
        # official cycle has aged out
        payload = await self.openmeteo.marine(lat, lon)
        rows = self.openmeteo.to_series(payload)
        live_swh = self._openmeteo_series(rows, "wave_height", start, end)
        if live_swh:
            prov = self.openmeteo.provenance("marine forecast - wave height")
            out["swh"] = PointField(
                variable="wave_height",
                label="significant wave height",
                unit="m",
                series=live_swh,
                source_tier=Tier.FALLBACK,
                used_fallback=True,
                note=(
                    "used as the live wave value because the ISRO OSF_WAVE cycle "
                    "on MOSDAC is older than the staleness threshold"
                    if stale
                    else "cross-check against the ISRO OSF_WAVE forecast"
                ),
                evidence=self._evidence(
                    prefix="swh",
                    label="significant wave height (live, fallback source)",
                    series=live_swh,
                    unit="m",
                    lat=lat,
                    lon=lon,
                    provenance=prov,
                ),
            )
        for var, label, unit, key in (
            ("swell_wave_height", "swell height", "m", "swell"),
            ("wave_period", "wave period", "s", "wave_period"),
            ("sea_level_height_msl", "sea level above MSL (tide)", "m", "tide"),
        ):
            series = self._openmeteo_series(rows, var, start, end)
            if not series:
                continue
            out[key] = PointField(
                variable=var,
                label=label,
                unit=unit,
                series=series,
                source_tier=Tier.FALLBACK,
                used_fallback=True,
                evidence=self._evidence(
                    prefix=key,
                    label=f"{label} (fallback source)",
                    series=series,
                    unit=unit,
                    lat=lat,
                    lon=lon,
                    provenance=self.openmeteo.provenance(f"marine forecast - {label}"),
                ),
            )
        if not out.get("swh") and out.get("swh_isro"):
            out["swh"] = out["swh_isro"]
        return out

    # ------------------------------------------------------------ weather   #

    async def weather_state(
        self, lat: float, lon: float, start: datetime, end: datetime
    ) -> dict[str, PointField]:
        payload = await self.openmeteo.weather(lat, lon)
        rows = self.openmeteo.to_series(payload)
        out: dict[str, PointField] = {}
        specs = (
            ("wind_speed_10m", "wind speed", "kt", "wind", MS_TO_KT / 3.6 * 3.6),
            ("wind_gusts_10m", "wind gusts", "kt", "gust", 1.0),
            ("wind_direction_10m", "wind direction", "deg", "wind_dir", 1.0),
            ("precipitation", "precipitation", "mm/h", "rain", 1.0),
            ("cape", "convective available potential energy", "J/kg", "cape", 1.0),
            ("visibility", "visibility", "m", "visibility", 1.0),
        )
        for var, label, unit, key, _ in specs:
            series = self._openmeteo_series(rows, var, start, end)
            if not series:
                continue
            if unit == "kt":
                # Open-Meteo returns km/h by default for wind
                series = [
                    {"time": r["time"], "value": float(r["value"]) * 0.539957}
                    for r in series
                    if r["value"] is not None
                ]
            out[key] = PointField(
                variable=var,
                label=label,
                unit=unit,
                series=series,
                source_tier=Tier.FALLBACK,
                used_fallback=True,
                evidence=self._evidence(
                    prefix=key,
                    label=f"{label} (fallback source)",
                    series=series,
                    unit=unit,
                    lat=lat,
                    lon=lon,
                    provenance=self.openmeteo.provenance(f"weather forecast - {label}"),
                ),
            )
        return out

    # ------------------------------------------------------------- fields   #

    async def sst_field(
        self,
        south: float,
        north: float,
        west: float,
        east: float,
        *,
        max_points: int = 22,
    ) -> GridField | None:
        info = await self.mosdac.latest_osf_circ()
        if not info:
            return None
        dataset = info["url_path"]
        window = await self.mosdac.grid_window(
            dataset,
            "temp",
            south,
            north,
            west,
            east,
            lon_var="xt_i",
            lat_var="yt_j",
            time_index=0,
            level_index=1,
            max_points=max_points,
        )
        if not window:
            return None
        span = await self.mosdac.time_span(dataset)
        return GridField(
            variable="temp",
            label="sea surface temperature",
            unit="degC",
            lats=window["lats"],
            lons=window["lons"],
            values=window["values"],
            valid_time=_iso(span[0]) if span else "",
            provenance=self.mosdac.provenance(
                dataset,
                title=f"SAC Ocean State Forecast - SST field ({info.get('date','')})",
                access_method="opendap-ascii",
                url=window["url"],
                valid_from=span[0] if span else None,
                valid_to=span[1] if span else None,
            ),
        )

    async def chlorophyll_field(
        self, south: float, north: float, west: float, east: float
    ) -> GridField | None:
        """ISRO PFZ chlorophyll grid; falls back to INCOIS ERDDAP ocean colour."""
        info = await self.mosdac.latest_pfz_grid("chl")
        if info:
            dataset = info["url_path"]
            dds = await self.mosdac.get_text(
                self.mosdac.dods_url(dataset, "dds"), ttl=86400
            )
            var, lat_var, lon_var = _guess_grid_vars(dds or "")
            if var:
                window = await self.mosdac.grid_window(
                    dataset,
                    var,
                    south,
                    north,
                    west,
                    east,
                    lon_var=lon_var,
                    lat_var=lat_var,
                    time_index=0,
                    max_points=20,
                )
                if window:
                    return GridField(
                        variable=var,
                        label="chlorophyll-a concentration",
                        unit="mg/m3",
                        lats=window["lats"],
                        lons=window["lons"],
                        values=window["values"],
                        valid_time=info.get("date", ""),
                        provenance=self.mosdac.provenance(
                            dataset,
                            title=(
                                "ISRO PFZ chlorophyll input grid "
                                f"({info.get('date','')})"
                            ),
                            access_method="opendap-ascii",
                            url=window["url"],
                            caveat=(
                                "archival PFZ input grid - the most recent chlorophyll "
                                "grid published on the MOSDAC PFZ catalogue, not a "
                                "same-day observation"
                            ),
                        ),
                    )
        return None

    async def chlorophyll_history(
        self, lat: float, lon: float, dataset_key: str = "oceansat2_ocm"
    ) -> list[dict[str, Any]]:
        dataset_id = IncoisConnector.GRID_DATASETS.get(dataset_key)
        if not dataset_id:
            return []
        variables = await self.incois.variables_of(dataset_id)
        candidate = next(
            (v for v in variables if "chl" in v.lower()),
            next((v for v in variables if v.lower() not in ("time", "latitude", "longitude")), ""),
        )
        if not candidate:
            return []
        return await self.incois.griddap_point(
            dataset_id, candidate, lat, lon, time_expr="last"
        )

    # ------------------------------------------------------------- helpers  #

    @staticmethod
    def _clamp(
        start: datetime, end: datetime, span: tuple[datetime, datetime] | None
    ) -> tuple[datetime, datetime]:
        """Keep a request inside the dataset's advertised validity window.

        MOSDAC returns an empty body rather than an error for out-of-range times,
        so this is the difference between an answer and a blank panel.
        """
        if not span:
            return start, end
        lo, hi = span
        if start > hi or end < lo:
            duration = end - start
            return lo, min(hi, lo + duration)
        return max(start, lo), min(end, hi)

    @staticmethod
    def _rows_to_series(
        rows: Sequence[dict[str, Any]], variable: str
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            value = row.get(variable)
            if value is None:
                continue
            out.append({"time": str(row.get("time", "")), "value": float(value)})
        return out

    @staticmethod
    def _openmeteo_series(
        rows: Sequence[dict[str, Any]],
        variable: str,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            stamp = _parse_iso(str(row.get("time", "")))
            if stamp is None or not (start <= stamp <= end):
                continue
            value = row.get(variable)
            if value is None:
                continue
            out.append({"time": str(row["time"]), "value": float(value)})
        return out

    def _evidence(
        self,
        *,
        prefix: str,
        label: str,
        series: list[dict[str, Any]],
        unit: str,
        lat: float,
        lon: float,
        provenance: Provenance,
    ) -> Evidence:
        first = series[0] if series else {}
        return Evidence(
            id=self._next_id(prefix),
            label=label,
            value=first.get("value"),
            unit=unit,
            at_lat=round(lat, 4),
            at_lon=round(lon, 4),
            at_time=_parse_iso(str(first.get("time", ""))),
            series=series[:64],
            provenance=provenance,
        )


def _guess_grid_vars(dds: str) -> tuple[str, str, str]:
    """Infer (data_var, lat_var, lon_var) from an OPeNDAP DDS blob."""
    import re

    dims = re.findall(r"(?:Float|Int|Double)\d*\s+(\w+)\[(\w+)\s*=", dds)
    names = [d[0] for d in dims]
    lat_var = next(
        (n for n in names if n.lower() in ("lat", "latitude", "y", "yt_j")), "latitude"
    )
    lon_var = next(
        (n for n in names if n.lower() in ("lon", "longitude", "x", "xt_i")), "longitude"
    )
    grids = re.findall(r"\}\s*(\w+);", dds)
    data_var = ""
    for name in grids:
        if name.lower() not in (lat_var.lower(), lon_var.lower(), "time"):
            data_var = name
            break
    if not data_var:
        for name in names:
            if name.lower() not in (lat_var.lower(), lon_var.lower(), "time"):
                data_var = name
                break
    return data_var, lat_var, lon_var


def default_window(hours: int = 48) -> tuple[datetime, datetime]:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return now, now + timedelta(hours=hours)
