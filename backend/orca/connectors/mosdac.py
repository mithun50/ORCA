"""MOSDAC / ISRO connector (THREDDS Data Server).

Root: https://mosdac.gov.in/live_data  (no authentication required)

Three access paths are used, each for what it is best at:

* **NCSS point** -> CSV time series at one lat/lon. Cheap, exact, used for every
  "conditions at my location" answer.
* **OPeNDAP ASCII** -> strided grid windows as plain text. Used for spatial
  reasoning (SST fronts, chlorophyll patches, nearest-PFZ search) without
  pulling a 3.8 GB file or needing a netCDF library.
* **WMS** -> raster tiles handed straight to the Leaflet client so the map shows
  the actual ISRO field, not a re-render.

Verified behaviours that the code depends on (2026-09-21):
  - `accept=csv` works for point requests and is rejected (HTTP 400) for grid
    requests, so grid windows must go through OPeNDAP.
  - `vertCoord=0` returns fill for OSF_CIRC; the first real level is `1` (1 m).
  - fill value is -1e34 for OSF_WAVE and -9.9999998e33 for OSF_CIRC.
  - the OSF_WAVE aggregate's time axis is `hours since 2026-4-21`; its true
    validity window must be read from `dataset.xml`, never assumed to be today.
"""

from __future__ import annotations

import asyncio
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from ..schemas import Provenance, Tier
from .base import HttpConnector

FILL_LIMIT = 1e30
_DASH_RE = re.compile(r"^-{10,}$")
_BLOCK_HEADER_RE = re.compile(r"^([A-Za-z0-9_.]+)\[([0-9\]\[]+)\]\s*$")
_ROW_PREFIX_RE = re.compile(r"^(?:\[\d+\])+,\s*")


def _is_fill(v: float | None) -> bool:
    return v is None or not math.isfinite(v) or abs(v) > FILL_LIMIT


def clean(v: float | None) -> float | None:
    return None if _is_fill(v) else float(v)


def parse_dods_ascii(text: str) -> dict[str, dict[str, Any]]:
    """Parse a THREDDS OPeNDAP `.ascii` response.

    Returns ``{short_name: {"shape": [...], "values": [flat floats]}}``.
    """
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if _DASH_RE.match(line.strip()):
            start = i + 1
            break
    out: dict[str, dict[str, Any]] = {}
    current: str | None = None
    for raw in lines[start:]:
        line = raw.strip()
        if not line:
            current = None
            continue
        header = _BLOCK_HEADER_RE.match(line)
        if header:
            name = header.group(1).split(".")[-1]
            shape = [int(d) for d in re.findall(r"\d+", header.group(2))]
            out[name] = {"shape": shape, "values": []}
            current = name
            continue
        if current is None:
            continue
        body = _ROW_PREFIX_RE.sub("", line)
        for token in body.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                out[current]["values"].append(float(token))
            except ValueError:
                pass
    return out


def parse_ncss_csv(text: str) -> list[dict[str, Any]]:
    """Parse THREDDS NCSS point CSV, stripping `[unit="m"]` from headers."""
    rows: list[dict[str, Any]] = []
    blocks = [b for b in text.split("\n\n") if b.strip()]
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        headers = []
        units = {}
        for col in lines[0].split(","):
            name = col.strip()
            unit = ""
            if "[" in name:
                unit_match = re.search(r'unit="([^"]*)"', name)
                unit = unit_match.group(1) if unit_match else ""
                name = name.split("[")[0]
            headers.append(name)
            units[name] = unit
        for line in lines[1:]:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(headers):
                continue
            row: dict[str, Any] = {"_units": units}
            for name, value in zip(headers, parts):
                if name == "time":
                    row["time"] = value
                    continue
                try:
                    row[name] = clean(float(value))
                except ValueError:
                    row[name] = value
            rows.append(row)
    return rows


def _merge_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """NCSS emits one CSV block per variable group; merge them on time."""
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("time", ""))
        slot = merged.setdefault(key, {"time": key, "_units": {}})
        slot["_units"].update(row.get("_units", {}))
        for k, v in row.items():
            if k in ("time", "_units"):
                continue
            if k not in slot or slot[k] is None:
                slot[k] = v
    return [merged[k] for k in sorted(merged)]


class MosdacConnector(HttpConnector):
    source_name = "mosdac"

    OSF_WAVE_DATASET = "OSF_WAVE/SAC_OSF_WAVE_10KM.nc"
    OSF_WAVE_VARS = (
        "SWH",
        "MWPER",
        "MWDIR",
        "HS01",
        "DIR01",
        "HS02",
        "DIR02",
        "UWIND",
        "VWIND",
    )
    OSF_CIRC_VARS = (
        "temp",
        "salinity",
        "eastward_ocean_wave_current",
        "northward_ocean_wave_current",
    )

    # ------------------------------------------------------------------ urls #

    def catalog_url(self, path: str) -> str:
        return f"{self.settings.mosdac_base}/catalog/{path.strip('/')}/catalog.xml"

    def ncss_url(self, dataset: str) -> str:
        return f"{self.settings.mosdac_base}/ncss/grid/{dataset}"

    def dods_url(self, dataset: str, ext: str) -> str:
        return f"{self.settings.mosdac_base}/dodsC/{dataset}.{ext}"

    def wms_url(self, dataset: str) -> str:
        return f"{self.settings.mosdac_base}/wms/{dataset}"

    # -------------------------------------------------------------- catalog #

    async def list_datasets(self, catalog_path: str) -> list[dict[str, str]]:
        """Return ``[{name, url_path, modified}]`` for one catalog."""
        text = await self.get_text(self.catalog_url(catalog_path), ttl=1800)
        if not text:
            return []
        items: list[dict[str, str]] = []
        for match in re.finditer(
            r'<dataset\s+name="([^"]+)"[^>]*urlPath="([^"]+)"[^>]*>(.*?)</dataset>'
            r"|<dataset\s+name=\"([^\"]+)\"[^>]*urlPath=\"([^\"]+)\"\s*/>",
            text,
            re.S,
        ):
            name = match.group(1) or match.group(4)
            url_path = match.group(2) or match.group(5)
            inner = match.group(3) or ""
            modified = ""
            mod = re.search(r'<date type="modified">([^<]+)</date>', inner)
            if mod:
                modified = mod.group(1)
            items.append({"name": name, "url_path": url_path, "modified": modified})
        return items

    async def latest_dated_dataset(
        self, catalog_path: str, pattern: str
    ) -> dict[str, str] | None:
        """Newest dataset in a catalog whose name matches a YYYYMMDD pattern."""
        datasets = await self.list_datasets(catalog_path)
        rx = re.compile(pattern)
        best: tuple[str, dict[str, str]] | None = None
        for ds in datasets:
            m = rx.search(ds["name"])
            if not m:
                continue
            stamp = m.group(1)
            if best is None or stamp > best[0]:
                best = (stamp, ds)
        if not best:
            return None
        ds = dict(best[1])
        ds["date"] = best[0]
        return ds

    async def latest_osf_circ(self) -> dict[str, str] | None:
        return await self.latest_dated_dataset(
            "OSF_CIRC", r"SAC_OSF_CIRC_10KM_(\d{8})\.nc"
        )

    async def latest_pfz_grid(self, kind: str) -> dict[str, str] | None:
        """kind is 'sst' or 'chl'."""
        return await self.latest_dated_dataset(
            f"pfz/{kind}", rf"pfz_{kind}_(\d{{8}})\.nc"
        )

    # ------------------------------------------------------------- metadata #

    async def time_span(self, dataset: str) -> tuple[datetime, datetime] | None:
        text = await self.get_text(
            f"{self.ncss_url(dataset)}/dataset.xml", ttl=1800
        )
        if not text:
            return None
        begin = re.search(r"<begin>([^<]+)</begin>", text)
        end = re.search(r"<end>([^<]+)</end>", text)
        if not (begin and end):
            return None
        try:
            return (
                datetime.fromisoformat(begin.group(1).replace("Z", "+00:00")),
                datetime.fromisoformat(end.group(1).replace("Z", "+00:00")),
            )
        except ValueError:
            return None

    async def wms_layers(self, dataset: str) -> list[str]:
        text = await self.get_text(
            self.wms_url(dataset),
            params={
                "service": "WMS",
                "version": "1.3.0",
                "request": "GetCapabilities",
            },
            ttl=3600,
        )
        if not text:
            return []
        names = [m.group(1) for m in re.finditer(r"<Name>([^<]+)</Name>", text)]
        return [n for n in names if "/" not in n and n != "WMS"]

    # ----------------------------------------------------------- point data #

    async def point_series(
        self,
        dataset: str,
        variables: Iterable[str],
        lat: float,
        lon: float,
        start: datetime,
        end: datetime,
        vert_coord: float | None = None,
    ) -> list[dict[str, Any]]:
        params: list[tuple[str, str]] = [("var", v) for v in variables]
        params += [
            ("latitude", f"{lat:.4f}"),
            ("longitude", f"{lon:.4f}"),
            ("time_start", start.strftime("%Y-%m-%dT%H:%M:%SZ")),
            ("time_end", end.strftime("%Y-%m-%dT%H:%M:%SZ")),
            ("accept", "csv"),
        ]
        if vert_coord is not None:
            params.append(("vertCoord", str(vert_coord)))
        # NCSS wants repeated `var=` keys, so the query string is built by hand
        query = "&".join(f"{k}={v}" for k, v in params)
        out = await self.fetch(f"{self.ncss_url(dataset)}?{query}", ttl=600)
        if not out.ok:
            return []
        return _merge_rows(parse_ncss_csv(out.text))

    async def point_series_nearest_wet(
        self,
        dataset: str,
        variables: Iterable[str],
        lat: float,
        lon: float,
        start: datetime,
        end: datetime,
        vert_coord: float | None = None,
        probe_var: str | None = None,
        max_rings: int = 3,
        step_deg: float = 0.25,
    ) -> tuple[list[dict[str, Any]], float, float, float]:
        """Sample the requested point; if it is land-masked, spiral outward.

        Coastal fishing locations frequently fall on a masked cell of a 10 km
        ocean model. Returning "no data" there would be useless to a fisherman,
        so ORCA moves to the nearest wet cell and reports how far it moved.
        """
        variables = list(variables)
        probe = probe_var or variables[0]

        async def attempt(la: float, lo: float) -> list[dict[str, Any]]:
            rows = await self.point_series(
                dataset, variables, la, lo, start, end, vert_coord
            )
            if any(r.get(probe) is not None for r in rows):
                return rows
            return []

        rows = await attempt(lat, lon)
        if rows:
            return rows, lat, lon, 0.0

        for ring in range(1, max_rings + 1):
            offset = step_deg * ring
            # prefer moving offshore (east in the Bay of Bengal, west in the
            # Arabian Sea) before trying the full ring
            candidates = [
                (lat, lon + offset) if lon > 77.0 else (lat, lon - offset),
                (lat + offset, lon),
                (lat - offset, lon),
                (lat, lon - offset) if lon > 77.0 else (lat, lon + offset),
                (lat + offset, lon + offset),
                (lat - offset, lon - offset),
            ]
            results = await asyncio.gather(
                *(attempt(la, lo) for la, lo in candidates)
            )
            for (la, lo), rows in zip(candidates, results):
                if rows:
                    moved = _haversine_km(lat, lon, la, lo)
                    return rows, la, lo, moved
        return [], lat, lon, 0.0

    # ------------------------------------------------------------ grid data #

    async def coord_arrays(
        self, dataset: str, lon_var: str, lat_var: str
    ) -> tuple[list[float], list[float]] | None:
        """Fetch (and disk-cache) the 1-D coordinate axes of a dataset."""
        key = f"{dataset}:{lon_var}:{lat_var}"
        cached = self.disk_get(key, ".coords.json", 86400)
        if cached:
            import json

            try:
                payload = json.loads(cached)
                return payload["lon"], payload["lat"]
            except (ValueError, KeyError):
                pass
        base = self.dods_url(dataset, "ascii")
        lon_text, lat_text = await asyncio.gather(
            self.get_text(f"{base}?{lon_var}", ttl=3600),
            self.get_text(f"{base}?{lat_var}", ttl=3600),
        )
        if not lon_text or not lat_text:
            return None
        lon_block = parse_dods_ascii(lon_text).get(lon_var)
        lat_block = parse_dods_ascii(lat_text).get(lat_var)
        if not lon_block or not lat_block:
            return None
        lons = lon_block["values"]
        lats = lat_block["values"]
        import json

        self.disk_put(key, ".coords.json", json.dumps({"lon": lons, "lat": lats}))
        return lons, lats

    @staticmethod
    def _index_range(
        axis: list[float], lo: float, hi: float
    ) -> tuple[int, int]:
        idx = [i for i, v in enumerate(axis) if lo <= v <= hi]
        if not idx:
            nearest = min(range(len(axis)), key=lambda i: abs(axis[i] - (lo + hi) / 2))
            return nearest, nearest
        return idx[0], idx[-1]

    async def grid_window(
        self,
        dataset: str,
        variable: str,
        south: float,
        north: float,
        west: float,
        east: float,
        *,
        lon_var: str,
        lat_var: str,
        time_index: int = 0,
        level_index: int | None = None,
        max_points: int = 26,
    ) -> dict[str, Any] | None:
        """Strided OPeNDAP window -> ``{lats, lons, values[[...]]}``.

        `max_points` caps each axis so a request stays small and fast; the
        stride is computed from the requested box.
        """
        axes = await self.coord_arrays(dataset, lon_var, lat_var)
        if not axes:
            return None
        lons, lats = axes
        x0, x1 = self._index_range(lons, west, east)
        y0, y1 = self._index_range(lats, south, north)
        xs = max(1, (x1 - x0) // max_points + 1)
        ys = max(1, (y1 - y0) // max_points + 1)
        slices = f"[{time_index}]"
        if level_index is not None:
            slices += f"[{level_index}]"
        slices += f"[{y0}:{ys}:{y1}][{x0}:{xs}:{x1}]"
        url = f"{self.dods_url(dataset, 'ascii')}?{variable}{slices}"
        text = await self.get_text(url, ttl=900)
        if not text:
            return None
        blocks = parse_dods_ascii(text)
        var_block = blocks.get(variable)
        if not var_block:
            return None
        out_lats = blocks.get(lat_var, {}).get("values") or [
            lats[i] for i in range(y0, y1 + 1, ys)
        ]
        out_lons = blocks.get(lon_var, {}).get("values") or [
            lons[i] for i in range(x0, x1 + 1, xs)
        ]
        ny, nx = len(out_lats), len(out_lons)
        flat = [clean(v) for v in var_block["values"]]
        if len(flat) < ny * nx:
            return None
        grid = [flat[r * nx : (r + 1) * nx] for r in range(ny)]
        return {
            "variable": variable,
            "lats": out_lats,
            "lons": out_lons,
            "values": grid,
            "url": url,
        }

    # ---------------------------------------------------------- provenance  #

    def provenance(
        self,
        dataset: str,
        *,
        title: str,
        access_method: str,
        url: str = "",
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        caveat: str = "",
    ) -> Provenance:
        stale = False
        note = ""
        if valid_to is not None:
            age_h = (
                datetime.now(timezone.utc) - valid_to
            ).total_seconds() / 3600.0
            if age_h > self.settings.stale_hours:
                stale = True
                note = (
                    f"forecast cycle ended {age_h / 24:.1f} days ago; "
                    "treated as archival, live values cross-checked "
                    "against the fallback tier"
                )
        return Provenance(
            agency="ISRO / MOSDAC (Space Applications Centre)",
            dataset=title,
            tier=Tier.ISRO,
            url=url or self.ncss_url(dataset),
            access_method=access_method,
            valid_from=valid_from,
            valid_to=valid_to,
            is_stale=stale,
            staleness_note=note,
            official=True,
            caveat=caveat,
        )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def default_window(hours: int = 48) -> tuple[datetime, datetime]:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return now, now + timedelta(hours=hours)
