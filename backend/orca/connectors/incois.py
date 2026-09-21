"""INCOIS / MoES connector.

Two very different surfaces:

* `erddap.incois.gov.in` is a real, open ERDDAP server (verified: 16 griddap and
  2 tabledap datasets). This is the machine-readable side and carries the ISRO
  ocean-colour heritage products (Oceansat-2 OCM, IRS-P4 OCM chlorophyll) plus
  Argo profiles.
* the INCOIS web portal (PFZ advisory, Ocean State Forecast bulletins) is
  JS-rendered and its GeoServer is behind a WAF that returns 403 to
  non-browser clients. ORCA does not fight the WAF. It ingests bulletin text as
  documents for the vector RAG and falls back to the seeded advisory corpus when
  the live page yields nothing parseable.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from ..schemas import Provenance, Tier
from .base import HttpConnector


class IncoisConnector(HttpConnector):
    source_name = "incois"

    #: griddap datasets confirmed present on the INCOIS ERDDAP
    GRID_DATASETS = {
        "oceansat2_ocm": "incois_oceansat2_datasets",
        "irs_chlorophyll": "IRS_chlorophyll_datasets",
        "argo_sst_weekly": "incois_argo_sst_weekly",
        "argo_10d_vam": "incois_argo_10d_VAM",
        "ascat_daily_wind": "ascat_daily_datasets",
        "avhrr_amsr_sst": "NOAA_AVHRR_AMSR_datasets",
        "value_added": "incois_valueadded_products_datasets",
    }
    ARGO_TABLE = "Indian_ARGO_Floats"

    # ------------------------------------------------------------- discovery #

    async def list_griddap(self) -> list[dict[str, str]]:
        payload = await self.get_json(
            f"{self.settings.incois_erddap_base}/griddap/index.json",
            params={"page": 1, "itemsPerPage": 200},
            ttl=86400,
        )
        return self._table_to_records(payload)

    async def dataset_info(self, dataset_id: str) -> list[dict[str, Any]]:
        payload = await self.get_json(
            f"{self.settings.incois_erddap_base}/info/{dataset_id}/index.json",
            ttl=86400,
        )
        return self._table_to_records(payload)

    async def variables_of(self, dataset_id: str) -> list[str]:
        rows = await self.dataset_info(dataset_id)
        return sorted(
            {
                r["Variable Name"]
                for r in rows
                if r.get("Row Type") == "variable" and r.get("Variable Name")
            }
        )

    async def time_coverage(self, dataset_id: str) -> tuple[str, str] | None:
        rows = await self.dataset_info(dataset_id)
        start = end = ""
        for r in rows:
            if r.get("Attribute Name") == "time_coverage_start":
                start = str(r.get("Value", ""))
            if r.get("Attribute Name") == "time_coverage_end":
                end = str(r.get("Value", ""))
        return (start, end) if start and end else None

    @staticmethod
    def _table_to_records(payload: Any) -> list[dict[str, Any]]:
        if not payload or "table" not in payload:
            return []
        cols = payload["table"].get("columnNames", [])
        return [dict(zip(cols, row)) for row in payload["table"].get("rows", [])]

    # ------------------------------------------------------------- griddap  #

    async def griddap_point(
        self,
        dataset_id: str,
        variable: str,
        lat: float,
        lon: float,
        *,
        time_expr: str = "last",
        extra_dims: str = "",
    ) -> list[dict[str, Any]]:
        """Nearest-cell value(s) using ERDDAP's `(value)` coordinate syntax."""
        query = (
            f"{variable}%5B({time_expr})%5D{extra_dims}"
            f"%5B({lat})%5D%5B({lon})%5D"
        )
        url = f"{self.settings.incois_erddap_base}/griddap/{dataset_id}.csv?{query}"
        text = await self.get_text(url, ttl=3600)
        if not text:
            return []
        return self._parse_erddap_csv(text)

    async def griddap_box(
        self,
        dataset_id: str,
        variable: str,
        south: float,
        north: float,
        west: float,
        east: float,
        *,
        time_expr: str = "last",
        stride: int = 4,
        extra_dims: str = "",
    ) -> list[dict[str, Any]]:
        query = (
            f"{variable}%5B({time_expr})%5D{extra_dims}"
            f"%5B({south}):{stride}:({north})%5D"
            f"%5B({west}):{stride}:({east})%5D"
        )
        url = f"{self.settings.incois_erddap_base}/griddap/{dataset_id}.csv?{query}"
        text = await self.get_text(url, ttl=3600)
        if not text:
            return []
        return self._parse_erddap_csv(text)

    async def argo_floats_near(
        self, lat: float, lon: float, radius_deg: float = 3.0, days: int = 30
    ) -> list[dict[str, Any]]:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        constraints = (
            f"&latitude%3E={lat - radius_deg}&latitude%3C={lat + radius_deg}"
            f"&longitude%3E={lon - radius_deg}&longitude%3C={lon + radius_deg}"
            f"&time%3E={since}"
        )
        url = (
            f"{self.settings.incois_erddap_base}/tabledap/{self.ARGO_TABLE}.csv"
            f"?{constraints.lstrip('&')}"
        )
        text = await self.get_text(url, ttl=3600)
        if not text:
            return []
        return self._parse_erddap_csv(text)[:200]

    @staticmethod
    def _parse_erddap_csv(text: str) -> list[dict[str, Any]]:
        """ERDDAP CSV is: header row, units row, then data."""
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) < 3:
            return []
        headers = [h.strip() for h in lines[0].split(",")]
        units = [u.strip() for u in lines[1].split(",")]
        rows: list[dict[str, Any]] = []
        for line in lines[2:]:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(headers):
                continue
            row: dict[str, Any] = {"_units": dict(zip(headers, units))}
            for key, value in zip(headers, parts):
                if value in ("", "NaN"):
                    row[key] = None
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
        return rows

    # ------------------------------------------------------------ advisories #

    async def pfz_advisory_text(self) -> str:
        """Best-effort text scrape of the public PFZ advisory page."""
        html = await self.get_text(self.settings.incois_pfz_page, ttl=3600)
        if not html:
            return ""
        return _visible_text(html)

    # ------------------------------------------------------------ provenance #

    def erddap_provenance(
        self, dataset_id: str, title: str, access_method: str = "erddap-csv"
    ) -> Provenance:
        return Provenance(
            agency="INCOIS / Ministry of Earth Sciences",
            dataset=title,
            tier=Tier.INCOIS,
            url=f"{self.settings.incois_erddap_base}/griddap/{dataset_id}.html",
            access_method=access_method,
            official=True,
        )

    def advisory_provenance(self, title: str) -> Provenance:
        return Provenance(
            agency="INCOIS / Ministry of Earth Sciences",
            dataset=title,
            tier=Tier.INCOIS,
            url=self.settings.incois_pfz_page,
            access_method="html",
            official=True,
        )


def _visible_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
    except Exception:  # pragma: no cover - bs4/lxml missing
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()
