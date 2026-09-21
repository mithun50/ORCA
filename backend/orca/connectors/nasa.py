"""NASA CMR connector - dataset discovery only.

The data-discovery agent needs to be able to answer "what else could answer this
question?" honestly. CMR is the catalogue of record for NASA Earth observation
collections, so ORCA queries it to name real alternative products (MODIS/VIIRS
ocean colour, MUR SST, PACE) with their concept IDs, rather than inventing
dataset names. ORCA does not pull NASA granules in the prototype - Indian
official products are preferred for the actual numbers.
"""

from __future__ import annotations

from typing import Any

from ..schemas import Provenance, Tier
from .base import HttpConnector


class NasaCmrConnector(HttpConnector):
    source_name = "nasa-cmr"

    async def search_collections(
        self, keyword: str, *, bbox: tuple[float, float, float, float] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "keyword": keyword,
            "page_size": limit,
            "sort_key": "-usage_score",
        }
        if bbox:
            params["bounding_box"] = ",".join(str(v) for v in bbox)
        payload = await self.get_json(
            f"{self.settings.nasa_cmr_base}/collections.json", params=params, ttl=86400
        )
        if not payload:
            return []
        entries = payload.get("feed", {}).get("entry", [])
        out = []
        for entry in entries:
            out.append(
                {
                    "concept_id": entry.get("id"),
                    "title": entry.get("dataset_id") or entry.get("title"),
                    "archive_center": entry.get("archive_center", ""),
                    "time_start": entry.get("time_start", ""),
                    "time_end": entry.get("time_end", ""),
                    "links": [
                        link.get("href")
                        for link in entry.get("links", [])[:2]
                        if link.get("href")
                    ],
                }
            )
        return out

    def provenance(self, dataset: str) -> Provenance:
        return Provenance(
            agency="NASA (Common Metadata Repository)",
            dataset=dataset,
            tier=Tier.FALLBACK,
            url=f"{self.settings.nasa_cmr_base}/collections.json",
            access_method="http-json",
            official=False,
            caveat="catalogue metadata used for dataset discovery, not for values",
        )
