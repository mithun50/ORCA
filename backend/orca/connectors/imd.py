"""IMD connector.

IMD's JSON APIs under `mausam.imd.gov.in/api/` all answer HTTP 401 without a
key that IMD issues on request. That path is implemented and switched on by
`ORCA_IMD_API_KEY`. With no key, ORCA reads the two public HTML products that
were verified reachable:

* `rsmcnewdelhi.imd.gov.in` - RSMC New Delhi tropical cyclone bulletins
* `mausam.imd.gov.in/imd_latest/contents/subdivisionwise-warning.php`
  - sub-division wise warnings, which is where the coastal "squally weather,
    fishermen are advised not to venture" text lives.

The extracted text is returned as advisory documents so the vector RAG can cite
the actual IMD wording instead of ORCA paraphrasing a warning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..schemas import Provenance, Tier
from .base import HttpConnector
from .incois import _visible_text

# Coastal meteorological sub-divisions, used to pick the relevant warning block.
COASTAL_SUBDIVISIONS = {
    "gujarat": ["Saurashtra", "Kutch", "Gujarat"],
    "maharashtra": ["Konkan", "Goa", "Madhya Maharashtra"],
    "goa": ["Konkan", "Goa"],
    "karnataka": ["Coastal Karnataka", "North Interior Karnataka"],
    "kerala": ["Kerala", "Mahe", "Lakshadweep"],
    "lakshadweep": ["Lakshadweep"],
    "tamil nadu": ["Tamil Nadu", "Puducherry", "Karaikal"],
    "puducherry": ["Tamil Nadu", "Puducherry"],
    "andhra pradesh": [
        "Coastal Andhra Pradesh",
        "Yanam",
        "Rayalaseema",
        "Andhra Pradesh",
    ],
    "odisha": ["Odisha"],
    "west bengal": ["Gangetic West Bengal", "West Bengal"],
    "andaman and nicobar": ["Andaman", "Nicobar"],
}

FISHERMEN_PATTERNS = (
    r"fisher(?:men|folk)[^.]{0,300}\.",
    r"not to venture[^.]{0,300}\.",
    r"squally weather[^.]{0,300}\.",
    r"rough to very rough sea[^.]{0,200}\.",
    r"high wave[^.]{0,200}\.",
)


@dataclass
class ImdDocument:
    title: str
    text: str
    url: str


class ImdConnector(HttpConnector):
    source_name = "imd"

    @property
    def has_api_key(self) -> bool:
        return bool(self.settings.imd_api_key)

    # ---------------------------------------------------------------- public #

    async def cyclone_bulletin(self) -> ImdDocument | None:
        html = await self.get_text(self.settings.imd_rsmc_url, ttl=1800)
        if not html:
            return None
        text = _visible_text(html)
        if not text:
            return None
        return ImdDocument(
            title="RSMC New Delhi tropical cyclone bulletin (page text)",
            text=text[:6000],
            url=self.settings.imd_rsmc_url,
        )

    async def subdivision_warnings(self) -> ImdDocument | None:
        html = await self.get_text(
            self.settings.imd_subdivision_warning_url, ttl=1800
        )
        if not html:
            return None
        text = _visible_text(html)
        if not text:
            return None
        return ImdDocument(
            title="IMD sub-division wise weather warning",
            text=text[:12000],
            url=self.settings.imd_subdivision_warning_url,
        )

    async def warnings_for_state(self, state: str) -> list[str]:
        """Sentences from the warning bulletin relevant to a coastal state."""
        doc = await self.subdivision_warnings()
        if not doc:
            return []
        keys = COASTAL_SUBDIVISIONS.get(state.lower().strip(), [state])
        hits: list[str] = []
        sentences = re.split(r"(?<=[.;])\s+", doc.text)
        for sentence in sentences:
            if any(k.lower() in sentence.lower() for k in keys):
                hits.append(sentence.strip())
        if not hits:
            for pattern in FISHERMEN_PATTERNS:
                hits += [m.group(0).strip() for m in re.finditer(pattern, doc.text, re.I)]
        seen: set[str] = set()
        unique: list[str] = []
        for h in hits:
            if h and h not in seen and len(h) > 25:
                seen.add(h)
                unique.append(h)
        return unique[:8]

    async def fishermen_warnings(self) -> list[str]:
        doc = await self.subdivision_warnings()
        if not doc:
            return []
        hits: list[str] = []
        for pattern in FISHERMEN_PATTERNS:
            hits += [m.group(0).strip() for m in re.finditer(pattern, doc.text, re.I)]
        return list(dict.fromkeys(hits))[:8]

    async def api_nowcast(self, district_id: str) -> Any | None:
        """Keyed endpoint. Returns None (not an error) when no key is set."""
        if not self.has_api_key:
            return None
        return await self.get_json(
            f"{self.settings.imd_api_base}/nowcast_district_api.php",
            params={"id": district_id},
            headers={"Authorization": f"Bearer {self.settings.imd_api_key}"},
            ttl=600,
        )

    # ------------------------------------------------------------ provenance #

    def provenance(self, dataset: str, url: str, caveat: str = "") -> Provenance:
        return Provenance(
            agency="India Meteorological Department",
            dataset=dataset,
            tier=Tier.IMD,
            url=url,
            access_method="html",
            official=True,
            caveat=caveat
            or (
                "parsed from IMD's public warning page; IMD's JSON API needs a "
                "department-issued key, set ORCA_IMD_API_KEY to use it"
            ),
        )
