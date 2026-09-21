"""Advisory retrieval and data discovery agents.

The advisory agent is the document leg of the multi-RAG. Before it searches it
pulls the live IMD warning page and the INCOIS PFZ page and injects their text
into the index, so a citation can point at today's bulletin rather than only at
the seeded reference corpus.

The discovery agent answers "which datasets can answer this?" from the real
catalogues, so the platform can say what it looked at and what else exists.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..schemas import Evidence, Tier
from ..services import Services
from .base import AgentContext


class AdvisoryAgent:
    name = "advisory"
    tools = (
        "imd.subdivision_warnings",
        "imd.cyclone_bulletin",
        "incois.pfz_advisory_text",
        "vector_rag.search",
    )

    def __init__(self, services: Services) -> None:
        self.services = services

    async def run(self, ctx: AgentContext) -> None:
        await self._ingest_live(ctx)

        with ctx.trace.timed(
            self.name,
            "search the advisory knowledge base",
            rationale=(
                "BM25 with marine synonym expansion over advisories, agency rules "
                "and today's bulletin text; official agency text is boosted over "
                "seeded reference material"
            ),
            tool="vector_rag.search",
            tool_args={"query": ctx.message[:160], "k": 5},
        ) as step:
            hits = self.services.vector_rag.search(ctx.message, k=5)
            found: list[dict[str, Any]] = []
            for hit in hits:
                evidence = Evidence(
                    id=f"doc-{hit.chunk.id}".replace("#", "-"),
                    label=hit.chunk.title,
                    text=hit.chunk.text[:900],
                    provenance=hit.chunk.provenance(),
                )
                ctx.evidence.add(evidence)
                found.append(
                    {
                        "title": hit.chunk.title,
                        "score": round(hit.score, 2),
                        "agency": hit.chunk.agency,
                        "tier": hit.chunk.tier.value,
                        "matched_terms": hit.matched_terms[:8],
                        "text": hit.chunk.text[:700],
                        "evidence_id": evidence.id,
                    }
                )
            ctx.findings.advisories = found
            step.evidence_ids = [f["evidence_id"] for f in found]
            step.outcome = (
                f"{len(found)} passages: "
                + "; ".join(f["title"][:52] for f in found[:3])
                if found
                else "no matching advisory text"
            )
            if not found:
                step.status = "degraded"

    async def _ingest_live(self, ctx: AgentContext) -> None:
        state = ctx.location.state if ctx.location else ""
        with ctx.trace.timed(
            self.name,
            "pull live official bulletins into the index",
            rationale=(
                "IMD's warning wording is the operative advice; ORCA quotes it "
                "rather than paraphrasing a warning"
            ),
            tool="imd.subdivision_warnings",
            tool_args={"state": state},
        ) as step:
            imd_warnings, imd_cyclone, pfz_text = await asyncio.gather(
                self.services.imd.subdivision_warnings(),
                self.services.imd.cyclone_bulletin(),
                self.services.incois.pfz_advisory_text(),
            )
            added = 0
            if imd_warnings:
                added += self.services.vector_rag.add_live_document(
                    doc_id="imd-warnings",
                    title=imd_warnings.title,
                    text=imd_warnings.text,
                    agency="India Meteorological Department",
                    tier=Tier.IMD,
                    url=imd_warnings.url,
                    tags=("advisory", "warning", "imd"),
                )
            if imd_cyclone:
                added += self.services.vector_rag.add_live_document(
                    doc_id="imd-cyclone",
                    title=imd_cyclone.title,
                    text=imd_cyclone.text,
                    agency="India Meteorological Department / RSMC New Delhi",
                    tier=Tier.IMD,
                    url=imd_cyclone.url,
                    tags=("advisory", "cyclone", "imd"),
                )
            if pfz_text:
                added += self.services.vector_rag.add_live_document(
                    doc_id="incois-pfz",
                    title="INCOIS Potential Fishing Zone advisory page",
                    text=pfz_text,
                    agency="INCOIS / Ministry of Earth Sciences",
                    tier=Tier.INCOIS,
                    url=self.services.settings.incois_pfz_page,
                    tags=("advisory", "pfz", "incois"),
                )
            step.outcome = f"{added} live passages indexed"
            if added == 0:
                step.status = "degraded"
                step.outcome = (
                    "no live bulletin text could be indexed; answering from the "
                    "seeded advisory corpus only"
                )
                ctx.note(
                    "Live agency bulletin pages could not be read for this "
                    "request, so advisory citations come from ORCA's seeded "
                    "reference corpus rather than today's bulletin."
                )


class DataDiscoveryAgent:
    name = "data-discovery"
    tools = ("mosdac.list_datasets", "incois.list_griddap", "nasa.search_collections")

    def __init__(self, services: Services) -> None:
        self.services = services

    async def run(self, ctx: AgentContext) -> None:
        with ctx.trace.timed(
            self.name,
            "discover candidate datasets",
            rationale=(
                "catalogues are queried live so the answer names datasets that "
                "actually exist right now, with their real update dates"
            ),
            tool="mosdac.list_datasets",
        ) as step:
            circ, wave, chl, sst, erddap = await asyncio.gather(
                self.services.mosdac.latest_osf_circ(),
                self.services.mosdac.list_datasets("OSF_WAVE"),
                self.services.mosdac.latest_pfz_grid("chl"),
                self.services.mosdac.latest_pfz_grid("sst"),
                self.services.incois.list_griddap(),
            )
            catalog: list[dict[str, Any]] = []
            if circ:
                catalog.append(
                    {
                        "agency": "ISRO / MOSDAC",
                        "dataset": "Ocean State Forecast - circulation (10 km)",
                        "id": circ["url_path"],
                        "latest": circ.get("date", ""),
                        "variables": "SST, salinity, mixed layer depth, currents",
                        "access": "THREDDS NCSS / OPeNDAP / WMS, no login",
                    }
                )
            if wave:
                catalog.append(
                    {
                        "agency": "ISRO / MOSDAC",
                        "dataset": "Ocean State Forecast - waves (10 km)",
                        "id": wave[0]["url_path"],
                        "latest": wave[0].get("modified", ""),
                        "variables": "SWH, wave period, wave direction, swell, winds",
                        "access": "THREDDS NCSS / OPeNDAP / WMS, no login",
                    }
                )
            for grid, label in ((sst, "PFZ input SST grid"), (chl, "PFZ input chlorophyll grid")):
                if grid:
                    catalog.append(
                        {
                            "agency": "ISRO / MOSDAC",
                            "dataset": label,
                            "id": grid["url_path"],
                            "latest": grid.get("date", ""),
                            "variables": "SST" if "SST" in label else "chlorophyll-a",
                            "access": "THREDDS OPeNDAP / WMS",
                        }
                    )
            for row in erddap[:6]:
                catalog.append(
                    {
                        "agency": "INCOIS / MoES",
                        "dataset": row.get("Title", ""),
                        "id": row.get("Dataset ID", ""),
                        "latest": "",
                        "variables": "",
                        "access": "ERDDAP griddap, open",
                    }
                )
            ctx.findings.catalog = catalog
            evidence = Evidence(
                id="catalog-summary",
                label="datasets available to answer this",
                value=len(catalog),
                unit="datasets",
                text="; ".join(f"{c['agency']}: {c['dataset']}" for c in catalog[:6]),
                provenance=self.services.mosdac.provenance(
                    "catalog",
                    title="live catalogue listing across MOSDAC and INCOIS ERDDAP",
                    access_method="catalog",
                ),
            )
            ctx.evidence.add(evidence)
            step.evidence_ids = [evidence.id]
            step.outcome = f"{len(catalog)} datasets found across MOSDAC and INCOIS"
            if not catalog:
                step.status = "degraded"
