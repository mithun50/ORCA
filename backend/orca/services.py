"""Service container.

One instance per process, built at startup. Holds the connector pool, the three
retrievers and the LLM client so agents receive them rather than constructing
their own (which would defeat the shared HTTP cache).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .config import Settings, get_settings
from .connectors.base import ConnectorRegistry, HttpConnector
from .connectors.gdacs import GdacsConnector
from .connectors.imd import ImdConnector
from .connectors.incois import IncoisConnector
from .connectors.marine_regions import MarineRegionsConnector
from .connectors.mosdac import MosdacConnector
from .connectors.nasa import NasaCmrConnector
from .connectors.openmeteo import OpenMeteoConnector
from .llm import LlmClient, get_llm
from .rag.geo_rag import GeoRag
from .rag.timeseries_rag import TimeseriesRag
from .rag.vector_rag import VectorRag

log = logging.getLogger("orca.services")


@dataclass
class Services:
    settings: Settings = field(default_factory=get_settings)
    registry: ConnectorRegistry = field(default_factory=ConnectorRegistry)

    mosdac: MosdacConnector = field(init=False)
    incois: IncoisConnector = field(init=False)
    imd: ImdConnector = field(init=False)
    openmeteo: OpenMeteoConnector = field(init=False)
    gdacs: GdacsConnector = field(init=False)
    marine_regions: MarineRegionsConnector = field(init=False)
    nasa: NasaCmrConnector = field(init=False)

    vector_rag: VectorRag = field(init=False)
    geo_rag: GeoRag = field(init=False)
    timeseries_rag: TimeseriesRag = field(init=False)
    llm: LlmClient = field(init=False)

    sessions: dict[str, list[dict[str, str]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mosdac = self.registry.add(MosdacConnector())  # type: ignore[assignment]
        self.incois = self.registry.add(IncoisConnector())  # type: ignore[assignment]
        self.imd = self.registry.add(ImdConnector())  # type: ignore[assignment]
        self.openmeteo = self.registry.add(OpenMeteoConnector())  # type: ignore[assignment]
        self.gdacs = self.registry.add(GdacsConnector())  # type: ignore[assignment]
        self.marine_regions = self.registry.add(  # type: ignore[assignment]
            MarineRegionsConnector()
        )
        self.nasa = self.registry.add(NasaCmrConnector())  # type: ignore[assignment]

        self.vector_rag = VectorRag()
        self.geo_rag = GeoRag(marine_regions=self.marine_regions)
        self.timeseries_rag = TimeseriesRag(
            mosdac=self.mosdac, openmeteo=self.openmeteo, incois=self.incois
        )
        self.llm = get_llm()

    # ----------------------------------------------------------------- setup #

    async def warm_up(self) -> dict[str, Any]:
        """Load indexes and pre-fetch slow, cacheable artefacts."""
        chunks = self.vector_rag.load_directory()
        zones = self.geo_rag.load_zones()
        provider = await self.llm.provider()
        eez = await self.geo_rag.ensure_eez()
        report = {
            "knowledge_chunks": chunks,
            "zone_features": zones,
            "llm_provider": provider,
            "eez_loaded": eez is not None,
        }
        log.info("ORCA warm-up: %s", report)
        return report

    async def shutdown(self) -> None:
        await HttpConnector.aclose()

    # -------------------------------------------------------------- session #

    def history(self, session_id: str) -> list[dict[str, str]]:
        return self.sessions.setdefault(session_id, [])

    def remember(self, session_id: str, role: str, text: str, **extra: str) -> None:
        turns = self.history(session_id)
        turns.append({"role": role, "text": text, **extra})
        del turns[:-12]  # keep the last dozen turns

    def last_location(self, session_id: str) -> tuple[float, float, str] | None:
        for turn in reversed(self.history(session_id)):
            if turn.get("lat") and turn.get("lon"):
                try:
                    return (
                        float(turn["lat"]),
                        float(turn["lon"]),
                        turn.get("place", ""),
                    )
                except (TypeError, ValueError):
                    continue
        return None

    # --------------------------------------------------------------- health #

    def health(self) -> dict[str, Any]:
        return {
            "sources": {
                name: {
                    "ok": c.health.ok_count,
                    "failed": c.health.fail_count,
                    "last_error": c.health.last_error,
                    "last_ok_ms": c.health.last_ok_ms,
                }
                for name, c in self.registry.instances.items()
            },
            "vector_rag": self.vector_rag.stats(),
            "geo_rag": self.geo_rag.stats(),
            "llm": self.llm.available_hint,
            "sessions": len(self.sessions),
        }


_services: Services | None = None


def get_services() -> Services:
    global _services
    if _services is None:
        _services = Services()
    return _services
