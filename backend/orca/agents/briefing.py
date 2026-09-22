"""Numbers-only briefing.

The prose answer reads naturally but buries the figures. This module rebuilds the
same retrieved data as four short groups of numbers, with the risk stated first,
so a skipper can read the decision and the values behind it in one glance instead
of parsing paragraphs.

Nothing here retrieves or derives anything. It reads what the agents already put
on `ctx.findings` and what the risk agent already decided, and reshapes it. That
matters: the briefing cannot disagree with the answer, because it is not a second
opinion, only a second layout of the same numbers.

Grouping follows where a value comes from rather than what it means:

* **ocean**     - sea state and the water column: waves, swell, SST, salinity,
  mixed layer, current, tide.
* **weather**   - the atmosphere: wind, gusts, rain, instability, visibility.
* **gis**       - geometry: position, EEZ, restricted zones, nearest shelter.
* **satellite** - which agency products the figures actually came from, so the
  reader can see at a glance how much of the answer rests on ISRO and INCOIS
  data rather than on a fallback model.
"""

from __future__ import annotations

from typing import Any

from ..schemas import Briefing, BriefingItem, Citation, RiskBand, Tier

#: risk flags, worst first, so a reader's eye lands on the problem
FLAG_DANGER = "danger"
FLAG_WATCH = "watch"

#: rule name -> the shortest honest phrasing of why it fired
DRIVER_PHRASE: dict[str, str] = {
    "swh_danger": "waves over the small-craft danger limit",
    "swh_caution": "waves over the caution limit",
    "wind_danger": "wind at or above near-gale",
    "wind_caution": "wind in the squally band",
    "gust_danger": "gusts strong enough to knock a small boat down",
    "convective_risk": "thunderstorm and lightning risk",
    "tropical_cyclone_distance": "tropical cyclone within range",
    "imd_warning_in_force": "IMD warning in force",
    "strong_current": "strong surface current",
    "inside_restricted_zone": "inside a restricted zone",
    "approaching_boundary": "closing on a boundary",
    "outside_eez": "outside the India EEZ",
    "swh_unavailable": "no wave height available",
}


def _fmt(value: Any, places: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if places == 0:
        return f"{number:.0f}"
    return f"{number:.{places}f}"


class BriefingBuilder:
    """Reshapes one request's findings into the numbers-only view."""

    def __init__(self, ctx: Any, citations: list[Citation]) -> None:
        self.ctx = ctx
        self.f = ctx.findings
        # marker lookup so a briefing row can point at the same source the
        # sentence did, rather than inventing its own numbering
        self._marker = {c.evidence_id: c.marker for c in citations}
        # tier comes from the evidence book, not the citation list: a figure the
        # prose did not need still has a source, and hiding it would make an
        # ISRO number look indistinguishable from a fallback one
        self._tier = {
            evidence.id: evidence.provenance.tier
            for evidence in ctx.evidence.all()
        }

    # ---------------------------------------------------------------- helpers #

    def _fields(self, block: str) -> dict[str, Any]:
        return (getattr(self.f, block, None) or {}).get("fields", {}) or {}

    def _eid(self, block: str, key: str) -> str:
        return (self._fields(block).get(key) or {}).get("evidence_id", "")

    def _item(
        self,
        label: str,
        value: Any,
        unit: str = "",
        *,
        places: int = 1,
        note: str = "",
        flag: str = "",
        evidence_id: str = "",
    ) -> BriefingItem | None:
        text = _fmt(value, places)
        if not text:
            return None
        return BriefingItem(
            label=label,
            value=text,
            unit=unit,
            note=note,
            flag=flag,
            marker=self._marker.get(evidence_id),
            tier=self._tier.get(evidence_id),
        )

    @staticmethod
    def _add(target: list[BriefingItem], item: BriefingItem | None) -> None:
        if item is not None:
            target.append(item)

    # ------------------------------------------------------------------ build #

    def build(self) -> Briefing:
        settings = self.ctx.services.settings
        risk = self.ctx.risk
        brief = Briefing()

        if risk is not None:
            brief.risk_band = risk.band.value
            brief.risk_score = risk.score
            brief.headline = risk.headline
            # only the rules that pushed the verdict, worst first
            bad = [
                f for f in risk.findings
                if f.band in (RiskBand.UNSAFE, RiskBand.CAUTION)
                and not f.rule.startswith("jev_")
            ]
            bad.sort(key=lambda f: 0 if f.band is RiskBand.UNSAFE else 1)
            brief.drivers = [
                DRIVER_PHRASE.get(f.rule, f.rule.replace("_", " ")) for f in bad
            ]

        self._ocean(brief, settings)
        self._weather(brief, settings)
        self._gis(brief)
        self._satellite(brief)
        self._missing(brief)
        return brief

    def _ocean(self, brief: Briefing, settings: Any) -> None:
        waves = self.f.waves or {}
        ocean = self.f.ocean or {}

        swh = waves.get("swh_now_m")
        peak = waves.get("swh_peak_m")
        flag = ""
        if swh is not None:
            worst = max(swh, peak) if peak is not None else swh
            if worst >= settings.swh_danger_m:
                flag = FLAG_DANGER
            elif worst >= settings.swh_caution_m:
                flag = FLAG_WATCH
        note = ""
        if peak is not None and swh is not None and peak - swh > 0.2:
            note = f"peak {_fmt(peak)}"
        self._add(brief.ocean, self._item(
            "waves", swh, "m", note=note, flag=flag,
            evidence_id=self._eid("waves", "swh"),
        ))

        period = (self._fields("waves").get("period") or {}).get("value")
        self._add(brief.ocean, self._item(
            "wave period", period, "s", places=0,
            evidence_id=self._eid("waves", "period"),
        ))
        if waves.get("wave_from"):
            brief.ocean.append(BriefingItem(
                label="wave dir", value=str(waves["wave_from"]),
                marker=self._marker.get(self._eid("waves", "swh")),
            ))

        sst = (self._fields("ocean").get("sst") or {}).get("value")
        self._add(brief.ocean, self._item(
            "sea temp", sst, "degC", evidence_id=self._eid("ocean", "sst"),
        ))

        salinity = (self._fields("ocean").get("salinity") or {}).get("value")
        self._add(brief.ocean, self._item(
            "salinity", salinity, "psu", evidence_id=self._eid("ocean", "salinity"),
        ))

        mld = (self._fields("ocean").get("mld") or {}).get("value")
        self._add(brief.ocean, self._item(
            "mixed layer", mld, "m", places=0,
            evidence_id=self._eid("ocean", "mld"),
        ))

        current = (self._fields("ocean").get("current") or {}).get("value")
        self._add(brief.ocean, self._item(
            "current", current, "cm/s", places=0,
            note=str(ocean.get("current_set") or ""),
            flag=FLAG_WATCH if (
                current is not None and current >= settings.current_caution_cms
            ) else "",
            evidence_id=self._eid("ocean", "current"),
        ))

        tide = waves.get("tide_now_m")
        low, high = waves.get("tide_low_m"), waves.get("tide_high_m")
        self._add(brief.ocean, self._item(
            "tide", tide, "m", places=2,
            note=(f"{_fmt(low, 1)} to {_fmt(high, 1)}" if low is not None and high is not None else ""),
            evidence_id=self._eid("waves", "tide"),
        ))

    def _weather(self, brief: Briefing, settings: Any) -> None:
        weather = self.f.weather or {}

        wind = weather.get("wind_kt")
        peak = weather.get("wind_peak_kt")
        flag = ""
        if wind is not None:
            worst = max(wind, peak) if peak is not None else wind
            if worst >= settings.wind_danger_kt:
                flag = FLAG_DANGER
            elif worst >= settings.wind_caution_kt:
                flag = FLAG_WATCH
        note = ""
        if peak is not None and wind is not None and peak - wind > 1:
            note = f"peak {_fmt(peak, 0)}"
        self._add(brief.weather, self._item(
            "wind", wind, "kt", places=0, note=note, flag=flag,
            evidence_id=self._eid("weather", "wind"),
        ))
        if weather.get("wind_from"):
            brief.weather.append(BriefingItem(
                label="wind dir", value=str(weather["wind_from"]),
                marker=self._marker.get(self._eid("weather", "wind")),
            ))

        gust = weather.get("gust_peak_kt")
        self._add(brief.weather, self._item(
            "gusts", gust, "kt", places=0,
            flag=FLAG_DANGER if (
                gust is not None and gust >= settings.gust_danger_kt
            ) else "",
            evidence_id=self._eid("weather", "gust"),
        ))

        rain = weather.get("rain_peak_mmh")
        self._add(brief.weather, self._item(
            "rain", rain, "mm/h", evidence_id=self._eid("weather", "rain"),
        ))

        convective = weather.get("convective_risk") or {}
        cape = convective.get("cape_j_per_kg", weather.get("cape_peak"))
        band = str(convective.get("band") or "")
        self._add(brief.weather, self._item(
            "CAPE", cape, "J/kg", places=0,
            note=(f"{band} storm risk" if band else ""),
            flag=(
                FLAG_DANGER if band == "high"
                else FLAG_WATCH if band == "moderate" else ""
            ),
            evidence_id=self._eid("weather", "cape"),
        ))

        visibility = weather.get("visibility_min_m")
        self._add(brief.weather, self._item(
            "visibility", visibility, "m", places=0,
            flag=FLAG_WATCH if (visibility is not None and visibility < 2000) else "",
            evidence_id=self._eid("weather", "visibility"),
        ))

    def _gis(self, brief: Briefing) -> None:
        geo = self.f.geo or {}
        location = self.ctx.location

        if location is not None:
            brief.gis.append(BriefingItem(
                label="position",
                value=f"{location.lat:.3f}N {location.lon:.3f}E",
                note=location.name if location.name else "",
            ))

        eez = geo.get("eez") or {}
        if eez.get("available"):
            inside = bool(eez.get("inside_india_eez"))
            self._add(brief.gis, self._item(
                "EEZ boundary", eez.get("distance_to_boundary_km"), "km",
                places=0,
                note=("inside India EEZ" if inside else "OUTSIDE India EEZ"),
                flag="" if inside else FLAG_DANGER,
                evidence_id="eez-status",
            ))

        for zone in (geo.get("zones") or [])[:3]:
            status = str(zone.get("status") or "")
            self._add(brief.gis, self._item(
                str(zone.get("name") or "zone"), zone.get("distance_km"), "km",
                places=0,
                note=f"{status} {zone.get('bearing', '')}".strip(),
                flag=(
                    FLAG_DANGER if status == "inside" and zone.get("danger")
                    else FLAG_WATCH if status in ("inside", "approaching") else ""
                ),
                evidence_id=str(zone.get("evidence_id") or ""),
            ))

        harbours = geo.get("harbours") or []
        if harbours:
            first = harbours[0]
            self._add(brief.gis, self._item(
                "nearest shelter", first.get("distance_km"), "km", places=0,
                note=f"{first.get('name', '')} {first.get('bearing', '')}".strip(),
                evidence_id=str(first.get("evidence_id") or ""),
            ))

        route = self.f.route or {}
        if route:
            best = route.get("recommended") or {}
            self._add(brief.gis, self._item(
                "route", best.get("length_km") or route.get("direct_km"), "km",
                places=0,
                note=str(route.get("initial_bearing") or ""),
            ))

    def _satellite(self, brief: Briefing) -> None:
        """Which agency products the figures rest on, best tier first."""
        official = {Tier.ISRO, Tier.INCOIS, Tier.IMD}
        seen: set[str] = set()
        for evidence in self.ctx.evidence.all():
            provenance = evidence.provenance
            if provenance.tier not in official:
                continue
            key = f"{provenance.agency}|{provenance.dataset}"
            if key in seen:
                continue
            seen.add(key)
            # dataset titles run long; keep the head, which is the product name
            label = (provenance.dataset or provenance.agency).split(" - ")[0]
            brief.satellite.append(BriefingItem(
                label=label[:44],
                value=(
                    _fmt(evidence.value, 1) if isinstance(evidence.value, (int, float))
                    else "text"
                ),
                unit=evidence.unit or "",
                note=(
                    "stale" if provenance.is_stale
                    else provenance.access_method or ""
                ),
                flag=FLAG_WATCH if provenance.is_stale else "",
                marker=self._marker.get(evidence.id),
                tier=provenance.tier,
            ))
        brief.satellite.sort(key=lambda i: (i.tier or Tier.SEED).value)

    def _missing(self, brief: Briefing) -> None:
        """Gaps, stated plainly. An absent number is not a safe number."""
        if self.f.waves and self.f.waves.get("swh_now_m") is None:
            brief.missing.append("wave height")
        if self.f.weather and self.f.weather.get("wind_kt") is None:
            brief.missing.append("wind")
        hazards = self.f.hazards or {}
        if hazards and hazards.get("cyclone") is None:
            degraded = self.ctx.services.registry.degraded()
            if any("gdacs" in d for d in degraded):
                brief.missing.append("cyclone check (GDACS unavailable)")
        for note in self.ctx.findings.notes:
            if "could not be read" in note or "unavailable" in note:
                brief.missing.append(note.split(".")[0][:70])
                break


def build_briefing(ctx: Any, citations: list[Citation]) -> Briefing:
    return BriefingBuilder(ctx, citations).build()


def briefing_text(brief: Briefing) -> str:
    """Compact plain-text form, for API callers and the voice path."""
    lines: list[str] = []
    band = brief.risk_band.upper()
    lines.append(f"RISK: {band}" + (f" ({brief.risk_score:.0f}/100)" if brief.risk_score else ""))
    if brief.drivers:
        lines.append("WHY: " + "; ".join(brief.drivers))

    def group(title: str, items: list[BriefingItem]) -> None:
        if not items:
            return
        parts = []
        for item in items:
            text = f"{item.label} {item.value}{(' ' + item.unit) if item.unit else ''}"
            if item.note:
                text += f" ({item.note})"
            if item.flag:
                text += f" !{item.flag}"
            parts.append(text)
        lines.append(f"{title}: " + ", ".join(parts))

    group("OCEAN", brief.ocean)
    group("WEATHER", brief.weather)
    group("GIS", brief.gis)
    group("SATELLITE", brief.satellite)
    if brief.missing:
        lines.append("MISSING: " + ", ".join(brief.missing))
    return "\n".join(lines)
