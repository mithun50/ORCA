"""Ocean analytics and weather intelligence agents.

Both are thin over the time-series RAG on purpose: retrieval policy (which
agency, which fallback, how to handle a masked cell) belongs in the retriever,
while the agents own interpretation - converting numbers into the sea-state
language a boat owner actually uses, and recording what that interpretation
rests on.
"""

from __future__ import annotations

import math
from typing import Any

from ..geo.geometry import compass
from ..schemas import ChartSeries
from ..services import Services
from .base import AgentContext

BEAUFORT = (
    (1.0, 0, "calm"),
    (3.0, 1, "light air"),
    (6.0, 2, "light breeze"),
    (10.0, 3, "gentle breeze"),
    (16.0, 4, "moderate breeze"),
    (21.0, 5, "fresh breeze"),
    (27.0, 6, "strong breeze"),
    (33.0, 7, "near gale"),
    (40.0, 8, "gale"),
    (47.0, 9, "strong gale"),
    (55.0, 10, "storm"),
)

SEA_STATE = (
    (0.1, "glassy"),
    (0.5, "smooth"),
    (1.25, "slight"),
    (2.5, "moderate"),
    (4.0, "rough"),
    (6.0, "very rough"),
    (9.0, "high"),
)


def beaufort(knots: float | None) -> tuple[int, str]:
    if knots is None:
        return -1, "unknown"
    for limit, force, label in BEAUFORT:
        if knots < limit:
            return force, label
    return 11, "violent storm"


def sea_state(swh_m: float | None) -> str:
    if swh_m is None:
        return "unknown"
    for limit, label in SEA_STATE:
        if swh_m < limit:
            return label
    return "very high"


class OceanAnalyticsAgent:
    """SST, currents, salinity, mixed layer depth and wave state at a point."""

    name = "ocean-analytics"
    tools = (
        "mosdac.osf_circ.ncss_point",
        "mosdac.osf_wave.ncss_point",
        "openmeteo.marine",
    )

    def __init__(self, services: Services) -> None:
        self.services = services

    async def run(self, ctx: AgentContext) -> None:
        rag = self.services.timeseries_rag

        with ctx.trace.timed(
            self.name,
            "retrieve ocean state from the ISRO Ocean State Forecast",
            rationale=(
                "MOSDAC OSF_CIRC is the official Indian ocean model, so it is "
                "queried before any non-Indian source"
            ),
            tool="mosdac.osf_circ.ncss_point",
            tool_args={
                "lat": round(ctx.lat, 3),
                "lon": round(ctx.lon, 3),
                "window": f"{ctx.start:%Y-%m-%dT%H:%MZ}..{ctx.end:%Y-%m-%dT%H:%MZ}",
                "vars": list(self.services.mosdac.OSF_CIRC_VARS) + ["hmxl"],
            },
        ) as step:
            ocean = await rag.ocean_state(ctx.lat, ctx.lon, ctx.start, ctx.end)
            if not ocean:
                step.status = "degraded"
                step.outcome = "no ocean-model values returned for this point"
            else:
                ids = ctx.evidence.add_many([f.evidence for f in ocean.values()])
                step.evidence_ids = ids
                sst = ocean.get("sst")
                bits = []
                if sst is not None:
                    bits.append(f"SST {sst.at_or_first(ctx.start):.2f} degC")
                current = ocean.get("current")
                if current is not None:
                    bits.append(f"current {current.at_or_first(ctx.start):.0f} cm/s")
                mld = ocean.get("mld")
                if mld is not None:
                    bits.append(f"mixed layer {mld.at_or_first(ctx.start):.0f} m")
                step.outcome = ", ".join(bits) or "retrieved"
                if sst is not None and sst.used_fallback:
                    step.status = "degraded"
                for pf in ocean.values():
                    if pf.note:
                        ctx.note(pf.note)
            ctx.findings.ocean = self._summarise_ocean(ocean, ctx)

        with ctx.trace.timed(
            self.name,
            "retrieve wave state",
            rationale=(
                "ISRO OSF_WAVE first for the official forecast, plus a live "
                "cross-check because that aggregate's cycle can be stale"
            ),
            tool="mosdac.osf_wave.ncss_point",
            tool_args={"lat": round(ctx.lat, 3), "lon": round(ctx.lon, 3)},
        ) as step:
            waves = await rag.wave_state(ctx.lat, ctx.lon, ctx.start, ctx.end)
            if not waves:
                step.status = "degraded"
                step.outcome = "no wave data available for this point"
            else:
                step.evidence_ids = ctx.evidence.add_many(
                    [f.evidence for f in waves.values()]
                )
                swh = waves.get("swh")
                peak = swh.max() if swh else None
                step.outcome = (
                    f"SWH now {swh.at_or_first(ctx.start):.2f} m, peak "
                    f"{peak:.2f} m in the window" if swh and peak else "retrieved"
                )
                isro = waves.get("swh_isro")
                if isro is not None and isro.evidence is not None:
                    if isro.evidence.provenance.is_stale:
                        step.status = "degraded"
                        ctx.note(
                            "The ISRO OSF_WAVE aggregate on MOSDAC is serving an "
                            "older forecast cycle, so the live wave height quoted "
                            "here is from the fallback model and the ISRO value is "
                            "shown alongside for comparison."
                        )
            ctx.findings.waves = self._summarise_waves(waves, ctx)

    # ------------------------------------------------------------ summaries #

    @staticmethod
    def _summarise_ocean(ocean: dict[str, Any], ctx: AgentContext) -> dict[str, Any]:
        out: dict[str, Any] = {"fields": {}}
        for key, pf in ocean.items():
            value = pf.at_or_first(ctx.start)
            out["fields"][key] = {
                "label": pf.label,
                "value": value,
                "unit": pf.unit,
                "min": pf.min(),
                "max": pf.max(),
                "mean": pf.mean(),
                "tier": pf.source_tier.value,
                "fallback": pf.used_fallback,
                "evidence_id": pf.evidence.id if pf.evidence else "",
            }
        current = ocean.get("current")
        if current and current.series:
            first = current.series[0]
            direction = first.get("direction_deg")
            if isinstance(direction, (int, float)):
                out["current_set"] = compass(float(direction))
                out["current_set_deg"] = round(float(direction), 1)
        if "sst" in out["fields"]:
            ctx.add_chart(
                ChartSeries(
                    id="chart-sst",
                    title="Sea surface temperature (ISRO Ocean State Forecast)",
                    unit="degC",
                    x=[row["time"] for row in ocean["sst"].series],
                    y=[row["value"] for row in ocean["sst"].series],
                    source=ocean["sst"].source_tier.value,
                )
            )
        if "mld" in out["fields"]:
            ctx.add_chart(
                ChartSeries(
                    id="chart-mld",
                    title="Mixed layer depth",
                    unit="m",
                    x=[row["time"] for row in ocean["mld"].series],
                    y=[row["value"] for row in ocean["mld"].series],
                    source=ocean["mld"].source_tier.value,
                )
            )
        return out

    def _summarise_waves(
        self, waves: dict[str, Any], ctx: AgentContext
    ) -> dict[str, Any]:
        settings = self.services.settings
        out: dict[str, Any] = {"fields": {}}
        for key, pf in waves.items():
            out["fields"][key] = {
                "label": pf.label,
                "value": pf.at_or_first(ctx.start),
                "unit": pf.unit,
                "max": pf.max(),
                "peak_time": pf.window_of_max(),
                "tier": pf.source_tier.value,
                "fallback": pf.used_fallback,
                "note": pf.note,
                "evidence_id": pf.evidence.id if pf.evidence else "",
            }
        swh = waves.get("swh")
        if swh:
            now_value = swh.at_or_first(ctx.start)
            out["sea_state"] = sea_state(now_value)
            out["swh_now_m"] = now_value
            out["swh_peak_m"] = swh.max()
            out["swh_peak_time"] = swh.window_of_max()
            ctx.add_chart(
                ChartSeries(
                    id="chart-swh",
                    title="Significant wave height",
                    unit="m",
                    x=[row["time"] for row in swh.series],
                    y=[row["value"] for row in swh.series],
                    threshold=settings.swh_caution_m,
                    threshold_label=f"caution above {settings.swh_caution_m} m",
                    source=swh.source_tier.value,
                )
            )
        isro = waves.get("swh_isro")
        if isro:
            out["swh_isro_m"] = isro.at_or_first(ctx.start)
            out["swh_isro_peak_m"] = isro.max()
        direction = waves.get("wave_dir_isro")
        if direction:
            heading = direction.at_or_first(ctx.start)
            if heading is not None:
                out["wave_from"] = compass(float(heading))
        tide = waves.get("tide")
        if tide and tide.series:
            out["tide_now_m"] = tide.at_or_first(ctx.start)
            out["tide_high_m"] = tide.max()
            out["tide_low_m"] = tide.min()
            out["tide_high_time"] = tide.window_of_max()
            ctx.add_chart(
                ChartSeries(
                    id="chart-tide",
                    title="Sea level above MSL (tide)",
                    unit="m",
                    x=[row["time"] for row in tide.series],
                    y=[row["value"] for row in tide.series],
                    source=tide.source_tier.value,
                )
            )
        return out


class WeatherIntelligenceAgent:
    """Wind, gusts, rain, convective potential and visibility."""

    name = "weather-intelligence"
    tools = ("openmeteo.forecast", "imd.subdivision_warnings")

    def __init__(self, services: Services) -> None:
        self.services = services

    async def run(self, ctx: AgentContext) -> None:
        settings = self.services.settings
        with ctx.trace.timed(
            self.name,
            "retrieve marine weather",
            rationale=(
                "IMD's JSON API needs a department-issued key, so the numeric "
                "wind field comes from the fallback tier while IMD's own warning "
                "wording is retrieved separately by the advisory agent"
            ),
            tool="openmeteo.forecast",
            tool_args={"lat": round(ctx.lat, 3), "lon": round(ctx.lon, 3)},
        ) as step:
            weather = await self.services.timeseries_rag.weather_state(
                ctx.lat, ctx.lon, ctx.start, ctx.end
            )
            if not weather:
                step.status = "failed"
                step.outcome = "no weather data available"
                ctx.findings.weather = {}
                return
            step.evidence_ids = ctx.evidence.add_many(
                [f.evidence for f in weather.values()]
            )
            wind = weather.get("wind")
            gust = weather.get("gust")
            force, label = beaufort(wind.at_or_first(ctx.start) if wind else None)
            step.outcome = (
                f"wind {wind.at_or_first(ctx.start):.0f} kt (force {force}, {label})"
                if wind
                else "retrieved"
            )
            if gust and gust.max() is not None:
                step.outcome += f", gusting to {gust.max():.0f} kt"

        out: dict[str, Any] = {"fields": {}}
        for key, pf in weather.items():
            out["fields"][key] = {
                "label": pf.label,
                "value": pf.at_or_first(ctx.start),
                "unit": pf.unit,
                "max": pf.max(),
                "peak_time": pf.window_of_max(),
                "tier": pf.source_tier.value,
                "evidence_id": pf.evidence.id if pf.evidence else "",
            }
        wind = weather.get("wind")
        if wind:
            force, label = beaufort(wind.at_or_first(ctx.start))
            out["wind_kt"] = wind.at_or_first(ctx.start)
            out["wind_peak_kt"] = wind.max()
            out["beaufort"] = force
            out["beaufort_label"] = label
            ctx.add_chart(
                ChartSeries(
                    id="chart-wind",
                    title="Wind speed at 10 m",
                    unit="kt",
                    x=[row["time"] for row in wind.series],
                    y=[row["value"] for row in wind.series],
                    threshold=settings.wind_caution_kt,
                    threshold_label=f"caution above {settings.wind_caution_kt} kt",
                    source=wind.source_tier.value,
                )
            )
        direction = weather.get("wind_dir")
        if direction:
            heading = direction.at_or_first(ctx.start)
            if heading is not None:
                out["wind_from"] = compass(float(heading))
        gust = weather.get("gust")
        if gust:
            out["gust_peak_kt"] = gust.max()
            out["gust_peak_time"] = gust.window_of_max()
        rain = weather.get("rain")
        if rain:
            out["rain_peak_mmh"] = rain.max()
            out["rain_total_mm"] = sum(rain.values())
        cape = weather.get("cape")
        if cape:
            out["cape_peak"] = cape.max()
            out["convective_risk"] = self._convective_risk(
                cape.max(), rain.max() if rain else None
            )
        visibility = weather.get("visibility")
        if visibility and visibility.min() is not None:
            out["visibility_min_m"] = visibility.min()
        ctx.findings.weather = out

    @staticmethod
    def _convective_risk(cape: float | None, rain: float | None) -> dict[str, Any]:
        """Thunderstorm/lightning proxy.

        India has no public lightning-strike API (IMD's Damini network is
        app-only), so ORCA reports a clearly-labelled proxy built from CAPE and
        rainfall rather than claiming an observed strike.
        """
        cape = cape or 0.0
        rain = rain or 0.0
        if cape >= 2500 and rain >= 2.0:
            band, text = "high", "conditions strongly favour thunderstorms with lightning"
        elif cape >= 3000:
            # deep instability on its own is enough to expect afternoon and
            # evening storms along the Indian coast, even with a dry model hour
            band, text = (
                "high",
                "deep convective instability, enough for thunderstorms to develop "
                "quickly even though little rain is forecast for this hour",
            )
        elif cape >= 1500 and rain >= 0.5:
            band, text = "moderate", "conditions favour isolated thunderstorms"
        elif cape >= 1500:
            band, text = (
                "moderate",
                "moderate convective instability with little forecast rain",
            )
        elif cape >= 1000:
            band, text = "low", "some convective instability, but little rain forecast"
        else:
            band, text = "minimal", "no significant convective instability"
        return {
            "band": band,
            "explanation": text,
            "cape_j_per_kg": round(cape, 0),
            "rain_peak_mm_per_h": round(rain, 2),
            "is_proxy": True,
            "proxy_note": (
                "derived indicator, not an observed lightning strike. IMD's Damini "
                "lightning network has no public API; treat this as a likelihood "
                "signal and check the IMD nowcast before sailing."
            ),
        }


def knots_from_ms(value: float | None) -> float | None:
    return None if value is None else value * 1.94384


def vector_magnitude(u: float | None, v: float | None) -> float | None:
    if u is None or v is None:
        return None
    return math.hypot(u, v)
