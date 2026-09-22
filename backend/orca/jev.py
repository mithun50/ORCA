"""TypeSafe AI Jev Decision Engine.

Integrates the Jev "System One" decision model (by TypeSafe AI) for structured,
programmatic judgments in high-consequence maritime scenarios:
- Choice: discrete classification (e.g. SAFE, CAUTION, UNSAFE)
- Noul: probabilistic boolean estimate in [0.0, 1.0] (e.g. boundary breach probability)
- Score: continuous numerical impact rating (e.g. 0 to 100 severity)

When ORCA_JEV_API_KEY or TYPESAFE_API_KEY is configured, queries the Jev
/v1/systemone endpoint. Otherwise, executes a deterministic domain emulator
so the system remains fully operational and reproducible offline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import get_settings

log = logging.getLogger("orca.jev")


@dataclass
class JevSafetyDecision:
    safety_verdict: str  # SAFE | CAUTION | UNSAFE
    action: str          # VENTURE_PERMITTED | EXERCISE_VIGILANCE | STAY_IN_HARBOR
    risk_score: float    # 0.0 to 100.0
    breach_probability: float  # 0.0 to 1.0
    capsizing_probability: float  # 0.0 to 1.0
    engine: str          # "jev-cloud" | "jev-rule-emulator"
    confidence: float = 0.95
    rationale: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)


class JevDecisionEngine:
    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.effective_jev_api_key)

    async def evaluate_safety(
        self,
        *,
        location_name: str,
        swh_m: float | None,
        wind_kt: float | None,
        gust_kt: float | None,
        cyclone_alert: bool = False,
        cyclone_dist_km: float | None = None,
        nearest_boundary_dist_km: float | None = None,
        restricted_zone_name: str | None = None,
        imd_warning_active: bool = False,
    ) -> JevSafetyDecision:
        """Evaluates marine safety state using Jev System-One model or rule emulator."""
        state_description = (
            f"Vessel operating near {location_name}. "
            f"Significant wave height: {swh_m if swh_m is not None else 'unknown'} m. "
            f"Wind speed: {wind_kt if wind_kt is not None else 'unknown'} knots (gusts {gust_kt if gust_kt is not None else 'unknown'} kt). "
            f"Active cyclone in north Indian Ocean: {cyclone_alert} (distance: {cyclone_dist_km if cyclone_dist_km is not None else 'N/A'} km). "
            f"Proximity to restricted maritime boundary ({restricted_zone_name or 'none'}): {nearest_boundary_dist_km if nearest_boundary_dist_km is not None else 'far'} km. "
            f"Official IMD small craft alert active: {imd_warning_active}."
        )

        if self.is_configured:
            cloud_decision = await self._query_jev_api(state_description)
            if cloud_decision:
                return cloud_decision

        return self._emulate_decision(
            swh_m=swh_m,
            wind_kt=wind_kt,
            gust_kt=gust_kt,
            cyclone_alert=cyclone_alert,
            cyclone_dist_km=cyclone_dist_km,
            boundary_dist_km=nearest_boundary_dist_km,
            imd_warning_active=imd_warning_active,
        )

    async def _query_jev_api(self, state: str) -> JevSafetyDecision | None:
        """Invokes TypeSafe AI's /v1/systemone endpoint."""
        key = self.settings.effective_jev_api_key
        url = f"{self.settings.jev_base.rstrip('/')}/systemone"
        payload = {
            "model": self.settings.jev_model or "jev-latest",
            "state": state,
            "questions": {
                "safety_verdict": {
                    "type": "choice",
                    "options": ["SAFE", "CAUTION", "UNSAFE"],
                    "instructions": "Classify maritime safety band for small artisanal and motorized fishing crafts."
                },
                "action": {
                    "type": "choice",
                    "options": ["VENTURE_PERMITTED", "EXERCISE_VIGILANCE", "STAY_IN_HARBOR"],
                    "instructions": "Recommended operational command."
                },
                "risk_score": {
                    "type": "score",
                    "min": 0,
                    "max": 100,
                    "instructions": "Overall numerical composite hazard severity score from 0 (calm) to 100 (extreme danger)."
                },
                "breach_probability": {
                    "type": "noul",
                    "instructions": "Is there a critical probability of violating restricted border or core marine protected area?"
                },
                "capsizing_probability": {
                    "type": "noul",
                    "instructions": "Is there a dangerous probability of craft capsizing or severe swamping from waves/wind?"
                }
            }
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                if resp.status_code != 200:
                    log.warning("Jev API returned %s: %s", resp.status_code, resp.text[:200])
                    return None
                data = resp.json()
                answers = data.get("answers", {})
                
                verdict = (answers.get("safety_verdict") or {}).get("choice", "CAUTION")
                action = (answers.get("action") or {}).get("choice", "EXERCISE_VIGILANCE")
                score = float((answers.get("risk_score") or {}).get("score", 50.0))
                breach_prob = float((answers.get("breach_probability") or {}).get("noul", 0.1))
                capsize_prob = float((answers.get("capsizing_probability") or {}).get("noul", 0.1))

                return JevSafetyDecision(
                    safety_verdict=verdict,
                    action=action,
                    risk_score=score,
                    breach_probability=breach_prob,
                    capsizing_probability=capsize_prob,
                    engine="jev-cloud",
                    confidence=float((answers.get("safety_verdict") or {}).get("confidence", 0.96)),
                    rationale=f"Jev System-One judgment: verdict={verdict}, score={score:.1f}/100",
                    raw_response=data,
                )
        except Exception as exc:
            log.warning("Failed calling Jev API (%s), falling back to emulator", exc)
            return None

    def _emulate_decision(
        self,
        *,
        swh_m: float | None,
        wind_kt: float | None,
        gust_kt: float | None,
        cyclone_alert: bool,
        cyclone_dist_km: float | None,
        boundary_dist_km: float | None,
        imd_warning_active: bool,
    ) -> JevSafetyDecision:
        """Deterministic safety judgment emulator replicating Jev decision schema."""
        score = 10.0
        breach_prob = 0.02
        capsize_prob = 0.02

        if boundary_dist_km is not None:
            if boundary_dist_km <= 1.5:
                breach_prob = 0.92
                score += 45.0
            elif boundary_dist_km <= 5.0:
                breach_prob = 0.65
                score += 25.0
            elif boundary_dist_km <= 10.0:
                breach_prob = 0.25
                score += 10.0

        if swh_m is not None:
            if swh_m >= self.settings.swh_danger_m:  # 2.5m
                capsize_prob = max(capsize_prob, 0.85)
                score += 40.0
            elif swh_m >= self.settings.swh_caution_m:  # 1.5m
                capsize_prob = max(capsize_prob, 0.40)
                score += 20.0

        if wind_kt is not None and wind_kt >= self.settings.wind_danger_kt:  # 27kt
            capsize_prob = max(capsize_prob, 0.80)
            score += 30.0
        elif wind_kt is not None and wind_kt >= self.settings.wind_caution_kt:  # 17kt
            score += 15.0

        if cyclone_alert:
            score += 40.0
            capsize_prob = max(capsize_prob, 0.90)

        if imd_warning_active:
            score += 25.0

        score = min(100.0, score)

        if cyclone_alert or capsize_prob >= 0.70 or breach_prob >= 0.80 or score >= 65.0:
            verdict = "UNSAFE"
            action = "STAY_IN_HARBOR"
        elif score >= 35.0 or capsize_prob >= 0.30 or breach_prob >= 0.40:
            verdict = "CAUTION"
            action = "EXERCISE_VIGILANCE"
        else:
            verdict = "SAFE"
            action = "VENTURE_PERMITTED"

        return JevSafetyDecision(
            safety_verdict=verdict,
            action=action,
            risk_score=round(score, 1),
            breach_probability=round(breach_prob, 2),
            capsizing_probability=round(capsize_prob, 2),
            engine="jev-rule-emulator",
            confidence=0.98,
            rationale=f"Evaluated state: verdict={verdict}, action={action}, score={score:.1f}/100",
        )


_jev_engine: JevDecisionEngine | None = None


def get_jev_engine() -> JevDecisionEngine:
    global _jev_engine
    if _jev_engine is None:
        _jev_engine = JevDecisionEngine()
    return _jev_engine
