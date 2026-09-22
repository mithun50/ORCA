"""Jev System-One structured decision engine (TypeSafe AI), via OpenRouter.

Jev is not a chat model. It is TypeSafe AI's first "System One" model: you send
one `state` plus a map of typed `questions`, and it returns typed `answers`
keyed by the same ids, with calibrated probabilities. It never generates prose.
That is exactly the right shape for a safety judgment that code has to act on.

The three primitives, per https://docs.typesafe.ai/api :

* **choice** - pick one option. `criteria` is a map of option to rubric text.
  The answer carries `choice`, a full `probabilities` map, and `confidence`.
* **score**  - rate against an ordered rubric. `criteria` is an ordered array of
  2 to 10 level descriptions. The answer is a probability-weighted float across
  the level indices, so it can land between levels, plus `legend`,
  `probabilities` and `confidence`.
* **noul**   - one yes/no judgment. The answer is the probability of yes, in
  [0, 1]. A noul answer carries **no** confidence field.

Two transports, one wire protocol:

* `jev_provider=openrouter` -> `POST {jev_base}/decisions`, model
  `typesafe/jev-1.13`. This is OpenRouter's Decisions API, which is separate
  from its OpenAI-compatible chat endpoint. Chat SDKs do not work against it.
  OpenRouter rejects the `typesafe/jev-latest` alias with a 400; only the
  versioned slug resolves there.
* `jev_provider=typesafe`   -> `POST {jev_base}/systemone`, model `jev-latest`.
  TypeSafe's own endpoint, which does accept the alias.

With no key, or if the call fails, times out or returns an off-schema answer, a
deterministic domain emulator runs instead, so the pipeline stays operational
and reproducible offline.

This judgment is advisory. `agents/risk.py` combines it with the threshold rules
by worst-band-wins, so Jev can make a verdict more conservative but never less.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import get_settings

log = logging.getLogger("orca.jev")

#: choice options, ordered least to most severe
VERDICTS: tuple[str, ...] = ("SAFE", "CAUTION", "UNSAFE")
ACTIONS: tuple[str, ...] = (
    "VENTURE_PERMITTED",
    "EXERCISE_VIGILANCE",
    "STAY_IN_HARBOR",
)

#: ordered score rubric. The answer is a weighted float over these indices.
SEVERITY_LEVELS: tuple[str, ...] = (
    "Benign. Calm sea, light wind, no warning in force, well inside Indian waters.",
    "Marginal. Sea or wind building toward the small-craft caution threshold.",
    "Hazardous for small craft. Caution thresholds exceeded, or a warning is in force.",
    "Dangerous. Danger thresholds exceeded, or a cyclone is within a few hundred km.",
    "Extreme. Survival conditions, or an imminent boundary or cyclone emergency.",
)

QUESTIONS: dict[str, dict[str, Any]] = {
    "safety_verdict": {
        "type": "choice",
        "instructions": (
            "Classify the maritime safety band for a small artisanal or "
            "motorised fishing craft operating in this state."
        ),
        "criteria": {
            "SAFE": "Conditions are workable for a small craft with normal seamanship.",
            "CAUTION": (
                "Workable only with vigilance, or unsafe for non-mechanised craft: "
                "sea or wind near advisory thresholds, or a warning in force."
            ),
            "UNSAFE": (
                "Should not put to sea: danger thresholds exceeded, a cyclone "
                "nearby, or a restricted boundary about to be breached."
            ),
        },
    },
    "action": {
        "type": "choice",
        "instructions": "Give the single operational command that follows from this state.",
        "criteria": {
            "VENTURE_PERMITTED": "Proceed as planned.",
            "EXERCISE_VIGILANCE": "Proceed only with heightened watch and an early return plan.",
            "STAY_IN_HARBOR": "Do not put to sea, or return immediately if already out.",
        },
    },
    "severity": {
        "type": "score",
        "instructions": "Rate the overall hazard severity for a small fishing craft.",
        "criteria": list(SEVERITY_LEVELS),
    },
    "breach_probability": {
        "type": "noul",
        "instructions": (
            "Is this craft at risk of crossing an international maritime boundary "
            "or entering a core marine protected area from this position?"
        ),
        "criteria": {
            "true": "A breach is likely without a deliberate course change.",
            "false": "The craft is comfortably clear of every restricted boundary.",
        },
    },
    "capsizing_probability": {
        "type": "noul",
        "instructions": (
            "Is this craft at risk of capsizing or severe swamping from the wave "
            "and wind state described?"
        ),
        "criteria": {
            "true": "Wave height or gusts could knock down or swamp a small craft.",
            "false": "Sea and wind are well within what a small craft handles.",
        },
    },
}


@dataclass
class JevSafetyDecision:
    safety_verdict: str  # SAFE | CAUTION | UNSAFE
    action: str          # VENTURE_PERMITTED | EXERCISE_VIGILANCE | STAY_IN_HARBOR
    risk_score: float    # 0.0 to 100.0
    breach_probability: float  # 0.0 to 1.0
    capsizing_probability: float  # 0.0 to 1.0
    engine: str          # "jev-openrouter" | "jev-typesafe" | "jev-rule-emulator"
    confidence: float = 0.95
    rationale: str = ""
    model: str = ""
    #: per-option probability distribution for the verdict, when Jev answered
    verdict_probabilities: dict[str, float] = field(default_factory=dict)
    #: usage reported by the endpoint. OpenRouter's decisions API returns a real
    #: dollar cost; output tokens are free on Jev so only input drives it.
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    elapsed_ms: int = 0
    raw_response: dict[str, Any] = field(default_factory=dict)


def _clamp(value: Any, low: float, high: float, default: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


class JevDecisionEngine:
    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.effective_jev_api_key)

    @property
    def provider(self) -> str:
        return (self.settings.jev_provider or "openrouter").lower()

    @property
    def model(self) -> str:
        if self.settings.jev_model:
            return self.settings.jev_model
        return "typesafe/jev-1.13" if self.provider == "openrouter" else "jev-latest"

    @property
    def endpoint(self) -> str:
        base = self.settings.jev_base.rstrip("/")
        path = "decisions" if self.provider == "openrouter" else "systemone"
        return f"{base}/{path}"

    def build_state(
        self,
        *,
        location_name: str,
        swh_m: float | None,
        wind_kt: float | None,
        gust_kt: float | None,
        cyclone_alert: bool,
        cyclone_dist_km: float | None,
        nearest_boundary_dist_km: float | None,
        restricted_zone_name: str | None,
        imd_warning_active: bool,
    ) -> dict[str, Any]:
        """Structured state. Jev accepts an object, which beats a prose blob."""
        return {
            "position": location_name,
            "sea_state": {
                "significant_wave_height_m": swh_m,
                "wind_speed_kt": wind_kt,
                "wind_gust_kt": gust_kt,
            },
            "thresholds_for_small_craft": {
                "wave_caution_m": self.settings.swh_caution_m,
                "wave_danger_m": self.settings.swh_danger_m,
                "wind_caution_kt": self.settings.wind_caution_kt,
                "wind_danger_kt": self.settings.wind_danger_kt,
                "gust_danger_kt": self.settings.gust_danger_kt,
            },
            "hazards": {
                "active_cyclone_north_indian_ocean": cyclone_alert,
                "cyclone_distance_km": cyclone_dist_km,
                "imd_small_craft_warning_active": imd_warning_active,
            },
            "boundaries": {
                "nearest_restricted_zone": restricted_zone_name,
                "distance_to_nearest_restricted_zone_km": nearest_boundary_dist_km,
            },
            "vessel": "small artisanal or motorised fishing craft, Indian waters",
        }

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
        """Evaluates marine safety using Jev System-One, or the rule emulator."""
        state = self.build_state(
            location_name=location_name,
            swh_m=swh_m,
            wind_kt=wind_kt,
            gust_kt=gust_kt,
            cyclone_alert=cyclone_alert,
            cyclone_dist_km=cyclone_dist_km,
            nearest_boundary_dist_km=nearest_boundary_dist_km,
            restricted_zone_name=restricted_zone_name,
            imd_warning_active=imd_warning_active,
        )

        if self.is_configured:
            decision = await self._query_jev_api(state)
            if decision:
                return decision

        return self._emulate_decision(
            swh_m=swh_m,
            wind_kt=wind_kt,
            gust_kt=gust_kt,
            cyclone_alert=cyclone_alert,
            cyclone_dist_km=cyclone_dist_km,
            boundary_dist_km=nearest_boundary_dist_km,
            imd_warning_active=imd_warning_active,
        )

    # ------------------------------------------------------------ transport #

    async def _query_jev_api(self, state: dict[str, Any]) -> JevSafetyDecision | None:
        """One System-One call: five typed questions against one state."""
        key = self.settings.effective_jev_api_key
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/mithun50/ORCA"
            headers["X-Title"] = "ORCA Marine Intelligence"
        payload = {
            "model": self.model,
            "state": state,
            "questions": QUESTIONS,
        }

        try:
            async with httpx.AsyncClient(timeout=self.settings.jev_timeout_s) as client:
                resp = await client.post(self.endpoint, headers=headers, json=payload)
                if resp.status_code != 200:
                    # 429 and 529 are documented as retryable; the emulator covers
                    # this turn rather than making the user wait on a backoff.
                    log.warning(
                        "Jev %s (%s) returned %s: %s",
                        self.provider,
                        self.model,
                        resp.status_code,
                        resp.text[:200],
                    )
                    return None
                body = resp.json()
        except Exception as exc:  # noqa: BLE001 - never raise into the risk agent
            log.warning("Jev call failed (%s); using the emulator", exc)
            return None

        return self._parse_answers(body)

    def _parse_answers(self, body: Any) -> JevSafetyDecision | None:
        if not isinstance(body, dict):
            return None
        answers = body.get("answers")
        if not isinstance(answers, dict):
            log.warning("Jev response had no answers map; using the emulator")
            return None

        verdict_ans = answers.get("safety_verdict") or {}
        action_ans = answers.get("action") or {}
        verdict = str(verdict_ans.get("choice", "")).strip().upper()
        action = str(action_ans.get("choice", "")).strip().upper()
        if verdict not in VERDICTS or action not in ACTIONS:
            log.warning(
                "Jev returned an off-schema verdict/action (%r/%r); using the emulator",
                verdict,
                action,
            )
            return None

        # score is a probability-weighted float over the level indices, so
        # normalise by the number of levels minus one to reach 0-100.
        severity_ans = answers.get("severity") or {}
        span = max(1, len(SEVERITY_LEVELS) - 1)
        severity = _clamp(severity_ans.get("score"), 0.0, float(span), 2.0)
        risk_score = round(100.0 * severity / span, 1)

        # noul answers carry no confidence field, by design
        breach = _clamp(
            (answers.get("breach_probability") or {}).get("noul"), 0.0, 1.0, 0.1
        )
        capsize = _clamp(
            (answers.get("capsizing_probability") or {}).get("noul"), 0.0, 1.0, 0.1
        )

        probabilities = verdict_ans.get("probabilities")
        probs = (
            {str(k): _clamp(v, 0.0, 1.0, 0.0) for k, v in probabilities.items()}
            if isinstance(probabilities, dict)
            else {}
        )
        confidence = _clamp(verdict_ans.get("confidence"), 0.0, 1.0, 0.9)
        model = str(body.get("model") or self.model)
        usage = body.get("usage") or {}
        cost = usage.get("cost")

        return JevSafetyDecision(
            safety_verdict=verdict,
            action=action,
            risk_score=risk_score,
            breach_probability=breach,
            capsizing_probability=capsize,
            engine=f"jev-{self.provider}",
            confidence=confidence,
            rationale=(
                f"Jev System-One: {verdict} / {action}, severity "
                f"{severity:.2f}/{span} ({risk_score:.0f}/100), verdict confidence "
                f"{confidence:.2f}"
            ),
            model=model,
            verdict_probabilities=probs,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
            elapsed_ms=int(body.get("elapsedMs") or 0),
            raw_response=body,
        )

    # ------------------------------------------------------------- emulator #

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
        """Deterministic judgment replicating the Jev answer schema offline."""
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
            if swh_m >= self.settings.swh_danger_m:
                capsize_prob = max(capsize_prob, 0.85)
                score += 40.0
            elif swh_m >= self.settings.swh_caution_m:
                capsize_prob = max(capsize_prob, 0.40)
                score += 20.0

        if wind_kt is not None and wind_kt >= self.settings.wind_danger_kt:
            capsize_prob = max(capsize_prob, 0.80)
            score += 30.0
        elif wind_kt is not None and wind_kt >= self.settings.wind_caution_kt:
            score += 15.0

        if gust_kt is not None and gust_kt >= self.settings.gust_danger_kt:
            capsize_prob = max(capsize_prob, 0.75)
            score += 20.0

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
            rationale=(
                f"Offline emulator: verdict={verdict}, action={action}, "
                f"score={score:.1f}/100"
            ),
            model="rule-emulator",
        )


_jev_engine: JevDecisionEngine | None = None


def get_jev_engine() -> JevDecisionEngine:
    global _jev_engine
    if _jev_engine is None:
        _jev_engine = JevDecisionEngine()
    return _jev_engine
