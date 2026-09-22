"""Intent segregation and knowledge-source routing.

This is the fan-out point of the whole system, and the node the n8n workflow
switches on. Given a raw utterance it decides:

1. **what is being asked** (`Intent`), and
2. **which knowledge sources can answer it** (`KnowledgeDomain` set).

Classification is lexical-first and LLM-second on purpose. A weighted keyword
model is deterministic, explainable in the trace, costs nothing and cannot
hallucinate a route. The LLM is consulted only when the lexical model is not
confident, and its verdict is recorded separately in the trace so a reviewer can
see which one decided.

Each `KnowledgeDomain` maps 1:1 onto a `/internal/retrieve/{domain}` endpoint,
which is what lets the n8n Switch node drive retrieval branch by branch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from ..schemas import Intent


class KnowledgeDomain(str, Enum):
    """One retrievable knowledge source. Maps to /internal/retrieve/{domain}."""

    ADVISORY = "advisory"      # document RAG: PFZ advisories, IMD warnings, rules
    OCEAN = "ocean"            # MOSDAC OSF: SST, currents, salinity, MLD, waves
    WEATHER = "weather"        # wind, gust, rain, CAPE, tide
    GEOSPATIAL = "geospatial"  # EEZ, IMBL, MPAs, harbours, geofencing
    HAZARD = "hazard"          # cyclone tracks, active alerts
    CATALOG = "catalog"        # dataset discovery across MOSDAC/INCOIS/NASA


# --------------------------------------------------------------------------- #
# lexical model
# --------------------------------------------------------------------------- #

#: (regex, intent, weight). Weights are additive; highest total wins.
RULES: tuple[tuple[str, Intent, float], ...] = (
    # potential fishing zone
    (r"\bpfz\b", Intent.PFZ_LOCATE, 4.0),
    (r"potential fishing zone", Intent.PFZ_LOCATE, 4.0),
    (r"\b(where|which)\b.{0,40}\b(fish|fishing|catch)\b", Intent.PFZ_LOCATE, 2.5),
    (r"\bnearest\b.{0,30}\b(fishing|zone|pfz)\b", Intent.PFZ_LOCATE, 3.0),
    (r"\bgood (fishing|catch)\b", Intent.PFZ_LOCATE, 2.0),
    (r"\bwhere.{0,20}(should|can) i (go|fish)\b", Intent.PFZ_LOCATE, 2.5),
    # go / no-go safety
    (r"\bis it safe\b", Intent.SAFETY_GO_NOGO, 4.0),
    (r"\b(safe|unsafe|risky|dangerous)\b.{0,30}\b(sea|venture|go out|sail|fish)\b",
     Intent.SAFETY_GO_NOGO, 3.0),
    (r"\bventure\b", Intent.SAFETY_GO_NOGO, 2.5),
    (r"\bcan i go (out|to sea|fishing)\b", Intent.SAFETY_GO_NOGO, 3.0),
    (r"\bshould i (go|sail|venture)\b", Intent.SAFETY_GO_NOGO, 3.0),
    (r"\bsafe to (go|sail|venture|fish)\b", Intent.SAFETY_GO_NOGO, 3.5),
    # conditions summary
    (r"\b(tide|tides)\b", Intent.CONDITIONS_SUMMARY, 2.0),
    (r"\bsea conditions?\b", Intent.CONDITIONS_SUMMARY, 2.5),
    (r"\b(weather|wind|wave|swell|current)s?\b.{0,30}\b(near|at|off|around)\b",
     Intent.CONDITIONS_SUMMARY, 1.5),
    (r"\bwhat (is|are) the\b.{0,40}\b(condition|weather|wave|wind|tide)",
     Intent.CONDITIONS_SUMMARY, 2.5),
    (r"\bsea state\b", Intent.CONDITIONS_SUMMARY, 2.0),
    (r"\bforecast\b", Intent.CONDITIONS_SUMMARY, 1.0),
    # hazard alerts
    (r"\b(cyclone|storm|depression|hurricane|typhoon)\b", Intent.HAZARD_ALERTS, 3.0),
    (r"\blightning\b", Intent.HAZARD_ALERTS, 3.5),
    (r"\bthunderstorm\b", Intent.HAZARD_ALERTS, 3.0),
    (r"\b(alert|warning|advisory)s?\b", Intent.HAZARD_ALERTS, 2.0),
    (r"\bany.{0,20}(alert|warning)\b", Intent.HAZARD_ALERTS, 3.0),
    (r"\bsquall\b", Intent.HAZARD_ALERTS, 2.5),
    # productivity scan (chl + sst regions)
    (r"\bchlorophyll\b", Intent.PRODUCTIVITY_SCAN, 3.0),
    (r"\bocean colour|ocean color\b", Intent.PRODUCTIVITY_SCAN, 2.0),
    (r"\b(high|favourable|favorable)\b.{0,40}\b(chlorophyll|sst|temperature)\b",
     Intent.PRODUCTIVITY_SCAN, 3.0),
    (r"\bwhich regions?\b", Intent.PRODUCTIVITY_SCAN, 2.0),
    (r"\bupwelling\b", Intent.PRODUCTIVITY_SCAN, 2.0),
    (r"\bfront(s|al)?\b", Intent.PRODUCTIVITY_SCAN, 1.5),
    # route planning
    (r"\broute\b", Intent.ROUTE_PLANNING, 3.5),
    (r"\bsafest (route|path|way|passage)\b", Intent.ROUTE_PLANNING, 4.0),
    (r"\bnavigat(e|ion)\b", Intent.ROUTE_PLANNING, 2.0),
    (r"\bfrom\b.{1,30}\bto\b", Intent.ROUTE_PLANNING, 1.5),
    (r"\bpassage plan\b", Intent.ROUTE_PLANNING, 3.0),
    # productivity diagnosis (causal why)
    (r"\bwhy\b.{0,50}\b(declin|drop|fall|reduc|less|poor|low)\w*\b",
     Intent.PRODUCTIVITY_DIAGNOSIS, 4.0),
    (r"\b(declin|decreas|reduc)\w*\b.{0,30}\b(fish|catch|productivity|yield)\b",
     Intent.PRODUCTIVITY_DIAGNOSIS, 3.5),
    (r"\bno fish\b", Intent.PRODUCTIVITY_DIAGNOSIS, 2.5),
    (r"\bcatch (is )?(down|poor|low)\b", Intent.PRODUCTIVITY_DIAGNOSIS, 3.0),
    # geofence
    (r"\b(imbl|maritime boundary|international boundary)\b", Intent.GEOFENCE_CHECK, 4.0),
    (r"\bgeofenc\w*\b", Intent.GEOFENCE_CHECK, 4.0),
    (r"\b(restricted|prohibited|protected|sanctuary|marine park)\b",
     Intent.GEOFENCE_CHECK, 3.0),
    (r"\bavoid\b.{0,30}\b(zone|area|region|water)s?\b", Intent.GEOFENCE_CHECK, 3.0),
    (r"\bwhich (zones?|areas?)\b.{0,40}\bavoid\b", Intent.GEOFENCE_CHECK, 3.5),
    (r"\b(sri lanka|bangladesh|pakistan)\b", Intent.GEOFENCE_CHECK, 1.5),
    (r"\beez\b", Intent.GEOFENCE_CHECK, 2.5),
    # data discovery
    (r"\bwhat (data|datasets?|products?)\b", Intent.DATA_DISCOVERY, 3.5),
    (r"\bwhich (satellite|sensor|dataset)\b", Intent.DATA_DISCOVERY, 3.0),
    (r"\b(mosdac|incois|oceansat|eos-?06|insat|scatsat)\b", Intent.DATA_DISCOVERY, 1.5),
    (r"\bdata sources?\b", Intent.DATA_DISCOVERY, 2.5),
    # small talk
    (r"^\s*(hi|hello|hey|namaste|vanakkam|thanks|thank you)\b", Intent.SMALL_TALK, 5.0),
    (r"\bwho are you\b|\bwhat can you do\b", Intent.SMALL_TALK, 4.0),
)

#: Which knowledge sources each intent needs, in the order they get fanned out.
INTENT_DOMAINS: dict[Intent, tuple[KnowledgeDomain, ...]] = {
    Intent.PFZ_LOCATE: (
        KnowledgeDomain.OCEAN,
        KnowledgeDomain.ADVISORY,
        KnowledgeDomain.GEOSPATIAL,
    ),
    Intent.SAFETY_GO_NOGO: (
        KnowledgeDomain.WEATHER,
        KnowledgeDomain.OCEAN,
        KnowledgeDomain.HAZARD,
        KnowledgeDomain.ADVISORY,
    ),
    Intent.CONDITIONS_SUMMARY: (
        KnowledgeDomain.OCEAN,
        KnowledgeDomain.WEATHER,
        KnowledgeDomain.ADVISORY,
    ),
    Intent.HAZARD_ALERTS: (
        KnowledgeDomain.HAZARD,
        KnowledgeDomain.WEATHER,
        KnowledgeDomain.ADVISORY,
    ),
    Intent.PRODUCTIVITY_SCAN: (
        KnowledgeDomain.OCEAN,
        KnowledgeDomain.GEOSPATIAL,
        KnowledgeDomain.ADVISORY,
    ),
    Intent.ROUTE_PLANNING: (
        KnowledgeDomain.WEATHER,
        KnowledgeDomain.OCEAN,
        KnowledgeDomain.GEOSPATIAL,
        KnowledgeDomain.HAZARD,
    ),
    Intent.PRODUCTIVITY_DIAGNOSIS: (
        KnowledgeDomain.OCEAN,
        KnowledgeDomain.ADVISORY,
        KnowledgeDomain.CATALOG,
    ),
    Intent.GEOFENCE_CHECK: (
        KnowledgeDomain.GEOSPATIAL,
        KnowledgeDomain.HAZARD,
        KnowledgeDomain.OCEAN,
    ),
    Intent.DATA_DISCOVERY: (KnowledgeDomain.CATALOG, KnowledgeDomain.ADVISORY),
    Intent.SMALL_TALK: (),
    Intent.UNKNOWN: (
        KnowledgeDomain.WEATHER,
        KnowledgeDomain.OCEAN,
        KnowledgeDomain.ADVISORY,
    ),
}

#: Which agent owns each domain, used to build the task list.
DOMAIN_AGENT: dict[KnowledgeDomain, str] = {
    KnowledgeDomain.ADVISORY: "advisory",
    KnowledgeDomain.OCEAN: "ocean-analytics",
    KnowledgeDomain.WEATHER: "weather-intelligence",
    KnowledgeDomain.GEOSPATIAL: "geospatial-reasoning",
    KnowledgeDomain.HAZARD: "risk-assessment",
    KnowledgeDomain.CATALOG: "data-discovery",
}

TIME_HINTS: tuple[tuple[str, str, int, int], ...] = (
    # pattern, label, start offset hours, duration hours
    (r"\bday after tomorrow\b", "day after tomorrow", 48, 24),
    (r"\btomorrow morning\b", "tomorrow morning", 24, 8),
    (r"\btomorrow evening\b|\btomorrow night\b", "tomorrow evening", 36, 8),
    (r"\btomorrow\b", "tomorrow", 24, 24),
    (r"\btonight\b", "tonight", 6, 12),
    (r"\bthis evening\b", "this evening", 3, 6),
    (r"\bthis (morning|afternoon)\b", "today", 0, 12),
    (r"\bnext (\d+) days?\b", "next few days", 0, 72),
    (r"\bthis week\b|\bcoming week\b", "this week", 0, 120),
    (r"\bnow\b|\bright now\b|\bcurrently\b", "now", 0, 6),
    (r"\btoday\b", "today", 0, 24),
    # Native-script time words. Without these a question asked in Kannada or
    # Tamil silently answers for "now" instead of tomorrow morning, which is a
    # wrong answer rather than a missing one.
    (r"ನಾಳೆ\s*ಬೆಳಿಗ್ಗೆ|நாளை\s*காலை|రేపు\s*ఉదయం|നാളെ\s*രാവിലെ|कल\s*सुबह",
     "tomorrow morning", 24, 8),
    (r"ನಾಳೆ\s*ಸಂಜೆ|நாளை\s*மாலை|రేపు\s*సాయంత్రం|നാളെ\s*വൈകുന്നേരം|कल\s*शाम",
     "tomorrow evening", 36, 8),
    (r"ನಾಳೆ|நாளை|రేపు|നാളെ|कल|আগামীকাল", "tomorrow", 24, 24),
    (r"ಇಂದು\s*ರಾತ್ರಿ|இன்று\s*இரவு|ఈ\s*రాత్రి|ഇന്ന്\s*രാത്രി|आज\s*रात", "tonight", 6, 12),
    (r"ಇಂದು|இன்று|ఈరోజు|ഇന്ന്|आज|আজ", "today", 0, 24),
    (r"ಈಗ|இப்போது|ఇప్పుడు|ഇപ്പോൾ|अभी", "now", 0, 6),
    (r"ಈ\s*ವಾರ|இந்த\s*வாரம்|ఈ\s*వారం|ഈ\s*ആഴ്ച|इस\s*हफ़्ते", "this week", 0, 120),
)

LANG_HINTS = {
    "ta": ("vanakkam", "meen", "kadal"),
    "hi": ("namaste", "samudra", "machhli"),
    "ml": ("kadal", "meen"),
}


@dataclass
class Classification:
    intent: Intent
    confidence: float
    domains: tuple[KnowledgeDomain, ...]
    matched_rules: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    time_label: str = "now"
    time_offset_h: int = 0
    time_duration_h: int = 24
    decided_by: str = "lexical"
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 3),
            "domains": [d.value for d in self.domains],
            "matched_rules": self.matched_rules,
            "scores": {k: round(v, 2) for k, v in self.scores.items()},
            "time_label": self.time_label,
            "decided_by": self.decided_by,
            "notes": self.notes,
        }


def classify(message: str) -> Classification:
    """Deterministic lexical intent segregation."""
    text = message.lower().strip()
    scores: dict[Intent, float] = {}
    matched: list[str] = []
    for pattern, intent, weight in RULES:
        if re.search(pattern, text):
            scores[intent] = scores.get(intent, 0.0) + weight
            matched.append(f"{intent.value}<-/{pattern}/ (+{weight})")

    if not scores:
        return Classification(
            intent=Intent.UNKNOWN,
            confidence=0.0,
            domains=INTENT_DOMAINS[Intent.UNKNOWN],
            matched_rules=[],
            notes="no lexical rule fired; escalating to the LLM classifier",
            **_time_hint(text),
        )

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_intent, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    # confidence: how dominant the winner is, saturating around a score of 6
    margin = (best_score - runner_up) / max(best_score, 1.0)
    confidence = min(0.98, 0.45 + 0.35 * margin + 0.12 * min(best_score / 6.0, 1.0))

    return Classification(
        intent=best_intent,
        confidence=confidence,
        domains=INTENT_DOMAINS.get(best_intent, INTENT_DOMAINS[Intent.UNKNOWN]),
        matched_rules=matched,
        scores={k.value: v for k, v in scores.items()},
        **_time_hint(text),
    )


def _time_hint(text: str) -> dict[str, object]:
    for pattern, label, offset, duration in TIME_HINTS:
        if re.search(pattern, text):
            return {
                "time_label": label,
                "time_offset_h": offset,
                "time_duration_h": duration,
            }
    return {"time_label": "now", "time_offset_h": 0, "time_duration_h": 24}


def domains_for(intent: Intent) -> tuple[KnowledgeDomain, ...]:
    return INTENT_DOMAINS.get(intent, INTENT_DOMAINS[Intent.UNKNOWN])


def agents_for(domains: Iterable[KnowledgeDomain]) -> list[str]:
    seen: list[str] = []
    for domain in domains:
        agent = DOMAIN_AGENT.get(domain)
        if agent and agent not in seen:
            seen.append(agent)
    return seen


def detect_language_hint(message: str) -> str:
    """Crude script/keyword hint. English-only responses in this prototype."""
    for char in message:
        code = ord(char)
        if 0x0900 <= code <= 0x097F:
            return "hi"
        if 0x0B80 <= code <= 0x0BFF:
            return "ta"
        if 0x0D00 <= code <= 0x0D7F:
            return "ml"
        if 0x0C00 <= code <= 0x0C7F:
            return "te"
        if 0x0A80 <= code <= 0x0AFF:
            return "gu"
        if 0x0980 <= code <= 0x09FF:
            return "bn"
        if 0x0C80 <= code <= 0x0CFF:
            return "kn"
    lowered = message.lower()
    for lang, words in LANG_HINTS.items():
        if any(word in lowered for word in words):
            return lang
    return "en"
