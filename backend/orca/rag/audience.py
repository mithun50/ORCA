"""Audience detection: who is asking, and how should the answer read?

The same retrieved facts serve very different readers. A fisherman at 4am wants
"waves about 1 metre, wind picking up by noon, go early". A researcher wants the
dataset name, the grid, the units and the caveat. An official wants the operative
advisory wording and the jurisdiction.

Getting this wrong is not cosmetic. Handing a fisherman "significant wave height
1.14 m, mixed layer depth 42 cm, tier4-fallback" is an answer they cannot use,
which in a safety tool is a failure.

Detection is lexical and deterministic for the same reasons the intent router is:
it is explainable in the trace, it costs nothing, and it cannot hallucinate a
persona. Signals are additive; the strongest total wins, and a query with no
signal at all defaults to the fisherman register, because that is the audience
the problem statement is written for and the one for whom a misjudged answer is
most costly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Audience(str, Enum):
    FISHERMAN = "fisherman"
    RESEARCHER = "researcher"
    OFFICIAL = "official"
    GENERAL = "general"


#: (regex, audience, weight)
AUDIENCE_RULES: tuple[tuple[str, Audience, float], ...] = (
    # --- fisherman: boat-owner vocabulary, first person, immediate action ---
    (r"\b(my|our) (boat|catamaran|vallam|fibre boat|trawler|craft|net|nets)\b",
     Audience.FISHERMAN, 4.0),
    (r"\b(can|should) i (go|sail|venture|fish|put out)\b", Audience.FISHERMAN, 3.0),
    (r"\b(is it safe|safe to (go|venture|sail|fish))\b", Audience.FISHERMAN, 2.0),
    (r"\b(catch|fishing|fish)\b", Audience.FISHERMAN, 1.5),
    (r"\b(today|tomorrow|tonight|morning|now)\b", Audience.FISHERMAN, 0.8),
    (r"\b(harbour|harbor|jetty|landing centre|landing center|shore)\b",
     Audience.FISHERMAN, 1.2),
    (r"\b(diesel|fuel|ice box|crew)\b", Audience.FISHERMAN, 2.0),
    (r"\b(nethu|naalai|meen|kadal|machhli|samudra|vanakkam|namaste)\b",
     Audience.FISHERMAN, 2.0),

    # --- researcher: dataset, method and provenance vocabulary ---
    (r"\b(dataset|datasets|data set|granule|netcdf|opendap|thredds|erddap|griddap)\b",
     Audience.RESEARCHER, 4.0),
    (r"\b(anomaly|anomalies|climatology|time series|timeseries|correlation|"
     r"regression|gradient|variance|standard deviation)\b", Audience.RESEARCHER, 3.5),
    (r"\b(chlorophyll|chl-a|sst|salinity|mixed layer|mld|upwelling|eddy|"
     r"thermocline|bathymetry)\b", Audience.RESEARCHER, 2.0),
    (r"\b(resolution|grid|spatial extent|temporal coverage|units|epsg|projection)\b",
     Audience.RESEARCHER, 3.0),
    (r"\b(oceansat|scatsat|insat|eos-?06|modis|sentinel|argo|avhrr)\b",
     Audience.RESEARCHER, 2.5),
    (r"\b(methodology|validate|validation|cite|citation|reference|doi|paper|study)\b",
     Audience.RESEARCHER, 3.0),
    (r"\b(why (has|did|is)|what (causes|caused|drives|explains))\b",
     Audience.RESEARCHER, 1.5),

    # --- official: enforcement, jurisdiction, coordination ---
    (r"\b(advisory|bulletin|circular|notification|gazette)\b", Audience.OFFICIAL, 2.5),
    (r"\b(coast guard|coastguard|fisheries department|district|state government|"
     r"authority|authorities|administration)\b", Audience.OFFICIAL, 3.5),
    (r"\b(enforce|enforcement|violation|prosecute|penalty|compliance|mandate)\b",
     Audience.OFFICIAL, 3.5),
    (r"\b(imbl|eez|jurisdiction|territorial waters|maritime boundary)\b",
     Audience.OFFICIAL, 2.0),
    (r"\b(evacuat\w+|disaster|relief|rescue|sop|protocol)\b", Audience.OFFICIAL, 3.0),
    (r"\b(issue|issued|should we (warn|advise|alert))\b", Audience.OFFICIAL, 2.0),
    (r"\b(all (boats|vessels|fishermen)|fleet|fishing community)\b",
     Audience.OFFICIAL, 2.5),
)

#: How the answer should read for each audience. Fed to the template layer and
#: to the model prompt, so both paths agree on register.
AUDIENCE_STYLE: dict[Audience, dict[str, object]] = {
    Audience.FISHERMAN: {
        "label": "fisherman or boat owner",
        "reading_level": "plain spoken, no jargon",
        "max_words": 130,
        "show_dataset_names": False,
        "show_units_in_full": False,
        "guidance": (
            "Talk like an experienced skipper talking to another. Short sentences. "
            "Lead with what to do, then why in everyday words. Say 'waves about a "
            "metre', not 'significant wave height 1.14 m'. Say 'wind picking up by "
            "noon', not 'wind peak 24 kt at 12:00Z'. Never use the words "
            "'significant wave height', 'CAPE', 'mixed layer depth', 'tier', "
            "'anomaly' or 'model run'. Use metres, knots and hours because those "
            "are the units a boat owner already thinks in. Mention the agency by "
            "its common name (ISRO, INCOIS, IMD) once, not per number."
        ),
    },
    Audience.RESEARCHER: {
        "label": "marine researcher or analyst",
        "reading_level": "technical, precise",
        "max_words": 260,
        "show_dataset_names": True,
        "show_units_in_full": True,
        "guidance": (
            "Be precise and quantitative. Keep the full variable names, units, "
            "dataset identifiers, grid and access method. State the validity window "
            "and every caveat, including staleness and whether a value is from an "
            "official product or a fallback model. Do not round away precision. "
            "Distinguish observed from derived quantities explicitly."
        ),
    },
    Audience.OFFICIAL: {
        "label": "coastal authority or fisheries official",
        "reading_level": "operational, decision oriented",
        "max_words": 200,
        "show_dataset_names": True,
        "show_units_in_full": True,
        "guidance": (
            "Lead with the operative decision and who it affects. Quote official "
            "advisory wording verbatim where it exists and name the issuing "
            "authority, because that is what carries legal weight. Be explicit "
            "about jurisdiction and about the limits of the underlying data. Keep "
            "derived indicators clearly separated from official advisories."
        ),
    },
    Audience.GENERAL: {
        "label": "general reader",
        "reading_level": "plain, lightly explanatory",
        "max_words": 170,
        "show_dataset_names": False,
        "show_units_in_full": True,
        "guidance": (
            "Plain language with a short gloss for any technical term you cannot "
            "avoid. Explain what a number means in practice rather than leaving it "
            "bare. Name the source agency so the reader knows it is official."
        ),
    },
}


@dataclass
class AudienceCall:
    audience: Audience
    confidence: float
    matched_rules: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    decided_by: str = "lexical"

    @property
    def style(self) -> dict[str, object]:
        return AUDIENCE_STYLE[self.audience]

    def as_dict(self) -> dict[str, object]:
        return {
            "audience": self.audience.value,
            "confidence": round(self.confidence, 3),
            "matched_rules": self.matched_rules,
            "scores": {k: round(v, 2) for k, v in self.scores.items()},
            "decided_by": self.decided_by,
        }


def detect_audience(message: str, *, depth_hint: int = 0) -> AudienceCall:
    """Decide who is asking.

    `depth_hint` is the number of turns already in this session. A long,
    elaborated query from someone several turns in reads more like an analyst
    than a skipper on deck, so it nudges away from the default register.
    """
    text = message.lower().strip()
    scores: dict[Audience, float] = {}
    matched: list[str] = []

    for pattern, audience, weight in AUDIENCE_RULES:
        if re.search(pattern, text):
            scores[audience] = scores.get(audience, 0.0) + weight
            matched.append(f"{audience.value}<-/{pattern[:38]}/ (+{weight})")

    # Query shape is a signal in itself. A long, clause-heavy question is not
    # how someone asks from a moving boat.
    words = len(text.split())
    if words >= 22:
        scores[Audience.RESEARCHER] = scores.get(Audience.RESEARCHER, 0.0) + 1.5
        matched.append(f"researcher<-long query ({words} words) (+1.5)")
    elif words <= 8:
        scores[Audience.FISHERMAN] = scores.get(Audience.FISHERMAN, 0.0) + 1.0
        matched.append(f"fisherman<-short query ({words} words) (+1.0)")
    if depth_hint >= 6:
        scores[Audience.RESEARCHER] = scores.get(Audience.RESEARCHER, 0.0) + 0.5
        matched.append(f"researcher<-deep session ({depth_hint} turns) (+0.5)")

    if not scores:
        return AudienceCall(
            audience=Audience.FISHERMAN,
            confidence=0.35,
            matched_rules=[],
            scores={},
            decided_by="default",
        )

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = (best_score - runner_up) / max(best_score, 1.0)
    confidence = min(0.97, 0.45 + 0.35 * margin + 0.12 * min(best_score / 6.0, 1.0))

    return AudienceCall(
        audience=best,
        confidence=confidence,
        matched_rules=matched,
        scores={k.value: v for k, v in scores.items()},
    )
