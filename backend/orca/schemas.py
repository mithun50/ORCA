"""Wire and internal schemas.

The contract that matters for the problem statement is `ChatResponse`: every
answer ships with the evidence it rests on, the reasoning steps that produced
it, and the map/chart payloads that visualise it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# provenance and evidence
# --------------------------------------------------------------------------- #

class Tier(str, Enum):
    """Where a number came from, in order of authority for Indian waters."""

    ISRO = "tier1-isro"
    INCOIS = "tier2-incois"
    IMD = "tier3-imd"
    FALLBACK = "tier4-fallback"
    DERIVED = "derived"
    SEED = "seed"


class Provenance(BaseModel):
    agency: str
    dataset: str
    tier: Tier
    url: str = ""
    access_method: str = ""  # ncss-point, opendap-ascii, wms, erddap, http-json, html
    retrieved_at: datetime = Field(default_factory=utcnow)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    is_stale: bool = False
    staleness_note: str = ""
    official: bool = True
    caveat: str = ""


class Evidence(BaseModel):
    """One retrievable fact, always attached to its provenance."""

    id: str
    label: str
    value: Any = None
    unit: str = ""
    at_lat: float | None = None
    at_lon: float | None = None
    at_time: datetime | None = None
    series: list[dict[str, Any]] = Field(default_factory=list)
    text: str = ""
    provenance: Provenance

    def one_line(self) -> str:
        if self.value is None:
            return f"{self.label}: {self.text[:160]}" if self.text else self.label
        if isinstance(self.value, float):
            return f"{self.label} = {self.value:.2f} {self.unit}".strip()
        return f"{self.label} = {self.value} {self.unit}".strip()


class Citation(BaseModel):
    """One numbered marker in the answer text, bound to the evidence behind it.

    The answer carries `[1]`, `[2]` markers inline. This is what each one points
    at, so a reader can check any claim without hunting through the evidence
    panel, and so the UI can make the marker clickable.
    """

    marker: int
    evidence_id: str
    label: str
    value: str = ""
    agency: str = ""
    dataset: str = ""
    tier: Tier = Tier.SEED
    url: str = ""
    official: bool = True
    is_stale: bool = False


# --------------------------------------------------------------------------- #
# reasoning trace
# --------------------------------------------------------------------------- #

class TraceStep(BaseModel):
    seq: int
    agent: str
    action: str
    rationale: str = ""
    tool: str = ""
    tool_args: dict[str, Any] = Field(default_factory=dict)
    outcome: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    duration_ms: int = 0
    status: Literal["ok", "degraded", "failed", "skipped"] = "ok"


# --------------------------------------------------------------------------- #
# intent / plan
# --------------------------------------------------------------------------- #

class Intent(str, Enum):
    PFZ_LOCATE = "pfz_locate"
    SAFETY_GO_NOGO = "safety_go_nogo"
    CONDITIONS_SUMMARY = "conditions_summary"
    HAZARD_ALERTS = "hazard_alerts"
    PRODUCTIVITY_SCAN = "productivity_scan"
    ROUTE_PLANNING = "route_planning"
    PRODUCTIVITY_DIAGNOSIS = "productivity_diagnosis"
    GEOFENCE_CHECK = "geofence_check"
    DATA_DISCOVERY = "data_discovery"
    SMALL_TALK = "small_talk"
    UNKNOWN = "unknown"


class Location(BaseModel):
    name: str = ""
    lat: float
    lon: float
    source: str = "gazetteer"  # gazetteer | explicit | session | default
    district: str = ""
    state: str = ""


class TimeWindow(BaseModel):
    label: str = "now"
    start: datetime
    end: datetime


class Task(BaseModel):
    agent: str
    goal: str
    why: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    intent: Intent
    confidence: float = 0.0
    location: Location | None = None
    destination: Location | None = None
    window: TimeWindow
    tasks: list[Task] = Field(default_factory=list)
    planner_notes: str = ""
    clarification_needed: str = ""


# --------------------------------------------------------------------------- #
# presentation
# --------------------------------------------------------------------------- #

class MapMarker(BaseModel):
    lat: float
    lon: float
    label: str
    kind: str = "info"  # info | good | caution | danger | vessel | pfz | harbour
    detail: str = ""


class MapLayer(BaseModel):
    id: str
    title: str
    kind: Literal["wms", "geojson", "markers", "line", "heat"]
    url: str = ""
    wms_layer: str = ""
    wms_time: str = ""
    wms_style: str = ""
    geojson: dict[str, Any] | None = None
    markers: list[MapMarker] = Field(default_factory=list)
    attribution: str = ""
    opacity: float = 0.75
    visible_by_default: bool = True
    legend: str = ""


class ChartSeries(BaseModel):
    id: str
    title: str
    unit: str = ""
    x: list[str] = Field(default_factory=list)
    y: list[float | None] = Field(default_factory=list)
    threshold: float | None = None
    threshold_label: str = ""
    source: str = ""


class RiskBand(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


class RiskFinding(BaseModel):
    rule: str
    band: RiskBand
    detail: str
    evidence_ids: list[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    band: RiskBand = RiskBand.UNKNOWN
    score: float = 0.0  # 0 calm .. 100 severe
    headline: str = ""
    findings: list[RiskFinding] = Field(default_factory=list)
    window_advice: str = ""
    jev_decision: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    lat: float | None = None
    lon: float | None = None
    place: str | None = None
    destination: str | None = None
    language: str = "en"
    audio_base64: str | None = None


class BriefingItem(BaseModel):
    """One number, as short as it can be said, with its risk flag and source."""

    label: str            # "waves", "wind", "SST"
    value: str            # already formatted, e.g. "1.1"
    unit: str = ""        # "m", "kt", "degC"
    note: str = ""        # very short qualifier, e.g. "building to 1.5"
    #: "" | "watch" | "danger" - why this number matters, not just what it is
    flag: str = ""
    marker: int | None = None   # citation marker in the answer text
    tier: Tier | None = None    # which tier the number came from


class Briefing(BaseModel):
    """The whole answer as numbers, grouped by where they come from.

    The prose answer reads well but buries the figures a skipper actually acts
    on. This is the same retrieved data with the sentences removed: four groups,
    every entry a number, and the risk stated first rather than last.
    """

    risk_band: str = "unknown"
    risk_score: float = 0.0
    headline: str = ""
    #: the rules that actually drove the band, one terse line each
    drivers: list[str] = Field(default_factory=list)
    #: sea state and water column
    ocean: list[BriefingItem] = Field(default_factory=list)
    #: atmosphere
    weather: list[BriefingItem] = Field(default_factory=list)
    #: position, boundaries, zones, shelter
    gis: list[BriefingItem] = Field(default_factory=list)
    #: which satellite and agency products the figures came from
    satellite: list[BriefingItem] = Field(default_factory=list)
    #: what could not be measured, so a gap is never mistaken for a safe value
    missing: list[str] = Field(default_factory=list)


class ModelRole(BaseModel):
    """Which model did which job on this request, and whether it actually ran."""

    role: str           # intent | audience | verdict | judgment | wording | voice
    job: str            # human readable description of the work
    provider: str = ""  # openrouter | gemini | typesafe | sarvam | rules | none
    model: str = ""
    used: bool = False
    detail: str = ""
    #: usage, when the provider reports it. `cost_usd` stays None rather than
    #: being estimated, so the UI can honestly say "not reported".
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    elapsed_ms: int = 0
    #: what drives the bill for this provider: tokens, characters, or nothing
    billing_unit: str = "none"


class CostSummary(BaseModel):
    """What this one prompt cost, as far as the providers actually reported."""

    total_usd: float = 0.0
    #: roles that ran but whose provider does not report a price
    unpriced_roles: list[str] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    tts_characters: int = 0
    #: True when every paid role reported a figure, so total_usd is complete
    complete: bool = True


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent: Intent
    confidence: float
    location: Location | None = None
    window: TimeWindow | None = None
    risk: RiskAssessment | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    #: inline [n] markers in `answer`, resolved to the evidence behind each one
    citations: list[Citation] = Field(default_factory=list)
    #: the same answer as numbers only, grouped by source, risk first
    briefing: Briefing = Field(default_factory=Briefing)
    trace: list[TraceStep] = Field(default_factory=list)
    layers: list[MapLayer] = Field(default_factory=list)
    charts: list[ChartSeries] = Field(default_factory=list)
    followups: list[str] = Field(default_factory=list)
    degraded_sources: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utcnow)
    llm_used: bool = False
    via_n8n: bool = False
    language: str = "en"
    #: What a voice turn was heard as. The user cannot check an answer they
    #: cannot see the question for, and a misheard word is the likeliest failure
    #: in a noisy wheelhouse, so this is always shown back to them.
    transcript: str = ""
    transcript_confidence: float = 0.0
    #: who the answer was written for, and how confident that call was
    audience: str = "fisherman"
    audience_confidence: float = 0.0
    #: full attribution: every model that could have run, and whether it did
    model_roles: list[ModelRole] = Field(default_factory=list)
    #: what this prompt cost, per the providers' own reporting
    cost: CostSummary = Field(default_factory=CostSummary)
    audio_base64: str | None = None


class AlertSubscription(BaseModel):
    subscriber_id: str
    lat: float
    lon: float
    place: str = ""
    channels: list[str] = Field(default_factory=lambda: ["webhook"])


class ProactiveAlert(BaseModel):
    subscriber_id: str
    severity: RiskBand
    title: str
    body: str
    location: Location
    evidence: list[Evidence] = Field(default_factory=list)
    issued_at: datetime = Field(default_factory=utcnow)
