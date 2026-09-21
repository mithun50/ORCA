"""Planner agent.

Turns an utterance into a plan: intent, location, time window and an ordered
task list of which specialised agents to run. It is the only agent allowed to
decide *what* happens; the rest decide *how*.

Autonomy with a brake on it: the lexical router decides when it is confident,
the LLM arbitrates when it is not, and if neither can pin down a location for a
question that needs one, the planner emits a clarification request instead of
answering about a made-up place.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..geo import gazetteer
from ..rag import router
from ..rag.router import KnowledgeDomain
from ..schemas import Intent, Location, Plan, Task, TimeWindow
from ..services import Services
from .base import Trace

#: intents that are meaningless without a position
NEEDS_LOCATION = {
    Intent.PFZ_LOCATE,
    Intent.SAFETY_GO_NOGO,
    Intent.CONDITIONS_SUMMARY,
    Intent.HAZARD_ALERTS,
    Intent.ROUTE_PLANNING,
    Intent.GEOFENCE_CHECK,
    Intent.PRODUCTIVITY_DIAGNOSIS,
}

GOALS: dict[KnowledgeDomain, str] = {
    KnowledgeDomain.OCEAN: "retrieve SST, currents, mixed layer and wave state",
    KnowledgeDomain.WEATHER: "retrieve wind, gusts, rain, convective and tide state",
    KnowledgeDomain.GEOSPATIAL: "check boundaries, protected zones and nearest harbours",
    KnowledgeDomain.HAZARD: "check cyclone tracks and active official warnings",
    KnowledgeDomain.ADVISORY: "retrieve matching advisories, rules and SOP text",
    KnowledgeDomain.CATALOG: "identify which datasets can answer this",
}

LLM_INTENT_PROMPT = """Classify this marine question into exactly one intent.

Intents:
- pfz_locate: where to fish, potential fishing zones
- safety_go_nogo: is it safe to go to sea
- conditions_summary: current or forecast tide, weather, wave, sea conditions
- hazard_alerts: cyclone, lightning, storm, official warnings
- productivity_scan: which regions have high chlorophyll or favourable SST
- route_planning: safest route or passage between two places
- productivity_diagnosis: why has fish catch or productivity declined
- geofence_check: maritime boundaries, restricted or protected zones to avoid
- data_discovery: which datasets or satellites are available
- small_talk: greeting or a question about the assistant itself

Question: {message}

Reply as JSON: {{"intent": "<one of the ids above>", "confidence": 0.0-1.0, "why": "<12 words>"}}
"""


class PlannerAgent:
    name = "planner"
    tools = ("lexical-router", "gazetteer", "llm-classifier")

    def __init__(self, services: Services) -> None:
        self.services = services

    async def plan(
        self,
        message: str,
        *,
        session_id: str,
        trace: Trace,
        lat: float | None = None,
        lon: float | None = None,
        place: str | None = None,
        destination: str | None = None,
    ) -> Plan:
        # ---- 1. segregate the input ------------------------------------- #
        with trace.timed(
            self.name,
            "segregate intent",
            rationale=(
                "weighted keyword rules first: deterministic, explainable and "
                "cannot hallucinate a route"
            ),
            tool="lexical-router",
            tool_args={"message": message[:160]},
        ) as step:
            classification = router.classify(message)
            step.outcome = (
                f"intent={classification.intent.value} "
                f"confidence={classification.confidence:.2f} "
                f"domains={[d.value for d in classification.domains]}"
            )
            if classification.matched_rules:
                step.outcome += f" rules={len(classification.matched_rules)}"

        if classification.confidence < 0.55:
            classification = await self._llm_arbitrate(message, classification, trace)

        # ---- 2. resolve where ------------------------------------------- #
        location = self._resolve_location(
            message, session_id=session_id, lat=lat, lon=lon, place=place, trace=trace
        )
        dest = self._resolve_destination(message, destination, location, trace)
        if dest and classification.intent in (
            Intent.CONDITIONS_SUMMARY,
            Intent.UNKNOWN,
        ):
            # "from X to Y" with no other signal is a routing question
            classification.intent = Intent.ROUTE_PLANNING
            classification.domains = router.domains_for(Intent.ROUTE_PLANNING)
            classification.notes += " reclassified as routing: two places named."

        # ---- 3. resolve when -------------------------------------------- #
        window = self._resolve_window(classification)

        # ---- 4. decompose ----------------------------------------------- #
        tasks = self._build_tasks(classification, has_destination=dest is not None)

        clarification = ""
        if classification.intent in NEEDS_LOCATION and (
            location is None or location.source == "default"
        ):
            clarification = (
                "I need a location for this. Tell me the harbour or landing "
                "centre you are sailing from, or send your latitude and longitude."
            )

        plan = Plan(
            intent=classification.intent,
            confidence=classification.confidence,
            location=location,
            destination=dest,
            window=window,
            tasks=tasks,
            planner_notes=(
                f"decided_by={classification.decided_by}; "
                f"time={classification.time_label}; "
                f"{classification.notes}".strip()
            ),
            clarification_needed=clarification,
        )

        with trace.timed(
            self.name,
            "decompose into agent tasks",
            rationale=(
                "each knowledge domain the intent needs becomes one specialised "
                "agent task; independent tasks run concurrently"
            ),
        ) as step:
            step.outcome = " -> ".join(t.agent for t in tasks) or "no tasks"
        return plan

    # ------------------------------------------------------------------ llm #

    async def _llm_arbitrate(
        self, message: str, current: router.Classification, trace: Trace
    ) -> router.Classification:
        with trace.timed(
            self.name,
            "escalate ambiguous intent to the LLM",
            rationale=(
                f"lexical confidence {current.confidence:.2f} is below the 0.55 "
                "threshold, so a second opinion is worth the latency"
            ),
            tool="llm-classifier",
        ) as step:
            payload = await self.services.llm.complete_json(
                LLM_INTENT_PROMPT.format(message=message)
            )
            if not payload or "intent" not in payload:
                step.status = "degraded"
                step.outcome = (
                    "no LLM available or unparseable reply; keeping the lexical "
                    f"verdict ({current.intent.value})"
                )
                return current
            try:
                intent = Intent(str(payload["intent"]).strip())
            except ValueError:
                step.status = "degraded"
                step.outcome = f"LLM returned an unknown intent: {payload['intent']!r}"
                return current
            confidence = float(payload.get("confidence", 0.6) or 0.6)
            step.outcome = (
                f"LLM says {intent.value} ({confidence:.2f}): "
                f"{payload.get('why', '')}"
            )
            current.intent = intent
            current.confidence = max(current.confidence, min(confidence, 0.9))
            current.domains = router.domains_for(intent)
            current.decided_by = "llm"
            return current

    # ------------------------------------------------------------- location #

    def _resolve_location(
        self,
        message: str,
        *,
        session_id: str,
        lat: float | None,
        lon: float | None,
        place: str | None,
        trace: Trace,
    ) -> Location | None:
        with trace.timed(
            self.name,
            "resolve location",
            rationale=(
                "precedence: GPS from the client, then coordinates typed in the "
                "message, then a gazetteer match, then the location from earlier "
                "in this conversation"
            ),
            tool="gazetteer",
        ) as step:
            if lat is not None and lon is not None:
                step.outcome = f"client position {lat:.3f}N {lon:.3f}E"
                return Location(name=place or "your position", lat=lat, lon=lon,
                                source="explicit")

            typed = gazetteer.parse_coords(message)
            if typed:
                step.outcome = f"coordinates parsed from the message {typed[0]:.3f}N {typed[1]:.3f}E"
                return Location(
                    name="the position you gave", lat=typed[0], lon=typed[1],
                    source="explicit",
                )

            if place:
                named = gazetteer.find_place(place)
                if named:
                    step.outcome = (
                        f"gazetteer matched {named.name} from the place field"
                    )
                    return Location(
                        name=named.name, lat=named.lat, lon=named.lon,
                        source="gazetteer", district=named.district,
                        state=named.state,
                    )

            # first place *mentioned* wins, so "from Chennai to Kakinada" keeps
            # Chennai as the origin rather than whichever name is longer
            mentioned = gazetteer.find_all_places(message)
            hit = mentioned[0] if mentioned else None
            if hit:
                step.outcome = (
                    f"gazetteer matched {hit.name} ({hit.district}, {hit.state})"
                    + (f", first of {len(mentioned)} places named" if len(mentioned) > 1 else "")
                )
                return Location(
                    name=hit.name, lat=hit.lat, lon=hit.lon, source="gazetteer",
                    district=hit.district, state=hit.state,
                )

            remembered = self.services.last_location(session_id)
            if remembered:
                step.outcome = (
                    f"reusing {remembered[2] or 'the location'} from earlier in "
                    "this conversation"
                )
                return Location(
                    name=remembered[2] or "your earlier location",
                    lat=remembered[0],
                    lon=remembered[1],
                    source="session",
                )

            step.status = "degraded"
            step.outcome = "no location found; will ask the user"
            fallback = gazetteer.DEFAULT_PLACE
            return Location(
                name=fallback.name,
                lat=fallback.lat,
                lon=fallback.lon,
                source="default",
                district=fallback.district,
                state=fallback.state,
            )

    def _resolve_destination(
        self,
        message: str,
        destination: str | None,
        origin: Location | None,
        trace: Trace,
    ) -> Location | None:
        if destination:
            hit = gazetteer.find_place(destination)
            if hit:
                return Location(
                    name=hit.name, lat=hit.lat, lon=hit.lon, source="explicit",
                    district=hit.district, state=hit.state,
                )
        hits = gazetteer.find_all_places(message)
        if len(hits) < 2:
            return None
        second = next(
            (h for h in hits if not origin or h.name != origin.name), None
        )
        if second is None:
            return None
        trace.add(
            self.name,
            "resolve destination",
            rationale="two gazetteer places in one sentence implies a passage",
            outcome=f"destination {second.name}",
        )
        return Location(
            name=second.name, lat=second.lat, lon=second.lon, source="gazetteer",
            district=second.district, state=second.state,
        )

    # ----------------------------------------------------------------- time #

    @staticmethod
    def _resolve_window(classification: router.Classification) -> TimeWindow:
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start = now + timedelta(hours=classification.time_offset_h)
        return TimeWindow(
            label=classification.time_label,
            start=start,
            end=start + timedelta(hours=classification.time_duration_h),
        )

    # ---------------------------------------------------------------- tasks #

    @staticmethod
    def _build_tasks(
        classification: router.Classification, *, has_destination: bool
    ) -> list[Task]:
        tasks: list[Task] = []
        for domain in classification.domains:
            agent = router.DOMAIN_AGENT.get(domain)
            if not agent:
                continue
            tasks.append(
                Task(
                    agent=agent,
                    goal=GOALS.get(domain, f"retrieve from {domain.value}"),
                    why=f"{classification.intent.value} needs the {domain.value} source",
                    args={"domain": domain.value},
                )
            )
        if classification.intent == Intent.PFZ_LOCATE:
            tasks.append(
                Task(
                    agent="pfz-analytics",
                    goal="detect thermal fronts and rank candidate fishing zones",
                    why="PFZ location requires spatial analysis, not a point lookup",
                    depends_on=["ocean-analytics"],
                )
            )
        if classification.intent == Intent.PRODUCTIVITY_SCAN:
            tasks.append(
                Task(
                    agent="pfz-analytics",
                    goal="scan the region for high chlorophyll and favourable SST",
                    why="the question is about regions, so a field scan is required",
                    depends_on=["ocean-analytics"],
                )
            )
        if classification.intent == Intent.PRODUCTIVITY_DIAGNOSIS:
            tasks.append(
                Task(
                    agent="diagnosis",
                    goal="correlate SST, mixed layer, chlorophyll and advisories",
                    why="a causal question needs multi-source correlation",
                    depends_on=["ocean-analytics"],
                )
            )
        if classification.intent == Intent.ROUTE_PLANNING and has_destination:
            tasks.append(
                Task(
                    agent="route-planner",
                    goal="score candidate tracks on sea state, wind and zone conflicts",
                    why="route choice needs conditions sampled along each track",
                    depends_on=["ocean-analytics", "weather-intelligence"],
                )
            )
        if classification.intent != Intent.SMALL_TALK:
            tasks.append(
                Task(
                    agent="risk-assessment",
                    goal="apply safety thresholds and produce an explainable verdict",
                    why="every operational answer carries a safety verdict",
                    depends_on=[t.agent for t in tasks],
                )
            )
            tasks.append(
                Task(
                    agent="visualisation",
                    goal="build map layers and charts for the retrieved evidence",
                    why="the answer must be shown on a map, not only described",
                    depends_on=["risk-assessment"],
                )
            )
        tasks.append(
            Task(
                agent="synthesis",
                goal="write the answer and attach evidence and reasoning",
                why="final step: one grounded reply with its supporting evidence",
                depends_on=[t.agent for t in tasks],
            )
        )
        # de-duplicate while preserving order
        seen: set[str] = set()
        unique: list[Task] = []
        for task in tasks:
            if task.agent in seen:
                continue
            seen.add(task.agent)
            unique.append(task)
        return unique
