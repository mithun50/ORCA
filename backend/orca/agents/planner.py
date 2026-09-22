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
from ..llm import LlmRole
from ..rag import router
from ..rag.router import KnowledgeDomain
from ..schemas import Intent, Location, Plan, Task, TimeWindow
from ..services import Services
from .base import Trace

#: How far from Indian waters a place can be and still be worth answering for.
#: A landing centre sits a little inland of the EEZ polygon's landward edge, and
#: the polygon itself is coarse, so a small tolerance is needed. Bengaluru is
#: ~290 km from the sea, so this comfortably separates coastal from inland.
COASTAL_TOLERANCE_KM = 30.0

#: UI affordance labels that must never be printed as though they were a place.
#: The browser sends these to say *how* a position was obtained, not where it is.
_UI_PIN_LABELS = {
    "picked on map", "device gps", "typed", "your position", "pinned",
    "map pin", "gps",
}

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


GEOCODE_PROMPT = """Locate this place for a marine question about Indian waters.

Place as the user wrote it: "{place}"
Full question for context: "{message}"

Reply as JSON only:
{{"name": "<the standard English name>",
  "lat": <decimal degrees, positive north>,
  "lon": <decimal degrees, positive east>,
  "country": "<country>",
  "kind": "<one of: harbour, landing-centre, beach, coastal-town, inland-city, sea-area, unknown>",
  "confident": <true or false>}}

Rules:
- Give the real coordinates. Do not invent a plausible-looking pair.
- If you genuinely do not know the place, set "confident": false and lat/lon to 0.
- "kind" is your reading of the place itself, not of the question.
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
        location, refusal = await self._resolve_location(
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
            # The resolver already worked out *why* it could not place the user,
            # so reuse that rather than second-guessing it here.
            clarification = refusal or (
                "I need a location for this. Tell me the harbour or landing "
                "centre you are sailing from, or send your latitude and "
                "longitude."
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
        provider, model = await self.services.llm.resolve_role(LlmRole.AGENTIC)
        with trace.timed(
            self.name,
            "escalate ambiguous intent to the agentic model",
            rationale=(
                f"lexical confidence {current.confidence:.2f} is below the 0.55 "
                "threshold, so a second opinion from a reasoning model is worth "
                "the latency"
            ),
            tool="llm-classifier",
            tool_args={"provider": provider, "model": model},
        ) as step:
            payload = await self.services.llm.complete_json(
                LLM_INTENT_PROMPT.format(message=message), role=LlmRole.AGENTIC
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

    async def _geocode_agent(
        self, phrase: str, message: str, trace: Trace
    ) -> tuple[Location | None, str, float | None]:
        """Resolve a place the gazetteer does not know, then verify it by geometry.

        Two steps, deliberately split. The model supplies coordinates, which is
        knowledge retrieval and the sort of thing it is good at. Whether those
        coordinates are at sea is then decided by measuring against the India EEZ
        polygon, which is geometry and not a matter of opinion. So a hallucinated
        harbour in the middle of the Deccan gets caught by the distance check
        rather than being taken on trust.

        Returns (location, verdict, distance_km) where verdict is one of
        "coastal", "inland", "unknown".
        """
        with trace.timed(
            self.name,
            f"look up {phrase!r}, which is not in the gazetteer",
            rationale=(
                "the built-in gazetteer covers 58 places; rather than refusing "
                "everything else, the agentic model supplies coordinates and the "
                "EEZ polygon decides whether they are at sea"
            ),
            tool="geocode-agent",
            tool_args={"place": phrase},
        ) as step:
            payload = await self.services.llm.complete_json(
                GEOCODE_PROMPT.format(place=phrase, message=message[:300]),
                role=LlmRole.AGENTIC,
                max_tokens=300,
            )
            if not payload or not payload.get("confident"):
                step.status = "degraded"
                step.outcome = (
                    "no coordinates could be resolved for this place"
                    if payload
                    else "no agentic model available to resolve it"
                )
                return None, "unknown", None

            try:
                lat = float(payload["lat"])
                lon = float(payload["lon"])
            except (KeyError, TypeError, ValueError):
                step.status = "degraded"
                step.outcome = "the reply did not contain usable coordinates"
                return None, "unknown", None

            if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
                step.status = "degraded"
                step.outcome = f"coordinates {lat},{lon} are not usable"
                return None, "unknown", None

            name = str(payload.get("name") or phrase).strip() or phrase
            distance = await self.services.geo_rag.distance_to_sea_km(lat, lon)
            if distance is None:
                step.status = "degraded"
                step.outcome = (
                    f"{name} resolved to {lat:.3f}N {lon:.3f}E but the EEZ polygon "
                    "is unavailable, so I cannot confirm it is at sea"
                )
                return None, "unknown", None

            location = Location(
                name=name, lat=lat, lon=lon, source="geocode-agent",
                state=str(payload.get("country") or ""),
            )
            if distance <= COASTAL_TOLERANCE_KM:
                step.outcome = (
                    f"{name} at {lat:.3f}N {lon:.3f}E is {distance:.0f} km from "
                    "Indian waters, close enough to answer for"
                )
                return location, "coastal", distance

            step.status = "degraded"
            step.outcome = (
                f"{name} at {lat:.3f}N {lon:.3f}E is {distance:.0f} km inland from "
                "Indian waters, so there is no sea state to report"
            )
            return location, "inland", distance

    async def _resolve_location(
        self,
        message: str,
        *,
        session_id: str,
        lat: float | None,
        lon: float | None,
        place: str | None,
        trace: Trace,
    ) -> tuple[Location | None, str]:
        """Resolve where the question is about.

        Returns (location, refusal). A refusal string is set when we deliberately
        decline to answer rather than substituting somewhere else, and it is what
        the user is shown.
        """
        with trace.timed(
            self.name,
            "resolve location",
            rationale=(
                "precedence: a place named in the question, then GPS from the "
                "client, then typed coordinates, then the gazetteer, then the "
                "geocoding agent verified against the EEZ polygon, then the "
                "location from earlier in this conversation"
            ),
            tool="gazetteer",
        ) as step:
            # A place named in the question is considered before a pinned
            # position. Otherwise "can I go to sea at Bengaluru" silently answers
            # about whatever pin is on the map, which ignores the question rather
            # than answering it.
            named_phrase = gazetteer.unresolved_place(message)
            gazetteer_hits = gazetteer.find_all_places(message)

            if named_phrase and not gazetteer_hits:
                located, verdict, distance = await self._geocode_agent(
                    named_phrase, message, trace
                )
                if verdict == "coastal" and located is not None:
                    step.outcome = (
                        f"{located.name} resolved by the geocoding agent and "
                        f"confirmed {distance:.0f} km from Indian waters"
                    )
                    return located, ""
                if verdict == "inland" and located is not None:
                    step.status = "degraded"
                    step.outcome = (
                        f"{located.name} is {distance:.0f} km inland; refusing to "
                        "answer about a coastal position instead"
                    )
                    return None, (
                        f"{located.name} is about {distance:.0f} km inland, so "
                        "there is no sea there to report on. If you are heading to "
                        "the coast, tell me the harbour or landing centre you will "
                        "sail from, or send a latitude and longitude."
                    )
                # the agent could not place it; fall back to the offline list
                offline = gazetteer.find_inland(message)
                if offline:
                    step.status = "degraded"
                    step.outcome = f"{offline[0]} is inland (offline list)"
                    return None, (
                        f"{offline[0]} is inland, so there is no sea there to "
                        f"report on. The nearest coast I cover is {offline[1]}, so "
                        f"you could ask \"is it safe off {offline[1]} tomorrow "
                        "morning\", or send a latitude and longitude."
                    )
                if lat is None or lon is None:
                    step.status = "degraded"
                    step.outcome = f"could not place {named_phrase!r}"
                    return None, (
                        f"I could not work out where \"{named_phrase}\" is, and I "
                        "am not going to answer about somewhere else instead. Give "
                        "me the nearest harbour or landing centre, or your "
                        "latitude and longitude."
                    )

            if lat is not None and lon is not None:
                # A pinned position is a coordinate, not a place name. Naming it
                # after a UI label produced answers like "off picked on map".
                label = (place or "").strip()
                if not label or label.lower() in _UI_PIN_LABELS:
                    label = f"{lat:.3f}N {lon:.3f}E"
                step.outcome = f"client position {lat:.3f}N {lon:.3f}E"
                return Location(name=label, lat=lat, lon=lon, source="explicit"), ""

            typed = gazetteer.parse_coords(message)
            if typed:
                step.outcome = f"coordinates parsed from the message {typed[0]:.3f}N {typed[1]:.3f}E"
                return Location(
                    name="the position you gave", lat=typed[0], lon=typed[1],
                    source="explicit",
                ), ""

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
                    ), ""

            # first place *mentioned* wins, so "from Chennai to Kakinada" keeps
            # Chennai as the origin rather than whichever name is longer
            mentioned = gazetteer_hits
            hit = mentioned[0] if mentioned else None
            if hit:
                step.outcome = (
                    f"gazetteer matched {hit.name} ({hit.district}, {hit.state})"
                    + (f", first of {len(mentioned)} places named" if len(mentioned) > 1 else "")
                )
                return Location(
                    name=hit.name, lat=hit.lat, lon=hit.lon, source="gazetteer",
                    district=hit.district, state=hit.state,
                ), ""

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
                ), ""

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
            ), ""

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
