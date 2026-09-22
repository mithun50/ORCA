"""FastAPI application.

Two ways in, deliberately:

* `POST /chat` - the whole pipeline in one call, run by the in-process
  orchestrator. This is what the UI uses and what works with nothing else
  running.
* `POST /internal/*` - the same pipeline broken into the four stages the n8n
  router workflow drives: classify, retrieve (per knowledge source), assess,
  synthesize. n8n owns the branching; the API owns the data access.

Both paths share the same agents, so the n8n route cannot drift from the
in-process one.

Security note for reviewers: these endpoints are unauthenticated, which is fine
for a local prototype but is not acceptable if this is exposed beyond localhost.
Put an API key or a reverse proxy with auth in front of it before any public
deployment, and lock `ORCA_CORS_ORIGINS` down to the real UI origin.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import base64
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agents.base import AgentContext, EvidenceBook, Findings, Trace
from .agents.orchestrator import Orchestrator
from .config import get_settings
from .jev import get_jev_engine
from .rag import router as intent_router
from .rag.router import KnowledgeDomain
from .schemas import ChatRequest, ChatResponse
from .services import get_services
from .voice import get_voice_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("orca.api")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    services = get_services()
    app.state.services = services
    app.state.orchestrator = Orchestrator(services)
    app.state.warm_up = await services.warm_up()
    yield
    await services.shutdown()


def get_orchestrator() -> Orchestrator:
    if not hasattr(app.state, "orchestrator") or app.state.orchestrator is None:
        services = get_services()
        app.state.services = services
        app.state.orchestrator = Orchestrator(services)
    return app.state.orchestrator


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description=(
        "Agentic marine intelligence for Indian waters. Intent segregation, "
        "multi-RAG retrieval over ISRO/INCOIS/IMD sources, explainable risk "
        "assessment."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# health and introspection
# --------------------------------------------------------------------------- #

@app.get("/health")
async def health() -> dict[str, Any]:
    services = get_services()
    llm_provider = await services.llm.provider()
    jev = get_jev_engine()
    return {
        "status": "ok",
        "warm_up": getattr(app.state, "warm_up", {}),
        "llm_provider": llm_provider,
        "llm_routing": await services.llm.routing(),
        "jev": {
            "configured": jev.is_configured,
            "provider": jev.provider,
            "model": jev.model,
            "endpoint": jev.endpoint,
        },
        "jev_configured": jev.is_configured,
        "sarvam_voice_configured": get_voice_client().is_configured,
        **services.health(),
    }


@app.get("/knowledge-sources")
async def knowledge_sources() -> dict[str, Any]:
    """What each routable knowledge source is and where it reads from."""
    return {
        "domains": [
            {
                "id": KnowledgeDomain.ADVISORY.value,
                "name": "Marine advisories and regulations",
                "retriever": "document RAG (BM25 + synonym expansion)",
                "reads": [
                    "IMD sub-division warnings (live HTML)",
                    "IMD RSMC New Delhi cyclone bulletin (live HTML)",
                    "INCOIS PFZ advisory page (live HTML)",
                    "seeded corpus: safety thresholds, PFZ basis, geofencing rules",
                ],
                "endpoint": "/internal/retrieve/advisory",
            },
            {
                "id": KnowledgeDomain.OCEAN.value,
                "name": "Ocean state",
                "retriever": "time-series RAG",
                "reads": [
                    "MOSDAC THREDDS OSF_CIRC: SST, salinity, MLD, currents (ISRO)",
                    "MOSDAC THREDDS OSF_WAVE: SWH, period, direction, swell (ISRO)",
                    "Open-Meteo marine as the live fallback",
                ],
                "endpoint": "/internal/retrieve/ocean",
            },
            {
                "id": KnowledgeDomain.WEATHER.value,
                "name": "Marine weather",
                "retriever": "time-series RAG",
                "reads": [
                    "wind, gusts, rain, CAPE, visibility",
                    "sea level (tide) from the marine model",
                ],
                "endpoint": "/internal/retrieve/weather",
            },
            {
                "id": KnowledgeDomain.GEOSPATIAL.value,
                "name": "Geospatial and geofencing",
                "retriever": "geospatial RAG (shapely)",
                "reads": [
                    "India EEZ polygon (Marine Regions WFS, cached)",
                    "seeded MPA / restricted / IMBL zone layer",
                    "coastal harbour gazetteer",
                ],
                "endpoint": "/internal/retrieve/geospatial",
            },
            {
                "id": KnowledgeDomain.HAZARD.value,
                "name": "Hazards and alerts",
                "retriever": "connector + rules",
                "reads": [
                    "GDACS tropical cyclone events (geometry)",
                    "IMD warning text (authoritative wording)",
                ],
                "endpoint": "/internal/retrieve/hazard",
            },
            {
                "id": KnowledgeDomain.CATALOG.value,
                "name": "Dataset discovery",
                "retriever": "live catalogue queries",
                "reads": [
                    "MOSDAC THREDDS catalogues",
                    "INCOIS ERDDAP griddap index",
                    "NASA CMR collections",
                ],
                "endpoint": "/internal/retrieve/catalog",
            },
        ],
        "intent_routing": {
            intent.value: [d.value for d in domains]
            for intent, domains in intent_router.INTENT_DOMAINS.items()
        },
    }


# --------------------------------------------------------------------------- #
# single-call chat (in-process orchestrator)
# --------------------------------------------------------------------------- #

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, via: str = "") -> ChatResponse:
    """Answer a question.

    `?via=n8n` hands the request to the n8n router workflow, which drives the
    same stages through the `/internal/*` endpoints. Without it, the in-process
    orchestrator runs the identical agent graph. If n8n is asked for but is not
    reachable, the request falls back to the in-process path and says so.
    """
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty")

    orchestrator: Orchestrator = get_orchestrator()

    if via.lower() == "n8n":
        routed = await _route_via_n8n(request)
        if routed is not None:
            return routed
        log.warning("n8n unreachable; falling back to the in-process orchestrator")
        response = await orchestrator.handle(request)
        response.degraded_sources.append(
            "n8n router unreachable, answered in-process instead"
        )
        return response

    return await orchestrator.handle(request)


async def _route_via_n8n(request: ChatRequest) -> ChatResponse | None:
    """POST the turn to the n8n webhook and return its ChatResponse."""
    if not settings.n8n_base:
        return None
    url = f"{settings.n8n_base.rstrip('/')}/webhook/orca-router"
    headers = {"Content-Type": "application/json"}
    if settings.n8n_webhook_token:
        headers["x-orca-token"] = settings.n8n_webhook_token
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                url, json=request.model_dump(mode="json"), headers=headers
            )
        if resp.status_code >= 400:
            log.warning("n8n webhook returned %s: %s", resp.status_code, resp.text[:300])
            return None
        payload = resp.json()
        if isinstance(payload, list) and payload:
            payload = payload[0]
        if not isinstance(payload, dict) or "answer" not in payload:
            log.warning("n8n webhook returned an unexpected shape")
            return None
        response = ChatResponse(**payload)
        response.via_n8n = True
        return response
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        log.warning("n8n routing failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# multimodal voice: Sarvam AI STT & TTS
# --------------------------------------------------------------------------- #

class TtsApiRequest(BaseModel):
    text: str
    language: str = "en"
    speaker: str | None = None


class SttApiResponse(BaseModel):
    transcript: str
    language_code: str
    detected_locale: str
    confidence: float
    provider: str


@app.post("/api/voice/stt", response_model=SttApiResponse)
async def api_voice_stt(
    file: UploadFile | None = None,
    audio_base64: str = Form(default=""),
    language: str = Form(default="unknown"),
) -> SttApiResponse:
    """Transcribes deck audio using Sarvam Saaras with diesel acoustic filtering."""
    voice_client = get_voice_client()
    if file is not None:
        audio_bytes = await file.read()
    elif audio_base64:
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception:
            raise HTTPException(status_code=422, detail="invalid base64 audio") from None
    else:
        raise HTTPException(status_code=422, detail="either audio file or audio_base64 is required")

    res = await voice_client.speech_to_text(audio_bytes, language_code=language)
    return SttApiResponse(
        transcript=res.transcript,
        language_code=res.language_code,
        detected_locale=res.detected_locale,
        confidence=res.confidence,
        provider=res.provider,
    )


@app.post("/api/voice/tts")
async def api_voice_tts(request: TtsApiRequest) -> dict[str, Any]:
    """Synthesizes coastal vernacular speech using Sarvam Bulbul."""
    voice_client = get_voice_client()
    res = await voice_client.text_to_speech(
        request.text, language_code=request.language, speaker=request.speaker
    )
    return {
        "audio_base64": res.audio_base64,
        "format": res.audio_format,
        "language": res.target_language,
        "speaker": res.speaker,
        "provider": res.provider,
    }


@app.post("/chat/voice", response_model=ChatResponse)
async def chat_voice(
    file: UploadFile | None = None,
    audio_base64: str = Form(default=""),
    language: str = Form(default="unknown"),
    session_id: str = Form(default="default"),
    lat: float | None = Form(default=None),
    lon: float | None = Form(default=None),
    place: str | None = Form(default=None),
) -> ChatResponse:
    """End-to-end hands-free voice query: Sarvam STT -> ORCA reasoning -> Sarvam TTS."""
    voice_client = get_voice_client()
    if file is not None:
        audio_bytes = await file.read()
    elif audio_base64:
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception:
            raise HTTPException(status_code=422, detail="invalid base64 audio") from None
    else:
        raise HTTPException(status_code=422, detail="either audio file or audio_base64 is required")

    stt_res = await voice_client.speech_to_text(audio_bytes, language_code=language)
    transcript = stt_res.transcript.strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="could not transcribe voice input")

    orchestrator: Orchestrator = get_orchestrator()
    chat_req = ChatRequest(
        message=transcript,
        session_id=session_id,
        lat=lat,
        lon=lon,
        place=place,
        language=stt_res.detected_locale,
    )
    response = await orchestrator.handle(chat_req)
    return response


# --------------------------------------------------------------------------- #
# staged endpoints for the n8n router workflow
# --------------------------------------------------------------------------- #

class ClassifyRequest(BaseModel):
    message: str
    session_id: str = "default"
    lat: float | None = None
    lon: float | None = None
    place: str | None = None
    destination: str | None = None


class ClassifyResponse(BaseModel):
    session_id: str
    message: str
    intent: str
    confidence: float
    decided_by: str
    domains: list[str]
    agents: list[str]
    matched_rules: list[str]
    scores: dict[str, float]
    language_hint: str
    location: dict[str, Any] | None
    destination: dict[str, Any] | None
    window: dict[str, str]
    clarification_needed: str
    trace: list[dict[str, Any]]


@app.post("/internal/classify", response_model=ClassifyResponse)
async def internal_classify(request: ClassifyRequest) -> ClassifyResponse:
    """Stage 1 for n8n: segregate the input and name the knowledge sources.

    The `domains` array is what the n8n Switch node branches on.
    """
    orchestrator: Orchestrator = get_orchestrator()
    trace = Trace()
    chat_request = ChatRequest(**request.model_dump())
    ctx = await orchestrator.build_context(chat_request, trace)
    lexical = intent_router.classify(request.message)
    domains = intent_router.domains_for(ctx.intent)
    return ClassifyResponse(
        session_id=request.session_id,
        message=request.message,
        intent=ctx.intent.value,
        confidence=round(ctx.plan.confidence, 3),
        decided_by="llm" if "decided_by=llm" in ctx.plan.planner_notes else "lexical",
        domains=[d.value for d in domains],
        agents=intent_router.agents_for(domains),
        matched_rules=lexical.matched_rules,
        scores={k: round(v, 2) for k, v in lexical.scores.items()},
        language_hint=ctx.language_hint,
        location=ctx.location.model_dump() if ctx.location else None,
        destination=ctx.destination.model_dump() if ctx.destination else None,
        window={
            "label": ctx.window.label if ctx.window else "now",
            "start": ctx.window.start.isoformat() if ctx.window else "",
            "end": ctx.window.end.isoformat() if ctx.window else "",
        },
        clarification_needed=ctx.plan.clarification_needed,
        trace=[step.model_dump(mode="json") for step in trace.steps],
    )


class RetrieveRequest(BaseModel):
    message: str
    session_id: str = "default"
    intent: str = ""
    lat: float | None = None
    lon: float | None = None
    place: str | None = None
    destination: str | None = None
    window_start: str = ""
    window_end: str = ""


class RetrieveResponse(BaseModel):
    domain: str
    agent: str
    findings: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    layers: list[dict[str, Any]] = Field(default_factory=list)
    charts: list[dict[str, Any]] = Field(default_factory=list)


DOMAIN_FINDING_KEYS: dict[str, tuple[str, ...]] = {
    KnowledgeDomain.OCEAN.value: ("ocean", "waves"),
    KnowledgeDomain.WEATHER.value: ("weather",),
    KnowledgeDomain.GEOSPATIAL.value: ("geo",),
    KnowledgeDomain.HAZARD.value: ("hazards",),
    KnowledgeDomain.ADVISORY.value: ("advisories",),
    KnowledgeDomain.CATALOG.value: ("catalog",),
}


@app.post("/internal/retrieve/{domain}", response_model=RetrieveResponse)
async def internal_retrieve(domain: str, request: RetrieveRequest) -> RetrieveResponse:
    """Stage 2 for n8n: retrieve from exactly one knowledge source.

    One branch of the n8n Switch calls this per domain, in parallel.
    """
    try:
        knowledge_domain = KnowledgeDomain(domain)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"unknown knowledge domain {domain!r}; valid: "
                + ", ".join(d.value for d in KnowledgeDomain)
            ),
        ) from None

    agent_name = intent_router.DOMAIN_AGENT[knowledge_domain]
    orchestrator: Orchestrator = get_orchestrator()
    trace = Trace()
    ctx = await orchestrator.build_context(
        ChatRequest(
            message=request.message,
            session_id=request.session_id,
            lat=request.lat,
            lon=request.lon,
            place=request.place,
            destination=request.destination,
        ),
        trace,
    )
    await orchestrator.run_agent(agent_name, ctx)

    findings: dict[str, Any] = {}
    for key in DOMAIN_FINDING_KEYS.get(domain, ()):
        findings[key] = getattr(ctx.findings, key)
    if ctx.findings.notes:
        findings["notes"] = ctx.findings.notes

    return RetrieveResponse(
        domain=domain,
        agent=agent_name,
        findings=findings,
        evidence=[e.model_dump(mode="json") for e in ctx.evidence.all()],
        trace=[s.model_dump(mode="json") for s in trace.steps],
        layers=[layer.model_dump(mode="json") for layer in ctx.layers],
        charts=[chart.model_dump(mode="json") for chart in ctx.charts],
    )


class SynthesizeRequest(BaseModel):
    message: str
    session_id: str = "default"
    lat: float | None = None
    lon: float | None = None
    place: str | None = None
    destination: str | None = None
    #: merged `findings` objects from the retrieve branches
    findings: list[dict[str, Any]] = Field(default_factory=list)
    #: merged `evidence` arrays from the retrieve branches
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    #: merged `trace` arrays, so the final response keeps the full n8n trace
    trace: list[dict[str, Any]] = Field(default_factory=list)
    #: set false to force deterministic template synthesis. The n8n critic uses
    #: this on its repair pass after it rejects an LLM rewrite.
    allow_llm: bool = True


@app.post("/internal/synthesize", response_model=ChatResponse)
async def internal_synthesize(request: SynthesizeRequest) -> ChatResponse:
    """Stage 3 and 4 for n8n: assess risk over the merged findings, then answer.

    n8n posts back everything its retrieve branches collected. This rebuilds the
    agent context from that payload and runs risk, visualisation and synthesis,
    so the reasoning trace in the final answer spans the whole n8n run.
    """
    from .schemas import Evidence, TraceStep

    orchestrator: Orchestrator = get_orchestrator()
    trace = Trace()
    ctx = await orchestrator.build_context(
        ChatRequest(
            message=request.message,
            session_id=request.session_id,
            lat=request.lat,
            lon=request.lon,
            place=request.place,
            destination=request.destination,
        ),
        trace,
    )

    # replay what the n8n branches already retrieved
    for step in request.trace:
        try:
            trace.steps.append(TraceStep(**step))
        except Exception:  # noqa: BLE001 - a malformed step must not break the answer
            continue
    trace.steps.sort(key=lambda s: s.seq)
    for i, step in enumerate(trace.steps, start=1):
        step.seq = i

    for item in request.evidence:
        try:
            ctx.evidence.add(Evidence(**item))
        except Exception:  # noqa: BLE001
            continue

    merged: dict[str, Any] = {}
    for block in request.findings:
        merged.update(block)
    for key in ("ocean", "waves", "weather", "geo", "hazards", "route", "pfz",
                "diagnosis"):
        if isinstance(merged.get(key), dict):
            setattr(ctx.findings, key, merged[key])
    for key in ("advisories", "catalog", "notes"):
        if isinstance(merged.get(key), list):
            setattr(ctx.findings, key, merged[key])

    trace.add(
        "orchestrator",
        "merge retrieval results from the n8n branches",
        rationale=(
            "n8n ran the retrieval fan-out; the API replays those findings so "
            "risk assessment and synthesis see exactly the same evidence as the "
            "in-process path"
        ),
        outcome=(
            f"{len(request.findings)} finding blocks, "
            f"{len(request.evidence)} evidence items, "
            f"{len(request.trace)} trace steps replayed"
        ),
    )

    await orchestrator.run_agent("risk-assessment", ctx)
    await orchestrator.run_agent("visualisation", ctx)
    ctx.findings.diagnosis["allow_llm"] = request.allow_llm
    await orchestrator.run_agent("synthesis", ctx)

    services = get_services()
    answer = ctx.findings.diagnosis.get("answer") or (
        "I could not put together an answer for that."
    )
    services.remember(ctx.session_id, "assistant", answer)
    return ChatResponse(
        session_id=ctx.session_id,
        answer=answer,
        intent=ctx.intent,
        confidence=ctx.plan.confidence,
        location=ctx.location,
        window=ctx.window,
        risk=ctx.risk,
        evidence=ctx.evidence.all(),
        citations=ctx.citations,
        trace=trace.steps,
        layers=ctx.layers,
        charts=ctx.charts,
        followups=ctx.followups,
        degraded_sources=services.registry.degraded(),
        llm_used=bool(ctx.findings.diagnosis.get("llm_used")),
        language=ctx.findings.diagnosis.get("language") or ctx.language_hint or "en",
        audience=ctx.audience.audience.value if ctx.audience else "fisherman",
        audience_confidence=ctx.audience.confidence if ctx.audience else 0.0,
        model_roles=orchestrator._model_roles(ctx),
        cost=orchestrator._cost_summary(orchestrator._model_roles(ctx), ctx),
        audio_base64=ctx.findings.diagnosis.get("audio_base64"),
    )


def run() -> None:  # pragma: no cover - entry point
    import uvicorn

    uvicorn.run(
        "orca.main:app", host=settings.host, port=settings.port, reload=False
    )


# The chat UI is a single static page with no build step, mounted last so it
# never shadows an API route.
_UI_DIR = Path(__file__).resolve().parents[2] / "frontend"
if _UI_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")
else:  # pragma: no cover
    log.warning("UI directory %s not found; serving API only", _UI_DIR)


if __name__ == "__main__":  # pragma: no cover
    run()
