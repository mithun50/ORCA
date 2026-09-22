"""Orchestrator.

Runs the plan the planner produced. Agents with no dependency on each other run
concurrently, which matters because every one of them is waiting on a
government server. Then risk, then visualisation, then synthesis, in that order,
because each depends on everything before it.

This in-process orchestrator is also the fallback path for the n8n workflow: n8n
drives the same stages through `/internal/*` endpoints, and if n8n is not running
the API answers `/chat` itself with identical logic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..rag import router
from ..schemas import ChatRequest, ChatResponse, Intent
from ..services import Services
from .advisory import AdvisoryAgent, DataDiscoveryAgent
from .base import AgentContext, EvidenceBook, Findings, Trace
from .geo import GeospatialReasoningAgent, RoutePlannerAgent
from .ocean import OceanAnalyticsAgent, WeatherIntelligenceAgent
from .planner import PlannerAgent
from .risk import RiskAssessmentAgent
from .synthesis import SynthesisAgent
from .viz import VisualisationAgent

log = logging.getLogger("orca.orchestrator")

#: agents that only read upstream data and can therefore run in parallel
PARALLEL_AGENTS = {
    "ocean-analytics",
    "weather-intelligence",
    "geospatial-reasoning",
    "advisory",
    "data-discovery",
}

#: agents that must run in this order after the parallel wave
SEQUENTIAL_AGENTS = ("route-planner", "risk-assessment", "visualisation", "synthesis")


class Orchestrator:
    def __init__(self, services: Services) -> None:
        self.services = services
        self.planner = PlannerAgent(services)
        self.agents: dict[str, Any] = {
            OceanAnalyticsAgent.name: OceanAnalyticsAgent(services),
            WeatherIntelligenceAgent.name: WeatherIntelligenceAgent(services),
            GeospatialReasoningAgent.name: GeospatialReasoningAgent(services),
            AdvisoryAgent.name: AdvisoryAgent(services),
            DataDiscoveryAgent.name: DataDiscoveryAgent(services),
            RoutePlannerAgent.name: RoutePlannerAgent(services),
            RiskAssessmentAgent.name: RiskAssessmentAgent(services),
            VisualisationAgent.name: VisualisationAgent(services),
            SynthesisAgent.name: SynthesisAgent(services),
        }

    # ------------------------------------------------------------------ main #

    async def handle(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        trace = Trace()
        ctx = await self.build_context(request, trace)

        if ctx.intent == Intent.SMALL_TALK:
            await self.agents["synthesis"].run(ctx)
            return self._respond(ctx, started)

        planned = [t.agent for t in ctx.plan.tasks]
        wave = [name for name in planned if name in PARALLEL_AGENTS]

        if wave:
            with trace.timed(
                "orchestrator",
                f"run {len(wave)} retrieval agents concurrently",
                rationale=(
                    "these agents have no dependency on each other and each waits "
                    "on a different government service, so running them in "
                    "parallel keeps the total latency near the slowest one"
                ),
                tool_args={"agents": wave},
            ) as step:
                results = await asyncio.gather(
                    *(self._safe_run(name, ctx) for name in wave),
                    return_exceptions=False,
                )
                failed = [name for name, ok in zip(wave, results) if not ok]
                step.outcome = (
                    f"{len(wave) - len(failed)}/{len(wave)} succeeded"
                    + (f"; failed: {', '.join(failed)}" if failed else "")
                )
                if failed:
                    step.status = "degraded"

        if "route-planner" in planned:
            await self._safe_run("route-planner", ctx)
        for name in ("risk-assessment", "visualisation", "synthesis"):
            await self._safe_run(name, ctx)

        return self._respond(ctx, started)

    async def build_context(self, request: ChatRequest, trace: Trace) -> AgentContext:
        """Planner pass only. Exposed so /internal/* can share it with n8n."""
        plan = await self.planner.plan(
            request.message,
            session_id=request.session_id,
            trace=trace,
            lat=request.lat,
            lon=request.lon,
            place=request.place,
            destination=request.destination,
        )
        ctx = AgentContext(
            message=request.message,
            session_id=request.session_id,
            plan=plan,
            trace=trace,
            evidence=EvidenceBook(),
            findings=Findings(),
            services=self.services,
            location=plan.location,
            destination=plan.destination,
            window=plan.window,
            language_hint=(
                request.language
                if request.language and request.language != "en"
                else router.detect_language_hint(request.message)
            ),
            history=self.services.history(request.session_id),
        )
        if plan.location:
            self.services.remember(
                request.session_id,
                "user",
                request.message,
                lat=str(plan.location.lat),
                lon=str(plan.location.lon),
                place=plan.location.name,
            )
        return ctx

    async def run_agent(self, name: str, ctx: AgentContext) -> bool:
        return await self._safe_run(name, ctx)

    # ---------------------------------------------------------------- guards #

    async def _safe_run(self, name: str, ctx: AgentContext) -> bool:
        agent = self.agents.get(name)
        if agent is None:
            ctx.trace.add(
                "orchestrator",
                f"agent {name} is not registered",
                status="skipped",
            )
            return False
        try:
            await agent.run(ctx)
            return True
        except Exception as exc:  # one agent must never sink the request
            log.exception("agent %s raised", name)
            ctx.trace.add(
                "orchestrator",
                f"agent {name} raised and was isolated",
                outcome=f"{type(exc).__name__}: {exc}",
                status="failed",
                rationale=(
                    "the request continues with the remaining agents and the "
                    "answer will say what is missing"
                ),
            )
            return False

    # -------------------------------------------------------------- response #

    def _respond(self, ctx: AgentContext, started: float) -> ChatResponse:
        answer = ctx.findings.diagnosis.get("answer") or (
            "I could not put together an answer for that. The reasoning trace "
            "shows which steps failed."
        )
        ctx.trace.add(
            "orchestrator",
            "request complete",
            outcome=(
                f"{len(ctx.evidence.all())} evidence items "
                f"({ctx.evidence.official_count()} from official agencies), "
                f"{len(ctx.trace.steps)} trace steps"
            ),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        self.services.remember(ctx.session_id, "assistant", answer)
        return ChatResponse(
            session_id=ctx.session_id,
            answer=answer,
            intent=ctx.intent,
            confidence=ctx.plan.confidence,
            location=ctx.location,
            window=ctx.window,
            risk=ctx.risk,
            evidence=ctx.evidence.all(),
            trace=ctx.trace.steps,
            layers=ctx.layers,
            charts=ctx.charts,
            followups=ctx.followups,
            degraded_sources=self.services.registry.degraded(),
            llm_used=bool(ctx.findings.diagnosis.get("llm_used")),
            language=ctx.findings.diagnosis.get("language") or ctx.language_hint or "en",
            audio_base64=ctx.findings.diagnosis.get("audio_base64"),
        )
