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
from ..rag.audience import detect_audience
from ..schemas import ChatRequest, ChatResponse, CostSummary, Intent, ModelRole
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

        # If the planner already knows it has to ask a question back, retrieving
        # is pointless: there is no position to retrieve for. Calling the
        # agencies with 0N 0E just produces HTTP 400s and a misleading
        # "degraded source" report on a request that was never going to answer.
        if ctx.plan.clarification_needed and ctx.location is None:
            ctx.trace.add(
                "orchestrator",
                "skipping retrieval, the planner needs a location first",
                rationale=(
                    "no position was resolved, so there is nothing to query the "
                    "agencies about; asking the user is the only useful next step"
                ),
                status="skipped",
                outcome=ctx.plan.clarification_needed,
            )
            await self._safe_run("synthesis", ctx)
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
        ctx.audience = detect_audience(
            request.message, depth_hint=len(ctx.history)
        )
        trace.add(
            "planner",
            f"reading the question as coming from a {ctx.audience.audience.value}",
            rationale=(
                "the same facts have to be worded differently for a boat owner, an "
                "analyst and an authority; getting this wrong produces an answer "
                "the reader cannot act on"
            ),
            tool="audience-router",
            tool_args={"signals": ctx.audience.matched_rules[:6]},
            outcome=(
                f"{ctx.audience.audience.value} at {ctx.audience.confidence:.2f} "
                f"({ctx.audience.decided_by}); style: "
                f"{ctx.audience.style['reading_level']}"
            ),
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

    @staticmethod
    def _model_roles(ctx: AgentContext) -> list[ModelRole]:
        """Full attribution: every model that could have run, and whether it did.

        Listing the ones that did not run matters as much as the ones that did. A
        reader needs to see that the verdict came from rules, not from a model,
        and that a translation did or did not happen.
        """
        roles: list[ModelRole] = []
        decided_by_llm = "decided_by=llm" in (ctx.plan.planner_notes or "")

        roles.append(
            ModelRole(
                role="intent",
                job="decide what is being asked and which sources can answer it",
                provider="openrouter" if decided_by_llm else "rules",
                model="agentic model" if decided_by_llm else "52 weighted regex rules",
                used=True,
                detail=(
                    "lexical confidence was below 0.55, so the agentic model "
                    "arbitrated"
                    if decided_by_llm
                    else "the keyword router was confident, so no model was needed"
                ),
            )
        )
        roles.append(
            ModelRole(
                role="audience",
                job="decide whether to answer as if to a fisherman, researcher or official",
                provider="rules",
                model="audience router",
                used=True,
                detail=(
                    f"{ctx.audience.audience.value} at "
                    f"{ctx.audience.confidence:.2f} ({ctx.audience.decided_by})"
                    if ctx.audience
                    else "defaulted"
                ),
            )
        )
        roles.append(
            ModelRole(
                role="verdict",
                job="decide the safety band",
                provider="rules",
                model="threshold rules, worst band wins",
                used=ctx.risk is not None,
                detail=(
                    f"{len(ctx.risk.findings)} rules evaluated, verdict "
                    f"{ctx.risk.band.value}"
                    if ctx.risk
                    else "no verdict was needed for this question"
                ),
            )
        )

        jev = (ctx.findings.hazards or {}).get("jev_decision") or {}
        if jev:
            live = jev.get("engine") != "jev-rule-emulator"
            roles.append(
                ModelRole(
                    role="judgment",
                    job="second opinion on safety as typed probabilities",
                    provider="typesafe" if live else "rules",
                    model=jev.get("model") or "rule emulator",
                    used=True,
                    detail=(
                        f"{jev.get('verdict')} / {jev.get('action')} at confidence "
                        f"{jev.get('confidence', 0):.2f}. Advisory only: it can make "
                        "the verdict stricter, never looser."
                    ),
                    input_tokens=int(jev.get("input_tokens") or 0),
                    output_tokens=int(jev.get("output_tokens") or 0),
                    cost_usd=jev.get("cost_usd"),
                    elapsed_ms=int(jev.get("elapsed_ms") or 0),
                    billing_unit="input tokens" if live else "none",
                )
            )

        syn = ctx.findings.diagnosis.get("synthesis_role") or {}
        discarded = bool(syn.get("billed_but_discarded"))
        roles.append(
            ModelRole(
                role="wording",
                job="write the answer in the reader's language and register",
                provider=syn.get("provider") or "none",
                model=syn.get("model") or "deterministic templates",
                used=bool(syn.get("used")),
                detail=(
                    "the model reworded the draft; every number and the verdict "
                    "were carried over unchanged"
                    if syn.get("used")
                    else (
                        "the model was called and billed, but its rewrite was "
                        "rejected for dropping a citation or the verdict, so the "
                        "template draft shipped instead"
                        if discarded
                        else "the template draft shipped as written, with no model involved"
                    )
                ),
                input_tokens=int(syn.get("input_tokens") or 0),
                output_tokens=int(syn.get("output_tokens") or 0),
                cost_usd=syn.get("cost_usd"),
                billing_unit=(
                    "tokens"
                    if syn.get("provider") in ("openrouter", "gemini", "openai")
                    else "none"
                ),
            )
        )

        tts = ctx.findings.diagnosis.get("tts_role") or {}
        if tts:
            live_tts = "sarvam" in str(tts.get("provider"))
            roles.append(
                ModelRole(
                    role="voice",
                    job="speak the answer aloud",
                    provider="sarvam" if live_tts else "offline",
                    model=tts.get("model") or "",
                    used=bool(tts.get("used")),
                    detail=(
                        f"voice {tts.get('speaker', '')} via {tts.get('provider', '')}"
                        + (
                            f", {tts.get('characters', 0)} characters in "
                            f"{tts.get('chunks', 1)} chunk(s)"
                            if live_tts
                            else ""
                        )
                    ),
                    cost_usd=None,  # Sarvam does not price the call in its response
                    billing_unit="characters" if live_tts else "none",
                )
            )
        return roles

    @staticmethod
    def _cost_summary(roles: list[ModelRole], ctx: AgentContext) -> CostSummary:
        """Total only what providers actually reported. Never estimate a price."""
        total = 0.0
        unpriced: list[str] = []
        in_tok = out_tok = 0
        for role in roles:
            in_tok += role.input_tokens
            out_tok += role.output_tokens
            if role.billing_unit == "none":
                continue
            if role.cost_usd is None:
                unpriced.append(role.role)
            else:
                total += role.cost_usd
        tts = ctx.findings.diagnosis.get("tts_role") or {}
        return CostSummary(
            total_usd=round(total, 8),
            unpriced_roles=unpriced,
            input_tokens=in_tok,
            output_tokens=out_tok,
            tts_characters=int(tts.get("characters") or 0),
            complete=not unpriced,
        )

    def _respond(self, ctx: AgentContext, started: float) -> ChatResponse:
        answer = ctx.findings.diagnosis.get("answer") or (
            "I could not put together an answer for that. The reasoning trace "
            "shows which steps failed."
        )
        roles = self._model_roles(ctx)
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
            citations=ctx.citations,
            trace=ctx.trace.steps,
            layers=ctx.layers,
            charts=ctx.charts,
            followups=ctx.followups,
            degraded_sources=self.services.registry.degraded(),
            llm_used=bool(ctx.findings.diagnosis.get("llm_used")),
            language=ctx.findings.diagnosis.get("language") or ctx.language_hint or "en",
            audience=ctx.audience.audience.value if ctx.audience else "fisherman",
            audience_confidence=ctx.audience.confidence if ctx.audience else 0.0,
            model_roles=roles,
            cost=self._cost_summary(roles, ctx),
            audio_base64=ctx.findings.diagnosis.get("audio_base64"),
        )
