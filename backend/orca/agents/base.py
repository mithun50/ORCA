"""Agent framework: shared context, evidence bookkeeping, reasoning trace.

Every agent is a small async class with one job, a declared set of tools it is
allowed to call, and an obligation to write what it did into the trace. The
orchestrator never lets an agent fail the request: an agent that cannot do its
job records a `degraded` or `failed` step and the synthesis agent tells the user
what is missing.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterator, Protocol

from ..schemas import (
    ChartSeries,
    Citation,
    Evidence,
    Intent,
    Location,
    MapLayer,
    Plan,
    RiskAssessment,
    TimeWindow,
    TraceStep,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..services import Services
    from ..rag.audience import AudienceCall
    from .synthesis import Citer

log = logging.getLogger("orca.agents")


class Trace:
    """Ordered, timed record of what the agents did and why."""

    def __init__(self) -> None:
        self.steps: list[TraceStep] = []

    def add(
        self,
        agent: str,
        action: str,
        *,
        rationale: str = "",
        tool: str = "",
        tool_args: dict[str, Any] | None = None,
        outcome: str = "",
        evidence_ids: list[str] | None = None,
        duration_ms: int = 0,
        status: str = "ok",
    ) -> TraceStep:
        step = TraceStep(
            seq=len(self.steps) + 1,
            agent=agent,
            action=action,
            rationale=rationale,
            tool=tool,
            tool_args=tool_args or {},
            outcome=outcome,
            evidence_ids=evidence_ids or [],
            duration_ms=duration_ms,
            status=status,  # type: ignore[arg-type]
        )
        self.steps.append(step)
        return step

    @contextmanager
    def timed(
        self, agent: str, action: str, *, rationale: str = "", tool: str = "",
        tool_args: dict[str, Any] | None = None,
    ) -> Iterator[TraceStep]:
        """Record a step and fill in its duration and status automatically."""
        step = self.add(
            agent, action, rationale=rationale, tool=tool, tool_args=tool_args
        )
        started = time.perf_counter()
        try:
            yield step
        except Exception as exc:  # agents must not break the request
            step.status = "failed"
            step.outcome = f"{type(exc).__name__}: {exc}"
            log.exception("agent %s step %s failed", agent, action)
        finally:
            step.duration_ms = int((time.perf_counter() - started) * 1000)


class EvidenceBook:
    """Deduplicating store that keeps evidence ids stable across agents."""

    def __init__(self) -> None:
        self._items: dict[str, Evidence] = {}

    def add(self, evidence: Evidence | None) -> str:
        if evidence is None:
            return ""
        if evidence.id in self._items:
            return evidence.id
        self._items[evidence.id] = evidence
        return evidence.id

    def add_many(self, items: list[Evidence | None]) -> list[str]:
        return [eid for eid in (self.add(i) for i in items) if eid]

    def get(self, evidence_id: str) -> Evidence | None:
        return self._items.get(evidence_id)

    def all(self) -> list[Evidence]:
        return list(self._items.values())

    def official_count(self) -> int:
        return sum(1 for e in self._items.values() if e.provenance.official)

    def stale_ids(self) -> list[str]:
        return [e.id for e in self._items.values() if e.provenance.is_stale]


@dataclass
class Findings:
    """Structured state the agents build up and synthesis reads."""

    ocean: dict[str, Any] = field(default_factory=dict)
    weather: dict[str, Any] = field(default_factory=dict)
    waves: dict[str, Any] = field(default_factory=dict)
    hazards: dict[str, Any] = field(default_factory=dict)
    geo: dict[str, Any] = field(default_factory=dict)
    pfz: dict[str, Any] = field(default_factory=dict)
    route: dict[str, Any] = field(default_factory=dict)
    advisories: list[dict[str, Any]] = field(default_factory=list)
    catalog: list[dict[str, Any]] = field(default_factory=list)
    diagnosis: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class AgentContext:
    """Everything an agent may read or write for one request."""

    message: str
    session_id: str
    plan: Plan
    trace: Trace
    evidence: EvidenceBook
    findings: Findings
    services: "Services"
    location: Location | None = None
    destination: Location | None = None
    window: TimeWindow | None = None
    risk: RiskAssessment | None = None
    layers: list[MapLayer] = field(default_factory=list)
    charts: list[ChartSeries] = field(default_factory=list)
    followups: list[str] = field(default_factory=list)
    language_hint: str = "en"
    history: list[dict[str, str]] = field(default_factory=list)
    #: who the answer is being written for, decided by the planner
    audience: "AudienceCall | None" = None
    #: numbered inline citations, populated by the synthesis agent
    citations: list[Citation] = field(default_factory=list)
    _citer: "Citer | None" = field(default=None, repr=False, compare=False)

    @property
    def citer(self) -> "Citer":
        """Lazily created so every draft builder shares one numbering run."""
        if self._citer is None:
            from .synthesis import Citer

            self._citer = Citer(self.evidence)
        return self._citer

    def reset_citer(self) -> None:
        """Start numbering again. Called before each draft attempt."""
        self._citer = None

    @property
    def intent(self) -> Intent:
        return self.plan.intent

    @property
    def lat(self) -> float:
        return self.location.lat if self.location else 0.0

    @property
    def lon(self) -> float:
        return self.location.lon if self.location else 0.0

    @property
    def start(self) -> datetime:
        assert self.window is not None
        return self.window.start

    @property
    def end(self) -> datetime:
        assert self.window is not None
        return self.window.end

    def note(self, text: str) -> None:
        if text and text not in self.findings.notes:
            self.findings.notes.append(text)

    def add_layer(self, layer: MapLayer) -> None:
        if not any(existing.id == layer.id for existing in self.layers):
            self.layers.append(layer)

    def add_chart(self, chart: ChartSeries) -> None:
        if chart.y and not any(existing.id == chart.id for existing in self.charts):
            self.charts.append(chart)


class Agent(Protocol):
    name: str
    tools: tuple[str, ...]

    async def run(self, ctx: AgentContext) -> None: ...
