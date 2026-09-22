"""Tests for the numbers-only briefing.

The briefing is a second layout of data the agents already retrieved, never a
second opinion, so the contract that matters is: it agrees with the risk
assessment, every figure carries a source, and a gap is reported rather than
omitted.
"""

from __future__ import annotations

import pytest
import httpx

from orca.main import app


BRIEF_QUERY = "Is it safe to venture out off Rameswaram tomorrow morning?"


async def _brief(message: str = BRIEF_QUERY) -> dict:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/chat", json={"message": message})
        assert resp.status_code == 200
        return resp.json()


@pytest.mark.asyncio
async def test_briefing_has_the_four_groups_and_a_risk_band():
    data = await _brief()
    brief = data["briefing"]

    for group in ("ocean", "weather", "gis", "satellite"):
        assert group in brief, f"{group} group missing from the briefing"
        assert isinstance(brief[group], list)

    assert brief["risk_band"] in ("safe", "caution", "unsafe", "unknown")
    # the briefing must not contradict the assessment it summarises
    assert brief["risk_band"] == data["risk"]["band"]
    assert brief["risk_score"] == data["risk"]["score"]


@pytest.mark.asyncio
async def test_briefing_is_numeric_and_sourced():
    data = await _brief()
    brief = data["briefing"]

    rows = brief["ocean"] + brief["weather"]
    assert rows, "a safety question must produce ocean or weather figures"

    for item in rows:
        assert item["label"], "every row needs a label"
        assert item["value"] != "", f"{item['label']} has no value"
        # a number without a source is exactly what the briefing exists to avoid
        assert item["tier"] or item["marker"] or item["unit"] == "", (
            f"{item['label']} carries neither a tier nor a citation"
        )
        assert item["flag"] in ("", "watch", "danger")


@pytest.mark.asyncio
async def test_briefing_drivers_match_the_rules_that_fired():
    """Whatever pushed the band must be named, and nothing else."""
    data = await _brief()
    brief = data["briefing"]
    bad = [
        f for f in data["risk"]["findings"]
        if f["band"] in ("unsafe", "caution") and not f["rule"].startswith("jev_")
    ]
    assert len(brief["drivers"]) == len(bad)
    if brief["risk_band"] in ("unsafe", "caution"):
        assert brief["drivers"], "a non-safe verdict must say what drove it"


@pytest.mark.asyncio
async def test_briefing_reports_position_in_gis():
    data = await _brief()
    labels = [i["label"] for i in data["briefing"]["gis"]]
    assert "position" in labels, "GIS must state where the answer is for"


@pytest.mark.asyncio
async def test_briefing_is_empty_for_small_talk():
    """No retrieval means no figures, and no invented ones either."""
    data = await _brief("hello")
    brief = data["briefing"]
    assert not brief["ocean"]
    assert not brief["weather"]
