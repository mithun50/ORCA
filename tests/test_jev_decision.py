"""Tests for TypeSafe AI Jev Decision Engine and Risk Agent Integration."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from orca.jev import JevDecisionEngine, get_jev_engine, JevSafetyDecision
from orca.schemas import RiskBand


@pytest.mark.asyncio
async def test_jev_emulator_calm_conditions():
    """Verify Jev decision emulator returns SAFE and VENTURE_PERMITTED in calm seas."""
    engine = JevDecisionEngine()
    engine.settings.jev_api_key = ""  # Force emulator

    decision = await engine.evaluate_safety(
        location_name="Chennai",
        swh_m=0.8,
        wind_kt=10.0,
        gust_kt=14.0,
        cyclone_alert=False,
        nearest_boundary_dist_km=25.0,
    )

    assert decision.safety_verdict == "SAFE"
    assert decision.action == "VENTURE_PERMITTED"
    assert decision.risk_score < 35.0
    assert decision.breach_probability < 0.1
    assert decision.capsizing_probability < 0.1
    assert decision.engine == "jev-rule-emulator"


@pytest.mark.asyncio
async def test_jev_emulator_dangerous_waves():
    """Verify Jev decision emulator returns UNSAFE and STAY_IN_HARBOR with waves >= 2.5m."""
    engine = JevDecisionEngine()
    engine.settings.jev_api_key = ""

    decision = await engine.evaluate_safety(
        location_name="Rameswaram",
        swh_m=3.1,
        wind_kt=28.0,
        gust_kt=35.0,
        cyclone_alert=False,
    )

    assert decision.safety_verdict == "UNSAFE"
    assert decision.action == "STAY_IN_HARBOR"
    assert decision.risk_score >= 65.0
    assert decision.capsizing_probability >= 0.70


@pytest.mark.asyncio
async def test_jev_emulator_boundary_breach():
    """Verify Jev decision flags high breach risk near international boundary."""
    engine = JevDecisionEngine()
    engine.settings.jev_api_key = ""

    decision = await engine.evaluate_safety(
        location_name="Palk Bay",
        swh_m=1.0,
        wind_kt=12.0,
        gust_kt=15.0,
        nearest_boundary_dist_km=0.8,
        restricted_zone_name="IMBL Palk Strait",
    )

    assert decision.safety_verdict == "UNSAFE"
    assert decision.breach_probability >= 0.80
    assert decision.action == "STAY_IN_HARBOR"


@pytest.mark.asyncio
async def test_jev_cloud_api_query():
    """Verify Jev cloud API payload and response parsing."""
    engine = JevDecisionEngine()
    engine.settings.jev_api_key = "typesafe-test-key"

    mock_resp_data = {
        "answers": {
            "safety_verdict": {"choice": "CAUTION", "confidence": 0.94},
            "action": {"choice": "EXERCISE_VIGILANCE", "confidence": 0.92},
            "risk_score": {"score": 45.0},
            "breach_probability": {"noul": 0.12},
            "capsizing_probability": {"noul": 0.38}
        }
    }

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.json = MagicMock(return_value=mock_resp_data)

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)) as mock_post:
        decision = await engine.evaluate_safety(
            location_name="Mangalore",
            swh_m=1.8,
            wind_kt=18.0,
            gust_kt=22.0,
        )

        assert decision.engine == "jev-cloud"
        assert decision.safety_verdict == "CAUTION"
        assert decision.action == "EXERCISE_VIGILANCE"
        assert decision.risk_score == 45.0
        assert decision.capsizing_probability == 0.38

        call_kwargs = mock_post.call_args.kwargs
        assert "Authorization" in call_kwargs["headers"]
        assert call_kwargs["headers"]["Authorization"] == "Bearer typesafe-test-key"
        assert call_kwargs["json"]["model"] == "jev-latest"
