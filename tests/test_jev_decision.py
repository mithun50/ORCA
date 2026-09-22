"""Tests for the Jev System-One decision engine and its risk-agent integration."""

from __future__ import annotations

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from orca.jev import JevDecisionEngine, get_jev_engine, JevSafetyDecision
from orca.schemas import RiskBand


def _offline_engine() -> JevDecisionEngine:
    """Engine with no credentials, so the offline emulator must run.

    `conftest.hermetic_settings` already blanks the credentials; this also
    asserts it, because a Jev test that silently reached the network would be
    both slow and billable.
    """
    engine = JevDecisionEngine()
    assert not engine.is_configured
    return engine


@pytest.mark.asyncio
async def test_jev_emulator_calm_conditions():
    """Verify Jev decision emulator returns SAFE and VENTURE_PERMITTED in calm seas."""
    engine = _offline_engine()

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
    engine = _offline_engine()

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
    engine = _offline_engine()

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
async def test_jev_openrouter_decisions_call():
    """Verify the Jev System-One request shape and typed answer parsing."""
    engine = JevDecisionEngine()
    engine.settings.jev_api_key = "sk-or-test-key"
    engine.settings.jev_provider = "openrouter"
    engine.settings.jev_base = "https://openrouter.ai/api/alpha"
    engine.settings.jev_model = "typesafe/jev-latest"

    # Shape per https://docs.typesafe.ai/api : choice carries probabilities and
    # confidence, score is a weighted float over level indices, noul carries
    # neither a confidence nor a type-specific extra.
    mock_resp_data = {
        "model": "jev-1.13.0",
        "answers": {
            "safety_verdict": {
                "type": "choice",
                "choice": "CAUTION",
                "probabilities": {"SAFE": 0.11, "CAUTION": 0.78, "UNSAFE": 0.11},
                "confidence": 0.81,
            },
            "action": {
                "type": "choice",
                "choice": "EXERCISE_VIGILANCE",
                "probabilities": {
                    "VENTURE_PERMITTED": 0.2,
                    "EXERCISE_VIGILANCE": 0.7,
                    "STAY_IN_HARBOR": 0.1,
                },
                "confidence": 0.74,
            },
            "severity": {
                "type": "score",
                "score": 2.0,  # level 2 of 0..4 -> 50/100
                "legend": {"0": "Benign", "1": "Marginal", "2": "Hazardous",
                           "3": "Dangerous", "4": "Extreme"},
                "probabilities": {"0": 0.0, "1": 0.1, "2": 0.8, "3": 0.1, "4": 0.0},
                "confidence": 0.88,
            },
            "breach_probability": {"type": "noul", "noul": 0.12},
            "capsizing_probability": {"type": "noul", "noul": 0.38},
        },
        "usage": {"input_tokens": 318, "output_tokens": 34},
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

        assert decision.engine == "jev-openrouter"
        assert decision.safety_verdict == "CAUTION"
        assert decision.action == "EXERCISE_VIGILANCE"
        assert decision.risk_score == 50.0  # score 2 of span 4
        assert decision.breach_probability == 0.12
        assert decision.capsizing_probability == 0.38
        assert decision.confidence == 0.81
        assert decision.model == "jev-1.13.0"
        assert decision.verdict_probabilities["CAUTION"] == 0.78

        # OpenRouter Decisions endpoint, not chat completions
        assert mock_post.call_args.args[0] == (
            "https://openrouter.ai/api/alpha/decisions"
        )
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["headers"]["Authorization"] == "Bearer sk-or-test-key"
        body = call_kwargs["json"]
        assert body["model"] == "typesafe/jev-latest"
        assert "messages" not in body  # System One, not a chat request
        assert isinstance(body["state"], dict)

        questions = body["questions"]
        assert questions["safety_verdict"]["type"] == "choice"
        # choice criteria is a map of option -> rubric, not an options list
        assert set(questions["safety_verdict"]["criteria"]) == {
            "SAFE", "CAUTION", "UNSAFE"
        }
        assert "options" not in questions["safety_verdict"]
        # score criteria is an ordered array of level descriptions
        assert isinstance(questions["severity"]["criteria"], list)
        assert 2 <= len(questions["severity"]["criteria"]) <= 10
        assert questions["breach_probability"]["type"] == "noul"


@pytest.mark.asyncio
async def test_jev_typesafe_transport_endpoint():
    """The typesafe transport posts to /systemone with the bare alias."""
    engine = JevDecisionEngine()
    engine.settings.jev_api_key = "ts-test-key"
    engine.settings.jev_provider = "typesafe"
    engine.settings.jev_base = "https://api.typesafe.ai/v1"
    engine.settings.jev_model = "jev-latest"

    assert engine.endpoint == "https://api.typesafe.ai/v1/systemone"
    assert engine.model == "jev-latest"


@pytest.mark.asyncio
async def test_jev_off_schema_answer_falls_back_to_emulator():
    """An answer outside the declared option set must not reach the risk agent."""
    engine = JevDecisionEngine()
    engine.settings.jev_api_key = "sk-or-test-key"
    engine.settings.jev_provider = "openrouter"

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.json = MagicMock(
        return_value={
            "model": "jev-1.13.0",
            "answers": {
                "safety_verdict": {"type": "choice", "choice": "PROBABLY_FINE"},
                "action": {"type": "choice", "choice": "EXERCISE_VIGILANCE"},
            },
        }
    )

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
        decision = await engine.evaluate_safety(
            location_name="Kochi", swh_m=3.2, wind_kt=30.0, gust_kt=40.0
        )

    assert decision.engine == "jev-rule-emulator"
    assert decision.safety_verdict == "UNSAFE"


@pytest.mark.asyncio
async def test_jev_http_error_falls_back_to_emulator():
    """A 429 or any non-200 must degrade to the emulator, not raise."""
    engine = JevDecisionEngine()
    engine.settings.jev_api_key = "sk-or-test-key"

    mock_resp = AsyncMock()
    mock_resp.status_code = 429
    mock_resp.text = "rate limited"

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
        decision = await engine.evaluate_safety(
            location_name="Chennai", swh_m=0.7, wind_kt=9.0, gust_kt=12.0
        )

    assert decision.engine == "jev-rule-emulator"
    assert decision.safety_verdict == "SAFE"
