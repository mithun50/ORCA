"""Runtime configuration for ORCA.

Everything has a working default so the prototype boots with no .env at all.
Optional integrations (LLM, Qdrant, n8n, IMD key) switch themselves off when
their settings are absent rather than raising.
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_prefix="ORCA_",
        extra="ignore",
    )

    # --- service ---
    app_name: str = "ORCA Marine Intelligence"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- upstream data services (verified 2026-09-21, see docs/DATA_SOURCES.md) ---
    mosdac_base: str = "https://mosdac.gov.in/live_data"
    incois_erddap_base: str = "https://erddap.incois.gov.in/erddap"
    incois_pfz_page: str = "https://incois.gov.in/MarineFisheries/PfzAdvisory"
    imd_rsmc_url: str = "https://rsmcnewdelhi.imd.gov.in/"
    imd_subdivision_warning_url: str = (
        "https://mausam.imd.gov.in/imd_latest/contents/subdivisionwise-warning.php"
    )
    imd_api_base: str = "https://mausam.imd.gov.in/api"
    imd_api_key: str = ""  # IMD issues these on request; 401 without one
    openmeteo_marine_base: str = "https://marine-api.open-meteo.com/v1/marine"
    openmeteo_forecast_base: str = "https://api.open-meteo.com/v1/forecast"
    gdacs_base: str = "https://www.gdacs.org/gdcsapi/api/events/geteventlist/SEARCH"
    marine_regions_wfs: str = "https://geo.vliz.be/geoserver/MarineRegions/wfs"
    nasa_cmr_base: str = "https://cmr.earthdata.nasa.gov/search"

    # --- http behaviour ---
    http_timeout_s: float = 45.0
    http_retries: int = 2
    user_agent: str = (
        "ORCA-MarineIntelligence/0.1 (SIH prototype; contact: orca@example.org)"
    )
    cache_ttl_s: int = 900
    cache_dir: str = str(REPO_ROOT / ".cache")

    # --- LLM (all optional) ---
    # auto | openrouter | gemini | openai | ollama | none
    llm_provider: str = "auto"
    llm_model: str = "llama3.1:8b"
    ollama_base: str = "http://localhost:11434"
    openai_api_key: str = ""
    openai_base: str = "https://api.openai.com/v1"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    openrouter_api_key: str = ""
    openrouter_base: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "anthropic/claude-3.5-sonnet"
    llm_timeout_s: float = 60.0

    # --- Jev / TypeSafe AI Decision Engine (optional) ---
    jev_api_key: str = ""
    jev_base: str = "https://api.typesafe.ai/v1"
    jev_model: str = "jev-latest"

    # --- Sarvam AI Voice Engine: STT & TTS (optional) ---
    sarvam_api_key: str = ""
    sarvam_base: str = "https://api.sarvam.ai"
    sarvam_stt_model: str = "saaras:v3"
    sarvam_tts_model: str = "bulbul:v1"
    sarvam_speaker: str = "meera"

    # --- retrieval ---
    qdrant_url: str = ""
    qdrant_collection: str = "orca_marine"
    knowledge_dir: str = str(REPO_ROOT / "knowledge")
    layers_dir: str = str(REPO_ROOT / "data" / "layers")

    # --- orchestration ---
    n8n_base: str = ""  # e.g. http://localhost:5678
    n8n_webhook_token: str = ""

    # --- risk thresholds (INCOIS/IMD small-craft advisory style) ---
    swh_caution_m: float = 1.5
    swh_danger_m: float = 2.5
    wind_caution_kt: float = 17.0  # ~ Beaufort 5
    wind_danger_kt: float = 27.0  # ~ Beaufort 7
    gust_danger_kt: float = 34.0
    current_caution_cms: float = 50.0
    stale_hours: int = 48

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def effective_gemini_api_key(self) -> str:
        return self.gemini_api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")

    @property
    def effective_openrouter_api_key(self) -> str:
        return self.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")

    @property
    def effective_sarvam_api_key(self) -> str:
        return self.sarvam_api_key or os.environ.get("SARVAM_API_KEY", "")

    @property
    def effective_jev_api_key(self) -> str:
        return self.jev_api_key or os.environ.get("JEV_API_KEY", "") or os.environ.get("TYPESAFE_API_KEY", "")


@lru_cache
def get_settings() -> Settings:
    return Settings()
