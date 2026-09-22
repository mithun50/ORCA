# ORCA data source inventory (verified 2026-09-21)

Every row below was probed with `curl` from this machine. "Verified" means the exact
URL shown returned real data, not that the page merely exists.

## Tier 1 - ISRO / MOSDAC (primary)

MOSDAC runs a public, unauthenticated THREDDS Data Server. The web root is
`https://mosdac.gov.in/live_data` (the catalog XML advertises `/thredds/...`
base paths, but the deployed prefix is `/live_data`). Services confirmed:

| Service | URL pattern | Status |
|---|---|---|
| Catalog | `/live_data/catalog/<path>/catalog.xml` | verified |
| NCSS grid metadata | `/live_data/ncss/grid/<ds>/dataset.xml` | verified |
| NCSS point (CSV) | `/live_data/ncss/grid/<ds>?var=..&latitude=..&longitude=..&accept=csv` | verified |
| NCSS grid (netCDF only) | `accept=netcdf` (csv rejected with HTTP 400) | verified |
| OPeNDAP | `/live_data/dodsC/<ds>.das` `.dds` `.ascii?var[i][j]` | verified |
| WMS | `/live_data/wms/<ds>?service=WMS&request=GetCapabilities` | verified |

### 1. Ocean State Forecast - circulation (SAC / MOSDAC)

- Catalog: `OSF_CIRC`, daily files `SAC_OSF_CIRC_10KM_YYYYMMDD.nc` (~3.8 GB each,
  subset server-side, never downloaded whole).
- Variables: `temp` (potential temperature, deg C), `salinity` (psu),
  `hmxl` (mixed layer depth, cm), `eastward_ocean_wave_current`,
  `northward_ocean_wave_current` (cm/s).
- Grid: stretched, `xt_i[560]` from 27.5E, `yt_j[400]` from 34.5S, ~0.1 deg over
  the Indian EEZ. 21 time steps, 6-hourly, 5 day horizon.
- Live check: `temp` at 13.0N 81.5E for 2026-09-21T00Z returned `28.977 degC`,
  eastward current `-7.26 cm/s`.
- This is ORCA's authoritative SST + currents + MLD source.

### 2. Ocean State Forecast - waves (SAC / MOSDAC)

- Catalog: `OSF_WAVE`, single aggregate `SAC_OSF_WAVE_10KM.nc`.
- Variables: `SWH`, `MWPER` (mean wave period Te), `MWDIR`, `HS01`/`DIR01`
  (primary swell), `HS02`/`DIR02` (secondary swell), `UWIND`, `VWIND`.
- Global 0-359E, 70S-70N, 6-hourly, 21 steps.
- Live check: `SWH` at 13.08N 80.35E returned `0.852 m`.
- Caveat: the aggregate's time axis is `hours since 2026-4-21`; the file content
  is a fixed forecast cycle, not a rolling one. ORCA reads the advertised
  TimeSpan from `dataset.xml`, labels the data with its true validity window,
  and marks it stale when the cycle is older than 48 h. When stale, the wave
  agent falls back to Tier 4 and says so in the evidence panel.

### 3. PFZ input grids (MOSDAC)

- `pfz/sst/pfz_sst_YYYYMMDD.nc` (latest 2024-12-13),
  `pfz/chl/pfz_chl_YYYYMMDD.nc` (latest 2024-01-20),
  `pfz/PFZ_possibility.xml`.
- Archival, used for the "why has productivity declined" historical reasoning and
  for demonstrating the PFZ front-detection algorithm on official ISRO grids.

### 4. Other MOSDAC catalogs available

`GSMAP_ISRO_RAIN` (hourly rainfall, archive to 2025-10-31),
`liveNewScatAnalyzedWind6hr` / `liveNewScatAnalyzedWind625` (EOS-06 / Oceansat-3
scatterometer analysed winds, 2026 folder present),
`Cyclone3DL1BSTD{1,4,8}km`, `CycloneArchiveImages`, `CycloneSCATARBImages`
(INSAT-3D/3DR/3S cyclone imagery), `ISW` (internal solitary waves),
`swot`, `live3{D,R,S}L2BSST` (INSAT L2B SST scans).

Known broken: `testArchive/3RIMG_AGG_L2B_SST.h5` and
`testArchive/EOS06SCAT_AGG_L4AW.nc` return HTTP 500 from NCSS.

## Tier 2 - INCOIS / MoES

| Source | URL | Status |
|---|---|---|
| INCOIS ERDDAP | `https://erddap.incois.gov.in/erddap` | verified, 16 griddap + 2 tabledap datasets |
| PFZ advisory page | `https://incois.gov.in/MarineFisheries/PfzAdvisory` | HTTP 200, JS-rendered |
| SARAT | `https://sarat.incois.gov.in/` | HTTP 200 |
| Live Access Server | `https://las.incois.gov.in/las/getUI.do` | HTTP 200 |
| INCOIS GeoServer | `https://incois.gov.in/geoserver/ows` | HTTP 403 - WAF blocks non-browser clients |

INCOIS ERDDAP datasets of interest: `incois_oceansat2_datasets` (Oceansat-2 OCM),
`IRS_chlorophyll_datasets` (IRS-P4 OCM chlorophyll), `incois_argo_sst_weekly`,
`incois_argo_10d_VAM` / `_McCreary` (T/S profiles), `Indian_ARGO_Floats`
(tabledap), `ascat_daily_datasets`, `NOAA_AVHRR_AMSR_datasets`,
`incois_valueadded_products_datasets`.

The GeoServer 403 is a WAF decision, not an outage. ORCA does not scrape around
it; the geospatial layers it needs (EEZ, IMBL) come from Tier 4 instead, and PFZ
advisory text is ingested as documents.

## Tier 3 - IMD

| Source | URL | Status |
|---|---|---|
| RSMC New Delhi (cyclone) | `https://rsmcnewdelhi.imd.gov.in/` | HTTP 200 HTML |
| Sub-division warnings | `https://mausam.imd.gov.in/imd_latest/contents/subdivisionwise-warning.php` | HTTP 200 HTML |
| NWP products | `https://nwp.imd.gov.in/` | HTTP 200 |
| `mausam.imd.gov.in/api/*` | `nowcast_district_api.php`, `current_wx_api.php`, `warnings_district_api.php` | HTTP 401 - needs an IMD-issued key |
| `city.imd.gov.in/api/cityweather.php` | | HTTP 401 |

IMD's JSON APIs require credentials that are issued on request. ORCA has a
connector for them behind `IMD_API_KEY`; with no key it parses the two public
HTML pages and degrades gracefully.

## Tier 4 - Gap fill (clearly labelled as non-ISRO in every response)

| Source | URL | Status | Used for |
|---|---|---|---|
| Open-Meteo Marine | `https://marine-api.open-meteo.com/v1/marine` | verified | live SWH, swell, `sea_level_height_msl` (tide), sea surface temp |
| Open-Meteo Forecast | `https://api.open-meteo.com/v1/forecast` | verified | wind, gusts, precip, CAPE, visibility |
| GDACS | `https://www.gdacs.org/gdcsapi/api/events/geteventlist/SEARCH?eventlist=TC` | **failing as of 2026-09-22** | tropical cyclone alert polygons |
| Marine Regions WFS | `https://geo.vliz.be/geoserver/MarineRegions/wfs` | verified (2.2 MB GeoJSON for India EEZ) | EEZ polygon, IMBL geofencing |
| NASA CMR | `https://cmr.earthdata.nasa.gov/search/collections.json` | verified | dataset discovery for the discovery agent |
| NOAA CoastWatch ERDDAP | `https://coastwatch.noaa.gov/erddap` | verified | science-quality chlorophyll cross-check |

The GDACS regression was found on 2026-09-22: the endpoint answers but the body
is empty, so JSON parsing fails with `Expecting value: line 3 column 1 (char 4)`.
The connector returns an empty event list, `gdacs` appears in
`degraded_sources`, and the `tropical_cyclone_distance` risk rule therefore
cannot fire. Nothing else compensates for it, which makes this the most
consequential open data gap in the system.

## AI services (not data sources, but external dependencies)

All optional. ORCA answers every supported query with none of them configured.

| Service | Endpoint | Role | Without a key |
|---|---|---|---|
| GLM 5.2 via OpenRouter | `POST https://openrouter.ai/api/v1/chat/completions`, model `z-ai/glm-5.2` | agentic: intent arbitration when lexical confidence < 0.55 | the lexical verdict stands |
| Google Gemini | `POST generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent` | synthesis: wording and regional language translation | the deterministic template draft is the answer |
| Jev System-One via OpenRouter | `POST https://openrouter.ai/api/alpha/decisions`, model `typesafe/jev-1.13` | typed safety decisions: choice, score, noul | offline rule emulator, labelled `jev-rule-emulator` |
| Jev System-One direct | `POST https://api.typesafe.ai/v1/systemone`, model `jev-latest` | same, alternate transport | same |
| Sarvam Saaras | `POST https://api.sarvam.ai/speech-to-text` | STT for coastal Indian languages | canned sample transcript, labelled `mock-saaras-offline` |
| Sarvam Bulbul | `POST https://api.sarvam.ai/text-to-speech` | TTS in the detected language | synthetic beep WAV, labelled `mock-bulbul-offline` |

Notes on Jev, since it is the least familiar of these:

- It is a decision model, not a chat model. One `state` plus a map of typed
  `questions` in, typed `answers` keyed by the same ids out. It does not generate
  prose, and chat-completions SDKs do not work against it.
- On OpenRouter it is served by the **Decisions API** at `/api/alpha/decisions`,
  which is a different endpoint from the OpenAI-compatible chat one. The wire
  protocol is near-identical to TypeSafe's own `/v1/systemone`. Note that
  OpenRouter only accepts the versioned slug `typesafe/jev-1.13`; the
  `typesafe/jev-latest` alias returns
  `400 Model typesafe/jev-latest does not exist` there, though TypeSafe's own
  endpoint accepts it. Verified live on 2026-09-22, which answered as
  `typesafe/jev-1.13-20260917`.
- `choice` takes `criteria` as a map of option to rubric text and answers with a
  full probability distribution plus `confidence`. `score` takes `criteria` as an
  ordered array of 2 to 10 level descriptions and answers with a
  probability-weighted float over the level indices, so it can land between
  levels. `noul` answers with a single yes-probability and carries **no**
  confidence field.
- TypeSafe publishes $0.042 per million input tokens with output free, a 32k
  context on the OpenRouter listing, and 70 to 500 ms latency, which is why ORCA
  can afford to call it on every safety verdict.
- Documented retryable statuses are 429 and 529. ORCA does not back off and
  retry inside a request; it falls through to the emulator so the user is not
  left waiting.

## Gaps with no public machine endpoint

- **Lightning**: IMD's Damini network has no public API. ORCA computes a
  convective-risk proxy from CAPE + precipitation + cloud-top proxies and labels
  it as a proxy, never as an observed strike.
- **Official daily PFZ advisory as data**: INCOIS publishes it as
  per-district bulletins behind a JS front end. ORCA ingests the bulletin text
  into the vector RAG and separately derives PFZ *candidates* from SST fronts
  and chlorophyll. Derived candidates are always tagged
  `ORCA-derived, not an official INCOIS advisory`.
- **MPA / ecologically sensitive zone boundaries**: seeded from a small
  hand-built GeoJSON of major Indian marine protected areas for the prototype.
