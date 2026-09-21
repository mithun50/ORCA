# ORCA

Agentic marine intelligence for Indian waters. Ask a question in English, get an
answer grounded in ISRO, INCOIS and IMD data, with the evidence and the reasoning
attached.

Built for the SIH problem statement on an Agentic AI conversational marine
platform. See `docs/ARCHITECTURE.md` for the diagrams and
`docs/DATA_SOURCES.md` for every endpoint that was probed and what it returned.

## What works right now

- **Intent segregation.** A weighted keyword router classifies the query into one
  of ten intents and names the knowledge sources that can answer it. The LLM is
  only consulted when lexical confidence drops below 0.55. Verified on all eight
  sample queries from the problem statement.
- **Multi-RAG retrieval.** Three retrievers behind one router: a BM25 document
  index over advisories, a shapely geospatial index for geofencing, and a
  time-series/grid retriever over MOSDAC THREDDS and INCOIS ERDDAP.
- **Live ISRO data.** MOSDAC's public THREDDS server at
  `mosdac.gov.in/live_data` needs no login. ORCA reads the SAC Ocean State
  Forecast for sea temperature, salinity, mixed layer depth, currents and waves.
- **Explainable risk.** The safety verdict comes from explicit threshold rules,
  each recorded with the evidence that triggered it. The LLM may rephrase the
  answer but cannot change a number or the verdict.
- **n8n as the router.** The same pipeline broken into stages that an n8n
  workflow drives: classify, fan out per knowledge source, retrieve, merge,
  synthesise.
- **Chat UI.** One static page with the map, the evidence panel and the reasoning
  trace.

## Run it

Two terminals. Python 3.13 and Docker Desktop.

### 1. The API and the UI

```powershell
cd C:\Projects\ORCA
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt

cd backend
..\.venv\Scripts\python.exe -m uvicorn orca.main:app --host 127.0.0.1 --port 8017
```

Open <http://127.0.0.1:8017/>. First start takes about 20 seconds: it indexes the
knowledge corpus and fetches the India EEZ polygon, which is then cached to disk.

Check it came up cleanly:

```powershell
curl.exe -s http://127.0.0.1:8017/health
```

You want to see `knowledge_chunks` above zero, `zone_features: 9` and
`eez_loaded: true`.

### 2. n8n

```powershell
cd C:\Projects\ORCA
docker compose up -d n8n
```

Open <http://localhost:5678>, then:

1. Workflows -> Import from File -> `n8n/workflows/orca_router.json`
   (the folder is also mounted inside the container at `/workflows`).
2. Activate the workflow.
3. Restart the API with the n8n base set so `/chat?via=n8n` has somewhere to go:

```powershell
$env:ORCA_N8N_BASE = "http://localhost:5678"
..\.venv\Scripts\python.exe -m uvicorn orca.main:app --host 127.0.0.1 --port 8017
```

4. In the UI, tick **route via n8n**. The answer comes back with a `via n8n`
   chip and the reasoning trace now spans the whole n8n run.

The n8n container reaches the API through `host.docker.internal:8017`, which is
the default for `ORCA_API_BASE` in the workflow. If you would rather run the API
in Docker too, `docker compose --profile full up -d` and set
`ORCA_API_BASE=http://api:8000` in the n8n service environment.

If n8n is not reachable, `/chat?via=n8n` falls back to the in-process
orchestrator and says so in `degraded_sources`. The demo never hard-fails on n8n.

### 3. Verify

```powershell
.\.venv\Scripts\python.exe scripts\verify_e2e.py --base http://127.0.0.1:8017
```

Runs all eight sample queries and asserts the intent, the knowledge sources
consulted, that evidence came back, that an official Indian agency source is
present where the query needs one, and that a reasoning trace was recorded.

## Optional: natural phrasing with an LLM

Everything works without one. With a provider configured, the synthesis agent
rephrases its draft; it cannot introduce a fact.

```powershell
# local, no key
ollama pull llama3.1:8b
$env:ORCA_LLM_MODEL = "llama3.1:8b"   # auto-detected on localhost:11434

# or hosted
$env:ORCA_LLM_PROVIDER = "gemini"
$env:ORCA_GEMINI_API_KEY = "..."
$env:ORCA_LLM_MODEL = "gemini-2.0-flash"
```

## API

| method | path | purpose |
|---|---|---|
| POST | `/chat` | full pipeline, in-process |
| POST | `/chat?via=n8n` | same, routed through the n8n workflow |
| POST | `/internal/classify` | stage 1: segregate intent, name the knowledge sources |
| POST | `/internal/retrieve/{domain}` | stage 2: retrieve from one source |
| POST | `/internal/synthesize` | stages 3 and 4: risk rules, then the answer |
| GET | `/knowledge-sources` | what each domain is and where it reads from |
| GET | `/health` | index sizes, per-source health, LLM provider |

Valid domains: `ocean`, `weather`, `geospatial`, `hazard`, `advisory`, `catalog`.

```powershell
$body = '{"message":"Is it safe to venture out off Rameswaram tomorrow morning?"}'
$body | Out-File "$env:TEMP\q.json" -Encoding utf8 -NoNewline
curl.exe -s -X POST http://127.0.0.1:8017/chat -H "Content-Type: application/json" --data-binary "@$env:TEMP\q.json"
```

## Honest limits

This is a prototype, and the following are real gaps rather than things that are
nearly done:

- **English only.** Language detection is wired in and reports a hint, but
  responses are English. Indian regional language output is not implemented.
- **Zone geometry is approximate.** `data/layers/marine_zones.geojson` is a
  hand-built envelope set for the demo. It is indicative, not survey grade, and
  must not be used for navigation. Production needs gazetted MoEFCC notifications
  and NHO chart data.
- **PFZ answers are physical context, not the official advisory.** INCOIS
  publishes the daily PFZ advisory as per-state bulletins for human readers.
  ORCA reports what the official SST, chlorophyll and mixed layer grids say and
  points the user at the bulletin. It does not claim to reproduce the advisory.
- **Lightning is a proxy.** IMD's Damini network has no public API. ORCA computes
  a CAPE and rainfall based likelihood and labels it as derived every time.
- **The MOSDAC OSF_WAVE aggregate serves a stale cycle.** Its time axis is fixed
  at `hours since 2026-4-21`. ORCA reads the real validity window from
  `dataset.xml`, marks the value stale, and quotes a live fallback beside it.
- **No authentication.** The API is open. Fine on localhost, not acceptable
  exposed. Put auth in front of it and narrow `ORCA_CORS_ORIGINS` first.
- **Route scoring is sea state and zone conflicts only.** No bathymetry, no
  shoals, no traffic separation, no vessel characteristics.
- **Conversation memory is in-process.** Restarting the API clears sessions.
