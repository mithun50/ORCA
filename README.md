# ORCA

Agentic marine intelligence for Indian waters. Ask a question in English, get an
answer grounded in ISRO, INCOIS and IMD data, with the evidence and the reasoning
attached.

Built for the SIH problem statement on an Agentic AI conversational marine
platform. See `docs/ARCHITECTURE.md` for the diagrams and
`docs/DATA_SOURCES.md` for every endpoint that was probed and what it returned.

## What works right now

- **Intent segregation.** A weighted keyword router classifies the query into one
  of ten intents and names the knowledge sources that can answer it. 52 regex
  rules, additive weights. GLM 5.2 is only consulted when lexical confidence
  drops below 0.55. Verified on all eight sample queries from the problem
  statement.
- **Multi-RAG retrieval.** Three retrievers behind one router: a BM25 document
  index over advisories, a shapely geospatial index for geofencing, and a
  time-series/grid retriever over MOSDAC THREDDS and INCOIS ERDDAP.
- **Live ISRO data.** MOSDAC's public THREDDS server at
  `mosdac.gov.in/live_data` needs no login. ORCA reads the SAC Ocean State
  Forecast for sea temperature, salinity, mixed layer depth, currents and waves.
- **Explainable risk.** The safety verdict comes from explicit threshold rules,
  each recorded with the evidence that triggered it. A model may rephrase the
  answer but cannot change a number or the verdict.
- **Jev System-One for structured judgment.** TypeSafe AI's decision model runs
  alongside the rules: five typed questions (two `choice`, one `score`, two
  `noul`) in one call, answered with calibrated probabilities rather than prose.
  Worst band wins, so Jev can only make a verdict more conservative. Served over
  OpenRouter's Decisions API, with a deterministic offline emulator when no key
  is set.
- **Role-routed models.** Two jobs, two models. Agentic work (intent arbitration)
  goes to GLM 5.2 on OpenRouter; natural language synthesis and regional
  translation go to a Google Gemini model. Either falls back to whatever
  credentials exist, then to templates.
- **Voice, hands-free.** Sarvam Saaras transcribes deck audio behind a 120 Hz
  high-pass filter that suppresses diesel rumble; Sarvam Bulbul speaks the answer
  back in the detected language.
- **n8n multi-agent orchestration.** A 24 node workflow: a supervisor plans, two
  gates decide whether to clarify or skip retrieval, six isolated specialist
  agents retrieve in parallel onto a shared blackboard, a reconciler folds their
  reports and names what failed, and a critic verifies the verdict survived
  before the answer ships.
- **Chat UI.** One static page with the map, the evidence panel and the reasoning
  trace.

## Run it

Two terminals. Python 3.13 and Docker Desktop.

### 1. The API and the UI

```powershell
cd C:\Users\nevil\Desktop\ORCA
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r backend\requirements.txt

cd backend
..\venv\Scripts\python.exe -m uvicorn orca.main:app --host 127.0.0.1 --port 8017
```

Open <http://127.0.0.1:8017/>. First start takes about 20 seconds: it indexes the
knowledge corpus and fetches the India EEZ polygon, which is then cached to disk.

Check it came up cleanly:

```powershell
curl.exe -s --noproxy "*" http://127.0.0.1:8017/health
```

You want to see `knowledge_chunks` above zero, `zone_features: 9` and
`eez_loaded: true`. `llm_routing` shows which model each role resolved to, and
`jev.endpoint` shows where the decision engine will be called.

### 2. n8n

```powershell
cd C:\Users\nevil\Desktop\ORCA
docker compose up -d n8n
```

Open <http://localhost:5678>, then:

1. Workflows -> Import from File -> `n8n/workflows/orca_router.json`
   (the folder is also mounted inside the container at `/workflows`).
2. Activate the workflow.
3. Restart the API with the n8n base set so `/chat?via=n8n` has somewhere to go:

```powershell
$env:ORCA_N8N_BASE = "http://localhost:5678"
..\venv\Scripts\python.exe -m uvicorn orca.main:app --host 127.0.0.1 --port 8017
```

4. In the UI, tick **route via n8n**. The answer comes back with a `via n8n`
   chip and the reasoning trace now spans the whole n8n run, including which
   agents were isolated and whether the critic accepted the wording.

The n8n container reaches the API through `host.docker.internal:8017`, which is
the default for `ORCA_API_BASE` in the workflow. If you would rather run the API
in Docker too, `docker compose --profile full up -d` and set
`ORCA_API_BASE=http://api:8000` in the n8n service environment.

If n8n is not reachable, `/chat?via=n8n` falls back to the in-process
orchestrator and says so in `degraded_sources`. The demo never hard-fails on n8n.

### 3. Verify

```powershell
.\venv\Scripts\python.exe scripts\verify_e2e.py --base http://127.0.0.1:8017
```

Runs all eight sample queries and asserts the intent, the knowledge sources
consulted, that evidence came back, that an official Indian agency source is
present where the query needs one, and that a reasoning trace was recorded.

The unit suite needs two extra packages that are not runtime dependencies:

```powershell
.\venv\Scripts\python.exe -m pip install pytest==9.0.2 pytest-asyncio==1.3.0
.\venv\Scripts\python.exe -m pytest tests -q
```

29 tests, covering the Jev wire protocol, LLM role routing, the Sarvam voice
paths and the API surface. They run offline and make no network calls.

## Optional: the models

Everything works without any of them. With a provider configured, the synthesis
agent rephrases its draft; it cannot introduce a fact.

```powershell
# both roles at once: GLM 5.2 for agentic work, Gemini for synthesis,
# and the same OpenRouter key also serves Jev
$env:ORCA_OPENROUTER_API_KEY = "sk-or-..."
$env:ORCA_GEMINI_API_KEY = "..."

# or pin them explicitly
$env:ORCA_LLM_AGENTIC_PROVIDER = "openrouter"
$env:ORCA_LLM_AGENTIC_MODEL = "z-ai/glm-5.2"
$env:ORCA_LLM_SYNTHESIS_PROVIDER = "gemini"
$env:ORCA_LLM_SYNTHESIS_MODEL = "gemini-3.5-flash"

# voice
$env:ORCA_SARVAM_API_KEY = "..."

# local, no key at all
ollama pull llama3.1:8b
$env:ORCA_LLM_PROVIDER = "auto"   # Ollama is picked up on localhost:11434
```

A role whose provider has no key falls back to whatever credentials do exist,
swapping the model as it swaps the provider, and then to templates. Set
`ORCA_LLM_PROVIDER=none` to disable both roles outright. `/health` reports what
each role resolved to. See `.env.example` for every setting.

## API

| method | path | purpose |
|---|---|---|
| POST | `/chat` | full pipeline, in-process |
| POST | `/chat?via=n8n` | same, routed through the n8n multi-agent workflow |
| POST | `/chat/voice` | Sarvam STT, then the full pipeline, then Sarvam TTS |
| POST | `/api/voice/stt` | transcribe only |
| POST | `/api/voice/tts` | synthesise speech only |
| POST | `/internal/classify` | stage 1: segregate intent, name the knowledge sources |
| POST | `/internal/retrieve/{domain}` | stage 2: retrieve from one source |
| POST | `/internal/synthesize` | stages 3 and 4: risk rules, then the answer |
| GET | `/knowledge-sources` | what each domain is and where it reads from |
| GET | `/health` | index sizes, per-source health, model routing, Jev and voice status |

Valid domains: `ocean`, `weather`, `geospatial`, `hazard`, `advisory`, `catalog`.
`/internal/synthesize` accepts `allow_llm: false` to force deterministic template
synthesis, which is what the n8n critic uses on its repair pass.

```powershell
$body = '{"message":"Is it safe to venture out off Rameswaram tomorrow morning?"}'
$body | Out-File "$env:TEMP\q.json" -Encoding utf8 -NoNewline
curl.exe -s --noproxy "*" -X POST http://127.0.0.1:8017/chat -H "Content-Type: application/json" --data-binary "@$env:TEMP\q.json"
```

## Honest limits

This is a prototype, and the following are real gaps rather than things that are
nearly done:

- **The non-English verdict guard is weak.** Regional language output is
  implemented: the script of the query selects the language and Gemini translates
  the draft. But the guard that checks an "unsafe" verdict survived the rewrite
  only understands English markers; for any other language it just checks the
  text is longer than 10 characters. A translated rewrite could therefore soften
  a "do not venture" verdict. The n8n critic blocks this on its own path by
  refusing to approve any non-English rewrite of an unsafe verdict, but plain
  `POST /chat` is still exposed. Fix this before anyone relies on it.
- **GDACS is down, so cyclone distance never fires.** As of 2026-09-22 the
  endpoint returns an empty body and JSON parsing fails. The connector degrades
  correctly and reports `gdacs` in `degraded_sources`, but the
  `tropical_cyclone_distance` rule cannot fire and nothing compensates.
- **Jev duplicates the threshold rules.** It judges the same wave, wind and
  cyclone numbers the rules already used, so the two can disagree inside one
  response. Worst-band-wins keeps the verdict conservative, but the findings list
  can read as self-contradictory. `risk.score` also saturates at 100 on any
  single unsafe finding, so read `band` and `findings` instead.
- **The offline voice path is a placeholder, not a demo.** With no Sarvam key,
  STT returns a fixed sample sentence whatever the audio contains, so
  `/chat/voice` answers a question nobody asked. TTS returns a 440 Hz beep, it
  runs on every text request as well as voice ones, and it was 69.5% of a 122 kB
  response. Both the beep generator and the acoustic filter are per-sample Python
  loops inside async handlers.
- **Zone geometry is approximate.** `data/layers/marine_zones.geojson` is a
  hand-built envelope set for the demo. It is indicative, not survey grade, and
  must not be used for navigation. Production needs gazetted MoEFCC notifications
  and NHO chart data. The modelled boundary-breach probability also uses the
  nearest zone of any kind, so a benign marine park a few km away inflates it.
- **PFZ answers are physical context, not the official advisory.** INCOIS
  publishes the daily PFZ advisory as per-state bulletins for human readers.
  ORCA reports what the official SST, chlorophyll and mixed layer grids say and
  points the user at the bulletin. It does not claim to reproduce the advisory.
- **Lightning is a proxy.** IMD's Damini network has no public API. ORCA computes
  a CAPE and rainfall based likelihood and labels it as derived every time.
- **The MOSDAC OSF_WAVE aggregate serves a stale cycle.** Its time axis is fixed
  at `hours since 2026-4-21`. ORCA reads the real validity window from
  `dataset.xml`, marks the value stale, and quotes a live fallback beside it.
- **No authentication.** The API is open and `host` defaults to `0.0.0.0`. Fine
  on localhost, not acceptable exposed. Put auth in front of it and narrow
  `ORCA_CORS_ORIGINS` first. CORS does not restrain non-browser clients.
- **Only the periphery is tested.** 29 tests cover voice, the Jev wire protocol,
  LLM role routing and the API surface. Nothing covers the intent router, BM25,
  the geofence index, the grid readers or the risk rules. There is no CI.
- **Route scoring is sea state and zone conflicts only.** No bathymetry, no
  shoals, no traffic separation, no vessel characteristics.
- **Conversation memory is in-process.** Restarting the API clears sessions.
