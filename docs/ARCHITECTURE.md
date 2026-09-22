# ORCA architecture

ORCA answers marine questions in natural language by segregating the query,
routing it to the knowledge sources that can actually answer it, retrieving from
official Indian agency services, applying explicit safety rules, and returning
the answer together with the evidence and the reasoning behind it.

Every number in the diagrams below was read out of the running system on
2026-09-22 (`/health`, a live `POST /chat`, and the 21-test suite). Where the
code and an earlier version of this document disagreed, the code won.

## 1. System overview

```mermaid
flowchart TB
    subgraph client["Client"]
        UI["Chat UI<br/>map, evidence panel, reasoning panel"]
        MIC["Deck microphone<br/>hands-free voice turn"]
    end

    subgraph voice["Voice layer (Sarvam AI, optional)"]
        STT["Saaras STT<br/>120 Hz high-pass pre-filter"]
        TTS["Bulbul TTS<br/>coastal vernacular voice"]
    end

    subgraph router["Orchestration"]
        N8N["n8n multi-agent workflow<br/>supervisor, 6 agents, critic"]
        INP["In-process orchestrator<br/>always available, also the fallback"]
    end

    subgraph api["FastAPI backend"]
        PLAN["Supervisor / planner<br/>intent, location, time, delegation"]
        AGENTS["Specialist agents<br/>ocean, weather, geospatial,<br/>hazard, advisory, discovery, route"]
        RISK["Risk agent<br/>threshold rules + Jev System-One"]
        SYN["Synthesis agent<br/>templates, then rephrase or translate"]
    end

    subgraph models["Models, all optional"]
        GLM["GLM 5.2 on OpenRouter<br/>agentic: intent arbitration"]
        GEM["Google Gemini<br/>synthesis: wording, translation"]
        JEV["Jev System-One<br/>typed safety decisions"]
    end

    subgraph rag["Multi-RAG retrieval"]
        TS["Time-series RAG<br/>gridded and point values"]
        GEO["Geospatial RAG<br/>shapely, distance ranked"]
        VEC["Document RAG<br/>BM25 + synonyms"]
        CAT["Catalogue retriever"]
    end

    subgraph sources["Upstream data"]
        MOS["ISRO / MOSDAC THREDDS<br/>OSF_CIRC, OSF_WAVE, PFZ grids"]
        INC["INCOIS<br/>ERDDAP, PFZ and OSF bulletins"]
        IMD["IMD<br/>warnings, RSMC cyclone bulletin"]
        FB["Fallback tier<br/>Open-Meteo, GDACS, Marine Regions, NASA CMR"]
    end

    MIC -->|"POST /chat/voice"| STT --> INP
    UI -->|POST /chat| INP
    UI -->|"POST /chat?via=n8n"| N8N
    N8N -->|"/internal/classify"| PLAN
    N8N -->|"/internal/retrieve/{domain}"| AGENTS
    N8N -->|"/internal/synthesize"| RISK
    N8N -.->|"unreachable: fall back,<br/>and say so in degraded_sources"| INP
    INP --> PLAN --> AGENTS --> RISK --> SYN
    PLAN -.-> GLM
    RISK -.-> JEV
    SYN -.-> GEM
    AGENTS --> TS & GEO & VEC & CAT
    TS --> MOS & INC & FB
    VEC --> IMD & INC
    GEO --> FB
    CAT --> MOS & INC & FB
    SYN --> TTS
    SYN -->|"answer + evidence + trace"| UI
    TTS -->|"audio_base64"| UI

    classDef isro fill:#f59e0b,stroke:#b45309,color:#1c1917
    classDef incois fill:#38bdf8,stroke:#0369a1,color:#0c1a25
    classDef imd fill:#a78bfa,stroke:#6d28d9,color:#12091f
    classDef fb fill:#64748b,stroke:#334155,color:#f8fafc
    classDef opt fill:#1e293b,stroke:#94a3b8,color:#e2e8f0
    class MOS isro
    class INC incois
    class IMD imd
    class FB fb
    class STT,TTS,GLM,GEM,JEV opt
```

## 2. Request flow: the multi-agent orchestration

A supervisor plans, six specialists retrieve in parallel against a shared
blackboard, a reconciler folds their reports together, rules decide the verdict,
and a critic verifies the answer before it ships. The same graph runs in-process
and in n8n.

```mermaid
sequenceDiagram
    autonumber
    actor U as Fisherman
    participant UI as Chat UI
    participant N8 as n8n orchestrator
    participant API as FastAPI
    participant SUP as Supervisor (planner)
    participant AG as Specialist agents
    participant K as Knowledge sources
    participant RK as Risk rules + Jev
    participant CR as Critic

    U->>UI: "Is it safe to venture out off Rameswaram tomorrow morning?"
    UI->>N8: POST /webhook/orca-router
    N8->>API: POST /internal/classify
    API->>SUP: 52 weighted regex rules, then GLM 5.2 if confidence < 0.55
    SUP-->>API: intent=safety_go_nogo, conf=0.92,<br/>domains=[weather, ocean, hazard, advisory],<br/>location=Rameswaram 9.27N 79.35E, window=tomorrow morning
    API-->>N8: plan + planner trace

    Note over N8: Gate 1 - answerable? No resolvable position<br/>for a question that needs one means ask, not guess.
    alt clarification needed
        N8-->>UI: ask the skipper to narrow it down
    end
    Note over N8: Gate 2 - any specialist needed?<br/>Small talk skips retrieval entirely.

    Note over N8,AG: Delegate one task per domain. Each agent is isolated:<br/>2 tries, then continue. One dead server degrades, never fails.
    par Weather Intelligence
        N8->>API: POST /internal/retrieve/weather
        API->>AG: weather-intelligence
        AG->>K: wind, gust, rain, CAPE, tide
    and Ocean Analytics
        N8->>API: POST /internal/retrieve/ocean
        API->>AG: ocean-analytics
        AG->>K: MOSDAC OSF_CIRC + OSF_WAVE
    and Hazard and Cyclone Watch
        N8->>API: POST /internal/retrieve/hazard
        API->>AG: risk-assessment
        AG->>K: GDACS geometry + IMD warning text
    and Advisory Documents
        N8->>API: POST /internal/retrieve/advisory
        API->>AG: advisory
        AG->>K: live bulletins into the BM25 index
    end
    K-->>AG: values and passages
    AG-->>N8: findings + evidence + trace per agent

    N8->>N8: reconcile: dedupe evidence by id,<br/>name the agents that failed, renumber the trace
    N8->>API: POST /internal/synthesize
    API->>RK: threshold rules, then five typed Jev questions
    RK-->>API: verdict (worst band wins), rules fired, evidence ids
    API->>API: template draft, then Gemini rewords or translates
    API-->>N8: ChatResponse + audio_base64

    N8->>CR: did the verdict survive the rewrite?
    alt critic rejects
        CR->>API: POST /internal/synthesize with allow_llm=false
        API-->>CR: deterministic draft, same numbers and verdict
    end
    CR-->>N8: approved answer
    N8-->>UI: answer + evidence + reasoning trace
    UI-->>U: verdict, numbers with agency labels, map, why
```

The observed run of this exact query returned `intent=safety_go_nogo`,
`confidence=0.92`, band `unsafe`, 22 evidence items and 16 trace steps, with
`gdacs` listed in `degraded_sources`.

### Why this is orchestration and not a pipeline

1. **Two supervisor gates.** Unanswerable questions go back for clarification
   and small talk skips retrieval. Neither wakes an agent or a government server.
2. **Agents are isolated.** Every specialist node carries
   `onError: continueRegularOutput` with two tries. The reconciler names what was
   lost so the answer can admit the gap.
3. **A shared blackboard.** The plan is resolved once. No specialist re-derives
   the place or the window, so they cannot disagree about what was asked.
4. **A critic with authority.** If a rewrite loses an unsafe verdict, or an
   answer arrives with no evidence, the critic rejects it and forces one
   deterministic re-run with `allow_llm: false`. Bounded to a single attempt.

## 3. Intent segregation to knowledge source

The classifier is a weighted keyword model: 52 regex rules, additive weights,
highest total wins. Deterministic and explainable. The LLM is consulted only
when lexical confidence is below 0.55, and the trace records which one decided.

```mermaid
flowchart LR
    Q["User query"] --> LEX["Weighted regex rules<br/>52 patterns, additive weights"]
    LEX -->|"confidence >= 0.55"| INT["Intent"]
    LEX -->|"confidence < 0.55<br/>or no rule fired"| LLM["LLM arbitration<br/>JSON intent only"]
    LLM --> INT
    LLM -.->|"no provider configured"| INT

    INT --> I1["pfz_locate (6 rules)"]
    INT --> I2["safety_go_nogo (6)"]
    INT --> I3["conditions_summary (6)"]
    INT --> I4["hazard_alerts (6)"]
    INT --> I5["productivity_scan (6)"]
    INT --> I6["route_planning (5)"]
    INT --> I7["productivity_diagnosis (4)"]
    INT --> I8["geofence_check (7)"]
    INT --> I9["data_discovery (4)"]
    INT --> I10["small_talk (2)<br/>answers with no retrieval"]
    INT --> I11["unknown<br/>safe default fan-out"]

    I1 --> D_OC & D_AD & D_GE
    I2 --> D_WE & D_OC & D_HA & D_AD
    I3 --> D_OC & D_WE & D_AD
    I4 --> D_HA & D_WE & D_AD
    I5 --> D_OC & D_GE & D_AD
    I6 --> D_WE & D_OC & D_GE & D_HA
    I7 --> D_OC & D_AD & D_CA
    I8 --> D_GE & D_HA & D_OC
    I9 --> D_CA & D_AD
    I11 --> D_WE & D_OC & D_AD

    D_OC["ocean<br/>ocean-analytics"]
    D_WE["weather<br/>weather-intelligence"]
    D_GE["geospatial<br/>geospatial-reasoning"]
    D_HA["hazard<br/>risk-assessment"]
    D_AD["advisory<br/>advisory"]
    D_CA["catalog<br/>data-discovery"]
```

Each of the six domains maps 1:1 onto `/internal/retrieve/{domain}`, which is
what lets the n8n Switch node drive retrieval branch by branch. A separate table
of 11 time-hint patterns resolves "tomorrow morning", "tonight", "this week" and
the rest into a concrete UTC window.

## 4. Multi-RAG: three retrievers, not one

A single vector store cannot answer "how far is the IMBL" or "what is the SST at
this point". ORCA runs three retrievers with different index types behind one
router.

```mermaid
flowchart TB
    ROUTE["Knowledge source router"]

    subgraph vec["Document RAG"]
        direction TB
        V1["BM25 index, k1/b weighted idf<br/>+ marine synonym expansion"]
        V2["18 chunks over 4 documents:<br/>seeded corpus + live IMD / INCOIS text"]
        V3["Official agency text boosted 1.15x"]
        V1 --> V2 --> V3
    end

    subgraph ts["Time-series and grid RAG"]
        direction TB
        T1["NCSS point query<br/>CSV series at one lat/lon"]
        T2["OPeNDAP ASCII window<br/>strided grid, no netCDF lib"]
        T3["Land-mask spiral<br/>nearest wet cell, distance reported"]
        T4["Validity clamp<br/>request pinned inside the dataset TimeSpan"]
        T1 --> T3 --> T4
        T2 --> T4
    end

    subgraph geo["Geospatial RAG"]
        direction TB
        G1["India EEZ polygon, cached from WFS"]
        G2["9 zones: 5 MPA, 2 IMBL,<br/>1 restricted, 1 eco-sensitive"]
        G3["Harbour gazetteer, 58 places"]
        G4["Point-in-polygon, distance,<br/>bearing, track intersection"]
        G1 --> G4
        G2 --> G4
        G3 --> G4
    end

    ROUTE --> vec & ts & geo
    vec --> EV["Evidence<br/>value + unit + provenance + tier"]
    ts --> EV
    geo --> EV
    EV --> RULES["Threshold rules"] --> ANS["Answer"]
```

Dense retrieval via Qdrant is optional (`backend/requirements-optional.txt`).
With it absent, `/health` reports `vector_rag.mode: lexical-bm25`, which is the
verified default.

## 5. Source precedence and degradation

Every value carries the tier it came from. The UI shows it, so a user can see
when an answer rests on a fallback rather than on an Indian official product.

```mermaid
flowchart LR
    NEED["Need a value"] --> T1{"ISRO product<br/>on MOSDAC?"}
    T1 -->|yes, cycle fresh| USE1["Use it<br/>tier1-isro"]
    T1 -->|"cycle stale (> 48 h)"| BOTH["Use the fallback for the live number,<br/>show the ISRO value beside it,<br/>label both"]
    T1 -->|no| T2{"INCOIS product?"}
    T2 -->|yes| USE2["Use it<br/>tier2-incois"]
    T2 -->|no| T3{"IMD warning text?"}
    T3 -->|yes| USE3["Quote it verbatim<br/>tier3-imd, overrides model numbers"]
    T3 -->|no| T4{"Free fallback exists?"}
    T4 -->|yes| USE4["Use it, mark official=false<br/>tier4-fallback"]
    T4 -->|no| GAP["Say the gap out loud<br/>no silent substitution"]

    classDef isro fill:#f59e0b,stroke:#b45309,color:#1c1917
    classDef incois fill:#38bdf8,stroke:#0369a1,color:#0c1a25
    classDef imd fill:#a78bfa,stroke:#6d28d9,color:#12091f
    classDef fb fill:#64748b,stroke:#334155,color:#f8fafc
    classDef gap fill:#f87171,stroke:#b91c1c,color:#1c1917
    class USE1,BOTH isro
    class USE2 incois
    class USE3 imd
    class USE4 fb
    class GAP gap
```

A connector never raises into the agent layer. It returns `None` or an empty
list, increments its own failure counter, and the failure surfaces in
`degraded_sources` on the response and under `sources` in `/health`. Retries
give up immediately on 4xx and back off on 5xx.

## 6. Agent collaboration on one request

Retrieval agents run concurrently because each waits on a different government
server. Risk, visualisation and synthesis are strictly ordered after them.

```mermaid
flowchart LR
    P["planner"] --> W["weather-intelligence"]
    P --> O["ocean-analytics"]
    P --> G["geospatial-reasoning"]
    P --> A["advisory"]
    P --> C["data-discovery"]
    W & O & G & A & C --> RT["route-planner<br/>only if a destination was resolved"]
    RT --> RK["risk-assessment"]
    W & O & G & A & C --> RK
    RK --> VZ["visualisation"] --> SY["synthesis"] --> OUT["ChatResponse"]

    subgraph parallel["one concurrent wave (asyncio.gather)"]
        W
        O
        G
        A
        C
    end
```

Each agent runs inside `_safe_run`. An agent that raises is isolated: the
exception is logged as a `failed` trace step, the remaining agents continue, and
the answer states what is missing. One dead upstream never sinks a request.

## 7. Safety verdict: threshold rules, then a structured judgment

The verdict is computed in code. The threshold rules read retrieved values and
each rule that fires is recorded with the evidence ids that triggered it. The
worst band wins; bands are never averaged, so one gale warning is not cancelled
out by calm seas.

Alongside them runs Jev, TypeSafe AI's System One model. Jev is not a chat model
and does not generate prose: you send one `state` plus a map of typed
`questions`, and it returns typed `answers` under the same ids with calibrated
probabilities. ORCA asks it five questions in a single call.

```mermaid
flowchart TB
    IN["Retrieved findings<br/>waves, weather, ocean, hazards, geo"] --> RULES

    subgraph RULES["Threshold rules (config-driven)"]
        direction TB
        R1["swh: caution >= 1.5 m, danger >= 2.5 m"]
        R2["wind: caution >= 17 kt, danger >= 27 kt"]
        R3["gust: danger >= 34 kt"]
        R4["convective proxy: CAPE + rain -> moderate / high"]
        R5["tropical cyclone: <= 300 km unsafe, <= 800 km caution"]
        R6["IMD warning text in force -> caution"]
        R7["surface current >= 50 cm/s -> caution"]
        R8["inside restricted zone / approaching boundary"]
        R9["outside the India EEZ -> unsafe"]
    end

    IN --> STATE["Build one Jev state object:<br/>position, sea state, thresholds,<br/>hazards, boundaries, vessel class"]

    subgraph JEVQ["Five typed Jev questions, one call"]
        direction TB
        Q1["safety_verdict (choice)<br/>SAFE | CAUTION | UNSAFE"]
        Q2["action (choice)<br/>VENTURE_PERMITTED | EXERCISE_VIGILANCE<br/>| STAY_IN_HARBOR"]
        Q3["severity (score)<br/>5 ordered levels, benign to extreme"]
        Q4["breach_probability (noul)<br/>boundary or MPA violation"]
        Q5["capsizing_probability (noul)<br/>capsize or severe swamping"]
    end

    STATE --> T{"key configured?"}
    T -->|yes| JEVQ
    T -->|no| EMU["Offline rule emulator<br/>same answer schema"]
    JEVQ --> JF["jev_* finding<br/>verdict + severity + probabilities<br/>+ verdict confidence"]
    EMU --> JF

    RULES --> WORST["Worst band wins"]
    JF --> WORST
    WORST --> BAND["band: safe | caution | unsafe | unknown"]
    BAND --> RA["RiskAssessment<br/>band, score, findings[], window_advice,<br/>jev_decision"]

    classDef warn fill:#f59e0b,stroke:#b45309,color:#1c1917
    class JF warn
```

Answer shapes, which are not uniform and matter when reading the code:

| primitive | request `criteria` | answer carries |
|---|---|---|
| `choice` | map of option to rubric text | `choice`, full `probabilities` map, `confidence` |
| `score` | ordered array of 2 to 10 level descriptions | `score` as a weighted float over level indices, `legend`, `probabilities`, `confidence` |
| `noul` | optional `{true, false}` descriptions | `noul` only, a yes probability. **No confidence field.** |

ORCA normalises the severity score by the number of levels minus one to reach
0-100. Any off-schema verdict, any non-200, and any transport error falls through
to the emulator rather than reaching the risk agent.

Two caveats a reviewer should know, both visible in a live response:

- Jev re-derives a verdict from the same wave, wind and cyclone numbers the
  threshold rules already used, so it can disagree with them. On the Rameswaram
  run the rules returned `unsafe` (high convective risk) while the emulator
  returned `SAFE / VENTURE_PERMITTED`. Worst-band-wins keeps the conservative
  verdict, but both appear in `risk.findings`.
- `risk.score` is `worst band weight + min(8 x bad findings, 15)`, so any single
  unsafe finding saturates it at 100. Read `band` and `findings`, not the score.

## 8. Synthesis, the verdict guard, and the critic

```mermaid
flowchart TB
    F["Findings + evidence + risk"] --> D["Per-intent template draft<br/>always runs, never fails"]
    D --> AL{"allow_llm?"}
    AL -->|"false (critic repair pass)"| KEEP["Keep the draft"]
    AL -->|true| Q{"synthesis provider configured?"}
    Q -->|no| KEEP
    Q -->|yes| L{"target language"}
    L -->|en| P1["Gemini rephrase prompt<br/>keep every number and the verdict"]
    L -->|"kn, ta, te, ml, hi,<br/>mr, gu, bn"| P2["Gemini translate and adapt prompt<br/>keep every number and the verdict"]
    P1 --> GD{"Verdict guard<br/>only enforced for English"}
    P2 --> GD
    GD -->|"unsafe verdict survived"| USE["Use the rewrite,<br/>llm_used = true"]
    GD -->|"verdict dropped or altered"| KEEP
    KEEP --> TTSQ
    USE --> TTSQ{"Sarvam TTS"}
    TTSQ --> OUT["answer + language + audio_base64"]
    OUT --> CR{"n8n critic<br/>second opinion"}
    CR -->|approved| SHIP["Ship it"]
    CR -->|"rejected: verdict lost, or a<br/>non-English rewrite it cannot read"| RETRY["Re-run with allow_llm=false<br/>bounded, one attempt"]
    RETRY --> D

    classDef risky fill:#f87171,stroke:#b91c1c,color:#1c1917
    class GD risky
```

The in-process guard (`SynthesisAgent._verdict_survived`) only fires when the
band is `unsafe`. For English it requires one of "do not", "don't", "avoid",
"stay in", "not safe", "unsafe", "remain in harbour", "postpone" to be present in
the rewrite. **For any other language it only checks that the text is longer than
10 characters**, so a translated rewrite can currently soften or flip an unsafe
verdict. This is a known defect, recorded in the README limits, not a design
decision.

The n8n critic partially compensates: because it cannot read a translated answer
either, it refuses to approve any non-English rewrite of an `unsafe` verdict and
forces the deterministic draft instead. That guard exists only on the n8n path,
so `POST /chat` without `?via=n8n` is still exposed.

## 9. Voice pipeline (Sarvam AI, optional)

```mermaid
sequenceDiagram
    autonumber
    actor U as Fisherman on deck
    participant UI as Chat UI
    participant API as FastAPI
    participant AF as AcousticFilter
    participant SV as Sarvam AI
    participant ORC as Orchestrator

    U->>UI: speaks in Kannada over engine noise
    UI->>API: POST /chat/voice (wav or audio_base64)
    API->>AF: 120 Hz single-pole high-pass
    Note over AF: suppresses 40-250 Hz diesel rumble;<br/>passthrough if not 16-bit RIFF PCM
    AF->>SV: Saaras speech-to-text
    alt SARVAM_API_KEY set
        SV-->>API: transcript + detected language_code
    else no key, or the call fails
        SV-->>API: canned sample transcript,<br/>provider = mock-saaras-offline
    end
    API->>ORC: ChatRequest(message=transcript, language=detected locale)
    ORC-->>API: ChatResponse with the deterministic verdict
    API->>SV: Bulbul text-to-speech in the detected language
    alt key set
        SV-->>API: audio, provider = sarvam-bulbul
    else no key
        SV-->>API: synthetic 440 Hz WAV,<br/>provider = mock-bulbul-offline
    end
    API-->>UI: answer + audio_base64
    UI-->>U: spoken answer, eyes-free
```

Language handling, as implemented:

| Layer | Languages |
|---|---|
| Script auto-detect from typed text | hi, ta, ml, te, gu, bn, kn (Devanagari resolves to hi, so Marathi is reported as Hindi) |
| LLM translation prompts | en, hi, bn, gu, kn, ml, mr, ta, te |
| Sarvam locale map | en, hi, bn, gu, kn, ml, mr, or, pa, ta, te |

Two things to keep in mind when demoing: TTS runs on **every** `/chat`
response, not only voice turns, and with no Sarvam key the mock placeholder beep
was 69.5% of a 122 kB response. The offline STT mock returns a fixed sample
sentence regardless of the audio supplied.

## 10. Explainability: what ships with every answer

```mermaid
flowchart TB
    ANS["ChatResponse"] --> A1["answer<br/>natural language"]
    ANS --> A2["risk<br/>band, score, every rule that fired,<br/>each linked to evidence ids, jev_decision"]
    ANS --> A3["evidence[]<br/>value, unit, position, time, series,<br/>agency, tier, access method,<br/>staleness, caveat"]
    ANS --> A4["trace[]<br/>agent, action, rationale, tool,<br/>tool args, outcome, ms, status"]
    ANS --> A5["layers[]<br/>MOSDAC WMS tiles + GeoJSON overlays"]
    ANS --> A6["charts[]<br/>series with threshold lines"]
    ANS --> A7["degraded_sources[]<br/>what failed on this request"]
    ANS --> A8["llm_used, language, via_n8n<br/>was the wording model-written, in what language,<br/>and which router answered"]
    ANS --> A9["followups[]<br/>next questions, intent aware"]
    ANS --> A10["audio_base64<br/>spoken answer"]
```

## 11. API surface

| method | path | purpose |
|---|---|---|
| POST | `/chat` | full pipeline, in-process |
| POST | `/chat?via=n8n` | same, routed through the n8n workflow, falls back in-process |
| POST | `/chat/voice` | STT, then the full pipeline, then TTS |
| POST | `/api/voice/stt` | transcribe only |
| POST | `/api/voice/tts` | synthesise speech only |
| POST | `/internal/classify` | stage 1: segregate intent, name the knowledge sources |
| POST | `/internal/retrieve/{domain}` | stage 2: retrieve from one source |
| POST | `/internal/synthesize` | stages 3 and 4: risk rules, then the answer |
| GET | `/knowledge-sources` | what each domain is and where it reads from |
| GET | `/health` | index sizes, per-source health, LLM / Jev / Sarvam status |

None of these are authenticated. That is acceptable on localhost and nowhere
else. `host` defaults to `0.0.0.0`, and CORS does not restrain non-browser
clients, so put auth in front of the API and narrow `ORCA_CORS_ORIGINS` before
exposing it.

## 12. Model routing: one job, one model

Three external AI services, all optional, all off by default. The system answers
every supported query without any of them.

```mermaid
flowchart TB
    subgraph roles["Role based LLM routing"]
        direction TB
        RA["AGENTIC role<br/>intent arbitration, judgment"] --> RAM["GLM 5.2 on OpenRouter<br/>z-ai/glm-5.2"]
        RS["SYNTHESIS role<br/>wording, regional translation"] --> RSM["Google Gemini<br/>gemini-3.5-flash"]
    end

    subgraph fb["Fallback chain, per role"]
        direction TB
        F0["role provider has a key?"] -->|yes| F1["use the role's model"]
        F0 -->|no| F2["generic auto-resolution:<br/>OpenRouter, Gemini, OpenAI, Ollama"]
        F2 --> F3["swap to that provider's default model,<br/>never send a slug it cannot serve"]
        F3 -->|nothing configured| F4["none -> deterministic templates"]
    end

    subgraph jevs["Jev System-One (decisions, not prose)"]
        direction TB
        J0["provider=openrouter"] --> J1["POST openrouter.ai/api/alpha/decisions<br/>model typesafe/jev-1.13"]
        J0b["provider=typesafe"] --> J2["POST api.typesafe.ai/v1/systemone<br/>model jev-latest"]
        J1 --> J3["typed answers by question id"]
        J2 --> J3
        J3 -->|"off-schema, non-200,<br/>timeout, or no key"| J4["offline rule emulator"]
    end

    subgraph sv["Sarvam AI (voice)"]
        direction TB
        S1["key set -> Saaras STT + Bulbul TTS"]
        S2["no key -> mock transcript + beep WAV"]
    end

    roles --> fb

    classDef glm fill:#38bdf8,stroke:#0369a1,color:#0c1a25
    classDef ggl fill:#f59e0b,stroke:#b45309,color:#1c1917
    class RAM glm
    class RSM ggl
```

Why split the roles: the two jobs have opposite requirements. Intent arbitration
is a reasoning problem where a wrong route wastes a whole retrieval fan-out, so
it gets a reasoning model. Rewording a draft that is already factually complete
is a fluency and multilingual problem where latency is felt by the user, so it
gets a fast Google model. Both are pinned per role rather than sharing one
global model.

`ORCA_LLM_PROVIDER` still gates everything: `none` disables both roles, `ggl` is
an alias for `gemini`. The fallback deliberately swaps the model when it swaps
the provider, so a Gemini-only deployment never tries to send `z-ai/glm-5.2` to
Google.

Jev is billed per input token with output free, and TypeSafe documents 70 to
500 ms responses, so it is cheap enough to call on every safety verdict. Both
transports share one wire protocol; only the path and the model alias differ.

## 13. Repository layout

```
ORCA/
  backend/orca/
    main.py                 FastAPI app, /chat, /chat/voice, /api/voice/*, /internal/*
    services.py             one container: connectors, retrievers, LLM, sessions
    schemas.py              Evidence, Provenance, TraceStep, RiskAssessment, ChatResponse
    config.py               settings, all with working defaults
    llm.py                  role routed LLM: GLM 5.2 agentic, Gemini synthesis, template fallback
    voice.py                Sarvam Saaras STT, Bulbul TTS, 120 Hz acoustic pre-filter
    jev.py                  Jev System-One decisions via OpenRouter, offline emulator fallback
    connectors/             mosdac, incois, imd, openmeteo, gdacs, marine_regions, nasa
    rag/
      router.py             intent segregation and knowledge-source routing
      vector_rag.py         BM25 document retrieval
      geo_rag.py            spatial retrieval and geofencing
      timeseries_rag.py     point and grid retrieval with source precedence
    agents/
      planner.py            intent, location, time, task decomposition
      ocean.py              ocean analytics + weather intelligence
      geo.py                geospatial reasoning + route planning
      risk.py               hazard retrieval + threshold rules + Jev judgment
      advisory.py           document retrieval + dataset discovery
      viz.py                map layers and charts
      synthesis.py          templated draft, LLM rephrase or translate, verdict guard, TTS
      orchestrator.py       runs the task graph
  frontend/index.html       chat UI, no build step, all API text HTML-escaped
  knowledge/                seeded advisory corpus indexed by the document RAG
  data/layers/              zone GeoJSON, cached EEZ
  n8n/workflows/            orca_router.json, the 24 node multi-agent orchestrator
  tests/                    29 tests: voice, Jev protocol, LLM role routing, API smoke
  scripts/verify_e2e.py     runs the eight sample queries and asserts routing
  docs/                     this file and DATA_SOURCES.md
```

## 14. The n8n orchestration graph

`n8n/workflows/orca_router.json`, 24 nodes. This is the node graph as imported,
so it can be read next to the canvas.

```mermaid
flowchart TB
    W["Chat in<br/>webhook"] --> SUP["Supervisor:<br/>segregate and plan"]
    SUP --> BB["Blackboard:<br/>shared context"]
    BB --> G1{"Gate: is the question<br/>answerable?"}
    G1 -->|"clarification needed"| RC["Ask the skipper<br/>to narrow it down"]
    G1 -->|"answerable"| G2{"Gate: does this need<br/>specialist agents?"}
    G2 -->|"no domains (small talk)"| SC["Shortcut:<br/>no retrieval needed"]
    G2 -->|"domains named"| DL["Delegate:<br/>one task per domain"]
    DL --> AS["Supervisor:<br/>assign to the owning agent"]

    AS --> A1["Agent:<br/>Ocean Analytics"]
    AS --> A2["Agent:<br/>Weather Intelligence"]
    AS --> A3["Agent:<br/>Geospatial Reasoning"]
    AS --> A4["Agent:<br/>Hazard and Cyclone Watch"]
    AS --> A5["Agent:<br/>Advisory Documents"]
    AS --> A6["Agent:<br/>Data Discovery"]

    A1 & A2 & A3 & A4 & A5 & A6 --> MG["Blackboard:<br/>collect agent reports"]
    MG --> RE["Reconciler:<br/>fold reports, flag gaps"]
    RE --> SY["Agent:<br/>Risk rules then Synthesis"]
    SC --> SY
    SY --> CR["Critic:<br/>verify the verdict survived"]
    CR --> G3{"Gate:<br/>critic approved?"}
    G3 -->|yes| OUT["Answer out"]
    G3 -->|no| RP["Repair:<br/>re-synthesise without the model"]
    RP --> RS["Agent: Synthesis<br/>(deterministic only)"]
    RS --> OUT

    classDef gate fill:#38bdf8,stroke:#0369a1,color:#0c1a25
    classDef agent fill:#f59e0b,stroke:#b45309,color:#1c1917
    classDef guard fill:#f87171,stroke:#b91c1c,color:#1c1917
    class G1,G2,G3 gate
    class A1,A2,A3,A4,A5,A6 agent
    class CR,RP,RS guard
```

Each of the six specialist nodes sets `retryOnFail` with two tries,
`alwaysOutputData`, and `onError: continueRegularOutput`, which is what makes the
fan-out survive a dead upstream. The merge node declares six inputs so it waits
for every branch before the reconciler runs.

## 15. Verified state and known gaps

Confirmed on 2026-09-22 against a running instance:

| Check | Result |
|---|---|
| `python -m pytest tests -q` | 29 passed, hermetic (a real `.env` cannot change the outcome) |
| `scripts/verify_e2e.py` | all checks passed across 8 sample queries |
| Warm-up | 18 knowledge chunks, 9 zone features, EEZ loaded |
| Gazetteer | 58 places |
| `POST /chat` sample query | HTTP 200, 22 evidence items, 16 trace steps |
| `POST /internal/synthesize` with `allow_llm=false` | HTTP 200, rewrite step reported `skipped` |
| Jev live call | `jev-openrouter`, answered as `typesafe/jev-1.13-20260917`: `CAUTION` at confidence 0.73, distribution `{CAUTION 0.82, UNSAFE 0.17, SAFE 0.01}`, severity 50.2/100, capsizing noul 0.48 |
| Worst-band-wins | rules said `unsafe`, Jev said `CAUTION`, final band stayed `unsafe` |
| GLM 5.2 rewrite | `llm_used: true` and the unsafe verdict survived ("Do not venture out off Rameswaram tomorrow morning") |
| n8n workflow | 24 nodes, no dangling or unreachable nodes, 6 isolated agents |
| Marine Regions WFS | 200, EEZ polygon cached to disk |
| GDACS | failing, `bad json` on an empty body |

Open gaps, in the order they matter:

1. The non-English verdict guard in `synthesis.py` is effectively a length check,
   so a translated rewrite can soften an unsafe verdict. The n8n critic blocks
   this on its own path; plain `POST /chat` is still exposed.
2. GDACS returns an empty body, so the cyclone distance rule cannot fire. The
   failure is reported in `degraded_sources` but nothing else compensates.
3. Jev is a second opinion over the same inputs and can contradict the threshold
   rules in the same response. Worst-band-wins keeps this safe but noisy.
4. `min_zone_dist` in `risk.py` is the nearest zone of any kind, so a benign MPA
   a few km away raises the modelled breach probability.
5. TTS and the acoustic filter are per-sample Python loops running inside async
   request handlers, and TTS runs on every text request.
6. No authentication on any endpoint.
7. The core is untested: no tests cover the intent router, BM25, the geofence
   index, the grid readers or the risk rules.

See the README for the full list, including the data-side limits.
