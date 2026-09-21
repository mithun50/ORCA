# ORCA architecture

ORCA answers marine questions in natural language by segregating the query,
routing it to the knowledge sources that can actually answer it, retrieving from
official Indian agency services, applying explicit safety rules, and returning
the answer together with the evidence and the reasoning behind it.

## 1. System overview

```mermaid
flowchart TB
    subgraph client["Client"]
        UI["Chat UI<br/>map, evidence panel, reasoning panel"]
    end

    subgraph router["Request router"]
        N8N["n8n workflow<br/>orca-router webhook"]
        INP["In-process orchestrator<br/>fallback path"]
    end

    subgraph api["FastAPI backend"]
        PLAN["Planner agent<br/>intent, location, time, task graph"]
        AGENTS["Specialised agents<br/>ocean, weather, geospatial,<br/>hazard, advisory, discovery, route"]
        RISK["Risk agent<br/>threshold rules"]
        SYN["Synthesis agent<br/>templates then LLM rephrase"]
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

    UI -->|POST /chat| INP
    UI -->|"POST /chat?via=n8n"| N8N
    N8N -->|"/internal/classify"| PLAN
    N8N -->|"/internal/retrieve/{domain}"| AGENTS
    N8N -->|"/internal/synthesize"| RISK
    INP --> PLAN --> AGENTS --> RISK --> SYN
    AGENTS --> TS & GEO & VEC & CAT
    TS --> MOS & INC & FB
    VEC --> IMD & INC
    GEO --> FB
    CAT --> MOS & INC & FB
    SYN -->|"answer + evidence + trace"| UI

    classDef isro fill:#f59e0b,stroke:#b45309,color:#1c1917
    classDef incois fill:#38bdf8,stroke:#0369a1,color:#0c1a25
    classDef imd fill:#a78bfa,stroke:#6d28d9,color:#12091f
    classDef fb fill:#64748b,stroke:#334155,color:#f8fafc
    class MOS isro
    class INC incois
    class IMD imd
    class FB fb
```

## 2. Request flow: segregate, route, retrieve, synthesise

This is the path the problem statement cares about. The query is classified
first, and classification decides which knowledge sources get called. Nothing is
retrieved speculatively.

```mermaid
sequenceDiagram
    autonumber
    actor U as Fisherman
    participant UI as Chat UI
    participant N8 as n8n router
    participant API as FastAPI
    participant P as Planner
    participant R as Retrieval agents
    participant K as Knowledge sources
    participant RK as Risk rules
    participant S as Synthesis

    U->>UI: "Is it safe to venture out tomorrow morning near Rameswaram?"
    UI->>N8: POST /webhook/orca-router
    N8->>API: POST /internal/classify
    API->>P: lexical rules, then LLM only if unsure
    P-->>API: intent=safety_go_nogo, conf=0.92,<br/>domains=[weather, ocean, hazard, advisory],<br/>location=Rameswaram 9.27N 79.35E, window=tomorrow morning
    API-->>N8: classification + planner trace

    Note over N8: Switch node fans out one branch per domain
    par weather
        N8->>API: POST /internal/retrieve/weather
        API->>R: weather-intelligence agent
        R->>K: wind, gust, rain, CAPE, tide
    and ocean
        N8->>API: POST /internal/retrieve/ocean
        API->>R: ocean-analytics agent
        R->>K: MOSDAC OSF_CIRC + OSF_WAVE
    and hazard
        N8->>API: POST /internal/retrieve/hazard
        API->>R: risk-assessment agent
        R->>K: GDACS cyclones + IMD warning text
    and advisory
        N8->>API: POST /internal/retrieve/advisory
        API->>R: advisory agent
        R->>K: live bulletins into the document index
    end
    K-->>R: values and passages
    R-->>N8: findings + evidence + trace per branch

    N8->>N8: merge and de-duplicate evidence by id
    N8->>API: POST /internal/synthesize
    API->>RK: apply SWH / wind / gust / cyclone / geofence rules
    RK-->>API: verdict CAUTION, rules that fired, evidence ids
    API->>S: template draft, then LLM rephrase only
    S-->>API: answer
    API-->>N8: ChatResponse
    N8-->>UI: answer + evidence + reasoning trace
    UI-->>U: verdict, numbers with agency labels, map, why
```

## 3. Intent segregation to knowledge source

The classifier is a weighted keyword model, deterministic and explainable. The
LLM is consulted only when lexical confidence is below 0.55, and the trace
records which one decided.

```mermaid
flowchart LR
    Q["User query"] --> LEX["Weighted keyword rules<br/>~60 patterns"]
    LEX -->|"confidence >= 0.55"| INT["Intent"]
    LEX -->|"confidence < 0.55"| LLM["LLM arbitration<br/>JSON intent only"]
    LLM --> INT

    INT --> I1["pfz_locate"]
    INT --> I2["safety_go_nogo"]
    INT --> I3["conditions_summary"]
    INT --> I4["hazard_alerts"]
    INT --> I5["productivity_scan"]
    INT --> I6["route_planning"]
    INT --> I7["productivity_diagnosis"]
    INT --> I8["geofence_check"]
    INT --> I9["data_discovery"]

    I1 --> D_OC & D_AD & D_GE
    I2 --> D_WE & D_OC & D_HA & D_AD
    I3 --> D_OC & D_WE & D_AD
    I4 --> D_HA & D_WE & D_AD
    I5 --> D_OC & D_GE & D_AD
    I6 --> D_WE & D_OC & D_GE & D_HA
    I7 --> D_OC & D_AD & D_CA
    I8 --> D_GE & D_HA & D_OC
    I9 --> D_CA & D_AD

    D_OC["ocean<br/>time-series RAG"]
    D_WE["weather<br/>time-series RAG"]
    D_GE["geospatial<br/>geo RAG"]
    D_HA["hazard<br/>connector + rules"]
    D_AD["advisory<br/>document RAG"]
    D_CA["catalog<br/>catalogue retriever"]
```

## 4. Multi-RAG: three retrievers, not one

A single vector store cannot answer "how far is the IMBL" or "what is the SST at
this point". ORCA runs three retrievers with different index types behind one
router.

```mermaid
flowchart TB
    ROUTE["Knowledge source router"]

    subgraph vec["Document RAG"]
        direction TB
        V1["BM25 index + marine synonym expansion"]
        V2["Chunks: seeded corpus +<br/>live IMD / INCOIS bulletin text"]
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
        G1["EEZ polygon, cached from WFS"]
        G2["MPA / restricted / IMBL layer"]
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

## 5. Source precedence and degradation

Every value carries the tier it came from. The UI shows it, so a user can see
when an answer rests on a fallback rather than on an Indian official product.

```mermaid
flowchart LR
    NEED["Need a value"] --> T1{"ISRO product<br/>on MOSDAC?"}
    T1 -->|yes, cycle fresh| USE1["Use it<br/>tier1-isro"]
    T1 -->|"cycle stale"| BOTH["Use the fallback for the live number,<br/>show the ISRO value beside it,<br/>label both"]
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

    subgraph parallel["one concurrent wave"]
        W
        O
        G
        A
        C
    end
```

## 7. Explainability: what ships with every answer

```mermaid
flowchart TB
    ANS["ChatResponse"] --> A1["answer<br/>natural language"]
    ANS --> A2["risk<br/>band, score, every rule that fired,<br/>each linked to evidence ids"]
    ANS --> A3["evidence[]<br/>value, unit, position, time, series,<br/>agency, tier, access method,<br/>staleness, caveat"]
    ANS --> A4["trace[]<br/>agent, action, rationale, tool,<br/>tool args, outcome, ms, status"]
    ANS --> A5["layers[]<br/>MOSDAC WMS tiles + GeoJSON overlays"]
    ANS --> A6["charts[]<br/>series with threshold lines"]
    ANS --> A7["degraded_sources[]<br/>what failed on this request"]
```

## 8. Repository layout

```
ORCA/
  backend/orca/
    main.py                 FastAPI app, /chat and the /internal/* stages for n8n
    services.py             one container: connectors, retrievers, LLM, sessions
    schemas.py              Evidence, Provenance, TraceStep, ChatResponse
    config.py               settings, all with working defaults
    llm.py                  optional Ollama / OpenAI / Gemini, template fallback
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
      risk.py               hazard retrieval + threshold rules
      advisory.py           document retrieval + dataset discovery
      viz.py                map layers and charts
      synthesis.py          templated draft, LLM rephrase, verdict guard
      orchestrator.py       runs the task graph
  frontend/index.html       chat UI, no build step
  knowledge/                seeded advisory corpus indexed by the document RAG
  data/layers/              zone GeoJSON, cached EEZ
  n8n/workflows/            orca_router.json
  scripts/verify_e2e.py     runs the eight sample queries and asserts routing
  docs/                     this file and DATA_SOURCES.md
```
