# Quran Knowledge Graph — Architecture Deep Dive

A complete walkthrough of what happens between the moment you type a question and the moment the cited answer appears on screen. All diagrams are [Mermaid](https://mermaid.js.org) — they render inline on GitHub, Obsidian, Notion, VS Code's markdown preview, and most modern viewers.

---

## The 5 architectural layers

```mermaid
graph TB
    subgraph L1["LAYER 1 — Browser (index.html)"]
        THREE[Three.js 3D scene<br/>Fibonacci sphere · halos · cluster]
        CHAT_UI[Chat panel<br/>SSE stream renderer]
        HOVER[Hover preview card]
    end

    subgraph L2["LAYER 2 — FastAPI server (app_free.py, port 8085)"]
        ENDPOINT["/chat endpoint<br/>Streams Server-Sent Events"]
        AGENT[Agent loop orchestrator<br/>up to 8 tool turns]
        PRIMING[Priming search<br/>&lt;200ms keyword Neo4j query]
        COMPRESS[Tool compressor<br/>trims results 60-70%]
    end

    subgraph L3["LAYER 3 — Model inference"]
        OLLAMA["Ollama localhost:11434<br/>Qwen3-8B default"]
        OR["OpenRouter<br/>gpt-oss-120b (free tier)"]
    end

    subgraph L4["LAYER 4 — Persistent data"]
        N4J[("Neo4j graph<br/>92K nodes · 403K edges<br/>localhost:7687")]
        CACHE[("Answer cache<br/>data/answer_cache.json<br/>500 entries")]
    end

    subgraph L5["LAYER 5 — ML helpers (loaded lazily)"]
        EMBED[all-MiniLM-L6-v2<br/>sentence embeddings]
        RERANK[ms-marco-MiniLM<br/>cross-encoder reranker]
        NLI[nli-deberta-v3-xsmall<br/>citation verifier]
    end

    THREE -->|user types| ENDPOINT
    ENDPOINT --> AGENT
    AGENT --> PRIMING
    AGENT --> COMPRESS
    AGENT --> OLLAMA
    AGENT --> OR
    AGENT --> N4J
    PRIMING --> N4J
    CACHE --> EMBED
    ENDPOINT -.SSE.->CHAT_UI
    ENDPOINT -.SSE.-> THREE
    HOVER -.reads.-> CACHE
```

**Layer 1** is a single-file SPA (`index.html`) — Three.js for the sphere, vanilla JS for the chat, 3d-force-graph for the dynamic cluster.

**Layer 2** is a lean FastAPI app (`app_free.py`). It doesn't generate anything itself — it orchestrates the agent loop and streams events back.

**Layer 3** is the "brain" — an LLM that decides which tools to call. Two options:
- **Ollama** runs models locally on your GPU (Qwen3-8B, 14B)
- **OpenRouter** brokers API access to bigger free-tier models (gpt-oss-120b, Qwen3-Coder-480B)

**Layer 4** is persistent storage — the Neo4j graph (the actual Quran data) and the JSON answer cache.

**Layer 5** is three small transformer models loaded on-demand for semantic search, reranking, and citation verification. They run on CPU in <500MB RAM each.

---

## End-to-end request flow

What happens when you type "What does the Quran say about patience?" and hit Send:

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Browser as Browser<br/>(index.html)
    participant Server as FastAPI<br/>(app_free.py)
    participant Cache as Answer Cache<br/>(JSON + embeddings)
    participant Neo4j as Neo4j Graph
    participant LLM as Qwen3-8B<br/>(Ollama)

    User->>Browser: Types question + hits Send
    Browser->>Server: POST /chat (deep_dive=false)
    Server-->>Browser: Opens SSE stream

    Note over Server,Cache: Phase 0 — Cache lookup
    Server->>Cache: Embed question (MiniLM)
    Cache->>Cache: Cosine similarity vs 500 entries
    Cache-->>Server: Top 3 similar past answers (if sim > 0.6)
    Server->>Server: Inject into system prompt as context

    Note over Server,Neo4j: Phase 1 — Priming (~200ms)
    Server->>Neo4j: Fast keyword search on user's message
    Neo4j-->>Server: 6 candidate verses
    Server-->>Browser: graph_update event
    Browser->>Browser: Cluster nodes + sphere halos appear INSTANTLY

    Note over Server,LLM: Phase 2 — Agent loop (1-5 turns)
    loop Until model stops calling tools
        Server->>LLM: Messages + tool schemas
        LLM-->>Server: {tool_calls: [search_keyword(...)]}
        Server->>Neo4j: Execute Cypher query
        Neo4j-->>Server: Verses + metadata
        Server->>Server: Compress (trim text for non-top verses)
        Server-->>Browser: tool event + graph_update event
        Browser->>Browser: Cluster grows, more halos light up
    end

    Note over Server,LLM: Phase 3 — Final answer
    Server->>LLM: "No more tools, write the answer"
    LLM-->>Server: Streaming markdown
    Server-->>Browser: text events (word-by-word)
    Browser->>Browser: Answer renders with [2:255] citations as tooltips

    Note over Server,Cache: Phase 4 — Citation check + save
    Server->>Server: Count unique [X:Y] citations
    alt Fewer than 5 citations
        Server->>LLM: "Add more citations" (retry once)
        LLM-->>Server: Richer answer
        Server-->>Browser: retry event (replaces previous)
    end
    Server->>Cache: Save question + answer + embedding
    Server->>Neo4j: Fetch text of every cited verse
    Neo4j-->>Server: Verse texts
    Server-->>Browser: done event + verse texts
    Browser->>Browser: Citations become hoverable tooltips

    User->>Browser: Hovers any cluster node
    Browser->>Browser: Preview card appears with Arabic + English
```

Let me break down each phase in detail.

---

## Phase 0 — Cache lookup (before the AI even starts)

The single most important optimization. Before running anything expensive, we check if we've already answered a similar question.

```mermaid
graph LR
    Q[New question:<br/>'What is patience?'] --> E[all-MiniLM-L6-v2<br/>384-dim embedding]
    E --> S[Cosine similarity<br/>vs 500 cached question embeddings]
    S --> T{Max similarity<br/>≥ 0.6?}
    T -->|Yes| INJECT["Inject top-3 past Q&A into system prompt:<br/><br/>Previously answered similar question:<br/>Q: What does the Quran say about patience?<br/>A: ...25 citations, thematic sections...<br/><br/>Use this as reference for the new question."]
    T -->|No| PLAIN[Use base system prompt unchanged]
    INJECT --> LLM[LLM sees rich context<br/>from premium past answers]
    PLAIN --> LLM
```

**Why this matters:** the cache was seeded with a big 120B model. When the local 8B model sees those cached answers as context, it produces answers **close to the 120B's quality** — because it's essentially paraphrasing + extending prior work, not generating from scratch.

---

## Phase 1 — Priming search (instant feedback, ~200ms)

Before the LLM runs at all, we do a quick keyword-based Neo4j query to surface 3-6 candidate verses. These appear as a cluster **within 200ms** so the user sees something happen immediately.

```mermaid
graph LR
    Q[User question] --> EXTRACT[Extract top 3 content words<br/>stripping stop words + question shells]
    EXTRACT --> CYPHER["Cypher:<br/>MATCH (k:Keyword)<br/>WHERE toLower(k.name) CONTAINS kw<br/>MATCH (k)&lt;-[:MENTIONS]-(v:Verse)<br/>RETURN v ORDER BY match_count DESC LIMIT 6"]
    CYPHER --> N4J[(Neo4j)]
    N4J --> RESP[6 candidate verses]
    RESP --> EVENT[graph_update SSE event]
    EVENT --> UI[Center cluster appears<br/>+ amber halos on sphere]
```

Completely independent of the LLM — just gives the user "something" to look at while the model thinks.

---

## Phase 2 — The agent tool loop

This is where the actual "intelligence" happens. The LLM is given a system prompt + 5 tool schemas and told to explore the knowledge graph to answer the question.

```mermaid
graph TB
    START[Agent loop begins] --> CALL[Send messages + tools to LLM]
    CALL --> RESP{LLM response}
    RESP -->|Tool calls present| EXEC[Execute each tool against Neo4j]
    RESP -->|Only text, no tools| CHECK{Enforced multi-tool?}
    CHECK -->|Yes, still need more| NUDGE[Inject 'You must call X tool' message]
    CHECK -->|No, done| FINAL[Move to final answer]
    NUDGE --> CALL
    EXEC --> COMPRESS[Compress results:<br/>top 6 verses keep full text,<br/>rest truncated to 60 chars]
    COMPRESS --> EMIT[Emit SSE events:<br/>tool + graph_update + etymology_panel]
    EMIT --> UPDATE[UI: cluster grows,<br/>more halos light up,<br/>tool indicator in chat]
    UPDATE --> APPEND[Append tool result<br/>to message history]
    APPEND --> CHECK_MAX{Turn &lt; 8?}
    CHECK_MAX -->|Yes| CALL
    CHECK_MAX -->|No| FINAL
```

### The 5 core tools (in the free version)

```mermaid
graph LR
    LLM{LLM picks<br/>which tool} --> SK[search_keyword<br/>'find all verses mentioning patience']
    LLM --> SS[semantic_search<br/>'find verses conceptually related to<br/>enduring hardship']
    LLM --> TT[traverse_topic<br/>'multi-keyword graph walk']
    LLM --> GV[get_verse<br/>'deep-dive on verse 2:155']
    LLM --> ES[explore_surah<br/>'themes of Al-Baqarah']

    SK --> Q1["Cypher:<br/>MATCH (v:Verse)-[:MENTIONS]->(k:Keyword)<br/>WHERE k.name = 'patience'<br/>RETURN v, k"]
    SS --> Q2["Vector index:<br/>Neo4j verse_embedding<br/>cosine similarity,<br/>top_k=20"]
    TT --> Q3["Multi-hop Cypher:<br/>direct keyword matches<br/>+ 1-hop neighbors<br/>+ 2-hop related verses"]
    GV --> Q4["Cypher:<br/>MATCH (v:Verse {reference: '2:155'})<br/>-[:RELATED_TO|SUPPORTS|...]->(other)<br/>RETURN all connections"]
    ES --> Q5["Cypher:<br/>MATCH (s:Sura {number: 2})<br/>-[:CONTAINS]->(v)<br/>RETURN themes"]

    Q1 --> N4J[(Neo4j)]
    Q2 --> N4J
    Q3 --> N4J
    Q4 --> N4J
    Q5 --> N4J
```

The full paid versions have **15 tools** including Arabic root analysis, word-level etymology, and polysemy lookups. The free version was trimmed to 5 because 14B local models choke on 15 tool schemas in their context window.

### What the LLM actually sees per turn

A typical 3-turn conversation has a message history that looks like this:

```mermaid
graph TB
    S1[System prompt<br/>~800 tokens<br/>Tool rules + citation mandate] --> S2
    S2[User: 'What does the Quran say about patience?'] --> S3
    S3[Assistant: 'I'll call search_keyword and semantic_search'<br/>+ tool_calls: 2] --> S4
    S4[Tool result 1: search_keyword<br/>15 verses, top 6 full text] --> S5
    S5[Tool result 2: semantic_search<br/>12 verses, top 6 full text] --> S6
    S6[Assistant: 'Let me also try traverse_topic'<br/>+ tool_calls: 1] --> S7
    S7[Tool result 3: traverse_topic<br/>20 verses across 2 hops] --> S8
    S8[Assistant: Final answer - 600 words, 12 citations]
```

Total context size: ~4,000 input tokens by the final turn.

---

## Phase 3 — Neo4j graph structure (what the tools actually query)

The Quran knowledge graph has 7 node types and 8+ relationship types:

```mermaid
graph TB
    SURA[Sura<br/>114 nodes] -->|CONTAINS| VERSE[Verse<br/>6,234 nodes<br/>text + arabicText + embedding]
    VERSE -->|MENTIONS<br/>41K edges<br/>TF-IDF weighted| KW[Keyword<br/>2,636 nodes]
    VERSE -->|MENTIONS_ROOT<br/>~100K edges<br/>surface forms + positions| AR[ArabicRoot<br/>1,223 nodes<br/>tri-literal roots]
    AR -->|DERIVES| LEMMA[Lemma<br/>4K+ nodes<br/>derived word forms]
    LEMMA -->|FOLLOWS_PATTERN| MP[MorphPattern<br/>100+ nodes<br/>wazn templates]
    AR -->|BELONGS_TO| SD[SemanticDomain<br/>30+ nodes<br/>mercy/knowledge/creation/etc.]

    VERSE -.->|RELATED_TO<br/>51K edges<br/>shared rare keywords<br/>capped 12/verse| VERSE
    VERSE -.->|SUPPORTS / ELABORATES /<br/>QUALIFIES / CONTRASTS / REPEATS<br/>typed semantic edges| VERSE
    VERSE -.->|NEXT_VERSE<br/>sequential order| VERSE
```

**Key numbers:**
- **6,234 verses** — every ayah of the Quran (excluding 9:128-129 per Khalifa)
- **2,636 keywords** — English lemmas weighted by TF-IDF rarity
- **1,223 Arabic roots** — tri-literal (3-letter) roots with surface forms in each verse
- **51K RELATED_TO edges** — verse-to-verse thematic connections (93.7% cross-surah)
- **Vector index** — 384-dim embeddings on every verse for semantic search

### What a tool result looks like

When `search_keyword('patience')` runs, Neo4j returns something like:

```json
{
  "keyword": "patience",
  "total_verses": 27,
  "by_surah": {
    "2": [
      {"verse_id": "2:153", "text": "O you who believe, seek help through steadfastness...", "score": 0.92},
      {"verse_id": "2:155", "text": "We will surely test you with some fear and hunger...", "score": 0.88}
    ],
    "3": [...],
    "16": [...]
  },
  "retrieval_quality": "good"
}
```

The tool compressor then rewrites this — keeps full text for the top 6 verses, trims the rest to 60 chars — before feeding it back to the LLM. This cuts tokens 60-70% while preserving the signal.

---

## Phase 4 — Citation check + retry (the hallucination guard)

```mermaid
graph TB
    ANS[Model produces answer] --> SCAN[Scan for [X:Y] citations with regex]
    SCAN --> COUNT[Count unique citations]
    COUNT --> QTYPE{Question type?}
    QTYPE -->|Simple lookup<br/>'verse 2:255'| ACCEPT[Accept, even 1 citation]
    QTYPE -->|Topical question<br/>'what about patience?'| CHECK{Citations ≥ 5?}
    CHECK -->|Yes| ACCEPT
    CHECK -->|No| RETRY[Inject: 'Your answer only has N citations.<br/>Add more thematic sections with more verse refs.']
    RETRY --> REGEN[LLM regenerates with same tool context]
    REGEN --> SCAN
    ACCEPT --> FETCH[Fetch full text of every cited verse from Neo4j]
    FETCH --> SSE[Stream done event with verse texts to browser]
    SSE --> UI[Citations become hoverable tooltips]
```

**Why this works:** the model has already run tools and has verses in its context. The "retry" doesn't re-search — it just re-writes the answer with more citation density from the verses already retrieved.

The full paid version (`app_full.py`) has **three additional hallucination checks**:
1. **Cross-encoder reranking** (retrieval_gate.py) — reranks tool results by relevance before the LLM sees them
2. **NLI verification** (citation_verifier.py) — checks that each cited verse actually *entails* the claim made about it
3. **Uncertainty probes** (uncertainty.py) — generates 5 answers via a cheap model, measures semantic entropy, abstains if high

---

## Phase 5 — The 3D visualization

Two layers that update in real-time as the agent loop runs:

```mermaid
graph TB
    subgraph STATIC["Static backdrop (rendered once at page load)"]
        PTS[6,234 point cloud<br/>Fibonacci sphere<br/>positioned by<br/>GALAXY_POSITIONS[verseId]]
        LABELS[114 surah label sprites<br/>fade with camera distance]
    end

    subgraph DYNAMIC["Dynamic layer (updates per SSE event)"]
        CLUSTER[Cluster nodes<br/>d3 force-directed<br/>at origin<br/>emerald spheres + labels]
        EDGES[Cluster edges<br/>with flowing particles]
        HALOS[Amber halo sprites<br/>at verse positions<br/>on the sphere surface]
        POINTS_TINT[Point cloud colors<br/>active verses → amber<br/>neighbors → faint amber]
    end

    SSE[SSE graph_update event<br/>nodes + links + active] --> EXTRACT[Parse verse IDs<br/>strip 'v:' prefix]
    EXTRACT --> ACTIVE[Active verse IDs]
    EXTRACT --> NEIGHBOR[Neighbor verse IDs]

    ACTIVE --> HALOS
    NEIGHBOR --> POINTS_TINT
    ACTIVE --> CLUSTER
    NEIGHBOR --> CLUSTER
```

**The insight:** the sphere is a *map of the entire Quran* — fixed, never moves. The cluster is a *dynamic working-memory view* — what the AI is thinking about *right now*. The halos are the bridge: they tell you "these cluster nodes live *there* on the map."

When you hover a cluster node:
1. Raycaster hits the 3d-force-graph node (not the point cloud)
2. Browser fetches verse data from local state (already fetched via `done` event)
3. Preview card positions near cursor with Arabic + English
4. Active highlight keywords get wrapped in `<mark>` tags

---

## What makes this different from generic RAG

Most "ask the Quran an AI" tools use retrieval-augmented generation:
1. Embed the question
2. Vector search for top-10 similar verses
3. Stuff them into a prompt
4. Ask GPT to answer

This app is **agentic**, not RAG:
1. The LLM *decides* which tool to use (keyword search? semantic search? graph traversal? Arabic root lookup?)
2. It runs multiple tools sequentially, using results from one to inform the next
3. Each tool returns **structured, typed data** — not just text chunks
4. The graph has **semantic relationships** (SUPPORTS, CONTRASTS, ELABORATES) the LLM can exploit

The result is much more like "a researcher with a powerful search tool" than "autocomplete with context." It's slower (15-60s per question) but produces answers with **25-40 citations** organized thematically — the kind of output you'd expect from a scholar, not a chatbot.

---

## Deployment modes

The same core architecture runs in four configurations:

| Mode | Model | Extra features | Cost per query | File |
|---|---|---|---|---|
| **Free** | Local Qwen3-8B via Ollama | None | $0 | `app_free.py` |
| **Lite** | Claude Haiku 4.5 | None | ~$0.04 | `app_lite.py` |
| **Standard** | Claude Sonnet 4.5 | Citation retry | ~$0.14 | `app.py` |
| **Full** | Claude Sonnet 4.5 | + uncertainty probes, NLI verification, cross-encoder rerank | ~$0.18 | `app_full.py` |

All four share:
- The same Neo4j graph
- The same 500-entry cache
- The same 3D UI
- The same tool definitions

---

## Source map (key files)

| File | Purpose |
|---|---|
| `index.html` | Browser frontend — 3D sphere, chat, hover cards, SSE consumer |
| `app_free.py` | FastAPI server, agent loop, tool dispatch, caching orchestration |
| `chat.py` | The 15 tool function bodies (Cypher queries) + default system prompt |
| `answer_cache.py` | Embedding-based cache lookup and save |
| `tool_compressor.py` | Trims tool results to save tokens |
| `retrieval_gate.py` | Cross-encoder reranking (paid mode) |
| `citation_verifier.py` | NLI-based citation checking (paid mode) |
| `uncertainty.py` | Semantic entropy uncertainty probe (paid mode) |
| `prompts/system_prompt_free.txt` | Instructions for the free-tier model |
| `data/answer_cache.json` | 500-entry cache |
| `pipeline_config.yaml` | All tunable knobs (top_k, thresholds, etc.) |

That's the whole picture. Every piece is replaceable — you could swap Qwen3 for Llama, Neo4j for PostgreSQL+pgvector, Ollama for vLLM. The architecture is deliberately modular so the moving pieces can evolve independently.
