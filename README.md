# Quran Knowledge Graph

An AI-powered Quran explorer connecting all 6,234 verses from Rashad Khalifa's translation (*The Final Testament*) through a Neo4j knowledge graph with Arabic morphology, etymology, and 3D visualization. Ask questions in natural language and get answers grounded in specific verse citations.

## What It Does

- **Conversational search** -- Ask anything about the Quran. An agentic LLM explores the graph using 15 tools and returns answers citing specific verses `[2:255]` that you can hover over to read the full text in English and Arabic.
- **Arabic root analysis** -- Trace connections through tri-literal roots. "Book" (kitab), "prescribed" (kutiba), and "wrote" (kataba) all share root k-t-b, revealing relationships invisible in English.
- **Word-level etymology** -- Look up any Arabic word's root, lemma, morphological pattern (wazn), and all Quranic occurrences. Explore semantic fields and polysemy (wujuh).
- **3D graph visualization** -- 6,234 verses arranged on a Fibonacci sphere by surah. Connections light up in real-time as the agent explores. Fly through with WASD controls.
- **Hallucination reduction pipeline** -- Five phases: retrieval gating, citation verification, constrained output, semantic entropy, and prompt refinement.
- **4 deployment modes** -- Full (max accuracy), Default (balanced), Lite (cheapest API), and Free (local Ollama, $0).

## Quick Start

### Prerequisites
- Python 3.10+
- Neo4j Desktop (Community Edition) with a database named `quran`
- Anthropic API key or OAuth token (for paid versions)
- Ollama (for free version only)

### Install
```bash
pip install anthropic neo4j fastapi uvicorn python-dotenv sentence-transformers scikit-learn nltk pyyaml numpy requests
```

### Configure
Create a `.env` file:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
ANTHROPIC_API_KEY=sk-ant-api03-your_key_here
# OR use OAuth token (takes priority over API key):
# ANTHROPIC_OAUTH_TOKEN=sk-ant-oat01-your_token_here
PDF_PATH=C:\path\to\your\quran.pdf
```

### Build the Graph
Run in order -- each step depends on the previous:
```bash
python parse_quran.py            # PDF -> data/verses.json
python build_graph.py            # TF-IDF keywords + edges -> CSVs
python import_neo4j.py           # CSVs -> Neo4j
python embed_verses.py           # Vector embeddings (all-MiniLM-L6-v2)
python migrate_graph.py          # Schema fixes, orphan verses, stopword cleanup
python load_arabic.py            # Arabic text (Hafs reading) -> Verse nodes
python build_arabic_roots.py     # Morphology -> ArabicRoot nodes + edges
python build_word_tokens.py      # Word-level parsing -> Lemma, MorphPattern nodes
python build_semantic_domains.py # Semantic field groupings
python build_wujuh.py            # Polysemy data
python import_etymology.py       # All etymology data -> Neo4j
python classify_edges.py         # Typed edges (SUPPORTS, ELABORATES, etc.)
```

### Run
```bash
python app.py          # Default version -- http://localhost:8081
python app_full.py     # Full hallucination reduction -- http://localhost:8083
python app_lite.py     # Cheapest API version (Haiku) -- http://localhost:8084
python app_free.py     # Free local version (Ollama) -- http://localhost:8085
```

## Deployment Modes

| Version | File | Port | Model | Cost/Question | Features |
|---------|------|------|-------|---------------|----------|
| **Default** | `app.py` | 8081 | Sonnet | ~$0.05-0.10 | Tool compression, answer cache |
| **Full** | `app_full.py` | 8083 | Sonnet | ~$0.15-0.35 | All hallucination reduction phases |
| **Lite** | `app_lite.py` | 8084 | Haiku | ~$0.01-0.03 | Cheapest API option, 1536 max tokens |
| **Free** | `app_free.py` | 8085 | Qwen 2.5 14B | $0.00 | Runs locally via Ollama |

The free version supports any Ollama model with tool-use:
```bash
python app_free.py --model qwen2.5:14b-instruct-q6_K
python app_free.py --model qwen2.5:32b-instruct-q3_K_M --port 8086
```

## Architecture

```
User Question
    |
POST /chat (FastAPI)
    |
+-- Answer Cache lookup (embedding similarity, data/answer_cache.json)
|   |
|   +-- Cache hit? -> inject past answers into system prompt
|
+-- [Full only] Semantic Entropy (5 Haiku probes -> uncertainty score)
|
Agentic Tool-Use Loop (up to 15 turns)
    |
    +-- LLM decides which tools to call
    +-- dispatch_tool() -> Neo4j Cypher queries
    +-- Tool Compressor strips results before feeding back (60-70% token savings)
    +-- Retrieval Gate reranks results (cross-encoder, lost-in-middle reorder)
    +-- Loop until LLM has enough context
    |
[Full only] Citation density check -> re-generate if < 30% cited
[Full only] NLI citation verification (DeBERTa entailment)
    |
SSE Event Stream --> Frontend (index.html)
    |
    +-- text: markdown deltas -> chat bubble
    +-- tool: expandable tool execution details
    +-- graph_update: nodes + links -> 3D visualization
    +-- etymology_panel: structured word analysis cards
    +-- done: batch verse texts for hoverable tooltips
    +-- error: error messages

Answer Cache save (question + response + cited verses)
```

## Cost Optimization

Two systems reduce API costs:

### Tool Result Compressor (`tool_compressor.py`)
Before feeding tool results back to the LLM, the compressor:
- Truncates verse text to 100 characters (from ~300-400)
- Strips Arabic text entirely (Claude doesn't need it to cite verses)
- Caps keyword lists at 8 items
- Drops embedding vectors

Typical savings: **60-70% fewer tokens per tool result**. A `search_keyword` result drops from ~19,000 chars to ~5,600 chars.

### Answer Cache (`answer_cache.py`)
Stores every Q&A pair with its embedding in `data/answer_cache.json`. On new questions:
- Searches for past answers with >= 60% similarity
- Injects up to 3 matching past answers into the system prompt
- The LLM can reuse verse citations and analysis instead of making full tool loops
- Near-duplicate questions (> 95% similarity) update the existing entry
- Shared across all 4 deployment modes
- Capped at 500 entries, oldest evicted first

Check cache stats: `GET /cache-stats` (on the default version at port 8081).

## Hallucination Reduction Pipeline

Five phases, all enabled in `app_full.py`:

| Phase | Module | What It Does | Cost |
|-------|--------|-------------|------|
| **2. Retrieval Gate** | `retrieval_gate.py` | Cross-encoder reranking (ms-marco-MiniLM-L-6-v2), lost-in-middle U-shape reorder, quality gating ("good"/"marginal"/"poor") | Free (local) |
| **3. Citation Verifier** | `citation_verifier.py` | NLI entailment check (nli-deberta-v3-xsmall) -- verifies each cited verse actually supports its claim | Free (local) |
| **4. Constrained Output** | `prompts/system_prompt.txt` | System prompt enforces mandatory citation rules, exhaustive search mandate, honesty about limits | Free (prompt) |
| **5. Uncertainty** | `uncertainty.py` | 5 Haiku probe responses -> semantic entropy. High entropy triggers caution in system prompt or abstention | ~$0.01-0.03 |
| **6. Citation Retry** | `app_full.py` | If < 30% of sentences have citations, re-generates with stricter grounding instructions | ~$0.05-0.15 |

## Graph Schema (Neo4j)

### Nodes

| Label | Count | Key Properties |
|-------|-------|----------------|
| `Verse` | 6,234 | `verseId`, `reference`, `surah`, `text`, `arabicText`, `arabicPlain`, `embedding` (384-dim) |
| `Keyword` | 2,636 | `keyword` (lemmatized) |
| `ArabicRoot` | 1,223 | `root` (spaced tri-literal), `rootBW` (Buckwalter), `gloss` |
| `Lemma` | 4,762 | `lemma`, `root`, `gloss`, `pattern` |
| `Sura` | 114 | `number`, `name` |
| `MorphPattern` | 38 | `pattern` (wazn), `label` |
| `SemanticDomain` | 28 | `domainId`, `nameEn`, `nameAr` |

### Relationships

| Type | Count | Description |
|------|-------|-------------|
| `IN_VERSE` | 77,400 | Word token -> Verse occurrence |
| `HAS_LEMMA` | 74,095 | Word token -> Lemma |
| `RELATED_BY_ROOT` | 73,666 | Verse <-> Verse (shared Arabic roots, TF-IDF) |
| `RELATED_TO` | 51,798 | Verse <-> Verse (shared English keywords, capped 12/verse, 93.7% cross-surah) |
| `MENTIONS` | 41,138 | Verse -> Keyword (TF-IDF weighted) |
| `MENTIONS_ROOT` | 36,366 | Verse -> ArabicRoot (with surface forms) |
| `FOLLOWS_PATTERN` | 24,579 | Word -> MorphPattern |
| `CONTAINS` | 6,234 | Sura -> Verse |
| `NEXT_VERSE` | 6,233 | Verse -> Verse (sequential) |
| `SUPPORTS` | 5,464 | Verse -> Verse (typed: independent evidence) |
| `DERIVES_FROM` | 4,128 | Lemma -> ArabicRoot |
| `CONTRASTS` | 697 | Verse -> Verse (typed: complementary perspectives) |
| `REPEATS` | 626 | Verse -> Verse (typed: near-verbatim) |
| `ELABORATES` | 357 | Verse -> Verse (typed: expands detail) |
| `QUALIFIES` | 247 | Verse -> Verse (typed: adds condition/exception) |
| `IN_DOMAIN` | 211 | ArabicRoot -> SemanticDomain |

### Indexes

| Name | Type | Target |
|------|------|--------|
| `verse_id` | UNIQUE | `Verse.verseId` |
| `verse_embedding` | VECTOR | `Verse.embedding` (cosine, 384-dim) |
| `kw_id` | UNIQUE | `Keyword.keyword` |
| `arabic_root_unique` | UNIQUE | `ArabicRoot.root` |
| Fulltext | FULLTEXT | `Verse.arabicPlain` |

## Claude's 15 Tools (`chat.py`)

### Search & Navigation (9 tools)

| Tool | Purpose |
|------|---------|
| `search_keyword(keyword)` | Find all verses mentioning a keyword (TF-IDF lemma match, grouped by surah) |
| `semantic_search(query, top_k)` | Conceptual search via vector embeddings (cosine similarity) |
| `get_verse(verse_id)` | Deep-dive into a verse: text, Arabic, keywords, connections, typed edges |
| `traverse_topic(keywords, hops)` | Multi-keyword search + graph traversal (1-2 hops via RELATED_TO) |
| `find_path(verse_id_1, verse_id_2)` | Shortest thematic path between two verses |
| `explore_surah(surah_number)` | Map a surah's content and cross-surah connections |
| `query_typed_edges(verse_id, types)` | Find verses by relationship type (SUPPORTS, ELABORATES, QUALIFIES, CONTRASTS, REPEATS) |
| `search_arabic_root(root)` | Find all verses containing a specific Arabic root (accepts Arabic or Buckwalter) |
| `compare_arabic_usage(root)` | Compare how different derived forms of one root are used across contexts |

### Etymology & Word Analysis (6 tools)

| Tool | Purpose |
|------|---------|
| `lookup_word(word)` | Look up any Arabic word -- root, lemma, pattern, morphology, occurrences |
| `explore_root_family(root)` | Full derivative tree: all lemmas, patterns, semantic domains from one root |
| `get_verse_words(verse_id)` | Word-by-word grammatical breakdown (root, lemma, pattern, POS, gloss) |
| `search_semantic_field(domain)` | Find all roots/words in a semantic domain (e.g. "mercy", "knowledge") |
| `lookup_wujuh(root)` | Show all distinct meanings (wujuh) of a polysemous root |
| `search_morphological_pattern(pattern)` | Find words by morphological pattern (wazn) or verbal form |

## 3D Visualization (`index.html`)

### Libraries
- Three.js r160 -- 3D rendering
- 3d-force-graph v1 -- force-directed layout
- marked.js v9 -- markdown rendering in chat bubbles

### Galaxy Layout
- 6,234 verses pre-positioned on a **Fibonacci sphere** (radius 600)
- 114 surah centers distributed via `fibSphere(i, 114, R)`
- Verses within each surah form perpendicular rings around the surah center
- Ring radius: `min(20 + n * 0.55, 90)` where n = verse count

### Node Colors
| Type | Color |
|------|-------|
| Verse | Hue by surah number |
| Keyword | Gold (#f59e0b) |
| ArabicRoot | Amber |
| Lemma | Teal |
| SemanticDomain | Purple |
| MorphPattern | Pink |

### Controls
- **Orbit**: drag
- **Zoom**: scroll
- **Fly**: WASD + Q/E (up/down)
- **Click node**: info card with verse text (English + Arabic)
- **Hover citation**: tooltip with full verse text

### Color Scheme (Dark Theme)
- Background: `#060a14`
- Cards: `#0f1a2e`
- Primary accent: `#10b981` (emerald)
- Secondary accent: `#f59e0b` (gold)

## Evaluation & Optimization

### QIS Score (Composite Metric)
```
QIS = 0.35 * citation_recall + 0.25 * citation_precision + 0.30 * grounding_rate + 0.10 * answer_relevance
```

### Test Dataset
`test_dataset.json` -- 218 questions (100 standard, 50 edge cases, 50 unanswerable), each with expected citations and difficulty rating.

### Running Evaluation
```bash
python evaluate.py                                  # All questions
python evaluate.py --ids q01_forgiveness,q05_nineteen --out results.json
```

### Autoresearch (Hyperparameter Optimization)
Optuna TPE Bayesian search over `pipeline_config.yaml` knobs:
```bash
python autoresearch.py --trials 50          # Claude API (~$2-3/trial)
python autoresearch_local.py --trials 100   # Ollama (free)
python autoresearch_dashboard.py            # Monitor at http://localhost:8082
```

Best result: QIS 0.1688 after 100 local trials (27% improvement over baseline).

## Configuration

All tunable parameters in `pipeline_config.yaml`:

```yaml
llm:
  model: claude-sonnet-4-5
  max_tokens: 3072
  temperature: 1.0

retrieval:
  semantic_search:
    default_top_k: 35
  traverse_topic:
    seed_limit: 15
    hop1_limit: 40
    max_hops: 2
  # ... per-tool limits

arabic:
  morphology_backend: corpus    # "corpus" | "camel" | "isri"
  root_min_verses: 2
  root_max_verses: 500

scoring:
  related_to_min_score: 0.3
  mentions_min_score: 0.04
```

Reload config at runtime: `config.reload()`

## File Structure

```
quran-graph-standalone/
  # ── App versions ──
  app.py                    Default (Sonnet, balanced)         :8081
  app_full.py               Full hallucination reduction       :8083
  app_lite.py               Cheapest (Haiku)                   :8084
  app_free.py               Free (local Ollama)                :8085
  server.py                 OpenAI-compatible API              :8100
  
  # ── Core ──
  chat.py                   15 tool functions + dispatch + system prompt
  config.py                 Config accessors from pipeline_config.yaml
  index.html                3D graph visualization (Three.js SPA)
  stats.html                Statistical insights dashboard
  
  # ── Cost optimization ──
  answer_cache.py           Embedding-based Q&A cache
  tool_compressor.py        Tool result compression (60-70% token savings)
  
  # ── Hallucination reduction ──
  retrieval_gate.py         Cross-encoder reranking + quality gating
  citation_verifier.py      NLI citation verification
  uncertainty.py            Semantic entropy (5 Haiku probes)
  
  # ── Data pipeline ──
  parse_quran.py            PDF -> data/verses.json
  build_graph.py            TF-IDF keywords + edges -> CSVs
  import_neo4j.py           CSVs -> Neo4j
  embed_verses.py           Vector embeddings
  migrate_graph.py          Schema fixes + quality improvements
  load_arabic.py            Arabic text (Hafs reading)
  build_arabic_roots.py     Morphology -> ArabicRoot nodes
  build_word_tokens.py      Word-level parsing -> Lemma nodes
  build_semantic_domains.py Semantic field groupings
  build_wujuh.py            Polysemy data
  import_etymology.py       All etymology data -> Neo4j
  classify_edges.py         Typed relationship classification
  
  # ── Evaluation ──
  evaluate.py               QIS scoring against test dataset
  autoresearch.py           Optuna optimization (Claude API)
  autoresearch_local.py     Optuna optimization (Ollama, free)
  autoresearch_dashboard.py Real-time optimization monitor    :8082
  test_dataset.json         218 test questions with expected citations
  
  # ── Config ──
  pipeline_config.yaml      All tunable parameters
  prompts/system_prompt.txt System prompt for Claude
  .env                      API keys + database credentials
  
  # ── Data ──
  data/
    verses.json             6,234 parsed verses
    verse_nodes.csv         Neo4j import: verses
    keyword_nodes.csv       Neo4j import: keywords
    verse_keyword_rels.csv  Neo4j import: MENTIONS edges
    verse_related_rels.csv  Neo4j import: RELATED_TO edges
    arabic_root_nodes.csv   Neo4j import: ArabicRoot nodes
    verse_root_rels.csv     Neo4j import: MENTIONS_ROOT edges
    quran-arabic-raw.json   Arabic text source (Hafs reading)
    quran-morphology.txt    Morphological analysis data
    answer_cache.json       Cached Q&A pairs (auto-generated)
```

## Tech Stack

- **Graph database**: Neo4j 5+ (Cypher, vector index, fulltext index)
- **AI (paid)**: Claude Sonnet/Haiku via Anthropic API (agentic tool-use)
- **AI (free)**: Qwen 2.5 via Ollama (local, tool-use compatible)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- **Reranking**: cross-encoder (ms-marco-MiniLM-L-6-v2)
- **NLI**: cross-encoder (nli-deberta-v3-xsmall)
- **NLP**: scikit-learn TF-IDF, NLTK WordNet lemmatization
- **Frontend**: Three.js r160, 3d-force-graph v1, marked.js v9, vanilla JS
- **Backend**: FastAPI + uvicorn, SSE streaming via daemon thread + SimpleQueue
- **Optimization**: Optuna TPE Bayesian search
- **Translation**: Rashad Khalifa's *Quran: The Final Testament* (6,234 verses)
- **Arabic**: Hafs reading (Uthmani script) from the Quranic Arabic Corpus

## Translation Context

This project uses Rashad Khalifa's translation exclusively. Verses 9:128-129 are excluded per this translation. The system prompt acknowledges Khalifa-specific interpretations (the mathematical miracle of 19, the Messenger of the Covenant). Arabic text is the Hafs reading (the dominant qira'a used by ~95% of the Muslim world).

## License

MIT
