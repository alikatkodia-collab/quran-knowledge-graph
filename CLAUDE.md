# CLAUDE.md

## Project Overview

Quran Knowledge Graph — an AI-powered system that connects all 6,234 Quran verses through a thematic knowledge graph, provides conversational exploration via Claude AI, and runs an autonomous research engine that discovers novel theological insights through syllogistic deduction.

## Architecture

### Tech Stack

- **Backend**: FastAPI + uvicorn (Python 3.11)
- **Database**: Neo4j (graph DB with 6,234 verse nodes, 2,661 keyword nodes, 106K+ edges)
- **AI**: Anthropic Claude API (Sonnet), sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- **NLP**: scikit-learn (TF-IDF), NLTK (lemmatization), spaCy (proposition extraction)
- **Frontend**: Vanilla JS, Three.js + 3d-force-graph, Chart.js, marked.js, Reveal.js
- **Deployment**: Docker, Heroku (Procfile), Railway, GitHub Pages (static)

### Graph Schema (Neo4j)

**Nodes:**
- `:Verse {verseId, surah, verseNum, surahName, text, embedding: vector(384)}`
- `:Keyword {keyword}`

**Relationships:**
- `(Verse)-[:MENTIONS {score}]->(Keyword)` — TF-IDF weighted
- `(Verse)-[:RELATED_TO {score}]-(Verse)` — Thematic similarity via shared keywords

## Repository Structure

```
├── app.py                  # Main FastAPI server (port 8081), SSE streaming chat + 3D viz
├── chat.py                 # Claude agent with 6 graph tools, system prompt, SSE dispatch
├── graph_qa.py             # Standalone Q&A mode (no web, CLI-based)
├── server.py               # OpenAI-compatible API wrapper (for Open WebUI)
├── serve.py                # Lightweight static server (no Neo4j required)
├── build_graph.py          # TF-IDF keyword extraction + edge building pipeline
├── parse_quran.py          # PDF → data/verses.json (6,234 verses)
├── embed_verses.py         # Generate verse embeddings → Neo4j vector index
├── import_neo4j.py         # Load CSVs into Neo4j with indexes
├── explore.py              # CLI graph explorer (verse, keyword, path, cluster)
├── ui.py                   # UI utilities
├── comparator_api.py       # API for parallel passages (/api/comparator/*)
├── deductions_api.py       # API for deductions + stats (/api/deductions/*)
├── index.html              # Main chat + 3D graph interface
├── deductions.html         # 3D meta-knowledge graph visualization
├── visualizations.html     # 9-chart statistical dashboard
├── stats.html              # Graph topology + keyword statistics
├── comparator.html         # Parallel passages viewer
├── explorer.html           # Tension/contradiction browser
├── presentation.html       # Reveal.js slide deck (16 slides)
├── data/                   # Generated data: verses.json, CSVs for Neo4j import
├── dist/                   # Pre-built static HTML for GitHub Pages
├── scripts/build_static.py # Static site builder for GitHub Pages
├── autoresearch/           # 9 concurrent infinite research loops (see below)
├── requirements.txt        # Python dependencies
├── package.json            # Node dep: pptxgenjs
├── Dockerfile              # Python 3.11 slim, port 8080
├── Procfile                # Heroku: uvicorn app:app
└── .gitignore
```

### Autoresearch Engine (`autoresearch/`)

9 concurrent infinite loops orchestrated by `continuous_runner.py`:

| Loop | File | Purpose |
|------|------|---------|
| 1 | `infinite_deduction.py` | Syllogistic deduction (~600/round), parameter mutation |
| 2 | `infinite_analysis.py` | Categorize deductions into 13 theological themes |
| 3 | `metagraph_loop.py` | Karpathy-style optimization of analysis parameters |
| 4 | `contradiction_loop.py` | Detect theological tensions/paradoxes |
| 5 | `deepening_loop.py` | Extend deductions into 4-5 hop chains |
| 6 | `narrative_loop.py` | Map narrative arcs within/across surahs |
| 7 | `rhetorical_loop.py` | Ring compositions, repetition patterns |
| 8 | `intertextuality_loop.py` | Parallel passages, progressive revelation |
| 9 | `linguistics_loop.py` | Word symmetries, co-occurrence networks |

Core engine: `deduction_engine.py` (proposition extraction, syllogistic rules, novelty scoring), `evaluate.py` (7-metric scoring system).

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet
python -m spacy download en_core_web_sm

# Data pipeline (one-time, sequential)
python parse_quran.py          # PDF → data/verses.json
python build_graph.py          # Builds CSVs in data/
python import_neo4j.py         # Import to Neo4j (requires running Neo4j instance)
python embed_verses.py         # Vector embeddings in Neo4j

# Run web server
python app.py                  # http://localhost:8081

# Run lightweight server (no Neo4j needed)
python serve.py                # Serves static pages + deductions/comparator APIs

# Run autoresearch
python autoresearch/continuous_runner.py --hours 0   # All 9 loops, forever

# CLI tools
python explore.py verse 2:255
python explore.py keyword covenant
python explore.py path 2:255 112:1
python graph_qa.py             # Interactive Q&A in terminal
```

## Environment Variables

- `ANTHROPIC_API_KEY` — Required for chat and autoresearch
- `NEO4J_URI` — Neo4j connection URI (default: `bolt://localhost:7687`)
- `NEO4J_USER` — Neo4j username (default: `neo4j`)
- `NEO4J_PASSWORD` — Neo4j password
- `PORT` — Server port for deployment (default: 8081 for app.py, 8080 for Docker)

## Key Conventions

### Code Style
- Python files use standard library patterns; no strict linter enforced
- HTML files are self-contained (inline CSS/JS) with a dark theme (`#060a14` background, `#10b981` emerald accent, `#f59e0b` gold accent)
- Frontend uses vanilla JS — no build tools or bundlers

### Chat Agent Pattern
- Claude is given 6 tools that query Neo4j: `search_keyword`, `get_verse`, `traverse_topic`, `find_path`, `explore_surah`, `semantic_search`
- Tool dispatch is in `chat.py` — each tool runs a Cypher query and returns JSON
- Responses stream via SSE with JSON events: `text_delta`, `tool_call`, `graph_update`, `done`, `error`
- Citations are mandatory: every claim must include `[surah:verse]` inline references

### Autoresearch Pattern
- Each loop follows a Karpathy-style monotonic ratchet: mutate parameters → run → evaluate → keep if improved
- Output files are JSONL/JSON in `autoresearch/` directory
- Round logs are TSV files tracking progress over time
- `continuous_runner.py` manages process lifecycle and restarts dead loops

### Data Pipeline
- `data/verses.json` is the source of truth for verse text
- CSVs (`verse_nodes.csv`, `keyword_nodes.csv`, `verse_keyword_rels.csv`, `verse_related_rels.csv`) are intermediate artifacts for Neo4j import
- Graph construction uses TF-IDF with custom Quranic stopwords, max 12 edges per verse

## Testing

No formal test suite exists. Quality is validated through:
- Autoresearch evaluation metrics (7-metric scoring in `evaluate.py`)
- `loop.py` and `retrieval_loop.py` ran 600+ and 1000+ experiments respectively against `benchmark.json`
- Manual verification of deductions and graph traversals

## Deployment

- **Docker**: `docker build -t quran-graph . && docker run -p 8080:8080 -e NEO4J_URI=... -e ANTHROPIC_API_KEY=... quran-graph`
- **Heroku**: Uses `Procfile` — `uvicorn app:app --host 0.0.0.0 --port $PORT`
- **GitHub Pages**: Run `scripts/build_static.py` to generate `dist/` (static only, no chat)
- **Railway**: Full stack with Neo4j add-on ($0-5/mo)
