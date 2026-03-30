# Quran Knowledge Graph + Autonomous Research Engine

An AI-powered Quran explorer that connects all 6,234 verses through a thematic knowledge graph, combined with an **autonomous research engine** inspired by [Karpathy's AutoResearch](https://github.com/karpathy/autoresearch) that continuously discovers novel theological insights through syllogistic deduction.

![Status](https://img.shields.io/badge/status-active-10b981?style=flat-square)
![Deductions](https://img.shields.io/badge/deductions-118%2C000%2B-3b82f6?style=flat-square)
![Loops](https://img.shields.io/badge/infinite_loops-9-f59e0b?style=flat-square)

## Live Metrics

| Metric | Value |
|--------|-------|
| **Deductions discovered** | 118,000+ |
| **Discovery rate** | ~118,000 deductions/hour |
| **Verses analyzed** | 6,234 (100%) |
| **Verses referenced** | 5,744 (92%) |
| **Theological categories** | 13 |
| **Theme connections** | 76 cross-theme bridges |
| **Surahs covered** | 114 (all) |
| **Concurrent research loops** | 9 |
| **Graph quality score** | 81.74 (optimized from 75.47) |
| **Cluster coherence** | 59.87 (improved from 35.53) |

## What It Does

### Knowledge Graph
- **Conversational search** — Ask anything about the Quran. Claude explores the graph using 6 tools and returns answers citing specific verses `[2:255]`.
- **3D graph visualization** — 6,234 verses on a sphere by surah. Connections light up in real-time.
- **Statistical dashboard** — Interactive graph topology, keyword frequencies, cross-surah connections.

### Autonomous Research Engine (AutoResearch)
Treats each Quranic verse as an **axiom** and autonomously computes **syllogistic deductions** to surface novel theological insights — running indefinitely with no human intervention.

**9 concurrent infinite loops:**

| Loop | What It Discovers | Rate |
|------|------------------|------|
| **Deduction** | Transitive chains, shared-subject synthesis, 3-hop thematic bridges | ~600/round |
| **Analysis** | Categorization, quality scoring, meta-knowledge graph | Every 60s |
| **Meta-Graph Optimization** | AutoResearch on the analysis parameters themselves | 100 exp/cycle |
| **Contradiction Detection** | Mercy/justice tensions, conditional vs absolute claims | ~50/round |
| **Cluster Deepening** | Extended 4-5 hop chains from best discoveries | ~20/round |
| **Narrative Arcs** | Thematic flow within and across surahs | 5-10 surahs/round |
| **Rhetorical Structure** | Ring compositions, repetition patterns, oath-evidence pairs | 5-10 surahs/round |
| **Intertextuality** | Parallel passages, progressive revelation, compelling connections | 3-5 topics/round |
| **Statistical Linguistics** | Vocabulary richness, word symmetries, co-occurrence networks | 10 surahs/round |

### Key Discoveries

1. **God's Nature is the structural hub** — The meta-knowledge graph shows "monotheism" connecting to ALL other themes, confirming tawhid as the Quran's organizing principle (10,695 deductions bridging to Prophecy alone).

2. **Physical rescue = Prophetic rescue** — The engine found that the Quran structurally parallels saving people from storms with saving them from spiritual darkness ([31:32] → [6:63] → [14:5]).

3. **Legislative spine** — Dietary law, warfare, marriage, and finance share a single "prohibition" backbone running through [5:3] → [9:29] → [4:25] → [2:282].

4. **Prostration as cosmic dividing line** — The act of prostration connects worship ([76:26]) to Satan's rebellion ([18:50]) to human social relationships ([60:1]).

5. **Most connected surah pair**: Surahs 2 (Heifer) and 7 (Purgatory) — 391 cross-deductions at 73.96 avg quality.

## Architecture

```
Parse PDF → Extract Keywords → Build Graph → Embed Verses → Store in Neo4j → Query via Claude
     ↓            ↓                ↓              ↓               ↓              ↓
verses.json   2,661 keywords  106,393 edges   384-dim vectors  Neo4j graph   6 agent tools

                        ↓ AutoResearch Layer ↓

Infinite Deduction Loop → Infinite Analysis → Meta-Graph Optimization
         ↓                      ↓                     ↓
  118,000+ deductions    13 categories          Self-tuning params
         ↓                      ↓
  Contradiction Detection   Narrative Arcs → Rhetorical Analysis
         ↓                      ↓                     ↓
  Mercy/justice tensions   Surah profiles    Ring compositions
         ↓
  Cluster Deepening → Intertextuality → Linguistics
         ↓                    ↓                ↓
  4-5 hop chains     Parallel passages   Word symmetries
```

## Web Interface

| Page | URL | Description |
|------|-----|-------------|
| Chat | `/` | AI chat + 3D verse graph |
| Stats | `/stats` | Statistical dashboard |
| Deductions | `/deductions` | 3D meta-knowledge graph + insights |
| Visualizations | `/visualizations` | 9-chart interactive dashboard |
| Presentation | `/presentation` | 16-slide Reveal.js deck |

## Quick Start

### Prerequisites
- Python 3.10+
- Neo4j (for chat features)
- Anthropic API key (for chat features)

### Install & Build
```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet
python -m spacy download en_core_web_sm

# Build the knowledge graph (one-time)
python parse_quran.py        # PDF → verses.json
python build_graph.py        # Extract keywords & edges
python import_neo4j.py       # Load into Neo4j
python embed_verses.py       # Generate embeddings
```

### Run the Web UI
```bash
python app.py                # http://localhost:8081
```

### Run the AutoResearch Engine
```bash
# Run all 9 loops indefinitely
python autoresearch/continuous_runner.py --hours 0

# Or run individual loops
python autoresearch/infinite_deduction.py --max-hours 0
python autoresearch/infinite_analysis.py --max-hours 0
python autoresearch/contradiction_loop.py --max-hours 0
```

### Run Parameter Optimization
```bash
# Optimize graph construction (Karpathy-style AutoResearch)
python autoresearch/loop.py --max-experiments 500 --strategy mixed

# Optimize retrieval pipeline
python autoresearch/retrieval_loop.py --max-experiments 1000
```

## AutoResearch Optimization Results

### Graph Construction (600 experiments)
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Cluster coherence | 35.53 | 59.87 | **+24.34** |
| Cross-surah ratio | 93.72 | 95.41 | +1.69 |
| Composite score | 75.47 | 81.74 | **+6.27** |

Key finding: **2.5x more edges per verse** (12→28) is the single biggest lever.

### Retrieval Pipeline (1,000 experiments)
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Composite score | 58.99 | 68.44 | **+9.45** |
| Precision | 19.0 | 31.7 | **+12.7** |

Key finding: **Less is more** — fewer, higher-quality results beat broad search.

## Tech Stack

- **Graph Database**: Neo4j (with vector index)
- **AI**: Claude Sonnet 4.5 via Anthropic API (agentic tool-use)
- **NLP**: scikit-learn TF-IDF, NLTK lemmatization, spaCy
- **Embeddings**: Sentence-transformers (all-MiniLM-L6-v2)
- **Frontend**: Three.js + ForceGraph3D, Chart.js, Reveal.js
- **Backend**: FastAPI + uvicorn

## Project Structure

```
├── app.py                    # FastAPI web server
├── chat.py                   # Claude agent with 6 graph tools
├── build_graph.py            # TF-IDF keyword extraction + edge building
├── deductions_api.py         # API endpoints for deductions UI
├── index.html                # Chat + 3D graph UI
├── deductions.html           # Deductions meta-graph UI
├── visualizations.html       # Chart.js dashboard
├── presentation.html         # Reveal.js slide deck
├── data/
│   ├── verses.json           # 6,234 verses
│   ├── verse_nodes.csv       # Graph nodes
│   ├── keyword_nodes.csv     # 2,661 keywords
│   ├── verse_keyword_rels.csv # MENTIONS edges
│   └── verse_related_rels.csv # RELATED_TO edges
├── autoresearch/
│   ├── continuous_runner.py  # Master orchestrator (9 loops)
│   ├── infinite_deduction.py # Syllogistic deduction engine
│   ├── infinite_analysis.py  # Real-time categorization + scoring
│   ├── metagraph_loop.py     # AutoResearch on analysis params
│   ├── contradiction_loop.py # Mercy/justice tension detection
│   ├── deepening_loop.py     # 4-5 hop chain extension
│   ├── narrative_loop.py     # Surah narrative arc analysis
│   ├── rhetorical_loop.py    # Ring composition + repetition
│   ├── intertextuality_loop.py # Parallel passages + echoes
│   ├── linguistics_loop.py   # Statistical linguistic analysis
│   ├── loop.py               # Graph param optimization
│   ├── retrieval_loop.py     # Retrieval param optimization
│   ├── evaluate.py           # 7-metric scoring engine
│   ├── benchmark.json        # 20 ground-truth clusters
│   ├── best_deductions.json  # Top 100 deductions
│   ├── meta_knowledge_graph.json # Theme-to-theme graph
│   ├── CURATED_INSIGHTS.md   # Human-readable analysis
│   └── claude_analysis.md    # Claude's reasoning about patterns
├── Dockerfile                # Container deployment
├── HOSTING_PLAN.md           # Deployment guide
└── scripts/
    └── build_static.py       # Static site builder for GitHub Pages
```

## Deployment

See [HOSTING_PLAN.md](HOSTING_PLAN.md) for full deployment options:
- **GitHub Pages** (free, static pages)
- **Railway.app** ($0-5/mo, full stack)
- **Docker Compose** (self-hosted)
- **Hybrid** (static + API)

## Inspired By

- [Karpathy's AutoResearch](https://github.com/karpathy/autoresearch) — autonomous ML experimentation framework
- The monotonic ratchet pattern: only improvements are kept, regressions discarded
- The principle of treating verses as formal axioms for syllogistic reasoning

## License

MIT
