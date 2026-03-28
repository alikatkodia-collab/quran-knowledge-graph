# Quran Knowledge Graph

An interactive AI-powered Quran explorer that connects all 6,234 verses from Rashad Khalifa's translation (*The Final Testament*) through a thematic knowledge graph. Ask questions in natural language and get answers grounded in actual verses, with a 3D visualization showing how they connect.

![Dark themed web app with chat on the left and 3D graph on the right](https://img.shields.io/badge/status-active-10b981?style=flat-square)

## What It Does

- **Conversational search** — Ask anything about the Quran. Claude explores the graph using 6 tools and returns answers citing specific verses `[2:255]` that you can hover over to read.
- **3D graph visualization** — 6,234 verses arranged on a sphere by surah. Connections light up in real-time as Claude explores. Fly through with WASD controls.
- **Statistical dashboard** — Interactive presentation of graph topology, keyword frequencies, cross-surah connections, and more.

## How It Works

### The Pipeline

1. **Parse** — Extract all 6,234 verses from the PDF translation into structured JSON
2. **Extract keywords** — TF-IDF vectorization with lemmatization produces 2,661 unique keywords and ~45,000 weighted verse-to-keyword relationships
3. **Build connections** — Shared rare keywords create ~51,700 verse-to-verse edges (capped at 12 per verse)
4. **Embed** — Every verse encoded into a 384-dim vector using `all-MiniLM-L6-v2` for semantic search
5. **Store** — Everything goes into Neo4j (graph database) with full-text and vector indexes
6. **Query** — Claude autonomously decides which tools to call, traverses the graph, deduplicates, and synthesizes answers

### Claude's Tools

| Tool | What it does |
|------|-------------|
| `search_keyword` | Exact keyword lookup across all verses |
| `semantic_search` | Meaning-based search via embeddings — finds "mercy" when you search "forgiveness" |
| `traverse_topic` | Multi-keyword search + 1-2 hop graph traversal |
| `get_verse` | Deep-dive into one verse and its connections |
| `find_path` | Shortest thematic path between any two verses |
| `explore_surah` | Map a full chapter and its cross-chapter links |

## Key Stats

| | |
|---|---|
| Verses | 6,234 across 114 surahs |
| Keywords | 2,661 extracted via TF-IDF |
| Verse-to-keyword edges | 45,064 |
| Verse-to-verse edges | 51,733 |
| Cross-surah connections | 93.7% of all edges |
| Avg keywords per verse | 7.27 |

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Neo4j Desktop](https://neo4j.com/download/) (Community Edition is fine)
- [Anthropic API key](https://console.anthropic.com/)

### Install

```bash
git clone https://github.com/alikatkodia-collab/quran-knowledge-graph.git
cd quran-knowledge-graph

pip install neo4j anthropic python-dotenv sentence-transformers scikit-learn nltk
npm install pptxgenjs
```

### Configure

Create a `.env` file:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=quran
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### Build the Graph

Run these in order (only needed once):

```bash
# 1. Parse the PDF into verses.json
python parse_quran.py

# 2. Extract keywords and build CSV edge lists
python build_graph.py

# 3. Import into Neo4j
python import_neo4j.py

# 4. Generate semantic embeddings
python embed_verses.py
```

### Run

```bash
# Web UI (recommended) — opens at http://localhost:8081
python app.py

# CLI chat
python chat.py

# Gradio UI — opens at http://localhost:7860
python ui.py

# OpenAI-compatible API — http://localhost:8100/v1
python server.py
```

## Project Structure

```
quran-knowledge-graph/
  parse_quran.py       — PDF to structured JSON
  build_graph.py       — TF-IDF keyword extraction + edge computation
  import_neo4j.py      — Batch import into Neo4j
  embed_verses.py      — Sentence-transformer embeddings + vector index
  chat.py              — Claude agent with 6 graph tools + system prompt
  app.py               — FastAPI web UI with streaming + 3D graph
  server.py            — OpenAI-compatible REST API
  ui.py                — Gradio chatbot interface
  explore.py           — CLI graph exploration tool
  index.html           — 3D graph visualization (Three.js + ForceGraph3D)
  stats.html           — Statistical insights dashboard
  make_stats_pptx.js   — PowerPoint stats generator
  data/
    verses.json              — All 6,234 parsed verses
    verse_nodes.csv          — Verse node data for Neo4j import
    keyword_nodes.csv        — Keyword node data
    verse_keyword_rels.csv   — MENTIONS edges with TF-IDF scores
    verse_related_rels.csv   — RELATED_TO edges with similarity scores
```

## Interesting Findings

- **93.7% of thematic connections are cross-surah** — the Quran is a deeply interconnected web, not 114 isolated chapters
- **Surah 12 (Joseph)** is uniquely isolated for a long surah (111 verses, 12.26 edges/verse) — its continuous narrative makes it thematically distinct
- **Surah 55 (Most Gracious)** is least densely connected (8.13 edges/verse) — its repetitive refrain creates internal similarity but less external connection
- **Surah 64 (Mutual Blaming)** is most densely connected (24.89 edges/verse) — its universal themes resonate across the entire Quran
- **Forbidden food passages** form the tightest cluster — 3 of the top 10 strongest verse pairs
- **The strongest verse pair** is 4:43 and 5:6 (both about ablution rules, score 3.82)

## Tech Stack

- **Graph database**: Neo4j (Cypher queries, vector index)
- **AI**: Claude claude-sonnet-4-5 via Anthropic API (agentic tool-use)
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`, 384-dim)
- **NLP**: scikit-learn TF-IDF, NLTK lemmatization
- **Frontend**: Three.js, ForceGraph3D, vanilla JS
- **Backend**: FastAPI, uvicorn
- **Translation**: Rashad Khalifa's *Quran: The Final Testament*

## License

MIT
