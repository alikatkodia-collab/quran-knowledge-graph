# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quran Knowledge Graph — an AI-powered Quran explorer connecting 6,234 verses (Rashad Khalifa's *The Final Testament*) through a Neo4j knowledge graph. Users ask natural language questions, Claude explores the graph via 15 tool-use functions, and answers are grounded in specific verse citations with 3D visualization.

## Running the Application

```bash
# Main web UI (FastAPI + 3D visualization)
python app.py                          # http://localhost:8081

# OpenAI-compatible API (for Open WebUI etc.)
python server.py                       # http://localhost:8100

# Evaluation
python evaluate.py                     # All test questions
python evaluate.py --ids q01_forgiveness,q05_nineteen --out results.json

# Autoresearch optimization
python autoresearch.py --trials 50     # Claude API (~$2-3/trial)
python autoresearch_local.py --trials 100  # Ollama (free)
python autoresearch_dashboard.py       # Monitor at http://localhost:8082
```

## Data Pipeline (build from scratch)

Run in order — each step depends on the previous:
```bash
python parse_quran.py          # PDF -> data/verses.json
python build_graph.py          # TF-IDF keywords + edges -> 4 CSVs
python import_neo4j.py         # CSVs -> Neo4j
python embed_verses.py         # Vector embeddings (all-MiniLM-L6-v2)
python migrate_graph.py        # Schema fixes, orphan verses, stopword cleanup
python load_arabic.py          # Arabic text (Hafs reading) -> Verse nodes
python build_arabic_roots.py   # Morphology -> ArabicRoot nodes + edges
python build_word_tokens.py    # Word-level parsing -> Lemma, MorphPattern nodes
python build_semantic_domains.py  # Semantic field groupings
python build_wujuh.py          # Polysemy data
python import_etymology.py     # All etymology data -> Neo4j
python classify_edges.py       # Typed edges (SUPPORTS, ELABORATES, etc.)
```

## Architecture

**Agentic, not RAG.** Claude drives exploration via a tool-use loop (up to 15 turns), choosing which graph tools to call based on the question. This is NOT a fixed retrieval pipeline.

### Request Flow
```
POST /chat (app.py:8081)
  -> daemon thread: Claude API agentic loop (chat.py)
    -> tool_use calls dispatch to 15 Neo4j query functions
    -> retrieval_gate.py: cross-encoder reranking + quality assessment
  -> SSE events: text | tool | graph_update | done | error
  -> index.html: chat bubbles + 3D graph + citation tooltips
```

### Key Subsystems

- **chat.py** — 15 tool functions + dispatch. The core of the system. Tools split into search/navigation (9) and etymology/morphology (6). Each returns structured dicts that Claude uses to form cited answers.
- **app.py** — FastAPI server. Bridges Claude's sync API with async SSE via `queue.SimpleQueue()` + daemon thread. Also runs citation density checks and NLI verification post-response.
- **config.py** — All pipeline parameters loaded from `pipeline_config.yaml`. Call `config.reload()` after modifying the YAML. Typed accessors: `cfg.llm_model()`, `cfg.semantic_default_top_k()`, etc.
- **retrieval_gate.py** — Phase 2 hallucination reduction. Cross-encoder reranking (ms-marco-MiniLM-L-6-v2), lost-in-middle U-shape reordering, quality gating ("good"/"marginal"/"poor").
- **citation_verifier.py** — Phase 3. NLI entailment (nli-deberta-v3-xsmall) to verify each citation actually supports its claim.
- **uncertainty.py** — Phase 5. Semantic entropy via 5 Haiku probes. Can trigger abstention on high-uncertainty questions.

### Neo4j Graph Schema

**Nodes:** Verse (6,234), Keyword (2,636), ArabicRoot (1,223), Lemma (4K+), MorphPattern (100+), SemanticDomain (30+), Sura (114)

**Key relationships:**
- `MENTIONS` (41K) — Verse->Keyword, TF-IDF weighted
- `RELATED_TO` (51K) — Verse<->Verse, shared rare keywords, capped 12/verse, 93.7% cross-surah
- `MENTIONS_ROOT` (~100K) — Verse->ArabicRoot, with surface forms
- Typed: `SUPPORTS`, `ELABORATES`, `QUALIFIES`, `CONTRASTS`, `REPEATS`
- Structural: `CONTAINS` (Sura->Verse), `NEXT_VERSE` (sequential order)

**Vector index:** `verse_embedding` (cosine, 384-dim, all-MiniLM-L6-v2)

### SSE Event Protocol

Events from `/chat` endpoint: `text` (markdown deltas), `tool` (tool execution details), `graph_update` (nodes+links for 3D viz), `done` (batch verse texts for tooltips), `error`.

### Frontend (index.html)

Single-file SPA. Three.js r160, 3d-force-graph, marked.js. All 6,234 verses pre-positioned on a Fibonacci sphere (114 surah clusters). Dark theme (#060a14 bg, #10b981 accent). WASD fly controls.

## Configuration

`pipeline_config.yaml` controls all tunable parameters:
- **llm** — model, max_tokens, temperature
- **retrieval** — per-tool limits (top_k, seed_limit, hop limits, fuzzy search params)
- **scoring** — min score thresholds for edges
- **evaluation** — composite weights for QIS score (citation_recall:0.35, citation_precision:0.25, grounding_rate:0.30, answer_relevance:0.10)

The autoresearch loop optimizes these via Optuna TPE (Bayesian search).

## Evaluation (QIS Score)

Composite metric maximized by autoresearch:
```
QIS = 0.35*citation_recall + 0.25*citation_precision + 0.30*grounding_rate + 0.10*answer_relevance
```

Test dataset: `test_dataset.json` — 218 questions (100 standard, 50 edge cases, 50 unanswerable). Each has `id`, `question`, `expected_citations`, `difficulty`.

## Environment

Requires `.env` with:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
ANTHROPIC_API_KEY=sk-ant-...
```

Neo4j database name: `quran` (set via `NEO4J_DATABASE` env var).

## Dependencies

Core: `anthropic`, `neo4j`, `fastapi`, `uvicorn`, `sentence-transformers`, `pyyaml`, `python-dotenv`, `nltk`, `scikit-learn`, `numpy`

Optional: `optuna` (autoresearch), `requests` (Ollama), `pdfminer` (PDF parsing), `gradio` (alt UI)

## Translation Context

This project uses Rashad Khalifa's translation exclusively (6,234 verses; 9:128-129 excluded). The system prompt acknowledges Khalifa-specific interpretations (mathematical miracle of 19, Messenger of the Covenant). Arabic text is Hafs reading (Uthmani script) from the Quranic Arabic Corpus.
