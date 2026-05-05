# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quran Knowledge Graph — an AI-powered Quran explorer connecting 6,234 verses (Rashad Khalifa's *The Final Testament*) through a Neo4j knowledge graph. Users ask natural-language questions, the agent explores the graph via 20+ tool functions, and answers are grounded in specific verse citations with 3D visualization.

## Running the Application

```bash
# Main web UI (FastAPI + 3D visualization, agentic loop, Ollama or OpenRouter free or Anthropic)
SEMANTIC_SEARCH_INDEX=verse_embedding_m3 \
RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
python app_free.py                     # http://localhost:8085  (RECOMMENDED)

python app.py                          # Anthropic-paid variant on :8081 (legacy)
python server.py                       # OpenAI-compatible API on :8100

# Eval harness (live /chat over a hand-curated set; see eval_v1.py + EVAL_QRCD_REPORT.md)
python eval_v1.py                      # 13-question end-to-end test
python eval_qrcd_retrieval.py          # retrieval-only QRCD A/B (3 backends)
python eval_qrcd_hipporag_sweep.py     # 36-config PPR ablation
python eval_ablation_retrieval.py      # per-stage retrieval pipeline ablation

# Autoresearch (older infra)
python autoresearch.py --trials 50     # Claude API (~$2-3/trial)
python autoresearch_local.py --trials 100  # Ollama (free)
```

## Data Pipeline (build from scratch)

Run in order — each step depends on the previous:
```bash
python parse_quran.py             # PDF -> data/verses.json
python build_graph.py             # TF-IDF keywords + edges -> 4 CSVs
python import_neo4j.py            # CSVs -> Neo4j
python embed_verses.py            # Vector embeddings (legacy MiniLM 384d)
python embed_verses_m3.py         # NEW: BGE-M3 1024d embeddings (EN + AR)  ← preferred
python migrate_graph.py           # Schema fixes, orphan verses, stopword cleanup
python load_arabic.py             # Arabic text (Hafs reading) -> Verse nodes
python build_arabic_roots.py      # Morphology -> ArabicRoot nodes + edges
python build_word_tokens.py       # Word-level parsing -> Lemma, MorphPattern nodes
python build_semantic_domains.py  # Semantic field groupings
python build_wujuh.py             # Polysemy data
python import_etymology.py        # All etymology data -> Neo4j
python classify_edges.py          # Typed edges (SUPPORTS, ELABORATES, etc.)

# Newer (2026-04+):
python build_code19_features.py        # Khalifa Code-19 arithmetic features
python build_fulltext_index.py         # BM25 indexes (verse_text + verse_arabic)
python build_concepts.py               # Porter-stem ER -> :Concept nodes
python import_mutashabihat.py          # CC0 mutashabihat -> :SIMILAR_PHRASE edges
python backfill_embedding_provenance.py
python backfill_bidirectional_tfidf.py # Sefaria-style from_tfidf/to_tfidf on MENTIONS
python backfill_retrieved_edges.py     # 32K+ :RETRIEVED edges from answer_cache
python analyze_graph_structure.py      # Degree distribution, betweenness, modularity
```

## Architecture

**Agentic, not RAG.** The model drives exploration via a tool-use loop (up to 15 turns), choosing which graph tools to call based on the question. Cross-encoder reranking, NLI/MiniCheck citation verification, and multilingual embeddings sit alongside, not in front of, the agent.

### Request Flow
```
POST /chat (app_free.py:8085)
  -> daemon thread: agentic loop (Anthropic API | Ollama | OpenRouter)
    -> tool_use calls dispatch via chat.py to 20+ Neo4j query functions
    -> retrieval_gate.py: cross-encoder rerank (multilingual) + lost-in-middle reorder
    -> tool-call cache (30s TTL) deduplicates identical calls within a turn
    -> reasoning_memory.py: writes Query/Trace/ToolCall/RETRIEVED subgraph
  -> SSE events: text | tool | graph_update | done | error | verification
  -> index.html: chat bubbles + 3D graph + citation tooltips
```

### Key Subsystems

- **chat.py** — 20+ tool functions + dispatch + tool-call cache. Tools split into:
    - **Search/retrieval (8)**: `search_keyword`, `semantic_search`, `traverse_topic`, `hybrid_search` (BM25+BGE-M3+RRF), `concept_search` (canonical-form expansion), `find_path`, `explore_surah`, `recall_similar_query` (past-trace playbook)
    - **Verse-level (3)**: `get_verse`, `get_verse_words`, `query_typed_edges`
    - **Etymology/morphology (6)**: `search_arabic_root`, `compare_arabic_usage`, `lookup_word`, `explore_root_family`, `search_semantic_field`, `lookup_wujuh`, `search_morphological_pattern`
    - **Code-19 + escape hatch (2)**: `get_code19_features`, `run_cypher` (read-only, denylist-guarded)
- **app_free.py** — FastAPI server. Bridges sync agent loop to async SSE via `queue.SimpleQueue` + daemon thread. Citation verification (NLI/MiniCheck/FActScore) is post-response and env-gated. Also exposes `/api/resolve_refs`, `/api/verse/<id>`, and `/quran_linker.js` (Sefaria-style auto-linker widget).
- **reasoning_memory.py** — Every query writes `(:Query)-[:TRIGGERED]->(:ReasoningTrace)-[:HAS_STEP]->(:ToolCall)` plus `(:ReasoningTrace)-[:RETRIEVED {tool, rank, turn}]->(:Verse)` edges that accumulate a learnable signal of which verses each tool returns. Also `(:ReasoningTrace)-[:HAS_CITATION_CHECK]->(:CitationCheck)` for NLI/MiniCheck verdicts.
- **citation_verifier.py** — NLI (`cross-encoder/nli-deberta-v3-xsmall`) + MiniCheck-FT5 (`flan-t5-large`) + FActScore-style atomic claim decomposition (LLM-driven). All env-gated.
- **retrieval_gate.py** — Cross-encoder reranking. Default model is now `BAAI/bge-reranker-v2-m3` (multilingual). Legacy `ms-marco-MiniLM-L-6-v2` was English-only and dropped QRCD hit@10 by 32 points on Arabic queries (see `eval_ablation_retrieval.py`).
- **ref_resolver.py** — Sefaria-style citation NER. Recognises `[2:255]`, `Quran X:Y`, `Surah Al-Baqarah verse 286`, `Ayat al-Kursi`, Arabic `سورة البقرة آية 255`, ranges, lists. Powers the JS auto-linker widget.
- **hipporag_traverse.py** — `hipporag_search()` (full PPR retrieval, NOT wired) + `ppr_rerank()` (post-seed re-ranker, available as helper). Negative QRCD result documented in `HIPPORAG_REPORT.md`.
- **uncertainty.py** — Phase 5 abstention via 5-Haiku semantic-entropy probes.
- **answer_cache.py** — 1,500+ entry cache; 0.98 cosine dedupe (was 0.95, was hiding growth); 5,000-entry cap (was 500). Inject relevant past answers as system-prompt context.

### Neo4j Graph Schema

**Nodes:** Verse (6,234), Keyword (2,636), Concept (2,388, NEW — Porter-stem ER over Keywords), ArabicRoot (1,223), Lemma (4,762+), MorphPattern (100+), SemanticDomain (30+), Sura (114), Query / ReasoningTrace / ToolCall / Answer / CitationCheck (reasoning memory).

**Verse properties (post-2026-04):** `verseId`, `text`, `arabicText`, `arabicPlain`, `surah`, `verseNum`, `surahName`, `embedding` (legacy MiniLM 384d), `embedding_m3` (BGE-M3 EN 1024d), `embedding_m3_ar` (BGE-M3 AR 1024d), `embedding_model`, `embedding_source_hash`, `embedded_at`, `position_in_sura`, `is_initial_verse`, `letter_qaf` / `letter_nun` / etc. (Code-19 features).

**Sura properties:** `number`, `verses_count`, `mysterious_letters`, `ml_letter_counts_json`, `ml_div_19_json`, `mod19_verse_count`.

**Key relationships:**
- `MENTIONS {score, from_tfidf, to_tfidf, data_source, generated_by}` (41K) — Verse->Keyword, Sefaria-style bidirectional TF-IDF
- `NORMALIZES_TO` (2,636) — Keyword->Concept (entity resolution layer)
- `RELATED_TO` (51K) — Verse<->Verse, shared rare keywords, capped 12/verse, 93.7% cross-surah
- `MENTIONS_ROOT` (~100K) — Verse->ArabicRoot, with surface forms
- `SIMILAR_PHRASE {dataSource: 'waqar144-mutashabiha'}` (3,270) — Verse<->Verse, CC0 mutashabihat
- Typed: `SUPPORTS`, `ELABORATES`, `QUALIFIES`, `CONTRASTS`, `REPEATS` (7K)
- Structural: `CONTAINS` (Sura->Verse), `NEXT_VERSE`
- Reasoning memory: `TRIGGERED`, `HAS_STEP {order}`, `PRODUCED`, `RETRIEVED {tool, rank, turn}`, `HAS_CITATION_CHECK`

**Vector indexes:**
- `verse_embedding` (cosine, 384-dim, all-MiniLM-L6-v2 — legacy)
- `verse_embedding_m3` (cosine, 1024-dim, BGE-M3 over English)
- `verse_embedding_m3_ar` (cosine, 1024-dim, BGE-M3 over Arabic)
- `query_embedding` (cosine, 384-dim, MiniLM over Query.text)

**Full-text indexes:**
- `verse_text_fulltext` (English analyzer over Verse.text — used by `hybrid_search`)
- `verse_arabic_fulltext` (Arabic analyzer over Verse.arabicPlain)

### SSE Event Protocol

Events from `/chat` endpoint: `text` (markdown deltas), `tool` (tool execution details), `graph_update` (nodes+links for 3D viz), `done` (batch verse texts for tooltips), `verification` (NLI/MiniCheck verdicts when enabled), `error`, `retry`.

### Frontend (index.html)

Single-file SPA. Three.js r160, 3d-force-graph, marked.js. All 6,234 verses pre-positioned on a Fibonacci sphere (114 surah clusters). Dark theme (#060a14 bg, #10b981 accent). WASD fly controls.

## Configuration

`pipeline_config.yaml` controls all tunable parameters. Most overrides are env vars (see below) — set them before launch.

## Environment

Requires `.env` with:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-v1-...   # for free-tier models + atomic decomposer
```

Neo4j database name: `quran` (set via `NEO4J_DATABASE` env var).

### Optional env vars

```bash
# Active vector index for tool_semantic_search & friends
SEMANTIC_SEARCH_INDEX=verse_embedding             # default (legacy MiniLM 384d)
                     |verse_embedding_m3          # BGE-M3 over English (RECOMMENDED, 1024d)
                     |verse_embedding_m3_ar       # BGE-M3 over Arabic (cross-lingual)

# Reranker (used by retrieval_gate)
RERANKER_MODEL=BAAI/bge-reranker-v2-m3            # default — multilingual, RECOMMENDED
              |cross-encoder/ms-marco-MiniLM-L-6-v2  # legacy English-only (HARMFUL on Arabic)
              |none / off / disabled              # skip reranking entirely
RERANK_DISABLED=1                                 # alternative kill switch

# Tool-call cache (Nixon "1-second cache" pattern in chat.py)
TOOL_CACHE_TTL_SEC=30                             # 0 disables; default 30s
TOOL_CACHE_MAX=256                                # FIFO eviction

# Post-response citation verification
ENABLE_CITATION_VERIFY=1
CITATION_VERIFIER_MODEL=nli|minicheck             # default nli
MINICHECK_MODEL=flan-t5-large                     # roberta-large | deberta-v3-large | flan-t5-large
MINICHECK_THRESHOLD=0.5                           # support probability cutoff
CITATION_DECOMPOSE=regex|atomic                   # FActScore-style decomposition (atomic uses OpenRouter)
DECOMPOSE_MODEL=openai/gpt-oss-120b:free
```

## Eval Harness (2026-04-28+)

- `data/eval_v1_results.json` + `data/eval_v1_results.md` — hand-curated 13-question end-to-end baseline
- `data/qrcd_retrieval_results.json` — QRCD retrieval A/B (MiniLM vs BGE-M3-EN vs BGE-M3-AR)
- `data/qrcd_hipporag_compare.json` — vanilla vs HippoRAG retrieval
- `data/qrcd_hipporag_sweep.json` — 36-config PPR ablation (definitive negative)
- `data/qrcd_hybrid_compare.json` — dense vs hybrid (BM25+BGE-M3 RRF)
- `data/qrcd_ablation.json` — per-stage retrieval pipeline ablation (revealed legacy reranker bug)
- `data/graph_stats.json` — degree distribution, betweenness, modularity (16 communities @ 0.5324)
- `data/code19_summary.json` — global Code-19 figures (6,346 = 19 × 334, etc.)

### Headline benchmark numbers
- QRCD MAP@10: legacy MiniLM = 0.028 → BGE-M3-EN = **0.139** (5×) vs AraBERT-base fine-tuned = 0.36 (lit)
- Multilingual reranker: hit@10 0.32 → 0.55 vs legacy English-only (which actively hurt Arabic queries)
- Cache: 500 → **1,500** entries (target hit), 97% long-form, 77% strong (≥10 cites)
- Tool-call cache: semantic_search **18,785× speedup** on hot calls (cold 18.8s → 0ms hit)

## Translation Context

This project uses Rashad Khalifa's translation exclusively (6,234 verses; 9:128–129 excluded as Khalifa flagged them as forged via the 19-based mathematical code). The system prompt acknowledges Khalifa-specific interpretations: the mathematical miracle of 19, the Messenger of the Covenant, and the rejection of hadith as a religious source. Arabic text is Hafs ʿan ʿĀṣim reading (Uthmani script) from the Quranic Arabic Corpus.

## Dependencies

Core: `anthropic`, `neo4j`, `fastapi`, `uvicorn`, `sentence-transformers`, `pyyaml`, `python-dotenv`, `nltk`, `scikit-learn`, `numpy`, `networkx`

For new features (mostly post-2026-04):
- `minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main` — citation verifier
- `BAAI/bge-m3` (sentence-transformers compatible) — multilingual embeddings
- `BAAI/bge-reranker-v2-m3` — multilingual reranker

Optional: `optuna` (autoresearch), `requests` (Ollama + OpenRouter), `pdfminer` (PDF parsing), `gradio` (alt UI)

## Reports & Documentation

- `RESEARCH_2026-04-27.md` — initial deep research on stack alternatives
- `RESEARCH_2026-04-28_DEEP.md` — deep-dive on Sefaria, MiniCheck, HippoRAG, GraphRAG-Bench, Tarteel QUL, Doha Dictionary
- `EVAL_QRCD_REPORT.md` — QRCD benchmark results
- `HIPPORAG_REPORT.md` — PPR implementation + negative result
- `AUTONOMOUS_RUN_2026-04-28.md` — wrap-up of the major autonomous build phase
- `WEEKEND_REPORT.md` — overnight cache-seeding from 500 → 1,500 entries
- `OVERNIGHT_REPORT.md` — earlier overnight seeding write-up
- `ARCHITECTURE.md` — end-to-end architecture with Mermaid diagrams
