# Quran Knowledge Graph — Project Summary

## What Is This?

A conversational AI system that lets you ask questions about the Quran in plain English and get answers grounded in specific verse citations, with a 3D visualization showing how verses connect to each other across the entire text.

It works by combining a **knowledge graph** (a database of connections between all 6,234 verses) with an **agentic AI** (Claude or a local model) that can autonomously explore that graph using 15 different search tools. Every claim in the AI's response is backed by a specific verse reference like `[2:255]` that you can hover over to read the full text in both English and Arabic.

This is not a simple chatbot with a copy of the Quran pasted into its context. The AI has no pre-loaded knowledge of the text — it discovers everything by querying the graph in real time, the same way a researcher would work through a concordance.

---

## The Knowledge Graph

### What's In It

The graph lives in a Neo4j database and contains **92,437 nodes** and **403,239 relationships**.

**Nodes (7 types):**

| Type | Count | What It Represents |
|------|-------|--------------------|
| Verse | 6,234 | Every verse in the Quran (Rashad Khalifa's translation). Each has English text, Arabic text (Hafs reading, Uthmani script), and a 384-dimensional vector embedding for semantic search. |
| Keyword | 2,636 | Lemmatized English keywords extracted via TF-IDF. Words like "forgiveness", "covenant", "messenger". |
| ArabicRoot | 1,223 | Tri-literal Arabic roots. The building blocks of Arabic vocabulary — e.g. root k-t-b (ك ت ب) produces "book", "wrote", "prescribed". |
| Lemma | 4,762 | Arabic dictionary forms. More specific than roots — a single root might produce dozens of lemmas with different meanings. |
| Sura | 114 | The 114 chapters of the Quran. |
| MorphPattern | 38 | Arabic morphological templates (awzan) like fa'il, maf'ul. These carry grammatical meaning — fa'il is an active participle, maf'ul is passive. |
| SemanticDomain | 28 | Conceptual groupings that cluster related roots. The "mercy" domain includes roots for compassion, clemency, pardon, and forgiveness. |

**Relationships (16 types):**

| Type | Count | What It Connects |
|------|-------|------------------|
| IN_VERSE | 77,400 | Word tokens to the verses they appear in |
| HAS_LEMMA | 74,095 | Word tokens to their dictionary forms |
| RELATED_BY_ROOT | 73,666 | Verse pairs that share Arabic roots (computed via TF-IDF on roots) |
| RELATED_TO | 51,798 | Verse pairs that share English keywords (capped at 12 per verse, 93.6% cross-surah) |
| MENTIONS | 41,138 | Verses to their TF-IDF keywords |
| MENTIONS_ROOT | 36,366 | Verses to the Arabic roots they contain |
| FOLLOWS_PATTERN | 24,579 | Words to their morphological patterns |
| CONTAINS | 6,234 | Suras to their verses |
| NEXT_VERSE | 6,233 | Sequential reading order |
| SUPPORTS | 5,464 | Verse A provides independent evidence for Verse B's claim |
| DERIVES_FROM | 4,128 | Lemmas to their parent roots |
| CONTRASTS | 697 | Verses presenting complementary perspectives on the same topic |
| REPEATS | 626 | Near-verbatim repetitions across different surahs |
| ELABORATES | 357 | Verse A expands on Verse B with more detail |
| QUALIFIES | 247 | Verse A adds a condition or exception to Verse B |
| IN_DOMAIN | 211 | Roots belonging to semantic domains |

### How It Was Built

The graph was constructed through a 12-step data pipeline:

1. **PDF Parsing** — Extracted all 6,234 verses from Rashad Khalifa's PDF translation, handling surah headers, verse numbering, multi-line verses, and footnotes.

2. **TF-IDF Keyword Extraction** — Built a vocabulary of 2,636 lemmatized keywords using scikit-learn TF-IDF with NLTK WordNet lemmatization. Each verse gets weighted MENTIONS edges to its keywords.

3. **Verse Similarity** — Computed pairwise verse similarity based on shared rare keywords. Verses that share unusual words (not common ones like "God" or "people") get RELATED_TO edges. Capped at 12 per verse to keep the graph navigable.

4. **Neo4j Import** — Batch imported all nodes and edges via CSV using Neo4j's LOAD CSV. Created uniqueness constraints and indexes.

5. **Vector Embeddings** — Encoded every verse with the all-MiniLM-L6-v2 sentence transformer (384 dimensions). Stored as a property on each Verse node with a Neo4j vector index for cosine similarity search.

6. **Schema Migration** — Unified dual-schema verse populations, rebuilt structural edges (CONTAINS, NEXT_VERSE), connected 33 orphan Muqatta'at verses, removed 25 generic stopword keywords and 3,926 noisy edges, added metadata.

7. **Arabic Text** — Downloaded the Hafs reading (Uthmani script) and attached it to every verse as `arabicText` (with diacritics, for display) and `arabicPlain` (stripped, for search).

8. **Arabic Root Morphology** — Used the Quranic Arabic Corpus morphology data to extract the tri-literal root of every word. Created 1,223 ArabicRoot nodes, 36,366 MENTIONS_ROOT edges (with surface forms), and 73,666 RELATED_BY_ROOT edges (a parallel similarity layer based on shared roots instead of English keywords).

9. **Word-Level Etymology** — Parsed every word into its lemma, morphological pattern, part of speech, and gloss. Created 4,762 Lemma nodes and 38 MorphPattern nodes with full occurrence tracking.

10. **Semantic Domains** — Grouped related roots into 28 conceptual fields (mercy, knowledge, creation, etc.).

11. **Polysemy (Wujuh)** — Mapped how the same root carries different meanings in different contexts, following the classical wujuh wa naza'ir tradition.

12. **Typed Edge Classification** — Analyzed all 51,798 RELATED_TO edges and classified them into semantic types: SUPPORTS (5,464), CONTRASTS (697), REPEATS (626), ELABORATES (357), QUALIFIES (247). This tells the AI not just that two verses are related, but *how* they relate.

### Why Two Similarity Layers?

The graph has two independent verse-to-verse connection systems:

- **RELATED_TO** (51,798 edges) — Based on shared English keywords. If two verses both mention "forgiveness" and "repentance", they're connected.
- **RELATED_BY_ROOT** (73,666 edges) — Based on shared Arabic roots. If two verses both contain forms of root r-h-m (mercy), they're connected — even if the English translations use completely different words ("compassionate" vs "merciful" vs "womb").

The Arabic layer is actually larger and catches connections invisible in English. For example, the word for "womb" (rahim) shares a root with "mercy" (rahmah) and "Most Merciful" (al-Rahman) — a theological connection between God's mercy and the nurturing of life that only appears at the Arabic root level.

---

## The AI Agent

### How It Works

When you ask a question, the AI doesn't just look it up — it runs an **agentic loop** where it decides which tools to call, processes the results, and calls more tools if needed. A typical question might involve 2-4 rounds of tool calls before the AI has enough context to write a thorough answer.

The system prompt instructs the AI to:
- Always search before answering (never rely on pre-trained knowledge)
- Use multiple search strategies per question (keyword + semantic + traversal)
- Cite every claim with inline `[surah:verse]` references
- Stay honest about what the text does and doesn't say
- Distinguish explicit statements from interpretation

### The 15 Tools

**Search & Navigation (9 tools):**

| Tool | What It Does |
|------|-------------|
| `search_keyword` | Exact keyword search across all verses, grouped by surah |
| `semantic_search` | Meaning-based search using vector embeddings — finds conceptually related verses even when they use different words |
| `get_verse` | Deep-dive into one verse: full text, Arabic, keywords, connections, typed relationships |
| `traverse_topic` | Multi-keyword search plus graph traversal — follows connections 1-2 hops out from seed verses |
| `find_path` | Finds the shortest thematic path between any two verses through the graph |
| `explore_surah` | Maps an entire chapter's content and its connections to other chapters |
| `query_typed_edges` | Finds verses by relationship type — "show me verses that SUPPORT this one" or "what QUALIFIES this rule?" |
| `search_arabic_root` | Finds all verses containing any form of a specific Arabic root |
| `compare_arabic_usage` | Shows how different derived forms of one root carry different meanings across contexts |

**Etymology & Word Analysis (6 tools):**

| Tool | What It Does |
|------|-------------|
| `lookup_word` | Look up any Arabic word — root, lemma, morphological pattern, all occurrences |
| `explore_root_family` | Full derivative tree from one root — all lemmas, patterns, semantic domains |
| `get_verse_words` | Word-by-word breakdown of a verse — each word's root, lemma, pattern, POS, English gloss |
| `search_semantic_field` | Find all roots in a conceptual domain (e.g. "mercy" returns roots for compassion, clemency, pardon) |
| `lookup_wujuh` | Show all distinct meanings a polysemous root carries in different contexts |
| `search_morphological_pattern` | Find words sharing a morphological pattern (e.g. all fa'il intensive adjectives) |

### Retrieval Gate

After tools return results, a **retrieval gate** (cross-encoder reranking using ms-marco-MiniLM-L-6-v2) reranks them by relevance to the original question, reorders them to avoid the "lost in the middle" problem (where models ignore information in the middle of long contexts), and assigns a quality rating ("good", "marginal", or "poor"). If results are poor, the AI is instructed to try a different approach.

---

## Hallucination Reduction

The system includes a 5-phase pipeline to prevent the AI from making things up:

### Phase 2: Retrieval Gate (`retrieval_gate.py`)
Cross-encoder reranking ensures the most relevant verses are surfaced first. Quality gating tells the AI when its search results are weak, prompting it to try alternative queries rather than fabricating an answer from thin data. Runs locally, no API cost.

### Phase 3: Citation Verification (`citation_verifier.py`)
After the AI responds, an NLI (Natural Language Inference) model (nli-deberta-v3-xsmall) checks whether each cited verse actually entails the claim it's attached to. If the AI says "[2:255] states that God never sleeps" — the verifier checks whether verse 2:255 actually says that. Runs locally, no API cost.

### Phase 4: Constrained Output (`prompts/system_prompt.txt`)
The system prompt contains strict rules: every factual claim must have a verse citation, never make a theological statement without evidence, distinguish between what a verse says and what you're interpreting, acknowledge when the Quran doesn't address a topic. No cost (it's just prompt text).

### Phase 5: Semantic Entropy (`uncertainty.py`)
Before answering, 5 quick probe responses are generated using a cheap model (Haiku). If the probes disagree significantly (high semantic entropy), the system flags high uncertainty and either adds a caution note to the system prompt or triggers abstention. Costs ~$0.01-0.03 per question.

### Phase 6: Citation Retry (`app_full.py`)
After the response is generated, a citation density check counts what fraction of factual sentences have verse citations. If below 30%, the entire response is regenerated with stricter grounding instructions. Costs ~$0.05-0.15 when triggered.

---

## Cost Optimization

Two systems reduce API costs:

### Tool Result Compressor (`tool_compressor.py`)
The biggest cost driver is tool results being fed back into the conversation. Each tool call returns detailed verse data, and on turn 3 of an agentic loop, the AI is re-reading everything from turns 1 and 2.

The compressor strips tool results before they go back into the conversation:
- Verse text truncated to 100 characters (from 300-400)
- Arabic text removed entirely (Claude doesn't need it to cite verses — it cites by reference)
- Keyword lists capped at 8 items
- Embedding vectors dropped

A typical `search_keyword` result drops from ~19,000 characters to ~5,600 — a **70% reduction**. The full uncompressed results are still used for the 3D graph visualization and etymology panels; only the conversation copy is compressed.

### Answer Cache (`answer_cache.py`)
Every question-answer pair is saved to `data/answer_cache.json` with an embedding of the question. When a new question comes in:
- The cache is searched for past answers with >= 60% semantic similarity
- Up to 3 matching past answers are injected into the system prompt
- The AI can reuse verse citations and analysis instead of making full tool loops
- Near-duplicate questions (> 95% similarity) update the existing entry
- The cache is shared across all 4 deployment modes
- Capped at 500 entries, oldest evicted first

Combined effect: **~50% cost reduction** on the paid versions, with repeat questions approaching zero marginal cost.

---

## Four Deployment Modes

| Version | Port | Model | Cost/Question | What's Included |
|---------|------|-------|---------------|-----------------|
| **Default** | 8081 | Claude Sonnet | ~$0.05-0.10 | Tool compression + answer cache. No hallucination reduction post-processing. Good balance of quality and cost. |
| **Full** | 8083 | Claude Sonnet | ~$0.15-0.35 | Everything: uncertainty probes, citation retry, NLI verification. Maximum accuracy, highest cost. |
| **Lite** | 8084 | Claude Haiku | ~$0.01-0.03 | Cheapest API option. Same tools and cache, but Haiku is ~10x cheaper per token than Sonnet. 1,536 max tokens. |
| **Free** | 8085 | Qwen 2.5 14B (Ollama) | $0.00 | Runs entirely on your GPU. Same 15 tools, same graph, same cache. Slower (10-30s vs 3-5s) but free. |

All four versions share the same Neo4j database, the same answer cache, and the same 15 tools. The only differences are the language model and which post-processing phases are enabled.

---

## The Frontend

### 3D Galaxy Visualization (`index.html`)

A single-file web application (1,651 lines) built with Three.js, 3d-force-graph, and marked.js. All 6,234 verses are pre-positioned on a **Fibonacci sphere** with 114 surah clusters — creating a galaxy-like visualization where each cluster is a chapter of the Quran.

When you ask a question, the AI's tool calls inject nodes and links into the 3D graph in real time. You can see which verses the AI is exploring, how they connect, and which keywords bridge them. Verse nodes glow when they're actively being referenced.

**Controls:**
- Orbit (drag), zoom (scroll), fly through (WASD + Q/E)
- Click any node to see its info card
- Hover over any `[surah:verse]` citation in the chat to read the full verse text in English and Arabic

**Visual design:** Dark theme (#060a14 background), emerald green accent (#10b981), gold for keywords (#f59e0b). Verse colors are distributed by surah number across the hue spectrum.

### Statistical Dashboard (`stats.html`)

An interactive presentation of graph topology: keyword frequency distributions, cross-surah connection density, surah connectivity rankings. Built with Chart.js.

---

## Evaluation System

### QIS Score
A composite metric that measures answer quality:
```
QIS = 0.35 * citation_recall    (did the AI find the expected verses?)
    + 0.25 * citation_precision  (were its citations relevant?)
    + 0.30 * grounding_rate      (what % of claims have citations?)
    + 0.10 * answer_relevance    (is the answer about what was asked?)
```

### Test Dataset
218 test questions in `test_dataset.json`: 100 standard questions, 50 edge cases, and 50 unanswerable questions (where the correct answer is "the Quran doesn't address this"). Each has expected verse citations and a difficulty rating.

### Autoresearch
Automated hyperparameter optimization using Optuna's TPE (Tree-structured Parzen Estimator) Bayesian search. It tunes the parameters in `pipeline_config.yaml` — things like how many results each tool returns, hop limits for graph traversal, TF-IDF score thresholds — to maximize the QIS score.

Two modes:
- **API mode** (`autoresearch.py`) — Uses Claude, costs ~$2-3 per trial
- **Local mode** (`autoresearch_local.py`) — Uses Ollama, free but slower

Best result: QIS 0.1688 after 100 local trials (27% improvement over baseline). A real-time monitoring dashboard (`autoresearch_dashboard.py`) shows trial progress, score charts, and best configuration at http://localhost:8082.

---

## Technical Stack

| Layer | Technology |
|-------|-----------|
| Graph database | Neo4j 5+ (Cypher queries, vector index, fulltext index) |
| AI (paid) | Claude Sonnet / Haiku via Anthropic API (agentic tool-use) |
| AI (free) | Qwen 2.5 14B via Ollama (local GPU inference with tool-use) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, 384 dimensions) |
| Reranking | cross-encoder (ms-marco-MiniLM-L-6-v2) |
| NLI verification | cross-encoder (nli-deberta-v3-xsmall) |
| NLP | scikit-learn TF-IDF, NLTK WordNet lemmatization |
| Backend | FastAPI + uvicorn, SSE streaming via daemon thread + SimpleQueue |
| Frontend | Three.js r160, 3d-force-graph v1, marked.js v9, vanilla JS |
| Optimization | Optuna TPE Bayesian search |
| Config | YAML (pipeline_config.yaml) with runtime reload |

---

## Codebase

~16,500 lines of Python + ~2,200 lines of HTML/JS across 31 files.

**Core application (6,359 lines):**
- `chat.py` (1,439) — 15 tool functions, dispatch, system prompt
- `app.py` (587) — Default FastAPI server
- `app_full.py` (616) — Full hallucination reduction server
- `app_lite.py` (558) — Haiku cheap server
- `app_free.py` (489) — Ollama free server
- `config.py` (145) — Config accessors
- `answer_cache.py` (175) — Q&A embedding cache
- `tool_compressor.py` (59) — Tool result compression
- `retrieval_gate.py` (164) — Cross-encoder reranking
- `citation_verifier.py` (130) — NLI citation checking
- `uncertainty.py` (126) — Semantic entropy probes
- `evaluate.py` (248) — QIS evaluation
- `autoresearch.py` (367) — API optimization
- `autoresearch_local.py` (404) — Local optimization
- `autoresearch_dashboard.py` (852) — Monitoring UI

**Data pipeline (3,610 lines):**
- `parse_quran.py` (278) — PDF extraction
- `build_graph.py` (269) — TF-IDF + edge computation
- `import_neo4j.py` (196) — Neo4j batch import
- `embed_verses.py` (121) — Vector embeddings
- `migrate_graph.py` (488) — Schema migration
- `load_arabic.py` (157) — Arabic text loading
- `build_arabic_roots.py` (507) — Root morphology
- `build_word_tokens.py` (638) — Word-level parsing
- `build_semantic_domains.py` (94) — Semantic fields
- `build_wujuh.py` (72) — Polysemy data
- `import_etymology.py` (344) — Etymology import
- `classify_edges.py` (446) — Edge type classification

**Frontend (2,157 lines):**
- `index.html` (1,651) — 3D graph + chat SPA
- `stats.html` (506) — Statistical dashboard

---

## Translation

This project uses Rashad Khalifa's translation of the Quran (*The Final Testament*) exclusively. This translation contains 6,234 verses — verses 9:128-129 are excluded, as Khalifa considered them fabricated additions. The system prompt acknowledges Khalifa-specific interpretations, including the mathematical miracle of 19 and the concept of the Messenger of the Covenant.

The Arabic text is the Hafs reading (the dominant qira'a used by approximately 95% of the Muslim world), sourced from the Quranic Arabic Corpus in Uthmani script.
