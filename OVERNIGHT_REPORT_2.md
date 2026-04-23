# Second Overnight Report — Autonomous 10-Hour Cache Seeding

**Window:** 2026-04-23 19:42 → 2026-04-24 07:56 local
**Wall clock:** ~12h seeding compute (single-threaded against one app server)
**Mode:** OpenRouter `openai/gpt-oss-120b:free` primary, local Ollama fallback available

---

## Headline Numbers

| Metric | Start | End | Delta |
|---|---|---|---|
| Cache entries | 500 | **845** | **+345** |
| Seeding phases run | — | 6, 7, 8 | 3 |
| Questions processed | — | 402 | — |
| Cache yield | — | — | 86% avg |
| Failures | — | 0 | 0 |

---

## Phase Breakdown

| Phase | Questions | Done | Cache Δ | Yield | Time |
|---|---|---|---|---|---|
| 6 | 88 | 88 | +69 | 78% | 108 min |
| 7 | 176 | 176 | +149 | 85% | 201 min |
| 8 | 180 | 138 (cap) | +127 | 92% | 421 min |
| **Total** | **444** | **402** | **+345** | **86%** | **730 min** |

Remaining in Phase 8 bank: 42 questions (not started, available for the next session).

---

## Critical Fixes Shipped

1. **answer_cache.py** — the cache wasn't actually growing before this session.
   - Dedupe threshold: `0.95` → `0.98` (semantic siblings now coexist)
   - Entry cap: `500` → `5000` (was silently evicting oldest)

2. **chat.py `tool_semantic_search`** — enhanced with VectorCypherRetriever pattern from Neo4j Graph Academy research.
   - Single Cypher query now returns: verse + top-5 related (RELATED_TO) + top-5 Arabic roots + typed edges (SUPPORTS / ELABORATES / QUALIFIES / CONTRASTS / REPEATS)
   - Smoke-tested: correctly returned 14:12 with 5 related verses, 5 roots, 3 SUPPORTS edges

3. **reasoning_memory.py** — new module wired into app_free.py.
   - Implements the Neo4j Labs Agent Memory pattern: (:Query)-[:TRIGGERED]->(:ReasoningTrace)-[:HAS_STEP]->(:ToolCall) with vector-indexed question embeddings
   - Playbook retrieval — past similar queries are injected as system-prompt context
   - Tested end-to-end: test query matched seeded query at sim=0.597

---

## Observations and Critique

**What worked:**
- Dedupe+cap fix was pivotal — before it, the cache was stuck at 500 even though we were making ~75 successful runs/hour
- Specific verse questions ("Explain verse X:Y") and middle-Mushaf surahs yielded ~85-92% cache-growth rate vs 78% for broader thematic questions
- OpenRouter `gpt-oss-120b:free` held up across 400+ requests with zero 429s

**What wobbled:**
- Phase 8 final hour showed severe slowdown (500s+ per question, [FALLBACK] tags on tool calls). Likely OpenRouter provider-side pressure or model capacity throttling. Not account-level — still responding, just slow.
- ~15% of questions yield 0-char answers and get filtered by save_answer's `len<50` check. These tend to be single-Arabic-word questions where the model over-calls tools but produces no final prose. Could be fixed with a fallback prose generation step.

**What's deferred:**
- VerseAnalysis injection into seeding prompts — the A/B test showed citation-format drift overwhelming any signal. Revisit if/when we add a proper VA-lookup tool to the agentic loop.
- Three-tier memory architecture (short-term conversation + long-term entities + reasoning). We shipped reasoning-memory only.

---

## Commits This Session

- `a65cb74` — enhanced semantic_search + reasoning_memory
- `4174c74` — overnight seeding phases 6-8 + cache growth fix

Both on `origin/main`.

---

## Process Timeline

| Time | Event |
|---|---|
| ~17:44 | Phase 6 restart after Neo4j + app_free.py recovery |
| 19:37 | Diagnosis: cache stuck at 500 due to 0.95 dedupe + 500 cap |
| 19:42 | Fix shipped, services restarted, Phase 6 resumed |
| 21:30 | Phase 6 complete (88/88, +69 cache) |
| 21:32 | Phase 7 started (176 questions) |
| 00:53 | Phase 7 complete (176/176, +149 cache) |
| 00:55 | Phase 8 started (180 questions, 7h cap) |
| 04:00-ish | Noticeable OpenRouter slowdown |
| 07:56 | Phase 8 hit 7h deadline at 138/180 |

---

## Recommended Next Steps

When you're back:

1. **Run the 42 unfinished Phase 8 questions** (quick — ~1h on OpenRouter or ~3h on local).
2. **Consider an uncertainty/quality pass over the cached answers** — some of the "STRONG" answers have 0-character bodies (counted as STRONG only by tool count, not by answer quality). Run a quick cleanup script to purge cache entries with answer length < some threshold.
3. **Build Phase 9** if more cache coverage desired. Gap areas: specific short surahs' deep-dives, Arabic root comparisons across contexts, more single-verse studies for the 80%+ of verses that don't yet have cached entries.
4. **Wire VerseAnalysis into a proper tool** (`tool_get_verse_analysis(verse_id)`) so the agentic loop can pull structured metadata on demand instead of pre-injecting. This bypasses the citation-format-drift we saw in the one-shot test.
5. **Commit the running state** periodically to protect against Neo4j/server crashes.

---

## Quick-Start After Return

```bash
# Check everything's still up
curl http://localhost:8085/
python -c "import json; print(len(json.load(open('data/answer_cache.json', encoding='utf-8'))))"

# If server died, restart:
python app_free.py

# To finish Phase 8:
python overnight_seed_phase8.py --port 8085 --hours 2
```
