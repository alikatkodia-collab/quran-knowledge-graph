# Legacy `verse_embedding` MiniLM index audit

Date: 2026-05-19
Scope: identify what still references the legacy 384-dim
`verse_embedding` vector index, categorise each reference, and bound
the deletion-task scope. **No deletions in this pass — pure audit.**

This audit closes part of Phase 7 item 26 in
[`docs/QKG_RETROFIT_PLAN.md`](../QKG_RETROFIT_PLAN.md):

> **Drop the legacy `verse_embedding` MiniLM vector index** — no
> production callers, measured harmful.

The "measured harmful" claim is grounded in `EVAL_QRCD_REPORT.md`:
BGE-M3-EN beats legacy MiniLM by ~5× on QRCD MAP@10 (n=22).

---

## Inventory

### A. Production-path references (would-fire if `SEMANTIC_SEARCH_INDEX=verse_embedding`)

| File | Line | What | Action |
| --- | ---: | --- | --- |
| [chat.py](../../chat.py) | 66 | Comment: "Override with `SEMANTIC_SEARCH_INDEX=verse_embedding` to use the legacy 384-dim index" | **Update** — remove the override-back-to-legacy framing once deletion lands |
| [chat.py](../../chat.py) | 71 | `_INDEX_TO_MODEL` map entry: `"verse_embedding": None` | **Drop** — closes the env-var escape hatch back to MiniLM |
| [reasoning_memory.py](../../reasoning_memory.py) | 335 | `os.environ.get("SEMANTIC_SEARCH_INDEX", "verse_embedding")` — default falls back to legacy if env var unset | **Update** — change default to `verse_embedding_m3` |
| [reasoning_memory.py](../../reasoning_memory.py) | 337 | Mapping `"verse_embedding" → "minilm-l6-v2"` for provenance writes | **Keep** — purely backward-reads; if Query nodes from before BGE-M3 still exist in Neo4j, their provenance attribution depends on this mapping |

### B. Build-pipeline references (the script that creates the index)

| File | Line | What | Action |
| --- | ---: | --- | --- |
| [embed_verses.py](../../embed_verses.py) | (whole file) | Creates the legacy 384-dim vector index + embeddings | **Delete** — the file ONLY targets the legacy index; `embed_verses_m3.py` is the replacement |

### C. Eval / research references (intentional A/B comparisons)

| File | Line | What | Action |
| --- | ---: | --- | --- |
| [eval_qrcd_retrieval.py](../../eval_qrcd_retrieval.py) | 82 | Lists MiniLM `verse_embedding` as one of three retrieval backends in the QRCD A/B | **Keep** — this script's job is to compare retrieval backends; the historical A/B was the empirical evidence for dropping legacy. Removing the comparator would erase the ability to re-verify the choice. Update the docstring at line 8 to flag the historical/reference-only nature |

### D. Documentation / docstring references (no code path)

| File | Line | What | Action |
| --- | ---: | --- | --- |
| [embed_verses_m3.py](../../embed_verses_m3.py) | 5 | Docstring: "property or the `verse_embedding` index. Instead it adds:" | **Keep** — explains why this script exists (replaces the legacy). Useful context |
| [model_registry.py](../../model_registry.py) | 15 | Comment: `# 384-dim, all-MiniLM-L6-v2 (legacy verse_embedding + Query + AnswerCache)` | **Keep with light edit** — MiniLM is still in use for `Query` (query_embedding 384-dim index) and `AnswerCache`. Strike "legacy verse_embedding" from the comment; the rest stays |

### E. Neo4j index itself (operator action)

The `verse_embedding` index on `Verse.embedding` exists in the live Neo4j database. Dropping it is a one-line Cypher DDL the operator runs:

```cypher
DROP INDEX verse_embedding IF EXISTS;
```

This frees ~6,234 × 384 × 4 bytes ≈ 9.5 MB of index storage + reduces write amplification on any future Verse updates.

The `Verse.embedding` *property itself* (the 384-dim float vectors) is a separate question. Recommendation: **keep the property** — small (9.5 MB) and useful for provenance / future-A-vs-B comparisons. The index is the costly bit; dropping the index alone is sufficient.

---

## Drop-task scope estimate

A bounded session to act on this audit:

```
fix-drop-legacy-minilm-index — ~30 LOC + 1 Neo4j DDL + 3 doc updates

Files touched:
  - chat.py: drop "verse_embedding" key from _INDEX_TO_MODEL; update
    the env-var override comment (lines ~60-67)
  - reasoning_memory.py: change default from "verse_embedding" to
    "verse_embedding_m3" at line 335
  - embed_verses.py: DELETE entirely
  - eval_qrcd_retrieval.py: update docstring at line 8 to note that
    the MiniLM comparator is kept for historical-A/B-reference
  - model_registry.py: light edit to comment at line 15
  - CLAUDE.md: remove the "Vector indexes" entry for verse_embedding;
    note in pipeline section that embed_verses.py is gone

Tests touched:
  - Audit tests/ for any reference to "verse_embedding" (not _m3)
    expecting the legacy path. Likely none, but verify.
  - Add a tiny test asserting the chat.py mapping no longer contains
    the legacy key (regression guard).

Operator action:
  - Run "DROP INDEX verse_embedding IF EXISTS" on local Neo4j
  - Verify queries against verse_embedding_m3 are unaffected
  - Decide whether to drop the Verse.embedding *property* too
    (recommendation: keep)

Estimated total time: ~45 min for the code session + ~5 min for
the operator's DDL + verification.
```

---

## Risks if dropped

- **Old Query nodes lose model attribution**: if any `:Query` node was created when `SEMANTIC_SEARCH_INDEX=verse_embedding` was active, its `embedding_model` provenance is "minilm-l6-v2" via the reasoning_memory map. **Mitigation**: keep the mapping entry in reasoning_memory.py:337 even though the default changes. The mapping is read-only at this point.
- **Eval reproducibility**: re-running `eval_qrcd_retrieval.py` against a Neo4j without the legacy index would fail. **Mitigation**: keep `embed_verses.py` in git history; if a future operator wants to re-run the historical A/B, they can `git show df0d2db:embed_verses.py > tmp.py && python tmp.py`.
- **Cache embeddings remain MiniLM**: `answer_cache.py` and the `query_embedding` index still use MiniLM. They are independent of the verse-embedding index and should not change.

## Risks if NOT dropped

- **Confusion**: new contributors see two vector indexes and one set of embeddings without clear guidance on which is "current." The retrofit plan explicitly flags this.
- **Footgun**: someone sets `SEMANTIC_SEARCH_INDEX=verse_embedding` on a future machine and gets the harmful behaviour silently.
- **Storage**: 9.5 MB in Neo4j and write amplification on every embedding refresh.

## Recommendation

**Fire the drop session.** The audit's findings make the deletion mechanical — no architectural decision pending. The only operator-side step is the one-line `DROP INDEX` DDL, which is reversible (re-create from the property if ever needed).

Time-sensitive? No — this is hygiene, not blocking. Can wait until a convenient operator window.

---

## How this audit was produced

- `grep` for all `verse_embedding` references in `*.py`, excluding `verse_embedding_m3` variants.
- Per-file inspection of each line for: code path / docstring / comment / data.
- Cross-reference with `docs/QKG_RETROFIT_PLAN.md` Phase 7 item 26.
- No code changes; no Neo4j queries; no LLM calls. ~25 min wall time, $0 cost.
