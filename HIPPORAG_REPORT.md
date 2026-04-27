# HippoRAG-style Traversal Evaluation

*Computed 2026-04-28. Implementation: `hipporag_traverse.py`. Eval: `eval_qrcd_hipporag.py`.*

## What was built

`hipporag_search(query)` combines three signals into a Personalized PageRank
over a Verse-only subgraph:

1. **Vector seeds**: top-K BGE-M3 hits (`alpha_vector` weight)
2. **Past-query seeds**: verses cited in answers to similar past `:Query`
   nodes from `reasoning_memory.py` (`beta_past` weight)
3. **Graph structure**: RELATED_TO + typed edges (SUPPORTS, ELABORATES,
   QUALIFIES, CONTRASTS, REPEATS), each weighted by edge type

PPR runs over a subgraph (~1k–2k nodes) extracted on-demand from the seeds.

## Result on QRCD (22 questions)

| metric    | vector-only | HippoRAG | delta   |
|-----------|-------------|----------|---------|
| hit@5     | 0.5455      | 0.3182   | **−0.23** |
| hit@10    | 0.6364      | 0.5000   | **−0.14** |
| hit@20    | 0.6364      | 0.6364   | +0.00 |
| recall@5  | 0.1252      | 0.0861   | −0.04 |
| recall@10 | 0.1316      | 0.0982   | −0.03 |
| recall@20 | 0.1428      | 0.1342   | −0.01 |
| **mrr**   | **0.4583**  | 0.2332   | **−0.23** |

**HippoRAG underperforms vanilla BGE-M3 retrieval on QRCD by every metric
except hit@20.**

## Why

Two compounding causes:

1. **QRCD is direct-lookup style.** Questions like "Who are the angels
   mentioned in the Quran?" have specific gold passages. PPR's strength is
   multi-hop reasoning ("connect concept A to concept B via verse C"). When
   the answer is a direct semantic match, mixing in graph-neighbors and
   past-query verses dilutes the strong signal.

2. **The past-query seed is essentially noise on QRCD.** Our
   `reasoning_memory.py` writes Query embeddings using `all-MiniLM-L6-v2`
   (English-only, 384-dim). QRCD questions are Arabic. When we embed an
   Arabic question with MiniLM and search the `query_embedding` index, the
   results have no meaningful similarity to the Arabic input — so the
   "past-query seed" is effectively random verses that happen to be cited
   in past English questions.

   **Fix to evaluate later**: re-embed all `:Query` nodes with BGE-M3 and
   use BGE-M3 for past-query lookup. Then the multilingual story holds
   end-to-end.

## What this means

- Don't wire `tool_hipporag_traverse` into `chat.py` yet. The
  implementation is fine, but the conditions for it to win aren't met.
- HippoRAG should help when:
  - Queries are multi-hop ("Compare X and Y in the Quran")
  - The reasoning graph has many similar past queries
  - Both past Q embeddings and current Q embeddings are in the same model space
- The infrastructure is reusable: PPR personalization can take any
  combination of seeds. Future seeds we could try:
  - Verses that share Arabic roots with query terms
  - Verses in the same SemanticDomain
  - Verses linked via tafsir-relation typed edges (when added)

## What was learned

- Negative result is informative — confirms the research-report caveat
  ("HippoRAG promising but unproven; needs eval"; wired-in only if it
  produces measurable lift).
- Surfaces an embedding-space-mixing bug we didn't have eyes on:
  `reasoning_memory.py` Query nodes are still in MiniLM space. Either
  re-embed them or document as a follow-up.
- The PPR machinery itself works correctly; the smoke test on
  `"How does the Quran teach patience?"` produced excellent results
  (top-5: 16:127, 70:5, 46:35, 42:43, 16:126 — all on patience).
  So this isn't a bug in the implementation, it's a mismatch with the
  benchmark.

## Re-test conditions (defer to later)

After re-embedding `:Query` nodes with BGE-M3:
1. Re-run `eval_qrcd_hipporag.py`
2. If still negative on QRCD, try a multi-hop benchmark instead
   (HotpotQA-Quran-style; doesn't exist yet, would have to construct)
3. If positive, wire `tool_hipporag_traverse` into chat.py as an
   alternative retrieval mode that the agent can choose for explicitly
   multi-hop questions.
