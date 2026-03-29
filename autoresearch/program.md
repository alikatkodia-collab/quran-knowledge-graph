# AutoResearch Program — Quran Knowledge Graph Optimization

## Objective

Maximize the **composite quality score** of the Quran Knowledge Graph by optimizing the parameters that control how the graph is built from 6,234 Quranic verses.

**Primary metric:** `composite_score` (weighted average of 7 sub-metrics, range 0-100)

**Target:** Improve from baseline 75.47 to >82.

## Architecture

```
train.py (PARAMS dict)  →  build graph CSVs  →  evaluate.py (7 metrics)  →  composite_score
     ↑                                                                          |
     └──── if improved: keep params, commit  ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←┘
           if not: revert params
```

## What the Agent Modifies

Only `autoresearch/train.py` — specifically the `PARAMS` dict at the top:

| Parameter | Current | Range | Effect |
|-----------|---------|-------|--------|
| min_df | 2 | 1-10 | Min verses a keyword must appear in |
| max_df | 300 | 50-1000 | Max verses before keyword is "too common" |
| min_tfidf_score | 0.04 | 0.01-0.15 | Min TF-IDF to create MENTIONS edge |
| max_features | 50000 | 5000-100000 | Max vocabulary size |
| max_edges_per_verse | 12 | 4-30 | Cap on RELATED_TO edges per verse |
| max_verse_freq | 300 | 50-1000 | Skip keywords in >N verses for RELATED_TO |
| edge_weight_method | geometric_mean | geometric_mean/harmonic_mean/min | How to combine TF-IDF scores |
| min_token_length | 3 | 2-4 | Minimum token character length |
| sublinear_tf | False | True/False | Logarithmic term frequency |
| ngram_max | 1 | 1-2 | Include bigrams? |
| norm | l2 | l1/l2 | TF-IDF normalization |
| lemma_verb_first | True | True/False | Lemmatization order |

## Evaluation Metrics (fixed, DO NOT modify)

| Metric | Weight | Baseline | Description |
|--------|--------|----------|-------------|
| cluster_coherence | 30% | 35.53 | Are known thematic clusters connected within 2 hops? |
| retrieval_recall | 20% | 79.96 | Can keyword search find known relevant verses? |
| cross_surah_ratio | 10% | 93.72 | % of edges connecting different surahs |
| edge_density | 10% | 99.72 | Edges/verse near ideal (~8) |
| keyword_coverage | 10% | 94.83 | Expected benchmark keywords present? |
| vocabulary_efficiency | 10% | 99.88 | Vocabulary size near ideal (~2700) |
| avg_edge_weight | 10% | 100.00 | Average connection strength |

## Research Priorities

### Priority 1: Cluster Coherence (35.53 → target >55)
This is the weakest metric by far. Ground-truth clusters (e.g., all ablution verses, all Noah-flood verses) should be connected within 2 hops. Approaches:
- **Increase max_edges_per_verse** to allow more connections per verse
- **Lower min_tfidf_score** to capture weaker but real thematic links
- **Increase max_verse_freq** to allow common-but-meaningful keywords to create edges
- **Try bigrams** (ngram_max=2) to capture multi-word concepts like "day judgment"
- **Try sublinear_tf=True** to reduce dominance of high-frequency terms

### Priority 2: Maintain Retrieval Recall (keep >75)
Don't sacrifice keyword search quality while improving clustering.

### Priority 3: Maintain Other Metrics (keep >85 each)
Don't break what works. Cross-surah ratio, edge density, and keyword coverage should stay high.

## Constraints

- Build time must stay under 60 seconds
- Do NOT modify evaluate.py or benchmark.json
- Do NOT modify the build() function logic — only PARAMS values
- The optimization loop handles mutation, build, eval, and keep/discard automatically

## Strategy Notes

- The ratchet is monotonic: only improvements are kept
- ~6 seconds per experiment = ~600 experiments per hour
- Focus 60% of mutations on the weakest metric (cluster_coherence)
- 40% random exploration to find unexpected improvements
- Expect diminishing returns after ~50-80 experiments
