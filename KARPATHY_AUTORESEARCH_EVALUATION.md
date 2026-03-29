# Evaluation: Quran Knowledge Graph + Karpathy's AutoResearch

## TL;DR

**Not a direct fit, but there are indirect and creative integration paths.**

Karpathy's AutoResearch is a narrow, metric-driven optimization loop for ML training code. This project is a knowledge retrieval and exploration system. They solve fundamentally different problems. However, there are several ways this project's architecture and data could enhance or be enhanced by AutoResearch-style thinking.

---

## What Is Karpathy's AutoResearch?

[AutoResearch](https://github.com/karpathy/autoresearch) (released March 2026) is an open-source tool that runs ML experiments autonomously in a loop:

1. An AI coding agent reads `train.py` and `program.md` (human-written research directions)
2. It forms a hypothesis, modifies `train.py`, runs a 5-minute training experiment
3. If `val_bpb` (validation bits-per-byte) improves, the change is kept; otherwise, `git reset`
4. Repeat indefinitely (~12 experiments/hour, ~100 overnight)

Key constraint: it optimizes **a single quantitative metric** (`val_bpb`) on a **single training script** (`train.py`). The human's job is writing `program.md` — plain-English research directions and constraints.

---

## Why It's NOT a Direct Fit

| AutoResearch | Quran Knowledge Graph |
|---|---|
| Optimizes ML training code | Retrieves and synthesizes knowledge |
| Single metric: `val_bpb` | No single optimization target |
| Modifies Python training scripts | Queries a Neo4j graph database |
| Output: faster/better model training | Output: grounded theological analysis |
| Domain: ML hyperparameters, architectures | Domain: Quranic text, thematic connections |

AutoResearch is designed to answer: *"Can I make this training loop faster/better?"*
This project answers: *"What does the Quran say about X, and how are themes connected?"*

---

## Where Integration IS Possible

### 1. AutoResearch-Style Loop for Graph Construction Parameters

The knowledge graph has tunable parameters that directly affect quality:

- **TF-IDF thresholds**: min_df=2, max_df=300, min_score=0.04
- **Edge cap**: max 12 connections per verse
- **Embedding model**: currently `all-MiniLM-L6-v2` (384-dim)
- **Semantic search top_k**: 40 results
- **Traversal hops**: 1-2 hops, top 60/40 results

You could build an AutoResearch-style loop that:
- Varies these parameters in `build_graph.py`
- Rebuilds the graph
- Evaluates quality via a benchmark (e.g., "does the graph correctly connect known thematically-related verses?")
- Keeps improvements, discards regressions

**Feasibility**: Medium. Requires defining a quantitative evaluation metric for graph quality, which is the hard part.

### 2. Optimizing the Agent's System Prompt and Tool Strategy

The chat agent (`chat.py`) has a complex system prompt and 6 tools. An AutoResearch-style loop could:
- Vary the system prompt instructions
- Adjust tool-use strategies (search order, deduplication, synthesis rules)
- Evaluate against a benchmark of questions with known good answers
- Keep prompt versions that produce better-cited, more comprehensive responses

**Feasibility**: High. This is essentially prompt optimization, which is well-suited to iterative experimentation.

### 3. Using the Knowledge Graph AS a Tool Inside AutoResearch

If someone is using AutoResearch to train a model on religious/philosophical text, the Quran Knowledge Graph could serve as an **evaluation oracle**:
- After each training iteration, query the knowledge graph to verify that the model's outputs about Quranic topics are factually grounded
- Use the graph's verse connections as ground truth for measuring a model's understanding of thematic relationships

**Feasibility**: Low-medium. Requires a custom AutoResearch setup targeting a text-understanding model rather than a generic LLM.

### 4. AutoResearch Loop for Embedding Quality

The project embeds all 6,234 verses using `all-MiniLM-L6-v2`. An optimization loop could:
- Try different embedding models or fine-tuning strategies
- Evaluate: "Do semantically similar verses end up closer in embedding space?"
- Use known thematic clusters (e.g., all verses about ablution, forbidden foods) as ground truth
- Iteratively improve the embedding quality

**Feasibility**: High. This is a classic metric-driven optimization that fits AutoResearch perfectly.

### 5. Improving AutoResearch's `program.md` with Structured Knowledge

AutoResearch's power depends on the quality of `program.md` — the human-written research directions. A knowledge graph approach could help write **better research programs**:
- Structure research directions as a graph of hypotheses and dependencies
- Use graph traversal to suggest unexplored research directions
- Track which areas of the search space have been explored vs. neglected

This is more of an architectural inspiration than a direct integration.

---

## The Strongest Integration Path

**Optimizing the retrieval pipeline parameters** (option 1 + 4 above) is the most natural fit:

```
program.md: "Optimize build_graph.py parameters to maximize
retrieval quality on the benchmark question set.
Target metric: answer_quality_score (0-100)."

Loop:
  1. Agent modifies TF-IDF thresholds, edge caps, embedding params
  2. Rebuild graph (fast — ~2 min on CPU)
  3. Run benchmark questions through the chat agent
  4. Score answers against gold-standard responses
  5. Keep if score improves, discard if not
```

This would require:
- A benchmark set of ~50-100 questions with gold-standard answers
- An automated scoring function (could use an LLM-as-judge approach)
- Wrapping the graph build + query pipeline into a single evaluable script

---

## Conclusion

| Aspect | Assessment |
|---|---|
| Direct drop-in integration | No |
| AutoResearch loop for graph params | Yes, with effort |
| AutoResearch loop for prompt optimization | Yes, natural fit |
| Knowledge graph as evaluation oracle | Possible but niche |
| Architectural inspiration both ways | Yes |

The Quran Knowledge Graph is a **knowledge retrieval system**, not an ML training pipeline. AutoResearch is an **ML training optimizer**. They live in different problem spaces. But the AutoResearch *philosophy* — autonomous, metric-driven iteration — can absolutely be applied to optimize this project's retrieval quality, and this project's structured knowledge could inform better research programs.

The most actionable next step would be building a **benchmark evaluation set** for the knowledge graph, which would unlock AutoResearch-style optimization of the entire retrieval pipeline.

---

## Sources

- [karpathy/autoresearch on GitHub](https://github.com/karpathy/autoresearch)
- [VentureBeat: AutoResearch overview](https://venturebeat.com/technology/andrej-karpathys-new-open-source-autoresearch-lets-you-run-hundreds-of-ai)
- [DataCamp: Guide to AutoResearch](https://www.datacamp.com/tutorial/guide-to-autoresearch)
- [SkyPilot: Scaling AutoResearch](https://blog.skypilot.co/scaling-autoresearch/)
- [The New Stack: 630-line script](https://thenewstack.io/karpathy-autonomous-experiment-loop/)
