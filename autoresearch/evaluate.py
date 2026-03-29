"""
Evaluation engine for the Quran Knowledge Graph AutoResearch loop.

Computes a composite quality score from the graph CSVs without Neo4j.
Uses ground-truth thematic clusters and retrieval queries as benchmarks.

DO NOT MODIFY THIS FILE — it is the fixed evaluation oracle.
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict

csv.field_size_limit(sys.maxsize)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.json")


def load_benchmark():
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_graph_csvs(data_dir=None):
    """Load all graph CSVs into memory. Returns dict of parsed data."""
    if data_dir is None:
        data_dir = os.path.join(BASE_DIR, "data")

    # verse nodes
    verses = {}
    with open(os.path.join(data_dir, "verse_nodes.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            verses[row["verseId"]] = row

    # keywords
    keywords = set()
    with open(os.path.join(data_dir, "keyword_nodes.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            keywords.add(row["keyword"])

    # verse-keyword edges (MENTIONS)
    mentions = defaultdict(list)  # verse_id -> [(keyword, score)]
    keyword_verses = defaultdict(list)  # keyword -> [(verse_id, score)]
    with open(os.path.join(data_dir, "verse_keyword_rels.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row["verseId"]
            kw = row["keyword"]
            score = float(row["score"])
            mentions[vid].append((kw, score))
            keyword_verses[kw].append((vid, score))

    # verse-verse edges (RELATED_TO)
    related = defaultdict(list)  # verse_id -> [(other_verse_id, score)]
    all_edges = []
    with open(os.path.join(data_dir, "verse_related_rels.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v1, v2, score = row["verseId1"], row["verseId2"], float(row["score"])
            related[v1].append((v2, score))
            related[v2].append((v1, score))
            all_edges.append((v1, v2, score))

    return {
        "verses": verses,
        "keywords": keywords,
        "mentions": mentions,
        "keyword_verses": keyword_verses,
        "related": related,
        "all_edges": all_edges,
    }


# ── Metric 1: Cluster Coherence ──────────────────────────────────────────────
# For each ground-truth cluster, what fraction of verse pairs are connected
# (directly or within 2 hops)?

def _bfs_reachable(related, start, max_hops=2):
    """BFS from start, return set of reachable verse IDs within max_hops."""
    visited = {start}
    frontier = [start]
    for _ in range(max_hops):
        next_frontier = []
        for vid in frontier:
            for neighbor, _ in related.get(vid, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
    return visited


def metric_cluster_coherence(graph, benchmark):
    """Score 0-100: how well ground-truth clusters are connected in the graph."""
    clusters = benchmark["clusters"]
    cluster_scores = []

    for cluster in clusters:
        verse_ids = [v for v in cluster["verses"] if v in graph["verses"]]
        if len(verse_ids) < 2:
            continue

        # For each pair, check if connected within 2 hops
        connected_pairs = 0
        total_pairs = 0
        for i in range(len(verse_ids)):
            reachable = _bfs_reachable(graph["related"], verse_ids[i], max_hops=2)
            for j in range(i + 1, len(verse_ids)):
                total_pairs += 1
                if verse_ids[j] in reachable:
                    connected_pairs += 1

        if total_pairs > 0:
            cluster_scores.append(connected_pairs / total_pairs)

    if not cluster_scores:
        return 0.0
    return 100.0 * sum(cluster_scores) / len(cluster_scores)


# ── Metric 2: Keyword Retrieval Recall ────────────────────────────────────────
# For each retrieval query, what fraction of expected verses have the keyword?

def metric_retrieval_recall(graph, benchmark):
    """Score 0-100: keyword search recall on benchmark queries."""
    queries = benchmark["retrieval_queries"]
    scores = []

    for q in queries:
        keyword = q["query"].lower()
        # Find all verses that mention this keyword (or lemmatized variant)
        found_verses = set()
        for kw in graph["keyword_verses"]:
            if kw == keyword or kw.startswith(keyword[:5]):
                for vid, _ in graph["keyword_verses"][kw]:
                    found_verses.add(vid)

        expected = set(q["expected_verses"])
        if not expected:
            continue
        recall = len(found_verses & expected) / len(expected)
        scores.append(recall)

    if not scores:
        return 0.0
    return 100.0 * sum(scores) / len(scores)


# ── Metric 3: Cross-Surah Connectivity ───────────────────────────────────────
# What fraction of RELATED_TO edges connect verses from different surahs?

def metric_cross_surah_ratio(graph, benchmark=None):
    """Score 0-100: fraction of edges that are cross-surah."""
    if not graph["all_edges"]:
        return 0.0

    cross = 0
    for v1, v2, _ in graph["all_edges"]:
        s1 = graph["verses"].get(v1, {}).get("surah", "")
        s2 = graph["verses"].get(v2, {}).get("surah", "")
        if s1 != s2:
            cross += 1

    return 100.0 * cross / len(graph["all_edges"])


# ── Metric 4: Edge Density Balance ───────────────────────────────────────────
# Penalize too few edges (disconnected) and too many (noise). Sweet spot matters.

def metric_edge_density(graph, benchmark=None):
    """Score 0-100: penalizes both under- and over-connectivity."""
    n_verses = len(graph["verses"])
    n_edges = len(graph["all_edges"])
    if n_verses == 0:
        return 0.0

    edges_per_verse = n_edges / n_verses
    # Sweet spot: 10-20 edges per verse for a knowledge graph
    # Wider bell curve centered at 14.0 to allow more connections
    ideal = 14.0
    sigma = 8.0
    score = math.exp(-0.5 * ((edges_per_verse - ideal) / sigma) ** 2)
    return 100.0 * score


# ── Metric 5: Keyword Quality ────────────────────────────────────────────────
# Check if expected keywords from benchmark clusters appear in the keyword set.

def metric_keyword_coverage(graph, benchmark):
    """Score 0-100: fraction of expected benchmark keywords found in graph."""
    all_expected = set()
    for cluster in benchmark["clusters"]:
        for kw in cluster.get("expected_keywords", []):
            all_expected.add(kw.lower())

    if not all_expected:
        return 0.0

    found = 0
    for expected_kw in all_expected:
        # Check exact match or prefix match (to handle lemmatization)
        for graph_kw in graph["keywords"]:
            if graph_kw == expected_kw or expected_kw.startswith(graph_kw) or graph_kw.startswith(expected_kw):
                found += 1
                break

    return 100.0 * found / len(all_expected)


# ── Metric 6: Vocabulary Size Efficiency ─────────────────────────────────────
# Too few keywords = underfitting, too many = noise

def metric_vocabulary_efficiency(graph, benchmark=None):
    """Score 0-100: penalizes extreme vocabulary sizes."""
    n_keywords = len(graph["keywords"])
    # Sweet spot: 2000-5000 keywords (wider range to allow bigrams)
    ideal = 3000
    sigma = 1500
    score = math.exp(-0.5 * ((n_keywords - ideal) / sigma) ** 2)
    return 100.0 * score


# ── Metric 7: Average Edge Weight ────────────────────────────────────────────
# Higher average weight = stronger, more meaningful connections

def metric_avg_edge_weight(graph, benchmark=None):
    """Score 0-100: normalized average edge weight."""
    if not graph["all_edges"]:
        return 0.0
    avg = sum(s for _, _, s in graph["all_edges"]) / len(graph["all_edges"])
    # Typical range 0.1-0.5, normalize to 0-100
    return min(100.0, 100.0 * avg / 0.4)


# ── Composite Score ──────────────────────────────────────────────────────────

WEIGHTS = {
    "cluster_coherence":   0.30,  # Most important: do thematic clusters connect?
    "retrieval_recall":    0.20,  # Can we find known relevant verses?
    "cross_surah_ratio":   0.10,  # Cross-surah connections show deep themes
    "edge_density":        0.10,  # Not too sparse, not too dense
    "keyword_coverage":    0.10,  # Do we capture expected keywords?
    "vocabulary_efficiency": 0.10,  # Right vocabulary size
    "avg_edge_weight":     0.10,  # Meaningful connections
}

METRICS = {
    "cluster_coherence":   metric_cluster_coherence,
    "retrieval_recall":    metric_retrieval_recall,
    "cross_surah_ratio":   metric_cross_surah_ratio,
    "edge_density":        metric_edge_density,
    "keyword_coverage":    metric_keyword_coverage,
    "vocabulary_efficiency": metric_vocabulary_efficiency,
    "avg_edge_weight":     metric_avg_edge_weight,
}


def evaluate(data_dir=None):
    """Run all metrics and return composite score + breakdown."""
    benchmark = load_benchmark()
    graph = load_graph_csvs(data_dir)

    results = {}
    for name, func in METRICS.items():
        results[name] = round(func(graph, benchmark), 4)

    composite = sum(results[name] * WEIGHTS[name] for name in WEIGHTS)
    results["composite_score"] = round(composite, 4)

    return results


if __name__ == "__main__":
    results = evaluate()
    print(f"\n{'='*60}")
    print(f"  QURAN KNOWLEDGE GRAPH — EVALUATION RESULTS")
    print(f"{'='*60}")
    for name, score in results.items():
        if name == "composite_score":
            continue
        weight = WEIGHTS.get(name, 0)
        print(f"  {name:30s} {score:7.2f}  (weight: {weight:.0%})")
    print(f"{'='*60}")
    print(f"  {'COMPOSITE SCORE':30s} {results['composite_score']:7.2f}")
    print(f"{'='*60}")
