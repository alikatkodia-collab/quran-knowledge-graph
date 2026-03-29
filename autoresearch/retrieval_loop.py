"""
AutoResearch Loop #2: Retrieval Pipeline Optimization

Optimizes the parameters that control how the chat agent searches and
retrieves verses from the knowledge graph. This is separate from the
graph construction optimization (loop.py) — it takes the graph as fixed
and optimizes the retrieval strategy.

Simulates the chat agent's search pipeline in-memory (no Neo4j needed):
  1. Keyword search (MENTIONS edges)
  2. Graph traversal (RELATED_TO edges, 1-2 hops)
  3. Result ranking and deduplication

Evaluates against benchmark queries: for each query, does the retrieval
pipeline find the expected verses?
"""

import copy
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime

csv.field_size_limit(sys.maxsize)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
AUTORESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_PATH = os.path.join(AUTORESEARCH_DIR, "benchmark.json")
RESULTS_TSV = os.path.join(AUTORESEARCH_DIR, "retrieval_results.tsv")
BEST_PARAMS_JSON = os.path.join(AUTORESEARCH_DIR, "best_retrieval_params.json")

# ══════════════════════════════════════════════════════════════════════════════
# TUNABLE RETRIEVAL PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

RETRIEVAL_PARAMS = {
    # Keyword search
    "keyword_prefix_match_len": 5,     # prefix length for fuzzy keyword matching
    "keyword_min_score": 0.0,          # minimum MENTIONS score to include a verse
    "keyword_max_results": 0,          # 0 = unlimited

    # Traversal
    "traverse_seed_limit": 30,         # max direct matches used as hop seeds
    "traverse_hop1_limit": 60,         # max 1-hop results
    "traverse_hop2_limit": 40,         # max 2-hop results
    "traverse_default_hops": 1,        # default hop count (1 or 2)
    "traverse_min_edge_score": 0.0,    # minimum RELATED_TO score to traverse

    # Ranking
    "hop1_score_boost": 0.5,           # multiply hop-1 score by this for ranking
    "hop2_score_boost": 0.25,          # multiply hop-2 score by this for ranking

    # Combined retrieval
    "dedup_strategy": "keep_highest",  # "keep_highest" or "sum_scores"
    "max_total_results": 200,          # max total verses to return across all methods
}

SEARCH_SPACE = {
    "keyword_prefix_match_len": (3, 8, "int", 1),
    "keyword_min_score":        (0.0, 0.1, "float", 0.01),
    "keyword_max_results":      (0, 200, "int", 20),
    "traverse_seed_limit":      (10, 100, "int", 10),
    "traverse_hop1_limit":      (20, 200, "int", 10),
    "traverse_hop2_limit":      (10, 100, "int", 10),
    "traverse_default_hops":    (1, 2, "int", 1),
    "traverse_min_edge_score":  (0.0, 0.5, "float", 0.05),
    "hop1_score_boost":         (0.1, 1.0, "float", 0.1),
    "hop2_score_boost":         (0.05, 0.5, "float", 0.05),
    "max_total_results":        (50, 500, "int", 50),
}

# ══════════════════════════════════════════════════════════════════════════════
# In-memory graph
# ══════════════════════════════════════════════════════════════════════════════

def load_graph():
    """Load graph from CSVs into memory."""
    verses = {}
    with open(os.path.join(DATA_DIR, "verse_nodes.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            verses[row["verseId"]] = row

    mentions = defaultdict(list)      # verse_id -> [(keyword, score)]
    keyword_verses = defaultdict(list) # keyword -> [(verse_id, score)]
    with open(os.path.join(DATA_DIR, "verse_keyword_rels.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid, kw, score = row["verseId"], row["keyword"], float(row["score"])
            mentions[vid].append((kw, score))
            keyword_verses[kw].append((vid, score))

    related = defaultdict(list)
    with open(os.path.join(DATA_DIR, "verse_related_rels.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v1, v2, score = row["verseId1"], row["verseId2"], float(row["score"])
            related[v1].append((v2, score))
            related[v2].append((v1, score))

    return {"verses": verses, "mentions": mentions, "keyword_verses": keyword_verses, "related": related}


# ══════════════════════════════════════════════════════════════════════════════
# Simulated retrieval pipeline (mirrors chat.py logic but in-memory)
# ══════════════════════════════════════════════════════════════════════════════

def search_keyword(graph, keyword, params):
    """Simulate tool_search_keyword: find verses mentioning a keyword."""
    kw_lower = keyword.lower().strip()
    prefix_len = params["keyword_prefix_match_len"]
    min_score = params["keyword_min_score"]
    max_results = params["keyword_max_results"]

    # Find matching keywords (exact or prefix)
    matched_keywords = set()
    for graph_kw in graph["keyword_verses"]:
        if graph_kw == kw_lower:
            matched_keywords.add(graph_kw)
        elif graph_kw[:prefix_len] == kw_lower[:prefix_len]:
            matched_keywords.add(graph_kw)

    results = {}  # verse_id -> best_score
    for kw in matched_keywords:
        for vid, score in graph["keyword_verses"][kw]:
            if score >= min_score:
                if vid not in results or score > results[vid]:
                    results[vid] = score

    # Sort by score descending
    sorted_results = sorted(results.items(), key=lambda x: -x[1])
    if max_results > 0:
        sorted_results = sorted_results[:max_results]

    return {vid: score for vid, score in sorted_results}


def traverse_topic(graph, keywords, params):
    """Simulate tool_traverse_topic: keyword search + graph traversal."""
    # Direct keyword matches
    direct = {}
    for kw in keywords:
        kw_results = search_keyword(graph, kw, params)
        for vid, score in kw_results.items():
            direct[vid] = direct.get(vid, 0) + score

    # Sort and limit seeds
    sorted_direct = sorted(direct.items(), key=lambda x: -x[1])
    seed_ids = [vid for vid, _ in sorted_direct[:params["traverse_seed_limit"]]]
    direct_ids = set(direct.keys())

    # 1-hop traversal
    hop1 = defaultdict(float)
    min_edge = params["traverse_min_edge_score"]
    for sid in seed_ids:
        for neighbor, score in graph["related"].get(sid, []):
            if neighbor not in direct_ids and score >= min_edge:
                hop1[neighbor] += score

    sorted_hop1 = sorted(hop1.items(), key=lambda x: -x[1])[:params["traverse_hop1_limit"]]
    hop1_ids = set(vid for vid, _ in sorted_hop1)

    # 2-hop traversal
    hop2 = defaultdict(int)
    hops = params["traverse_default_hops"]
    if hops >= 2:
        exclude = direct_ids | hop1_ids
        for h1_id, _ in sorted_hop1:
            for neighbor, score in graph["related"].get(h1_id, []):
                if neighbor not in exclude and score >= min_edge:
                    hop2[neighbor] += 1
        sorted_hop2 = sorted(hop2.items(), key=lambda x: -x[1])[:params["traverse_hop2_limit"]]
    else:
        sorted_hop2 = []

    # Combine with score boosts
    combined = {}
    for vid, score in sorted_direct:
        combined[vid] = score
    for vid, score in sorted_hop1:
        boosted = score * params["hop1_score_boost"]
        if params["dedup_strategy"] == "sum_scores" and vid in combined:
            combined[vid] += boosted
        elif vid not in combined or boosted > combined[vid]:
            combined[vid] = boosted
    for vid, count in sorted_hop2:
        boosted = count * params["hop2_score_boost"]
        if params["dedup_strategy"] == "sum_scores" and vid in combined:
            combined[vid] += boosted
        elif vid not in combined or boosted > combined[vid]:
            combined[vid] = boosted

    # Limit total results
    sorted_combined = sorted(combined.items(), key=lambda x: -x[1])
    if params["max_total_results"] > 0:
        sorted_combined = sorted_combined[:params["max_total_results"]]

    return {vid: score for vid, score in sorted_combined}


def full_retrieval(graph, query, params):
    """Simulate the full retrieval pipeline for a query."""
    keywords = query.lower().split()
    # Remove very short words
    keywords = [k for k in keywords if len(k) >= 3]

    # Method 1: Individual keyword searches
    keyword_results = {}
    for kw in keywords:
        for vid, score in search_keyword(graph, kw, params).items():
            if vid not in keyword_results or score > keyword_results[vid]:
                keyword_results[vid] = score

    # Method 2: Traverse with all keywords combined
    traverse_results = traverse_topic(graph, keywords, params)

    # Combine
    combined = {}
    for vid, score in keyword_results.items():
        combined[vid] = score
    for vid, score in traverse_results.items():
        if params["dedup_strategy"] == "sum_scores" and vid in combined:
            combined[vid] += score
        elif vid not in combined or score > combined[vid]:
            combined[vid] = score

    return set(combined.keys())


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_retrieval(graph, params):
    """Evaluate retrieval quality against benchmark queries."""
    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)

    # Metric 1: Query recall — for each benchmark query, what fraction of expected verses are found?
    query_scores = []
    for q in benchmark["retrieval_queries"]:
        found = full_retrieval(graph, q["query"], params)
        expected = set(q["expected_verses"])
        if not expected:
            continue
        recall = len(found & expected) / len(expected)
        query_scores.append(recall)

    query_recall = 100.0 * sum(query_scores) / len(query_scores) if query_scores else 0

    # Metric 2: Cluster retrieval — can we find cluster members using cluster keywords?
    cluster_scores = []
    for cluster in benchmark["clusters"]:
        cluster_keywords = cluster.get("expected_keywords", [])
        if not cluster_keywords:
            continue
        # Use first 3 keywords as query
        query_kws = cluster_keywords[:3]
        found = set()
        for kw in query_kws:
            found |= full_retrieval(graph, kw, params)
        expected = set(cluster["verses"])
        if not expected:
            continue
        recall = len(found & expected) / len(expected)
        cluster_scores.append(recall)

    cluster_recall = 100.0 * sum(cluster_scores) / len(cluster_scores) if cluster_scores else 0

    # Metric 3: Precision proxy — ratio of expected vs total found (penalize noise)
    precision_scores = []
    for q in benchmark["retrieval_queries"]:
        found = full_retrieval(graph, q["query"], params)
        expected = set(q["expected_verses"])
        if not found:
            continue
        # How many of found are in expected?
        precision = len(found & expected) / len(found) if found else 0
        precision_scores.append(precision)

    precision = 100.0 * sum(precision_scores) / len(precision_scores) if precision_scores else 0

    # Metric 4: Result size efficiency — not too many, not too few
    sizes = []
    for q in benchmark["retrieval_queries"]:
        found = full_retrieval(graph, q["query"], params)
        sizes.append(len(found))

    avg_size = sum(sizes) / len(sizes) if sizes else 0
    # Sweet spot: 50-150 results per query
    size_score = 100.0 * math.exp(-0.5 * ((avg_size - 100) / 60) ** 2)

    # Composite (weighted)
    composite = (
        0.35 * query_recall +
        0.30 * cluster_recall +
        0.20 * precision +
        0.15 * size_score
    )

    return {
        "query_recall": round(query_recall, 4),
        "cluster_recall": round(cluster_recall, 4),
        "precision": round(precision, 4),
        "size_efficiency": round(size_score, 4),
        "avg_result_size": round(avg_size, 1),
        "composite_score": round(composite, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Mutation
# ══════════════════════════════════════════════════════════════════════════════

def mutate_retrieval_params(params, n_mutations=None):
    if n_mutations is None:
        n_mutations = random.choice([1, 1, 2, 2, 3])

    new_params = copy.deepcopy(params)
    keys = list(SEARCH_SPACE.keys())
    chosen = random.sample(keys, min(n_mutations, len(keys)))

    mutations = []
    for key in chosen:
        spec = SEARCH_SPACE[key]
        old_val = new_params.get(key)

        if spec[2] == "int":
            lo, hi, _, step = spec
            delta = random.choice([-1, 1]) * step * random.randint(1, 3)
            new_val = max(lo, min(hi, int(old_val + delta)))
        elif spec[2] == "float":
            lo, hi, _, step = spec
            delta = random.choice([-1, 1]) * step * random.uniform(0.5, 3.0)
            new_val = round(max(lo, min(hi, old_val + delta)), 4)
        else:
            continue

        mutations.append(f"{key}: {old_val} -> {new_val}")
        new_params[key] = new_val

    return new_params, mutations


# ══════════════════════════════════════════════════════════════════════════════
# Main Loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Retrieval pipeline optimization")
    parser.add_argument("--max-experiments", type=int, default=500)
    args = parser.parse_args()

    print("=" * 70)
    print("  RETRIEVAL PIPELINE — AutoResearch Optimization Loop")
    print("=" * 70)

    print("\nLoading graph into memory...")
    graph = load_graph()
    print(f"  {len(graph['verses'])} verses, {len(graph['keyword_verses'])} keywords, "
          f"{sum(len(v) for v in graph['related'].values()) // 2} edges")

    # Baseline
    print("\nEvaluating baseline retrieval...")
    current_params = copy.deepcopy(RETRIEVAL_PARAMS)
    baseline = evaluate_retrieval(graph, current_params)
    best_score = baseline["composite_score"]
    best_params = copy.deepcopy(current_params)

    print(f"\n  Baseline breakdown:")
    for k, v in baseline.items():
        print(f"    {k:25s} {v}")
    print(f"\n  BASELINE: {best_score:.4f}")

    improvements = 0
    start_time = time.time()

    # Initialize results file
    if not os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, "w") as f:
            f.write("experiment\ttimestamp\tscore\tbest_score\tkept\tmutations\tparams\n")

    print(f"\n{'='*70}")
    print(f"  Starting retrieval optimization ({args.max_experiments} experiments)...")
    print(f"{'='*70}\n")

    for i in range(1, args.max_experiments + 1):
        new_params, mutations = mutate_retrieval_params(best_params)

        if new_params == best_params:
            new_params, mutations = mutate_retrieval_params(best_params, n_mutations=3)

        try:
            results = evaluate_retrieval(graph, new_params)
            new_score = results["composite_score"]
        except Exception as e:
            print(f"[exp {i:4d}] FAILED: {e}")
            continue

        delta = new_score - best_score

        if new_score > best_score:
            improvements += 1
            print(f"[exp {i:4d}] IMPROVED! {best_score:.4f} -> {new_score:.4f} (+{delta:.4f}) "
                  f"| {', '.join(mutations)}")
            for k, v in results.items():
                if k != "composite_score":
                    print(f"          {k:25s} {v}")

            best_score = new_score
            best_params = copy.deepcopy(new_params)

            with open(BEST_PARAMS_JSON, "w") as f:
                json.dump({"score": best_score, "params": best_params, "experiment": i}, f, indent=2)

            with open(RESULTS_TSV, "a") as f:
                f.write(f"{i}\t{datetime.now().isoformat()}\t{new_score:.4f}\t{best_score:.4f}\t"
                        f"True\t{'; '.join(mutations)}\t{json.dumps(new_params)}\n")
        else:
            if i % 50 == 0:
                print(f"[exp {i:4d}] best: {best_score:.4f} (no improvement in this batch)")

            with open(RESULTS_TSV, "a") as f:
                f.write(f"{i}\t{datetime.now().isoformat()}\t{new_score:.4f}\t{best_score:.4f}\t"
                        f"False\t{'; '.join(mutations)}\t{json.dumps(new_params)}\n")

        if i % 100 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed * 3600
            print(f"\n  --- Progress: {i}/{args.max_experiments}, {improvements} improvements, "
                  f"best: {best_score:.4f}, rate: {rate:.0f} exp/hr ---\n")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  RETRIEVAL OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Experiments:   {args.max_experiments}")
    print(f"  Improvements:  {improvements}")
    print(f"  Final score:   {best_score:.4f}")
    print(f"  Time:          {total_time/60:.1f} minutes")
    print(f"  Rate:          {args.max_experiments/total_time*3600:.0f} exp/hr")
    print(f"\n  Best params: {BEST_PARAMS_JSON}")
    print(f"  Results log: {RESULTS_TSV}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
