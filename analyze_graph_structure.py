"""
analyze_graph_structure.py — Network Science measurements over the Quran graph.

What it computes (over RELATED_TO + typed edges between Verses):
  1. Degree distribution + power-law fit indicator (alpha estimate)
  2. Top-K hub verses by degree (theological "load-bearing" candidates)
  3. Top-K bridge verses by approximate betweenness centrality (sampled)
  4. Communities via greedy modularity + modularity score
     (Louvain would be better but python-louvain isn't installed; modularity score
      is comparable. Higher = more well-defined communities.)
  5. Average shortest path length (sample-based, full APSP is too slow)
  6. Connected components

Output: data/graph_stats.json — one snapshot, with timestamp + edge counts.

Why this matters:
  - Hub verses are good candidates for special UI treatment + retrieval boosting
  - Bridge verses connect different thematic clusters and are easy to miss
  - Modularity tells us whether community-based retrieval would even be meaningful
  - Power-law fit indicates whether the graph is scale-free (likely)

Run: python analyze_graph_structure.py
"""

import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

import networkx as nx

load_dotenv()

URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
USER     = os.getenv("NEO4J_USER",     "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "")
DB       = os.getenv("NEO4J_DATABASE", "quran")

OUT = Path(__file__).parent / "data" / "graph_stats.json"

# Approx betweenness sample size (full BC is O(VE) ~3e8 ops on this graph)
BC_SAMPLE = 500
# Avg shortest path sample size
APSP_SAMPLE = 200
# Top-K hubs/bridges to surface
TOP_K = 30


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_graph(driver) -> nx.Graph:
    """Pull all RELATED_TO + typed edges between Verse nodes."""
    print("Fetching edges from Neo4j...")
    with driver.session(database=DB) as s:
        # Untyped semantic similarity edges
        related = s.run("""
            MATCH (a:Verse)-[r:RELATED_TO]->(b:Verse)
            RETURN a.verseId AS src, b.verseId AS dst, r.score AS w
        """).data()
        # Typed thematic edges
        typed = s.run("""
            MATCH (a:Verse)-[r]->(b:Verse)
            WHERE type(r) IN ['SUPPORTS','ELABORATES','QUALIFIES','CONTRASTS','REPEATS']
            RETURN a.verseId AS src, b.verseId AS dst, type(r) AS t
        """).data()

    print(f"  RELATED_TO edges: {len(related):,}")
    print(f"  typed edges: {len(typed):,}")

    g = nx.Graph()  # undirected — semantic similarity is symmetric in spirit
    for r in related:
        if r["src"] and r["dst"]:
            g.add_edge(r["src"], r["dst"], weight=float(r.get("w") or 1.0), kind="related_to")
    for r in typed:
        if r["src"] and r["dst"]:
            # If already an edge, augment it with typed metadata; otherwise add
            if g.has_edge(r["src"], r["dst"]):
                g[r["src"]][r["dst"]].setdefault("types", []).append(r["t"])
            else:
                g.add_edge(r["src"], r["dst"], weight=1.0, kind="typed", types=[r["t"]])
    return g


def degree_distribution(g: nx.Graph):
    deg = dict(g.degree())
    counts = Counter(deg.values())
    return deg, counts


def estimate_powerlaw_alpha(degrees: list[int], min_deg: int = 2) -> dict:
    """
    Naive Hill-style estimator for the power-law exponent alpha.
    alpha ≈ 1 + n / sum(ln(d_i / (min_deg - 0.5)))
    A fit-quality proxy: skew of log-log degree counts.
    """
    sample = [d for d in degrees if d >= min_deg]
    n = len(sample)
    if n < 50:
        return {"alpha": None, "n_used": n, "min_deg": min_deg, "note": "sample too small"}
    s = sum(math.log(d / (min_deg - 0.5)) for d in sample)
    alpha = 1.0 + n / s if s > 0 else None
    return {
        "alpha": round(alpha, 3) if alpha else None,
        "n_used": n,
        "min_deg": min_deg,
        "interpretation": (
            "scale-free-ish (typical 2.0-3.0)" if alpha and 2.0 <= alpha <= 3.0
            else ("heavy-tailed but not classical scale-free"
                  if alpha and alpha < 2.0 else "thin-tailed / not scale-free")
        ) if alpha else None,
    }


def main():
    print(f"Connecting to Neo4j ({URI}, db={DB})...")
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    driver.verify_connectivity()
    print("  Connected OK\n")

    g = fetch_graph(driver)
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    print(f"\nGraph: {n_nodes:,} nodes, {n_edges:,} unique undirected edges\n")

    stats = {
        "computed_at": _now_iso(),
        "nodes": n_nodes,
        "edges": n_edges,
        "edge_density": round(2 * n_edges / (n_nodes * (n_nodes - 1)), 6) if n_nodes > 1 else 0,
    }

    # ── Components ───────────────────────────────────────────────────────────
    print("Connected components ...")
    comps = list(nx.connected_components(g))
    comps.sort(key=len, reverse=True)
    stats["connected_components"] = {
        "count": len(comps),
        "largest_size": len(comps[0]) if comps else 0,
        "isolated_nodes": sum(1 for c in comps if len(c) == 1),
    }
    print(f"  {len(comps)} components, largest = {len(comps[0]):,}, isolated = {stats['connected_components']['isolated_nodes']}")

    # Use giant component for path/modularity work
    giant = g.subgraph(comps[0]).copy()

    # ── Degree distribution ──────────────────────────────────────────────────
    print("\nDegree distribution ...")
    deg, counts = degree_distribution(g)
    deg_values = sorted(deg.values(), reverse=True)
    stats["degree"] = {
        "max": deg_values[0] if deg_values else 0,
        "mean": round(sum(deg_values) / len(deg_values), 2) if deg_values else 0,
        "median": deg_values[len(deg_values) // 2] if deg_values else 0,
        "p90": deg_values[int(len(deg_values) * 0.10)] if deg_values else 0,
        "p99": deg_values[int(len(deg_values) * 0.01)] if deg_values else 0,
        "histogram_top10": [
            {"degree": d, "verses": c}
            for d, c in counts.most_common(10)
        ],
    }
    print(f"  max={stats['degree']['max']}, mean={stats['degree']['mean']}, median={stats['degree']['median']}, p90={stats['degree']['p90']}, p99={stats['degree']['p99']}")

    stats["powerlaw"] = estimate_powerlaw_alpha(deg_values)
    print(f"  power-law alpha estimate: {stats['powerlaw']}")

    # ── Top-K hubs ───────────────────────────────────────────────────────────
    print(f"\nTop {TOP_K} hub verses by degree ...")
    top_hubs = sorted(deg.items(), key=lambda kv: -kv[1])[:TOP_K]
    stats["top_hubs"] = [{"verseId": v, "degree": d} for v, d in top_hubs]
    for v, d in top_hubs[:10]:
        print(f"  [{v}] degree={d}")

    # ── Approx betweenness centrality ────────────────────────────────────────
    print(f"\nApprox betweenness centrality (k={BC_SAMPLE} sample) ...")
    t0 = time.time()
    bc = nx.betweenness_centrality(giant, k=BC_SAMPLE, normalized=True, seed=42)
    bc_top = sorted(bc.items(), key=lambda kv: -kv[1])[:TOP_K]
    stats["top_bridges"] = [{"verseId": v, "betweenness": round(b, 5)} for v, b in bc_top]
    print(f"  computed in {time.time() - t0:.1f}s")
    for v, b in bc_top[:10]:
        print(f"  [{v}] betweenness={b:.4f}")

    # ── Communities + modularity (greedy modularity, no python-louvain dep) ──
    print("\nCommunities via greedy modularity (slow but no extra deps) ...")
    t0 = time.time()
    try:
        communities = list(nx.community.greedy_modularity_communities(giant, weight="weight"))
        modularity = nx.community.modularity(giant, communities, weight="weight")
        sizes = sorted([len(c) for c in communities], reverse=True)
        stats["communities"] = {
            "method": "greedy_modularity",
            "count": len(communities),
            "modularity": round(modularity, 4),
            "size_distribution": {
                "max": sizes[0] if sizes else 0,
                "median": sizes[len(sizes) // 2] if sizes else 0,
                "min": sizes[-1] if sizes else 0,
                "top10": sizes[:10],
            },
            "interpretation": (
                "well-defined communities" if modularity >= 0.4
                else "moderately defined" if modularity >= 0.3
                else "weakly defined — community-based retrieval may not help"
            ),
        }
        print(f"  {len(communities)} communities, modularity={modularity:.4f}")
        print(f"  ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"  community detection failed: {e}")
        stats["communities"] = {"error": str(e)}

    # ── Avg shortest path (sampled) ─────────────────────────────────────────
    print(f"\nAvg shortest path length (sample of {APSP_SAMPLE} pairs) ...")
    nodes = list(giant.nodes)
    random.seed(42)
    sample_pairs = [(random.choice(nodes), random.choice(nodes))
                    for _ in range(APSP_SAMPLE)]
    sample_pairs = [(a, b) for a, b in sample_pairs if a != b]
    lengths = []
    for a, b in sample_pairs:
        try:
            lengths.append(nx.shortest_path_length(giant, a, b))
        except nx.NetworkXNoPath:
            pass
    if lengths:
        stats["avg_shortest_path"] = {
            "sampled_pairs": len(lengths),
            "mean": round(sum(lengths) / len(lengths), 3),
            "max": max(lengths),
        }
        print(f"  mean = {stats['avg_shortest_path']['mean']}, max = {stats['avg_shortest_path']['max']}")
    else:
        stats["avg_shortest_path"] = {"error": "no reachable pairs found"}

    # ── Write ─────────────────────────────────────────────────────────────────
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}")

    driver.close()
    print("Done.")


if __name__ == "__main__":
    main()
