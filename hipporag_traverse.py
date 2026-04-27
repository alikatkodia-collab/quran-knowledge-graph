"""
hipporag_traverse.py — HippoRAG-style retrieval combining vector search,
graph structure, and reasoning-memory.

HippoRAG (OSU-NLP-Group, NeurIPS 2024) uses Personalized PageRank over a
knowledge graph where seed nodes come from passage-level dense retrieval.
We adapt the idea to the Quran Knowledge Graph:

  Personalization vector =
    α * (top-K verses by BGE-M3 semantic similarity)
    + β * (verses cited in answers to similar past Queries)

  Edges considered:
    Verse -[:RELATED_TO]- Verse           (51K shared-keyword edges)
    Verse -[:SUPPORTS|ELABORATES|...]- Verse  (7K typed edges)
    Verse <-[:NEXT_VERSE]- Verse              (sequential)

PPR is run over this Verse-only subgraph; the result is a score per verse
that combines direct topicality with structural and historical signals.

Why this could outperform plain semantic search:
  - past_verses captures "what the system found useful for similar questions"
  - graph traversal surfaces verses thematically near a hit but not directly
    matching the query string
  - PPR's random-walk semantics naturally weights well-connected hubs

To use as a tool in chat.py:
  from hipporag_traverse import hipporag_search
  result = hipporag_search(session, query, top_k=20)

CLI smoke test:
  python hipporag_traverse.py "How does the Quran teach patience?"
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dotenv import load_dotenv
from neo4j import GraphDatabase
import networkx as nx

load_dotenv()

URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
USER     = os.getenv("NEO4J_USER",     "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "")
DB       = os.getenv("NEO4J_DATABASE", "quran")

# Index for verse semantic search — defaults to BGE-M3 if available
VERSE_INDEX = os.environ.get("SEMANTIC_SEARCH_INDEX", "verse_embedding_m3")
QUERY_INDEX = "query_embedding"   # MiniLM 384-dim past-query vector index


# ── lazy embedders ───────────────────────────────────────────────────────────
_verse_embedder = None
_query_embedder = None  # for past-Query similarity

def _get_verse_embedder():
    global _verse_embedder
    if _verse_embedder is None:
        from sentence_transformers import SentenceTransformer
        if VERSE_INDEX in ("verse_embedding_m3", "verse_embedding_m3_ar"):
            m = SentenceTransformer("BAAI/bge-m3")
            m.max_seq_length = 512
        else:
            m = SentenceTransformer("all-MiniLM-L6-v2")
        _verse_embedder = m
    return _verse_embedder

def _get_query_embedder():
    global _query_embedder
    if _query_embedder is None:
        from sentence_transformers import SentenceTransformer
        _query_embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _query_embedder


# ── core ─────────────────────────────────────────────────────────────────────

def _vector_seed(session, query: str, top_k: int = 30) -> list[tuple[str, float]]:
    """Vector-search top-K verses using the configured index."""
    m = _get_verse_embedder()
    vec = m.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
    rows = session.run(
        "CALL db.index.vector.queryNodes($idx, $k, $vec) YIELD node, score "
        "WHERE node.verseId IS NOT NULL "
        "RETURN node.verseId AS id, score ORDER BY score DESC",
        idx=VERSE_INDEX, k=top_k, vec=vec,
    ).data()
    return [(r["id"], float(r["score"])) for r in rows]


def _past_query_seed(session, query: str, top_k: int = 5,
                     min_sim: float = 0.7) -> list[tuple[str, float]]:
    """
    Find verses cited in answers to similar past Query nodes.
    Uses MiniLM 384-dim because that's what reasoning_memory.py wrote.
    """
    m = _get_query_embedder()
    vec = m.encode(query, normalize_embeddings=True).tolist()
    try:
        rows = session.run(
            "CALL db.index.vector.queryNodes($idx, $k, $vec) "
            "YIELD node, score WHERE score >= $min "
            "MATCH (node)-[:PRODUCED]->(a:Answer) "
            "UNWIND a.cited_verses AS vid "
            "RETURN vid AS id, score, count(*) AS occurrences",
            idx=QUERY_INDEX, k=top_k, vec=vec, min=min_sim,
        ).data()
    except Exception:
        # Index may not exist or be empty
        return []
    # Aggregate: a verse cited in 3 of 5 similar past queries is stronger
    agg: dict[str, float] = defaultdict(float)
    for r in rows:
        agg[r["id"]] += float(r["score"]) * float(r["occurrences"])
    return sorted(agg.items(), key=lambda x: -x[1])


def _build_subgraph(session, seed_ids: set[str], hops: int = 1,
                    max_neighbors: int = 12) -> nx.Graph:
    """
    Pull seed verses + their k-hop neighborhood from Neo4j.
    Edges weighted by edge type and (where present) score.
    """
    if not seed_ids:
        return nx.Graph()

    # Use Cypher to expand from seeds via RELATED_TO + typed edges, capped
    rows = session.run("""
        MATCH (v:Verse) WHERE v.verseId IN $seeds
        OPTIONAL MATCH (v)-[r:RELATED_TO]-(o:Verse)
        WITH v, r, o LIMIT 5000
        RETURN v.verseId AS src, type(r) AS et, o.verseId AS dst,
               coalesce(r.score, 1.0) AS w
    """, seeds=list(seed_ids)).data()

    typed_rows = session.run("""
        MATCH (v:Verse) WHERE v.verseId IN $seeds
        OPTIONAL MATCH (v)-[r:SUPPORTS|ELABORATES|QUALIFIES|CONTRASTS|REPEATS]-(o:Verse)
        WITH v, r, o LIMIT 5000
        RETURN v.verseId AS src, type(r) AS et, o.verseId AS dst,
               1.5 AS w
    """, seeds=list(seed_ids)).data()

    g = nx.Graph()
    for r in rows + typed_rows:
        if r["src"] and r["dst"]:
            g.add_edge(r["src"], r["dst"], weight=float(r["w"] or 1.0))
        elif r["src"]:
            g.add_node(r["src"])

    # Make sure every seed is in the graph even if it has no edges
    for s in seed_ids:
        if s not in g:
            g.add_node(s)

    return g


def hipporag_search(session, query: str,
                    top_k_seed_verses: int = 30,
                    top_k_past_queries: int = 5,
                    alpha_vector: float = 0.7,
                    beta_past: float = 0.3,
                    final_top_k: int = 20,
                    pagerank_alpha: float = 0.5,
                    return_breakdown: bool = False) -> dict:
    """
    Run HippoRAG-style retrieval. Returns top-K verses with PPR scores.

    Args:
        session         : Neo4j session
        query           : user question
        top_k_seed_verses : how many verses to seed from vector search
        top_k_past_queries: how many similar past queries to probe
        alpha_vector    : weight on direct semantic seed (sums to ~1 with beta)
        beta_past       : weight on past-query-derived seed
        final_top_k     : cap output
        pagerank_alpha  : random-restart probability (1 - damping). Lower =
                          more weight on personalization.
        return_breakdown: also return the seed components for debugging
    """
    t0 = time.time()
    vec_seeds = _vector_seed(session, query, top_k=top_k_seed_verses)
    past_seeds = _past_query_seed(session, query, top_k=top_k_past_queries)

    # Build personalization vector
    personalization: dict[str, float] = defaultdict(float)
    for vid, score in vec_seeds:
        personalization[vid] += alpha_vector * score
    for vid, score in past_seeds:
        personalization[vid] += beta_past * score

    if not personalization:
        return {"query": query, "results": [], "warning": "no seeds"}

    # Build subgraph from union of seeds
    seed_ids = set(personalization.keys())
    g = _build_subgraph(session, seed_ids)

    # Make sure ALL seed nodes are in the graph (PPR requires it)
    for vid in seed_ids:
        if vid not in g:
            g.add_node(vid)

    # Run PPR
    pers_filtered = {n: w for n, w in personalization.items() if n in g}
    if not pers_filtered:
        return {"query": query, "results": [], "warning": "seeds not in graph"}

    try:
        pr = nx.pagerank(g, alpha=1 - pagerank_alpha,
                         personalization=pers_filtered, weight="weight",
                         max_iter=200, tol=1e-6)
    except Exception as e:
        return {"query": query, "results": [], "error": f"PPR failed: {e}"}

    ranked = sorted(pr.items(), key=lambda x: -x[1])[:final_top_k]
    elapsed = round(time.time() - t0, 2)

    out = {
        "query": query,
        "elapsed_sec": elapsed,
        "n_subgraph_nodes": g.number_of_nodes(),
        "n_subgraph_edges": g.number_of_edges(),
        "results": [{"verse_id": v, "ppr_score": round(s, 6)} for v, s in ranked],
    }
    if return_breakdown:
        out["vector_seed"] = [{"verse_id": v, "score": round(s, 4)} for v, s in vec_seeds[:10]]
        out["past_query_seed"] = [{"verse_id": v, "score": round(s, 4)} for v, s in past_seeds[:10]]
    return out


# ── CLI smoke test ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default="How does the Quran teach patience?")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--breakdown", action="store_true")
    args = ap.parse_args()

    print(f"Connecting to Neo4j ({URI}, db={DB})...")
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    driver.verify_connectivity()
    print(f"  OK, using verse index: {VERSE_INDEX}")

    print(f"\nQuery: {args.query!r}")

    with driver.session(database=DB) as s:
        r = hipporag_search(s, args.query, final_top_k=args.top_k,
                            return_breakdown=args.breakdown)
    print(f"\n  elapsed: {r.get('elapsed_sec')}s")
    print(f"  subgraph: {r.get('n_subgraph_nodes')} nodes, {r.get('n_subgraph_edges')} edges")
    print(f"  top-{len(r.get('results', []))}:")
    for i, hit in enumerate(r.get("results", []), 1):
        print(f"    {i:2}. [{hit['verse_id']}] ppr={hit['ppr_score']}")

    if args.breakdown:
        print("\n  vector seeds (top-10):")
        for h in r.get("vector_seed", []):
            print(f"    [{h['verse_id']}] sim={h['score']}")
        print("  past-query seeds (top-10):")
        for h in r.get("past_query_seed", []):
            print(f"    [{h['verse_id']}] from-past={h['score']}")

    driver.close()


if __name__ == "__main__":
    main()
