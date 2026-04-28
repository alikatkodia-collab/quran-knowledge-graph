"""
build_concepts.py — Eifrem-pattern entity resolution over Keyword nodes.

Today's graph has 2,636 Keyword nodes that are surface tokens. "patient",
"patience", "patiently" are separate keywords for what is clearly one
concept. This script collapses surface-form variants into canonical
:Concept nodes and links them.

Schema added (additive — Keyword nodes stay):
  (:Concept {name, n_keywords, n_verses, total_weight, source})
  (:Keyword)-[:NORMALIZES_TO]->(:Concept)

Algorithm:
  1. Group Keywords by Porter stem (English lemmatisation surrogate)
  2. For each group of size ≥ 2, mint a Concept named after the highest-
     weighted member ("patient" wins over "patience" if it has more weight)
  3. Singletons also get Concept nodes (so every Keyword has a Concept) —
     this lets queries always go through the Concept layer

Run: python build_concepts.py
"""

import os
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dotenv import load_dotenv
from neo4j import GraphDatabase
from nltk.stem import PorterStemmer

load_dotenv()
URI = os.getenv("NEO4J_URI"); USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD"); DB = os.getenv("NEO4J_DATABASE", "quran")


def main():
    print(f"Connecting to Neo4j ({URI}, db={DB})...")
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    driver.verify_connectivity()
    print("  OK")

    print("\nLoading Keywords with mention stats...")
    with driver.session(database=DB) as s:
        rows = s.run("""
            MATCH (k:Keyword)<-[m:MENTIONS]-(v:Verse)
            RETURN k.keyword AS kw,
                   count(DISTINCT v) AS n_verses,
                   sum(coalesce(m.score, 0.0)) AS total_weight
        """).data()
    print(f"  {len(rows):,} keywords with mentions")

    # Stem and group
    stemmer = PorterStemmer()
    by_stem: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        stem = stemmer.stem(r["kw"].lower())
        by_stem[stem].append(r)

    multi = [s for s, members in by_stem.items() if len(members) > 1]
    print(f"  {len(multi):,} stems with 2+ surface variants (real ER targets)")
    print(f"  {len(by_stem) - len(multi):,} singleton stems")

    # Sample a few multis to sanity-check
    print("\n  Sample multi-keyword groups:")
    sample_stems = sorted(multi, key=lambda s: -sum(m["n_verses"] for m in by_stem[s]))[:8]
    for stem in sample_stems:
        members = by_stem[stem]
        labels = sorted([m["kw"] for m in members])
        total_v = sum(m["n_verses"] for m in members)
        print(f"    [{stem}] -> {labels[:5]} ({total_v} total verses)")

    # Mint Concepts
    print("\nWriting :Concept nodes + :NORMALIZES_TO edges...")
    BATCH = 200
    concepts = []
    for stem, members in by_stem.items():
        # name the concept after the highest-weighted member
        members.sort(key=lambda m: -m["total_weight"])
        canonical_name = members[0]["kw"]
        concepts.append({
            "stem": stem,
            "name": canonical_name,
            "n_keywords": len(members),
            "n_verses": sum(m["n_verses"] for m in members),
            "total_weight": float(sum(m["total_weight"] for m in members)),
            "keywords": [m["kw"] for m in members],
        })

    # First create constraint
    with driver.session(database=DB) as s:
        s.run("CREATE CONSTRAINT concept_stem_unique IF NOT EXISTS "
              "FOR (c:Concept) REQUIRE c.stem IS UNIQUE")

    with driver.session(database=DB) as s:
        for i in range(0, len(concepts), BATCH):
            chunk = concepts[i:i + BATCH]
            s.run("""
                UNWIND $rows AS row
                MERGE (c:Concept {stem: row.stem})
                SET c.name = row.name,
                    c.n_keywords = row.n_keywords,
                    c.n_verses = row.n_verses,
                    c.total_weight = row.total_weight,
                    c.source = 'porter-stem'
                WITH c, row
                UNWIND row.keywords AS kw
                MATCH (k:Keyword {keyword: kw})
                MERGE (k)-[:NORMALIZES_TO]->(c)
            """, rows=chunk)
            done = min(i + BATCH, len(concepts))
            print(f"  {done}/{len(concepts)}", end="\r", flush=True)
    print(f"  {len(concepts)}/{len(concepts)} OK              ")

    # Verification
    print("\nVerifying...")
    with driver.session(database=DB) as s:
        n_c = s.run("MATCH (c:Concept) RETURN count(c) AS n").single()["n"]
        n_e = s.run("MATCH ()-[r:NORMALIZES_TO]->() RETURN count(r) AS n").single()["n"]
        # Sample: patience concept
        sample = s.run("""
            MATCH (c:Concept)
            WHERE c.stem = 'patienc' OR c.name CONTAINS 'patien'
            OPTIONAL MATCH (k:Keyword)-[:NORMALIZES_TO]->(c)
            RETURN c.name AS name, c.n_keywords AS n_kw, c.n_verses AS n_v,
                   collect(k.keyword) AS keywords
        """).data()
    print(f"  Concept nodes: {n_c:,}")
    print(f"  NORMALIZES_TO edges: {n_e:,}")
    if sample:
        for r in sample:
            print(f"  patience-cluster: name={r['name']}, n_kw={r['n_kw']}, "
                  f"n_v={r['n_v']}, surface_forms={r['keywords']}")

    driver.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
