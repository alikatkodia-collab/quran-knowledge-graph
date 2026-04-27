"""
import_mutashabihat.py — load Waqar144's open mutashabihat dataset into the
Quran Knowledge Graph as :SIMILAR_PHRASE edges between Verse nodes.

Source: https://github.com/Waqar144/Quran_Mutashabihat_Data (CC0)
File:   data/mutashabihat_raw.json (already downloaded)

Format of source: dict[chapter -> list[entry]] where entry = {
    "src": {"ayah": <int|list[int]>},
    "muts": [{"ayah": <int>}, ...],
    "ctx": <int, optional>
}
where `ayah` is the ABSOLUTE 1-indexed verse number across the whole Quran
(not surah:verse). 1 = 1:1, 7 = 1:7, 8 = 2:1, etc.

Khalifa's translation excludes 9:128 and 9:129. Edges that touch those
verses are silently skipped.

Output:
  Neo4j edges (:Verse)-[:SIMILAR_PHRASE {
      dataSource: 'waqar144-mutashabiha',
      ctx: <int|null>
  }]->(:Verse)
  (undirected — emitted in both directions)

Usage:  python import_mutashabihat.py
"""

import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
URI = os.getenv("NEO4J_URI"); USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD"); DB = os.getenv("NEO4J_DATABASE", "quran")

SOURCE = "waqar144-mutashabiha"
KHALIFA_EXCLUDED = {"9:128", "9:129"}


def build_absolute_to_verseid(driver) -> dict[int, str]:
    """Walk verses in canonical order (surah, verseNum) and assign 1-indexed
    absolute positions. The Waqar144 dataset uses the same ordering."""
    with driver.session(database=DB) as s:
        rows = s.run("""
            MATCH (v:Verse)
            WHERE v.verseId IS NOT NULL
            RETURN v.verseId AS id, v.surah AS s, v.verseNum AS vn
            ORDER BY v.surah, v.verseNum
        """).data()
    # The Waqar144 dataset's absolute numbering INCLUDES 9:128 and 9:129
    # (because it's based on the Hafs canonical text). Khalifa's database
    # excludes them. To make the absolute index match Waqar144's, insert
    # placeholder slots for the missing two verses at their canonical position.
    out: dict[int, str] = {}
    abs_idx = 0
    seen_9_127 = False
    for r in rows:
        abs_idx += 1
        # When we just emitted 9:127, the next two slots in the canonical
        # numbering are 9:128 and 9:129 — which we don't have. Skip them.
        if seen_9_127 and r["s"] == 10 and r["vn"] == 1:
            # Burn the two missing absolute positions
            abs_idx += 2
            seen_9_127 = False
        out[abs_idx] = r["id"]
        if r["s"] == 9 and r["vn"] == 127:
            seen_9_127 = True
    return out


def main():
    print(f"Connecting to Neo4j ({URI}, db={DB})...")
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    driver.verify_connectivity()
    print("  OK")

    print("\nBuilding absolute→verseId map...")
    abs_map = build_absolute_to_verseid(driver)
    print(f"  {len(abs_map):,} absolute slots mapped")

    print("\nLoading mutashabihat_raw.json...")
    raw = json.loads(Path("data/mutashabihat_raw.json").read_text(encoding="utf-8"))
    def _as_list(x):
        return x if isinstance(x, list) else [x]

    rows = []
    for chap, entries in raw.items():
        for e in entries:
            srcs = _as_list(e["src"]["ayah"])
            ctx = e.get("ctx")
            for s_abs in srcs:
                for m in e.get("muts", []):
                    for d_abs in _as_list(m["ayah"]):
                        rows.append({"src_abs": int(s_abs), "dst_abs": int(d_abs), "ctx": ctx})

    print(f"  {len(rows):,} edge candidates")

    # Resolve to verseIds
    edges = []
    skipped_unknown = 0
    skipped_excluded = 0
    for row in rows:
        s_id = abs_map.get(row["src_abs"])
        d_id = abs_map.get(row["dst_abs"])
        if not s_id or not d_id:
            skipped_unknown += 1
            continue
        if s_id in KHALIFA_EXCLUDED or d_id in KHALIFA_EXCLUDED:
            skipped_excluded += 1
            continue
        if s_id == d_id:
            continue  # self-link
        edges.append({"src": s_id, "dst": d_id, "ctx": row["ctx"]})

    print(f"  resolved: {len(edges):,}")
    print(f"  skipped (unknown abs): {skipped_unknown}")
    print(f"  skipped (Khalifa-excluded): {skipped_excluded}")

    # Write to Neo4j (undirected — emit both directions for query convenience)
    print("\nWriting :SIMILAR_PHRASE edges...")
    BATCH = 500
    with driver.session(database=DB) as s:
        for i in range(0, len(edges), BATCH):
            chunk = edges[i:i+BATCH]
            s.run("""
                UNWIND $rows AS row
                MATCH (a:Verse {verseId: row.src})
                MATCH (b:Verse {verseId: row.dst})
                MERGE (a)-[r1:SIMILAR_PHRASE]->(b)
                SET r1.dataSource = $src, r1.ctx = row.ctx
                MERGE (b)-[r2:SIMILAR_PHRASE]->(a)
                SET r2.dataSource = $src, r2.ctx = row.ctx
            """, rows=chunk, src=SOURCE)
            done = min(i + BATCH, len(edges))
            print(f"  {done}/{len(edges)}", end="\r", flush=True)
    print(f"  {len(edges)}/{len(edges)} OK             ")

    # Verify
    with driver.session(database=DB) as s:
        n = s.run("""
            MATCH (a:Verse)-[r:SIMILAR_PHRASE {dataSource: $src}]->(b:Verse)
            RETURN count(r) AS n
        """, src=SOURCE).single()["n"]
        sample = s.run("""
            MATCH (a:Verse {verseId: '1:1'})-[r:SIMILAR_PHRASE]->(b:Verse)
            RETURN b.verseId AS dst, b.text AS text LIMIT 3
        """).data()
    print(f"\n  total SIMILAR_PHRASE edges: {n:,}")
    print(f"  sample (1:1 → ...):")
    for r in sample:
        print(f"    [{r['dst']}] {r['text'][:80]}")

    driver.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
