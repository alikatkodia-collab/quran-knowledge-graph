"""
backfill_embedding_provenance.py

One-shot: stamp existing Verse.embedding values with provenance metadata
(model name, dim, source hash, timestamp) without re-running the model.

Use this once after upgrading from the legacy embed_verses.py to the
versioned embed_verses.py. After this runs, future embed_verses.py runs
will correctly skip verses whose hash is already current.

Assumes existing embeddings were produced by `all-MiniLM-L6-v2` at 384-dim
(matching what app_free.py + reasoning_memory.py + answer_cache.py already use).

Run:  python backfill_embedding_provenance.py
"""

import hashlib
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
USER     = os.getenv("NEO4J_USER",     "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "")
DB       = os.getenv("NEO4J_DATABASE", "quran")

ASSUMED_MODEL = "all-MiniLM-L6-v2"
ASSUMED_DIM   = 384
WRITE_BATCH   = 200


def compute_source_hash(model_name: str, dim: int, text: str) -> str:
    payload = f"{model_name}|{dim}|{text}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def main():
    print(f"Connecting to Neo4j ({URI}, db={DB})...")
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    driver.verify_connectivity()
    print("  Connected OK")

    print("\nFinding Verse nodes with embedding but no provenance...")
    with driver.session(database=DB) as s:
        rows = s.run("""
            MATCH (v:Verse)
            WHERE v.embedding IS NOT NULL
              AND (v.embedding_model IS NULL
                   OR v.embedding_source_hash IS NULL)
            RETURN v.verseId AS id, v.text AS text
            ORDER BY v.verseId
        """).data()
    print(f"  {len(rows):,} verses need stamping")
    if not rows:
        print("  All verses already have provenance metadata. Nothing to do.")
        driver.close()
        return

    ts = datetime.now(timezone.utc).isoformat()
    rows_param = [
        {
            "id": r["id"],
            "src_hash": compute_source_hash(ASSUMED_MODEL, ASSUMED_DIM, r["text"] or ""),
        }
        for r in rows
    ]

    print(f"\nWriting provenance ({ASSUMED_MODEL}, {ASSUMED_DIM}-dim, {ts})...")
    with driver.session(database=DB) as s:
        for i in range(0, len(rows_param), WRITE_BATCH):
            batch = rows_param[i:i + WRITE_BATCH]
            s.run("""
                UNWIND $rows AS row
                MATCH (v:Verse {verseId: row.id})
                SET v.embedding_model = $model,
                    v.embedding_dim = $dim,
                    v.embedding_source_hash = row.src_hash,
                    v.embedded_at = datetime($ts)
            """, rows=batch, model=ASSUMED_MODEL, dim=ASSUMED_DIM, ts=ts)
            done = min(i + WRITE_BATCH, len(rows_param))
            print(f"  {done}/{len(rows_param)}", end="\r")
    print(f"  {len(rows_param)}/{len(rows_param)} OK          ")

    # Verification
    print("\nVerifying...")
    with driver.session(database=DB) as s:
        n = s.run("""
            MATCH (v:Verse)
            WHERE v.embedding IS NOT NULL
              AND v.embedding_model IS NOT NULL
              AND v.embedding_source_hash IS NOT NULL
            RETURN count(v) AS n
        """).single()["n"]
    print(f"  {n:,} verses now have full provenance.")

    driver.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
