"""
embed_verses.py — embed all Quran verses and store in Neo4j vector index.

Model: all-MiniLM-L6-v2  (384-dim, ~80MB download, runs on CPU in ~1 min)
Index: verse_embedding (cosine similarity)

This script is idempotent. Each verse is stamped with provenance metadata:
  - embedding_model            (e.g. "all-MiniLM-L6-v2")
  - embedding_dim              (e.g. 384)
  - embedding_source_hash      (sha1 of "model|dim|verse_text") — recompute trigger
  - embedded_at                (ISO 8601 timestamp)

On re-run, verses whose `embedding_source_hash` is already current are skipped.
This means safe re-runs after partial failures and selective re-embedding when
the model or text changes.

CLI:
  python embed_verses.py                # embed all stale/missing verses
  python embed_verses.py --force        # re-embed everything regardless
  python embed_verses.py --model X      # use a different model (forces version bump)
"""

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
USER     = os.getenv("NEO4J_USER",     "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "")
DB       = os.getenv("NEO4J_DATABASE", "quran")
DEFAULT_MODEL = "all-MiniLM-L6-v2"
BATCH    = 256   # embed this many verses at once


def compute_source_hash(model_name: str, dim: int, text: str) -> str:
    """Stable hash of (model, dim, text) — changes invalidate the embedding."""
    payload = f"{model_name}|{dim}|{text}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"sentence-transformers model name (default: {DEFAULT_MODEL})")
    ap.add_argument("--force", action="store_true",
                    help="Re-embed every verse regardless of existing source hash")
    args = ap.parse_args()

    model_name = args.model

    print(f"Connecting to Neo4j ({URI}, db={DB})...")
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    driver.verify_connectivity()
    print("  Connected OK")

    # ── load model first so we know dim ──────────────────────────────────────
    print(f"\nLoading model '{model_name}'...")
    model = SentenceTransformer(model_name)
    dim   = model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {dim}")

    # ── load verse texts + existing provenance ───────────────────────────────
    print("\nLoading verses + existing provenance...")
    with driver.session(database=DB) as s:
        rows = s.run("""
            MATCH (v:Verse) WHERE v.verseId IS NOT NULL
            RETURN v.verseId AS id, v.text AS text,
                   v.embedding_model AS model, v.embedding_dim AS dim,
                   v.embedding_source_hash AS src_hash
            ORDER BY v.verseId
        """).data()
    print(f"  {len(rows):,} verses loaded")

    # ── decide which need embedding ──────────────────────────────────────────
    todo_ids, todo_texts = [], []
    skipped = 0
    for r in rows:
        target_hash = compute_source_hash(model_name, dim, r["text"])
        if not args.force and r.get("src_hash") == target_hash:
            skipped += 1
            continue
        todo_ids.append(r["id"])
        todo_texts.append(r["text"])

    print(f"  to embed: {len(todo_ids):,} · already current (skipped): {skipped:,}")
    if not todo_ids:
        print("  Everything is up to date. Nothing to do.")
        driver.close()
        return

    # ── embed ─────────────────────────────────────────────────────────────────
    print(f"\nEmbedding {len(todo_texts):,} verses in batches of {BATCH}...")
    embeddings = model.encode(
        todo_texts,
        batch_size=BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit vectors → cosine = dot product
        convert_to_numpy=True,
    )
    print(f"  Done. Shape: {embeddings.shape}")

    # ── create vector index ───────────────────────────────────────────────────
    print("\nCreating vector index...")
    with driver.session(database=DB) as s:
        try:
            s.run(f"""
                CREATE VECTOR INDEX verse_embedding IF NOT EXISTS
                FOR (v:Verse) ON (v.embedding)
                OPTIONS {{indexConfig: {{
                  `vector.dimensions`: {dim},
                  `vector.similarity_function`: 'cosine'
                }}}}
            """)
            print("  Index created (or already exists)")
        except Exception as e:
            print(f"  Index creation warning: {e}")

    # ── write embeddings + provenance to Neo4j ───────────────────────────────
    print(f"\nWriting embeddings + provenance to Neo4j...")
    ts = _now_iso()
    WRITE_BATCH = 100
    with driver.session(database=DB) as s:
        for i in range(0, len(todo_ids), WRITE_BATCH):
            batch_ids  = todo_ids[i:i + WRITE_BATCH]
            batch_texts = todo_texts[i:i + WRITE_BATCH]
            batch_embs = embeddings[i:i + WRITE_BATCH]
            rows_param = [
                {
                    "id": vid,
                    "emb": emb.tolist(),
                    "src_hash": compute_source_hash(model_name, dim, txt),
                }
                for vid, txt, emb in zip(batch_ids, batch_texts, batch_embs)
            ]
            s.run("""
                UNWIND $rows AS row
                MATCH (v:Verse {verseId: row.id})
                CALL db.create.setNodeVectorProperty(v, 'embedding', row.emb)
                SET v.embedding_model = $model_name,
                    v.embedding_dim = $dim,
                    v.embedding_source_hash = row.src_hash,
                    v.embedded_at = datetime($ts)
            """, rows=rows_param, model_name=model_name, dim=dim, ts=ts)
            done = min(i + WRITE_BATCH, len(todo_ids))
            print(f"  {done}/{len(todo_ids)}", end="\r")
    print(f"  {len(todo_ids)}/{len(todo_ids)} OK          ")

    # ── verify ────────────────────────────────────────────────────────────────
    print("\nVerifying — sample query: 'God forgives the repentant'")
    query_vec = model.encode(
        ["God forgives the repentant"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].tolist()

    with driver.session(database=DB) as s:
        results = s.run("""
            CALL db.index.vector.queryNodes('verse_embedding', 5, $vec)
            YIELD node, score
            RETURN node.verseId AS id, node.surahName AS surahName,
                   score, node.text AS text
        """, vec=query_vec).data()

    print("Top 5 semantically similar verses:")
    for r in results:
        print(f"  [{r['id']}] ({r['surahName']}) score={r['score']:.4f}")
        print(f"    {r['text'][:100]}...")

    driver.close()
    print("\nDone. Semantic search is ready.")

if __name__ == "__main__":
    main()
