"""
embed_verses.py — One-time job: embed all Quran verses and store in Neo4j vector index.

Model: all-MiniLM-L6-v2  (384-dim, ~80MB download, runs on CPU in ~1 min)
Index: verse_embedding (cosine similarity)

Run once:  python embed_verses.py
"""

import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
USER     = os.getenv("NEO4J_USER",     "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "")
DB       = os.getenv("NEO4J_DATABASE", "quran")
MODEL    = "all-MiniLM-L6-v2"
BATCH    = 256   # embed this many verses at once

def main():
    print(f"Connecting to Neo4j ({URI}, db={DB})...")
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    driver.verify_connectivity()
    print("  Connected OK")

    # ── load verse texts ──────────────────────────────────────────────────────
    print("Loading verses...")
    with driver.session(database=DB) as s:
        rows = s.run(
            "MATCH (v:Verse) WHERE v.verseId IS NOT NULL "
            "RETURN v.verseId AS id, v.text AS text ORDER BY v.verseId"
        ).data()
    print(f"  {len(rows):,} verses loaded")

    ids   = [r["id"]   for r in rows]
    texts = [r["text"] for r in rows]

    # ── embed ─────────────────────────────────────────────────────────────────
    print(f"\nLoading model '{MODEL}'...")
    model = SentenceTransformer(MODEL)
    dim   = model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {dim}")

    print(f"Embedding {len(texts):,} verses in batches of {BATCH}...")
    embeddings = model.encode(
        texts,
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

    # ── write embeddings to Neo4j ─────────────────────────────────────────────
    print(f"\nWriting embeddings to Neo4j...")
    WRITE_BATCH = 100
    with driver.session(database=DB) as s:
        for i in range(0, len(ids), WRITE_BATCH):
            batch_ids  = ids[i:i + WRITE_BATCH]
            batch_embs = embeddings[i:i + WRITE_BATCH]
            rows_param = [
                {"id": vid, "emb": emb.tolist()}
                for vid, emb in zip(batch_ids, batch_embs)
            ]
            s.run("""
                UNWIND $rows AS row
                MATCH (v:Verse {verseId: row.id})
                CALL db.create.setNodeVectorProperty(v, 'embedding', row.emb)
            """, rows=rows_param)
            done = min(i + WRITE_BATCH, len(ids))
            print(f"  {done}/{len(ids)}", end="\r")
    print(f"  {len(ids)}/{len(ids)} OK          ")

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
