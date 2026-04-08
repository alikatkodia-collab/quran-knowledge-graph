"""
Step 3 — Import the 4 CSV files into Neo4j using the Python driver.

Uses batch UNWIND inserts — no offline requirement, works with Neo4j 4.x and 5.x.
Run with Neo4j Desktop DBMS running (green/active state).

Creates:
  - (:Verse {verseId, surah, verseNum, surahName, text})
  - (:Keyword {keyword})
  - (:Verse)-[:MENTIONS {score}]->(:Keyword)
  - (:Verse)-[:RELATED_TO {score}]-(:Verse)

Indexes created for fast lookup:
  - Verse.verseId (unique)
  - Keyword.keyword (unique)
  - Verse.surah
  - Verse.surahName
"""

import csv
import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Increase CSV field size limit for long verse texts
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

load_dotenv()

URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER     = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "")

BATCH_SIZE = 500

# ── Helpers ───────────────────────────────────────────────────────────────────

def read_csv(path: str) -> list[dict]:
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def run_batched(session, query: str, rows: list[dict], label: str):
    total = len(rows)
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        session.run(query, rows=batch)
        print(f"  {label}: {min(i + BATCH_SIZE, total):,}/{total:,}", end='\r')
    print(f"  {label}: {total:,}/{total:,} OK")


# ── Schema setup ──────────────────────────────────────────────────────────────

SCHEMA_QUERIES = [
    "CREATE CONSTRAINT verse_id IF NOT EXISTS FOR (v:Verse) REQUIRE v.verseId IS UNIQUE",
    "CREATE CONSTRAINT keyword_id IF NOT EXISTS FOR (k:Keyword) REQUIRE k.keyword IS UNIQUE",
    "CREATE INDEX verse_surah IF NOT EXISTS FOR (v:Verse) ON (v.surah)",
    "CREATE INDEX verse_surah_name IF NOT EXISTS FOR (v:Verse) ON (v.surahName)",
]

def setup_schema(session):
    print("Setting up schema / constraints...")
    for q in SCHEMA_QUERIES:
        try:
            session.run(q)
        except Exception as e:
            # Older Neo4j may use different syntax — try fallback
            if "IF NOT EXISTS" in q:
                q2 = q.replace(" IF NOT EXISTS", "")
                try:
                    session.run(q2)
                except Exception:
                    pass  # Already exists
    print("  Schema ready OK")


# ── Import functions ──────────────────────────────────────────────────────────

def import_verse_nodes(session, path: str):
    rows = read_csv(path)
    print(f"\nImporting verse nodes ({len(rows):,})...")
    query = """
    UNWIND $rows AS row
    MERGE (v:Verse {verseId: row.verseId})
    SET v.surah     = toInteger(row.surah),
        v.verseNum  = toInteger(row.verseNum),
        v.surahName = row.surahName,
        v.text      = row.text
    """
    run_batched(session, query, rows, "Verses")


def import_keyword_nodes(session, path: str):
    rows = read_csv(path)
    print(f"\nImporting keyword nodes ({len(rows):,})...")
    query = """
    UNWIND $rows AS row
    MERGE (k:Keyword {keyword: row.keyword})
    """
    run_batched(session, query, rows, "Keywords")


def import_mentions(session, path: str):
    rows = read_csv(path)
    print(f"\nImporting MENTIONS edges ({len(rows):,})...")
    # Convert score to float
    for r in rows:
        r['score'] = float(r['score'])
    query = """
    UNWIND $rows AS row
    MATCH (v:Verse {verseId: row.verseId})
    MATCH (k:Keyword {keyword: row.keyword})
    MERGE (v)-[r:MENTIONS]->(k)
    SET r.score = row.score
    """
    run_batched(session, query, rows, "MENTIONS")


def import_related(session, path: str):
    rows = read_csv(path)
    print(f"\nImporting RELATED_TO edges ({len(rows):,})...")
    for r in rows:
        r['score'] = float(r['score'])
    query = """
    UNWIND $rows AS row
    MATCH (v1:Verse {verseId: row.verseId1})
    MATCH (v2:Verse {verseId: row.verseId2})
    MERGE (v1)-[r:RELATED_TO]-(v2)
    SET r.score = row.score
    """
    run_batched(session, query, rows, "RELATED_TO")


# ── Verification ──────────────────────────────────────────────────────────────

def verify(session):
    print("\n" + "="*50)
    print("IMPORT VERIFICATION")
    print("="*50)
    counts = {
        "Verse nodes":    "MATCH (v:Verse) RETURN count(v) AS n",
        "Keyword nodes":  "MATCH (k:Keyword) RETURN count(k) AS n",
        "MENTIONS edges": "MATCH ()-[r:MENTIONS]->() RETURN count(r) AS n",
        "RELATED_TO edges": "MATCH ()-[r:RELATED_TO]-() RETURN count(r)/2 AS n",
    }
    for label, query in counts.items():
        result = session.run(query).single()
        print(f"  {label:20s}: {result['n']:,}")

    # Sample lookup
    print("\nSample verse lookup [2:255] (Throne Verse):")
    result = session.run(
        "MATCH (v:Verse {verseId: '2:255'}) RETURN v.text AS text"
    ).single()
    if result:
        print(f"  {result['text'][:100]}...")
    else:
        print("  NOT FOUND — check parsing")

    print("\nSample graph traversal from [2:255]:")
    result = session.run("""
        MATCH (v:Verse {verseId: '2:255'})-[:RELATED_TO]-(related:Verse)
        RETURN related.verseId AS id, related.text AS text
        LIMIT 5
    """)
    for row in result:
        print(f"  [{row['id']}] {row['text'][:70]}...")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print(f"Connecting to Neo4j at {URI}...")
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    try:
        driver.verify_connectivity()
        print("  Connected OK")
    except Exception as e:
        print(f"  Connection failed: {e}")
        print("\n  Make sure Neo4j Desktop DBMS is running (green status).")
        raise

    with driver.session() as session:
        setup_schema(session)
        import_verse_nodes(session,   "data/verse_nodes.csv")
        import_keyword_nodes(session, "data/keyword_nodes.csv")
        import_mentions(session,      "data/verse_keyword_rels.csv")
        import_related(session,       "data/verse_related_rels.csv")
        verify(session)

    driver.close()
    print("\nOK Import complete. Neo4j is ready.")
