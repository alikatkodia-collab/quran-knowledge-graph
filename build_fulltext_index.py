"""
build_fulltext_index.py — create Neo4j full-text (BM25) indexes over
Verse.text (English) and Verse.arabicPlain (Arabic). One-shot.

Used by tool_hybrid_search in chat.py to fuse BM25 + dense BGE-M3 hits
via reciprocal rank fusion (RRF).

Run: python build_fulltext_index.py
"""

import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

load_dotenv()
URI = os.getenv("NEO4J_URI"); USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD"); DB = os.getenv("NEO4J_DATABASE", "quran")


def main():
    print(f"Connecting to Neo4j ({URI}, db={DB})...")
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    driver.verify_connectivity()
    print("  OK")

    print("\nCreating full-text indexes...")
    with driver.session(database=DB) as s:
        # English text index
        s.run("""
            CREATE FULLTEXT INDEX verse_text_fulltext IF NOT EXISTS
            FOR (v:Verse) ON EACH [v.text]
            OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}}
        """)
        print("  verse_text_fulltext (English) OK")

        # Arabic plain (diacritics-stripped) index
        s.run("""
            CREATE FULLTEXT INDEX verse_arabic_fulltext IF NOT EXISTS
            FOR (v:Verse) ON EACH [v.arabicPlain]
            OPTIONS {indexConfig: {`fulltext.analyzer`: 'arabic'}}
        """)
        print("  verse_arabic_fulltext (Arabic) OK")

        # Wait for indexes to populate (Neo4j async-builds)
        s.run("CALL db.awaitIndexes(60)")
        print("  indexes populated")

    # Verification
    print("\nVerifying with sample queries...")
    with driver.session(database=DB) as s:
        rows = s.run("""
            CALL db.index.fulltext.queryNodes('verse_text_fulltext', 'patience trial')
            YIELD node, score
            RETURN node.verseId AS v, score, node.text AS text
            LIMIT 5
        """).data()
    print("\n  English BM25 top-5 for 'patience trial':")
    for r in rows:
        print(f"    [{r['v']}] score={r['score']:.3f}  {r['text'][:80]}")

    with driver.session(database=DB) as s:
        ar_rows = s.run("""
            CALL db.index.fulltext.queryNodes('verse_arabic_fulltext', 'صبر')
            YIELD node, score
            RETURN node.verseId AS v, score
            LIMIT 5
        """).data()
    print("\n  Arabic BM25 top-5 for 'صبر' (patience):")
    for r in ar_rows:
        print(f"    [{r['v']}] score={r['score']:.3f}")

    driver.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
