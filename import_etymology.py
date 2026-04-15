"""
Import etymology layer data into Neo4j.

Imports WordToken, Lemma, MorphPattern, SemanticDomain nodes and all
relationships from CSV files produced by:
  - build_word_tokens.py
  - build_semantic_domains.py
  - build_wujuh.py

Uses the same UNWIND batch pattern as import_neo4j.py.

Usage:
    py import_etymology.py                     # full import
    py import_etymology.py --skip-tokens       # skip WordToken (fast re-import of metadata)
    py import_etymology.py --verify-only       # just check counts
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DB       = os.getenv("NEO4J_DATABASE", "quran")

BATCH_SIZE = 500


def run_batched(session, query: str, rows: list, label: str):
    """Execute a Cypher query in batches using UNWIND."""
    total = len(rows)
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        session.run(query, rows=batch)
        if (i // BATCH_SIZE + 1) % 20 == 0:
            done = min(i + BATCH_SIZE, total)
            print(f"    {label}: {done:,}/{total:,}")
    print(f"    {label}: {total:,}/{total:,} done")


def load_csv(filename: str) -> list[dict]:
    """Load a CSV file as a list of dicts."""
    path = DATA_DIR / filename
    if not path.exists():
        print(f"  WARNING: {filename} not found, skipping")
        return []
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def setup_schema(session):
    """Create constraints and indexes for etymology nodes."""
    print("  Creating constraints and indexes...")

    constraints = [
        "CREATE CONSTRAINT word_token_id IF NOT EXISTS FOR (w:WordToken) REQUIRE w.tokenId IS UNIQUE",
        "CREATE CONSTRAINT lemma_id IF NOT EXISTS FOR (l:Lemma) REQUIRE l.lemma IS UNIQUE",
        "CREATE CONSTRAINT morph_pattern_id IF NOT EXISTS FOR (p:MorphPattern) REQUIRE p.pattern IS UNIQUE",
        "CREATE CONSTRAINT semantic_domain_id IF NOT EXISTS FOR (d:SemanticDomain) REQUIRE d.domainId IS UNIQUE",
    ]
    indexes = [
        "CREATE INDEX word_token_root IF NOT EXISTS FOR (w:WordToken) ON (w.root)",
        "CREATE INDEX word_token_lemma IF NOT EXISTS FOR (w:WordToken) ON (w.lemma)",
        "CREATE INDEX word_token_arabic IF NOT EXISTS FOR (w:WordToken) ON (w.arabicClean)",
        "CREATE INDEX word_token_bw IF NOT EXISTS FOR (w:WordToken) ON (w.translitBW)",
        "CREATE INDEX word_token_verse IF NOT EXISTS FOR (w:WordToken) ON (w.verseId)",
        "CREATE INDEX word_token_pos IF NOT EXISTS FOR (w:WordToken) ON (w.pos)",
        "CREATE INDEX lemma_root IF NOT EXISTS FOR (l:Lemma) ON (l.root)",
    ]

    for cypher in constraints + indexes:
        session.run(cypher)
    print("    Done")


def import_word_tokens(session):
    """Import WordToken nodes from CSV."""
    rows = load_csv("word_token_nodes.csv")
    if not rows:
        return

    print(f"  Importing {len(rows):,} WordToken nodes...")
    run_batched(session, """
        UNWIND $rows AS row
        MERGE (w:WordToken {tokenId: row.tokenId})
        SET w.verseId = row.verseId,
            w.wordPos = toInteger(row.wordPos),
            w.arabicText = row.arabicText,
            w.arabicClean = row.arabicClean,
            w.translitBW = row.translitBW,
            w.root = row.root,
            w.lemma = row.lemma,
            w.pos = row.pos,
            w.morphFeatures = row.morphFeatures,
            w.wazn = row.wazn
    """, rows, "WordToken nodes")


def import_lemma_nodes(session):
    """Import Lemma nodes from CSV."""
    rows = load_csv("lemma_nodes.csv")
    if not rows:
        return

    print(f"  Importing {len(rows):,} Lemma nodes...")
    run_batched(session, """
        UNWIND $rows AS row
        MERGE (l:Lemma {lemma: row.lemma})
        SET l.lemmaBW = row.lemmaBW,
            l.root = row.root,
            l.glossEn = row.glossEn,
            l.pos = row.pos,
            l.verseCount = toInteger(row.verseCount)
    """, rows, "Lemma nodes")


def import_pattern_nodes(session):
    """Import MorphPattern nodes from CSV."""
    rows = load_csv("morph_pattern_nodes.csv")
    if not rows:
        return

    print(f"  Importing {len(rows):,} MorphPattern nodes...")
    run_batched(session, """
        UNWIND $rows AS row
        MERGE (p:MorphPattern {pattern: row.pattern})
        SET p.patternBW = row.patternBW,
            p.label = row.label,
            p.meaningTendency = row.meaningTendency
    """, rows, "MorphPattern nodes")


def import_domain_nodes(session):
    """Import SemanticDomain nodes from CSV."""
    rows = load_csv("semantic_domain_nodes.csv")
    if not rows:
        return

    print(f"  Importing {len(rows):,} SemanticDomain nodes...")
    run_batched(session, """
        UNWIND $rows AS row
        MERGE (d:SemanticDomain {domainId: row.domainId})
        SET d.nameEn = row.nameEn,
            d.nameAr = row.nameAr,
            d.description = row.description
    """, rows, "SemanticDomain nodes")


def import_word_verse_rels(session):
    """Import WordToken → Verse relationships."""
    rows = load_csv("word_verse_rels.csv")
    if not rows:
        return

    print(f"  Importing {len(rows):,} IN_VERSE relationships...")
    run_batched(session, """
        UNWIND $rows AS row
        MATCH (w:WordToken {tokenId: row.tokenId})
        MATCH (v:Verse {verseId: row.verseId})
        MERGE (w)-[:IN_VERSE]->(v)
    """, rows, "IN_VERSE rels")


def import_word_lemma_rels(session):
    """Import WordToken → Lemma relationships."""
    rows = load_csv("word_lemma_rels.csv")
    if not rows:
        return

    print(f"  Importing {len(rows):,} HAS_LEMMA relationships...")
    run_batched(session, """
        UNWIND $rows AS row
        MATCH (w:WordToken {tokenId: row.tokenId})
        MATCH (l:Lemma {lemma: row.lemma})
        MERGE (w)-[:HAS_LEMMA]->(l)
    """, rows, "HAS_LEMMA rels")


def import_lemma_root_rels(session):
    """Import Lemma → ArabicRoot relationships."""
    rows = load_csv("lemma_root_rels.csv")
    if not rows:
        return

    print(f"  Importing {len(rows):,} DERIVES_FROM relationships...")
    run_batched(session, """
        UNWIND $rows AS row
        MATCH (l:Lemma {lemma: row.lemma})
        MATCH (r:ArabicRoot {root: row.root})
        MERGE (l)-[:DERIVES_FROM]->(r)
    """, rows, "DERIVES_FROM rels")


def import_word_pattern_rels(session):
    """Import WordToken → MorphPattern relationships."""
    rows = load_csv("word_pattern_rels.csv")
    if not rows:
        return

    print(f"  Importing {len(rows):,} FOLLOWS_PATTERN relationships...")
    run_batched(session, """
        UNWIND $rows AS row
        MATCH (w:WordToken {tokenId: row.tokenId})
        MATCH (p:MorphPattern {pattern: row.pattern})
        MERGE (w)-[:FOLLOWS_PATTERN]->(p)
    """, rows, "FOLLOWS_PATTERN rels")


def import_root_domain_rels(session):
    """Import ArabicRoot → SemanticDomain relationships."""
    rows = load_csv("root_domain_rels.csv")
    if not rows:
        return

    print(f"  Importing {len(rows):,} IN_DOMAIN relationships...")
    run_batched(session, """
        UNWIND $rows AS row
        MATCH (r:ArabicRoot {root: row.root})
        MATCH (d:SemanticDomain {domainId: row.domainId})
        MERGE (r)-[:IN_DOMAIN]->(d)
    """, rows, "IN_DOMAIN rels")


def update_verse_word_counts(session):
    """Set Verse.wordCount from WordToken counts."""
    print("  Computing verse word counts...")
    session.run("""
        MATCH (w:WordToken)-[:IN_VERSE]->(v:Verse)
        WITH v, count(w) AS wc
        SET v.wordCount = wc
    """)
    result = session.run("""
        MATCH (v:Verse) WHERE v.wordCount IS NOT NULL
        RETURN count(v) AS c, avg(v.wordCount) AS avg_wc
    """).single()
    print(f"    Updated {result['c']:,} verses, avg words/verse: {result['avg_wc']:.1f}")


def verify(session):
    """Print count verification for all etymology nodes and rels."""
    print("\n  Verification:")
    queries = [
        ("WordToken nodes", "MATCH (w:WordToken) RETURN count(w) AS c"),
        ("Lemma nodes", "MATCH (l:Lemma) RETURN count(l) AS c"),
        ("MorphPattern nodes", "MATCH (p:MorphPattern) RETURN count(p) AS c"),
        ("SemanticDomain nodes", "MATCH (d:SemanticDomain) RETURN count(d) AS c"),
        ("IN_VERSE rels", "MATCH ()-[r:IN_VERSE]->() RETURN count(r) AS c"),
        ("HAS_LEMMA rels", "MATCH ()-[r:HAS_LEMMA]->() RETURN count(r) AS c"),
        ("DERIVES_FROM rels", "MATCH ()-[r:DERIVES_FROM]->() RETURN count(r) AS c"),
        ("FOLLOWS_PATTERN rels", "MATCH ()-[r:FOLLOWS_PATTERN]->() RETURN count(r) AS c"),
        ("IN_DOMAIN rels", "MATCH ()-[r:IN_DOMAIN]->() RETURN count(r) AS c"),
    ]
    for label, query in queries:
        result = session.run(query).single()
        print(f"    {label:25s} {result['c']:>8,}")

    # Sample check: word-by-word of 1:1
    print("\n  Sample — verse 1:1 word tokens:")
    results = session.run("""
        MATCH (w:WordToken {verseId: '1:1'})
        OPTIONAL MATCH (w)-[:HAS_LEMMA]->(l:Lemma)
        OPTIONAL MATCH (l)-[:DERIVES_FROM]->(r:ArabicRoot)
        RETURN w.tokenId AS tid, w.arabicText AS ar,
               w.pos AS pos, l.lemma AS lem, r.root AS root, r.gloss AS gloss
        ORDER BY w.wordPos
    """).data()
    for r in results:
        print(f"    [{r['tid']}] {r['ar']:>15} pos={r['pos']:<10} "
              f"lemma={r['lem'] or '':12} root={r['root'] or '':<6} "
              f"gloss={r['gloss'] or ''}")


def main():
    sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description='Import etymology data into Neo4j')
    parser.add_argument('--skip-tokens', action='store_true',
                        help='Skip WordToken import (faster for metadata updates)')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only run verification queries')
    args = parser.parse_args()

    from neo4j import GraphDatabase

    print("Etymology Neo4j Import")
    print("=" * 60)
    print(f"\nConnecting to {NEO4J_URI} (database: {NEO4J_DB})...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("  Connected.")

    with driver.session(database=NEO4J_DB) as session:
        if args.verify_only:
            verify(session)
        else:
            # Schema
            print("\n[1] Setting up schema...")
            setup_schema(session)

            # Nodes
            print("\n[2] Importing nodes...")
            if not args.skip_tokens:
                import_word_tokens(session)
            else:
                print("  Skipping WordToken import (--skip-tokens)")
            import_lemma_nodes(session)
            import_pattern_nodes(session)
            import_domain_nodes(session)

            # Relationships
            print("\n[3] Importing relationships...")
            if not args.skip_tokens:
                import_word_verse_rels(session)
                import_word_lemma_rels(session)
                import_word_pattern_rels(session)
            import_lemma_root_rels(session)
            import_root_domain_rels(session)

            # Computed properties
            print("\n[4] Computing derived properties...")
            if not args.skip_tokens:
                update_verse_word_counts(session)

            # Verify
            print("\n[5] Verification...")
            verify(session)

    driver.close()
    print("\nDone.")


if __name__ == '__main__':
    main()
