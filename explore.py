"""
Step 5 | Standalone Quran graph explorer (no API key needed).

Usage:
    py explore.py verse 2:255
        -> show verse text + all directly connected verses with shared keywords

    py explore.py keyword covenant
        -> all verses mentioning "covenant", grouped by surah

    py explore.py path 2:255 112:1
        -> shortest thematic path between two verses through the graph

    py explore.py cluster 36
        -> all verses in Surah 36 and their cross-surah connections
"""

import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Reuse lemmatizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_graph import tokenize_and_lemmatize

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

DIVIDER = "-" * 70


# -- Commands ------------------------------------------------------------------

def cmd_verse(session, verse_id: str):
    """Show a verse and all its directly connected verses with shared keywords."""
    # Normalize input: accept "2:255" or "2 255"
    verse_id = verse_id.replace(' ', ':')

    result = session.run(
        "MATCH (v:Verse {verseId: $id}) RETURN v", id=verse_id
    ).single()
    if not result:
        print(f"Verse [{verse_id}] not found.")
        return

    v = result['v']
    print(f"\n[{v['verseId']}] {v['surahName']} (Surah {v['surah']})")
    print(DIVIDER)
    print(v['text'])
    print()

    # Keywords this verse mentions
    kw_result = session.run("""
        MATCH (v:Verse {verseId: $id})-[r:MENTIONS]->(k:Keyword)
        RETURN k.keyword AS kw, r.score AS score
        ORDER BY r.score DESC
        LIMIT 20
    """, id=verse_id)
    keywords = [(r['kw'], r['score']) for r in kw_result]
    print(f"Keywords ({len(keywords)}): {', '.join(k for k, _ in keywords)}")
    print()

    # Connected verses
    conn_result = session.run("""
        MATCH (v:Verse {verseId: $id})-[r:RELATED_TO]-(other:Verse)
        RETURN other.verseId AS otherId, other.surahName AS surahName,
               other.text AS text, r.score AS score
        ORDER BY r.score DESC
        LIMIT 15
    """, id=verse_id)
    connections = list(conn_result)

    print(f"Connected verses ({len(connections)}):")
    print(DIVIDER)
    for row in connections:
        # Find shared keywords
        shared = session.run("""
            MATCH (v1:Verse {verseId: $v1})-[:MENTIONS]->(k:Keyword)<-[:MENTIONS]-(v2:Verse {verseId: $v2})
            RETURN k.keyword AS kw
            LIMIT 5
        """, v1=verse_id, v2=row['otherId'])
        shared_kws = [r['kw'] for r in shared]
        print(f"\n  [{row['otherId']}] {row['surahName']}")
        print(f"  Shared keywords: {', '.join(shared_kws)}")
        print(f"  {row['text'][:120]}{'...' if len(row['text']) > 120 else ''}")


def cmd_keyword(session, raw_keyword: str):
    """Show all verses mentioning a keyword, grouped by surah."""
    # Lemmatize the input keyword
    lemmas = tokenize_and_lemmatize(raw_keyword)
    if not lemmas:
        print(f"Keyword '{raw_keyword}' was filtered as a stopword.")
        return
    keyword = lemmas[0]

    # Check if keyword exists
    exists = session.run(
        "MATCH (k:Keyword {keyword: $kw}) RETURN k", kw=keyword
    ).single()
    if not exists:
        print(f"Keyword '{keyword}' (from '{raw_keyword}') not found in graph.")
        # Suggest similar
        similar = session.run("""
            MATCH (k:Keyword)
            WHERE k.keyword STARTS WITH $prefix
            RETURN k.keyword AS kw LIMIT 10
        """, prefix=keyword[:4])
        suggestions = [r['kw'] for r in similar]
        if suggestions:
            print(f"  Similar keywords: {', '.join(suggestions)}")
        return

    result = session.run("""
        MATCH (k:Keyword {keyword: $kw})<-[r:MENTIONS]-(v:Verse)
        RETURN v.surah AS surah, v.surahName AS surahName,
               v.verseId AS verseId, v.text AS text, r.score AS score
        ORDER BY v.surah, v.verseNum
    """, kw=keyword)
    rows = list(result)

    if not rows:
        print(f"No verses found for keyword '{keyword}'.")
        return

    print(f"\nKeyword: '{keyword}'  ({len(rows)} verses)")
    print(DIVIDER)

    # Group by surah
    current_surah = None
    for row in rows:
        if row['surah'] != current_surah:
            current_surah = row['surah']
            print(f"\nSurah {row['surah']}: {row['surahName']}")
            print("  " + "-" * 50)
        print(f"  [{row['verseId']}] (score: {row['score']:.4f})")
        print(f"  {row['text'][:120]}{'...' if len(row['text']) > 120 else ''}")
        print()


def cmd_path(session, v1_id: str, v2_id: str):
    """Find the shortest thematic path between two verses."""
    v1_id = v1_id.replace(' ', ':')
    v2_id = v2_id.replace(' ', ':')

    # Verify both exist
    for vid in [v1_id, v2_id]:
        if not session.run("MATCH (v:Verse {verseId: $id}) RETURN v", id=vid).single():
            print(f"Verse [{vid}] not found.")
            return

    result = session.run("""
        MATCH (v1:Verse {verseId: $v1}), (v2:Verse {verseId: $v2}),
              path = shortestPath((v1)-[:RELATED_TO*..6]-(v2))
        RETURN path, length(path) AS hops
        LIMIT 1
    """, v1=v1_id, v2=v2_id).single()

    if not result:
        print(f"No path found between [{v1_id}] and [{v2_id}] within 6 hops.")
        print("These verses may not be thematically connected in the graph.")
        return

    path = result['path']
    hops = result['hops']
    nodes = path.nodes

    print(f"\nShortest thematic path: [{v1_id}] -> [{v2_id}]  ({hops} hops)")
    print(DIVIDER)

    for i, node in enumerate(nodes):
        verse_id = node['verseId']
        print(f"\nStep {i+1}: [{verse_id}] {node['surahName']}")
        print(f"  {node['text'][:120]}{'...' if len(node['text']) > 120 else ''}")

        if i < len(nodes) - 1:
            next_id = nodes[i+1]['verseId']
            shared = session.run("""
                MATCH (v1:Verse {verseId: $v1})-[:MENTIONS]->(k:Keyword)<-[:MENTIONS]-(v2:Verse {verseId: $v2})
                RETURN k.keyword AS kw LIMIT 5
            """, v1=verse_id, v2=next_id)
            kws = [r['kw'] for r in shared]
            print(f"  | shared keywords: {', '.join(kws)}")


def cmd_cluster(session, surah_num: int):
    """Show all verses in a surah and their cross-surah connections."""
    result = session.run("""
        MATCH (v:Verse {surah: $surah})
        RETURN v.verseId AS verseId, v.surahName AS surahName, v.text AS text, v.verseNum AS verseNum
        ORDER BY v.verseNum
    """, surah=surah_num)
    verses = list(result)

    if not verses:
        print(f"No verses found for Surah {surah_num}.")
        return

    surah_name = verses[0]['surahName']
    print(f"\nSurah {surah_num}: {surah_name}  ({len(verses)} verses)")
    print(DIVIDER)

    # Internal structure
    print("\nVerses:")
    for v in verses:
        print(f"  [{v['verseId']}] {v['text'][:90]}{'...' if len(v['text']) > 90 else ''}")

    # Cross-surah connections
    verse_ids = [v['verseId'] for v in verses]
    cross_result = session.run("""
        UNWIND $verseIds AS vid
        MATCH (v:Verse {verseId: vid})-[r:RELATED_TO]-(other:Verse)
        WHERE other.surah <> $surah
        WITH other.surah AS otherSurah, other.surahName AS otherName,
             collect(DISTINCT vid) AS fromVerses,
             collect(DISTINCT other.verseId) AS toVerses,
             count(*) AS connections
        RETURN otherSurah, otherName, fromVerses, toVerses, connections
        ORDER BY connections DESC
        LIMIT 20
    """, verseIds=verse_ids, surah=surah_num)
    cross = list(cross_result)

    print(f"\nCross-surah connections (top 20):")
    print(DIVIDER)
    for row in cross:
        print(f"\n  Surah {row['otherSurah']}: {row['otherName']}  ({row['connections']} connections)")
        from_sample = ', '.join(f"[{v}]" for v in row['fromVerses'][:3])
        to_sample   = ', '.join(f"[{v}]" for v in row['toVerses'][:3])
        print(f"  From this surah: {from_sample}{'...' if len(row['fromVerses']) > 3 else ''}")
        print(f"  To that surah:   {to_sample}{'...' if len(row['toVerses']) > 3 else ''}")


# -- Entry point ---------------------------------------------------------------

USAGE = """
Quran Graph Explorer
Usage:
  py explore.py verse 2:255          -> verse details + connections
  py explore.py keyword covenant      -> all verses with keyword, by surah
  py explore.py path 2:255 112:1     -> shortest thematic path between verses
  py explore.py cluster 36           -> Surah 36 internal + cross-surah map
"""

def main():
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(0)

    cmd = sys.argv[1].lower()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"Cannot connect to Neo4j: {e}")
        print("Make sure Neo4j Desktop DBMS is running.")
        sys.exit(1)

    with driver.session() as session:
        if cmd == "verse":
            if len(sys.argv) < 3:
                print("Usage: py explore.py verse <surah:verse>  e.g. 2:255")
                sys.exit(1)
            cmd_verse(session, sys.argv[2])

        elif cmd == "keyword":
            if len(sys.argv) < 3:
                print("Usage: py explore.py keyword <word>  e.g. covenant")
                sys.exit(1)
            cmd_keyword(session, ' '.join(sys.argv[2:]))

        elif cmd == "path":
            if len(sys.argv) < 4:
                print("Usage: py explore.py path <v1> <v2>  e.g. 2:255 112:1")
                sys.exit(1)
            cmd_path(session, sys.argv[2], sys.argv[3])

        elif cmd == "cluster":
            if len(sys.argv) < 3:
                print("Usage: py explore.py cluster <surah_number>  e.g. 36")
                sys.exit(1)
            try:
                surah_num = int(sys.argv[2])
            except ValueError:
                print("Surah number must be an integer.")
                sys.exit(1)
            cmd_cluster(session, surah_num)

        else:
            print(f"Unknown command: '{cmd}'")
            print(USAGE)

    driver.close()


if __name__ == "__main__":
    main()
