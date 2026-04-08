"""
Step 4 — Conversational Q&A using the Quran knowledge graph + Claude.

Usage:
    py graph_qa.py

For each question:
1. Extract keywords (lemmatized, stopword-filtered)
2. Find verses that MENTION those keywords (ranked by TF-IDF score)
3. Traverse RELATED_TO 1-2 hops from seed verses
4. Rank: direct match > 1-hop > 2-hop
5. Take top 15-20 verses as context
6. Send to Claude with the system prompt
7. Print answer with inline citations + Sources section
"""

import os
import re
import sys
from collections import defaultdict
from dotenv import load_dotenv
from neo4j import GraphDatabase
import anthropic

def _load_env(path=".env"):
    """Load .env manually — dotenv can silently drop long values."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():  # force-set non-empty values, overriding bad dotenv parse
                    os.environ[k.strip()] = v.strip()

# Reuse lemmatizer from build_graph
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_graph import tokenize_and_lemmatize

load_dotenv()
_load_env(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
MODEL          = "claude-sonnet-4-5"

SYSTEM_PROMPT = """You are a Quran scholar assistant. Answer questions using only the verses \
provided as context. Always cite verse references inline like [2:255]. \
When graph-connected verses are provided, explain how they relate to the question \
even if not an obvious match — these connections reveal thematic depth. \
Be thorough but concise. If the context doesn't contain enough information to answer \
the question, say so clearly rather than speculating."""

TOP_K          = 20     # max total verses to send to Claude
MAX_DIRECT     = 10     # max direct keyword-match verses
MAX_HOP1       = 6      # max 1-hop traversal verses
MAX_HOP2       = 4      # max 2-hop traversal verses


# ── Neo4j queries ─────────────────────────────────────────────────────────────

def find_direct_matches(session, keywords: list[str], limit: int = MAX_DIRECT) -> list[dict]:
    """Find verses that MENTION any of the given keywords, ranked by total score."""
    if not keywords:
        return []
    result = session.run("""
        UNWIND $keywords AS kw
        MATCH (k:Keyword {keyword: kw})<-[r:MENTIONS]-(v:Verse)
        WITH v, sum(r.score) AS total_score, collect(kw) AS matched_kws
        RETURN v.verseId AS verseId, v.surah AS surah, v.surahName AS surahName,
               v.text AS text, total_score, matched_kws
        ORDER BY total_score DESC
        LIMIT $limit
    """, keywords=keywords, limit=limit)
    return [dict(r) for r in result]


def find_hop1_verses(session, seed_ids: list[str], exclude_ids: set, limit: int = MAX_HOP1) -> list[dict]:
    """1-hop RELATED_TO traversal from seed verses."""
    if not seed_ids:
        return []
    result = session.run("""
        UNWIND $seedIds AS seedId
        MATCH (seed:Verse {verseId: seedId})-[r:RELATED_TO]-(neighbor:Verse)
        WHERE NOT neighbor.verseId IN $excludeIds
        WITH neighbor, sum(r.score) AS hop_score, collect(seedId) AS via_seeds
        RETURN neighbor.verseId AS verseId, neighbor.surah AS surah,
               neighbor.surahName AS surahName, neighbor.text AS text,
               hop_score, via_seeds
        ORDER BY hop_score DESC
        LIMIT $limit
    """, seedIds=seed_ids, excludeIds=list(exclude_ids), limit=limit)
    return [dict(r) for r in result]


def find_hop2_verses(session, hop1_ids: list[str], exclude_ids: set, limit: int = MAX_HOP2) -> list[dict]:
    """2-hop RELATED_TO traversal."""
    if not hop1_ids:
        return []
    result = session.run("""
        UNWIND $hop1Ids AS h1Id
        MATCH (h1:Verse {verseId: h1Id})-[:RELATED_TO]-(h2:Verse)
        WHERE NOT h2.verseId IN $excludeIds
        WITH h2, count(*) AS connections
        RETURN h2.verseId AS verseId, h2.surah AS surah,
               h2.surahName AS surahName, h2.text AS text,
               connections AS hop_score, [] AS via_seeds
        ORDER BY connections DESC
        LIMIT $limit
    """, hop1Ids=hop1_ids, excludeIds=list(exclude_ids), limit=limit)
    return [dict(r) for r in result]


def get_shared_keywords(session, v1_id: str, v2_id: str) -> list[str]:
    """Get keywords shared between two verses."""
    result = session.run("""
        MATCH (v1:Verse {verseId: $v1})-[:MENTIONS]->(k:Keyword)<-[:MENTIONS]-(v2:Verse {verseId: $v2})
        RETURN k.keyword AS kw
        LIMIT 5
    """, v1=v1_id, v2=v2_id)
    return [r['kw'] for r in result]


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(direct: list[dict], hop1: list[dict], hop2: list[dict]) -> str:
    """Format retrieved verses as context for Claude."""
    lines = []

    if direct:
        lines.append("## Direct keyword matches:")
        for v in direct:
            lines.append(f"[{v['verseId']}] ({v['surahName']}) {v['text']}")
            lines.append("")

    if hop1:
        lines.append("## Thematically connected verses (1 hop):")
        for v in hop1:
            lines.append(f"[{v['verseId']}] ({v['surahName']}) {v['text']}")
            lines.append("")

    if hop2:
        lines.append("## Thematically connected verses (2 hops):")
        for v in hop2:
            lines.append(f"[{v['verseId']}] ({v['surahName']}) {v['text']}")
            lines.append("")

    return '\n'.join(lines)


def build_sources_section(
    keywords: list[str],
    direct: list[dict],
    hop1: list[dict],
    hop2: list[dict],
    session
) -> str:
    lines = ["\n" + "─"*60, "SOURCES", "─"*60]

    lines.append(f"\nKeywords extracted: {', '.join(keywords) if keywords else '(none)'}")

    if direct:
        lines.append(f"\nDirect keyword matches ({len(direct)}):")
        for v in direct:
            kws = ', '.join(v.get('matched_kws', []))
            lines.append(f"  [{v['verseId']}] {v['surahName']} — keywords: {kws}")

    if hop1:
        lines.append(f"\n1-hop graph connections ({len(hop1)}):")
        for v in hop1:
            seeds = v.get('via_seeds', [])
            for seed_id in seeds[:2]:
                shared = get_shared_keywords(session, seed_id, v['verseId'])
                if shared:
                    lines.append(f"  [{v['verseId']}] via [{seed_id}] — shared: {', '.join(shared)}")
                    break
            else:
                lines.append(f"  [{v['verseId']}] {v['surahName']}")

    if hop2:
        lines.append(f"\n2-hop graph connections ({len(hop2)}):")
        for v in hop2:
            lines.append(f"  [{v['verseId']}] {v['surahName']}")

    return '\n'.join(lines)


# ── Main Q&A loop ─────────────────────────────────────────────────────────────

def answer_question(
    question: str,
    session,
    client: anthropic.Anthropic,
    conversation_history: list[dict],
) -> str:
    # 1. Extract keywords
    keywords = list(set(tokenize_and_lemmatize(question)))
    print(f"\n  Keywords: {keywords}")

    # 2. Direct matches
    direct = find_direct_matches(session, keywords)
    direct_ids = {v['verseId'] for v in direct}

    # 3. 1-hop traversal
    hop1 = find_hop1_verses(session, list(direct_ids), direct_ids)
    hop1_ids = {v['verseId'] for v in hop1}

    # 4. 2-hop traversal
    hop2 = find_hop2_verses(session, list(hop1_ids), direct_ids | hop1_ids)

    print(f"  Retrieved: {len(direct)} direct, {len(hop1)} 1-hop, {len(hop2)} 2-hop")

    if not direct and not hop1:
        return "I couldn't find relevant verses for this question in the graph. Try rephrasing with different keywords."

    # 5. Build context
    context = build_context(direct, hop1, hop2)

    # 6. Call Claude with conversation history
    user_message = f"Context from the Quran knowledge graph:\n\n{context}\n\nQuestion: {question}"
    conversation_history.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=conversation_history,
    )
    answer = response.content[0].text

    # Add assistant response to history
    conversation_history.append({"role": "assistant", "content": answer})

    # 7. Build sources
    sources = build_sources_section(keywords, direct, hop1, hop2, session)

    return answer + sources


def main():
    print("Quran Knowledge Graph Q&A")
    print("=" * 60)
    print(f"Model: {MODEL}")

    if not ANTHROPIC_KEY or ANTHROPIC_KEY == "your_api_key_here":
        print("\n⚠  Set ANTHROPIC_API_KEY in your .env file first.")
        sys.exit(1)

    print(f"\nConnecting to Neo4j at {NEO4J_URI}...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        print("  Connected OK")
    except Exception as e:
        print(f"  Connection failed: {e}")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    conversation_history = []

    print("\nType your question (or 'quit' to exit, 'clear' to reset history):\n")

    with driver.session() as session:
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not question:
                continue
            if question.lower() in ('quit', 'exit', 'q'):
                print("Goodbye.")
                break
            if question.lower() == 'clear':
                conversation_history.clear()
                print("  Conversation history cleared.\n")
                continue

            print("\nSearching graph...", end='')
            answer = answer_question(question, session, client, conversation_history)
            print(f"\n\nAssistant:\n{answer}\n")

    driver.close()


if __name__ == "__main__":
    main()
