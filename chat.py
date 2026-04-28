"""
Quran Graph Chat — conversational LLM agent with graph tools.

Claude drives the exploration. Instead of a fixed keyword→query→answer pipeline,
Claude decides which tools to call, how deep to traverse, and when it has enough
context to answer well.

Usage:
    py chat.py

Tools available to Claude:
    search_keyword(keyword)              → verses mentioning a keyword, by surah
    get_verse(verse_id)                  → verse text + connected verses + shared keywords
    traverse_topic(keywords, hops)       → multi-keyword search + graph traversal
    find_path(verse_id_1, verse_id_2)    → shortest thematic path between two verses
    explore_surah(surah_number)          → all verses in a surah + cross-surah connections
"""

import json
import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase
import anthropic
import config as cfg

# ── env loading ────────────────────────────────────────────────────────────────

def _load_env(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ[k.strip()] = v.strip()

load_dotenv()
_load_env()

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_TOKEN = os.getenv("ANTHROPIC_OAUTH_TOKEN", "")
MODEL          = cfg.llm_model()

# ── reuse lemmatizer ───────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_graph import tokenize_and_lemmatize

# ── semantic model (loaded once, reused across requests) ───────────────────────
#
# Two-index design: pick which embedding model + index `tool_semantic_search`
# uses via the SEMANTIC_SEARCH_INDEX env var. Default keeps the legacy
# all-MiniLM-L6-v2 path so nothing breaks; flip to BGE-M3 by setting:
#   SEMANTIC_SEARCH_INDEX=verse_embedding_m3
# (after running `python embed_verses_m3.py` to populate it)

_SEMANTIC_INDEX = os.environ.get("SEMANTIC_SEARCH_INDEX", "verse_embedding").strip()

# Map: index name -> embedding model name
_INDEX_TO_MODEL = {
    "verse_embedding":      None,                # uses cfg.embedding_model() (= all-MiniLM-L6-v2)
    "verse_embedding_m3":   "BAAI/bge-m3",       # English BGE-M3
    "verse_embedding_m3_ar": "BAAI/bge-m3",      # Arabic BGE-M3 — same model, different index
}

_sem_models = {}     # name -> SentenceTransformer

def _get_sem_model_for(index_name: str = None):
    """Return the right SentenceTransformer for the active vector index."""
    idx = index_name or _SEMANTIC_INDEX
    name = _INDEX_TO_MODEL.get(idx)
    if name is None:
        name = cfg.embedding_model()
    if name not in _sem_models:
        from sentence_transformers import SentenceTransformer
        _sem_models[name] = SentenceTransformer(name)
    return _sem_models[name]

# Back-compat alias
def _get_sem_model():
    return _get_sem_model_for(_SEMANTIC_INDEX)

# ── graph tool implementations ─────────────────────────────────────────────────

def tool_search_keyword(session, keyword: str) -> dict:
    """Find ALL verses mentioning a keyword, grouped by surah."""
    lemmas = tokenize_and_lemmatize(keyword)
    if not lemmas:
        return {"error": f"'{keyword}' is a stopword or too short — try a different word"}
    kw = lemmas[0]

    exists = session.run("MATCH (k:Keyword {keyword: $kw}) RETURN k.keyword AS kw", kw=kw).single()
    if not exists:
        # Fuzzy fallback: suggest similar keywords
        similar = [r["kw"] for r in session.run(
            "MATCH (k:Keyword) WHERE k.keyword STARTS WITH $prefix RETURN k.keyword AS kw LIMIT $lim",
            prefix=kw[:cfg.search_keyword_fuzzy_prefix()], lim=cfg.search_keyword_fuzzy_limit()
        )]
        return {
            "error": f"Keyword '{kw}' not in graph",
            "suggestions": similar,
            "tip": "Try one of the suggested keywords above"
        }

    # No LIMIT — return every verse that mentions this keyword
    rows = list(session.run("""
        MATCH (k:Keyword {keyword: $kw})<-[r:MENTIONS]-(v:Verse)
        RETURN v.surah AS surah, v.surahName AS surahName,
               v.verseId AS verseId, v.text AS text, v.arabicText AS arabic,
               r.score AS score
        ORDER BY r.score DESC
    """, kw=kw))

    by_surah = {}
    for row in rows:
        sname = f"Surah {row['surah']}: {row['surahName']}"
        by_surah.setdefault(sname, []).append({
            "verse_id": row["verseId"],
            "score": round(row["score"], 4),
            "text": row["text"],
            "arabic_text": row["arabic"] or "",
        })

    return {
        "keyword": kw,
        "total_verses": len(rows),
        "by_surah": by_surah
    }


def tool_get_verse(session, verse_id: str) -> dict:
    """Get a verse's full text, its keywords, and all directly connected verses."""
    verse_id = verse_id.strip().replace(" ", ":")
    row = session.run("MATCH (v:Verse {verseId: $id}) RETURN v", id=verse_id).single()
    if not row:
        return {"error": f"Verse [{verse_id}] not found. Format: surah:verse e.g. 2:255"}

    v = row["v"]
    keywords = [r["kw"] for r in session.run("""
        MATCH (v:Verse {verseId: $id})-[r:MENTIONS]->(k:Keyword)
        RETURN k.keyword AS kw ORDER BY r.score DESC LIMIT $lim
    """, id=verse_id, lim=cfg.get_verse_keyword_limit())]

    neighbours = list(session.run("""
        MATCH (v:Verse {verseId: $id})-[r:RELATED_TO]-(other:Verse)
        RETURN other.verseId AS otherId, other.surahName AS surahName,
               other.text AS text, r.score AS score
        ORDER BY r.score DESC LIMIT $lim
    """, id=verse_id, lim=cfg.get_verse_neighbour_limit()))

    # Batch: fetch shared keywords for ALL neighbours in one query
    neighbour_ids = [r["otherId"] for r in neighbours]
    shared_map = {}
    if neighbour_ids:
        shared_rows = session.run("""
            UNWIND $otherIds AS oid
            MATCH (v1:Verse {verseId: $v1})-[:MENTIONS]->(k:Keyword)<-[:MENTIONS]-(v2:Verse {verseId: oid})
            WITH oid, collect(k.keyword)[..$skLim] AS kws
            RETURN oid, kws
        """, v1=verse_id, otherIds=neighbour_ids, skLim=cfg.get_verse_shared_kw_limit())
        for sr in shared_rows:
            shared_map[sr["oid"]] = sr["kws"]

    connected = [{
        "verse_id": r["otherId"],
        "surah_name": r["surahName"],
        "text": r["text"],
        "shared_keywords": shared_map.get(r["otherId"], []),
        "connection_score": round(r["score"], 4)
    } for r in neighbours]

    # Fetch top Arabic roots for this verse
    arabic_roots = [{"root": r["root"], "gloss": r["gloss"] or "", "forms": r["forms"] or []}
                    for r in session.run("""
        MATCH (v:Verse {verseId: $id})-[m:MENTIONS_ROOT]->(r:ArabicRoot)
        RETURN r.root AS root, r.gloss AS gloss, m.forms AS forms
        ORDER BY m.count DESC LIMIT 10
    """, id=verse_id)]

    # Typed edges summary (SUPPORTS, ELABORATES, etc.)
    typed_rows = list(session.run("""
        MATCH (v:Verse {verseId: $id})-[r]-(other:Verse)
        WHERE type(r) IN ['SUPPORTS','ELABORATES','QUALIFIES','CONTRASTS','REPEATS']
        RETURN other.verseId AS otherId, type(r) AS relType,
               r.score AS score, r.confidence AS confidence
        ORDER BY r.score DESC LIMIT 15
    """, id=verse_id))
    typed_edges = {}
    for tr in typed_rows:
        typed_edges.setdefault(tr["relType"], []).append({
            "verse_id": tr["otherId"],
            "score": round(tr["score"], 4) if tr["score"] else None,
        })

    return {
        "verse_id": verse_id,
        "surah": v["surah"],
        "surah_name": v["surahName"],
        "text": v["text"],
        "arabic_text": v.get("arabicText", "") or "",
        "arabic_roots": arabic_roots,
        "keywords": keywords,
        "connected_verses": connected,
        "typed_edges": typed_edges,
    }


def tool_traverse_topic(session, keywords: list[str], hops: int = 1) -> dict:
    """Multi-keyword search + graph traversal to map a topic."""
    hops = max(1, min(hops, 2))
    lemmas = list(set(
        lemma
        for kw in keywords
        for lemma in tokenize_and_lemmatize(kw)
    ))
    if not lemmas:
        return {"error": "All keywords were stopwords. Use more specific terms."}

    # Direct matches — no limit, return ALL verses matching any of the keywords
    direct = list(session.run("""
        UNWIND $keywords AS kw
        MATCH (k:Keyword {keyword: kw})<-[r:MENTIONS]-(v:Verse)
        WITH v, sum(r.score) AS total, collect(kw) AS matched
        RETURN v.verseId AS verseId, v.surahName AS surahName,
               v.text AS text, total, matched
        ORDER BY total DESC
    """, keywords=lemmas))

    direct_ids = [r["verseId"] for r in direct]
    direct_results = [{
        "verse_id": r["verseId"],
        "surah_name": r["surahName"],
        "text": r["text"],
        "matched_keywords": r["matched"],
        "score": round(r["total"], 4)
    } for r in direct]

    seed_ids = direct_ids[:cfg.traverse_seed_limit()]

    # 1-hop — top 60 thematically connected verses not already in direct matches
    hop1 = list(session.run("""
        UNWIND $seedIds AS sid
        MATCH (seed:Verse {verseId: sid})-[r:RELATED_TO]-(n:Verse)
        WHERE NOT n.verseId IN $allDirectIds
        WITH n, sum(r.score) AS score, collect(sid) AS via
        RETURN n.verseId AS verseId, n.surahName AS surahName,
               n.text AS text, score, via
        ORDER BY score DESC LIMIT $lim
    """, seedIds=seed_ids, allDirectIds=direct_ids, lim=cfg.traverse_hop1_limit()))

    hop1_ids = [r["verseId"] for r in hop1]
    hop1_results = [{
        "verse_id": r["verseId"],
        "surah_name": r["surahName"],
        "text": r["text"],
        "connected_via": r["via"],
        "score": round(r["score"], 4)
    } for r in hop1]

    hop2_results = []
    if hops >= 2:
        exclude = set(direct_ids + hop1_ids)
        hop2 = list(session.run("""
            UNWIND $h1Ids AS h1
            MATCH (h1v:Verse {verseId: h1})-[:RELATED_TO]-(h2:Verse)
            WHERE NOT h2.verseId IN $exclude
            WITH h2, count(*) AS connections
            RETURN h2.verseId AS verseId, h2.surahName AS surahName,
                   h2.text AS text, connections
            ORDER BY connections DESC LIMIT $lim
        """, h1Ids=hop1_ids, exclude=list(exclude), lim=cfg.traverse_hop2_limit()))
        hop2_results = [{
            "verse_id": r["verseId"],
            "surah_name": r["surahName"],
            "text": r["text"]
        } for r in hop2]

    return {
        "keywords_used": lemmas,
        "direct_matches": direct_results,
        "hop_1_connections": hop1_results,
        "hop_2_connections": hop2_results,
        "total_verses_found": len(direct_results) + len(hop1_results) + len(hop2_results)
    }


def tool_find_path(session, verse_id_1: str, verse_id_2: str) -> dict:
    """Find the shortest thematic path between two verses through the graph."""
    v1 = verse_id_1.strip().replace(" ", ":")
    v2 = verse_id_2.strip().replace(" ", ":")

    for vid in [v1, v2]:
        if not session.run("MATCH (v:Verse {verseId: $id}) RETURN v", id=vid).single():
            return {"error": f"Verse [{vid}] not found"}

    result = session.run("""
        MATCH (v1:Verse {verseId: $v1}), (v2:Verse {verseId: $v2}),
              path = shortestPath((v1)-[:RELATED_TO*..%d]-(v2))""" % cfg.find_path_max_depth() + """
        RETURN path, length(path) AS hops LIMIT 1
    """, v1=v1, v2=v2).single()

    if not result:
        return {"error": f"No path found between [{v1}] and [{v2}] within {cfg.find_path_max_depth()} hops"}

    nodes = result["path"].nodes

    # Batch: fetch bridge keywords for ALL consecutive pairs in one query
    pairs = [{"a": nodes[i]["verseId"], "b": nodes[i + 1]["verseId"]}
             for i in range(len(nodes) - 1)]
    bridge_map = {}
    if pairs:
        bridge_rows = session.run("""
            UNWIND $pairs AS pair
            MATCH (a:Verse {verseId: pair.a})-[:MENTIONS]->(k:Keyword)<-[:MENTIONS]-(b:Verse {verseId: pair.b})
            WITH pair.a AS fromId, pair.b AS toId, collect(k.keyword)[..$bkLim] AS kws
            RETURN fromId, toId, kws
        """, pairs=pairs, bkLim=cfg.find_path_bridge_kw_limit())
        for br in bridge_rows:
            bridge_map[(br["fromId"], br["toId"])] = br["kws"]

    steps = []
    for i, node in enumerate(nodes):
        step = {
            "step": i + 1,
            "verse_id": node["verseId"],
            "surah_name": node["surahName"],
            "text": node["text"]
        }
        if i < len(nodes) - 1:
            step["bridge_keywords"] = bridge_map.get(
                (node["verseId"], nodes[i + 1]["verseId"]), [])
        steps.append(step)

    return {
        "from": v1,
        "to": v2,
        "hops": result["hops"],
        "path": steps
    }


def tool_explore_surah(session, surah_number: int) -> dict:
    """Get all verses in a surah and its top cross-surah thematic connections."""
    verses = list(session.run("""
        MATCH (v:Verse {surah: $s})
        RETURN v.verseId AS verseId, v.text AS text, v.verseNum AS verseNum
        ORDER BY v.verseNum
    """, s=surah_number))

    if not verses:
        return {"error": f"Surah {surah_number} not found (valid range: 1-114)"}

    surah_name = session.run(
        "MATCH (v:Verse {surah: $s}) RETURN v.surahName AS name LIMIT 1", s=surah_number
    ).single()["name"]

    verse_ids = [v["verseId"] for v in verses]
    cross = list(session.run("""
        UNWIND $vids AS vid
        MATCH (v:Verse {verseId: vid})-[:RELATED_TO]-(other:Verse)
        WHERE other.surah <> $s
        WITH other.surah AS otherSurah, other.surahName AS otherName, count(*) AS connections
        RETURN otherSurah, otherName, connections
        ORDER BY connections DESC LIMIT $lim
    """, vids=verse_ids, s=surah_number, lim=cfg.explore_surah_cross_limit()))

    return {
        "surah": surah_number,
        "surah_name": surah_name,
        "verse_count": len(verses),
        "verses": [{"verse_id": v["verseId"], "text": v["text"]} for v in verses],
        "top_cross_surah_connections": [{
            "surah": r["otherSurah"],
            "surah_name": r["otherName"],
            "connections": r["connections"]
        } for r in cross]
    }


def tool_search_arabic_root(session, root: str) -> dict:
    """Find all verses containing a specific Arabic tri-literal root."""
    root = root.strip()

    # Check if root exists
    row = session.run(
        "MATCH (r:ArabicRoot {root: $root}) RETURN r.root AS root, r.gloss AS gloss, r.verseCount AS vc",
        root=root,
    ).single()

    if not row:
        # Try Buckwalter lookup
        row = session.run(
            "MATCH (r:ArabicRoot {rootBW: $bw}) RETURN r.root AS root, r.gloss AS gloss, r.verseCount AS vc",
            bw=root,
        ).single()

    if not row:
        # Fuzzy: suggest roots starting with same first letter
        similar = [r["root"] + (" (" + r["gloss"] + ")" if r["gloss"] else "")
                   for r in session.run(
            "MATCH (r:ArabicRoot) WHERE r.root STARTS WITH $prefix RETURN r.root, r.gloss LIMIT 8",
            prefix=root[:1] if root else "",
        )]
        return {"error": f"Root '{root}' not found", "suggestions": similar}

    actual_root = row["root"]
    gloss = row["gloss"] or ""

    verses = list(session.run("""
        MATCH (r:ArabicRoot {root: $root})<-[m:MENTIONS_ROOT]-(v:Verse)
        RETURN v.verseId AS verseId, v.surah AS surah, v.surahName AS surahName,
               v.text AS text, v.arabicText AS arabic,
               m.count AS count, m.forms AS forms
        ORDER BY m.count DESC, v.surah, v.verseNum
    """, root=actual_root))

    by_surah = {}
    for r in verses:
        sname = f"Surah {r['surah']}: {r['surahName']}"
        by_surah.setdefault(sname, []).append({
            "verse_id": r["verseId"],
            "text": r["text"],
            "arabic_text": r["arabic"] or "",
            "root_count": r["count"],
            "forms_used": r["forms"] or [],
        })

    return {
        "root": actual_root,
        "gloss": gloss,
        "total_verses": len(verses),
        "by_surah": by_surah,
    }


def tool_compare_arabic_usage(session, root: str) -> dict:
    """Compare how different forms of an Arabic root are used across the Quran."""
    root = root.strip()

    row = session.run(
        "MATCH (r:ArabicRoot {root: $root}) RETURN r.root AS root, r.gloss AS gloss",
        root=root,
    ).single()
    if not row:
        row = session.run(
            "MATCH (r:ArabicRoot {rootBW: $bw}) RETURN r.root AS root, r.gloss AS gloss",
            bw=root,
        ).single()
    if not row:
        return {"error": f"Root '{root}' not found. Use search_arabic_root for suggestions."}

    actual_root = row["root"]
    gloss = row["gloss"] or ""

    verses = list(session.run("""
        MATCH (r:ArabicRoot {root: $root})<-[m:MENTIONS_ROOT]-(v:Verse)
        RETURN v.verseId AS verseId, v.surah AS surah, v.surahName AS surahName,
               v.text AS text, v.arabicText AS arabic, m.forms AS forms
        ORDER BY v.surah, v.verseNum
    """, root=actual_root))

    # Group by surface form
    by_form = {}
    for r in verses:
        for form in (r["forms"] or []):
            by_form.setdefault(form, []).append({
                "verse_id": r["verseId"],
                "surah_name": r["surahName"],
                "text": r["text"],
                "arabic_text": r["arabic"] or "",
            })

    # Limit forms and verses per form
    try:
        max_forms = cfg.raw().get("arabic", {}).get("compare_arabic_usage", {}).get("max_forms_per_root", 20)
        max_per_form = cfg.raw().get("arabic", {}).get("compare_arabic_usage", {}).get("max_verses_per_form", 10)
    except Exception:
        max_forms, max_per_form = 20, 10

    form_summary = []
    for form, vlist in sorted(by_form.items(), key=lambda x: -len(x[1]))[:max_forms]:
        form_summary.append({
            "form": form,
            "total_occurrences": len(vlist),
            "sample_verses": vlist[:max_per_form],
        })

    return {
        "root": actual_root,
        "gloss": gloss,
        "total_verses": len(verses),
        "total_forms": len(by_form),
        "forms": form_summary,
    }


def tool_query_typed_edges(session, verse_id: str, edge_type: str = None) -> dict:
    """Query verses connected by a specific relationship type (SUPPORTS, ELABORATES, etc.)."""
    verse_id = verse_id.strip().replace(" ", ":")
    row = session.run("MATCH (v:Verse {verseId: $id}) RETURN v", id=verse_id).single()
    if not row:
        return {"error": f"Verse [{verse_id}] not found. Format: surah:verse e.g. 2:255"}

    valid_types = ['SUPPORTS', 'ELABORATES', 'QUALIFIES', 'CONTRASTS', 'REPEATS']

    if edge_type:
        edge_type = edge_type.upper()
        if edge_type not in valid_types:
            return {"error": f"Unknown edge type '{edge_type}'. Valid: {valid_types}"}
        rows = list(session.run("""
            MATCH (v:Verse {verseId: $id})-[r:""" + edge_type + """]-(other:Verse)
            RETURN other.verseId AS otherId, other.surahName AS surahName,
                   other.text AS text, other.arabicText AS arabic,
                   r.score AS score, r.confidence AS confidence,
                   $etype AS relType
            ORDER BY r.score DESC LIMIT 12
        """, id=verse_id, etype=edge_type))
    else:
        rows = list(session.run("""
            MATCH (v:Verse {verseId: $id})-[r]-(other:Verse)
            WHERE type(r) IN ['SUPPORTS','ELABORATES','QUALIFIES','CONTRASTS','REPEATS']
            RETURN other.verseId AS otherId, other.surahName AS surahName,
                   other.text AS text, other.arabicText AS arabic,
                   type(r) AS relType, r.score AS score
            ORDER BY r.score DESC LIMIT 20
        """, id=verse_id))

    by_type = {}
    for r in rows:
        rtype = r["relType"]
        by_type.setdefault(rtype, []).append({
            "verse_id": r["otherId"],
            "surah_name": r["surahName"],
            "text": r["text"],
            "arabic_text": r["arabic"] or "",
            "score": round(r["score"], 4) if r["score"] else None,
        })

    return {
        "verse_id": verse_id,
        "edge_type_filter": edge_type,
        "total_results": len(rows),
        "by_type": by_type,
    }


def tool_semantic_search(session, query: str, top_k: int = 40) -> dict:
    """
    Find verses conceptually similar to a query using vector embeddings,
    then enrich each hit with connected graph structure in the same pass.

    Returns per verse:
      - similarity score
      - text + surah context
      - related verses (RELATED_TO edges, top 5)
      - Arabic roots present (top 5)
      - typed edges (SUPPORTS / ELABORATES / QUALIFIES / CONTRASTS / REPEATS)

    This is a VectorCypherRetriever pattern — one tool call gets semantic
    hits plus their graph context, reducing the need for follow-up get_verse
    calls on every hit.
    """
    # Pick model + index dynamically (env-driven so we can A/B test BGE-M3)
    model = _get_sem_model_for(_SEMANTIC_INDEX)
    vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()

    cypher = """
        CALL db.index.vector.queryNodes($index, $k, $vec)
        YIELD node, score
        WHERE node.verseId IS NOT NULL
        WITH node, score ORDER BY score DESC
        // Graph enrichment in the same pass (VectorCypherRetriever pattern)
        OPTIONAL MATCH (node)-[:RELATED_TO]-(related:Verse)
        WITH node, score, collect(DISTINCT related.reference)[0..5] AS related_verses
        OPTIONAL MATCH (node)-[:MENTIONS_ROOT]->(root:ArabicRoot)
        WITH node, score, related_verses,
             collect(DISTINCT root.root)[0..5] AS arabic_roots
        OPTIONAL MATCH (node)-[typed:SUPPORTS|ELABORATES|QUALIFIES|CONTRASTS|REPEATS]-(te:Verse)
        WITH node, score, related_verses, arabic_roots,
             [x IN collect(DISTINCT {type: type(typed), target: te.reference})
              WHERE x.type IS NOT NULL AND x.target IS NOT NULL][0..5] AS typed_edges
        RETURN node.verseId AS verseId, node.surahName AS surahName,
               node.surah AS surah, node.text AS text, score,
               related_verses, arabic_roots, typed_edges
    """
    results = session.run(cypher, index=_SEMANTIC_INDEX, k=top_k, vec=vec).data()

    by_surah = {}
    for r in results:
        sname = f"Surah {r['surah']}: {r['surahName']}"
        entry = {
            "verse_id": r["verseId"],
            "similarity": round(r["score"], 4),
            "text": r["text"],
        }
        # Only include enrichment fields when non-empty — keeps payload lean
        if r["related_verses"]:
            entry["related_verses"] = r["related_verses"]
        if r["arabic_roots"]:
            entry["arabic_roots"] = r["arabic_roots"]
        if r["typed_edges"]:
            entry["typed_edges"] = r["typed_edges"]
        by_surah.setdefault(sname, []).append(entry)

    return {
        "query": query,
        "total_verses": len(results),
        "note": ("Results ranked by semantic similarity. Each hit includes its "
                 "related verses, Arabic roots present, and any typed edges "
                 "(SUPPORTS/ELABORATES/QUALIFIES/CONTRASTS/REPEATS) in one pass."),
        "by_surah": by_surah,
    }


# ── etymology tools ─────────────────────────────────────────────────────────────

def tool_lookup_word(session, word: str) -> dict:
    """Look up an Arabic word — root, lemma, pattern, morphology, occurrences."""
    results = session.run("""
        MATCH (w:WordToken)
        WHERE w.arabicClean CONTAINS $word
           OR w.translitBW CONTAINS $word
           OR w.lemma CONTAINS $word
        WITH w LIMIT 200
        OPTIONAL MATCH (w)-[:HAS_LEMMA]->(l:Lemma)
        OPTIONAL MATCH (l)-[:DERIVES_FROM]->(r:ArabicRoot)
        OPTIONAL MATCH (w)-[:FOLLOWS_PATTERN]->(p:MorphPattern)
        RETURN w.tokenId AS tokenId, w.verseId AS verseId,
               w.arabicText AS arabicText, w.pos AS pos,
               w.morphFeatures AS morphFeatures, w.wazn AS wazn,
               l.lemma AS lemma, l.glossEn AS gloss,
               r.root AS root, r.gloss AS rootGloss,
               p.pattern AS pattern, p.label AS patternLabel,
               p.meaningTendency AS meaningTendency
        ORDER BY w.verseId
    """, word=word).data()

    if not results:
        return {"word": word, "found": False, "message": "Word not found in the Quran"}

    # Group by lemma
    by_lemma = {}
    for r in results:
        lem = r['lemma'] or r['arabicText']
        if lem not in by_lemma:
            by_lemma[lem] = {
                "lemma": lem,
                "root": r['root'] or '',
                "rootGloss": r['rootGloss'] or r['gloss'] or '',
                "pattern": r['pattern'] or r['wazn'] or '',
                "patternLabel": r['patternLabel'] or '',
                "meaningTendency": r['meaningTendency'] or '',
                "pos": r['pos'],
                "occurrences": [],
            }
        if len(by_lemma[lem]['occurrences']) < 50:
            by_lemma[lem]['occurrences'].append({
                "verse_id": r['verseId'],
                "token_id": r['tokenId'],
                "arabic": r['arabicText'],
                "morphology": r['morphFeatures'],
            })

    return {
        "word": word,
        "found": True,
        "total_occurrences": len(results),
        "lemmas": list(by_lemma.values()),
    }


def tool_explore_root_family(session, root: str) -> dict:
    """Full derivative tree of a root — all lemmas grouped by pattern."""
    results = session.run("""
        MATCH (r:ArabicRoot)
        WHERE r.root = $root OR r.rootBW = $root
        OPTIONAL MATCH (l:Lemma)-[:DERIVES_FROM]->(r)
        OPTIONAL MATCH (w:WordToken)-[:HAS_LEMMA]->(l)
        OPTIONAL MATCH (w)-[:FOLLOWS_PATTERN]->(p:MorphPattern)
        OPTIONAL MATCH (r)-[:IN_DOMAIN]->(d:SemanticDomain)
        RETURN r.root AS root, r.gloss AS rootGloss, r.verseCount AS rootVerseCount,
               l.lemma AS lemma, l.glossEn AS lemmaGloss, l.pos AS lemmaPos,
               l.verseCount AS lemmaVerseCount,
               w.verseId AS verseId, w.arabicText AS arabicText,
               p.pattern AS pattern, p.label AS patternLabel,
               d.domainId AS domainId, d.nameEn AS domainName
        ORDER BY l.lemma, w.verseId
    """, root=root).data()

    if not results:
        return {"root": root, "found": False, "message": "Root not found"}

    root_info = {
        "root": results[0]['root'],
        "gloss": results[0]['rootGloss'] or '',
        "verse_count": results[0]['rootVerseCount'] or 0,
    }

    # Collect domains
    domains = {}
    for r in results:
        if r['domainId']:
            domains[r['domainId']] = r['domainName']

    # Group by lemma then pattern
    lemma_map = {}
    for r in results:
        lem = r['lemma']
        if not lem:
            continue
        if lem not in lemma_map:
            lemma_map[lem] = {
                "lemma": lem,
                "gloss": r['lemmaGloss'] or '',
                "pos": r['lemmaPos'] or '',
                "verse_count": r['lemmaVerseCount'] or 0,
                "pattern": r['pattern'] or '',
                "pattern_label": r['patternLabel'] or '',
                "sample_verses": [],
            }
        if r['verseId'] and len(lemma_map[lem]['sample_verses']) < 5:
            # Avoid duplicate verses
            existing = {v['verse_id'] for v in lemma_map[lem]['sample_verses']}
            if r['verseId'] not in existing:
                lemma_map[lem]['sample_verses'].append({
                    "verse_id": r['verseId'],
                    "arabic": r['arabicText'] or '',
                })

    return {
        "root": root_info,
        "found": True,
        "semantic_domains": domains,
        "total_lemmas": len(lemma_map),
        "lemmas": sorted(lemma_map.values(), key=lambda x: -x['verse_count']),
    }


def tool_get_verse_words(session, verse_id: str) -> dict:
    """Word-by-word breakdown of a verse."""
    results = session.run("""
        MATCH (w:WordToken {verseId: $vid})
        OPTIONAL MATCH (w)-[:HAS_LEMMA]->(l:Lemma)
        OPTIONAL MATCH (l)-[:DERIVES_FROM]->(r:ArabicRoot)
        OPTIONAL MATCH (w)-[:FOLLOWS_PATTERN]->(p:MorphPattern)
        RETURN w.tokenId AS tokenId, w.wordPos AS wordPos,
               w.arabicText AS arabicText, w.pos AS pos,
               w.morphFeatures AS morphFeatures, w.wazn AS wazn,
               l.lemma AS lemma, l.glossEn AS gloss,
               r.root AS root, r.gloss AS rootGloss,
               p.pattern AS pattern, p.label AS patternLabel
        ORDER BY w.wordPos
    """, vid=verse_id).data()

    if not results:
        return {"verse_id": verse_id, "found": False, "message": "Verse not found or no word tokens"}

    # Also get the verse text
    verse = session.run("""
        MATCH (v:Verse {verseId: $vid})
        RETURN v.text AS text, v.surahName AS surahName, v.surah AS surah
    """, vid=verse_id).single()

    words = []
    for r in results:
        words.append({
            "position": r['wordPos'],
            "arabic": r['arabicText'],
            "root": r['root'] or '',
            "root_gloss": r['rootGloss'] or r['gloss'] or '',
            "lemma": r['lemma'] or '',
            "pos": r['pos'],
            "pattern": r['pattern'] or r['wazn'] or '',
            "pattern_label": r['patternLabel'] or '',
            "morphology": r['morphFeatures'] or '',
        })

    return {
        "verse_id": verse_id,
        "found": True,
        "surah_name": verse['surahName'] if verse else '',
        "translation": verse['text'] if verse else '',
        "word_count": len(words),
        "words": words,
    }


def tool_search_semantic_field(session, domain: str) -> dict:
    """Find all roots and words in a semantic domain."""
    results = session.run("""
        MATCH (d:SemanticDomain)
        WHERE d.domainId = $domain
           OR d.nameEn CONTAINS $domain
           OR d.nameAr CONTAINS $domain
        WITH d LIMIT 1
        OPTIONAL MATCH (r:ArabicRoot)-[:IN_DOMAIN]->(d)
        OPTIONAL MATCH (l:Lemma)-[:DERIVES_FROM]->(r)
        RETURN d.domainId AS domainId, d.nameEn AS nameEn,
               d.nameAr AS nameAr, d.description AS description,
               r.root AS root, r.gloss AS rootGloss, r.verseCount AS rootVerseCount,
               l.lemma AS lemma, l.glossEn AS lemmaGloss, l.verseCount AS lemmaVerseCount
        ORDER BY r.verseCount DESC, l.verseCount DESC
    """, domain=domain.lower()).data()

    if not results or not results[0]['domainId']:
        # Try fuzzy match
        all_domains = session.run("""
            MATCH (d:SemanticDomain)
            RETURN d.domainId AS id, d.nameEn AS name, d.nameAr AS nameAr
        """).data()
        return {
            "domain": domain,
            "found": False,
            "available_domains": [{"id": d['id'], "name": d['name'], "nameAr": d['nameAr']} for d in all_domains],
        }

    domain_info = {
        "domainId": results[0]['domainId'],
        "nameEn": results[0]['nameEn'],
        "nameAr": results[0]['nameAr'],
        "description": results[0]['description'],
    }

    # Group by root
    root_map = {}
    for r in results:
        if not r['root']:
            continue
        if r['root'] not in root_map:
            root_map[r['root']] = {
                "root": r['root'],
                "gloss": r['rootGloss'] or '',
                "verse_count": r['rootVerseCount'] or 0,
                "lemmas": [],
            }
        if r['lemma']:
            existing = {l['lemma'] for l in root_map[r['root']]['lemmas']}
            if r['lemma'] not in existing:
                root_map[r['root']]['lemmas'].append({
                    "lemma": r['lemma'],
                    "gloss": r['lemmaGloss'] or '',
                    "verse_count": r['lemmaVerseCount'] or 0,
                })

    return {
        "domain": domain_info,
        "found": True,
        "total_roots": len(root_map),
        "roots": sorted(root_map.values(), key=lambda x: -x['verse_count']),
    }


def tool_lookup_wujuh(session, root: str) -> dict:
    """Show all distinct meanings (wujuh) of a polysemous word/root."""
    # First check if we have wujuh data stored as node properties
    # For now, load from the CSV data that was imported
    import csv
    from pathlib import Path
    wujuh_csv = Path(__file__).parent / "data" / "wujuh_entries.csv"

    if not wujuh_csv.exists():
        return {"root": root, "found": False, "message": "Wujuh data not loaded"}

    # Search for matching root
    matches = []
    with open(wujuh_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['root'] == root or root in row.get('lemma', ''):
                matches.append(row)

    if not matches:
        # Try to find verses with this root to provide context even without wujuh data
        verse_data = session.run("""
            MATCH (r:ArabicRoot)
            WHERE r.root = $root OR r.rootBW = $root
            OPTIONAL MATCH (v:Verse)-[:MENTIONS_ROOT]->(r)
            RETURN r.root AS root, r.gloss AS gloss,
                   v.verseId AS verseId, v.text AS text
            ORDER BY v.verseId LIMIT 10
        """, root=root).data()

        if verse_data:
            return {
                "root": root,
                "found": False,
                "message": f"No wujuh (polysemy) data for this root yet. Root '{verse_data[0]['root']}' ({verse_data[0]['gloss']}) appears in {len(verse_data)} sample verses.",
                "sample_verses": [{"verse_id": v['verseId'], "text": v['text']} for v in verse_data if v['verseId']],
            }
        return {"root": root, "found": False, "message": "Root not found"}

    senses = []
    for m in matches:
        sample_verses = json.loads(m.get('sampleVerses', '[]'))
        # Fetch verse texts for each sample
        verse_texts = {}
        if sample_verses:
            vt_results = session.run("""
                UNWIND $vids AS vid
                MATCH (v:Verse {verseId: vid})
                RETURN v.verseId AS verseId, v.text AS text
            """, vids=sample_verses[:5]).data()
            verse_texts = {v['verseId']: v['text'] for v in vt_results}

        senses.append({
            "sense_id": m['senseId'],
            "meaning_en": m['meaningEn'],
            "meaning_ar": m.get('meaningAr', ''),
            "sample_verses": [
                {"verse_id": vid, "text": verse_texts.get(vid, '')}
                for vid in sample_verses[:5]
            ],
        })

    return {
        "root": root,
        "lemma": matches[0].get('lemma', ''),
        "found": True,
        "total_senses": len(senses),
        "senses": senses,
    }


def tool_search_morphological_pattern(session, pattern: str = None,
                                       pos: str = None,
                                       verb_form: str = None) -> dict:
    """Find words by morphological pattern, POS, or verbal form."""
    conditions = []
    params = {}

    if pattern:
        conditions.append("(p.pattern = $pattern OR p.patternBW = $pattern)")
        params['pattern'] = pattern
    if pos:
        conditions.append("w.pos = $pos")
        params['pos'] = pos.upper()
    if verb_form:
        # verb_form could be "IV", "4", "Form IV", etc.
        vf_num = verb_form.replace('Form ', '').replace('form ', '')
        roman_to_num = {'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5',
                        'VI': '6', 'VII': '7', 'VIII': '8', 'IX': '9', 'X': '10', 'XI': '11'}
        vf_num = roman_to_num.get(vf_num, vf_num)
        conditions.append("w.morphFeatures CONTAINS $vf_str")
        params['vf_str'] = f'"vf": "{vf_num}"'

    if not conditions:
        return {"error": "Provide at least one of: pattern, pos, verb_form"}

    where_clause = " AND ".join(conditions)

    if pattern:
        query = f"""
            MATCH (w:WordToken)-[:FOLLOWS_PATTERN]->(p:MorphPattern)
            WHERE {where_clause}
            OPTIONAL MATCH (w)-[:HAS_LEMMA]->(l:Lemma)
            OPTIONAL MATCH (l)-[:DERIVES_FROM]->(r:ArabicRoot)
            RETURN DISTINCT w.arabicText AS arabic, w.pos AS pos,
                   l.lemma AS lemma, r.root AS root, r.gloss AS gloss,
                   p.pattern AS pattern, p.label AS patternLabel,
                   count(w) AS occurrences
            ORDER BY occurrences DESC
            LIMIT 100
        """
    else:
        query = f"""
            MATCH (w:WordToken)
            WHERE {where_clause}
            OPTIONAL MATCH (w)-[:HAS_LEMMA]->(l:Lemma)
            OPTIONAL MATCH (l)-[:DERIVES_FROM]->(r:ArabicRoot)
            OPTIONAL MATCH (w)-[:FOLLOWS_PATTERN]->(p:MorphPattern)
            RETURN DISTINCT w.arabicText AS arabic, w.pos AS pos,
                   l.lemma AS lemma, r.root AS root, r.gloss AS gloss,
                   p.pattern AS pattern, p.label AS patternLabel,
                   count(w) AS occurrences
            ORDER BY occurrences DESC
            LIMIT 100
        """

    results = session.run(query, **params).data()

    if not results:
        return {"found": False, "message": "No words match the given criteria",
                "pattern": pattern, "pos": pos, "verb_form": verb_form}

    # Group by root
    by_root = {}
    for r in results:
        root_key = r['root'] or 'unknown'
        if root_key not in by_root:
            by_root[root_key] = {
                "root": r['root'] or '',
                "gloss": r['gloss'] or '',
                "words": [],
            }
        by_root[root_key]['words'].append({
            "arabic": r['arabic'],
            "lemma": r['lemma'] or '',
            "pos": r['pos'],
            "pattern": r['pattern'] or '',
            "occurrences": r['occurrences'],
        })

    return {
        "found": True,
        "pattern": pattern,
        "pos": pos,
        "verb_form": verb_form,
        "total_distinct_words": len(results),
        "by_root": sorted(by_root.values(), key=lambda x: -sum(w['occurrences'] for w in x['words'])),
    }


def tool_recall_similar_query(session, query: str, top_k: int = 3,
                              min_sim: float = 0.65) -> dict:
    """
    Surface past similar queries from the reasoning_memory subgraph as a
    "playbook" for the current question. Uses the query_embedding vector
    index (MiniLM 384-dim) on (:Query) nodes.

    For each similar past query returns:
      - the question text
      - similarity score
      - the answer that was produced (cited verses + brief excerpt)
      - the tool sequence that led to that answer (which tools, in what order)

    Use when:
      - a question feels familiar / repetitive ("how is patience taught?",
        "what about prayer?")
      - you want to know what graph paths worked before
      - you want to check if the cache likely has a relevant answer

    Don't use when the question is novel / domain-specific or the agent
    has already retrieved a strong answer via direct tools.

    Returns dict with: ok, query, similar_queries: [...]
    """
    if not query or not isinstance(query, str):
        return {"ok": False, "error": "query must be a non-empty string"}

    # Embed the input query with the MiniLM model that wrote the index
    try:
        from sentence_transformers import SentenceTransformer
        m = _sem_models.get("all-MiniLM-L6-v2")
        if m is None:
            m = SentenceTransformer("all-MiniLM-L6-v2")
            _sem_models["all-MiniLM-L6-v2"] = m
        vec = m.encode(query, normalize_embeddings=True).tolist()
    except Exception as e:
        return {"ok": False, "error": f"embed failed: {e}"}

    try:
        rows = session.run("""
            CALL db.index.vector.queryNodes('query_embedding', $k, $vec)
            YIELD node, score WHERE score >= $min
            OPTIONAL MATCH (node)-[:PRODUCED]->(a:Answer)
            OPTIONAL MATCH (node)-[:TRIGGERED]->(t:ReasoningTrace)
            OPTIONAL MATCH (t)-[hs:HAS_STEP]->(tc:ToolCall)
            WITH node, score, a, t,
                 collect({
                     order: hs.order, turn: tc.turn,
                     tool_name: tc.tool_name, args: tc.args_json,
                     ok: tc.ok, summary: tc.summary
                 }) AS steps
            RETURN node.text AS past_question,
                   node.timestamp AS ts,
                   round(score, 4) AS similarity,
                   a.text AS answer_text,
                   a.cited_verses AS cited_verses,
                   t.citation_count AS n_cites,
                   t.status AS status,
                   [s IN steps WHERE s.tool_name IS NOT NULL] AS tool_sequence
            ORDER BY similarity DESC
        """, k=top_k, vec=vec, min=min_sim).data()
    except Exception as e:
        return {"ok": False, "error": f"index query failed: {e}",
                "hint": "query_embedding index may be missing; "
                        "ensure reasoning_memory.ensure_schema() ran"}

    out = []
    for r in rows:
        # Trim answer for compactness — agent gets a hint, not the full text
        answer_excerpt = (r.get("answer_text") or "")[:600]
        out.append({
            "past_question": r["past_question"],
            "similarity": r["similarity"],
            "status": r.get("status"),
            "n_citations": r.get("n_cites"),
            "cited_verses": r.get("cited_verses") or [],
            "answer_excerpt": answer_excerpt,
            "tool_sequence": [
                {"tool": s["tool_name"], "args": s["args"][:120], "ok": s["ok"]}
                for s in (r["tool_sequence"] or [])
            ][:10],   # cap at 10 tool steps
        })
    return {
        "ok": True,
        "query": query,
        "n_similar": len(out),
        "similar_queries": out,
        "note": ("These are the closest past queries this agent has answered. "
                 "Use the tool_sequence as a hint for which tools to call. "
                 "The answer_excerpt is from a past run and may need verification."),
    }


def tool_run_cypher(session, query: str, params: dict = None,
                    row_limit: int = 100) -> dict:
    """
    Execute a read-only Cypher query against the Quran graph.

    Safety wrapper:
      - denylists write/admin clauses (CREATE, DELETE, MERGE, SET, REMOVE,
        DETACH, LOAD CSV, CALL apoc.refactor, CALL dbms, CALL db.create*)
      - injects/enforces a `LIMIT` clause if not present
      - timeouts via underlying Neo4j session config
      - returns rows + a brief schema header so the LLM can self-correct

    Use cases this unlocks:
      - Long-tail user questions the 15 specialised tools don't cover
      - Aggregations across the graph (count, group, percentile)
      - Multi-hop traversals not covered by find_path / explore_root_family
      - Schema introspection ("how many ArabicRoot nodes are there?")

    Returns dict with: ok, rows, columns, n_rows, query_executed, error?
    """
    if not query or not isinstance(query, str):
        return {"ok": False, "error": "query must be a non-empty string"}

    q = query.strip()
    q_upper = q.upper()

    # Denylist of write / admin clauses
    forbidden = [
        r"\bCREATE\b", r"\bMERGE\b", r"\bDELETE\b", r"\bDETACH\b",
        r"\bSET\b", r"\bREMOVE\b", r"\bLOAD\s+CSV\b",
        r"\bCALL\s+APOC\.REFACTOR", r"\bCALL\s+APOC\.PERIODIC",
        r"\bCALL\s+APOC\.NODES\.DELETE", r"\bCALL\s+APOC\.LOAD",
        r"\bCALL\s+DBMS", r"\bDROP\b", r"\bCALL\s+DB\.CREATE",
        r"\bCALL\s+DB\.DROP", r"\bCALL\s+DB\.INDEX\.VECTOR\.CREATE",
        r"\bUSE\s+", r"\bSTART\b", r"\bSTOP\b", r"\bCONSTRAINT\b",
        r"\bINDEX\s+ON\b", r"\bINDEX\s+FOR\b",
    ]
    import re as _re
    for pat in forbidden:
        if _re.search(pat, q, _re.IGNORECASE):
            return {
                "ok": False,
                "error": f"forbidden clause detected (matched {pat!r})",
                "hint": "tool_run_cypher is read-only. Use only MATCH / "
                        "OPTIONAL MATCH / RETURN / WITH / WHERE / ORDER BY / "
                        "LIMIT / UNWIND / collect() / count() / etc."
            }

    # Soft cap: if the query has no LIMIT clause, append one
    if not _re.search(r"\bLIMIT\b\s+\d+", q, _re.IGNORECASE):
        q = q.rstrip(";").rstrip() + f" LIMIT {min(int(row_limit or 100), 500)}"

    try:
        result = session.run(q, **(params or {}))
        rows = result.data()
    except Exception as e:
        return {"ok": False, "error": str(e)[:300], "query_executed": q}

    # Best-effort columns from first row
    cols = list(rows[0].keys()) if rows else []
    return {
        "ok": True,
        "query_executed": q,
        "n_rows": len(rows),
        "columns": cols,
        "rows": rows[:row_limit],
        "note": ("Read-only Cypher. Use MATCH/OPTIONAL MATCH and RETURN. "
                 "Don't include CREATE/MERGE/DELETE/SET/REMOVE/DETACH."),
    }


def tool_get_code19_features(session, scope: str, target: str = None) -> dict:
    """
    Retrieve Khalifa-style Code-19 mathematical features.

    Args:
        scope: "global" | "sura" | "verse"
        target: required for scope="sura" (sura number, e.g. "50")
                or scope="verse" (verseId, e.g. "2:255")

    Returns counts and divisibility-by-19 indicators that have been precomputed
    by build_code19_features.py over the immutable Arabic text. These figures
    are arithmetic (not interpretation) and cannot be hallucinated.
    """
    if scope == "global":
        # Read the precomputed summary file
        from pathlib import Path
        import json as _json
        p = Path(__file__).parent / "data" / "code19_summary.json"
        if not p.exists():
            return {"error": "data/code19_summary.json not found — run build_code19_features.py"}
        try:
            data = _json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            return {"error": f"failed to read code19_summary.json: {e}"}
        return {
            "found": True,
            "scope": "global",
            "khalifa_total_verses": data.get("khalifa_total"),
            "khalifa_total_div_19": data.get("khalifa_total_div_19"),
            "quotient": data.get("khalifa_total_div_by_19_quotient"),
            "numbered_verses": data.get("total_numbered_verses"),
            "unnumbered_basmalahs": data.get("unnumbered_basmalahs"),
            "num_surahs": data.get("num_surahs"),
            "num_surahs_div_19": data.get("num_surahs_div_19"),
            "num_mysterious_letter_surahs": data.get("num_mysterious_letter_surahs"),
            "global_letter_counts": data.get("global_letter_counts"),
            "key_examples": [
                {"sura": 50, "letters": "Q", "letter_counts": data["per_mysterious_letter_surah"].get("50", {}).get("letter_counts")},
                {"sura": 42, "letters": "HMASQ", "letter_counts": data["per_mysterious_letter_surah"].get("42", {}).get("letter_counts")},
                {"sura": 68, "letters": "N", "letter_counts": data["per_mysterious_letter_surah"].get("68", {}).get("letter_counts")},
                {"sura": 38, "letters": "S", "letter_counts": data["per_mysterious_letter_surah"].get("38", {}).get("letter_counts")},
                {"sura": 36, "letters": "YSin", "letter_counts": data["per_mysterious_letter_surah"].get("36", {}).get("letter_counts")},
            ],
        }

    elif scope == "sura":
        if target is None:
            return {"error": "scope='sura' requires target (sura number, e.g. '50')"}
        try:
            num = int(str(target).split(":")[0])
        except ValueError:
            return {"error": f"invalid sura target: {target}"}
        row = session.run("""
            MATCH (su:Sura {number: $n})
            RETURN su.number AS num, su.verses_count AS verses,
                   su.mysterious_letters AS ml,
                   su.ml_letter_counts_json AS counts_json,
                   su.ml_div_19_json AS div19_json,
                   su.mod19_verse_count AS mod19_vc
        """, n=num).single()
        if row is None or row["num"] is None:
            return {"found": False, "error": f"Sura {num} not found or features not stamped"}
        import json as _json
        counts = _json.loads(row["counts_json"]) if row["counts_json"] else {}
        div19 = _json.loads(row["div19_json"]) if row["div19_json"] else {}
        return {
            "found": True,
            "scope": "sura",
            "sura": num,
            "verses_count": row["verses"],
            "mysterious_letters": row["ml"],
            "letter_counts": counts,
            "all_letter_counts_div_19": all(div19.values()) if div19 else None,
            "per_letter_div_19": div19,
            "verses_count_mod_19": row["mod19_vc"],
        }

    elif scope == "verse":
        if target is None:
            return {"error": "scope='verse' requires target (verseId, e.g. '2:255')"}
        row = session.run("""
            MATCH (v:Verse {verseId: $vid})
            RETURN v.verseId AS id, v.surah AS sura, v.verseNum AS vn,
                   v.position_in_sura AS pos,
                   v.is_initial_verse AS init,
                   v.ar_char_count AS char_count, v.ar_word_count AS word_count,
                   v.letter_alif AS alif, v.letter_lam AS lam, v.letter_mim AS mim,
                   v.letter_ra AS ra, v.letter_sad AS sad, v.letter_kaf AS kaf,
                   v.letter_ha AS ha, v.letter_ha_heavy AS ha_heavy,
                   v.letter_ya AS ya, v.letter_ain AS ain, v.letter_ta AS ta,
                   v.letter_sin AS sin, v.letter_qaf AS qaf, v.letter_nun AS nun
        """, vid=str(target)).single()
        if row is None or row["id"] is None:
            return {"found": False, "error": f"verse {target} not found"}
        return {
            "found": True,
            "scope": "verse",
            "verse_id": row["id"],
            "sura": row["sura"],
            "verseNum": row["vn"],
            "position_in_sura": row["pos"],
            "is_initial_verse": row["init"],
            "ar_char_count": row["char_count"],
            "ar_word_count": row["word_count"],
            "letter_counts": {
                "alif": row["alif"], "lam": row["lam"], "mim": row["mim"],
                "ra": row["ra"], "sad": row["sad"], "kaf": row["kaf"],
                "ha": row["ha"], "ha_heavy": row["ha_heavy"],
                "ya": row["ya"], "ain": row["ain"], "ta": row["ta"],
                "sin": row["sin"], "qaf": row["qaf"], "nun": row["nun"],
            },
        }

    else:
        return {"error": f"unknown scope: {scope}; use 'global' | 'sura' | 'verse'"}


# ── tool schema (Anthropic tool_use format) ────────────────────────────────────

TOOLS = [
    {
        "name": "search_keyword",
        "description": (
            "Search for all Quran verses that mention a specific keyword. "
            "Results are grouped by surah and ranked by relevance (TF-IDF score). "
            "Use this to find where a concept appears across the Quran."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "A single word to search for (e.g. 'covenant', 'prayer', 'abraham')"
                }
            },
            "required": ["keyword"]
        }
    },
    {
        "name": "get_verse",
        "description": (
            "Get a specific verse by its ID, along with its keywords and all directly "
            "connected verses in the knowledge graph. Shows what the verse is thematically "
            "linked to and which keywords they share. Use this to deeply explore a single verse."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "verse_id": {
                    "type": "string",
                    "description": "Verse ID in surah:verse format, e.g. '2:255' or '36:1'"
                }
            },
            "required": ["verse_id"]
        }
    },
    {
        "name": "traverse_topic",
        "description": (
            "Explore a topic using multiple keywords + graph traversal. "
            "Finds direct keyword matches first, then expands outward through "
            "thematic connections (1-2 hops) to find related verses that wouldn't "
            "show up in a keyword search. Best for broad topic exploration."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of keywords describing the topic (e.g. ['covenant', 'abraham', 'prophet'])"
                },
                "hops": {
                    "type": "integer",
                    "description": "How many graph hops to traverse: 1 (faster, tighter focus) or 2 (broader, more connections). Default: 1",
                    "default": 1
                }
            },
            "required": ["keywords"]
        }
    },
    {
        "name": "find_path",
        "description": (
            "Find the shortest thematic path between two verses through the knowledge graph. "
            "Shows the chain of connected verses and the keywords that bridge each step. "
            "Useful for discovering unexpected connections between concepts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "verse_id_1": {
                    "type": "string",
                    "description": "Starting verse ID, e.g. '2:255'"
                },
                "verse_id_2": {
                    "type": "string",
                    "description": "Ending verse ID, e.g. '112:1'"
                }
            },
            "required": ["verse_id_1", "verse_id_2"]
        }
    },
    {
        "name": "explore_surah",
        "description": (
            "Get all verses in a specific surah (chapter) and a map of its strongest "
            "thematic connections to other surahs. Use this to understand a surah's "
            "content and how it relates to the rest of the Quran."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "surah_number": {
                    "type": "integer",
                    "description": "Surah number between 1 and 114"
                }
            },
            "required": ["surah_number"]
        }
    },
    {
        "name": "semantic_search",
        "description": (
            "Find verses that are CONCEPTUALLY similar to a query using vector embeddings. "
            "Unlike search_keyword which requires exact word matches, this finds verses that "
            "express the same idea even with completely different vocabulary. "
            "Use this to catch verses about redemption, divine mercy, being freed from sin, "
            "etc. when searching for 'forgiveness' — or any concept where the idea matters "
            "more than the exact words. Always use this alongside keyword search for full coverage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A phrase or sentence describing the concept to search for, e.g. 'God forgiving and accepting repentance' or 'patience in hardship'"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of most similar verses to return (default 40, max 80)",
                    "default": 40
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "query_typed_edges",
        "description": (
            "Find verses connected to a given verse by a specific relationship type: "
            "SUPPORTS (independent evidence), ELABORATES (expands with detail), "
            "QUALIFIES (adds condition/exception), CONTRASTS (complementary perspective), "
            "REPEATS (near-verbatim across surahs). Use this after get_verse to understand "
            "HOW verses relate, not just THAT they relate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "verse_id": {
                    "type": "string",
                    "description": "Verse reference e.g. '2:255'"
                },
                "edge_type": {
                    "type": "string",
                    "enum": ["SUPPORTS", "ELABORATES", "QUALIFIES", "CONTRASTS", "REPEATS"],
                    "description": "Optional: filter to one relationship type. Omit to get all typed edges."
                }
            },
            "required": ["verse_id"]
        }
    },
    {
        "name": "search_arabic_root",
        "description": (
            "Find all Quran verses containing a specific Arabic tri-literal root. "
            "Arabic roots connect words that share a common origin — e.g. root k-t-b "
            "(ك ت ب) yields kitab/book, kataba/wrote, maktub/written. Use this to trace "
            "how an Arabic concept appears across the entire Quran with all its derived forms. "
            "Accepts Arabic script (e.g. 'رحم') or Buckwalter transliteration (e.g. 'rHm')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "root": {
                    "type": "string",
                    "description": "Arabic root in Arabic letters (e.g. 'رحم', 'كتب', 'علم') or Buckwalter (e.g. 'rHm', 'ktb', 'Elm')"
                }
            },
            "required": ["root"]
        }
    },
    {
        "name": "compare_arabic_usage",
        "description": (
            "Compare how different forms of an Arabic root are used across the Quran. "
            "Shows each derived word form and the verses where it appears, revealing how "
            "the same root carries different meanings in different contexts. For example, "
            "root r-H-m (رحم) yields raHman (most gracious), raHim (merciful), raHmah (mercy) — "
            "each used in different theological contexts. Use this for linguistic analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "root": {
                    "type": "string",
                    "description": "Arabic root to analyse (e.g. 'رحم' or 'rHm')"
                }
            },
            "required": ["root"]
        }
    },
    {
        "name": "lookup_word",
        "description": (
            "Look up any Arabic word in the Quran. Returns the word's trilateral root, "
            "lemma, morphological pattern (wazn), part of speech, grammatical features, "
            "and all verse occurrences. Accepts Arabic script or Buckwalter transliteration. "
            "Use this when a user asks about a specific Arabic word."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "word": {
                    "type": "string",
                    "description": "Arabic word to look up (e.g. 'رحيم' or 'rHym')"
                }
            },
            "required": ["word"]
        }
    },
    {
        "name": "explore_root_family",
        "description": (
            "Show the full derivative tree of an Arabic root — all lemmas derived from it, "
            "grouped by morphological pattern, with semantic domain membership and sample verses. "
            "Reveals how a single root generates a family of related words. "
            "Use this to explore the semantic architecture of a root."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "root": {
                    "type": "string",
                    "description": "Arabic root to explore (e.g. 'رحم' or 'rHm')"
                }
            },
            "required": ["root"]
        }
    },
    {
        "name": "get_verse_words",
        "description": (
            "Get a complete word-by-word grammatical breakdown of a Quranic verse. "
            "Returns each word with its Arabic text, root, lemma, morphological pattern, "
            "English gloss, part of speech, and grammatical features (person, gender, number, case). "
            "Use this when a user wants to understand the grammar or etymology of a specific verse."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "verse_id": {
                    "type": "string",
                    "description": "Verse ID in surah:verse format (e.g. '1:1')"
                }
            },
            "required": ["verse_id"]
        }
    },
    {
        "name": "search_semantic_field",
        "description": (
            "Find all Arabic roots and words that belong to a semantic domain (e.g. 'mercy', "
            "'knowledge', 'creation'). Returns the domain's roots with their derived lemmas "
            "and verse counts. Based on al-Isfahani's classical categorization. "
            "Use this to explore how the Quran expresses a concept through multiple roots."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Semantic domain name or Arabic name (e.g. 'mercy', 'رحمة', 'knowledge')"
                }
            },
            "required": ["domain"]
        }
    },
    {
        "name": "lookup_wujuh",
        "description": (
            "Show all distinct meanings (wujuh) of a polysemous root across the Quran. "
            "Based on the classical Islamic discipline of al-wujuh wa al-naza'ir. "
            "Returns each sense with its meaning and sample verses. "
            "Use this when exploring how the same word carries different meanings in different contexts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "root": {
                    "type": "string",
                    "description": "Arabic root to look up wujuh for (e.g. 'هدي' or 'hdy')"
                }
            },
            "required": ["root"]
        }
    },
    {
        "name": "search_morphological_pattern",
        "description": (
            "Find Quranic words by morphological pattern (wazn), part of speech, or verbal form. "
            "For example, find all words on the فَعِيل pattern (intensive adjectives like رحيم, عليم), "
            "or all Form IV verbs (أَفْعَلَ — causative). Results grouped by root. "
            "Use this to study how morphological patterns shape meaning."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Morphological pattern in Arabic (e.g. 'فَعِيل', 'فَعُول') or Buckwalter"
                },
                "pos": {
                    "type": "string",
                    "description": "Part of speech filter (e.g. 'V.PERF', 'N', 'ADJ', 'ACT_PCPL')"
                },
                "verb_form": {
                    "type": "string",
                    "description": "Verbal form number or Roman numeral (e.g. '4', 'IV', 'X')"
                }
            }
        }
    },
    {
        "name": "recall_similar_query",
        "description": (
            "Surface past similar queries this agent has answered, with the "
            "tools they used and the answers they produced. Use when the "
            "current question feels familiar / repetitive — the past run is "
            "a playbook, not a final answer. Past tool sequences are hints "
            "for which retrieval paths worked. Don't use as the only "
            "retrieval method; verify answer_excerpt with direct tools."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "The current question text"},
                "top_k": {"type": "integer", "default": 3,
                          "description": "How many past matches to return"},
                "min_sim": {"type": "number", "default": 0.65,
                            "description": "Minimum cosine similarity (MiniLM)"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "run_cypher",
        "description": (
            "Execute a READ-ONLY Cypher query against the Quran graph. "
            "Use this for the long-tail of questions the specialised tools "
            "don't cover: aggregations, custom multi-hop traversals, schema "
            "introspection, percentile queries, etc. "
            "Schema reminder — Verse(verseId, surah, verseNum, text, arabicText, "
            "arabicPlain, surahName, embedding_m3 (1024d BGE-M3)), "
            "Sura(number, name), Keyword(keyword), ArabicRoot(root), Lemma(lemma), "
            "MorphPattern(pattern), SemanticDomain(name). "
            "Edges: MENTIONS{score, from_tfidf, to_tfidf}, RELATED_TO{score}, "
            "MENTIONS_ROOT, SIMILAR_PHRASE, NEXT_VERSE, CONTAINS, "
            "and typed: SUPPORTS, ELABORATES, QUALIFIES, CONTRASTS, REPEATS. "
            "FORBIDDEN: CREATE/MERGE/DELETE/SET/REMOVE/DETACH/LOAD CSV. Always "
            "include a LIMIT — one will be appended if missing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Cypher query (read-only). Example: "
                                   "'MATCH (v:Verse)-[:MENTIONS]->(k:Keyword {keyword: \"patience\"}) "
                                   "RETURN v.verseId, v.text ORDER BY v.surah LIMIT 20'"
                },
                "row_limit": {
                    "type": "integer",
                    "description": "Max rows to return (default 100, max 500)",
                    "default": 100
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_code19_features",
        "description": (
            "Retrieve Khalifa-style Code-19 mathematical features (verse counts, "
            "mysterious-letter frequencies, divisibility-by-19 indicators). These "
            "are arithmetic over the immutable Arabic text and CANNOT be hallucinated — "
            "use this when discussing the mathematical miracle of 19, the count of "
            "Q in Surahs 50/42, the count of N in Surah 68, or any claim that "
            "Khalifa makes about verse-arithmetic. "
            "scope='global' returns project-wide totals; scope='sura' takes a sura "
            "number; scope='verse' takes a verseId like '2:255'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["global", "sura", "verse"],
                    "description": "What level to query"
                },
                "target": {
                    "type": "string",
                    "description": "For scope='sura': sura number (e.g. '50'). For scope='verse': verseId (e.g. '2:255'). Omit for scope='global'."
                }
            },
            "required": ["scope"]
        }
    }
]

SYSTEM_PROMPT = cfg.system_prompt()


# ── agentic tool-use loop ──────────────────────────────────────────────────────

# ── tool-result cache (Nixon "1-second cache transforms your API" pattern) ────
#
# Same agentic loop often calls identical tools multiple times across turns.
# A short-TTL LRU cache avoids re-querying Neo4j for the same args inside a
# single conversation. Off by default for safety; flip on with env vars.
#
#   TOOL_CACHE_TTL_SEC=30   (0 = disabled, default = 30s)
#   TOOL_CACHE_MAX=256      (max entries, FIFO-on-insert eviction)
#
# Stats are tracked in _TOOL_CACHE_STATS — call get_tool_cache_stats() to read.

import threading

_TOOL_CACHE_TTL = float(os.environ.get("TOOL_CACHE_TTL_SEC", "30"))
_TOOL_CACHE_MAX = int(os.environ.get("TOOL_CACHE_MAX", "256"))
_TOOL_CACHE: dict = {}             # key -> (expires_at_ts, result_json)
_TOOL_CACHE_LOCK = threading.Lock()
_TOOL_CACHE_STATS = {"hits": 0, "misses": 0, "stores": 0, "evictions": 0,
                     "expired": 0, "skipped": 0}


def _tool_cache_key(tool_name: str, tool_input: dict, user_query: str | None) -> str:
    """Stable key over (tool, args, user_query, active vector index)."""
    args_json = json.dumps(tool_input or {}, sort_keys=True, ensure_ascii=False)
    uq = (user_query or "")[:200]   # cap to avoid pathological keys
    return f"{tool_name}|{_SEMANTIC_INDEX}|{args_json}|{uq}"


def _tool_cache_get(key: str):
    if _TOOL_CACHE_TTL <= 0:
        return None
    import time
    now = time.time()
    with _TOOL_CACHE_LOCK:
        entry = _TOOL_CACHE.get(key)
        if entry is None:
            _TOOL_CACHE_STATS["misses"] += 1
            return None
        expires, value = entry
        if expires < now:
            del _TOOL_CACHE[key]
            _TOOL_CACHE_STATS["expired"] += 1
            _TOOL_CACHE_STATS["misses"] += 1
            return None
        _TOOL_CACHE_STATS["hits"] += 1
        return value


def _tool_cache_put(key: str, value: str, has_error: bool = False):
    if _TOOL_CACHE_TTL <= 0 or has_error:
        if has_error:
            _TOOL_CACHE_STATS["skipped"] += 1
        return
    import time
    expires = time.time() + _TOOL_CACHE_TTL
    with _TOOL_CACHE_LOCK:
        # FIFO-style eviction if over cap
        if len(_TOOL_CACHE) >= _TOOL_CACHE_MAX:
            # drop the oldest (smallest expires_at)
            try:
                oldest_key = min(_TOOL_CACHE, key=lambda k: _TOOL_CACHE[k][0])
                del _TOOL_CACHE[oldest_key]
                _TOOL_CACHE_STATS["evictions"] += 1
            except ValueError:
                pass
        _TOOL_CACHE[key] = (expires, value)
        _TOOL_CACHE_STATS["stores"] += 1


def get_tool_cache_stats() -> dict:
    """Return a snapshot of cache stats — useful for observability."""
    with _TOOL_CACHE_LOCK:
        total = _TOOL_CACHE_STATS["hits"] + _TOOL_CACHE_STATS["misses"]
        hit_rate = _TOOL_CACHE_STATS["hits"] / total if total else 0.0
        return {
            **dict(_TOOL_CACHE_STATS),
            "size": len(_TOOL_CACHE),
            "max_size": _TOOL_CACHE_MAX,
            "ttl_sec": _TOOL_CACHE_TTL,
            "hit_rate": round(hit_rate, 4),
            "total_lookups": total,
        }


def clear_tool_cache():
    """Drop everything in the cache. For tests + cache busting after writes."""
    with _TOOL_CACHE_LOCK:
        _TOOL_CACHE.clear()
        for k in _TOOL_CACHE_STATS:
            _TOOL_CACHE_STATS[k] = 0


def dispatch_tool(session, tool_name: str, tool_input: dict, user_query: str = None) -> str:
    """Call the appropriate graph function, apply retrieval gating, return JSON."""
    cache_key = _tool_cache_key(tool_name, tool_input, user_query)
    cached = _tool_cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        if tool_name == "search_keyword":
            result = tool_search_keyword(session, **tool_input)
        elif tool_name == "get_verse":
            result = tool_get_verse(session, **tool_input)
        elif tool_name == "traverse_topic":
            result = tool_traverse_topic(session, **tool_input)
        elif tool_name == "find_path":
            result = tool_find_path(session, **tool_input)
        elif tool_name == "explore_surah":
            result = tool_explore_surah(session, **tool_input)
        elif tool_name == "semantic_search":
            result = tool_semantic_search(session, **tool_input)
        elif tool_name == "query_typed_edges":
            result = tool_query_typed_edges(session, **tool_input)
        elif tool_name == "search_arabic_root":
            result = tool_search_arabic_root(session, **tool_input)
        elif tool_name == "compare_arabic_usage":
            result = tool_compare_arabic_usage(session, **tool_input)
        elif tool_name == "lookup_word":
            result = tool_lookup_word(session, **tool_input)
        elif tool_name == "explore_root_family":
            result = tool_explore_root_family(session, **tool_input)
        elif tool_name == "get_verse_words":
            result = tool_get_verse_words(session, **tool_input)
        elif tool_name == "search_semantic_field":
            result = tool_search_semantic_field(session, **tool_input)
        elif tool_name == "lookup_wujuh":
            result = tool_lookup_wujuh(session, **tool_input)
        elif tool_name == "search_morphological_pattern":
            result = tool_search_morphological_pattern(session, **tool_input)
        elif tool_name == "get_code19_features":
            result = tool_get_code19_features(session, **tool_input)
        elif tool_name == "run_cypher":
            result = tool_run_cypher(session, **tool_input)
        elif tool_name == "recall_similar_query":
            result = tool_recall_similar_query(session, **tool_input)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        # Retrieval quality gating: rerank + assess + reorder
        if user_query and tool_name in ("search_keyword", "semantic_search", "traverse_topic"):
            try:
                from retrieval_gate import gate_tool_result
                result = gate_tool_result(user_query, tool_name, result)
            except Exception as e:
                result["gate_error"] = str(e)

    except Exception as e:
        result = {"error": str(e)}
    payload = json.dumps(result, ensure_ascii=False)
    has_error = isinstance(result, dict) and "error" in result
    _tool_cache_put(cache_key, payload, has_error=has_error)
    return payload


def run_agent_turn(
    user_message: str,
    conversation: list,
    session,
    client: anthropic.Anthropic,
) -> str:
    """
    Run one full agent turn: may involve multiple tool calls before final response.
    Returns the final text response.
    """
    conversation.append({"role": "user", "content": user_message})

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=cfg.llm_max_tokens(),
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=conversation,
        )

        # Collect any text blocks for display
        text_parts = [b.text for b in response.content if b.type == "text" and b.text.strip()]

        # If no tool calls, we're done
        if response.stop_reason != "tool_use":
            final_text = "\n".join(text_parts) if text_parts else "(no response)"
            conversation.append({"role": "assistant", "content": response.content})
            return final_text

        # Process tool calls
        tool_results = []
        tool_names = []
        for block in response.content:
            if block.type == "tool_use":
                tool_names.append(f"{block.name}({json.dumps(block.input)})")
                print(f"  [tool] {block.name}({json.dumps(block.input)[:80]}{'...' if len(json.dumps(block.input)) > 80 else ''})")
                result_str = dispatch_tool(session, block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

        # Add assistant turn + tool results to conversation
        conversation.append({"role": "assistant", "content": response.content})
        conversation.append({"role": "user", "content": tool_results})
        # Loop back for Claude to process tool results


# ── main loop ──────────────────────────────────────────────────────────────────

def main():
    print("Quran Knowledge Graph — Conversational Agent")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print("Claude can freely explore the graph using tools.")
    print()

    if not ANTHROPIC_KEY and not ANTHROPIC_TOKEN:
        print("Set ANTHROPIC_API_KEY or ANTHROPIC_OAUTH_TOKEN in your .env file first.")
        sys.exit(1)

    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        print("  Connected OK")
    except Exception as e:
        print(f"  Connection failed: {e}")
        sys.exit(1)

    if ANTHROPIC_TOKEN:
        client = anthropic.Anthropic(auth_token=ANTHROPIC_TOKEN)
    else:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    conversation = []

    print("\nType your question (or 'quit' / 'clear'):\n")

    with driver.session() as session:
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye.")
                break
            if question.lower() == "clear":
                conversation.clear()
                print("  Conversation cleared.\n")
                continue

            print()
            answer = run_agent_turn(question, conversation, session, client)
            print(f"\nClaude:\n{answer}\n")
            print("-" * 60)

    driver.close()


if __name__ == "__main__":
    main()
